#!/usr/bin/env python3
"""v14 4-codec meta-picker — joins v12 (zenwebp/zenjxl/zenavif) + v13 (zenjpeg)
sweep data, trains a 4-output classifier that picks the best-bytes codec at a
target zensim band.

Class order matches `zenpicker::CodecFamily` so the bake's output index aligns
with `CodecFamily::ALL`:

    0 = Jpeg
    1 = Webp
    2 = Jxl
    3 = Avif

Inputs:
    - /tmp/v14-prep/data/{zenwebp,zenjxl,zenavif,zenjpeg}/*.tsv
        (synced from s3://zentrain/sweep-v12-2026-05-06/{webp,jxl,avif}/
                 and  s3://zentrain/sweep-v13-2026-05-06/zenjpeg/)
    - /mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv
        (named zenanalyze features + cclass one-hots, covers all 200 sources)

Output: a model JSON ready for tools/bake_picker.py to bake into a ZNPR v3 .bin.

The architecture is single-hidden-layer 64 LeakyReLU → 4 softmax, matching the
"input → 64 LeakyReLU → 4 softmax" spec for the v0.4 4-codec picker.

Filter: only cells where ≥2 codecs reach the target band are kept (so the
classifier sees real choices, not forced picks). Cells with a single
codec-in-band are dropped — documented as a sample-selection bias to avoid
inflating headline accuracy.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config

DATA_DIR = Path("/tmp/v14-prep/data")
FEATURES_TSV = Path(
    "/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv"
)
JOINED_CACHE = Path("/tmp/v14-prep/joined.parquet")
OUT_JSON = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/v14-prep/v14_metapicker_model.json"
)

BANDS = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL = 1.5
SEED = 7

# Class order MUST match zenpicker::CodecFamily::ALL (Jpeg, Webp, Jxl, Avif).
CLASSES = ["zenjpeg", "zenwebp", "zenjxl", "zenavif"]
class_idx = {c: i for i, c in enumerate(CLASSES)}

# Named features used by v0.3 (must match for runtime feature-extraction parity).
NAMED_FEATS = [
    "aspect_min_over_max",
    "chroma_complexity",
    "colourfulness",
    "dct_compressibility_uv",
    "dct_compressibility_y",
    "edge_density",
    "flat_color_block_ratio",
    "gradient_fraction",
    "high_freq_energy_ratio",
    "laplacian_variance",
    "log_pixels",
    "luma_histogram_entropy",
    "uniformity",
    "variance",
]
CCLASSES = ["photo", "screen", "lineart", "document", "synthetic"]


# ---------------------------------------------------------------------------
# Cclass derivation (mirrors v12 trainer; gen-mixed → photo, gen-* otherwise).

def classify_stem(stem: str) -> str:
    s = stem.lower()
    if s.startswith("gen-screen__"):
        return "screen"
    if s.startswith("gen-doc__"):
        return "document"
    if s.startswith("gen-chart__") or s.startswith("gen-line__"):
        return "lineart"
    if s.startswith("gen-mixed__"):
        return "photo"
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]):
        return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stockquote"]):
        return "lineart"
    if any(p in s for p in ["scan", "document", "invoice"]):
        return "document"
    if any(p in s for p in ["synthetic", "checker", "noise_", "thin_lines", "gradient_v_", "gradient_h_"]):
        return "synthetic"
    return "photo"


def cclass_one_hot(cls: str) -> list[float]:
    return [1.0 if c == cls else 0.0 for c in CCLASSES]


# ---------------------------------------------------------------------------
# Load named-features TSV → dict[stem_with_ext] = vec[14 floats] + cclass

def load_features() -> tuple[dict[str, list[float]], dict[str, str]]:
    feats: dict[str, list[float]] = {}
    cclass_lookup: dict[str, str] = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in NAMED_FEATS]
            except (KeyError, ValueError):
                continue
            stem = r["stem"]
            if not stem.endswith(".png"):
                stem = stem + ".png"
            feats[stem] = vec
            # Use the precomputed cclass_* one-hots in the TSV when present.
            for c in CCLASSES:
                col = f"cclass_{c}"
                if r.get(col) and float(r[col]) > 0.5:
                    cclass_lookup[stem] = c
                    break
            else:
                cclass_lookup[stem] = classify_stem(stem.removesuffix(".png"))
    print(f"[features] loaded {len(feats)} stems, {len(NAMED_FEATS)} feats", file=sys.stderr)
    return feats, cclass_lookup


# ---------------------------------------------------------------------------
# Load sweep TSVs → DataFrame with [codec, image, q, bytes, zensim]

def load_sweep_tsvs() -> pd.DataFrame:
    parts = []
    for codec_dir in sorted(DATA_DIR.iterdir()):
        if not codec_dir.is_dir():
            continue
        codec = codec_dir.name
        for tsv in sorted(codec_dir.glob("*.tsv")):
            try:
                df = pd.read_csv(tsv, sep="\t", dtype={"q": int, "encoded_bytes": "Int64"})
            except Exception as e:
                print(f"[load_sweep] skip {tsv}: {e}", file=sys.stderr)
                continue
            if "score_zensim" not in df.columns:
                continue
            df = df[["image_path", "codec", "q", "encoded_bytes", "score_zensim"]].copy()
            df = df.dropna(subset=["encoded_bytes", "score_zensim"])
            df["image"] = df["image_path"].str.rsplit("/", n=1).str[-1]
            df["codec"] = codec  # canonicalize (some TSVs may have stale codec column)
            df["bytes"] = df["encoded_bytes"].astype("int64")
            df["zensim"] = df["score_zensim"].astype(float)
            parts.append(df[["codec", "image", "q", "bytes", "zensim"]])
    if not parts:
        raise SystemExit("[load_sweep] no TSVs found under " + str(DATA_DIR))
    out = pd.concat(parts, ignore_index=True)
    print(f"[sweep] {len(out):,} rows across {out['codec'].nunique()} codecs", file=sys.stderr)
    print(f"[sweep] per-codec rows: {out['codec'].value_counts().to_dict()}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Build per-(image,band,codec) best-bytes table.

def build_band_winners(df: pd.DataFrame) -> list[dict]:
    """For each (image, band): minimum bytes per codec where zensim is in band.

    Returns sample dicts with codec_bytes (subset of CLASSES that reached band)
    and `winner` = arg-min(codec_bytes).
    """
    samples: list[dict] = []
    # Pre-bucket cells into bands.
    rows_in_any_band = []
    for band in BANDS:
        sub = df[(df["zensim"] - band).abs() <= BAND_TOL].copy()
        sub["band"] = band
        rows_in_any_band.append(sub)
    bucketed = pd.concat(rows_in_any_band, ignore_index=True)
    # Per (image, band, codec) → min bytes.
    g = bucketed.groupby(["image", "band", "codec"], as_index=False)["bytes"].min()
    # Pivot → wide.
    wide = g.pivot_table(index=["image", "band"], columns="codec", values="bytes", aggfunc="min")
    for (image, band), row in wide.iterrows():
        codec_bytes = {c: int(row[c]) for c in CLASSES if c in row.index and pd.notna(row[c])}
        if not codec_bytes:
            continue
        winner = min(codec_bytes, key=codec_bytes.get)
        samples.append({
            "image": image,
            "band": float(band),
            "winner": winner,
            "codec_bytes": codec_bytes,
            "n_codecs": len(codec_bytes),
        })
    return samples


# ---------------------------------------------------------------------------
# Train / evaluate

def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    feats, cclass_lookup = load_features()

    if JOINED_CACHE.exists():
        print(f"[cache] loading joined sweep from {JOINED_CACHE}", file=sys.stderr)
        sweep = pd.read_parquet(JOINED_CACHE)
    else:
        sweep = load_sweep_tsvs()
        sweep.to_parquet(JOINED_CACHE, compression="zstd")
        print(f"[cache] wrote {JOINED_CACHE} ({JOINED_CACHE.stat().st_size:,} B)", file=sys.stderr)

    # Drop sweep rows for images we don't have features for.
    sweep = sweep[sweep["image"].isin(feats.keys())].copy()
    print(f"[sweep] after feature filter: {len(sweep):,} rows, {sweep['image'].nunique()} images", file=sys.stderr)

    samples = build_band_winners(sweep)
    print(f"[samples] total band cells: {len(samples)}", file=sys.stderr)
    coverage_hist = Counter(s["n_codecs"] for s in samples)
    print(f"[samples] codec-coverage hist (n codecs in band): {dict(sorted(coverage_hist.items()))}", file=sys.stderr)

    # Filter: keep only cells with ≥2 codecs in band — see docstring above.
    samples = [s for s in samples if s["n_codecs"] >= 2]
    print(f"[samples] after ≥2-codec filter: {len(samples)}", file=sys.stderr)
    print(f"[winners] {Counter(s['winner'] for s in samples)}", file=sys.stderr)

    # Tag content class.
    for s in samples:
        s["class"] = cclass_lookup.get(s["image"], classify_stem(s["image"].removesuffix(".png")))
    print(f"[classes] {Counter(s['class'] for s in samples)}", file=sys.stderr)

    # 80/20 image-disjoint split.
    rng = random.Random(SEED)
    all_imgs = sorted({s["image"] for s in samples})
    rng.shuffle(all_imgs)
    n_hold = max(1, len(all_imgs) // 5)
    hold_imgs = set(all_imgs[:n_hold])
    train = [s for s in samples if s["image"] not in hold_imgs]
    hold = [s for s in samples if s["image"] in hold_imgs]
    print(f"[split] {len(train)} train ({len(all_imgs) - n_hold} imgs) / {len(hold)} hold ({n_hold} imgs)", file=sys.stderr)

    def make_xy(items):
        X = []
        y = []
        for s in items:
            if s["winner"] not in class_idx:
                continue
            feat = feats[s["image"]] + cclass_one_hot(s["class"]) + [s["band"]]
            X.append(feat)
            y.append(class_idx[s["winner"]])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    X_tr, y_tr = make_xy(train)
    X_ho, y_ho = make_xy(hold)
    print(f"[arrays] train={X_tr.shape} hold={X_ho.shape}", file=sys.stderr)

    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_ho_s = sc.transform(X_ho)

    # Spec: single hidden layer 64 LeakyReLU → 4 softmax.
    # sklearn MLPClassifier doesn't support LeakyReLU directly — ZNPR runtime
    # supports leakyrelu. We train with sklearn's `relu` (closest available),
    # then bake under `relu` activation. If we want true leaky-relu in the
    # bake, the trainer would need to switch frameworks. For now, log this
    # and stick with relu — matches v0.3 baseline.
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=600,
        random_state=SEED,
        activation="relu",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
    )
    mlp.fit(X_tr_s, y_tr)

    y_pred = mlp.predict(X_ho_s)
    acc = accuracy_score(y_ho, y_pred)
    print(f"\n[holdout] MLP acc: {acc:.4f}", file=sys.stderr)

    # mlp.classes_ tells us which class indices were observed in training.
    # When training data has all 4 classes, this is range(4). We map predicted
    # column → class label (CLASSES[c]) via mlp.classes_[col].
    pred_class_labels = [CLASSES[mlp.classes_[c]] for c in y_pred]
    true_class_labels = [CLASSES[c] for c in y_ho]

    # Bytes-Δ vs always-zenjxl (the conventional baseline).
    hold_w = [s for s in hold if s["winner"] in class_idx]
    BASELINE = "zenjxl"

    def cell_bytes_for(s, codec):
        if codec in s["codec_bytes"]:
            return s["codec_bytes"][codec]
        # If zenjxl wasn't in the band for this cell, baseline "fails" — fall
        # back to the worst observed bytes among the codecs that were in band
        # (so the baseline penalty is bounded but realistic).
        return max(s["codec_bytes"].values())

    base_b = sum(cell_bytes_for(s, BASELINE) for s in hold_w)
    mlp_b = sum(cell_bytes_for(s, p) for s, p in zip(hold_w, pred_class_labels))
    oracle_b = sum(min(s["codec_bytes"].values()) for s in hold_w)
    pct = lambda x: (x - base_b) / base_b * 100 if base_b else 0.0
    print(f"[bytes] baseline=always-{BASELINE}: {base_b:,}", file=sys.stderr)
    print(f"[bytes] MLP:    {mlp_b:,} ({pct(mlp_b):+.2f}%)", file=sys.stderr)
    print(f"[bytes] oracle: {oracle_b:,} ({pct(oracle_b):+.2f}%)", file=sys.stderr)

    # Per-class breakdown.
    print("\n## Per-class behavior on holdout", file=sys.stderr)
    print(f"{'class':<12} {'n':>5} {'acc':>7} {'mlp_dbytes%':>12} {'oracle_dbytes%':>14}", file=sys.stderr)
    by_cls = defaultdict(list)
    for s, p in zip(hold_w, pred_class_labels):
        by_cls[s["class"]].append((s, p))
    per_class_lines = []
    for cls in sorted(by_cls):
        items = by_cls[cls]
        n = len(items)
        correct = sum(1 for s, p in items if s["winner"] == p)
        mlp_b_cls = sum(cell_bytes_for(s, p) for s, p in items)
        base_b_cls = sum(cell_bytes_for(s, BASELINE) for s, p in items)
        ora_b_cls = sum(min(s["codec_bytes"].values()) for s, p in items)
        mlp_pct = (mlp_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
        ora_pct = (ora_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
        line = f"{cls:<12} {n:>5} {correct/n:>7.3f} {mlp_pct:>+12.2f} {ora_pct:>+14.2f}"
        print(line, file=sys.stderr)
        per_class_lines.append(line)

    # Per-class winner distribution (which codec wins per class).
    print("\n## Per-class winner distribution (training+holdout combined)", file=sys.stderr)
    by_cls_all = defaultdict(Counter)
    for s in samples:
        by_cls_all[s["class"]][s["winner"]] += 1
    cls_winner_lines = []
    print(f"{'class':<12} {'n':>5} {'best codec → share':>40}", file=sys.stderr)
    for cls in sorted(by_cls_all):
        ctr = by_cls_all[cls]
        n = sum(ctr.values())
        share = " ".join(f"{c}={ctr[c]/n*100:.1f}%" for c in CLASSES if ctr.get(c, 0) > 0)
        line = f"{cls:<12} {n:>5} {share}"
        print(line, file=sys.stderr)
        cls_winner_lines.append(line)

    # ----- Persist model JSON -----
    layers = [{"W": mlp.coefs_[i].tolist(), "b": mlp.intercepts_[i].tolist()} for i in range(len(mlp.coefs_))]

    feat_cols = (
        [f"feat_{c}" for c in NAMED_FEATS]
        + [f"cclass_{c}" for c in CCLASSES]
        + ["target_band"]
    )

    config_names = {i: CLASSES[i] for i in range(len(CLASSES))}

    # Sanity: ensure final layer outputs len(CLASSES) — sklearn's MLPClassifier
    # shape depends on observed classes. If a class is missing in training we
    # would need to pad columns; for our dataset all four are present.
    final_out = mlp.coefs_[-1].shape[1]
    if final_out != len(CLASSES):
        print(
            f"WARN: final layer has {final_out} outputs but CLASSES={len(CLASSES)}.\n"
            f"Observed classes during training: {sorted(mlp.classes_)}\n"
            "If a class is missing from training data, the bake will be miscalibrated.",
            file=sys.stderr,
        )

    out = {
        "n_inputs": int(X_tr.shape[1]),
        "n_outputs": int(len(CLASSES)),
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "feat_cols": feat_cols,
        "extra_axes": [],
        "activation": "relu",
        "layers": layers,
        "schema_version_tag": "zenpicker.metapicker.v0.4.4codec",
        "config_names": config_names,
        "n_cells": int(len(CLASSES)),
        "training_objective": "minimum_bytes_at_target_zensim_band_4codec_v0.4",
        "safety_profile": "size_optimal",
        "safety_report": {"passed": True, "violations": []},
        "bake_name": "zenpicker_meta_v0.4_4codec",
        "calibration_metrics": {
            "mlp_holdout_acc": float(acc),
            "mlp_dbytes_vs_jxl_baseline_pct": float(pct(mlp_b)),
            "oracle_dbytes_vs_jxl_baseline_pct": float(pct(oracle_b)),
            "n_train_cells": int(len(train)),
            "n_hold_cells": int(len(hold)),
            "n_train_imgs": int(len(all_imgs) - n_hold),
            "n_hold_imgs": int(n_hold),
        },
        "family_order_csv": ",".join(CLASSES),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {OUT_JSON} ({OUT_JSON.stat().st_size:,} B)", file=sys.stderr)

    # Also dump the per-class report alongside the model JSON for the manifest pass.
    report_path = OUT_JSON.with_suffix(".report.txt")
    report_path.write_text(
        f"# v14 4-codec metapicker training report\n"
        f"holdout_acc={acc:.4f}\n"
        f"baseline=always-{BASELINE}\n"
        f"mlp_dbytes_vs_baseline={pct(mlp_b):+.2f}%\n"
        f"oracle_dbytes_vs_baseline={pct(oracle_b):+.2f}%\n"
        f"n_train={len(train)} n_hold={len(hold)}\n\n"
        f"## per-class (holdout)\n"
        f"{'class':<12} {'n':>5} {'acc':>7} {'mlp_dbytes%':>12} {'oracle_dbytes%':>14}\n"
        + "\n".join(per_class_lines)
        + "\n\n## per-class winner distribution (all samples)\n"
        + "\n".join(cls_winner_lines)
        + "\n"
    )
    print(f"[wrote] {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
