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

DEDUP-C (2026-05-26): shared scaffolding extracted to
`zentrain/tools/_metapicker_lib.py`. This wrapper now owns only v14-specific
concerns: which codecs, sklearn model + training, baseline.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Make `zentrain/tools/_metapicker_lib.py` importable from this script.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "zentrain" / "tools"))
from _metapicker_lib import (
    BANDS_DEFAULT,
    BAND_TOL_DEFAULT,
    SEED_DEFAULT,
    CCLASSES,
    classify_stem,
    cclass_one_hot,
    load_features,
    load_sweep_tsvs,
    build_band_winners,
    image_disjoint_split,
    cell_bytes_for,
    bytes_delta_vs_baseline,
    format_per_class_report,
    format_per_class_winner_distribution,
    write_metapicker_json,
)

# ---------------------------------------------------------------------------
# Config — v14-specific.

DATA_DIR = Path("/tmp/v14-prep/data")
FEATURES_TSV = Path(
    "/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv"
)
JOINED_CACHE = Path("/tmp/v14-prep/joined.parquet")
OUT_JSON = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/v14-prep/v14_metapicker_model.json"
)

BANDS = BANDS_DEFAULT
BAND_TOL = BAND_TOL_DEFAULT
SEED = SEED_DEFAULT

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


# ---------------------------------------------------------------------------
# Train / evaluate

def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    feats, cclass_lookup = load_features(FEATURES_TSV, NAMED_FEATS)

    if JOINED_CACHE.exists():
        print(f"[cache] loading joined sweep from {JOINED_CACHE}", file=sys.stderr)
        sweep = pd.read_parquet(JOINED_CACHE)
    else:
        sweep = load_sweep_tsvs(DATA_DIR)
        sweep.to_parquet(JOINED_CACHE, compression="zstd")
        print(f"[cache] wrote {JOINED_CACHE} ({JOINED_CACHE.stat().st_size:,} B)", file=sys.stderr)

    # Drop sweep rows for images we don't have features for.
    sweep = sweep[sweep["image"].isin(feats.keys())].copy()
    print(f"[sweep] after feature filter: {len(sweep):,} rows, {sweep['image'].nunique()} images", file=sys.stderr)

    samples = build_band_winners(sweep, BANDS, BAND_TOL, CLASSES)
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
    train, hold, all_imgs, hold_imgs = image_disjoint_split(samples, seed=SEED)
    n_hold = len(hold_imgs)
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

    # Bytes-Δ vs always-zenjxl (the conventional baseline).
    hold_w = [s for s in hold if s["winner"] in class_idx]
    BASELINE = "zenjxl"
    base_b, mlp_b, oracle_b, pct = bytes_delta_vs_baseline(hold_w, pred_class_labels, BASELINE)
    print(f"[bytes] baseline=always-{BASELINE}: {base_b:,}", file=sys.stderr)
    print(f"[bytes] MLP:    {mlp_b:,} ({pct(mlp_b):+.2f}%)", file=sys.stderr)
    print(f"[bytes] oracle: {oracle_b:,} ({pct(oracle_b):+.2f}%)", file=sys.stderr)

    # Per-class breakdown.
    print("\n## Per-class behavior on holdout", file=sys.stderr)
    print(f"{'class':<12} {'n':>5} {'acc':>7} {'mlp_dbytes%':>12} {'oracle_dbytes%':>14}", file=sys.stderr)
    per_class_lines = format_per_class_report(hold_w, pred_class_labels, cclass_lookup, BASELINE)
    for line in per_class_lines:
        print(line, file=sys.stderr)

    # Per-class winner distribution (which codec wins per class).
    print("\n## Per-class winner distribution (training+holdout combined)", file=sys.stderr)
    cls_winner_lines = format_per_class_winner_distribution(samples, CLASSES)
    print(f"{'class':<12} {'n':>5} {'best codec → share':>40}", file=sys.stderr)
    for line in cls_winner_lines:
        print(line, file=sys.stderr)

    # ----- Persist model JSON -----
    layers = [{"W": mlp.coefs_[i].tolist(), "b": mlp.intercepts_[i].tolist()} for i in range(len(mlp.coefs_))]

    feat_cols = (
        [f"feat_{c}" for c in NAMED_FEATS]
        + [f"cclass_{c}" for c in CCLASSES]
        + ["target_band"]
    )

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

    write_metapicker_json(
        OUT_JSON,
        n_inputs=int(X_tr.shape[1]),
        classes=CLASSES,
        scaler_mean=sc.mean_.tolist(),
        scaler_scale=sc.scale_.tolist(),
        feat_cols=feat_cols,
        layers=layers,
        activation="relu",
        schema_version_tag="zenpicker.metapicker.v0.4.4codec",
        bake_name="zenpicker_meta_v0.4_4codec",
        training_objective="minimum_bytes_at_target_zensim_band_4codec_v0.4",
        calibration_metrics={
            "mlp_holdout_acc": float(acc),
            "mlp_dbytes_vs_jxl_baseline_pct": float(pct(mlp_b)),
            "oracle_dbytes_vs_jxl_baseline_pct": float(pct(oracle_b)),
            "n_train_cells": int(len(train)),
            "n_hold_cells": int(len(hold)),
            "n_train_imgs": int(len(all_imgs) - n_hold),
            "n_hold_imgs": int(n_hold),
        },
    )
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
