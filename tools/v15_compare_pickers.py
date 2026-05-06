#!/usr/bin/env python3
"""Compare v0.3 (3-codec), v0.4 (4-codec), v0.5 (5-codec) meta-pickers on
a SHARED holdout split.

Pipeline:
    1. Load the v15 joined sweep (covers all 5 codecs).
    2. Build per-(image, band) cells with codec_bytes for ALL codecs.
    3. Take the deterministic 80/20 image-disjoint split (seed=7).
       This is the v15 split; v14 used the same seed and split function
       but its sample set was 4-codec; its inferences here are evaluated
       on the 5-codec holdout cells.
    4. For each pre-trained model JSON, run forward inference on the holdout
       cells and compute:
         - argmax-accuracy (only counted on cells where the model's
           predicted codec is among the cell's available codecs; cells
           where the model predicts a codec that didn't reach the band
           are still counted in bytes via the worst-bytes fallback).
         - bytes Δ vs always-zenjxl.
         - per-class winner share + per-class accuracy + bytes Δ.
    5. Print a side-by-side comparison table.

Usage:
    python3 tools/v15_compare_pickers.py
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

REPO = Path("/home/lilith/work/zen/zenanalyze")
JOINED_CACHE = Path("/tmp/v15-prep/joined.parquet")
FEATURES_TSV = Path(
    "/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv"
)
BANDS = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL = 1.5
SEED = 7

NAMED_FEATS = [
    "aspect_min_over_max", "chroma_complexity", "colourfulness",
    "dct_compressibility_uv", "dct_compressibility_y", "edge_density",
    "flat_color_block_ratio", "gradient_fraction", "high_freq_energy_ratio",
    "laplacian_variance", "log_pixels", "luma_histogram_entropy",
    "uniformity", "variance",
]
CCLASSES = ["photo", "screen", "lineart", "document", "synthetic"]


# Models to compare. Each entry must list the codec families the model
# outputs in the same order as the bake's softmax columns.
MODELS = [
    {
        "name": "v0.3 (3-codec)",
        "json": Path("/tmp/v12_metapicker_v0.3_model.json"),
        "classes": ["zenwebp", "zenjxl", "zenavif"],
    },
    {
        "name": "v0.4 (4-codec)",
        "json": Path("/tmp/v14-prep/v14_metapicker_model.json"),
        "classes": ["zenjpeg", "zenwebp", "zenjxl", "zenavif"],
    },
    {
        "name": "v0.5 (5-codec)",
        "json": Path("/tmp/v15-prep/v15_metapicker_model.json"),
        "classes": ["zenjpeg", "zenwebp", "zenjxl", "zenavif", "zenpng"],
    },
]


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
            for c in CCLASSES:
                col = f"cclass_{c}"
                if r.get(col) and float(r[col]) > 0.5:
                    cclass_lookup[stem] = c
                    break
            else:
                cclass_lookup[stem] = classify_stem(stem.removesuffix(".png"))
    return feats, cclass_lookup


def build_band_winners(df: pd.DataFrame, classes_all: list[str]) -> list[dict]:
    """Build cells for ALL 5 codecs (so each model's eval has access to its
    universe). Cells with <2 codecs in band are kept (so v0.5 PNG-only
    cells appear); each model's accuracy is conditioned on the cells
    where its codec set has ≥1 entry in band."""
    samples: list[dict] = []
    rows_in_any_band = []
    for band in BANDS:
        sub = df[(df["zensim"] - band).abs() <= BAND_TOL].copy()
        sub["band"] = band
        rows_in_any_band.append(sub)
    bucketed = pd.concat(rows_in_any_band, ignore_index=True)
    g = bucketed.groupby(["image", "band", "codec"], as_index=False)["bytes"].min()
    wide = g.pivot_table(index=["image", "band"], columns="codec", values="bytes", aggfunc="min")
    for (image, band), row in wide.iterrows():
        codec_bytes = {c: int(row[c]) for c in classes_all if c in row.index and pd.notna(row[c])}
        if not codec_bytes:
            continue
        winner = min(codec_bytes, key=codec_bytes.get)
        samples.append({
            "image": image, "band": float(band),
            "winner": winner, "codec_bytes": codec_bytes,
            "n_codecs": len(codec_bytes),
        })
    return samples


def leaky_relu(x, slope=0.01):
    return np.where(x > 0, x, slope * x)


def relu(x):
    return np.maximum(x, 0)


def forward(model: dict, X: np.ndarray) -> np.ndarray:
    """Numpy forward pass. Standardize, then dense → activation → dense ...
    Final layer is identity (raw logits). Returns argmax indices."""
    mean = np.array(model["scaler_mean"], dtype=np.float32)
    scale = np.array(model["scaler_scale"], dtype=np.float32)
    Z = ((X - mean) / scale).astype(np.float32)
    act = model.get("activation", "relu").lower()
    af = leaky_relu if act in ("leakyrelu", "leaky_relu") else relu
    layers = model["layers"]
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float32)
        b = np.asarray(layer["b"], dtype=np.float32)
        Z = Z @ W + b
        if i < len(layers) - 1:
            Z = af(Z)
    return np.argmax(Z, axis=1)


def evaluate(model_entry: dict, hold: list[dict], feats: dict, cclass_lookup: dict,
             baseline: str = "zenjxl") -> dict:
    """Forward through model, compute bytes Δ + per-class breakdown."""
    model = json.loads(model_entry["json"].read_text())
    classes = model_entry["classes"]
    n_in = model["n_inputs"]
    # Sanity: feat layout = 14 named + 5 cclass + 1 band = 20.
    assert n_in == 20, f"model {model_entry['name']} has unexpected n_inputs={n_in}"

    X = []
    for s in hold:
        cls = cclass_lookup.get(s["image"],
                                 classify_stem(s["image"].removesuffix(".png")))
        feat = feats[s["image"]] + cclass_one_hot(cls) + [s["band"]]
        X.append(feat)
    X = np.array(X, dtype=np.float32)
    pred_idx = forward(model, X)
    pred_codecs = [classes[i] for i in pred_idx]

    def cell_bytes_for(s, codec):
        if codec in s["codec_bytes"]:
            return s["codec_bytes"][codec]
        # Fallback: max bytes among in-band codecs (worst-case penalty).
        return max(s["codec_bytes"].values())

    base_b = sum(cell_bytes_for(s, baseline) for s in hold)
    mlp_b = sum(cell_bytes_for(s, p) for s, p in zip(hold, pred_codecs))
    oracle_b = sum(min(s["codec_bytes"].values()) for s in hold)
    pct = lambda x: (x - base_b) / base_b * 100 if base_b else 0.0

    # Accuracy: cell prediction matches the cell's argmin codec.
    correct = sum(1 for s, p in zip(hold, pred_codecs) if p == s["winner"])
    acc = correct / max(1, len(hold))

    # Per-class.
    per_class = {}
    by_cls = defaultdict(list)
    for s, p in zip(hold, pred_codecs):
        cls = cclass_lookup.get(s["image"],
                                 classify_stem(s["image"].removesuffix(".png")))
        by_cls[cls].append((s, p))
    for cls, items in by_cls.items():
        n = len(items)
        cor = sum(1 for s, p in items if s["winner"] == p)
        m_b = sum(cell_bytes_for(s, p) for s, p in items)
        b_b = sum(cell_bytes_for(s, baseline) for s, _ in items)
        o_b = sum(min(s["codec_bytes"].values()) for s, _ in items)
        per_class[cls] = {
            "n": n, "acc": cor / n,
            "mlp_pct": (m_b - b_b) / b_b * 100 if b_b else 0,
            "oracle_pct": (o_b - b_b) / b_b * 100 if b_b else 0,
        }

    # Per-codec accuracy (when each is true winner).
    per_codec = {}
    by_codec = defaultdict(list)
    for s, p in zip(hold, pred_codecs):
        by_codec[s["winner"]].append(p)
    for c, ps in by_codec.items():
        per_codec[c] = {
            "n": len(ps),
            "acc": sum(1 for p in ps if p == c) / max(1, len(ps)),
        }

    pred_dist = Counter(pred_codecs)

    return {
        "name": model_entry["name"],
        "acc": acc,
        "base_b": base_b, "mlp_b": mlp_b, "oracle_b": oracle_b,
        "mlp_pct": pct(mlp_b), "oracle_pct": pct(oracle_b),
        "per_class": per_class,
        "per_codec": per_codec,
        "pred_dist": pred_dist,
    }


def main() -> int:
    feats, cclass_lookup = load_features()
    if not JOINED_CACHE.exists():
        sys.stderr.write(f"missing {JOINED_CACHE}; run tools/v15_metapicker_train.py first\n")
        return 1
    sweep = pd.read_parquet(JOINED_CACHE)
    sweep = sweep[sweep["image"].isin(feats.keys())].copy()

    classes_all = ["zenjpeg", "zenwebp", "zenjxl", "zenavif", "zenpng"]
    samples = build_band_winners(sweep, classes_all)

    # Use SAME filter as v15 trainer for the comparison: ≥2 codecs in band.
    samples = [s for s in samples if s["n_codecs"] >= 2]

    # Image-disjoint 80/20 split with deterministic seed (matches v15).
    rng = random.Random(SEED)
    all_imgs = sorted({s["image"] for s in samples})
    rng.shuffle(all_imgs)
    n_hold = max(1, len(all_imgs) // 5)
    hold_imgs = set(all_imgs[:n_hold])
    hold = [s for s in samples if s["image"] in hold_imgs]

    print(f"# holdout: {len(hold)} cells over {n_hold} images "
          f"(seed={SEED}, ≥2-codec filter)\n", file=sys.stdout)
    print(f"# holdout winner distribution: "
          f"{dict(Counter(s['winner'] for s in hold))}\n", file=sys.stdout)

    results = []
    for entry in MODELS:
        if not entry["json"].exists():
            print(f"# SKIP {entry['name']}: missing {entry['json']}", file=sys.stdout)
            continue
        r = evaluate(entry, hold, feats, cclass_lookup)
        results.append(r)

    # Headline table.
    print("## Headline (holdout = same 40 images, 142 cells, ≥2-codec filter)\n")
    print(f"{'model':<18} {'acc':>7} {'bytes':>14} {'Δ vs jxl':>10} {'oracle Δ':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<18} {r['acc']:>7.4f} {r['mlp_b']:>14,} "
              f"{r['mlp_pct']:>+9.2f}% {r['oracle_pct']:>+9.2f}%")
    print()

    # Per-class comparison.
    all_classes = sorted({c for r in results for c in r["per_class"]})
    print("## Per-class behavior on holdout (mlp_dbytes vs always-jxl)\n")
    header = f"{'class':<12} {'n':>4}"
    for r in results:
        header += f"  {r['name']:<18}"
    print(header)
    print("-" * len(header))
    for cls in all_classes:
        n = next((r["per_class"][cls]["n"] for r in results if cls in r["per_class"]), 0)
        line = f"{cls:<12} {n:>4}"
        for r in results:
            pc = r["per_class"].get(cls)
            if pc:
                line += f"  acc={pc['acc']:.3f} {pc['mlp_pct']:+6.2f}%  "
            else:
                line += f"  {'—':<18}"
        print(line)
    print()

    # Per-codec accuracy.
    all_codecs = ["zenjpeg", "zenwebp", "zenjxl", "zenavif", "zenpng"]
    print("## Per-codec accuracy (when each codec is the true winner)\n")
    header = f"{'codec':<10} {'n':>4}"
    for r in results:
        header += f"  {r['name']:<18}"
    print(header)
    print("-" * len(header))
    for c in all_codecs:
        n_total = next((r["per_codec"][c]["n"] for r in results if c in r["per_codec"]), 0)
        if n_total == 0:
            continue
        line = f"{c:<10} {n_total:>4}"
        for r in results:
            pc = r["per_codec"].get(c)
            if pc:
                line += f"  acc={pc['acc']:.3f} (n={pc['n']:>3})  "
            else:
                line += f"  {'—':<18}"
        print(line)
    print()

    # Prediction distribution.
    print("## Prediction distribution on holdout\n")
    for r in results:
        n = sum(r["pred_dist"].values())
        share = " ".join(f"{c}={r['pred_dist'].get(c,0)/n*100:.1f}%"
                         for c in all_codecs if r["pred_dist"].get(c, 0) > 0)
        print(f"  {r['name']:<18} {share}")

    # Oracle ceiling (same regardless of model).
    print(f"\n# oracle ceiling on this holdout: {results[0]['oracle_pct']:+.2f}% vs always-zenjxl")
    print(f"# always-zenjxl baseline bytes: {results[0]['base_b']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
