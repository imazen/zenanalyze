#!/usr/bin/env python3
"""
Holdout-image multi-size generalization probe (zenanalyze#42 follow-up).

The existing `size_invariance_probe.py` uses the same 4 size_classes the
picker was trained on (tiny/small/medium/large). That tests *within-
bucket* correctness but doesn't tell us whether the picker generalizes
to **novel sizes** — the structural concern raised by the size-
generalization audit (`/mnt/v/output/zenpicker/size_generalization_audit_2026-04-30.md`).

This probe:

1. Reads a features TSV that contains the *same* images at **many sizes**
   spanning log-pixels uniformly (e.g. 64, 96, 128, 192, 256, 384, 512,
   768, 1024, 1536, 2048 — 11 sizes per image, generated via
   `zenjpeg_pareto_calibrate --features-only --sizes <list>`).
2. Loads a picker model JSON + manifest.
3. For each (image, size) row:
   - Standardizes features, runs the MLP forward pass
   - Records argmin cell index over the bytes_log subrange
   - Records top-1/top-2 gap (confidence) in log-bytes space
   - Counts features falling outside the manifest's
     `feature_bounds_p01_p99` (OOD rate)
4. Aggregates **per size** (across images): mean confidence, mean OOD
   rate, fraction of cells visited.
5. Aggregates **per image** (across sizes): how many distinct cells are
   chosen for the same image as size varies; mean confidence drift.

Without ground-truth optimal cells (which would require a fresh
encode-and-measure pareto run, ~hours), this probe answers the question
"does the picker know it's uncertain at novel sizes?" rather than "is
the picker correct at novel sizes?". A well-generalizing picker should
have:

- Comparable confidence at novel sizes vs trained sizes
- Low OOD rate (features stay in distribution)
- Reasonable argmin stability (similar images should land on similar
  cells regardless of size, modulo the trained size_class effects)

A picker that overfits to the 4 trained buckets will show:

- Confidence collapse at intermediate sizes
- High OOD rate at extreme or off-bucket sizes
- Erratic argmin flipping per image as size sweeps

Usage:
    # Step 1: generate dense-size features with zenjpeg
    cd ~/work/zen/zenjpeg
    cargo run --release -p zenjpeg --features 'target-zq trellis parallel' \\
        --example zq_pareto_calibrate -- \\
        --features-only \\
        --max-images 30 \\
        --sizes 64,96,128,192,256,384,512,768,1024,1536,2048 \\
        --features-output benchmarks/holdout_dense_sizes.tsv \\
        --output benchmarks/.unused.tsv

    # Step 2: run the probe
    python3 tools/holdout_size_sweep_probe.py \\
        --model benchmarks/zq_bytes_hybrid_v2_2.json \\
        --features-tsv benchmarks/holdout_dense_sizes.tsv \\
        --report-out /mnt/v/output/zenpicker/holdout_size_sweep_<date>.md
"""

import argparse
import csv
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_features_tsv(path: Path, feat_cols: list[str]):
    """Returns rows as list of dicts: {image, width, height, log_pixels, feats: np.array}."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        col_names = reader.fieldnames or []
        feat_idxs = []
        for c in feat_cols:
            if c not in col_names:
                sys.exit(f"feat col {c!r} missing from {path}")
            feat_idxs.append(c)
        for r in reader:
            try:
                feats = np.array(
                    [float(r[c]) if r[c] not in ("", "nan", "NaN") else np.nan for c in feat_idxs],
                    dtype=np.float32,
                )
            except (KeyError, ValueError) as e:
                sys.exit(f"bad row in {path}: {e}")
            w = int(r.get("width", 0))
            h = int(r.get("height", 0))
            rows.append(
                {
                    "image": r["image_path"],
                    "size_class": r.get("size_class", ""),
                    "width": w,
                    "height": h,
                    "log_pixels": float(np.log(max(w * h, 1))),
                    "feats": feats,
                }
            )
    return rows


def relu_forward(x: np.ndarray, layers: list[dict]) -> np.ndarray:
    cur = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float32)
        b = np.asarray(layer["b"], dtype=np.float32)
        cur = cur @ W + b
        if i != last:
            cur = np.maximum(cur, 0.0)
    return cur


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def build_engineered_input(
    feat_vec: np.ndarray,
    feat_cols: list[str],
    extra_axes: list[str],
    width: int,
    height: int,
    target_zq: float,
) -> np.ndarray:
    """Construct the full engineered input vector from raw features +
    image dimensions + zq target. Mirrors the v2.1 schema's extra_axes
    layout (size one-hot + size polys + zq cross-terms + icc bytes)."""
    out = np.zeros(len(feat_cols) + len(extra_axes), dtype=np.float32)
    out[: len(feat_cols)] = feat_vec
    n_pixels = max(width * height, 1)
    log_pixels = float(np.log(n_pixels))
    log_pixels_sq = log_pixels * log_pixels
    zq_norm = target_zq / 100.0
    zq_norm_sq = zq_norm * zq_norm
    base = len(feat_cols)
    # Build extra axes by name lookup — order from manifest preserved.
    name_to_value: dict[str, float] = {}
    # Size class one-hot (4 buckets) — derive size_class from log_pixels.
    # tiny ≤ 64×64 ≈ 4096 px ≈ log 8.32; small ≤ 256×256 ≈ 65k ≈ log 11;
    # medium ≤ 1024×1024 ≈ 1M ≈ log 13.8; otherwise large.
    if n_pixels <= 8192:
        sc = "tiny"
    elif n_pixels <= 131_072:
        sc = "small"
    elif n_pixels <= 1_572_864:
        sc = "medium"
    else:
        sc = "large"
    for s in ("tiny", "small", "medium", "large"):
        name_to_value[f"size_{s}"] = 1.0 if sc == s else 0.0
    name_to_value["log_pixels"] = log_pixels
    name_to_value["log_pixels_sq"] = log_pixels_sq
    name_to_value["zq_norm"] = zq_norm
    name_to_value["zq_norm_sq"] = zq_norm_sq
    name_to_value["zq_norm_x_log_pixels"] = zq_norm * log_pixels
    for col in feat_cols:
        name_to_value[f"zq_x_{col}"] = zq_norm * float(feat_vec[feat_cols.index(col)])
    name_to_value["icc_bytes"] = 0.0  # not in TSV; safe default

    for i, axis in enumerate(extra_axes):
        out[base + i] = float(name_to_value.get(axis, 0.0))
    return out


def pick(model: dict, x: np.ndarray, n_cells: int) -> tuple[int, float, np.ndarray]:
    """Forward pass + argmin over bytes_log subrange. Returns
    (cell_idx, top1_top2_log_gap, full_output)."""
    mean = np.asarray(model["scaler_mean"], dtype=np.float32)
    scale = np.asarray(model["scaler_scale"], dtype=np.float32)
    xs = (x - mean) / scale
    out = relu_forward(xs, model["layers"])
    bytes_log = out[:n_cells]
    sorted_idx = np.argsort(bytes_log)
    best = int(sorted_idx[0])
    second = float(bytes_log[sorted_idx[1]]) if n_cells > 1 else float(bytes_log[0])
    gap = second - float(bytes_log[best])
    return best, gap, out


def count_ood(feat_vec: np.ndarray, bounds: list[dict]) -> int:
    """Returns count of features whose value is outside [low, high] from
    feature_bounds_p01_p99. Aligned to feat_cols ordering."""
    n = 0
    for i, b in enumerate(bounds):
        v = feat_vec[i]
        lo = b.get("low", float("-inf"))
        hi = b.get("high", float("inf"))
        if not np.isfinite(v):
            continue
        if v < lo or v > hi:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--features-tsv", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest JSON (default: <model>.manifest.json)")
    ap.add_argument(
        "--target-zqs",
        default="50,75,90,94",
        help="Comma-separated target_zq values to probe per (image, size). "
        "Default mid + high range where pickers usually land.",
    )
    ap.add_argument("--report-out", type=Path, required=True)
    args = ap.parse_args()

    model = json.loads(args.model.read_text())
    manifest_path = args.manifest or args.model.with_suffix(".manifest.json")
    if not manifest_path.exists():
        # Fall back: try `<base>.manifest.json` next to .json
        sys.exit(f"manifest not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    feat_cols = list(manifest["feat_cols"])
    extra_axes = list(manifest["extra_axes"])
    n_cells = int(manifest.get("hybrid_heads", {}).get("n_cells", 0))
    if n_cells == 0:
        # Fall back to model.n_outputs assuming pure-categorical
        n_cells = int(model.get("n_cells", model["n_outputs"]))
    bounds = manifest.get("feature_bounds_p01_p99", [])
    if not bounds:
        sys.stderr.write(
            "WARNING: no feature_bounds_p01_p99 in manifest — OOD counts will be 0\n"
        )

    target_zqs = [float(z) for z in args.target_zqs.split(",")]

    rows = load_features_tsv(args.features_tsv, feat_cols)
    sys.stderr.write(f"loaded {len(rows)} (image, size) rows from {args.features_tsv}\n")

    # Group rows by image so we can compute per-image cell-stability
    # across sizes.
    by_image: dict[str, list] = defaultdict(list)
    for r in rows:
        by_image[r["image"]].append(r)
    sys.stderr.write(f"  → {len(by_image)} unique images\n\n")

    # Per-(size, zq) buckets for aggregation.
    per_size_zq: dict[tuple[int, float], list[dict]] = defaultdict(list)
    per_image_zq: dict[tuple[str, float], list[dict]] = defaultdict(list)

    for r in rows:
        for tz in target_zqs:
            x = build_engineered_input(
                r["feats"], feat_cols, extra_axes, r["width"], r["height"], tz
            )
            cell, gap, full = pick(model, x, n_cells)
            ood = count_ood(r["feats"], bounds)
            entry = {
                "image": r["image"],
                "width": r["width"],
                "height": r["height"],
                "n_pixels": r["width"] * r["height"],
                "log_pixels": r["log_pixels"],
                "size_class": r["size_class"],
                "target_zq": tz,
                "cell": cell,
                "gap": gap,
                "ood_count": ood,
                "ood_frac": ood / max(len(bounds), 1),
            }
            per_size_zq[(r["width"] * r["height"], tz)].append(entry)
            per_image_zq[(r["image"], tz)].append(entry)

    # Aggregate per (n_pixels, target_zq).
    summaries: list[dict] = []
    for (npx, tz), entries in sorted(per_size_zq.items()):
        n = len(entries)
        gaps = np.array([e["gap"] for e in entries])
        ood_fracs = np.array([e["ood_frac"] for e in entries])
        cells_visited = len({e["cell"] for e in entries})
        summaries.append(
            {
                "n_pixels": npx,
                "target_zq": tz,
                "n": n,
                "mean_gap": float(gaps.mean()),
                "p10_gap": float(np.percentile(gaps, 10)) if n else 0.0,
                "mean_ood_frac": float(ood_fracs.mean()),
                "max_ood_frac": float(ood_fracs.max()),
                "cells_visited": cells_visited,
            }
        )

    # Per-image cell stability across sizes (for each zq).
    image_stability: list[dict] = []
    for (img, tz), entries in per_image_zq.items():
        cells = [e["cell"] for e in entries]
        n_distinct = len(set(cells))
        image_stability.append(
            {
                "image": img,
                "target_zq": tz,
                "n_sizes": len(entries),
                "n_distinct_cells": n_distinct,
                "all_same_cell": n_distinct == 1,
            }
        )

    # Build report.
    today = datetime.date.today().isoformat()
    lines: list[str] = []
    lines.append("# Holdout multi-size generalization probe\n\n")
    lines.append(f"Date: {today}  •  model: `{args.model}`\n\n")
    lines.append(f"Features TSV: `{args.features_tsv}` ({len(rows)} rows, {len(by_image)} images)\n")
    lines.append(f"target_zqs: {target_zqs}\n\n")

    lines.append("## Per-size summary\n\n")
    lines.append(
        "| n_pixels | target_zq | n | mean conf gap | p10 conf gap | mean OOD% | max OOD% | distinct cells |\n"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summaries:
        lines.append(
            f"| {s['n_pixels']:,} | {s['target_zq']:.0f} | {s['n']} | "
            f"{s['mean_gap']:.3f} | {s['p10_gap']:.3f} | "
            f"{s['mean_ood_frac'] * 100:.1f}% | {s['max_ood_frac'] * 100:.1f}% | "
            f"{s['cells_visited']} |\n"
        )

    # Cell-stability roll-up across sizes per zq.
    lines.append("\n## Per-image cell stability (across all sizes, per target_zq)\n\n")
    by_zq_stability: dict[float, list[dict]] = defaultdict(list)
    for s in image_stability:
        by_zq_stability[s["target_zq"]].append(s)
    lines.append("| target_zq | n_images | all-same-cell rate | mean distinct cells per image |\n")
    lines.append("|---:|---:|---:|---:|\n")
    for tz, items in sorted(by_zq_stability.items()):
        same = sum(1 for it in items if it["all_same_cell"])
        mean_distinct = float(np.mean([it["n_distinct_cells"] for it in items]))
        lines.append(
            f"| {tz:.0f} | {len(items)} | "
            f"{same / len(items) * 100:.1f}% ({same}/{len(items)}) | "
            f"{mean_distinct:.2f} |\n"
        )

    # Worst-case OOD images (top 5).
    lines.append("\n## Top-5 worst OOD-rate (image, size, zq) cases\n\n")
    all_entries: list[dict] = []
    for entries in per_size_zq.values():
        all_entries.extend(entries)
    all_entries.sort(key=lambda e: e["ood_frac"], reverse=True)
    lines.append("| OOD% | image | size | target_zq | cell | conf gap |\n|---:|---|---:|---:|---:|---:|\n")
    for e in all_entries[:5]:
        lines.append(
            f"| {e['ood_frac'] * 100:.1f}% | `{Path(e['image']).name}` | "
            f"{e['width']}×{e['height']} | {e['target_zq']:.0f} | "
            f"{e['cell']} | {e['gap']:.3f} |\n"
        )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("".join(lines))
    sys.stderr.write(f"\nwrote {args.report_out}\n")
    print("".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
