#!/usr/bin/env python3
"""
Held-out A/B for v0.4 picker via SWEEP TABLE LOOKUP.

We have:
- Trained picker JSON (sklearn MLP) at benchmarks/<codec>_hybrid_v04_<DATE>.json
- Sweep TSV (zentrain schema, post-adapter) with bytes & metric for every
  (image, cell, q) covered.

Instead of re-encoding through the runtime closed-loop, this A/B does
table-lookup against the sweep:
  picker_bytes(img, q)  := sweep[img, picker_cell(img), q].bytes
  default_bytes(img, q) := sweep[img, default_cell, q].bytes
  delta = (picker - default) / default

For "picker" we apply the trained MLP's argmin-over-cells head on
bytes_log to choose the cell. Scalar heads (tune / distance) are
ignored at this layer — we just measure cell-choice quality.

This is a SOFT A/B: no closed-loop re-encode, no target_zensim
iteration. It measures one specific question — "does the picker
choose better cells than the default?". For the v0.4 reduced grid
(2 cells per codec) this is the relevant question; the closed-loop
SHIP gate is a separate harness.

Usage:
  python3 tools/holdout_ab_lookup.py \\
      --model benchmarks/zenavif_hybrid_v04_2026-05-04.json \\
      --features ~/work/zen/zenavif/benchmarks/zenavif_features_2026-05-04_v04_full.tsv \\
      --pareto ~/work/zen/zenavif/benchmarks/zenavif_pareto_2026-05-04_v04_full.tsv \\
      --codec zenavif \\
      --default-cell speed6 \\
      --out-md ~/work/zen/zenavif/benchmarks/picker_v0.4_holdout_ab_2026-05-04.md \\
      --out-tsv /tmp/zenavif_v04_ab.tsv \\
      --holdout-frac 0.2 --seed 7
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def forward(model_json, x):
    """numpy forward pass on the trainer's MLP JSON dump.

    Layer keys are 'W' (n_in, n_out) and 'b' (n_out,) — matching
    sklearn's coefs_ orientation (input × hidden), not Keras's.
    """
    h = (np.asarray(x, dtype=np.float64) - np.asarray(model_json["scaler_mean"])) / np.asarray(model_json["scaler_scale"])
    layers = model_json["layers"]
    activation = model_json.get("activation", "relu")
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"])      # shape (n_in, n_out)
        b = np.asarray(layer["b"])      # shape (n_out,)
        h = h @ W + b
        if i < len(layers) - 1:
            if activation == "relu":
                h = relu(h)
            elif activation == "identity":
                pass
            else:
                raise ValueError(f"unsupported activation {activation}")
    return h


# Engineering pipeline matching train_hybrid.py:945-985 (Xe layout).
# 51 raw feats + 4 size_oh + 5 poly + 51 (zq_norm × feats) + 1 icc placeholder = 112
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}


def engineer_xe(feats_transformed, size_class, zq, w, h):
    f = np.asarray(feats_transformed, dtype=np.float64)
    log_px = math.log(max(1, w * h))
    size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float64)
    if size_class in SIZE_INDEX:
        size_oh[SIZE_INDEX[size_class]] = 1.0
    zq_norm = zq / 100.0
    poly = np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px])
    return np.concatenate([f, size_oh, poly, zq_norm * f, np.array([0.0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--pareto", required=True)
    ap.add_argument("--codec", required=True, choices=["zenavif", "zenjxl"])
    ap.add_argument("--default-cell", required=True,
                    help="e.g. 'speed6' for zenavif, 'effort7' for zenjxl")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    with open(args.model) as f:
        model = json.load(f)
    feat_cols_used = model["feat_cols"]
    n_cells = model.get("n_cells")
    hh = model.get("hybrid_heads_manifest", {})
    cell_names = [c["label"] for c in hh.get("cells", [])]
    output_layout = hh.get("output_layout", {"bytes_log": [0, n_cells]})
    bytes_range = output_layout.get("bytes_log", [0, n_cells])
    print(f"[model] feat_cols={len(feat_cols_used)} cells={cell_names} bytes_range={bytes_range}", file=sys.stderr)

    # Load features TSV
    features_by_img = {}
    with open(args.features) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            img = row["image_path"]
            try:
                # Numeric size_class encoding helpers
                sc = row.get("size_class", "unknown")
                vec = []
                for c in feat_cols_used:
                    v = row.get(c, "")
                    if v == "" or v is None:
                        vec.append(0.0)
                    else:
                        vec.append(float(v))
                features_by_img[img] = (vec, sc, int(row.get("width", "0") or 0), int(row.get("height", "0") or 0))
            except Exception as e:
                print(f"[features] skip {img}: {e}", file=sys.stderr)
    print(f"[features] loaded {len(features_by_img)} rows", file=sys.stderr)

    # Apply feature transforms (the MLP was trained on transformed inputs).
    # feature_transforms is a parallel list to feat_cols, one entry per feature.
    feat_transforms = model.get("feature_transforms", ["identity"] * len(feat_cols_used))

    def transform_row(vec):
        out = list(vec)
        for i in range(len(feat_cols_used)):
            t = feat_transforms[i] if i < len(feat_transforms) else "identity"
            v = out[i]
            if t == "log":
                out[i] = math.log(max(v, 1e-9))
            elif t == "log1p":
                out[i] = math.log1p(max(v, 0.0))
            else:
                pass  # identity
        return out

    n_inputs_expected = model["n_inputs"]
    print(f"[model] n_inputs={n_inputs_expected}, raw feat cols={len(feat_cols_used)}", file=sys.stderr)

    # Load pareto sweep — index by (image, cell, q) -> bytes
    # cell name format: per codec, e.g. 'speed6' or 'effort7'
    sweep = {}  # (image, cell, q) -> (bytes, zensim)
    cells_seen = defaultdict(int)
    with open(args.pareto) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            img = row["image_path"]
            q = int(row["q"])
            if args.codec == "zenavif":
                cell = f"speed{row['speed']}"
            else:
                cell = f"effort{row['effort']}"
            try:
                bytes_ = int(row["bytes"])
                z = float(row["zensim"])
            except (KeyError, ValueError):
                continue
            sweep[(img, cell, q)] = (bytes_, z)
            cells_seen[cell] += 1
    print(f"[sweep] {len(sweep)} rows, cells: {dict(cells_seen)}", file=sys.stderr)

    # Holdout split
    rng = random.Random(args.seed)
    all_imgs = sorted(set(img for img, *_ in sweep.keys()))
    rng.shuffle(all_imgs)
    n_holdout = max(1, int(len(all_imgs) * args.holdout_frac))
    holdout = set(all_imgs[:n_holdout])
    print(f"[split] holdout {len(holdout)} of {len(all_imgs)} images", file=sys.stderr)

    n_outputs = model["n_outputs"]
    print(f"[output] n_outputs={n_outputs}", file=sys.stderr)

    Z_TARGETS = list(range(30, 96, 5))

    rows = []
    misses = 0
    for img in sorted(holdout):
        if img not in features_by_img:
            misses += 1
            continue
        raw_vec, sc, w, h = features_by_img[img]
        feats_t = transform_row(raw_vec)

        for zq in Z_TARGETS:
            x = engineer_xe(feats_t, sc, zq, w, h)
            if len(x) != n_inputs_expected:
                print(f"[engineer] x has {len(x)} but model expects {n_inputs_expected} — aborting", file=sys.stderr)
                sys.exit(2)
            try:
                out = forward(model, x)
            except Exception as e:
                print(f"[forward] {img} zq={zq}: {e}", file=sys.stderr)
                continue
            # bytes_log slot per output_layout (lower = better bytes)
            b_lo, b_hi = bytes_range
            picker_cell_idx = int(np.argmin(out[b_lo:b_hi]))
            picker_cell = cell_names[picker_cell_idx]

            # Lookup actual sweep result for picker's chosen cell at zq
            picker_data = sweep.get((img, picker_cell, zq))
            default_data = sweep.get((img, args.default_cell, zq))
            if picker_data is None or default_data is None:
                continue
            pb, pz = picker_data
            db, dz = default_data
            delta_bytes = (pb - db) / db if db > 0 else 0.0
            rows.append({
                "image": img,
                "zq": zq,
                "picker_cell": picker_cell,
                "default_cell": args.default_cell,
                "picker_bytes": pb,
                "default_bytes": db,
                "picker_zensim": pz,
                "default_zensim": dz,
                "delta_bytes_pct": delta_bytes * 100,
                "delta_zensim_pp": pz - dz,
            })
    if misses:
        print(f"[features] {misses} held-out images had no features row", file=sys.stderr)

    # Write per-row TSV
    if rows:
        with open(args.out_tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[ab] wrote {len(rows)} rows to {args.out_tsv}", file=sys.stderr)

    # Aggregate verdict
    if not rows:
        print("[ab] NO ROWS — no overlap between holdout images and sweep+features.", file=sys.stderr)
        sys.exit(1)

    def stats(rs):
        deltas = [r["delta_bytes_pct"] for r in rs]
        zs = [r["delta_zensim_pp"] for r in rs]
        return {
            "n": len(rs),
            "mean_dbytes_pct": np.mean(deltas) if deltas else 0,
            "median_dbytes_pct": np.median(deltas) if deltas else 0,
            "win_rate": sum(1 for d in deltas if d < -0.1) / len(deltas) if deltas else 0,
            "mean_dzensim_pp": np.mean(zs) if zs else 0,
        }

    # Bands
    band_low = [r for r in rows if r["zq"] < 50]
    band_mid = [r for r in rows if 50 <= r["zq"] < 75]
    band_high = [r for r in rows if r["zq"] >= 75]
    overall = stats(rows)
    s_low, s_mid, s_high = stats(band_low), stats(band_mid), stats(band_high)

    # Per-cell picker preference
    picker_cells = defaultdict(int)
    for r in rows:
        picker_cells[r["picker_cell"]] += 1

    # Verdict: SHIP if mean_dbytes < -0.5% AND mean_dzensim_pp > -0.5
    ship = overall["mean_dbytes_pct"] < -0.5 and overall["mean_dzensim_pp"] > -0.5
    verdict = "SHIP" if ship else "HOLD"

    md = []
    md.append(f"# {args.codec} v0.4 picker held-out A/B (table-lookup)\n")
    md.append(f"\n**Verdict: {verdict}**\n\n")
    md.append(f"- Holdout: {len(holdout)} of {len(all_imgs)} images (frac={args.holdout_frac}, seed={args.seed})\n")
    md.append(f"- Method: table-lookup over the v0.4 sweep TSV; picker chooses cell, default cell = {args.default_cell}\n")
    md.append(f"- Cells in sweep: {dict(cells_seen)}\n")
    md.append(f"- Picker cell preference (held-out): {dict(picker_cells)}\n\n")

    md.append("## Per-band results\n\n")
    md.append("| band | n | mean Δbytes % | median Δbytes % | win rate (Δ<-0.1%) | mean Δzensim pp |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for label, s in (("zq30..49 (low)", s_low), ("zq50..74 (mid)", s_mid), ("zq75..95 (high)", s_high), ("overall", overall)):
        md.append(f"| {label} | {s['n']} | {s['mean_dbytes_pct']:+.2f} | {s['median_dbytes_pct']:+.2f} | {s['win_rate']*100:.1f}% | {s['mean_dzensim_pp']:+.2f} |\n")

    md.append("\n## Reading\n")
    md.append(f"- A {verdict} picker should beat the default cell on bytes at matched quality.\n")
    md.append(f"- Δbytes < 0 means picker is smaller. Δzensim_pp > 0 means picker is sharper.\n")
    md.append(f"- This is a TABLE-LOOKUP A/B: it does not measure closed-loop target_zensim convergence. The closed-loop SHIP gate is a separate harness.\n")
    md.append(f"- The v0.4 sweep grid is reduced (2 cells); the binary search space is small. Mean overhead is bounded above by the cell delta at any (img, q).\n")

    with open(args.out_md, "w") as f:
        f.write("".join(md))

    print(f"\n[verdict] {verdict}", file=sys.stderr)
    print(f"  overall n={overall['n']} mean Δbytes={overall['mean_dbytes_pct']:+.2f}% Δzensim={overall['mean_dzensim_pp']:+.2f}pp", file=sys.stderr)
    print(f"  bands: low n={s_low['n']} Δ={s_low['mean_dbytes_pct']:+.2f}%; mid n={s_mid['n']} Δ={s_mid['mean_dbytes_pct']:+.2f}%; high n={s_high['n']} Δ={s_high['mean_dbytes_pct']:+.2f}%", file=sys.stderr)
    print(f"  wrote {args.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
