#!/usr/bin/env python3
"""Distance-band held-out A/B for zenjxl pickers.

Variant of holdout_ab_lookup.py that keys the sweep by (image, cell, distance)
instead of (image, cell, q), since zenjxl uses distance as the quality axis
(not q — the JXL CLI overrides q with distance when distance is set).

Bands by distance: tight=[0.05..1.0], mid=[1.0..3.0], loose=[3.0..15].
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

SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}


def relu(x):
    return np.maximum(x, 0.0)


def forward(model_json, x):
    h = (np.asarray(x, dtype=np.float64) - np.asarray(model_json["scaler_mean"])) / np.asarray(model_json["scaler_scale"])
    layers = model_json["layers"]
    activation = model_json.get("activation", "relu")
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"])
        b = np.asarray(layer["b"])
        h = h @ W + b
        if i < len(layers) - 1:
            if activation == "relu":
                h = relu(h)
    return h


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
    ap.add_argument("--default-cell", required=True, help="e.g. effort7")
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
    n_inputs_expected = model["n_inputs"]
    print(f"[model] cells={cell_names} n_inputs={n_inputs_expected}", file=sys.stderr)

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
        return out

    # Load features
    features_by_img = {}
    with open(args.features) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            img = row["image_path"]
            try:
                vec = []
                for c in feat_cols_used:
                    v = row.get(c, "")
                    vec.append(0.0 if v in ("", None) else float(v))
                features_by_img[img] = (vec, row.get("size_class", "unknown"),
                                          int(row.get("width", "0") or 0),
                                          int(row.get("height", "0") or 0))
            except Exception:
                pass
    print(f"[features] loaded {len(features_by_img)}", file=sys.stderr)

    # Load pareto — key by (img, cell, distance) since q is dummy
    sweep = {}
    cells_seen = defaultdict(int)
    distances_seen = set()
    with open(args.pareto) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            img = row["image_path"]
            try:
                effort = int(row["effort"])
                distance = float(row["distance"])
                bytes_ = int(row["bytes"])
                z = float(row["zensim"])
            except (KeyError, ValueError):
                continue
            cell = f"effort{effort}"
            key = (img, cell, round(distance, 4))
            sweep[key] = (bytes_, z)
            cells_seen[cell] += 1
            distances_seen.add(round(distance, 4))
    print(f"[sweep] {len(sweep)} rows, cells: {dict(cells_seen)}, distances: {sorted(distances_seen)}", file=sys.stderr)

    # Holdout split
    rng = random.Random(args.seed)
    all_imgs = sorted(set(img for img, *_ in sweep.keys()))
    rng.shuffle(all_imgs)
    n_holdout = max(1, int(len(all_imgs) * args.holdout_frac))
    holdout = set(all_imgs[:n_holdout])
    print(f"[split] holdout {len(holdout)} of {len(all_imgs)}", file=sys.stderr)

    # For each (image, distance), pick picker's chosen cell + compare to default
    # Use distance as the quality axis. Map each distance to a band:
    def band(d):
        if d <= 1.0: return "tight"
        if d <= 3.0: return "mid"
        return "loose"

    # zq input to picker: still need to construct an Xe vector. We use a midpoint
    # zq=75 since the model was trained with q=75 dummy. Picker's choice is then
    # invariant to the zq input but conditioned on the image features. (This is
    # a limitation we accept; the model wasn't trained with zq as a meaningful axis.)

    rows = []
    misses = 0
    for img in sorted(holdout):
        if img not in features_by_img:
            misses += 1
            continue
        raw_vec, sc, w, h = features_by_img[img]
        feats_t = transform_row(raw_vec)
        x = engineer_xe(feats_t, sc, 75, w, h)
        if len(x) != n_inputs_expected:
            print(f"[engineer] dim mismatch", file=sys.stderr)
            sys.exit(2)
        out = forward(model, x)
        b_lo, b_hi = bytes_range
        picker_cell_idx = int(np.argmin(out[b_lo:b_hi]))
        picker_cell = cell_names[picker_cell_idx]

        # For each distance level in the sweep, compare picker_cell vs default_cell
        for distance in sorted(distances_seen):
            picker_data = sweep.get((img, picker_cell, distance))
            default_data = sweep.get((img, args.default_cell, distance))
            if picker_data is None or default_data is None:
                continue
            pb, pz = picker_data
            db, dz = default_data
            delta_bytes = (pb - db) / db if db > 0 else 0.0
            rows.append({
                "image": img,
                "distance": distance,
                "band": band(distance),
                "picker_cell": picker_cell,
                "default_cell": args.default_cell,
                "picker_bytes": pb,
                "default_bytes": db,
                "picker_zensim": pz,
                "default_zensim": dz,
                "delta_bytes_pct": delta_bytes * 100,
                "delta_zensim_pp": pz - dz,
            })

    if rows:
        with open(args.out_tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[ab] wrote {len(rows)} rows to {args.out_tsv}", file=sys.stderr)

    if not rows:
        print("[ab] NO ROWS", file=sys.stderr)
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

    band_tight = [r for r in rows if r["band"] == "tight"]
    band_mid = [r for r in rows if r["band"] == "mid"]
    band_loose = [r for r in rows if r["band"] == "loose"]
    overall = stats(rows)

    picker_cells = defaultdict(int)
    for r in rows:
        picker_cells[r["picker_cell"]] += 1

    ship = overall["mean_dbytes_pct"] < -0.5 and overall["mean_dzensim_pp"] > -0.5
    verdict = "SHIP" if ship else "HOLD"

    md = []
    md.append(f"# zenjxl held-out A/B (distance-banded)\n\n**Verdict: {verdict}**\n\n")
    md.append(f"- Holdout: {len(holdout)} images (frac={args.holdout_frac}, seed={args.seed}); {len(rows)} (image, distance) cells.\n")
    md.append(f"- Default cell: {args.default_cell}\n")
    md.append(f"- Picker cell preference: {dict(picker_cells)}\n\n")
    md.append("## Per-band results\n\n")
    md.append("| band (distance) | n | mean Δbytes % | median Δbytes % | win rate | mean Δzensim_pp |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for label, s in (("tight (0.05..1.0)", stats(band_tight)),
                     ("mid (1.0..3.0)", stats(band_mid)),
                     ("loose (3.0..15)", stats(band_loose)),
                     ("overall", overall)):
        md.append(f"| {label} | {s['n']} | {s['mean_dbytes_pct']:+.2f} | {s['median_dbytes_pct']:+.2f} | {s['win_rate']*100:.1f}% | {s['mean_dzensim_pp']:+.2f} |\n")

    with open(args.out_md, "w") as f:
        f.write("".join(md))

    print(f"\n[verdict] {verdict}", file=sys.stderr)
    print(f"  overall n={overall['n']} mean Δbytes={overall['mean_dbytes_pct']:+.2f}% Δzensim={overall['mean_dzensim_pp']:+.2f}pp", file=sys.stderr)


if __name__ == "__main__":
    main()
