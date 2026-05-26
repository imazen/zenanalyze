#!/usr/bin/env python3
"""Distance-aware picker prototype on v05c zenjxl data.

Question: with the same v05c data the v0.5 picker trained on, can we build
a SIMPLE picker that beats effort=7 default in the speed-safe regime?

Picker design:
- Input: image features + distance (the actual JXL quality knob)
- Output: which (effort, noise) cell to use at that distance
- Teacher signal: per (image, distance), the cell that minimizes bytes
  subject to (zensim ≥ default - 0.05) AND (encode_ms ≤ default × 1.05).
  If no cell beats default, label = effort=7 (i.e., picker should not
  pick anything else).

We use a HistGradientBoostingClassifier — far simpler than zentrain's
hybrid-heads MLP, and good for the categorical-output structure we want.

This is NOT a production picker. It's a proof that the v05c data
contains enough signal for a properly-trained picker to ship a real
speed-safe win against the static default. If this prototype shows
clean wins on a held-out split, the v0.5 picker just had bad teacher
labels — no new sweep needed.
"""
from __future__ import annotations

# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: picker_v06_classifier_prototype.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v0.6 zenjxl classifier prototype on v05c data (early R&D).\n"
    "         Use: tools/v14_metapicker_train.py for cross-codec routing, or zentrain/tools/train_hybrid.py for per-codec picking.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv, json, sys, math, random
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

FEATURES_TSV = Path(
    "/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv"
)
SWEEP_TSV = Path("/home/lilith/sweep-data/zenjxl_v05c.tsv")
DEFAULT_EFFORT = 7

# Default-relative tolerances for "safe" alternatives
ZENSIM_TOL = 0.05  # pp; allow tiny regression
SPEED_TOL = 1.05   # ratio; allow 5% slower
BYTES_GAIN = 0.99  # require >=1% byte savings


def load_features():
    feats = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        keep_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        for r in rdr:
            try:
                vec = []
                for c in keep_cols:
                    v = r.get(c, "")
                    vec.append(0.0 if v in ("", None) else float(v))
                feats[r["image_path"]] = (vec, keep_cols)
            except Exception:
                continue
    print(f"[features] {len(feats)} images, {len(keep_cols)} features", file=sys.stderr)
    return feats, keep_cols


def load_sweep():
    rows = []
    with open(SWEEP_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                if not r["encoded_bytes"] or not r["score_zensim"]:
                    continue
                bytes_ = int(r["encoded_bytes"])
                ms = float(r["encode_ms"])
                z = float(r["score_zensim"])
                k = json.loads(r["knob_tuple_json"])
                if k.get("noise") is True:
                    continue
                d = round(float(k["distance"]), 4)
                e = int(k["effort"])
                # Sweep paths are absolute worker paths; features TSV uses basename
                img_key = r["image_path"].rsplit("/", 1)[-1]
                rows.append((img_key, d, e, bytes_, ms, z))
            except Exception:
                continue
    print(f"[sweep] {len(rows)} cells (noise=False only)", file=sys.stderr)
    return rows


def build_dataset(feats, sweep_rows):
    """Per (image, distance), build the teacher label.

    label = best_safe_effort_index in [3, 5, 7, 9] tier.
    """
    EFFORTS = [1, 3, 5, 7, 9]
    EFFORT_TO_IDX = {e: i for i, e in enumerate(EFFORTS)}

    by_id = defaultdict(dict)  # (image, distance) -> {effort: (bytes, ms, zensim)}
    for img, d, e, b, ms, z in sweep_rows:
        by_id[(img, d)][e] = (b, ms, z)

    X = []  # feature + log(distance) input
    y = []  # cell label
    image_holdout_marker = []
    raw = []  # for analysis

    n_default_label = 0
    n_safe_alt_label = 0

    feat_cols = None

    for (img, dist), cells in by_id.items():
        if img not in feats:
            continue
        if DEFAULT_EFFORT not in cells:
            continue
        b_def, ms_def, z_def = cells[DEFAULT_EFFORT]
        # Find best safe alternative
        best_label = EFFORT_TO_IDX[DEFAULT_EFFORT]
        best_bytes = b_def
        for e in EFFORTS:
            if e == DEFAULT_EFFORT or e not in cells:
                continue
            b, ms, z = cells[e]
            if (b < best_bytes * BYTES_GAIN
                and z >= z_def - ZENSIM_TOL
                and ms <= ms_def * SPEED_TOL):
                if b < best_bytes:
                    best_bytes = b
                    best_label = EFFORT_TO_IDX[e]
        if best_label == EFFORT_TO_IDX[DEFAULT_EFFORT]:
            n_default_label += 1
        else:
            n_safe_alt_label += 1

        vec, cols = feats[img]
        if feat_cols is None:
            feat_cols = cols
        # input = features + log(distance)
        x = vec + [math.log(max(dist, 0.01))]
        X.append(x)
        y.append(best_label)
        image_holdout_marker.append(img)
        raw.append((img, dist, cells))

    print(
        f"[teacher] {len(X)} (image, distance) cells | "
        f"default={n_default_label} ({100*n_default_label/len(X):.1f}%) | "
        f"safe-alt={n_safe_alt_label} ({100*n_safe_alt_label/len(X):.1f}%)",
        file=sys.stderr,
    )
    print(f"[teacher] effort distribution in labels: {Counter(EFFORTS[lab] for lab in y)}", file=sys.stderr)

    return (
        np.array(X, dtype=np.float64),
        np.array(y, dtype=np.int64),
        image_holdout_marker,
        EFFORTS,
        feat_cols,
        raw,
    )


def main():
    feats, _ = load_features()
    sweep_rows = load_sweep()
    X, y, marker, EFFORTS, feat_cols, raw = build_dataset(feats, sweep_rows)

    # Image-level 80/20 split (no leakage)
    rng = random.Random(7)
    all_imgs = sorted(set(marker))
    rng.shuffle(all_imgs)
    n_holdout = int(len(all_imgs) * 0.2)
    holdout = set(all_imgs[:n_holdout])
    train_idx = [i for i, m in enumerate(marker) if m not in holdout]
    val_idx = [i for i, m in enumerate(marker) if m in holdout]
    print(f"[split] train rows={len(train_idx)} val rows={len(val_idx)}", file=sys.stderr)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=8,
        learning_rate=0.05,
        random_state=7,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    print(f"[train] train acc: {clf.score(X_tr, y_tr):.3f} val acc: {clf.score(X_va, y_va):.3f}", file=sys.stderr)

    pred = clf.predict(X_va)
    print(f"[predict] val pred distribution: {Counter(EFFORTS[p] for p in pred)}", file=sys.stderr)
    print(f"[predict] val truth distribution: {Counter(EFFORTS[t] for t in y_va)}", file=sys.stderr)

    # A/B against effort=7 default on held-out cells
    n = 0
    n_default_pick = 0
    n_safe_alt_pick = 0
    n_pick_matches_oracle = 0
    n_oracle_safe_alt = 0
    dbytes = []
    dzensim = []
    dms = []
    for i, (img, dist, cells) in enumerate(raw):
        if img not in holdout:
            continue
        n += 1
        b_def, ms_def, z_def = cells[DEFAULT_EFFORT]
        truth_e = EFFORTS[y_va[len(dbytes) if False else 0]]  # unused

        # Use the same idx mapping
        local_idx = val_idx.index(i) if i in val_idx else None
        if local_idx is None:
            continue
        picked_e = EFFORTS[pred[local_idx]]
        oracle_e = EFFORTS[y_va[local_idx]]

        if picked_e not in cells:
            continue  # picker chose unsupported cell

        b, ms, z = cells[picked_e]
        dbytes.append(100 * (b - b_def) / b_def)
        dzensim.append(z - z_def)
        dms.append(100 * (ms - ms_def) / ms_def)
        if picked_e == DEFAULT_EFFORT:
            n_default_pick += 1
        else:
            n_safe_alt_pick += 1
        if picked_e == oracle_e:
            n_pick_matches_oracle += 1
        if oracle_e != DEFAULT_EFFORT:
            n_oracle_safe_alt += 1

    print()
    print(f"## A/B: distance-aware picker over v05c (held-out {len(holdout)} images, {n} cells)")
    print(f"   Default cell: effort={DEFAULT_EFFORT}, noise=False")
    print(f"   Picker decisions: {n_default_pick} default ({100*n_default_pick/n:.1f}%), "
          f"{n_safe_alt_pick} safe-alt ({100*n_safe_alt_pick/n:.1f}%)")
    print(f"   Oracle had safe-alt available on {n_oracle_safe_alt} cells ({100*n_oracle_safe_alt/n:.1f}%)")
    print(f"   Picker matched oracle on {n_pick_matches_oracle} cells ({100*n_pick_matches_oracle/n:.1f}%)")
    print()
    print(f"   Bytes vs default:  mean {mean(dbytes):+.3f}%  median {median(dbytes):+.3f}%")
    print(f"   Zensim vs default: mean {mean(dzensim):+.4f}pp  median {median(dzensim):+.4f}pp")
    print(f"   Encode ms vs def:  mean {mean(dms):+.2f}%  median {median(dms):+.2f}%")
    print()
    print(f"## Same picker but ONLY counting cells where picker != default:")
    if n_safe_alt_pick:
        alt_dbytes = [d for d, p in zip(dbytes, [pred[val_idx.index(i)] for i, (img, _, _) in enumerate(raw) if img in holdout][:len(dbytes)]) if EFFORTS[p] != DEFAULT_EFFORT]
        # simpler: zip pred and dbytes per index
        alt_only_dbytes = []
        alt_only_dzensim = []
        alt_only_dms = []
        for i in range(len(dbytes)):
            # the i-th element of dbytes corresponds to the i-th held-out cell
            pass

    # Per-distance band breakdown
    bands = {"tight (≤1.0)": [], "mid (1..3)": [], "loose (>3)": []}
    band_dms = {"tight (≤1.0)": [], "mid (1..3)": [], "loose (>3)": []}
    band_dz = {"tight (≤1.0)": [], "mid (1..3)": [], "loose (>3)": []}
    j = 0
    for i, (img, dist, cells) in enumerate(raw):
        if img not in holdout:
            continue
        if i not in val_idx:
            continue
        local_idx = val_idx.index(i)
        picked_e = EFFORTS[pred[local_idx]]
        if picked_e not in cells or DEFAULT_EFFORT not in cells:
            continue
        b_def, ms_def, z_def = cells[DEFAULT_EFFORT]
        b, ms, z = cells[picked_e]
        d_pct = 100 * (b - b_def) / b_def
        dm_pct = 100 * (ms - ms_def) / ms_def
        dz_pp = z - z_def
        if dist <= 1.0:
            bands["tight (≤1.0)"].append(d_pct); band_dms["tight (≤1.0)"].append(dm_pct); band_dz["tight (≤1.0)"].append(dz_pp)
        elif dist <= 3.0:
            bands["mid (1..3)"].append(d_pct); band_dms["mid (1..3)"].append(dm_pct); band_dz["mid (1..3)"].append(dz_pp)
        else:
            bands["loose (>3)"].append(d_pct); band_dms["loose (>3)"].append(dm_pct); band_dz["loose (>3)"].append(dz_pp)

    print()
    print(f"## Per-distance band breakdown")
    print(f"{'band':18s} {'n':>5s} {'mean Δbytes':>12s} {'mean Δzensim':>13s} {'mean Δms':>10s}")
    for band in ["tight (≤1.0)", "mid (1..3)", "loose (>3)"]:
        d = bands[band]
        if not d:
            continue
        print(f"{band:18s} {len(d):>5d} {mean(d):>+12.3f}% {mean(band_dz[band]):>+13.4f}pp {mean(band_dms[band]):>+10.2f}%")


if __name__ == "__main__":
    main()
