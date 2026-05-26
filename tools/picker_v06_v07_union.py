#!/usr/bin/env python3
"""Unified v06+v07 picker — full cell taxonomy spans both sweeps.

The picker learns, for each (image, distance), which cell from the
COMBINED taxonomy gives the best safe-alt vs the v06 default
(effort=7, biters=0, ziters=0). Cells from v07 include extra knobs
(force_strategy, max_strategy_size, progressive, gaborish, patches,
lz77, lf_frame, pixel_domain_loss).

Training set: rows from BOTH sweeps. For (img, dist) where only v06
has data, the picker can only choose v06 cells; for (img, dist) where
only v07 has data, only v07 cells. For overlap, the picker can choose
from the full union.

Per-image holdout (same images held out from both sweeps).
"""
from __future__ import annotations

# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: picker_v06_v07_union.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v0.6+v0.7 unified picker (early R&D, both sweeps combined).\n"
    "         Use: tools/v14_metapicker_train.py or zentrain/tools/train_hybrid.py.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv, json, math, random, sys
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

import numpy as np

V06_TSV = Path("/home/lilith/sweep-data/zenjxl_v06.tsv")
V07_DIR = Path("/home/lilith/sweep-data/v07")
FEATURES_TSV = Path(
    "/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv"
)
OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/picker_v06_v07_union_report.md")

DEFAULT_EFFORT = 7
ZENSIM_TOL = 0.05
SPEED_TOL = 1.05
BYTES_GAIN = 0.99


def load_features():
    feats = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        keep = [c for c in rdr.fieldnames if c.startswith("feat_")]
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in keep]
                w = float(r.get("width", 0) or 0)
                h = float(r.get("height", 0) or 0)
                feats[r["image_path"]] = (vec, w, h)
            except Exception:
                continue
    return feats, keep


def parse_row(r, src):
    if not r["encoded_bytes"] or not r["score_zensim"]:
        return None
    try:
        k = json.loads(r["knob_tuple_json"])
    except Exception:
        return None
    if k.get("noise") is True:
        return None
    return {
        "src": src,
        "image": r["image_path"].rsplit("/", 1)[-1],
        "distance": round(float(k["distance"]), 4),
        "effort": int(k["effort"]),
        "biters": int(k.get("butteraugli_iters", 0)),
        "ziters": int(k.get("zensim_iters", 0)),
        "force_strategy": k.get("force_strategy"),
        "max_strategy_size": k.get("max_strategy_size"),
        "progressive": k.get("progressive", "single"),
        "gaborish": k.get("gaborish", True),
        "patches": k.get("patches"),
        "lz77": k.get("lz77"),
        "lf_frame": k.get("lf_frame"),
        "pixel_domain_loss": k.get("pixel_domain_loss", True),
        "bytes": int(r["encoded_bytes"]),
        "ms": float(r["encode_ms"]),
        "zensim": float(r["score_zensim"]),
    }


def load_all():
    rows = []
    with open(V06_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            row = parse_row(r, "v06")
            if row:
                rows.append(row)
    if V07_DIR.exists():
        for tsv in sorted(V07_DIR.glob("*.tsv")):
            with open(tsv) as f:
                rdr = csv.DictReader(f, delimiter="\t")
                for r in rdr:
                    row = parse_row(r, "v07")
                    if row:
                        rows.append(row)
    return rows


def cell_key(r):
    """Compact cell tuple — the union of v06 + v07 knobs."""
    return (
        r["effort"], r["biters"], r["ziters"],
        r["force_strategy"], r["max_strategy_size"], r["progressive"],
        r["gaborish"], r["patches"], r["lz77"], r["lf_frame"],
        r["pixel_domain_loss"],
    )


def is_safe(c, default):
    return (
        c["bytes"] < default["bytes"] * BYTES_GAIN
        and c["ms"] <= default["ms"] * SPEED_TOL
        and c["zensim"] >= default["zensim"] - ZENSIM_TOL
    )


def main():
    feats, _ = load_features()
    rows = load_all()
    print(f"[features] {len(feats)} images", file=sys.stderr)
    print(f"[rows] total {len(rows)} (v06: {sum(1 for r in rows if r['src']=='v06')}, v07: {sum(1 for r in rows if r['src']=='v07')})", file=sys.stderr)

    by_id = defaultdict(dict)
    for r in rows:
        c = cell_key(r)
        # If duplicate cell from both sweeps, keep the smaller bytes (best result)
        existing = by_id[(r["image"], r["distance"])].get(c)
        if existing is None or r["bytes"] < existing["bytes"]:
            by_id[(r["image"], r["distance"])][c] = r

    # Cells contain mixed None/bool/str/int — sort by str repr to avoid type errors.
    all_cells = sorted({c for cs in by_id.values() for c in cs.keys()}, key=lambda c: repr(c))
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    print(f"[teacher] {len(by_id)} (img, dist) pairs, {len(cell_to_idx)} unique cells in union", file=sys.stderr)

    # Default = (effort=7, biters=0, ziters=0, all v07-only knobs at v06-default values)
    default_cell = (DEFAULT_EFFORT, 0, 0, None, None, "single", True, None, None, None, True)
    if default_cell not in cell_to_idx:
        # Pick whatever cell with effort=7, biters=ziters=0 exists
        candidates = [c for c in all_cells if c[0] == DEFAULT_EFFORT and c[1] == 0 and c[2] == 0]
        if not candidates:
            print(f"ERROR: no default cell candidates", file=sys.stderr)
            return
        default_cell = candidates[0]
    print(f"[default] {default_cell}", file=sys.stderr)

    # Build dataset
    X, y, raw, image_for_row = [], [], [], []
    for (img, dist), cells in by_id.items():
        if img not in feats or default_cell not in cells:
            continue
        feat_vec, w, h = feats[img]
        default = cells[default_cell]
        best_label = cell_to_idx[default_cell]
        best_data = default
        for c, d in cells.items():
            if c == default_cell:
                continue
            if is_safe(d, default) and d["bytes"] < best_data["bytes"]:
                best_data = d
                best_label = cell_to_idx[c]
        log_dist = math.log(max(dist, 0.01))
        log_px = math.log(max(w * h, 1.0))
        x = feat_vec + [log_dist, log_px]
        X.append(x)
        y.append(best_label)
        raw.append((img, dist, cells, default_cell))
        image_for_row.append(img)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"[teacher] dataset {X.shape}, {len(set(y))} unique labels", file=sys.stderr)

    # Image-level holdout
    rng = random.Random(7)
    all_imgs = sorted(set(image_for_row))
    rng.shuffle(all_imgs)
    n_holdout = int(len(all_imgs) * 0.2)
    holdout = set(all_imgs[:n_holdout])
    tr_idx = np.array([i for i, m in enumerate(image_for_row) if m not in holdout])
    va_idx = np.array([i for i, m in enumerate(image_for_row) if m in holdout])
    print(f"[split] {len(all_imgs)} imgs, {n_holdout} holdout, {len(tr_idx)} tr / {len(va_idx)} va cells", file=sys.stderr)

    # MLP train
    import torch, torch.nn as nn
    torch.manual_seed(7)
    n_in = X.shape[1]
    n_classes = len(cell_to_idx)
    print(f"[mlp] in={n_in} out={n_classes}", file=sys.stderr)

    mu = X[tr_idx].mean(axis=0)
    sd = X[tr_idx].std(axis=0) + 1e-9
    X_n = (X - mu) / sd

    # Class weights
    label_counts = np.bincount(y[tr_idx], minlength=n_classes).astype(np.float32)
    cw = (label_counts.sum() / (label_counts + 1.0)).clip(0.5, 5.0)

    net = nn.Sequential(
        nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(cw))

    X_tr_t = torch.from_numpy(X_n[tr_idx]); y_tr_t = torch.from_numpy(y[tr_idx])
    X_va_t = torch.from_numpy(X_n[va_idx]); y_va_t = torch.from_numpy(y[va_idx])
    n = X_tr_t.shape[0]

    best_acc = 0.0
    best_state = None
    bad = 0
    for epoch in range(150):
        perm = torch.randperm(n)
        net.train()
        for s in range(0, n, 512):
            ix = perm[s:s + 512]
            opt.zero_grad()
            loss = loss_fn(net(X_tr_t[ix]), y_tr_t[ix])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va_acc = float((net(X_va_t).argmax(1) == y_va_t).float().mean())
        if va_acc > best_acc + 1e-4:
            best_acc = va_acc; best_state = {k: v.clone() for k, v in net.state_dict().items()}; bad = 0
        else:
            bad += 1
        if bad >= 25: break
    net.load_state_dict(best_state)
    with torch.no_grad():
        pred = net(X_va_t).argmax(1).numpy()

    # Evaluate
    inv = {v: k for k, v in cell_to_idx.items()}
    dbytes, dzen, dms = [], [], []
    n_default, n_alt = 0, 0
    pick_dist = Counter()
    pick_v07_extras = 0  # count picks that use a v07-only knob combo
    for k_local, i in enumerate(va_idx):
        img, dist, cells, def_cell = raw[i]
        picked_cell = inv[pred[k_local]]
        if picked_cell not in cells:
            continue
        df = cells[def_cell]; pk = cells[picked_cell]
        dbytes.append(100 * (pk["bytes"] - df["bytes"]) / df["bytes"])
        dzen.append(pk["zensim"] - df["zensim"])
        dms.append(100 * (pk["ms"] - df["ms"]) / df["ms"])
        if picked_cell == def_cell: n_default += 1
        else: n_alt += 1
        pick_dist[picked_cell[0]] += 1  # by effort
        # Check if using v07-only knobs
        _, _, _, fs, mss, prog, gab, pat, lz, lf, pdl = picked_cell
        if (fs is not None or mss is not None or prog != "single"
            or gab is False or pat is True or lz is False
            or lf is True or pdl is False):
            pick_v07_extras += 1

    print(f"\n## Unified v06+v07 picker A/B")
    print(f"   train acc: {best_acc:.3f}")
    print(f"   n held-out cells: {len(dbytes)}")
    print(f"   default picks: {n_default} ({100*n_default/len(dbytes):.1f}%)")
    print(f"   alt picks:     {n_alt} ({100*n_alt/len(dbytes):.1f}%)")
    print(f"   alts using v07-only knobs: {pick_v07_extras} ({100*pick_v07_extras/len(dbytes):.1f}%)")
    print(f"   mean Δbytes:  {mean(dbytes):+.3f}%")
    print(f"   mean Δzensim: {mean(dzen):+.4f}pp")
    print(f"   mean Δms:     {mean(dms):+.2f}%")
    print(f"   pick effort distribution: {dict(pick_dist)}")

    with open(OUTPUT, "w") as f:
        f.write(f"# Unified v06+v07 picker — {len(rows)} cells, {len(all_imgs)} imgs, {n_holdout} holdout\n\n")
        f.write(f"- Cell taxonomy: {len(cell_to_idx)} cells (union of v06 effort×biters×ziters + v07 extras)\n")
        f.write(f"- Default cell: {default_cell}\n")
        f.write(f"- Architecture: 256x128 dropout=0.2, weight decay 1e-5\n")
        f.write(f"- Held-out 248 imgs (seed 7), {len(va_idx)} cells\n\n")
        f.write(f"## A/B vs default\n\n")
        f.write(f"- val acc: {best_acc:.3f}\n")
        f.write(f"- mean Δbytes: {mean(dbytes):+.3f}%\n")
        f.write(f"- mean Δzensim: {mean(dzen):+.4f}pp\n")
        f.write(f"- mean Δms: {mean(dms):+.2f}%\n")
        f.write(f"- default picks: {100*n_default/len(dbytes):.1f}%\n")
        f.write(f"- alt picks: {100*n_alt/len(dbytes):.1f}%\n")
        f.write(f"- alt picks using v07-only knobs: {100*pick_v07_extras/len(dbytes):.1f}%\n")
        f.write(f"- pick effort distribution: {dict(pick_dist)}\n")
    print(f"\n[wrote] {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
