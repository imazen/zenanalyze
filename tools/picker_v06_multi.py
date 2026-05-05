#!/usr/bin/env python3
"""Multi-target picker training for v06+ sweep data.

Trains 6 picker variants in parallel:
  - target=zensim      mask=on   arch=mlp_classifier
  - target=zensim      mask=on   arch=histgb_classifier
  - target=butter_max  mask=on   arch=mlp_classifier
  - target=butter_p3   mask=on   arch=mlp_classifier
  - target=multi       mask=on   arch=mlp_classifier   (zensim + butter weighted)
  - target=zensim      mask=off  arch=mlp_classifier   (control)

For each: train teacher labels per (image, distance) cell, train classifier
on (features ⊕ log distance), evaluate on held-out 248-image split via
table lookup against the actual sweep cells.

Reports: per-variant Δbytes vs effort=7 default, Δ-target-metric, Δms,
pick distribution, classification accuracy. Best one wins.
"""
from __future__ import annotations
import csv, json, math, random, sys
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, median

import numpy as np

# Default args; override via sys.argv[1] = sweep TSV path
SWEEP_TSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/lilith/sweep-data/zenjxl_v06.tsv")
FEATURES_TSV = Path(
    "/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv"
)
OUTPUT_REPORT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/picker_v06_multi_report.md")

DEFAULT_EFFORT = 7
EFFORTS = [1, 3, 5, 7, 9]
EFFORT_TO_IDX = {e: i for i, e in enumerate(EFFORTS)}

# Default safety constraints (used in mask=on variants)
ZENSIM_TOL = 0.05  # pp; allow tiny regression vs default
SPEED_TOL = 1.05   # ratio; allow 5% slower
BYTES_GAIN = 0.99  # require >=1% byte savings

VARIANTS = [
    # (label, target_metric, mask_on, arch, target_direction)
    # target_direction: "higher" means higher is better (zensim); "lower" means lower is better (butter)
    ("zensim_mask_mlp",       "zensim",            True,  "mlp",     "higher"),
    ("zensim_mask_histgb",    "zensim",            True,  "histgb",  "higher"),
    ("butter_max_mask_mlp",   "butteraugli_max",   True,  "mlp",     "lower"),
    ("butter_p3_mask_mlp",    "butteraugli_pnorm3",True,  "mlp",     "lower"),
    ("multi_mask_mlp",        "multi",             True,  "mlp",     "mixed"),  # weighted zensim + butter
    ("zensim_nomask_mlp",     "zensim",            False, "mlp",     "higher"),
]


def load_features():
    feats = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        keep = [c for c in rdr.fieldnames if c.startswith("feat_")]
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in keep]
                feats[r["image_path"]] = vec
            except Exception:
                continue
    return feats, keep


def load_sweep():
    """Load sweep TSV. Returns rows with all available metric columns."""
    rows = []
    with open(SWEEP_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        # Detect available metric columns
        metric_cols = [c for c in rdr.fieldnames if c.startswith("score_")]
        print(f"[sweep] metric cols available: {metric_cols}", file=sys.stderr)
        for r in rdr:
            try:
                if not r["encoded_bytes"] or not r["score_zensim"]:
                    continue
                k = json.loads(r["knob_tuple_json"])
                if k.get("noise") is True:
                    continue
                row = {
                    "image": r["image_path"].rsplit("/", 1)[-1],
                    "distance": round(float(k["distance"]), 4),
                    "effort": int(k["effort"]),
                    "biters": int(k.get("butteraugli_iters", 0)),
                    "ziters": int(k.get("zensim_iters", 0)),
                    "bytes": int(r["encoded_bytes"]),
                    "ms": float(r["encode_ms"]),
                }
                for c in metric_cols:
                    if r.get(c):
                        try:
                            row[c.replace("score_", "")] = float(r[c])
                        except ValueError:
                            pass
                rows.append(row)
            except Exception:
                continue
    return rows


def is_safe(c, default, target, target_dir):
    """Safe alternative if: bytes >=1% smaller AND target not regressed AND speed within 5% of default."""
    if c["bytes"] >= default["bytes"] * BYTES_GAIN:
        return False
    if c["ms"] > default["ms"] * SPEED_TOL:
        return False
    if target == "multi":
        # Composite: zensim must not regress + butter must not regress
        z_ok = c.get("zensim", -1e9) >= default.get("zensim", -1e9) - ZENSIM_TOL
        b_ok = c.get("butteraugli_max", 1e9) <= default.get("butteraugli_max", 1e9) * 1.02
        return z_ok and b_ok
    val = c.get(target, None)
    def_val = default.get(target, None)
    if val is None or def_val is None:
        return True  # no info: trust the bytes/speed gate
    if target_dir == "higher":
        return val >= def_val - ZENSIM_TOL
    else:
        return val <= def_val * 1.02  # 2% tolerance for butter


def alt_better(c1, c2, target, target_dir):
    """Is c1 a 'better' alt than c2 (smaller bytes, ties broken by target)?"""
    if c1["bytes"] != c2["bytes"]:
        return c1["bytes"] < c2["bytes"]
    if target == "multi":
        return c1.get("zensim", 0) > c2.get("zensim", 0)
    val1 = c1.get(target, 0); val2 = c2.get(target, 0)
    return (val1 > val2) if target_dir == "higher" else (val1 < val2)


def build_dataset(feats, sweep_rows, target, target_dir, mask_on):
    """Per (image, distance), assign label = best safe-alt cell index (or default).

    Cell = (effort, biters, ziters) — flat tuple. Build cell index from data.
    """
    by_id = defaultdict(dict)
    for r in sweep_rows:
        cell = (r["effort"], r["biters"], r["ziters"])
        by_id[(r["image"], r["distance"])][cell] = r

    # Build cell index
    all_cells = sorted({c for cells in by_id.values() for c in cells.keys()})
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    default_cell = (DEFAULT_EFFORT, 0, 0)
    if default_cell not in cell_to_idx:
        # fall back to (default_effort, anything)
        candidates = [c for c in all_cells if c[0] == DEFAULT_EFFORT]
        default_cell = candidates[0] if candidates else all_cells[0]
    print(f"[teacher] target={target} mask={mask_on} default={default_cell} n_cells={len(all_cells)}", file=sys.stderr)

    X, y, raw, image_for_row = [], [], [], []
    for (img, dist), cells in by_id.items():
        if img not in feats or default_cell not in cells:
            continue
        default = cells[default_cell]
        # Pick best label
        best_label = cell_to_idx[default_cell]
        best_cell_data = default
        if mask_on:
            for cell, data in cells.items():
                if cell == default_cell:
                    continue
                if is_safe(data, default, target, target_dir):
                    if alt_better(data, best_cell_data, target, target_dir):
                        best_cell_data = data
                        best_label = cell_to_idx[cell]
        else:
            # No mask: pick min bytes cell that doesn't regress target by 0.05/2%
            for cell, data in cells.items():
                if cell == default_cell:
                    continue
                if data["bytes"] >= best_cell_data["bytes"]:
                    continue
                # No safety mask; just check target sanity
                val = data.get(target if target != "multi" else "zensim")
                def_val = default.get(target if target != "multi" else "zensim")
                if val is None or def_val is None:
                    continue
                if target_dir == "higher" and val < def_val - 5.0:  # major regress
                    continue
                if target_dir == "lower" and val > def_val * 1.5:  # major regress
                    continue
                best_cell_data = data
                best_label = cell_to_idx[cell]

        x = feats[img] + [math.log(max(dist, 0.01))]
        X.append(x)
        y.append(best_label)
        raw.append((img, dist, cells, default_cell))
        image_for_row.append(img)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), raw, image_for_row, cell_to_idx, default_cell


def train_mlp_classifier(X_tr, y_tr, X_va, y_va, n_classes):
    import torch, torch.nn as nn
    torch.manual_seed(7)

    n_in = X_tr.shape[1]
    net = nn.Sequential(
        nn.Linear(n_in, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )

    label_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    cw = (label_counts.sum() / (label_counts + 1.0)).clip(0.5, 5.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(cw))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    X_tr_t = torch.from_numpy(X_tr); y_tr_t = torch.from_numpy(y_tr)
    X_va_t = torch.from_numpy(X_va); y_va_t = torch.from_numpy(y_va)
    n = X_tr_t.shape[0]

    best_va = 0.0
    best_state = None
    bad = 0
    for epoch in range(80):
        perm = torch.randperm(n)
        net.train()
        for s in range(0, n, 512):
            ix = perm[s:s+512]
            opt.zero_grad()
            loss = loss_fn(net(X_tr_t[ix]), y_tr_t[ix])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va_acc = float((net(X_va_t).argmax(1) == y_va_t).float().mean())
        if va_acc > best_va + 1e-4:
            best_va = va_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 20: break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(X_va_t).argmax(1).numpy(), best_va


def train_histgb_classifier(X_tr, y_tr, X_va, y_va, n_classes):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05,
        random_state=7, class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    acc = float((pred == y_va).mean())
    return pred, acc


def evaluate(pred, raw, val_idx, image_for_row, holdout, target, default_cell, cell_to_idx):
    """Compute Δbytes, Δtarget, Δms vs default cell across held-out (img, distance) cells."""
    inv_cell = {v: k for k, v in cell_to_idx.items()}
    dbytes, dtarget, dms = [], [], []
    n_default_pick = 0
    n_alt_pick = 0
    pick_dist = Counter()
    for k_local, i in enumerate(val_idx):
        img, dist, cells, def_cell = raw[i]
        picked_idx = pred[k_local]
        picked_cell = inv_cell[picked_idx]
        if picked_cell not in cells or def_cell not in cells:
            continue
        b_def = cells[def_cell]["bytes"]; ms_def = cells[def_cell]["ms"]
        t_def = cells[def_cell].get(target if target != "multi" else "zensim")
        b = cells[picked_cell]["bytes"]; ms = cells[picked_cell]["ms"]
        t = cells[picked_cell].get(target if target != "multi" else "zensim")
        dbytes.append(100 * (b - b_def) / b_def)
        dms.append(100 * (ms - ms_def) / ms_def)
        if t is not None and t_def is not None:
            dtarget.append(t - t_def)
        if picked_cell == def_cell:
            n_default_pick += 1
        else:
            n_alt_pick += 1
        pick_dist[picked_cell[0]] += 1  # by effort
    return {
        "n": len(dbytes),
        "n_default_pick": n_default_pick,
        "n_alt_pick": n_alt_pick,
        "mean_dbytes_pct": mean(dbytes) if dbytes else 0,
        "mean_dtarget": mean(dtarget) if dtarget else 0,
        "mean_dms_pct": mean(dms) if dms else 0,
        "pick_dist_by_effort": dict(pick_dist),
    }


def main():
    feats, _ = load_features()
    print(f"[features] {len(feats)} images", file=sys.stderr)

    sweep_rows = load_sweep()
    print(f"[sweep] {len(sweep_rows)} cells", file=sys.stderr)

    # Image-level holdout split
    rng = random.Random(7)
    all_imgs = sorted({r["image"] for r in sweep_rows if r["image"] in feats})
    rng.shuffle(all_imgs)
    n_holdout = int(len(all_imgs) * 0.2)
    holdout = set(all_imgs[:n_holdout])
    print(f"[split] {len(all_imgs)} images, {n_holdout} holdout", file=sys.stderr)

    report_lines = ["# v06 picker variant comparison\n\n"]
    report_lines.append(f"- {len(sweep_rows)} cells, {len(all_imgs)} images, {n_holdout} held out (seed 7)\n")
    report_lines.append(f"- Default cell: (effort={DEFAULT_EFFORT}, biters=0, ziters=0), noise=False\n")
    report_lines.append(f"- Safety mask: bytes <99% AND target not regressed AND ms <=105%\n\n")
    report_lines.append("| variant | acc | n | mean Δbytes | mean Δtarget | mean Δms | default % | top picks |\n")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")

    for label, target, mask_on, arch, target_dir in VARIANTS:
        try:
            X, y, raw, image_for_row, cell_to_idx, default_cell = build_dataset(
                feats, sweep_rows, target, target_dir, mask_on,
            )
            n_cells = len(cell_to_idx)
            tr_idx = np.array([i for i, m in enumerate(image_for_row) if m not in holdout])
            va_idx = np.array([i for i, m in enumerate(image_for_row) if m in holdout])
            if len(tr_idx) == 0 or len(va_idx) == 0:
                report_lines.append(f"| {label} | n/a | 0 | n/a | n/a | n/a | n/a | empty split |\n")
                continue
            mu = X[tr_idx].mean(axis=0)
            sd = X[tr_idx].std(axis=0) + 1e-9
            X_n = (X - mu) / sd

            if arch == "mlp":
                pred, acc = train_mlp_classifier(X_n[tr_idx], y[tr_idx], X_n[va_idx], y[va_idx], n_cells)
            else:
                pred, acc = train_histgb_classifier(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], n_cells)

            r = evaluate(pred, raw, va_idx, image_for_row, holdout, target, default_cell, cell_to_idx)
            top_picks = ", ".join(f"e{e}={n}" for e, n in sorted(r["pick_dist_by_effort"].items(), key=lambda x: -x[1])[:5])
            default_pct = 100 * r["n_default_pick"] / r["n"] if r["n"] else 0
            report_lines.append(
                f"| {label} | {acc:.3f} | {r['n']} | {r['mean_dbytes_pct']:+.3f}% | "
                f"{r['mean_dtarget']:+.4f} | {r['mean_dms_pct']:+.2f}% | {default_pct:.1f}% | {top_picks} |\n"
            )
            print(f"\n=== {label} ===", file=sys.stderr)
            print(f"  acc={acc:.3f} dbytes={r['mean_dbytes_pct']:+.3f}% dtarget={r['mean_dtarget']:+.4f} dms={r['mean_dms_pct']:+.2f}%", file=sys.stderr)
        except Exception as e:
            report_lines.append(f"| {label} | n/a | n/a | ERROR | {e} | n/a | n/a | n/a |\n")
            print(f"  ERROR: {e}", file=sys.stderr)

    with open(OUTPUT_REPORT, "w") as f:
        f.writelines(report_lines)
    print(f"\n[wrote] {OUTPUT_REPORT}", file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
