#!/usr/bin/env python3
"""Re-run v06 zensim_mask_histgb champion + audit per-content-class behavior."""
from __future__ import annotations

# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: v06_champ_per_class.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v0.6 zenjxl champion / per-content-class audit (one-off recovery investigation).\n"
    "         Use: Per-class auditing is now covered by zentrain/tools/validate_schema.py + _picker_lib's load_or_build_dataset cache. v0.6 sweep data at /home/lilith/sweep-data/zenjxl_v06.tsv may not be staged.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

V06_TSV = Path("/home/lilith/sweep-data/zenjxl_v06.tsv")
FEATURES_TSV = Path("/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv")

DEFAULT_EFFORT = 7
ZENSIM_TOL = 0.05
SPEED_TOL = 1.05
BYTES_GAIN = 0.99
SEED = 7

def classify(stem):
    s = stem.lower()
    if "/synthetic/" in s or s.startswith("synth_") or "synthetic" in s and "synthetic" not in s.split('/')[0]: pass
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]): return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stock"]): return "lineart"
    if any(p in s for p in ["scan", "document", "invoice", "page-"]): return "document"
    return "photo"

def load_features():
    feats = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        feat_cols = [c for c in rdr.fieldnames if c.startswith('feat_')]
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in feat_cols]
                key = r['image_path'].rsplit('/', 1)[-1]
                feats[key] = (vec, classify(key))
            except Exception:
                continue
    return feats

def load_sweep():
    rows = []
    with open(V06_TSV) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            try:
                if not r['encoded_bytes'] or not r['score_zensim']: continue
                k = json.loads(r['knob_tuple_json'])
                rows.append({
                    'image': r['image_path'].rsplit('/', 1)[-1],
                    'effort': int(k['effort']),
                    'biters': int(k.get('butteraugli_iters', 0)),
                    'ziters': int(k.get('zensim_iters', 0)),
                    'distance': round(float(k['distance']), 4),
                    'bytes': int(r['encoded_bytes']),
                    'ms': float(r['encode_ms']),
                    'zensim': float(r['score_zensim']),
                })
            except Exception:
                continue
    return rows

def main():
    feats = load_features()
    rows = load_sweep()

    by_id = defaultdict(dict)
    for r in rows:
        by_id[(r['image'], r['distance'])][(r['effort'], r['biters'], r['ziters'])] = r

    cells = sorted({(r['effort'], r['biters'], r['ziters']) for r in rows})
    cell_idx = {c: i for i, c in enumerate(cells)}

    samples = []
    for (img, dist), d in by_id.items():
        if img not in feats: continue
        if (DEFAULT_EFFORT, 0, 0) not in d: continue
        default = d[(DEFAULT_EFFORT, 0, 0)]
        best_cell = (DEFAULT_EFFORT, 0, 0)
        best_bytes = default['bytes']
        for c, r in d.items():
            if c == (DEFAULT_EFFORT, 0, 0): continue
            if (r['bytes'] < default['bytes'] * BYTES_GAIN
                and r['ms'] <= default['ms'] * SPEED_TOL
                and r['zensim'] >= default['zensim'] - ZENSIM_TOL
                and r['bytes'] < best_bytes):
                best_bytes = r['bytes']
                best_cell = c
        vec, cls = feats[img]
        samples.append({
            'image': img, 'class': cls, 'features': vec, 'log_dist': np.log(dist),
            'best_cell_idx': cell_idx[best_cell],
            'cells_dict': d, 'default_bytes': default['bytes'],
            'default_ms': default['ms'], 'default_zensim': default['zensim'],
        })

    print(f"[samples] {len(samples)}, classes: {Counter(s['class'] for s in samples)}", file=sys.stderr)

    # Image-level holdout split
    import random
    rng = random.Random(SEED)
    by_cls_imgs = defaultdict(set)
    for s in samples:
        by_cls_imgs[s['class']].add(s['image'])
    hold_imgs = set()
    for cls, imgs in by_cls_imgs.items():
        imgs = sorted(imgs); rng.shuffle(imgs)
        hold_imgs.update(imgs[:max(1, len(imgs) // 5)])

    train = [s for s in samples if s['image'] not in hold_imgs]
    hold = [s for s in samples if s['image'] in hold_imgs]

    def make_xy(items):
        X = np.array([s['features'] + [s['log_dist']] for s in items])
        y = np.array([s['best_cell_idx'] for s in items])
        return X, y

    X_tr, y_tr = make_xy(train)
    X_ho, y_ho = make_xy(hold)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr); X_ho_s = sc.transform(X_ho)

    clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_ho_s)

    # Per-class breakdown
    print("\n## v06 zensim_mask_histgb per-class behavior on HOLDOUT")
    print(f"\n{'class':<12s} {'n':>5} {'Δbytes%':>10} {'Δzensim_pp':>12} {'Δms%':>10} {'def_pick%':>10}")
    by_cls = defaultdict(list)
    for s, p in zip(hold, y_pred):
        by_cls[s['class']].append((s, p))

    overall_dbytes_sum = 0
    overall_n = 0
    for cls in sorted(by_cls):
        items = by_cls[cls]
        n = len(items)
        total_default_b = total_picked_b = total_default_ms = total_picked_ms = 0
        zensim_deltas = []
        n_default_pick = 0
        for s, p in items:
            pick_cell = cells[p]
            if pick_cell == (DEFAULT_EFFORT, 0, 0):
                n_default_pick += 1
            r = s['cells_dict'].get(pick_cell, s['cells_dict'][(DEFAULT_EFFORT, 0, 0)])
            total_default_b += s['default_bytes']
            total_picked_b += r['bytes']
            total_default_ms += s['default_ms']
            total_picked_ms += r['ms']
            zensim_deltas.append(r['zensim'] - s['default_zensim'])
        dbytes = (total_picked_b - total_default_b) / total_default_b * 100
        dms = (total_picked_ms - total_default_ms) / total_default_ms * 100
        dzensim = np.mean(zensim_deltas)
        def_pct = 100 * n_default_pick / n
        print(f"{cls:<12s} {n:>5d} {dbytes:>+10.3f} {dzensim:>+12.4f} {dms:>+10.3f} {def_pct:>10.1f}")
        overall_dbytes_sum += dbytes * n
        overall_n += n
    print(f"\nweighted-avg Δbytes: {overall_dbytes_sum/overall_n:+.3f}%")

if __name__ == '__main__':
    main()
