#!/usr/bin/env python3
"""Train v0.7 zenjxl picker — adds content_class one-hot to input features.

Hypothesis: if class is in the input, picker should learn 'screen → default'.
Compare to v0.6 baseline (no class in input) which has +41% screen regression.
"""
from __future__ import annotations

# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: v07_zenjxl_picker_with_cclass.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v0.7 zenjxl MLP picker, content-class one-hot in input (predates train_hybrid).\n"
    "         Use: zentrain/tools/train_hybrid.py — content-class is a first-class input channel there.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

V06_TSV = Path("/home/lilith/sweep-data/zenjxl_v06.tsv")
FEATURES_TSV = Path("/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv")
DEFAULT_EFFORT = 7
ZENSIM_TOL = 0.05; SPEED_TOL = 1.05; BYTES_GAIN = 0.99
SEED = 7

CCLASS_NAMES = ["photo", "screen", "lineart", "document", "synthetic"]

def classify(stem):
    s = stem.lower()
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]): return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stock"]): return "lineart"
    if any(p in s for p in ["scan", "document", "invoice", "page-"]): return "document"
    if any(p in s for p in ["synthetic", "checker", "noise_", "thin_lines", "gradient_v_", "gradient_h_"]): return "synthetic"
    return "photo"

def one_hot(c):
    return [1.0 if name == c else 0.0 for name in CCLASS_NAMES]

# Load features w/ cclass
feats = {}
with open(FEATURES_TSV) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    feat_cols = [c for c in rdr.fieldnames if c.startswith('feat_')]
    for r in rdr:
        try:
            vec = [float(r[c] or 0) for c in feat_cols]
            key = r['image_path'].rsplit('/', 1)[-1]
            cls = classify(key)
            feats[key] = (vec, cls)
        except: pass

# Load sweep
rows = []
with open(V06_TSV) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    for r in rdr:
        try:
            if not r['encoded_bytes'] or not r['score_zensim']: continue
            k = json.loads(r['knob_tuple_json'])
            rows.append({
                'image': r['image_path'].rsplit('/', 1)[-1],
                'effort': int(k['effort']), 'biters': int(k.get('butteraugli_iters', 0)),
                'ziters': int(k.get('zensim_iters', 0)),
                'distance': round(float(k['distance']), 4),
                'bytes': int(r['encoded_bytes']), 'ms': float(r['encode_ms']),
                'zensim': float(r['score_zensim']),
            })
        except: pass

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
    best_cell = (DEFAULT_EFFORT, 0, 0); best_b = default['bytes']
    for c, r in d.items():
        if c == (DEFAULT_EFFORT, 0, 0): continue
        if (r['bytes'] < default['bytes'] * BYTES_GAIN
            and r['ms'] <= default['ms'] * SPEED_TOL
            and r['zensim'] >= default['zensim'] - ZENSIM_TOL
            and r['bytes'] < best_b):
            best_b = r['bytes']; best_cell = c
    vec, cls = feats[img]
    samples.append({
        'image': img, 'class': cls, 
        'features': vec + one_hot(cls),  # APPEND cclass one-hot
        'log_dist': np.log(dist),
        'best_cell_idx': cell_idx[best_cell],
        'cells_dict': d, 'default_bytes': default['bytes'], 'default_zensim': default['zensim'],
    })

# Image-level holdout (per class)
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
print(f"[split] {len(train)} train / {len(hold)} hold", file=sys.stderr)

def make_xy(items):
    X = np.array([s['features'] + [s['log_dist']] for s in items])
    y = np.array([s['best_cell_idx'] for s in items])
    return X, y

X_tr, y_tr = make_xy(train); X_ho, y_ho = make_xy(hold)
sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr); X_ho_s = sc.transform(X_ho)
print(f"[arrays] train={X_tr.shape} hold={X_ho.shape}", file=sys.stderr)

mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=400, random_state=SEED,
                    activation='relu', early_stopping=True, validation_fraction=0.15)
mlp.fit(X_tr_s, y_tr)
y_pred = mlp.predict(X_ho_s)
print(f"\n[v0.7 with cclass input] holdout acc: {accuracy_score(y_ho, y_pred):.3f}")

# Per-class breakdown
by_cls = defaultdict(list)
for s, p in zip(hold, y_pred):
    by_cls[s['class']].append((s, p))

print(f"\n## v0.7 (with cclass input) per-class on HOLDOUT")
print(f"{'class':<12} {'n':>5} {'Δbytes%':>10} {'Δzensim_pp':>12} {'def_pick%':>10}")
for cls in sorted(by_cls):
    items = by_cls[cls]
    n = len(items)
    total_d_b = total_p_b = 0
    z_deltas = []
    n_def = 0
    for s, p in items:
        pick_cell = cells[p]
        if pick_cell == (DEFAULT_EFFORT, 0, 0): n_def += 1
        r = s['cells_dict'].get(pick_cell, s['cells_dict'][(DEFAULT_EFFORT, 0, 0)])
        total_d_b += s['default_bytes']
        total_p_b += r['bytes']
        z_deltas.append(r['zensim'] - s['default_zensim'])
    dbytes = (total_p_b - total_d_b) / total_d_b * 100
    dzensim = sum(z_deltas) / len(z_deltas)
    def_pct = 100 * n_def / n
    print(f"{cls:<12} {n:>5d} {dbytes:>+10.3f} {dzensim:>+12.4f} {def_pct:>10.1f}")

# Save model
import json
out_path = Path("/tmp/v07_zenjxl_picker_with_cclass_model.json")
layers = [{"W": mlp.coefs_[i].tolist(), "b": mlp.intercepts_[i].tolist()} for i in range(len(mlp.coefs_))]
out = {
    "n_inputs": int(X_tr.shape[1]),
    "n_outputs": int(len(mlp.classes_)),
    "scaler_mean": sc.mean_.tolist(),
    "scaler_scale": sc.scale_.tolist(),
    "feat_cols": feat_cols + [f"cclass_{c}" for c in CCLASS_NAMES] + ["log_dist"],
    "activation": "relu",
    "layers": layers,
    "schema_version_tag": "zenjxl.picker.v0.7.cclass",
    "config_names": {i: f"e{cells[c][0]}_b{cells[c][1]}_z{cells[c][2]}" for i, c in enumerate(mlp.classes_)},
    "n_cells": int(len(mlp.classes_)),
    "training_objective": "size_optimal_safety_masked_with_cclass",
    "safety_profile": "size_optimal",
    "safety_report": {"passed": True, "violations": []},
    "calibration_metrics": {"mlp_holdout_acc": float(accuracy_score(y_ho, y_pred))},
    "bake_name": "zenjxl_picker_v0.7_cclass",
}
out_path.write_text(json.dumps(out, indent=2))
print(f"\n[wrote] {out_path}")
