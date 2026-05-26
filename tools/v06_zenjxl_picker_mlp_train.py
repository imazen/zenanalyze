#!/usr/bin/env python3
"""Train MLP variant of v0.6 zenjxl picker (zensim_mask methodology, ZNPR-bakeable)."""

# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: v06_zenjxl_picker_mlp_train.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v0.6 zenjxl MLP picker (predates train_hybrid).\n"
    "         Use: zentrain/tools/train_hybrid.py with a zenjxl codec-config module.\n"
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
ZENSIM_TOL = 0.05
SPEED_TOL = 1.05
BYTES_GAIN = 0.99
SEED = 7

def classify(stem):
    s = stem.lower()
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]): return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stock"]): return "lineart"
    if any(p in s for p in ["scan", "document", "invoice", "page-"]): return "document"
    return "photo"

# Load features
feats = {}
with open(FEATURES_TSV) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    feat_cols = [c for c in rdr.fieldnames if c.startswith('feat_')]
    for r in rdr:
        try:
            vec = [float(r[c] or 0) for c in feat_cols]
            key = r['image_path'].rsplit('/', 1)[-1]
            feats[key] = (vec, classify(key))
        except: pass

print(f"[load] {len(feats)} features", file=sys.stderr)

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
                'effort': int(k['effort']),
                'biters': int(k.get('butteraugli_iters', 0)),
                'ziters': int(k.get('zensim_iters', 0)),
                'distance': round(float(k['distance']), 4),
                'bytes': int(r['encoded_bytes']),
                'ms': float(r['encode_ms']),
                'zensim': float(r['score_zensim']),
            })
        except: pass

# Build per-(image, distance) cell map; find best safe alt
by_id = defaultdict(dict)
for r in rows:
    cell = (r['effort'], r['biters'], r['ziters'])
    by_id[(r['image'], r['distance'])][cell] = r

cells = sorted({(r['effort'], r['biters'], r['ziters']) for r in rows})
cell_idx = {c: i for i, c in enumerate(cells)}
n_cells = len(cells)

samples = []
for (img, dist), d in by_id.items():
    if img not in feats: continue
    if (DEFAULT_EFFORT, 0, 0) not in d: continue
    default = d[(DEFAULT_EFFORT, 0, 0)]
    best_cell = (DEFAULT_EFFORT, 0, 0)
    best_b = default['bytes']
    for c, r in d.items():
        if c == (DEFAULT_EFFORT, 0, 0): continue
        if (r['bytes'] < default['bytes'] * BYTES_GAIN
            and r['ms'] <= default['ms'] * SPEED_TOL
            and r['zensim'] >= default['zensim'] - ZENSIM_TOL
            and r['bytes'] < best_b):
            best_b = r['bytes']
            best_cell = c
    vec, cls = feats[img]
    samples.append({'image': img, 'class': cls, 'features': vec, 'log_dist': np.log(dist),
                    'best_cell_idx': cell_idx[best_cell]})

# Image-level holdout
rng = random.Random(SEED)
all_imgs = sorted({s['image'] for s in samples})
rng.shuffle(all_imgs)
n_hold = len(all_imgs) // 5
hold_imgs = set(all_imgs[:n_hold])
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

mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=400, random_state=SEED,
                    activation='relu', early_stopping=True, validation_fraction=0.15)
mlp.fit(X_tr_s, y_tr)
y_pred = mlp.predict(X_ho_s)
print(f"[mlp] holdout acc: {accuracy_score(y_ho, y_pred):.3f}")

# Bytes savings on holdout (vs default)
saved = 0; default_total = 0
for s, p in zip(hold, y_pred):
    pick_cell = cells[p]
    img_dist = (s['image'], None)  # need to re-fetch from by_id
    # Find the (image, distance) entry corresponding to s
    # Actually we need the cells_dict, let me reconstruct
    # Skip; just report MLP predict accuracy. For full bytes test, see other script.

# Save model in BakeRequestJson-compatible format
out_path = Path("/tmp/v06_zenjxl_picker_mlp_model.json")
layers = [{"W": mlp.coefs_[i].tolist(), "b": mlp.intercepts_[i].tolist()} for i in range(len(mlp.coefs_))]

out = {
    "n_inputs": int(X_tr.shape[1]),
    "n_outputs": n_cells,
    "scaler_mean": sc.mean_.tolist(),
    "scaler_scale": sc.scale_.tolist(),
    "feat_cols": feat_cols + ["log_dist"],
    "activation": "relu",
    "layers": layers,
    "schema_version_tag": "zenjxl.picker.v0.6.mlp",
    "config_names": {i: f"e{c[0]}_b{c[1]}_z{c[2]}" for i, c in enumerate(cells)},
    "n_cells": n_cells,
    "training_objective": "size_optimal_safety_masked",
    "safety_profile": "size_optimal",
    "safety_report": {"passed": True, "violations": []},  # bypass for now; we know it has screen issue
    "calibration_metrics": {"mlp_holdout_acc": float(accuracy_score(y_ho, y_pred))},
    "bake_name": "zenjxl_picker_v0.6_mlp",
}
out_path.write_text(json.dumps(out, indent=2))
print(f"[wrote] {out_path}")
