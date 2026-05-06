#!/usr/bin/env python3
"""End-to-end Pareto simulation:
  v06 default → v0.7b picker → with per-class encoder rule (patches/gaborish from v08)
  
For each (image, distance) on holdout:
  - default cell:    effort=7, biters=0, ziters=0, patches=False, gaborish=True (jxl-encoder defaults)
  - v0.7b pick:      MLP picks (effort, biters, ziters); per-class rule fixes (patches, gaborish)
  - v0.6 picker:     baseline for comparison
  
Compare bytes / zensim / encode time to default.
"""
import csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

V06_TSV = Path("/home/lilith/sweep-data/zenjxl_v06.tsv")
V08_DIR = Path("/home/lilith/sweep-data/v08")
FEATURES_TSV = Path("/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv")
SEED = 7
DEFAULT_EFFORT = 7

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

# Load v06 sweep
v06_rows = []
with open(V06_TSV) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    for r in rdr:
        try:
            if not r['encoded_bytes'] or not r['score_zensim']: continue
            k = json.loads(r['knob_tuple_json'])
            v06_rows.append({
                'image': r['image_path'].rsplit('/', 1)[-1],
                'effort': int(k['effort']), 'biters': int(k.get('butteraugli_iters', 0)),
                'ziters': int(k.get('zensim_iters', 0)),
                'distance': round(float(k['distance']), 4),
                'bytes': int(r['encoded_bytes']), 'ms': float(r['encode_ms']),
                'zensim': float(r['score_zensim']),
            })
        except: pass

# Load v08 sweep (has patches/gaborish/pdl)
v08_rows = []
for chunk in V08_DIR.glob('*.tsv'):
    with open(chunk) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            try:
                if not r['encoded_bytes'] or not r['score_zensim']: continue
                k = json.loads(r['knob_tuple_json'])
                v08_rows.append({
                    'image': r['image_path'].rsplit('/', 1)[-1],
                    'effort': int(k['effort']), 'biters': int(k.get('butteraugli_iters', 0)),
                    'distance': round(float(k['distance']), 4),
                    'patches': k.get('patches'), 'gaborish': k.get('gaborish'),
                    'pdl': k.get('pixel_domain_loss'),
                    'bytes': int(r['encoded_bytes']), 'ms': float(r['encode_ms']),
                    'zensim': float(r['score_zensim']),
                })
            except: pass

# Build v06 cell map: (image, distance) → {(eff, biters, ziters): row}
v06_by_id = defaultdict(dict)
for r in v06_rows:
    v06_by_id[(r['image'], r['distance'])][(r['effort'], r['biters'], r['ziters'])] = r

# Build v08 cell map: (image, distance, effort, biters) → {(patches, gaborish, pdl): row}
v08_by_id = defaultdict(dict)
for r in v08_rows:
    v08_by_id[(r['image'], r['distance'], r['effort'], r['biters'])][(r['patches'], r['gaborish'], r['pdl'])] = r

# Train v0.7b (excluding screen)
import random
rng = random.Random(SEED)
samples = []
ZENSIM_TOL = 0.05; SPEED_TOL = 1.05; BYTES_GAIN = 0.99
v06_cells = sorted({(r['effort'], r['biters'], r['ziters']) for r in v06_rows})
cell_idx = {c: i for i, c in enumerate(v06_cells)}

for (img, dist), d in v06_by_id.items():
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
        'features': vec + one_hot(cls),
        'log_dist': np.log(dist),
        'best_cell_idx': cell_idx[best_cell],
        'cells_dict': d,
        'distance': dist,
    })

by_cls_imgs = defaultdict(set)
for s in samples: by_cls_imgs[s['class']].add(s['image'])
hold_imgs = set()
for cls, imgs in by_cls_imgs.items():
    imgs = sorted(imgs); rng.shuffle(imgs)
    hold_imgs.update(imgs[:max(1, len(imgs) // 5)])
train_nonscreen = [s for s in samples if s['image'] not in hold_imgs and s['class'] != 'screen']
hold = [s for s in samples if s['image'] in hold_imgs]

X_tr = np.array([s['features'] + [s['log_dist']] for s in train_nonscreen])
y_tr = np.array([s['best_cell_idx'] for s in train_nonscreen])
sc = StandardScaler().fit(X_tr)
mlp = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=400, random_state=SEED,
                    activation='relu', early_stopping=True, validation_fraction=0.15)
mlp.fit(sc.transform(X_tr), y_tr)

# Per-class encoder rule
def encoder_rule(cls):
    if cls in ('screen', 'synthetic'):
        return (True, False, False)  # patches=T, gaborish=F, pdl=F
    return (False, True, False)  # default jxl-encoder

# Simulate: for each holdout cell, compute bytes/zensim under 3 strategies
strategies = {
    'default': lambda s: s['cells_dict'][(DEFAULT_EFFORT, 0, 0)],
    'v0.7b_only': None,  # special handling below
    'v0.7b+encoder_rule': None,
}

# Per-class bookkeeping
per_class_default_b = defaultdict(int)
per_class_default_z = defaultdict(list)
per_class_v07b_b = defaultdict(int)
per_class_v07b_z = defaultdict(list)
per_class_v07b_rule_b = defaultdict(int)
per_class_v07b_rule_z = defaultdict(list)
per_class_n = defaultdict(int)

for s in hold:
    cls = s['class']
    per_class_n[cls] += 1
    
    # default
    d = s['cells_dict'][(DEFAULT_EFFORT, 0, 0)]
    per_class_default_b[cls] += d['bytes']
    per_class_default_z[cls].append(d['zensim'])
    
    # v0.7b: gate screen → default
    if cls == 'screen':
        v07b_pick = (DEFAULT_EFFORT, 0, 0)
    else:
        X = sc.transform(np.array([s['features'] + [s['log_dist']]]))
        idx = mlp.predict(X)[0]
        v07b_pick = v06_cells[idx]
    v07b_row = s['cells_dict'].get(v07b_pick, d)
    per_class_v07b_b[cls] += v07b_row['bytes']
    per_class_v07b_z[cls].append(v07b_row['zensim'])
    
    # v0.7b + encoder rule (using v08 data for patches/gaborish)
    e, bi, zi = v07b_pick
    rule = encoder_rule(cls)
    v08_lookup = (s['image'], s['distance'], e, bi)
    if v08_lookup in v08_by_id and rule in v08_by_id[v08_lookup]:
        rule_row = v08_by_id[v08_lookup][rule]
        per_class_v07b_rule_b[cls] += rule_row['bytes']
        per_class_v07b_rule_z[cls].append(rule_row['zensim'])
    else:
        # Fall back to v0.7b row if v08 data unavailable
        per_class_v07b_rule_b[cls] += v07b_row['bytes']
        per_class_v07b_rule_z[cls].append(v07b_row['zensim'])

print(f"\n## End-to-end Pareto simulation — v0.6 default vs v0.7b vs v0.7b+rule\n")
print(f"{'class':<12} {'n':>5} {'default_b':>12} {'v0.7b Δ%':>10} {'v0.7b+rule Δ%':>14}")
totals = {'default_b': 0, 'v07b_b': 0, 'v07b_rule_b': 0, 'n': 0}
for cls in sorted(per_class_n):
    n = per_class_n[cls]
    d_b = per_class_default_b[cls]
    v_b = per_class_v07b_b[cls]
    r_b = per_class_v07b_rule_b[cls]
    v_pct = (v_b - d_b) / d_b * 100 if d_b else 0
    r_pct = (r_b - d_b) / d_b * 100 if d_b else 0
    print(f"{cls:<12} {n:>5d} {d_b:>12d} {v_pct:>+9.3f}% {r_pct:>+13.3f}%")
    totals['default_b'] += d_b; totals['v07b_b'] += v_b; totals['v07b_rule_b'] += r_b; totals['n'] += n
print()
v_o = (totals['v07b_b'] - totals['default_b']) / totals['default_b'] * 100
r_o = (totals['v07b_rule_b'] - totals['default_b']) / totals['default_b'] * 100
print(f"{'OVERALL':<12} {totals['n']:>5d} {totals['default_b']:>12d} {v_o:>+9.3f}% {r_o:>+13.3f}%")
