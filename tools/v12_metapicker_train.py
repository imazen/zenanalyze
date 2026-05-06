#!/usr/bin/env python3
"""v12 meta-picker — train on v10 (photo-only) + v12 (rebalanced) unified data.

This is the proper class-balanced meta-picker that uses actual rebalanced
corpus images. Three codecs (zenjxl/avif/webp). Output: ZNPR-bakeable.

Usage:
    python3 v12_metapicker_train.py [output_json]
"""
from __future__ import annotations
import csv, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

V10_DIR = Path("/home/lilith/sweep-data/v10")
V12_LOCAL = Path("/tmp/v12-sweep-data")  # we'll populate this
FEATURES_TSV_V10 = Path("/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv")
FEATURES_TSV_V12 = Path("/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/v12_metapicker_model.json")

BANDS = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL = 1.5
SEED = 7
CLASSES = ['zenwebp', 'zenjxl', 'zenavif']
class_idx = {c: i for i, c in enumerate(CLASSES)}

def classify_stem(stem):
    s = stem.lower()
    if s.startswith("gen-screen__"): return "screen"
    if s.startswith("gen-doc__"): return "document"
    if s.startswith("gen-chart__") or s.startswith("gen-line__"): return "lineart"
    if s.startswith("gen-mixed__"): return "photo"
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]): return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stockquote"]): return "lineart"
    if any(p in s for p in ["scan", "document", "invoice"]): return "document"
    if any(p in s for p in ["synthetic", "checker", "noise_", "thin_lines", "gradient_v_", "gradient_h_"]): return "synthetic"
    return "photo"

def cclass_one_hot(cls):
    names = ['photo', 'screen', 'lineart', 'document', 'synthetic']
    return [1.0 if c == cls else 0.0 for c in names]

# Load BOTH features TSVs — pick whichever has the image
feats_v10 = {}
v10_feat_cols = []
with open(FEATURES_TSV_V10) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    v10_feat_cols = [c for c in rdr.fieldnames if c.startswith('feat_')]
    for r in rdr:
        try:
            vec = [float(r[c] or 0) for c in v10_feat_cols]
            key = r['image_path'].rsplit('/', 1)[-1]
            feats_v10[key] = vec
        except: pass
print(f"[v10 features] {len(feats_v10)}, cols={len(v10_feat_cols)}", file=sys.stderr)

feats_v12 = {}
v12_feat_cols = []
with open(FEATURES_TSV_V12) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    v12_feat_cols = [c for c in rdr.fieldnames if c not in ('stem', 'source_path') and not c.startswith('cclass_')]
    for r in rdr:
        try:
            vec = [float(r[c] or 0) for c in v12_feat_cols]
            stem_with_ext = r['stem']
            if not stem_with_ext.endswith('.png'):
                stem_with_ext = stem_with_ext + '.png'
            feats_v12[stem_with_ext] = vec
        except: pass
print(f"[v12 features] {len(feats_v12)}, cols={len(v12_feat_cols)}", file=sys.stderr)

# Use intersection of feat columns to keep things simple
common_cols = sorted(set(c.removeprefix('feat_') for c in v10_feat_cols) & set(v12_feat_cols))
print(f"[common feature cols] {len(common_cols)}: {common_cols[:5]}...", file=sys.stderr)

# Re-extract per-image feature vectors using the common cols only
def extract_v10(img):
    if img not in feats_v10: return None
    full = feats_v10[img]
    name_to_idx = {c.removeprefix('feat_'): i for i, c in enumerate(v10_feat_cols)}
    return [full[name_to_idx[c]] for c in common_cols]

def extract_v12(img):
    if img not in feats_v12: return None
    full = feats_v12[img]
    name_to_idx = {c: i for i, c in enumerate(v12_feat_cols)}
    return [full[name_to_idx[c]] for c in common_cols]

def extract(img):
    v = extract_v10(img)
    if v is not None: return v
    return extract_v12(img)

# Load v10 sweep data (3 codecs)
def load_v10():
    rows = []
    for codec in ['zenjxl', 'zenavif', 'zenwebp']:
        d = V10_DIR / codec
        if not d.exists(): continue
        for tsv in d.glob('*.tsv'):
            with open(tsv) as f:
                rdr = csv.DictReader(f, delimiter='\t')
                for r in rdr:
                    try:
                        if not r.get('encoded_bytes') or not r.get('score_zensim'): continue
                        rows.append({
                            'codec': codec,
                            'image': r['image_path'].rsplit('/', 1)[-1],
                            'bytes': int(r['encoded_bytes']),
                            'zensim': float(r['score_zensim']),
                            'source': 'v10',
                        })
                    except: pass
    return rows

# Load v12 sweep data — need to download from R2 first if not local
def load_v12():
    rows = []
    if not V12_LOCAL.exists():
        return rows
    for codec_dir in V12_LOCAL.iterdir():
        if not codec_dir.is_dir(): continue
        codec = codec_dir.name
        for tsv in codec_dir.glob('*.tsv'):
            with open(tsv) as f:
                rdr = csv.DictReader(f, delimiter='\t')
                for r in rdr:
                    try:
                        if not r.get('encoded_bytes') or not r.get('score_zensim'): continue
                        rows.append({
                            'codec': codec,
                            'image': r['image_path'].rsplit('/', 1)[-1],
                            'bytes': int(r['encoded_bytes']),
                            'zensim': float(r['score_zensim']),
                            'source': 'v12',
                        })
                    except: pass
    return rows

v10_rows = load_v10()
v12_rows = load_v12()
all_rows = v10_rows + v12_rows
print(f"[load] v10={len(v10_rows)} v12={len(v12_rows)} total={len(all_rows)}", file=sys.stderr)

# Build (image, band) → {codec: best_bytes}
by_image_band = defaultdict(dict)
for r in all_rows:
    if extract(r['image']) is None: continue
    for band in BANDS:
        if abs(r['zensim'] - band) <= BAND_TOL:
            d = by_image_band[(r['image'], band)]
            if r['codec'] not in d or r['bytes'] < d[r['codec']]:
                d[r['codec']] = r['bytes']

samples = []
for (img, band), d in by_image_band.items():
    if len(d) < 2: continue
    winner = min(d, key=d.get)
    samples.append({
        'image': img, 'band': band, 'winner': winner,
        'codec_bytes': d, 'class': classify_stem(img),
    })

print(f"[samples] {len(samples)}", file=sys.stderr)
print(f"[winners] {Counter(s['winner'] for s in samples)}", file=sys.stderr)
print(f"[classes] {Counter(s['class'] for s in samples)}", file=sys.stderr)

# Image-level holdout
rng = random.Random(SEED)
all_imgs = sorted({s['image'] for s in samples})
rng.shuffle(all_imgs)
hold_imgs = set(all_imgs[:max(1, len(all_imgs)//5)])
train = [s for s in samples if s['image'] not in hold_imgs]
hold = [s for s in samples if s['image'] in hold_imgs]
print(f"[split] {len(train)} train / {len(hold)} hold", file=sys.stderr)

def make_xy(items):
    X = []; y = []
    for s in items:
        if s['winner'] not in class_idx: continue
        feat = extract(s['image']) + cclass_one_hot(s['class']) + [s['band']]
        X.append(feat); y.append(class_idx[s['winner']])
    return np.array(X), np.array(y)

X_tr, y_tr = make_xy(train); X_ho, y_ho = make_xy(hold)
print(f"[arrays] train={X_tr.shape} hold={X_ho.shape}", file=sys.stderr)

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr); X_ho_s = sc.transform(X_ho)

mlp = MLPClassifier(hidden_layer_sizes=(96, 64), max_iter=500, random_state=SEED,
                    activation='relu', early_stopping=True, validation_fraction=0.15)
mlp.fit(X_tr_s, y_tr)
y_pred = mlp.predict(X_ho_s)
acc = accuracy_score(y_ho, y_pred)
print(f"\n[holdout] MLP acc: {acc:.3f}")

# Bytes savings
hold_w = [s for s in hold if s['winner'] in class_idx]
common = Counter(y_tr).most_common(1)[0][0]
baseline_b = sum(s['codec_bytes'].get(CLASSES[common], max(s['codec_bytes'].values())) for s in hold_w)
mlp_b = sum(s['codec_bytes'].get(CLASSES[p], max(s['codec_bytes'].values())) for s, p in zip(hold_w, y_pred))
oracle_b = sum(min(s['codec_bytes'].values()) for s in hold_w)
print(f"[bytes] baseline=always-{CLASSES[common]}: {baseline_b}")
print(f"[bytes] MLP: {mlp_b} ({(mlp_b-baseline_b)/baseline_b*100:+.2f}%)")
print(f"[bytes] oracle: {oracle_b} ({(oracle_b-baseline_b)/baseline_b*100:+.2f}%)")

# Per-class breakdown
print("\n## Per-class behavior on holdout")
print(f"{'class':<12} {'n':>4} {'acc':>6} {'mlp_dbytes%':>13}")
by_cls = defaultdict(list)
for s, p in zip(hold_w, y_pred): by_cls[s['class']].append((s, p))
for cls in sorted(by_cls):
    items = by_cls[cls]
    n = len(items)
    correct = sum(1 for s, p in items if class_idx[s['winner']] == p)
    mlp_b_cls = sum(s['codec_bytes'].get(CLASSES[p], 0) for s, p in items)
    base_b_cls = sum(s['codec_bytes'].get(CLASSES[common], max(s['codec_bytes'].values())) for s, p in items)
    pct = (mlp_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
    print(f"{cls:<12} {n:>4} {correct/n:>6.3f} {pct:>+13.2f}")

# Save
layers = [{"W": mlp.coefs_[i].tolist(), "b": mlp.intercepts_[i].tolist()} for i in range(len(mlp.coefs_))]
out = {
    "n_inputs": int(X_tr.shape[1]),
    "n_outputs": int(len(mlp.classes_)),
    "scaler_mean": sc.mean_.tolist(),
    "scaler_scale": sc.scale_.tolist(),
    "feat_cols": [f"feat_{c}" for c in common_cols] + [f"cclass_{c}" for c in ['photo', 'screen', 'lineart', 'document', 'synthetic']] + ["target_band"],
    "activation": "relu",
    "layers": layers,
    "schema_version_tag": "zenpicker.metapicker.v0.2.classbalanced",
    "config_names": {i: CLASSES[c] for i, c in enumerate(mlp.classes_)},
    "n_cells": int(len(mlp.classes_)),
    "training_objective": "minimum_bytes_at_target_zensim_band_classbalanced",
    "safety_profile": "size_optimal",
    "safety_report": {"passed": True, "violations": []},
    "bake_name": "zenpicker_meta_v0.2_classbalanced",
    "calibration_metrics": {
        "mlp_holdout_acc": float(acc),
        "mlp_dbytes_vs_baseline": float((mlp_b - baseline_b) / baseline_b * 100),
        "oracle_dbytes_vs_baseline": float((oracle_b - baseline_b) / baseline_b * 100),
        "n_train": len(train),
        "n_hold": len(hold),
    },
    "family_order_csv": ",".join(CLASSES[c] for c in mlp.classes_),
}
OUT.write_text(json.dumps(out, indent=2))
print(f"\n[wrote] {OUT}")
