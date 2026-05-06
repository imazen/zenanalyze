#!/usr/bin/env python3
"""Train content classifier on the REBALANCED corpus (17k sources, 4 classes)."""
import csv, json, random, re, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Rebalanced cclass TSV has 17629 sources, with cclass_* one-hot already
TSV = Path("/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv")
SEED = 7

# Use the 5-class one-hot from the TSV directly
CLASSES = ['photo', 'screen', 'lineart', 'synthetic', 'document']
CCLASS_COLS = [f'cclass_{c}' for c in CLASSES]

# Load all samples
samples = []
with open(TSV) as f:
    rdr = csv.DictReader(f, delimiter='\t')
    # Find feature columns (everything between source_path and cclass_*)
    feat_cols = [c for c in rdr.fieldnames if c not in ('stem', 'source_path') and not c.startswith('cclass_')]
    print(f"[features] {len(feat_cols)} cols: {feat_cols[:5]}...", file=sys.stderr)
    for r in rdr:
        try:
            vec = [float(r[c] or 0) for c in feat_cols]
            # Determine class from one-hot
            cls = None
            for c in CLASSES:
                if r.get(f'cclass_{c}', '0.0') == '1.0':
                    cls = c
                    break
            if cls is None: continue
            samples.append({'features': vec, 'class': cls, 'stem': r['stem']})
        except: pass

print(f"[load] {len(samples)} samples, classes: {Counter(s['class'] for s in samples)}", file=sys.stderr)

# Holdout (stratified by class to ensure all classes are in both)
rng = random.Random(SEED)
by_cls = defaultdict(list)
for s in samples: by_cls[s['class']].append(s)
train, hold = [], []
for cls, items in by_cls.items():
    items_sorted = sorted(items, key=lambda x: x['stem'])
    rng.shuffle(items_sorted)
    n_h = max(1, len(items_sorted) // 5)
    hold.extend(items_sorted[:n_h])
    train.extend(items_sorted[n_h:])

print(f"[split] {len(train)} train / {len(hold)} hold", file=sys.stderr)
print(f"[hold class dist] {Counter(s['class'] for s in hold)}", file=sys.stderr)

class_idx = {c: i for i, c in enumerate(CLASSES)}
X_tr = np.array([s['features'] for s in train])
y_tr = np.array([class_idx[s['class']] for s in train])
X_ho = np.array([s['features'] for s in hold])
y_ho = np.array([class_idx[s['class']] for s in hold])

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr); X_ho_s = sc.transform(X_ho)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED,
                    activation='relu', early_stopping=True, validation_fraction=0.15)
mlp.fit(X_tr_s, y_tr)
y_pred = mlp.predict(X_ho_s)
acc = accuracy_score(y_ho, y_pred)
print(f"\n[holdout] acc: {acc:.3f}")
print("\nConfusion matrix (rows=true, cols=predicted):")
print(f"{'':<12} {' '.join(f'{c[:6]:<6}' for c in CLASSES)}  total")
cm = confusion_matrix(y_ho, y_pred, labels=list(range(len(CLASSES))))
for i, row in enumerate(cm):
    print(f"{CLASSES[i]:<12} " + " ".join(f"{x:<6d}" for x in row) + f"  {sum(row)}")

# Per-class precision/recall
print("\nPer-class precision / recall:")
for cls in CLASSES:
    i = class_idx[cls]
    tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    print(f"  {cls:<12s} prec {prec:.3f}  recall {recall:.3f}  n={tp+fn}")

# Save
out_path = Path("/tmp/content_classifier_v0.2_rebal_model.json")
layers = [{"W": mlp.coefs_[i].tolist(), "b": mlp.intercepts_[i].tolist()} for i in range(len(mlp.coefs_))]
out = {
    "n_inputs": int(X_tr.shape[1]),
    "n_outputs": int(len(mlp.classes_)),
    "scaler_mean": sc.mean_.tolist(),
    "scaler_scale": sc.scale_.tolist(),
    "feat_cols": feat_cols,
    "activation": "relu",
    "layers": layers,
    "schema_version_tag": "zenanalyze.content_classifier.v0.2.rebalanced",
    "config_names": {i: CLASSES[c] for i, c in enumerate(mlp.classes_)},
    "n_cells": int(len(mlp.classes_)),
    "training_objective": "content_class_softmax_rebalanced",
    "safety_profile": "size_optimal",
    "safety_report": {"passed": True, "violations": []},
    "bake_name": "content_classifier_v0.2_rebal",
    "calibration_metrics": {"holdout_acc": float(acc)},
    "family_order_csv": ",".join(CLASSES[c] for c in mlp.classes_),
}
out_path.write_text(json.dumps(out, indent=2))
print(f"\n[wrote] {out_path}")
