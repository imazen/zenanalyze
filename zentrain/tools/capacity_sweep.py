#!/usr/bin/env python3
"""
MLP capacity + cross-term sweep on the 8-feature reduced schema.

The v1.1 student takes +1.0pp vs the 19-feature v1.0 student because
reducing 19→8 features also halves the engineered cross-term count,
dropping MLP input dim 48→26. This script searches for an
architecture+feature recipe that closes the gap without re-adding
zenanalyze tiers.

Method:
1. Train HistGB teacher *once* on simple inputs (8 raw feats + size
   onehot + log_pixels + zq). Cache its predictions. This is the
   ceiling: the teacher's quality is what the student is trying to
   match.
2. Iterate over student variants (architecture × cross-term recipe)
   on the same cached teacher targets. Train, evaluate, record.
3. Report: variant comparison table, best variant, gap to teacher.

Each student fit is ~2-5 min vs the teacher's ~10 min (run once).
Total wall-time ~30-45 min for ~10 variants.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")
OUT_LOG = Path("benchmarks/zq_distill_capacity_sweep_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_distill_capacity_sweep_2026-04-29.json")
CACHE = Path("/tmp/zq_capacity_sweep_teacher_cache.npz")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

KEEP_FEATURES = [
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
]

CONFIG_NAMES: dict = {}


# ---------- Data loading (same as distill scripts) ----------


def load_pareto(path):
    rows = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                cid = int(r["config_id"])
                bytes_v = int(r["bytes"])
                zensim_v = float(r["zensim"])
            except (ValueError, KeyError):
                continue
            CONFIG_NAMES.setdefault(cid, r["config_name"])
            key = (r["image_path"], r["size_class"], int(r["width"]), int(r["height"]))
            rows[key].append({"config_id": cid, "bytes": bytes_v, "zensim": zensim_v})
    return rows


def load_features(path):
    feats = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        cols = [c for c in KEEP_FEATURES if c in all_cols]
        for r in rdr:
            feats[(r["image_path"], r["size_class"])] = np.array(
                [float(r[c]) for c in cols], dtype=np.float32
            )
    return feats, cols


# ---------- Cross-term recipes (the knobs we vary) ----------


def make_simple_features(f, log_px, size_oh, zq_norm):
    """Simple input vector — what the teacher trains on (14 dims)."""
    return np.concatenate([f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)])


def make_engineered_v1(f, log_px, size_oh, zq_norm):
    """v1.1 (current) recipe: 8 + 4 + 5 + 8 + 1 = 26 dims."""
    return np.concatenate([
        f,
        size_oh,
        np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px], dtype=np.float32),
        zq_norm * f,
        np.array([0.0], dtype=np.float32),  # icc placeholder
    ])


def make_engineered_v2_zqsq(f, log_px, size_oh, zq_norm):
    """v2: add zq²×feat cross terms. 26 + 8 = 34 dims."""
    return np.concatenate([
        f,
        size_oh,
        np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px], dtype=np.float32),
        zq_norm * f,
        zq_norm * zq_norm * f,
        np.array([0.0], dtype=np.float32),
    ])


def make_engineered_v3_logpx(f, log_px, size_oh, zq_norm):
    """v3: add log_pixels×feat cross terms. 26 + 8 = 34 dims."""
    return np.concatenate([
        f,
        size_oh,
        np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px], dtype=np.float32),
        zq_norm * f,
        log_px * f,
        np.array([0.0], dtype=np.float32),
    ])


def make_engineered_v4_full(f, log_px, size_oh, zq_norm):
    """v4: zq²×feat AND log_px×feat. 26 + 16 = 42 dims."""
    return np.concatenate([
        f,
        size_oh,
        np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px], dtype=np.float32),
        zq_norm * f,
        zq_norm * zq_norm * f,
        log_px * f,
        np.array([0.0], dtype=np.float32),
    ])


def make_engineered_v5_pairs(f, log_px, size_oh, zq_norm):
    """v5: zq×feat plus a small set of feat×feat pairs (the most
    informative pairs from a quick correlation guess: cb×cr, hf×entropy,
    var×edge). 26 + 3 = 29 dims."""
    pairs = np.array(
        [
            f[4] * f[5],  # cb_sharpness × cr_sharpness
            f[6] * f[7],  # hf_energy_ratio × luma_entropy
            f[0] * f[1],  # variance × edge_density
        ],
        dtype=np.float32,
    )
    return np.concatenate([
        f,
        size_oh,
        np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px], dtype=np.float32),
        zq_norm * f,
        pairs,
        np.array([0.0], dtype=np.float32),
    ])


CROSS_RECIPES = {
    "v1_baseline": make_engineered_v1,
    "v2_zqsq": make_engineered_v2_zqsq,
    "v3_logpx": make_engineered_v3_logpx,
    "v4_full": make_engineered_v4_full,
    "v5_pairs": make_engineered_v5_pairs,
}


# ---------- Architecture variants ----------

ARCHITECTURES = {
    "h64x2":   (64, 64),
    "h96x2":   (96, 96),
    "h128x2":  (128, 128),
    "h64x3":   (64, 64, 64),
    "h96x3":   (96, 96, 96),
}


# ---------- Build datasets per recipe ----------


def build_dataset_for_recipe(pareto, feats, recipe_fn):
    n_configs = max(CONFIG_NAMES) + 1
    Xs_rows, Xe_rows, Y_rows, meta = [], [], [], []
    for (image, size, w, h), samples in pareto.items():
        feat_key = (image, size)
        if feat_key not in feats:
            continue
        f = feats[feat_key]
        log_px = math.log(max(1, w * h))
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        size_oh[SIZE_INDEX[size]] = 1.0

        per_cfg = defaultdict(lambda: defaultdict(lambda: math.inf))
        for s in samples:
            for zq in ZQ_TARGETS:
                if s["zensim"] >= zq and s["bytes"] < per_cfg[zq][s["config_id"]]:
                    per_cfg[zq][s["config_id"]] = s["bytes"]

        for zq in ZQ_TARGETS:
            if not per_cfg[zq]:
                continue
            zq_norm = zq / 100.0
            xs = make_simple_features(f, log_px, size_oh, zq_norm)
            xe = recipe_fn(f, log_px, size_oh, zq_norm)
            y = np.full(n_configs, np.nan, dtype=np.float32)
            for cfg, b in per_cfg[zq].items():
                if b > 0 and not math.isinf(b):
                    y[cfg] = math.log(b)
            Xs_rows.append(xs)
            Xe_rows.append(xe)
            Y_rows.append(y)
            meta.append((image, size, zq))
    return np.stack(Xs_rows), np.stack(Xe_rows), np.stack(Y_rows), meta


def evaluate(Y_pred, Y_actual, meta):
    overheads, correct = [], 0
    for i in range(Y_pred.shape[0]):
        actual = Y_actual[i]
        pred = Y_pred[i]
        m = ~np.isnan(actual)
        if not np.any(m):
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        a = int(np.argmin(ab))
        p = int(np.argmin(pb))
        if p == a:
            correct += 1
        overheads.append((ab[p] - ab[a]) / ab[a])
    arr = np.array(overheads)
    return {
        "n": int(len(arr)),
        "argmin_acc": correct / len(arr),
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
    }


# ---------- Teacher (run once, cached) ----------


def train_teacher(Xs_tr, Y_tr, Xs_va, n_configs):
    sys.stderr.write(f"\nTraining teacher on simple inputs (dim={Xs_tr.shape[1]})...\n")
    teachers = []
    cfg_means = np.nanmean(Y_tr, axis=0)
    for cfg in range(n_configs):
        mask = ~np.isnan(Y_tr[:, cfg])
        if mask.sum() < 50:
            teachers.append(None)
            continue
        gbm = HistGradientBoostingRegressor(
            max_iter=400, max_depth=8, learning_rate=0.05,
            l2_regularization=0.5, random_state=SEED,
        )
        gbm.fit(Xs_tr[mask], Y_tr[mask, cfg])
        teachers.append(gbm)
        if (cfg + 1) % 30 == 0:
            sys.stderr.write(f"  {cfg+1}/{n_configs} teachers trained\n")

    soft_tr = np.zeros((Xs_tr.shape[0], n_configs), dtype=np.float32)
    soft_va = np.zeros((Xs_va.shape[0], n_configs), dtype=np.float32)
    for cfg in range(n_configs):
        if teachers[cfg] is None:
            soft_tr[:, cfg] = cfg_means[cfg]
            soft_va[:, cfg] = cfg_means[cfg]
        else:
            soft_tr[:, cfg] = teachers[cfg].predict(Xs_tr)
            soft_va[:, cfg] = teachers[cfg].predict(Xs_va)
    return soft_tr, soft_va


# ---------- Student fit ----------


def fit_student(Xe_tr, soft_tr, Xe_va, hidden_layers, label):
    sys.stderr.write(f"\nStudent {label}: arch={hidden_layers}, dim={Xe_tr.shape[1]}\n")
    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    student = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=2e-3,
        batch_size=512,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-6,
        random_state=SEED,
        verbose=False,
    )
    student.fit(Xe_tr_s, soft_tr)
    Y_va_pred = student.predict(Xe_va_s)
    n_params = sum(c.size + i.size for c, i in zip(student.coefs_, student.intercepts_))
    return Y_va_pred, n_params, student.n_iter_, student.loss_


# ---------- Main ----------


def main():
    sys.stderr.write(f"Loading {PARETO}...\n")
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features\n")

    # Build the train/val split once on the v1 recipe; the split
    # depends only on image identity, not feature engineering.
    Xs, Xe_v1, Y, meta = build_dataset_for_recipe(pareto, feats, make_engineered_v1)
    n_configs = Y.shape[1]

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(train_idx)}, val rows: {len(val_idx)}\n")
    meta_va = [meta[i] for i in val_idx]

    Xs_tr, Xs_va = Xs[train_idx], Xs[val_idx]
    Y_tr, Y_va = Y[train_idx], Y[val_idx]

    # Teacher: train once or load from cache.
    if CACHE.exists():
        sys.stderr.write(f"\nLoading teacher predictions from cache: {CACHE}\n")
        cached = np.load(CACHE)
        soft_tr = cached["soft_tr"]
        soft_va = cached["soft_va"]
        if soft_tr.shape != (Xs_tr.shape[0], n_configs):
            sys.stderr.write(f"  cache shape mismatch — retraining\n")
            soft_tr, soft_va = train_teacher(Xs_tr, Y_tr, Xs_va, n_configs)
            np.savez(CACHE, soft_tr=soft_tr, soft_va=soft_va)
    else:
        soft_tr, soft_va = train_teacher(Xs_tr, Y_tr, Xs_va, n_configs)
        np.savez(CACHE, soft_tr=soft_tr, soft_va=soft_va)
        sys.stderr.write(f"Cached teacher predictions to {CACHE}\n")

    teacher_metrics = evaluate(soft_va, Y_va, meta_va)
    sys.stderr.write(
        f"\nTeacher (ceiling): mean {teacher_metrics['mean_pct']:.2f}% "
        f"argmin {teacher_metrics['argmin_acc']:.1%}\n"
    )

    # Sweep students.
    results = []
    for recipe_name, recipe_fn in CROSS_RECIPES.items():
        # Rebuild Xe with this recipe (same train/val split).
        _, Xe_recipe, _, _ = build_dataset_for_recipe(pareto, feats, recipe_fn)
        Xe_tr_r, Xe_va_r = Xe_recipe[train_idx], Xe_recipe[val_idx]

        for arch_name, arch in ARCHITECTURES.items():
            label = f"{recipe_name}/{arch_name}"
            try:
                Y_pred, n_params, n_iter, loss = fit_student(
                    Xe_tr_r, soft_tr, Xe_va_r, arch, label
                )
                m = evaluate(Y_pred, Y_va, meta_va)
                results.append({
                    "label": label,
                    "recipe": recipe_name,
                    "arch": arch_name,
                    "input_dim": Xe_recipe.shape[1],
                    "hidden": list(arch),
                    "n_params": n_params,
                    "kb_f32": round(n_params * 4 / 1024, 1),
                    "kb_f16": round(n_params * 2 / 1024, 1),
                    "n_iter": n_iter,
                    "loss": float(loss),
                    "metrics": m,
                })
                sys.stderr.write(
                    f"  {label}: mean {m['mean_pct']:.2f}% argmin {m['argmin_acc']:.1%}  "
                    f"params={n_params} ({n_params*2/1024:.1f}KB f16)\n"
                )
            except Exception as e:
                sys.stderr.write(f"  {label}: FAILED — {e}\n")

    # Sort by mean overhead.
    results.sort(key=lambda r: r["metrics"]["mean_pct"])

    out = {
        "teacher": teacher_metrics,
        "students": results,
        "n_configs": n_configs,
        "feat_cols": feat_cols,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = []
    def w(s):
        lines.append(s)
        sys.stderr.write(s + "\n")

    w("\n# MLP capacity + cross-term sweep — 8-feature reduced schema")
    w(f"Train rows: {len(train_idx)}, val rows: {len(val_idx)}")
    w(f"Teacher (HistGB ceiling): mean {teacher_metrics['mean_pct']:.2f}%  "
      f"argmin {teacher_metrics['argmin_acc']:.1%}")
    w(f"Existing v1.1 student: mean 8.20% argmin 10.5% (h64x2, recipe=v1_baseline, 13688 params)")
    w("")
    w("## Student variants (sorted by mean overhead, best first)")
    w(f"{'rank':>4s} {'recipe/arch':25s} {'in_dim':>7s} {'params':>8s} {'kb_f16':>7s} "
      f"{'mean':>7s} {'p50':>7s} {'p90':>7s} {'argmin':>7s}")
    w("-" * 95)
    for rank, r in enumerate(results, 1):
        m = r["metrics"]
        w(f"{rank:>4d} {r['label']:25s} {r['input_dim']:>7d} {r['n_params']:>8d} "
          f"{r['kb_f16']:>6.1f} K {m['mean_pct']:>6.2f}% {m['p50_pct']:>6.2f}% "
          f"{m['p90_pct']:>6.2f}% {m['argmin_acc']:>6.1%}")

    if results:
        best = results[0]
        gap_to_teacher = best["metrics"]["mean_pct"] - teacher_metrics["mean_pct"]
        w(f"\n## Best: {best['label']}")
        w(f"  mean overhead: {best['metrics']['mean_pct']:.2f}% (gap to teacher: {gap_to_teacher:+.2f}pp)")
        w(f"  argmin_acc:    {best['metrics']['argmin_acc']:.1%}")
        w(f"  size:          {best['kb_f16']} KB f16, {best['kb_f32']} KB f32")
        w(f"  arch:          {best['hidden']}, input_dim={best['input_dim']}")

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
