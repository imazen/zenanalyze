#!/usr/bin/env python3
"""
Feature ablation for the Pareto controller's HistGB model.

Goal: rank zenanalyze features by their contribution to picker
accuracy, then drop the ones whose accuracy contribution doesn't
justify their compute cost on the hot path.

Method:
1. Train baseline HistGB (all 19 features) — measure mean argmin
   overhead.
2. For each feature, retrain with that feature dropped (replaced
   with constant 0). Measure overhead delta.
3. Forward greedy: starting from the empty set, repeatedly add the
   feature whose inclusion gives the largest overhead reduction.
   Record the Pareto curve (n_features kept vs overhead).

The forward greedy is the actionable output: it tells us "with K
features, here's the best K and here's the overhead." User picks K
based on per-feature compute cost in zenanalyze.

Inputs:
- benchmarks/zq_pareto_2026-04-29.tsv
- benchmarks/zq_pareto_features_2026-04-29.tsv

Outputs:
- benchmarks/zq_feature_ablation_2026-04-29.log
- benchmarks/zq_feature_ablation_2026-04-29.json
"""

import argparse
import csv
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# Defaults — overridden by `--codec-config <module>` (recommended) so the
# script is codec-agnostic in shape. Module-level placeholders keep the
# helpers below readable.
PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")
OUT_LOG = Path("benchmarks/zq_feature_ablation_2026-04-29.log")
OUT_JSON = Path("benchmarks/zq_feature_ablation_2026-04-29.json")

ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))


def load_codec_config(name: str):
    """Bind the codec config module's PARETO/FEATURES/ZQ_TARGETS to
    this script's module-level names. The OUT_LOG / OUT_JSON paths
    derive from the codec's OUT_LOG basename (suffix `_ablation`)
    so ablation artifacts ship next to the trained model."""
    global PARETO, FEATURES, OUT_LOG, OUT_JSON, ZQ_TARGETS
    mod = importlib.import_module(name)
    PARETO = Path(mod.PARETO)
    FEATURES = Path(mod.FEATURES)
    ZQ_TARGETS = list(mod.ZQ_TARGETS)
    base = Path(mod.OUT_LOG)
    OUT_LOG = base.with_name(base.stem + "_ablation" + base.suffix)
    OUT_JSON = base.with_name(base.stem + "_ablation.json")
    return mod
SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

# Light HistGB tuned for speed across many ablation points. Quality
# drops by ~1-2pp vs the production fit (which uses max_iter=400,
# max_depth=8) but the *ranking* across feature ablations is stable
# — same features end up most/least useful at lower compute. Total
# wall-time at 100/4: ~50 min for baseline + 19 LOO. Forward-greedy
# phase skipped — LOO ranking is enough to identify which features
# can be dropped, and forward greedy adds another ~120 min.
HISTGB_KW = dict(
    max_iter=100,
    max_depth=4,
    learning_rate=0.1,
    l2_regularization=0.5,
    random_state=SEED,
)
# Skip forward greedy if True (LOO is sufficient for the ranking).
SKIP_GREEDY = True

CONFIG_NAMES: dict = {}


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
        cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        for r in rdr:
            feats[(r["image_path"], r["size_class"])] = np.array(
                [float(r[c]) for c in cols], dtype=np.float32
            )
    return feats, cols


def build_dataset(pareto, feats, feat_cols):
    """X: (n, 19 + 4 + 1) = 19 features + size onehot + zq.

    Y: log-bytes per config; np.nan if config doesn't reach zq.
    meta: (image, size, zq) per row.
    """
    n_configs = max(CONFIG_NAMES) + 1
    X_rows, Y_rows, meta = [], [], []
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
            x = np.concatenate([f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)])
            y = np.full(n_configs, np.nan, dtype=np.float32)
            for cfg, b in per_cfg[zq].items():
                if b > 0 and not math.isinf(b):
                    y[cfg] = math.log(b)
            X_rows.append(x)
            Y_rows.append(y)
            meta.append((image, size, zq))
    return np.stack(X_rows), np.stack(Y_rows), meta


def evaluate_argmin(Y_pred, Y_actual, meta, mask):
    n = Y_pred.shape[0]
    overheads = []
    correct = 0
    per_zq = defaultdict(list)
    for i in range(n):
        actual = Y_actual[i]
        pred = Y_pred[i]
        m = (~np.isnan(actual)) & mask
        if not np.any(m):
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        actual_best = int(np.argmin(ab))
        pred_best = int(np.argmin(pb))
        if pred_best == actual_best:
            correct += 1
        ov = (ab[pred_best] - ab[actual_best]) / ab[actual_best]
        overheads.append(ov)
        per_zq[meta[i][2]].append(ov)
    if not overheads:
        return None
    arr = np.array(overheads)
    return {
        "n": int(len(arr)),
        "argmin_acc": correct / len(arr),
        "mean_pct": float(100 * arr.mean()),
        "p50_pct": float(100 * np.percentile(arr, 50)),
        "p90_pct": float(100 * np.percentile(arr, 90)),
    }


def train_all_configs(X_tr, Y_tr, n_configs):
    """Train one HistGB per config, return list of estimators (None for
    cells with too few rows). Used by permutation-importance ablation."""
    teachers = [None] * n_configs
    for cfg in range(n_configs):
        mask = ~np.isnan(Y_tr[:, cfg])
        if mask.sum() < 50:
            continue
        gbm = HistGradientBoostingRegressor(**HISTGB_KW)
        gbm.fit(X_tr[mask], Y_tr[mask, cfg])
        teachers[cfg] = gbm
    return teachers


def predict_with_teachers(teachers, X, Y_tr_for_fallback):
    """Predict per-config log-bytes using `teachers`; columns whose
    teacher is None get filled with the per-column nanmean of Y_tr."""
    n_configs = len(teachers)
    Y_pred = np.zeros((X.shape[0], n_configs), dtype=np.float32)
    for cfg, t in enumerate(teachers):
        if t is None:
            fill = float(np.nanmean(Y_tr_for_fallback[:, cfg])) if n_configs else 0.0
            if math.isnan(fill):
                fill = 0.0
            Y_pred[:, cfg] = fill
        else:
            Y_pred[:, cfg] = t.predict(X)
    return Y_pred


def permutation_importance_analysis(
    teachers,
    X_va,
    Y_va,
    Y_tr,
    meta_va,
    feat_cols,
    n_configs,
    seed=SEED,
    n_repeats=3,
):
    """Permutation feature importance for the Pareto controller.

    Train teachers once (caller does this), then for each feature
    permute that column on the validation set, run inference, record
    the mean-overhead delta vs the unpermuted baseline. ~50× faster
    than retrain-LOO and the ranking is well-known to track LOO
    closely on tree ensembles.

    Returns:
      baseline (dict) — metrics on unpermuted X_va
      loo_results (dict[name → metrics]) — same shape as the
        retrain-LOO output so downstream code is unchanged. Each
        entry's `mean_pct` is the *permuted* mean overhead (i.e.
        what the model produces when that feature is shuffled).
    """
    rng = np.random.default_rng(seed)
    Y_pred_base = predict_with_teachers(teachers, X_va, Y_tr)
    baseline = evaluate_argmin(Y_pred_base, Y_va, meta_va, np.ones(n_configs, dtype=bool))

    n_feat = len(feat_cols)
    loo_results = {}
    for fi in range(n_feat):
        # Average across n_repeats permutations to denoise.
        ovs, accs = [], []
        for _ in range(n_repeats):
            X_perm = X_va.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            Y_pred = predict_with_teachers(teachers, X_perm, Y_tr)
            m = evaluate_argmin(Y_pred, Y_va, meta_va, np.ones(n_configs, dtype=bool))
            if m is None:
                continue
            ovs.append(m["mean_pct"])
            accs.append(m["argmin_acc"])
        if not ovs:
            loo_results[feat_cols[fi]] = baseline
            continue
        loo_results[feat_cols[fi]] = {
            "n": baseline["n"],
            "argmin_acc": float(np.mean(accs)),
            "mean_pct": float(np.mean(ovs)),
            "p50_pct": baseline["p50_pct"],
            "p90_pct": baseline["p90_pct"],
        }
    return baseline, loo_results


def train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask_cols, n_configs):
    """Train HistGB per-config using only the columns where mask_cols is True.

    Returns metrics dict.
    """
    X_tr_m = X_tr[:, mask_cols]
    X_va_m = X_va[:, mask_cols]

    Y_pred = np.zeros_like(Y_va)
    for cfg in range(n_configs):
        mask = ~np.isnan(Y_tr[:, cfg])
        if mask.sum() < 50:
            Y_pred[:, cfg] = np.nanmean(Y_tr[:, cfg]) if mask.any() else 0.0
            continue
        gbm = HistGradientBoostingRegressor(**HISTGB_KW)
        gbm.fit(X_tr_m[mask], Y_tr[mask, cfg])
        Y_pred[:, cfg] = gbm.predict(X_va_m)

    return evaluate_argmin(Y_pred, Y_va, meta_va, np.ones(n_configs, dtype=bool))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec-config",
        default=None,
        help="Python module name exporting PARETO/FEATURES/OUT_*/ZQ_TARGETS. "
        "Same contract as train_hybrid.py. Without this flag the script falls "
        "back to the legacy hardcoded paths.",
    )
    parser.add_argument(
        "--method",
        choices=["loo", "permutation"],
        default="permutation",
        help="Importance estimation method. `permutation` (default) trains the "
        "baseline once then shuffles each feature column on the val set — "
        "~50× faster than retrain-LOO with near-identical ranking. `loo` "
        "retrains the model with each feature dropped (legacy; ground truth "
        "for ranking but slow).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Permutation repeats per feature (averaged). Default 3 — high "
        "enough to denoise small datasets without inflating wall-time.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any feature has Δ < --max-negative-delta-pp. "
        "Auto-enabled when the CI environment variable is set. A feature whose "
        "Δ is *negative* under permutation importance — i.e., shuffling that "
        "feature *improves* the picker — is noise the model overfit; ranking "
        "negatives indicates schema needs pruning before ship.",
    )
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Override --strict / CI for intentionally noisy features.",
    )
    parser.add_argument(
        "--max-negative-delta-pp",
        type=float,
        default=-0.05,
        help="Threshold (in pp) below which a Δ counts as a strict-mode "
        "violation. Default -0.05pp — features that improve the picker by "
        "more than 0.05pp when shuffled are flagged. Set to e.g. -0.20 for "
        "looser tolerance.",
    )
    args = parser.parse_args()
    is_ci = bool(os.environ.get("CI"))
    strict = (args.strict or is_ci) and not args.allow_unsafe
    if args.codec_config:
        load_codec_config(args.codec_config)
    sys.stderr.write(f"Loading {PARETO}...\n")
    sys.stderr.write(f"        {FEATURES}\n")
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features\n")

    X, Y, meta = build_dataset(pareto, feats, feat_cols)
    n_configs = Y.shape[1]
    sys.stderr.write(f"Decision rows: {len(X)} × {n_configs} configs\n")

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(train_idx)}, val rows: {len(val_idx)}\n")

    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]
    meta_va = [meta[i] for i in val_idx]

    n_feat = len(feat_cols)
    n_total = X.shape[1]  # n_feat + 4 (size onehot) + 1 (zq)
    feat_indices = list(range(n_feat))  # we ablate features only; keep size+zq
    fixed_indices = list(range(n_feat, n_total))

    # ============ Phase 1: baseline (all features) ============
    if args.method == "permutation":
        sys.stderr.write(f"\n[1/3] Baseline + per-config teachers (all {n_feat} features)...\n")
        teachers = train_all_configs(X_tr, Y_tr, n_configs)
        baseline_pred = predict_with_teachers(teachers, X_va, Y_tr)
        baseline = evaluate_argmin(
            baseline_pred, Y_va, meta_va, np.ones(n_configs, dtype=bool)
        )
        sys.stderr.write(
            f"  baseline: mean overhead {baseline['mean_pct']:.2f}% "
            f"argmin_acc {baseline['argmin_acc']:.1%}\n"
        )

        sys.stderr.write(
            f"\n[2/3] Permutation importance across {n_feat} features "
            f"({args.n_repeats} repeats each)...\n"
        )
        _, loo_results = permutation_importance_analysis(
            teachers,
            X_va,
            Y_va,
            Y_tr,
            meta_va,
            feat_cols,
            n_configs,
            seed=SEED,
            n_repeats=args.n_repeats,
        )
        for fi in range(n_feat):
            m = loo_results[feat_cols[fi]]
            delta = m["mean_pct"] - baseline["mean_pct"]
            sys.stderr.write(
                f"  permute {feat_cols[fi]:35s}: {m['mean_pct']:.2f}% "
                f"(Δ {delta:+.2f}pp)\n"
            )
    else:
        sys.stderr.write(f"\n[1/3] Baseline (all {n_feat} features)...\n")
        baseline_mask = np.ones(n_total, dtype=bool)
        baseline = train_and_eval(
            X_tr, Y_tr, X_va, Y_va, meta_va, baseline_mask, n_configs
        )
        sys.stderr.write(
            f"  baseline: mean overhead {baseline['mean_pct']:.2f}% "
            f"argmin_acc {baseline['argmin_acc']:.1%}\n"
        )

        sys.stderr.write(f"\n[2/3] Leave-one-out across {n_feat} features...\n")
        loo_results = {}
        for fi in range(n_feat):
            mask = np.ones(n_total, dtype=bool)
            mask[fi] = False
            m = train_and_eval(
                X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs
            )
            loo_results[feat_cols[fi]] = m
            delta = m["mean_pct"] - baseline["mean_pct"]
            sys.stderr.write(
                f"  drop {feat_cols[fi]:35s}: {m['mean_pct']:.2f}% "
                f"(Δ {delta:+.2f}pp)\n"
            )

    # Sort by Δ (largest hurt = most important).
    loo_sorted = sorted(loo_results.items(), key=lambda kv: -kv[1]["mean_pct"])
    sys.stderr.write("\n  Most important (drop hurts most):\n")
    for name, m in loo_sorted[:5]:
        sys.stderr.write(f"    {name}: drop → {m['mean_pct']:.2f}% (Δ {m['mean_pct'] - baseline['mean_pct']:+.2f}pp)\n")
    sys.stderr.write("  Least important (drop hurts least):\n")
    for name, m in loo_sorted[-5:]:
        sys.stderr.write(f"    {name}: drop → {m['mean_pct']:.2f}% (Δ {m['mean_pct'] - baseline['mean_pct']:+.2f}pp)\n")

    # ============ Strict-gate: flag negative-Δ features ============
    # A feature whose permutation Δ is negative is *noise the model
    # overfit*. Listing them is a 100% should-be-dropped signal that
    # the schema needs pruning before ship.
    negatives = []
    for name, m in loo_sorted:
        delta = m["mean_pct"] - baseline["mean_pct"]
        if delta < args.max_negative_delta_pp:
            negatives.append({"feature": name, "delta_pp": delta})
    if negatives:
        sys.stderr.write(
            "\n" + "=" * 70 + "\n"
            f"  ⚠ {len(negatives)} FEATURE(S) IMPROVE THE PICKER WHEN SHUFFLED — overfit signal\n"
            + "=" * 70 + "\n"
        )
        for n in negatives:
            sys.stderr.write(
                f"  • {n['feature']:35s} Δ {n['delta_pp']:+.2f}pp\n"
            )
        sys.stderr.write(
            "=" * 70 + "\n"
            "  Drop these from KEEP_FEATURES and retrain. Threshold: "
            f"{args.max_negative_delta_pp:+.2f}pp.\n"
        )
    else:
        sys.stderr.write("\n✓ No features improve the picker when shuffled.\n")

    # ============ Phase 3: forward greedy (optional) ============
    greedy_curve = []
    if SKIP_GREEDY:
        sys.stderr.write("\n[3/3] Forward greedy SKIPPED (LOO ranking is sufficient).\n")
        run_greedy = False
    else:
        sys.stderr.write("\n[3/3] Forward greedy (add the feature that helps most until plateau)...\n")
        run_greedy = True
        selected = set()
        # K=0 baseline (no features, just size onehot + zq).
        mask0 = np.zeros(n_total, dtype=bool)
        mask0[fixed_indices] = True
        m0 = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask0, n_configs)
        greedy_curve.append({"k": 0, "selected": [], "metrics": m0})
        sys.stderr.write(f"  K=0 (size+zq only): {m0['mean_pct']:.2f}%\n")

        candidate_pool = [feat_cols.index(name) for name, _ in loo_sorted[:12]]
        candidate_pool_set = set(candidate_pool)

    while run_greedy and candidate_pool_set:
        best_fi = None
        best_metrics = None
        for fi in candidate_pool_set:
            mask = np.zeros(n_total, dtype=bool)
            mask[fixed_indices] = True
            for s in selected:
                mask[s] = True
            mask[fi] = True
            m = train_and_eval(X_tr, Y_tr, X_va, Y_va, meta_va, mask, n_configs)
            if best_metrics is None or m["mean_pct"] < best_metrics["mean_pct"]:
                best_fi = fi
                best_metrics = m
        selected.add(best_fi)
        candidate_pool_set.remove(best_fi)
        greedy_curve.append(
            {
                "k": len(selected),
                "selected": [feat_cols[i] for i in sorted(selected)],
                "added": feat_cols[best_fi],
                "metrics": best_metrics,
            }
        )
        sys.stderr.write(
            f"  K={len(selected):2d} (+{feat_cols[best_fi]:30s}): {best_metrics['mean_pct']:.2f}% "
            f"argmin_acc {best_metrics['argmin_acc']:.1%}\n"
        )

    # ============ Persist ============
    out = {
        "baseline": baseline,
        "leave_one_out": loo_results,
        "leave_one_out_ranked": [
            {"feature": name, "mean_pct": m["mean_pct"], "delta_pp": m["mean_pct"] - baseline["mean_pct"]}
            for name, m in loo_sorted
        ],
        "greedy_curve": greedy_curve,
        "feat_cols": feat_cols,
        "histgb_params": HISTGB_KW,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    lines = []

    def w(s):
        lines.append(s)
        sys.stderr.write(s + "\n")

    w("\n# Feature ablation — Pareto controller HistGB")
    w(f"Train rows: {len(X_tr)}, val rows: {len(X_va)}")
    w(f"HistGB: {HISTGB_KW}")
    w("")
    w(f"## Baseline (all {n_feat} features)")
    w(f"mean overhead: {baseline['mean_pct']:.2f}%  argmin_acc: {baseline['argmin_acc']:.1%}")
    w("")
    w("## Leave-one-out (sorted by Δ vs baseline, largest hurt first)")
    w(f"{'feature':40s} {'overhead':>10s} {'Δ vs base':>10s}")
    w("-" * 65)
    for name, m in loo_sorted:
        delta = m["mean_pct"] - baseline["mean_pct"]
        w(f"{name:40s} {m['mean_pct']:>9.2f}%  {delta:>+8.2f}pp")
    w("")
    w("## Forward greedy curve (cumulative best feature subsets)")
    w(f"{'K':>3s} {'overhead':>10s} {'argmin_acc':>11s}  added")
    w("-" * 65)
    for entry in greedy_curve:
        m = entry["metrics"]
        added = entry.get("added", "(none)")
        w(f"{entry['k']:>3d} {m['mean_pct']:>9.2f}%  {m['argmin_acc']:>10.1%}   +{added}")

    OUT_LOG.write_text("\n".join(lines))

    # Strict-gate exit. Logs already written for inspection; only the
    # exit code signals failure.
    if negatives and strict:
        sys.stderr.write(
            f"\nstrict mode: exiting 1 with {len(negatives)} negative-Δ feature(s). "
            "Re-run with --allow-unsafe to override.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
