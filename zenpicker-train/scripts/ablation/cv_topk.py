#!/usr/bin/env python3
"""K-fold (by image) cross-validation of the zenjpeg picker feature ablation.

Extends the single-split `ablation.py` study to a CV-confirmed top-K. The
2026-06-01 ablation found pruning to ~20-40 source features REDUCES held-out
byte-overhead, but on ONE image split — a point estimate. This harness:

  1. Partitions the *distinct images* into K folds (deterministic, sorted).
  2. For each fold f (held out as val, the other K-1 folds as train):
       - fit the full-108 per-cell HistGB teacher on train,
       - permutation-importance rank all 108 source features on the fold's
         val rows (same metric as ablation.py: argmin-acc degradation primary,
         srocc + overhead tiebreak),
  3. Averages the per-fold importance ranks into a CV importance ranking
     (rank-aggregation across folds — robust to a single fold's noise).
  4. For each K in --ks and each fold f, re-fit the teacher on the fold's
     train rows using the TOP-K features *of that fold's own ranking*
     (no leakage: the val fold never informs its own feature selection),
     and score the fold's val rows. Reports per-K the CV mean +- std of
     {argmin-acc, overhead, p95-overhead, bytes-srocc} across folds.

The teacher + decision metric are byte-identical to ablation.py /
teacher_soft_targets.py (per-cell HistGradientBoostingRegressor, masked
argmin over predicted bytes_log, overhead vs the true oracle).

Outputs:
  --cv-importance  cv_importance.tsv   (feat, mean_rank, per-fold imp stats)
  --cv-topk        cv_topk.tsv         (K, mean/std of each metric across folds)
  --ranking-npz    cv_ranking.npz      (the aggregated ranked index, for bake build)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

# Teacher hyperparameters — identical to teacher_soft_targets.py / ablation.py.
HISTGB = dict(
    max_iter=400,
    max_depth=8,
    learning_rate=0.05,
    l2_regularization=0.5,
    random_state=0xCAFE,
)
MIN_CELL_ROWS = 50


def load_cache(path: Path):
    z = np.load(path, allow_pickle=True)
    return dict(
        features=z["features"],            # (n, 109) [108 feat + zq_norm]
        bytes_log=z["bytes_log"],          # (n, 36) NaN where unreachable
        reach=z["reach"],                  # (n, 36) bool
        target_zq=z["target_zq"],
        image_ids=z["image_ids"],
        feature_names=list(z["feature_names"]),  # 108 real zenanalyze names
        feat_col_names=list(z["feat_col_names"]),  # feat_0..feat_107
        cell_labels=list(z["cell_labels"]),
    )


def train_teacher(X_tr, bl_tr, n_cells):
    models = [None] * n_cells
    fallback = np.zeros(n_cells)
    for c in range(n_cells):
        y = bl_tr[:, c]
        mask = np.isfinite(y)
        fm = float(np.nanmean(y)) if mask.any() else 0.0
        fallback[c] = fm if np.isfinite(fm) else 0.0
        if mask.sum() < MIN_CELL_ROWS:
            continue
        m = HistGradientBoostingRegressor(**HISTGB)
        m.fit(X_tr[mask], y[mask])
        models[c] = m
    return models, fallback


def predict_all(models, fallback, X, n_cells):
    pred = np.zeros((X.shape[0], n_cells), dtype=np.float64)
    for c in range(n_cells):
        if models[c] is None:
            pred[:, c] = fallback[c]
        else:
            pred[:, c] = models[c].predict(X)
    return pred


def decision_metrics(pred, truth_bl, reach):
    correct = 0
    scored = 0
    overheads = []
    pool_pred = []
    pool_true = []
    for i in range(pred.shape[0]):
        reachable = np.where(reach[i])[0]
        if reachable.size == 0:
            continue
        scored += 1
        pc = reachable[np.argmin(pred[i, reachable])]
        tc = reachable[np.argmin(truth_bl[i, reachable])]
        if pc == tc:
            correct += 1
        best = float(np.exp(truth_bl[i, tc]))
        pick = float(np.exp(truth_bl[i, pc]))
        if best > 0 and np.isfinite(pick):
            overheads.append(pick / best - 1.0)
        pool_pred.extend(pred[i, reachable].tolist())
        pool_true.extend(truth_bl[i, reachable].tolist())
    argmin_acc = correct / scored if scored else 0.0
    mean_oh = float(np.mean(overheads)) if overheads else float("nan")
    p95_oh = float(np.percentile(overheads, 95)) if overheads else float("nan")
    if len(pool_pred) > 2:
        srocc = float(spearmanr(pool_pred, pool_true).correlation)
    else:
        srocc = float("nan")
    return dict(
        argmin_acc=argmin_acc,
        overhead_mean=mean_oh,
        overhead_p95=p95_oh,
        bytes_srocc=srocc,
        n_val_scored=scored,
    )


def make_folds(image_ids, k, seed):
    """Deterministic by-image K folds. Sort distinct images, shuffle with a
    fixed seed, then round-robin assign so fold sizes are balanced."""
    distinct = sorted(set(image_ids.tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(distinct))
    distinct = [distinct[i] for i in perm]
    fold_of = {}
    for i, im in enumerate(distinct):
        fold_of[im] = i % k
    fold = np.array([fold_of[im] for im in image_ids], dtype=np.int64)
    return fold, distinct


def fit_eval_subset(C, feat_idx, tr_mask, val_mask):
    """Train teacher on tr_mask rows with chosen source-feature subset (+zq_norm
    always), eval on val_mask rows."""
    n_cells = len(C["cell_labels"])
    n_src = len(C["feature_names"])
    keep = list(feat_idx) + [n_src]  # + zq_norm (last input col)
    X = C["features"][:, keep]
    bl = C["bytes_log"]
    reach = C["reach"]
    models, fallback = train_teacher(X[tr_mask], bl[tr_mask], n_cells)
    pred = predict_all(models, fallback, X[val_mask], n_cells)
    m = decision_metrics(pred, bl[val_mask], reach[val_mask])
    return m, models, fallback


def permutation_importance_fold(C, models, fallback, tr_mask, val_mask, repeats, seed):
    """Permutation importance of all 108 source features on this fold's val,
    using the fold's full-108 teacher. Returns per-feature mean importance."""
    n_cells = len(C["cell_labels"])
    n_src = len(C["feature_names"])
    keep = list(range(n_src)) + [n_src]
    X_val = C["features"][val_mask][:, keep].copy()
    bl_val = C["bytes_log"][val_mask]
    reach_val = C["reach"][val_mask]
    base = decision_metrics(predict_all(models, fallback, X_val, n_cells), bl_val, reach_val)
    rng = np.random.default_rng(seed)
    imp_arg = np.zeros(n_src)
    imp_oh = np.zeros(n_src)
    imp_sr = np.zeros(n_src)
    for j in range(n_src):
        da, do, ds = [], [], []
        for _ in range(repeats):
            Xp = X_val.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[perm, j]
            mm = decision_metrics(predict_all(models, fallback, Xp, n_cells), bl_val, reach_val)
            da.append(base["argmin_acc"] - mm["argmin_acc"])
            do.append(mm["overhead_mean"] - base["overhead_mean"])
            ds.append(base["bytes_srocc"] - mm["bytes_srocc"])
        imp_arg[j] = float(np.mean(da))
        imp_oh[j] = float(np.mean(do))
        imp_sr[j] = float(np.mean(ds))
    return imp_arg, imp_oh, imp_sr, base


def rank_fold(imp_arg, imp_oh, imp_sr):
    """Order features within a fold by the ablation.py key: argmin importance
    desc, then srocc importance desc, then overhead importance desc.
    Returns the ranked feature indices (best first) and a rank per feature."""
    n = len(imp_arg)
    order = sorted(range(n), key=lambda j: (-imp_arg[j], -imp_sr[j], -imp_oh[j]))
    rank = np.zeros(n, dtype=np.int64)
    for r, j in enumerate(order):
        rank[j] = r  # 0 = most important
    return order, rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--ks", default="20,30,40,50,60")
    ap.add_argument("--cv-importance", type=Path, required=True)
    ap.add_argument("--cv-topk", type=Path, required=True)
    ap.add_argument("--ranking-npz", type=Path, required=True)
    args = ap.parse_args()

    C = load_cache(args.cache)
    n_src = len(C["feature_names"])
    image_ids = C["image_ids"]
    fold, distinct = make_folds(image_ids, args.folds, args.seed)
    sys.stderr.write(
        f"[cv] {len(distinct)} images, {args.folds} folds; "
        f"sizes={[int((fold==f).sum()) for f in range(args.folds)]} rows/fold\n"
    )

    ks = [int(k) for k in args.ks.split(",")]
    t0 = time.time()

    # Per-fold: full-108 teacher, permutation importance, fold ranking.
    per_fold_rank = np.zeros((args.folds, n_src), dtype=np.int64)
    per_fold_order = []
    per_fold_imp_arg = np.zeros((args.folds, n_src))
    per_fold_imp_oh = np.zeros((args.folds, n_src))
    per_fold_imp_sr = np.zeros((args.folds, n_src))
    full108_metrics = []  # per fold, the full-108 baseline on that fold's val

    for f in range(args.folds):
        val_mask = fold == f
        tr_mask = fold != f
        sys.stderr.write(
            f"[cv] fold {f}: {tr_mask.sum()} train rows / {val_mask.sum()} val rows; "
            f"training full-108 teacher...\n"
        )
        m_full, models, fallback = fit_eval_subset(C, list(range(n_src)), tr_mask, val_mask)
        full108_metrics.append(m_full)
        sys.stderr.write(
            f"[cv] fold {f} full-108: argmin={m_full['argmin_acc']:.4f} "
            f"overhead={m_full['overhead_mean']:.4f} srocc={m_full['bytes_srocc']:.4f} "
            f"({time.time()-t0:.0f}s)\n"
        )
        imp_arg, imp_oh, imp_sr, _ = permutation_importance_fold(
            C, models, fallback, tr_mask, val_mask, args.repeats, args.seed + f
        )
        per_fold_imp_arg[f] = imp_arg
        per_fold_imp_oh[f] = imp_oh
        per_fold_imp_sr[f] = imp_sr
        order, rank = rank_fold(imp_arg, imp_oh, imp_sr)
        per_fold_order.append(order)
        per_fold_rank[f] = rank
        sys.stderr.write(f"[cv] fold {f} permutation importance done ({time.time()-t0:.0f}s)\n")

    # Aggregate ranking: mean rank across folds (lower = more important).
    mean_rank = per_fold_rank.mean(axis=0)
    agg_order = sorted(range(n_src), key=lambda j: (mean_rank[j], -per_fold_imp_arg[:, j].mean()))

    # Write CV importance TSV.
    args.cv_importance.parent.mkdir(parents=True, exist_ok=True)
    with open(args.cv_importance, "w") as fh:
        fh.write(
            "agg_rank\tfeat_idx\tfeat_name\tmean_fold_rank\tmean_imp_argmin\t"
            "std_imp_argmin\tmean_imp_overhead\tmean_imp_srocc\t"
            + "\t".join(f"rank_fold{f}" for f in range(args.folds))
            + "\n"
        )
        for ar, j in enumerate(agg_order):
            fh.write(
                f"{ar}\t{j}\t{C['feat_col_names'][j]}|{C['feature_names'][j]}\t"
                f"{mean_rank[j]:.3f}\t{per_fold_imp_arg[:, j].mean():.6f}\t"
                f"{per_fold_imp_arg[:, j].std():.6f}\t{per_fold_imp_oh[:, j].mean():.6f}\t"
                f"{per_fold_imp_sr[:, j].mean():.6f}\t"
                + "\t".join(str(int(per_fold_rank[f, j])) for f in range(args.folds))
                + "\n"
            )
    sys.stderr.write(f"[cv] wrote CV importance -> {args.cv_importance}\n")

    # Per-K CV: for each fold, select top-K from THAT fold's own ranking
    # (no val leakage), re-fit on fold train, score fold val. Aggregate.
    rows = []
    # Full-108 reference across folds.
    full_arg = np.array([m["argmin_acc"] for m in full108_metrics])
    full_oh = np.array([m["overhead_mean"] for m in full108_metrics])
    full_p95 = np.array([m["overhead_p95"] for m in full108_metrics])
    full_sr = np.array([m["bytes_srocc"] for m in full108_metrics])
    rows.append(("full108", n_src, full_arg, full_oh, full_p95, full_sr))

    for k in ks:
        k = min(k, n_src)
        arg = np.zeros(args.folds)
        oh = np.zeros(args.folds)
        p95 = np.zeros(args.folds)
        sr = np.zeros(args.folds)
        for f in range(args.folds):
            val_mask = fold == f
            tr_mask = fold != f
            keep = per_fold_order[f][:k]  # fold-own ranking, no leakage
            m, _, _ = fit_eval_subset(C, keep, tr_mask, val_mask)
            arg[f] = m["argmin_acc"]
            oh[f] = m["overhead_mean"]
            p95[f] = m["overhead_p95"]
            sr[f] = m["bytes_srocc"]
        rows.append((f"top{k}", k, arg, oh, p95, sr))
        sys.stderr.write(
            f"[cv] top{k}: overhead {oh.mean():.4f}+-{oh.std():.4f} "
            f"argmin {arg.mean():.4f}+-{arg.std():.4f} "
            f"srocc {sr.mean():.4f}+-{sr.std():.4f} ({time.time()-t0:.0f}s)\n"
        )

    args.cv_topk.parent.mkdir(parents=True, exist_ok=True)
    with open(args.cv_topk, "w") as fh:
        fh.write(
            "label\tK\toverhead_mean\toverhead_std\targmin_mean\targmin_std\t"
            "p95_mean\tp95_std\tsrocc_mean\tsrocc_std\tn_folds\n"
        )
        for label, k, arg, oh, p95, sr in rows:
            fh.write(
                f"{label}\t{k}\t{oh.mean():.5f}\t{oh.std():.5f}\t"
                f"{arg.mean():.5f}\t{arg.std():.5f}\t"
                f"{p95.mean():.5f}\t{p95.std():.5f}\t"
                f"{sr.mean():.5f}\t{sr.std():.5f}\t{args.folds}\n"
            )
    sys.stderr.write(f"[cv] wrote CV top-K -> {args.cv_topk}\n")

    # Persist the aggregated ranked order for the bake-build step.
    np.savez(
        args.ranking_npz,
        agg_ranked_idx=np.asarray(agg_order, dtype=np.int64),
        agg_ranked_feat_col=np.asarray([C["feat_col_names"][j] for j in agg_order]),
        agg_ranked_feat_name=np.asarray([C["feature_names"][j] for j in agg_order]),
        mean_rank=mean_rank,
        per_fold_rank=per_fold_rank,
        feat_col_names=np.asarray(C["feat_col_names"]),
        feature_names=np.asarray(C["feature_names"]),
    )
    sys.stderr.write(f"[cv] wrote ranking npz -> {args.ranking_npz}\n")

    # Print a compact summary JSON.
    summary = {
        "folds": args.folds,
        "topk": {
            label: {
                "K": k,
                "overhead_mean": float(oh.mean()),
                "overhead_std": float(oh.std()),
                "argmin_mean": float(arg.mean()),
                "srocc_mean": float(sr.mean()),
            }
            for label, k, arg, oh, p95, sr in rows
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
