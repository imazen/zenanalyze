#!/usr/bin/env python3
"""Feature-ablation / LOO study on the zenjpeg source-feature picker.

Operates on the materialized dataset cache from `build_dataset.py`.
The teacher is the per-cell `HistGradientBoostingRegressor` exactly as
in `zenanalyze/zenpicker-train/scripts/teacher_soft_targets.py`. The
held-out DECISION metric is the teacher script's `argmin_overhead` plus
a pooled bytes-SROCC.

Subcommands:
  permute   train full-108 teacher once, permutation-importance rank all
            108 source features on held-out val (cheap O(features) pass).
  topk      retrain the teacher on top-K source features for several K,
            emit K vs {argmin-acc, overhead, bytes-SROCC} held-out.
  baseline  just train full-108 + dropped-8-dead + dropped-17-deadcorpus,
            report held-out metrics (sanity: dead drops change nothing).
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

# Teacher hyperparameters — identical to teacher_soft_targets.py defaults.
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
        split=z["split"],                  # 0=train 1=val
        feature_names=list(z["feature_names"]),  # 108 source-feature names
        cell_labels=list(z["cell_labels"]),
    )


def train_teacher(X_tr, bl_tr, n_cells):
    """Per-cell HistGB on reaching train rows; returns (models, fallback)."""
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
    """Dense soft bytes_log predictions for all rows (teacher_predict_all)."""
    pred = np.zeros((X.shape[0], n_cells), dtype=np.float64)
    for c in range(n_cells):
        if models[c] is None:
            pred[:, c] = fallback[c]
        else:
            pred[:, c] = models[c].predict(X)
    return pred


def decision_metrics(pred, truth_bl, reach):
    """The teacher script's argmin_overhead + a pooled bytes-SROCC.

    pred/truth_bl/reach are (n_val, n_cells). Decision is masked argmin
    over predicted bytes_log; oracle is masked argmin over TRUE bytes_log.
    Overhead uses TRUE bytes (exp of true bytes_log) of pick vs oracle.
    bytes-SROCC = Spearman over all (row, reachable-cell) pairs of
    (pred_bl, true_bl) — the regression-quality signal.
    """
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


def fit_and_eval(C, feat_idx, label="", verbose=True):
    """Train teacher on the chosen source-feature subset + zq_norm; eval val.

    feat_idx: indices into the 108 source features to KEEP. zq_norm (the
    last input column) is ALWAYS kept (it's the requested-quality axis,
    not under ablation).
    """
    n_cells = len(C["cell_labels"])
    n_src = len(C["feature_names"])
    keep = list(feat_idx) + [n_src]  # + zq_norm column
    X = C["features"][:, keep]
    bl = C["bytes_log"]
    reach = C["reach"]
    is_tr = C["split"] == 0
    is_val = C["split"] == 1
    t0 = time.time()
    models, fallback = train_teacher(X[is_tr], bl[is_tr], n_cells)
    pred_val = predict_all(models, fallback, X[is_val], n_cells)
    m = decision_metrics(pred_val, bl[is_val], reach[is_val])
    m["n_features"] = len(feat_idx)
    m["fit_sec"] = time.time() - t0
    m["n_cells_with_teacher"] = sum(1 for x in models if x is not None)
    if verbose:
        sys.stderr.write(
            f"[fit{(' '+label) if label else ''}] K={len(feat_idx):3d} "
            f"argmin={m['argmin_acc']:.4f} overhead={m['overhead_mean']:.4f} "
            f"p95={m['overhead_p95']:.3f} bytes_srocc={m['bytes_srocc']:.4f} "
            f"({m['fit_sec']:.1f}s, {m['n_cells_with_teacher']}/{n_cells} cells)\n"
        )
    return m, models, fallback


def cmd_permute(args):
    C = load_cache(args.cache)
    n_src = len(C["feature_names"])
    n_cells = len(C["cell_labels"])
    is_tr = C["split"] == 0
    is_val = C["split"] == 1

    # Train full-108 teacher once.
    sys.stderr.write("[permute] training full-108 teacher...\n")
    all_idx = list(range(n_src))
    base_m, models, fallback = fit_and_eval(C, all_idx, label="FULL-108")

    # Baseline val predictions (unperturbed).
    keep = all_idx + [n_src]
    X_val = C["features"][is_val][:, keep].copy()
    bl_val = C["bytes_log"][is_val]
    reach_val = C["reach"][is_val]
    base_pred = predict_all(models, fallback, X_val, n_cells)
    base = decision_metrics(base_pred, bl_val, reach_val)
    sys.stderr.write(
        f"[permute] baseline (recomputed): argmin={base['argmin_acc']:.4f} "
        f"overhead={base['overhead_mean']:.4f} bytes_srocc={base['bytes_srocc']:.4f}\n"
    )

    # Per-image std on the per-image population (for reporting).
    # Use first-row-per-image to get the true per-image feature std.
    img = C["image_ids"]
    first = {}
    for i, im in enumerate(img):
        if im not in first:
            first[im] = i
    fr = sorted(first.values())
    Fimg = C["features"][fr, :n_src]
    src_std = Fimg.std(axis=0)

    n_rep = args.repeats
    rng = np.random.default_rng(args.seed)
    rows = []
    t_start = time.time()
    for j in range(n_src):
        d_arg = []
        d_oh = []
        d_sr = []
        for rep in range(n_rep):
            Xp = X_val.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[perm, j]  # shuffle ONLY feature j (col j of keep == src j)
            pp = predict_all(models, fallback, Xp, n_cells)
            mm = decision_metrics(pp, bl_val, reach_val)
            d_arg.append(base["argmin_acc"] - mm["argmin_acc"])      # >0 = important
            d_oh.append(mm["overhead_mean"] - base["overhead_mean"]) # >0 = important (overhead up)
            d_sr.append(base["bytes_srocc"] - mm["bytes_srocc"])     # >0 = important
        rows.append(
            dict(
                feat_idx=j,
                feat_name=C["feature_names"][j],
                std=float(src_std[j]),
                imp_argmin=float(np.mean(d_arg)),
                imp_argmin_sd=float(np.std(d_arg)),
                imp_overhead=float(np.mean(d_oh)),
                imp_overhead_sd=float(np.std(d_oh)),
                imp_srocc=float(np.mean(d_sr)),
                imp_srocc_sd=float(np.std(d_sr)),
            )
        )
        if (j + 1) % 10 == 0:
            sys.stderr.write(
                f"[permute] {j+1}/{n_src} features done "
                f"({time.time()-t_start:.0f}s elapsed)\n"
            )

    # Rank by argmin-acc degradation (primary), tiebreak by srocc deg, then overhead.
    rows.sort(key=lambda r: (-r["imp_argmin"], -r["imp_srocc"], -r["imp_overhead"]))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(
            "rank\tfeat_idx\tfeat_name\tstd\timp_argmin\timp_argmin_sd\t"
            "imp_overhead\timp_overhead_sd\timp_srocc\timp_srocc_sd\n"
        )
        for rank, r in enumerate(rows, 1):
            f.write(
                f"{rank}\t{r['feat_idx']}\t{r['feat_name']}\t{r['std']:.6g}\t"
                f"{r['imp_argmin']:.6f}\t{r['imp_argmin_sd']:.6f}\t"
                f"{r['imp_overhead']:.6f}\t{r['imp_overhead_sd']:.6f}\t"
                f"{r['imp_srocc']:.6f}\t{r['imp_srocc_sd']:.6f}\n"
            )
    sys.stderr.write(f"[permute] wrote ranking -> {out}\n")

    # Persist the ranked order for topk to consume + the baseline metrics.
    np.savez(
        args.ranking_npz,
        ranked_idx=np.asarray([r["feat_idx"] for r in rows]),
        ranked_names=np.asarray([r["feat_name"] for r in rows]),
        imp_argmin=np.asarray([r["imp_argmin"] for r in rows]),
        baseline_argmin=base["argmin_acc"],
        baseline_overhead=base["overhead_mean"],
        baseline_srocc=base["bytes_srocc"],
    )
    sys.stderr.write(f"[permute] wrote ranking npz -> {args.ranking_npz}\n")
    print(json.dumps({"baseline": base, "n_repeats": n_rep}, indent=2))


def cmd_topk(args):
    C = load_cache(args.cache)
    r = np.load(args.ranking_npz, allow_pickle=True)
    ranked_idx = list(r["ranked_idx"])
    base_argmin = float(r["baseline_argmin"])
    base_overhead = float(r["baseline_overhead"])
    base_srocc = float(r["baseline_srocc"])

    ks = [int(k) for k in args.ks.split(",")]
    n_src = len(C["feature_names"])
    rows = []
    # Full-108 reference (re-fit so timing + cells are comparable).
    full_m, _, _ = fit_and_eval(C, list(range(n_src)), label="FULL-108-ref")
    rows.append(("full108", n_src, full_m))
    for k in ks:
        k = min(k, n_src)
        keep = ranked_idx[:k]
        m, _, _ = fit_and_eval(C, keep, label=f"top{k}")
        rows.append((f"top{k}", k, m))

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(
            "label\tK\targmin_acc\tdelta_argmin_vs108\toverhead_mean\t"
            "delta_overhead_vs108\tbytes_srocc\tdelta_srocc_vs108\toverhead_p95\tn_val_scored\n"
        )
        for label, k, m in rows:
            f.write(
                f"{label}\t{k}\t{m['argmin_acc']:.5f}\t{m['argmin_acc']-base_argmin:+.5f}\t"
                f"{m['overhead_mean']:.5f}\t{m['overhead_mean']-base_overhead:+.5f}\t"
                f"{m['bytes_srocc']:.5f}\t{m['bytes_srocc']-base_srocc:+.5f}\t"
                f"{m['overhead_p95']:.4f}\t{m['n_val_scored']}\n"
            )
    sys.stderr.write(f"[topk] wrote curve -> {out}\n")
    print(json.dumps({"full108": full_m,
                      "baseline_ref": {"argmin": base_argmin,
                                        "overhead": base_overhead,
                                        "srocc": base_srocc}}, indent=2))


def cmd_baseline(args):
    """Sanity: full-108 vs drop-8-dead vs drop-17-deadcorpus, held-out."""
    C = load_cache(args.cache)
    n_src = len(C["feature_names"])
    names = C["feature_names"]
    DEAD8 = {"feat_peak_luminance_nits", "feat_p99_luminance_nits",
             "feat_hdr_headroom_stops", "feat_hdr_pixel_fraction",
             "feat_hdr_present", "feat_alpha_present",
             "feat_alpha_used_fraction", "feat_alpha_bimodal_score"}
    # 17 std==0 on this corpus (computed in build): indices fixed.
    src_std = None
    img = C["image_ids"]
    first = {}
    for i, im in enumerate(img):
        first.setdefault(im, i)
    Fimg = C["features"][sorted(first.values()), :n_src]
    src_std = Fimg.std(axis=0)
    dead_corpus = [j for j in range(n_src) if src_std[j] == 0.0]

    all_idx = list(range(n_src))
    keep_no8 = [j for j in range(n_src) if names[j] not in DEAD8]
    keep_no17 = [j for j in range(n_src) if j not in dead_corpus]

    results = {}
    m_full, _, _ = fit_and_eval(C, all_idx, label="FULL-108")
    results["full108"] = m_full
    m8, _, _ = fit_and_eval(C, keep_no8, label="drop8dead")
    results["drop8dead_keep100"] = m8
    m17, _, _ = fit_and_eval(C, keep_no17, label=f"drop{len(dead_corpus)}deadcorpus")
    results[f"drop{len(dead_corpus)}deadcorpus_keep{len(keep_no17)}"] = m17

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    sys.stderr.write(f"[baseline] wrote -> {out}\n")
    print(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("permute")
    p.add_argument("--cache", type=Path, default=Path("cache/picker_ds.npz"))
    p.add_argument("--out", type=Path, default=Path("importance_ranking.tsv"))
    p.add_argument("--ranking-npz", type=Path, default=Path("cache/ranking.npz"))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_permute)

    p = sub.add_parser("topk")
    p.add_argument("--cache", type=Path, default=Path("cache/picker_ds.npz"))
    p.add_argument("--ranking-npz", type=Path, default=Path("cache/ranking.npz"))
    p.add_argument("--out", type=Path, default=Path("topk_curve.tsv"))
    p.add_argument("--ks", default="100,80,60,40,20")
    p.set_defaults(func=cmd_topk)

    p = sub.add_parser("baseline")
    p.add_argument("--cache", type=Path, default=Path("cache/picker_ds.npz"))
    p.add_argument("--out", type=Path, default=Path("baseline_dead_drop.json"))
    p.set_defaults(func=cmd_baseline)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
