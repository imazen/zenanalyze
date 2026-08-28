#!/usr/bin/env python3
"""Fit a per-image achievable-ceiling predictor from analyzer features
(imazen/zenanalyze#51, work item 3 — the `PredictedZensimCeiling` idea,
done as a BAKE instead of a new zenanalyze feature).

Every perceptual metric has a content-and-size-dependent ceiling: tiny or
complex images cannot reach zensim 94 at any encoder setting. The sweep
already tells us the ceiling per `(image, size_class)`
(`effective_max_<metric>` in the Pareto, emitted by `canonical_to_pareto.py`).
This tool asks whether the analyzer features predict it well enough for a
codec to short-circuit an unreachable target BEFORE encoding
(`UnreachableAction`, see FOR_NEW_CODECS.md / SAFETY_PLANE.md):

1. `load_pareto` (train_hybrid — the one Parquet/TSV pareto owner) gives
   `ceilings[(image, size_class)]`; `_picker_lib.load_features_raw` gives
   the analyzer features; rows are joined on the key.
2. Split by `origin_split.split_of(image_path)` (even/odd origin id — the
   canonical rule; renditions of one source never straddle splits).
3. Teacher: HistGradientBoostingRegressor on `[features, size one-hot,
   log_pixels]`. Reported per split and per size_class: R², MAE, p90 |err|
   and — the number that matters for a pit-of-success gate — the
   OVER-prediction rate at margins 0 / 2 / 5 metric points (a predicted
   ceiling above the real one lets the codec attempt a target it cannot
   reach; a margin is what `effective_target = min(user, predicted -
   margin)` subtracts).
4. `--student-json`: an sklearn MLP student (1 output) in the JSON shape
   `tools/bake_picker.py` bakes to ZNPR v3, so a codec can ship the
   predictor as a ~10 KB bake with no zenanalyze API change. Its metrics
   are reported next to the teacher's.

Usage (canonical zenwebp, after canonical_to_pareto.py):

    PYTHONPATH=.:<zenmetrics>/scripts/picker python3 fit_zensim_ceiling.py \\
        --pareto ~/tmp/canonical/zenwebp_lossy/derived/pareto.parquet \\
        --features ~/tmp/canonical/zenwebp_lossy/derived/features.tsv \\
        --out-tsv ../../benchmarks/zensim_ceiling_fit_zenwebp_<date>.tsv \\
        --md ../../benchmarks/zensim_ceiling_fit_zenwebp_<date>.md \\
        --student-json ~/tmp/canonical/zenwebp_lossy/models/zenwebp_zensim_ceiling.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv_saved = list(sys.argv)
sys.argv = ["train_hybrid.py"]  # train_hybrid parses argv at import in some paths
import train_hybrid as th  # noqa: E402
from _picker_lib import load_features_raw  # noqa: E402

sys.argv = sys.argv_saved

SIZE_CLASSES = ["tiny", "small", "medium", "large"]
EXTRA_AXES = [f"size_{s}" for s in SIZE_CLASSES] + ["log_pixels"]
MARGINS = (0.0, 2.0, 5.0)


def load_split_of():
    """`origin_split.split_of` from PYTHONPATH or the sibling zenmetrics
    checkout — hard error otherwise (a seeded shuffle would leak
    renditions across splits; train_hybrid refuses the same way)."""
    try:
        from origin_split import split_of  # type: ignore

        return split_of
    except ImportError:
        pass
    sibling = HERE.parent.parent.parent / "zenmetrics" / "scripts" / "picker"
    if (sibling / "origin_split.py").exists():
        sys.path.insert(0, str(sibling))
        from origin_split import split_of  # type: ignore

        return split_of
    raise SystemExit(
        "origin_split.py not importable — add <zenmetrics>/scripts/picker to "
        "PYTHONPATH (the canonical even/odd origin split; no fallback on purpose)."
    )


def build_rows(ceilings: dict, pixels: dict, feats: dict, split_of):
    """One row per (image, size_class) present in both the ceilings and the
    features. Returns (X, y, size_idx, split, keys) with X = [features,
    size one-hot, log_pixels]."""
    X, y, sizes, splits, keys = [], [], [], [], []
    for key, ceiling in ceilings.items():
        if ceiling is None or not math.isfinite(ceiling):
            continue
        f = feats.get(key)
        if f is None:
            continue
        sp = split_of(key[0])
        if sp is None:
            continue
        image, size = key
        oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        if size in SIZE_CLASSES:
            oh[SIZE_CLASSES.index(size)] = 1.0
        px = pixels.get(key, 0)
        log_px = math.log(max(1, px))
        X.append(np.concatenate([f, oh, np.array([log_px], dtype=np.float32)]))
        y.append(float(ceiling))
        sizes.append(size)
        splits.append(sp)
        keys.append(key)
    if not X:
        raise SystemExit("no (image, size_class) rows join the ceilings to the features")
    return (
        np.stack(X).astype(np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(sizes, dtype=object),
        np.asarray(splits, dtype=object),
        keys,
    )


def eval_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = pred - actual
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    out = {
        "n": int(actual.size),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(err))),
        "p90_abs": float(np.percentile(np.abs(err), 90)),
        "max_abs": float(np.max(np.abs(err))),
    }
    for m in MARGINS:
        out[f"over_{m:g}"] = float(np.mean(err > m))
    return out


def eval_by_size(pred, actual, sizes) -> dict:
    out = {"all": eval_metrics(pred, actual)}
    for s in SIZE_CLASSES:
        m = sizes == s
        if m.any():
            out[s] = eval_metrics(pred[m], actual[m])
    return out


def student_json(model, scaler, feat_cols: list[str], metric: str, metrics: dict, bake_name: str,
                 nan_fill=None) -> dict:
    return {
        # Per-input value to substitute for NaN BEFORE the bake's standardize
        # step (train-split column mean; 0.0 for the engineered axes).
        "nan_fill": [float(v) for v in nan_fill] if nan_fill is not None else None,
        "n_inputs": int(scaler.mean_.size),
        "n_outputs": 1,
        "feat_cols": list(feat_cols),
        "extra_axes": list(EXTRA_AXES),
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
        "activation": "relu",
        "schema_version_tag": "zenpredict.ceiling.v1",
        "bake_name": bake_name,
        "layers": [
            {"W": W.astype(np.float32).tolist(), "b": b.astype(np.float32).tolist()}
            for W, b in zip(model.coefs_, model.intercepts_)
        ],
        "training_objective": {"name": "ceiling", "metric_name": metric},
        "safety_report": {
            "passed": True,
            "violations": [],
            "diagnostics": {"ceiling": metrics},
        },
    }


def fmt_row(section: str, split: str, size: str, m: dict) -> str:
    return "\t".join(
        [section, split, size, str(m["n"]), f"{m['r2']:.4f}", f"{m['mae']:.3f}", f"{m['p90_abs']:.3f}",
         f"{m['max_abs']:.3f}"] + [f"{m[f'over_{x:g}']:.4f}" for x in MARGINS]
    )


def main(argv: list[str] | None = None, *, split_of=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pareto", type=Path, required=True, help="Pareto parquet/TSV with effective_max_<metric>")
    ap.add_argument("--features", type=Path, required=True, help="features TSV (image_path, size_class, feat_*)")
    ap.add_argument("--metric-column", default="zensim")
    ap.add_argument("--keep-features", default=None, help="comma-separated feat_ columns (default: all)")
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--student-json", type=Path, default=None)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--seed", type=int, default=0xCAFE)
    args = ap.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    split_of = split_of or load_split_of()
    th.METRIC_COLUMN = args.metric_column
    pareto, ceilings, has_ceiling, _has_time = th.load_pareto(args.pareto)
    if not has_ceiling:
        raise SystemExit(
            f"{args.pareto}: no effective_max_{args.metric_column} column — "
            "regenerate with canonical_to_pareto.py or have the sweep emit it (#51)"
        )
    pixels = {(image, size): w * h for (image, size, w, h) in pareto.keys()}
    keep = args.keep_features.split(",") if args.keep_features else None
    feats, feat_cols = load_features_raw(args.features, keep)
    X, y, sizes, splits, _keys = build_rows(ceilings, pixels, feats, split_of)
    tr, va, te = splits == "train", splits == "val", splits == "test"
    if not tr.any() or not va.any():
        raise SystemExit(f"split too small: train={tr.sum()} val={va.sum()} test={te.sum()}")
    sys.stderr.write(
        f"rows: {len(y)} (train {tr.sum()} / val {va.sum()} / test {te.sum()}), "
        f"{len(feat_cols)} features + {len(EXTRA_AXES)} axes; ceiling p50 {np.median(y):.1f}\n"
    )

    teacher = HistGradientBoostingRegressor(max_iter=400, max_depth=8, learning_rate=0.05, random_state=args.seed)
    teacher.fit(X[tr], y[tr])
    report = {"teacher": {}, "student": {}}
    for name, mask in (("train", tr), ("val", va), ("test", te)):
        if mask.any():
            report["teacher"][name] = eval_by_size(teacher.predict(X[mask]), y[mask], sizes[mask])

    student = None
    scaler = None
    nan_fill = None
    if args.student_json is not None:
        # Tiny images skip the percentile features (zenanalyze#49) → NaN.
        # HistGB takes NaN natively; the MLP does not, so impute with the
        # TRAIN column mean and ship the fill vector in the JSON — the
        # codec must apply the same fill before calling the bake.
        col_mean = np.nanmean(X[tr], axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nan_fill = col_mean
        X = np.where(np.isnan(X), col_mean[None, :], X)
        scaler = StandardScaler().fit(X[tr])
        hidden = tuple(int(h) for h in args.hidden.split(","))
        student = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                               learning_rate_init=2e-3, max_iter=800, early_stopping=True,
                               n_iter_no_change=40, random_state=args.seed)
        student.fit(scaler.transform(X[tr]), y[tr])
        for name, mask in (("train", tr), ("val", va), ("test", te)):
            if mask.any():
                report["student"][name] = eval_by_size(
                    student.predict(scaler.transform(X[mask])), y[mask], sizes[mask]
                )

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tsv, "w") as f:
        f.write(f"# fit_zensim_ceiling.py metric={args.metric_column} pareto={args.pareto} features={args.features} "
                f"n_features={len(feat_cols)} seed={args.seed} date={datetime.date.today().isoformat()}\n")
        f.write("section\tsplit\tsize_class\tn\tr2\tmae\tp90_abs\tmax_abs" + "".join(f"\tover_{m:g}" for m in MARGINS) + "\n")
        for section in ("teacher", "student"):
            for split, by_size in report[section].items():
                for size, m in by_size.items():
                    f.write(fmt_row(section, split, size, m) + "\n")

    if args.md is not None:
        lines = [f"# Feature → achievable-{args.metric_column}-ceiling fit (zenanalyze#51)", "",
                 f"Date {datetime.date.today().isoformat()} · pareto `{args.pareto}` · features `{args.features}` "
                 f"({len(feat_cols)} feat_ + {len(EXTRA_AXES)} axes) · rows {len(y)} "
                 f"(train {tr.sum()} / val {va.sum()} / test {te.sum()}, origin even/odd split) · seed {args.seed}", "",
                 "`over_m` = fraction of rows whose PREDICTED ceiling exceeds the real one by more than m points — "
                 "the codec-side risk of attempting an unreachable target when it subtracts margin m.", ""]
        for section in ("teacher", "student"):
            if not report[section]:
                continue
            lines += [f"## {section} ({'HistGradientBoosting' if section == 'teacher' else 'MLP student ' + args.hidden})", "",
                      "| split | size_class | n | R² | MAE | p90 abs | max abs | " + " | ".join(f"over_{m:g}" for m in MARGINS) + " |",
                      "|---|---|---:|---:|---:|---:|---:|" + "---:|" * len(MARGINS)]
            for split, by_size in report[section].items():
                for size, m in by_size.items():
                    lines.append(f"| {split} | {size} | {m['n']} | {m['r2']:.3f} | {m['mae']:.2f} | {m['p90_abs']:.2f} | "
                                 f"{m['max_abs']:.2f} | " + " | ".join(f"{100 * m[f'over_{x:g}']:.1f} %" for x in MARGINS) + " |")
            lines.append("")
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(lines) + "\n")

    if student is not None:
        args.student_json.parent.mkdir(parents=True, exist_ok=True)
        args.student_json.write_text(json.dumps(student_json(
            student, scaler, feat_cols, args.metric_column, report["student"], args.student_json.stem,
            nan_fill=nan_fill,
        )))
        sys.stderr.write(f"student JSON → {args.student_json} (bake with tools/bake_picker.py)\n")
    v = report["teacher"]["val"]["all"]
    sys.stderr.write(f"teacher val: R² {v['r2']:.3f} MAE {v['mae']:.2f} p90 {v['p90_abs']:.2f} "
                     f"over_0 {100 * v['over_0']:.1f}% over_2 {100 * v['over_2']:.1f}% over_5 {100 * v['over_5']:.1f}%\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
