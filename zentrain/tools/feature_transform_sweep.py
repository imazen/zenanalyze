#!/usr/bin/env python3
"""Per-feature input-transform sweep for picker training.

Adapts zensim's V_20 input-shaping screen
(`zensim/scripts/v_next/v0_20_feature_transform_greedy_screen.py`) to
the picker workload's argmin objective. zensim screens transforms by
their Pearson lift vs human MOS scores; the picker doesn't have MOS
scores — its label is per-cell `bytes_log` and the gold-standard
metric is held-out **val argmin accuracy**.

Two-tier methodology mirroring zensim's:

1. **Fast Pearson screen.** For each `(feat_col × transform × params)`:
   compute `|Pearson(transform(feat), bytes_log[:, c])|` averaged
   over reachable cells, then lift vs identity. ~CPU-seconds per
   codec.

2. **End-to-end confirmation.** Train the student MLP twice:
   - **Baseline** — apply the codec's current `FEATURE_TRANSFORMS`.
   - **Recommended** — apply the screen's per-feature winners.
   Compare val argmin accuracy + mean overhead. The recommendation is
   only adopted if it's a measured win. ~minutes per codec on RTX-class
   GPU.

Beyond zensim's screen vocabulary:
- All 9 transforms in `zenpredict::FeatureTransform`
  (identity, log, log1p, signed_log1p, signed_sqrt, signed_cbrt,
  clip_then_log1p, winsor_p99, quantile_bins) are exercised, **with
  parameter sweeps**. zensim's screen also covers these.
- Training-safety gates (NaN propagation, log(<0), log1p(<-1)) match
  zensim — the runtime apply path doesn't drop bad rows.

## Usage

    PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \\
        python3 zentrain/tools/feature_transform_sweep.py \\
            --codec-config zenjpeg_picker_config \\
            --out /home/lilith/work/zen/zenanalyze/benchmarks/feat_xform_zenjpeg_2026-05-17 \\
            --epochs 100 \\
            --confirm

## Output

Under `--out` directory:
- `screen_results.tsv` — every (feat, transform, params) row with its
  Pearson aggregate + lift vs identity. Sortable.
- `recommended_transforms.py` — Python snippet drop-in for the codec
  config: `FEATURE_TRANSFORMS = {...}` + optional params dict.
- `confirmation_summary.json` — baseline vs recommended end-to-end
  metrics (only when `--confirm` is passed).
- `summary.md` — human-readable report.

## Notes

The screen is "per-feature greedy" — each feature gets its individually
best transform, ignoring cross-feature interactions. True combinatorial
search over 9 transforms × 50 features is infeasible (8^50). The
end-to-end confirmation step catches catastrophic interactions
(e.g. two transforms that look good individually but conflict with
each other in the MLP's gradient).

For truly optimal combos, run with `--confirm` and inspect the
end-to-end delta. If the recommended set loses to baseline, fall back
to the codec's existing `FEATURE_TRANSFORMS` and triage which feature
introduced the regression (one-feature-at-a-time ablation, queued).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

import train_hybrid as TH


# ---------------------------------------------------------------------------
# Transform library — mirror zenpredict::FeatureTransform variants exactly.
# ---------------------------------------------------------------------------


def t_identity(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return x


def t_log(x: np.ndarray, _params: list[float]) -> np.ndarray:
    out = np.full_like(x, np.nan)
    pos = x > 0
    out[pos] = np.log(x[pos])
    return out


def t_log1p(x: np.ndarray, _params: list[float]) -> np.ndarray:
    out = np.full_like(x, np.nan)
    valid = x >= 0
    out[valid] = np.log1p(x[valid])
    return out


def t_signed_log1p(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def t_signed_sqrt(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def t_signed_cbrt(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.cbrt(np.abs(x))


def t_clip_then_log1p(x: np.ndarray, params: list[float]) -> np.ndarray:
    eps = params[0] if params else 0.0
    return np.log1p(np.maximum(0.0, x - eps))


def t_winsor_p99(x: np.ndarray, params: list[float]) -> np.ndarray:
    if len(params) < 2:
        return x
    return np.clip(x, params[0], params[1])


def t_quantile_bins(x: np.ndarray, params: list[float]) -> np.ndarray:
    if len(params) < 2:
        return x
    edges = np.asarray(params, dtype=np.float64)
    idx = np.zeros_like(x, dtype=np.float64)
    for edge in edges:
        idx += (x >= edge).astype(np.float64)
    return idx / len(edges)


TRANSFORMS: dict[str, Callable[[np.ndarray, list[float]], np.ndarray]] = {
    "identity": t_identity,
    "log": t_log,
    "log1p": t_log1p,
    "signed_log1p": t_signed_log1p,
    "signed_sqrt": t_signed_sqrt,
    "signed_cbrt": t_signed_cbrt,
    "clip_then_log1p": t_clip_then_log1p,
    "winsor_p99": t_winsor_p99,
    "quantile_bins": t_quantile_bins,
}


# ---------------------------------------------------------------------------
# Stacked transforms (two-step compositions)
# ---------------------------------------------------------------------------
#
# `clip_then_log1p` is effectively a single-sided stack already (subtract
# noise floor → clip non-negative → log1p). The compositions below are
# genuinely different:
#
# - `winsor_p99 → log1p`: clip BOTH tails to [p1, p99], then log1p. Distinct
#   from `clip_then_log1p` (which only subtracts a lower bound and doesn't
#   clip the upper tail).
# - `signed_log1p → winsor_p99`: compress with sign-preserving log1p first,
#   then clip the log-domain outliers. Useful when the post-log distribution
#   still has fat tails.
# - `clip_then_log1p → winsor_p99`: noise-floor + log compress, then clip
#   remaining log-domain outliers. Three-step.
# - `signed_cbrt → winsor_p99`: mild variance stabilization then clip.
#
# Stacks involving `quantile_bins` are excluded — bin index is rank-only,
# so any monotonic outer transform is a no-op on the rank order. Stacks
# with `identity` on either side reduce to a single transform.
#
# Each stack accepts (outer_params, inner_params). For simplicity the
# concatenated params list is `inner_params + outer_params`, with the
# split documented per-stack in `STACK_PARAM_SPLITS`.


def _make_stack(inner_fn, outer_fn, inner_n_params: int):
    """Factory: returns a transform fn that applies `inner_fn` then
    `outer_fn`, splitting the param list at `inner_n_params`."""
    def stacked(x: np.ndarray, params: list[float]) -> np.ndarray:
        inner_p = list(params[:inner_n_params])
        outer_p = list(params[inner_n_params:])
        y = inner_fn(x, inner_p)
        # Convert any NaN from the inner stage to a sane default so the
        # outer transform's percentile-based params (computed at sweep
        # time) don't choke on NaN inputs.
        if np.issubdtype(y.dtype, np.floating):
            y = np.where(np.isfinite(y), y, np.nan)
        return outer_fn(y, outer_p)
    return stacked


STACKS: dict[str, Callable[[np.ndarray, list[float]], np.ndarray]] = {
    # (inner, outer, inner_n_params)
    "winsor_then_log1p":            _make_stack(t_winsor_p99, t_log1p, 2),
    "winsor_then_log":              _make_stack(t_winsor_p99, t_log, 2),
    "winsor_then_signed_cbrt":      _make_stack(t_winsor_p99, t_signed_cbrt, 2),
    "winsor_then_signed_log1p":     _make_stack(t_winsor_p99, t_signed_log1p, 2),
    "log1p_then_winsor":            _make_stack(t_log1p, t_winsor_p99, 0),
    "log_then_winsor":              _make_stack(t_log, t_winsor_p99, 0),
    "signed_log1p_then_winsor":     _make_stack(t_signed_log1p, t_winsor_p99, 0),
    "signed_cbrt_then_winsor":      _make_stack(t_signed_cbrt, t_winsor_p99, 0),
    "clip_then_log1p_then_winsor":  _make_stack(t_clip_then_log1p, t_winsor_p99, 1),
}

# How many parameters the inner stage consumes; the rest go to the outer.
STACK_INNER_NPARAMS: dict[str, int] = {
    "winsor_then_log1p": 2,
    "winsor_then_log": 2,
    "winsor_then_signed_cbrt": 2,
    "winsor_then_signed_log1p": 2,
    "log1p_then_winsor": 0,
    "log_then_winsor": 0,
    "signed_log1p_then_winsor": 0,
    "signed_cbrt_then_winsor": 0,
    "clip_then_log1p_then_winsor": 1,
}


def sweep_for(name: str, col: np.ndarray) -> list[list[float]]:
    """Param-vector candidates for a given transform + feature column.
    Single empty list for non-parameterized variants. Matches the
    zensim v0_20 screen's sweep schedules verbatim. Stacks combine
    inner + outer param sweeps; for the outer (winsor) step we
    recompute percentiles on the inner-transformed column at sweep
    time."""
    valid = col[np.isfinite(col)]
    if valid.size == 0:
        return [[]]
    if name == "clip_then_log1p":
        pcts = [5, 10, 25, 50, 75]
        return [[float(np.percentile(valid, p))] for p in pcts]
    if name == "winsor_p99":
        bounds = [(1, 99), (5, 95), (10, 90), (25, 75)]
        return [
            [float(np.percentile(valid, lo)), float(np.percentile(valid, hi))]
            for (lo, hi) in bounds
        ]
    if name == "quantile_bins":
        edges = [
            float(np.percentile(valid, p))
            for p in [12.5, 25, 37.5, 50, 62.5, 75, 87.5]
        ]
        return [edges]
    # Stacked variants: enumerate inner × outer param sweeps. Recompute
    # the OUTER winsor's percentiles on the inner-transformed column
    # so the clip bounds are in the right space (e.g. log-domain).
    if name in STACKS:
        return _stack_sweep_for(name, col)
    return [[]]


def _stack_sweep_for(name: str, col: np.ndarray) -> list[list[float]]:
    """Build the combinatorial param sweep for a stacked transform.

    For each inner-param config: compute the inner output, derive the
    outer winsor's percentile-based bounds on THAT distribution, then
    yield the concatenated `[inner_params..., outer_p1, outer_p99]`
    list. The outer winsor's bounds are picked from the standard
    (1,99) / (5,95) / (10,90) / (25,75) grid in the inner-transformed
    space.

    Stacks whose outer is non-winsor (winsor_then_*) sweep just the
    inner's params; the outer (log / signed_cbrt / log1p / signed_log1p)
    is unparameterized.
    """
    inner_n = STACK_INNER_NPARAMS[name]
    # Decide inner sweep based on inner identity.
    if name.startswith("winsor_then_"):
        # Inner = winsor_p99 (2 params); outer is unparameterized.
        valid = col[np.isfinite(col)]
        bounds = [(1, 99), (5, 95), (10, 90), (25, 75)]
        return [
            [float(np.percentile(valid, lo)), float(np.percentile(valid, hi))]
            for (lo, hi) in bounds
        ]
    if name == "clip_then_log1p_then_winsor":
        # Inner = clip_then_log1p (1 param ε), outer = winsor (2 params on
        # the log-transformed distribution).
        valid = col[np.isfinite(col)]
        eps_candidates = [
            float(np.percentile(valid, p)) for p in [5, 10, 25, 50, 75]
        ]
        out: list[list[float]] = []
        for eps in eps_candidates:
            # Apply inner once at this ε, compute outer percentiles on the
            # result. Outer sweep is the same (1,99) / (5,95) grid.
            y = t_clip_then_log1p(col, [eps])
            y_valid = y[np.isfinite(y)]
            if y_valid.size == 0:
                continue
            for (lo, hi) in [(1, 99), (5, 95), (10, 90)]:
                out.append([
                    eps,
                    float(np.percentile(y_valid, lo)),
                    float(np.percentile(y_valid, hi)),
                ])
        return out
    # log1p_then_winsor, log_then_winsor, signed_*_then_winsor:
    # inner unparameterized; outer winsor sweeps percentiles on the
    # inner-transformed distribution.
    valid = col[np.isfinite(col)]
    inner_fn_map = {
        "log1p_then_winsor": t_log1p,
        "log_then_winsor": t_log,
        "signed_log1p_then_winsor": t_signed_log1p,
        "signed_cbrt_then_winsor": t_signed_cbrt,
    }
    inner_fn = inner_fn_map[name]
    y = inner_fn(col, [])
    y_valid = y[np.isfinite(y)]
    if y_valid.size == 0:
        return [[]]
    bounds = [(1, 99), (5, 95), (10, 90), (25, 75)]
    return [
        [float(np.percentile(y_valid, lo)), float(np.percentile(y_valid, hi))]
        for (lo, hi) in bounds
    ]


# Merged transform vocabulary — singles + stacks. Screen iterates this.
ALL_TRANSFORMS: dict[str, Callable[[np.ndarray, list[float]], np.ndarray]] = {
    **TRANSFORMS,
    **STACKS,
}


# ---------------------------------------------------------------------------
# Screen scoring — aggregate Pearson lift across cells
# ---------------------------------------------------------------------------


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """|Pearson| over rows where both `a` and `b` are finite.
    Returns NaN when no rows survive."""
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n < 5:
        return float("nan")
    aa = a[finite]
    bb = b[finite]
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return 0.0
    c = float(np.corrcoef(aa, bb)[0, 1])
    return abs(c) if math.isfinite(c) else float("nan")


def aggregate_cell_pearson(
    feat_col: np.ndarray,
    bytes_log: np.ndarray,
    reach: np.ndarray,
) -> float:
    """Average |Pearson(feat, bytes_log[:, c])| over cells where the
    cell has ≥ 5 reachable rows. Aggregates per-cell signal so we
    catch features that distinguish *any* cell well, not just every
    cell uniformly. Returns NaN if no cell has enough samples."""
    n_cells = bytes_log.shape[1]
    cell_corrs: list[float] = []
    for c in range(n_cells):
        mask = reach[:, c]
        if mask.sum() < 5:
            continue
        corr = safe_pearson(feat_col[mask], bytes_log[mask, c])
        if not math.isnan(corr):
            cell_corrs.append(corr)
    if not cell_corrs:
        return float("nan")
    # Aggregate via mean (each cell weighs equally — codec cells already
    # represent meaningful encoder-config buckets, so per-cell signal is
    # the natural unit). Max-aggregate alternative tracked as a future
    # diagnostic axis.
    return float(np.mean(cell_corrs))


def screen_one_feature(
    feat_name: str,
    feat_col: np.ndarray,
    bytes_log: np.ndarray,
    reach: np.ndarray,
) -> dict:
    """Sweep every transform × param config on one feature column and
    return the per-feature winner — a record with the transform token,
    params, baseline (identity) score, transformed score, and lift."""
    finite = feat_col[np.isfinite(feat_col)]
    feat_min = float(finite.min()) if finite.size else 0.0
    baseline = aggregate_cell_pearson(feat_col, bytes_log, reach)
    best = {
        "feat_name": feat_name,
        "best_transform": "identity",
        "best_params": [],
        "baseline_score": baseline,
        "transformed_score": baseline,
        "lift": 0.0,
    }
    if math.isnan(baseline):
        return best

    for token, fn in ALL_TRANSFORMS.items():
        # Training-safety gates for raw-input log/log1p (single-step
        # variants). Stacks whose inner is log/log1p inherit the same
        # gate; stacks whose inner is winsor/signed_*/clip_then_log1p
        # are safe across the full real line by construction.
        if token == "log" and feat_min <= 0.0:
            continue
        if token == "log1p" and feat_min <= -1.0:
            continue
        if token == "log_then_winsor" and feat_min <= 0.0:
            continue
        if token == "log1p_then_winsor" and feat_min <= -1.0:
            continue
        for params in sweep_for(token, feat_col):
            tx = fn(feat_col, params)
            score = aggregate_cell_pearson(tx, bytes_log, reach)
            if math.isnan(score):
                continue
            lift = score - baseline
            if lift > best["lift"] + 1e-9:
                best = {
                    "feat_name": feat_name,
                    "best_transform": token,
                    "best_params": list(params),
                    "baseline_score": baseline,
                    "transformed_score": score,
                    "lift": lift,
                }
    return best


# ---------------------------------------------------------------------------
# Data loading — reuse train_hybrid pipeline
# ---------------------------------------------------------------------------


def load_codec_dataset(args) -> tuple[dict, dict, dict, dict, dict, list, dict, list]:
    """Run train_hybrid's data pipeline; return everything the sweep
    needs. `feat_cols` here is the codec's KEEP_FEATURES (post-filter
    against the available columns)."""
    TH.load_codec_config(args.codec_config)
    sys.stderr.write(f"[load] codec config: {args.codec_config}\n")
    pareto, ceilings, hcc, htc = TH.load_pareto(TH.PARETO)
    feats, feat_cols, feat_transforms = TH.load_features(TH.FEATURES)
    cells, _, c2c, parsed_all = TH.build_cell_index()
    time_baselines = (
        TH.compute_time_baselines(pareto) if htc else {}
    )
    (Xs, Xe, bytes_log, scalars, reach, meta, _tl, _ml, _infeas) = TH.build_dataset(
        pareto, feats, feat_cols, cells, c2c, parsed_all,
        ceilings=(ceilings if hcc else None),
        time_baselines=time_baselines if time_baselines else None,
    )

    # Image-level train/val split, same seed as train_hybrid main().
    rng = np.random.default_rng(args.seed)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * 0.20))
    val_set = set(images[:n_val])
    tr = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    va = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(
        f"[load] {len(Xs)} decision rows, {Xs.shape[1]} Xs cols, "
        f"{Xe.shape[1]} Xe cols. Train: {len(tr)}, Val: {len(va)}\n"
    )

    return {
        "Xs": Xs, "Xe": Xe,
        "bytes_log": bytes_log, "reach": reach,
        "meta": meta, "scalars": scalars,
        "cells": cells, "config_to_cell": c2c, "parsed_all": parsed_all,
        "feat_cols": feat_cols, "feat_transforms": feat_transforms,
        "train_idx": tr, "val_idx": va,
        "feats": feats,
    }


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------


def run_screen(data: dict) -> list[dict]:
    """Per-feature Pearson lift screen over all TRANSFORMS × param
    configs. Returns one record per feature, sorted by lift desc."""
    feat_cols = data["feat_cols"]
    bytes_log_tr = data["bytes_log"][data["train_idx"]]
    reach_tr = data["reach"][data["train_idx"]]
    # The codec's raw feature values live at the front of Xs:
    #   Xs = [feat_values (n_feats), size_oh (4), log_px, zq_norm]
    Xs_tr = data["Xs"][data["train_idx"]]
    n_feats = len(feat_cols)
    rows = []
    t0 = time.monotonic()
    for i, name in enumerate(feat_cols):
        col = Xs_tr[:, i].astype(np.float64)
        rec = screen_one_feature(name, col, bytes_log_tr, reach_tr)
        rec["feat_idx"] = i
        rows.append(rec)
    sys.stderr.write(
        f"[screen] {n_feats} features × {len(TRANSFORMS)} transforms "
        f"in {time.monotonic() - t0:.1f}s\n"
    )
    rows.sort(key=lambda r: -r["lift"])
    return rows


def write_screen_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "feat_idx", "feat_name", "best_transform", "params_csv",
            "baseline_score", "transformed_score", "lift",
        ])
        for r in rows:
            params_csv = ",".join(f"{v:.6g}" for v in r["best_params"]) \
                if r["best_params"] else ""
            w.writerow([
                r["feat_idx"], r["feat_name"], r["best_transform"], params_csv,
                f"{r['baseline_score']:.6f}",
                f"{r['transformed_score']:.6f}",
                f"{r['lift']:.6f}",
            ])


def write_recommended_py(rows: list[dict], path: Path, codec_name: str,
                          min_lift: float) -> None:
    """Write a Python snippet drop-in for the codec config.

    Emits:
      FEATURE_TRANSFORMS = {feat_name: "transform_token", ...}
      FEATURE_TRANSFORM_PARAMS = {feat_name: [param0, param1, ...], ...}
    """
    winners = [r for r in rows if r["lift"] >= min_lift
               and r["best_transform"] != "identity"]
    lines = [
        f"# Generated by feature_transform_sweep.py (zensim v0_20-style screen).",
        f"# Codec: {codec_name}; lift filter: >= {min_lift}",
        f"# Per-feature Pearson lift winners across the full",
        f"# zenpredict::FeatureTransform vocabulary.",
        "",
        "FEATURE_TRANSFORMS = {",
    ]
    for r in winners:
        lines.append(
            f'    "{r["feat_name"]}": "{r["best_transform"]}",   '
            f'# lift={r["lift"]:+.4f}'
        )
    lines.append("}")
    # Params dict — only entries that actually need params.
    have_params = [r for r in winners if r["best_params"]]
    if have_params:
        lines.append("")
        lines.append("FEATURE_TRANSFORM_PARAMS = {")
        for r in have_params:
            ps = ", ".join(f"{v:.6g}" for v in r["best_params"])
            lines.append(f'    "{r["feat_name"]}": [{ps}],')
        lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# End-to-end confirmation — train baseline + recommended student MLPs
# ---------------------------------------------------------------------------


def apply_transforms_to_features(
    data: dict,
    transforms_map: dict[str, str],
    params_map: dict[str, list[float]],
) -> dict:
    """Return a copy of `data` with Xs / Xe rebuilt under the given
    per-feature transforms. The transforms are applied to the codec's
    raw feature columns (positions 0..n_feats-1 in Xs); size_oh +
    log_px + zq_norm cross-terms in Xe are untouched.

    Implementation note: we don't re-run build_dataset (~10 s). We
    rebuild Xs / Xe in-place from `feats` + meta — same shape, faster.
    """
    n_feats = len(data["feat_cols"])
    Xs = data["Xs"].copy()
    Xe = data["Xe"].copy()
    # Rebuild Xs[:, :n_feats] and the cross-term block in Xe
    # under the new per-feature transforms. The cross-term block in Xe
    # is `zq_norm × feat[i]` placed AFTER the size_oh/log_px/zq cross-
    # terms; we need to detect its offset.
    #
    # Layout of Xe (from train_hybrid.build_dataset):
    #   [f (n_feats), size_oh (4), [log_px, log_px², zq, zq², zq×log_px] (5),
    #    zq_norm × f (n_feats), icc_placeholder (1)]
    cross_start = n_feats + 4 + 5
    cross_end = cross_start + n_feats
    zq_norm_col = data["Xs"][:, n_feats + 4 + 1]  # in Xs layout: log_px, zq_norm
    for i, name in enumerate(data["feat_cols"]):
        if name not in transforms_map:
            continue
        token = transforms_map[name]
        params = params_map.get(name, [])
        fn = ALL_TRANSFORMS[token]
        new_col = fn(Xs[:, i].astype(np.float64), params).astype(np.float32)
        # NaN guard — replace any NaN with the mean of the finite
        # entries so the scaler doesn't blow up.
        finite = np.isfinite(new_col)
        if not finite.all():
            fill = np.nanmean(new_col[finite]) if finite.any() else 0.0
            new_col = np.where(finite, new_col, fill).astype(np.float32)
        Xs[:, i] = new_col
        Xe[:, i] = new_col
        Xe[:, cross_start + i] = (zq_norm_col * new_col).astype(np.float32)
    return {**data, "Xs": Xs, "Xe": Xe}


def train_and_eval_student(
    data: dict,
    epochs: int = 100,
    hidden_sizes: tuple[int, ...] = (192, 192, 192),
    seed: int = 0xCAFE,
    device: str | None = None,
) -> dict:
    """End-to-end confirmation: train a leakyrelu student against
    HistGB teacher soft targets, evaluate val argmin acc + mean
    overhead. Returns the metric record.

    This is the same shape `train_hybrid.py` produces but compressed —
    fewer epochs, no safety report, no bake JSON. Fast enough that
    we can A/B baseline vs recommended in ~minutes total.
    """
    import torch
    import torch.nn as nn

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr = data["train_idx"]
    va = data["val_idx"]
    Xs_tr = data["Xs"][tr]
    Xe_tr = data["Xe"][tr]
    Xe_va = data["Xe"][va]
    bl_tr = data["bytes_log"][tr]
    bl_va = data["bytes_log"][va]
    rch_tr = data["reach"][tr]
    rch_va = data["reach"][va]
    cells = data["cells"]
    n_cells = len(cells)
    meta_va = [data["meta"][i] for i in va]

    # Teacher — HistGB per cell (same as train_hybrid.train_teacher_per_cell
    # path; we restrict to bytes head only for speed in the screen).
    sys.stderr.write("  [confirm] training HistGB teacher (bytes head only)\n")
    t0 = time.monotonic()
    from _picker_lib import HISTGB_FAST, train_teachers_per_cell_parallel
    t_bytes = train_teachers_per_cell_parallel(
        Xs_tr, bl_tr, rch_tr, params=HISTGB_FAST, label="bytes",
    )
    fallback = np.nanmean(bl_tr, axis=0)
    teacher_pred_tr = TH.teacher_predict_all(t_bytes, Xs_tr, fallback, n_cells)
    teacher_pred_va = TH.teacher_predict_all(t_bytes, data["Xs"][va],
                                              fallback, n_cells)
    sys.stderr.write(
        f"  [confirm] teacher done in {time.monotonic() - t0:.1f}s\n"
    )

    # Teacher diagnostic: val argmin acc as an upper bound for the student.
    all_mask = np.ones(n_cells, dtype=bool)
    t_argmin = TH.evaluate_argmin(
        teacher_pred_va, bl_va, rch_va, meta_va, all_mask
    )

    # Student — small leakyrelu MLP on (Xe, teacher soft bytes).
    sys.stderr.write(
        f"  [confirm] training student (hidden={hidden_sizes}, "
        f"epochs={epochs}) on {device}\n"
    )
    n_in = Xe_tr.shape[1]
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(0.01))
        prev = h
    layers.append(nn.Linear(prev, n_cells))
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.from_numpy(Xe_tr.astype(np.float32)).to(device)
    Yt = torch.from_numpy(teacher_pred_tr.astype(np.float32)).to(device)
    Xv = torch.from_numpy(Xe_va.astype(np.float32)).to(device)
    Yv = torch.from_numpy(teacher_pred_va.astype(np.float32)).to(device)
    B = 4096
    t0 = time.monotonic()
    best_val = math.inf
    bad = 0
    patience = 20
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        for b in range(0, len(perm), B):
            sel = perm[b:b + B]
            pred = net(Xt[sel])
            target = Yt[sel]
            mask = torch.isfinite(target)
            diff = torch.where(mask, pred - target, torch.zeros_like(pred))
            loss = (diff * diff).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pred_v = net(Xv)
            mask_v = torch.isfinite(Yv)
            diff_v = torch.where(mask_v, pred_v - Yv, torch.zeros_like(pred_v))
            val_mse = ((diff_v * diff_v).sum() / mask_v.sum().clamp(min=1)).item()
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    fit_s = time.monotonic() - t0
    net.eval()
    with torch.no_grad():
        student_pred_va = net(Xv).cpu().numpy()
    s_argmin = TH.evaluate_argmin(
        student_pred_va, bl_va, rch_va, meta_va, all_mask
    )
    return {
        "teacher_argmin_acc": float(t_argmin.get("argmin_acc", 0)),
        "teacher_mean_overhead_pct": float(t_argmin.get("mean_pct", 0)),
        "student_argmin_acc": float(s_argmin.get("argmin_acc", 0)),
        "student_mean_overhead_pct": float(s_argmin.get("mean_pct", 0)),
        "student_p99_overhead_pct": float(s_argmin.get("p99_pct", 0)),
        "fit_seconds": float(fit_s),
        "best_val_mse": float(best_val),
        "epochs_ran": int(epoch + 1),
        "device": device,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--codec-config", required=True,
                    help="Codec config module (e.g. zenjpeg_picker_config)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory")
    ap.add_argument("--min-lift", type=float, default=0.005,
                    help="Emit recommendation rows with lift >= this "
                    "(default 0.005, matches zensim's screen)")
    ap.add_argument("--confirm", action="store_true",
                    help="Train baseline vs recommended student MLPs "
                    "and measure end-to-end val argmin acc delta")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Confirmation training epochs (default 100)")
    ap.add_argument("--hidden", default="192,192,192",
                    help="Confirmation student hidden widths")
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0xCAFE)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Phase 1: load + screen.
    data = load_codec_dataset(args)
    rows = run_screen(data)

    screen_path = args.out / "screen_results.tsv"
    write_screen_tsv(rows, screen_path)
    sys.stderr.write(f"[write] {screen_path}\n")

    rec_path = args.out / "recommended_transforms.py"
    write_recommended_py(rows, rec_path, args.codec_config, args.min_lift)
    sys.stderr.write(f"[write] {rec_path}\n")

    n_winners = sum(1 for r in rows if r["lift"] >= args.min_lift
                    and r["best_transform"] != "identity")
    n_total = len(rows)
    sys.stderr.write(
        f"\n[screen] {n_winners}/{n_total} features have lift >= {args.min_lift}\n"
    )
    sys.stderr.write("Top 10 by lift:\n")
    for r in rows[:10]:
        params_csv = ",".join(f"{v:.6g}" for v in r["best_params"]) \
            if r["best_params"] else "-"
        sys.stderr.write(
            f"  {r['feat_name']:<40s} {r['best_transform']:<16s} "
            f"params={params_csv:<32s} "
            f"base={r['baseline_score']:.3f} tx={r['transformed_score']:.3f} "
            f"lift={r['lift']:+.4f}\n"
        )

    # Phase 2: end-to-end confirmation (optional).
    confirmation: dict = {}
    if args.confirm:
        sys.stderr.write("\n[confirm] phase 2: end-to-end A/B vs baseline\n")
        hidden = tuple(int(x) for x in args.hidden.split(","))

        # Baseline — apply the codec's CURRENT FEATURE_TRANSFORMS.
        baseline_transforms = dict(getattr(TH, "FEATURE_TRANSFORMS", {}))
        baseline_params: dict[str, list[float]] = {}
        sys.stderr.write(
            f"[baseline] codec's current FEATURE_TRANSFORMS: "
            f"{len(baseline_transforms)} entries\n"
        )
        baseline_data = apply_transforms_to_features(
            data, baseline_transforms, baseline_params
        )
        baseline_metrics = train_and_eval_student(
            baseline_data, epochs=args.epochs, hidden_sizes=hidden, seed=args.seed,
        )

        # Recommended — apply screen winners.
        rec_transforms = {
            r["feat_name"]: r["best_transform"]
            for r in rows if r["lift"] >= args.min_lift
            and r["best_transform"] != "identity"
        }
        rec_params = {
            r["feat_name"]: r["best_params"]
            for r in rows if r["lift"] >= args.min_lift and r["best_params"]
        }
        sys.stderr.write(
            f"[recommended] screen winners: {len(rec_transforms)} entries\n"
        )
        rec_data = apply_transforms_to_features(data, rec_transforms, rec_params)
        rec_metrics = train_and_eval_student(
            rec_data, epochs=args.epochs, hidden_sizes=hidden, seed=args.seed,
        )

        confirmation = {
            "baseline": baseline_metrics,
            "recommended": rec_metrics,
            "delta": {
                "argmin_acc_pp": rec_metrics["student_argmin_acc"]
                                  - baseline_metrics["student_argmin_acc"],
                "mean_overhead_pp": rec_metrics["student_mean_overhead_pct"]
                                     - baseline_metrics["student_mean_overhead_pct"],
            },
            "baseline_transforms": baseline_transforms,
            "recommended_transforms": rec_transforms,
            "recommended_params": rec_params,
        }
        sys.stderr.write(
            "\n=== A/B summary ===\n"
            f"  baseline    argmin_acc {baseline_metrics['student_argmin_acc']:.1%}  "
            f"mean_overhead {baseline_metrics['student_mean_overhead_pct']:.2f}%\n"
            f"  recommended argmin_acc {rec_metrics['student_argmin_acc']:.1%}  "
            f"mean_overhead {rec_metrics['student_mean_overhead_pct']:.2f}%\n"
            f"  delta       argmin_acc {confirmation['delta']['argmin_acc_pp']:+.4f}pp  "
            f"mean_overhead {confirmation['delta']['mean_overhead_pp']:+.4f}pp\n"
        )
        (args.out / "confirmation_summary.json").write_text(
            json.dumps(confirmation, indent=2)
        )
        sys.stderr.write(
            f"[write] {args.out / 'confirmation_summary.json'}\n"
        )

    # Summary markdown.
    md_lines = [
        f"# Feature-transform sweep — {args.codec_config}",
        "",
        f"Generated by `feature_transform_sweep.py` "
        f"({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}). "
        f"Methodology: zensim v0_20 greedy Pearson screen "
        "(`scripts/v_next/v0_20_feature_transform_greedy_screen.py`) "
        "adapted to picker per-cell `bytes_log` labels.",
        "",
        f"## Screen results — {n_winners} / {n_total} features have "
        f"lift ≥ {args.min_lift}",
        "",
        "| Feature | Best transform | Params | Baseline | Transformed | Lift |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        if r["best_transform"] == "identity":
            continue
        params_csv = ",".join(f"{v:.6g}" for v in r["best_params"]) \
            if r["best_params"] else "—"
        md_lines.append(
            f"| `{r['feat_name']}` | `{r['best_transform']}` | "
            f"`{params_csv}` | {r['baseline_score']:.4f} | "
            f"{r['transformed_score']:.4f} | {r['lift']:+.4f} |"
        )
    if confirmation:
        md_lines += [
            "",
            "## End-to-end confirmation",
            "",
            "Trained baseline (codec's current `FEATURE_TRANSFORMS`) vs "
            "recommended (per-feature screen winners). Same teacher, "
            "same student architecture, same val split.",
            "",
            "| | Baseline | Recommended | Δ |",
            "|---|---:|---:|---:|",
            f"| Val argmin accuracy | {confirmation['baseline']['student_argmin_acc']:.1%} "
            f"| {confirmation['recommended']['student_argmin_acc']:.1%} "
            f"| **{confirmation['delta']['argmin_acc_pp']:+.4f}pp** |",
            f"| Val mean overhead | {confirmation['baseline']['student_mean_overhead_pct']:.2f}% "
            f"| {confirmation['recommended']['student_mean_overhead_pct']:.2f}% "
            f"| **{confirmation['delta']['mean_overhead_pp']:+.4f}pp** |",
        ]
    md_lines += [
        "",
        "## Recommended snippet (drop into codec config)",
        "",
        "```python",
        Path(rec_path).read_text().rstrip("\n"),
        "```",
        "",
        "## Caveats",
        "",
        "- **Greedy per-feature.** The screen scores transforms one "
        "feature at a time. Cross-feature interactions are NOT captured "
        "in the Pearson aggregate. The end-to-end confirmation step "
        "(`--confirm`) catches interaction-driven regressions.",
        "- **Pearson aggregate over cells.** A feature that strongly "
        "separates 2/12 cells but is noise on the other 10 will look "
        "weaker than a feature that distinguishes all cells modestly. "
        "Picker argmin only needs the *one* right cell — the gold-"
        "standard confirmation captures this; the screen is a fast "
        "filter, not a final answer.",
        "- **Parameter sweeps are coarse.** `clip_then_log1p` ε at "
        "[p5,p10,p25,p50,p75], `winsor_p99` bounds at (1,99)/(5,95)/"
        "(10,90)/(25,75), `quantile_bins` at 8 fixed edges. Production "
        "tuning may want denser sweeps for the top-N candidate features.",
    ]
    (args.out / "summary.md").write_text("\n".join(md_lines) + "\n")
    sys.stderr.write(f"[write] {args.out / 'summary.md'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
