#!/usr/bin/env python3
"""
Hybrid-heads picker training — codec-agnostic in shape.

Splits the codec's full config grid into:
  - N categorical cells (some combination of discrete-only axes:
    color_mode, subsampling, trellis_on/off, sa_piecewise, …)
  - K continuous predictions per cell (e.g. chroma_scale, lambda,
    effort) extracted from the within-cell-optimal config

For each (image, size, target_zq), compute the within-cell optimal:
  bytes(cell)        = min bytes over configs in cell that reach zq
  scalar_<i>(cell)   = scalar value of the within-cell optimal config

Train an MLP with `(K + 1) × N` outputs:
  bytes head:   N log-bytes outputs  (categorical pick target)
  scalar heads: N×K continuous outputs

At inference:
  Y = picker.predict(features)
  bytes_log = Y[0..N]
  scalar_<i> = Y[N*(i+1) .. N*(i+2)]   for i in 0..K
  cell_idx = argmin(bytes_log, mask=allowed_cells)
  encoder_config = build_from(CELLS[cell_idx], scalar_<i>[cell_idx], …)

The model learns *Pareto-optimal scalars* per cell. The codec
consumer clamps to caller constraints at inference.

# Codec-side adapter

The codec supplies one Python module that exports:

    PARETO          — Path to the Pareto sweep TSV
    FEATURES        — Path to the analyzer features TSV
    OUT_JSON        — Output path for the trained model JSON
    OUT_LOG         — Output path for the training summary
    KEEP_FEATURES   — list[str] of `feat_*` column names to use
    ZQ_TARGETS      — list[int] of target_zq values to model
    parse_config_name(name: str) -> dict
        Returns a dict with at least:
          - one or more `categorical_axes` (hashable values)
          - one or more `scalar_axes` (float values; sentinel for
            "not applicable" cells, e.g. lambda=0 in noT cells)

Run with:
    python3 train_hybrid.py --codec-config zenjpeg_picker_config

A reference codec config lives at `examples/zenjpeg_picker_config.py`.
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
from sklearn.neural_network import MLPRegressor


# --- LeakyReLU student via PyTorch -------------------------------------------
# sklearn's MLPRegressor only supports {identity, logistic, tanh, relu}, so a
# leakyrelu run falls through to a small PyTorch student that mimics the
# sklearn fit / predict / coefs_ / intercepts_ surface. This lets the rest of
# train_hybrid (safety_check, diagnostics, JSON serialization) work unchanged.
def _train_torch_leakyrelu_student(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    hidden_layer_sizes: tuple,
    lr: float,
    batch_size: int,
    max_iter: int,
    seed: int,
    val_frac: float = 0.1,
    n_iter_no_change: int = 30,
    tol: float = 1e-6,
    leaky_slope: float = 0.01,
):
    """Drop-in for `MLPRegressor.fit` returning an object that exposes
    `.coefs_`, `.intercepts_`, `.predict`, `.loss_`, `.n_iter_`. The
    network shape, init, loss (MSE), optimizer (Adam), and early-stopping
    schedule mirror sklearn's defaults so the comparison stays apples-to-apples.
    """
    import torch  # lazy import — only needed when --activation leakyrelu
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hold out an internal val slice for early stopping (matches sklearn's
    # validation_fraction). The split is row-shuffled, not image-aware —
    # the outer image-level holdout is already separated upstream.
    n = X_tr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xt = torch.from_numpy(X_tr[tr_idx].astype(np.float32))
    Yt = torch.from_numpy(Y_tr[tr_idx].astype(np.float32))
    Xv = torch.from_numpy(X_tr[val_idx].astype(np.float32))
    Yv = torch.from_numpy(Y_tr[val_idx].astype(np.float32))

    n_in = X_tr.shape[1]
    n_out = Y_tr.shape[1]
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden_layer_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(negative_slope=leaky_slope))
        prev = h
    layers.append(nn.Linear(prev, n_out))
    net = nn.Sequential(*layers)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    last_loss = float("inf")
    for epoch in range(max_iter):
        net.train()
        perm_e = torch.randperm(Xt.shape[0])
        for i in range(0, Xt.shape[0], batch_size):
            idx = perm_e[i : i + batch_size]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xv), Yv).item()
        last_loss = v
        if v < best_val - tol:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= n_iter_no_change:
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    # Build sklearn-compatible coefs_ / intercepts_ lists.
    coefs_: list[np.ndarray] = []
    intercepts_: list[np.ndarray] = []
    for layer in net:
        if isinstance(layer, nn.Linear):
            # PyTorch nn.Linear stores weight as (out_features, in_features);
            # sklearn's coefs_[i] is (in_features, out_features) — transpose.
            coefs_.append(layer.weight.detach().cpu().numpy().T.astype(np.float64))
            intercepts_.append(layer.bias.detach().cpu().numpy().astype(np.float64))

    class _TorchStudent:
        def __init__(self):
            self.coefs_ = coefs_
            self.intercepts_ = intercepts_
            self.loss_ = best_val
            self.n_iter_ = epoch + 1
            self._net = net

        def predict(self, X: np.ndarray) -> np.ndarray:
            self._net.eval()
            with torch.no_grad():
                t = torch.from_numpy(X.astype(np.float32))
                return self._net(t).cpu().numpy().astype(np.float32)

    return _TorchStudent()
from sklearn.preprocessing import StandardScaler

SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

CONFIG_NAMES: dict = {}

# These are bound from the loaded codec config in main(). Module-
# level placeholders so the helper functions below can name them.
PARETO: Path
FEATURES: Path
OUT_LOG: Path
OUT_JSON: Path
ZQ_TARGETS: list
KEEP_FEATURES: list
parse_config_name = None  # type: ignore[assignment]

# Codec-driven axis schema. The codec config exports CATEGORICAL_AXES
# (list[str] — keys of parsed dict that form the cell tuple) and
# SCALAR_AXES (list[str] — keys of parsed dict that become per-cell
# scalar prediction heads). Default to zenjpeg's shape so configs that
# pre-date the explicit declaration keep working.
CATEGORICAL_AXES: list = ["color", "sub", "trellis_on", "sa"]
SCALAR_AXES: list = ["chroma_scale", "lambda"]
# Per-axis sentinel values for "not applicable" rows. The trainer
# masks rows where actual_value <= sentinel out of that axis's per-cell
# regression so the model doesn't learn from sentinel placeholders.
# Default mirrors zenjpeg's lambda<=0 → trellis-off semantics.
SCALAR_SENTINELS: dict = {"lambda": 0.0}
# Per-axis (min, max) ranges shown in the training log next to RMSE,
# purely for human readability. Optional.
SCALAR_DISPLAY_RANGES: dict = {
    "chroma_scale": (0.6, 1.5),
    "lambda": (8.0, 25.0),
}

# Quality-metric column on the pareto TSV that the picker is trained
# against. Defaults to "zensim" for back-compat with existing zenjpeg
# bakes. Codec configs that target butteraugli, ssim2, dssim, etc.
# override this. Per-bake choice — different bakes per metric.
METRIC_COLUMN: str = "zensim"

# Direction of the metric. Reachability: a config "reaches" the target
# when its metric value satisfies the direction-appropriate inequality:
#   - "higher_better": metric >= target  (zensim, ssim2, psnr, …)
#   - "lower_better":  metric <= target  (butteraugli, dssim, mse, …)
METRIC_DIRECTION: str = "higher_better"

# Encode-time column on the pareto TSV. Picker training optionally adds
# a per-cell `time_log` head from this column for the time_budgeted
# objective. Defaults to "encode_ms" (matches existing harnesses).
TIME_COLUMN: str = "encode_ms"


def load_codec_config(name: str):
    """Import a codec-config module and bind its exports to module-level
    names this script consumes. The codec module must define:
      PARETO, FEATURES, OUT_JSON, OUT_LOG, ZQ_TARGETS, KEEP_FEATURES,
      parse_config_name(name: str) -> dict.

    Optional codec-config exports (recommended for non-zenjpeg codecs):
      CATEGORICAL_AXES: list[str] — parsed-dict keys forming the cell
                                    tuple. Defaults to zenjpeg's shape
                                    `["color", "sub", "trellis_on", "sa"]`.
      SCALAR_AXES:      list[str] — parsed-dict keys that become per-cell
                                    scalar prediction heads. Defaults to
                                    `["chroma_scale", "lambda"]`.
      SCALAR_SENTINELS: dict[str, float] — per-axis sentinels. Rows with
                                    `actual_value <= sentinel` are masked
                                    out of that axis's per-cell teacher.
                                    Defaults to `{"lambda": 0.0}`.
      SCALAR_DISPLAY_RANGES: dict[str, (float, float)] — log formatting.

    `parse_config_name` returns a dict whose keys partition into:
      - categorical axes (hashable values, used to form cells)
      - scalar axes (float values; sentinel allowed for "not
        applicable" — e.g. lambda=0.0 in trellis-off cells)
    """
    global PARETO, FEATURES, OUT_LOG, OUT_JSON
    global ZQ_TARGETS, KEEP_FEATURES, parse_config_name
    global CATEGORICAL_AXES, SCALAR_AXES, SCALAR_SENTINELS, SCALAR_DISPLAY_RANGES
    global METRIC_COLUMN, METRIC_DIRECTION, TIME_COLUMN
    mod = importlib.import_module(name)
    PARETO = Path(mod.PARETO)
    FEATURES = Path(mod.FEATURES)
    OUT_LOG = Path(mod.OUT_LOG)
    OUT_JSON = Path(mod.OUT_JSON)
    ZQ_TARGETS = list(mod.ZQ_TARGETS)
    KEEP_FEATURES = list(mod.KEEP_FEATURES)
    parse_config_name = mod.parse_config_name
    # Optional axis schema — fall back to module defaults (zenjpeg shape)
    # when the codec config doesn't declare. Pre-existing zenjpeg config
    # keeps working without changes.
    if hasattr(mod, "CATEGORICAL_AXES"):
        CATEGORICAL_AXES = list(mod.CATEGORICAL_AXES)
    if hasattr(mod, "SCALAR_AXES"):
        SCALAR_AXES = list(mod.SCALAR_AXES)
    if hasattr(mod, "SCALAR_SENTINELS"):
        SCALAR_SENTINELS = dict(mod.SCALAR_SENTINELS)
    elif hasattr(mod, "CATEGORICAL_AXES") or hasattr(mod, "SCALAR_AXES"):
        # Codec explicitly declared schema → don't inherit zenjpeg's
        # lambda sentinel by default.
        SCALAR_SENTINELS = {}
    if hasattr(mod, "SCALAR_DISPLAY_RANGES"):
        SCALAR_DISPLAY_RANGES = dict(mod.SCALAR_DISPLAY_RANGES)
    elif hasattr(mod, "CATEGORICAL_AXES") or hasattr(mod, "SCALAR_AXES"):
        SCALAR_DISPLAY_RANGES = {}
    # Quality metric column + direction. Optional — defaults to zensim
    # / higher_better which match every existing zenjpeg / zenwebp /
    # zenavif config. Codecs targeting butteraugli / dssim override
    # both (`METRIC_COLUMN = "butteraugli"`, `METRIC_DIRECTION = "lower_better"`).
    if hasattr(mod, "METRIC_COLUMN"):
        METRIC_COLUMN = str(mod.METRIC_COLUMN)
    if hasattr(mod, "METRIC_DIRECTION"):
        d = str(mod.METRIC_DIRECTION).lower()
        if d not in ("higher_better", "lower_better"):
            raise ValueError(
                f"METRIC_DIRECTION must be 'higher_better' or 'lower_better', got {d!r}"
            )
        METRIC_DIRECTION = d
    if hasattr(mod, "TIME_COLUMN"):
        TIME_COLUMN = str(mod.TIME_COLUMN)
    return mod


# ---------- Config-name parser (codec-supplied) ----------
#
# `parse_config_name` is bound by `load_codec_config()`. The codec
# config module owns the regex/pattern that parses its own
# config-name convention into a dict of categorical + scalar axes.
# See `examples/zenjpeg_picker_config.py` for a reference.
#
# (Intentionally empty — function lives in the codec config.)


def _placeholder_parse_config_name(name: str) -> dict:
    """Stub returning the same shape as `parse_config_name` would.
    Only here so static analysis can see the contract; never called.
    """
    return {
        "color": "",
        "sub": "",
        "sa": False,
        "trellis_on": False,
        "lambda": 0.0,
        "chroma_scale": 0.0,
    }


def categorical_key(parsed: dict) -> tuple:
    """The cell-forming tuple, driven by the codec's CATEGORICAL_AXES.

    For zenjpeg (default): `(color, sub, trellis_on, sa)`.
    For zenwebp: `(method, segments)`.
    """
    return tuple(parsed[axis] for axis in CATEGORICAL_AXES)


def cell_label_from_key(key: tuple) -> str:
    """Build a human-readable label by joining axis values with `_`.

    For zenjpeg `(ycbcr, 444, True, False)` → `ycbcr_444_True_False`.
    For zenwebp `(4, 1)` → `m4_seg1` via the zenjpeg-historical
    short-hand (color/sub/trellis/sa get special-cased so the label
    matches what the existing zenjpeg report expects). For unknown
    axis schemas we just stringify each component and join with `_`.
    """
    # Special-case the zenjpeg axis order so the report keeps
    # producing labels like `ycbcr_444_trellis_sa` (matches v2.1 logs).
    if list(CATEGORICAL_AXES) == ["color", "sub", "trellis_on", "sa"]:
        color, sub, trellis_on, sa = key
        sa_tag = "_sa" if sa else ""
        trel_tag = "trellis" if trellis_on else "noT"
        return f"{color}_{sub}_{trel_tag}{sa_tag}"
    # Generic path — `{axis}{value}` per component.
    return "_".join(_render_axis_value(axis, v) for axis, v in zip(CATEGORICAL_AXES, key))


def _render_axis_value(axis: str, value) -> str:
    """Compact label for a categorical axis value."""
    if isinstance(value, bool):
        return f"{axis}={int(value)}"
    if isinstance(value, (int, float)):
        return f"{axis}{value}"
    return str(value)


# ---------- Data loading ----------


def load_pareto(path):
    """Load the Pareto sweep TSV.

    Returns `(rows, ceilings, has_ceiling_column, has_time_column)` where:
      - rows: `{(image_path, size_class, w, h) -> [{"config_id", "bytes",
        "zensim"[, "time_ms"]}]}`. The `"zensim"` key holds the value
        from `METRIC_COLUMN` (which defaults to "zensim" but can be
        any quality column the codec targets — butteraugli, dssim, …).
        Direction is interpreted via `METRIC_DIRECTION`.
      - ceilings: `{(image_path, size_class) -> effective_max_zensim}` —
        only populated if the sweep TSV declares the
        `effective_max_zensim` column. The value is the max metric score
        physically achievable for that image at that size, computed by
        the sweep harness as a byproduct (typically `max(metric)` across
        all configs at the highest q values when METRIC_DIRECTION is
        higher_better, or `min(metric)` for lower_better).
      - has_ceiling_column: True iff the sweep TSV declared
        `effective_max_zensim`.
      - has_time_column: True iff the sweep TSV declared the
        configured TIME_COLUMN. When True, rows carry a `"time_ms"`
        key (float, milliseconds). Drives the time_budgeted objective's
        per-cell time head.

    See imazen/zenanalyze#51 for the cross-codec design context.
    """
    rows = defaultdict(list)
    ceilings: dict = {}
    has_ceiling_column = False
    has_time_column = False
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        has_ceiling_column = (
            rdr.fieldnames is not None and "effective_max_zensim" in rdr.fieldnames
        )
        has_time_column = (
            rdr.fieldnames is not None and TIME_COLUMN in rdr.fieldnames
        )
        if METRIC_COLUMN not in (rdr.fieldnames or []):
            raise ValueError(
                f"pareto TSV {path} is missing METRIC_COLUMN={METRIC_COLUMN!r}; "
                f"available columns: {rdr.fieldnames}"
            )
        for r in rdr:
            try:
                cid = int(r["config_id"])
                bytes_v = int(r["bytes"])
                metric_v = float(r[METRIC_COLUMN])
            except (ValueError, KeyError):
                continue
            CONFIG_NAMES.setdefault(cid, r["config_name"])
            key = (r["image_path"], r["size_class"], int(r["width"]), int(r["height"]))
            row = {"config_id": cid, "bytes": bytes_v, "zensim": metric_v}
            if has_time_column:
                t_raw = r.get(TIME_COLUMN, "")
                if t_raw:
                    try:
                        row["time_ms"] = float(t_raw)
                    except ValueError:
                        pass
            rows[key].append(row)
            if has_ceiling_column:
                cv = r.get("effective_max_zensim", "")
                if cv:
                    try:
                        # Per-(image, size) — value is identical across
                        # all (config, q) rows of that key. Take first.
                        ceil_key = (r["image_path"], r["size_class"])
                        if ceil_key not in ceilings:
                            ceilings[ceil_key] = float(cv)
                    except ValueError:
                        pass
    return rows, ceilings, has_ceiling_column, has_time_column


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


# ---------- Build categorical cell mapping ----------


def build_cell_index():
    """Return:
       cells: list of dicts describing each cell (in stable order).
              Each carries `id`, `label`, `member_config_ids`, plus
              one entry per CATEGORICAL_AXES axis with its value.
       cell_id_by_key: {tuple -> int}
       config_to_cell: {config_id -> cell_id}
       config_to_parsed: {config_id -> parsed dict}
    """
    parsed_all = {}
    for cid, name in CONFIG_NAMES.items():
        parsed_all[cid] = parse_config_name(name)

    keys = sorted({categorical_key(p) for p in parsed_all.values()})
    cell_id_by_key = {k: i for i, k in enumerate(keys)}

    cells = []
    for k in keys:
        label = cell_label_from_key(k)
        members = [cid for cid, p in parsed_all.items() if categorical_key(p) == k]
        cell = {
            "id": cell_id_by_key[k],
            "label": label,
            "member_config_ids": sorted(members),
        }
        # Carry each categorical axis value back into the cell dict so
        # downstream consumers (codec runtime, manifest readers) can
        # reconstruct the encoder config from the cell index alone.
        for axis, value in zip(CATEGORICAL_AXES, k):
            cell[axis] = value
        cells.append(cell)

    config_to_cell = {cid: cell_id_by_key[categorical_key(p)] for cid, p in parsed_all.items()}
    return cells, cell_id_by_key, config_to_cell, parsed_all


# ---------- Build training dataset ----------


def rows_have_time(pareto):
    """True iff any row in the pareto dict carries a `time_ms` key.
    Used to decide whether to emit the per-cell time_log head."""
    for samples in pareto.values():
        for s in samples:
            if "time_ms" in s:
                return True
        # Only need to inspect the first nonempty sample list — rows in
        # one (image, size) cell come from the same TSV pass, so the
        # column-presence is uniform.
        if samples:
            break
    return False


def compute_time_baselines(pareto):
    """Median `time_ms` per size_class across all samples in the
    dataset. Returned as `{size_class: median_ms}`. Used to compute
    the per-(image, size_class) budget when --time-budget-multiplier > 0
    filters within-cell candidates."""
    by_size: dict = defaultdict(list)
    for (image, size, w, h), samples in pareto.items():
        for s in samples:
            t = s.get("time_ms")
            if t is not None and not math.isinf(t) and t > 0:
                by_size[size].append(t)
    baselines = {}
    for sz, vals in by_size.items():
        if vals:
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            baselines[sz] = (
                vals_sorted[mid]
                if len(vals_sorted) % 2 == 1
                else 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
            )
    return baselines


def build_dataset(
    pareto,
    feats,
    feat_cols,
    cells,
    config_to_cell,
    parsed_all,
    ceilings=None,
    *,
    time_budget_multiplier: float = 0.0,
    time_baselines: dict | None = None,
    emit_metric_head: bool = False,
):
    """Per (image, size, zq) row, compute within-cell optimal:
       bytes_log[c]    = log(min bytes in cell c over configs that reach zq)
       scalars[axis][c] = scalar value of the within-cell optimal for axis
       reachable[c]    = 1 if any config in cell c reached zq, 0 otherwise

    `ceilings`: optional `{(image, size_class) -> effective_max_zensim}`.
    When provided, skips `target_zq > effective_max_zensim[image, size]
    + CEILING_MARGIN` cells — those targets are physically unreachable
    for that (image, size) and produce only data-starvation noise. See
    imazen/zenanalyze#51.

    Returns (Xs, Xe, bytes_log, scalars, reach, meta, time_log,
    metric_log, infeasible) where:
      - `scalars` is `dict[axis_name -> ndarray(n_rows, n_cells)]`.
      - `time_log` is `ndarray(n_rows, n_cells)` of `log(encode_ms)` for
        the within-cell-best config (NaN where cell did not reach
        target). None when the sweep TSV has no time column.
      - `metric_log` is `ndarray(n_rows, n_cells)` of
        `log(metric_value)` (or `metric_value` for direction-agnostic
        targets) for the within-cell-best config. None when
        `emit_metric_head=False`.
      - `infeasible` is `{(image, size_class): True}` for (image, size)
        pairs where every cell is over the time budget at every zq
        target. Empty when no budget filter is in effect. Drives the
        BUDGET_INFEASIBLE safety gate.

    `time_budget_multiplier` (default 0.0 = no filter): when > 0, only
    configs whose `time_ms <= time_baselines[size_class] * multiplier`
    are eligible as within-cell candidates. `time_baselines` must be
    provided (use `compute_time_baselines(pareto)`).
    """
    n_cells = len(cells)
    has_time = bool(rows_have_time(pareto))
    apply_budget = time_budget_multiplier > 0 and has_time and time_baselines is not None
    Xs_rows, Xe_rows = [], []
    bytes_log_rows, reach_rows = [], []
    time_log_rows = [] if has_time else None
    metric_log_rows = [] if emit_metric_head else None
    scalar_rows = {axis: [] for axis in SCALAR_AXES}
    meta = []
    infeasible: dict = {}
    # Drop zq targets above ceiling - this margin. The margin lets
    # the picker still see borderline cells where some images do reach
    # the target and others don't.
    CEILING_MARGIN = 0.0
    skipped_above_ceiling = 0

    for (image, size, w, h), samples in pareto.items():
        feat_key = (image, size)
        if feat_key not in feats:
            continue
        f = feats[feat_key]
        log_px = math.log(max(1, w * h))
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        size_oh[SIZE_INDEX[size]] = 1.0

        # Per-image zensim ceiling (only when sweep TSV declared it).
        ceiling = None
        if ceilings:
            ceiling = ceilings.get((image, size))

        # Group samples by config to track per-config best.
        # (one config can have multiple q values; pareto-best for each
        # cell at each zq target is the cheapest config that crosses zq.)
        by_cfg = defaultdict(list)
        for s in samples:
            by_cfg[s["config_id"]].append(s)

        for zq in ZQ_TARGETS:
            # Ceiling-aware skip: targets above the per-image achievable
            # max are physically unreachable (every cell will fail) —
            # skip them so the picker doesn't waste capacity on
            # impossible rows. See imazen/zenanalyze#51.
            if ceiling is not None and zq > ceiling + CEILING_MARGIN:
                skipped_above_ceiling += 1
                continue
            cell_bytes = [math.inf] * n_cells
            cell_time = [math.inf] * n_cells if has_time else None
            cell_metric = [math.nan] * n_cells if emit_metric_head else None
            cell_scalars = {axis: [math.nan] * n_cells for axis in SCALAR_AXES}
            cell_reach = [False] * n_cells

            # Per-(image, size) budget gate. When apply_budget is on,
            # only candidates with time ≤ this value are considered.
            budget_ms = math.inf
            if apply_budget:
                base = time_baselines.get(size)  # type: ignore[union-attr]
                if base is not None:
                    budget_ms = base * time_budget_multiplier

            # Track whether ANY config (ignoring budget) would have
            # reached this zq for this (image, size). Used to attribute
            # post-filter failures to the budget vs to physical
            # unreachability.
            any_unfiltered_reach = False

            for cfg_id, hits in by_cfg.items():
                # Cheapest sample for this config that reaches the target
                # (direction-aware: higher_better → metric ≥ zq;
                # lower_better → metric ≤ zq) AND within budget.
                best_b = math.inf
                best_t: float = math.inf
                best_m: float = math.nan
                for s in hits:
                    if METRIC_DIRECTION == "higher_better":
                        reaches = s["zensim"] >= zq
                    else:
                        reaches = s["zensim"] <= zq
                    if not reaches:
                        continue
                    any_unfiltered_reach = True
                    t_ms = s.get("time_ms", math.inf) if has_time else math.inf
                    if apply_budget and t_ms > budget_ms:
                        continue
                    if s["bytes"] < best_b:
                        best_b = s["bytes"]
                        if has_time:
                            best_t = t_ms
                        if emit_metric_head:
                            best_m = s["zensim"]
                if math.isinf(best_b):
                    continue
                c = config_to_cell[cfg_id]
                if best_b < cell_bytes[c]:
                    cell_bytes[c] = best_b
                    if has_time:
                        cell_time[c] = best_t
                    if emit_metric_head:
                        cell_metric[c] = best_m
                    p = parsed_all[cfg_id]
                    for axis in SCALAR_AXES:
                        cell_scalars[axis][c] = p[axis]
                    cell_reach[c] = True

            if not any(cell_reach):
                # Mark as budget-infeasible only when the unfiltered
                # version would have reached. Otherwise this is a
                # physically-unreachable target, not a budget problem.
                if apply_budget and any_unfiltered_reach:
                    infeasible[(image, size)] = True
                continue

            zq_norm = zq / 100.0
            # Engineered input vector — same as v1.1 student to keep
            # the comparison apples-to-apples.
            xs = np.concatenate([f, size_oh, np.array([log_px, zq_norm], dtype=np.float32)])
            xe = np.concatenate([
                f,
                size_oh,
                np.array(
                    [log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px],
                    dtype=np.float32,
                ),
                zq_norm * f,
                np.array([0.0], dtype=np.float32),  # icc placeholder
            ])

            bytes_log = np.array(
                [math.log(b) if not math.isinf(b) else math.nan for b in cell_bytes],
                dtype=np.float32,
            )
            reach = np.array(cell_reach, dtype=bool)

            Xs_rows.append(xs)
            Xe_rows.append(xe)
            bytes_log_rows.append(bytes_log)
            if time_log_rows is not None:
                # log(time_ms); NaN where the cell didn't reach.
                tlog = np.array(
                    [
                        math.log(t) if (t is not None and not math.isinf(t) and t > 0)
                        else math.nan
                        for t in cell_time  # type: ignore[union-attr]
                    ],
                    dtype=np.float32,
                )
                time_log_rows.append(tlog)
            if metric_log_rows is not None:
                # For higher_better metrics (zensim, ssim2), values are
                # always > 0 — log is fine. For lower_better (butteraugli),
                # also > 0 in practice (distance is non-negative). Use
                # log everywhere for numeric stability across metrics.
                # NaN preserved for unreached cells.
                mlog = np.array(
                    [
                        math.log(m) if (not math.isnan(m) and m > 0) else math.nan
                        for m in cell_metric  # type: ignore[union-attr]
                    ],
                    dtype=np.float32,
                )
                metric_log_rows.append(mlog)
            for axis in SCALAR_AXES:
                scalar_rows[axis].append(np.array(cell_scalars[axis], dtype=np.float32))
            reach_rows.append(reach)
            meta.append((image, size, zq))

    scalars = {axis: np.stack(scalar_rows[axis]) for axis in SCALAR_AXES}
    time_log = np.stack(time_log_rows) if time_log_rows else None
    metric_log = np.stack(metric_log_rows) if metric_log_rows else None
    return (
        np.stack(Xs_rows),
        np.stack(Xe_rows),
        np.stack(bytes_log_rows),
        scalars,
        np.stack(reach_rows),
        meta,
        time_log,
        metric_log,
        infeasible,
    )


# ---------- Evaluation ----------


def evaluate_argmin(pred_bytes_log, actual_bytes_log, reach, meta, mask):
    """Categorical argmin over allowed reachable cells."""
    rows = evaluate_argmin_per_row(pred_bytes_log, actual_bytes_log, reach, meta, mask)
    if not rows:
        return {"n": 0, "argmin_acc": 0.0, "mean_pct": 0.0, "p50_pct": 0.0, "p90_pct": 0.0}
    overheads = np.array([r["overhead"] for r in rows], dtype=np.float64)
    correct = sum(1 for r in rows if r["pick"] == r["actual_best"])
    return {
        "n": int(len(overheads)),
        "argmin_acc": correct / len(overheads),
        "mean_pct": float(100 * overheads.mean()),
        "p50_pct": float(100 * np.percentile(overheads, 50)),
        "p90_pct": float(100 * np.percentile(overheads, 90)),
        "p95_pct": float(100 * np.percentile(overheads, 95)),
        "p99_pct": float(100 * np.percentile(overheads, 99)),
        "max_pct": float(100 * overheads.max()),
    }


def evaluate_argmin_per_row(pred_bytes_log, actual_bytes_log, reach, meta, mask):
    """Like `evaluate_argmin` but returns the per-row breakdown so the
    safety-report code can stratify by zq / size_class and surface
    worst-case images. Each entry: image, size_class, zq, pick,
    actual_best, overhead, predicted_bytes, actual_bytes."""
    n_rows = pred_bytes_log.shape[0]
    out = []
    for i in range(n_rows):
        actual = actual_bytes_log[i]
        pred = pred_bytes_log[i]
        m = reach[i] & mask
        if not np.any(m):
            continue
        ab = np.where(m, np.exp(actual), np.inf)
        pb = np.where(m, np.exp(np.clip(pred, -30, 30)), np.inf)
        a = int(np.argmin(ab))
        p = int(np.argmin(pb))
        out.append({
            "image": meta[i][0],
            "size_class": meta[i][1],
            "zq": int(meta[i][2]),
            "pick": p,
            "actual_best": a,
            "overhead": float((ab[p] - ab[a]) / ab[a]),
            "predicted_bytes": float(ab[p]),
            "actual_best_bytes": float(ab[a]),
        })
    return out


def evaluate_scalars(pred_scalars, actual_scalars, reach):
    """Per-axis RMSE + MAE on scalar predictions, over reachable cells
    (where the target exists). Rows below SCALAR_SENTINELS[axis] (when
    declared) are excluded — for example zenjpeg's lambda<=0 marks
    trellis-off cells where the lambda value is a placeholder.

    `pred_scalars` and `actual_scalars` are dicts keyed by axis name,
    each mapping to an ndarray of shape `(n_rows, n_cells)`.

    Returns a flat dict like:
        {axis: rmse, axis+"_mae": mae, ...}
    so existing code that reads `metrics["chroma_scale"]` keeps working.
    """
    out = {}
    for axis in SCALAR_AXES:
        pred = pred_scalars[axis]
        actual = actual_scalars[axis]
        sentinel = SCALAR_SENTINELS.get(axis, None)
        diffs = []
        for i in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                if not reach[i, c]:
                    continue
                a = actual[i, c]
                if math.isnan(a):
                    continue
                if sentinel is not None and a <= sentinel:
                    continue
                diffs.append(pred[i, c] - a)
        arr = np.array(diffs, dtype=np.float64) if diffs else np.array([0.0])
        out[axis] = float(np.sqrt((arr ** 2).mean()))
        out[axis + "_mae"] = float(np.abs(arr).mean())
    return out


def evaluate_per_cell_r2(pred, actual, reach):
    """Per-cell R² over reachable cells. Returns
    `{"per_cell": [r2 or None per cell], "median": float, "min": float}`.

    R² = 1 - SS_res / SS_tot. None when the cell has < 5 reached samples
    or zero variance in the target. Used by the TIME_HEAD_R2 and
    METRIC_HEAD_R2 safety gates.
    """
    n_rows, n_cells = pred.shape
    per_cell = []
    valid = []
    for c in range(n_cells):
        a_vals = []
        p_vals = []
        for i in range(n_rows):
            if not reach[i, c]:
                continue
            a = actual[i, c]
            p = pred[i, c]
            if math.isnan(a) or math.isnan(p):
                continue
            a_vals.append(a)
            p_vals.append(p)
        if len(a_vals) < 5:
            per_cell.append(None)
            continue
        a_arr = np.asarray(a_vals, dtype=np.float64)
        p_arr = np.asarray(p_vals, dtype=np.float64)
        ss_tot = float(((a_arr - a_arr.mean()) ** 2).sum())
        if ss_tot <= 1e-12:
            per_cell.append(None)
            continue
        ss_res = float(((a_arr - p_arr) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
        per_cell.append(r2)
        valid.append(r2)
    return {
        "per_cell": per_cell,
        "median": float(np.median(valid)) if valid else float("nan"),
        "min": float(min(valid)) if valid else float("nan"),
        "n_valid": len(valid),
    }


# ---------- Diagnostics + safety report ----------

# Default thresholds. Codec configs override by exporting
# `SAFETY_THRESHOLDS = {...}`. Values are conservative — defaults that
# would have caught the v2.1 384³ overfit and the wide-schema strict
# regression seen during the 2026-04-29 retrain.
DEFAULT_SAFETY_THRESHOLDS = dict(
    # Train/val gap > X pp ⇒ overfit. Pick smaller than typical
    # production gap; the v2.0 baseline trains at ~2pp gap.
    max_train_val_gap_pp=2.0,
    # Held-out argmin accuracy must clear this floor.
    min_argmin_acc=0.30,
    # Held-out mean overhead ceiling.
    max_mean_overhead_pct=10.0,
    # No single zq band may have a p99 overhead this bad.
    max_per_zq_p99_overhead_pct=80.0,
    # No single image-size class may have a p99 overhead this bad.
    # Size invariance is a *safety property*: the picker must be
    # near-optimal at every (width, height), not just on average.
    # Tiny images (≤64×64) historically tail worst — tight per-bin
    # gates catch a tiny-class blowup that the global mean would
    # absorb. See SAFETY_PLANE.md → "Size invariance is a safety
    # property" and the size_invariance_probe.py post-bake gate.
    max_per_size_p99_overhead_pct=80.0,
    # Each (size_class, target_zq) training cell must have at least
    # this many rows. Below this, the teacher fits noise in a
    # corner of the size×quality grid the picker still has to serve
    # at inference. The codec's pareto sweep MUST emit rows for
    # tiny / small / medium / large per image (see
    # FOR_NEW_CODECS.md Step 1.5); this gate fires when the sweep
    # silently skips a size class for a chunk of the corpus.
    min_train_rows_per_size_zq=50,
    # No single (image, size, zq) row may overshoot by more than
    # this. Catches catastrophic individual failures.
    max_single_row_overhead_pct=200.0,
    # Each cell must have at least this many member configs in the
    # training data; below this the teacher fits noise.
    min_cell_member_configs=3,
    # Each cell must have at least this many train rows AFTER the
    # reach mask. Below this the teacher returns None and the
    # student falls back to a constant — picker can't actually pick
    # this cell with confidence.
    min_cell_reach_train_rows=50,
    # MLP weight sanity. Beyond these, training is broken.
    max_dead_neuron_fraction=0.30,
    max_layer_weight_ratio=1000.0,
    # zensim_strict-only: at least one cell must remain safe at the
    # top of the zq grid (so the picker isn't always falling through
    # to KnownGoodFallback).
    min_safe_cells_at_top_zq=1,
    # The sweep TSV must declare per-(image, size_class) zensim
    # ceilings (`effective_max_zensim` column) when the codec's
    # ZQ_TARGETS grid extends above this threshold. Without ceilings,
    # the trainer can't tell DATA_STARVED_SIZE (sweep harness skipped
    # cells) apart from physically-unreachable rows (perceptual
    # metric saturated below target). Set to None to disable. See
    # imazen/zenanalyze#51 for the cross-codec design context.
    require_ceiling_above_zq=85,
    # Time / metric head R² floors (held-out, per cell median). Below
    # these the picker can't trust the head's predictions for
    # inference-time budget filtering or quality-constraint enforcement.
    # Only checked when the corresponding head is trained.
    min_time_head_r2=0.6,
    min_metric_head_r2=0.6,
    # Fraction of (image, size) pairs where every cell is over budget
    # at every zq target. When higher than this, the budget is too
    # tight for the corpus — the picker has nothing to recommend.
    max_budget_infeasible_fraction=0.05,
)


def compute_feature_bounds(feats, train_keys, feat_cols):
    """Per-feature distribution stats over the **training** image set.

    Computed once at bake time and shipped in the manifest so codecs
    can detect out-of-distribution inputs at runtime and fall through
    to a `KnownGoodFallback` rescue rather than letting the MLP
    extrapolate silently.

    Each entry is a dict with `min, p01, p25, p50, p75, p99, max,
    mean, std` — codec picks which pair to compile into its
    `FEATURE_BOUNDS` const. Default recommendation: `(p01, p99)` so
    the gate fires only on truly extreme inputs (≈2% miss rate at
    train-distribution boundaries by construction).
    """
    keys_seen = [k for k in train_keys if k in feats]
    if not keys_seen:
        return {}
    arr = np.stack([feats[k] for k in keys_seen]).astype(np.float64)
    out = {}
    for i, col in enumerate(feat_cols):
        v = arr[:, i]
        v_finite = v[np.isfinite(v)]
        if v_finite.size == 0:
            out[col] = {
                "min": None, "p01": None, "p25": None, "p50": None,
                "p75": None, "p99": None, "max": None,
                "mean": None, "std": None, "n": 0,
            }
            continue
        out[col] = {
            "min": float(v_finite.min()),
            "p01": float(np.percentile(v_finite, 1)),
            "p25": float(np.percentile(v_finite, 25)),
            "p50": float(np.percentile(v_finite, 50)),
            "p75": float(np.percentile(v_finite, 75)),
            "p99": float(np.percentile(v_finite, 99)),
            "max": float(v_finite.max()),
            "mean": float(v_finite.mean()),
            "std": float(v_finite.std()),
            "n": int(v_finite.size),
        }
    return out


def count_train_rows_by_size_zq(meta_tr, size_classes, zq_targets):
    """Count training rows per (size_class, zq) cell.

    Size invariance discipline (see SAFETY_PLANE.md): the picker is a
    feature-vector-in / argmin-out function — it has no notion of
    image dimensions at runtime, so the codec must populate every
    (size_class, target_zq) cell with enough training data for the
    teacher to learn from. A sparsely-sampled cell silently trains
    noise into a corner of the size × quality grid the picker still
    has to serve at inference. This counter feeds the
    `DATA_STARVED_SIZE` safety violation.

    Returns: {size_class: {zq: int}} with every declared
    `size_classes × zq_targets` combination present (zero when the
    sweep emitted no rows for that cell)."""
    out = {sz: {int(zq): 0 for zq in zq_targets} for sz in size_classes}
    for _img, sz, zq in meta_tr:
        if sz in out and int(zq) in out[sz]:
            out[sz][int(zq)] += 1
    return out


def stratify_overheads(per_row):
    """Group per-row overhead entries by (zq, size_class). Returns
    {zq: {size_class: stats_dict}} and a flat per-zq aggregate."""
    by_zq = {}
    by_size = {}
    by_zq_size = {}
    for r in per_row:
        zq = r["zq"]
        sz = r["size_class"]
        by_zq.setdefault(zq, []).append(r["overhead"])
        by_size.setdefault(sz, []).append(r["overhead"])
        by_zq_size.setdefault((zq, sz), []).append(r["overhead"])

    def stats(arr):
        a = np.array(arr, dtype=np.float64)
        return {
            "n": int(len(a)),
            "mean_pct": float(100 * a.mean()),
            "p50_pct": float(100 * np.percentile(a, 50)),
            "p90_pct": float(100 * np.percentile(a, 90)),
            "p95_pct": float(100 * np.percentile(a, 95)),
            "p99_pct": float(100 * np.percentile(a, 99)),
            "max_pct": float(100 * a.max()),
        }

    return (
        {zq: stats(v) for zq, v in by_zq.items()},
        {sz: stats(v) for sz, v in by_size.items()},
        {f"{zq}/{sz}": stats(v) for (zq, sz), v in by_zq_size.items()},
    )


def worst_case_rows(per_row, top_pct=1.0, max_n=20):
    """Top-`top_pct`% rows by overhead, capped at `max_n` for the log."""
    if not per_row:
        return []
    threshold = np.percentile([r["overhead"] for r in per_row], 100 - top_pct)
    bad = sorted(
        (r for r in per_row if r["overhead"] >= threshold),
        key=lambda r: -r["overhead"],
    )
    out = []
    for r in bad[:max_n]:
        out.append({
            "image": r["image"],
            "size_class": r["size_class"],
            "zq": r["zq"],
            "pick": r["pick"],
            "actual_best": r["actual_best"],
            "overhead_pct": float(100 * r["overhead"]),
        })
    return out


def per_cell_diagnostics(
    cells, pred_bytes_log_va, actual_bytes_log_va, reach_va, n_cells
):
    """For each cell: training-time row count, member config count,
    calibration delta (predicted mean vs actual mean log-bytes on val
    rows where the cell was reachable). Big delta ⇒ systematic bias."""
    out = []
    for c in range(n_cells):
        mask = reach_va[:, c]
        if not mask.any():
            out.append({
                "cell": c,
                "label": cells[c]["label"],
                "n_member_configs": len(cells[c]["member_config_ids"]),
                "n_val_reach_rows": 0,
                "predicted_mean_log_bytes": None,
                "actual_mean_log_bytes": None,
                "calibration_delta": None,
            })
            continue
        pmean = float(np.nanmean(pred_bytes_log_va[mask, c]))
        amean = float(np.nanmean(actual_bytes_log_va[mask, c]))
        out.append({
            "cell": c,
            "label": cells[c]["label"],
            "n_member_configs": len(cells[c]["member_config_ids"]),
            "n_val_reach_rows": int(mask.sum()),
            "predicted_mean_log_bytes": pmean,
            "actual_mean_log_bytes": amean,
            "calibration_delta": pmean - amean,
        })
    return out


def scan_mlp_weights(student, X_va):
    """Static + dynamic checks on the student MLP. Returns dict.

    Static (from coefs_): NaN/Inf, max-to-median weight ratio per
    layer.

    Dynamic (forward pass on val): dead-neuron fraction (output
    variance ~0 across val rows) — catches collapsed neurons that
    never contribute to predictions."""
    nan_in_weights = False
    inf_in_weights = False
    layer_ratios = []
    for layer_w in student.coefs_:
        if not np.isfinite(layer_w).all():
            nan_in_weights = nan_in_weights or bool(np.isnan(layer_w).any())
            inf_in_weights = inf_in_weights or bool(np.isinf(layer_w).any())
        absw = np.abs(layer_w)
        med = float(np.median(absw)) if absw.size else 0.0
        mx = float(absw.max()) if absw.size else 0.0
        layer_ratios.append({"max": mx, "median": med, "ratio": mx / max(med, 1e-12)})

    # Dynamic: forward each hidden layer up to (but not including)
    # the regression head and find neurons whose output variance is ~0.
    activations = X_va.copy()
    dead_total = 0
    n_total = 0
    for li, (W, b) in enumerate(zip(student.coefs_, student.intercepts_)):
        z = activations @ W + b
        is_hidden = li < len(student.coefs_) - 1
        if is_hidden:
            # ReLU activation
            a = np.maximum(z, 0.0)
            var = a.var(axis=0)
            dead_total += int((var < 1e-10).sum())
            n_total += a.shape[1]
            activations = a
        else:
            activations = z
    dead_frac = (dead_total / n_total) if n_total else 0.0

    nan_in_predictions = bool(np.isnan(activations).any() or np.isinf(activations).any())

    return {
        "nan_in_weights": nan_in_weights,
        "inf_in_weights": inf_in_weights,
        "nan_in_predictions": nan_in_predictions,
        "dead_neuron_fraction": float(dead_frac),
        "n_dead_neurons": int(dead_total),
        "n_total_hidden_neurons": int(n_total),
        "per_layer_weight_ratio": layer_ratios,
        "max_layer_weight_ratio": max((r["ratio"] for r in layer_ratios), default=0.0),
    }


def safety_check(diag, thresholds, objective: str):
    """Compile violations from the diagnostics dict against the
    threshold dict. Returns (passed, violations_list)."""
    v = []

    val = diag["argmin"]["val"]
    train = diag["argmin"]["train"]
    gap = val["mean_pct"] - train["mean_pct"]
    if gap > thresholds["max_train_val_gap_pp"]:
        v.append(
            f"OVERFIT: train→val mean gap {gap:+.2f}pp "
            f"(train {train['mean_pct']:.2f}% vs val {val['mean_pct']:.2f}%) "
            f"> threshold {thresholds['max_train_val_gap_pp']:.2f}pp"
        )

    if val["argmin_acc"] < thresholds["min_argmin_acc"]:
        v.append(
            f"LOW_ARGMIN: val argmin_acc {val['argmin_acc']:.1%} "
            f"< threshold {thresholds['min_argmin_acc']:.1%}"
        )

    if val["mean_pct"] > thresholds["max_mean_overhead_pct"]:
        v.append(
            f"HIGH_OVERHEAD: val mean overhead {val['mean_pct']:.2f}% "
            f"> threshold {thresholds['max_mean_overhead_pct']:.2f}%"
        )

    for zq, m in diag["by_zq"].items():
        if m["p99_pct"] > thresholds["max_per_zq_p99_overhead_pct"]:
            v.append(
                f"PER_ZQ_TAIL: zq={zq} p99 overhead {m['p99_pct']:.1f}% "
                f"> threshold {thresholds['max_per_zq_p99_overhead_pct']:.1f}%"
            )

    # Size invariance: the picker must be near-optimal at every
    # image (width, height), not just on the global average. Tiny
    # images historically tail worst (small absolute headers
    # dominate per-pixel cost) — a per-size p99 ceiling is the
    # in-trainer counterpart to size_invariance_probe.py's
    # post-bake stability check.
    for sz, m in diag.get("by_size", {}).items():
        if m["p99_pct"] > thresholds["max_per_size_p99_overhead_pct"]:
            v.append(
                f"PER_SIZE_TAIL: size_class={sz} p99 overhead {m['p99_pct']:.1f}% "
                f"> threshold {thresholds['max_per_size_p99_overhead_pct']:.1f}% "
                f"(picker is not size-invariant — see SAFETY_PLANE.md)"
            )

    # Data-starvation gate per (size_class, target_zq) training
    # cell. Catches sweep harnesses that silently skip a size
    # class for a chunk of the corpus, leaving the picker with
    # too few examples to learn from at that (size, quality)
    # corner. Codec's harness MUST emit rows for tiny / small /
    # medium / large per image (FOR_NEW_CODECS.md Step 1.5).
    starved = []
    for sz, by_zq in diag.get("train_rows_by_size_zq", {}).items():
        for zq, n in by_zq.items():
            if n < thresholds["min_train_rows_per_size_zq"]:
                starved.append((sz, zq, n))
    if starved:
        # Surface the worst (lowest-n) cells; capping at 6 lines
        # keeps the log readable when a whole size class is missing.
        starved.sort(key=lambda t: t[2])
        examples = ", ".join(
            f"{sz}/zq{zq}={n}" for (sz, zq, n) in starved[:6]
        )
        more = f" (+{len(starved) - 6} more)" if len(starved) > 6 else ""
        v.append(
            f"DATA_STARVED_SIZE: {len(starved)} (size_class, zq) cell(s) "
            f"have train rows < {thresholds['min_train_rows_per_size_zq']}: "
            f"{examples}{more}"
        )

    # Sweep-side ceiling discipline. When the codec's ZQ_TARGETS grid
    # extends above `require_ceiling_above_zq`, the sweep TSV MUST
    # declare per-(image, size_class) zensim ceilings so the trainer
    # can tell DATA_STARVED_SIZE (sweep harness gap) apart from
    # physically-unreachable rows (perceptual metric saturated below
    # target). Without this, every codec re-discovers the same lesson
    # the hard way — silent miscalibration at small+high-zq corners.
    # See imazen/zenanalyze#51.
    require_ceiling_above_zq = thresholds.get("require_ceiling_above_zq")
    sweep_ceilings = diag.get("sweep_ceilings", {})
    if require_ceiling_above_zq is not None and sweep_ceilings:
        max_target_zq = sweep_ceilings.get("max_target_zq", 0)
        if max_target_zq > require_ceiling_above_zq and not sweep_ceilings.get(
            "has_effective_max_zensim", False
        ):
            v.append(
                f"UNCAPPED_ZQ_GRID: ZQ_TARGETS includes zq={max_target_zq} > "
                f"{require_ceiling_above_zq} but Pareto TSV has no "
                f"`effective_max_zensim` column. Trainer can't tell physically-"
                f"unreachable cells apart from sweep gaps; DATA_STARVED_SIZE "
                f"warnings cannot be diagnosed honestly. Either lower "
                f"max(ZQ_TARGETS) below {require_ceiling_above_zq + 1} or have "
                f"the codec sweep harness emit `effective_max_zensim` per "
                f"(image, size_class). See imazen/zenanalyze#51."
            )

    if diag["worst_case"]:
        worst = diag["worst_case"][0]
        if worst["overhead_pct"] > thresholds["max_single_row_overhead_pct"]:
            v.append(
                f"WORST_ROW: {worst['image']} @ {worst['size_class']}/zq{worst['zq']} "
                f"overhead {worst['overhead_pct']:.1f}% "
                f"> threshold {thresholds['max_single_row_overhead_pct']:.1f}%"
            )

    for c in diag["per_cell"]:
        if c["n_member_configs"] < thresholds["min_cell_member_configs"]:
            v.append(
                f"DATA_STARVED_CELL: cell {c['cell']} ({c['label']}) has "
                f"{c['n_member_configs']} member configs "
                f"< threshold {thresholds['min_cell_member_configs']}"
            )

    mlp = diag["mlp"]
    if mlp["nan_in_weights"]:
        v.append("NAN_WEIGHTS: student MLP layer weights contain NaN")
    if mlp["inf_in_weights"]:
        v.append("INF_WEIGHTS: student MLP layer weights contain Inf")
    if mlp["nan_in_predictions"]:
        v.append("NAN_PREDICTIONS: student MLP produced NaN/Inf on val")
    if mlp["dead_neuron_fraction"] > thresholds["max_dead_neuron_fraction"]:
        v.append(
            f"DEAD_NEURONS: {mlp['dead_neuron_fraction']:.1%} of hidden neurons "
            f"have ~0 variance on val "
            f"> threshold {thresholds['max_dead_neuron_fraction']:.1%}"
        )
    if mlp["max_layer_weight_ratio"] > thresholds["max_layer_weight_ratio"]:
        v.append(
            f"WEIGHT_BLOWUP: max/median weight ratio {mlp['max_layer_weight_ratio']:.0f} "
            f"> threshold {thresholds['max_layer_weight_ratio']:.0f}"
        )

    if objective == "zensim_strict" and "reach_safety" in diag:
        # Highest zq band must have at least one safe cell, otherwise
        # zensim_strict callers above that band always fall through.
        top_zq = max((int(z) for z in diag["reach_safety"]["by_zq"].keys()), default=0)
        if top_zq:
            top = diag["reach_safety"]["by_zq"][str(top_zq)]
            n_safe = sum(1 for s in top["safe"] if s)
            if n_safe < thresholds["min_safe_cells_at_top_zq"]:
                v.append(
                    f"NO_SAFE_CELL_AT_TOP_ZQ: zq={top_zq} has {n_safe} safe cells "
                    f"< threshold {thresholds['min_safe_cells_at_top_zq']} — "
                    "zensim_strict picker can't reach the top of the zq grid"
                )

    # Time / metric head R² gates — only checked when the head exists.
    thr_time_r2 = thresholds.get("min_time_head_r2", 0.0)
    if diag.get("time_head_r2") is not None and thr_time_r2 > 0:
        med = diag["time_head_r2"]["median"]
        if not math.isnan(med) and med < thr_time_r2:
            v.append(
                f"TIME_HEAD_R2: median per-cell R² {med:.3f} "
                f"< threshold {thr_time_r2:.3f} — time predictions "
                f"too noisy for inference-time budget filtering"
            )
    thr_metric_r2 = thresholds.get("min_metric_head_r2", 0.0)
    if diag.get("metric_head_r2") is not None and thr_metric_r2 > 0:
        med = diag["metric_head_r2"]["median"]
        if not math.isnan(med) and med < thr_metric_r2:
            v.append(
                f"METRIC_HEAD_R2: median per-cell R² {med:.3f} "
                f"< threshold {thr_metric_r2:.3f} — metric predictions "
                f"too noisy to enforce quality constraints"
            )
    # Budget feasibility — only meaningful when budget filter is on.
    thr_budget = thresholds.get("max_budget_infeasible_fraction", 1.0)
    frac = diag.get("budget_infeasible_fraction", 0.0)
    if frac > thr_budget:
        v.append(
            f"BUDGET_INFEASIBLE: {frac:.1%} of (image, size) pairs have "
            f"no in-budget cell at any zq target "
            f"> threshold {thr_budget:.1%} — budget too tight for corpus"
        )

    return (len(v) == 0, v)


# ---------- Train ----------


def train_teacher_per_cell(
    Xs_tr,
    bytes_log_tr,
    scalars_tr,
    reach_tr,
    n_cells,
    params=None,
    bytes_quantile=None,
    time_log_tr=None,
    metric_log_tr=None,
):
    """Per-cell HistGB regressors for: bytes_log + each scalar axis
    (+ optional time_log when `time_log_tr` is provided).

    `scalars_tr` is a dict `{axis_name: ndarray(n_rows, n_cells)}`.
    `time_log_tr` is `ndarray(n_rows, n_cells)` of `log(encode_ms)` for
    the within-cell-best config; NaN where the cell didn't reach target.
    Returns `(teachers_bytes, teachers_per_axis, scalar_means,
    teachers_time, time_means)` — the time entries are `(None, None)`
    when `time_log_tr is None`.

    Per-axis sentinel mask: when SCALAR_SENTINELS[axis] is declared,
    rows where actual_value <= sentinel are excluded from that axis's
    training (matches zenjpeg's lambda<=0 → trellis-off semantics).

    (1 + len(SCALAR_AXES)) teachers per cell × n_cells × ~5 s each.
    With `train_teachers_per_cell_parallel` on a 16-core box it ends
    up ~30 s per head pass. ~12× speedup vs the pre-2026-04-29 serial
    loop.

    `params` defaults to `HISTGB_FULL` (production training). Pass
    `HISTGB_FAST` for iteration / ablation runs.

    `bytes_quantile`: when not None, switches the bytes head to
    quantile regression at that q (e.g. 0.99). Used by the
    `zensim_strict` safety profile so the bytes head predicts the
    worst-case-safe cost, not the mean. Scalar heads always stay at
    mean regression — they predict the within-cell-optimal scalar
    conditional on the cell being chosen.
    """
    from _picker_lib import HISTGB_FULL, train_teachers_per_cell_parallel

    if params is None:
        params = HISTGB_FULL

    # Per-axis fallback means computed from sentinel-filtered values.
    scalar_means = {}
    for axis in SCALAR_AXES:
        arr = scalars_tr[axis]
        sentinel = SCALAR_SENTINELS.get(axis, None)
        if sentinel is not None:
            scalar_means[axis] = np.nanmean(np.where(arr > sentinel, arr, np.nan), axis=0)
        else:
            scalar_means[axis] = np.nanmean(arr, axis=0)

    # Bytes head — per-cell reach mask (cell achieved target_zq).
    bytes_params = dict(params)
    if bytes_quantile is not None:
        bytes_params["loss"] = "quantile"
        bytes_params["quantile"] = bytes_quantile
    teachers_bytes = train_teachers_per_cell_parallel(
        Xs_tr, bytes_log_tr, reach_tr, params=bytes_params, label="bytes"
    )

    # Scalar heads — same reach mask, plus per-axis sentinel mask
    # when declared.
    teachers_per_axis = {}
    for axis in SCALAR_AXES:
        arr = scalars_tr[axis]
        sentinel = SCALAR_SENTINELS.get(axis, None)
        extra_mask = arr > sentinel if sentinel is not None else None
        teachers_per_axis[axis] = train_teachers_per_cell_parallel(
            Xs_tr, arr, reach_tr,
            extra_mask=extra_mask, params=params, label=axis,
        )

    # Time head (optional) — same reach mask. Codec runtime can apply a
    # budget filter at inference using these predictions.
    teachers_time = None
    time_means = None
    if time_log_tr is not None:
        # Mean-regression for time (no quantile by default — bytes head's
        # quantile mode is for risk-tail bytes, not time predictions).
        teachers_time = train_teachers_per_cell_parallel(
            Xs_tr, time_log_tr, reach_tr, params=params, label="time",
        )
        time_means = np.nanmean(time_log_tr, axis=0)

    # Metric head (optional) — predicts the achieved metric value
    # (log-space) for the within-cell-best config. Codec runtime uses
    # this to enforce the user's quality constraint (e.g., bfly ≤ target).
    teachers_metric = None
    metric_means = None
    if metric_log_tr is not None:
        teachers_metric = train_teachers_per_cell_parallel(
            Xs_tr, metric_log_tr, reach_tr, params=params, label="metric",
        )
        metric_means = np.nanmean(metric_log_tr, axis=0)

    return (
        teachers_bytes,
        teachers_per_axis,
        scalar_means,
        teachers_time,
        time_means,
        teachers_metric,
        metric_means,
    )


def teacher_predict_all(teachers, Xs, fallback_means, n_cells):
    out = np.zeros((Xs.shape[0], n_cells), dtype=np.float32)
    for c in range(n_cells):
        if teachers[c] is None:
            out[:, c] = fallback_means[c] if not math.isnan(fallback_means[c]) else 0.0
        else:
            out[:, c] = teachers[c].predict(Xs)
    return out


def compute_reach_safe_cells(
    bytes_log_tr,
    reach_tr,
    meta_tr,
    n_cells,
    zq_targets,
    threshold: float,
) -> dict:
    """Per-target_zq, return the per-cell empirical reach rate and
    the boolean safety mask (`reach_rate >= threshold`).

    Used by the `zensim_strict` profile: cells whose historical reach
    rate at a target_zq band is below `threshold` (default 0.99) are
    masked out at inference. Codec consumers AND this gate with their
    caller mask before argmin.

    Returns:
        {
          "threshold": float,
          "by_zq": {str(zq): {"reach_rate": [f32; n_cells],
                              "safe": [bool; n_cells]}},
        }
    """
    out = {"threshold": float(threshold), "by_zq": {}}
    for zq in zq_targets:
        zq_rows = [i for i, m in enumerate(meta_tr) if m[2] == zq]
        if not zq_rows:
            continue
        zq_idx = np.array(zq_rows)
        rch = reach_tr[zq_idx]
        rate = rch.mean(axis=0).astype(np.float32)
        safe = (rate >= threshold).tolist()
        out["by_zq"][str(zq)] = {
            "reach_rate": [float(x) for x in rate],
            "safe": [bool(x) for x in safe],
        }
    return out


def main():
    # Declare globals first — `--seed` help text reads `SEED` and
    # later code may rebind it, so the `global` declaration must come
    # before any read of these names per Python scoping rules.
    global SEED, OUT_JSON, OUT_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec-config",
        required=True,
        help="Python module name exporting PARETO/FEATURES/OUT_*/ZQ_TARGETS/"
        "KEEP_FEATURES/parse_config_name. Example: zenjpeg_picker_config (which "
        "must be importable on PYTHONPATH).",
    )
    parser.add_argument(
        "--objective",
        choices=["size_optimal", "zensim_strict"],
        default="size_optimal",
        help="Safety profile. `size_optimal` (default) trains the bytes head "
        "with mean log-bytes regression — minimum mean cost subject to reach. "
        "`zensim_strict` trains with quantile regression at --bytes-quantile "
        "(default 0.99) and emits a per-zq reach-rate gate; cells whose "
        "empirical reach rate is below --reach-threshold at a given target "
        "are masked out at inference.",
    )
    parser.add_argument(
        "--bytes-quantile",
        type=float,
        default=0.99,
        help="Quantile for the bytes head when --objective=zensim_strict. "
        "Default 0.99: bytes prediction is the p99 worst-case cost so "
        "argmin biases toward configs that are safe at the tail.",
    )
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=0.99,
        help="Per-cell empirical reach-rate floor for the zensim_strict "
        "safety gate. Cells with reach_rate < threshold at a given "
        "target_zq are excluded from the runtime mask. Default 0.99.",
    )
    parser.add_argument(
        "--metric-column",
        default=None,
        help="Override codec config's METRIC_COLUMN. Pareto-TSV column "
        "name for the quality metric the picker is trained against "
        "(e.g., 'butteraugli', 'ssim2', 'dssim'). Default: codec config "
        "value (or 'zensim' if unset).",
    )
    parser.add_argument(
        "--metric-direction",
        choices=["higher_better", "lower_better"],
        default=None,
        help="Override codec config's METRIC_DIRECTION. 'higher_better' "
        "for zensim/ssim2/psnr; 'lower_better' for butteraugli/dssim/mse. "
        "Default: codec config value.",
    )
    parser.add_argument(
        "--time-budget-multiplier",
        type=float,
        default=0.0,
        help="When > 0, applies a budget filter at label-extraction time: "
        "only configs with time ≤ baseline_ms[size_class] × multiplier "
        "are eligible as within-cell candidates. baseline_ms is the "
        "median time per size_class. Default 0 (no filter).",
    )
    parser.add_argument(
        "--emit-metric-head",
        action="store_true",
        help="Train an extra per-cell `metric_log` head (predicts the "
        "achieved metric value of the within-cell-best config). Codec "
        "runtime uses this to enforce the user's quality constraint. "
        "Requires the metric column be available in the pareto TSV.",
    )
    parser.add_argument(
        "--out-suffix",
        default=None,
        help="Override the OUT_JSON / OUT_LOG basename suffix. Defaults to "
        "the codec config's OUT_JSON for size_optimal, and "
        "<basename>_zensim_strict for zensim_strict.",
    )
    parser.add_argument(
        "--hidden",
        default="128,128",
        help="Comma-separated hidden layer widths for the student MLP. "
        "Default '128,128' matches the v2.0 baseline. Try '256,256' or "
        "'256,256,256' when the input layer grows past ~50 cross-termed "
        "inputs (e.g. v2.1's 35-feature schema feeds ~80 inputs into the "
        "MLP — 128x128 is undersized).",
    )
    parser.add_argument(
        "--dump-overheads",
        type=Path,
        default=None,
        help="If set, write a per-row val overhead CSV to this path "
        "(image, size_class, zq, pick, actual_best, overhead). "
        "Future violin / KDE plots feed off this; safety_report only "
        "carries summary percentiles.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "leakyrelu"],
        default="relu",
        help="Hidden-layer activation. relu (default) trains via "
        "sklearn `MLPRegressor.fit`, which is single-threaded for "
        "our matmul size and TYPICALLY 10–20× SLOWER than the "
        "leakyrelu path. **Use `--activation leakyrelu` for new "
        "bakes** — it falls through to a PyTorch student with "
        "negative_slope=0.01 (same MLP shape, same Adam/lr/batch/"
        "early-stopping schedule) and finishes in seconds-to-minutes "
        "instead of minutes-to-hours. Both produce a "
        "`student.coefs_/intercepts_` surface so safety_check, "
        "diagnostics, and JSON serialization work the same way. "
        "Keep `relu` only when you need bit-identical reproduction "
        "of a pre-leakyrelu sklearn-trained baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed override (default: {SEED:#x}). The student "
        "training uses this to seed init + dropout; the train/val "
        "image-level split also keys off this seed via "
        "`np.random.default_rng(SEED)`. Multi-seed sweeps for "
        "experiments like LeakyReLU-vs-ReLU pass --seed N.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any safety threshold is violated. "
        "Auto-enabled when the CI environment variable is set. The "
        "JSON output is still written (with safety_report.passed=false) "
        "so reviewers can inspect; bake_picker.py then refuses to bake "
        "unless --allow-unsafe is also passed there.",
    )
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Override the strict gate even when --strict / CI is set. "
        "Use only when a violation is intentional and reviewed.",
    )
    args = parser.parse_args()
    hidden_layer_sizes = tuple(int(x) for x in args.hidden.split(","))
    is_ci = bool(os.environ.get("CI"))
    strict = (args.strict or is_ci) and not args.allow_unsafe
    # Per-run seed override — falls back to the module-level SEED so
    # default behavior is unchanged. (`global SEED` already declared
    # at top of `main()`.)
    if args.seed is not None:
        SEED = args.seed
        sys.stderr.write(f"  seed override: SEED={SEED:#x}\n")
    load_codec_config(args.codec_config)

    # CLI overrides — take precedence over codec config defaults.
    global METRIC_COLUMN, METRIC_DIRECTION
    if args.metric_column is not None:
        METRIC_COLUMN = args.metric_column
        sys.stderr.write(f"  CLI override: METRIC_COLUMN={METRIC_COLUMN!r}\n")
    if args.metric_direction is not None:
        METRIC_DIRECTION = args.metric_direction
        sys.stderr.write(f"  CLI override: METRIC_DIRECTION={METRIC_DIRECTION!r}\n")

    # Per-objective output naming. The codec config defines the
    # baseline OUT_JSON/OUT_LOG; we suffix when training a non-default
    # safety profile so both bakes can co-exist. `global OUT_JSON,
    # OUT_LOG` already declared at top of `main()`.
    if args.out_suffix is not None:
        suffix = args.out_suffix
    elif args.objective == "zensim_strict":
        suffix = "_zensim_strict"
    else:
        suffix = ""
    if suffix:
        OUT_JSON = OUT_JSON.with_name(OUT_JSON.stem + suffix + OUT_JSON.suffix)
        OUT_LOG = OUT_LOG.with_name(OUT_LOG.stem + suffix + OUT_LOG.suffix)
    sys.stderr.write(
        f"Training objective: {args.objective}\n"
        f"  bytes head loss: "
        f"{'quantile q=' + str(args.bytes_quantile) if args.objective == 'zensim_strict' else 'mean (squared error)'}\n"
        f"  reach gate: "
        f"{'>= ' + str(args.reach_threshold) + ' per zq band' if args.objective == 'zensim_strict' else 'none (any reachable cell allowed)'}\n"
        f"  output JSON: {OUT_JSON}\n"
    )

    sys.stderr.write(f"Loading {PARETO}...\n")
    pareto, ceilings, has_ceiling_column, has_time_column = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(
        f"  metric column: {METRIC_COLUMN} ({METRIC_DIRECTION})\n"
        f"  time column:   {TIME_COLUMN} ({'present' if has_time_column else 'absent'})\n"
    )
    if has_ceiling_column:
        n_with_ceiling = sum(1 for v in ceilings.values() if v is not None)
        sys.stderr.write(
            f"Loaded {len(pareto)} cells × {len(feat_cols)} features  "
            f"({n_with_ceiling} (image, size_class) pairs declare effective_max_zensim)\n"
        )
    else:
        sys.stderr.write(
            f"Loaded {len(pareto)} cells × {len(feat_cols)} features  "
            f"(sweep TSV has NO effective_max_zensim column — see imazen/zenanalyze#51)\n"
        )

    cells, cell_id_by_key, config_to_cell, parsed_all = build_cell_index()
    n_cells = len(cells)
    sys.stderr.write(f"\nCategorical cells: {n_cells}\n")
    for c in cells:
        sys.stderr.write(f"  {c['id']:>2d}: {c['label']:30s}  ({len(c['member_config_ids'])} configs)\n")

    # Time baselines per size_class (median ms across all configs).
    # Only used when --time-budget-multiplier > 0; we always compute so
    # we can record them in the manifest for the codec runtime.
    time_baselines = compute_time_baselines(pareto) if has_time_column else {}
    if args.time_budget_multiplier > 0 and not has_time_column:
        sys.stderr.write(
            f"  WARNING: --time-budget-multiplier={args.time_budget_multiplier} "
            f"but pareto TSV has no '{TIME_COLUMN}' column — budget filter "
            f"will be a no-op\n"
        )
    if args.time_budget_multiplier > 0 and time_baselines:
        sys.stderr.write(
            f"  budget per size_class (median × {args.time_budget_multiplier}): "
            + ", ".join(
                f"{sz}={time_baselines[sz] * args.time_budget_multiplier:.1f}ms"
                for sz in sorted(time_baselines)
            )
            + "\n"
        )

    (
        Xs,
        Xe,
        bytes_log,
        scalars,
        reach,
        meta,
        time_log,
        metric_log,
        infeasible,
    ) = build_dataset(
        pareto,
        feats,
        feat_cols,
        cells,
        config_to_cell,
        parsed_all,
        ceilings=(ceilings if has_ceiling_column else None),
        time_budget_multiplier=args.time_budget_multiplier,
        time_baselines=time_baselines if time_baselines else None,
        emit_metric_head=args.emit_metric_head,
    )
    sys.stderr.write(
        f"\nDecision rows: {len(Xs)}; Xs={Xs.shape[1]}, Xe={Xe.shape[1]}, n_cells={n_cells}"
        + (" (+ time_log)" if time_log is not None else "")
        + (" (+ metric_log)" if metric_log is not None else "")
        + (f" (BUDGET_INFEASIBLE: {len(infeasible)} (image, size) pairs)"
           if infeasible else "")
        + "\n"
    )
    n_scalar_axes = len(SCALAR_AXES)
    has_time_head = time_log is not None
    has_metric_head = metric_log is not None
    # Output blocks per cell: bytes + (time?) + (metric?) + scalar axes.
    output_dim = (
        1
        + (1 if has_time_head else 0)
        + (1 if has_metric_head else 0)
        + n_scalar_axes
    ) * n_cells

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    tr = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    va = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(tr)}, val rows: {len(va)}\n")

    Xs_tr, Xs_va = Xs[tr], Xs[va]
    Xe_tr, Xe_va = Xe[tr], Xe[va]
    bl_tr, bl_va = bytes_log[tr], bytes_log[va]
    scalars_tr = {axis: scalars[axis][tr] for axis in SCALAR_AXES}
    scalars_va = {axis: scalars[axis][va] for axis in SCALAR_AXES}
    rch_tr, rch_va = reach[tr], reach[va]
    meta_va = [meta[i] for i in va]
    time_log_tr = time_log[tr] if time_log is not None else None
    time_log_va = time_log[va] if time_log is not None else None
    metric_log_tr = metric_log[tr] if metric_log is not None else None
    metric_log_va = metric_log[va] if metric_log is not None else None

    # --- Teacher
    bytes_quantile = args.bytes_quantile if args.objective == "zensim_strict" else None
    (
        t_bytes,
        t_per_axis,
        scalar_means,
        t_time,
        time_means,
        t_metric,
        metric_means,
    ) = train_teacher_per_cell(
        Xs_tr, bl_tr, scalars_tr, rch_tr, n_cells,
        bytes_quantile=bytes_quantile,
        time_log_tr=time_log_tr,
        metric_log_tr=metric_log_tr,
    )
    sys.stderr.write("\nGenerating teacher soft targets (val + train)...\n")
    bytes_pred_tr = teacher_predict_all(t_bytes, Xs_tr, np.nanmean(bl_tr, axis=0), n_cells)
    bytes_pred_va = teacher_predict_all(t_bytes, Xs_va, np.nanmean(bl_tr, axis=0), n_cells)
    scalar_pred_tr = {
        axis: teacher_predict_all(t_per_axis[axis], Xs_tr, scalar_means[axis], n_cells)
        for axis in SCALAR_AXES
    }
    scalar_pred_va = {
        axis: teacher_predict_all(t_per_axis[axis], Xs_va, scalar_means[axis], n_cells)
        for axis in SCALAR_AXES
    }
    if has_time_head:
        time_means_safe = np.where(np.isnan(time_means), 0.0, time_means)
        time_pred_tr = teacher_predict_all(t_time, Xs_tr, time_means_safe, n_cells)
        time_pred_va = teacher_predict_all(t_time, Xs_va, time_means_safe, n_cells)
    else:
        time_pred_tr = None
        time_pred_va = None
    if has_metric_head:
        metric_means_safe = np.where(np.isnan(metric_means), 0.0, metric_means)
        metric_pred_tr = teacher_predict_all(t_metric, Xs_tr, metric_means_safe, n_cells)
        metric_pred_va = teacher_predict_all(t_metric, Xs_va, metric_means_safe, n_cells)
    else:
        metric_pred_tr = None
        metric_pred_va = None

    all_mask = np.ones(n_cells, dtype=bool)
    teacher_argmin = evaluate_argmin(bytes_pred_va, bl_va, rch_va, meta_va, all_mask)
    teacher_scalars = evaluate_scalars(scalar_pred_va, scalars_va, rch_va)
    sys.stderr.write(
        f"\nTeacher metrics: argmin mean overhead {teacher_argmin['mean_pct']:.2f}% "
        f"argmin_acc {teacher_argmin['argmin_acc']:.1%}\n"
    )
    sys.stderr.write(
        "  scalar RMSE: " + "  ".join(
            f"{axis} {teacher_scalars[axis]:.4f}" for axis in SCALAR_AXES
        ) + "\n"
    )

    # --- Student
    # Soft targets: bytes + (time?) + (one block per scalar axis), each
    # block n_cells wide. Layout matches `output_layout` emitted in the
    # bake manifest.
    soft_blocks = [bytes_pred_tr]
    if time_pred_tr is not None:
        soft_blocks.append(time_pred_tr)
    if metric_pred_tr is not None:
        soft_blocks.append(metric_pred_tr)
    soft_blocks.extend(scalar_pred_tr[axis] for axis in SCALAR_AXES)
    soft_tr = np.concatenate(soft_blocks, axis=1)
    hidden_repr = "x".join(str(x) for x in hidden_layer_sizes)
    sys.stderr.write(
        f"\nTraining MLP student (hidden={hidden_repr}, output_dim={soft_tr.shape[1]})...\n"
    )

    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    if args.activation == "leakyrelu":
        sys.stderr.write("  using PyTorch backend (LeakyReLU(0.01))\n")
        student = _train_torch_leakyrelu_student(
            X_tr=Xe_tr_s,
            Y_tr=soft_tr,
            hidden_layer_sizes=hidden_layer_sizes,
            lr=2e-3,
            batch_size=512,
            max_iter=500,
            seed=SEED,
            n_iter_no_change=30,
            tol=1e-6,
        )
    else:
        student = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
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
    sys.stderr.write(f"  trained, final loss={student.loss_:.4f}, n_iter={student.n_iter_}\n")

    Y_va_pred = student.predict(Xe_va_s)
    pred_bytes = Y_va_pred[:, :n_cells]
    student_pred_scalars = {
        axis: Y_va_pred[:, (i + 1) * n_cells : (i + 2) * n_cells]
        for i, axis in enumerate(SCALAR_AXES)
    }

    student_argmin = evaluate_argmin(pred_bytes, bl_va, rch_va, meta_va, all_mask)
    student_scalars = evaluate_scalars(student_pred_scalars, scalars_va, rch_va)
    sys.stderr.write(
        f"\nStudent metrics: argmin mean overhead {student_argmin['mean_pct']:.2f}% "
        f"argmin_acc {student_argmin['argmin_acc']:.1%}\n"
    )
    sys.stderr.write(
        "  scalar RMSE: " + "  ".join(
            f"{axis} {student_scalars[axis]:.4f}" for axis in SCALAR_AXES
        ) + "\n"
    )

    # --- Diagnostics: also evaluate student on TRAIN to detect overfit
    Y_tr_pred = student.predict(Xe_tr_s)
    pred_bytes_tr = Y_tr_pred[:, :n_cells]
    meta_tr = [meta[i] for i in tr]
    student_argmin_tr = evaluate_argmin(pred_bytes_tr, bl_tr, rch_tr, meta_tr, all_mask)
    sys.stderr.write(
        f"  train: mean overhead {student_argmin_tr['mean_pct']:.2f}% "
        f"argmin_acc {student_argmin_tr['argmin_acc']:.1%} "
        f"(gap to val: {student_argmin['mean_pct'] - student_argmin_tr['mean_pct']:+.2f}pp)\n"
    )

    # Per-row val breakdown for stratification
    val_per_row = evaluate_argmin_per_row(
        pred_bytes, bl_va, rch_va, meta_va, all_mask
    )
    by_zq, by_size, by_zq_size = stratify_overheads(val_per_row)
    worst = worst_case_rows(val_per_row, top_pct=1.0, max_n=20)

    if args.dump_overheads is not None:
        # CSV for downstream plotting. One row per (image, size, zq)
        # val decision. Overhead is the relative cost vs the per-row
        # oracle minimum (0.0 = picker matched the optimum).
        args.dump_overheads.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_overheads.open("w") as fh:
            fh.write("image\tsize_class\tzq\tpick\tactual_best\toverhead\n")
            for r in val_per_row:
                fh.write(
                    f"{r['image']}\t{r['size_class']}\t{r['zq']}\t"
                    f"{r['pick']}\t{r['actual_best']}\t{r['overhead']}\n"
                )
        sys.stderr.write(f"  wrote per-row overheads → {args.dump_overheads}\n")
    per_cell = per_cell_diagnostics(cells, pred_bytes, bl_va, rch_va, n_cells)
    mlp_health = scan_mlp_weights(student, Xe_va_s)

    # Per-feature distribution bounds over the train split — shipped
    # in the manifest so codecs can do runtime OOD checks.
    train_keys = {(meta[i][0], meta[i][1]) for i in tr}
    feature_bounds = compute_feature_bounds(feats, train_keys, feat_cols)

    # --- Per-zq reach-rate gate (zensim_strict only; recorded
    # always so the manifest is shape-stable across profiles)
    reach_safety = compute_reach_safe_cells(
        bl_tr, rch_tr, meta_tr, n_cells, ZQ_TARGETS, args.reach_threshold
    )

    # Size-invariance discipline: count training rows per
    # (size_class, zq) so the safety gate can flag a starved sweep
    # corner before the picker ships a model that can't actually
    # serve every (width, height) the codec is asked to encode.
    train_rows_by_size_zq = count_train_rows_by_size_zq(
        meta_tr, SIZE_CLASSES, ZQ_TARGETS
    )

    # --- Optional time + metric head R² (held-out, per cell)
    time_head_r2 = None
    if has_time_head:
        time_head_r2 = evaluate_per_cell_r2(time_pred_va, time_log_va, rch_va)
    metric_head_r2 = None
    if has_metric_head:
        metric_head_r2 = evaluate_per_cell_r2(metric_pred_va, metric_log_va, rch_va)
    # Budget infeasible fraction: (image, size) pairs where every cell
    # is over budget at every zq target. Denominator is total candidate
    # (image, size) pairs in the pareto (not just meta — meta excludes
    # pairs that never reached any zq, which is exactly the infeasible
    # set when budget filter is active).
    total_pairs = len({(image, size) for (image, size, _w, _h) in pareto.keys()})
    budget_infeasible_fraction = (
        len(infeasible) / total_pairs
        if (infeasible and total_pairs > 0)
        else 0.0
    )

    # --- Safety report: assemble + check thresholds
    diag = {
        "argmin": {"train": student_argmin_tr, "val": student_argmin},
        "by_zq": by_zq,
        "by_size": by_size,
        "by_zq_size": by_zq_size,
        "train_rows_by_size_zq": train_rows_by_size_zq,
        "worst_case": worst,
        "per_cell": per_cell,
        "mlp": mlp_health,
        "feature_bounds": feature_bounds,
        "reach_safety": reach_safety,
        "time_head_r2": time_head_r2,
        "metric_head_r2": metric_head_r2,
        "budget_infeasible_fraction": budget_infeasible_fraction,
        # Sweep ceilings: did the codec's harness emit
        # `effective_max_zensim` in the Pareto TSV? Drives the
        # UNCAPPED_ZQ_GRID safety gate. See imazen/zenanalyze#51.
        "sweep_ceilings": {
            "has_effective_max_zensim": bool(has_ceiling_column),
            "n_with_ceiling": int(
                sum(1 for v in ceilings.values() if v is not None)
            ) if has_ceiling_column else 0,
            "max_target_zq": int(max(ZQ_TARGETS)) if ZQ_TARGETS else 0,
        },
    }
    thresholds = dict(DEFAULT_SAFETY_THRESHOLDS)
    codec_thresholds = getattr(
        sys.modules.get(parse_config_name.__module__, sys.modules[__name__]),
        "SAFETY_THRESHOLDS",
        None,
    )
    if codec_thresholds:
        thresholds.update(codec_thresholds)
    passed, violations = safety_check(diag, thresholds, args.objective)
    safety_report = {
        "passed": passed,
        "violations": violations,
        "thresholds": thresholds,
        "diagnostics": diag,
    }
    if violations:
        sys.stderr.write(
            "\n" + "=" * 70 + "\n"
            "  ⚠ SAFETY VIOLATIONS DETECTED — picker may produce dangerous results\n"
            + "=" * 70 + "\n"
        )
        for v in violations:
            sys.stderr.write(f"  • {v}\n")
        sys.stderr.write("=" * 70 + "\n")
    else:
        sys.stderr.write("\n✓ All safety thresholds passed.\n")

    # --- Persist
    n_params = sum(c.size + i.size for c, i in zip(student.coefs_, student.intercepts_))
    # Output layout: bytes_log first, then (optional) time_log, then one
    # block per scalar axis. Each block is n_cells wide. Codec runtime
    # reads the manifest to find which slice of the output vector is
    # which head.
    output_layout = {"bytes_log": [0, n_cells]}
    next_block = 1
    if has_time_head:
        output_layout["time_log"] = [next_block * n_cells, (next_block + 1) * n_cells]
        next_block += 1
    if has_metric_head:
        output_layout["metric_log"] = [next_block * n_cells, (next_block + 1) * n_cells]
        next_block += 1
    for axis in SCALAR_AXES:
        output_layout[axis] = [next_block * n_cells, (next_block + 1) * n_cells]
        next_block += 1
    # Sentinel record for runtime — codec config supplies values.
    sentinels_for_manifest = {
        axis: float(SCALAR_SENTINELS[axis])
        for axis in SCALAR_AXES if axis in SCALAR_SENTINELS
    }
    out = {
        "n_inputs": int(Xe.shape[1]),
        "n_outputs": output_dim,
        "n_cells": n_cells,
        "safety_profile": args.objective,
        "config_names": {int(k): v for k, v in CONFIG_NAMES.items()},
        "feat_cols": feat_cols,
        "scaler_mean": scaler.mean_.tolist(),
        # `scaler_scale` stores sklearn's `StandardScaler.scale_`
        # directly — that attribute IS the standard deviation
        # (`np.sqrt(var_)`). The Rust runtime in
        # `zenpredict::inference` divides by this on every forward
        # pass — same operation sklearn's `transform` applies, same
        # operation `scaler.fit_transform(X_tr)` applied to produce
        # the standardized inputs the MLP was trained on.
        "scaler_scale": scaler.scale_.tolist(),
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(student.coefs_, student.intercepts_)
        ],
        "activation": args.activation,
        "hybrid_heads_manifest": {
            "n_cells": n_cells,
            "cells": cells,
            "categorical_axes": list(CATEGORICAL_AXES),
            "scalar_axes": list(SCALAR_AXES),
            "output_layout": output_layout,
            "scalar_sentinels": sentinels_for_manifest,
            # Back-compat alias for runtime code that still reads the
            # old key. New code should use scalar_sentinels["lambda"].
            "lambda_notrellis_sentinel": (
                sentinels_for_manifest.get("lambda")
                if "lambda" in SCALAR_AXES
                else getattr(
                    sys.modules.get(parse_config_name.__module__, sys.modules[__name__]),
                    "LAMBDA_NOTRELLIS_SENTINEL",
                    0.0,
                )
            ),
        },
        "training_objective": {
            "name": args.objective,
            "bytes_quantile": (
                args.bytes_quantile if args.objective == "zensim_strict" else None
            ),
            "reach_threshold": args.reach_threshold,
            # Metric the picker was trained against. Codec runtime checks
            # this matches the metric the caller is targeting before
            # using the bake.
            "metric_name": METRIC_COLUMN,
            "metric_direction": METRIC_DIRECTION,
            "time_column": TIME_COLUMN if has_time_head else None,
            "has_time_head": bool(has_time_head),
            "has_metric_head": bool(has_metric_head),
            # Budget filter applied at label-extraction time. When > 0,
            # within-cell candidates were restricted to time ≤
            # baseline_ms[size_class] × this multiplier. Codec runtime
            # should apply the same filter at inference for parity.
            "time_budget_multiplier": float(args.time_budget_multiplier),
            # Median time per size_class — codec runtime needs these to
            # compute budgets at inference matching how labels were
            # extracted.
            "time_baselines_ms": {str(k): float(v) for k, v in time_baselines.items()},
            # Count of (image, size) pairs where every cell is over
            # budget at every zq target. Drives BUDGET_INFEASIBLE.
            "n_infeasible_pairs": int(len(infeasible)),
        },
        "reach_safety": reach_safety,
        "teacher_metrics": {"argmin": teacher_argmin, "scalars": teacher_scalars},
        "student_metrics": {"argmin": student_argmin, "scalars": student_scalars},
        "safety_report": safety_report,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    sys.stderr.write(
        f"\nWrote {OUT_JSON} ({n_params} weights, {n_params*2/1024:.1f} KB f16)\n"
    )

    # --- Report
    lines = []
    def w(s):
        lines.append(s)
        sys.stderr.write(s + "\n")

    scalar_axes_label = ", ".join(SCALAR_AXES) if SCALAR_AXES else "none"
    w(f"\n# Hybrid-heads picker — categorical bytes + scalar ({scalar_axes_label})")
    w(f"Safety profile: {args.objective}")
    if args.objective == "zensim_strict":
        w(f"  bytes head: quantile q={args.bytes_quantile}")
        w(f"  reach gate: cells with reach_rate < {args.reach_threshold} per zq are masked")
        # Quick summary of how many cells survive the gate at each zq
        # band — useful sanity-check during training.
        for zq_str, info in sorted(
            reach_safety["by_zq"].items(), key=lambda kv: int(kv[0])
        ):
            n_safe = sum(1 for s in info["safe"] if s)
            w(
                f"    zq={int(zq_str):3d}: {n_safe:>2d}/{n_cells} cells safe "
                f"(rates: min {min(info['reach_rate']):.2f}, "
                f"max {max(info['reach_rate']):.2f})"
            )
    else:
        w("  bytes head: mean (squared error)")
        w("  reach gate: none — any reachable cell allowed at inference")
    w(f"Train rows: {len(tr)}, val rows: {len(va)}")
    w(f"n_cells: {n_cells}, output_dim: {output_dim}")
    arch_str = " -> ".join(
        [str(Xe.shape[1])] + [str(h) for h in hidden_layer_sizes] + [str(output_dim)]
    )
    w(f"Student: MLP {arch_str}, "
      f"{n_params} params (~{n_params*2/1024:.1f} KB f16)")
    w("")
    w("## Categorical cells")
    for c in cells:
        w(f"  {c['id']:>2d}: {c['label']:30s}  ({len(c['member_config_ids'])} member configs)")
    w("")
    w("## Argmin (categorical) — vs reachable per-row optimal")
    w(f"  Teacher: mean {teacher_argmin['mean_pct']:.2f}%  argmin_acc {teacher_argmin['argmin_acc']:.1%}")
    w(f"  Student: mean {student_argmin['mean_pct']:.2f}%  argmin_acc {student_argmin['argmin_acc']:.1%}")
    w("")
    if SCALAR_AXES:
        w("## Scalar regression RMSE")
        for axis in SCALAR_AXES:
            range_lo, range_hi = SCALAR_DISPLAY_RANGES.get(axis, (None, None))
            range_str = f", range {range_lo}..{range_hi}" if range_lo is not None else ""
            w(f"  Teacher {axis} RMSE: {teacher_scalars[axis]:.4f}  "
              f"(MAE {teacher_scalars[axis + '_mae']:.4f}{range_str})")
            w(f"  Student {axis} RMSE: {student_scalars[axis]:.4f}  "
              f"(MAE {student_scalars[axis + '_mae']:.4f})")

    OUT_LOG.write_text("\n".join(lines))

    # --- Strict-gate exit. We've already written the JSON + log
    # so reviewers can inspect; only the *exit code* signals the
    # failure. This shape keeps CI red without blocking diagnosis.
    if violations and strict:
        sys.stderr.write(
            f"\nstrict mode: exiting 1 with {len(violations)} unresolved safety "
            f"violation(s). Re-run with --allow-unsafe to override.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
