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
from sklearn.tree import DecisionTreeClassifier


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
    hard_example_mode: str = "none",
    hard_example_alpha: float = 1.0,
    hard_example_ema_window: int = 5,
    hard_example_clip: float = 10.0,
):
    """Drop-in for `MLPRegressor.fit` returning an object that exposes
    `.coefs_`, `.intercepts_`, `.predict`, `.loss_`, `.n_iter_`. The
    network shape, init, loss (MSE), optimizer (Adam), and early-stopping
    schedule mirror sklearn's defaults so the comparison stays apples-to-apples.

    Hard-example weighting (when `hard_example_mode != "none"`):
      After each epoch's parameter updates, compute the per-row squared
      disagreement `mean_j (student_pred_j - Y_tr_j)²` over the FULL
      internal training slice (one extra O(n) forward pass per epoch).
      Maintain an EMA across the last N epochs with α_ema = 1 / N. Per-
      row weight =
          clip(1 + α · ema[i] / median(ema), 1/clip, clip)
      For the first `hard_example_ema_window` epochs the row weights stay
      uniform (no signal yet to weight by). Loss is
      `mean(weight_b · mean_j (pred_bj − target_bj)²)`. Weighting is
      internal-only — the val-loss used for early stopping is the
      unweighted MSE so the stopping criterion stays comparable to the
      uniform run.
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

    # Hard-example weighting state — uniform until activated (after N
    # epochs of EMA warmup). When `hard_example_mode == "none"`, this
    # state is allocated but never touched.
    use_hew = hard_example_mode == "emae"
    n_tr = Xt.shape[0]
    row_weights = torch.ones(n_tr, dtype=torch.float32)
    # EMA of per-row squared disagreement. NaN = "no observation yet";
    # flipped to real values after the first full-train forward pass.
    disagree_ema = torch.full((n_tr,), float("nan"), dtype=torch.float32)
    ema_alpha = 1.0 / max(1, hard_example_ema_window)
    w_min = 1.0 / max(hard_example_clip, 1e-6)
    w_max = hard_example_clip

    for epoch in range(max_iter):
        net.train()
        perm_e = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm_e[i : i + batch_size]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            if use_hew:
                pred = net(xb)
                # mean over output dims, then per-row reweight, then mean
                # over batch. When row_weights == 1.0 everywhere this is
                # exactly `MSELoss()(pred, yb)` to float roundoff.
                per_row_mse = ((pred - yb) ** 2).mean(dim=1)
                loss = (per_row_mse * row_weights[idx]).mean()
            else:
                loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xv), Yv).item()
            if use_hew:
                # One full-train forward to update disagreement EMA.
                # O(n_tr) per epoch — same order as one extra minibatch
                # sweep — and dominates the weighting overhead vs the
                # per-batch reweight. Still tiny vs total epoch cost.
                pred_full = net(Xt)
                d = ((pred_full - Yt) ** 2).mean(dim=1)
                first = torch.isnan(disagree_ema)
                disagree_ema = torch.where(
                    first, d, (1.0 - ema_alpha) * disagree_ema + ema_alpha * d
                )
                # Warmup: first N epochs run uniform weights (the EMA
                # has fewer than `window` samples blended in so its
                # values are noisy / biased toward the very first
                # observations). After `hard_example_ema_window` full-
                # train passes, flip to weighted.
                if epoch + 1 >= hard_example_ema_window:
                    med = disagree_ema.median().clamp(min=1e-12)
                    raw_w = 1.0 + hard_example_alpha * disagree_ema / med
                    row_weights = raw_w.clamp(min=w_min, max=w_max)
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


def _predict_via_coefs(student, X: np.ndarray, activation: str) -> np.ndarray:
    """Numpy forward pass over `student.coefs_/intercepts_`.

    Used after per-head loss normalization mutates the final layer so that
    both backends (sklearn `MLPRegressor` and the torch LeakyReLU student)
    produce predictions consistent with the baked weights. The torch
    student's internal `nn.Sequential` is NOT updated when we mutate
    `coefs_`/`intercepts_`, so calling its native `predict()` after the
    rescale would still return standardized-unit values; routing through
    this helper guarantees natural-unit output for downstream metrics,
    diagnostics, and CSV dumps.

    `activation` selects the hidden-layer nonlinearity. Output layer is
    always linear (regression head).
    """
    a = X.astype(np.float64, copy=False)
    n_layers = len(student.coefs_)
    for li, (W, b) in enumerate(zip(student.coefs_, student.intercepts_)):
        z = a @ W + b
        if li < n_layers - 1:
            if activation == "leakyrelu":
                a = np.where(z > 0, z, 0.01 * z)
            else:
                a = np.maximum(z, 0.0)
        else:
            a = z
    return a.astype(np.float32)


from sklearn.preprocessing import StandardScaler

SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
_SIZE_CANON = ["tiny", "small", "medium", "large"]


def _scope_size_classes(present_order):
    """Reassign the SIZE_CLASSES / SIZE_INDEX grid to the size classes the
    corpus actually covers (canonical tiny<small<medium<large order).

    A web-focused corpus tops out at medium (<=1 MP); the picker can't learn a
    size class with zero renditions, and the DATA_STARVED_SIZE gate would fire
    on an absent size forever (no amount of data fixes a size the corpus never
    contained). Scoping the grid to present sizes is the honest fix: the picker
    models exactly the sizes it has data for, and larger images map to the
    nearest modeled size at inference. Returns the new SIZE_CLASSES.
    """
    global SIZE_CLASSES, SIZE_INDEX
    canon = [s for s in _SIZE_CANON if s in set(present_order)]
    if canon:
        SIZE_CLASSES = canon
        SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}
    return SIZE_CLASSES


HOLDOUT_FRAC = 0.20
SEED = 0xCAFE

CONFIG_NAMES: dict = {}

# zenanalyze-api reuse-key provenance, optionally declared by the codec config
# (`ANALYSIS_PROVENANCE`). Bound in the loader; empty -> no stamps -> the baked
# model safely runs its own feature pass (no reuse). See tools/_provenance.py.
ANALYSIS_PROVENANCE: dict = {}


def _feature_provenance_block(features_path) -> str | None:
    """The fine-grained ``zenanalyze-provenance/1`` block for the FEATURES table —
    from its Parquet key-value metadata, else a ``<stem>.provenance`` sidecar (the
    TSV case). ``None`` if the table is unstamped (legacy / no `api` extractor)."""
    from _provenance import provenance_from_parquet, read_provenance_sidecar

    p = Path(features_path)
    if p.suffix.lower() in (".parquet", ".pq"):
        block = provenance_from_parquet(p)
        if block:
            return block
    return read_provenance_sidecar(p)


def _check_provenance_agreement(block: str, declared: dict) -> None:
    """Warn (don't fail) if the codec config's declared coarse stamps disagree with
    the FEATURES table's actual serialization provenance — a declared
    `analyzer_version` / `feature_config_hash` that mismatches the bytes on disk
    means the reuse-key stamps would misdescribe the training features."""
    from _provenance import parse_provenance_block

    try:
        parsed = parse_provenance_block(block)
    except ValueError as e:
        sys.stderr.write(f"  [provenance] WARNING: unparseable FEATURES block: {e}\n")
        return
    dav = declared.get("analyzer_version")
    if dav and dav.split(".")[:2] != parsed["analyzer_version"].split(".")[:2]:
        sys.stderr.write(
            f"  [provenance] WARNING: declared analyzer_version {dav!r} disagrees "
            f"with FEATURES provenance {parsed['analyzer_version']!r} (major.minor)\n"
        )
    if "feature_config_hash" in declared and declared["feature_config_hash"] != parsed["config_hash"]:
        sys.stderr.write(
            f"  [provenance] WARNING: declared feature_config_hash "
            f"{declared['feature_config_hash']} disagrees with FEATURES provenance "
            f"config_hash {parsed['config_hash']}\n"
        )

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

# Quality-tolerance reach band for the p50 "hug the RD curve, allow both-axis
# error" objective. When > 0, a cell counts as reaching target zq if its
# achieved quality is within REACH_UNDERSHOOT metric points BELOW the target
# (higher-is-better) — letting the picker trade up to that much quality for a
# cheaper RD-frontier point. 0.0 = strict quality-safe (p90 / historical
# default). Set per-run via --reach-undershoot or a codec config's
# REACH_UNDERSHOOT. Read inside the pareto reach construction.
#
# STATUS (measured 2026-06-29): this is the reach-band HALF of the p50 objective.
# A complete p50 also needs an RD-DISTANCE overhead metric (pick vs the per-image
# frontier AT THE ACHIEVED quality) so the gate CREDITS byte-savings-via-quality-
# drift instead of charging it against the now-cheaper band-oracle. Without that
# metric the band makes the overhead look WORSE (jpeg δ=3: K=1 9.34%→9.98%).
# AND — even complete — the band only helps ON-frontier quality-overshoot tails;
# the multi-cell tail (jpeg/webp/avif) is OFF-frontier SUBSAMPLING mis-picks
# (wrong 420/444 at the same quality), which no RD metric forgives. (Full-budget
# chroma re-extraction tested 2026-06-29 too: medium chroma features moved 0.0-
# 0.1% — sampling is NOT the bottleneck; the chroma features are precise but not
# discriminative enough.) Kept as a default-off scaffold for the p90/p50 dial +
# on-frontier cases; see benchmarks/picker_k1_cross_codec_2026-06-29.md.
REACH_UNDERSHOOT: float = 0.0

# K-verify: how many top-predicted cells the DEPLOYED codec encodes + keeps the
# best of (by actual bytes). 1 = single-encode K=1 (the default; most codecs).
# A codec that can afford 2-3 encodes (e.g. zenjpeg) sets VERIFY_K > 1 — the
# gate then evaluates the picker as best-of-top-K, matching deployment, so the
# mean/p99/worst reflect what the codec actually ships. Set via --verify-k or a
# codec config's VERIFY_K. Read inside evaluate_argmin_per_row.
VERIFY_K: int = 1

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

# Per-feature pre-standardize transform. Codec configs declare a
# {feat_name: "log" | "log1p" | "identity"} dict; the trainer applies
# the transform once at feature-load time, BEFORE the StandardScaler
# fit. Standardization captures post-transform mean/scale, so the
# scaler's interpretation matches what the network was trained on.
#
# The selected transform per feature is recorded in the model JSON
# under top-level `feature_transforms` (parallel to `feat_cols`); the
# bake step threads it into the v3 artifact for runtime symmetry.
# Until the codec runtimes consume `feature_transforms`, fresh bakes
# only round-trip correctly when the runtime applies an identical
# pre-transform — codec configs that omit FEATURE_TRANSFORMS (or set
# it empty) train identically to pre-#52 behaviour.
FEATURE_TRANSFORMS: dict = {}

# Per-feature parameter vectors for parameterized variants. Keyed by
# feat name. Required when FEATURE_TRANSFORMS uses `clip_then_log1p`
# (1 param ε), `winsor_p99` (2 params [p1, p99]), or `quantile_bins`
# (N params [edges asc]). Codec configs that only use non-parameterized
# variants (identity / log / log1p / signed_*) leave this empty.
FEATURE_TRANSFORM_PARAMS: dict = {}

# Per-output OutputSpec metadata, keyed by head name (e.g. "bytes_log",
# or any SCALAR_AXES name). Threaded verbatim into the model JSON so
# downstream `inject_v3_specs.py` / `bake_picker.py` can expand them
# into the per-output array without re-importing the codec config.
# Empty default → behaviour identical to pre-#52 (no per-output
# bounds/discrete_set/transform applied at the runtime).
OUTPUT_SPECS: dict = {}

# Per-output sparse hand-tune overrides. List of `{idx, value}` dicts
# in the laid-out output index space. `value=None` forces
# OutputValue::Default at runtime.
SPARSE_OVERRIDES: list = []


def load_codec_config(name: str, drop_features=None):
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
    global METRIC_COLUMN, METRIC_DIRECTION, TIME_COLUMN, REACH_UNDERSHOOT, VERIFY_K
    global FEATURE_TRANSFORMS, FEATURE_TRANSFORM_PARAMS, OUTPUT_SPECS, SPARSE_OVERRIDES
    global ANALYSIS_PROVENANCE
    mod = importlib.import_module(name)
    PARETO = Path(mod.PARETO)
    FEATURES = Path(mod.FEATURES)
    OUT_LOG = Path(mod.OUT_LOG)
    OUT_JSON = Path(mod.OUT_JSON)
    ZQ_TARGETS = list(mod.ZQ_TARGETS)
    KEEP_FEATURES = list(mod.KEEP_FEATURES)
    # Leave-one-out (LOO) ablation hook: drop named features before any downstream
    # use. Spearman/correlation cleanup only catches monotonic redundancy; LOO
    # retrains without each feature to measure its true marginal contribution.
    if drop_features:
        drop = {f.strip() for f in drop_features if f and f.strip()}
        unknown = drop - set(KEEP_FEATURES)
        if unknown:
            sys.stderr.write(
                f"  load_codec_config: --drop-features had {len(unknown)} name(s) "
                f"not in KEEP_FEATURES (ignored): {sorted(unknown)[:5]}\n"
            )
        _before = len(KEEP_FEATURES)
        KEEP_FEATURES = [f for f in KEEP_FEATURES if f not in drop]
        sys.stderr.write(
            f"  load_codec_config: dropped {_before - len(KEEP_FEATURES)} feature(s) "
            f"-> KEEP_FEATURES {_before} -> {len(KEEP_FEATURES)}\n"
        )
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
    # Optional p50 RD-hugging quality-tolerance band (--reach-undershoot in main
    # overrides this). A codec config sets REACH_UNDERSHOOT > 0 for its p50
    # variant; absent / 0 = the strict p90 quality-safe default.
    if hasattr(mod, "REACH_UNDERSHOOT"):
        REACH_UNDERSHOOT = float(mod.REACH_UNDERSHOOT)
    # Optional K-verify (--verify-k in main overrides). A codec config that can
    # afford 2-3 encodes sets VERIFY_K > 1 so the gate evaluates best-of-top-K.
    if hasattr(mod, "VERIFY_K"):
        VERIFY_K = int(mod.VERIFY_K)
    # Optional — zenanalyze-api reuse-key provenance: which zenanalyze version +
    # AnalysisQuery config extracted this codec's features. Recorded into the
    # baked model so it can reuse a shared feature Offer. Undeclared -> no stamps
    # (safe own-pass). See tools/_provenance.py for the declaration shape.
    if hasattr(mod, "ANALYSIS_PROVENANCE"):
        ANALYSIS_PROVENANCE = dict(mod.ANALYSIS_PROVENANCE)
    # Optional — FEATURE_GROUPS validator. Codec configs declare
    # mutual-exclusion groups based on the cross-codec dendrogram +
    # LOO data; the validator enforces `count(KEEP ∩ group) ≤
    # max_picked` per group. See benchmarks/feature_groups_cross_codec
    # _2026-05-02.md for the methodology and the 6 perfect-Jaccard
    # cross-codec clusters that motivated the design.
    if hasattr(mod, "FEATURE_GROUPS"):
        validate_keep_features(KEEP_FEATURES, mod.FEATURE_GROUPS)
    # Optional — per-feature pre-standardize transform. Validated
    # below at feature-load time (unknown transform names raise).
    if hasattr(mod, "FEATURE_TRANSFORMS"):
        FEATURE_TRANSFORMS = dict(mod.FEATURE_TRANSFORMS)
    else:
        FEATURE_TRANSFORMS = {}
    # Optional — per-feature parameter vectors for parameterized
    # transforms (clip_then_log1p / winsor_p99 / quantile_bins).
    # Validated at feature-load time (wrong arity / missing for
    # parameterized variant raises).
    if hasattr(mod, "FEATURE_TRANSFORM_PARAMS"):
        FEATURE_TRANSFORM_PARAMS = dict(mod.FEATURE_TRANSFORM_PARAMS)
    else:
        FEATURE_TRANSFORM_PARAMS = {}
    # Optional — per-output OutputSpec metadata + sparse overrides.
    # Threaded straight through to the bake artifact.
    if hasattr(mod, "OUTPUT_SPECS"):
        OUTPUT_SPECS = dict(mod.OUTPUT_SPECS)
    else:
        OUTPUT_SPECS = {}
    if hasattr(mod, "SPARSE_OVERRIDES"):
        SPARSE_OVERRIDES = list(mod.SPARSE_OVERRIDES)
    else:
        SPARSE_OVERRIDES = []
    return mod


def validate_keep_features(keep, groups):
    """Enforce per-group `max_picked` constraints on KEEP_FEATURES.

    Each entry in `groups` is `{"members": [feat_name, ...],
    "max_picked": int}`. The validator counts overlap between
    `KEEP_FEATURES` and `group["members"]`; if it exceeds
    `max_picked`, raise `ValueError` with a pointer to the cull docs.

    Groups with `max_picked >= len(members)` are unconstrained
    (allow all members). The validator is silent on success.
    """
    keep_set = set(keep)
    violations = []
    for name, group in groups.items():
        members = list(group.get("members", []))
        max_picked = int(group.get("max_picked", len(members)))
        if max_picked >= len(members):
            continue  # unconstrained
        picked = sorted(keep_set & set(members))
        if len(picked) > max_picked:
            violations.append(
                f"  group {name!r}: max_picked={max_picked}, "
                f"got {len(picked)}: {picked}"
            )
    if violations:
        raise ValueError(
            "FEATURE_GROUPS validator failed — KEEP_FEATURES violates "
            "mutual-exclusion constraints:\n"
            + "\n".join(violations)
            + "\n\nPick one feature per group per the cross-codec "
            "dendrogram analysis. See "
            "benchmarks/feature_groups_cross_codec_2026-05-02.md "
            "for group definitions and the LOO ranking that informs "
            "which member to keep."
        )


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


def _read_table_columns(path: Path):
    """Read a TSV / CSV / Parquet file column-wise.

    Auto-detects Parquet by `.parquet` / `.pq` suffix; everything else
    routes through pyarrow's CSV reader with TAB delimiter.

    For multi-GB pareto sweeps the **column-wise interface is
    materially faster than row-wise**. csv.DictReader on the zenwebp
    21.8M-row combined pareto takes 68 s; pyarrow CSV takes 3 s;
    Parquet (zstd) takes 1.9 s. The 36× speedup only materializes
    if downstream code consumes columns directly instead of
    reconstructing per-row dicts (which loses most of the gain to
    Python dict allocation overhead).

    See `~/work/claudehints/topics/parquet-vs-tsv.md` for the
    project-wide format convention and `benchmarks/tsv_to_parquet.py`
    for the converter.

    Returns
    -------
    fieldnames : list[str]
    columns : dict[str, list_or_array]
        Each value is one of:
        - `list` of Python objects (strings, mixed-type, etc.)
        - `numpy.ndarray` for numeric columns (int64, float64)
        Callers should index `columns[name][i]` per row in a tight
        Python loop, NOT reconstruct per-row dicts.
    """
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        table = pq.read_table(path)
    else:
        import pyarrow.csv as pa_csv

        table = pa_csv.read_csv(
            path,
            parse_options=pa_csv.ParseOptions(delimiter="\t"),
            convert_options=pa_csv.ConvertOptions(strings_can_be_null=True),
        )

    fieldnames = list(table.column_names)
    cols: dict = {}
    for name in fieldnames:
        col = table.column(name)
        # Numeric columns → numpy (zero-copy where possible). String
        # columns → Python list. Heuristic: integer/float types use
        # to_numpy(); everything else uses to_pylist().
        try:
            if col.type.bit_width:  # integer / float (raises for non-numeric)
                cols[name] = col.to_numpy(zero_copy_only=False)
            else:
                cols[name] = col.to_pylist()
        except (AttributeError, ValueError):
            cols[name] = col.to_pylist()
    return fieldnames, cols


def _factorize_keys(ip_idx, sc_idx, width_np, height_np):
    """Factorize the 4-tuple per-row key into (unique_keys_int, key_id).

    Fast path: bitpack the four int columns into one int64 when their
    observed value ranges fit in 63 bits combined, then run a 1D
    np.unique. This is ~13× faster than np.unique(axis=0) on 3.5M
    rows (2.8s vs 38s on the zq pareto). Fallback uses a structured
    view, which matches axis=0 in cost.

    Returns the unpacked 2D unique-key array (one row per unique key,
    columns are [ip_idx, sc_idx, width, height]) plus key_id (one
    int per input row).
    """
    maxes = [int(arr.max()) + 1 for arr in (ip_idx, sc_idx, width_np, height_np)]
    bits = [max(1, (m - 1).bit_length()) for m in maxes]
    if sum(bits) <= 63:
        b_ip, b_sc, b_w, b_h = bits
        s_ip = b_sc + b_w + b_h
        s_sc = b_w + b_h
        s_w = b_h
        key_packed = (
            (ip_idx.astype(np.int64) << s_ip)
            | (sc_idx.astype(np.int64) << s_sc)
            | (width_np.astype(np.int64) << s_w)
            | height_np.astype(np.int64)
        )
        unique_packed, key_id = np.unique(key_packed, return_inverse=True)
        ip_u = ((unique_packed >> s_ip) & ((1 << b_ip) - 1)).astype(np.int64)
        sc_u = ((unique_packed >> s_sc) & ((1 << b_sc) - 1)).astype(np.int64)
        w_u = ((unique_packed >> s_w) & ((1 << b_w) - 1)).astype(np.int64)
        h_u = (unique_packed & ((1 << b_h) - 1)).astype(np.int64)
        return np.stack([ip_u, sc_u, w_u, h_u], axis=1), key_id
    arr2d = np.ascontiguousarray(
        np.stack([ip_idx, sc_idx, width_np, height_np], axis=1)
    )
    sd = np.dtype(
        [("a", np.int64), ("b", np.int64), ("c", np.int64), ("d", np.int64)]
    )
    sv = arr2d.view(sd).ravel()
    unique_struct, key_id = np.unique(sv, return_inverse=True)
    return (
        np.stack(
            [
                unique_struct["a"],
                unique_struct["b"],
                unique_struct["c"],
                unique_struct["d"],
            ],
            axis=1,
        ),
        key_id,
    )


def load_pareto(path):
    """Load the Pareto sweep TSV / Parquet (columnar return shape).

    Returns `(rows, ceilings, has_ceiling_column, has_time_column)` where:
      - rows: `{(image_path, size_class, w, h) -> {"config_id": int64[],
        "bytes": int64[], "zensim": float64[], "time_ms": float64[] | None}}`.
        Each per-key entry is a small dict of typed numpy arrays —
        one element per (config, q) sample for that (image, size).
        Downstream code iterates the arrays directly (vectorized
        groupby / reach mask) instead of looping over per-row dicts.
      - ceilings: `{(image_path, size_class) -> effective_max_zensim}`.
      - has_ceiling_column / has_time_column: presence flags.

    Implementation note: Parquet paths take an Arrow-native fast path
    that filters in Arrow, dictionary-encodes the string key columns,
    then factorizes the per-row key via a bitpack. Measured cold-cache
    wall on the 3.5M-row zenjpeg zq pareto drops from 71s (old
    pylist + Python dict loop) to ~14s (~5×). The TSV/CSV path still
    routes through `_read_table_columns` (`pyarrow.csv.read_csv`) and
    then converges with the Parquet path at the factorize step, so
    they share the bitpack speedup.

    See imazen/zenanalyze#51 for the cross-codec design context.
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        import pyarrow.compute as pc

        table = pq.read_table(path)
        fieldnames = list(table.column_names)
        has_ceiling_column = "effective_max_zensim" in fieldnames
        has_time_column = TIME_COLUMN in fieldnames
        if METRIC_COLUMN not in fieldnames:
            raise ValueError(
                f"pareto file {path} is missing METRIC_COLUMN={METRIC_COLUMN!r}; "
                f"available columns: {fieldnames}"
            )

        valid = pc.is_finite(table[METRIC_COLUMN])
        valid = pc.and_(valid, pc.greater_equal(table["config_id"], 0))
        valid = pc.and_(valid, pc.greater(table["bytes"], 0))
        valid = pc.and_(valid, pc.is_finite(table["width"]))
        valid = pc.and_(valid, pc.is_finite(table["height"]))
        table = pc.filter(table, valid)
        n = len(table)
        if n == 0:
            return {}, {}, has_ceiling_column, has_time_column

        config_id_np = table["config_id"].to_numpy().astype(np.int64)
        bytes_np = table["bytes"].to_numpy().astype(np.int64)
        metric_np = table[METRIC_COLUMN].to_numpy().astype(np.float64)
        width_np = table["width"].to_numpy().astype(np.int64)
        height_np = table["height"].to_numpy().astype(np.int64)
        time_np = (
            table[TIME_COLUMN].to_numpy().astype(np.float64)
            if has_time_column else None
        )
        ceil_np = (
            table["effective_max_zensim"].to_numpy().astype(np.float64)
            if has_ceiling_column else None
        )

        def _enc(name):
            col = table[name].combine_chunks()
            enc = pc.dictionary_encode(col)
            return (
                enc.indices.to_numpy().astype(np.int64),
                enc.dictionary.to_pylist(),
            )

        ip_idx, ip_vocab = _enc("image_path")
        sc_idx, sc_vocab = _enc("size_class")
        cn_idx, cn_vocab = _enc("config_name")
    else:
        # TSV / CSV path. _read_table_columns already routes through
        # pyarrow.csv internally and returns numpy arrays for numeric
        # columns + python lists for strings; we then mirror the
        # Parquet path's downstream steps.
        fieldnames, cols = _read_table_columns(Path(path))
        has_ceiling_column = "effective_max_zensim" in fieldnames
        has_time_column = TIME_COLUMN in fieldnames
        if METRIC_COLUMN not in fieldnames:
            raise ValueError(
                f"pareto file {path} is missing METRIC_COLUMN={METRIC_COLUMN!r}; "
                f"available columns: {fieldnames}"
            )

        metric_np = np.asarray(cols[METRIC_COLUMN], dtype=np.float64)
        config_id_f = np.asarray(cols["config_id"], dtype=np.float64)
        bytes_f = np.asarray(cols["bytes"], dtype=np.float64)
        width_f = np.asarray(cols["width"], dtype=np.float64)
        height_f = np.asarray(cols["height"], dtype=np.float64)
        time_np = (
            np.asarray(cols[TIME_COLUMN], dtype=np.float64)
            if has_time_column else None
        )
        ceil_np = (
            np.asarray(cols["effective_max_zensim"], dtype=np.float64)
            if has_ceiling_column else None
        )
        valid = (
            np.isfinite(metric_np)
            & np.isfinite(config_id_f) & (config_id_f >= 0)
            & np.isfinite(bytes_f) & (bytes_f > 0)
            & np.isfinite(width_f) & np.isfinite(height_f)
        )
        n_dropped = int((~valid).sum())
        if n_dropped:
            metric_np = metric_np[valid]
            config_id_f = config_id_f[valid]
            bytes_f = bytes_f[valid]
            width_f = width_f[valid]
            height_f = height_f[valid]
            if time_np is not None:
                time_np = time_np[valid]
            if ceil_np is not None:
                ceil_np = ceil_np[valid]
        config_id_np = config_id_f.astype(np.int64)
        bytes_np = bytes_f.astype(np.int64)
        width_np = width_f.astype(np.int64)
        height_np = height_f.astype(np.int64)
        n = len(width_np)
        if n == 0:
            return {}, {}, has_ceiling_column, has_time_column

        image_path_full = cols["image_path"]
        size_class_full = cols["size_class"]
        config_name_full = cols["config_name"]
        if n_dropped:
            valid_list = valid.tolist()
            image_path = [v for v, keep in zip(image_path_full, valid_list) if keep]
            size_class = [v for v, keep in zip(size_class_full, valid_list) if keep]
            config_name = [v for v, keep in zip(config_name_full, valid_list) if keep]
        else:
            image_path = image_path_full
            size_class = size_class_full
            config_name = config_name_full

        # Build per-column dictionary indices via pandas-free factorize
        # to share the downstream bitpack path with the Parquet branch.
        ip_vocab: list = []
        ip_lookup: dict = {}
        ip_idx_arr = np.empty(n, dtype=np.int64)
        for i, v in enumerate(image_path):
            idx = ip_lookup.get(v)
            if idx is None:
                idx = len(ip_vocab)
                ip_lookup[v] = idx
                ip_vocab.append(v)
            ip_idx_arr[i] = idx
        ip_idx = ip_idx_arr
        sc_vocab = []
        sc_lookup: dict = {}
        sc_idx_arr = np.empty(n, dtype=np.int64)
        for i, v in enumerate(size_class):
            idx = sc_lookup.get(v)
            if idx is None:
                idx = len(sc_vocab)
                sc_lookup[v] = idx
                sc_vocab.append(v)
            sc_idx_arr[i] = idx
        sc_idx = sc_idx_arr
        cn_vocab = []
        cn_lookup: dict = {}
        cn_idx_arr = np.empty(n, dtype=np.int64)
        for i, v in enumerate(config_name):
            idx = cn_lookup.get(v)
            if idx is None:
                idx = len(cn_vocab)
                cn_lookup[v] = idx
                cn_vocab.append(v)
            cn_idx_arr[i] = idx
        cn_idx = cn_idx_arr

    # Factorize per-row keys via bitpack (5× faster than np.unique axis=0).
    unique_keys, key_id = _factorize_keys(ip_idx, sc_idx, width_np, height_np)
    keys = [
        (ip_vocab[int(k[0])], sc_vocab[int(k[1])], int(k[2]), int(k[3]))
        for k in unique_keys
    ]

    # Record config_name on first sighting of each cid (vectorized).
    config_names = CONFIG_NAMES
    cid_order = np.argsort(config_id_np, kind="stable")
    cid_sorted = config_id_np[cid_order]
    first_in_group = np.concatenate(([True], cid_sorted[1:] != cid_sorted[:-1]))
    first_idxs = cid_order[first_in_group]
    for i in first_idxs:
        cid = int(config_id_np[i])
        if cid not in config_names:
            config_names[cid] = cn_vocab[int(cn_idx[i])]

    # Sort by key_id so contiguous slabs == per-key entries.
    sort_idx = np.argsort(key_id, kind="stable")
    key_id_s = key_id[sort_idx]
    boundaries = np.flatnonzero(np.diff(key_id_s)) + 1
    boundaries = np.concatenate(([0], boundaries, [n]))

    config_id_s = config_id_np[sort_idx]
    bytes_s = bytes_np[sort_idx]
    metric_s = metric_np[sort_idx]
    time_s = time_np[sort_idx] if time_np is not None else None
    ceil_s = ceil_np[sort_idx] if ceil_np is not None else None

    rows: dict = {}
    ceilings: dict = {}
    for gi in range(len(boundaries) - 1):
        lo = int(boundaries[gi])
        hi = int(boundaries[gi + 1])
        k = keys[int(key_id_s[lo])]
        entry: dict = {
            "config_id": config_id_s[lo:hi],
            "bytes": bytes_s[lo:hi],
            "zensim": metric_s[lo:hi],
        }
        if time_s is not None:
            entry["time_ms"] = time_s[lo:hi]
        rows[k] = entry
        if ceil_s is not None:
            ceil_key = (k[0], k[1])
            if ceil_key not in ceilings:
                slab = ceil_s[lo:hi]
                ok = np.isfinite(slab)
                if ok.any():
                    ceilings[ceil_key] = float(slab[ok][0])
    return rows, ceilings, has_ceiling_column, has_time_column


_VALID_FEATURE_TRANSFORMS = {
    "identity",
    "log",
    "log1p",
    "signed_log1p",
    "signed_sqrt",
    "signed_cbrt",
    "clip_then_log1p",
    "winsor_p99",
    "quantile_bins",
    # Stacked variants (zenpredict 0.2.1+).
    "winsor_then_log",
    "winsor_then_log1p",
    "winsor_then_signed_cbrt",
    "signed_cbrt_then_winsor",
    "clip_then_log1p_then_winsor",
}


def _apply_feature_transform(
    name: str,
    transform: str,
    value: float,
    params: list[float] | None = None,
) -> float:
    """Apply a per-feature pre-standardize transform. Matches the
    runtime variants in [`zenpredict::FeatureTransform`] verbatim.

    Non-parameterized variants:
      - "identity"      — no-op.
      - "log"           — natural log; non-positive values clamp to
                          log(1e-12).
      - "log1p"         — log(1+x); negatives clip to 0.
      - "signed_log1p"  — sign(x) · log1p(|x|).
      - "signed_sqrt"   — sign(x) · sqrt(|x|).
      - "signed_cbrt"   — sign(x) · cbrt(|x|).

    Parameterized variants (`params` required):
      - "clip_then_log1p" — 1 param [ε]: log1p(max(0, x − ε)).
      - "winsor_p99"     — 2 params [p1, p99]: clamp x to [p1, p99].
      - "quantile_bins"  — N params [edges sorted asc]: bin index / N.

    Unknown transform names or wrong param counts raise — codec
    configs are authoritative, a typo silently breaking training is
    worse than crashing the run.
    """
    if transform == "identity":
        return value
    if transform == "log":
        if value <= 0.0:
            return math.log(1e-12)
        return math.log(value)
    if transform == "log1p":
        if value < 0.0:
            return 0.0
        return math.log1p(value)
    if transform == "signed_log1p":
        s = 1.0 if value >= 0.0 else -1.0
        return s * math.log1p(abs(value))
    if transform == "signed_sqrt":
        s = 1.0 if value >= 0.0 else -1.0
        return s * math.sqrt(abs(value))
    if transform == "signed_cbrt":
        s = 1.0 if value >= 0.0 else -1.0
        # math.cbrt available 3.11+; fall back to copysign(abs**(1/3), x).
        try:
            return s * (abs(value) ** (1.0 / 3.0))
        except (OverflowError, ValueError):
            return 0.0
    if transform == "clip_then_log1p":
        eps = (params[0] if params else 0.0)
        y = value - eps
        if y < 0.0:
            y = 0.0
        return math.log1p(y)
    if transform == "winsor_p99":
        if not params or len(params) < 2:
            return value
        lo, hi = params[0], params[1]
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value
    if transform == "quantile_bins":
        if not params:
            return value
        idx = 0.0
        for edge in params:
            if value >= edge:
                idx += 1.0
        return idx / len(params)
    if transform == "winsor_then_log":
        # 2 params [p1, p99]; p1 must be > 0 (validated at bake time).
        if not params or len(params) < 2:
            return value
        lo, hi = params[0], params[1]
        y = lo if value < lo else (hi if value > hi else value)
        return math.log(y) if y > 0.0 else math.log(1e-12)
    if transform == "winsor_then_log1p":
        # 2 params [p1, p99]; p1 > -1.
        if not params or len(params) < 2:
            return value
        lo, hi = params[0], params[1]
        y = lo if value < lo else (hi if value > hi else value)
        return math.log1p(max(y, -0.9999))
    if transform == "winsor_then_signed_cbrt":
        if not params or len(params) < 2:
            return value
        lo, hi = params[0], params[1]
        y = lo if value < lo else (hi if value > hi else value)
        s = 1.0 if y >= 0.0 else -1.0
        try:
            return s * (abs(y) ** (1.0 / 3.0))
        except (OverflowError, ValueError):
            return 0.0
    if transform == "signed_cbrt_then_winsor":
        # signed_cbrt first, then winsor in cbrt-space (q1, q99).
        if not params or len(params) < 2:
            return value
        s = 1.0 if value >= 0.0 else -1.0
        try:
            y = s * (abs(value) ** (1.0 / 3.0))
        except (OverflowError, ValueError):
            y = 0.0
        q1, q99 = params[0], params[1]
        if y < q1:
            return q1
        if y > q99:
            return q99
        return y
    if transform == "clip_then_log1p_then_winsor":
        # 3 params [eps, q1, q99]. clip_then_log1p produces non-negative
        # output; winsor in that log1p-space.
        if not params or len(params) < 3:
            return value
        eps = params[0]
        y = value - eps
        if y < 0.0:
            y = 0.0
        z = math.log1p(y)
        q1, q99 = params[1], params[2]
        if z < q1:
            return q1
        if z > q99:
            return q99
        return z
    raise ValueError(
        f"unknown FEATURE_TRANSFORMS['{name}'] = {transform!r}; "
        f"valid values: {sorted(_VALID_FEATURE_TRANSFORMS)}"
    )


def load_features(path):
    from _picker_lib import _canonical_feat, _is_feature_col

    feats = {}
    fieldnames, columns = _read_table_columns(Path(path))
    all_cols = [c for c in fieldnames if _is_feature_col(c)]
    # Match the (bare `feat_*`) KEEP_FEATURES against possibly-qualified
    # `name@hex8` columns by canonical name; keep the actual column name so the
    # bake carries the qualified identity through. (Mirrors _picker_lib.)
    by_canon = {}
    for c in all_cols:
        by_canon.setdefault(_canonical_feat(c), c)
    cols = [by_canon[k] for kf in KEEP_FEATURES if (k := _canonical_feat(kf)) in by_canon]
    # Per-column transforms and (optional) params from the codec
    # config. Params come from FEATURE_TRANSFORM_PARAMS — a
    # {feat_name: [param0, param1, ...]} dict on the codec config
    # module. Missing params for a parameterized transform raise:
    # a typo there silently turning winsor_p99 into a no-op is
    # worse than crashing.
    transforms = []
    transform_params = []
    PARAM_REQUIREMENTS = {
        "clip_then_log1p": (1, None),  # exactly 1 param
        "winsor_p99": (2, 2),          # exactly 2 params
        "quantile_bins": (1, None),    # ≥ 1 edge
        "winsor_then_log": (2, 2),
        "winsor_then_log1p": (2, 2),
        "winsor_then_signed_cbrt": (2, 2),
        "signed_cbrt_then_winsor": (2, 2),
        "clip_then_log1p_then_winsor": (3, 3),
    }
    for c in cols:
        t = FEATURE_TRANSFORMS.get(c, "identity") if FEATURE_TRANSFORMS else "identity"
        if t not in _VALID_FEATURE_TRANSFORMS:
            raise ValueError(
                f"FEATURE_TRANSFORMS[{c!r}] = {t!r} not in "
                f"{sorted(_VALID_FEATURE_TRANSFORMS)}"
            )
        params = list(FEATURE_TRANSFORM_PARAMS.get(c, [])) \
            if FEATURE_TRANSFORM_PARAMS else []
        req = PARAM_REQUIREMENTS.get(t)
        if req is not None:
            min_p, max_p = req
            if len(params) < min_p or (max_p is not None and len(params) > max_p):
                raise ValueError(
                    f"FEATURE_TRANSFORM_PARAMS[{c!r}] = {params!r} but "
                    f"transform {t!r} requires "
                    f"{'exactly ' + str(min_p) if max_p == min_p else 'at least ' + str(min_p)} param(s)"
                )
        elif params:
            raise ValueError(
                f"FEATURE_TRANSFORM_PARAMS[{c!r}] = {params!r} but "
                f"transform {t!r} takes no params"
            )
        transforms.append(t)
        transform_params.append(params)
    n_transformed = sum(1 for t in transforms if t != "identity")
    if n_transformed:
        n_with_params = sum(1 for p in transform_params if p)
        sys.stderr.write(
            f"FEATURE_TRANSFORMS active on {n_transformed}/{len(cols)} "
            f"columns ({n_with_params} with params): "
            f"{[(c, t) for c, t in zip(cols, transforms) if t != 'identity']}\n"
        )
    image_path = columns["image_path"]
    size_class = columns["size_class"]
    n = len(image_path)
    # Content-aware NaN policy (zenanalyze #49 + the 2026-06-28 tiny-image fix).
    # Percentile/content features (laplacian_variance_p*, aq_map_p*, noise_floor_*,
    # quant_survival_*, luma_kurtosis) are undefined for an image too small to
    # satisfy their min-sample floor. The fix is UPSTREAM, in the feature pipeline:
    # for too-small renditions, mirror-tile the content up to >=128 px and
    # re-extract, then fill ONLY the NaN columns from the tiled run (native-primary
    # + tiled-fill). This gives each tiny image its OWN content-derived percentile
    # values (validated 2026-06-28: all 13 KEEP NaN-features recover near the
    # large-size ground truth; see /mnt/v/output/picker-feature-size-audit-2026-06-28).
    #
    # We deliberately do NOT constant-fill here: a constant makes every too-small
    # image identical in those features → a degenerate tiny picker that PASSES
    # DATA_STARVED with meaningless inputs (gate-gaming). If a KEEP feature is
    # STILL NaN at this point, the upstream tiled-fill did not recover it — we
    # REPORT it loudly (measured) and drop the row, rather than silently filling.
    n_dropped = 0
    nan_cols: dict = {}
    for i in range(n):
        vals = []
        has_nan = False
        for c, t, p in zip(cols, transforms, transform_params):
            v = columns[c][i]
            if v == "" or v is None:
                has_nan = True
                nan_cols[c] = nan_cols.get(c, 0) + 1
                vals.append(float("nan"))
                continue
            try:
                fv = float(v)
            except (ValueError, TypeError):
                has_nan = True
                nan_cols[c] = nan_cols.get(c, 0) + 1
                vals.append(float("nan"))
                continue
            if fv != fv:
                has_nan = True
                nan_cols[c] = nan_cols.get(c, 0) + 1
            elif t != "identity":
                fv = _apply_feature_transform(c, t, fv, p)
            vals.append(fv)
        if has_nan:
            n_dropped += 1
            continue
        feats[(image_path[i], size_class[i])] = np.array(vals, dtype=np.float32)
    if n_dropped:
        sys.stderr.write(
            f"WARNING: dropped {n_dropped}/{n} (image, size) keys with residual NaN in "
            f"KEEP features AFTER upstream mirror-tiled fill — these features were NOT "
            f"recovered by tiling and are REPORTED (never constant-filled): "
            f"{dict(sorted(nan_cols.items(), key=lambda kv: -kv[1]))}. If this is "
            f"non-trivial, fix the extraction or exclude the feature — do NOT "
            f"constant-fill (it games DATA_STARVED with a degenerate picker).\n"
        )
    # Stash the resolved params alongside the transform names so main() can emit
    # them in the bake JSON without re-reading the codec config.
    load_features._last_transform_params = transform_params  # type: ignore[attr-defined]
    return feats, cols, transforms


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
    """True iff the pareto's columnar dict carries a `time_ms` array.
    Column presence is uniform across keys (one TSV pass produced all
    of them), so checking any one nonempty key is enough."""
    for samples in pareto.values():
        return "time_ms" in samples
    return False


def compute_time_baselines(pareto):
    """Median `time_ms` per size_class across all samples in the
    dataset. Returned as `{size_class: median_ms}`. Used to compute
    the per-(image, size_class) budget when --time-budget-multiplier > 0
    filters within-cell candidates.

    Vectorized over the columnar `pareto` shape — `samples["time_ms"]`
    is a numpy float64 array per key; we filter to finite positive
    values and accumulate per size_class.
    """
    by_size: dict = defaultdict(list)
    for (image, size, w, h), samples in pareto.items():
        t_arr = samples.get("time_ms")
        if t_arr is None:
            continue
        ok = np.isfinite(t_arr) & (t_arr > 0)
        if ok.any():
            by_size[size].append(t_arr[ok])
    baselines = {}
    for sz, chunks in by_size.items():
        if chunks:
            all_t = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            baselines[sz] = float(np.median(all_t))
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
    safety_default_cell_idx: int | None = None,
    safety_speed_tol: float = 1.05,
    safety_bytes_min_gain: float = 0.99,
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

    higher_is_better = METRIC_DIRECTION == "higher_better"
    # Pre-compute per-config → cell map as a numpy lookup so we can
    # gather the cell index for a vector of config_ids in one op.
    if config_to_cell:
        max_cid = max(config_to_cell.keys())
        cell_lookup = np.full(max_cid + 1, -1, dtype=np.int64)
        for cid, c in config_to_cell.items():
            cell_lookup[cid] = c
        scalar_lookup = {
            axis: np.full(max_cid + 1, np.nan, dtype=np.float64)
            for axis in SCALAR_AXES
        }
        for cid, p in parsed_all.items():
            for axis in SCALAR_AXES:
                v = p.get(axis)
                if v is not None:
                    scalar_lookup[axis][cid] = v
    else:
        cell_lookup = np.empty(0, dtype=np.int64)
        scalar_lookup = {axis: np.empty(0, dtype=np.float64) for axis in SCALAR_AXES}

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

        # Columnar samples — typed numpy arrays from load_pareto.
        cfg_arr = samples["config_id"]
        bytes_arr = samples["bytes"]
        metric_arr = samples["zensim"]
        time_arr = samples.get("time_ms") if has_time else None
        if len(cfg_arr) == 0:
            continue

        # Sort once by config_id; "per-config best" becomes contiguous
        # slicing instead of a defaultdict(list) loop. Stable sort
        # preserves intra-config q ordering for debugging clarity.
        sort_idx = np.argsort(cfg_arr, kind="stable")
        cfg_s = cfg_arr[sort_idx]
        bytes_sorted = bytes_arr[sort_idx]
        metric_sorted = metric_arr[sort_idx]
        time_sorted = time_arr[sort_idx] if time_arr is not None else None

        # Group boundaries within the sorted array.
        diffs = np.flatnonzero(np.diff(cfg_s)) + 1
        group_starts = np.concatenate(([0], diffs))
        group_ends = np.concatenate((diffs, [len(cfg_s)]))
        unique_cfgs = cfg_s[group_starts]

        for zq in ZQ_TARGETS:
            # Ceiling-aware skip: targets above the per-image achievable
            # max are physically unreachable (every cell will fail) —
            # skip them so the picker doesn't waste capacity on
            # impossible rows. See imazen/zenanalyze#51.
            if ceiling is not None and zq > ceiling + CEILING_MARGIN:
                skipped_above_ceiling += 1
                continue

            # Per-(image, size) budget gate. When apply_budget is on,
            # only candidates with time ≤ this value are considered.
            budget_ms = math.inf
            if apply_budget:
                base = time_baselines.get(size)  # type: ignore[union-attr]
                if base is not None:
                    budget_ms = base * time_budget_multiplier

            # Vectorized reach mask across the WHOLE key (sorted by
            # config), then within-budget mask AND'd in. Per-config
            # slices below see the right entries.
            #
            # Quality-tolerance reach band (p50 RD-hugging, "allow both-axis
            # error"): relax the target by REACH_UNDERSHOOT metric points so the
            # picker MAY undershoot quality by up to that much to land on a
            # cheaper RD-frontier point — the oracle + the pick are then both
            # measured over the band, so a near-frontier point at slightly-off
            # quality is ~0 bytes-overhead instead of a fixed-quality "miss".
            # REACH_UNDERSHOOT = 0 is the strict quality-safe path (p90 / the
            # historical default). The band caps the quality error at
            # REACH_UNDERSHOOT, so the "other axis" drift stays bounded.
            _zq_reach = (
                (zq - REACH_UNDERSHOOT) if higher_is_better else (zq + REACH_UNDERSHOOT)
            )
            if higher_is_better:
                reach_full = metric_sorted >= _zq_reach
            else:
                reach_full = metric_sorted <= _zq_reach
            any_unfiltered_reach = bool(reach_full.any())
            if apply_budget and time_sorted is not None:
                reach_after_budget = reach_full & (time_sorted <= budget_ms)
            else:
                reach_after_budget = reach_full

            cell_bytes = np.full(n_cells, math.inf, dtype=np.float64)
            cell_time = (
                np.full(n_cells, math.inf, dtype=np.float64)
                if has_time else None
            )
            cell_metric = (
                np.full(n_cells, math.nan, dtype=np.float64)
                if emit_metric_head else None
            )
            cell_scalars = {
                axis: np.full(n_cells, math.nan, dtype=np.float64)
                for axis in SCALAR_AXES
            }
            cell_reach = np.zeros(n_cells, dtype=bool)

            # Iterate per-config groups (typically 60-120 cfgs). Each
            # group's reach + bytes are array slices; argmin is
            # vectorized C-speed via np.where + .argmin().
            int_max = np.iinfo(np.int64).max
            for gi in range(len(unique_cfgs)):
                lo = int(group_starts[gi])
                hi = int(group_ends[gi])
                g_reach = reach_after_budget[lo:hi]
                if not g_reach.any():
                    continue
                g_bytes = bytes_sorted[lo:hi]
                masked_b = np.where(g_reach, g_bytes, int_max)
                local_argmin = int(masked_b.argmin())
                best_b = int(g_bytes[local_argmin])
                cfg_id = int(unique_cfgs[gi])
                c = (
                    int(cell_lookup[cfg_id])
                    if 0 <= cfg_id < len(cell_lookup) else -1
                )
                if c < 0:
                    continue
                if best_b < cell_bytes[c]:
                    cell_bytes[c] = best_b
                    if has_time and time_sorted is not None and cell_time is not None:
                        cell_time[c] = float(time_sorted[lo + local_argmin])
                    if emit_metric_head and cell_metric is not None:
                        cell_metric[c] = float(metric_sorted[lo + local_argmin])
                    for axis in SCALAR_AXES:
                        cell_scalars[axis][c] = scalar_lookup[axis][cfg_id]
                    cell_reach[c] = True

            if not cell_reach.any():
                # Mark as budget-infeasible only when the unfiltered
                # version would have reached. Otherwise this is a
                # physically-unreachable target, not a budget problem.
                if apply_budget and any_unfiltered_reach:
                    infeasible[(image, size)] = True
                continue

            # Safety mask: when --safety-default-cell is set, hide
            # alternative cells that would either slow encoding by
            # > safety_speed_tol or fail to deliver
            # safety_bytes_min_gain bytes savings vs the default.
            if (
                safety_default_cell_idx is not None
                and 0 <= safety_default_cell_idx < n_cells
                and cell_reach[safety_default_cell_idx]
                and has_time
                and cell_time is not None
            ):
                d_idx = safety_default_cell_idx
                d_bytes = float(cell_bytes[d_idx])
                d_time = float(cell_time[d_idx])
                bytes_ceiling = d_bytes * safety_bytes_min_gain
                time_ceiling = (
                    d_time * safety_speed_tol
                    if not math.isinf(d_time)
                    else math.inf
                )
                others = cell_reach.copy()
                others[d_idx] = False
                too_slow = cell_time > time_ceiling
                no_real_gain = cell_bytes >= bytes_ceiling
                kill = others & (too_slow | no_real_gain)
                if kill.any():
                    cell_bytes[kill] = math.inf
                    cell_reach[kill] = False
                    cell_time[kill] = math.inf
                    if emit_metric_head and cell_metric is not None:
                        cell_metric[kill] = math.nan
                    for axis in SCALAR_AXES:
                        cell_scalars[axis][kill] = math.nan
                # If masking eliminated every alternative AND default
                # is the only reachable cell, keep going — picker will
                # learn to pick default for this row, which is correct.

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

            # Vectorized log: unreachable cells → NaN.
            bytes_log = np.where(
                np.isinf(cell_bytes),
                math.nan,
                np.log(np.where(cell_bytes > 0, cell_bytes, 1.0)),
            ).astype(np.float32)
            reach = cell_reach.copy()

            Xs_rows.append(xs)
            Xe_rows.append(xe)
            bytes_log_rows.append(bytes_log)
            if time_log_rows is not None and cell_time is not None:
                # log(time_ms); NaN where the cell didn't reach or
                # time is non-positive / inf.
                valid_time = (cell_time > 0) & np.isfinite(cell_time)
                tlog = np.where(
                    valid_time,
                    np.log(np.where(valid_time, cell_time, 1.0)),
                    math.nan,
                ).astype(np.float32)
                time_log_rows.append(tlog)
            if metric_log_rows is not None and cell_metric is not None:
                # NaN preserved for unreached cells. zensim/ssim2 always
                # > 0; butteraugli also > 0 in practice — log everywhere
                # for numeric stability across metrics.
                valid_metric = (~np.isnan(cell_metric)) & (cell_metric > 0)
                mlog = np.where(
                    valid_metric,
                    np.log(np.where(valid_metric, cell_metric, 1.0)),
                    math.nan,
                ).astype(np.float32)
                metric_log_rows.append(mlog)
            for axis in SCALAR_AXES:
                scalar_rows[axis].append(cell_scalars[axis].astype(np.float32))
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


def evaluate_argmin(pred_bytes_log, actual_bytes_log, reach, meta, mask, veto=None):
    """Categorical argmin over allowed reachable cells.

    `veto` (optional `(n_rows, n_cells)` bool mask): cells the deployed picker
    is forbidden from choosing for that row (feature-gated knob-veto safety
    bounds, see `derive_knob_vetoes`). The oracle/actual-best is computed over
    the UN-vetoed reachable set, so overhead reflects the true achievable
    optimum vs the vetoed pick. `veto=None` → byte-identical to the un-vetoed
    behavior."""
    rows = evaluate_argmin_per_row(pred_bytes_log, actual_bytes_log, reach, meta, mask, veto=veto)
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


def evaluate_argmin_per_row(pred_bytes_log, actual_bytes_log, reach, meta, mask, veto=None):
    """Like `evaluate_argmin` but returns the per-row breakdown so the
    safety-report code can stratify by zq / size_class and surface
    worst-case images. Each entry: image, size_class, zq, pick,
    actual_best, overhead, predicted_bytes, actual_bytes.

    `veto` (optional `(n_rows, n_cells)` bool): cells the deployed picker may
    NOT choose for that row. The oracle (`actual_best`) is the argmin over the
    UN-vetoed reachable set `reach[i] & mask`; the pick is the argmin over
    `reach[i] & mask & ~veto[i]`, falling back to the un-vetoed reachable set if
    every reachable cell is vetoed (never strand a row). Overhead = vetoed-pick
    bytes vs true-oracle bytes. `veto=None` → byte-identical to the prior
    behavior."""
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
        a = int(np.argmin(ab))  # true oracle over reachable&allowed (veto NOT applied)
        if veto is not None:
            m_pick = m & ~veto[i]
            # fall back to the un-vetoed reachable set if the veto strands the row
            pb_pick = np.where(m_pick, pb, np.inf) if np.any(m_pick) else pb
        else:
            pb_pick = pb
        if VERIFY_K <= 1:
            p = int(np.argmin(pb_pick))
        else:
            # K-verify (VERIFY_K > 1): the deployed codec encodes the top-VERIFY_K
            # cheapest-PREDICTED reachable cells and keeps the best by ACTUAL
            # bytes. `p == a` iff the oracle is among the verified top-K, so the
            # overhead + argmin_acc reflect the codec's real K-encode behavior
            # (not the single-pick K=1). Predicted ties broken stably.
            order = np.argsort(pb_pick, kind="stable")
            topk = [int(c) for c in order if np.isfinite(pb_pick[c])][:VERIFY_K]
            p = min(topk, key=lambda c: ab[c]) if topk else int(np.argmin(pb_pick))
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


def _veto_feature_matrix(meta_rows, feats, feat_cols):
    """Per-row feature matrix (n_rows, n_feat), looked up by (image, size).
    NaN/missing → 0.0 (matches the runtime's NaN handling; a row that has no
    features simply can't fire a feature-gated rule)."""
    n_rows = len(meta_rows)
    F = np.zeros((n_rows, len(feat_cols)), dtype=np.float64)
    for i, mr in enumerate(meta_rows):
        fv = feats.get((mr[0], mr[1]))
        if fv is not None:
            F[i] = np.nan_to_num(np.asarray(fv, dtype=np.float64), nan=0.0)
    return F


def _axis_value_arrays(cells, categorical_axes):
    """{axis -> (n_cells,) object array of each cell's value for that axis}."""
    return {
        ax: np.array([c.get(ax) for c in cells], dtype=object)
        for ax in categorical_axes
    }


def derive_knob_vetoes(
    cells,
    categorical_axes,
    pred_tr,
    bl_tr,
    rch_tr,
    meta_tr,
    feats,
    feat_cols,
    all_mask,
    # Candidate-generation threshold: a (axis,value) is a veto candidate when
    # its best-cell overhead exceeds this. 0.40 aligns with the tightened
    # per-zq/size p99<40 gate (was 0.50) so the deriver can see + bound the
    # 40-100% tail that the gate rejects.
    cat_overhead=0.40,
    mean_budget_pp=0.40,
):
    """Greedily derive feature-gated per-(categorical-axis-value) safety vetoes
    that bound the single-encode picker's worst-case RD overhead.

    The K=1 (pure-argmin) picker catastrophically mis-sets a categorical toggle
    (chroma/bd/qm) on a tiny fraction of images, blowing the WORST_ROW overhead
    far past the 200% gate. A veto is a rule "forbid value V on axis A when
    feature F </> thr"; firing it removes the offending cells from the picker's
    reachable set for that row WITHOUT touching the oracle, so worst-case is
    bounded while the achievable optimum is unchanged.

    Port of /tmp/systematic_veto.py (validated to reproduce the avif tail
    numbers). For each (axis, value V): fit a depth-1 stump predicting "V is
    catastrophic for this row" (best-V-cell overhead vs oracle > `cat_overhead`),
    yielding a candidate feature threshold + direction. Then greedily
    forward-select the candidates that most reduce a weighted tail score
    (`1000*>200 + 30*>150 + >100 + 0.5*max`) at a cumulative mean-overhead cost
    <= `mean_budget_pp`, evaluated on the TRAIN picker (`pred_tr`). It does NOT
    stop at train-pass: the catastrophic worst row MOVES across splits, so it
    minimizes the whole train tail to cover every catastrophe mode
    (mode-completeness) and leave val/test margin; it stops only when no
    remaining candidate reduces the score within budget. `mean_budget_pp` 0.40
    admits the 2nd veto (the binding constraint — the sub/chroma veto costs
    ~0.25pp train mean and is rejected at 0.20, leaving val@334%).

    Returns `list[{"axis", "value", "feat", "op", "threshold"}]` (op is "<"/">").
    """
    n_rows = pred_tr.shape[0]
    n_cells = len(cells)
    if n_rows == 0 or n_cells == 0 or not categorical_axes:
        return []

    axis_vals = _axis_value_arrays(cells, categorical_axes)
    # Augment the veto candidate features with the quality target (zq): the
    # surviving catastrophic mis-sets are quality-dependent (high-zq) and image-
    # feature-only stumps cannot separate them. A "__zq__" veto gates on the
    # target zq (which the deploy knows), e.g. "veto config X at zq>80".
    veto_feat_cols = list(feat_cols) + ["__zq__"]
    fidx = {c: i for i, c in enumerate(veto_feat_cols)}
    F = _veto_feature_matrix(meta_tr, feats, feat_cols)
    F = np.hstack([F, np.array([[float(mr[2])] for mr in meta_tr], dtype=np.float64)])

    base = rch_tr & all_mask[None, :]
    AB = np.where(base, np.exp(bl_tr), np.inf)
    PB = np.where(base, np.exp(np.clip(pred_tr, -30, 30)), np.inf)
    oracle = AB.min(axis=1)  # true per-row oracle (never vetoed)

    def value_overhead(cols):
        # best reachable cell carrying this axis value vs the true oracle
        if cols.size == 0:
            return np.full(n_rows, np.inf)
        return (AB[:, cols].min(axis=1) - oracle) / oracle

    # --- candidate vetoes: one depth-1 stump per (axis, value) ---
    cands = []
    for ax in categorical_axes:
        av = axis_vals[ax]
        values = {v for v in av.tolist() if v is not None}
        for V in sorted(values, key=lambda x: (str(type(x)), str(x))):
            cols = np.where(av == V)[0]
            vo = value_overhead(cols)
            ok = np.isfinite(vo)
            y = (vo > cat_overhead).astype(int)
            if int(y[ok].sum()) < 20:
                continue
            clf = DecisionTreeClassifier(
                max_depth=1, class_weight="balanced", min_samples_leaf=300
            ).fit(F[ok], y[ok])
            fi = int(clf.tree_.feature[0])
            if fi < 0:  # no split found
                continue
            thr = float(clf.tree_.threshold[0])
            # which side of the split is the catastrophic (class-1) side?
            lc = clf.tree_.value[1][0][1] / clf.tree_.value[1][0].sum()
            rc = clf.tree_.value[2][0][1] / clf.tree_.value[2][0].sum()
            op = "<" if lc > rc else ">"
            cands.append(
                {"axis": ax, "value": V, "feat": veto_feat_cols[fi], "op": op, "threshold": thr}
            )

    def fire(rule):
        fv = F[:, fidx[rule["feat"]]]
        return (fv < rule["threshold"]) if rule["op"] == "<" else (fv > rule["threshold"])

    def allowed_mask(rules):
        a = base.copy()
        for r in rules:
            a = a & ~(fire(r)[:, None] & (axis_vals[r["axis"]] == r["value"])[None, :])
        stranded = ~a.any(axis=1)
        if stranded.any():
            a[stranded] = base[stranded]  # never strand a row
        return a

    def metrics(rules):
        a = allowed_mask(rules)
        pa = np.where(a, PB, np.inf)
        if VERIFY_K <= 1:
            pk = pa.argmin(axis=1)
            achieved = AB[np.arange(n_rows), pk]
        else:
            # K-verify: best ACTUAL bytes among the K cheapest-PREDICTED allowed
            # cells — the SAME best-of-top-K the deployed gate scores, so the
            # deriver bounds the K-verify tail (not the K=1 tail it would
            # otherwise chase past the verify pass).
            k = min(VERIFY_K, pa.shape[1])
            topk = np.argpartition(pa, k - 1, axis=1)[:, :k]
            ab_top = np.take_along_axis(AB, topk, axis=1)
            pa_top = np.take_along_axis(pa, topk, axis=1)
            achieved = np.where(np.isfinite(pa_top), ab_top, np.inf).min(axis=1)
        ov = (achieved - oracle) / oracle * 100.0
        return {
            "mean": float(ov.mean()),
            "mx": float(ov.max()),
            "n200": int((ov > 200).sum()),
            "n150": int((ov > 150).sum()),
            "n100": int((ov > 100).sum()),
            # n40 = rows over the tightened per-zq / per-size p99 gate (40%).
            # The deriver must target THIS band — the gate binds on p99<40 and
            # worst<100, not the old >200 catastrophe band — or it derives 0
            # vetoes for a picker whose mean is fine but whose 40-100% tail
            # fails the gate (the webp case, 2026-06-29).
            "n40": int((ov > 40).sum()),
        }

    # Weighted tail score, ALIGNED TO THE TIGHTENED GATE (max_single_row<100 +
    # per-zq/size p99<40). Bound the worst-row gate first (n200 catastrophes,
    # then n100 = the hard worst<100 gate), then the p99-driving 40-100% band
    # (n40), then the absolute max. Greedily add the veto that most reduces it.
    # Do NOT stop at train-pass: the catastrophic worst row MOVES across splits
    # (a qm-only veto cleared train@196% but left val@333% on a sub/chroma
    # mis-set), so minimize the WHOLE train tail to cover every catastrophe
    # mode and leave val/test margin. Terminates when no candidate helps within
    # mean_budget_pp. (Pre-2026-06-29 this targeted only >100/150/200 and
    # derived 0 vetoes for tail-only-failing pickers like webp, whose 40-100%
    # band is exactly what the p99<40 gate rejects — n40 fixes that.)
    def tail_score(m):
        return (
            1000.0 * m["n200"]
            + 200.0 * m["n100"]
            + 5.0 * m["n40"]
            + 0.5 * m["mx"]
        )

    base_m = metrics([])
    chosen = []
    cur = base_m
    while True:
        best = None
        for c in cands:
            if c in chosen:
                continue
            m = metrics(chosen + [c])
            if m["mean"] - base_m["mean"] > mean_budget_pp:
                continue
            gain = tail_score(cur) - tail_score(m)
            if gain > 1.0 and (best is None or gain > best[0]):
                best = (gain, c, m)
        if best is None:
            break
        chosen.append(best[1])
        cur = best[2]
        r = best[1]
        sys.stderr.write(
            f"  + knob-veto {r['axis']}={r['value']} when {r['feat']} {r['op']} "
            f"{r['threshold']:.4g}  -> train max={cur['mx']:.0f}% >200={cur['n200']} "
            f">150={cur['n150']} >100={cur['n100']} mean={cur['mean']:.2f}%\n"
        )
    sys.stderr.write(
        f"  derive_knob_vetoes: {len(chosen)} veto(s) selected from {len(cands)} "
        f"candidate(s); train tail base max={base_m['mx']:.0f}% (>200={base_m['n200']}) "
        f"-> vetoed max={cur['mx']:.0f}% (>200={cur['n200']}) "
        f"mean {base_m['mean']:.2f}%->{cur['mean']:.2f}%\n"
    )
    return chosen


def build_veto_mask(vetoes, meta_rows, feats, feat_cols, cells, categorical_axes):
    """Build the `(n_rows, n_cells)` bool veto mask for a split from the rules
    returned by `derive_knob_vetoes`. For each row `meta=(img, size, zq)`, look
    up `feats[(img, size)]`; for every rule whose feature condition fires, veto
    all cells carrying that rule's (axis == value). Rows with no features key
    get no veto (graceful)."""
    n_rows = len(meta_rows)
    n_cells = len(cells)
    veto = np.zeros((n_rows, n_cells), dtype=bool)
    if not vetoes:
        return veto
    fidx = {c: i for i, c in enumerate(feat_cols)}
    axis_vals = _axis_value_arrays(cells, categorical_axes)
    for i, mr in enumerate(meta_rows):
        fv = feats.get((mr[0], mr[1]))
        if fv is not None:
            fv = np.nan_to_num(np.asarray(fv, dtype=np.float64), nan=0.0)
        for r in vetoes:
            if r["feat"] == "__zq__":
                # quality-target veto: gate on the target zq (meta[2]), which
                # the deploy knows. Covers the quality-dependent catastrophes
                # that image-feature-only vetoes cannot separate.
                val = float(mr[2])
            else:
                if fv is None:
                    continue  # no features for this row -> no feature veto
                col = fidx.get(r["feat"])
                if col is None:
                    continue
                val = fv[col]
            fires = (val < r["threshold"]) if r["op"] == "<" else (val > r["threshold"])
            if fires:
                veto[i] |= axis_vals[r["axis"]] == r["value"]
    return veto


def evaluate_topk_verify(pred_bytes_log, actual_bytes_log, reach, mask, ks=(1, 2, 3, 4, 5, 6, 7)):
    """Predict-top-K-then-verify picker design.

    A pure content picker (K=1 = raw argmin) leaves a residual oracle gap
    (here ~2.4%). This measures the alternative *narrow-by-content,
    finalize-by-RD-check* design: rank the cells by the picker's PREDICTED
    bytes, take the K predicted-cheapest reachable cells, then — simulating a
    real encode of just those K — pick the one with min ACTUAL bytes among
    them. Overhead is vs the true per-row oracle (min actual over all
    reachable cells). K=1 reproduces evaluate_argmin's overhead; K=n_reachable
    -> 0% (the oracle is always inside the verified set).

    Per K it reports mean/p50/p90/p99/max overhead %, the hit-rate (fraction
    of rows whose oracle cell falls within the predicted top-K), and the mean
    number of cells actually verified (<=K, capped by the reachable count).
    This is exactly the data needed to answer "can a content picker + a small
    encode-verify step reach <=1% achieved RD, and at what K (encode budget)?"
    """
    n_rows = pred_bytes_log.shape[0]
    acc = {k: {"ovh": [], "hit": 0, "verified": []} for k in ks}
    n_used = 0
    for i in range(n_rows):
        m = reach[i] & mask
        if not np.any(m):
            continue
        n_used += 1
        ab = np.where(m, np.exp(actual_bytes_log[i]), np.inf)
        pb = np.where(m, np.exp(np.clip(pred_bytes_log[i], -30, 30)), np.inf)
        oracle = int(np.argmin(ab))
        oracle_bytes = float(ab[oracle])
        order = np.argsort(pb, kind="stable")  # predicted-cheapest first; unreachable (inf) sort last
        n_reach = int(np.count_nonzero(m))
        for k in ks:
            kk = min(k, n_reach)
            topk = order[:kk]
            best_actual = float(ab[topk].min())
            acc[k]["ovh"].append((best_actual - oracle_bytes) / oracle_bytes)
            acc[k]["hit"] += int(oracle in set(int(x) for x in topk))
            acc[k]["verified"].append(kk)
    out = {}
    for k in ks:
        ovh = np.asarray(acc[k]["ovh"]) if acc[k]["ovh"] else np.zeros(1)
        out[k] = {
            "mean_pct": float(100 * ovh.mean()),
            "p50_pct": float(100 * np.percentile(ovh, 50)),
            "p90_pct": float(100 * np.percentile(ovh, 90)),
            "p99_pct": float(100 * np.percentile(ovh, 99)),
            "max_pct": float(100 * ovh.max()),
            "hit_rate": float(acc[k]["hit"] / max(n_used, 1)),
            "mean_verified": float(np.mean(acc[k]["verified"])) if acc[k]["verified"] else 0.0,
            "n_rows": int(n_used),
        }
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
    # LOOSE sanity floor on held-out argmin accuracy — catches a degenerate /
    # near-random picker only. It is NOT the quality gate: exact-match argmin
    # penalizes RD-equivalent NEAR-TIES (many configs land within ~1 byte-% of
    # the per-image optimum, so picking a near-tie counts as a "miss" at ~0 RD
    # cost). The real quality gate is `max_mean_overhead_pct` (RD overhead — the
    # quantity actually deployed) plus the per-zq / per-size p99 tails. Demoted
    # from 0.30 → 0.10 on 2026-06-28 after measuring that webp's K=1 RD overhead
    # (3.37%) is LOWER than jpeg's (6.41%, which PASSED at 36% argmin): argmin_acc
    # anti-correlated with RD quality, so a 30% argmin gate rejected the MORE
    # RD-efficient picker. A pure content picker deploys via top-K-verify (the
    # oracle is in the top-K; mean overhead collapses to <1% by K=5), so raw-K=1
    # exact-match was never the deployed quantity. See CLEAN_PICKER_PROGRAM.md.
    min_argmin_acc=0.10,
    # Held-out mean overhead ceiling.
    max_mean_overhead_pct=5.0,
    # No single zq band may have a p99 overhead this bad.
    max_per_zq_p99_overhead_pct=40.0,
    # No single image-size class may have a p99 overhead this bad.
    # Size invariance is a *safety property*: the picker must be
    # near-optimal at every (width, height), not just on average.
    # Tiny images (≤64×64) historically tail worst — tight per-bin
    # gates catch a tiny-class blowup that the global mean would
    # absorb. See SAFETY_PLANE.md → "Size invariance is a safety
    # property" and the size_invariance_probe.py post-bake gate.
    max_per_size_p99_overhead_pct=40.0,
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
    max_single_row_overhead_pct=100.0,
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


# Canonical pixel-AREA (width×height) thresholds defining the size classes.
# MUST match scripts/picker/omni_to_pareto.py::size_class (the labeller of
# the Pareto parquet's `size_class` column) and the runtime size discriminant
# in zenpredict::UnachievableZones. tiny ≤ 64², small ≤ 256², medium ≤ 1024².
SIZE_CLASS_PIXEL_UPPER = [
    ("tiny", 4096.0),
    ("small", 65536.0),
    ("medium", 1048576.0),
    ("large", float("inf")),
]


def compute_unachievable_zones(
    meta_tr, rch_tr, train_rows_by_size_zq, size_classes, zq_targets,
    min_rows, scalar_axes, scalar_display_ranges,
):
    """Per size class, the declared unachievable-zone + fallback knobset.

    `meta_tr` holds one row per *reachable* (image, size, zq), so the per-size
    density ceiling — the highest zq with ≥ `min_rows` reachable train images —
    is the boundary above which the target is physically unreachable for that
    size. This is the SAME ceiling the zone-aware DATA_STARVED_SIZE gate
    exempts, so the declared zone ≡ the exempted tail ≡ the runtime fallback
    region (one ceiling, three uses — the skip is described, not silent).

    Fallback knobset = the cell most reachable at the ceiling zq (the
    quality-leader for that size) at max scalar (best achievable quality). The
    runtime (`zenpredict::UnachievableZones::resolve`) maps an image's
    `feat_pixel_count` → size class and, when `target_zq > ceiling_zq`, encodes
    this knobset instead of an unreachable argmin pick.

    Returns a list of zone dicts {size_class, pixel_upper, ceiling_zq,
    fallback_cell, fallback_scalar}, ascending by pixel_upper. Only sizes with
    a real unreachable tail (`ceiling < max(zq_targets)`) get a zone; sizes
    reachable across the whole grid, or under-supplied everywhere (the gate
    flags those), are omitted.
    """
    import numpy as _np

    upper = {name: ub for name, ub in SIZE_CLASS_PIXEL_UPPER}
    canon_order = [name for name, _ in SIZE_CLASS_PIXEL_UPPER]
    present = [s for s in canon_order if s in size_classes]
    if not present:
        return []
    # The largest present size class catches every larger image at runtime
    # (the JSON's "clamping larger images to the last" contract).
    pixel_upper = {
        s: (float("inf") if i == len(present) - 1 else upper[s])
        for i, s in enumerate(present)
    }
    # One scalar axis → bake its display max (highest effort/quality = best
    # achievable). Zero or many axes → NaN sentinel ("use predicted scalar").
    fallback_scalar = float("nan")
    if len(scalar_axes) == 1:
        rng = scalar_display_ranges.get(scalar_axes[0])
        if rng:
            fallback_scalar = float(rng[1])

    top_zq = max(int(z) for z in zq_targets)
    zones = []
    for s in present:
        by_zq = train_rows_by_size_zq.get(s, {})
        ok = [int(z) for z, n in by_zq.items() if n >= min_rows]
        if not ok:
            continue  # under-supplied at every zq → gate flags it, no zone
        ceiling = max(ok)
        if ceiling >= top_zq:
            continue  # whole grid reachable → no unachievable tail
        # Fallback cell = the cell reachable in the most train rows at the
        # ceiling zq for this size (the quality leader at the ceiling).
        idx = [
            i for i, m in enumerate(meta_tr)
            if m[1] == s and int(m[2]) == ceiling
        ]
        if idx:
            reach_counts = _np.asarray(rch_tr)[idx, :].sum(axis=0)
            fallback_cell = int(_np.argmax(reach_counts))
        else:
            fallback_cell = 0
        zones.append({
            "size_class": s,
            "pixel_upper": pixel_upper[s],
            "ceiling_zq": float(ceiling),
            "fallback_cell": fallback_cell,
            "fallback_scalar": fallback_scalar,
        })
    return zones


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
            f"< sanity floor {thresholds['min_argmin_acc']:.1%} "
            f"(degenerate/near-random picker — NOT the quality gate; RD quality "
            f"is gated by max_mean_overhead_pct + per-zq/size p99, which charge "
            f"~nothing for RD-equivalent near-ties)"
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

    # Data-starvation gate per (size_class, target_zq) training cell —
    # ZONE-AWARE. `meta_tr` holds one row per *reachable* (image, size, zq),
    # so `train_rows_by_size_zq[S][zq]` is the count of size-S images that can
    # physically reach target zq. High-zq starvation is therefore the
    # *reachability tail* (fewer images of a size reach higher quality — e.g.
    # medium photos top out near SSIMULACRA2 ~94, tiny ceilings are bimodal),
    # NOT a sweep harness gap. A genuine gap instead shows as a size starved at
    # LOW zq too (the harness skipped that size for a chunk of the corpus).
    #
    # So per size: ceiling = the highest zq with >= min_rows reachable images.
    #   - zq > ceiling  → physically-unreachable tail → EXEMPT (info, not a
    #     violation). The bake declares these as unachievable zones carrying a
    #     fallback knobset (zenpicker.unachievable_zones), so the skip is a
    #     described, deploy-honored boundary — not a silent gap.
    #   - zq <= ceiling but starved → a hole below the ceiling (non-monotonic
    #     gap) → FLAG. A size with NO zq >= min_rows (ceiling = None) is
    #     under-supplied everywhere → FLAG every cell.
    # Codec harness MUST still emit rows for tiny/small/medium/large per image
    # (FOR_NEW_CODECS.md Step 1.5) across the *reachable* band.
    min_rows = thresholds["min_train_rows_per_size_zq"]
    rows_by_sz = diag.get("train_rows_by_size_zq", {})
    starved = []        # genuine under-supply (violations)
    exempt_tail = []    # physically-unreachable high-zq tail (info)
    for sz, by_zq in rows_by_sz.items():
        ok_zqs = [int(zq) for zq, n in by_zq.items() if n >= min_rows]
        ceiling = max(ok_zqs) if ok_zqs else None
        for zq, n in by_zq.items():
            zq = int(zq)
            if n >= min_rows:
                continue
            if ceiling is not None and zq > ceiling:
                exempt_tail.append((sz, zq, n))
            else:
                starved.append((sz, zq, n))
    # Record the exemptions in the report (provenance) + log them — the
    # boundary is visible, never a silent runtime skip.
    diag["data_starved_exempt_tail"] = sorted(
        ({"size_class": sz, "zq": zq, "n_reachable": n} for (sz, zq, n) in exempt_tail),
        key=lambda d: (d["size_class"], d["zq"]),
    )
    if exempt_tail:
        exempt_tail.sort(key=lambda t: (t[0], t[1]))
        ex = ", ".join(f"{sz}/zq{zq}={n}" for (sz, zq, n) in exempt_tail[:8])
        more = f" (+{len(exempt_tail) - 8} more)" if len(exempt_tail) > 8 else ""
        sys.stderr.write(
            f"  ℹ DATA_STARVED_SIZE exemption: {len(exempt_tail)} (size,zq) cell(s) "
            f"above the per-size achievable ceiling — physically-unreachable "
            f"high-zq tail, declared as unachievable zones w/ fallback knobset "
            f"(not a sweep gap): {ex}{more}\n"
        )
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
            f"under-supplied at/below the per-size achievable ceiling "
            f"(genuine gap, not the unreachable tail) — train rows < {min_rows}: "
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
        n_cfg = c["n_member_configs"]
        if n_cfg < thresholds["min_cell_member_configs"]:
            # A cell with exactly ONE member config is a deliberate fixed
            # reference anchor (e.g. jxl `vd_lean` = the lone `vd-e7_lean_def`
            # preset, `vd_libjxl` = `vd-e7_libjxl_def`), NOT a thin scalar
            # ladder: there is no scalar curve to learn, the scalar head
            # trivially emits the one config's value. The
            # >= min_cell_member_configs rule targets cells that SHOULD carry a
            # swept ladder but came up thin — it does not apply to a single
            # deterministic point (which the oracle still picks, e.g. lean wins
            # ~5% of jxl-lossy decisions). Exempt + report; a 2-config cell is
            # still flagged (genuinely thin).
            if n_cfg == 1:
                sys.stderr.write(
                    f"  ℹ DATA_STARVED_CELL exemption: cell {c['cell']} "
                    f"({c['label']}) is a single-config reference anchor "
                    f"(1 member config — fixed knobset, no scalar ladder to "
                    f"learn); not flagged.\n"
                )
                continue
            v.append(
                f"DATA_STARVED_CELL: cell {c['cell']} ({c['label']}) has "
                f"{n_cfg} member configs "
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
        "--reach-undershoot",
        type=float,
        default=None,
        help="Quality-tolerance reach band for the p50 'hug the RD curve, "
        "allow both-axis error' objective: a cell reaches target zq if its "
        "achieved quality is within this many metric points BELOW target, so "
        "the picker may trade up to this much quality for a cheaper RD-frontier "
        "point. 0 = strict quality-safe (p90, the default). Overrides a codec "
        "config's REACH_UNDERSHOOT when given.",
    )
    parser.add_argument(
        "--verify-k",
        type=int,
        default=None,
        help="K-verify: evaluate the picker as best-of-top-K (the codec encodes "
        "the K cheapest-predicted reachable cells, keeps the best by actual "
        "bytes). 1 = single-encode K=1 (default). Set 2-3 for codecs that can "
        "afford a few encodes (e.g. zenjpeg). The safety gate then reflects the "
        "deployed K. Overrides a codec config's VERIFY_K when given.",
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
        "--pareto",
        default=None,
        help="Override codec config's PARETO path (the sweep TSV/parquet to "
        "train on). Lets a caller point the trainer at a freshly-merged pareto "
        "without editing the codec config (e.g. a re-sweep's omni_to_pareto "
        "output). Default: codec config value.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Override codec config's FEATURES path (the variant_name-keyed "
        "feature TSV joined to the pareto). Pair with --pareto so both inputs "
        "come from the same fresh run. Default: codec config value.",
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
        default="leakyrelu",
        help="Hidden-layer activation. leakyrelu (default) routes to "
        "a PyTorch student with negative_slope=0.01 (same MLP shape, "
        "same Adam/lr/batch/early-stopping schedule as the legacy "
        "sklearn path) and finishes in seconds-to-minutes. relu falls "
        "through to sklearn `MLPRegressor.fit`, which is "
        "single-threaded for our matmul size and TYPICALLY 10–20× "
        "SLOWER. Keep `relu` only when you need bit-identical "
        "reproduction of a pre-leakyrelu sklearn-trained baseline. "
        "Both produce a `student.coefs_/intercepts_` surface so "
        "safety_check, diagnostics, and JSON serialization work the "
        "same way.",
    )
    parser.add_argument(
        "--hard-example-weighting",
        choices=["none", "emae"],
        default="none",
        help="Per-row reweighting of the distill MSE loss based on a "
        "moving average of per-row student-vs-teacher squared "
        "disagreement. `none` (default) leaves the loss unweighted. "
        "`emae` (Exponential Moving Average + median normalisation) "
        "maintains disagree_ema[i] across the last "
        "--hard-example-ema-window epochs, then per-row weight = "
        "clip(1 + α · ema[i] / median(ema), 1/clip, clip). The first "
        "N epochs use uniform weights (no signal yet). LeakyReLU "
        "backend only — the sklearn ReLU path ignores this. See "
        "`benchmarks/hard_example_weighting_2026-05-17.md` for the "
        "measurement that justifies the default.",
    )
    parser.add_argument(
        "--hard-example-alpha",
        type=float,
        default=1.0,
        help="Strength multiplier for the hard-example weight. "
        "weight = clip(1 + α · ema / median(ema), 1/clip, clip). "
        "α=0 → uniform; α=1 (default) → at the median disagreement "
        "the weight is 2.0. Larger α biases harder toward hard rows.",
    )
    parser.add_argument(
        "--hard-example-ema-window",
        type=int,
        default=5,
        help="EMA window (in epochs) for the per-row disagreement "
        "tracker. `α_ema = 1 / window`. Default 5. First `window` "
        "epochs run uniform weights so the EMA can warm up.",
    )
    parser.add_argument(
        "--hard-example-clip",
        type=float,
        default=10.0,
        help="Per-row weight clip range = [1/clip, clip]. Default "
        "10.0 → weights ∈ [0.1, 10.0]. Prevents one outlier row from "
        "dominating the gradient.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override OUT_JSON path entirely (codec-config OUT_JSON "
        "ignored). Use when running the trainer from a directory "
        "where the codec-config relative output paths would write "
        "into the wrong repo (e.g. running the zenjpeg config from "
        "the zenjpeg checkout but redirecting bench outputs to "
        "zenanalyze).",
    )
    parser.add_argument(
        "--out-log",
        type=Path,
        default=None,
        help="Override OUT_LOG path entirely (codec-config OUT_LOG "
        "ignored). See --out-json.",
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
        "--drop-features",
        type=str,
        default=None,
        help="Comma-separated feature names to REMOVE from the codec "
        "config's KEEP_FEATURES before training. Drives leave-one-out "
        "(LOO) feature ablation: retrain with one feature dropped, "
        "compare val mean overhead to the full-set baseline. Unknown "
        "names are ignored with a stderr warning.",
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
    parser.add_argument(
        "--safety-default-cell",
        default=None,
        help="Cell label (matches `cell_label_from_key` output, e.g. "
        "'effort7' for an effort-only taxonomy) to anchor a per-row "
        "safety mask. When set, the per-(image, zq) teacher hides "
        "alternative cells whose min-bytes config either takes more "
        "than --safety-speed-tol times the default's encode time, OR "
        "fails to deliver a >= (1 - --safety-bytes-min-gain) bytes "
        "savings vs the default. Forces the picker to default unless "
        "an alternative is meaningfully smaller AND not slower. "
        "Default off (preserves prior teacher behaviour).",
    )
    parser.add_argument(
        "--safety-speed-tol",
        type=float,
        default=1.05,
        help="Speed tolerance multiplier for the safety mask "
        "(default 1.05 = alternative may be at most 5%% slower).",
    )
    parser.add_argument(
        "--safety-bytes-min-gain",
        type=float,
        default=0.99,
        help="Minimum bytes-shrink ratio for an alternative to count "
        "as 'meaningfully smaller' under the safety mask "
        "(default 0.99 = alternative bytes must be < 99%% of default).",
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
    load_codec_config(
        args.codec_config,
        drop_features=(args.drop_features.split(",") if args.drop_features else None),
    )

    # CLI overrides — take precedence over codec config defaults.
    global METRIC_COLUMN, METRIC_DIRECTION, PARETO, FEATURES, REACH_UNDERSHOOT, VERIFY_K
    if args.reach_undershoot is not None:
        REACH_UNDERSHOOT = float(args.reach_undershoot)
        sys.stderr.write(f"  CLI override: REACH_UNDERSHOOT={REACH_UNDERSHOOT}\n")
    if args.verify_k is not None:
        VERIFY_K = int(args.verify_k)
        sys.stderr.write(f"  CLI override: VERIFY_K={VERIFY_K}\n")
    if VERIFY_K > 1:
        sys.stderr.write(
            f"  K-verify active: gate evaluates best-of-top-{VERIFY_K} (codec "
            f"encodes {VERIFY_K} cheapest-predicted cells, keeps the best).\n"
        )
    if REACH_UNDERSHOOT > 0:
        sys.stderr.write(
            f"  p50 RD-hugging band active: reach allows up to "
            f"{REACH_UNDERSHOOT:g} metric-point quality undershoot to hug the "
            f"RD curve (both-axis error bounded by the band).\n"
        )
    if args.pareto is not None:
        PARETO = Path(args.pareto)
        sys.stderr.write(f"  CLI override: PARETO={PARETO}\n")
    if args.features is not None:
        FEATURES = Path(args.features)
        sys.stderr.write(f"  CLI override: FEATURES={FEATURES}\n")
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
    # --out-json / --out-log override entirely (applied AFTER suffix so
    # a caller passing both gets the explicit path verbatim, not a
    # suffix-mutated version).
    if args.out_json is not None:
        OUT_JSON = Path(args.out_json)
        sys.stderr.write(f"  CLI override: OUT_JSON={OUT_JSON}\n")
    if args.out_log is not None:
        OUT_LOG = Path(args.out_log)
        sys.stderr.write(f"  CLI override: OUT_LOG={OUT_LOG}\n")
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
    # Scope the size grid to the size classes present in the corpus (or an
    # explicit PICKER_SIZE_CLASSES=tiny,small,medium override). Absent sizes
    # (e.g. `large` in a <=1 MP web corpus) are excluded so the DATA_STARVED_SIZE
    # gate enforces coverage only for sizes the sweep actually produced — a size
    # the corpus never contained is not a "silent sweep skip" to flag.
    _env_sc = os.environ.get("PICKER_SIZE_CLASSES", "").strip()
    if _env_sc:
        _present_order = [s.strip() for s in _env_sc.split(",") if s.strip()]
    else:
        _present_order = sorted(
            {sz for (_i, sz, _w, _h) in pareto.keys()},
            key=lambda s: _SIZE_CANON.index(s) if s in _SIZE_CANON else 99,
        )
    _before = list(SIZE_CLASSES)
    _scope_size_classes(_present_order)
    if SIZE_CLASSES != _before:
        sys.stderr.write(
            f"SIZE_CLASSES scoped to {SIZE_CLASSES} (was {_before}; size classes "
            f"absent from the corpus excluded — picker models only swept sizes)\n"
        )
    feats, feat_cols, feat_transforms = load_features(FEATURES)
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

    safety_default_cell_idx = None
    if args.safety_default_cell:
        cell_labels = [c["label"] for c in cells]
        try:
            safety_default_cell_idx = cell_labels.index(args.safety_default_cell)
        except ValueError:
            sys.stderr.write(
                f"--safety-default-cell {args.safety_default_cell!r} not in cell taxonomy "
                f"{cell_labels}\n"
            )
            sys.exit(2)
        sys.stderr.write(
            f"  safety mask anchor: cell '{args.safety_default_cell}' "
            f"(idx {safety_default_cell_idx}); "
            f"speed_tol={args.safety_speed_tol}, "
            f"bytes_min_gain={args.safety_bytes_min_gain}\n"
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
        safety_default_cell_idx=safety_default_cell_idx,
        safety_speed_tol=args.safety_speed_tol,
        safety_bytes_min_gain=args.safety_bytes_min_gain,
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

    # CANONICAL split — by ORIGIN image, last-digit parity. origin_split.py
    # (zenmetrics/scripts/picker) is the ONE source of truth: {0,2,4,6,8}=train,
    # {1,3,5}=val, {7,9}=test; every sizing/crop/encode derivative inherits the
    # origin's bucket so nothing leaks. Replaces the old seeded per-rendition 20%
    # shuffle (per-rendition → scale leakage; random → irreproducible). Train only
    # ever sees even-origin content. See docs/CLEAN_PICKER_PROGRAM.md.
    try:
        from origin_split import origin_stem as _origin_stem, split_of as _origin_split_of
    except ImportError as _e:
        raise SystemExit(
            "train_hybrid needs the canonical origin_split on PYTHONPATH "
            "(zenmetrics/scripts/picker/origin_split.py) — refusing to fall back to a "
            f"leaky random split. Add scripts/picker to PYTHONPATH. ({_e})"
        )
    _spl = [_origin_split_of(m[0]) for m in meta]
    n_unsplit = sum(1 for s in _spl if s is None)
    tr = np.array([i for i, s in enumerate(_spl) if s == "train"])
    va = np.array([i for i, s in enumerate(_spl) if s == "val"])
    te = np.array([i for i, s in enumerate(_spl) if s == "test"])
    n_origins = len({_origin_stem(m[0]) for m in meta})
    n_rend = len({m[0] for m in meta})
    sys.stderr.write(
        f"Origin even/odd split (train_hybrid via origin_split.py): "
        f"train {len(tr)} / val {len(va)} / test {len(te)} rows "
        f"({n_origins} origins, {n_rend} renditions; {n_unsplit} unsplittable rows dropped)\n"
    )
    if len(va) == 0:
        raise SystemExit(
            "0 validation rows (no origins ending in 1/3/5). Train on the full "
            "imazen-26 corpus, not a train-biased even-only set."
        )

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
    # Held-out TEST slices (origins ending 7/9) — evaluated after val, never trained on.
    Xe_te = Xe[te]
    bl_te = bytes_log[te]
    rch_te = reach[te]
    meta_te = [meta[i] for i in te]

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

    # Per-head loss normalization (mathematically equivalent to inverse-loss-
    # weighted training in natural-space):
    #
    # The MLP trains under MSE on a flat output vector concatenating heads of
    # very different scales — log-bytes (~10..14), log-time (~-3..3), and
    # narrow scalars like filter_sharpness (0..7) or multi_pass_stats (0..1).
    # Raw MSE gradient is dominated by the wide-range heads, leaving the
    # narrow ones unfit (581% / 2770% normalized RMSE observed pre-fix).
    #
    # Fix: standardize each scalar block to (label - μ) / σ before fit, then
    # absorb the inverse affine into the final-layer weights AFTER fit:
    #     coefs_[-1][:, i] *= σ_block
    #     intercepts_[-1][i] = intercepts_[-1][i] * σ_block + μ_block
    # The post-fit network produces predictions in natural units, so
    # `OUTPUT_SPECS` (bounds/round/discrete_set) and the bake artifact (which
    # captures `coefs_/intercepts_` directly) work unchanged. Bytes/time/
    # metric log heads stay untouched — they're already log-space and their
    # downstream consumers expect that.
    #
    # Each scalar block is n_cells columns; the (μ, σ) used here are scalar
    # (one per block, computed over all rows × cells in the block) so the
    # rescale is a single multiply per output column. Using per-cell (μ_c,
    # σ_c) within a block would also be correct but adds no value — the
    # teachers already produce sensible per-cell predictions; we only need
    # to balance the relative gradient magnitude across heads.
    scalar_block_starts: list[tuple[int, int, str, float, float]] = []
    next_block = 1
    if time_pred_tr is not None:
        next_block += 1
    if metric_pred_tr is not None:
        next_block += 1
    for axis in SCALAR_AXES:
        start = next_block * n_cells
        end = (next_block + 1) * n_cells
        block = soft_tr[:, start:end]
        mu = float(np.mean(block))
        sigma = float(np.std(block))
        # Guard against zero-variance heads (constant teacher predictions).
        # σ=0 means the block already contributes nothing to the gradient
        # and absorbing it would divide by zero — skip rescaling, leave
        # labels unchanged.
        if sigma < 1e-12:
            sys.stderr.write(
                f"  per-head norm: scalar block '{axis}' has σ≈0 "
                f"(μ={mu:.4f}); skipping rescale\n"
            )
            scalar_block_starts.append((start, end, axis, 0.0, 1.0))
        else:
            soft_tr[:, start:end] = (block - mu) / sigma
            scalar_block_starts.append((start, end, axis, mu, sigma))
            sys.stderr.write(
                f"  per-head norm: scalar block '{axis}' "
                f"μ={mu:.4f} σ={sigma:.4f} (standardized for fit)\n"
            )
        next_block += 1

    hidden_repr = "x".join(str(x) for x in hidden_layer_sizes)
    sys.stderr.write(
        f"\nTraining MLP student (hidden={hidden_repr}, output_dim={soft_tr.shape[1]})...\n"
    )

    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    Xe_te_s = scaler.transform(Xe_te) if len(te) else Xe_va_s[:0]
    if args.hard_example_weighting != "none" and args.activation != "leakyrelu":
        sys.stderr.write(
            "  WARNING: --hard-example-weighting is leakyrelu-only; "
            f"ignored under --activation={args.activation}\n"
        )
    if args.activation == "leakyrelu":
        sys.stderr.write("  using PyTorch backend (LeakyReLU(0.01))\n")
        if args.hard_example_weighting != "none":
            sys.stderr.write(
                f"  hard-example weighting: mode={args.hard_example_weighting} "
                f"α={args.hard_example_alpha} ema_window={args.hard_example_ema_window} "
                f"clip=[{1.0/max(args.hard_example_clip,1e-6):.3f},"
                f"{args.hard_example_clip:.3f}]\n"
            )
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
            hard_example_mode=args.hard_example_weighting,
            hard_example_alpha=args.hard_example_alpha,
            hard_example_ema_window=args.hard_example_ema_window,
            hard_example_clip=args.hard_example_clip,
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

    # Post-fit: absorb the inverse standardization (label = ŷ_std * σ + μ)
    # into the final layer so the network's forward pass produces natural-
    # unit predictions. After this point `student.coefs_/intercepts_` ARE
    # the natural-unit weights — both the bake artifact (which serializes
    # those arrays directly into `layers`) and our numpy forward pass below
    # see the rescaled values. For sklearn `MLPRegressor`, `predict()` uses
    # `coefs_/intercepts_` so it stays consistent. For the torch student
    # the underlying `nn.Sequential` is NOT mutated — we therefore route
    # all downstream prediction through `_predict_via_coefs` (numpy) so
    # both backends produce predictions matching what gets shipped.
    final_W = student.coefs_[-1]   # shape: (hidden_last, output_dim)
    final_b = student.intercepts_[-1]  # shape: (output_dim,)
    for start, end, axis, mu, sigma in scalar_block_starts:
        if sigma == 1.0 and mu == 0.0:
            continue  # σ≈0 path — labels were left unchanged, nothing to absorb
        final_W[:, start:end] *= sigma
        final_b[start:end] = final_b[start:end] * sigma + mu

    Y_va_pred = _predict_via_coefs(student, Xe_va_s, args.activation)
    # Per-output (p01, p99) on held-out validation predictions -- consumed by
    # bake_picker.py's encode_output_bounds() for the codec's OOD-on-output
    # runtime check. Computed over the RAW model output (every output
    # neuron: bytes head + any scalar heads), matching `n_outputs` generically
    # so this covers pickers with scalar axes too, not just jxl-modular's
    # bytes-only (SCALAR_AXES=[]) shape. Without this, bake_picker.py falls
    # back to open ±inf sentinels and the OOD check is a silent no-op (found
    # 2026-07-03 baking the jxl-modular picker).
    output_bounds_p01 = np.percentile(Y_va_pred, 1, axis=0)
    output_bounds_p99 = np.percentile(Y_va_pred, 99, axis=0)
    output_bounds_computed = [
        {"p01": float(lo), "p99": float(hi)}
        for lo, hi in zip(output_bounds_p01, output_bounds_p99)
    ]
    pred_bytes = Y_va_pred[:, :n_cells]
    # Use the same per-block offsets we computed for normalization above —
    # this fixes a latent bug where the original `(i+1)*n_cells` indexing
    # silently mis-sliced when `has_time_head` or `has_metric_head` shifted
    # the scalar blocks downstream.
    student_pred_scalars = {
        axis: Y_va_pred[:, start:end]
        for start, end, axis, _mu, _sigma in scalar_block_starts
    }

    student_argmin = evaluate_argmin(pred_bytes, bl_va, rch_va, meta_va, all_mask)
    student_scalars = evaluate_scalars(student_pred_scalars, scalars_va, rch_va)
    sys.stderr.write(
        f"\nStudent metrics: argmin mean overhead {student_argmin['mean_pct']:.2f}% "
        f"argmin_acc {student_argmin['argmin_acc']:.1%}\n"
    )

    # --- Top-K-verify: does narrowing-by-content + encode-verifying the K
    # predicted-cheapest cells reach <=1% achieved RD, and at what K?
    student_topk = evaluate_topk_verify(pred_bytes, bl_va, rch_va, all_mask)
    sys.stderr.write(
        "\nTop-K-verify (rank by predicted bytes, encode-verify K cheapest, pick min actual):\n"
    )
    for k in sorted(student_topk):
        r = student_topk[k]
        sys.stderr.write(
            f"  K={k}: mean {r['mean_pct']:.3f}%  p50 {r['p50_pct']:.3f}%  "
            f"p90 {r['p90_pct']:.3f}%  p99 {r['p99_pct']:.2f}%  max {r['max_pct']:.2f}%  "
            f"oracle-in-topK {r['hit_rate']:.1%}  (~{r['mean_verified']:.1f} encodes/row)\n"
        )
    sys.stderr.write(
        "  scalar RMSE: " + "  ".join(
            f"{axis} {student_scalars[axis]:.4f}" for axis in SCALAR_AXES
        ) + "\n"
    )

    # --- Diagnostics: also evaluate student on TRAIN to detect overfit
    Y_tr_pred = _predict_via_coefs(student, Xe_tr_s, args.activation)
    pred_bytes_tr = Y_tr_pred[:, :n_cells]
    meta_tr = [meta[i] for i in tr]
    from collections import Counter as _DBGC
    sys.stderr.write(f"DEBUG meta_tr size dist: {dict(_DBGC(m[1] for m in meta_tr))}\n")
    sys.stderr.write(f"DEBUG SIZE_CLASSES at gate: {SIZE_CLASSES}\n")
    student_argmin_tr = evaluate_argmin(pred_bytes_tr, bl_tr, rch_tr, meta_tr, all_mask)
    sys.stderr.write(
        f"  train: mean overhead {student_argmin_tr['mean_pct']:.2f}% "
        f"argmin_acc {student_argmin_tr['argmin_acc']:.1%} "
        f"(gap to val: {student_argmin['mean_pct'] - student_argmin_tr['mean_pct']:+.2f}pp)\n"
    )

    # --- Feature-gated knob-veto safety bounds ---------------------------------
    # The pure-argmin (K=1) single-encode picker catastrophically mis-sets a
    # categorical toggle on a tiny fraction of images, blowing WORST_ROW past the
    # 200% gate. Derive per-(axis,value) feature-gated vetoes on the TRAIN picker
    # that bound the worst case (they shrink the picker's reachable set per row;
    # the oracle is never vetoed), then apply them to val/test so the reported
    # numbers AND the WORST_ROW gate reflect the DEPLOYED (vetoed) picker. The
    # exact rules ship in the manifest (`knob_vetoes`) for the runtime to enforce.
    knob_vetoes = derive_knob_vetoes(
        cells, list(CATEGORICAL_AXES), pred_bytes_tr, bl_tr, rch_tr, meta_tr,
        feats, feat_cols, all_mask,
    )
    veto_va = build_veto_mask(
        knob_vetoes, meta_va, feats, feat_cols, cells, list(CATEGORICAL_AXES)
    )
    veto_te = (
        build_veto_mask(knob_vetoes, meta_te, feats, feat_cols, cells, list(CATEGORICAL_AXES))
        if len(te) else None
    )
    # Override the raw (un-vetoed) val student_argmin computed above with the
    # vetoed picker so diag / manifest / summary report the deployed model.
    # `student_argmin_tr` (train) is intentionally left UN-vetoed: it is the
    # overfit reference (the raw-train vs raw-val gap logged just above), not a
    # deployed-picker number.
    if knob_vetoes:
        student_argmin = evaluate_argmin(
            pred_bytes, bl_va, rch_va, meta_va, all_mask, veto=veto_va
        )
        sys.stderr.write(
            f"  after {len(knob_vetoes)} knob-veto(s): val mean overhead "
            f"{student_argmin['mean_pct']:.2f}% (max {student_argmin['max_pct']:.1f}%, "
            f"p99 {student_argmin['p99_pct']:.1f}%)\n"
        )

    # --- Held-out TEST (origins ending 7/9) — the honest generalization number.
    # NEVER trained or tuned on; reported alongside val so the gap val→test shows
    # any val overfit. See docs/CLEAN_PICKER_PROGRAM.md.
    student_argmin_te = None
    student_topk_te = None
    if len(te):
        Y_te_pred = _predict_via_coefs(student, Xe_te_s, args.activation)
        pred_bytes_te = Y_te_pred[:, :n_cells]
        student_argmin_te = evaluate_argmin(
            pred_bytes_te, bl_te, rch_te, meta_te, all_mask, veto=veto_te
        )
        student_topk_te = evaluate_topk_verify(pred_bytes_te, bl_te, rch_te, all_mask)
        sys.stderr.write(
            f"\n  TEST (7/9 origins): argmin mean {student_argmin_te['mean_pct']:.2f}% "
            f"argmin_acc {student_argmin_te['argmin_acc']:.1%} "
            f"(val→test gap {student_argmin_te['mean_pct'] - student_argmin['mean_pct']:+.2f}pp)\n"
        )
        for k in sorted(student_topk_te):
            r = student_topk_te[k]
            sys.stderr.write(
                f"    TEST K={k}: mean {r['mean_pct']:.3f}%  p90 {r['p90_pct']:.3f}%  "
                f"oracle-in-topK {r['hit_rate']:.1%}\n"
            )
    else:
        sys.stderr.write("  TEST: 0 rows (no 7/9 origins in this corpus)\n")

    # Per-row val breakdown for stratification. Vetoed so the WORST_ROW gate
    # (fed by `worst` below) reflects the deployed (knob-vetoed) picker.
    val_per_row = evaluate_argmin_per_row(
        pred_bytes, bl_va, rch_va, meta_va, all_mask, veto=veto_va
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
        # Also save the raw eval arrays so safety-bound / mask experiments
        # can re-run the argmin offline under arbitrary per-cell masks
        # without retraining. Gated on --dump-overheads; additive.
        _npz = args.dump_overheads.with_suffix(".eval.npz")
        _arrs = dict(
            pred_bytes=pred_bytes, actual_bytes=bl_va, reach=rch_va,
            all_mask=all_mask,
            images=np.array([r[0] for r in meta_va]),
            sizes=np.array([r[1] for r in meta_va]),
            zqs=np.array([int(r[2]) for r in meta_va]),
            cells=np.array([str(c) for c in cells]),
        )
        if 'pred_bytes_te' in dir() and len(te):
            _arrs.update(
                te_pred_bytes=pred_bytes_te, te_actual_bytes=bl_te, te_reach=rch_te,
                te_images=np.array([r[0] for r in meta_te]),
                te_sizes=np.array([r[1] for r in meta_te]),
                te_zqs=np.array([int(r[2]) for r in meta_te]),
            )
        # TRAIN arrays too — knob-veto rules are DERIVED on train, so offline
        # tuning of the veto greedy (matching the deployed derivation) needs the
        # train picker, not just val/test.
        _arrs.update(
            tr_pred_bytes=pred_bytes_tr, tr_actual_bytes=bl_tr, tr_reach=rch_tr,
            tr_images=np.array([r[0] for r in meta_tr]),
            tr_sizes=np.array([r[1] for r in meta_tr]),
            tr_zqs=np.array([int(r[2]) for r in meta_tr]),
        )
        np.savez_compressed(_npz, **_arrs)
        sys.stderr.write(f"  wrote eval arrays → {_npz}\n")
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

    # Merge codec-specific safety thresholds over the defaults up-front so the
    # unachievable-zone descriptor below uses the SAME min_train_rows_per_size_zq
    # the DATA_STARVED_SIZE gate later exempts on (declared-zone ≡ exempted-tail).
    thresholds = dict(DEFAULT_SAFETY_THRESHOLDS)
    # Verify-codec gate (VERIFY_K > 1, user-approved 2026-06-29): a K-verify codec
    # encodes K candidates and keeps the best, so its worst-of-K tolerance is a
    # different regime than the K=1 tightening (worst<100 / p99<40). Loosen the
    # single-row + per-bin p99 ceilings for verify codecs; K=1 codecs keep the
    # tightened bars. Applied BEFORE the codec's explicit SAFETY_THRESHOLDS so a
    # codec can still override either way.
    if VERIFY_K > 1:
        thresholds["max_single_row_overhead_pct"] = 160.0
        thresholds["max_per_zq_p99_overhead_pct"] = 42.0
        thresholds["max_per_size_p99_overhead_pct"] = 42.0
    _codec_thresholds = getattr(
        sys.modules.get(parse_config_name.__module__, sys.modules[__name__]),
        "SAFETY_THRESHOLDS",
        None,
    )
    if _codec_thresholds:
        thresholds.update(_codec_thresholds)

    # Declared unachievable zones + fallback knobsets (the deploy-honored
    # complement to the zone-aware DATA_STARVED_SIZE gate). Same per-size
    # density ceiling the gate exempts above → declared-zone ≡ exempted-tail ≡
    # runtime fallback region. Shipped in hybrid_heads_manifest →
    # zenpicker.unachievable_zones metadata by bake_picker.
    unachievable_zones = compute_unachievable_zones(
        meta_tr, rch_tr, train_rows_by_size_zq, SIZE_CLASSES, ZQ_TARGETS,
        thresholds["min_train_rows_per_size_zq"],
        SCALAR_AXES, SCALAR_DISPLAY_RANGES,
    )
    if unachievable_zones:
        sys.stderr.write(
            "  ℹ unachievable zones declared (size_class: ceiling_zq → "
            "fallback cell@scalar): "
            + ", ".join(
                f"{z['size_class']}:>{z['ceiling_zq']:.0f}→c{z['fallback_cell']}"
                f"@{z['fallback_scalar']:.0f}" for z in unachievable_zones
            )
            + "\n"
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
        # Feature-gated knob-veto safety bounds derived on TRAIN and applied to
        # the val/test numbers + the WORST_ROW gate above. Empty list when no
        # veto was needed/found. Also shipped in hybrid_heads_manifest for the
        # runtime to enforce on the deployed picker.
        "knob_vetoes": knob_vetoes,
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
    # `thresholds` already merged (defaults + codec overrides) above, before
    # the unachievable-zone descriptor, so the zone ceiling and the gate
    # exemption share one min_train_rows_per_size_zq.
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
    # Build a per-input list of feature_transforms (length = n_inputs).
    # Per-feat_col transforms come from FEATURE_TRANSFORMS; the engineered
    # axes downstream (size_oh, log_px, zq_norm, interactions, icc
    # placeholder) get explicit `identity` entries so the runtime's
    # parser (which strict-checks `len(transforms) == n_inputs` via
    # parse_feature_transforms in zenpredict/src/feature_transform.rs)
    # accepts the bake. Earlier bakes emitted a length-len(feat_cols)
    # array and now fail to parse with `feature_transforms length 51 !=
    # expected 112` — see the v0.6 rebake.
    n_inputs_total = int(Xe.shape[1])
    feat_transform_list = [
        FEATURE_TRANSFORMS.get(c, "identity") if FEATURE_TRANSFORMS else "identity"
        for c in feat_cols
    ] + ["identity"] * (n_inputs_total - len(feat_cols))
    # Parallel array of per-input parameter vectors. Engineered axes
    # (size_oh / log_px / zq_norm / interactions / icc) take an empty
    # param list — they're always Identity-transformed. Only the
    # codec's natural feat_cols can carry params.
    feat_transform_params_list: list[list[float]] = []
    for c in feat_cols:
        params = list(FEATURE_TRANSFORM_PARAMS.get(c, [])) \
            if FEATURE_TRANSFORM_PARAMS else []
        feat_transform_params_list.append(params)
    feat_transform_params_list.extend(
        [] for _ in range(n_inputs_total - len(feat_cols))
    )
    # Expand OUTPUT_SPECS (keyed by head name) to a per-output-index
    # array of length n_outputs, in `output_layout` order. Codecs that
    # define every head in OUTPUT_SPECS get a full per-output array;
    # configs that omit OUTPUT_SPECS emit an empty list (bake_picker
    # reads that as "no per-output specs", ZNPR v3 round-trips raw).
    output_specs_array: list[dict] = []
    if OUTPUT_SPECS:
        # Default specs for trainer-internal heads (time_log, metric_log)
        # the codec configs don't need to declare. These are log-space
        # diagnostic outputs; identity transform with very wide bounds.
        DEFAULT_HEAD_SPECS = {
            "time_log": {"bounds": [-30.0, 30.0], "transform": "identity"},
            "metric_log": {"bounds": [-30.0, 30.0], "transform": "identity"},
        }
        per_idx: list[dict | None] = [None] * output_dim
        missing_heads = []
        for head_name, span in output_layout.items():
            spec = OUTPUT_SPECS.get(head_name) or DEFAULT_HEAD_SPECS.get(head_name)
            if spec is None:
                missing_heads.append(head_name)
                continue
            start, end = int(span[0]), int(span[1])
            for i in range(start, end):
                per_idx[i] = dict(spec)
        if missing_heads:
            sys.stderr.write(
                f"WARNING: OUTPUT_SPECS is set but missing entries for "
                f"heads {missing_heads}; emitting empty output_specs "
                f"(bake will treat as raw passthrough).\n"
            )
            output_specs_array = []
        elif any(s is None for s in per_idx):
            # Defensive: layout span didn't cover full range — should not
            # happen but emit empty rather than partial.
            sys.stderr.write(
                "WARNING: output_layout did not cover all n_outputs; "
                "emitting empty output_specs.\n"
            )
            output_specs_array = []
        else:
            output_specs_array = [dict(s) for s in per_idx]  # type: ignore[arg-type]

    out = {
        "n_inputs": int(Xe.shape[1]),
        "n_outputs": output_dim,
        "n_cells": n_cells,
        "safety_profile": args.objective,
        "config_names": {int(k): v for k, v in CONFIG_NAMES.items()},
        "feat_cols": feat_cols,
        # Size-class grid the picker was actually trained on (the corpus-present
        # subset of tiny/small/medium/large). The size_class one-hot inside the
        # engineered input vector has this length; the runtime maps an image's
        # pixel count to one of these (clamping larger images to the last).
        "size_classes": list(SIZE_CLASSES),
        # Per-feat_col pre-standardize transform (parallel array to
        # `feat_cols`). Runtime not yet consuming this — fresh bakes
        # with non-identity transforms produce wrong predictions until
        # the codec runtime applies the same transform pre-scaler. See
        # imazen/zenanalyze#52.
        "feature_transforms": feat_transform_list,
        # Per-input parameter vector for the parameterized transforms
        # (clip_then_log1p / winsor_p99 / quantile_bins). Parallel
        # array to feature_transforms. Length = n_inputs. Empty list
        # for non-parameterized variants and engineered axes.
        # bake_picker.py forwards into the runtime's
        # `zentrain.feature_transform_params` metadata key when any
        # entry is non-empty.
        "feature_transform_params": feat_transform_params_list,
        # Per-output OutputSpec metadata in n_outputs order. Empty when
        # OUTPUT_SPECS is unset on the codec config.
        "output_specs": output_specs_array,
        # Sparse hand-tune overrides — list of {idx, value} dicts.
        "sparse_overrides": list(SPARSE_OVERRIDES),
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
            # K-verify: the codec should encode the top-`verify_k` cheapest-
            # predicted reachable cells and keep the best by actual bytes. 1 =
            # single-encode (most codecs). The safety report's overhead/p99/worst
            # were evaluated at this K, so the deployment must honor it.
            "verify_k": int(VERIFY_K),
            "output_layout": output_layout,
            # Feature-gated knob-veto safety bounds the runtime must enforce on
            # the deployed single-encode picker: for each rule, when
            # feat[`feat`] `op` `threshold` holds for the image, forbid every
            # cell whose `axis` == `value`. Derived on TRAIN, validated to bound
            # the WORST_ROW overhead. Empty list = no veto.
            "knob_vetoes": knob_vetoes,
            # Size-discriminated unachievable zones + fallback knobsets. Per
            # size class: the achievable zq ceiling (above it the target is
            # physically unreachable) + the fallback cell+scalar (best-
            # achievable knobset). bake_picker emits these as the
            # zenpicker.unachievable_zones metadata; the runtime
            # (UnachievableZones::resolve) maps feat_pixel_count → size class
            # and returns the fallback when target_zq exceeds the ceiling,
            # instead of an unreachable argmin. Deploy-side complement to the
            # zone-aware DATA_STARVED_SIZE gate. Empty list = every grid target
            # reachable for every size.
            "unachievable_zones": unachievable_zones,
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
        "output_bounds": output_bounds_computed,
    }
    # zenanalyze-api reuse-key stamps from the codec config's ANALYSIS_PROVENANCE
    # (outside-in: which zenanalyze + config extracted the features). bake_picker.py
    # forwards these to the ZNPR metadata; absent -> unstamped -> safe own-pass.
    from _provenance import stamps_from_provenance

    out.update(stamps_from_provenance(ANALYSIS_PROVENANCE))
    # Fine-grained serialization provenance of the training features themselves
    # (per-feature version hashes + descriptor framing), recorded verbatim so a
    # future run can validate reuse feature-by-feature. Carried from the FEATURES
    # Parquet metadata / sidecar; absent -> simply not recorded (legacy-safe).
    feat_block = _feature_provenance_block(FEATURES)
    if feat_block:
        out["feature_provenance"] = feat_block
        _check_provenance_agreement(feat_block, ANALYSIS_PROVENANCE)
        sys.stderr.write("  [provenance] recorded FEATURES serialization provenance\n")
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
