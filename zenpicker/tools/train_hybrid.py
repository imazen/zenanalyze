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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
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


def load_codec_config(name: str):
    """Import a codec-config module and bind its exports to module-level
    names this script consumes. The codec module must define:
      PARETO, FEATURES, OUT_JSON, OUT_LOG, ZQ_TARGETS, KEEP_FEATURES,
      parse_config_name(name: str) -> dict.

    `parse_config_name` returns a dict whose keys partition into:
      - categorical axes (hashable values, used to form cells via
        `categorical_key()`)
      - scalar axes (float values; sentinel allowed for "not
        applicable" — e.g. lambda=0.0 in trellis-off cells)

    The `categorical_key` and the scalar-axis list are determined by
    the codec config; this script reads them via the codec's
    `CATEGORICAL_AXES` and `SCALAR_AXES` constants when present.
    """
    global PARETO, FEATURES, OUT_LOG, OUT_JSON
    global ZQ_TARGETS, KEEP_FEATURES, parse_config_name
    mod = importlib.import_module(name)
    PARETO = Path(mod.PARETO)
    FEATURES = Path(mod.FEATURES)
    OUT_LOG = Path(mod.OUT_LOG)
    OUT_JSON = Path(mod.OUT_JSON)
    ZQ_TARGETS = list(mod.ZQ_TARGETS)
    KEEP_FEATURES = list(mod.KEEP_FEATURES)
    parse_config_name = mod.parse_config_name
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
    """The (color, sub, trellis_on, sa) tuple. xyb cells always have sa=False."""
    return (parsed["color"], parsed["sub"], parsed["trellis_on"], parsed["sa"])


# ---------- Data loading ----------


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


# ---------- Build categorical cell mapping ----------


def build_cell_index():
    """Return:
       cells: list of dicts describing each cell (in stable order)
       cell_id_by_key: {(color, sub, trellis_on, sa) -> int}
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
        color, sub, trellis_on, sa = k
        # Pick a representative human-readable label
        sa_tag = "_sa" if sa else ""
        trel_tag = "trellis" if trellis_on else "noT"
        label = f"{color}_{sub}_{trel_tag}{sa_tag}"
        # Find member configs
        members = [cid for cid, p in parsed_all.items() if categorical_key(p) == k]
        cells.append(
            {
                "id": cell_id_by_key[k],
                "label": label,
                "color": color,
                "sub": sub,
                "trellis_on": trellis_on,
                "sa": sa,
                "member_config_ids": sorted(members),
            }
        )

    config_to_cell = {cid: cell_id_by_key[categorical_key(p)] for cid, p in parsed_all.items()}
    return cells, cell_id_by_key, config_to_cell, parsed_all


# ---------- Build training dataset ----------


def build_dataset(pareto, feats, feat_cols, cells, config_to_cell, parsed_all):
    """Per (image, size, zq) row, compute within-cell optimal:
       bytes_log[c]    = log(min bytes in cell c over configs that reach zq)
       chroma_scale[c] = chroma of the within-cell optimal
       lambda[c]       = lambda of the within-cell optimal
       reachable[c]    = 1 if any config in cell c reached zq, 0 otherwise
    """
    n_cells = len(cells)
    Xs_rows, Xe_rows = [], []
    bytes_log_rows, chroma_rows, lambda_rows, reach_rows = [], [], [], []
    meta = []

    for (image, size, w, h), samples in pareto.items():
        feat_key = (image, size)
        if feat_key not in feats:
            continue
        f = feats[feat_key]
        log_px = math.log(max(1, w * h))
        size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float32)
        size_oh[SIZE_INDEX[size]] = 1.0

        # Group samples by config to track per-config best.
        # (one config can have multiple q values; pareto-best for each
        # cell at each zq target is the cheapest config that crosses zq.)
        by_cfg = defaultdict(list)
        for s in samples:
            by_cfg[s["config_id"]].append(s)

        for zq in ZQ_TARGETS:
            cell_bytes = [math.inf] * n_cells
            cell_cs = [math.nan] * n_cells
            cell_lam = [math.nan] * n_cells
            cell_reach = [False] * n_cells

            for cfg_id, hits in by_cfg.items():
                # Cheapest sample for this config that reaches zq.
                best_b = math.inf
                for s in hits:
                    if s["zensim"] >= zq and s["bytes"] < best_b:
                        best_b = s["bytes"]
                if math.isinf(best_b):
                    continue
                c = config_to_cell[cfg_id]
                if best_b < cell_bytes[c]:
                    cell_bytes[c] = best_b
                    p = parsed_all[cfg_id]
                    cell_cs[c] = p["chroma_scale"]
                    cell_lam[c] = p["lambda"]
                    cell_reach[c] = True

            if not any(cell_reach):
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
            chroma = np.array(cell_cs, dtype=np.float32)
            lam = np.array(cell_lam, dtype=np.float32)
            reach = np.array(cell_reach, dtype=bool)

            Xs_rows.append(xs)
            Xe_rows.append(xe)
            bytes_log_rows.append(bytes_log)
            chroma_rows.append(chroma)
            lambda_rows.append(lam)
            reach_rows.append(reach)
            meta.append((image, size, zq))

    return (
        np.stack(Xs_rows),
        np.stack(Xe_rows),
        np.stack(bytes_log_rows),
        np.stack(chroma_rows),
        np.stack(lambda_rows),
        np.stack(reach_rows),
        meta,
    )


# ---------- Evaluation ----------


def evaluate_argmin(pred_bytes_log, actual_bytes_log, reach, meta, mask):
    """Categorical argmin over allowed reachable cells."""
    n_rows = pred_bytes_log.shape[0]
    overheads, correct = [], 0
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


def evaluate_scalars(pred_chroma, actual_chroma, pred_lam, actual_lam, reach):
    """RMSE on the chroma_scale and lambda predictions, computed over
    reachable cells only (where the targets exist).
    """
    rmse = {}
    cs_diff = []
    lam_diff = []
    for i in range(pred_chroma.shape[0]):
        for c in range(pred_chroma.shape[1]):
            if not reach[i, c]:
                continue
            cs_diff.append(pred_chroma[i, c] - actual_chroma[i, c])
            # lambda only meaningful when trellis_on (target != sentinel)
            if not math.isnan(actual_lam[i, c]) and actual_lam[i, c] > 0:
                lam_diff.append(pred_lam[i, c] - actual_lam[i, c])
    cs_arr = np.array(cs_diff, dtype=np.float64)
    lam_arr = np.array(lam_diff, dtype=np.float64) if lam_diff else np.array([0.0])
    rmse["chroma_scale"] = float(np.sqrt((cs_arr ** 2).mean()))
    rmse["chroma_scale_mae"] = float(np.abs(cs_arr).mean())
    rmse["lambda"] = float(np.sqrt((lam_arr ** 2).mean()))
    rmse["lambda_mae"] = float(np.abs(lam_arr).mean())
    return rmse


# ---------- Train ----------


def train_teacher_per_cell(
    Xs_tr,
    bytes_log_tr,
    chroma_tr,
    lam_tr,
    reach_tr,
    n_cells,
    params=None,
    bytes_quantile=None,
):
    """Per-cell HistGB regressors for: bytes_log, chroma_scale, lambda.

    Three teachers per cell × 12 cells × ~5 s each = ~3 min serial.
    With `train_teachers_per_cell_parallel` from `_zq_picker_lib.py`
    on a 16-core box, drops to ~30 s × 3 = ~90 s. ~12× speedup vs the
    pre-2026-04-29 serial loop.

    `params` defaults to `HISTGB_FULL` (production training). Pass
    `HISTGB_FAST` for iteration / ablation runs.

    `bytes_quantile`: when not None, switches the bytes head to
    quantile regression at that q (e.g. 0.99). Used by the
    `zensim_strict` safety profile so the bytes head predicts the
    worst-case-safe cost, not the mean. Chroma_scale and lambda
    heads always stay at mean regression — they predict the within-
    cell-optimal scalar conditional on the cell being chosen.
    """
    from _picker_lib import HISTGB_FULL, train_teachers_per_cell_parallel

    if params is None:
        params = HISTGB_FULL

    cs_means = np.nanmean(chroma_tr, axis=0)
    lam_means = np.nanmean(np.where(lam_tr > 0, lam_tr, np.nan), axis=0)

    # Bytes head — per-cell reach mask (cell achieved target_zq).
    bytes_params = dict(params)
    if bytes_quantile is not None:
        bytes_params["loss"] = "quantile"
        bytes_params["quantile"] = bytes_quantile
    teachers_bytes = train_teachers_per_cell_parallel(
        Xs_tr, bytes_log_tr, reach_tr, params=bytes_params, label="bytes"
    )

    # Chroma head — same reach mask (chroma_scale only meaningful
    # when the cell actually reaches target).
    teachers_chroma = train_teachers_per_cell_parallel(
        Xs_tr, chroma_tr, reach_tr, params=params, label="chroma"
    )

    # Lambda head — additional mask: lambda > 0 (trellis-on rows
    # only). noT cells will fall through to the < 50 row check
    # inside the worker and return None.
    lambda_extra_mask = lam_tr > 0
    teachers_lambda = train_teachers_per_cell_parallel(
        Xs_tr, lam_tr, reach_tr, extra_mask=lambda_extra_mask, params=params, label="lambda"
    )

    return teachers_bytes, teachers_chroma, teachers_lambda, cs_means, lam_means


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
        "--out-suffix",
        default=None,
        help="Override the OUT_JSON / OUT_LOG basename suffix. Defaults to "
        "the codec config's OUT_JSON for size_optimal, and "
        "<basename>_zensim_strict for zensim_strict.",
    )
    args = parser.parse_args()
    load_codec_config(args.codec_config)

    # Per-objective output naming. The codec config defines the
    # baseline OUT_JSON/OUT_LOG; we suffix when training a non-default
    # safety profile so both bakes can co-exist.
    global OUT_JSON, OUT_LOG
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
    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(f"Loaded {len(pareto)} cells × {len(feat_cols)} features\n")

    cells, cell_id_by_key, config_to_cell, parsed_all = build_cell_index()
    n_cells = len(cells)
    sys.stderr.write(f"\nCategorical cells: {n_cells}\n")
    for c in cells:
        sys.stderr.write(f"  {c['id']:>2d}: {c['label']:30s}  ({len(c['member_config_ids'])} configs)\n")

    Xs, Xe, bytes_log, chroma, lam, reach, meta = build_dataset(
        pareto, feats, feat_cols, cells, config_to_cell, parsed_all
    )
    sys.stderr.write(
        f"\nDecision rows: {len(Xs)}; Xs={Xs.shape[1]}, Xe={Xe.shape[1]}, n_cells={n_cells}\n"
    )

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
    cs_tr, cs_va = chroma[tr], chroma[va]
    lam_tr, lam_va = lam[tr], lam[va]
    rch_tr, rch_va = reach[tr], reach[va]
    meta_va = [meta[i] for i in va]

    # --- Teacher
    bytes_quantile = args.bytes_quantile if args.objective == "zensim_strict" else None
    t_bytes, t_chroma, t_lambda, cs_means, lam_means = train_teacher_per_cell(
        Xs_tr, bl_tr, cs_tr, lam_tr, rch_tr, n_cells, bytes_quantile=bytes_quantile
    )
    sys.stderr.write("\nGenerating teacher soft targets (val + train)...\n")
    bytes_pred_tr = teacher_predict_all(t_bytes, Xs_tr, np.nanmean(bl_tr, axis=0), n_cells)
    bytes_pred_va = teacher_predict_all(t_bytes, Xs_va, np.nanmean(bl_tr, axis=0), n_cells)
    chroma_pred_tr = teacher_predict_all(t_chroma, Xs_tr, cs_means, n_cells)
    chroma_pred_va = teacher_predict_all(t_chroma, Xs_va, cs_means, n_cells)
    lam_pred_tr = teacher_predict_all(t_lambda, Xs_tr, lam_means, n_cells)
    lam_pred_va = teacher_predict_all(t_lambda, Xs_va, lam_means, n_cells)

    all_mask = np.ones(n_cells, dtype=bool)
    teacher_argmin = evaluate_argmin(bytes_pred_va, bl_va, rch_va, meta_va, all_mask)
    teacher_scalars = evaluate_scalars(chroma_pred_va, cs_va, lam_pred_va, lam_va, rch_va)
    sys.stderr.write(
        f"\nTeacher metrics: argmin mean overhead {teacher_argmin['mean_pct']:.2f}% "
        f"argmin_acc {teacher_argmin['argmin_acc']:.1%}\n"
    )
    sys.stderr.write(
        f"  scalar RMSE: chroma {teacher_scalars['chroma_scale']:.4f}  "
        f"lambda {teacher_scalars['lambda']:.3f}\n"
    )

    # --- Student
    # Soft targets: 12 bytes + 12 chroma + 12 lambda = 36 outputs
    soft_tr = np.concatenate([bytes_pred_tr, chroma_pred_tr, lam_pred_tr], axis=1)
    sys.stderr.write(f"\nTraining MLP student (hidden=128x2, output_dim={soft_tr.shape[1]})...\n")

    scaler = StandardScaler()
    Xe_tr_s = scaler.fit_transform(Xe_tr)
    Xe_va_s = scaler.transform(Xe_va)
    student = MLPRegressor(
        hidden_layer_sizes=(128, 128),
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
    pred_chroma = Y_va_pred[:, n_cells : 2 * n_cells]
    pred_lambda = Y_va_pred[:, 2 * n_cells : 3 * n_cells]

    student_argmin = evaluate_argmin(pred_bytes, bl_va, rch_va, meta_va, all_mask)
    student_scalars = evaluate_scalars(pred_chroma, cs_va, pred_lambda, lam_va, rch_va)
    sys.stderr.write(
        f"\nStudent metrics: argmin mean overhead {student_argmin['mean_pct']:.2f}% "
        f"argmin_acc {student_argmin['argmin_acc']:.1%}\n"
    )
    sys.stderr.write(
        f"  scalar RMSE: chroma {student_scalars['chroma_scale']:.4f}  "
        f"lambda {student_scalars['lambda']:.3f}\n"
    )

    # --- Per-zq reach-rate gate (zensim_strict only; recorded
    # always so the manifest is shape-stable across profiles)
    meta_tr = [meta[i] for i in tr]
    reach_safety = compute_reach_safe_cells(
        bl_tr, rch_tr, meta_tr, n_cells, ZQ_TARGETS, args.reach_threshold
    )

    # --- Persist
    n_params = sum(c.size + i.size for c, i in zip(student.coefs_, student.intercepts_))
    out = {
        "n_inputs": int(Xe.shape[1]),
        "n_outputs": 3 * n_cells,
        "n_cells": n_cells,
        "safety_profile": args.objective,
        "config_names": {int(k): v for k, v in CONFIG_NAMES.items()},
        "feat_cols": feat_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(student.coefs_, student.intercepts_)
        ],
        "activation": "relu",
        "hybrid_heads_manifest": {
            "n_cells": n_cells,
            "cells": cells,
            "output_layout": {
                "bytes_log": [0, n_cells],
                "chroma_scale": [n_cells, 2 * n_cells],
                "lambda": [2 * n_cells, 3 * n_cells],
            },
            "lambda_notrellis_sentinel": getattr(
                sys.modules.get(parse_config_name.__module__, sys.modules[__name__]),
                "LAMBDA_NOTRELLIS_SENTINEL",
                0.0,
            ),
        },
        "training_objective": {
            "name": args.objective,
            "bytes_quantile": (
                args.bytes_quantile if args.objective == "zensim_strict" else None
            ),
            "reach_threshold": args.reach_threshold,
        },
        "reach_safety": reach_safety,
        "teacher_metrics": {"argmin": teacher_argmin, "scalars": teacher_scalars},
        "student_metrics": {"argmin": student_argmin, "scalars": student_scalars},
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

    w("\n# Hybrid-heads picker — categorical bytes + scalar (chroma_scale, lambda)")
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
    w(f"n_cells: {n_cells}, output_dim: {3 * n_cells}")
    w(f"Student: MLP {Xe.shape[1]} -> 128 -> 128 -> {3 * n_cells}, "
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
    w("## Scalar regression RMSE")
    w(f"  Teacher chroma_scale RMSE: {teacher_scalars['chroma_scale']:.4f}  "
      f"(MAE {teacher_scalars['chroma_scale_mae']:.4f}, range 0.6..1.5)")
    w(f"  Teacher lambda RMSE:       {teacher_scalars['lambda']:.3f}   "
      f"(MAE {teacher_scalars['lambda_mae']:.3f}, range 8..25)")
    w(f"  Student chroma_scale RMSE: {student_scalars['chroma_scale']:.4f}  "
      f"(MAE {student_scalars['chroma_scale_mae']:.4f})")
    w(f"  Student lambda RMSE:       {student_scalars['lambda']:.3f}   "
      f"(MAE {student_scalars['lambda_mae']:.3f})")

    OUT_LOG.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
