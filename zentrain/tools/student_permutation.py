#!/usr/bin/env python3
"""
Permutation feature importance for hybrid-heads student MLPs.

Designed for models produced by `train_hybrid.py` (per-cell argmin,
not per-config HistGB). Loads the trained student JSON, runs forward
passes on the validation set, permutes each feature column, and
reports the per-feature overhead delta.

**O(n_features × val_rows × forward_pass_cost)** — for typical
picker sizes (~40 features, ~2 K val rows, 96→192→192→192→80 MLP)
this completes in **seconds**, not hours. The existing
`feature_ablation.py --method=permutation` retrains 1 HistGB per
*config*, which explodes when the picker has scalar heads
(110 K configs × 100 features × N repeats = ~30 min wedge before
the first per-feature delta is even measurable).

This tool is the "fast path" — the trained student already knows
which features it uses; we just probe it directly.

Usage:

    python3 student_permutation.py \\
        --codec-config rav1e_picker_config \\
        --model-json benchmarks/rav1e_picker_v0_1.json \\
        --output benchmarks/rav1e_student_perm.json \\
        --n-repeats 5

Outputs a JSON like:

    {
      "schema_version": 1,
      "model_path": "...",
      "baseline": { "mean_pct": 3.88, "argmin_acc": 0.238, ... },
      "per_feature": {
        "feat_variance": { "mean_pct_permuted": 5.13, "delta_pp": 1.25, ... },
        ...
      },
      "ranked_by_delta": [
        ["feat_dct_compressibility_y", 1.91],
        ["feat_variance",              1.25],
        ...
      ],
      "cull_candidates": ["feat_log_aspect_abs", ...]    # delta < 0.05 pp
    }
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Codec-config plumbing — shared with feature_ablation.py /
# train_hybrid.py
# ---------------------------------------------------------------------

PARETO: Path = None  # type: ignore[assignment]
FEATURES: Path = None  # type: ignore[assignment]
KEEP_FEATURES: list[str] = []  # type: ignore[assignment]
ZQ_TARGETS: list[int] = []  # type: ignore[assignment]
parse_config_name = None  # type: ignore[assignment]


def load_codec_config(name: str) -> None:
    global PARETO, FEATURES, KEEP_FEATURES, ZQ_TARGETS, parse_config_name
    mod = importlib.import_module(name)
    PARETO = mod.PARETO
    FEATURES = mod.FEATURES
    KEEP_FEATURES = list(mod.KEEP_FEATURES)
    ZQ_TARGETS = list(mod.ZQ_TARGETS)
    parse_config_name = mod.parse_config_name


# ---------------------------------------------------------------------
# Data loading — minimal version, doesn't need the full per-config
# Y-tensor that train_hybrid builds. Per-row optimum is enough for
# overhead computation against the picker's cell choice.
# ---------------------------------------------------------------------

SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}


def load_features(path: Path) -> tuple[dict, list[str]]:
    feats: dict[tuple[str, str], np.ndarray] = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        cols = [c for c in KEEP_FEATURES if c in all_cols]
        n_dropped = 0
        for r in rdr:
            vals = []
            has_nan = False
            for c in cols:
                v = r[c]
                if v == "" or v is None:
                    has_nan = True
                    vals.append(float("nan"))
                else:
                    fv = float(v)
                    if fv != fv:
                        has_nan = True
                    vals.append(fv)
            if has_nan:
                # Tiny images skip percentile features (zenanalyze #49).
                # At inference these go through the OOD-bounds fallback;
                # the student MLP was never trained on them. Drop them
                # from the permutation eval set too.
                n_dropped += 1
                continue
            feats[(r["image_path"], r["size_class"])] = np.array(
                vals, dtype=np.float32
            )
        if n_dropped:
            sys.stderr.write(
                f"Dropped {n_dropped} (image, size) keys with NaN "
                f"feature values.\n"
            )
    return feats, cols


def load_pareto(path: Path) -> dict:
    """For each (image, size_class), collect all (config_name, q,
    bytes, zensim) rows.
    """
    out: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                bytes_v = int(r["bytes"])
                z = float(r["zensim"])
            except (ValueError, KeyError, TypeError):
                continue
            key = (
                r["image_path"],
                r["size_class"],
                int(r["width"]),
                int(r["height"]),
            )
            out[key].append(
                {
                    "config_name": r["config_name"],
                    "bytes": bytes_v,
                    "zensim": z,
                }
            )
    return out


def build_eval_dataset(
    pareto: dict,
    feats: dict,
    feat_cols: list[str],
    n_cells: int,
    cell_for_config: dict[str, int],
) -> tuple[np.ndarray, list[float], list[int], list[float], list[float], list[tuple]]:
    """For each (image, size, target_zq), compute:
      - raw_feats[n_feat]
      - log_px (float)
      - target_zq (int)
      - row_optimal_bytes (smallest bytes any reachable cell achieved)
      - per_cell_best_bytes[n_cells] — smallest bytes within that cell
        achieving zensim ≥ target_zq, or +inf if unreachable

    Returns parallel arrays + meta tuple list for held-out splitting.
    """
    raw_rows: list[np.ndarray] = []
    log_px_rows: list[float] = []
    target_zq_rows: list[int] = []
    row_opt_rows: list[float] = []
    cell_bytes_rows: list[np.ndarray] = []
    meta_rows: list[tuple] = []

    for (image, size_class, w, h), members in pareto.items():
        f = feats.get((image, size_class))
        if f is None:
            continue
        log_px = math.log(max(1, w * h))

        for tz in ZQ_TARGETS:
            cell_best = np.full(n_cells, np.inf, dtype=np.float32)
            row_opt = float("inf")
            any_reach = False
            for m in members:
                if m["zensim"] < tz:
                    continue
                ci = cell_for_config.get(m["config_name"])
                if ci is None:
                    continue
                if m["bytes"] < cell_best[ci]:
                    cell_best[ci] = m["bytes"]
                if m["bytes"] < row_opt:
                    row_opt = m["bytes"]
                any_reach = True
            if not any_reach:
                continue
            raw_rows.append(f)
            log_px_rows.append(log_px)
            target_zq_rows.append(int(tz))
            row_opt_rows.append(row_opt)
            cell_bytes_rows.append(cell_best)
            meta_rows.append((image, size_class, int(tz), int(w), int(h)))

    raw = np.array(raw_rows, dtype=np.float32)
    cell_bytes = np.array(cell_bytes_rows, dtype=np.float32)
    return raw, log_px_rows, target_zq_rows, row_opt_rows, cell_bytes, meta_rows


# ---------------------------------------------------------------------
# Forward pass — pure numpy, matches train_hybrid's PyTorch student
# (Linear → LeakyReLU(0.01) chain ending in linear regression heads).
# ---------------------------------------------------------------------


def make_forward_fn(model_json: dict):
    """Build a (raw_feats, w, h, target_zq) → output[n_outputs] callable.

    The MLP itself sees the engineered 96-dim input (raw_feats(n) +
    size_oh(4) + [log_px, log_px², zq_norm, zq_norm², zq_norm·log_px]
    + zq_norm·raw_feats(n) + [icc=0]). This factory bakes the
    engineering + standardization + layer chain into one closure.
    """
    layers = model_json["layers"]
    activation = model_json["activation"]
    scaler_mean = np.asarray(model_json["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(model_json["scaler_scale"], dtype=np.float32)
    n_outputs = int(model_json["n_outputs"])
    n_inputs_expected = int(model_json["n_inputs"])

    if activation not in ("relu", "leakyrelu"):
        raise ValueError(f"unsupported activation {activation}")
    leaky_slope = 0.01

    Ws = [np.asarray(l["W"], dtype=np.float32) for l in layers]
    bs = [np.asarray(l["b"], dtype=np.float32) for l in layers]

    def engineer(raw: np.ndarray, w: int, h: int, tz: int) -> np.ndarray:
        n = (w * h) if (w * h) > 0 else 1
        log_px = math.log(n)
        zq_norm = tz / 100.0
        if n < 64 * 64:
            size_oh = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif n < 256 * 256:
            size_oh = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        elif n < 1024 * 1024:
            size_oh = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        else:
            size_oh = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        zlx = np.array(
            [log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px],
            dtype=np.float32,
        )
        return np.concatenate([raw, size_oh, zlx, zq_norm * raw, np.array([0.0], dtype=np.float32)])

    def forward_one(raw: np.ndarray, w: int, h: int, tz: int) -> np.ndarray:
        x = engineer(raw, w, h, tz)
        if x.shape[0] != n_inputs_expected:
            raise ValueError(
                f"engineered length {x.shape[0]} != model n_inputs {n_inputs_expected} "
                f"(raw_feats={raw.shape[0]} — KEEP_FEATURES count must match the model's training run)"
            )
        x = (x - scaler_mean) / scaler_scale
        for i, (W, b) in enumerate(zip(Ws, bs)):
            x = x @ W + b
            if i < len(Ws) - 1:
                # hidden activations
                if activation == "leakyrelu":
                    x = np.where(x > 0, x, leaky_slope * x)
                else:
                    x = np.maximum(x, 0)
        if x.shape[0] != n_outputs:
            raise ValueError(f"output length {x.shape[0]} != n_outputs {n_outputs}")
        return x

    def forward_batch(
        raws: np.ndarray, ws: list[int], hs: list[int], tzs: list[int]
    ) -> np.ndarray:
        out = np.empty((len(raws), n_outputs), dtype=np.float32)
        for i in range(len(raws)):
            out[i] = forward_one(raws[i], ws[i], hs[i], tzs[i])
        return out

    return forward_batch


# ---------------------------------------------------------------------
# Cell mapping — matches train_hybrid's cell construction so we can
# argmin over the same axes the trained model uses.
# ---------------------------------------------------------------------


def build_cell_map(model_json: dict) -> tuple[int, dict[str, int]]:
    """Return (n_cells, dict[config_name → cell_id]).

    Walks the manifest's `categorical_cells` and uses train_hybrid's
    bytes-head ordering to recover the per-config → cell index map.
    """
    raw = model_json.get("config_names", [])
    if isinstance(raw, dict):
        # train_hybrid emits {config_id_str → config_name_str}.
        config_names: list[str] = list(raw.values())
    else:
        # Older shape: bare list of names.
        config_names = list(raw)
    n_cells = int(model_json["n_cells"])

    if not config_names:
        raise ValueError("model JSON missing `config_names`")

    # parse_config_name → categorical key tuple → cell_id
    # Same logic as train_hybrid.build_cell_index, abbreviated.
    mod = sys.modules[parse_config_name.__module__]
    categorical_axes = getattr(mod, "CATEGORICAL_AXES", ["speed"])

    parsed_all = {cn: parse_config_name(cn) for cn in config_names}

    def cat_key(p):
        return tuple(p.get(a) for a in categorical_axes)

    keys = sorted({cat_key(p) for p in parsed_all.values()})
    cell_id_for_key = {k: i for i, k in enumerate(keys)}

    if len(keys) != n_cells:
        # Sanity check — model and config say the same number of cells.
        sys.stderr.write(
            f"WARN: model n_cells={n_cells}, computed cells={len(keys)} from {len(config_names)} configs\n"
        )

    cell_for_config = {cn: cell_id_for_key[cat_key(p)] for cn, p in parsed_all.items()}
    return len(keys), cell_for_config


# ---------------------------------------------------------------------
# Permutation importance loop
# ---------------------------------------------------------------------


def evaluate(
    forward_batch,
    raw_va: np.ndarray,
    ws: list[int],
    hs: list[int],
    tzs: list[int],
    cell_bytes_va: np.ndarray,
    row_opt_va: list[float],
) -> dict:
    """Forward, argmin over bytes_log head, compute mean overhead %
    + argmin_acc.
    """
    out = forward_batch(raw_va, ws, hs, tzs)
    n_cells = cell_bytes_va.shape[1]
    bytes_log_pred = out[:, :n_cells]  # the bytes head
    overheads: list[float] = []
    correct = 0
    for i, opt in enumerate(row_opt_va):
        # Picker chooses the cell with smallest predicted bytes,
        # restricted to cells that the row could actually achieve.
        reachable = np.isfinite(cell_bytes_va[i])
        if not reachable.any():
            continue
        masked_pred = np.where(reachable, bytes_log_pred[i], np.inf)
        cell = int(np.argmin(masked_pred))
        actual_bytes = float(cell_bytes_va[i, cell])
        if not math.isfinite(actual_bytes):
            continue
        oh = (actual_bytes - opt) / opt * 100.0
        overheads.append(oh)
        # Argmin accuracy: did we pick a cell that matches the
        # row-optimal bytes (within float precision)?
        if abs(actual_bytes - opt) < max(1.0, opt * 1e-6):
            correct += 1
    if not overheads:
        return {"n": 0, "mean_pct": float("nan"), "argmin_acc": float("nan")}
    arr = np.asarray(overheads)
    return {
        "n": len(overheads),
        "mean_pct": float(np.mean(arr)),
        "p50_pct": float(np.median(arr)),
        "p90_pct": float(np.percentile(arr, 90)),
        "argmin_acc": float(correct / len(overheads)),
    }


def permutation_importance(
    forward_batch,
    raw_va: np.ndarray,
    ws: list[int],
    hs: list[int],
    tzs: list[int],
    cell_bytes_va: np.ndarray,
    row_opt_va: list[float],
    feat_cols: list[str],
    n_repeats: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    baseline = evaluate(forward_batch, raw_va, ws, hs, tzs, cell_bytes_va, row_opt_va)
    per_feat: dict[str, dict] = {}
    for fi, name in enumerate(feat_cols):
        means: list[float] = []
        accs: list[float] = []
        for _ in range(n_repeats):
            perm = raw_va.copy()
            perm[:, fi] = rng.permutation(perm[:, fi])
            m = evaluate(forward_batch, perm, ws, hs, tzs, cell_bytes_va, row_opt_va)
            if m["n"] == 0 or math.isnan(m["mean_pct"]):
                continue
            means.append(m["mean_pct"])
            accs.append(m["argmin_acc"])
        if not means:
            per_feat[name] = {**baseline, "delta_pp": 0.0}
            continue
        mean_perm = float(np.mean(means))
        per_feat[name] = {
            "mean_pct_permuted": mean_perm,
            "argmin_acc_permuted": float(np.mean(accs)),
            "delta_pp": mean_perm - baseline["mean_pct"],
        }
    return baseline, per_feat


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--model-json", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.20,
        help="Per-image holdout fraction for the val set. Same default "
        "as train_hybrid + feature_ablation so val rows are comparable.",
    )
    ap.add_argument(
        "--cull-threshold-pp",
        type=float,
        default=0.05,
        help="Δ_pp below which a feature is flagged as cull candidate. "
        "Default 0.05 — features that don't change overhead by more "
        "than 0.05 percentage points when shuffled aren't earning their keep.",
    )
    args = ap.parse_args()

    load_codec_config(args.codec_config)
    sys.stderr.write(f"Loading PARETO  = {PARETO}\n")
    sys.stderr.write(f"        FEATURES = {FEATURES}\n")
    sys.stderr.write(f"        MODEL    = {args.model_json}\n")

    pareto = load_pareto(PARETO)
    feats, feat_cols = load_features(FEATURES)
    sys.stderr.write(
        f"Loaded {len(pareto)} (image, size) cells × {len(feat_cols)} features\n"
    )

    model_json = json.loads(args.model_json.read_text())
    if int(model_json["n_inputs"]) != 2 * len(feat_cols) + 10:
        sys.stderr.write(
            f"WARN: model n_inputs={model_json['n_inputs']} but engineering "
            f"would produce {2 * len(feat_cols) + 10} for {len(feat_cols)} feat_cols — "
            "likely KEEP_FEATURES drift since training. Aborting.\n"
        )
        sys.exit(2)

    n_cells, cell_for_config = build_cell_map(model_json)
    sys.stderr.write(f"Model has {n_cells} cells, {len(cell_for_config)} configs\n")

    raw, log_px, tzs, row_opt, cell_bytes, meta = build_eval_dataset(
        pareto, feats, feat_cols, n_cells, cell_for_config
    )
    sys.stderr.write(f"Built {len(raw)} (image, size, target_zq) eval rows\n")

    # Held-out by image hash (matches train_hybrid + feature_ablation).
    rng = np.random.default_rng(args.seed)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * args.holdout_frac))
    val_set = set(images[:n_val])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Val rows: {len(val_idx)} (across {len(val_set)} images)\n")

    raw_va = raw[val_idx]
    cell_bytes_va = cell_bytes[val_idx]
    row_opt_va = [row_opt[i] for i in val_idx]
    ws_va = [meta[i][3] for i in val_idx]
    hs_va = [meta[i][4] for i in val_idx]
    tzs_va = [meta[i][2] for i in val_idx]

    forward_batch = make_forward_fn(model_json)

    sys.stderr.write(
        f"\nRunning permutation importance ({args.n_repeats} repeats × {len(feat_cols)} features)...\n"
    )
    baseline, per_feat = permutation_importance(
        forward_batch,
        raw_va,
        ws_va,
        hs_va,
        tzs_va,
        cell_bytes_va,
        row_opt_va,
        feat_cols,
        args.n_repeats,
        args.seed,
    )

    sys.stderr.write(
        f"\nBaseline: mean overhead {baseline['mean_pct']:.2f}% argmin_acc {baseline['argmin_acc']*100:.1f}%\n"
    )

    ranked = sorted(
        per_feat.items(), key=lambda kv: kv[1].get("delta_pp", 0.0), reverse=True
    )
    cull = [n for n, m in ranked if m.get("delta_pp", 0.0) < args.cull_threshold_pp]

    print(f"\n{'feature':<40} {'Δ_pp':>10} {'mean_perm%':>10} {'verdict':>8}")
    print("-" * 72)
    for name, m in ranked:
        delta = m.get("delta_pp", 0.0)
        mean_perm = m.get("mean_pct_permuted", baseline["mean_pct"])
        verdict = "KEEP" if delta >= args.cull_threshold_pp else "cull?"
        print(f"{name:<40} {delta:>+9.3f}pp {mean_perm:>9.2f}% {verdict:>8}")

    print(f"\nKEEP: {len(feat_cols) - len(cull)}    CULL: {len(cull)}")
    if cull:
        print(f"\nCull candidates (Δ_pp < {args.cull_threshold_pp}):")
        for c in cull:
            print(f"  - {c}")

    out = {
        "schema_version": 1,
        "codec_config": args.codec_config,
        "model_path": str(args.model_json),
        "feat_cols": feat_cols,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "cull_threshold_pp": args.cull_threshold_pp,
        "n_eval_rows": len(raw),
        "n_val_rows": int(len(val_idx)),
        "baseline": baseline,
        "per_feature": per_feat,
        "ranked_by_delta": [(n, m.get("delta_pp", 0.0)) for n, m in ranked],
        "cull_candidates": cull,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    sys.stderr.write(f"\nWrote {args.output}\n")


if __name__ == "__main__":
    main()
