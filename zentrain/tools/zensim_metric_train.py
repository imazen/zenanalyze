#!/usr/bin/env python3
"""Train zensim's perceptual-metric MLP and bake to ZNPR v3.

Companion to ``train_hybrid.py`` (codec-picker trainer). Where the
hybrid trainer predicts a categorical cell + K scalar continuous
heads, this trainer predicts a single regression head (the zensim
quality / distance score) given a feature vector.

This trainer is the **Python-side research / ablation home** for
zensim metric work. The parallel Rust trainer at
``zensim-validate/src/bin/zensim_mlp_train.rs`` remains
authoritative for V_X SHIP bakes. Both trainers coexist:

  - Rust trainer = V_X ship bakes (per-step group-weighted pair
    sampling, used by recipes ``recipe_v0_16.sh`` and onward).
  - This Python trainer = research / ablation / cycle experiments;
    uses batched RankNet with per-row MSE weighting. The mechanisms
    differ on ``train_weight`` semantics (per cycle-13 finding);
    pick the trainer that matches the recipe semantic you need.

This module was the SCAFFOLDED Phase-3 trainer from the 2026-05-08
recovery cycle. The ports landed 2026-05-21 (task #201):

  PORT 1 (FOUNDATION):
    * ``train_loop`` end-to-end wiring (RankNet + MSE + TV + multi-
      dataset val SROCC + val-policy=min selection)
    * ``bake_to_znpr`` ZNPR-v3 JSON pipeline (shells to
      ``zenpredict-bake``; passes feature_transforms metadata
      through; LZ4 + i8 + zerobias-aware)

  PORT 2 (DCT_HF APPENDER):
    * ``attach_zenanalyze_features`` joins additional zenanalyze
      features by ``ref_basename`` from a TSV / Parquet sidecar.

  PORT 3 (MAGNITUDE-MATCHING LOSS):
    * ``magnitude_match_term`` adds a magnitude term
      ``λ · mean(|α·|target| − |pred‖²)`` to the loss.

  PORT 4 (SAMPLER BIAS — low-band oversample):
    * ``--low-band-oversample`` ratio. Builds a sampling probability
      weight per row from each row's ``target`` band; pairs sampled
      via ``torch.multinomial(weight)`` for the RankNet term.

  PORTS 5-7 (FILM HEADS / MOE / MULTI-TARGET LOSS):
    * Not implemented this pass. Multi-target supervision IS
      implementable via the ``--target-col`` argument by switching
      targets across epochs, but FiLM/MoE require an architectural
      branch that's better landed in the Rust trainer.

Canonical training data (per zensim/CLAUDE.md):
  /mnt/v/zen/zensim-training/canonical-2026-05-21/
    train/safesyn.parquet           196,086 × 372 features
    train/kadid.parquet              10,125 × 372 features (DMOS)
    train/tid.parquet                 3,000 × 372 features (MOS)
    train/konjnd-dense.parquet       20,160 × 372 features (PJND)
    val/cid22.parquet                 4,292 × 372 features (MCOS)
    val/{kadid,tid,konjnd,aic3}.parquet  — validation only

Bake target: ZNPR v3 always. Per zensim/CLAUDE.md "ZNPR v2
PROHIBITED" (2026-05-15). Use the local
``~/work/zen/zenanalyze/target/release/zenpredict-bake`` binary,
NOT the published 0.1.0 (which writes v2).

Example invocation (smoke test against canonical-2026-05-21):

    python3 zentrain/tools/zensim_metric_train.py \\
        --train safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet:1.0 \\
        --train kadid:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/kadid.parquet:0.3 \\
        --train tid:/mnt/v/zen/zensim-training/canonical-2026-05-21/train/tid.parquet:0.3 \\
        --val cid22:/mnt/v/zen/zensim-training/canonical-2026-05-21/val/cid22.parquet \\
        --val kadid:/mnt/v/zen/zensim-training/canonical-2026-05-21/val/kadid.parquet \\
        --val tid:/mnt/v/zen/zensim-training/canonical-2026-05-21/val/tid.parquet \\
        --target-col human_score \\
        --hidden 128 --epochs 50 --lr 1e-3 --batch-size 16384 \\
        --rank-weight 0.5 --val-policy min --seed 1 \\
        --out-dir /tmp/zensim_recovery_phase3 --tag smoke

The bake step is triggered with ``--bake-out PATH``; without it the
trainer emits ``model.pt``, ``scaler.npz``, ``meta.json``, and
``predictions_val.parquet`` only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import stats as sstats


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_corpus(path: Path, target_col: str, n_features: int,
                feature_prefix: str = "f") -> pd.DataFrame:
    """Load one corpus parquet / CSV into the canonical-in-memory
    schema:

        ref_basename : str
        target       : float (the active --target-col value)
        feat_0..feat_N-1 : float

    The canonical-2026-05-21 parquets carry feature columns ``f0..f371``
    (or ``f0..f299`` for the LARGE variant). Older CSVs may carry
    ``f0..f227`` (228-feat base) or ``f0..f299`` (300-feat extended).

    All rows are read up-front; downstream code subsets via boolean
    masks. If the parquet doesn't carry ``target_col`` (or that column
    is entirely null), raises SystemExit — silent training against
    nulls is forbidden per zensim/CLAUDE.md ("Operational rules" §1).
    """
    if path.suffix in (".parquet", ".pq"):
        df = pq.read_table(str(path)).to_pandas()
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise SystemExit(f"unsupported corpus suffix: {path}")
    if "ref_basename" not in df.columns:
        # Older CSV format used ``image_basename``.
        for alt in ("image_basename", "ref_name", "stem"):
            if alt in df.columns:
                df = df.rename(columns={alt: "ref_basename"})
                break
        else:
            raise SystemExit(
                f"{path}: no ref_basename / image_basename / ref_name / stem")
    if target_col not in df.columns:
        raise SystemExit(
            f"{path}: target column {target_col!r} not found (got "
            f"{sorted([c for c in df.columns if not c.startswith(feature_prefix)])[:20]})")
    if df[target_col].isna().all():
        raise SystemExit(
            f"{path}: target column {target_col!r} is entirely null — "
            f"this parquet does not carry that target. Per zensim/"
            f"CLAUDE.md the trainer must NOT silently learn to predict "
            f"nulls.")
    feat_cols = [f"{feature_prefix}{i}" for i in range(n_features)]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path}: missing {len(missing)} feature cols (expected "
            f"{feature_prefix}0..{feature_prefix}{n_features - 1}); "
            f"first missing: {missing[:5]}")
    keep_cols = ["ref_basename", target_col] + feat_cols
    # Preserve additional target columns for multi-target experiments.
    extra_targets = [c for c in (
        "cvvdp_score", "cvvdp_log_norm", "iwssim", "iwssim_log_norm",
        "ssim2_gpu", "ssim2_log_norm", "pjnd_target",
    ) if c in df.columns]
    keep = keep_cols + [c for c in extra_targets if c not in keep_cols]
    out = df[keep].copy()
    out = out.rename(columns={target_col: "target"})
    # Rename feat cols to canonical ``feat_*`` to match older v_next code.
    out = out.rename(columns={
        c: f"feat_{i}" for i, c in enumerate(feat_cols)})
    return out


def attach_zenanalyze_features(df: pd.DataFrame, sidecar_path: Path,
                                feature_names: list[str]) -> pd.DataFrame:
    """Append N zenanalyze features (e.g. ``dct_compressibility_y``) to
    the ``feat_*`` columns, keyed by ``ref_basename`` joined to the
    sidecar's ``stem`` column.

    Port from ``zensim-validate/src/main.rs::expand_with_zenanalyze_features``
    (v07-e1-ablation branch). The Rust impl reads TSV, builds a
    ``BTreeMap<String, [f32; N]>`` keyed by ``stem``, and appends to
    each row.

    Behaviour:
      - The sidecar may be ``.tsv``, ``.csv``, ``.parquet``, or
        ``.pq``. It MUST carry a ``stem`` column (or ``ref_basename``)
        plus all of ``feature_names``.
      - Joins are LEFT-joins keyed by ``ref_basename`` → sidecar stem.
        Rows whose ``ref_basename`` is not in the sidecar get NaN in
        the appended feature columns. The caller is responsible for
        deciding whether to drop those rows (the standardize step
        below replaces post-z-norm NaNs with 0, so they survive as
        "neutral" rows but lose signal from the appended features).
      - The appended features are renamed to ``feat_{n_prior + i}``
        so they extend the original feature index.
      - Logs how many rows joined vs how many got NaN.

    Returns the augmented dataframe. Does NOT mutate the input.
    """
    if sidecar_path.suffix in (".parquet", ".pq"):
        sc = pq.read_table(str(sidecar_path)).to_pandas()
    elif sidecar_path.suffix == ".tsv":
        sc = pd.read_csv(sidecar_path, sep="\t")
    elif sidecar_path.suffix == ".csv":
        sc = pd.read_csv(sidecar_path)
    else:
        raise SystemExit(f"unsupported sidecar suffix: {sidecar_path}")
    # Accept ``stem`` or ``ref_basename`` as the join key.
    key = "ref_basename" if "ref_basename" in sc.columns else "stem"
    if key not in sc.columns:
        raise SystemExit(
            f"{sidecar_path}: sidecar needs a 'stem' or 'ref_basename' "
            f"column (got {list(sc.columns)[:10]})")
    missing = [c for c in feature_names if c not in sc.columns]
    if missing:
        raise SystemExit(
            f"{sidecar_path}: missing feature(s) {missing[:5]}")
    n_prior = sum(1 for c in df.columns if c.startswith("feat_"))
    # Rename sidecar feature cols to feat_{n_prior+i}.
    rename_map = {name: f"feat_{n_prior + i}"
                  for i, name in enumerate(feature_names)}
    sub = sc[[key] + feature_names].rename(
        columns={key: "ref_basename", **rename_map})
    merged = df.merge(sub, on="ref_basename", how="left")
    n_nan_per_row = merged[[
        f"feat_{n_prior + i}" for i in range(len(feature_names))]]\
        .isna().any(axis=1).sum()
    print(f"  attach_zenanalyze_features: appended "
          f"{len(feature_names)} feature(s) to df "
          f"({n_prior} → {n_prior + len(feature_names)}); "
          f"{n_nan_per_row:,} / {len(merged):,} rows had NaN in the "
          f"appended block (joined on ref_basename)", flush=True)
    return merged


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def ranknet_loss(pred: torch.Tensor, target: torch.Tensor,
                  groups: torch.Tensor, max_total_pairs: int = 4096,
                  row_sample_weights: torch.Tensor | None = None,
                  ) -> torch.Tensor:
    """Vectorized pair-uniform RankNet loss within same-group pairs.

    Same implementation as the v_next trainer (10-30× faster than the
    per-group Python loop). When ``row_sample_weights`` is supplied,
    pair endpoints are sampled via ``torch.multinomial(w)`` instead of
    uniform ``torch.randint`` — this is the lever for low-band
    oversampling (port 4).
    """
    device = pred.device
    n = pred.size(0)
    if n < 2:
        return torch.zeros((), device=device)
    num_unique = int(groups.unique().numel())
    n_candidates = min(max_total_pairs * max(num_unique, 1) * 2,
                       max_total_pairs * 256)
    n_candidates = min(n_candidates, 1_048_576)
    if row_sample_weights is not None and (row_sample_weights != 1.0).any():
        w = row_sample_weights.clamp_min(0.0)
        if w.sum() > 0:
            i = torch.multinomial(w, n_candidates, replacement=True)
            j = torch.multinomial(w, n_candidates, replacement=True)
        else:
            i = torch.randint(0, n, (n_candidates,), device=device)
            j = torch.randint(0, n, (n_candidates,), device=device)
    else:
        i = torch.randint(0, n, (n_candidates,), device=device)
        j = torch.randint(0, n, (n_candidates,), device=device)
    keep = (groups[i] == groups[j]) & (i < j)
    i, j = i[keep], j[keep]
    if i.numel() == 0:
        return torch.zeros((), device=device)
    if i.numel() > max_total_pairs:
        sel = torch.randperm(i.numel(), device=device)[:max_total_pairs]
        i, j = i[sel], j[sel]
    p_diff = pred[i] - pred[j]
    t_diff = target[i] - target[j]
    nz = t_diff != 0
    if nz.sum() == 0:
        return torch.zeros((), device=device)
    sign = torch.sign(t_diff[nz])
    return torch.nn.functional.softplus(-sign * p_diff[nz]).mean()


def magnitude_match_term(pred: torch.Tensor, target: torch.Tensor,
                          lam: float, alpha: float) -> torch.Tensor:
    """Magnitude-matching penalty (port 3, from v04-mlp).

    Form: ``λ · mean((|α · target| − |pred|)²)``.

    Encourages the model to keep predicted magnitudes proportional to
    target magnitudes — alongside the RankNet loss that only enforces
    relative ordering, this anchors absolute scale so the runtime
    output isn't free to drift.

    α controls slope; λ the loss weight. A typical value (from v04-mlp
    recipes): α=30, λ=0.1 when target is on a 0..1 scale (CID22 MCOS).
    For 0..100 scale, scale α down accordingly (α=0.3, λ=0.1).

    Per-row absolute value (not batch-mean) preserves the
    intent of the v04-mlp Rust impl: each row contributes its own
    magnitude error, then averaged.
    """
    if lam <= 0.0:
        return torch.zeros((), device=pred.device)
    return lam * torch.mean(
        (pred.abs() - alpha * target.abs()) ** 2)


def build_low_band_sample_weights(target: torch.Tensor,
                                   oversample_ratio: float,
                                   low_band_cutoff: float = 50.0
                                   ) -> torch.Tensor:
    """Per-row sample-weight builder for the low-band oversample
    sampler (port 4).

    Form: rows whose ``target < low_band_cutoff`` get weight
    ``oversample_ratio``; rows with ``target >= low_band_cutoff`` get
    weight 1.0. The result is passed to ``torch.multinomial`` in the
    RankNet pair sampler.

    With ``oversample_ratio=4.0`` and a corpus where 10% of rows are
    low-band, low-band rows go from contributing ~10% of pairs to
    ``4×10/(4×10 + 90) = 31%`` — a 3× lift in low-band ranking signal.

    Default cutoff 50 matches the V0_4 "B0+B1 = score<50" definition
    used in train_v_next_mlp.py's ``--low-q-boost`` flag.
    """
    if oversample_ratio <= 1.0:
        return torch.ones_like(target)
    return torch.where(
        target < low_band_cutoff,
        torch.full_like(target, float(oversample_ratio)),
        torch.ones_like(target))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ZensimMLP(torch.nn.Module):
    """The canonical zensim metric architecture:
        n_in → [hidden_i, LeakyReLU(0.01)]* → 1

    Default hidden=[128]. For FiLM/MoE variants, subclass and override
    ``forward()`` — but per zensim/CLAUDE.md the architectural
    branches belong in the Rust trainer for ship bakes; this Python
    trainer stays single-MLP for research / ablation.
    """

    def __init__(self, n_in: int, hidden: list[int], init: str = "kaiming"):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
        if init == "glorot":
            # Glorot/Xavier-normal init matches the Rust trainer's
            # ``std = sqrt(2 / (n_in + n_out))``.
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    fan_in, fan_out = m.weight.shape[1], m.weight.shape[0]
                    std = (2.0 / (fan_in + fan_out)) ** 0.5
                    torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Standardize
# ---------------------------------------------------------------------------

def standardize(X_train: np.ndarray, X_val_list: list[np.ndarray]
                ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Per-feature Z-score on the training split, applied to all val
    splits. Replaces NaN/inf with 0 post-standardization (some
    features are degenerate at certain content classes and don't
    carry signal). Returns standardized arrays + (mean, std)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    Xt = (X_train - mean) / std
    Xt[~np.isfinite(Xt)] = 0.0
    Xv_list = []
    for X_val in X_val_list:
        Xv = (X_val - mean) / std
        Xv[~np.isfinite(Xv)] = 0.0
        Xv_list.append(Xv.astype(np.float32))
    return (Xt.astype(np.float32), Xv_list,
            mean.astype(np.float32), std.astype(np.float32))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All settings consumed by ``train_loop`` (port 1)."""
    target_col: str = "human_score"
    feature_prefix: str = "f"
    n_features: int = 372
    hidden: tuple[int, ...] = (128,)
    epochs: int = 100
    batch_size: int = 16384
    lr: float = 1e-3
    weight_decay: float = 1e-5
    rank_weight: float = 0.5
    tv_weight: float = 0.0
    magnitude_match_lambda: float = 0.0
    magnitude_match_alpha: float = 30.0
    low_band_oversample: float = 1.0
    low_band_cutoff: float = 50.0
    seed: int = 0
    val_policy: str = "min"  # ``min`` or ``mean``
    loss_kind: str = "mse_rank"  # ``mse`` | ``ranknet`` | ``mse_rank``
    init: str = "kaiming"
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"


def srocc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    rho, _ = sstats.spearmanr(a[finite], b[finite])
    return float(rho)


def train_loop(cfg: TrainConfig,
                train_dfs: dict[str, tuple[pd.DataFrame, float]],
                val_dfs: dict[str, pd.DataFrame],
                device: torch.device) -> tuple[ZensimMLP, dict, dict]:
    """End-to-end training loop (port 1).

    Inputs:
      cfg         — TrainConfig knobs.
      train_dfs   — {dataset_name: (df, train_weight)} for training corpora.
                     Each df has columns ``ref_basename, target,
                     feat_0..feat_{N-1}``.
      val_dfs     — {dataset_name: df} for validation corpora (no weight
                     — val never contributes to the loss).
      device      — torch.device.

    Returns:
      (best_model, metrics, scaler_dict)

    Where ``metrics`` includes ``best_epoch, best_sel_metric,
    per_dataset_val_srocc, train_secs``; ``scaler_dict`` carries
    ``mean, std`` arrays for the bake step.

    Selection metric per ``cfg.val_policy``: ``min`` selects on the
    worst per-dataset val SROCC; ``mean`` selects on the mean.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    feat_cols = [f"feat_{i}" for i in range(cfg.n_features)]

    # Materialize train arrays (per-dataset, then concat).
    Xt_blocks, yt_blocks, wt_blocks, gt_blocks, dst_blocks = [], [], [], [], []
    group_offset = 0
    for name, (df, train_w) in train_dfs.items():
        # Per-dataset group ids = factorized ref_basename, offset to be
        # globally unique across datasets.
        codes, _ = pd.factorize(df["ref_basename"], sort=False)
        codes = codes.astype(np.int64) + group_offset
        group_offset = int(codes.max()) + 1 if codes.size > 0 else group_offset
        Xb = df[feat_cols].to_numpy(dtype=np.float32)
        yb = df["target"].to_numpy(dtype=np.float32)
        wb = np.full(len(df), train_w, dtype=np.float32)
        # Drop rows with NaN target.
        finite_y = np.isfinite(yb)
        finite_x = np.isfinite(Xb).all(axis=1)
        keep = finite_y & finite_x
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"  {name}: dropped {n_drop:,} rows with NaN/inf",
                  flush=True)
        Xt_blocks.append(Xb[keep])
        yt_blocks.append(yb[keep])
        wt_blocks.append(wb[keep])
        gt_blocks.append(codes[keep])
        dst_blocks.append(np.full(int(keep.sum()), name, dtype=object))
    X_train = np.concatenate(Xt_blocks, axis=0) if Xt_blocks else np.zeros((0, cfg.n_features), dtype=np.float32)
    y_train = np.concatenate(yt_blocks, axis=0) if yt_blocks else np.zeros((0,), dtype=np.float32)
    w_train = np.concatenate(wt_blocks, axis=0) if wt_blocks else np.zeros((0,), dtype=np.float32)
    g_train = np.concatenate(gt_blocks, axis=0) if gt_blocks else np.zeros((0,), dtype=np.int64)
    dst_train = np.concatenate(dst_blocks, axis=0) if dst_blocks else np.zeros((0,), dtype=object)

    # Materialize val arrays (per-dataset, kept separate for per-dataset SROCC).
    val_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    Xv_inputs = []
    val_order = list(val_dfs.keys())
    for name in val_order:
        df = val_dfs[name]
        Xb = df[feat_cols].to_numpy(dtype=np.float32)
        yb = df["target"].to_numpy(dtype=np.float32)
        finite = np.isfinite(yb) & np.isfinite(Xb).all(axis=1)
        Xv_inputs.append(Xb[finite])
        val_arrays[name] = (Xb[finite], yb[finite])

    # Standardize on train; apply to all vals.
    X_train_z, X_vals_z_list, scaler_mean, scaler_std = standardize(
        X_train, [val_arrays[n][0] for n in val_order])
    val_arrays_z = {n: (X_vals_z_list[i], val_arrays[n][1])
                    for i, n in enumerate(val_order)}

    print(f"Train: {len(X_train):,} rows over {len(train_dfs)} datasets, "
          f"{len(set(g_train.tolist())):,} groups", flush=True)
    print(f"Val: {sum(v[1].size for v in val_arrays.values()):,} rows "
          f"over {len(val_arrays)} datasets", flush=True)

    n_in = X_train_z.shape[1]
    model = ZensimMLP(n_in, list(cfg.hidden), init=cfg.init).to(device)
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    else:
        sched = None

    Xt = torch.from_numpy(X_train_z).to(device)
    yt = torch.from_numpy(y_train).to(device)
    wt = torch.from_numpy(w_train).to(device)
    gt = torch.from_numpy(g_train).to(device)
    val_tensors = {n: (torch.from_numpy(arr).to(device), y)
                   for n, (arr, y) in val_arrays_z.items()}

    # Pre-compute low-band sample weights once (target is fixed).
    ranknet_weights = build_low_band_sample_weights(
        yt, cfg.low_band_oversample, cfg.low_band_cutoff
    ) if cfg.low_band_oversample > 1.0 else None

    best = {"epoch": -1, "sel_metric": -2.0, "state": None,
            "per_ds_srocc": {}}
    n = Xt.size(0)
    bs = min(cfg.batch_size, n)
    t0 = time.time()

    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            x = Xt[idx]
            y = yt[idx]
            w = wt[idx]
            g = gt[idx]
            pred = model(x)
            # Per-row weighted MSE (synthetic = 1.0, KADID/TID = 0.3).
            mse = ((pred - y) ** 2 * w).sum() / w.sum().clamp_min(1e-6)
            if cfg.loss_kind == "mse":
                loss = mse
            elif cfg.loss_kind == "ranknet":
                # Subset of the row sample weights for this batch.
                rw = ranknet_weights[idx] if ranknet_weights is not None else None
                loss = ranknet_loss(pred, y, g, row_sample_weights=rw)
            elif cfg.loss_kind == "mse_rank":
                rw = ranknet_weights[idx] if ranknet_weights is not None else None
                rk = ranknet_loss(pred, y, g, row_sample_weights=rw)
                loss = mse + cfg.rank_weight * rk
            else:
                raise ValueError(cfg.loss_kind)
            if cfg.magnitude_match_lambda > 0.0:
                mm = magnitude_match_term(
                    pred, y, cfg.magnitude_match_lambda,
                    cfg.magnitude_match_alpha)
                loss = loss + mm
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        if sched is not None:
            sched.step()

        # Val.
        model.eval()
        per_ds_srocc: dict[str, float] = {}
        with torch.no_grad():
            for name, (Xv, yv) in val_tensors.items():
                pv = model(Xv).detach().cpu().numpy()
                per_ds_srocc[name] = srocc(pv, yv)
        if cfg.val_policy == "min":
            vals = [v for v in per_ds_srocc.values() if not math.isnan(v)]
            sel = float(min(vals)) if vals else float("-inf")
        else:
            vals = [v for v in per_ds_srocc.values() if not math.isnan(v)]
            sel = float(np.mean(vals)) if vals else float("-inf")
        if sel > best["sel_metric"]:
            best = {"epoch": ep, "sel_metric": sel,
                    "per_ds_srocc": dict(per_ds_srocc),
                    "state": {k: v.detach().clone().cpu()
                              for k, v in model.state_dict().items()}}
        per_ds_str = "  ".join(f"{k}={v:+.4f}" for k, v in per_ds_srocc.items())
        print(f"  epoch {ep:3d}  train_loss={ep_loss/max(1,n_batches):.4f}  "
              f"sel={sel:+.4f}  {per_ds_str}", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    train_secs = time.time() - t0
    metrics = {
        "best_epoch": best["epoch"],
        "best_sel_metric": best["sel_metric"],
        "per_ds_val_srocc": best["per_ds_srocc"],
        "train_secs": round(train_secs, 1),
        "n_train": int(len(X_train)),
        "n_val_total": sum(v[1].size for v in val_arrays.values()),
        "n_features": n_in,
    }
    scaler = {"mean": scaler_mean, "std": scaler_std}
    return model, metrics, scaler


# ---------------------------------------------------------------------------
# Bake — ZNPR v3 via the JSON pipeline
# ---------------------------------------------------------------------------

def _find_bake_bin() -> Path:
    """Locate the v3-emitting ``zenpredict-bake`` binary.

    Per zensim/CLAUDE.md "ZNPR v2 PROHIBITED" (2026-05-15) we MUST
    write v3. The local ``~/work/zen/zenanalyze/target/release/
    zenpredict-bake`` (built from current main) emits v3. The
    published 0.1.0 binary writes v2 — never use it.
    """
    candidates = [
        Path.home() / "work/zen/zenanalyze/target/release/zenpredict-bake",
        Path.home() / "work/zen/zenanalyze--recovery-phase3/target/release/zenpredict-bake",
        Path.home() / ".cargo/bin/zenpredict-bake",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def bake_to_znpr(model: ZensimMLP, scaler_mean: np.ndarray,
                  scaler_std: np.ndarray, n_inputs: int,
                  out_path: Path, *,
                  flip_output: bool = True,
                  metadata: dict[str, str] | None = None,
                  bake_bin: Path | None = None,
                  zerobias_tau: float = 0.0,
                  compressed: bool = False,
                  optimize: bool = False,
                  keep_json: bool = False) -> None:
    """Bake the trained model to ZNPR v3 via the
    ``zenpredict-bake <input.json> <output.bin>`` CLI.

    Mirrors ``zensim/scripts/v_next/bake_to_znpr.py`` but operates on
    an in-memory model + scaler rather than reloading from disk.

    Per zensim/CLAUDE.md:
      - ZNPR v2 is BANNED — uses the v3 path.
      - JSON pipeline is mandated — never emit ZNPR wire format from
        Python directly.
      - ``flip_output=True`` rewrites the final linear layer to emit
        ``100 − W·x − b`` so the bake output is a distance score
        (low = high quality). zensim's score-mapping then applies
        ``100 − a·d^b`` with ``a=1, b=1`` to invert back to quality
        scale at runtime. Use ``flip_output=False`` when the target
        was MOS-shaped (high = high quality) and the bake is meant to
        be consumed score-shaped.

    metadata dict gets stamped into the ``zentrain.*`` metadata block
    of the bake so ``zenpredict-inspect`` can surface training
    provenance at debug time.
    """
    sd = model.state_dict()
    keys = sorted(sd.keys(),
                   key=lambda k: int(k.split(".")[1]))
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    cur_w = None
    for k in keys:
        if k.endswith(".weight"):
            cur_w = sd[k]
        elif k.endswith(".bias"):
            assert cur_w is not None
            pairs.append((cur_w, sd[k]))
            cur_w = None
    last = len(pairs) - 1
    layers = []
    for i, (W, b) in enumerate(pairs):
        if i == last and flip_output:
            W = -W
            b = 100.0 - b
        out_dim, in_dim = W.shape
        activation = "identity" if i == last else "leakyrelu"
        layers.append({
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "activation": activation,
            "dtype": "f32",
            "weights": W.t().contiguous().flatten().tolist(),
            "biases": b.flatten().tolist(),
        })
    if scaler_mean.shape[0] != n_inputs:
        raise SystemExit(
            f"scaler_mean has {scaler_mean.shape[0]} entries, model "
            f"expects {n_inputs}")
    meta_entries = []
    if metadata:
        for k, v in metadata.items():
            meta_entries.append(
                {"key": str(k), "type": "utf8", "text": str(v)})
    meta_entries.append({"key": "zensim.profile", "type": "utf8",
                         "text": "zensim-preview-v0.5-recovery-phase3"})
    req = {
        "schema_hash": 0,
        "flags": 0,
        "scaler_mean": scaler_mean.astype(np.float32).tolist(),
        "scaler_scale": scaler_std.astype(np.float32).tolist(),
        "layers": layers,
        "feature_bounds": [],
        "metadata": meta_entries,
        "output_specs": [],
        "discrete_sets": [],
        "sparse_overrides": [],
    }
    if zerobias_tau > 0.0:
        req["zerobias_tau"] = float(zerobias_tau)
    if compressed:
        req["compressed"] = True
    if optimize:
        req["optimize"] = True

    bake_bin = Path(bake_bin) if bake_bin else _find_bake_bin()
    if not bake_bin.is_file():
        raise SystemExit(
            f"zenpredict-bake not found at {bake_bin}; run "
            f"`cargo build --release -p zenpredict-bake` in zenanalyze first.")

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_json:
        json_out = out_path.with_suffix(".bake.json")
    else:
        json_fd, json_out_str = tempfile.mkstemp(suffix=".bake.json",
                                                  text=True)
        os.close(json_fd)
        json_out = Path(json_out_str)
    json_out.write_text(json.dumps(req))
    print(f"  bake: BakeRequestJson at {json_out} "
          f"({json_out.stat().st_size:,} bytes); calling {bake_bin}",
          flush=True)
    res = subprocess.run(
        [str(bake_bin), str(json_out), str(out_path)],
        capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"zenpredict-bake exit {res.returncode}")
    if not keep_json:
        try:
            json_out.unlink()
        except OSError:
            pass
    print(f"  bake: wrote {out_path} ({out_path.stat().st_size:,} bytes)",
          flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_corpus_spec(spec: str, *, with_weight: bool = False
                       ) -> tuple[str, Path, float]:
    """Parse a ``name:path[:weight]`` corpus argument. ``with_weight``
    requires the third field for train corpora."""
    parts = spec.split(":")
    if with_weight:
        if len(parts) == 3:
            return parts[0], Path(parts[1]), float(parts[2])
        raise SystemExit(
            f"--train spec expects NAME:PATH:WEIGHT, got {spec!r}")
    if len(parts) == 2:
        return parts[0], Path(parts[1]), 1.0
    if len(parts) == 3:
        return parts[0], Path(parts[1]), float(parts[2])
    raise SystemExit(
        f"--val spec expects NAME:PATH (or NAME:PATH:WEIGHT), got {spec!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Zensim metric MLP trainer (recovery-phase-3 port).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--train", action="append", required=True,
                    metavar="NAME:PATH:WEIGHT",
                    help="Training corpus parquet/CSV with feature "
                         "columns f0..fN-1 + a target column. WEIGHT "
                         "scales the per-row MSE (synthetic = 1.0, "
                         "KADID/TID human-MOS = 0.3 per V0_4 recipe).")
    ap.add_argument("--corpus-target-scale", action="append", default=[],
                    metavar="NAME:FACTOR",
                    help="Recovery-champion port: multiply --target-col "
                         "by FACTOR for the named training corpus before "
                         "loss computation. Used to harmonize corpora "
                         "carrying the same column on incompatible "
                         "scales (e.g. konjnd's human_score is [-65, 96] "
                         "while safesyn/kadid/tid are [0, 1]; passing "
                         "konjnd:0.01 puts konjnd on a comparable scale).")
    ap.add_argument("--val", action="append", default=[],
                    metavar="NAME:PATH",
                    help="Validation corpus parquet/CSV. Selection "
                         "metric is computed per dataset; "
                         "--val-policy=min picks the epoch maximizing "
                         "the worst per-dataset SROCC.")
    ap.add_argument("--target-col", default="human_score",
                    help="Target column to predict. canonical-2026-05-21 "
                         "parquets carry many targets; human_score is "
                         "the per-corpus native anchor.")
    ap.add_argument("--n-features", type=int, default=372,
                    help="Number of feature columns f0..fN-1. "
                         "canonical-2026-05-21 = 372; 300 for the "
                         "LARGE variant; 228 for legacy basic features.")
    ap.add_argument("--feature-prefix", default="f")
    ap.add_argument("--hidden", default="128",
                    help="Comma-separated hidden widths.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--rank-weight", type=float, default=0.5)
    ap.add_argument("--tv-weight", type=float, default=0.0,
                    help="Per-curve monotonicity TV regularizer "
                         "(currently a no-op in the recovery-phase-3 "
                         "port; planned port from v_next).")
    ap.add_argument("--magnitude-match-lambda", type=float, default=0.0,
                    help="Port 3: weight on the magnitude-matching "
                         "loss term λ · mean((|α·target| − |pred|)²).")
    ap.add_argument("--magnitude-match-alpha", type=float, default=30.0,
                    help="Port 3: slope α for the magnitude-matching "
                         "term. α=30 when target on [0,1], α=0.3 on "
                         "[0,100].")
    ap.add_argument("--low-band-oversample", type=float, default=1.0,
                    help="Port 4: RankNet sampler bias. Rows with "
                         "target < --low-band-cutoff get this much "
                         "sampling weight (default 1.0 = uniform).")
    ap.add_argument("--low-band-cutoff", type=float, default=50.0,
                    help="Port 4: threshold for low-band classification. "
                         "Default 50 matches V0_4 'B0+B1 = score<50'.")
    ap.add_argument("--val-policy", default="min", choices=["min", "mean"])
    ap.add_argument("--loss", default="mse_rank",
                    choices=["mse", "ranknet", "mse_rank"])
    ap.add_argument("--init", default="kaiming", choices=["kaiming", "glorot"])
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adam"])
    ap.add_argument("--lr-schedule", default="cosine",
                    choices=["constant", "cosine"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/recovery-phase3/runs",
                    help="Where to write run artifacts.")
    ap.add_argument("--tag", default="recovery_phase3",
                    help="Run-dir suffix.")
    ap.add_argument("--zenanalyze-sidecar",
                    help="Port 2: TSV / Parquet with additional features. "
                         "Format: PATH:FEAT1,FEAT2,... — features "
                         "named FEAT1,FEAT2,... in the sidecar are "
                         "joined by ref_basename and appended after "
                         "f0..fN-1.")
    ap.add_argument("--bake-out",
                    help="If set, after training also bake the model "
                         "to a ZNPR v3 .bin at this path.")
    ap.add_argument("--no-flip-output", action="store_true",
                    help="Skip the final-layer 100−x flip in the bake. "
                         "Default: ON (target is on 0..100 quality "
                         "scale, bake flipped to distance scale).")
    ap.add_argument("--bake-zerobias-tau", type=float, default=0.0)
    ap.add_argument("--bake-compress", action="store_true")
    ap.add_argument("--bake-optimize", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_specs = [_parse_corpus_spec(s, with_weight=True) for s in args.train]
    val_specs = [_parse_corpus_spec(s, with_weight=False) for s in args.val]

    # Parse --corpus-target-scale NAME:FACTOR entries.
    corpus_scales: dict[str, float] = {}
    for entry in args.corpus_target_scale:
        if ":" not in entry:
            raise SystemExit(f"--corpus-target-scale: bad spec {entry!r} "
                             "(want NAME:FACTOR, e.g. konjnd:0.01)")
        name, factor_str = entry.split(":", 1)
        corpus_scales[name] = float(factor_str)

    print(f"Loading {len(train_specs)} train corpora + "
          f"{len(val_specs)} val corpora...", flush=True)
    train_dfs: dict[str, tuple[pd.DataFrame, float]] = {}
    for name, path, w in train_specs:
        df = load_corpus(path, args.target_col, args.n_features,
                          feature_prefix=args.feature_prefix)
        if name in corpus_scales:
            scale = corpus_scales[name]
            df["target"] = df["target"] * scale
            print(f"  train/{name}: {len(df):,} rows, weight={w}, "
                  f"target_scale={scale}", flush=True)
        else:
            print(f"  train/{name}: {len(df):,} rows, weight={w}", flush=True)
        train_dfs[name] = (df, w)
    val_dfs: dict[str, pd.DataFrame] = {}
    for name, path, _ in val_specs:
        df = load_corpus(path, args.target_col, args.n_features,
                          feature_prefix=args.feature_prefix)
        print(f"  val/{name}: {len(df):,} rows", flush=True)
        val_dfs[name] = df

    # Port 2 — zenanalyze sidecar.
    if args.zenanalyze_sidecar:
        spec_path, feat_str = args.zenanalyze_sidecar.split(":", 1)
        feat_names = [s.strip() for s in feat_str.split(",") if s.strip()]
        sidecar_path = Path(spec_path)
        print(f"Port 2: attaching {len(feat_names)} zenanalyze "
              f"features from {sidecar_path}", flush=True)
        for name in list(train_dfs.keys()):
            df, w = train_dfs[name]
            train_dfs[name] = (attach_zenanalyze_features(
                df, sidecar_path, feat_names), w)
        for name in list(val_dfs.keys()):
            val_dfs[name] = attach_zenanalyze_features(
                val_dfs[name], sidecar_path, feat_names)
        n_features = args.n_features + len(feat_names)
    else:
        n_features = args.n_features

    cfg = TrainConfig(
        target_col=args.target_col,
        feature_prefix=args.feature_prefix,
        n_features=n_features,
        hidden=tuple(int(h) for h in args.hidden.split(",")),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rank_weight=args.rank_weight,
        tv_weight=args.tv_weight,
        magnitude_match_lambda=args.magnitude_match_lambda,
        magnitude_match_alpha=args.magnitude_match_alpha,
        low_band_oversample=args.low_band_oversample,
        low_band_cutoff=args.low_band_cutoff,
        seed=args.seed,
        val_policy=args.val_policy,
        loss_kind=args.loss,
        init=args.init,
        optimizer=args.optimizer,
        lr_schedule=args.lr_schedule,
    )
    print(f"Config: {cfg}", flush=True)

    model, metrics, scaler = train_loop(cfg, train_dfs, val_dfs, device)
    print(f"\nBest: epoch {metrics['best_epoch']} "
          f"sel_metric={metrics['best_sel_metric']:+.4f}; "
          f"per-ds val SROCC = "
          f"{ {k: f'{v:+.4f}' for k,v in metrics['per_ds_val_srocc'].items()} }",
          flush=True)

    # Persist run artifacts.
    ts = time.strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.out_dir) / f"{ts}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.savez(run_dir / "scaler.npz",
              mean=scaler["mean"], std=scaler["std"])
    meta_blob = {"config": asdict(cfg), "metrics": metrics,
                  "argv": sys.argv,
                  "target_col": args.target_col,
                  "n_features": n_features,
                  "train_specs": [(n, str(p), w) for n, p, w in train_specs],
                  "val_specs": [(n, str(p)) for n, p, _ in val_specs]}
    (run_dir / "meta.json").write_text(json.dumps(meta_blob, indent=2,
                                                    default=str))
    # Predictions parquet on val.
    model.eval()
    feat_cols = [f"feat_{i}" for i in range(n_features)]
    pred_frames = []
    with torch.no_grad():
        for name, df in val_dfs.items():
            X = df[feat_cols].to_numpy(dtype=np.float32)
            # Re-apply the trained scaler.
            mean = scaler["mean"]
            std = scaler["std"]
            Xz = (X - mean) / std
            Xz[~np.isfinite(Xz)] = 0.0
            pv = model(torch.from_numpy(Xz.astype(np.float32)).to(device)
                        ).cpu().numpy()
            sub = df[["ref_basename", "target"]].copy()
            sub["pred"] = pv
            sub["dataset"] = name
            pred_frames.append(sub)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df.to_parquet(run_dir / "predictions_val.parquet",
                        compression="zstd", compression_level=9)
    print(f"Wrote {run_dir}/")

    if args.bake_out:
        print(f"\nBaking to ZNPR v3 at {args.bake_out}...", flush=True)
        flip = not args.no_flip_output
        bake_to_znpr(model, scaler["mean"], scaler["std"], n_features,
                      Path(args.bake_out),
                      flip_output=flip,
                      metadata={"train.tag": args.tag,
                                "train.seed": str(args.seed),
                                "train.epochs": str(args.epochs),
                                "train.target_col": args.target_col,
                                "train.n_features": str(n_features),
                                "train.hidden": args.hidden,
                                "train.best_sel_metric":
                                    f"{metrics['best_sel_metric']:.4f}",
                                "train.recipe": "recovery-phase-3"},
                      zerobias_tau=args.bake_zerobias_tau,
                      compressed=args.bake_compress,
                      optimize=args.bake_optimize)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
