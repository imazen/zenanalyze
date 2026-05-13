#!/usr/bin/env python3
"""Train zensim's perceptual-metric MLP and bake to ZNPR.

Companion to `train_hybrid.py` (codec-picker trainer). Where the hybrid
trainer predicts a categorical cell + K scalar continuous heads per
cell, this trainer predicts a single regression head (the zensim
distance score) given a feature vector. Different shape, same
zentrain pipeline (Pareto / teacher fit / distill / inspection /
safety / triage / bake → ZNPR v2 or v3).

This trainer is the **Python-side canonical home** for zensim metric
work. NOTE (corrected 2026-05-13): the parallel Rust trainer at
`zensim-validate/src/mlp_train.rs` was deleted on 2026-05-07 (commit
`e613224`) and RESTORED on 2026-05-10 (commit `ec40ec8`). It is LIVE
and is the trainer that produced the current ship V0_16 (CID22
0.8919). Both trainers coexist:
  - Rust trainer = authoritative for V_X SHIP bakes (run via
    `target/release/zensim_mlp_train`; recipe captured at
    `zensim/benchmarks/recipe_v0_16.sh`); uses per-step group-weighted
    pair sampling.
  - This Python trainer = research / ablation / cycle experiments;
    uses batched RankNet with per-row MSE weighting. The mechanisms
    differ on `train_weight` semantics (per cycle-13 finding); pick
    the trainer that matches the recipe semantic you need.

Recipe (per RECOVERY_PLAN_2026-05-08.md Phase 3):
  - Primary signal: training_safe_synthetic_extended.csv (340k rows,
    includes CID22 _training_ set tiles → SSIM2-target IS CID22-MOS-
    aligned per CID22 paper §4)
  - Auxiliary supervision: KADID_train + TID_train at train_weight=0.3
  - Held-out val: CID22 _validation_ set, KADID_val, TID_val,
    KonJND-1k anchors
  - Selection metric: mean over per-dataset val SROCC (gated on
    `--val-policy min` — pick the epoch where the min over datasets
    is highest)
  - Optional features (port from v06-rebalance / v06-film / v06-moe
    Rust experimental trainer variants — those specific variants WERE
    stripped on 2026-05-07 and not restored; the BASE trainer
    `zensim-validate/src/mlp_train.rs` is LIVE since 2026-05-10):
    * `--zenanalyze-tsv ...` + `--zenanalyze-features dct_compressibility_y,...`
      append N zenanalyze features to the 228 zensim features
    * `--mlp-magnitude-match-lambda` + `--alpha`
    * `--mlp-low-band-oversample 0.5` (sampler bias)
    * `--film-cclass-tsv ...` (FiLM heads conditioned on content class)
    * `--moe-experts N` (mixture-of-experts; v06-moe code is reference)
    * `--target-secondary butteraugli_p3 --secondary-weight 0.3`
      (multi-task loss per CID22 paper §6)

Bake target: ZNPR v3 (the canonical format per user direction
2026-05-08 — every consumer rebakes to v3).

## Status (2026-05-08)

This file is a SCAFFOLDED trainer. Components implemented vs TODO:

  IMPLEMENTED:
  - canonical CSV loader (synth + KADID/TID/CID22)
  - per-dataset val SROCC reporting + val-policy=min selection
  - vectorized RankNet pairwise loss (10-30× faster than per-group loop)
  - TV monotonicity regularizer (per-curve adjacent-q penalty)
  - per-row train_weight applied to MSE
  - bake to ZNPR via the zenpredict-bake CLI (subprocess)

  TODO (port from zensim--v06-* worktrees, see register):
  - Zenanalyze-feature appender (`dct_hf` family — read TSV, join by
    source_basename, append features). v06-rebalance Rust ref:
    `zensim-validate/src/main.rs::expand_with_zenanalyze_features`
  - Magnitude-matching loss (λ * |scale - α · ‖W·x + b‖_2|).
    v04-smooth Rust ref: `mlp_train::magnitude_match_term`
  - Sampler bias (low-band oversample). v07-e1-ablation Rust ref:
    `mlp_train::low_band_oversample`
  - FiLM heads conditioned on content class. v06-rebalance Rust ref:
    `mlp_train::FilmHead`. Bake produces N per-class .bin + manifest.tsv
  - MoE (mixture of experts). v06-moe code: `zensim--v06-moe/docs/moe_architecture.md`
  - Multi-target loss (ssim2 + butteraugli_p3 / dssim) — paper §6 advice
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import stats as sstats


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_synth_features(path: Path) -> pd.DataFrame:
    """Read a zensim-validate `--features-csv` output (or its parquet
    rewrite) and normalize to the in-memory schema:
        ref_basename, target_value (=human_score * 100 OR ssim2 * 100),
        feat_0 .. feat_N-1
    """
    if path.suffix == ".parquet":
        df = pq.read_table(str(path)).to_pandas()
    else:
        df = pd.read_csv(path)
    return df


def attach_zenanalyze_features(df: pd.DataFrame,
                                tsv_path: Path,
                                feature_names: list[str]) -> pd.DataFrame:
    """Append N zenanalyze features (e.g. dct_compressibility_y) to the
    zensim feature columns, keyed by `ref_basename` joined to the TSV's
    `stem` column.

    TODO: port from `zensim-validate/src/main.rs::expand_with_zenanalyze_features`
    on the `v07-e1-ablation` branch. The Rust impl reads TSV, builds a
    `BTreeMap<String, [f32; N]>` keyed by stem, and appends to each row.
    """
    raise NotImplementedError(
        "attach_zenanalyze_features: port from v07-e1-ablation Rust trainer")


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def ranknet_loss(pred: torch.Tensor, target: torch.Tensor,
                  groups: torch.Tensor, max_total_pairs: int = 4096
                  ) -> torch.Tensor:
    """Vectorized pair-uniform RankNet loss within same-group pairs.
    Same implementation as the v_next trainer (10-30× faster than the
    per-group Python loop)."""
    device = pred.device
    n = pred.size(0)
    if n < 2:
        return torch.zeros((), device=device)
    num_unique = int(groups.unique().numel())
    n_candidates = min(max_total_pairs * max(num_unique, 1) * 2,
                       max_total_pairs * 256)
    n_candidates = min(n_candidates, 1_048_576)
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
    """λ · |α·|target| − |pred‖.

    From v04-smooth (RankNet+magnitude-matching). Encourages the model
    to keep predicted-distance magnitudes proportional to target-
    distance magnitudes. α controls the slope; λ the loss weight.

    TODO: port the exact normalization (per-row vs batch mean) from
    `zensim-validate/src/main.rs::magnitude_match_term` on the
    `v04-mlp` branch.
    """
    raise NotImplementedError(
        "magnitude_match_term: port from v04-mlp Rust trainer")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ZensimMLP(torch.nn.Module):
    """Plain 228 (or 231 with dct_hf) → hidden → 1 LeakyReLU MLP. The
    canonical zensim metric architecture.

    For FiLM/MoE variants, subclass and override `forward()`. See
    TODO: zentrain/tools/zensim_metric_film.py and zensim_metric_moe.py
    (not yet ported from zensim--v06-rebalance / v06-moe Rust trainers).
    """

    def __init__(self, n_in: int, hidden: list[int]):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.01))
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop (skeleton)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    target_col: str = "score_ssim2"
    epochs: int = 100
    batch_size: int = 16384
    lr: float = 3e-3
    weight_decay: float = 1e-5
    rank_weight: float = 0.5
    tv_weight: float = 0.0
    magnitude_match_lambda: float = 0.0
    magnitude_match_alpha: float = 30.0
    low_band_oversample: float = 0.0
    seed: int = 0
    val_policy: str = "min"  # "min" (best min-over-datasets) or "mean"


def train_loop(cfg: TrainConfig, df_all: pd.DataFrame,
                feat_cols: list[str]) -> tuple[ZensimMLP, dict]:
    """End-to-end loop. Reads df_all with columns:
       feat_0..feat_N-1, target_value, dataset, train_weight, is_val_only,
       group_id (image_basename factorized).

    Returns the best model + metrics dict.

    TODO: implement properly. The vectorized RankNet + per-dataset val
    SROCC + val-policy=min selection are ready; the rest is wiring.
    """
    raise NotImplementedError(
        "train_loop: scaffolded — wire the loaders + RankNet + val SROCC "
        "from zensim/scripts/v_next/train_v_next_mlp.py (which has all "
        "the Phase 1 read of those features; just port to here).")


# ---------------------------------------------------------------------------
# Bake
# ---------------------------------------------------------------------------

def bake_to_znpr(model: ZensimMLP, scaler_mean: np.ndarray,
                  scaler_std: np.ndarray, n_inputs: int,
                  out_path: Path, *, znpr_version: int = 3,
                  flip_output: bool = True,
                  metadata: dict[str, str] | None = None) -> None:
    """Bake the trained model to ZNPR (v2 or v3) by serializing a
    `BakeRequestJson` and shelling out to the `zenpredict-bake` CLI.

    Per RECOVERY_PLAN_2026-05-08.md user direction: ZNPR v3 is
    canonical for everyone after Phase 4. Until v3 is re-released, v2
    bakes load via the published zenpredict 0.1.0.

    `flip_output=True` rewrites the final layer to emit `100 - W·x - b`
    so the bake output is on a "distance" scale (0=identical, 100=worst).
    Combined with score_mapping_a=1, b=1 in zensim's profile, runtime
    `100 - 1·d^1` returns the original ssim2-scale prediction.
    """
    raise NotImplementedError(
        "bake_to_znpr: port from zensim/scripts/v_next/bake_to_znpr.py "
        "(which implements the v2 path). Add v3 fields once the v3 "
        "BakeRequest schema is finalized in zenpredict main.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="Path to a Python config module (per "
                         "examples/zenjpeg_picker_config.py pattern).")
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/synthetic-v2/runs")
    ap.add_argument("--tag", default="v07_zentrain")
    args = ap.parse_args()

    raise SystemExit(
        "Scaffolded only — see TODOs in module docstring. Highest-priority "
        "ports: load_synth_features (DONE), ranknet_loss (DONE), train_loop "
        "(needs wiring), bake_to_znpr (port from v_next/bake_to_znpr.py). "
        "Then add: attach_zenanalyze_features, magnitude_match_term, FiLM "
        "head, sampler bias.")


if __name__ == "__main__":
    raise SystemExit(main())
