#!/usr/bin/env python3
"""
Multi-codec shared-trunk MLP picker trainer.

Trains ONE shared trunk MLP on the union of all codecs' Pareto data,
with per-codec output heads (different `n_cells`, different cell
semantics, different scalar axes per codec). Emits one
`*_picker.json` per codec in the existing distill-JSON shape that
`bake_picker.py` consumes — no bake or runtime changes.

# Why this exists

Per-codec piecemeal training (zenjpeg, zenwebp, zenjxl, zenavif each
in isolation) doesn't exploit cross-codec signal. The image features
are codec-agnostic; a shared trunk learns a single feature embedding,
codec-specific heads decode it per codec. Multi-task tabular learning
typically wins ~0-1 pp argmin-acc on the data-poor codecs without
hurting the data-rich ones.

# Architecture

- Each codec's data pipeline runs unchanged via `train_hybrid` helpers
  (`load_pareto`, `load_features`, `build_cell_index`, `build_dataset`,
  `train_teacher_per_cell`, `teacher_predict_all`). The trainer per-
  codec teacher emits soft targets exactly like the single-codec path.
- Trunk input: union of all codecs' feat_cols, plus a presence-mask
  channel per feat_col (1.0 = codec used this feature, 0.0 = missing),
  plus size one-hot (4), `log_px`, `zq_norm`, and a codec one-hot.
- The trunk is `Linear(in → h0) -> LeakyReLU -> Linear(h0 → h1) ->
  LeakyReLU -> ...`. Each codec gets a final `Linear(h_last → n_out_i)`
  head with `n_out_i = n_cells_i * (1 + len(SCALAR_AXES_i))`.
- Training: AdamW, MSE on teacher soft targets, cosine LR schedule
  with warmup, early-stop on a held-out val image-level split. Each
  batch contains a balanced mix from every codec (weighted sampler).

# Output: per-codec bake JSON

To stay 100% compatible with `bake_picker.py`, we produce one JSON
per codec where `layers = [trunk_layer_1, ..., trunk_layer_K,
codec_head]`. That sequence IS a normal MLP — the bake layer
serializer treats it transparently. The per-codec `feat_cols`,
`scaler_mean`, `scaler_scale`, `n_inputs`, etc. all reflect the
*union* schema (since that's the input the trunk consumes), but
the JSON shape matches the existing single-codec bakes.

# Smoke test

    python3 train_multi_codec.py \\
        --codec zenjpeg=examples/zenjpeg_picker_config.py:/home/lilith/work/zen/zenjpeg \\
        --codec zenwebp=examples/zenwebp_picker_config.py:/home/lilith/work/zen/zenwebp \\
        --out-dir benchmarks/multi_codec_2026-05-17 \\
        --hidden 96,48 \\
        --epochs 200

See `benchmarks/multi_codec_bench_2026-05-17.md` for the v1 results.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Need the picker_lib helpers + train_hybrid surface. Both live in
# zentrain/tools/; this script lives there too, so straight import works.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import train_hybrid as TH  # noqa: E402
import _picker_lib as PL  # noqa: E402


# ----------------------------------------------------------------------
# Codec config loading (path-based — no PYTHONPATH dependence)
# ----------------------------------------------------------------------

def _load_codec_config_from_path(config_path: Path) -> Any:
    """Import a codec config module from an explicit path (no PYTHONPATH).
    Mirrors the modules that `train_hybrid.load_codec_config(name)`
    would import.
    """
    spec = importlib.util.spec_from_file_location(
        f"_picker_cfg_{config_path.stem}", config_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CodecConfigBundle:
    """Holds one codec's loaded module + resolved paths.

    Resolves the codec config's relative `PARETO` / `FEATURES` paths
    against a `data_root` (the sibling repo). We never `cd` — the
    trainer reads from absolute paths only.
    """

    def __init__(self, codec_name: str, config_path: Path, data_root: Path):
        self.codec_name = codec_name
        self.config_path = config_path
        self.data_root = data_root
        self.module = _load_codec_config_from_path(config_path)

        def _resolve(p: Path) -> Path:
            return (data_root / p).resolve() if not p.is_absolute() else p

        self.pareto_path = _resolve(Path(self.module.PARETO))
        self.features_path = _resolve(Path(self.module.FEATURES))
        for attr in ("pareto_path", "features_path"):
            p = getattr(self, attr)
            if not p.exists():
                raise FileNotFoundError(
                    f"[{codec_name}] {attr}={p} does not exist "
                    f"(data_root={data_root})"
                )

    # Convenience accessors with documented defaults.
    @property
    def keep_features(self) -> list[str]:
        return list(self.module.KEEP_FEATURES)

    @property
    def zq_targets(self) -> list[int]:
        return list(self.module.ZQ_TARGETS)

    @property
    def categorical_axes(self) -> list[str]:
        return list(getattr(self.module, "CATEGORICAL_AXES",
                            ["color", "sub", "trellis_on", "sa"]))

    @property
    def scalar_axes(self) -> list[str]:
        return list(getattr(self.module, "SCALAR_AXES",
                            ["chroma_scale", "lambda"]))

    @property
    def scalar_sentinels(self) -> dict:
        if hasattr(self.module, "SCALAR_SENTINELS"):
            return dict(self.module.SCALAR_SENTINELS)
        return {"lambda": 0.0} if "lambda" in self.scalar_axes else {}

    @property
    def metric_column(self) -> str:
        return str(getattr(self.module, "METRIC_COLUMN", "zensim"))

    @property
    def metric_direction(self) -> str:
        return str(getattr(self.module, "METRIC_DIRECTION", "higher_better"))

    @property
    def time_column(self) -> str:
        return str(getattr(self.module, "TIME_COLUMN", "encode_ms"))

    @property
    def feature_transforms(self) -> dict:
        return dict(getattr(self.module, "FEATURE_TRANSFORMS", {}))

    @property
    def output_specs(self) -> dict:
        return dict(getattr(self.module, "OUTPUT_SPECS", {}))

    @property
    def sparse_overrides(self) -> list:
        return list(getattr(self.module, "SPARSE_OVERRIDES", []))

    @property
    def parse_config_name(self):
        return self.module.parse_config_name


# ----------------------------------------------------------------------
# Per-codec data extraction
# ----------------------------------------------------------------------

def _bind_globals_for_codec(bundle: CodecConfigBundle) -> None:
    """Bind train_hybrid module-level globals to this codec's config.

    train_hybrid's load_pareto/load_features/build_cell_index/
    build_dataset all consume module-level globals (PARETO/FEATURES,
    CATEGORICAL_AXES/SCALAR_AXES, METRIC_COLUMN/METRIC_DIRECTION/
    TIME_COLUMN, KEEP_FEATURES, ZQ_TARGETS, parse_config_name, etc.).
    To avoid forking those functions, we mutate the globals before
    calling them and restore between codecs. This is the same trick
    `load_codec_config(name)` uses in train_hybrid itself.
    """
    TH.PARETO = bundle.pareto_path
    TH.FEATURES = bundle.features_path
    TH.KEEP_FEATURES = bundle.keep_features
    TH.ZQ_TARGETS = bundle.zq_targets
    TH.parse_config_name = bundle.parse_config_name
    TH.CATEGORICAL_AXES = bundle.categorical_axes
    TH.SCALAR_AXES = bundle.scalar_axes
    TH.SCALAR_SENTINELS = bundle.scalar_sentinels
    TH.METRIC_COLUMN = bundle.metric_column
    TH.METRIC_DIRECTION = bundle.metric_direction
    TH.TIME_COLUMN = bundle.time_column
    TH.FEATURE_TRANSFORMS = bundle.feature_transforms
    TH.OUTPUT_SPECS = bundle.output_specs
    TH.SPARSE_OVERRIDES = bundle.sparse_overrides
    # Reset the CONFIG_NAMES dict — load_pareto fills it from the TSV,
    # but it accumulates across codec runs if we don't clear.
    TH.CONFIG_NAMES = {}


def extract_codec_dataset(bundle: CodecConfigBundle) -> dict:
    """Run train_hybrid's pipeline on one codec and return the inputs,
    soft-target teachers, and metadata needed by the shared-trunk fit.

    Returns a dict with:
      - codec_name, n_cells, cells, feat_cols (codec-specific subset)
      - Xs       (n_rows, n_feats + 4 + 2)  raw simple input
      - bytes_log, scalars, reach, meta
      - teachers_bytes, teachers_per_axis, scalar_means
      - feat_values: per-(image, size) feature dict (for OOD bounds + union build)
      - per-feat transforms
    """
    sys.stderr.write(f"\n=== [{bundle.codec_name}] loading {bundle.pareto_path.name} ===\n")
    _bind_globals_for_codec(bundle)
    pareto, ceilings, has_ceiling, has_time = TH.load_pareto(bundle.pareto_path)
    feats, feat_cols, feat_transforms = TH.load_features(bundle.features_path)
    cells, cell_id_by_key, config_to_cell, parsed_all = TH.build_cell_index()
    n_cells = len(cells)
    sys.stderr.write(
        f"[{bundle.codec_name}] {len(pareto)} pareto cells, {len(feat_cols)} feats, {n_cells} hybrid cells\n"
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
    ) = TH.build_dataset(
        pareto, feats, feat_cols, cells, config_to_cell, parsed_all,
        ceilings=ceilings if has_ceiling else None,
        time_budget_multiplier=0.0,
        time_baselines=None,
        emit_metric_head=False,
        safety_default_cell_idx=None,
    )
    sys.stderr.write(
        f"[{bundle.codec_name}] decision rows: {len(Xs)}, Xs cols: {Xs.shape[1]}\n"
    )

    # Image-level holdout (per codec — each codec's train/val split is
    # deterministic from its own image list using the global seed).
    rng = np.random.default_rng(0xCAFE)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * 0.20))
    val_set = set(images[:n_val])
    train_idx = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    val_idx = np.array([i for i, m in enumerate(meta) if m[0] in val_set])

    # Per-codec teacher on the codec's own simple input. The teacher is
    # the source of truth for soft targets — image-feature-only inputs,
    # so the shared trunk's input layout is irrelevant here.
    Xs_tr = Xs[train_idx]
    bl_tr = bytes_log[train_idx]
    rch_tr = reach[train_idx]
    scalars_tr = {ax: scalars[ax][train_idx] for ax in bundle.scalar_axes}
    sys.stderr.write(f"[{bundle.codec_name}] training per-codec teacher...\n")
    (
        t_bytes, t_per_axis, scalar_means, _, _, _, _
    ) = TH.train_teacher_per_cell(
        Xs_tr, bl_tr, scalars_tr, rch_tr, n_cells,
        params=PL.HISTGB_FAST,  # iteration-fast preset; FULL would be slow on 4-codec smoke
        time_log_tr=None,
        metric_log_tr=None,
    )

    return {
        "codec_name": bundle.codec_name,
        "bundle": bundle,
        "n_cells": n_cells,
        "cells": cells,
        "feat_cols": feat_cols,
        "feat_transforms": feat_transforms,
        "Xs": Xs,
        "Xe": Xe,
        "bytes_log": bytes_log,
        "scalars": scalars,
        "reach": reach,
        "meta": meta,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "feats": feats,
        "teachers_bytes": t_bytes,
        "teachers_per_axis": t_per_axis,
        "scalar_means": scalar_means,
        "config_names": dict(TH.CONFIG_NAMES),
        "scalar_axes": list(bundle.scalar_axes),
        "scalar_sentinels": bundle.scalar_sentinels,
    }


# ----------------------------------------------------------------------
# Union schema + shared-trunk input vector
# ----------------------------------------------------------------------

def build_union_schema(codec_data: list[dict]) -> dict:
    """Compute the union of all codecs' feat_cols. Each codec gets a
    stable mapping `feat_name -> union_index`. The shared trunk's input
    vector is the union vector, with feat values from the codec's
    feature dict in their slots and 0 elsewhere; a parallel presence-
    mask vector marks which features are real vs imputed.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for cd in codec_data:
        for c in cd["feat_cols"]:
            if c not in seen_set:
                seen.append(c)
                seen_set.add(c)
    union_idx = {c: i for i, c in enumerate(seen)}
    sys.stderr.write(
        f"\n[union] {len(seen)} total features across {len(codec_data)} codecs "
        f"(per-codec sizes: "
        + ", ".join(f"{cd['codec_name']}={len(cd['feat_cols'])}" for cd in codec_data)
        + ")\n"
    )
    return {"feat_cols": seen, "idx": union_idx}


SIZE_CLASSES = TH.SIZE_CLASSES
SIZE_INDEX = TH.SIZE_INDEX


def build_trunk_input(
    codec_data: list[dict],
    union: dict,
    n_codecs: int,
) -> dict:
    """Build the shared-trunk input matrix per codec.

    Input layout:
      [ union_feat (n_union, 0 where codec doesn't have it),
        presence_mask (n_union, 1 where codec has it),
        size_oh (4),
        log_px, zq_norm,
        codec_oh (n_codecs) ]
    """
    n_union = len(union["feat_cols"])
    n_inputs = n_union * 2 + 4 + 2 + n_codecs
    per_codec_X: list[np.ndarray] = []
    for ci, cd in enumerate(codec_data):
        bundle = cd["bundle"]
        # Map this codec's feat_cols → union positions.
        col_positions = np.array([union["idx"][c] for c in cd["feat_cols"]],
                                 dtype=np.int64)
        n_rows = cd["Xs"].shape[0]
        Xc = np.zeros((n_rows, n_inputs), dtype=np.float32)
        # The codec's simple Xs already has layout:
        #   [feat_values (n_codec_feats), size_oh (4), log_px, zq_norm]
        n_feats = len(cd["feat_cols"])
        feat_block = cd["Xs"][:, :n_feats]
        size_oh_block = cd["Xs"][:, n_feats:n_feats + 4]
        log_px_zq_block = cd["Xs"][:, n_feats + 4:n_feats + 6]
        # Scatter into union slots.
        Xc[:, col_positions] = feat_block
        # Presence mask: 1 in this codec's slots.
        Xc[:, n_union + col_positions] = 1.0
        # Size onehot + log_px + zq_norm.
        Xc[:, n_union * 2:n_union * 2 + 4] = size_oh_block
        Xc[:, n_union * 2 + 4:n_union * 2 + 6] = log_px_zq_block
        # Codec onehot.
        Xc[:, n_union * 2 + 6 + ci] = 1.0
        per_codec_X.append(Xc)
        sys.stderr.write(
            f"[trunk-input] {bundle.codec_name}: {Xc.shape} "
            f"(presence={int(n_feats)}/{n_union})\n"
        )
    return {
        "n_inputs": n_inputs,
        "n_union": n_union,
        "n_codecs": n_codecs,
        "per_codec_X": per_codec_X,
    }


# ----------------------------------------------------------------------
# Soft target generation (per codec)
# ----------------------------------------------------------------------

def build_soft_targets(cd: dict) -> dict:
    """Compute per-codec soft targets matching the single-codec
    train_hybrid layout: `[bytes_log_block | scalar_axis_1 | ... |
    scalar_axis_K]`, each block n_cells wide.

    Returns soft_tr (n_train_rows, output_dim) and the (mu, sigma)
    blocks we'll absorb into the head later. Block standardization
    matches the single-codec trainer's per-head normalization fix
    (see train_hybrid:2249).
    """
    Xs = cd["Xs"]
    n_cells = cd["n_cells"]
    train_idx = cd["train_idx"]
    val_idx = cd["val_idx"]
    Xs_tr = Xs[train_idx]
    Xs_va = Xs[val_idx]
    bytes_pred_tr = TH.teacher_predict_all(
        cd["teachers_bytes"], Xs_tr, np.nanmean(cd["bytes_log"][train_idx], axis=0), n_cells
    )
    bytes_pred_va = TH.teacher_predict_all(
        cd["teachers_bytes"], Xs_va, np.nanmean(cd["bytes_log"][train_idx], axis=0), n_cells
    )
    scalar_pred_tr: dict[str, np.ndarray] = {}
    scalar_pred_va: dict[str, np.ndarray] = {}
    for axis in cd["scalar_axes"]:
        scalar_pred_tr[axis] = TH.teacher_predict_all(
            cd["teachers_per_axis"][axis], Xs_tr,
            cd["scalar_means"][axis], n_cells,
        )
        scalar_pred_va[axis] = TH.teacher_predict_all(
            cd["teachers_per_axis"][axis], Xs_va,
            cd["scalar_means"][axis], n_cells,
        )
    soft_blocks = [bytes_pred_tr]
    for axis in cd["scalar_axes"]:
        soft_blocks.append(scalar_pred_tr[axis])
    soft_tr = np.concatenate(soft_blocks, axis=1)

    # Per-block standardization for scalar heads only (bytes stays log-space).
    scalar_block_starts: list[tuple[int, int, str, float, float]] = []
    for bi, axis in enumerate(cd["scalar_axes"], start=1):
        start = bi * n_cells
        end = (bi + 1) * n_cells
        block = soft_tr[:, start:end]
        mu = float(np.mean(block))
        sigma = float(np.std(block))
        if sigma < 1e-12:
            scalar_block_starts.append((start, end, axis, 0.0, 1.0))
        else:
            soft_tr[:, start:end] = (block - mu) / sigma
            scalar_block_starts.append((start, end, axis, mu, sigma))

    # Also build the val soft target without standardization (for
    # diagnostic teacher metrics on the held-out side).
    soft_va_blocks = [bytes_pred_va]
    for axis in cd["scalar_axes"]:
        soft_va_blocks.append(scalar_pred_va[axis])
    soft_va = np.concatenate(soft_va_blocks, axis=1)

    return {
        "soft_tr": soft_tr,
        "soft_va": soft_va,
        "bytes_pred_va": bytes_pred_va,
        "scalar_pred_va": scalar_pred_va,
        "scalar_block_starts": scalar_block_starts,
        "output_dim": soft_tr.shape[1],
    }


# ----------------------------------------------------------------------
# Shared-trunk MLP (PyTorch)
# ----------------------------------------------------------------------

class SharedTrunkMLP:
    """Trunk + per-codec head. Single optimizer, single forward pass per
    codec batch.

    We expose `.coefs_` / `.intercepts_` per codec by composing the
    trunk weights with that codec's head weights — both bake_picker and
    the runtime see a plain MLP.
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: tuple[int, ...],
        head_dims: dict[str, int],  # codec_name -> n_outputs
        leaky_slope: float = 0.01,
        seed: int = 0xCAFE,
    ):
        import torch
        import torch.nn as nn

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._torch = torch
        self._nn = nn

        # Trunk = Linear(in, h0) -> LReLU -> Linear(h0, h1) -> LReLU -> ...
        # Final trunk activation is LeakyReLU; per-codec head is a single
        # Linear from h_last to that codec's output dim.
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({
            name: nn.Linear(prev, n_out) for name, n_out in head_dims.items()
        })
        self.hidden_sizes = hidden_sizes
        self.head_dims = head_dims
        self.n_inputs = n_inputs
        self.leaky_slope = leaky_slope

        # Wire trunk + heads into one ParameterList for the optimizer.
        self._all_params = list(self.trunk.parameters())
        for h in self.heads.values():
            self._all_params.extend(h.parameters())

    def forward(self, X, codec_name: str):
        z = self.trunk(X)
        return self.heads[codec_name](z)

    def export_codec(self, codec_name: str) -> dict:
        """Materialize a codec's effective MLP: `coefs_` and `intercepts_`
        in the sklearn convention (W is (in_features, out_features)) so
        the JSON shape exactly matches what bake_picker expects.
        """
        torch = self._torch
        nn = self._nn

        # Walk the trunk Sequential, collecting Linear weights only.
        # LeakyReLU is implicit between linears (recorded in `activation`).
        coefs: list[np.ndarray] = []
        intercepts: list[np.ndarray] = []
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                W = m.weight.detach().cpu().numpy().T.astype(np.float64)  # (in, out)
                b = m.bias.detach().cpu().numpy().astype(np.float64)
                coefs.append(W)
                intercepts.append(b)
        head = self.heads[codec_name]
        Wh = head.weight.detach().cpu().numpy().T.astype(np.float64)
        bh = head.bias.detach().cpu().numpy().astype(np.float64)
        coefs.append(Wh)
        intercepts.append(bh)

        return {"coefs_": coefs, "intercepts_": intercepts}


def _predict_codec(model: SharedTrunkMLP, X: np.ndarray,
                   codec_name: str) -> np.ndarray:
    """Forward pass through trunk + that codec's head; returns NumPy."""
    torch = model._torch
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32))
        Y_t = model.forward(X_t, codec_name)
        return Y_t.cpu().numpy()


def train_shared_trunk(
    model: SharedTrunkMLP,
    codec_train_payloads: list[dict],  # one dict per codec: X, soft, val_X, val_soft
    *,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    epochs: int = 200,
    n_iter_no_change: int = 30,
    warmup_frac: float = 0.05,
    seed: int = 0xCAFE,
) -> dict:
    """Train the shared-trunk MLP with codec-balanced sampling.

    Each "step" picks `batch_size` rows from EACH codec independently
    (so 4 codecs × 512 rows/step = 2048 rows per backward). The loss
    is `sum_{codec} MSE(pred_codec, target_codec)`. Per-codec MSE is
    averaged over rows × outputs — this keeps codecs with more output
    dimensions from dominating gradient (e.g. zenjpeg's 16 cells
    don't overpower zenwebp's 6 cells just because it has more
    columns).
    """
    torch = model._torch
    nn = model._nn

    rng = np.random.default_rng(seed)

    # Build per-codec train tensors + val tensors. Move to CPU torch
    # tensors once (training set is small enough — picker work fits in
    # main RAM by design).
    train_tensors: list[dict] = []
    for p in codec_train_payloads:
        X_tr = torch.from_numpy(p["X_tr"].astype(np.float32))
        Y_tr = torch.from_numpy(p["soft_tr"].astype(np.float32))
        X_va = torch.from_numpy(p["X_va"].astype(np.float32))
        Y_va = torch.from_numpy(p["soft_va"].astype(np.float32))
        # Per-codec label variance — divides MSE so each codec's loss
        # contributes proportionally regardless of output scale. The
        # bytes_log head dominates raw variance (zenwebp bytes range
        # ~10..18 ~~ var 16; standardized scalar heads have var ~1).
        # Without this normalization, codecs whose bytes_log spans a
        # wider range get gradient priority and the rest collapse —
        # exactly what we saw on the 2-codec v0 run (zenwebp loss
        # 800+, zenjpeg 35). Variance is computed over train labels
        # only (no leak from val).
        loss_scale = float(Y_tr.var().item())
        if not (loss_scale > 1e-9):
            loss_scale = 1.0
        train_tensors.append({
            "codec": p["codec"],
            "X_tr": X_tr, "Y_tr": Y_tr, "X_va": X_va, "Y_va": Y_va,
            "n_tr": X_tr.shape[0],
            "loss_scale": loss_scale,
        })

    optimizer = torch.optim.AdamW(model._all_params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Per-epoch steps: enough to cover the LARGEST codec's training set
    # once (each codec runs through its own data via random sampling
    # with replacement at the same step count).
    n_tr_max = max(t["n_tr"] for t in train_tensors)
    steps_per_epoch = max(1, n_tr_max // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def _lr_at_step(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        # Cosine annealing.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    sys.stderr.write(
        f"\n[trunk] training: hidden={model.hidden_sizes}, "
        f"epochs={epochs}, steps/epoch={steps_per_epoch}, "
        f"batch_size={batch_size} per codec, lr={lr}, wd={weight_decay}\n"
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    best_state = {k: v.detach().clone() for k, v in
                  model.trunk.state_dict().items()}
    best_heads = {
        name: {k: v.detach().clone() for k, v in h.state_dict().items()}
        for name, h in model.heads.items()
    }

    step = 0
    t0 = time.monotonic()
    for ep in range(epochs):
        # Per-codec random index permutation for this epoch (sampled with
        # replacement when n_tr < steps_per_epoch * batch_size, which is
        # rare on real Pareto sweeps but harmless).
        codec_perms = {}
        for t in train_tensors:
            n_total = steps_per_epoch * batch_size
            if n_total <= t["n_tr"]:
                codec_perms[t["codec"]] = rng.permutation(t["n_tr"])[:n_total]
            else:
                # Sample with replacement to fill.
                codec_perms[t["codec"]] = rng.integers(0, t["n_tr"], size=n_total)

        for s in range(steps_per_epoch):
            cur_lr = _lr_at_step(step)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            optimizer.zero_grad()
            total_loss = 0.0
            for t in train_tensors:
                idx = codec_perms[t["codec"]][s * batch_size:(s + 1) * batch_size]
                idx_t = torch.from_numpy(idx.astype(np.int64))
                Xb = t["X_tr"].index_select(0, idx_t)
                Yb = t["Y_tr"].index_select(0, idx_t)
                pred = model.forward(Xb, t["codec"])
                loss = loss_fn(pred, Yb) / t["loss_scale"]
                total_loss = total_loss + loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model._all_params, max_norm=5.0)
            optimizer.step()
            step += 1

        # Val MSE per codec. Use the same variance-normalized loss for
        # the aggregate so early-stop tracks the joint objective the
        # optimizer minimizes (not a raw-scale sum that the wide-range
        # codec would dominate).
        val_losses_raw = []
        val_losses_norm = []
        with torch.no_grad():
            for t in train_tensors:
                pred_va = model.forward(t["X_va"], t["codec"])
                raw = loss_fn(pred_va, t["Y_va"]).item()
                val_losses_raw.append(raw)
                val_losses_norm.append(raw / t["loss_scale"])
        val_losses = val_losses_norm
        val_total = sum(val_losses_norm) / len(val_losses_norm)
        if val_total < best_val - 1e-6:
            best_val = val_total
            best_epoch = ep
            epochs_since_improve = 0
            best_state = {k: v.detach().clone() for k, v in
                          model.trunk.state_dict().items()}
            best_heads = {
                name: {k: v.detach().clone() for k, v in h.state_dict().items()}
                for name, h in model.heads.items()
            }
        else:
            epochs_since_improve += 1
        if ep % 10 == 0 or ep == epochs - 1:
            per_codec = "  ".join(
                f"{t['codec']}:{vl:.4f}" for t, vl in zip(train_tensors, val_losses)
            )
            sys.stderr.write(
                f"  ep {ep:3d} lr={cur_lr:.2e} val_mean={val_total:.4f}  {per_codec}"
                f"  (best ep={best_epoch} val={best_val:.4f}, "
                f"no_improve={epochs_since_improve})\n"
            )
        if epochs_since_improve >= n_iter_no_change:
            sys.stderr.write(
                f"[trunk] early stop at ep {ep} (best ep {best_epoch}, "
                f"val={best_val:.4f})\n"
            )
            break

    # Restore best weights.
    model.trunk.load_state_dict(best_state)
    for name, sd in best_heads.items():
        model.heads[name].load_state_dict(sd)

    elapsed = time.monotonic() - t0
    sys.stderr.write(
        f"[trunk] training done in {elapsed:.1f}s. Best val_mean={best_val:.4f} at ep {best_epoch}\n"
    )
    return {"best_val": best_val, "best_epoch": best_epoch, "elapsed_s": elapsed}


# ----------------------------------------------------------------------
# Per-codec eval + JSON emission
# ----------------------------------------------------------------------

def evaluate_codec(
    model: SharedTrunkMLP,
    cd: dict,
    soft: dict,
    X: np.ndarray,
    split: str = "val",
) -> dict:
    """Run argmin eval on the shared-trunk model's predictions for this
    codec, mirroring train_hybrid's `evaluate_argmin` / `evaluate_scalars`
    so the numbers are directly comparable to the single-codec baseline.

    `split` is "train" or "val" — selects which row index set (and thus
    which slices of meta / bytes_log / reach / scalars) to grade against.
    """
    n_cells = cd["n_cells"]
    if split == "val":
        idx = cd["val_idx"]
    elif split == "train":
        idx = cd["train_idx"]
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    Y_pred = _predict_codec(model, X, cd["codec_name"])

    # Apply inverse-standardization for scalar blocks (we standardized
    # at train time; need to reverse for evaluate).
    for start, end, axis, mu, sigma in soft["scalar_block_starts"]:
        if sigma == 0.0 and mu == 0.0:
            continue
        Y_pred[:, start:end] = Y_pred[:, start:end] * sigma + mu

    pred_bytes = Y_pred[:, :n_cells]
    scalar_preds = {}
    for start, end, axis, _mu, _sigma in soft["scalar_block_starts"]:
        scalar_preds[axis] = Y_pred[:, start:end]

    meta_sel = [cd["meta"][i] for i in idx]
    bl_sel = cd["bytes_log"][idx]
    rch_sel = cd["reach"][idx]
    scalars_sel = {ax: cd["scalars"][ax][idx] for ax in cd["scalar_axes"]}
    all_mask = np.ones(n_cells, dtype=bool)
    # Bind globals so evaluate_* honor this codec's axes.
    TH.SCALAR_AXES = cd["scalar_axes"]
    TH.SCALAR_SENTINELS = cd["scalar_sentinels"]
    argmin = TH.evaluate_argmin(pred_bytes, bl_sel, rch_sel, meta_sel, all_mask)
    sc = TH.evaluate_scalars(scalar_preds, scalars_sel, rch_sel)
    return {
        "argmin": argmin,
        "scalars": sc,
        "Y_pred": Y_pred,
        "pred_bytes": pred_bytes,
        "scalar_preds": scalar_preds,
    }


def write_codec_json(
    cd: dict,
    soft: dict,
    model: SharedTrunkMLP,
    union: dict,
    n_codecs: int,
    codec_idx: int,
    all_codec_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    out_path: Path,
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    """Emit one per-codec picker JSON matching the single-codec bake
    shape (the keys bake_picker.py reads).

    The trunk + this codec's head get serialized as a normal MLP
    `layers` list. bake_picker is agnostic to where the weights came
    from — it just packs them as zenpredict ZNPR v3.
    """
    bundle = cd["bundle"]
    n_cells = cd["n_cells"]
    exported = model.export_codec(cd["codec_name"])
    # Absorb the per-scalar-block (mu, sigma) standardization into the
    # head's final layer so the bake produces natural-unit predictions
    # at inference (same trick train_hybrid uses post-fit).
    final_W = exported["coefs_"][-1]    # (h_last, n_outputs)
    final_b = exported["intercepts_"][-1]
    for start, end, axis, mu, sigma in soft["scalar_block_starts"]:
        if sigma == 0.0 and mu == 0.0:
            continue
        final_W[:, start:end] *= sigma
        final_b[start:end] = final_b[start:end] * sigma + mu

    # Build output_layout in train_hybrid's order: bytes_log first, then
    # each scalar axis. n_outputs = (1 + len(SCALAR_AXES)) * n_cells.
    output_dim = (1 + len(cd["scalar_axes"])) * n_cells
    output_layout = {"bytes_log": [0, n_cells]}
    next_block = 1
    for axis in cd["scalar_axes"]:
        output_layout[axis] = [next_block * n_cells,
                               (next_block + 1) * n_cells]
        next_block += 1

    # Feature transforms: the shared-trunk input does NOT include the
    # codec's per-feature transforms. The trunk gets union-standardized
    # values; we don't apply log/log1p there because a feature may be
    # transformed differently across codecs (different runtime
    # consumers). For runtime parity, the per-codec runtime would need
    # to populate the union vector with raw values pre-transform and
    # also know the trunk's input layout. **Caveat documented in
    # benchmarks/multi_codec_bench_2026-05-17.md** — codecs that depend
    # on log/log1p transforms today won't be runtime-compatible with
    # the shared trunk until the runtime is taught the trunk's input
    # layout. The smoke test reports overheads with the trainer's
    # transforms-disabled view as the apples-to-apples comparison.
    n_inputs = exported["coefs_"][0].shape[0]
    n_union = len(union["feat_cols"])
    feat_transform_list = ["identity"] * n_inputs

    # output_specs: expand bundle's OUTPUT_SPECS by head name → per-output array.
    output_specs_dict = bundle.output_specs
    output_specs_array: list[dict] = []
    if output_specs_dict:
        per_idx: list[dict | None] = [None] * output_dim
        missing = []
        for head_name, span in output_layout.items():
            spec = output_specs_dict.get(head_name)
            if spec is None:
                missing.append(head_name)
                continue
            start, end = int(span[0]), int(span[1])
            for i in range(start, end):
                per_idx[i] = dict(spec)
        if missing or any(s is None for s in per_idx):
            output_specs_array = []
        else:
            output_specs_array = [dict(s) for s in per_idx]  # type: ignore[arg-type]

    cells_dump = []
    for c in cd["cells"]:
        cells_dump.append({k: (list(v) if isinstance(v, (list, tuple)) else v)
                           for k, v in c.items()})

    sentinels_for_manifest = {
        axis: float(cd["scalar_sentinels"][axis])
        for axis in cd["scalar_axes"] if axis in cd["scalar_sentinels"]
    }

    # Build extra_axes list so bake_picker can name the engineered
    # input slots. Layout per `build_trunk_input`:
    #   [union_feats (n_union)] -> covered by feat_cols (above)
    #   [presence_mask (n_union)] -> presence_<feat_name>
    #   [size_oh (4)]            -> size_tiny / small / medium / large
    #   [log_pixels, zq_norm]    -> log_pixels, zq_norm
    #   [codec_onehot (n_codecs)] -> codec_<name>
    extra_axes = (
        [f"presence_{c}" for c in union["feat_cols"]]
        + ["size_tiny", "size_small", "size_medium", "size_large"]
        + ["log_pixels", "zq_norm"]
        + [f"codec_{n}" for n in all_codec_names]
    )

    out = {
        "n_inputs": int(n_inputs),
        "n_outputs": int(output_dim),
        "n_cells": int(n_cells),
        "safety_profile": "size_optimal",
        "config_names": {int(k): v for k, v in cd["config_names"].items()},
        # feat_cols for the bake = the UNION layout. The trunk
        # consumes `[union_feats, presence_mask, size_oh, log_px,
        # zq_norm, codec_oh]`; bake_picker reads feat_cols only for
        # schema_hash derivation. Runtime parity requires a matching
        # union-layout adapter (see TODO above + bench writeup).
        "feat_cols": list(union["feat_cols"]),
        "extra_axes": extra_axes,
        "feature_transforms": feat_transform_list,
        "output_specs": output_specs_array,
        "sparse_overrides": list(bundle.sparse_overrides),
        "scaler_mean": [float(x) for x in scaler_mean],
        "scaler_scale": [float(x) for x in scaler_scale],
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(exported["coefs_"], exported["intercepts_"])
        ],
        "activation": "leakyrelu",
        "hybrid_heads_manifest": {
            "n_cells": int(n_cells),
            "cells": cells_dump,
            "categorical_axes": list(bundle.categorical_axes),
            "scalar_axes": list(cd["scalar_axes"]),
            "output_layout": output_layout,
            "scalar_sentinels": sentinels_for_manifest,
            "lambda_notrellis_sentinel": (
                sentinels_for_manifest.get("lambda", 0.0)
            ),
        },
        "training_objective": {
            "name": "multi_codec_shared_trunk",
            "metric_name": bundle.metric_column,
            "metric_direction": bundle.metric_direction,
        },
        # Multi-codec-specific bookkeeping (additive — bake_picker
        # ignores unknown keys; runtime would treat this as informational).
        "multi_codec": {
            "codec_index": int(codec_idx),
            "n_codecs": int(n_codecs),
            "n_union_features": int(n_union),
            "codec_name": cd["codec_name"],
            "codec_feat_cols": list(cd["feat_cols"]),
            "trunk_hidden": list(model.hidden_sizes),
        },
        "student_metrics": {
            "argmin": val_metrics["argmin"],
            "scalars": val_metrics["scalars"],
        },
        "train_metrics": train_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    sys.stderr.write(f"  wrote {out_path} ({len(json.dumps(out)) / 1024:.1f} KB)\n")


def write_joint_bake_json(
    codec_data: list[dict],
    soft_targets: list[dict],
    model: SharedTrunkMLP,
    union: dict,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    out_path: Path,
    per_codec_metrics: dict | None = None,
) -> dict:
    """Emit ONE joint bake JSON encapsulating the whole multi-codec
    model — shared trunk + every codec's head concatenated into one
    MLP — with the ZNPR v3.2 `multi_codec_schema` block populated.

    The bake artifact is consumed by an external caller via
    `Picker::predict_multi_codec(codec_id, codec_features, size_class,
    log_pixels, zq_norm)`. Codec crates don't depend on this bake;
    only the external orchestrator does.

    Joint output layout — codec heads concatenated in input order:

        [codec_0 head outputs (n_out_0) | codec_1 head outputs (n_out_1) | ...]

    Per-codec `output_range = (offset_i, offset_i + n_out_i)`.

    Returns the JSON dict (also written to `out_path`).
    """
    torch = model._torch
    nn = model._nn

    # Walk the shared trunk collecting Linear weight + bias.
    trunk_coefs: list[np.ndarray] = []
    trunk_intercepts: list[np.ndarray] = []
    for m in model.trunk:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy().T.astype(np.float64)  # (in, out)
            b = m.bias.detach().cpu().numpy().astype(np.float64)
            trunk_coefs.append(W)
            trunk_intercepts.append(b)
    if not trunk_coefs:
        raise SystemExit("joint bake: trunk has no Linear layers — refusing to write")

    h_last = trunk_coefs[-1].shape[1]

    # Concatenate per-codec heads horizontally into a single Linear
    # (h_last → sum(n_out_i)). Absorb each codec's per-block scalar
    # standardization (μ, σ) into its head columns so the joint bake
    # produces natural-unit outputs — identical to what the per-codec
    # bake does today.
    head_W_blocks: list[np.ndarray] = []
    head_b_blocks: list[np.ndarray] = []
    per_codec_output_ranges: list[tuple[int, int]] = []
    offset = 0
    for cd, soft in zip(codec_data, soft_targets):
        codec_name = cd["codec_name"]
        head = model.heads[codec_name]
        Wh = head.weight.detach().cpu().numpy().T.astype(np.float64)  # (h_last, n_out_codec)
        bh = head.bias.detach().cpu().numpy().astype(np.float64)
        # Inverse-standardize scalar blocks within this codec's outputs.
        for start, end, axis, mu, sigma in soft["scalar_block_starts"]:
            if sigma == 0.0 and mu == 0.0:
                continue
            Wh[:, start:end] *= sigma
            bh[start:end] = bh[start:end] * sigma + mu
        n_out_codec = Wh.shape[1]
        head_W_blocks.append(Wh)
        head_b_blocks.append(bh)
        per_codec_output_ranges.append((offset, offset + n_out_codec))
        offset += n_out_codec
    joint_head_W = np.concatenate(head_W_blocks, axis=1)  # (h_last, sum_n_out)
    joint_head_b = np.concatenate(head_b_blocks)
    assert joint_head_W.shape[0] == h_last, (
        f"joint head row count {joint_head_W.shape[0]} != trunk out {h_last}"
    )

    coefs = list(trunk_coefs) + [joint_head_W]
    intercepts = list(trunk_intercepts) + [joint_head_b]

    n_inputs = coefs[0].shape[0]
    output_dim = coefs[-1].shape[1]
    n_union = len(union["feat_cols"])
    n_codecs = len(codec_data)

    # extra_axes: presence_<feat>, size_oh, log_pixels, zq_norm, codec_<name>.
    extra_axes = (
        [f"presence_{c}" for c in union["feat_cols"]]
        + ["size_tiny", "size_small", "size_medium", "size_large"]
        + ["log_pixels", "zq_norm"]
        + [f"codec_{cd['codec_name']}" for cd in codec_data]
    )

    # multi_codec_schema block — per-codec map + union_feat_count.
    per_codec_block = []
    for ci, cd in enumerate(codec_data):
        codec_name = cd["codec_name"]
        # Map this codec's feat_cols to its union slot index.
        union_slots = [int(union["idx"][c]) for c in cd["feat_cols"]]
        lo, hi = per_codec_output_ranges[ci]
        n_cells = cd["n_cells"]
        n_heads = 1 + len(cd["scalar_axes"])  # bytes + scalar axes
        per_codec_block.append({
            "codec_name": codec_name,
            "union_slot_for_codec_feat": union_slots,
            "output_range": [lo, hi],
            "head_n_cells": int(n_cells),
            "head_n_heads": int(n_heads),
        })

    # Build per-codec hybrid_heads_manifest summary so the bake artifact
    # carries the cell layout for each codec (consumer must demux by
    # codec_id and pick the relevant slice).
    codec_manifests: list[dict] = []
    for ci, (cd, soft) in enumerate(zip(codec_data, soft_targets)):
        lo, hi = per_codec_output_ranges[ci]
        n_cells = cd["n_cells"]
        cells_dump = []
        for c in cd["cells"]:
            cells_dump.append({
                k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in c.items()
            })
        # output_layout within this codec's output_range — relative
        # offsets (consumer adds `lo` to get the absolute index).
        within = {"bytes_log": [0, n_cells]}
        next_block = 1
        for axis in cd["scalar_axes"]:
            within[axis] = [next_block * n_cells, (next_block + 1) * n_cells]
            next_block += 1
        codec_manifests.append({
            "codec_name": cd["codec_name"],
            "codec_index": int(ci),
            "output_range": [int(lo), int(hi)],
            "n_cells": int(n_cells),
            "categorical_axes": list(cd["bundle"].categorical_axes),
            "scalar_axes": list(cd["scalar_axes"]),
            "cells": cells_dump,
            "output_layout_relative": within,
            "config_names": {int(k): v for k, v in cd["config_names"].items()},
            "codec_feat_cols": list(cd["feat_cols"]),
        })

    # n_inputs == 2*U + 6 + n_codecs (per the trunk-input layout).
    expected = 2 * n_union + 6 + n_codecs
    if n_inputs != expected:
        raise SystemExit(
            f"joint bake: trunk in_dim {n_inputs} != expected "
            f"{expected} = 2*n_union({n_union}) + 6 + n_codecs({n_codecs})"
        )

    out = {
        "n_inputs": int(n_inputs),
        "n_outputs": int(output_dim),
        "safety_profile": "size_optimal",
        "feat_cols": list(union["feat_cols"]),
        "extra_axes": extra_axes,
        "feature_transforms": ["identity"] * n_inputs,
        "output_specs": [],     # see TODO in write_codec_json on per-codec specs
        "sparse_overrides": [],
        "scaler_mean": [float(x) for x in scaler_mean],
        "scaler_scale": [float(x) for x in scaler_scale],
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(coefs, intercepts)
        ],
        "activation": "leakyrelu",
        # ZNPR v3.2 multi-codec section — forwarded verbatim by
        # bake_picker.py → zenpredict_bake::MultiCodecSchemaJson →
        # the runtime's `multi_codec_schema` header section.
        "multi_codec_schema": {
            "union_feat_count": int(n_union),
            "per_codec": per_codec_block,
        },
        # Joint-bake-specific bookkeeping (additive; bake_picker
        # ignores unknown keys at the top level).
        "joint_multi_codec": {
            "n_codecs": int(n_codecs),
            "n_union_features": int(n_union),
            "trunk_hidden": list(model.hidden_sizes),
            "codec_order": [cd["codec_name"] for cd in codec_data],
            "per_codec_manifests": codec_manifests,
        },
        "training_objective": {
            "name": "multi_codec_shared_trunk_joint",
        },
    }
    if per_codec_metrics is not None:
        out["per_codec_metrics"] = per_codec_metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    sys.stderr.write(
        f"  wrote JOINT bake JSON {out_path} "
        f"({len(json.dumps(out)) / 1024:.1f} KB, n_inputs={n_inputs}, "
        f"output_dim={output_dim}, n_codecs={n_codecs})\n"
    )
    return out


# ----------------------------------------------------------------------
# CLI driver
# ----------------------------------------------------------------------

def _parse_codec_arg(s: str) -> tuple[str, Path, Path]:
    """Parse `<name>=<config_path>:<data_root>`. Examples:
       zenjpeg=examples/zenjpeg_picker_config.py:/home/lilith/work/zen/zenjpeg
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--codec expects 'name=config_path:data_root', got {s!r}"
        )
    name, rest = s.split("=", 1)
    if ":" not in rest:
        raise argparse.ArgumentTypeError(
            f"--codec expects 'name=config_path:data_root' (missing ':'), got {s!r}"
        )
    config_path_s, data_root_s = rest.rsplit(":", 1)
    config_path = Path(config_path_s)
    data_root = Path(data_root_s)
    if not config_path.is_absolute():
        # Resolve config_path against this script's parent (zentrain/tools).
        # We allow `examples/...` shorthand for the standard layout.
        candidate = (_HERE.parent / config_path).resolve()
        if candidate.exists():
            config_path = candidate
        else:
            config_path = config_path.resolve()
    if not data_root.is_absolute():
        data_root = data_root.resolve()
    return name, config_path, data_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a shared-trunk MLP picker across multiple codecs.",
    )
    parser.add_argument(
        "--codec", action="append", type=_parse_codec_arg, required=True,
        help="Repeatable. Format: name=config_path:data_root  "
             "(e.g. zenjpeg=examples/zenjpeg_picker_config.py:/home/lilith/work/zen/zenjpeg). "
             "config_path may be relative to zentrain/.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write per-codec JSON files into.",
    )
    parser.add_argument(
        "--hidden", type=str, default="96,48",
        help="Comma-separated trunk hidden sizes (default 96,48).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0xCAFE)
    parser.add_argument(
        "--out-name", type=str, default="multi_codec",
        help="Filename prefix for the per-codec bake JSONs.",
    )
    parser.add_argument(
        "--joint-bake", type=Path, default=None,
        help="Optional path to write the joint bake JSON (single file "
             "encapsulating the shared trunk + every codec's head, with "
             "the ZNPR v3.2 multi_codec_schema block populated). "
             "Consumed by bake_picker.py to emit one .bin that an "
             "external caller invokes via Picker::predict_multi_codec.",
    )
    args = parser.parse_args()

    hidden = tuple(int(x) for x in args.hidden.split(","))
    bundles: list[CodecConfigBundle] = []
    for name, config_path, data_root in args.codec:
        bundles.append(CodecConfigBundle(name, config_path, data_root))

    # --- Phase 1: per-codec data + teachers
    codec_data: list[dict] = []
    for b in bundles:
        codec_data.append(extract_codec_dataset(b))

    # --- Phase 2: union schema + trunk input
    union = build_union_schema(codec_data)
    trunk_input = build_trunk_input(codec_data, union, n_codecs=len(codec_data))

    # --- Phase 3: per-codec soft targets
    soft_targets: list[dict] = []
    for cd in codec_data:
        soft_targets.append(build_soft_targets(cd))

    # --- Phase 4: shared input scaler. Fit on the concatenation of all
    # codecs' train rows so the trunk sees standardized inputs regardless
    # of codec.
    from sklearn.preprocessing import StandardScaler  # local import — sklearn already used elsewhere
    train_X_all = []
    for cd, Xc in zip(codec_data, trunk_input["per_codec_X"]):
        train_X_all.append(Xc[cd["train_idx"]])
    train_X_concat = np.concatenate(train_X_all, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_X_concat)
    scaled_train = []
    scaled_val = []
    for cd, Xc in zip(codec_data, trunk_input["per_codec_X"]):
        scaled_train.append(scaler.transform(Xc[cd["train_idx"]]))
        scaled_val.append(scaler.transform(Xc[cd["val_idx"]]))

    # --- Phase 5: build trunk + heads, train
    head_dims = {cd["codec_name"]: soft_targets[i]["output_dim"]
                 for i, cd in enumerate(codec_data)}
    n_inputs = trunk_input["n_inputs"]
    sys.stderr.write(
        f"\n[arch] trunk inputs={n_inputs}, hidden={hidden}, "
        f"heads={head_dims}\n"
    )
    model = SharedTrunkMLP(n_inputs, hidden, head_dims, seed=args.seed)

    payloads = []
    for cd, soft, X_tr_s, X_va_s in zip(codec_data, soft_targets, scaled_train, scaled_val):
        payloads.append({
            "codec": cd["codec_name"],
            "X_tr": X_tr_s,
            "soft_tr": soft["soft_tr"],
            "X_va": X_va_s,
            "soft_va": soft["soft_va"],
        })

    train_summary = train_shared_trunk(
        model, payloads,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        seed=args.seed,
    )

    # --- Phase 6: per-codec eval + JSON emission
    all_val_metrics: dict[str, dict] = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (cd, soft, X_va_s) in enumerate(zip(codec_data, soft_targets, scaled_val)):
        # Train-side metrics for the gap report.
        train_eval = evaluate_codec(model, cd, soft, scaled_train[i], split="train")
        val_eval = evaluate_codec(model, cd, soft, X_va_s, split="val")
        # Restore globals for any further evaluation calls (paranoia).
        _bind_globals_for_codec(cd["bundle"])
        sys.stderr.write(
            f"\n[{cd['codec_name']}] student VAL  argmin_acc={val_eval['argmin'].get('argmin_acc', 0):.1%}  "
            f"mean_overhead={val_eval['argmin'].get('mean_pct', 0):.2f}%  "
            f"p90={val_eval['argmin'].get('p90_pct', 0):.2f}%  "
            f"n={val_eval['argmin'].get('n', 0)}\n"
            f"[{cd['codec_name']}] student TRAIN argmin_acc={train_eval['argmin'].get('argmin_acc', 0):.1%}  "
            f"mean_overhead={train_eval['argmin'].get('mean_pct', 0):.2f}%  "
            f"(gap to val: {val_eval['argmin'].get('mean_pct', 0) - train_eval['argmin'].get('mean_pct', 0):+.2f}pp)\n"
        )
        all_val_metrics[cd["codec_name"]] = {
            "val": val_eval["argmin"], "train": train_eval["argmin"],
            "val_scalars": val_eval["scalars"],
        }
        out_path = args.out_dir / f"{args.out_name}_{cd['codec_name']}_picker.json"
        write_codec_json(
            cd, soft, model, union, n_codecs=len(codec_data),
            codec_idx=i,
            all_codec_names=[c["codec_name"] for c in codec_data],
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_, out_path=out_path,
            train_metrics={"argmin": train_eval["argmin"]},
            val_metrics={"argmin": val_eval["argmin"],
                         "scalars": val_eval["scalars"]},
        )

    summary = {
        "training": train_summary,
        "codecs": all_val_metrics,
        "trunk_hidden": list(hidden),
        "n_codecs": len(codec_data),
        "n_union_features": len(union["feat_cols"]),
        "n_inputs": n_inputs,
    }
    summary_path = args.out_dir / f"{args.out_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    sys.stderr.write(f"\nWrote summary → {summary_path}\n")

    # Joint bake JSON — one model encapsulating the whole multi-codec
    # picker (trunk + all codec heads concatenated) with the v3.2
    # multi_codec_schema block. External callers consume the resulting
    # .bin via Picker::predict_multi_codec; codec crates stay
    # zenanalyze/zenpredict-free.
    if args.joint_bake is not None:
        write_joint_bake_json(
            codec_data, soft_targets, model, union,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            out_path=args.joint_bake,
            per_codec_metrics=all_val_metrics,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
