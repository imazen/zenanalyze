#!/usr/bin/env python3
"""Multi-codec joint training → per-codec distillation.

Runs `train_multi_codec`'s shared-trunk pipeline, then for each codec
fits a per-codec MLP student against the joint model's predictions on
that codec's *natural* (engineered, Xe) feature schema. Emits per-codec
JSON files in `train_hybrid.py`'s exact output shape so `bake_picker.py`
consumes them unchanged — no runtime changes anywhere in zenpredict /
zenpicker.

Why this exists
---------------
The joint shared-trunk model needs a multi-codec-aware runtime (union
feature vector + presence mask + codec onehot). Per the 2026-05-17
"value optimality for codec configuration" call: measure how much
accuracy we lose when distilling the joint model back to per-codec
students that ship through the existing ZNPR v3 runtime. If the
distilled students recover most of the joint win, ship distilled — no
runtime change. If the gap is too wide, we'll wire a `Picker::
predict_multi_codec` helper into zenpredict and ship the joint model
directly.

Architecture
------------
- Phase 1-6: identical to `train_multi_codec.main()` (per-codec data +
  teachers, union schema, trunk input, joint training, codec-balanced
  AdamW, per-codec evaluation).
- Phase 7 (this script's addition):
    a. For each codec, forward all rows (train+val) through the joint
       model to get teacher predictions in standardized output space.
    b. Inverse-standardize scalar blocks → natural-space teacher labels
       (same convention as `train_hybrid`'s post-fit absorption).
    c. Train a per-codec PyTorch MLP on
       `(Xe_codec, joint_teacher_predictions_for_codec)` with
       per-head standardization (matching `train_hybrid`'s student).
    d. Inverse-standardize at write time so the JSON's `coefs_` /
       `intercepts_` produce natural-unit outputs (matches what
       `bake_picker.py` expects).
    e. Evaluate val argmin acc + mean overhead on the same val split.
    f. Emit per-codec JSON in `train_hybrid.py:write_distill_json`
       shape: feat_cols (codec's natural), extra_axes (cross-terms +
       size onehot + log_px + zq_norm + icc placeholder), n_inputs,
       n_outputs, scaler_mean/scale, layers, hybrid_heads_manifest,
       output_specs, sparse_overrides.

Usage
-----
Same CLI surface as `train_multi_codec.py` plus distill knobs:

    PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \\
        python3 zentrain/tools/distill_multi_to_per_codec.py \\
            --codec zenjpeg=examples/zenjpeg_picker_config.py:/home/lilith/work/zen/zenjpeg \\
            --codec zenwebp=examples/zenwebp_picker_config.py:/home/lilith/work/zen/zenwebp \\
            --codec zenavif=examples/zenavif_picker_config.py:/home/lilith/work/zen/zenavif \\
            --out-dir /home/lilith/work/zen/zenanalyze/benchmarks/multi_codec_2026-05-17 \\
            --hidden 96,48 --epochs 200 \\
            --student-hidden 192,192,192 --student-epochs 200

The output dir gets `{prefix}_{codec}_picker.json` for each codec
(distilled student in bake_picker shape), plus `{prefix}_distill_summary.md`
comparing single-codec baseline (= per-codec teacher-only) vs joint
direct vs distilled per-codec.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Reuse all the joint-training plumbing.
import train_multi_codec as TMC
import train_hybrid as TH


# ----------------------------------------------------------------------
# Per-codec distilled student
# ----------------------------------------------------------------------

class DistilledStudent(nn.Module):
    """Single-codec MLP — same shape as `train_hybrid`'s leakyrelu student.

    Trained on the joint model's predictions for one codec (natural-space
    teacher labels). Outputs per-block standardized values during training;
    we absorb the inverse affine into the final-layer weights before
    serializing so the exported MLP produces natural-unit outputs and
    `bake_picker.py` consumes it unchanged.
    """

    def __init__(self, n_in: int, hidden_sizes: tuple[int, ...], n_out: int,
                 leaky_slope: float = 0.01, seed: int = 0xCAFE) -> None:
        super().__init__()
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
        self.hidden_sizes = hidden_sizes
        self.n_in = n_in
        self.n_out = n_out
        self.leaky_slope = leaky_slope

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


def _export_distilled(student: DistilledStudent) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Walk the Sequential, return (coefs_, intercepts_) in sklearn convention
    (W: in × out)."""
    coefs: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []
    for m in student.net:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy().T.astype(np.float64)
            b = m.bias.detach().cpu().numpy().astype(np.float64)
            coefs.append(W)
            intercepts.append(b)
    return coefs, intercepts


# ----------------------------------------------------------------------
# Build per-codec distillation labels from the joint model
# ----------------------------------------------------------------------

def joint_predictions_for_codec(
    model: TMC.SharedTrunkMLP,
    X_codec_scaled: np.ndarray,
    cd: dict,
    soft_pkg: dict,
) -> np.ndarray:
    """Forward the codec's full rows through the joint model, then
    invert the soft-target standardization (per-block μ/σ from
    `build_soft_targets`) so the result is in natural units:
      [bytes_log (n_cells) | scalar_axis_1 (n_cells) | ...]
    """
    Y_std = TMC._predict_codec(model, X_codec_scaled, cd["codec_name"])
    Y_nat = Y_std.copy()
    for (start, end, axis, mu, sigma) in soft_pkg["scalar_block_starts"]:
        if sigma == 0.0 and mu == 0.0:
            continue
        Y_nat[:, start:end] = Y_nat[:, start:end] * sigma + mu
    return Y_nat


# ----------------------------------------------------------------------
# Train distilled student (same shape as train_hybrid's leakyrelu path)
# ----------------------------------------------------------------------

def train_distilled_student(
    Xe_tr: np.ndarray,
    Xe_va: np.ndarray,
    Y_tr_nat: np.ndarray,
    Y_va_nat: np.ndarray,
    n_cells: int,
    scalar_axes: list[str],
    *,
    hidden_sizes: tuple[int, ...] = (192, 192, 192),
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 4096,
    epochs: int = 200,
    n_iter_no_change: int = 30,
    seed: int = 0xCAFE,
    device: str | None = None,
) -> dict:
    """Distill on (Xe, joint-teacher-labels). Mirrors train_hybrid's
    `_train_torch_leakyrelu_student` shape: PyTorch leakyrelu MLP,
    AdamW, cosine schedule, early stop on val MSE. Per-head
    standardization for scalar blocks (bytes_log block stays log-space).
    Inverse affine is absorbed into the final-layer weights at the end
    so the exported MLP produces natural-unit outputs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)

    # Per-block standardization on the labels (input is the codec's
    # natural Xe — we let the bake-side scaler handle that).
    n_blocks = 1 + len(scalar_axes)  # bytes + each scalar
    block_starts: list[tuple[int, int, str, float, float]] = []
    Y_tr = Y_tr_nat.copy()
    Y_va = Y_va_nat.copy()  # kept natural; only Y_tr is standardized for fit
    # bytes block (block 0) is left in log-space natural units — same as
    # train_hybrid. Standardize scalar blocks only.
    for bi, axis in enumerate(scalar_axes, start=1):
        start = bi * n_cells
        end = (bi + 1) * n_cells
        block = Y_tr[:, start:end]
        mu = float(np.nanmean(block))
        sigma = float(np.nanstd(block))
        if sigma < 1e-12:
            block_starts.append((start, end, axis, 0.0, 1.0))
        else:
            Y_tr[:, start:end] = (block - mu) / sigma
            block_starts.append((start, end, axis, mu, sigma))

    student = DistilledStudent(Xe_tr.shape[1], hidden_sizes, Y_tr.shape[1],
                               seed=seed).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.from_numpy(Xe_tr.astype(np.float32)).to(device)
    Yt = torch.from_numpy(Y_tr.astype(np.float32)).to(device)
    Xv = torch.from_numpy(Xe_va.astype(np.float32)).to(device)
    Yv_nat = torch.from_numpy(Y_va_nat.astype(np.float32)).to(device)

    # For internal early stop, build a standardized val target using the
    # same per-block (μ, σ) we computed from train labels.
    Yv_std = Y_va_nat.copy()
    for (start, end, axis, mu, sigma) in block_starts:
        if sigma > 0.0:
            Yv_std[:, start:end] = (Yv_std[:, start:end] - mu) / sigma
    Yv = torch.from_numpy(Yv_std.astype(np.float32)).to(device)

    n = len(Xt)
    best_val = math.inf
    best_state: dict | None = None
    bad = 0

    t0 = time.monotonic()
    epoch = 0
    for epoch in range(epochs):
        student.train()
        perm = torch.randperm(n, device=device)
        n_batches = max(1, n // batch_size)
        epoch_loss = 0.0
        for b in range(n_batches):
            sel = perm[b * batch_size:(b + 1) * batch_size]
            pred = student(Xt[sel])
            # MSE over the standardized output. NaN-aware mask: any row
            # with a NaN label dropped from that head's contribution.
            target = Yt[sel]
            mask = torch.isfinite(target)
            diff = torch.where(mask, pred - target, torch.zeros_like(pred))
            denom = mask.sum().clamp(min=1)
            loss = (diff * diff).sum() / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        epoch_loss /= n_batches

        student.eval()
        with torch.no_grad():
            pred_v = student(Xv)
            mask_v = torch.isfinite(Yv)
            diff_v = torch.where(mask_v, pred_v - Yv, torch.zeros_like(pred_v))
            denom_v = mask_v.sum().clamp(min=1)
            val_mse = ((diff_v * diff_v).sum() / denom_v).item()

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            bad = 0
            best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
        else:
            bad += 1
            if bad >= n_iter_no_change:
                sys.stderr.write(
                    f"  [distill] early stop at epoch {epoch} "
                    f"(val_mse {val_mse:.4f} no improvement {n_iter_no_change} epochs)\n"
                )
                break
        if epoch % 20 == 0 or epoch < 3:
            sys.stderr.write(
                f"  [distill] epoch {epoch:3d}  tr_loss {epoch_loss:.4f}  "
                f"val_mse {val_mse:.4f}  lr {opt.param_groups[0]['lr']:.2e}\n"
            )

    elapsed = time.monotonic() - t0
    if best_state is not None:
        student.load_state_dict(best_state)
    sys.stderr.write(
        f"  [distill] fit done in {elapsed:.1f}s ({epoch + 1} epochs), "
        f"best val_mse {best_val:.4f}\n"
    )

    # Final eval on val in NATURAL space — invert the per-block
    # standardization on the student's predictions.
    student.eval()
    with torch.no_grad():
        pred_v_std = student(Xv).cpu().numpy()
    pred_v_nat = pred_v_std.copy()
    for (start, end, axis, mu, sigma) in block_starts:
        if sigma == 0.0 and mu == 0.0:
            continue
        pred_v_nat[:, start:end] = pred_v_nat[:, start:end] * sigma + mu

    # Absorb the inverse-standardization into the final-layer weights
    # so the exported MLP produces natural-unit outputs. Match the
    # convention used in train_hybrid's main():
    #   coefs_[-1][:, i] *= sigma; intercepts_[-1][i] = old * sigma + mu
    coefs, intercepts = _export_distilled(student)
    last_W = coefs[-1].copy()
    last_b = intercepts[-1].copy()
    for (start, end, axis, mu, sigma) in block_starts:
        if sigma == 0.0 and mu == 0.0:
            continue
        last_W[:, start:end] = last_W[:, start:end] * sigma
        last_b[start:end] = last_b[start:end] * sigma + mu
    coefs[-1] = last_W
    intercepts[-1] = last_b

    return {
        "coefs_": coefs,
        "intercepts_": intercepts,
        "elapsed_s": elapsed,
        "epochs_ran": epoch + 1,
        "best_val_mse": best_val,
        "pred_v_natural": pred_v_nat,
        "block_starts": block_starts,
        "hidden_sizes": list(hidden_sizes),
    }


# ----------------------------------------------------------------------
# Emit per-codec JSON in train_hybrid output shape
# ----------------------------------------------------------------------

def write_distilled_codec_json(
    cd: dict,
    coefs: list[np.ndarray],
    intercepts: list[np.ndarray],
    val_argmin: dict,
    train_argmin: dict | None,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    extra_axes: list[str],
    out_path: Path,
    distill_meta: dict,
) -> None:
    """Per-codec JSON matching train_hybrid's distill-output shape so
    bake_picker.py consumes it without changes.

    The codec runtime sees a plain MLP that takes the codec's natural
    feature vector (engineered Xe = feat_cols + cross terms + size_oh +
    log_px + zq_norm + icc) and outputs `(1 + len(scalar_axes)) ×
    n_cells` floats.
    """
    feat_cols = cd["feat_cols"]
    n_cells = cd["n_cells"]
    scalar_axes = cd["scalar_axes"]
    n_inputs = coefs[0].shape[0]
    n_outputs = coefs[-1].shape[1]
    assert n_outputs == (1 + len(scalar_axes)) * n_cells, (
        f"output dim mismatch: coefs[-1].shape[1]={n_outputs}, "
        f"expected (1+{len(scalar_axes)})*{n_cells}={(1 + len(scalar_axes)) * n_cells}"
    )

    # Hybrid-heads manifest: list of (codec config, scalar values) per
    # cell, plus output layout block boundaries.
    cells_meta = []
    for c in cd["cells"]:
        cells_meta.append({
            "id": c["id"],
            "label": c["label"],
            "key": list(c["key"]),
            "member_config_ids": list(c["member_config_ids"]),
        })

    output_layout = {
        "bytes_log": {"start": 0, "end": n_cells},
    }
    for bi, axis in enumerate(scalar_axes, start=1):
        output_layout[axis] = {"start": bi * n_cells, "end": (bi + 1) * n_cells}

    payload = {
        "n_inputs": int(n_inputs),
        "n_outputs": int(n_outputs),
        "feat_cols": list(feat_cols),
        "extra_axes": list(extra_axes),
        "scaler_mean": scaler_mean.tolist(),
        "scaler_scale": scaler_scale.tolist(),
        "layers": [
            {
                "W": W.tolist(),
                "b": b.tolist(),
                "activation": ("leakyrelu" if i < len(coefs) - 1 else "identity"),
            }
            for i, (W, b) in enumerate(zip(coefs, intercepts))
        ],
        "hybrid_heads_manifest": {
            "n_cells": n_cells,
            "scalar_axes": list(scalar_axes),
            "cells": cells_meta,
            "output_layout": output_layout,
        },
        "metrics": {
            "val_argmin": val_argmin,
            "train_argmin": train_argmin,
            "distill": distill_meta,
        },
        "source": "distill_multi_to_per_codec",
    }
    out_path.write_text(json.dumps(payload, indent=2))


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------

def evaluate_predictions(
    cd: dict,
    pred_nat: np.ndarray,
    split_idx: np.ndarray,
) -> dict:
    """Run argmin / scalar evals on pre-computed natural-space predictions
    for one codec on the given split's rows."""
    n_cells = cd["n_cells"]
    meta_sel = [cd["meta"][i] for i in split_idx]
    bl_sel = cd["bytes_log"][split_idx]
    rch_sel = cd["reach"][split_idx]
    scalars_sel = {ax: cd["scalars"][ax][split_idx] for ax in cd["scalar_axes"]}
    pred_bytes = pred_nat[:, :n_cells]
    scalar_preds = {}
    for bi, axis in enumerate(cd["scalar_axes"], start=1):
        scalar_preds[axis] = pred_nat[:, bi * n_cells:(bi + 1) * n_cells]
    all_mask = np.ones(n_cells, dtype=bool)
    TH.SCALAR_AXES = cd["scalar_axes"]
    TH.SCALAR_SENTINELS = cd["scalar_sentinels"]
    argmin = TH.evaluate_argmin(pred_bytes, bl_sel, rch_sel, meta_sel, all_mask)
    scalars = TH.evaluate_scalars(scalar_preds, scalars_sel, rch_sel)
    return {"argmin": argmin, "scalars": scalars}


# ----------------------------------------------------------------------
# Single-codec baseline (no multi-codec at all — for comparison)
# ----------------------------------------------------------------------

def teacher_only_baseline(cd: dict) -> dict:
    """Per-codec baseline: just the HistGB teacher's predictions on val.
    This is what `train_hybrid.py` would report before the MLP student
    step, and is a tight lower bound on what any sensible student
    achieves on this codec alone.
    """
    n_cells = cd["n_cells"]
    val_idx = cd["val_idx"]
    Xs_va = cd["Xs"][val_idx]
    fallback = np.nanmean(cd["bytes_log"][cd["train_idx"]], axis=0)
    pred_bytes_va = TH.teacher_predict_all(
        cd["teachers_bytes"], Xs_va, fallback, n_cells,
    )
    scalar_preds_va: dict[str, np.ndarray] = {}
    for axis in cd["scalar_axes"]:
        scalar_preds_va[axis] = TH.teacher_predict_all(
            cd["teachers_per_axis"][axis], Xs_va,
            cd["scalar_means"][axis], n_cells,
        )
    meta_sel = [cd["meta"][i] for i in val_idx]
    bl_sel = cd["bytes_log"][val_idx]
    rch_sel = cd["reach"][val_idx]
    scalars_sel = {ax: cd["scalars"][ax][val_idx] for ax in cd["scalar_axes"]}
    all_mask = np.ones(n_cells, dtype=bool)
    TH.SCALAR_AXES = cd["scalar_axes"]
    TH.SCALAR_SENTINELS = cd["scalar_sentinels"]
    argmin = TH.evaluate_argmin(pred_bytes_va, bl_sel, rch_sel, meta_sel, all_mask)
    scalars = TH.evaluate_scalars(scalar_preds_va, scalars_sel, rch_sel)
    return {"argmin": argmin, "scalars": scalars}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--codec", action="append", type=TMC._parse_codec_arg, required=True,
        help="Repeatable. Format: name=config_path:data_root",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="distill",
                        help="Prefix for per-codec JSONs (default 'distill')")
    parser.add_argument("--hidden", default="96,48",
                        help="Trunk hidden widths (joint model)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Joint training epochs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0xCAFE)
    parser.add_argument("--student-hidden", default="192,192,192",
                        help="Per-codec distilled student hidden widths")
    parser.add_argument("--student-epochs", type=int, default=200)
    parser.add_argument("--student-batch-size", type=int, default=4096)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--student-weight-decay", type=float, default=1e-5)
    args = parser.parse_args()

    trunk_hidden = tuple(int(x) for x in args.hidden.split(","))
    student_hidden = tuple(int(x) for x in args.student_hidden.split(","))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Phases 1-3 from train_multi_codec.main() --------
    bundles = [
        TMC.CodecConfigBundle(name, cfg, data_root)
        for name, cfg, data_root in args.codec
    ]
    codec_data: list[dict] = []
    for b in bundles:
        codec_data.append(TMC.extract_codec_dataset(b))

    union = TMC.build_union_schema(codec_data)
    trunk_input = TMC.build_trunk_input(codec_data, union, n_codecs=len(codec_data))

    soft_targets: list[dict] = []
    for cd in codec_data:
        soft_targets.append(TMC.build_soft_targets(cd))

    # -------- Phase 4: shared input scaler --------
    from sklearn.preprocessing import StandardScaler
    train_X_all = []
    for cd, Xc in zip(codec_data, trunk_input["per_codec_X"]):
        train_X_all.append(Xc[cd["train_idx"]])
    train_X_concat = np.concatenate(train_X_all, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_X_concat)
    scaled_train, scaled_val, scaled_full = [], [], []
    for cd, Xc in zip(codec_data, trunk_input["per_codec_X"]):
        scaled_train.append(scaler.transform(Xc[cd["train_idx"]]))
        scaled_val.append(scaler.transform(Xc[cd["val_idx"]]))
        scaled_full.append(scaler.transform(Xc))

    # -------- Phase 5: joint shared-trunk training --------
    head_dims = {
        cd["codec_name"]: soft_targets[i]["output_dim"]
        for i, cd in enumerate(codec_data)
    }
    n_inputs = trunk_input["n_inputs"]
    sys.stderr.write(
        f"\n[joint] trunk inputs={n_inputs} hidden={trunk_hidden} heads={head_dims}\n"
    )
    model = TMC.SharedTrunkMLP(n_inputs, trunk_hidden, head_dims, seed=args.seed)

    payloads = []
    for cd, soft, X_tr_s, X_va_s in zip(codec_data, soft_targets, scaled_train, scaled_val):
        payloads.append({
            "codec": cd["codec_name"],
            "X_tr": X_tr_s,
            "soft_tr": soft["soft_tr"],
            "X_va": X_va_s,
            "soft_va": soft["soft_va"],
        })
    joint_summary = TMC.train_shared_trunk(
        model, payloads,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, epochs=args.epochs,
        seed=args.seed,
    )

    # -------- Phase 6: joint-direct evaluation --------
    joint_metrics: dict[str, dict] = {}
    for i, (cd, soft, X_va_s) in enumerate(zip(codec_data, soft_targets, scaled_val)):
        ev = TMC.evaluate_codec(model, cd, soft, X_va_s, split="val")
        joint_metrics[cd["codec_name"]] = ev["argmin"]
        TMC._bind_globals_for_codec(cd["bundle"])

    # -------- Phase 7: per-codec distillation --------
    distill_metrics: dict[str, dict] = {}
    baseline_metrics: dict[str, dict] = {}
    for i, (cd, soft, X_full_s) in enumerate(zip(codec_data, soft_targets, scaled_full)):
        codec = cd["codec_name"]
        sys.stderr.write(f"\n=== [{codec}] distill ===\n")
        TMC._bind_globals_for_codec(cd["bundle"])

        # Baseline: teacher-only val argmin (no MLP student at all).
        baseline = teacher_only_baseline(cd)
        baseline_metrics[codec] = baseline["argmin"]

        # Joint predictions on ALL rows (train+val) in natural-space.
        Y_joint = joint_predictions_for_codec(model, X_full_s, cd, soft)
        Y_tr_nat = Y_joint[cd["train_idx"]]
        Y_va_nat = Y_joint[cd["val_idx"]]

        Xe_tr = cd["Xe"][cd["train_idx"]]
        Xe_va = cd["Xe"][cd["val_idx"]]

        # Per-codec input scaler — match train_hybrid: standardize Xe.
        student_scaler = StandardScaler()
        Xe_tr_s = student_scaler.fit_transform(Xe_tr)
        Xe_va_s = student_scaler.transform(Xe_va)

        distill_result = train_distilled_student(
            Xe_tr_s, Xe_va_s, Y_tr_nat, Y_va_nat,
            n_cells=cd["n_cells"],
            scalar_axes=cd["scalar_axes"],
            hidden_sizes=student_hidden,
            lr=args.student_lr,
            weight_decay=args.student_weight_decay,
            batch_size=args.student_batch_size,
            epochs=args.student_epochs,
            seed=args.seed,
        )
        # Val argmin in natural space.
        ev = evaluate_predictions(cd, distill_result["pred_v_natural"], cd["val_idx"])
        distill_metrics[codec] = ev["argmin"]

        # Extra axes for the bake JSON — matches train_hybrid's
        # engineered Xe layout: log_px, log_px², zq_norm, zq_norm²,
        # zq_norm × log_px, then per-feat zq_norm cross-terms, then icc.
        extra_axes = (
            ["size_tiny", "size_small", "size_medium", "size_large",
             "log_px", "log_px_sq", "zq_norm", "zq_norm_sq",
             "zq_norm_x_log_px"]
            + [f"zq_norm_x_{c}" for c in cd["feat_cols"]]
            + ["icc_bytes"]
        )
        out_path = args.out_dir / f"{args.out_prefix}_{codec}_picker.json"
        write_distilled_codec_json(
            cd,
            coefs=distill_result["coefs_"],
            intercepts=distill_result["intercepts_"],
            val_argmin=ev["argmin"],
            train_argmin=None,
            scaler_mean=student_scaler.mean_,
            scaler_scale=student_scaler.scale_,
            extra_axes=extra_axes,
            out_path=out_path,
            distill_meta={
                "hidden_sizes": distill_result["hidden_sizes"],
                "epochs_ran": distill_result["epochs_ran"],
                "fit_seconds": distill_result["elapsed_s"],
                "best_val_mse_standardized": distill_result["best_val_mse"],
                "joint_trunk_hidden": list(trunk_hidden),
                "joint_n_codecs": len(codec_data),
            },
        )
        sys.stderr.write(
            f"[{codec}] WROTE {out_path}\n"
            f"   teacher only: argmin_acc={baseline['argmin'].get('argmin_acc', 0):.1%} "
            f"mean={baseline['argmin'].get('mean_pct', 0):.2f}%\n"
            f"   joint direct: argmin_acc={joint_metrics[codec].get('argmin_acc', 0):.1%} "
            f"mean={joint_metrics[codec].get('mean_pct', 0):.2f}%\n"
            f"   distilled:    argmin_acc={ev['argmin'].get('argmin_acc', 0):.1%} "
            f"mean={ev['argmin'].get('mean_pct', 0):.2f}%\n"
        )

    # -------- Phase 8: comparison report --------
    md_path = args.out_dir / f"{args.out_prefix}_summary.md"
    lines = [
        f"# Multi-codec joint → per-codec distill ({len(codec_data)} codecs)",
        "",
        f"Generated by `distill_multi_to_per_codec.py` "
        f"({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}). ",
        f"Joint trunk: {list(trunk_hidden)}, distill student: "
        f"{list(student_hidden)}.",
        "",
        "## Held-out val argmin accuracy + mean overhead",
        "",
        "| Codec | Teacher only | Joint direct | Distilled | "
        "Joint vs Teacher | Distilled vs Joint |",
        "|---|---|---|---|---|---|",
    ]
    for cd in codec_data:
        c = cd["codec_name"]
        tea = baseline_metrics[c]
        joi = joint_metrics[c]
        dis = distill_metrics[c]
        delta_jt = joi.get("argmin_acc", 0) - tea.get("argmin_acc", 0)
        delta_dj = dis.get("argmin_acc", 0) - joi.get("argmin_acc", 0)
        lines.append(
            f"| {c} | "
            f"{tea.get('argmin_acc', 0):.1%} / {tea.get('mean_pct', 0):.2f}% | "
            f"{joi.get('argmin_acc', 0):.1%} / {joi.get('mean_pct', 0):.2f}% | "
            f"{dis.get('argmin_acc', 0):.1%} / {dis.get('mean_pct', 0):.2f}% | "
            f"{delta_jt:+.1%} acc | {delta_dj:+.1%} acc |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Joint vs Teacher** is the headline accuracy lift from joint "
        "training. Positive = joint training found cross-codec signal that "
        "helped this codec.",
        "- **Distilled vs Joint** is the loss from distillation. ≥0 = "
        "distillation captured all the joint win, can ship distilled with "
        "no runtime change. Negative = some joint signal didn't make it "
        "through the per-codec schema.",
        "",
        "- If `Distilled vs Joint` is within −1 pp of 0 for each codec: "
        "ship distilled. No runtime change.",
        "- If gap > 1-2 pp on a codec that matters: wire the "
        "`Picker::predict_multi_codec(codec_id, codec_features)` helper "
        "into `zenpredict` and ship the joint model directly.",
        "",
        "Per-codec JSONs in this directory match `bake_picker.py`'s "
        "single-codec input shape — drop into the corresponding codec "
        "crate's `include_bytes!` exactly like the existing v2.1 bakes.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    sys.stderr.write(f"\nWrote {md_path}\n")

    summary_json = {
        "joint_training": joint_summary,
        "metrics": {
            c: {
                "teacher_only": baseline_metrics[c],
                "joint_direct": joint_metrics[c],
                "distilled": distill_metrics[c],
            }
            for c in [cd["codec_name"] for cd in codec_data]
        },
        "trunk_hidden": list(trunk_hidden),
        "student_hidden": list(student_hidden),
        "n_codecs": len(codec_data),
        "n_union_features": len(union["feat_cols"]),
    }
    summary_json_path = args.out_dir / f"{args.out_prefix}_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2))
    sys.stderr.write(f"Wrote {summary_json_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
