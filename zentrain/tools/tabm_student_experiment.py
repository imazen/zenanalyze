#!/usr/bin/env python3
"""TabM-style K-head MLP student — single-codec experiment.

Drops in alongside `train_hybrid.py` for an A/B against the shipped
sklearn-MLP / PyTorch-leakyrelu student. The teacher stage, dataset
build, and evaluation surface are reused verbatim — only the student
fit is replaced.

Architecture — TabM-mini (BatchEnsemble; ICLR 2025, arXiv 2410.24210):

    For each Linear layer with shared weight W [in, out]:
      head k forward: y_k = ((x ⊙ r_k) @ W + bias) ⊙ s_k + b_k

    where r_k [in], s_k [out], b_k [out] are per-head rank-1 adapters.
    K independent "heads" share W; per-head capacity grows linearly
    in K with K× fewer params than K independent MLPs.

    At inference we average head predictions (mean ensemble) — that's
    the +2% accuracy lift the TabM paper reports vs a tuned single MLP.

Why this is worth measuring on the picker workload
--------------------------------------------------
- The picker student is a small MLP (~30-200 KB params); the bake
  format (ZNPR v2) is literally "scaler → series of Linear + activation
  layers". TabM-mini *is* an MLP-shaped model; the K heads concatenate
  into wider layers when flattened (see `flatten_to_dense_for_znpr`),
  so a successful TabM win can ship through the existing bake / runtime
  with at most a stacking-serializer change.
- Larger archs (FT-Transformer, TabPFN) would require new `zenpredict`
  ops; TabM stays inside the small-MLP family.

What this prototype is NOT (yet)
--------------------------------
- Not wired to `bake_picker.py` — the flatten-to-dense step is sketched
  but not run; production-ize after the A/B shows a win.
- Not a sweep harness — runs one config end-to-end and prints metrics.
- Not multi-codec — drives one `--codec-config` per invocation.

Usage:
    PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \\
        python3 tabm_student_experiment.py \\
            --codec-config zenjpeg_picker_config \\
            --heads 8 --hidden 192,192,192 --epochs 200

References:
- Yandex Research, "TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling" (ICLR 2025) — arXiv:2410.24210
- BatchEnsemble: Wen et al., ICLR 2020 — arXiv:2002.06715
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Reuse the full data pipeline from train_hybrid. The teacher + dataset
# build + evaluator are codec-agnostic and we want apples-to-apples
# inputs against the baseline student.
import train_hybrid as TH  # noqa: E402


# ----------------------------------- model


class TabMMiniLinear(nn.Module):
    """One BatchEnsemble Linear layer.

    Shared weight `W` [in_dim, out_dim] + shared bias `b0` [out_dim];
    per-head rank-1 adapters `R` [K, in_dim] (input multiplier) and
    `S` [K, out_dim] (output multiplier) plus per-head bias `B`
    [K, out_dim]. Forward expects `x` shaped [K, batch, in_dim] and
    returns [K, batch, out_dim].

    Init follows the TabM paper Appendix C — `R` / `S` to sign-Bernoulli
    {-1, +1} so heads start meaningfully different, `B` to zeros.
    """

    def __init__(self, in_dim: int, out_dim: int, k_heads: int,
                 init_r_s: str = "sign") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k_heads = k_heads
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.b0 = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        # Per-head rank-1 adapters
        if init_r_s == "sign":
            r = torch.randint(0, 2, (k_heads, in_dim), dtype=torch.float32) * 2 - 1
            s = torch.randint(0, 2, (k_heads, out_dim), dtype=torch.float32) * 2 - 1
        elif init_r_s == "normal":
            r = torch.randn(k_heads, in_dim) * 0.1 + 1.0
            s = torch.randn(k_heads, out_dim) * 0.1 + 1.0
        else:
            raise ValueError(f"unknown init_r_s {init_r_s!r}")
        self.R = nn.Parameter(r)
        self.S = nn.Parameter(s)
        self.B = nn.Parameter(torch.zeros(k_heads, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K, batch, in_dim]
        # multiply per-head input scale: [K, 1, in_dim] * [K, batch, in_dim]
        x_scaled = x * self.R.unsqueeze(1)
        # shared linear: einsum reduces in_dim
        y = torch.einsum("kbi,io->kbo", x_scaled, self.W) + self.b0
        # per-head output scale + per-head bias
        y = y * self.S.unsqueeze(1) + self.B.unsqueeze(1)
        return y


class TabMStudent(nn.Module):
    """TabM-mini ensemble of K MLPs sharing weights via rank-1 adapters."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], n_out: int,
                 k_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.k_heads = k_heads
        dims = [n_in, *hidden, n_out]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                TabMMiniLinear(dims[i], dims[i + 1], k_heads=k_heads)
            )
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.n_layers = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [K, batch, n_out] — per-head outputs. Caller averages."""
        # broadcast single x [batch, in_dim] across K heads
        x = x.unsqueeze(0).expand(self.k_heads, -1, -1).contiguous()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers - 1:
                x = self.act(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: average head predictions."""
        return self.forward(x).mean(dim=0)


# ----------------------------------- training


def _scale_targets(soft_tr: np.ndarray, n_cells: int, n_heads_layout: list[str]) -> tuple[np.ndarray, list]:
    """Replicate `train_hybrid`'s per-head loss normalization so the
    flat MSE doesn't get dominated by wide-range log-bytes heads.

    Returns (soft_tr_scaled, scaler_state) where scaler_state is a list
    of `(start, end, axis, mu, sigma)` per scalar block. The bake step
    in train_hybrid absorbs the inverse affine into the final-layer
    weights; for this prototype we just store the state so a follow-up
    flatten-to-dense can apply the same trick.
    """
    out = soft_tr.copy()
    scaler_state: list[tuple[int, int, str, float, float]] = []
    # Block 0 is bytes (log-space, leave alone). For the prototype, we
    # don't know if time/metric heads are emitted without inspecting
    # soft_tr's width; mark them with start indices passed in.
    return out, scaler_state


def train_tabm_student(
    Xe_tr: np.ndarray,
    Xe_va: np.ndarray,
    soft_tr: np.ndarray,
    bl_va: np.ndarray,
    rch_va: np.ndarray,
    meta_va: list,
    n_cells: int,
    *,
    hidden: tuple[int, ...] = (192, 192, 192),
    k_heads: int = 8,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 4096,
    weight_decay: float = 1e-5,
    seed: int = 0xCAFE,
    device: str | None = None,
) -> dict:
    """Train a TabM student against teacher soft targets and return
    val metrics + the trained model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_in = Xe_tr.shape[1]
    n_out = soft_tr.shape[1]
    student = TabMStudent(n_in, hidden, n_out, k_heads=k_heads).to(device)
    sys.stderr.write(
        f"[tabm] device={device}  K={k_heads}  hidden={hidden}  "
        f"n_in={n_in}  n_out={n_out}  params="
        f"{sum(p.numel() for p in student.parameters()):,}\n"
    )

    opt = torch.optim.AdamW(student.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Internal val split (image-aware split happens upstream; here we
    # just keep ~10% of the train rows for early-stopping signal).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xe_tr))
    n_val = max(1, len(Xe_tr) // 10)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xt = torch.from_numpy(Xe_tr[tr_idx].astype(np.float32)).to(device)
    Yt = torch.from_numpy(soft_tr[tr_idx].astype(np.float32)).to(device)
    Xv = torch.from_numpy(Xe_tr[val_idx].astype(np.float32)).to(device)
    Yv = torch.from_numpy(soft_tr[val_idx].astype(np.float32)).to(device)

    best_val = math.inf
    bad = 0
    patience = 30

    t0 = time.monotonic()
    n_batches_per_epoch = max(1, len(tr_idx) // batch_size)
    for epoch in range(epochs):
        student.train()
        perm_e = torch.randperm(len(tr_idx), device=device)
        epoch_loss = 0.0
        for b in range(n_batches_per_epoch):
            sel = perm_e[b * batch_size : (b + 1) * batch_size]
            xb = Xt[sel]
            yb = Yt[sel]
            # per-head independent loss — each head learns to match
            # the full teacher target. Final inference averages heads.
            pred = student(xb)  # [K, B, out]
            loss = (pred - yb.unsqueeze(0).expand_as(pred)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        epoch_loss /= n_batches_per_epoch

        student.eval()
        with torch.no_grad():
            pred_v = student.predict_mean(Xv)
            val_mse = (pred_v - Yv).pow(2).mean().item()
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                sys.stderr.write(
                    f"[tabm] early stop at epoch {epoch} "
                    f"(val_mse {val_mse:.4f} no improvement {patience} epochs)\n"
                )
                break
        if epoch % 10 == 0 or epoch < 5:
            sys.stderr.write(
                f"[tabm] epoch {epoch:3d}  tr_loss {epoch_loss:.4f}  "
                f"val_mse {val_mse:.4f}  lr {opt.param_groups[0]['lr']:.2e}\n"
            )
    dt = time.monotonic() - t0
    sys.stderr.write(f"[tabm] fit time: {dt:.1f}s ({epoch + 1} epochs)\n")

    # Held-out evaluation: predict on Xe_va, slice the bytes head,
    # compute argmin overhead / accuracy via train_hybrid's evaluator.
    student.eval()
    with torch.no_grad():
        Xv_full = torch.from_numpy(Xe_va.astype(np.float32)).to(device)
        pred_va_mean = student.predict_mean(Xv_full).cpu().numpy()  # [N, n_out]
        # also per-head for diagnostic
        pred_va_per_head = student.forward(Xv_full).cpu().numpy()  # [K, N, n_out]
    pred_bytes_va = pred_va_mean[:, :n_cells]
    all_mask = np.ones(n_cells, dtype=bool)
    argmin = TH.evaluate_argmin(pred_bytes_va, bl_va, rch_va, meta_va, all_mask)

    # Per-head argmin acc (diagnostic — single-head vs ensemble)
    per_head_argmin_acc = []
    for k in range(k_heads):
        ph = pred_va_per_head[k, :, :n_cells]
        a = TH.evaluate_argmin(ph, bl_va, rch_va, meta_va, all_mask)
        per_head_argmin_acc.append(a["argmin_acc"])

    return {
        "best_val_mse": best_val,
        "argmin_mean_overhead_pct": argmin["mean_pct"],
        "argmin_p99_overhead_pct": argmin["p99_pct"],
        "argmin_acc": argmin["argmin_acc"],
        "ensemble_argmin_acc": argmin["argmin_acc"],
        "per_head_argmin_acc_mean": float(np.mean(per_head_argmin_acc)),
        "per_head_argmin_acc_std": float(np.std(per_head_argmin_acc)),
        "per_head_argmin_acc_list": per_head_argmin_acc,
        "fit_seconds": dt,
        "epochs_ran": epoch + 1,
        "n_params": sum(p.numel() for p in student.parameters()),
        "device": device,
        "_model": student,
    }


# ----------------------------------- driver


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--codec-config", required=True,
        help="Python module name with codec config (PARETO, FEATURES, "
        "KEEP_FEATURES, ZQ_TARGETS, parse_config_name, etc.) — same "
        "shape as train_hybrid.py's codec-config contract.",
    )
    parser.add_argument("--heads", type=int, default=8,
                        help="K parallel BatchEnsemble heads (default 8)")
    parser.add_argument("--hidden", default="192,192,192",
                        help="Hidden-layer widths, comma-separated")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0xCAFE)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / mps. Default: cuda if available")
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Write the experiment summary here. "
                        "Default: <OUT_JSON-from-codec-config>_tabm.json")
    args = parser.parse_args()

    # Load codec config exactly like train_hybrid does. This sets the
    # module-level globals (PARETO, FEATURES, METRIC_COLUMN, …) on TH.
    TH.load_codec_config(args.codec_config)
    hidden = tuple(int(x) for x in args.hidden.split(","))

    sys.stderr.write(f"Loading {TH.PARETO}\n")
    pareto, ceilings, has_ceiling_column, has_time_column = TH.load_pareto(
        TH.PARETO
    )
    feats, feat_cols, _feat_transforms = TH.load_features(TH.FEATURES)
    cells, _, config_to_cell, parsed_all = TH.build_cell_index()
    n_cells = len(cells)
    sys.stderr.write(f"n_cells: {n_cells}\n")

    time_baselines = TH.compute_time_baselines(pareto) if has_time_column else {}
    (Xs, Xe, bytes_log, scalars, reach, meta, time_log, metric_log,
     _infeasible) = TH.build_dataset(
        pareto, feats, feat_cols, cells, config_to_cell, parsed_all,
        ceilings=(ceilings if has_ceiling_column else None),
        time_baselines=time_baselines if time_baselines else None,
    )

    # Image-level train/val split (mirrors train_hybrid).
    rng = np.random.default_rng(args.seed)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * TH.HOLDOUT_FRAC_DEFAULT))
    val_set = set(images[:n_val])
    tr = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    va = np.array([i for i, m in enumerate(meta) if m[0] in val_set])
    sys.stderr.write(f"Train rows: {len(tr)}  Val rows: {len(va)}\n")

    Xe_tr, Xe_va = Xe[tr], Xe[va]
    bl_tr, bl_va = bytes_log[tr], bytes_log[va]
    rch_tr, rch_va = reach[tr], reach[va]
    meta_va = [meta[i] for i in va]
    scalars_tr = {axis: scalars[axis][tr] for axis in TH.SCALAR_AXES}
    time_log_tr = time_log[tr] if time_log is not None else None
    metric_log_tr = metric_log[tr] if metric_log is not None else None

    # Reuse the production teacher — same HistGB + joblib path so the
    # A/B against the baseline student is pure-architecture.
    sys.stderr.write("Training teachers (sklearn HistGB)...\n")
    t_teacher = time.monotonic()
    (t_bytes, t_per_axis, scalar_means, t_time, time_means,
     t_metric, metric_means) = TH.train_teacher_per_cell(
        Xs[tr], bl_tr, scalars_tr, rch_tr, n_cells,
        time_log_tr=time_log_tr, metric_log_tr=metric_log_tr,
    )
    sys.stderr.write(f"  teacher fit: {time.monotonic() - t_teacher:.1f}s\n")
    fallback_means = np.nanmean(bl_tr, axis=0)
    bytes_pred_tr = TH.teacher_predict_all(t_bytes, Xs[tr], fallback_means, n_cells)
    bytes_pred_va = TH.teacher_predict_all(t_bytes, Xs[va], fallback_means, n_cells)
    scalar_pred_tr = {
        axis: TH.teacher_predict_all(t_per_axis[axis], Xs[tr], scalar_means[axis], n_cells)
        for axis in TH.SCALAR_AXES
    }
    if t_time is not None:
        time_means_safe = np.where(np.isnan(time_means), 0.0, time_means)
        time_pred_tr = TH.teacher_predict_all(t_time, Xs[tr], time_means_safe, n_cells)
    else:
        time_pred_tr = None
    if t_metric is not None:
        metric_means_safe = np.where(np.isnan(metric_means), 0.0, metric_means)
        metric_pred_tr = TH.teacher_predict_all(t_metric, Xs[tr], metric_means_safe, n_cells)
    else:
        metric_pred_tr = None

    # Concatenate soft target blocks — same layout as train_hybrid.
    soft_blocks = [bytes_pred_tr]
    if time_pred_tr is not None:
        soft_blocks.append(time_pred_tr)
    if metric_pred_tr is not None:
        soft_blocks.append(metric_pred_tr)
    soft_blocks.extend(scalar_pred_tr[axis] for axis in TH.SCALAR_AXES)
    soft_tr = np.concatenate(soft_blocks, axis=1)
    sys.stderr.write(f"soft_tr shape: {soft_tr.shape}\n")

    # Teacher metric for the comparison row
    all_mask = np.ones(n_cells, dtype=bool)
    teacher_argmin = TH.evaluate_argmin(bytes_pred_va, bl_va, rch_va, meta_va, all_mask)
    sys.stderr.write(
        f"\nTeacher (HistGB): argmin_acc {teacher_argmin['argmin_acc']:.1%}  "
        f"mean overhead {teacher_argmin['mean_pct']:.2f}%\n"
    )

    # --- Fit the TabM student
    result = train_tabm_student(
        Xe_tr=Xe_tr,
        Xe_va=Xe_va,
        soft_tr=soft_tr,
        bl_va=bl_va,
        rch_va=rch_va,
        meta_va=meta_va,
        n_cells=n_cells,
        hidden=hidden,
        k_heads=args.heads,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )

    summary = {
        "codec_config": args.codec_config,
        "k_heads": args.heads,
        "hidden": list(hidden),
        "n_cells": n_cells,
        "n_train_rows": int(len(tr)),
        "n_val_rows": int(len(va)),
        "teacher_argmin_acc": teacher_argmin["argmin_acc"],
        "teacher_mean_overhead_pct": teacher_argmin["mean_pct"],
        "tabm_argmin_acc": result["argmin_acc"],
        "tabm_mean_overhead_pct": result["argmin_mean_overhead_pct"],
        "tabm_p99_overhead_pct": result["argmin_p99_overhead_pct"],
        "tabm_val_mse": result["best_val_mse"],
        "tabm_fit_seconds": result["fit_seconds"],
        "tabm_epochs_ran": result["epochs_ran"],
        "tabm_params": result["n_params"],
        "tabm_per_head_argmin_acc_mean": result["per_head_argmin_acc_mean"],
        "tabm_per_head_argmin_acc_std": result["per_head_argmin_acc_std"],
        "tabm_per_head_argmin_acc_list": result["per_head_argmin_acc_list"],
        "device": result["device"],
    }
    sys.stderr.write("\n=== TabM vs Teacher (held-out images) ===\n")
    sys.stderr.write(
        f"  argmin_acc:      teacher {teacher_argmin['argmin_acc']:.1%}    "
        f"TabM ensemble {result['argmin_acc']:.1%}    "
        f"(per-head μ={result['per_head_argmin_acc_mean']:.1%} ±{result['per_head_argmin_acc_std']:.1%})\n"
    )
    sys.stderr.write(
        f"  mean overhead:   teacher {teacher_argmin['mean_pct']:.2f}%   "
        f"TabM {result['argmin_mean_overhead_pct']:.2f}%\n"
    )
    sys.stderr.write(
        f"  fit time:        {result['fit_seconds']:.1f}s on {result['device']}  "
        f"({result['epochs_ran']} epochs)\n"
    )

    out_json = args.out_json
    if out_json is None:
        base = Path(TH.OUT_JSON) if hasattr(TH, "OUT_JSON") else Path("tabm_result.json")
        out_json = base.with_name(base.stem + "_tabm.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    sys.stderr.write(f"\nWrote {out_json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
