#!/usr/bin/env python3
"""
LeakyReLU vs ReLU comparison for the zenpicker hybrid-heads MLP.

Trains two PyTorch students on the same image-level holdout split as
`train_hybrid.py` (HOLDOUT_FRAC=0.20, SEED=0xCAFE), one with ReLU
hidden activations and one with LeakyReLU(alpha=0.01). Reports val
argmin mean overhead, argmin accuracy, and dead-neuron rate for each.

This is a *student-only* experiment — no teacher distillation, both
students learn directly from the raw bytes_log + scalar targets.
That makes the comparison narrower than the production train_hybrid
pipeline (which uses HistGradientBoostingRegressor as the teacher),
but answers the immediate "does the activation choice matter on
this task?" question apples-to-apples.

Output: writes a comparison report to
`/mnt/v/output/zenpicker/leakyrelu_vs_relu_<date>.md` and prints
the headline numbers to stdout.

Usage:
    python3 tools/leakyrelu_experiment.py \
        --codec-config zenjpeg_picker_config \
        --hidden 192,192,192 \
        --epochs 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "zenpicker" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "zenpicker" / "examples"))

# Import everything we need from train_hybrid as a library.
import train_hybrid as th  # noqa: E402

SEED = 0xCAFE
HOLDOUT_FRAC = 0.20


class HybridMLP(nn.Module):
    """Hybrid-heads MLP with configurable hidden activation."""

    def __init__(
        self, n_inputs: int, hidden: tuple[int, ...], n_outputs: int, activation: str
    ):
        super().__init__()
        layers = []
        prev = n_inputs
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if activation == "leakyrelu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_student(
    Xe_tr: np.ndarray,
    Y_tr: np.ndarray,
    Xe_va: np.ndarray,
    Y_va: np.ndarray,
    hidden: tuple[int, ...],
    activation: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[HybridMLP, list[float]]:
    """Train one student. Returns (model, val_loss_history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    Xt = torch.from_numpy(Xe_tr.astype(np.float32)).to(device)
    Yt = torch.from_numpy(Y_tr.astype(np.float32)).to(device)
    Xv = torch.from_numpy(Xe_va.astype(np.float32)).to(device)
    Yv = torch.from_numpy(Y_va.astype(np.float32)).to(device)
    n_in = Xe_tr.shape[1]
    n_out = Y_tr.shape[1]
    model = HybridMLP(n_in, hidden, n_out, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    n_train = Xt.shape[0]
    val_loss_history: list[float] = []
    best_val = float("inf")
    bad_epochs = 0
    patience = 15
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv), Yv).item()
        val_loss_history.append(v)
        if v < best_val - 1e-6:
            best_val = v
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            sys.stderr.write(
                f"    [{activation}] early stop ep={ep + 1} val_loss={v:.4f}\n"
            )
            break
    return model, val_loss_history


def predict_np(model: HybridMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32))).numpy()


def dead_neuron_rate(
    model: HybridMLP, X: np.ndarray, activation: str, n_probes: int = 1000
) -> dict:
    """Fraction of hidden neurons that never fire on a sample of inputs."""
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(n_probes, X.shape[0]), replace=False)
    Xs = torch.from_numpy(X[idx].astype(np.float32))
    layers = list(model.net)
    per_layer: list[dict] = []
    activated_threshold = 0.0
    with torch.no_grad():
        cur = Xs
        for j, layer in enumerate(layers):
            cur = layer(cur)
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                # On a leaky-relu, "dead" means: pre-activation always
                # negative (so output is always negative * alpha).
                # Easier to test the pre-activation directly — a
                # neuron is dead if its pre-act is always ≤ 0.
                # We'll use a threshold of 0 on the post-activation
                # for ReLU, and on the pre-activation for LeakyReLU
                # (which we recompute from prev linear layer).
                if isinstance(layer, nn.ReLU):
                    fires = (cur > activated_threshold).any(dim=0).cpu().numpy()
                else:
                    # cur is post-LeakyReLU. Dead = max output ≤ 0
                    # (i.e. pre-act always ≤ 0 since alpha > 0).
                    fires = (cur > activated_threshold).any(dim=0).cpu().numpy()
                dead = int((~fires).sum())
                total = int(len(fires))
                per_layer.append(
                    {
                        "layer": j,
                        "dead": dead,
                        "total": total,
                        "rate": dead / total if total > 0 else 0.0,
                    }
                )
    total_dead = sum(p["dead"] for p in per_layer)
    total_n = sum(p["total"] for p in per_layer)
    return {
        "per_layer": per_layer,
        "combined_rate": total_dead / total_n if total_n > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--hidden", default="192,192,192")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--report-out", default="/mnt/v/output/zenpicker/leakyrelu_vs_relu_2026-04-30.md")
    args = ap.parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(","))

    th.load_codec_config(args.codec_config)

    sys.stderr.write(f"Loading {th.PARETO}...\n")
    pareto = th.load_pareto(th.PARETO)
    feats, feat_cols = th.load_features(th.FEATURES)
    sys.stderr.write(f"  pareto={len(pareto)} feat_cols={len(feat_cols)}\n")

    cells, cell_id_by_key, config_to_cell, parsed_all = th.build_cell_index()
    n_cells = len(cells)
    sys.stderr.write(f"  n_cells={n_cells}\n")

    Xs, Xe, bytes_log, scalars, reach, meta = th.build_dataset(
        pareto, feats, feat_cols, cells, config_to_cell, parsed_all
    )
    sys.stderr.write(f"  decision rows: {len(Xs)}\n")

    rng = np.random.default_rng(SEED)
    images = sorted({m[0] for m in meta})
    rng.shuffle(images)
    n_val = max(1, int(len(images) * HOLDOUT_FRAC))
    val_set = set(images[:n_val])
    tr = np.array([i for i, m in enumerate(meta) if m[0] not in val_set])
    va = np.array([i for i, m in enumerate(meta) if m[0] in val_set])

    Xe_tr_raw, Xe_va_raw = Xe[tr], Xe[va]
    bl_tr, bl_va = bytes_log[tr], bytes_log[va]
    rch_va = reach[va]
    meta_va = [meta[i] for i in va]
    scalars_va = {axis: scalars[axis][va] for axis in th.SCALAR_AXES}

    # Standardize inputs.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xe_tr = scaler.fit_transform(Xe_tr_raw)
    Xe_va = scaler.transform(Xe_va_raw)

    # Build target: bytes_log per cell (NaN-imputed with column mean
    # so the MSE loss has finite gradients on unreached cells), then
    # scalar heads per axis. Same shape as the production hybrid
    # student output: (n_rows, (1 + n_scalar_axes) * n_cells).
    bl_tr_imp = bl_tr.copy()
    col_mean_bl = np.nanmean(bl_tr, axis=0)
    nan_mask = np.isnan(bl_tr_imp)
    bl_tr_imp[nan_mask] = np.take(col_mean_bl, np.where(nan_mask)[1])
    Y_tr = bl_tr_imp.astype(np.float32)
    Y_va_bl = bl_va.copy()
    nan_mask_va = np.isnan(Y_va_bl)
    Y_va_bl[nan_mask_va] = np.take(col_mean_bl, np.where(nan_mask_va)[1])
    Y_va = Y_va_bl.astype(np.float32)
    for axis in th.SCALAR_AXES:
        s_tr = scalars[axis][tr].copy()
        col_mean_sc = np.nanmean(s_tr, axis=0)
        nm = np.isnan(s_tr)
        s_tr[nm] = np.take(col_mean_sc, np.where(nm)[1])
        s_va = scalars[axis][va].copy()
        nm_va = np.isnan(s_va)
        s_va[nm_va] = np.take(col_mean_sc, np.where(nm_va)[1])
        Y_tr = np.concatenate([Y_tr, s_tr.astype(np.float32)], axis=1)
        Y_va = np.concatenate([Y_va, s_va.astype(np.float32)], axis=1)

    sys.stderr.write(
        f"  Xe_tr={Xe_tr.shape} Y_tr={Y_tr.shape}  Xe_va={Xe_va.shape}\n"
    )

    all_mask = np.ones(n_cells, dtype=bool)
    results: dict[str, dict] = {}
    for activation in ["relu", "leakyrelu"]:
        sys.stderr.write(f"\n=== training {activation} student ===\n")
        t0 = time.time()
        model, val_history = train_student(
            Xe_tr,
            Y_tr,
            Xe_va,
            Y_va,
            hidden,
            activation,
            args.epochs,
            args.batch_size,
            SEED,
        )
        elapsed = time.time() - t0
        sys.stderr.write(f"  trained in {elapsed:.1f}s, final val_loss={val_history[-1]:.4f}\n")

        Y_va_pred = predict_np(model, Xe_va)
        pred_bytes = Y_va_pred[:, :n_cells]
        student_argmin = th.evaluate_argmin(pred_bytes, bl_va, rch_va, meta_va, all_mask)
        scalar_pred_va = {
            axis: Y_va_pred[:, (i + 1) * n_cells : (i + 2) * n_cells]
            for i, axis in enumerate(th.SCALAR_AXES)
        }
        student_scalars = th.evaluate_scalars(scalar_pred_va, scalars_va, rch_va)
        dead = dead_neuron_rate(model, Xe_va, activation)
        results[activation] = {
            "argmin": student_argmin,
            "scalars": student_scalars,
            "dead": dead,
            "elapsed_s": elapsed,
            "final_val_loss": val_history[-1],
        }
        sys.stderr.write(
            f"  val: mean overhead {student_argmin['mean_pct']:.2f}% "
            f"acc {student_argmin['argmin_acc']:.1%} dead {dead['combined_rate']:.1%}\n"
        )

    # Write report.
    relu = results["relu"]
    leaky = results["leakyrelu"]

    def fmt_pp(a, b, decimals=2):
        d = b - a
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.{decimals}f}pp"

    report = []
    report.append("# LeakyReLU vs ReLU — student-only comparison\n")
    report.append(f"Date: 2026-04-30  •  hidden={args.hidden}  •  epochs={args.epochs}\n")
    report.append("\n## Methodology\n")
    report.append(
        "Both students trained from scratch with PyTorch on the same\n"
        "image-level 80/20 holdout split (SEED=0xCAFE) used by\n"
        "`train_hybrid.py`. **No teacher distillation** — both students\n"
        "learn directly from the raw bytes_log + scalar targets, with\n"
        "NaN imputation by column mean. Adam, lr=2e-3, batch=512,\n"
        "patience=15. The comparison is narrower than the production\n"
        "pipeline (which uses HistGradientBoostingRegressor as a\n"
        "teacher) but answers the activation question apples-to-apples.\n"
        "\n"
        "Hidden activation = ReLU vs LeakyReLU(alpha=0.01); output is\n"
        "always Identity. Same MLP shape, same seed, same data split.\n"
    )
    report.append("\n## Headline numbers\n")
    report.append("| Metric | ReLU | LeakyReLU | Δ |\n|---|---:|---:|---:|\n")
    rm = relu["argmin"]
    lm = leaky["argmin"]
    report.append(
        f"| Mean overhead (val) | {rm['mean_pct']:.2f}% | {lm['mean_pct']:.2f}% | "
        f"{fmt_pp(rm['mean_pct'], lm['mean_pct'])} |\n"
    )
    report.append(
        f"| p99 overhead (val) | {rm['p99_pct']:.2f}% | {lm['p99_pct']:.2f}% | "
        f"{fmt_pp(rm['p99_pct'], lm['p99_pct'])} |\n"
    )
    report.append(
        f"| max overhead (val) | {rm['max_pct']:.2f}% | {lm['max_pct']:.2f}% | "
        f"{fmt_pp(rm['max_pct'], lm['max_pct'])} |\n"
    )
    report.append(
        f"| argmin accuracy | {rm['argmin_acc']:.4f} | {lm['argmin_acc']:.4f} | "
        f"{(lm['argmin_acc'] - rm['argmin_acc']) * 100:+.2f}pp |\n"
    )
    report.append(
        f"| dead-neuron rate | {relu['dead']['combined_rate']:.1%} | "
        f"{leaky['dead']['combined_rate']:.1%} | "
        f"{(leaky['dead']['combined_rate'] - relu['dead']['combined_rate']) * 100:+.1f}pp |\n"
    )
    report.append(
        f"| final val_loss | {relu['final_val_loss']:.4f} | "
        f"{leaky['final_val_loss']:.4f} | "
        f"{leaky['final_val_loss'] - relu['final_val_loss']:+.4f} |\n"
    )
    report.append(
        f"| training time | {relu['elapsed_s']:.1f}s | "
        f"{leaky['elapsed_s']:.1f}s | |\n"
    )
    report.append("\n## Per-layer dead-neuron breakdown\n")
    report.append("| Layer | ReLU dead/total | LeakyReLU dead/total |\n|---|---|---|\n")
    for r, l in zip(relu["dead"]["per_layer"], leaky["dead"]["per_layer"]):
        report.append(
            f"| {r['layer']} | {r['dead']}/{r['total']} ({r['rate']:.1%}) | "
            f"{l['dead']}/{l['total']} ({l['rate']:.1%}) |\n"
        )

    report.append("\n## Scalar RMSE (lower is better)\n")
    report.append("| Axis | ReLU | LeakyReLU | Δ |\n|---|---:|---:|---:|\n")
    for axis in th.SCALAR_AXES:
        r_rmse = relu["scalars"][axis]
        l_rmse = leaky["scalars"][axis]
        report.append(
            f"| {axis} | {r_rmse:.4f} | {l_rmse:.4f} | "
            f"{l_rmse - r_rmse:+.4f} |\n"
        )

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(report))
    sys.stderr.write(f"\nwrote {out_path}\n")
    print("".join(report))


if __name__ == "__main__":
    sys.exit(main() or 0)
