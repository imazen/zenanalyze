#!/usr/bin/env python3
"""v0.2 zenjpeg picker — hybrid regression+classification.

Replaces v0.1's 6-axis multi-head softmax with a more honest model:

    Input:  14 zenanalyze named features
          + 5 content_class one-hot
          + 1 target_zensim (continuous, not band one-hot)

    Hidden: 64 → 64 LeakyReLU

    Outputs:
      ├── subsampling           : 3-way softmax  (categorical)
      ├── progressive_mode      : 2-way softmax  (categorical)
      ├── aq_enabled            : 1 sigmoid      (boolean)
      ├── auto_optimize         : 1 sigmoid      (boolean)
      ├── chroma_distance_scale : 1 scalar       (log-space regression)
      └── q                     : 1 scalar       (log-space regression)

Training data: union of v15r (1.79M cells, full knob grid, sparse chroma
coverage [0.7, 1.2]) and v15rc (514K cells, dense chroma [0.4..2.0],
other knobs pinned). Targets are oracle (min-bytes-in-band) cells per
(image, target_zensim).

Loss: multi_head_CE + λ·BCE + μ·MSE(log).

Holdout: image-disjoint 80/20, seed=7. Reports per-class bytes Δ vs
zenjpeg encoder default (no `--expert` overrides; reference TSV is
v15r-baseline/baseline.tsv with 18,639 cells).
"""
from __future__ import annotations


# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: v0_2_zenjpeg_picker_train.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         Hybrid regression+classification picker (v0.2 era, pre-train_hybrid).\n"
    "         Use: zentrain/tools/train_hybrid.py (codec-agnostic, the canonical trainer since v15) for new work, or v15_zenjpeg_picker_train.py for the most recent zenjpeg-specific predecessor.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config

ROOT = Path("/mnt/v/zen/zensim-training/2026-05-07")
SWEEP_DIRS = [
    ROOT / "v15r-prep/data/zenjpeg",
    ROOT / "v15rc-prep/data/zenjpeg",
]
BASELINE_TSV = ROOT / "v15r-baseline/baseline.tsv"
FEATURES_TSV = ROOT / "v15r-prep/features_v15r_combined.tsv"

OUT_JSON = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "/home/lilith/work/zen/zenanalyze/benchmarks/zenjpeg_picker_v0.2_2026-05-07.json"
)

# Continuous target zensim values to sample per image. Spans the production-
# quality band (q=70-95-ish range maps to roughly zensim 70-90 on most
# 1024px content).
TARGET_ZENSIMS = [70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0]
BAND_TOL = 1.5
SEED = 7

NAMED_FEATS = [
    "aspect_min_over_max",
    "chroma_complexity",
    "colourfulness",
    "dct_compressibility_uv",
    "dct_compressibility_y",
    "edge_density",
    "flat_color_block_ratio",
    "gradient_fraction",
    "high_freq_energy_ratio",
    "laplacian_variance",
    "log_pixels",
    "luma_histogram_entropy",
    "uniformity",
    "variance",
]
CCLASSES = [
    "illustration_or_logo",
    "illustration_or_screen",
    "photo_natural_or_detailed",
    "photo_or_illustration",
    "photo_wide_gamut",
]

# Categorical axis vocabularies (must match what the encoder accepts).
SUBSAMPLING_VALUES = ["444", "422", "420"]
PROGRESSIVE_VALUES = ["baseline", "progressive_search"]

# Scalar regression bounds (clamp predictions; train in log-space).
CHROMA_MIN, CHROMA_MAX = 0.1, 5.0
Q_MIN, Q_MAX = 5.0, 95.0

# Loss weights.
LAMBDA_BOOL = 1.0       # BCE for aq/auto
MU_CHROMA = 0.5         # MSE-log on chroma_distance_scale
MU_Q = 0.5              # MSE-log on q
GAMMA_REGRET = 0.5      # bytes-aware regret (computed against oracle bytes,
                        # not via outer-product since scalars complicate that)


# ---------------------------------------------------------------------------
# Loaders

def load_features() -> tuple[dict[str, list[float]], dict[str, str]]:
    feats: dict[str, list[float]] = {}
    cclass: dict[str, str] = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                vec = [
                    float(r.get(c) or r.get(f"feat_{c}") or 0.0)
                    for c in NAMED_FEATS
                ]
            except (KeyError, ValueError):
                continue
            stem = r.get("image_path") or r.get("stem") or r.get("path") or ""
            if not stem:
                continue
            stem = Path(stem).name
            cls = r.get("content_class", "")
            feats[stem] = vec
            cclass[stem] = cls
            # Extension fallback (gif → png-converted).
            if stem.endswith(".png"):
                gif_stem = stem[: -4] + ".gif"
                feats.setdefault(gif_stem, vec)
                cclass.setdefault(gif_stem, cls)
    print(f"[features] {len(feats)} stems × {len(NAMED_FEATS)} feats", file=sys.stderr)
    print(f"[features] cclass dist: {dict(Counter(cclass.values()))}", file=sys.stderr)
    return feats, cclass


def load_sweep(features: dict) -> pd.DataFrame:
    """Load and concat all sweep TSVs. Drops rows with missing scores or
    images that have no zenanalyze features."""
    parts = []
    for sweep_dir in SWEEP_DIRS:
        tsvs = sorted(sweep_dir.glob("*.tsv"))
        for tsv in tsvs:
            try:
                df = pd.read_csv(tsv, sep="\t",
                                 dtype={"q": "Int64", "encoded_bytes": "Int64"})
            except Exception as e:
                print(f"[load] skip {tsv.name}: {e}", file=sys.stderr)
                continue
            need = {"image_path", "q", "knob_tuple_json", "encoded_bytes",
                    "score_zensim"}
            if not need.issubset(df.columns):
                continue
            df = df.dropna(subset=["encoded_bytes", "score_zensim", "knob_tuple_json"])
            df["stem"] = df["image_path"].astype(str).str.rsplit("/", n=1).str[-1]
            df = df[df["stem"].isin(features.keys())]
            parts.append(df)
    if not parts:
        raise SystemExit(f"[load] no usable TSVs under {SWEEP_DIRS}")
    out = pd.concat(parts, ignore_index=True)
    print(f"[load] {len(out):,} cells, {out['stem'].nunique()} unique images",
          file=sys.stderr)
    # Parse knob_tuple_json once.
    out["knob"] = out["knob_tuple_json"].map(json.loads)
    out["k_subsampling"] = out["knob"].map(lambda k: k.get("subsampling", "420"))
    out["k_progressive"] = out["knob"].map(lambda k: k.get("progressive_mode", "baseline"))
    out["k_aq"] = out["knob"].map(lambda k: bool(k.get("aq_enabled", False)))
    out["k_auto"] = out["knob"].map(lambda k: bool(k.get("auto_optimize", False)))
    out["k_chroma"] = out["knob"].map(lambda k: float(k.get("chroma_distance_scale", 1.0)))
    return out


def build_labels(sweep: pd.DataFrame, features: dict, cclass: dict) -> pd.DataFrame:
    """Per (image, target_zensim) → oracle cell. Continuous target sampled
    from TARGET_ZENSIMS."""
    rows = []
    n_no_band = 0
    sweep_by_image = sweep.groupby("stem", sort=False)
    for stem, df in sweep_by_image:
        for tgt in TARGET_ZENSIMS:
            in_band = df[df["score_zensim"] >= tgt - BAND_TOL]
            if in_band.empty:
                n_no_band += 1
                continue
            best = in_band.loc[in_band["encoded_bytes"].idxmin()]
            rows.append({
                "stem": stem,
                "cclass": cclass.get(stem, "unknown"),
                "target": tgt,
                "bytes_oracle": int(best["encoded_bytes"]),
                "zensim_actual": float(best["score_zensim"]),
                "label_subs": SUBSAMPLING_VALUES.index(best["k_subsampling"]),
                "label_prog": PROGRESSIVE_VALUES.index(best["k_progressive"]),
                "label_aq": float(best["k_aq"]),
                "label_auto": float(best["k_auto"]),
                "label_chroma": float(best["k_chroma"]),
                "label_q": float(best["q"]),
            })
    print(f"[labels] kept {len(rows):,} (image,target) cells "
          f"({n_no_band:,} skipped: target unreachable)", file=sys.stderr)
    return pd.DataFrame(rows)


def load_baseline_lookup() -> dict[tuple[str, float], int]:
    """Encoder-default baseline: per (stem, target_zensim) → bytes at
    smallest q reaching target."""
    base = pd.read_csv(BASELINE_TSV, sep="\t",
                      dtype={"q": "Int64", "encoded_bytes": "Int64"})
    base = base.dropna(subset=["encoded_bytes", "score_zensim"])
    base["stem"] = base["image_path"].astype(str).str.rsplit("/", n=1).str[-1]
    out = {}
    for stem, df in base.groupby("stem"):
        for tgt in TARGET_ZENSIMS:
            in_band = df[df["score_zensim"] >= tgt - BAND_TOL].sort_values("q")
            if in_band.empty:
                continue
            out[(stem, tgt)] = int(in_band["encoded_bytes"].iloc[0])
    print(f"[baseline] {len(out):,} (stem,target) → bytes_default entries",
          file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Model

class HybridPicker(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # Heads
        self.head_subs = nn.Linear(hidden, len(SUBSAMPLING_VALUES))
        self.head_prog = nn.Linear(hidden, len(PROGRESSIVE_VALUES))
        self.head_aq = nn.Linear(hidden, 1)
        self.head_auto = nn.Linear(hidden, 1)
        self.head_chroma = nn.Linear(hidden, 1)  # outputs log_chroma
        self.head_q = nn.Linear(hidden, 1)       # outputs log_q

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.1)
        h = F.leaky_relu(self.fc2(h), 0.1)
        return {
            "subs": self.head_subs(h),
            "prog": self.head_prog(h),
            "aq": self.head_aq(h).squeeze(-1),
            "auto": self.head_auto(h).squeeze(-1),
            "log_chroma": self.head_chroma(h).squeeze(-1),
            "log_q": self.head_q(h).squeeze(-1),
        }


def predict_knobs(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Argmax/sigmoid/exp the raw outputs to recoverable knob values."""
    return {
        "subs_idx": out["subs"].argmax(dim=-1),
        "prog_idx": out["prog"].argmax(dim=-1),
        "aq": (torch.sigmoid(out["aq"]) > 0.5).float(),
        "auto": (torch.sigmoid(out["auto"]) > 0.5).float(),
        "chroma": torch.exp(out["log_chroma"]).clamp(CHROMA_MIN, CHROMA_MAX),
        "q": torch.exp(out["log_q"]).clamp(Q_MIN, Q_MAX),
    }


# ---------------------------------------------------------------------------
# Holdout eval — bytes lookup per predicted knob set

def lookup_bytes_for_knobs(sweep: pd.DataFrame, stem: str,
                           target_zensim: float,
                           knob_pred: dict) -> int | None:
    """For the picker's predicted knobs (categorical exact, scalars rounded
    to nearest grid value), find the cell on the sweep grid that matches AND
    reaches target_zensim, return its bytes. None if no in-grid match exists."""
    df = sweep[sweep["stem"] == stem]
    if df.empty:
        return None
    in_band = df[df["score_zensim"] >= target_zensim - BAND_TOL]
    if in_band.empty:
        return None
    # Snap predicted scalars to the available grid values (because we only
    # have measured bytes at sweep grid points).
    chroma_grid = sorted(in_band["k_chroma"].unique())
    q_grid = sorted(in_band["q"].unique())
    chroma_snap = min(chroma_grid, key=lambda c: abs(c - knob_pred["chroma"]))
    # Pick the smallest q that meets target zensim AT the snapped chroma +
    # categorical knobs. If no such cell exists, fall back to any in-band
    # bytes-min cell with matching categorical knobs.
    sub_v = SUBSAMPLING_VALUES[knob_pred["subs_idx"]]
    prog_v = PROGRESSIVE_VALUES[knob_pred["prog_idx"]]
    mask = (
        (in_band["k_subsampling"] == sub_v)
        & (in_band["k_progressive"] == prog_v)
        & (in_band["k_aq"] == bool(knob_pred["aq"]))
        & (in_band["k_auto"] == bool(knob_pred["auto"]))
        & (np.isclose(in_band["k_chroma"], chroma_snap, atol=0.001))
    )
    matched = in_band[mask]
    if matched.empty:
        # Fall back: nearest by chroma only (categoricals may not be in grid).
        chroma_only = in_band[np.isclose(in_band["k_chroma"], chroma_snap, atol=0.001)]
        if chroma_only.empty:
            return None
        return int(chroma_only["encoded_bytes"].min())
    return int(matched["encoded_bytes"].min())


# ---------------------------------------------------------------------------
# Train

def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    feats, cclass = load_features()
    sweep = load_sweep(feats)
    labels = build_labels(sweep, feats, cclass)
    baseline = load_baseline_lookup()

    # Image-disjoint split.
    images = sorted(labels["stem"].unique())
    random.shuffle(images)
    n_train = int(0.8 * len(images))
    train_imgs = set(images[:n_train])
    test_imgs = set(images[n_train:])
    print(f"[split] {len(train_imgs)} train / {len(test_imgs)} test images",
          file=sys.stderr)

    train_df = labels[labels["stem"].isin(train_imgs)].copy()
    test_df = labels[labels["stem"].isin(test_imgs)].copy()
    print(f"[split] {len(train_df):,} train / {len(test_df):,} test cells",
          file=sys.stderr)

    # Build feature matrices.
    cclass_idx = {c: i for i, c in enumerate(CCLASSES)}

    def row_to_input(r) -> list[float]:
        f = list(feats[r["stem"]])
        oh = [0.0] * len(CCLASSES)
        ci = cclass_idx.get(r["cclass"])
        if ci is not None:
            oh[ci] = 1.0
        # target_zensim normalized to roughly [-1, 1] (production band 70-90)
        tz = (r["target"] - 80.0) / 10.0
        return f + oh + [tz]

    X_tr = torch.tensor(np.stack([row_to_input(r) for _, r in train_df.iterrows()]),
                        dtype=torch.float32)
    X_te = torch.tensor(np.stack([row_to_input(r) for _, r in test_df.iterrows()]),
                        dtype=torch.float32)

    Y_tr = {
        "subs":   torch.tensor(train_df["label_subs"].values, dtype=torch.long),
        "prog":   torch.tensor(train_df["label_prog"].values, dtype=torch.long),
        "aq":     torch.tensor(train_df["label_aq"].values, dtype=torch.float32),
        "auto":   torch.tensor(train_df["label_auto"].values, dtype=torch.float32),
        "chroma": torch.tensor(train_df["label_chroma"].values, dtype=torch.float32),
        "q":      torch.tensor(train_df["label_q"].values, dtype=torch.float32),
    }
    Y_te = {
        "subs":   torch.tensor(test_df["label_subs"].values, dtype=torch.long),
        "prog":   torch.tensor(test_df["label_prog"].values, dtype=torch.long),
        "aq":     torch.tensor(test_df["label_aq"].values, dtype=torch.float32),
        "auto":   torch.tensor(test_df["label_auto"].values, dtype=torch.float32),
        "chroma": torch.tensor(test_df["label_chroma"].values, dtype=torch.float32),
        "q":      torch.tensor(test_df["label_q"].values, dtype=torch.float32),
    }

    # Standardize features (mean/std on train).
    mu = X_tr.mean(dim=0); sd = X_tr.std(dim=0).clamp(min=1e-6)
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    n_in = X_tr.shape[1]
    model = HybridPicker(n_in)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    EPOCHS = 800
    BATCH = 1024
    PATIENCE = 60
    n_train_cells = len(train_df)
    print(f"[train] max_epochs={EPOCHS} batch={BATCH} cells={n_train_cells} "
          f"n_in={n_in} patience={PATIENCE}", file=sys.stderr)

    def loss_fn(out, y):
        ce = (
            F.cross_entropy(out["subs"], y["subs"])
            + F.cross_entropy(out["prog"], y["prog"])
        )
        bce = (
            F.binary_cross_entropy_with_logits(out["aq"], y["aq"])
            + F.binary_cross_entropy_with_logits(out["auto"], y["auto"])
        )
        # Log-space MSE on scalars
        log_chroma_target = torch.log(y["chroma"].clamp(min=1e-3))
        log_q_target = torch.log(y["q"].clamp(min=1e-3))
        mse_chroma = F.mse_loss(out["log_chroma"], log_chroma_target)
        mse_q = F.mse_loss(out["log_q"], log_q_target)
        return ce + LAMBDA_BOOL * bce + MU_CHROMA * mse_chroma + MU_Q * mse_q

    best_test_loss = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train_cells)
        total_loss = 0.0
        for i in range(0, n_train_cells, BATCH):
            idx = perm[i:i + BATCH]
            xb = X_tr_n[idx]
            yb = {k: Y_tr[k][idx] for k in Y_tr}
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            out = model(X_te_n)
            test_loss = loss_fn(out, Y_te).item()
        if test_loss < best_test_loss - 1e-4:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 25 == 0 or bad_epochs >= PATIENCE or epoch == EPOCHS - 1:
            tag = " *" if bad_epochs == 0 else ""
            print(f"[epoch {epoch:4d}] train={total_loss/n_train_cells:.4f} "
                  f"test={test_loss:.4f}{tag}", file=sys.stderr)
        if bad_epochs >= PATIENCE:
            print(f"[early-stop] epoch {epoch}, best test_loss={best_test_loss:.4f}",
                  file=sys.stderr)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Holdout eval.
    model.eval()
    with torch.no_grad():
        out = model(X_te_n)
        preds = predict_knobs(out)
        preds_np = {k: v.numpy() for k, v in preds.items()}

    # Per-axis accuracy (categorical heads only)
    print(f"[holdout] per-axis accuracy:", file=sys.stderr)
    print(f"  subsampling      = {(preds_np['subs_idx'] == Y_te['subs'].numpy()).mean():.3f}",
          file=sys.stderr)
    print(f"  progressive_mode = {(preds_np['prog_idx'] == Y_te['prog'].numpy()).mean():.3f}",
          file=sys.stderr)
    print(f"  aq_enabled       = {(preds_np['aq'] == Y_te['aq'].numpy()).mean():.3f}",
          file=sys.stderr)
    print(f"  auto_optimize    = {(preds_np['auto'] == Y_te['auto'].numpy()).mean():.3f}",
          file=sys.stderr)

    # Scalar regression error
    chroma_err = np.abs(preds_np["chroma"] - test_df["label_chroma"].values)
    q_err = np.abs(preds_np["q"] - test_df["label_q"].values)
    print(f"  chroma MAE  = {chroma_err.mean():.3f}  median={np.median(chroma_err):.3f}",
          file=sys.stderr)
    print(f"  q MAE       = {q_err.mean():.2f}  median={np.median(q_err):.1f}",
          file=sys.stderr)

    # Bytes Δ vs encoder default per (cclass, target)
    print(f"\n[holdout] bytes Δ vs encoder default:", file=sys.stderr)
    sliceacc = defaultdict(lambda: [0, 0, 0, 0])  # cls/target → [pred, def, oracle, n]
    n_no_default = n_no_pred = 0
    for i, (_, r) in enumerate(test_df.iterrows()):
        b_default = baseline.get((r["stem"], r["target"]))
        if b_default is None:
            n_no_default += 1
            continue
        knob_pred = {k: int(preds_np[k][i]) if k.endswith("_idx") else float(preds_np[k][i])
                     for k in preds_np}
        b_pred = lookup_bytes_for_knobs(sweep, r["stem"], r["target"], knob_pred)
        if b_pred is None:
            n_no_pred += 1
            b_pred = b_default
        b_oracle = int(r["bytes_oracle"])
        for key in [(r["cclass"], "ALL"), (r["cclass"], r["target"]), ("ALL", "ALL")]:
            sliceacc[key][0] += b_pred
            sliceacc[key][1] += b_default
            sliceacc[key][2] += b_oracle
            sliceacc[key][3] += 1

    print(f"  (skipped {n_no_default} no-default-in-band, {n_no_pred} no-pred-grid-match)",
          file=sys.stderr)
    print(f"  {'cclass':<28} {'target':>7} {'pred Δ':>8} {'oracle':>8}  n", file=sys.stderr)
    for key in sorted(sliceacc.keys(),
                      key=lambda k: (k[0] != "ALL", str(k[0]), str(k[1]))):
        b_pred, b_def, b_orc, n = sliceacc[key]
        if b_def == 0: continue
        dp = (b_pred / b_def - 1) * 100
        do = (b_orc / b_def - 1) * 100
        print(f"  {str(key[0]):<28} {str(key[1]):>7} {dp:>+7.2f}%  {do:>+7.2f}% {n:>5d}",
              file=sys.stderr)

    # Emit model JSON
    state = model.state_dict()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "schema": "zenpicker-hybrid-v0.2",
            "input_features": NAMED_FEATS + [f"cclass_{c}" for c in CCLASSES] + ["target_zensim_norm"],
            "feature_means": mu.tolist(),
            "feature_stds": sd.tolist(),
            "categorical_axes": {
                "subsampling": SUBSAMPLING_VALUES,
                "progressive_mode": PROGRESSIVE_VALUES,
            },
            "boolean_axes": ["aq_enabled", "auto_optimize"],
            "scalar_axes": {
                "chroma_distance_scale": {"min": CHROMA_MIN, "max": CHROMA_MAX, "log_space": True},
                "q": {"min": Q_MIN, "max": Q_MAX, "log_space": True},
            },
            "activation": "leakyrelu",
            "alpha": 0.1,
            "layers": [
                {"weight": state["fc1.weight"].tolist(), "bias": state["fc1.bias"].tolist()},
                {"weight": state["fc2.weight"].tolist(), "bias": state["fc2.bias"].tolist()},
            ],
            "heads": {
                k: {
                    "weight": state[f"head_{k.split('_')[0]}.weight"].tolist(),
                    "bias":   state[f"head_{k.split('_')[0]}.bias"].tolist(),
                }
                for k in ["subs", "prog", "aq", "auto", "chroma", "q"]
            },
            "metrics": {
                "n_train_images": len(train_imgs),
                "n_test_images": len(test_imgs),
                "best_test_loss": float(best_test_loss),
                "bytes_delta_overall_pct": (sliceacc[("ALL", "ALL")][0] /
                                            max(1, sliceacc[("ALL", "ALL")][1]) - 1) * 100
                if sliceacc[("ALL", "ALL")][1] else 0.0,
            },
        }, f, indent=2)
    print(f"\n[done] wrote {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
