#!/usr/bin/env python3
"""v15 zenjpeg per-codec picker — trains a multi-head softmax over the 6 free
knob axes from the v15 dense sweep (983 images × 19 q × 96 knob configs).

Goal: at runtime, given (zenanalyze named features, target_zensim_band) pick
the (q, knob_tuple) that minimizes encoded_bytes while reaching the band.

Knob axes (all from v15 grid; pinned axes are NOT predicted):

    subsampling           {444, 422, 420}                — 3 classes
    progressive_mode      {baseline, progressive_search} — 2
    effort                {1, 2}                         — 2
    chroma_distance_scale {0.7, 1.2}                     — 2
    aq_enabled            {false, true}                  — 2
    auto_optimize         {false, true}                  — 2

    pinned: sharp_yuv=true, optimize_huffman=true, deringing=false,
            quant_source=jpegli

Output: 13 logits (3+2+2+2+2+2) split into 6 softmax heads. Training uses
multi-head CE on the per-axis argmin-bytes label, plus a bytes-aware regret
penalty on the joint config (so the model learns axis interactions, not just
marginals).

Q is NOT predicted: at runtime the picker outputs the knob_tuple, and the
caller does a 1D bisect on q to hit the target zensim. The training labels
treat (image, target_band) as a unit and pick the joint argmin-bytes config.

Honest holdout: image-disjoint 80/20 split, seed=7. We report on the holdout:

    - per-axis accuracy
    - top-1 joint-config accuracy
    - bytes-Δ vs default-knob baseline (default = subsampling=420,
      progressive_mode=baseline, effort=1, chroma_distance_scale=1.2,
      aq_enabled=false, auto_optimize=false — the closest in-grid match
      to libjpeg/jpegli "out of the box").
    - oracle bytes-Δ (held-out floor: lowest achievable if the picker
      were perfect)

Inputs:
    - /tmp/v15-prep/data/zenjpeg/*.tsv       (sweep TSVs)
    - /tmp/v15-prep/features_v15.tsv         (zenanalyze 14-feature output;
                                              keyed by basename(image_path))

Output: model JSON consumable by tools/bake_picker.py for ZNPR v3 baking.
"""
from __future__ import annotations


# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: v15_zenjpeg_picker_train.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v15 zenjpeg per-codec picker (single-codec trainer using v15 dense sweep).\n"
    "         Use: zentrain/tools/train_hybrid.py with the zenjpeg codec-config module + _picker_lib pipeline.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv
import json
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

DATA_DIR = Path("/tmp/v15r-prep/data/zenjpeg")
FEATURES_TSV = Path("/tmp/v15r-prep/features_v15r_combined.tsv")
# 5 OpenAI-tagged content classes from the v15 curated manifest.
CCLASSES = [
    "illustration_or_logo",
    "illustration_or_screen",
    "photo_natural_or_detailed",
    "photo_or_illustration",
    "photo_wide_gamut",
]
JOINED_CACHE = Path("/tmp/v15-prep/zenjpeg_joined.parquet")
OUT_JSON = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/v15-prep/v15_zenjpeg_picker_model.json"
)

BANDS = [70.0, 75.0, 80.0, 85.0]  # zenjpeg rarely exceeds 90 even at q95
BAND_TOL = 1.5
SEED = 7

# 14 zenanalyze named features (must match runtime extraction order).
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

# Free knob axes — order is the model output head order. The bake & runtime
# must match this order.
AXES: list[tuple[str, list]] = [
    ("subsampling", ["444", "422", "420"]),
    ("progressive_mode", ["baseline", "progressive_search"]),
    ("effort", [1, 2]),
    ("chroma_distance_scale", [0.7, 1.2]),
    ("aq_enabled", [False, True]),
    ("auto_optimize", [False, True]),
]
AXIS_NAMES = [a[0] for a in AXES]
AXIS_VALUES = [a[1] for a in AXES]
AXIS_SIZES = [len(v) for v in AXIS_VALUES]
N_OUT = sum(AXIS_SIZES)  # 13

# Default-knob baseline (closest in-grid match to "out of the box").
DEFAULT_KNOB = {
    "subsampling": "420",
    "progressive_mode": "baseline",
    "effort": 1,
    "chroma_distance_scale": 1.2,
    "aq_enabled": False,
    "auto_optimize": False,
}


# ---------------------------------------------------------------------------
# Loaders

def load_features() -> tuple[dict[str, list[float]], dict[str, str]]:
    """Read zenanalyze features TSV → (feats, cclass).

    Key is basename(image_path), with extension-fallback (.gif↔.png) since
    failing-to-decode gifs were converted to PNGs and re-extracted with .png
    basenames. The sweep TSV uses the original .gif basename, so we register
    feature rows under both extensions when applicable.
    """
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
            # Extension fallback: register .png feature rows for gif-static
            # entries under their original .gif basename (sweep TSV uses .gif).
            if stem.endswith(".png"):
                gif_stem = stem[: -4] + ".gif"
                feats[gif_stem] = vec
                cclass[gif_stem] = cls
    print(f"[features] loaded {len(feats)} stems × {len(NAMED_FEATS)} feats", file=sys.stderr)
    cls_counts = Counter(cclass.values())
    print(f"[features] cclass: {dict(cls_counts)}", file=sys.stderr)
    return feats, cclass


def parse_knob_json(s: str) -> dict:
    return json.loads(s)


def axis_label(knob: dict, axis_name: str, axis_values: list) -> int | None:
    v = knob.get(axis_name)
    for i, want in enumerate(axis_values):
        if v == want:
            return i
    return None  # value outside our axis set — drop


def load_sweep_tsvs() -> pd.DataFrame:
    """Load sweep TSVs, accepting either CPU-metric (`score_zensim`) or GPU-
    metric (`score_zensim_gpu`) column names. CPU and GPU implementations
    produce numerically near-identical zensim scores so cells from both
    backends can be combined for training."""
    parts = []
    tsvs = sorted(DATA_DIR.glob("*.tsv"))
    if not tsvs:
        raise SystemExit(f"[load_sweep] no TSVs under {DATA_DIR}")
    n_cpu = n_gpu = 0
    for tsv in tsvs:
        try:
            df = pd.read_csv(tsv, sep="\t", dtype={"q": int, "encoded_bytes": "Int64"})
        except Exception as e:
            print(f"[load_sweep] skip {tsv}: {e}", file=sys.stderr)
            continue
        zensim_col = None
        if "score_zensim" in df.columns:
            zensim_col = "score_zensim"
            n_cpu += 1
        elif "score_zensim_gpu" in df.columns:
            zensim_col = "score_zensim_gpu"
            n_gpu += 1
        else:
            continue
        need = {"image_path", "q", "knob_tuple_json", "encoded_bytes", zensim_col}
        if not need.issubset(df.columns):
            continue
        df = df.dropna(subset=["encoded_bytes", zensim_col, "knob_tuple_json"])
        df["image"] = df["image_path"].astype(str).str.rsplit("/", n=1).str[-1]
        df["bytes"] = df["encoded_bytes"].astype("int64")
        df["zensim"] = df[zensim_col].astype(float)
        parts.append(df[["image", "q", "knob_tuple_json", "bytes", "zensim"]])
    if not parts:
        raise SystemExit("[load_sweep] no usable TSVs")
    out = pd.concat(parts, ignore_index=True)
    print(f"[load_sweep] {len(out):,} cells across {out['image'].nunique()} images (cpu_tsvs={n_cpu} gpu_tsvs={n_gpu})", file=sys.stderr)
    return out


import itertools

# Joint-config index: 96 ordered tuples, each is (a0,a1,a2,a3,a4,a5).
JOINT_CONFIGS: list[tuple[int, ...]] = list(itertools.product(*[range(s) for s in AXIS_SIZES]))
JOINT_INDEX: dict[tuple[int, ...], int] = {t: i for i, t in enumerate(JOINT_CONFIGS)}
N_JOINT = len(JOINT_CONFIGS)  # 96


def knob_to_joint_idx(knob: dict) -> int | None:
    axis_idxs = []
    for an, av in AXES:
        i = axis_label(knob, an, av)
        if i is None:
            return None
        axis_idxs.append(i)
    return JOINT_INDEX.get(tuple(axis_idxs))


# ---------------------------------------------------------------------------
# Build (image, target_band) → best (q, knob_tuple) labels + bytes-by-config table.

def build_labels(sweep: pd.DataFrame, features: dict[str, list[float]]):
    """For each (image, band), find the min-bytes (q, knob) cell s.t.
    abs(zensim - band) <= BAND_TOL OR zensim >= band (i.e. picker can pick a
    config that lands in or above the band)."""
    rows = []
    sweep_by_image = sweep.groupby("image", sort=False)
    n_in = 0
    n_no_feat = 0
    n_no_band = 0
    bytes_table: list[np.ndarray] = []  # one [96] vector per (image, band) cell
    for image, df in sweep_by_image:
        if image not in features:
            n_no_feat += 1
            continue
        for band in BANDS:
            in_band = df[df["zensim"] >= band - BAND_TOL]
            if len(in_band) == 0:
                n_no_band += 1
                continue
            # Per-knob min-bytes-in-band → 96-element vector. inf where
            # that knob never reaches the band at any q.
            tbl = np.full(N_JOINT, np.inf, dtype=np.float32)
            min_by_knob = in_band.groupby("knob_tuple_json", sort=False)["bytes"].min()
            for knob_str, b in min_by_knob.items():
                k = knob_to_joint_idx(parse_knob_json(knob_str))
                if k is not None:
                    tbl[k] = float(b)
            if not np.isfinite(tbl).any():
                n_no_band += 1
                continue
            best_k = int(np.argmin(tbl))
            best_axes = JOINT_CONFIGS[best_k]
            best_bytes = int(tbl[best_k])
            # Find a representative (q, knob_json, zensim) for this argmin
            # config — used in evaluation lookups below.
            best_knob = {
                an: AXIS_VALUES[ai][best_axes[ai_pos]]
                for ai_pos, (an, ai) in enumerate([(an, ai) for ai, an in enumerate(AXIS_NAMES)])
            }
            best_rows = in_band[in_band["knob_tuple_json"].map(
                lambda s: knob_to_joint_idx(parse_knob_json(s)) == best_k
            )]
            best_row = best_rows.loc[best_rows["bytes"].idxmin()]
            rows.append({
                "image": image,
                "band": band,
                "q": int(best_row["q"]),
                "bytes_oracle": best_bytes,
                "zensim_oracle": float(best_row["zensim"]),
                "knob_json": best_row["knob_tuple_json"],
                "bytes_row": len(bytes_table),  # index into bytes_table
                **{f"label_{an}": best_axes[ai] for ai, an in enumerate(AXIS_NAMES)},
            })
            bytes_table.append(tbl)
            n_in += 1
    print(f"[labels] kept {n_in:,} (image,band) cells; {n_no_feat:,} images had no features; {n_no_band:,} (image,band) had no in-band config", file=sys.stderr)
    out = pd.DataFrame(rows)
    out.attrs["bytes_table"] = np.stack(bytes_table) if bytes_table else np.zeros((0, N_JOINT), dtype=np.float32)
    return out


def lookup_default_baseline(sweep: pd.DataFrame, image: str, band: float) -> int | None:
    """Bytes for default-knob picker at this (image,band): bisect q on the
    default knob_tuple, return min-bytes-in-band; None if default never reaches band."""
    df = sweep[sweep["image"] == image]
    if df.empty:
        return None
    # Filter to default knob.
    def matches_default(s: str) -> bool:
        k = parse_knob_json(s)
        return all(k.get(an) == DEFAULT_KNOB[an] for an in AXIS_NAMES)
    mask = df["knob_tuple_json"].map(matches_default)
    df_def = df[mask]
    if df_def.empty:
        return None
    in_band = df_def[df_def["zensim"] >= band - BAND_TOL]
    if in_band.empty:
        return None
    return int(in_band["bytes"].min())


def lookup_predicted_bytes(sweep: pd.DataFrame, image: str, band: float, knob_pred: dict) -> int | None:
    """Bytes for predicted knob_tuple at this (image,band): bisect q on the
    predicted knob, return min-bytes-in-band; None if predicted knob never
    reaches band."""
    df = sweep[sweep["image"] == image]
    if df.empty:
        return None
    def matches_pred(s: str) -> bool:
        k = parse_knob_json(s)
        return all(k.get(an) == knob_pred[an] for an in AXIS_NAMES)
    mask = df["knob_tuple_json"].map(matches_pred)
    df_pred = df[mask]
    if df_pred.empty:
        return None
    in_band = df_pred[df_pred["zensim"] >= band - BAND_TOL]
    if in_band.empty:
        return None
    return int(in_band["bytes"].min())


# ---------------------------------------------------------------------------
# Model

class MultiHeadPicker(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, N_OUT)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.1)
        h = F.leaky_relu(self.fc2(h), 0.1)
        return self.head(h)  # raw logits, split per-axis for softmax

    def axis_logits(self, logits):
        out = []
        cursor = 0
        for sz in AXIS_SIZES:
            out.append(logits[..., cursor:cursor + sz])
            cursor += sz
        return out


# ---------------------------------------------------------------------------
# Train

def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    feats, cclass = load_features()
    sweep = load_sweep_tsvs()
    labels = build_labels(sweep, feats)
    labels["cclass"] = labels["image"].map(cclass).fillna("unknown")
    bytes_table_np = labels.attrs["bytes_table"]
    print(f"[bytes-table] {bytes_table_np.shape}", file=sys.stderr)

    # Image-disjoint split.
    images = sorted(labels["image"].unique())
    random.shuffle(images)
    n_train = int(0.8 * len(images))
    train_images = set(images[:n_train])
    test_images = set(images[n_train:])
    print(f"[split] {len(train_images)} train / {len(test_images)} test images (image-disjoint, seed={SEED})", file=sys.stderr)

    # Build features matrix per (image, band).
    # Layout: [14 named feats] + [5 cclass one-hot] + [4 target-band one-hot]
    band_to_idx = {b: i for i, b in enumerate(BANDS)}
    cclass_to_idx = {c: i for i, c in enumerate(CCLASSES)}

    def row_features(r) -> list[float]:
        f = list(feats[r["image"]])
        cls_oh = [0.0] * len(CCLASSES)
        ci = cclass_to_idx.get(r["cclass"])
        if ci is not None:
            cls_oh[ci] = 1.0
        band_oh = [0.0] * len(BANDS)
        band_oh[band_to_idx[r["band"]]] = 1.0
        return f + cls_oh + band_oh

    labels["feat_vec"] = labels.apply(row_features, axis=1)

    train_df = labels[labels["image"].isin(train_images)].copy()
    test_df = labels[labels["image"].isin(test_images)].copy()
    print(f"[split] {len(train_df):,} train cells / {len(test_df):,} test cells", file=sys.stderr)

    n_in = len(NAMED_FEATS) + len(CCLASSES) + len(BANDS)
    X_tr = torch.tensor(np.stack(train_df["feat_vec"].values), dtype=torch.float32)
    X_te = torch.tensor(np.stack(test_df["feat_vec"].values), dtype=torch.float32)
    Y_tr = {
        an: torch.tensor(train_df[f"label_{an}"].values, dtype=torch.long)
        for an in AXIS_NAMES
    }
    Y_te = {
        an: torch.tensor(test_df[f"label_{an}"].values, dtype=torch.long)
        for an in AXIS_NAMES
    }

    # Bytes-table tensors for regret loss. Use bytes / bytes_oracle ratio so
    # everything is in [1, ~3] range; +inf entries (knob never reaches band)
    # become a very large penalty.
    btab_tr = torch.tensor(bytes_table_np[train_df["bytes_row"].values], dtype=torch.float32)
    btab_te = torch.tensor(bytes_table_np[test_df["bytes_row"].values], dtype=torch.float32)
    boracle_tr = torch.tensor(train_df["bytes_oracle"].values, dtype=torch.float32)
    boracle_te = torch.tensor(test_df["bytes_oracle"].values, dtype=torch.float32)
    # Replace +inf with 3x oracle bytes (a hard penalty without nan-poisoning).
    big = boracle_tr.unsqueeze(1) * 3.0
    btab_tr = torch.where(torch.isinf(btab_tr), big, btab_tr)
    btab_te = torch.where(torch.isinf(btab_te), boracle_te.unsqueeze(1) * 3.0, btab_te)
    # Joint index tensor for soft-regret outer product. [N_JOINT, n_axes].
    JOINT_IDX_T = torch.tensor(JOINT_CONFIGS, dtype=torch.long)

    # Standardize features (mean/std on train only).
    mu = X_tr.mean(dim=0)
    sd = X_tr.std(dim=0).clamp(min=1e-6)
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    model = MultiHeadPicker(n_in)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    EPOCHS = 600
    BATCH = 1024
    PATIENCE = 60  # epochs without holdout-loss improvement before stopping
    REGRET_LAMBDA = 1.0  # weight of bytes-aware regret in the joint loss
    n_train_cells = len(train_df)
    print(f"[train] max_epochs={EPOCHS} batch={BATCH} cells={n_train_cells} n_in={n_in} patience={PATIENCE} regret_lambda={REGRET_LAMBDA}", file=sys.stderr)

    def soft_regret_loss(heads, btab, boracle):
        """Soft regret = E[bytes_pred / bytes_oracle - 1] under per-axis softmax.
        heads: list of [B, axis_size] logits
        btab:  [B, N_JOINT] bytes for each joint config
        boracle: [B] oracle bytes per cell
        """
        probs = [F.softmax(h, dim=-1) for h in heads]  # [B, axis_size_d]
        # joint_prob[b, k] = prod_d probs[d][b, JOINT_CONFIGS[k][d]]
        # Build via gather + product.
        # Shape per axis after gather: [B, N_JOINT]
        per_axis = [
            probs[d].gather(1, JOINT_IDX_T[:, d].unsqueeze(0).expand(probs[d].size(0), -1))
            for d in range(len(AXIS_NAMES))
        ]
        joint_prob = per_axis[0]
        for d in range(1, len(per_axis)):
            joint_prob = joint_prob * per_axis[d]
        # Expected bytes / bytes_oracle. (B, N_JOINT) * (B, N_JOINT) → (B,)
        ratios = btab / boracle.unsqueeze(1)
        expected = (joint_prob * ratios).sum(dim=1)  # [B]
        return (expected - 1.0).mean()

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
            yb = {an: Y_tr[an][idx] for an in AXIS_NAMES}
            logits = model(xb)
            heads = model.axis_logits(logits)
            ce = sum(F.cross_entropy(h, yb[an]) for h, an in zip(heads, AXIS_NAMES))
            regret = soft_regret_loss(heads, btab_tr[idx], boracle_tr[idx])
            loss = ce + REGRET_LAMBDA * regret
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        # Eval each epoch (cheap on tiny model).
        model.eval()
        with torch.no_grad():
            logits = model(X_te_n)
            heads = model.axis_logits(logits)
            test_loss = sum(F.cross_entropy(h, Y_te[an]).item() for h, an in zip(heads, AXIS_NAMES))
            accs = []
            for h, an in zip(heads, AXIS_NAMES):
                pred = h.argmax(dim=-1)
                acc = (pred == Y_te[an]).float().mean().item()
                accs.append(acc)
        if test_loss < best_test_loss - 1e-4:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch % 25 == 0 or bad_epochs >= PATIENCE or epoch == EPOCHS - 1:
            tag = " *" if bad_epochs == 0 else ""
            print(f"[epoch {epoch:4d}] train-loss={total_loss/n_train_cells:.4f} test-loss={test_loss:.4f} test-acc={['%.3f'%a for a in accs]}{tag}", file=sys.stderr)
        if bad_epochs >= PATIENCE:
            print(f"[early-stop] patience exceeded at epoch {epoch} (best test-loss={best_test_loss:.4f})", file=sys.stderr)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final holdout report.
    model.eval()
    with torch.no_grad():
        logits = model(X_te_n)
        heads = model.axis_logits(logits)
        preds_per_axis = [h.argmax(dim=-1).numpy() for h in heads]

    # Joint-config bytes Δ vs default baseline.
    total_pred_bytes = 0
    total_def_bytes = 0
    total_oracle_bytes = 0
    n_eval = 0
    n_no_def = 0
    n_no_pred = 0
    # Per-(cclass, band) bytes accumulators for slicing.
    slice_pred = defaultdict(int)
    slice_def = defaultdict(int)
    slice_oracle = defaultdict(int)
    slice_n = defaultdict(int)
    for i, (_, r) in enumerate(test_df.iterrows()):
        knob_pred = {
            an: AXIS_VALUES[k][int(preds_per_axis[k][i])]
            for k, an in enumerate(AXIS_NAMES)
        }
        b_pred = lookup_predicted_bytes(sweep, r["image"], r["band"], knob_pred)
        b_def = lookup_default_baseline(sweep, r["image"], r["band"])
        b_oracle = int(r["bytes_oracle"])
        if b_def is None:
            n_no_def += 1
            continue
        if b_pred is None:
            # Predicted config doesn't reach the band → caller would have to
            # bump q on the predicted knob until it does, or fall back to
            # default. Charge the default bytes (no-improvement penalty).
            n_no_pred += 1
            b_pred = b_def
        total_pred_bytes += b_pred
        total_def_bytes += b_def
        total_oracle_bytes += b_oracle
        n_eval += 1
        slice_pred[(r["cclass"], r["band"])] += b_pred
        slice_def[(r["cclass"], r["band"])] += b_def
        slice_oracle[(r["cclass"], r["band"])] += b_oracle
        slice_n[(r["cclass"], r["band"])] += 1
        slice_pred[("ALL", r["band"])] += b_pred
        slice_def[("ALL", r["band"])] += b_def
        slice_oracle[("ALL", r["band"])] += b_oracle
        slice_n[("ALL", r["band"])] += 1
        slice_pred[(r["cclass"], "ALL")] += b_pred
        slice_def[(r["cclass"], "ALL")] += b_def
        slice_oracle[(r["cclass"], "ALL")] += b_oracle
        slice_n[(r["cclass"], "ALL")] += 1
    delta_pred = (total_pred_bytes / total_def_bytes - 1.0) * 100.0
    delta_oracle = (total_oracle_bytes / total_def_bytes - 1.0) * 100.0
    print(f"[holdout] n_eval={n_eval} (skipped {n_no_def} no-default-in-band, {n_no_pred} predicted-knob-out-of-band)", file=sys.stderr)
    print(f"[holdout] bytes Δ vs default-knob baseline (overall):", file=sys.stderr)
    print(f"          predicted: {delta_pred:+.2f}%", file=sys.stderr)
    print(f"          oracle:    {delta_oracle:+.2f}%   (floor)", file=sys.stderr)
    print(f"[holdout] slice (cclass × band): bytes Δ% pred / oracle  (n cells)", file=sys.stderr)
    keys = sorted(slice_n.keys(), key=lambda k: (k[0] != "ALL", str(k[0]), str(k[1])))
    for k in keys:
        if slice_def[k] == 0:
            continue
        dp = (slice_pred[k] / slice_def[k] - 1.0) * 100.0
        do = (slice_oracle[k] / slice_def[k] - 1.0) * 100.0
        print(f"          {str(k[0]):28s} band={k[1]:>4}  pred={dp:+6.2f}%  oracle={do:+6.2f}%  n={slice_n[k]}", file=sys.stderr)

    # Emit model JSON for bake_picker.py.
    state = model.state_dict()
    model_json = {
        "schema": "zenpicker-multihead-v1",
        "knob_axes": [{"name": n, "values": [str(v).lower() if isinstance(v, bool) else v for v in vs]} for n, vs in AXES],
        "input_features": NAMED_FEATS + [f"cclass_{c}" for c in CCLASSES] + [f"target_band_{b}" for b in BANDS],
        "feature_means": mu.tolist(),
        "feature_stds": sd.tolist(),
        "activation": "leakyrelu",
        "alpha": 0.1,
        "layers": [
            {"weight": state["fc1.weight"].tolist(), "bias": state["fc1.bias"].tolist()},
            {"weight": state["fc2.weight"].tolist(), "bias": state["fc2.bias"].tolist()},
            {"weight": state["head.weight"].tolist(), "bias": state["head.bias"].tolist(), "activation": "identity"},
        ],
        "metrics": {
            "n_train_images": len(train_images),
            "n_test_images": len(test_images),
            "n_train_cells": int(n_train_cells),
            "n_test_cells": int(n_eval),
            "per_axis_accuracy": {an: float(np.mean(preds_per_axis[k] == Y_te[an].numpy())) for k, an in enumerate(AXIS_NAMES)},
            "bytes_delta_vs_default_pct": float(delta_pred),
            "bytes_delta_oracle_vs_default_pct": float(delta_oracle),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(model_json, f, indent=2)
    print(f"[done] wrote {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
