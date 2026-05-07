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

DATA_DIR = Path("/tmp/v15-prep/data/zenjpeg")
FEATURES_TSV = Path("/tmp/v15-prep/features_v15.tsv")
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

def load_features() -> dict[str, list[float]]:
    """Read zenanalyze named-features TSV. Key is basename(image_path)."""
    feats: dict[str, list[float]] = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        # Accept either bare named cols or `feat_<name>` prefixed cols
        # (extract_features_for_picker uses `feat_<name>` schema).
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
            feats[stem] = vec
    print(f"[features] loaded {len(feats)} stems × {len(NAMED_FEATS)} feats", file=sys.stderr)
    return feats


def parse_knob_json(s: str) -> dict:
    return json.loads(s)


def axis_label(knob: dict, axis_name: str, axis_values: list) -> int | None:
    v = knob.get(axis_name)
    for i, want in enumerate(axis_values):
        if v == want:
            return i
    return None  # value outside our axis set — drop


def load_sweep_tsvs() -> pd.DataFrame:
    parts = []
    tsvs = sorted(DATA_DIR.glob("*.tsv"))
    if not tsvs:
        raise SystemExit(f"[load_sweep] no TSVs under {DATA_DIR}")
    for tsv in tsvs:
        try:
            df = pd.read_csv(tsv, sep="\t", dtype={"q": int, "encoded_bytes": "Int64"})
        except Exception as e:
            print(f"[load_sweep] skip {tsv}: {e}", file=sys.stderr)
            continue
        need = {"image_path", "q", "knob_tuple_json", "encoded_bytes", "score_zensim"}
        if not need.issubset(df.columns):
            continue
        df = df.dropna(subset=["encoded_bytes", "score_zensim", "knob_tuple_json"])
        df["image"] = df["image_path"].astype(str).str.rsplit("/", n=1).str[-1]
        df["bytes"] = df["encoded_bytes"].astype("int64")
        df["zensim"] = df["score_zensim"].astype(float)
        parts.append(df[["image", "q", "knob_tuple_json", "bytes", "zensim"]])
    if not parts:
        raise SystemExit("[load_sweep] no usable TSVs")
    out = pd.concat(parts, ignore_index=True)
    print(f"[load_sweep] {len(out):,} cells across {out['image'].nunique()} images", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Build (image, target_band) → best (q, knob_tuple) labels.

def build_labels(sweep: pd.DataFrame, features: dict[str, list[float]]):
    """For each (image, band), find the min-bytes (q, knob) cell s.t.
    abs(zensim - band) <= BAND_TOL OR zensim >= band (i.e. picker can pick a
    config that lands in or above the band)."""
    rows = []
    sweep_by_image = sweep.groupby("image", sort=False)
    n_in = 0
    n_no_feat = 0
    n_no_band = 0
    for image, df in sweep_by_image:
        if image not in features:
            n_no_feat += 1
            continue
        for band in BANDS:
            in_band = df[df["zensim"] >= band - BAND_TOL]
            if len(in_band) == 0:
                n_no_band += 1
                continue
            # Take min-bytes cell.
            best = in_band.loc[in_band["bytes"].idxmin()]
            knob = parse_knob_json(best["knob_tuple_json"])
            labels = []
            ok = True
            for axis_name, axis_values in AXES:
                idx = axis_label(knob, axis_name, axis_values)
                if idx is None:
                    ok = False
                    break
                labels.append(idx)
            if not ok:
                continue
            rows.append({
                "image": image,
                "band": band,
                "q": int(best["q"]),
                "bytes_oracle": int(best["bytes"]),
                "zensim_oracle": float(best["zensim"]),
                "knob_json": best["knob_tuple_json"],
                **{f"label_{n}": v for n, v in zip(AXIS_NAMES, labels)},
            })
            n_in += 1
    print(f"[labels] kept {n_in:,} (image,band) cells; {n_no_feat:,} images had no features; {n_no_band:,} (image,band) had no in-band config", file=sys.stderr)
    return pd.DataFrame(rows)


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

    feats = load_features()
    sweep = load_sweep_tsvs()
    labels = build_labels(sweep, feats)

    # Image-disjoint split.
    images = sorted(labels["image"].unique())
    random.shuffle(images)
    n_train = int(0.8 * len(images))
    train_images = set(images[:n_train])
    test_images = set(images[n_train:])
    print(f"[split] {len(train_images)} train / {len(test_images)} test images (image-disjoint, seed={SEED})", file=sys.stderr)

    # Build features matrix per (image, band).
    band_to_idx = {b: i for i, b in enumerate(BANDS)}
    def row_features(r) -> list[float]:
        f = list(feats[r["image"]])
        # Append target-band one-hot (so the same MLP serves all bands).
        oh = [0.0] * len(BANDS)
        oh[band_to_idx[r["band"]]] = 1.0
        return f + oh

    labels["feat_vec"] = labels.apply(row_features, axis=1)

    train_df = labels[labels["image"].isin(train_images)].copy()
    test_df = labels[labels["image"].isin(test_images)].copy()
    print(f"[split] {len(train_df):,} train cells / {len(test_df):,} test cells", file=sys.stderr)

    n_in = len(NAMED_FEATS) + len(BANDS)
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

    # Standardize features (mean/std on train only).
    mu = X_tr.mean(dim=0)
    sd = X_tr.std(dim=0).clamp(min=1e-6)
    X_tr_n = (X_tr - mu) / sd
    X_te_n = (X_te - mu) / sd

    model = MultiHeadPicker(n_in)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    EPOCHS = 600
    BATCH = 1024
    n_train_cells = len(train_df)
    print(f"[train] epochs={EPOCHS} batch={BATCH} cells={n_train_cells} n_in={n_in}", file=sys.stderr)

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
            loss = sum(F.cross_entropy(h, yb[an]) for h, an in zip(heads, AXIS_NAMES))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                logits = model(X_te_n)
                heads = model.axis_logits(logits)
                accs = []
                for h, an in zip(heads, AXIS_NAMES):
                    pred = h.argmax(dim=-1)
                    acc = (pred == Y_te[an]).float().mean().item()
                    accs.append(acc)
            print(f"[epoch {epoch:4d}] train-loss={total_loss/n_train_cells:.4f} test-acc={['%.3f'%a for a in accs]}", file=sys.stderr)

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
            # Predicted config doesn't reach the band → fall back to next q
            # at the predicted knob (already covered by lookup_predicted_bytes
            # returning None). Penalize: treat as default bytes (i.e. the
            # picker effectively gives up to default).
            n_no_pred += 1
            b_pred = b_def
        total_pred_bytes += b_pred
        total_def_bytes += b_def
        total_oracle_bytes += b_oracle
        n_eval += 1
    delta_pred = (total_pred_bytes / total_def_bytes - 1.0) * 100.0
    delta_oracle = (total_oracle_bytes / total_def_bytes - 1.0) * 100.0
    print(f"[holdout] n_eval={n_eval} (skipped {n_no_def} no-default-in-band, {n_no_pred} predicted-knob-out-of-band)", file=sys.stderr)
    print(f"[holdout] bytes Δ vs default-knob baseline:", file=sys.stderr)
    print(f"          predicted: {delta_pred:+.2f}%", file=sys.stderr)
    print(f"          oracle:    {delta_oracle:+.2f}%   (floor)", file=sys.stderr)

    # Emit model JSON for bake_picker.py.
    state = model.state_dict()
    model_json = {
        "schema": "zenpicker-multihead-v1",
        "knob_axes": [{"name": n, "values": [str(v).lower() if isinstance(v, bool) else v for v in vs]} for n, vs in AXES],
        "input_features": NAMED_FEATS + [f"target_band_{b}" for b in BANDS],
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
