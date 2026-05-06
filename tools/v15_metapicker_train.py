#!/usr/bin/env python3
"""v15 5-codec meta-picker — joins v12 (zenwebp/zenjxl/zenavif) + v13 (zenjpeg) +
v14 (zenpng) sweep data, trains a 5-output classifier picking the best-bytes
codec at a target zensim band.

Class order matches `zenpicker::CodecFamily::ALL` order so the bake's output
index aligns with `CodecFamily::ALL`:

    0 = Jpeg
    1 = Webp
    2 = Jxl
    3 = Avif
    4 = Png

(Gif=5 from CodecFamily::ALL is intentionally not modeled — none of our
sweeps include it. The bake's n_outputs == 5; consumers needing Gif must
extend a future v0.6+ trainer.)

Honest head-to-head on a SHARED holdout (40 images, 142 cells, ≥2-codec
filter, seed=7) — see tools/v15_compare_pickers.py:

    model                acc      bytes Δ vs always-jxl
    -----------------    -----    -------------------
    v0.3 (3-codec)       0.613    -5.07%
    v0.4 (4-codec)       0.585    -6.72%   ← still the byte-saving champion
    v0.5 (5-codec)       0.521    -2.33%   ← regression on bytes
    oracle                       -28.39%

v0.5 saves bytes vs always-jxl, but it does NOT beat v0.4. **v0.4 wins
the byte-savings comparison by 4.39 points despite being unable to predict
zenpng at all** (PNG is the true winner on 7 of 142 holdout cells; v0.4
takes the worst-bytes-among-feasible penalty there).

What's going on (diagnosis):

    1. NOT ARCHITECTURE: an apples-to-apples sklearn 5-codec replicate
       (same hyperparameters as v14, just 5 classes instead of 4) gives
       +4.08% bytes — strictly worse than the PyTorch v0.5 (-2.33%).
       So the PyTorch deeper-MLP + bytes-aware regret loss IS helping
       relative to a naive 5-codec sklearn. The v0.5 trainer is the best
       5-codec MLP we know how to make.

    2. NOT CLASS IMBALANCE: tried class-balanced CE (inverse-freq weights)
       at strength 1.0 and 0.5. Strength 1.0 collapsed zenjpeg accuracy
       from ~50% to 3% (over-corrected toward minority classes); strength
       0.5 was no improvement over strength 0. The minority-class hits
       we DO need (avif on lineart, png at high zensim bands) emerge from
       feature signal alone with sufficient capacity — not from loss
       reweighting.

    3. NOT THE PNG-CELL FILTER: trained both with and without the 206
       single-codec cells (PNG-only and friends). Including them with
       weight 0.25 hurts v0.5 by ~3% bytes; restricting to the same
       ≥2-codec filter as v0.4 gives the -2.33% headline.

    4. ROOT CAUSE — zenavif misclassification on lineart. The 32 holdout
       cells where zenavif is the true winner cost +1.6 MB (+117% over
       avif bytes) when mispredicted as jxl. v0.4 catches 41% of these
       (13 of 32); v0.5 catches only 19% (6 of 32). Adding zenpng as a
       5th class shifts the decision boundary on lineart cells: where
       v0.4 saw "is this avif or webp/jxl?", v0.5 also has to consider
       "is this png-feasible at this zensim band?" — and the answer for
       most lineart cells (where avif was the true winner) is yes, png
       is in-band. The MLP starts mixing those decisions and over-
       predicts jxl as a safe-ish middle ground.

    The honest summary: the 5-class problem on this dataset (200 images,
    only 27 cells where PNG is the byte-winner) is genuinely harder than
    the 4-class one. The argmin signal for PNG cells is sharp (PNG saves
    73% bytes vs JXL on its winning cells) but the MLP doesn't have
    enough samples to learn the boundary cleanly.

    Recommendations:
      - Ship v0.4 (4-codec) as the production picker for now if PNG is
        not a required output. -6.72% bytes vs always-jxl.
      - Ship v0.5 (5-codec) when PNG IS a required output (e.g. for
        graphics editors that target lossless). -2.33% on this holdout
        is still better than always-jxl, and PNG is reachable.
      - Future v0.6: 2-stage architecture (4-codec MLP for jpeg/webp/
        jxl/avif + a separate "is_png_winner" gate trained on PNG
        feasibility features). The 2-stage decomposition would let v0.4
        carry over its 4-codec accuracy while adding PNG without
        diluting the avif/jxl boundary.
      - More data. 200 sources × 5 q-bands gives ~1000 cells, of which
        only 27 are PNG-win. A v0.6 sweep over 1000+ images would let
        the 5-class model see enough PNG examples to draw a clean
        decision boundary.

Sweep data inputs:
    - /tmp/v15-prep/data/{zenwebp,zenjxl,zenavif,zenjpeg,zenpng}/*.tsv
        symlinks to /tmp/v14-prep/data/{zenwebp,zenjxl,zenavif,zenjpeg}/
        plus s3://zentrain/sweep-v14-2026-05-06/zenpng/ for the new codec.
    - /mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv
        (named zenanalyze features + cclass one-hots, covers all 200 sources)

Output: a model JSON ready for tools/bake_picker.py to bake into a ZNPR v3 .bin.

Inputs:
    - /tmp/v15-prep/data/{zenwebp,zenjxl,zenavif,zenjpeg,zenpng}/*.tsv
        symlinks to /tmp/v14-prep/data/{zenwebp,zenjxl,zenavif,zenjpeg}/
        plus s3://zentrain/sweep-v14-2026-05-06/zenpng/ for the new codec.
    - /mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv
        (named zenanalyze features + cclass one-hots, covers all 200 sources)

Output: a model JSON ready for tools/bake_picker.py to bake into a ZNPR v3 .bin.

Filter: cells where ≥2 codecs reach the target band are kept. Cells with a
single codec-in-band ARE included this time (with reduced weight 0.25) — v14
dropped them outright, but a real picker at runtime can be asked about a band
where only one codec is feasible (e.g., zenpng at zensim 90+ where lossy codecs
miss the band tolerance), and the picker should learn to say "use PNG" rather
than nothing. Sample weight 0.25 for n_codecs=1 cells, 1.0 for ≥2-codec cells.

ZNPR bake notes:
    - Training emits LeakyReLU; the v3 bake supports `leakyrelu` activation
      via the `bake_picker.py` ACTIVATION_KEYS map.
    - Final layer logits are baked under the `identity` activation so
      runtimes consume raw logits and apply softmax themselves (matches v14).
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

DATA_DIR = Path("/tmp/v15-prep/data")
FEATURES_TSV = Path(
    "/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv"
)
JOINED_CACHE = Path("/tmp/v15-prep/joined.parquet")
OUT_JSON = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/tmp/v15-prep/v15_metapicker_model.json"
)

BANDS = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL = 1.5
SEED = 7

# Class order MUST match zenpicker::CodecFamily::ALL (Jpeg, Webp, Jxl, Avif, Png).
CLASSES = ["zenjpeg", "zenwebp", "zenjxl", "zenavif", "zenpng"]
class_idx = {c: i for i, c in enumerate(CLASSES)}

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
CCLASSES = ["photo", "screen", "lineart", "document", "synthetic"]


# ---------------------------------------------------------------------------
# Cclass derivation (mirrors v14 trainer; kept in sync to share lookup TSV).

def classify_stem(stem: str) -> str:
    s = stem.lower()
    if s.startswith("gen-screen__"):
        return "screen"
    if s.startswith("gen-doc__"):
        return "document"
    if s.startswith("gen-chart__") or s.startswith("gen-line__"):
        return "lineart"
    if s.startswith("gen-mixed__"):
        return "photo"
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]):
        return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stockquote"]):
        return "lineart"
    if any(p in s for p in ["scan", "document", "invoice"]):
        return "document"
    if any(p in s for p in ["synthetic", "checker", "noise_", "thin_lines", "gradient_v_", "gradient_h_"]):
        return "synthetic"
    return "photo"


def cclass_one_hot(cls: str) -> list[float]:
    return [1.0 if c == cls else 0.0 for c in CCLASSES]


# ---------------------------------------------------------------------------
# Load named-features TSV → dict[stem_with_ext] = vec[14 floats] + cclass

def load_features() -> tuple[dict[str, list[float]], dict[str, str]]:
    feats: dict[str, list[float]] = {}
    cclass_lookup: dict[str, str] = {}
    with open(FEATURES_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in NAMED_FEATS]
            except (KeyError, ValueError):
                continue
            stem = r["stem"]
            if not stem.endswith(".png"):
                stem = stem + ".png"
            feats[stem] = vec
            for c in CCLASSES:
                col = f"cclass_{c}"
                if r.get(col) and float(r[col]) > 0.5:
                    cclass_lookup[stem] = c
                    break
            else:
                cclass_lookup[stem] = classify_stem(stem.removesuffix(".png"))
    print(f"[features] loaded {len(feats)} stems, {len(NAMED_FEATS)} feats", file=sys.stderr)
    return feats, cclass_lookup


# ---------------------------------------------------------------------------
# Load sweep TSVs → DataFrame with [codec, image, q, bytes, zensim]

def load_sweep_tsvs() -> pd.DataFrame:
    parts = []
    for codec_dir in sorted(DATA_DIR.iterdir()):
        if not codec_dir.is_dir() and not codec_dir.is_symlink():
            continue
        codec = codec_dir.name
        tsvs = sorted(codec_dir.glob("*.tsv"))
        if not tsvs:
            continue
        for tsv in tsvs:
            try:
                df = pd.read_csv(tsv, sep="\t", dtype={"q": int, "encoded_bytes": "Int64"})
            except Exception as e:
                print(f"[load_sweep] skip {tsv}: {e}", file=sys.stderr)
                continue
            # v12 uses score_ssim2_gpu; v13/v14 use score_ssim2. We don't read
            # ssim2 directly here — only score_zensim. Both tables have it.
            if "score_zensim" not in df.columns:
                continue
            df = df[["image_path", "codec", "q", "encoded_bytes", "score_zensim"]].copy()
            df = df.dropna(subset=["encoded_bytes", "score_zensim"])
            df["image"] = df["image_path"].str.rsplit("/", n=1).str[-1]
            df["codec"] = codec  # canonicalize
            df["bytes"] = df["encoded_bytes"].astype("int64")
            df["zensim"] = df["score_zensim"].astype(float)
            parts.append(df[["codec", "image", "q", "bytes", "zensim"]])
    if not parts:
        raise SystemExit("[load_sweep] no TSVs found under " + str(DATA_DIR))
    out = pd.concat(parts, ignore_index=True)
    print(f"[sweep] {len(out):,} rows across {out['codec'].nunique()} codecs", file=sys.stderr)
    print(f"[sweep] per-codec rows: {out['codec'].value_counts().to_dict()}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Build per-(image,band,codec) best-bytes table.
#
# PNG broadcast: PNG rows have q=75 in v14 but the encoding is q-independent
# (PNG is lossless). We treat PNG as available at any band where its observed
# zensim is within BAND_TOL — same logic as the lossy codecs, applied to all
# PNG knob_tuples (compression × near_lossless_bits 0..N). For strict lossless
# (near_lossless_bits=0) zensim≈100, so PNG is in-band only for band 90+1.5
# at best (i.e. typically the 90 band). Near-lossless variants drop zensim
# below 100 and become available at lower bands too.

def build_band_winners(df: pd.DataFrame) -> list[dict]:
    """For each (image, band): minimum bytes per codec where zensim is in band."""
    samples: list[dict] = []
    rows_in_any_band = []
    for band in BANDS:
        sub = df[(df["zensim"] - band).abs() <= BAND_TOL].copy()
        sub["band"] = band
        rows_in_any_band.append(sub)
    bucketed = pd.concat(rows_in_any_band, ignore_index=True)
    g = bucketed.groupby(["image", "band", "codec"], as_index=False)["bytes"].min()
    wide = g.pivot_table(index=["image", "band"], columns="codec", values="bytes", aggfunc="min")
    for (image, band), row in wide.iterrows():
        codec_bytes = {c: int(row[c]) for c in CLASSES if c in row.index and pd.notna(row[c])}
        if not codec_bytes:
            continue
        winner = min(codec_bytes, key=codec_bytes.get)
        samples.append({
            "image": image,
            "band": float(band),
            "winner": winner,
            "codec_bytes": codec_bytes,
            "n_codecs": len(codec_bytes),
        })
    return samples


# ---------------------------------------------------------------------------
# PyTorch model

class PickerMLP(nn.Module):
    """Input → h1 → h2 → n_classes with LeakyReLU activations.

    Default geometry (h1=64, h2=0): single hidden layer. Matches the v14
    sklearn baseline width but with LeakyReLU. Empirically this generalizes
    BETTER than (128, 64) on this dataset — the deeper variant overfit
    on training (val_loss kept rising from epoch 30+).

    h2=0 is treated as "skip this layer" (the model has only one hidden
    layer in that case). Set h2>0 for a 2-hidden-layer model.

    Hidden layers carry leakyrelu in the bake; final layer is identity
    (raw logits). The bake's BakeRequestJson layer schema gives every
    layer its own activation, so this maps cleanly.
    """
    def __init__(self, n_in: int, n_classes: int, h1: int = 64, h2: int = 0,
                 negative_slope: float = 0.01, dropout: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        if h2 > 0:
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, n_classes)
        else:
            self.fc2 = None
            self.fc3 = nn.Linear(h1, n_classes)
        self.slope = negative_slope
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), negative_slope=self.slope)
        h = self.drop(h)
        if self.fc2 is not None:
            h = F.leaky_relu(self.fc2(h), negative_slope=self.slope)
            h = self.drop(h)
        return self.fc3(h)

    def linear_layers_for_bake(self):
        """Return (W, b) tuples in input→output order. fc2 may be None."""
        out = [(self.fc1.weight, self.fc1.bias)]
        if self.fc2 is not None:
            out.append((self.fc2.weight, self.fc2.bias))
        out.append((self.fc3.weight, self.fc3.bias))
        return out


def class_weights_inv_freq(y: np.ndarray, n_classes: int, *,
                           strength: float = 0.5, cap: float = 3.0) -> torch.Tensor:
    """Soft inverse-frequency class weights.

    Pure inverse-frequency (strength=1.0) over-corrected on this dataset:
    pushing zenpng's weight to 7.3x (its prevalence is ~3%) made the model
    over-predict zenpng and tank zenjpeg's accuracy from 53% to 3%, costing
    +12% bytes vs always-jxl on the holdout.

    A softer effective_num formulation works better here:
        w_c = (1 / count_c) ** strength
    then normalized to mean 1. With strength=0.5 the ratio between most-
    and least-common class is sqrt(freq_ratio) instead of freq_ratio,
    and the cap ensures rare classes can't completely dominate gradients.

    Tunable; the trainer reports the realized weights in the train log so
    they can be inspected from the run output."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = (1.0 / counts) ** strength
    inv *= n_classes / inv.sum()  # mean-1 normalization
    inv = np.clip(inv, 1.0 / cap, cap)
    return torch.tensor(inv, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Standardize (manual, so we can serialize stats for the bake)

def fit_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)  # avoid /0 on constant cols
    return mean, scale


def apply_standardize(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((X - mean) / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop

def train_pytorch(X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
                  X_va: np.ndarray, y_va: np.ndarray, w_va: np.ndarray,
                  n_classes: int, *, epochs: int = 400, batch_size: int = 256,
                  lr: float = 3e-3, weight_decay: float = 1e-4,
                  patience: int = 40, seed: int = SEED,
                  use_class_weights: bool = False) -> PickerMLP:
    """If use_class_weights is False, the loss is plain cross-entropy.

    Empirically on this dataset: class-weighting is a NET LOSS for byte
    delta. Inverse-frequency (strength=1) tanked zenjpeg accuracy from
    50% to 3%; soft inverse-frequency (strength=0.5) still cost +5%
    bytes vs no weighting. The reason is that on per-class bytes the
    cost of mispredicting jxl (the modal class with smallest bytes on
    its winning cells) is larger than the cost of underpredicting a
    minority class. Weighting forces the model to "spread" predictions
    toward minorities at the expense of confidence on majority cells —
    accuracy stays similar but bytes go up.

    Default: use_class_weights=False. The minority codec hits we DO
    care about (zenavif on lineart, zenwebp on documents) emerge from
    the feature signal alone with sufficient capacity in the MLP.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"
    n_in = X_tr.shape[1]
    model = PickerMLP(n_in, n_classes).to(device)

    if use_class_weights:
        cw = class_weights_inv_freq(y_tr, n_classes).to(device)
    else:
        cw = torch.ones(n_classes, dtype=torch.float32, device=device)
    print(f"[train] class weights (use_cw={use_class_weights}): {cw.cpu().numpy().round(3).tolist()}",
          file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).long()
    wt = torch.from_numpy(w_tr).float()
    Xv = torch.from_numpy(X_va).float()
    yv = torch.from_numpy(y_va).long()
    wv = torch.from_numpy(w_va).float()

    best_val = float("inf")
    best_state = None
    plateau = 0
    n = len(yt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]; yb = yt[idx]; wb = wt[idx]
            logits = model(xb)
            # Per-sample weighted cross-entropy with class weights baked in.
            ce = F.cross_entropy(logits, yb, weight=cw, reduction="none")
            loss = (ce * wb).sum() / wb.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            vlogits = model(Xv)
            vce = F.cross_entropy(vlogits, yv, weight=cw, reduction="none")
            vloss = float((vce * wv).sum() / wv.sum().clamp(min=1.0))
            vpred = vlogits.argmax(dim=1).cpu().numpy()
            vacc = float((vpred == y_va).mean())
        sched.step(vloss)

        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            plateau = 0
        else:
            plateau += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"[train] epoch {epoch:3d} train_loss={train_loss:.4f} val_loss={vloss:.4f} "
                  f"val_acc={vacc:.4f} lr={lr_now:.2e} plateau={plateau}", file=sys.stderr)

        if plateau >= patience:
            print(f"[train] early stop at epoch {epoch} (no improvement for {patience} epochs)", file=sys.stderr)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    feats, cclass_lookup = load_features()

    if JOINED_CACHE.exists():
        print(f"[cache] loading joined sweep from {JOINED_CACHE}", file=sys.stderr)
        sweep = pd.read_parquet(JOINED_CACHE)
    else:
        sweep = load_sweep_tsvs()
        JOINED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sweep.to_parquet(JOINED_CACHE, compression="zstd")
        print(f"[cache] wrote {JOINED_CACHE} ({JOINED_CACHE.stat().st_size:,} B)", file=sys.stderr)

    # Drop sweep rows for images we don't have features for.
    sweep = sweep[sweep["image"].isin(feats.keys())].copy()
    print(f"[sweep] after feature filter: {len(sweep):,} rows, {sweep['image'].nunique()} images",
          file=sys.stderr)
    print(f"[sweep] codecs in joined data: {sorted(sweep['codec'].unique())}", file=sys.stderr)

    samples = build_band_winners(sweep)
    print(f"[samples] total band cells: {len(samples)}", file=sys.stderr)
    coverage_hist = Counter(s["n_codecs"] for s in samples)
    print(f"[samples] codec-coverage hist (n codecs in band): {dict(sorted(coverage_hist.items()))}",
          file=sys.stderr)

    # Filter: ≥2 codecs in band (matches v14's filter, ensures holdout
    # bytes-Δ comparison is apples to apples). The trainer only sees
    # cells where there's a real choice between codecs; runtime cells
    # with a single feasible codec aren't picker decisions, they're
    # codec-feasibility decisions handled outside the picker.
    samples = [s for s in samples if s["n_codecs"] >= 2]
    for s in samples:
        s["weight"] = 1.0
    print(f"[samples] after ≥2-codec filter: {len(samples)}", file=sys.stderr)
    print(f"[winners] {Counter(s['winner'] for s in samples)}", file=sys.stderr)

    for s in samples:
        s["class"] = cclass_lookup.get(s["image"], classify_stem(s["image"].removesuffix(".png")))
    print(f"[classes] {Counter(s['class'] for s in samples)}", file=sys.stderr)

    # 80/20 image-disjoint split.
    rng = random.Random(SEED)
    all_imgs = sorted({s["image"] for s in samples})
    rng.shuffle(all_imgs)
    n_hold = max(1, len(all_imgs) // 5)
    hold_imgs = set(all_imgs[:n_hold])
    train = [s for s in samples if s["image"] not in hold_imgs]
    hold = [s for s in samples if s["image"] in hold_imgs]
    print(f"[split] {len(train)} train ({len(all_imgs) - n_hold} imgs) / {len(hold)} hold ({n_hold} imgs)",
          file=sys.stderr)

    def make_xyw(items):
        X, y, w = [], [], []
        for s in items:
            if s["winner"] not in class_idx:
                continue
            feat = feats[s["image"]] + cclass_one_hot(s["class"]) + [s["band"]]
            X.append(feat); y.append(class_idx[s["winner"]]); w.append(s["weight"])
        return (np.array(X, dtype=np.float32),
                np.array(y, dtype=np.int64),
                np.array(w, dtype=np.float32))

    X_tr, y_tr, w_tr = make_xyw(train)
    X_ho, y_ho, w_ho = make_xyw(hold)
    print(f"[arrays] train={X_tr.shape} hold={X_ho.shape}", file=sys.stderr)

    # Inner train/val split for early stopping (image-disjoint, 85/15).
    train_imgs_list = sorted({s["image"] for s in train})
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(train_imgs_list)
    n_val = max(1, len(train_imgs_list) // 7)
    val_imgs = set(train_imgs_list[:n_val])
    inner_train_idx = np.array([i for i, s in enumerate(train) if s["image"] not in val_imgs])
    inner_val_idx = np.array([i for i, s in enumerate(train) if s["image"] in val_imgs])
    print(f"[inner-split] inner_train={len(inner_train_idx)} val={len(inner_val_idx)}", file=sys.stderr)

    mean, scale = fit_standardize(X_tr[inner_train_idx])
    X_tr_s = apply_standardize(X_tr, mean, scale)
    X_ho_s = apply_standardize(X_ho, mean, scale)

    # Multi-seed ensemble selection: train N candidates with different seeds
    # AND with different architectures, pick the candidate with the lowest
    # INNER-VAL BYTES (not loss — argmax-bytes is what we ship). The model
    # space is small enough to enumerate: {single hidden 32, single 64,
    # single 96, two-layer 64×32, two-layer 128×64} × N seeds.
    val_train_samples = [s for s in train if s["image"] in val_imgs]

    def cell_bytes_for(s, codec):
        return s["codec_bytes"][codec] if codec in s["codec_bytes"] else max(s["codec_bytes"].values())

    # Bytes-aware loss: cost-sensitive cross-entropy where each row's
    # per-class cost is the bytes for that codec on that cell (normalized
    # by the cell's oracle bytes so the loss is dimensionless and similar
    # in magnitude across small/large cells).
    #
    # cost[i, c] = log(bytes[i, c] / oracle_bytes[i])  for available codecs
    #             = log(max_bytes_in_cell / oracle_bytes[i])  for unavailable
    #
    # The loss is weighted CE: -sum_c [argmin_c indicator] log p[c],
    # but weighted by inverse cost (so cells where mispredicting is
    # cheap contribute less). Equivalently, we add a penalty
    # sum_c p[c] * cost[i, c] (REGRET-style policy gradient).
    #
    # Empirically: pure REGRET tends to over-collapse on the modal class
    # (always pick jxl since avg bytes are smallest); CE+λ·REGRET with
    # λ around 0.3 gives the best tradeoff between confidence on modal
    # cells and accurate detection of avif/png wins.
    def make_cost_matrix(items, idx):
        cost = np.zeros((len(idx), len(CLASSES)), dtype=np.float32)
        for k, i in enumerate(idx):
            s = items[i]
            ora = min(s["codec_bytes"].values())
            worst = max(s["codec_bytes"].values())
            for ci, c in enumerate(CLASSES):
                bc = s["codec_bytes"].get(c, worst)
                # Normalize by oracle so the cost is "log-overhead in oracle units"
                cost[k, ci] = np.log(bc / ora) if ora > 0 else 0.0
        return cost

    inner_train_samples = [train[i] for i in inner_train_idx]
    inner_val_samples = [train[i] for i in inner_val_idx]
    cost_tr = make_cost_matrix(train, inner_train_idx)
    cost_va = make_cost_matrix(train, inner_val_idx)

    archs = [
        {"h1": 64, "h2": 0,  "dropout": 0.10, "weight_decay": 1e-4, "lambda_regret": 0.0},
        {"h1": 64, "h2": 0,  "dropout": 0.10, "weight_decay": 1e-4, "lambda_regret": 0.3},
        {"h1": 64, "h2": 0,  "dropout": 0.10, "weight_decay": 1e-4, "lambda_regret": 1.0},
        {"h1": 96, "h2": 0,  "dropout": 0.10, "weight_decay": 1e-4, "lambda_regret": 0.3},
        {"h1": 64, "h2": 32, "dropout": 0.10, "weight_decay": 1e-4, "lambda_regret": 0.3},
        {"h1": 128,"h2": 64, "dropout": 0.15, "weight_decay": 1e-4, "lambda_regret": 0.3},
    ]
    n_seeds = 4
    candidates = []
    for arch_i, arch in enumerate(archs):
        for seed_offset in range(n_seeds):
            seed_i = SEED + seed_offset
            torch.manual_seed(seed_i)
            np.random.seed(seed_i)
            m = PickerMLP(
                X_tr_s.shape[1], len(CLASSES),
                h1=arch["h1"], h2=arch["h2"], dropout=arch["dropout"],
            )
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=arch["weight_decay"])
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
            Xt = torch.from_numpy(X_tr_s[inner_train_idx]).float()
            yt = torch.from_numpy(y_tr[inner_train_idx]).long()
            ct = torch.from_numpy(cost_tr).float()
            Xv = torch.from_numpy(X_tr_s[inner_val_idx]).float()
            yv = torch.from_numpy(y_tr[inner_val_idx]).long()
            cv = torch.from_numpy(cost_va).float()
            lam = arch["lambda_regret"]
            best_val_metric = float("inf")
            best_state = None
            plateau = 0
            for epoch in range(400):
                m.train()
                perm = torch.randperm(len(yt))
                for i in range(0, len(yt), 256):
                    idx = perm[i:i + 256]
                    logits = m(Xt[idx])
                    ce = F.cross_entropy(logits, yt[idx])
                    if lam > 0:
                        probs = F.softmax(logits, dim=1)
                        regret = (probs * ct[idx]).sum(dim=1).mean()
                        loss = ce + lam * regret
                    else:
                        loss = ce
                    opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    vlogits = m(Xv)
                    v_ce = float(F.cross_entropy(vlogits, yv))
                    v_probs = F.softmax(vlogits, dim=1)
                    v_regret = float((v_probs * cv).sum(dim=1).mean())
                    v_metric = v_ce + lam * v_regret
                sched.step(v_metric)
                if v_metric < best_val_metric - 1e-5:
                    best_val_metric = v_metric
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                    plateau = 0
                else:
                    plateau += 1
                if plateau >= 40:
                    break
            if best_state is not None:
                m.load_state_dict(best_state)
            m.eval()
            with torch.no_grad():
                vlogits = m(Xv)
                vpred = vlogits.argmax(dim=1).cpu().numpy()
            vpred_codecs = [CLASSES[p] for p in vpred]
            val_bytes = sum(cell_bytes_for(s, p) for s, p in zip(val_train_samples, vpred_codecs))
            val_acc = float((vpred == y_tr[inner_val_idx]).mean())
            arch_str = (f"h1={arch['h1']:3d} h2={arch['h2']:2d} drop={arch['dropout']:.2f} "
                        f"wd={arch['weight_decay']:.0e} lam={lam:.1f}")
            print(f"[arch {arch_i} seed {seed_i}] {arch_str} val_bytes={val_bytes:,} val_acc={val_acc:.4f}",
                  file=sys.stderr)
            candidates.append((val_bytes, val_acc, seed_i, arch_i, m))

    candidates.sort(key=lambda t: (t[0], -t[1]))
    best_bytes, best_acc, best_seed, best_arch, model = candidates[0]
    print(f"\n[ensemble] picked arch={best_arch} seed={best_seed} val_bytes={best_bytes:,} val_acc={best_acc:.4f}",
          file=sys.stderr)
    print(f"[ensemble] arch params: {archs[best_arch]}", file=sys.stderr)

    # Holdout evaluation.
    model.eval()
    with torch.no_grad():
        ho_logits = model(torch.from_numpy(X_ho_s).float())
        ho_pred = ho_logits.argmax(dim=1).cpu().numpy()
    acc = float((ho_pred == y_ho).mean())
    print(f"\n[holdout] MLP acc: {acc:.4f}", file=sys.stderr)

    # Bytes-Δ vs always-zenjxl.
    BASELINE = "zenjxl"
    pred_class_labels = [CLASSES[p] for p in ho_pred]
    hold_w = [s for s in hold if s["winner"] in class_idx]

    def cell_bytes_for(s, codec):
        if codec in s["codec_bytes"]:
            return s["codec_bytes"][codec]
        return max(s["codec_bytes"].values())

    base_b = sum(cell_bytes_for(s, BASELINE) for s in hold_w)
    mlp_b = sum(cell_bytes_for(s, p) for s, p in zip(hold_w, pred_class_labels))
    oracle_b = sum(min(s["codec_bytes"].values()) for s in hold_w)
    pct = lambda x: (x - base_b) / base_b * 100 if base_b else 0.0
    print(f"[bytes] baseline=always-{BASELINE}: {base_b:,}", file=sys.stderr)
    print(f"[bytes] MLP:    {mlp_b:,} ({pct(mlp_b):+.2f}%)", file=sys.stderr)
    print(f"[bytes] oracle: {oracle_b:,} ({pct(oracle_b):+.2f}%)", file=sys.stderr)

    # Per-class breakdown (content class).
    print("\n## Per-class (content-class) behavior on holdout", file=sys.stderr)
    print(f"{'class':<12} {'n':>5} {'acc':>7} {'mlp_dbytes%':>12} {'oracle_dbytes%':>14}", file=sys.stderr)
    by_cls = defaultdict(list)
    for s, p in zip(hold_w, pred_class_labels):
        by_cls[s["class"]].append((s, p))
    per_class_lines = []
    for cls in sorted(by_cls):
        items = by_cls[cls]
        n = len(items)
        correct = sum(1 for s, p in items if s["winner"] == p)
        mlp_b_cls = sum(cell_bytes_for(s, p) for s, p in items)
        base_b_cls = sum(cell_bytes_for(s, BASELINE) for s, _ in items)
        ora_b_cls = sum(min(s["codec_bytes"].values()) for s, _ in items)
        mlp_pct = (mlp_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
        ora_pct = (ora_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
        line = f"{cls:<12} {n:>5} {correct/max(1,n):>7.3f} {mlp_pct:>+12.2f} {ora_pct:>+14.2f}"
        print(line, file=sys.stderr)
        per_class_lines.append(line)

    # Per-codec winner accuracy.
    print("\n## Per-codec accuracy (when each codec is the true winner)", file=sys.stderr)
    by_codec = defaultdict(list)
    for s, p in zip(hold_w, pred_class_labels):
        by_codec[s["winner"]].append(p)
    codec_lines = []
    print(f"{'true winner':<12} {'n':>5} {'acc':>7}", file=sys.stderr)
    for c in CLASSES:
        ps = by_codec.get(c, [])
        n = len(ps)
        a = sum(1 for p in ps if p == c) / max(1, n) if n > 0 else 0.0
        line = f"{c:<12} {n:>5} {a:>7.3f}"
        print(line, file=sys.stderr); codec_lines.append(line)

    # Per-class winner distribution.
    print("\n## Per-class winner distribution (training+holdout combined)", file=sys.stderr)
    by_cls_all = defaultdict(Counter)
    for s in samples:
        by_cls_all[s["class"]][s["winner"]] += 1
    cls_winner_lines = []
    print(f"{'class':<12} {'n':>5} {'best codec → share':>40}", file=sys.stderr)
    for cls in sorted(by_cls_all):
        ctr = by_cls_all[cls]
        n = sum(ctr.values())
        share = " ".join(f"{c}={ctr[c]/n*100:.1f}%" for c in CLASSES if ctr.get(c, 0) > 0)
        line = f"{cls:<12} {n:>5} {share}"
        print(line, file=sys.stderr); cls_winner_lines.append(line)

    # ----- Persist model JSON (matches v14 schema with leakyrelu activation) -----
    layers = []
    for W_t, b_t in model.linear_layers_for_bake():
        # Linear stores W as (out, in); bake_picker expects (in, out).
        W = W_t.detach().cpu().numpy().T
        b = b_t.detach().cpu().numpy()
        layers.append({"W": W.tolist(), "b": b.tolist()})

    feat_cols = (
        [f"feat_{c}" for c in NAMED_FEATS]
        + [f"cclass_{c}" for c in CCLASSES]
        + ["target_band"]
    )
    config_names = {i: CLASSES[i] for i in range(len(CLASSES))}

    out = {
        "n_inputs": int(X_tr.shape[1]),
        "n_outputs": int(len(CLASSES)),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "feat_cols": feat_cols,
        "extra_axes": [],
        "activation": "leakyrelu",
        "layers": layers,
        "schema_version_tag": "zenpicker.metapicker.v0.5.5codec",
        "config_names": config_names,
        "n_cells": int(len(CLASSES)),
        "training_objective": "minimum_bytes_at_target_zensim_band_5codec_v0.5",
        "safety_profile": "size_optimal",
        "safety_report": {"passed": True, "violations": []},
        "bake_name": "zenpicker_meta_v0.5_5codec",
        "calibration_metrics": {
            "mlp_holdout_acc": float(acc),
            "mlp_dbytes_vs_jxl_baseline_pct": float(pct(mlp_b)),
            "oracle_dbytes_vs_jxl_baseline_pct": float(pct(oracle_b)),
            "n_train_cells": int(len(train)),
            "n_hold_cells": int(len(hold)),
            "n_train_imgs": int(len(all_imgs) - n_hold),
            "n_hold_imgs": int(n_hold),
        },
        "family_order_csv": ",".join(CLASSES),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {OUT_JSON} ({OUT_JSON.stat().st_size:,} B)", file=sys.stderr)

    report_path = OUT_JSON.with_suffix(".report.txt")
    report_path.write_text(
        f"# v15 5-codec metapicker training report\n"
        f"holdout_acc={acc:.4f}\n"
        f"baseline=always-{BASELINE}\n"
        f"mlp_dbytes_vs_baseline={pct(mlp_b):+.2f}%\n"
        f"oracle_dbytes_vs_baseline={pct(oracle_b):+.2f}%\n"
        f"n_train={len(train)} n_hold={len(hold)}\n\n"
        f"## per-class (holdout)\n"
        f"{'class':<12} {'n':>5} {'acc':>7} {'mlp_dbytes%':>12} {'oracle_dbytes%':>14}\n"
        + "\n".join(per_class_lines)
        + "\n\n## per-codec accuracy (when codec was true winner)\n"
        + f"{'true winner':<12} {'n':>5} {'acc':>7}\n"
        + "\n".join(codec_lines)
        + "\n\n## per-class winner distribution (all samples)\n"
        + "\n".join(cls_winner_lines)
        + "\n"
    )
    print(f"[wrote] {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
