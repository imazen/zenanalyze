"""
Shared helpers for the meta-picker training pipeline (cross-codec
classifier). Codec-agnostic: every meta-picker version (v12 3-codec,
v14 4-codec, v15 5-codec, …) imports these helpers and supplies its
own CLASSES list + model/training mechanism. Per-version wrapper
scripts stay short and only own per-version concerns (which codecs,
which architecture, which training framework).

Lives at `zenanalyze/zentrain/tools/_metapicker_lib.py`. Sibling to
`_picker_lib.py` (per-codec picker training). Kept separate because
the meta-picker concern (cross-codec classifier learning argmin-bytes
at a target zensim band) is structurally different from the per-codec
picker concern (cross-config knob-tuple regression to bytes/quality).

Extracted from v12/v14/v15 metapicker_train.py + v15_compare_pickers.py
on 2026-05-26 as part of DEDUP Tier-1 #5 (verified synthesis,
`~/work/zen/zensim/benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md`).
Pre-extraction: v12 (253 LOC) + v14 (430 LOC) + v15 (839 LOC) +
v15_compare_pickers (357 LOC) each duplicated ~60-80% of the
scaffolding below. `classify_stem` was byte-identical v14↔v15;
load_features/load_sweep_tsvs/build_band_winners/cclass_one_hot were
near-identical; per-class report writers were copy-pasted.

What lives here (shared scaffolding):

- `classify_stem(stem)` — stem prefix → content class (photo/screen/
  lineart/document/synthetic). Byte-identical v12/v14/v15.
- `cclass_one_hot(cls)` — content-class one-hot vector.
- `load_features(tsv_path, named_feats)` — named-features TSV →
  {stem_with_ext: vec[float]} + cclass lookup.
- `load_sweep_tsvs(data_dir)` — sweep TSVs under codec/ subdirs →
  DataFrame[codec, image, q, bytes, zensim].
- `build_band_winners(df, bands, band_tol, classes)` — per-(image,
  band) min-bytes-per-codec table with winner.
- `image_disjoint_split(samples, seed, holdout_frac)` — deterministic
  image-disjoint train/holdout split.
- `cell_bytes_for(s, codec)` — bytes lookup with worst-case fallback.
- `bytes_delta_vs_baseline(hold, pred_codecs, baseline)` —
  always-baseline vs MLP vs oracle bytes Δ on a holdout.
- `format_per_class_report(...)` — per-class breakdown lines.
- `format_per_class_winner_distribution(...)` — winner share by
  content class.
- `write_metapicker_json(...)` — ZNPR-bakeable JSON writer.
- `forward_metapicker(model, X)` — numpy forward-pass for a baked
  metapicker JSON (LeakyReLU or ReLU). Used by comparators.

What does NOT live here (intentional per-version variation):

- Per-version `CLASSES` list (3/4/5 codecs) — passed in.
- Per-version model/training framework (sklearn MLPClassifier vs
  PyTorch + AdamW + multi-arch ensemble) — wrappers own this.
- Per-version sample filter (e.g. v14: ≥2-codec strict, v15: ≥2
  with optional weight=0.25 single-codec) — wrappers own this.
- Per-version data prep (v12: two-TSV merge; v14/v15: single TSV) —
  wrappers own their data sources.

Per-version wiring example (v14):

    from _metapicker_lib import (
        BANDS, BAND_TOL, SEED, CCLASSES,
        classify_stem, cclass_one_hot,
        load_features, load_sweep_tsvs, build_band_winners,
        image_disjoint_split, cell_bytes_for,
        write_metapicker_json,
    )

    CLASSES = ["zenjpeg", "zenwebp", "zenjxl", "zenavif"]
    NAMED_FEATS = [...]
    feats, cclass_lookup = load_features(FEATURES_TSV, NAMED_FEATS)
    sweep = load_sweep_tsvs(DATA_DIR)
    samples = build_band_winners(sweep, BANDS, BAND_TOL, CLASSES)
    train, hold = image_disjoint_split(samples)
    # ... train sklearn MLPClassifier on (X, y) ...
    write_metapicker_json(OUT, ...)
"""

from __future__ import annotations

import csv
import json
import sys
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Defaults shared across v12/v14/v15.

BANDS_DEFAULT: list[float] = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL_DEFAULT: float = 1.5
SEED_DEFAULT: int = 7
HOLDOUT_FRAC_DEFAULT: float = 0.20

# Content classes used by every version. Per CodecFamily/cclass docs
# (see `zenanalyze/src/tier1.rs` and the cclass one-hot wiring in v12).
CCLASSES: list[str] = ["photo", "screen", "lineart", "document", "synthetic"]


# ---------------------------------------------------------------------------
# Cclass derivation — byte-identical across v12/v14/v15. Kept here as the
# single source so future versions don't re-fork the regex list.

def classify_stem(stem: str) -> str:
    """Map an image stem to a content class.

    Byte-identical impl across the v12/v14/v15 trainers (verified by
    diff before extraction). Used both by trainers AND the comparator
    (`v15_compare_pickers.py`).
    """
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


def cclass_one_hot(cls: str, cclasses: list[str] = CCLASSES) -> list[float]:
    """Content-class one-hot vector. Defaults to `CCLASSES`."""
    return [1.0 if c == cls else 0.0 for c in cclasses]


# ---------------------------------------------------------------------------
# Feature loader — load a named-features TSV with a precomputed cclass.

def load_features(
    features_tsv: Path,
    named_feats: list[str],
    *,
    cclasses: list[str] = CCLASSES,
    require_png_ext: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Load `features_tsv` → ({stem: feat_vec}, {stem: cclass}).

    - `named_feats`: list of column names to extract per row.
    - `require_png_ext`: append '.png' to stem if missing (v14/v15 do
      this so the lookup key matches the sweep-side `image` column).
    - cclass priority: use precomputed `cclass_*` one-hot column if
      present (>0.5), else fall back to `classify_stem(stem)`.

    Rows missing any named feature (KeyError / ValueError) are silently
    skipped — preserves v14/v15 behaviour.
    """
    feats: dict[str, list[float]] = {}
    cclass_lookup: dict[str, str] = {}
    with open(features_tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                vec = [float(r[c] or 0) for c in named_feats]
            except (KeyError, ValueError):
                continue
            stem = r["stem"]
            if require_png_ext and not stem.endswith(".png"):
                stem = stem + ".png"
            feats[stem] = vec
            for c in cclasses:
                col = f"cclass_{c}"
                if r.get(col) and float(r[col]) > 0.5:
                    cclass_lookup[stem] = c
                    break
            else:
                cclass_lookup[stem] = classify_stem(stem.removesuffix(".png"))
    if verbose:
        print(f"[features] loaded {len(feats)} stems, {len(named_feats)} feats",
              file=sys.stderr)
    return feats, cclass_lookup


# ---------------------------------------------------------------------------
# Sweep loader — load codec/<*.tsv> sweep data into a unified DataFrame.

def load_sweep_tsvs(data_dir: Path, *, verbose: bool = True):
    """Load sweep TSVs under `data_dir/<codec>/*.tsv` → DataFrame.

    Returned columns: [codec, image, q, bytes, zensim]. The `codec`
    column is canonicalized to the parent dir name (so a stale `codec`
    column in the TSV is overridden — matches v14/v15 behaviour).
    Skips any TSV that lacks `score_zensim`. Skips rows missing
    `encoded_bytes` or `score_zensim`. Includes symlinked codec dirs
    (v15 symlinks into v14's data dir).

    Requires pandas; raises ImportError if not installed.
    """
    import pandas as pd

    parts = []
    for codec_dir in sorted(data_dir.iterdir()):
        # v15 added symlinked codec dirs; accept those too.
        if not codec_dir.is_dir() and not codec_dir.is_symlink():
            continue
        codec = codec_dir.name
        tsvs = sorted(codec_dir.glob("*.tsv"))
        if not tsvs:
            continue
        for tsv in tsvs:
            try:
                df = pd.read_csv(tsv, sep="\t",
                                 dtype={"q": int, "encoded_bytes": "Int64"})
            except Exception as e:
                print(f"[load_sweep] skip {tsv}: {e}", file=sys.stderr)
                continue
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
        raise SystemExit("[load_sweep] no TSVs found under " + str(data_dir))
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"[sweep] {len(out):,} rows across {out['codec'].nunique()} codecs",
              file=sys.stderr)
        print(f"[sweep] per-codec rows: {out['codec'].value_counts().to_dict()}",
              file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Band-winner builder — per-(image,band) min-bytes-per-codec + winner.

def build_band_winners(
    df,
    bands: list[float] = BANDS_DEFAULT,
    band_tol: float = BAND_TOL_DEFAULT,
    classes: list[str] | None = None,
) -> list[dict]:
    """For each (image, band): {codec: min_bytes_in_band}, winner.

    - `classes`: codec list to restrict to (drops rows for codecs
      outside the list, e.g. v14 excludes zenpng). None = all codecs.

    Returns list of dicts: {image, band, winner, codec_bytes, n_codecs}.

    Requires pandas.
    """
    import pandas as pd

    samples: list[dict] = []
    rows_in_any_band = []
    for band in bands:
        sub = df[(df["zensim"] - band).abs() <= band_tol].copy()
        sub["band"] = band
        rows_in_any_band.append(sub)
    bucketed = pd.concat(rows_in_any_band, ignore_index=True)
    g = bucketed.groupby(["image", "band", "codec"], as_index=False)["bytes"].min()
    wide = g.pivot_table(index=["image", "band"], columns="codec",
                         values="bytes", aggfunc="min")
    for (image, band), row in wide.iterrows():
        if classes is None:
            codec_bytes = {c: int(row[c]) for c in row.index if pd.notna(row[c])}
        else:
            codec_bytes = {c: int(row[c]) for c in classes
                           if c in row.index and pd.notna(row[c])}
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
# Train/holdout split — image-disjoint, deterministic.

def image_disjoint_split(
    samples: list[dict],
    *,
    seed: int = SEED_DEFAULT,
    holdout_frac: float = HOLDOUT_FRAC_DEFAULT,
) -> tuple[list[dict], list[dict], list[str], set[str]]:
    """Image-disjoint train/holdout split.

    Returns (train, hold, all_imgs_sorted, hold_imgs).
    """
    rng = random.Random(seed)
    all_imgs = sorted({s["image"] for s in samples})
    rng.shuffle(all_imgs)
    n_hold = max(1, int(len(all_imgs) * holdout_frac))
    hold_imgs = set(all_imgs[:n_hold])
    train = [s for s in samples if s["image"] not in hold_imgs]
    hold = [s for s in samples if s["image"] in hold_imgs]
    return train, hold, all_imgs, hold_imgs


# ---------------------------------------------------------------------------
# Cell-bytes lookup — bytes for `codec` on cell `s`, with worst-case
# fallback when the codec didn't reach the band. Identical across v14/v15.

def cell_bytes_for(s: dict, codec: str) -> int:
    """Bytes for `codec` on cell `s`; max(codec_bytes) if codec is OOB."""
    if codec in s["codec_bytes"]:
        return s["codec_bytes"][codec]
    return max(s["codec_bytes"].values())


# ---------------------------------------------------------------------------
# Bytes-Δ reporting — always-baseline vs mlp vs oracle on a holdout.

def bytes_delta_vs_baseline(
    hold: list[dict],
    pred_codecs: list[str],
    baseline: str = "zenjxl",
) -> tuple[int, int, int, Callable[[float], float]]:
    """Return (base_b, mlp_b, oracle_b, pct_fn).

    `pct_fn(x) -> (x - base_b) / base_b * 100`. Useful for callers that
    want to compute Δ for additional partitions (per-class).
    """
    base_b = sum(cell_bytes_for(s, baseline) for s in hold)
    mlp_b = sum(cell_bytes_for(s, p) for s, p in zip(hold, pred_codecs))
    oracle_b = sum(min(s["codec_bytes"].values()) for s in hold)
    pct = lambda x: (x - base_b) / base_b * 100 if base_b else 0.0
    return base_b, mlp_b, oracle_b, pct


# ---------------------------------------------------------------------------
# Per-class breakdown report — v14/v15 emit this verbatim.

def format_per_class_report(
    hold: list[dict],
    pred_codecs: list[str],
    cclass_lookup: dict[str, str] | None = None,
    baseline: str = "zenjxl",
) -> list[str]:
    """Return list of formatted lines (one per content class).

    Cclass priority: `cclass_lookup[image]` if present, else
    `classify_stem(image[:-4])`. Matches v14/v15 logic.
    """
    by_cls = defaultdict(list)
    for s, p in zip(hold, pred_codecs):
        if cclass_lookup is not None and s["image"] in cclass_lookup:
            cls = cclass_lookup[s["image"]]
        elif "class" in s:
            cls = s["class"]
        else:
            cls = classify_stem(s["image"].removesuffix(".png"))
        by_cls[cls].append((s, p))
    lines: list[str] = []
    for cls in sorted(by_cls):
        items = by_cls[cls]
        n = len(items)
        correct = sum(1 for s, p in items if s["winner"] == p)
        mlp_b_cls = sum(cell_bytes_for(s, p) for s, p in items)
        base_b_cls = sum(cell_bytes_for(s, baseline) for s, _ in items)
        ora_b_cls = sum(min(s["codec_bytes"].values()) for s, _ in items)
        mlp_pct = (mlp_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
        ora_pct = (ora_b_cls - base_b_cls) / base_b_cls * 100 if base_b_cls else 0
        line = f"{cls:<12} {n:>5} {correct/max(1,n):>7.3f} {mlp_pct:>+12.2f} {ora_pct:>+14.2f}"
        lines.append(line)
    return lines


def format_per_class_winner_distribution(
    samples: list[dict],
    classes: list[str],
) -> list[str]:
    """Per-class winner share (training+holdout combined). v14/v15."""
    by_cls_all = defaultdict(Counter)
    for s in samples:
        cls = s.get("class")
        if cls is None:
            cls = classify_stem(s["image"].removesuffix(".png"))
        by_cls_all[cls][s["winner"]] += 1
    lines: list[str] = []
    for cls in sorted(by_cls_all):
        ctr = by_cls_all[cls]
        n = sum(ctr.values())
        share = " ".join(f"{c}={ctr[c]/n*100:.1f}%"
                         for c in classes if ctr.get(c, 0) > 0)
        line = f"{cls:<12} {n:>5} {share}"
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Per-codec accuracy (used by v15 + the comparator).

def format_per_codec_accuracy(
    hold: list[dict],
    pred_codecs: list[str],
    classes: list[str],
) -> list[str]:
    """Per-true-winner accuracy lines."""
    by_codec = defaultdict(list)
    for s, p in zip(hold, pred_codecs):
        by_codec[s["winner"]].append(p)
    lines: list[str] = []
    for c in classes:
        ps = by_codec.get(c, [])
        n = len(ps)
        a = sum(1 for p in ps if p == c) / max(1, n) if n > 0 else 0.0
        line = f"{c:<12} {n:>5} {a:>7.3f}"
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# ZNPR-bakeable JSON writer.

def write_metapicker_json(
    out_path: Path,
    *,
    n_inputs: int,
    classes: list[str],
    scaler_mean: list[float],
    scaler_scale: list[float],
    feat_cols: list[str],
    layers: list[dict],
    activation: str,
    schema_version_tag: str,
    bake_name: str,
    training_objective: str,
    calibration_metrics: dict,
    safety_profile: str = "size_optimal",
    extra_axes: list[str] | None = None,
) -> None:
    """Write ZNPR-bakeable picker JSON. Schema matches v14/v15 (and v12
    when given the right activation + bake_name)."""
    out = {
        "n_inputs": int(n_inputs),
        "n_outputs": int(len(classes)),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "feat_cols": feat_cols,
        "extra_axes": extra_axes if extra_axes is not None else [],
        "activation": activation,
        "layers": layers,
        "schema_version_tag": schema_version_tag,
        "config_names": {i: classes[i] for i in range(len(classes))},
        "n_cells": int(len(classes)),
        "training_objective": training_objective,
        "safety_profile": safety_profile,
        "safety_report": {"passed": True, "violations": []},
        "bake_name": bake_name,
        "calibration_metrics": calibration_metrics,
        "family_order_csv": ",".join(classes),
    }
    out_path.write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------------------
# Numpy forward-pass for a baked metapicker JSON. Used by comparators.
#
# This is a domain-specific subset of the deferred Tier-1 #6 follow-on
# `_predict_lib.forward`. Metapicker JSONs always carry standardize +
# dense-MLP-with-uniform-activation + identity-on-final-layer schema, so
# the impl is tighter than the general ZNPR runtime helper would be.
# When `_predict_lib.py` ships, this should delegate to it.

def _leaky_relu(x: np.ndarray, slope: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, slope * x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def forward_metapicker(model: dict, X: np.ndarray) -> np.ndarray:
    """Numpy forward-pass for a baked metapicker JSON.

    Standardize, then dense → activation → ... → identity (raw logits).
    Returns argmax indices. Activation is `model['activation']`;
    `leakyrelu` / `leaky_relu` use slope=0.01, anything else uses ReLU.

    Used by `v15_compare_pickers.py`.
    """
    mean = np.array(model["scaler_mean"], dtype=np.float32)
    scale = np.array(model["scaler_scale"], dtype=np.float32)
    Z = ((X - mean) / scale).astype(np.float32)
    act = model.get("activation", "relu").lower()
    af = _leaky_relu if act in ("leakyrelu", "leaky_relu") else _relu
    layers = model["layers"]
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float32)
        b = np.asarray(layer["b"], dtype=np.float32)
        Z = Z @ W + b
        if i < len(layers) - 1:
            Z = af(Z)
    return np.argmax(Z, axis=1)


# ---------------------------------------------------------------------------
# __all__ — explicit re-export list so wrappers can `from _metapicker_lib
# import *` without pulling in pandas/numpy unexpectedly.

__all__ = [
    "BANDS_DEFAULT",
    "BAND_TOL_DEFAULT",
    "SEED_DEFAULT",
    "HOLDOUT_FRAC_DEFAULT",
    "CCLASSES",
    "classify_stem",
    "cclass_one_hot",
    "load_features",
    "load_sweep_tsvs",
    "build_band_winners",
    "image_disjoint_split",
    "cell_bytes_for",
    "bytes_delta_vs_baseline",
    "format_per_class_report",
    "format_per_class_winner_distribution",
    "format_per_codec_accuracy",
    "write_metapicker_json",
    "forward_metapicker",
]
