#!/usr/bin/env python3
"""tile_fill_tiny_features.py — content-aware NaN fill for picker features on
too-small renditions, via mirror-tiled re-extraction (2026-06-28 tiny-image fix).

THE PROBLEM
-----------
Percentile / windowed content features (laplacian_variance_p*, aq_map_p*,
noise_floor_*, quant_survival_*, luma_kurtosis, ...) are NaN ("") for an image
too small to satisfy their per-feature minimum-sample / minimum-block floor
(zenanalyze #49). `train_hybrid.py` used to DROP those (image, size) rows, which
silently starved the (tiny, target_zq) corner of the size×quality grid the picker
must serve → DATA_STARVED_SIZE → bake refused. A *constant* fill (0.0 / feature
min) clears the gate but makes every too-small image IDENTICAL in those features
→ a degenerate tiny-image picker that can't distinguish content (gate-gaming).

THE FIX (native-primary + mirror-tiled-fill)
---------------------------------------------
The percentiles are NaN only because there are too few blocks/windows — not
because the content lacks structure. So for a too-small rendition we MIRROR-TILE
the content up to >= MIN_TILE_DIM px (alternating horizontal/vertical flips =
seamless; NO plain repeat, which would inject false edges at the seams and inflate
laplacian_variance), then RE-EXTRACT. The tiled image has the SAME per-block
statistics as the original content, just enough samples to define the percentiles.
We then fill ONLY the NaN columns from the tiled run, keeping every valid native
feature unchanged (native is correct at native size). Each too-small image gets
its OWN content-derived percentile values.

MIN_TILE_DIM = 128 was chosen by measurement (see
/mnt/v/output/picker-feature-size-audit-2026-06-28): mirror-tiling to 96px already
recovers ALL 97 extracted features (64px leaves 30 NaN); 128 is a safe margin.

CONSISTENCY (train == inference)
--------------------------------
This is a feature-extraction step, so the picker RUNTIME must apply the SAME
mirror-tile-before-extract to too-small images at inference (codec
`extract_raw_features_rgb8`: if a feature is missing/NaN, mirror-tile to
>= MIN_TILE_DIM and re-extract; native-primary fill). The `mirror_tile` algorithm
below is the canonical spec — the Rust inference path MUST produce byte-identical
tiled pixels (it is integer flips + concat, deterministic given (w, h)).

Residual NaN after tiling is REPORTED, never constant-filled (per the no-silent-
bad-values policy). On the 2026-06-28 corpus only `spectral_slope_y` stays NaN for
1 rendition — and it is NOT in any picker KEEP_FEATURES set, so no picker feature
ships a missing/constant value.

Usage:
  tile_fill_tiny_features.py \
     --native-tsv  <native features TSV (variant_name or image_path keyed)> \
     --corpus-dir  <dir with the rendition PNGs> [--corpus-dir ...] \
     --extractor   <path to extract_features_for_picker binary> \
     --out         <combined_features_vn_tiled.tsv> [--min-tile-dim 128]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

MIN_TILE_DIM_DEFAULT = 128


def mirror_tile(arr: np.ndarray, min_dim: int) -> np.ndarray:
    """Mirror-tile (alternating flips, seamless) each axis up to >= min_dim px.

    CANONICAL SPEC — the codec inference path must match this exactly:
      - tile counts: ny = ceil(min_dim/h) if h<min_dim else 1; nx likewise on w.
      - tile (i, j): the source, flipped left-right iff i is odd, flipped
        up-down iff j is odd. Concatenate columns then rows. No crop (full
        tiles; result is >= min_dim in each tiled axis).
    """
    h, w = arr.shape[:2]
    ny = -(-min_dim // h) if h < min_dim else 1
    nx = -(-min_dim // w) if w < min_dim else 1
    rows = []
    for j in range(ny):
        cols = []
        for i in range(nx):
            t = arr
            if i % 2 == 1:
                t = t[:, ::-1]
            if j % 2 == 1:
                t = t[::-1, :]
            cols.append(t)
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def _norm_feat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename qualified `name@hex8` feature cols to `feat_<name>` (idempotent)."""
    ren = {c: "feat_" + c.split("@")[0] for c in df.columns if "@" in c}
    return df.rename(columns=ren)


def size_class(px: float) -> str:
    if px <= 64 * 64:
        return "tiny"
    if px <= 256 * 256:
        return "small"
    if px <= 1024 * 1024:
        return "medium"
    return "large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-tsv", action="append", required=True,
                    help="native features TSV(s); repeat per corpus")
    ap.add_argument("--corpus-dir", action="append", required=True,
                    help="rendition PNG dir(s), parallel to --native-tsv")
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-tile-dim", type=int, default=MIN_TILE_DIM_DEFAULT)
    ap.add_argument("--scratch", default="/tmp/tile_fill")
    args = ap.parse_args()
    assert len(args.native_tsv) == len(args.corpus_dir), "--native-tsv/--corpus-dir count mismatch"
    os.makedirs(args.scratch, exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../../../zenmetrics/scripts/picker"))
    import origin_split as osp  # noqa: E402

    tables = []
    for tsv, d in zip(args.native_tsv, args.corpus_dir):
        df = _norm_feat_cols(pd.read_csv(tsv, sep="\t"))
        if "variant_name" not in df.columns:
            df["variant_name"] = df["image_path"].map(
                lambda p: os.path.basename(str(p))[:-4]
                if str(p).endswith(".png") else os.path.basename(str(p)))
        df["__dir"] = d
        tables.append(df)
    featc = sorted(set.intersection(*[{c for c in t.columns if c.startswith("feat_")} for t in tables]))
    print(f"[tile-fill] common feat cols: {len(featc)}", flush=True)

    # 1) mirror-tile every rendition that has a NaN feature, re-extract.
    man = os.path.join(args.scratch, "manifest.tsv")
    rows = ["sha256\tsplit\tcontent_class\tsource\tpath"]
    seen = set()
    for df in tables:
        d = df["__dir"].iloc[0]
        for _, r in df[df[featc].isna().any(axis=1)].iterrows():
            vn = r["variant_name"]
            src = os.path.join(d, vn + ".png")
            if vn in seen or not os.path.exists(src):
                continue
            seen.add(vn)
            a = np.asarray(Image.open(src).convert("RGB"))
            Image.fromarray(mirror_tile(a, args.min_tile_dim)).save(
                os.path.join(args.scratch, vn + ".png"))
            rows.append(f"\t\t\t\t{os.path.join(args.scratch, vn + '.png')}")
    open(man, "w").write("\n".join(rows) + "\n")
    print(f"[tile-fill] NaN renditions mirror-tiled to >= {args.min_tile_dim}px: {len(seen)}", flush=True)
    subprocess.run([args.extractor, "--manifest", man, "--output",
                    os.path.join(args.scratch, "tiled.tsv"), "--sizes", "0"], check=True)
    tiled = _norm_feat_cols(pd.read_csv(os.path.join(args.scratch, "tiled.tsv"), sep="\t"))
    tiled["variant_name"] = tiled["image_path"].map(lambda p: os.path.basename(str(p))[:-4])
    tfill = {r["variant_name"]: r for _, r in tiled.iterrows()}

    # 2) native-primary + tiled-fill: fill ONLY NaN feat cols from the tiled run.
    VN = ["variant_name", "image_path", "image_sha", "split", "content_class",
          "source", "size_class", "width", "height"] + featc
    out_parts = []
    n_filled = 0
    for df in tables:
        df = df.copy()
        for idx, r in df[df[featc].isna().any(axis=1)].iterrows():
            tf = tfill.get(r["variant_name"])
            if tf is None:
                continue
            for c in featc:
                if pd.isna(r[c]) and c in tf and pd.notna(tf[c]):
                    df.at[idx, c] = tf[c]
            n_filled += 1
        df["split"] = df["variant_name"].map(osp.split_of)
        df["size_class"] = (df["width"].astype(float) * df["height"].astype(float)).map(size_class)
        for c in ("image_path", "image_sha", "content_class", "source"):
            if c not in df.columns:
                df[c] = ""
        out_parts.append(df[[c for c in VN if c in df.columns]])
    out = pd.concat(out_parts, ignore_index=True).drop_duplicates("variant_name", keep="first")
    resid = out[featc].isna().sum()
    resid = {k: int(v) for k, v in resid.items() if v > 0}
    print(f"[tile-fill] tiled-filled renditions: {n_filled}; combined rows: {len(out)}", flush=True)
    print(f"[tile-fill] residual-NaN feat cols (REPORTED, never constant-filled): "
          f"{resid if resid else 'NONE'}", flush=True)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"[tile-fill] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
