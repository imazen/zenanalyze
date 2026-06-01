#!/usr/bin/env python3
"""Materialize the zenjpeg source-feature picker dataset, FAITHFULLY
mirroring the Rust `build_picker_dataset` in
`zenanalyze/zenpicker-train/src/pareto_dataset.rs`.

One row per (image, target_zq) over the ZQ grid (step-5 0..70, step-2
70..100). For each row + each of the 36 categorical cells (derived from
`knob_tuple_json`):
    bytes_log[cell] = ln(min encoded_bytes over rows in this cell whose
                         score_zensim >= target_zq)
    reach[cell]     = any such row exists
Features = per-image feature vector (108) + zq_norm (= target_zq/100) as
the LAST column. Rows where NO cell reaches the target are dropped (the
ceiling-aware skip). Grouped-by-image deterministic split (val_frac of
distinct images, sorted, held out — matches `grouped_split_picker`).

Outputs a single .npz cache so every ablation run reuses the build:
    features  (n_rows, 109)  float64  [108 feat + zq_norm]
    bytes_log (n_rows, 36)   float64  NaN where unreachable
    reach     (n_rows, 36)   bool
    target_zq (n_rows,)      int64
    image_ids (n_rows,)      <U..  (the per-row image identifier)
    split     (n_rows,)      int64  0=train 1=val
    feature_names (108,)     <U..
    cell_labels (36,)        <U..
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CANON = ["subsampling", "progressive", "sharp_yuv", "effort"]


def render_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def cell_key_from_knob(knob_json: str) -> str:
    """Port of `cell_key_from_knob` — canonical-ordered knob tuple."""
    try:
        obj = json.loads(knob_json)
    except Exception:
        return f"raw:{knob_json}"
    if not isinstance(obj, dict):
        return f"raw:{knob_json}"
    parts = []
    for k in CANON:
        if k in obj:
            parts.append(f"{k}={render_scalar(obj[k])}")
    for k in sorted(kk for kk in obj if kk not in CANON):
        parts.append(f"{k}={render_scalar(obj[k])}")
    return "|".join(parts)


def default_zq_targets() -> list[int]:
    v = list(range(0, 70, 5))
    v.extend(range(70, 101, 2))
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            "/mnt/v/zen/picker-dense-full-2026-05-27/parquet/"
            "picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet"
        ),
    )
    ap.add_argument(
        "--feature-order",
        type=Path,
        default=Path(
            "/mnt/v/zen/picker-dense-full-2026-05-27/parquet/feature_order.txt"
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--codec-filter", default="zenjpeg")
    args = ap.parse_args()

    feat_names = {}
    for line in args.feature_order.read_text().splitlines():
        i, n = line.split("\t")
        feat_names[i] = n

    t = pq.read_table(args.parquet)
    names = t.schema.names
    feat_cols = sorted(
        [n for n in names if n.startswith("feat_")],
        key=lambda c: int(c.split("_")[1]),
    )
    n_feat = len(feat_cols)
    d = t.to_pydict()

    image = d["image_basename"]
    codec = d.get("codec", ["zenjpeg"] * len(image))
    knob = d["knob_tuple_json"]
    score = np.asarray(d["score_zensim"], dtype=np.float64)
    bytes_v = np.asarray(d["encoded_bytes"], dtype=np.float64)
    F = np.column_stack(
        [np.asarray(d[c], dtype=np.float64) for c in feat_cols]
    )

    want = args.codec_filter.lower() if args.codec_filter else None

    # Drop non-finite / non-positive bytes; apply codec filter; collect cell set.
    keep = []
    cell_keys = [None] * len(image)
    for r in range(len(image)):
        if want is not None and want not in str(codec[r]).lower():
            continue
        s, b = score[r], bytes_v[r]
        if not np.isfinite(s) or not np.isfinite(b) or b <= 0.0:
            continue
        if not np.all(np.isfinite(F[r])):
            continue
        cell_keys[r] = cell_key_from_knob(knob[r])
        keep.append(r)

    cell_labels = sorted({cell_keys[r] for r in keep})
    n_cells = len(cell_labels)
    cell_index = {c: i for i, c in enumerate(cell_labels)}
    sys.stderr.write(f"[build] {len(keep)} raw rows kept, {n_cells} cells\n")

    # Group kept rows by image.
    by_image: dict[str, list[int]] = {}
    for r in keep:
        by_image.setdefault(image[r], []).append(r)

    zq_targets = default_zq_targets()

    feats_out = []
    bytes_log_out = []
    reach_out = []
    image_ids = []
    target_zq = []

    for im in sorted(by_image):
        rows = by_image[im]
        # per-image feature vector = first row's features
        f = F[rows[0]]
        for zq in zq_targets:
            cell_min = np.full(n_cells, np.inf)
            cell_reach = np.zeros(n_cells, dtype=bool)
            for r in rows:
                if score[r] >= zq:
                    c = cell_index[cell_keys[r]]
                    if bytes_v[r] < cell_min[c]:
                        cell_min[c] = bytes_v[r]
                        cell_reach[c] = True
            if not cell_reach.any():
                continue
            row_feat = np.empty(n_feat + 1)
            row_feat[:n_feat] = f
            row_feat[n_feat] = zq / 100.0
            feats_out.append(row_feat)
            bl = np.where(cell_reach, np.log(np.where(cell_reach, cell_min, 1.0)), np.nan)
            bytes_log_out.append(bl)
            reach_out.append(cell_reach.copy())
            image_ids.append(im)
            target_zq.append(zq)

    features = np.asarray(feats_out, dtype=np.float64)
    bytes_log = np.asarray(bytes_log_out, dtype=np.float64)
    reach = np.asarray(reach_out, dtype=bool)
    image_ids = np.asarray(image_ids)
    target_zq = np.asarray(target_zq, dtype=np.int64)
    sys.stderr.write(
        f"[build] materialized {features.shape[0]} (image,zq) rows x "
        f"{features.shape[1]} inputs ({n_feat} feat + zq_norm), {n_cells} cells\n"
    )

    # Grouped-by-image split: sort distinct images, take val_frac as val.
    distinct = sorted(set(image_ids.tolist()))
    n_val = round(len(distinct) * args.val_frac)
    n_val = max(1 if len(distinct) > 1 else 0, min(n_val, len(distinct)))
    val_imgs = set(distinct[:n_val])
    split = np.array([1 if im in val_imgs else 0 for im in image_ids], dtype=np.int64)
    sys.stderr.write(
        f"[build] split: {len(distinct)} images, {n_val} val images; "
        f"{(split==0).sum()} train rows / {(split==1).sum()} val rows\n"
    )

    feature_names_full = [feat_names.get(c, c) for c in feat_cols]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        bytes_log=bytes_log,
        reach=reach,
        target_zq=target_zq,
        image_ids=image_ids,
        split=split,
        feature_names=np.asarray(feature_names_full),
        feat_col_names=np.asarray(feat_cols),
        cell_labels=np.asarray(cell_labels),
    )
    sys.stderr.write(f"[build] wrote cache -> {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
