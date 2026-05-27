#!/usr/bin/env python3
"""Assemble the dense-q picker training parquet from a zen-metrics sweep.

Joins the per-cell artifacts the sweep produced into the single parquet
schema `zenpicker-train` consumes (`build_picker_dataset`):

    image_basename : utf8      (per-(image,size) identity)
    codec          : utf8
    q              : int64
    knob_tuple_json: utf8       (canonical sorted JSON: subsampling|progressive|sharp_yuv|effort)
    score_zensim   : float64    (the reach-ladder quality target, higher=better)
    encoded_bytes  : float64
    encoded_sha256 : utf8       (content-addressed pointer into artifacts/)
    feat_0..feat_N : float64    (372 zensim features, per-image)

Per zensim/CLAUDE.md "always persist encoded variants": every encoded
file is content-addressed by sha256 into <out_root>/artifacts/<sha>.jpg
and the row references that sha. ALL metric variants from the sweep
Pareto TSV are carried through as extra `metric_*` columns (not just the
reach-ladder score) so a future learned-metric pass has them.

score_zensim source: the sweep's own `score_zensim_gpu`/`zensim_score`
columns are RAW (unmapped, can be negative AND non-monotone — the shipped
zensim metric has a documented correctness defect on photo content). We
instead use the batch-path ssim2-gpu score (correct-monotone 0-100) as
the reach-ladder target. zensim is trained to predict ssim2, so this is a
sound stand-in; the column keeps the picker's expected name `score_zensim`.

Inputs per size class `sz`:
  --pareto   pareto_<sz>.tsv         (encoded_bytes, encoded_filename, raw scores)
  --features features/feat_<sz>.parquet (feat_0..feat_N + zensim_score)
  --encoded-dir encoded/<sz>/         (the .jpg files to content-address)
  --ssim2    score_pairs_ssim2.tsv    (mapped ssim2-gpu, all sizes; joined by dist_path)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--sizes", default="sz512,sz256")
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--codec", default="zenjpeg")
    args = ap.parse_args()

    root = args.out_root
    art_dir = os.path.join(root, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    # ssim2 reach-ladder scores keyed by encoded dist_path (absolute).
    ssim2_path = os.path.join(root, "score_pairs_ssim2.tsv")
    ssim2_by_dist = {}
    for r in read_tsv(ssim2_path):
        ssim2_by_dist[os.path.abspath(r["dist_path"])] = float(r["ssim2_gpu"])

    out_rows = []
    n_feat = None
    metric_cols = set()

    for sz in args.sizes.split(","):
        sz = sz.strip()
        if not sz:
            continue
        pareto_path = os.path.join(root, f"pareto_{sz}.tsv")
        feat_path = os.path.join(root, "features", f"feat_{sz}.parquet")
        enc_dir = os.path.join(root, "encoded", sz)
        if not (os.path.exists(pareto_path) and os.path.exists(feat_path)):
            print(f"skip {sz}: missing pareto/feature file", file=sys.stderr)
            continue

        # Feature parquet → per-(image_path,q,knob) feature vector.
        ft = pq.read_table(feat_path)
        fcols = sorted(
            [n for n in ft.schema.names if n.startswith("feat_")],
            key=lambda n: int(n.split("_")[1]),
        )
        if n_feat is None:
            n_feat = len(fcols)
        fd = ft.to_pydict()
        feat_by_key = {}
        for i in range(ft.num_rows):
            key = (fd["image_path"][i], int(fd["q"][i]), fd["knob_tuple_json"][i])
            feat_by_key[key] = [float(fd[c][i]) for c in fcols]

        for r in read_tsv(pareto_path):
            img_path = r["image_path"]
            q = int(r["q"])
            knob = r["knob_tuple_json"]
            enc_fn = r.get("encoded_filename", "").strip()
            if not enc_fn:
                continue
            enc_path = os.path.join(enc_dir, enc_fn)
            if not os.path.exists(enc_path):
                continue
            dist_abs = os.path.abspath(enc_path)
            score = ssim2_by_dist.get(dist_abs)
            if score is None:
                continue  # no reach-ladder score → drop
            fk = (img_path, q, knob)
            feat = feat_by_key.get(fk)
            if feat is None:
                continue

            # Content-address the encoded bytes.
            sha = sha256_file(enc_path)
            art_path = os.path.join(art_dir, f"{sha}.jpg")
            if not os.path.exists(art_path):
                shutil.copy2(enc_path, art_path)

            image_basename = f"{os.path.splitext(os.path.basename(img_path))[0]}@{sz}"
            row = {
                "image_basename": image_basename,
                "codec": args.codec,
                "q": q,
                "knob_tuple_json": knob,
                "score_zensim": score,
                "encoded_bytes": float(r["encoded_bytes"]),
                "encoded_sha256": sha,
                "size_class": sz,
            }
            # carry ALL metric variants from the pareto TSV
            for k, v in r.items():
                if k.startswith("score_"):
                    try:
                        row[f"metric_{k[len('score_'):]}"] = float(v)
                        metric_cols.add(f"metric_{k[len('score_'):]}")
                    except (ValueError, TypeError):
                        pass
            for j, fv in enumerate(feat):
                row[f"feat_{j}"] = fv
            out_rows.append(row)

    if not out_rows:
        print("ERROR: no rows assembled", file=sys.stderr)
        sys.exit(1)

    # Build a stable column order.
    base = [
        "image_basename",
        "codec",
        "q",
        "knob_tuple_json",
        "score_zensim",
        "encoded_bytes",
        "encoded_sha256",
        "size_class",
    ]
    metric_list = sorted(metric_cols)
    feat_list = [f"feat_{j}" for j in range(n_feat)]
    cols = base + metric_list + feat_list

    arrays = {}
    for c in cols:
        if c in ("image_basename", "codec", "knob_tuple_json", "encoded_sha256", "size_class"):
            arrays[c] = pa.array([r[c] for r in out_rows], pa.string())
        elif c == "q":
            arrays[c] = pa.array([r[c] for r in out_rows], pa.int64())
        else:
            arrays[c] = pa.array([r.get(c, float("nan")) for r in out_rows], pa.float64())
    table = pa.table(arrays)

    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    pq.write_table(table, args.out_parquet, compression="zstd")
    print(
        f"wrote {table.num_rows} rows x {table.num_columns} cols -> {args.out_parquet}\n"
        f"  features: {n_feat}  metric variants: {metric_list}\n"
        f"  artifacts: {len(glob.glob(os.path.join(art_dir, '*.jpg')))} content-addressed .jpg",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
