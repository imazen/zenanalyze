#!/usr/bin/env python3
"""Cluster source images on zenanalyze content features and pick K
centroid-nearest representatives.

Implements the zensim/CLAUDE.md "Dense sampling for trained models"
stratification rule: pick representative sources via k-means on a
feature-space embedding (the ``feat_*`` columns), choosing the
centroid-nearest member of each cluster rather than random sampling.

Input is the TSV produced by ``extract_features_for_picker`` (one row
per image, ``feat_<name>`` columns). Output is a newline-separated list
of chosen image paths plus a small JSON sidecar with the cluster
assignment + size of each cluster (singletons are interesting outliers
worth keeping, not noise).

Pure stdlib + numpy/sklearn (offline build-side tool, no runtime dep).
"""
import argparse
import json
import sys

import numpy as np


def read_tsv(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    return header, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="extract_features_for_picker TSV")
    ap.add_argument("--k", type=int, default=8, help="number of clusters / reps")
    ap.add_argument("--out-list", required=True, help="newline-separated chosen image paths")
    ap.add_argument("--out-json", required=True, help="cluster assignment sidecar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    header, rows = read_tsv(args.features)
    idx = {n: i for i, n in enumerate(header)}
    feat_cols = [n for n in header if n.startswith("feat_")]
    path_col = idx["image_path"]

    # Build feature matrix, dropping non-finite / empty cells (NaN -> col median).
    raw = np.full((len(rows), len(feat_cols)), np.nan, dtype=np.float64)
    paths = []
    for r, row in enumerate(rows):
        paths.append(row[path_col])
        for c, fc in enumerate(feat_cols):
            v = row[idx[fc]]
            if v == "":
                continue
            try:
                raw[r, c] = float(v)
            except ValueError:
                pass
    col_med = np.nanmedian(raw, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    inds = np.where(~np.isfinite(raw))
    raw[inds] = np.take(col_med, inds[1])

    # Drop zero-variance columns (e.g. size features that are constant on a
    # uniform-size sample) so they don't dominate / break z-scoring.
    std = raw.std(axis=0)
    keep = std > 1e-9
    X = raw[:, keep]
    kept_names = [fc for fc, k in zip(feat_cols, keep) if k]
    # Standardize so no single feature scale dominates the Euclidean k-means.
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    from sklearn.cluster import KMeans

    k = min(args.k, len(rows))
    km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    labels = km.fit_predict(X)

    chosen = []
    clusters = []
    for cl in range(k):
        members = np.where(labels == cl)[0]
        if members.size == 0:
            continue
        # centroid-nearest member
        d = np.linalg.norm(X[members] - km.cluster_centers_[cl], axis=1)
        pick = members[int(np.argmin(d))]
        chosen.append(paths[pick])
        clusters.append(
            {
                "cluster": cl,
                "size": int(members.size),
                "rep_path": paths[pick],
                "rep_dist": float(d.min()),
            }
        )

    with open(args.out_list, "w") as f:
        for p in chosen:
            f.write(p + "\n")
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "k": k,
                "n_samples": len(rows),
                "n_features_used": int(keep.sum()),
                "features_used": kept_names,
                "clusters": sorted(clusters, key=lambda c: -c["size"]),
            },
            f,
            indent=2,
        )
    print(f"chose {len(chosen)} representatives from {len(rows)} samples, k={k}", file=sys.stderr)
    for c in sorted(clusters, key=lambda c: -c["size"]):
        print(f"  cluster {c['cluster']:2d}  size={c['size']:4d}  {c['rep_path']}", file=sys.stderr)


if __name__ == "__main__":
    main()
