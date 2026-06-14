#!/usr/bin/env python3
"""Coverage-vs-K ablation: select a feasible representative subset of
imazen-26 `(image, crop)` sources for picker-MLP encode-sweep training.

Feature extraction is cheap; the *downstream* (encode-sweeping each source
across q × size × config to train picker MLPs) is ~100× costlier, so we
cluster the sources in zenanalyze CONTENT-feature space (geometry features
excluded — size is densified on the chosen reps, not selected here) and keep
the centroid-nearest representative of each of K clusters (the CLAUDE.md
"pick representative sources via k-means, not random" rule).

Phase 1 (`--ks`): sweep K, print coverage — variance explained + the
distribution of each unit's distance to its cluster centroid (how far a
typical corpus source sits from its nearest representative). Pick the knee.
Phase 2 (`--select-k`): final k-means at K, emit the representative manifest.

Unit = native rows (one feature vector per (image, crop) at full crop res).
"""
import argparse
import sys
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

GEOM = ("pixel_count", "log_pixels", "bitmap_bytes", "min_dim", "max_dim",
        "aspect", "block_misalign", "log_padded", "channel_count")


def load(parquet):
    pf = pq.ParquetFile(parquet)
    feats = [n for n in pf.schema.names if n.startswith("feat_")]
    content = [n for n in feats if not any(k in n for k in GEOM)]
    t = pq.read_table(
        parquet,
        columns=["image_path", "crop_label", "content_class", "size_class"] + content,
    )
    t = t.filter(pc.equal(t["size_class"], "native"))
    paths = t["image_path"].to_pylist()
    crops = t["crop_label"].to_pylist()
    cc = t["content_class"].to_pylist()
    M = np.empty((t.num_rows, len(content)), np.float64)
    for j, name in enumerate(content):
        col = t[name].to_numpy(zero_copy_only=False).astype(np.float64)
        med = np.nanmedian(col)
        med = med if np.isfinite(med) else 0.0
        M[:, j] = np.where(np.isfinite(col), col, med)
    std = M.std(0)
    keep = std > 1e-9
    M = M[:, keep]
    mu, sd = M.mean(0), M.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (M - mu) / sd
    return Z, paths, crops, cc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ks", default="100,200,300,500,750,1000,1500")
    ap.add_argument("--select-k", type=int, default=0)
    ap.add_argument("--out-manifest", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    Z, paths, crops, cc = load(a.parquet)
    n = Z.shape[0]
    total_ss = ((Z - Z.mean(0)) ** 2).sum()
    print(f"# units={n}  content_feats={Z.shape[1]}", file=sys.stderr)

    if not a.select_k:
        print("K\tvar_explained\tdist_p50\tdist_p95\tdist_max\tsingletons")
        for K in (int(x) for x in a.ks.split(",")):
            if K >= n:
                continue
            km = KMeans(n_clusters=K, n_init=3, random_state=a.seed).fit(Z)
            ve = 1 - km.inertia_ / total_ss
            d = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
            sing = int((np.bincount(km.labels_, minlength=K) == 1).sum())
            print(f"{K}\t{ve:.4f}\t{np.percentile(d,50):.3f}\t"
                  f"{np.percentile(d,95):.3f}\t{d.max():.3f}\t{sing}")
        return

    K = a.select_k
    km = KMeans(n_clusters=K, n_init=10, random_state=a.seed).fit(Z)
    sizes = np.bincount(km.labels_, minlength=K)
    rows = []
    for c in range(K):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1)
        r = int(idx[d.argmin()])
        rows.append((paths[r], crops[r], cc[r], c, int(sizes[c])))
    out = a.out_manifest or f"/mnt/v/output/imazen-26-features/imazen26_representatives_K{K}_2026-06-14.tsv"
    with open(out, "w") as f:
        f.write("image_path\tcrop_label\tcontent_class\tcluster_id\tcluster_size\n")
        for p, cr, klass, cid, sz in rows:
            f.write(f"{p}\t{cr}\t{klass}\t{cid}\t{sz}\n")
    # content-class coverage of the selected set vs the full population
    from collections import Counter
    selc = Counter(r[2] for r in rows)
    allc = Counter(cc)
    print(f"selected {len(rows)} representatives -> {out}", file=sys.stderr)
    print("content_class\tselected\tcorpus_units\tsel_share\tcorpus_share", file=sys.stderr)
    for klass in sorted(allc):
        print(f"{klass}\t{selc.get(klass,0)}\t{allc[klass]}\t"
              f"{selc.get(klass,0)/len(rows):.3f}\t{allc[klass]/n:.3f}", file=sys.stderr)
    print(f"singleton clusters (outliers kept): {int((sizes==1).sum())}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
