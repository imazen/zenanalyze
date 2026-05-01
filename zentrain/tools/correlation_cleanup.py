#!/usr/bin/env python3
"""
Correlation cleanup — pre-flight gate before model training / ablation.

Identifies pairs of features that are mathematically (or
near-mathematically) collinear, picks one canonical anchor per
cluster, and writes a drop-list. Catches features like
`feat_log_pixels`, `feat_sqrt_pixels`, `feat_log_padded_pixels_8`
that are pure transforms of `feat_pixel_count` — these have
Spearman = 1.0 with the canonical anchor by construction, so any
tree-based learner will route through one and ignore the rest.

Why this is the right pre-flight step:
  - Cheap. No model training. ~milliseconds at our corpus sizes.
  - Deterministic. Same input → same drop list.
  - Catches the "redundant transform" failure mode (zenanalyze
    issue #59) without needing a 103-retrain LOO pass.
  - Output feeds straight into `feature_ablation.py --keep-list`
    so the expensive pass works on the post-cleanup feature set.

Methodology:
  1. Load features TSV (codec config supplies path).
  2. Compute pairwise Spearman correlation across the entire
     feature matrix. Spearman (rank) is the right choice — `log(x)`
     and `x` have Spearman = 1.0 even though Pearson < 1.0.
  3. Cluster: features with |Spearman| >= --threshold (default
     0.99) form a cluster.
  4. Pick the canonical anchor in each cluster:
       - If `--prefer-list` matches one cluster member, that one
         wins. Lets callers force `feat_pixel_count` over its
         transforms.
       - Otherwise: shortest name (a heuristic for "most generic
         signal"), with a deterministic tie-break by alphabetical
         order.
  5. Emit the drop list + a per-cluster report so the operator
     can audit before applying.

Pit-of-success defaults:
  - Threshold 0.99 (not 0.95). Below 0.99 you start dropping
    features that are correlated but distinct (e.g.
    `cb_sharpness` vs `cr_sharpness` ≈ 0.92 — they're related but
    not redundant). 0.99+ is "the same signal in different units."
  - --prefer-list defaults to: the canonical raw signals
    (`feat_pixel_count`, `feat_min_dim`, `feat_max_dim`,
    `feat_variance`, `feat_edge_density`, ...) so log/sqrt/padded
    transforms get dropped in favor of their parent.

Usage:
  PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \\
    python3 <zenanalyze>/zentrain/tools/correlation_cleanup.py \\
        --codec-config zenjpeg_picker_config \\
        --threshold 0.99
"""

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np


# Default canonical anchors. Caller's --prefer-list extends this.
DEFAULT_CANONICAL_ANCHORS = [
    "feat_pixel_count",
    "feat_min_dim",
    "feat_max_dim",
    "feat_variance",
    "feat_edge_density",
    "feat_chroma_complexity",
    "feat_uniformity",
    "feat_distinct_color_bins",
    "feat_laplacian_variance",
    "feat_aq_map_mean",
    "feat_noise_floor_y",
    "feat_quant_survival_y",
    "feat_patch_fraction",
]


def load_features(path: Path):
    """Load features TSV. Returns (matrix, feat_cols, n_rows).

    matrix: numpy float32 array, shape (n_rows, n_features), one
    row per (image, size_class). feat_cols ordered to match.
    """
    rows = []
    feat_cols = None
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        feat_cols = all_cols
        n_dropped = 0
        for r in rdr:
            row_vals = []
            has_nan = False
            for c in feat_cols:
                v = r[c]
                if v == "" or v is None:
                    row_vals.append(float("nan"))
                    has_nan = True
                else:
                    fv = float(v)
                    if fv != fv:  # NaN sentinel from analyzer
                        has_nan = True
                    row_vals.append(fv)
            if has_nan:
                # Percentile features emit NaN when an image is too small
                # to satisfy the per-feature minimum-sample-count floor
                # (zenanalyze #49). For correlation analysis we want a
                # dense feature matrix, so drop those rows.
                n_dropped += 1
                continue
            rows.append(row_vals)
    if n_dropped:
        sys.stderr.write(
            f"Dropped {n_dropped} rows with NaN feature values "
            f"(tiny images skipping percentile features).\n"
        )
    return np.asarray(rows, dtype=np.float32), feat_cols, len(rows)


def detect_constant_columns(X: np.ndarray, feat_cols: list[str], min_unique: int = 2):
    """Find features that have fewer than `min_unique` distinct values
    across the corpus. These are NOT redundancy candidates — they're
    a corpus-coverage gap. The feature might carry signal but the
    corpus doesn't exercise it.

    Returns a list of dicts: `{name, n_unique, value}` for features
    that are effectively constant.
    """
    out = []
    for j, name in enumerate(feat_cols):
        col = X[:, j]
        n_unique = len(np.unique(col))
        if n_unique < min_unique:
            value = float(col[0]) if len(col) else float("nan")
            out.append({"name": name, "n_unique": n_unique, "value": value})
    return out


def detect_low_variance_columns(X: np.ndarray, feat_cols: list[str], min_unique: int = 5):
    """Features with <5 unique values that aren't fully constant.
    A separate bucket — these CAN feed correlation but the rank-based
    Spearman is degenerate (lots of ties), so any 'redundancy' claim
    against them is suspect."""
    out = []
    for j, name in enumerate(feat_cols):
        col = X[:, j]
        n_unique = len(np.unique(col))
        if 2 <= n_unique < min_unique:
            out.append({"name": name, "n_unique": n_unique})
    return out


def spearman_corr_matrix(X: np.ndarray) -> np.ndarray:
    """Spearman rank correlation across columns of X. O(n*p^2)
    where n = rows, p = columns. Symmetric, diag = 1.0.

    Caller MUST filter constant columns out before calling this —
    constants produce trivial Spearman = 1 with each other (rank of a
    constant is undefined → all-zero post-centering → cross-product
    of zero vectors).
    """
    # Rank each column. argsort.argsort gives ranks 0..n-1; ties
    # get distinct ranks (good enough for our 99%+ threshold —
    # tied-rank correction matters more for borderline cases).
    n, p = X.shape
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    # Center + scale so that row-wise mean = 0, std = 1; then
    # cross-product / n gives correlation.
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(ranks, axis=0)
    norms[norms == 0] = 1.0  # constant column → corr = 0 with everything
    ranks /= norms
    corr = ranks.T @ ranks
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def cluster_by_correlation(corr: np.ndarray, threshold: float):
    """Union-find clustering. Two features go into the same cluster
    if |corr| >= threshold. Returns a list of clusters (each a
    list of column indices); singletons are skipped."""
    p = corr.shape[0]
    parent = list(range(p))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(p):
        for j in range(i + 1, p):
            if abs(corr[i, j]) >= threshold:
                union(i, j)

    clusters = {}
    for i in range(p):
        clusters.setdefault(find(i), []).append(i)
    return [c for c in clusters.values() if len(c) > 1]


def pick_anchor(cluster_names, prefer_list):
    """Pick one canonical anchor from a cluster.

    Order of preference:
      1. First match in `prefer_list` (callers can force a specific
         feature to win).
      2. Shortest name (heuristic: 'feat_pixel_count' beats
         'feat_log_padded_pixels_8' because the bare signal is
         usually canonical).
      3. Alphabetical (deterministic tie-break).
    """
    for p in prefer_list:
        if p in cluster_names:
            return p
    by_short = sorted(cluster_names, key=lambda n: (len(n), n))
    return by_short[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--codec-config",
        help="Python module exporting `FEATURES = Path(...)` (the path "
        "to the per-image features TSV).",
    )
    ap.add_argument(
        "--features-tsv",
        type=Path,
        help="Direct path override; supersedes --codec-config.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="|Spearman| threshold for clustering. Default 0.99 — below "
        "0.99 you start dropping features that are correlated but "
        "distinct.",
    )
    ap.add_argument(
        "--prefer",
        action="append",
        default=[],
        help="Force this feature to win cluster anchoring if present. "
        "Repeatable; appended to DEFAULT_CANONICAL_ANCHORS.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        help="Write a machine-readable cluster report here. Defaults "
        "to <features-tsv stem>_correlation_cleanup.json next to the "
        "TSV.",
    )
    ap.add_argument(
        "--out-drop-list",
        type=Path,
        help="Write the drop list (one feature per line) here. "
        "Suitable for `feature_ablation.py --skip-list`.",
    )
    args = ap.parse_args()

    if args.features_tsv:
        features_path = args.features_tsv
    elif args.codec_config:
        sys.path.insert(0, "zentrain/tools")
        sys.path.insert(0, "zentrain/examples")
        mod = importlib.import_module(args.codec_config)
        features_path = Path(mod.FEATURES)
    else:
        sys.stderr.write("error: pass --codec-config or --features-tsv\n")
        sys.exit(1)

    if not features_path.exists():
        sys.stderr.write(f"error: {features_path} does not exist\n")
        sys.exit(1)

    sys.stderr.write(f"Loading {features_path}...\n")
    X, feat_cols, n_rows = load_features(features_path)
    sys.stderr.write(
        f"Loaded {n_rows} rows × {len(feat_cols)} features\n"
    )

    # Separate constants (corpus gap) from non-constant (real
    # candidates for correlation analysis). Two-bucket reporting —
    # the failure modes look identical to a naive Spearman pass but
    # require different remediation.
    constants = detect_constant_columns(X, feat_cols, min_unique=2)
    constant_names = {c["name"] for c in constants}
    low_var = detect_low_variance_columns(X, feat_cols, min_unique=5)
    sys.stderr.write(
        f"Constant columns (corpus gap, NOT redundancy): {len(constants)}\n"
    )
    sys.stderr.write(
        f"Low-variance columns (n_unique < 5): {len(low_var)}\n"
    )

    # Strip constants before computing correlation — they trivially
    # collide with each other at Spearman = 1 and pollute the cluster
    # output. They're a separate, more important finding.
    nonconst_idx = [i for i, n in enumerate(feat_cols) if n not in constant_names]
    X_nc = X[:, nonconst_idx]
    feat_cols_nc = [feat_cols[i] for i in nonconst_idx]

    sys.stderr.write(
        f"Computing Spearman correlation ({len(feat_cols_nc)}x{len(feat_cols_nc)}, "
        f"constants stripped)...\n"
    )
    corr = spearman_corr_matrix(X_nc)

    clusters_idx = cluster_by_correlation(corr, args.threshold)
    sys.stderr.write(
        f"Found {len(clusters_idx)} multi-feature clusters at "
        f"|Spearman| >= {args.threshold}\n"
    )

    prefer_list = args.prefer + DEFAULT_CANONICAL_ANCHORS

    report = []
    drop_list = []
    for cluster in clusters_idx:
        names = [feat_cols_nc[i] for i in cluster]
        anchor = pick_anchor(names, prefer_list)
        dropped = sorted(n for n in names if n != anchor)
        min_corr = 1.0
        for i in cluster:
            for j in cluster:
                if i != j:
                    min_corr = min(min_corr, abs(corr[i, j]))
        # Per-anchor n_unique tells callers whether this cluster's
        # signal is well-exercised by the corpus.
        anchor_idx_nc = feat_cols_nc.index(anchor)
        anchor_unique = int(len(np.unique(X_nc[:, anchor_idx_nc])))
        report.append({
            "anchor": anchor,
            "anchor_n_unique": anchor_unique,
            "dropped": dropped,
            "cluster_size": len(names),
            "min_abs_corr_in_cluster": float(min_corr),
        })
        drop_list.extend(dropped)

    report.sort(key=lambda r: r["anchor"])
    drop_list.sort()

    print("# Correlation cleanup report")
    print(f"# Source: {features_path}")
    print(f"# Threshold: |Spearman| >= {args.threshold}")
    print(f"# {n_rows} rows × {len(feat_cols)} features")
    print()

    if constants:
        print(
            f"## CORPUS GAP — {len(constants)} constant column(s) "
            f"(zero variance across the whole corpus)"
        )
        print(
            f"# These are NOT redundancy candidates. The feature might"
        )
        print(
            f"# carry real signal — the corpus just doesn't exercise it."
        )
        print(
            f"# Fix the corpus before drawing conclusions about these."
        )
        print()
        for c in constants:
            print(f"  constant: {c['name']:40s} = {c['value']}")
        print()

    if low_var:
        print(
            f"## LOW-VARIANCE — {len(low_var)} column(s) with < 5 unique values"
        )
        print(
            f"# Spearman is rank-based; <5 unique values means tied"
        )
        print(
            f"# ranks dominate. Any redundancy claim against these is"
        )
        print(
            f"# suspect. Treat as 'corpus coverage borderline'."
        )
        print()
        for lv in low_var:
            print(f"  low_var: {lv['name']:40s} n_unique = {lv['n_unique']}")
        print()

    if not report:
        print("## No multi-feature redundancy clusters found.")
        print()
    else:
        print(
            f"## REDUNDANCY — {len(clusters_idx)} cluster(s), {len(drop_list)} drop candidate(s)"
        )
        print(
            f"# These features are ≥ {args.threshold} Spearman-correlated."
        )
        print(
            f"# Tree learners route signal through one and ignore the rest."
        )
        print()
        for cluster in report:
            print(
                f"### Anchor: {cluster['anchor']}  "
                f"(cluster size {cluster['cluster_size']}, "
                f"anchor unique values {cluster['anchor_n_unique']}, "
                f"min |corr| {cluster['min_abs_corr_in_cluster']:.4f})"
            )
            for d in cluster["dropped"]:
                print(f"  drop: {d}")
            print()

    out_json = args.out_json or features_path.with_name(
        features_path.stem + "_correlation_cleanup.json"
    )
    out_json.write_text(
        json.dumps(
            {
                "source": str(features_path),
                "threshold": args.threshold,
                "n_rows": n_rows,
                "n_features_in": len(feat_cols),
                "n_features_dropped": len(drop_list),
                "n_constant": len(constants),
                "n_low_variance": len(low_var),
                "constant_columns": constants,
                "low_variance_columns": low_var,
                "clusters": report,
                "drop_list": drop_list,
            },
            indent=2,
        )
    )
    sys.stderr.write(f"Wrote {out_json}\n")

    if args.out_drop_list:
        args.out_drop_list.write_text("\n".join(drop_list) + ("\n" if drop_list else ""))
        sys.stderr.write(f"Wrote {args.out_drop_list}\n")


if __name__ == "__main__":
    main()
