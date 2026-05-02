"""Build a hierarchical-clustering dendrogram + reordered correlation
heatmap + VIF table for a features TSV.

Outputs (paths take a `--label` slug — e.g., `zenwebp`, `zenjxl_lossy`):
- benchmarks/feature_groups_<label>_dendrogram_2026-05-02.svg
- benchmarks/feature_groups_<label>_heatmap_2026-05-02.svg
- benchmarks/feature_groups_<label>_clusters_2026-05-02.tsv

Usage:
    python3 benchmarks/feature_groups_2026-05-02.py \\
        --tsv <path> --label <slug>

Defaults to the zenwebp combined_filled TSV with label `zenwebp`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


ZA_ROOT = Path("/home/lilith/work/zen/zenanalyze")
ZW_ROOT = Path("/home/lilith/work/zen/zenwebp")
DEFAULT_TSV = ZW_ROOT / "benchmarks/zenwebp_pareto_features_2026-05-01_combined_filled.tsv"

# Feature-name prefix → semantic group label. Used as a sanity overlay
# on the auto-clusters; agreement validates, disagreement is interesting.
SEMANTIC_PREFIXES = [
    ("aq_map", "aq_map_distribution"),
    ("noise_floor_y", "noise_floor_y_dist"),
    ("noise_floor_uv", "noise_floor_uv_dist"),
    ("quant_survival_y", "quant_survival_y_dist"),
    ("quant_survival_uv", "quant_survival_uv_dist"),
    ("laplacian_variance", "laplacian_dist"),
    ("cb_horiz_sharpness", "cb_sharpness_axis"),
    ("cb_vert_sharpness", "cb_sharpness_axis"),
    ("cb_peak_sharpness", "cb_sharpness_axis"),
    ("cb_sharpness", "cb_sharpness_axis"),
    ("cr_horiz_sharpness", "cr_sharpness_axis"),
    ("cr_vert_sharpness", "cr_sharpness_axis"),
    ("cr_peak_sharpness", "cr_sharpness_axis"),
    ("cr_sharpness", "cr_sharpness_axis"),
    ("dct_compressibility", "dct_compressibility"),
    ("hdr_", "hdr"),
    ("wide_gamut", "wide_gamut"),
    ("gamut_coverage", "gamut_coverage"),
    ("alpha_", "alpha"),
    ("palette_", "palette"),
    ("distinct_color_bins", "palette"),
    ("indexed_palette_width", "palette"),
    ("log_padded_pixels", "block_grid_surface"),
    ("block_misalignment", "block_misalign"),
    ("pixel_count", "resolution"),
    ("log_pixels", "resolution"),
    ("bitmap_bytes", "resolution"),
    ("min_dim", "shape_dim"),
    ("max_dim", "shape_dim"),
    ("aspect_min_over_max", "aspect"),
    ("log_aspect_abs", "aspect"),
    ("edge_density", "edge"),
    ("edge_slope_stdev", "edge"),
    ("patch_fraction", "patch"),
    ("flat_color", "flat_color"),
    ("uniformity", "uniformity"),
    ("gradient_fraction", "gradient_fraction"),
    ("variance_spread", "variance_spread"),
    ("variance", "variance"),
    ("colourfulness", "colourfulness"),
    ("chroma_complexity", "chroma_complexity"),
    ("chroma_kurtosis", "chroma_kurtosis"),
    ("luma_kurtosis", "luma_kurtosis"),
    ("luma_histogram_entropy", "luma_histogram_entropy"),
    ("high_freq_energy_ratio", "high_freq_energy_ratio"),
    ("skin_tone_fraction", "skin"),
    ("grayscale_score", "grayscale"),
    ("is_grayscale", "grayscale"),
    ("channel_count", "channel_count"),
    ("effective_bit_depth", "effective_bit_depth"),
    ("peak_luminance_nits", "hdr"),
    ("p99_luminance_nits", "hdr"),
]


def semantic_prefix(feat: str) -> str:
    name = feat.removeprefix("feat_")
    for prefix, label in SEMANTIC_PREFIXES:
        if name.startswith(prefix):
            return label
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default=str(DEFAULT_TSV),
                        help="Input features TSV with feat_* columns.")
    parser.add_argument("--label", default="zenwebp",
                        help="Slug used in output filenames "
                             "(feature_groups_<label>_*.svg/.tsv).")
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    label = args.label
    out_dendro = ZA_ROOT / f"benchmarks/feature_groups_{label}_dendrogram_2026-05-02.svg"
    out_heatmap = ZA_ROOT / f"benchmarks/feature_groups_{label}_heatmap_2026-05-02.svg"
    out_clusters = ZA_ROOT / f"benchmarks/feature_groups_{label}_clusters_2026-05-02.tsv"

    sys.stderr.write(f"[groups] reading {tsv_path} (label={label})\n")
    df = pd.read_csv(tsv_path, sep="\t")
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    sys.stderr.write(f"[groups] {len(df)} rows × {len(feat_cols)} features\n")

    # Drop near-constant cols (n_unique < 2) — they break Spearman.
    keep = []
    dropped_const = []
    for c in feat_cols:
        n_unique = df[c].nunique(dropna=True)
        if n_unique < 2:
            dropped_const.append(c)
            continue
        keep.append(c)
    sys.stderr.write(
        f"[groups] dropped {len(dropped_const)} constant cols: "
        f"{', '.join(dropped_const[:6])}{'...' if len(dropped_const) > 6 else ''}\n"
    )

    X = df[keep].to_numpy(dtype=float)
    n_features = X.shape[1]

    # Spearman correlation matrix.
    sys.stderr.write("[groups] computing Spearman correlation...\n")
    corr, _ = spearmanr(X, axis=0, nan_policy="omit")
    if not isinstance(corr, np.ndarray):  # 2-feature edge case
        corr = np.array([[1.0, corr], [corr, 1.0]])
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 1.0)
    np.clip(abs_corr, 0.0, 1.0, out=abs_corr)

    dist = 1.0 - abs_corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # ensure symmetry

    sys.stderr.write("[groups] hierarchical clustering (average linkage)...\n")
    condensed = squareform(dist, checks=False)
    Z = sch.linkage(condensed, method="average")

    # Cut at multiple thresholds. Linkage distance = 1 - |ρ|; threshold
    # corresponds to "everything tighter than ρ ≥ threshold goes
    # together".
    THRESHOLDS = [0.99, 0.95, 0.90, 0.85]
    cluster_ids: dict[float, np.ndarray] = {}
    for t in THRESHOLDS:
        cluster_ids[t] = sch.fcluster(Z, t=1.0 - t, criterion="distance")
        n_clusters = len(np.unique(cluster_ids[t]))
        sys.stderr.write(f"  cut at ρ≥{t}: {n_clusters} clusters\n")

    # Variance Inflation Factor.
    sys.stderr.write("[groups] computing VIF (per-feature OLS R²)...\n")
    # Standardize for numerical stability.
    X_std = X.copy()
    finite_mask = np.isfinite(X_std).all(axis=1)
    X_std = X_std[finite_mask]
    means = X_std.mean(axis=0)
    stds = X_std.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X_norm = (X_std - means) / stds

    vif = np.zeros(n_features)
    for i in range(n_features):
        # OLS: predict X[:, i] from all others.
        others = np.delete(np.arange(n_features), i)
        A = X_norm[:, others]
        b = X_norm[:, i]
        # Add intercept (already mean-zero, but safety).
        A_ = np.column_stack([A, np.ones(A.shape[0])])
        # Use lstsq for numerical stability.
        coefs, *_ = np.linalg.lstsq(A_, b, rcond=None)
        pred = A_ @ coefs
        ss_res = np.sum((b - pred) ** 2)
        ss_tot = np.sum((b - b.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        r2 = float(min(max(r2, 0.0), 0.999_999))
        vif[i] = 1.0 / (1.0 - r2)
        if (i + 1) % 20 == 0:
            sys.stderr.write(f"  VIF {i + 1}/{n_features}\n")

    # Per-feature TSV.
    rows = []
    for i, name in enumerate(keep):
        rows.append({
            "feature": name,
            "semantic_prefix": semantic_prefix(name),
            "vif": vif[i],
            "cluster_at_0_99": int(cluster_ids[0.99][i]),
            "cluster_at_0_95": int(cluster_ids[0.95][i]),
            "cluster_at_0_90": int(cluster_ids[0.90][i]),
            "cluster_at_0_85": int(cluster_ids[0.85][i]),
        })
    out_df = pd.DataFrame(rows).sort_values(["cluster_at_0_85", "cluster_at_0_90", "cluster_at_0_95", "vif"])
    out_df.to_csv(out_clusters, sep="\t", index=False, float_format="%.4f")
    sys.stderr.write(f"[groups] wrote {out_clusters}\n")

    # Dendrogram.
    sys.stderr.write("[groups] rendering dendrogram...\n")
    fig_h = max(8.0, n_features * 0.20)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sch.dendrogram(
        Z,
        labels=keep,
        orientation="left",
        leaf_font_size=7,
        color_threshold=1.0 - 0.95,
        ax=ax,
    )
    ax.axvline(1.0 - 0.99, color="red", linestyle="--", linewidth=0.8, label="ρ≥0.99")
    ax.axvline(1.0 - 0.95, color="orange", linestyle="--", linewidth=0.8, label="ρ≥0.95")
    ax.axvline(1.0 - 0.90, color="goldenrod", linestyle=":", linewidth=0.8, label="ρ≥0.90")
    ax.axvline(1.0 - 0.85, color="gray", linestyle=":", linewidth=0.8, label="ρ≥0.85")
    ax.set_xlabel(r"Distance: $1 - |\rho_\mathrm{Spearman}|$  (smaller = tighter cluster)")
    ax.set_title(
        f"{label} features hierarchical clustering — average linkage on "
        f"$1 - |\\rho|$, n={len(df)} pareto rows, {n_features} features"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dendro, format="svg")
    plt.close(fig)
    sys.stderr.write(f"[groups] wrote {out_dendro}\n")

    # Reordered heatmap.
    sys.stderr.write("[groups] rendering reordered correlation heatmap...\n")
    leaf_order = sch.leaves_list(Z)
    abs_corr_reordered = abs_corr[np.ix_(leaf_order, leaf_order)]
    labels_reordered = [keep[i] for i in leaf_order]

    fig_size = max(10.0, n_features * 0.18)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(abs_corr_reordered, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(labels_reordered, rotation=90, fontsize=5)
    ax.set_yticklabels(labels_reordered, fontsize=5)
    ax.set_title(
        f"$|\\rho_\\mathrm{{Spearman}}|$ heatmap, dendrogram-ordered "
        f"({label}, n={len(df)})"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$|\rho|$", rotation=0, labelpad=12)
    fig.tight_layout()
    fig.savefig(out_heatmap, format="svg")
    plt.close(fig)
    sys.stderr.write(f"[groups] wrote {out_heatmap}\n")

    # Console summary: clusters at 0.95 with > 1 member.
    sys.stderr.write("\n[groups] redundancy clusters at ρ≥0.95 (size > 1):\n")
    df95 = pd.DataFrame({"feat": keep, "cluster": cluster_ids[0.95]})
    by_cluster = df95.groupby("cluster")["feat"].apply(list)
    for cid, members in by_cluster.items():
        if len(members) <= 1:
            continue
        prefixes = sorted(set(semantic_prefix(m) for m in members))
        flag = "  " if len(prefixes) == 1 else " *"  # * = cross-prefix
        sys.stderr.write(f"{flag} cluster #{cid:03d} ({len(members)}): {', '.join(members)}\n")
        if len(prefixes) > 1:
            sys.stderr.write(f"     mixed prefixes: {prefixes}\n")

    # VIF tail summary.
    sys.stderr.write("\n[groups] highest-VIF features (top 15):\n")
    for _, r in out_df.nlargest(15, "vif").iterrows():
        sys.stderr.write(
            f"  VIF={r['vif']:>9.1f}  {r['feature']:<35s}  "
            f"prefix={r['semantic_prefix']}  c95={int(r['cluster_at_0_95'])}\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
