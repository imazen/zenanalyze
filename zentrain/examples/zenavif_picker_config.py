"""
Codec config for zenavif's hybrid-heads picker — phase 1a sweep
(rav1e backend, ssim-tune, qm/vaq/strength fixed).

Used by `zentrain/tools/feature_ablation.py`,
`zentrain/tools/correlation_cleanup.py`, and
`zentrain/tools/train_hybrid.py`. Defines paths to the Pareto sweep
+ features TSVs, the feature subset the picker consumes, the
target_zq grid, and the regex that parses zenavif's config_name
strings.

Run from the zenavif checkout:

    cd ~/work/zen/zenavif
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/correlation_cleanup.py \\
            --codec-config zenavif_picker_config

    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/feature_ablation.py \\
            --codec-config zenavif_picker_config --method permutation

Cell taxonomy (CATEGORICAL_AXES — form cells):
  - speed ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  → 10 cells

Phase 1a holds qm=1, vaq=0, strength=1.0, tune=1 fixed; later phases
will add those axes. Once the sweep extends, this config grows
CATEGORICAL_AXES with `qm`, `vaq`, `tune` and SCALAR_AXES with
`vaq_strength` (continuous).

Scalar prediction heads (SCALAR_AXES):
  - none — only the q axis varies inside each cell, and that's the
    target_zq the picker is asked to hit.

So 10 cells × 1 bytes_log = 10 output dimensions.

The Pareto sweep emits config_name strings of the form:
  `s{speed}_q{q}_qm{qm}_vaq{vaq}_strength{strength}_tune{tune}`
e.g. `s1_q5_qm1_vaq0_strength1.0_tune1`,
     `s10_q100_qm1_vaq0_strength1.0_tune1`.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------- Paths ----------

# zenavif/benchmarks/dev/rav1e_phase1a_pareto.rs writes here. Bump
# dates when re-running the sweep. The 2026-04-30 pareto is unchanged
# (encode results still valid); the features file was re-extracted on
# 2026-05-01 against the post-cull zenanalyze 0.1.0 schema (97 → 99
# features: dropped 17 culled features, added 2 new shape/smoothness
# features — luma_kurtosis, gradient_fraction_smooth. The other 3
# (chroma_kurtosis, uniformity_smooth, flat_color_smooth) were Tier-0
# redundant on cross-codec ablation and removed before re-extraction).
PARETO = Path("benchmarks/rav1e_phase1a_2026-04-30.tsv")
FEATURES = Path("benchmarks/rav1e_phase1a_features_2026-05-01.tsv")

OUT_JSON = Path("benchmarks/rav1e_phase1a_hybrid_2026-05-01.json")
OUT_LOG = Path("benchmarks/rav1e_phase1a_hybrid_2026-05-01.log")


# ---------- Schema ----------

# Start with the broad zenanalyze v0.2 feature set (103 features).
# Run the four-tier ablation pipeline (zentrain/ABLATION.md) and
# tighten this list once cross-codec evidence supports specific
# drops. Until then, hand the picker every signal we have so the
# permutation-importance ranking is computed on the full surface.
# LOO multi-seed cull (2026-05-03, #52): 16 features whose mean ΔOH ≥
# +0.10pp AND ≥ +0.5σ across 5 seeds — removing them improves overall
# overhead consistently. Commented in-place with the LOO numbers so a
# future re-run can confirm the drops.
KEEP_FEATURES = [
    # Tier 1 (sparse stripe)
    "feat_variance",
    # DROPPED: feat_edge_density (mean +0.150pp, σ 0.212)
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    # DROPPED: feat_cr_sharpness (mean +0.100pp, σ 0.170)
    "feat_uniformity",
    "feat_flat_color_block_ratio",
    "feat_colourfulness",
    "feat_laplacian_variance",
    "feat_variance_spread",
    "feat_distinct_color_bins",
    # DROPPED: feat_palette_density (mean +0.288pp, σ 0.406 — biggest cull)
    "feat_cb_horiz_sharpness",  # NOTE: also +0.214pp/0.301σ — kept for now (cb_*_sharpness group coverage)
    "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness",
    "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness",
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
    # Tier 3 (sampled DCT blocks)
    "feat_dct_compressibility_y",
    "feat_dct_compressibility_uv",
    "feat_patch_fraction",
    "feat_patch_fraction_fast",
    "feat_quant_survival_y",
    # DROPPED: feat_quant_survival_uv (mean +0.200pp, σ 0.250)
    # DROPPED: feat_aq_map_mean (mean +0.170pp, σ 0.332)
    "feat_aq_map_std",
    # DROPPED: feat_aq_map_p50 (mean +0.166pp, σ 0.296)
    # DROPPED: feat_aq_map_p75 (mean +0.232pp, σ 0.135)
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    # DROPPED: feat_aq_map_p99 (mean +0.200pp, σ 0.351)
    "feat_noise_floor_y",
    "feat_noise_floor_uv",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p75",
    "feat_noise_floor_y_p90",
    "feat_noise_floor_uv_p25",
    "feat_noise_floor_uv_p50",
    # DROPPED: feat_noise_floor_uv_p75 (mean +0.220pp, σ 0.387)
    # DROPPED: feat_noise_floor_uv_p90 (mean +0.160pp, σ 0.197)
    # DROPPED: feat_laplacian_variance_p50 (mean +0.244pp, σ 0.369)
    # DROPPED: feat_laplacian_variance_p75 (mean +0.218pp, σ 0.179)
    # DROPPED: feat_laplacian_variance_p90 (mean +0.194pp, σ 0.252)
    "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak",
    "feat_quant_survival_y_p10",
    "feat_quant_survival_y_p25",
    "feat_quant_survival_y_p50",
    # DROPPED: feat_quant_survival_y_p75 (mean +0.242pp, σ 0.194)
    "feat_quant_survival_uv_p10",
    "feat_gradient_fraction",
    "feat_grayscale_score",
    "feat_edge_slope_stdev",
    # New experimental shape / smoothness features (post-cull addition,
    # zenanalyze 0.1.0 — 2 of 5 kept 2026-05-01 after Tier-0 ablation
    # removed chroma_kurtosis / uniformity_smooth / flat_color_smooth
    # as redundant).
    "feat_luma_kurtosis",
    # DROPPED: feat_gradient_fraction_smooth (mean +0.102pp, σ 0.164)
    # Dimension / shape
    "feat_pixel_count",
    "feat_min_dim",
    "feat_max_dim",
    "feat_aspect_min_over_max",
    "feat_log_aspect_abs",
    "feat_channel_count",
    # Alpha (corpus-conditional)
    "feat_alpha_present",
    "feat_alpha_used_fraction",
    "feat_alpha_bimodal_score",
]

# Zq target grid: production-relevant range. q < 30 is extreme-low
# quality that's rarely shipped; q > 95 saturates on tiny images
# regardless of codec setting (see #51 + #61).
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema ----------

CATEGORICAL_AXES = ["speed"]
SCALAR_AXES: list[str] = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}


# ---------- Per-feature pre-standardize transform ----------
#
# log: log-distributed positive features (pixel_count, dimensions).
# log1p: heavy-tailed positives (variance, laplacian variance, edge
#        slope stdev). aq_map_std also long-tailed.
# identity (default): bounded ratios, sharpness, scores, alpha bools.
FEATURE_TRANSFORMS = {
    "feat_pixel_count": "log",
    "feat_min_dim": "log",
    "feat_max_dim": "log",
    "feat_variance": "log1p",
    "feat_variance_spread": "log1p",
    "feat_laplacian_variance": "log1p",
    "feat_laplacian_variance_p99": "log1p",
    "feat_laplacian_variance_peak": "log1p",
    "feat_edge_slope_stdev": "log1p",
    "feat_aq_map_std": "log1p",
}


# ---------- ZNPR v3 output post-processing ----------
#
# Phase 1a holds qm/vaq/strength/tune fixed; the only output head is
# bytes_log per cell. SCALAR_AXES is empty so no further heads.
# Once phase 2 lands speed-as-scalar (or vaq_strength as continuous),
# add discrete_set+round (speed: 0..10) and identity+bounds for
# vaq_strength (~0.5..2.0).
OUTPUT_SPECS = {
    "bytes_log": {
        "bounds": [0.0, 30.0],
        "transform": "identity",
    },
}

SPARSE_OVERRIDES: list = []


# ---------- Config-name parser ----------

# Format: s{speed}_q{q}_qm{qm}_vaq{vaq}_strength{strength}_tune{tune}
# Examples:
#   s1_q5_qm1_vaq0_strength1.0_tune1
#   s10_q100_qm1_vaq0_strength1.0_tune1
_CONFIG_RE = re.compile(
    r"^s(?P<speed>\d+)_q(?P<q>\d+)"
    r"_qm(?P<qm>\d+)_vaq(?P<vaq>\d+)"
    r"_strength(?P<strength>[\d.]+)_tune(?P<tune>\d+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a zenavif config name into categorical + scalar axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenavif config name: {name}")
    return {
        "speed": int(m.group("speed")),
        # Phase 1a holds qm/vaq/strength/tune fixed; included here so
        # downstream tooling sees the full axis dict.
        "qm": int(m.group("qm")),
        "vaq": int(m.group("vaq")),
        "vaq_strength": float(m.group("strength")),
        "tune": int(m.group("tune")),
    }
