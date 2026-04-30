"""
Codec config for zenwebp's hybrid-heads picker.

Used by `zenpicker/tools/train_hybrid.py`. Defines paths to the Pareto
sweep + features TSVs, the feature subset the picker consumes, the
target_zq grid, the regex that parses zenwebp's config_name strings,
and the explicit axis schema (categorical cells × scalar prediction
heads).

Run training from the zenwebp checkout:

    cd ~/work/zen/zenwebp
    PYTHONPATH=~/work/zen/zenanalyze/zenpicker/examples:~/work/zen/zenanalyze/zenpicker/tools \\
        python3 ~/work/zen/zenanalyze/zenpicker/tools/train_hybrid.py \\
            --codec-config zenwebp_picker_config

zenwebp's encoder knobs (LossyConfig builder):
  - method (0..6)               — RD optimization tier
  - sns_strength (0..100)       — spatial noise shaping
  - filter_strength (0..100)    — loop filter strength
  - filter_sharpness (0..7)     — loop filter sharpness
  - segments (1..4)             — adaptive quantization segment count
  - quality (0..100)            — primary q dial; varied during the
                                  Pareto sweep, fed in as target_zq at
                                  picker inference (NOT a scalar head)

Cell taxonomy (CATEGORICAL_AXES — form cells):
  - method ∈ {4, 5, 6}    — m0-3 omitted (below production quality floor)
  - segments ∈ {1, 4}     — 2/3 omitted (rarely Pareto-winners)
  → 6 cells

Scalar prediction heads (SCALAR_AXES):
  - sns_strength
  - filter_strength
  - filter_sharpness

So 6 cells × (1 bytes_log + 3 scalars) = 24 output dimensions.

The Pareto sweep emits config_name strings of the form:
  `m{method}_seg{segments}_sns{sns}_fs{filter_strength}_sh{filter_sharpness}`
e.g. `m4_seg1_sns0_fs0_sh0`, `m6_seg4_sns100_fs60_sh6`.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------- Paths ----------

# zenwebp/dev/zenwebp_pareto.rs writes here. Bump dates when re-running
# the sweep on a new corpus or with a new config grid.
PARETO = Path("benchmarks/zenwebp_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zenwebp_pareto_features_2026-04-29.tsv")

OUT_JSON = Path("benchmarks/zenwebp_hybrid_2026-04-29.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_2026-04-29.log")


# ---------- Schema ----------

# "Safe prune" feature schema from the 2026-04-29 permutation-ablation
# pass (`tools/feature_ablation.py --method permutation`).
#
# An aggressive 22-feature prune (drop everything with Δ < +0.05pp)
# regressed the per-cell hybrid-heads student from 2.01% → 2.43% —
# the ablation's rankings come from a per-config bytes regression
# which has different signal structure than per-cell scalar regression.
# Mid-tier features (Δ +0.01..+0.04pp) help the hybrid-heads scalar
# heads even when they don't pay for themselves in the per-config
# baseline.
#
# So we only drop features the ablation flagged as **guaranteed
# zero-or-negative** on this corpus — these are safe to drop on any
# SDR/non-alpha-heavy training data:
#   - 8 HDR / wide-gamut signals (Δ = 0.00pp; corpus is SDR)
#   - 3 alpha signals (Δ = 0.00pp; corpus has minimal alpha variance)
#   - 6 negative-Δ features (model improved without them — redundant
#     with feat_uniformity / feat_laplacian_variance / feat_aq_map_*):
#       feat_variance (-0.03), feat_dct_compressibility_y (-0.03),
#       feat_dct_compressibility_uv (-0.01), feat_luma_histogram_entropy
#       (-0.01), feat_cr_horiz_sharpness (-0.03),
#       feat_cb_vert_sharpness (-0.04), feat_gradient_fraction (-0.09)
#
# Re-run ablation when the corpus changes substantially (HDR
# coverage, alpha-heavy training data, etc.).
KEEP_FEATURES = [
    # Top tier — each drop hurts > +0.39pp
    "feat_uniformity",
    "feat_aq_map_mean",
    "feat_edge_density",
    "feat_distinct_color_bins",
    "feat_laplacian_variance",
    # +0.10..+0.20pp
    "feat_aq_map_std",
    "feat_cr_sharpness",
    "feat_edge_slope_stdev",
    "feat_flat_color_block_ratio",
    "feat_palette_density",
    "feat_high_freq_energy_ratio",
    "feat_chroma_complexity",
    "feat_patch_fraction",
    "feat_text_likelihood",
    # +0.05..+0.09pp
    "feat_cb_sharpness",
    "feat_cb_horiz_sharpness",
    "feat_variance_spread",
    "feat_cr_peak_sharpness",
    "feat_natural_likelihood",
    "feat_grayscale_score",
    "feat_colourfulness",
    "feat_noise_floor_y",
    # +0.01..+0.04pp — kept because dropping them regressed the
    # per-cell student even though the per-config ablation said
    # they were marginal
    "feat_cr_vert_sharpness",
    "feat_cb_peak_sharpness",
    "feat_noise_floor_uv",
    "feat_skin_tone_fraction",
    "feat_screen_content_likelihood",
    "feat_indexed_palette_width",
    "feat_line_art_score",
    "feat_palette_fits_in_256",
]

# Zq target grid: step 5 from 0..70 + step 2 from 70..100. Same shape as
# zenjpeg — denser in the perceptibility band where 1-2 zensim points
# matter for production decisions.
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))


# ---------- Axis schema ----------

# Categorical axes (form cells via the tuple of these dict-key values).
CATEGORICAL_AXES = ["method", "segments"]

# Scalar prediction heads (per-cell continuous outputs the trainer
# regresses against the within-cell-optimal config's value).
SCALAR_AXES = ["sns_strength", "filter_strength", "filter_sharpness"]

# No sentinel axes — every reachable cell row carries a meaningful
# scalar value for every axis, so leave SCALAR_SENTINELS empty.
SCALAR_SENTINELS: dict = {}

# Display-only ranges for the training log.
SCALAR_DISPLAY_RANGES = {
    "sns_strength": (0, 100),
    "filter_strength": (0, 100),
    "filter_sharpness": (0, 7),
}


# ---------- Config-name parser ----------

# Format: m{method}_seg{segments}_sns{sns}_fs{filter_strength}_sh{filter_sharpness}
# Examples:
#   m4_seg1_sns0_fs0_sh0      → method=4, segments=1, sns=0,   fs=0,  sh=0
#   m6_seg4_sns100_fs60_sh6   → method=6, segments=4, sns=100, fs=60, sh=6
_CONFIG_RE = re.compile(
    r"^m(?P<method>\d+)_seg(?P<seg>\d+)"
    r"_sns(?P<sns>\d+)_fs(?P<fs>\d+)_sh(?P<sh>\d+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a zenwebp config name into categorical + scalar axes.

    Returns a dict with keys matching CATEGORICAL_AXES + SCALAR_AXES.
    """
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenwebp config name: {name}")
    return {
        "method": int(m.group("method")),
        "segments": int(m.group("seg")),
        "sns_strength": float(m.group("sns")),
        "filter_strength": float(m.group("fs")),
        "filter_sharpness": float(m.group("sh")),
    }
