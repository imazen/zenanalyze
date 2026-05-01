"""
Codec config for zenwebp's hybrid-heads picker — v0.2 schema with the
post-#42 zenanalyze dimension + percentile feature blocks. Pruned to
36 features via the 2026-04-30 ablation.

Used by `zentrain/tools/train_hybrid.py`. Defines paths to the Pareto
sweep + features TSVs, the feature subset the picker consumes, the
target_zq grid, the regex that parses zenwebp's config_name strings,
and the explicit axis schema (categorical cells × scalar prediction
heads).

Run training from the zenwebp checkout:

    cd ~/work/zen/zenwebp
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/train_hybrid.py \\
            --codec-config zenwebp_picker_config

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
PARETO = Path("benchmarks/zenwebp_pareto_2026-04-30_combined.tsv")
FEATURES = Path("benchmarks/zenwebp_pareto_features_2026-04-30_combined.tsv")

OUT_JSON = Path("benchmarks/zenwebp_hybrid_2026-04-30.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_2026-04-30.log")


# ---------- Schema ----------

# Pruned schema from the 2026-04-30 v0.2 ablation pass (Δ ≥ +0.05pp on
# permutation importance). 36 features that earn their keep — sorted
# by importance descending.
KEEP_FEATURES = [
    # Top tier (Δ ≥ +0.20pp)
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance",
    "feat_quant_survival_y",
    "feat_cb_sharpness",
    "feat_pixel_count",
    "feat_uniformity",
    "feat_distinct_color_bins",
    "feat_cr_sharpness",
    "feat_edge_density",
    "feat_noise_floor_y_p50",
    "feat_luma_histogram_entropy",
    # Mid tier (Δ +0.10..+0.20pp)
    "feat_natural_likelihood",
    "feat_quant_survival_y_p50",
    "feat_noise_floor_uv_p50",
    "feat_aq_map_mean",
    "feat_cr_horiz_sharpness",
    "feat_min_dim",
    "feat_edge_slope_stdev",
    "feat_laplacian_variance_p90",
    "feat_patch_fraction",
    "feat_max_dim",
    "feat_aspect_min_over_max",
    "feat_aq_map_p75",
    # Low tier (Δ +0.05..+0.10pp)
    "feat_cb_horiz_sharpness",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_uv",
    "feat_chroma_complexity",
    "feat_quant_survival_y_p75",
    "feat_aq_map_std",
    "feat_gradient_fraction",
    "feat_noise_floor_y_p75",
    "feat_screen_content_likelihood",
    "feat_high_freq_energy_ratio",
    "feat_colourfulness",
    "feat_quant_survival_uv",
]

# Zq target grid: production-relevant range. q < 30 corresponds to
# extreme-low quality that's rarely shipped (the per-zq-tail safety
# gate fired at zq=5 with 84.8% p99 overhead — that's just low-q
# noise, not a picker miscalibration). Cap below at 30 to drop the
# noise floor; cap above at 95 to stay below where data starvation
# bites tiny/small images (see imazen/zenanalyze#51).
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema ----------

CATEGORICAL_AXES = ["method", "segments"]
SCALAR_AXES = ["sns_strength", "filter_strength", "filter_sharpness"]
SCALAR_SENTINELS: dict = {}
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
    """Decompose a zenwebp config name into categorical + scalar axes."""
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
