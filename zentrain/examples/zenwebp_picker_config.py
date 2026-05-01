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

# zenwebp/dev/zenwebp_pareto.rs writes here. The v0.2 expansion adds
# partition_limit, multi_pass_stats, and cost_model to the encode grid
# (576 configs × 30 q × ~1264 image-instances = ~22M cells).
PARETO = Path("benchmarks/zenwebp_pareto_2026-05-01.tsv")
FEATURES = Path("benchmarks/zenwebp_pareto_features_2026-05-01.tsv")

OUT_JSON = Path("benchmarks/zenwebp_hybrid_2026-05-01.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_2026-05-01.log")


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


# ---------- Axis schema (v0.2) ----------
#
# Cells: method × segments × cost_model = 3 × 2 × 2 = 12 cells.
# Scalar heads: sns_strength + filter_strength + filter_sharpness +
#               partition_limit + multi_pass_stats. multi_pass_stats
#               is binary (0/1); only m4 cells have meaningful training
#               signal for it (m5/m6 don't honor the flag — see
#               LossyConfig::with_multi_pass_stats docstring).
#               Picker output for that head on m5/m6 cells is ignored
#               at runtime via the codec's `--method` gating.
# cost_model encoded as bool: 0 = ZenwebpDefault, 1 = StrictLibwebpParity.
CATEGORICAL_AXES = ["method", "segments", "cost_model_strict"]
SCALAR_AXES = [
    "sns_strength",
    "filter_strength",
    "filter_sharpness",
    "partition_limit",
    "multi_pass_stats",
]
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES = {
    "sns_strength": (0, 100),
    "filter_strength": (0, 100),
    "filter_sharpness": (0, 7),
    "partition_limit": (0, 100),
    "multi_pass_stats": (0, 1),
}


# ---------- Config-name parser ----------

# Format (v0.2):
#   m{method}_seg{seg}_cm{cm}_sns{sns}_fs{fs}_sh{sh}_pl{pl}_mp{mp}
# Examples:
#   m4_seg1_cm0_sns0_fs0_sh0_pl0_mp0
#       → method=4, segments=1, cost_model_strict=False,
#         sns=0, fs=0, sh=0, partition_limit=0, multi_pass_stats=False
#   m6_seg4_cm1_sns100_fs60_sh6_pl100_mp0
#       → method=6, segments=4, cost_model_strict=True,
#         sns=100, fs=60, sh=6, partition_limit=100, multi_pass_stats=False
_CONFIG_RE = re.compile(
    r"^m(?P<method>\d+)_seg(?P<seg>\d+)_cm(?P<cm>[01])"
    r"_sns(?P<sns>\d+)_fs(?P<fs>\d+)_sh(?P<sh>\d+)"
    r"_pl(?P<pl>\d+)_mp(?P<mp>[01])$"
)
# Legacy v0.1 format (2026-04-29 / 2026-04-30 bakes): no
# cost_model_strict / partition_limit / multi_pass_stats axes.
# Older bakes on disk are still useful for ablation / comparison runs;
# parse_config_name accepts both formats so student_permutation.py and
# friends can re-load them without re-training.
_CONFIG_RE_V01 = re.compile(
    r"^m(?P<method>\d+)_seg(?P<seg>\d+)"
    r"_sns(?P<sns>\d+)_fs(?P<fs>\d+)_sh(?P<sh>\d+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a zenwebp config name into categorical + scalar
    axes. Categorical axes are method, segments, cost_model_strict;
    everything else is a scalar prediction head.

    Accepts both:
      - v0.2: ``m4_seg1_cm0_sns0_fs0_sh0_pl0_mp0`` (current bakes)
      - v0.1: ``m4_seg1_sns0_fs0_sh0`` (2026-04-29 / 2026-04-30 bakes
        before cm/pl/mp axes were added). v0.1 rows default
        cost_model_strict=False, partition_limit=0, multi_pass_stats=0
        — i.e. they slot into the v0.2 cell that matches the legacy
        defaults so a v0.1 bake's outputs are interpretable under the
        v0.2 schema.
    """
    m = _CONFIG_RE.match(name)
    if m:
        return {
            "method": int(m.group("method")),
            "segments": int(m.group("seg")),
            "cost_model_strict": bool(int(m.group("cm"))),
            "sns_strength": float(m.group("sns")),
            "filter_strength": float(m.group("fs")),
            "filter_sharpness": float(m.group("sh")),
            "partition_limit": float(m.group("pl")),
            "multi_pass_stats": float(m.group("mp")),
        }
    m = _CONFIG_RE_V01.match(name)
    if m:
        return {
            "method": int(m.group("method")),
            "segments": int(m.group("seg")),
            "cost_model_strict": False,
            "sns_strength": float(m.group("sns")),
            "filter_strength": float(m.group("fs")),
            "filter_sharpness": float(m.group("sh")),
            "partition_limit": 0.0,
            "multi_pass_stats": 0.0,
        }
    raise ValueError(f"unparseable zenwebp config name: {name}")
