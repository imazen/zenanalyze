"""
Codec config for zenavif's hybrid-heads picker — v0.4 (data-starvation
remediation).

Background: v0.3 picker shipped at val argmin acc 23.3%, mean overhead
5.60%, train→val gap +2.08pp = OVERFIT, data-starved (200 imgs ×
89.6k Pareto rows × 10 cells × 20 outputs).

v0.4 plan (per benchmarks/picker_v0.4_data_starvation_spec_zenavif_zenjxl.md
§3.1, §3.2, §3.4):
  - Corpus expanded to 587 imgs (mlp-tune-fast full).
  - Cell taxonomy collapsed: speed only, restricted to {3,5,7,9}.
  - Tune is now a scalar head (binary, snap-to-{0,1} post-decode).
  - Output dim count: 4 cells × (1 bytes_log + 1 tune) = 8 outputs.
  - Student MLP sized 100→32→16 (~4k params) per Hsu 10x rule.

Sweep grid (zen-metrics CLI):
  q ∈ {15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90}
  speed ∈ {3,5,7,9}, tune ∈ {0,1}
  Total: 587 × 16 × 4 × 2 = 75,008 encodes.

Score column: zensim (real XYB-Butteraugli zensim, NOT ssim2-as-zensim
substitution; the published zen-metrics 0.3.0 sweep emits real zensim).

Run from any directory (paths are absolute):

    cd ~/work/zen/zenavif
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/correlation_cleanup.py \\
            --codec-config zenavif_picker_config_v04

    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/feature_ablation.py \\
            --codec-config zenavif_picker_config_v04 --method permutation

Cell taxonomy (CATEGORICAL_AXES — form cells):
  - speed ∈ {3, 5, 7, 9}  → 4 cells

Scalar prediction heads (SCALAR_AXES):
  - tune ∈ {0, 1}  binary, snap post-decode

So 4 cells × (1 bytes_log + 1 tune) = 8 output dimensions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ---------- Paths ----------

# Bump these once the 2026-05-04 sweep finishes (vast.ai instances
# 36108332). The adapter writes to zenavif/benchmarks/.
_ZENAVIF = Path(os.path.expanduser("~/work/zen/zenavif"))
PARETO = _ZENAVIF / "benchmarks" / "zenavif_pareto_2026-05-04_v04_full.tsv"
FEATURES = _ZENAVIF / "benchmarks" / "zenavif_features_2026-05-04_v04_full.tsv"

_ZENANALYZE = Path(os.path.expanduser("~/work/zen/zenanalyze"))
OUT_JSON = _ZENANALYZE / "benchmarks" / "zenavif_hybrid_v04_2026-05-04.json"
OUT_LOG = _ZENANALYZE / "benchmarks" / "zenavif_hybrid_v04_2026-05-04.log"


# ---------- Schema ----------

# v0.4 KEEP_FEATURES — same post-LOO cull as v0.3 (52 features,
# committed at #43 LOO consensus), unchanged because the same image
# corpus drives the cull. Drop list mirrors v0.3:
#   feat_edge_density, feat_cr_sharpness, feat_palette_density,
#   feat_cb_horiz_sharpness, feat_quant_survival_uv, feat_aq_map_mean,
#   feat_aq_map_p50, feat_aq_map_p75, feat_aq_map_p99,
#   feat_noise_floor_uv_p75, feat_noise_floor_uv_p90,
#   feat_laplacian_variance_p50, feat_laplacian_variance_p75,
#   feat_laplacian_variance_p90, feat_quant_survival_y_p75,
#   feat_gradient_fraction_smooth.
KEEP_FEATURES = [
    # Tier 1 (sparse stripe)
    "feat_variance",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_uniformity",
    "feat_flat_color_block_ratio",
    "feat_colourfulness",
    "feat_laplacian_variance",
    "feat_variance_spread",
    "feat_distinct_color_bins",
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
    "feat_aq_map_std",
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    "feat_noise_floor_y",
    "feat_noise_floor_uv",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p75",
    "feat_noise_floor_y_p90",
    "feat_noise_floor_uv_p25",
    "feat_noise_floor_uv_p50",
    "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak",
    "feat_quant_survival_y_p10",
    "feat_quant_survival_y_p25",
    "feat_quant_survival_y_p50",
    "feat_quant_survival_uv_p10",
    "feat_gradient_fraction",
    "feat_grayscale_score",
    "feat_edge_slope_stdev",
    "feat_luma_kurtosis",
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
    # NOTE: 50 features in this list. v0.3 was 52 (post-LOO #43); the
    # 2 removed here (cb_horiz, palette_density) were already commented
    # out in v0.3 — keeping the v0.4 list explicit reduces drift.
]

# Zq target grid — same as v0.3 (production-relevant range).
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema ----------

# v0.4 collapsed taxonomy: speed-only cells, tune as scalar.
CATEGORICAL_AXES = ["speed"]
SCALAR_AXES: list[str] = ["tune"]
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {
    "tune": (0.0, 1.0),
}


# ---------- Per-feature pre-standardize transform ----------

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
# v0.4 has tune as a binary scalar head — round to {0, 1} at decode
# time via discrete_set + bounds.
OUTPUT_SPECS = {
    "bytes_log": {
        "bounds": [0.0, 30.0],
        "transform": "identity",
    },
    "tune": {
        "bounds": [0.0, 1.0],
        "transform": "discrete_set",
        "discrete_set": [0, 1],
    },
}

SPARSE_OVERRIDES: list = []


# ---------- Config-name parser ----------

# Format from zenmetrics_sweep_adapter.py: s{speed}_q{q}_t{tune}
# Example: s5_q75_t1
_CONFIG_RE = re.compile(
    r"^s(?P<speed>\d+)_q(?P<q>\d+)_t(?P<tune>\d+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a v0.4 zenavif config name into axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenavif v0.4 config name: {name}")
    return {
        "speed": int(m.group("speed")),
        "tune": int(m.group("tune")),
    }
