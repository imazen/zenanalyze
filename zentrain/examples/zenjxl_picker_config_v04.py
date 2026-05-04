"""
Codec config for zenjxl's hybrid-heads picker — v0.4 (data-starvation
remediation).

Background: v0.3 picker shipped at val argmin acc 11.2%, mean overhead
4.57%, train→val gap +0.81pp = HEAVY data-starve (100 imgs ×
610k Pareto rows × 16 cells × 64 outputs). 16 cells × 100 unique
images is roughly 6 imgs/cell after train/val split.

v0.4 plan (per benchmarks/picker_v0.4_data_starvation_spec_zenavif_zenjxl.md
§3.1, §3.2, §3.4 + Open Question §7.5):
  - Corpus expanded to 587 imgs (mlp-tune-fast full).
  - zenjxl 0.2.1 published doesn't expose with_internal_params for the
    lossy path, so the cell taxonomy is restricted to the public knob
    surface: effort and distance only. ac_intensity, gaborish, patches,
    enhanced_clustering are NOT accessible via the public API in 0.2.1,
    so the original spec's 16-cell collapse to {(effort, ac_intensity)}
    is reduced further to {effort} cells with {distance} as scalar.
  - Cells: effort ∈ {3,5,7,9}  → 4 cells.
  - Scalars: distance (continuous, 0.5..12.0).
  - Total: 4 cells × (1 bytes_log + 1 distance) = 8 outputs.
  - Student MLP sized 100→48→20 (~6.5k params) per spec §3.4.

Sweep grid (zen-metrics CLI):
  q ∈ {5,10,15,20,25,30,40,50,60,70,80,90,95}
  distance ∈ {0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0}
  effort ∈ {3, 5, 7, 9}
  Total: 587 × 13 × 7 × 4 = 213,668 encodes.

Score column: ssim2 (zen-metrics 0.3.0's `score_ssim2`). Same metric
substitution caveat as v0.3: numerically distinct from real zensim,
DO NOT mix overhead numbers across codecs that use real zensim.

Run from any directory (paths are absolute):

    cd ~/work/zen/zenjxl
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/correlation_cleanup.py \\
            --codec-config zenjxl_picker_config_v04

    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/feature_ablation.py \\
            --codec-config zenjxl_picker_config_v04 --method permutation
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ---------- Paths ----------

# Bump these once the 2026-05-04 sweep finishes (vast.ai instance
# 36108335). The adapter writes to zenjxl/benchmarks/.
_ZENJXL = Path(os.path.expanduser("~/work/zen/zenjxl"))
PARETO = _ZENJXL / "benchmarks" / "zenjxl_pareto_2026-05-04_v04_full.tsv"
FEATURES = _ZENJXL / "benchmarks" / "zenjxl_features_2026-05-04_v04_full.tsv"

_ZENANALYZE = Path(os.path.expanduser("~/work/zen/zenanalyze"))
OUT_JSON = _ZENANALYZE / "benchmarks" / "zenjxl_hybrid_v04_2026-05-04.json"
OUT_LOG = _ZENANALYZE / "benchmarks" / "zenjxl_hybrid_v04_2026-05-04.log"


# ---------- Schema ----------

# v0.4 KEEP_FEATURES — start from v0.3's full set (68 features). The
# v0.3 LOO multiseed kept 64; we re-run LOO on the expanded corpus
# before deciding whether to drop any further (Tier 1.5 step in the
# trainer pipeline).
KEEP_FEATURES = [
    # Tier 1 (sparse stripe)
    "feat_variance",
    "feat_edge_density",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_uniformity",
    "feat_flat_color_block_ratio",
    "feat_colourfulness",
    "feat_laplacian_variance",
    "feat_variance_spread",
    "feat_distinct_color_bins",
    "feat_palette_density",
    "feat_cb_horiz_sharpness",
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
    "feat_quant_survival_uv",
    "feat_aq_map_mean",
    "feat_aq_map_std",
    "feat_aq_map_p50",
    "feat_aq_map_p75",
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    "feat_aq_map_p99",
    "feat_noise_floor_y",
    "feat_noise_floor_uv",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p75",
    "feat_noise_floor_y_p90",
    "feat_noise_floor_uv_p25",
    "feat_noise_floor_uv_p50",
    "feat_noise_floor_uv_p75",
    "feat_noise_floor_uv_p90",
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance_p90",
    "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak",
    "feat_quant_survival_y_p10",
    "feat_quant_survival_y_p25",
    "feat_quant_survival_y_p50",
    "feat_quant_survival_y_p75",
    "feat_quant_survival_uv_p10",
    "feat_gradient_fraction",
    "feat_grayscale_score",
    "feat_edge_slope_stdev",
    "feat_luma_kurtosis",
    "feat_gradient_fraction_smooth",
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

# Zq target grid — full production span; zenjxl handles low-q better
# than zenavif so we go down to 25.
ZQ_TARGETS = list(range(25, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema ----------

# v0.4 collapsed taxonomy: effort cells, distance scalar.
CATEGORICAL_AXES = ["effort"]
SCALAR_AXES: list[str] = ["distance"]
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {
    "distance": (0.5, 12.0),
}


# ---------- Per-feature pre-standardize transform ----------

FEATURE_TRANSFORMS = {
    "feat_pixel_count": "log",
    "feat_min_dim": "log",
    "feat_max_dim": "log",
    "feat_variance": "log1p",
    "feat_variance_spread": "log1p",
    "feat_laplacian_variance": "log1p",
    "feat_laplacian_variance_p50": "log1p",
    "feat_laplacian_variance_p75": "log1p",
    "feat_laplacian_variance_p90": "log1p",
    "feat_laplacian_variance_p99": "log1p",
    "feat_laplacian_variance_peak": "log1p",
    "feat_edge_slope_stdev": "log1p",
    "feat_aq_map_std": "log1p",
}


# ---------- ZNPR v3 output post-processing ----------
#
# Distance is continuous in jxl-encoder's API (0.0..15.0 typical).
# v0.4 picker outputs distance as a continuous scalar; downstream
# clamps to [0.5, 12.0] (the swept range).
OUTPUT_SPECS = {
    "bytes_log": {
        "bounds": [0.0, 30.0],
        "transform": "identity",
    },
    "distance": {
        "bounds": [0.5, 12.0],
        "transform": "identity",
    },
}

SPARSE_OVERRIDES: list = []


# ---------- Config-name parser ----------

# Format from zenmetrics_sweep_adapter.py: e{effort}_d{distance}_q{q}
# Example: e5_d2.0_q75
_CONFIG_RE = re.compile(
    r"^e(?P<effort>\d+)_d(?P<distance>[\d.?]+)_q(?P<q>\d+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a v0.4 zenjxl config name into axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenjxl v0.4 config name: {name}")
    return {
        "effort": int(m.group("effort")),
        "distance": float(m.group("distance")) if m.group("distance") not in ("?", "") else 1.0,
    }
