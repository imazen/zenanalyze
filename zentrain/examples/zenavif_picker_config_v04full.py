"""zenavif v0.4-full picker config — matches the actual full-grid sweep.

Sweep grid (see /tmp/chunks_full_v04.jsonl):
  q ∈ {5,10,15,...,95}    (19 q values)
  speed ∈ {3, 5, 7, 9}    (4 speeds)

Cell taxonomy: speed-only (4 cells). No `tune` axis (zen-metrics CLI
doesn't expose one). Output: 1 bytes_log per cell + 1 time_log per cell
= 8 outputs.
"""
import os
import re
from pathlib import Path

# Paths
_ZENAVIF = Path(os.path.expanduser("~/work/zen/zenavif"))
PARETO = _ZENAVIF / "benchmarks" / "zenavif_pareto_v04full_2026-05-04_adapted.tsv"
FEATURES = _ZENAVIF / "benchmarks" / "zenavif_features_v04full_2026-05-04.tsv"
_ZENANALYZE = Path(os.path.expanduser("~/work/zen/zenanalyze"))
OUT_JSON = _ZENANALYZE / "benchmarks" / "zenavif_hybrid_v04full_2026-05-04.json"
OUT_LOG = _ZENANALYZE / "benchmarks" / "zenavif_hybrid_v04full_2026-05-04.log"

# Reuse v04 KEEP_FEATURES (same image domain)
KEEP_FEATURES = [
    "feat_variance", "feat_chroma_complexity", "feat_cb_sharpness",
    "feat_uniformity", "feat_flat_color_block_ratio", "feat_colourfulness",
    "feat_laplacian_variance", "feat_variance_spread",
    "feat_distinct_color_bins", "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness", "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness", "feat_cr_peak_sharpness",
    "feat_high_freq_energy_ratio", "feat_luma_histogram_entropy",
    "feat_dct_compressibility_y", "feat_dct_compressibility_uv",
    "feat_patch_fraction", "feat_patch_fraction_fast",
    "feat_quant_survival_y", "feat_aq_map_std", "feat_aq_map_p90",
    "feat_aq_map_p95", "feat_noise_floor_y", "feat_noise_floor_uv",
    "feat_noise_floor_y_p25", "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p75", "feat_noise_floor_y_p90",
    "feat_noise_floor_uv_p25", "feat_noise_floor_uv_p50",
    "feat_laplacian_variance_p99", "feat_laplacian_variance_peak",
    "feat_quant_survival_y_p10", "feat_quant_survival_y_p25",
    "feat_quant_survival_y_p50", "feat_quant_survival_uv_p10",
    "feat_gradient_fraction", "feat_grayscale_score",
    "feat_edge_slope_stdev", "feat_luma_kurtosis",
    "feat_pixel_count", "feat_min_dim", "feat_max_dim",
    "feat_aspect_min_over_max", "feat_log_aspect_abs", "feat_channel_count",
    "feat_alpha_present", "feat_alpha_used_fraction", "feat_alpha_bimodal_score",
]

ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))

# Cell taxonomy: speed only (4 cells)
CATEGORICAL_AXES = ["speed"]
SCALAR_AXES: list[str] = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}

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

OUTPUT_SPECS = {
    "bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"},
}
SPARSE_OVERRIDES: list = []

# Config name: s{speed}_q{q}  (no tune; trainer/adapter sets tune=0 absent)
_CONFIG_RE = re.compile(r"^s(?P<speed>\d+)_q(?P<q>\d+)(?:_t(?P<tune>\d+))?$")
def parse_config_name(name: str) -> dict:
    m = _CONFIG_RE.match(name)
    if not m: raise ValueError(f"unparseable v0.4full zenavif config: {name}")
    return {"speed": int(m.group("speed"))}
