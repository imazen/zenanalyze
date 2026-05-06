"""zenjxl v0.4-full picker config — matches the actual full-grid sweep.

Sweep grid:
  q ∈ {75}                       (single dummy value; distance overrides)
  effort ∈ {3, 5, 7, 9}          (4 efforts)
  distance ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0}  (11)

Cell taxonomy: effort-only (4 cells). Distance is the scalar head
(continuous; controls quality target).
"""
import os, re
from pathlib import Path

_ZENJXL = Path(os.path.expanduser("~/work/zen/zenjxl"))
PARETO = _ZENJXL / "benchmarks" / "zenjxl_pareto_v06_2026-05-06_adapted.tsv"
FEATURES = _ZENJXL / "benchmarks" / "zenjxl_features_v04full_2026-05-04.tsv"
_ZENANALYZE = Path(os.path.expanduser("~/work/zen/zenanalyze"))
OUT_JSON = _ZENANALYZE / "benchmarks" / "zenjxl_hybrid_v06_2026-05-06.json"
OUT_LOG = _ZENANALYZE / "benchmarks" / "zenjxl_hybrid_v06_2026-05-06.log"

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

CATEGORICAL_AXES = ["effort"]
SCALAR_AXES = ["distance"]
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES = {"distance": (0.5, 12.0)}

FEATURE_TRANSFORMS = {
    "feat_pixel_count": "log", "feat_min_dim": "log", "feat_max_dim": "log",
    "feat_variance": "log1p", "feat_variance_spread": "log1p",
    "feat_laplacian_variance": "log1p", "feat_laplacian_variance_p99": "log1p",
    "feat_laplacian_variance_peak": "log1p", "feat_edge_slope_stdev": "log1p",
    "feat_aq_map_std": "log1p",
}

OUTPUT_SPECS = {
    "bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"},
    "distance":  {"bounds": [0.5, 12.0], "transform": "identity"},
}
SPARSE_OVERRIDES: list = []

_CONFIG_RE = re.compile(r"^e(?P<effort>\d+)(?:_d(?P<distance>[\d.?]+))?_q(?P<q>\d+)$")
def parse_config_name(name: str) -> dict:
    m = _CONFIG_RE.match(name)
    if not m: raise ValueError(f"unparseable v0.4full zenjxl config: {name}")
    return {
        "effort":   int(m.group("effort")),
        "distance": float(m.group("distance")) if m.group("distance") not in (None,"?","") else 1.0,
    }
