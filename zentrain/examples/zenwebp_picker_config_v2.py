"""zenwebp picker config v2 — adopts the v14+z_rmse sweep winners.

Same KEEP_FEATURES + paths as the production zenwebp_picker_config,
but replaces FEATURE_TRANSFORMS with the per-feature winners from
the 2026-05-17 v14 transform sweep
(`benchmarks/feat_xform_zenwebp_v14_2026-05-17/`) and adds
FEATURE_TRANSFORM_PARAMS for the parameterized variants.

Measured: +24.54 pp argmin acc / −2.62 pp mean overhead at the screen,
single seed `0xCAFE`. Multi-seed confirmation pending.

This is the production-ready config — uses the original features TSV
(no HVS — HVS features regress per
`benchmarks/hvs_features_picker_eval_2026-05-17.md`).
"""

from __future__ import annotations

from pathlib import Path

from zenwebp_picker_config import (  # noqa: F401
    PARETO, FEATURES,
    ZQ_TARGETS, CATEGORICAL_AXES, SCALAR_AXES, SCALAR_SENTINELS,
    SCALAR_DISPLAY_RANGES, FEATURE_GROUPS,
    TIME_COLUMN,
    OUTPUT_SPECS, SPARSE_OVERRIDES,
    parse_config_name,
    KEEP_FEATURES,
)

OUT_JSON = Path("benchmarks/zenwebp_hybrid_v2_2026-05-17.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_v2_2026-05-17.log")

# Per-feature transform winners from
# benchmarks/feat_xform_zenwebp_v14_2026-05-17/recommended_transforms.py
FEATURE_TRANSFORMS = {
    "feat_patch_fraction": "winsor_then_signed_cbrt",
    "feat_cr_horiz_sharpness": "winsor_then_log",
    "feat_noise_floor_uv_p50": "signed_cbrt_then_winsor",
    "feat_high_freq_energy_ratio": "quantile_bins",
    "feat_quant_survival_uv": "winsor_then_log",
    "feat_colourfulness": "clip_then_log1p_then_winsor",
    "feat_chroma_complexity": "winsor_then_log",
    "feat_distinct_color_bins": "signed_cbrt",
    "feat_gradient_fraction": "winsor_then_log",
    "feat_aq_map_std": "winsor_then_log",
    "feat_edge_density": "clip_then_log1p",
    "feat_noise_floor_y_p25": "signed_cbrt",
    "feat_noise_floor_y_p50": "signed_cbrt",
    "feat_quant_survival_y": "clip_then_log1p",
    "feat_quant_survival_y_p75": "clip_then_log1p",
    "feat_noise_floor_uv": "winsor_then_signed_cbrt",
    "feat_aq_map_p75": "signed_cbrt",
    "feat_quant_survival_y_p50": "clip_then_log1p",
    "feat_laplacian_variance_p50": "signed_cbrt",
    "feat_max_dim": "clip_then_log1p",
    "feat_laplacian_variance": "signed_cbrt",
    "feat_pixel_count": "clip_then_log1p",
    "feat_edge_slope_stdev": "signed_cbrt",
    "feat_cb_sharpness": "winsor_then_log",
    "feat_min_dim": "clip_then_log1p",
    "feat_laplacian_variance_p90": "clip_then_log1p",
    "feat_laplacian_variance_p75": "signed_cbrt",
    "feat_cb_horiz_sharpness": "winsor_then_log",
    "feat_noise_floor_y_p75": "signed_cbrt",
    "feat_aq_map_mean": "log",
    "feat_aspect_min_over_max": "winsor_then_log",
}

FEATURE_TRANSFORM_PARAMS = {
    "feat_patch_fraction": [0.0, 0.018555],
    "feat_cr_horiz_sharpness": [0.0005, 0.25653],
    "feat_noise_floor_uv_p50": [0.0, 0.754174],
    "feat_high_freq_energy_ratio": [
        0.068333, 0.088514, 0.10695, 0.12901,
        0.152232, 0.183683, 0.275646,
    ],
    "feat_quant_survival_uv": [0.000306, 0.058346],
    "feat_colourfulness": [4.75928, 0.0, 4.77076],
    "feat_chroma_complexity": [0.013007, 0.245517],
    "feat_gradient_fraction": [0.037109, 0.71875],
    "feat_aq_map_std": [0.641201, 1.23842],
    "feat_edge_density": [0.307634],
    "feat_quant_survival_y": [0.137814],
    "feat_quant_survival_y_p75": [0.222222],
    "feat_noise_floor_uv": [0.0, 0.136827],
    "feat_quant_survival_y_p50": [0.126984],
    "feat_max_dim": [5.54518],
    "feat_pixel_count": [10.7437],
    "feat_cb_sharpness": [0.000498, 0.031879],
    "feat_min_dim": [5.2575],
    "feat_laplacian_variance_p90": [4.29046],
    "feat_cb_horiz_sharpness": [0.0001305, 0.05774],
    "feat_aspect_min_over_max": [0.474609, 1.0],
}
