"""Seed-stable transform recommendation.

Built by zentrain/tools/seed_stable_screen.py from 3 seed dirs:
  - benchmarks/multiseed_zenwebp_v14_2026-05-17/seed_0xcafe
  - benchmarks/multiseed_zenwebp_v14_2026-05-17/seed_0xbeef
  - benchmarks/multiseed_zenwebp_v14_2026-05-17/seed_0xface

Strict-majority threshold: ≥0.51 agreement.
Features recommended in any seed: 32
Features in stable output: 31
Dropped (no clear majority): 1
Dropped (low coverage):      0
"""

FEATURE_TRANSFORMS = {
    'feat_aq_map_mean': 'log',
    'feat_aq_map_p75': 'signed_cbrt',
    'feat_aq_map_std': 'winsor_then_log',
    'feat_aspect_min_over_max': 'winsor_then_log',
    'feat_cb_horiz_sharpness': 'winsor_then_log',
    'feat_cb_sharpness': 'winsor_then_log',
    'feat_chroma_complexity': 'winsor_then_log',
    'feat_colourfulness': 'clip_then_log1p_then_winsor',
    'feat_cr_horiz_sharpness': 'winsor_then_log',
    'feat_distinct_color_bins': 'signed_cbrt',
    'feat_edge_density': 'clip_then_log1p',
    'feat_edge_slope_stdev': 'signed_cbrt',
    'feat_gradient_fraction': 'winsor_then_log',
    'feat_high_freq_energy_ratio': 'winsor_then_log',
    'feat_laplacian_variance': 'signed_cbrt',
    'feat_laplacian_variance_p50': 'signed_cbrt',
    'feat_laplacian_variance_p75': 'signed_cbrt',
    'feat_laplacian_variance_p90': 'clip_then_log1p',
    'feat_max_dim': 'clip_then_log1p',
    'feat_min_dim': 'clip_then_log1p',
    'feat_noise_floor_uv': 'winsor_then_signed_cbrt',
    'feat_noise_floor_uv_p50': 'winsor_then_signed_cbrt',
    'feat_noise_floor_y_p25': 'signed_cbrt',
    'feat_noise_floor_y_p50': 'signed_cbrt',
    'feat_noise_floor_y_p75': 'signed_cbrt',
    'feat_patch_fraction': 'winsor_then_signed_cbrt',
    'feat_pixel_count': 'clip_then_log1p',
    'feat_quant_survival_uv': 'winsor_then_log',
    'feat_quant_survival_y': 'clip_then_log1p',
    'feat_quant_survival_y_p50': 'clip_then_log1p',
    'feat_quant_survival_y_p75': 'clip_then_log1p',
}

FEATURE_TRANSFORM_PARAMS = {
    'feat_aq_map_std': [0.644739, 1.24941],
    'feat_aspect_min_over_max': [0.474609, 1],
    'feat_cb_horiz_sharpness': [0.000156833, 0.058007],
    'feat_cb_sharpness': [0.000521, 0.0321315],
    'feat_chroma_complexity': [0.0139673, 0.245517],
    'feat_colourfulness': [5.04115, 0, 4.76837],
    'feat_cr_horiz_sharpness': [0.000600333, 0.254917],
    'feat_edge_density': [0.31036],
    'feat_gradient_fraction': [0.0360512, 0.713867],
    'feat_high_freq_energy_ratio': [0.0751955, 0.244223],
    'feat_laplacian_variance_p90': [4.29046],
    'feat_max_dim': [5.54518],
    'feat_min_dim': [5.2575],
    'feat_noise_floor_uv': [0, 0.139839],
    'feat_noise_floor_uv_p50': [0, 0.417598],
    'feat_patch_fraction': [0, 0.0158857],
    'feat_pixel_count': [10.7116],
    'feat_quant_survival_uv': [0.000513667, 0.0568167],
    'feat_quant_survival_y': [0.139787],
    'feat_quant_survival_y_p50': [0.132275],
    'feat_quant_survival_y_p75': [0.216931],
}
