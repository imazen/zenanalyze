"""Seed-stable transform recommendation.

Built by zentrain/tools/seed_stable_screen.py from 3 seed dirs:
  - benchmarks/multiseed_zenjpeg_v14_2026-05-17/seed_0xcafe
  - benchmarks/multiseed_zenjpeg_v14_2026-05-17/seed_0xbeef
  - benchmarks/multiseed_zenjpeg_v14_2026-05-17/seed_0xface

Strict-majority threshold: ≥0.51 agreement.
Features recommended in any seed: 42
Features in stable output: 31
Dropped (no clear majority): 0
Dropped (low coverage):      11
"""

FEATURE_TRANSFORMS = {
    'feat_aq_map_mean': 'clip_then_log1p',
    'feat_aq_map_p75': 'clip_then_log1p_then_winsor',
    'feat_aq_map_p90': 'clip_then_log1p_then_winsor',
    'feat_aq_map_std': 'clip_then_log1p',
    'feat_aspect_min_over_max': 'log',
    'feat_cb_peak_sharpness': 'winsor_p99',
    'feat_chroma_complexity': 'winsor_then_log',
    'feat_colourfulness': 'winsor_then_log',
    'feat_cr_horiz_sharpness': 'winsor_then_log',
    'feat_cr_peak_sharpness': 'winsor_then_log',
    'feat_cr_sharpness': 'winsor_then_log',
    'feat_cr_vert_sharpness': 'winsor_p99',
    'feat_dct_compressibility_uv': 'winsor_p99',
    'feat_edge_slope_stdev': 'log',
    'feat_flat_color_block_ratio': 'clip_then_log1p',
    'feat_gradient_fraction': 'clip_then_log1p',
    'feat_gradient_fraction_smooth': 'clip_then_log1p_then_winsor',
    'feat_grayscale_score': 'winsor_then_log',
    'feat_laplacian_variance_p50': 'winsor_then_log',
    'feat_laplacian_variance_p75': 'winsor_then_log',
    'feat_laplacian_variance_p90': 'clip_then_log1p',
    'feat_luma_histogram_entropy': 'clip_then_log1p',
    'feat_luma_kurtosis': 'clip_then_log1p_then_winsor',
    'feat_noise_floor_y_p50': 'winsor_then_log',
    'feat_noise_floor_y_p90': 'winsor_then_log',
    'feat_patch_fraction_fast': 'winsor_then_log',
    'feat_quant_survival_uv': 'winsor_p99',
    'feat_quant_survival_y_p10': 'winsor_then_log',
    'feat_uniformity': 'clip_then_log1p',
    'feat_variance': 'clip_then_log1p_then_winsor',
    'feat_variance_spread': 'clip_then_log1p',
}

FEATURE_TRANSFORM_PARAMS = {
    'feat_aq_map_mean': [2.68991],
    'feat_aq_map_p75': [3.82676, 0, 0.84905],
    'feat_aq_map_p90': [4.46044, 0, 0.562031],
    'feat_aq_map_std': [1.30142],
    'feat_cb_peak_sharpness': [0, 3],
    'feat_chroma_complexity': [0, 0.320382],
    'feat_colourfulness': [0, 127.572],
    'feat_cr_horiz_sharpness': [0.0062, 0.0882567],
    'feat_cr_peak_sharpness': [2, 15.3333],
    'feat_cr_sharpness': [0.00365833, 0.0105277],
    'feat_cr_vert_sharpness': [0.008165, 0.064785],
    'feat_dct_compressibility_uv': [3.18587, 12.0563],
    'feat_flat_color_block_ratio': [0.191059],
    'feat_gradient_fraction': [0.467285],
    'feat_gradient_fraction_smooth': [0.834867, 0, 0.0395758],
    'feat_grayscale_score': [0.000788333, 0.857646],
    'feat_laplacian_variance_p50': [0, 3.41109],
    'feat_laplacian_variance_p75': [0, 4.61613],
    'feat_laplacian_variance_p90': [2.13231],
    'feat_luma_histogram_entropy': [3.66793],
    'feat_luma_kurtosis': [17.2163, 0, 3.537],
    'feat_noise_floor_y_p50': [0, 1],
    'feat_noise_floor_y_p90': [0.560663, 1],
    'feat_patch_fraction_fast': [0, 0.05013],
    'feat_quant_survival_uv': [0.0057895, 0.021531],
    'feat_quant_survival_y_p10': [0, 0.132275],
    'feat_uniformity': [0.59607],
    'feat_variance': [6.70698, 0, 1.10996],
    'feat_variance_spread': [0.888346],
}
