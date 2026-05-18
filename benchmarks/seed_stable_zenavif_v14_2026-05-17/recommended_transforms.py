"""Seed-stable transform recommendation.

Built by zentrain/tools/seed_stable_screen.py from 3 seed dirs:
  - benchmarks/multiseed_zenavif_v14_2026-05-17/seed_0xcafe
  - benchmarks/multiseed_zenavif_v14_2026-05-17/seed_0xbeef
  - benchmarks/multiseed_zenavif_v14_2026-05-17/seed_0xface

Strict-majority threshold: ≥0.51 agreement.
Features recommended in any seed: 42
Features in stable output: 31
Dropped (no clear majority): 8
Dropped (low coverage):      3
"""

FEATURE_TRANSFORMS = {
    'feat_aq_map_p90': 'winsor_p99',
    'feat_aq_map_p95': 'winsor_p99',
    'feat_aq_map_std': 'clip_then_log1p',
    'feat_aspect_min_over_max': 'winsor_then_log',
    'feat_cb_horiz_sharpness': 'winsor_then_signed_cbrt',
    'feat_cb_peak_sharpness': 'winsor_then_log',
    'feat_cb_sharpness': 'clip_then_log1p_then_winsor',
    'feat_cb_vert_sharpness': 'winsor_p99',
    'feat_chroma_complexity': 'winsor_then_log',
    'feat_colourfulness': 'clip_then_log1p_then_winsor',
    'feat_cr_horiz_sharpness': 'winsor_then_log',
    'feat_dct_compressibility_uv': 'winsor_p99',
    'feat_dct_compressibility_y': 'clip_then_log1p_then_winsor',
    'feat_flat_color_block_ratio': 'clip_then_log1p',
    'feat_gradient_fraction': 'clip_then_log1p',
    'feat_grayscale_score': 'winsor_then_log',
    'feat_high_freq_energy_ratio': 'winsor_then_log',
    'feat_laplacian_variance_p99': 'clip_then_log1p_then_winsor',
    'feat_laplacian_variance_peak': 'quantile_bins',
    'feat_log_aspect_abs': 'winsor_then_log',
    'feat_luma_histogram_entropy': 'clip_then_log1p',
    'feat_min_dim': 'clip_then_log1p',
    'feat_noise_floor_uv': 'winsor_then_log',
    'feat_noise_floor_uv_p50': 'winsor_then_log',
    'feat_noise_floor_y_p75': 'winsor_p99',
    'feat_noise_floor_y_p90': 'quantile_bins',
    'feat_patch_fraction_fast': 'winsor_then_log',
    'feat_quant_survival_y': 'clip_then_log1p',
    'feat_quant_survival_y_p25': 'winsor_then_log',
    'feat_uniformity': 'clip_then_log1p_then_winsor',
    'feat_variance_spread': 'clip_then_log1p',
}

FEATURE_TRANSFORM_PARAMS = {
    'feat_aq_map_p90': [4.54425, 5.03537],
    'feat_aq_map_p95': [4.80438, 5.1709],
    'feat_aq_map_std': [0.81211],
    'feat_aspect_min_over_max': [0.75, 1],
    'feat_cb_horiz_sharpness': [0.002495, 0.02136],
    'feat_cb_peak_sharpness': [0, 4],
    'feat_cb_sharpness': [0.00456275, 0, 0.0203448],
    'feat_cb_vert_sharpness': [0.0025925, 0.0215792],
    'feat_chroma_complexity': [0, 0.318672],
    'feat_colourfulness': [30.1589, 0, 4.15606],
    'feat_cr_horiz_sharpness': [0.010425, 0.0865758],
    'feat_dct_compressibility_uv': [3.37852, 15.3357],
    'feat_dct_compressibility_y': [20.3894, 0, 3.94088],
    'feat_flat_color_block_ratio': [0.183451],
    'feat_gradient_fraction': [0.433594],
    'feat_grayscale_score': [0.000216367, 0.976028],
    'feat_high_freq_energy_ratio': [0.1223, 0.277661],
    'feat_laplacian_variance_p99': [5.17248, 0, 0.254598],
    'feat_laplacian_variance_peak': [5.54518, 5.54518, 5.54518, 5.54518, 5.54518, 5.54518, 5.54518],
    'feat_log_aspect_abs': [0, 0.409379],
    'feat_luma_histogram_entropy': [3.69903],
    'feat_min_dim': [5.54518],
    'feat_noise_floor_uv': [0, 0.144945],
    'feat_noise_floor_uv_p50': [0, 0.37163],
    'feat_noise_floor_y_p75': [0.652496, 1],
    'feat_noise_floor_y_p90': [0.995525, 1, 1, 1, 1, 1, 1],
    'feat_patch_fraction_fast': [0, 0.044689],
    'feat_quant_survival_y': [0.0645493],
    'feat_quant_survival_y_p25': [0, 0.153439],
    'feat_uniformity': [0.577543, 0, 0.236515],
    'feat_variance_spread': [0.843616],
}
