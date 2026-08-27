# Feature consumption inventory

Which zenanalyze features each downstream bake consumes, aggregated from the bake artifacts by `tools/feature_inventory.py` (zenanalyze#41). Regenerate with `just feature-inventory`; do not edit by hand.

- generated: 2026-08-27
- command: `tools/feature_inventory.py --universe target/inventory/universe.txt --inspect-bin target/release/zenpredict-inspect --label zenjpeg-a-v3-shipped=../zenjpeg/zenjpeg/src/encode/picker_data/feature_order.txt --label zenavif-rav1e-v0.1.1-shipped=../zenavif/src/models/rav1e_picker_v0_1_1.bin --label zenjpeg-v0.5-modesfull=../zenjpeg/benchmarks/zenjpeg_picker_v0.5_modesfull-tiled-evenodd_2026-06-28.manifest.json --label zenjpeg-lossy-ssim2-v0.1=../zenjpeg/benchmarks/zenjpeg_lossy_ssim2_picker_v0.1_K3_cleansplit_2026-06-29.manifest.json --label zenavif-lossy-zensim-v0.1=../zenavif/benchmarks/pickers/zenavif_lossy_mlp_zensim_v0.1_2026-06-28.manifest.json --label zenjxl-lossy-ssim2-v0.1=../zenjxl/benchmarks/zenjxl_lossy_ssim2_picker_v0.1_cleansplit_2026-06-29.manifest.json --label zenjpeg-v2.1-full=zentrain/testdata/zenjpeg_picker_v2.1_full.manifest.json --label zenjxl-v0.7b=benchmarks/zenjxl_picker_v0.7b_2026-05-06.manifest.json --label meta-v0.5-5codec=benchmarks/zenpicker_meta_v0.5_5codec_2026-05-06.manifest.json --out docs/feature-consumption.md`

## Bakes

| label | source | features | schema_hash | tag / version |
|---|---|---:|---|---|
| zenjpeg-a-v3-shipped | `…/src/encode/picker_data/feature_order.txt` | 108 |  |  |
| zenavif-rav1e-v0.1.1-shipped | `…/zenavif/src/models/rav1e_picker_v0_1_1.bin` | 43 | 0x9ff47e14b8c8ff0f | zentrain.v1.generic / zenanalyze 0.2.0 |
| zenjpeg-v0.5-modesfull | `…/zen/zenjpeg/benchmarks/zenjpeg_picker_v0.5_modesfull-tiled-evenodd_2026-06-28.manifest.json` | 50 | 0x676f03b6e7401d90 | zentrain.v1.generic |
| zenjpeg-lossy-ssim2-v0.1 | `…/zen/zenjpeg/benchmarks/zenjpeg_lossy_ssim2_picker_v0.1_K3_cleansplit_2026-06-29.manifest.json` | 67 | 0x884858d204d3bb8e | zentrain.v1.generic |
| zenavif-lossy-zensim-v0.1 | `…/zenavif/benchmarks/pickers/zenavif_lossy_mlp_zensim_v0.1_2026-06-28.manifest.json` | 50 | 0x676f03b6e7401d90 | zentrain.v1.generic |
| zenjxl-lossy-ssim2-v0.1 | `…/zen/zenjxl/benchmarks/zenjxl_lossy_ssim2_picker_v0.1_cleansplit_2026-06-29.manifest.json` | 97 | 0x424a9cc3c7b6b31d | zentrain.v1.generic |
| zenjpeg-v2.1-full | `zentrain/testdata/zenjpeg_picker_v2.1_full.manifest.json` | 35 | 0xdea793c422abe3db | zenpicker.v1.generic |
| zenjxl-v0.7b | `benchmarks/zenjxl_picker_v0.7b_2026-05-06.manifest.json` | 98 | 0x5896532033934e16 | zenjxl.picker.v0.7b.gated |
| meta-v0.5-5codec | `benchmarks/zenpicker_meta_v0.5_5codec_2026-05-06.manifest.json` | 20 | 0x30b45862aa501ff0 | zenpicker.metapicker.v0.5.5codec |

## Consumption matrix (121 features × 9 bakes)

Sorted by number of consuming bakes, then name.

| feature | n | zenjpeg-a-v3-shipped | zenavif-rav1e-v0.1.1-shipped | zenjpeg-v0.5-modesfull | zenjpeg-lossy-ssim2-v0.1 | zenavif-lossy-zensim-v0.1 | zenjxl-lossy-ssim2-v0.1 | zenjpeg-v2.1-full | zenjxl-v0.7b | meta-v0.5-5codec |
|---|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `feat_chroma_complexity` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_colourfulness` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_dct_compressibility_uv` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_dct_compressibility_y` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_edge_density` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_flat_color_block_ratio` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_gradient_fraction` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_high_freq_energy_ratio` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_laplacian_variance` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_luma_histogram_entropy` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_uniformity` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_variance` | 9 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_aq_map_mean` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_aq_map_std` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_aspect_min_over_max` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| `feat_cb_horiz_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cb_peak_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cb_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cb_vert_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_horiz_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_peak_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_vert_sharpness` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_edge_slope_stdev` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_grayscale_score` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_log_pixels` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| `feat_noise_floor_uv` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_noise_floor_y` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_patch_fraction_fast` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_quant_survival_uv` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_quant_survival_y` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_variance_spread` | 8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_alpha_bimodal_score` | 7 | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_alpha_used_fraction` | 7 | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_channel_count` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_distinct_color_bins` | 6 | ✓ | ✓ |  | ✓ |  | ✓ | ✓ | ✓ |  |
| `feat_gradient_fraction_smooth` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p50` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p75` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p90` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p99` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_peak` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_luma_kurtosis` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_pixel_count` | 6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p75` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p90` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p95` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p99` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_log_aspect_abs` | 5 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p50` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p90` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_patch_fraction` | 5 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p10` | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_skin_tone_fraction` | 5 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_alpha_present` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_bitmap_bytes` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_block_misalignment_32` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_block_misalignment_8` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_is_grayscale` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p1` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p10` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p5` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_log_padded_pixels_16` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_log_padded_pixels_32` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_log_padded_pixels_8` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_max_dim` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_min_dim` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_palette_density` | 4 | ✓ | ✓ |  |  |  |  | ✓ | ✓ |  |
| `feat_palette_fits_in_256` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_palette_log2_size` | 4 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_aq_map_p1` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_aq_map_p10` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_aq_map_p5` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_aq_map_p50` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_chroma_luma_covariance_cb` | 3 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_chroma_luma_covariance_cr` | 3 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_info_weight_mean` | 3 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_noise_floor_uv_p25` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_uv_p50` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_uv_p75` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_uv_p90` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p1` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p10` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p25` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p5` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p75` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_orientation_energy_ratio` | 3 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_quant_survival_uv_p10` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_uv_p25` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_uv_p50` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_uv_p75` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p1` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p25` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p5` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p50` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p75` | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_spectral_slope_y` | 3 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_cclass_document` | 2 |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_lineart` | 2 |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_photo` | 2 |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_screen` | 2 |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_synthetic` | 2 |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_info_weight_p90` | 2 | ✓ |  |  |  |  | ✓ |  |  |  |
| `feat_line_art_score` | 2 |  | ✓ |  |  |  |  | ✓ |  |  |
| `feat_effective_bit_depth` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_gamut_coverage_p3` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_gamut_coverage_srgb` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_hdr_headroom_stops` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_hdr_pixel_fraction` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_hdr_present` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_log_dist` | 1 |  |  |  |  |  |  |  | ✓ |  |
| `feat_log_max_dim` | 1 |  | ✓ |  |  |  |  |  |  |  |
| `feat_log_min_dim` | 1 |  | ✓ |  |  |  |  |  |  |  |
| `feat_natural_likelihood` | 1 |  | ✓ |  |  |  |  |  |  |  |
| `feat_p99_luminance_nits` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_peak_luminance_nits` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_screen_content_likelihood` | 1 |  | ✓ |  |  |  |  |  |  |  |
| `feat_target_band` | 1 |  |  |  |  |  |  |  |  | ✓ |
| `feat_text_likelihood` | 1 |  | ✓ |  |  |  |  |  |  |  |
| `feat_wide_gamut_fraction` | 1 | ✓ |  |  |  |  |  |  |  |  |
| `feat_wide_gamut_peak` | 1 | ✓ |  |  |  |  |  |  |  |  |

## Shared by ≥ 2 bakes (104)

First-class hot-path candidates per #41.

- `feat_chroma_complexity` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_colourfulness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_dct_compressibility_uv` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_dct_compressibility_y` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_edge_density` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_flat_color_block_ratio` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_gradient_fraction` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_high_freq_energy_ratio` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_laplacian_variance` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_luma_histogram_entropy` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_uniformity` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_variance` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_aq_map_mean` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_aq_map_std` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_aspect_min_over_max` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cb_horiz_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cb_peak_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cb_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cb_vert_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cr_horiz_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cr_peak_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cr_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_cr_vert_sharpness` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_edge_slope_stdev` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_grayscale_score` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_log_pixels` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b, meta-v0.5-5codec
- `feat_noise_floor_uv` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_noise_floor_y` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_patch_fraction_fast` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_quant_survival_uv` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_quant_survival_y` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_variance_spread` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_alpha_bimodal_score` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_alpha_used_fraction` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_channel_count` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_distinct_color_bins` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_gradient_fraction_smooth` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p50` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p75` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p90` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p99` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_peak` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_luma_kurtosis` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_pixel_count` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p75` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p90` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p95` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p99` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_log_aspect_abs` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p50` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p90` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_patch_fraction` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_y_p10` — zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_skin_tone_fraction` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_alpha_present` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_bitmap_bytes` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_block_misalignment_32` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_block_misalignment_8` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_is_grayscale` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p1` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p10` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_laplacian_variance_p5` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_log_padded_pixels_16` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_log_padded_pixels_32` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_log_padded_pixels_8` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_max_dim` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_min_dim` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_palette_density` — zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v2.1-full, zenjxl-v0.7b
- `feat_palette_fits_in_256` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_palette_log2_size` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p1` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p10` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p5` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_aq_map_p50` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_chroma_luma_covariance_cb` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1
- `feat_chroma_luma_covariance_cr` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1
- `feat_info_weight_mean` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1
- `feat_noise_floor_uv_p25` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_uv_p50` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_uv_p75` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_uv_p90` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p1` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p10` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p25` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p5` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_noise_floor_y_p75` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_orientation_energy_ratio` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1
- `feat_quant_survival_uv_p10` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_uv_p25` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_uv_p50` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_uv_p75` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_y_p1` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_y_p25` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_y_p5` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_y_p50` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_quant_survival_y_p75` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b
- `feat_spectral_slope_y` — zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1
- `feat_cclass_document` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_lineart` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_photo` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_screen` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_synthetic` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_info_weight_p90` — zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1
- `feat_line_art_score` — zenavif-rav1e-v0.1.1-shipped, zenjpeg-v2.1-full

## Consumed by exactly one bake

- **zenjpeg-a-v3-shipped** (10): `feat_peak_luminance_nits`, `feat_p99_luminance_nits`, `feat_hdr_headroom_stops`, `feat_hdr_pixel_fraction`, `feat_wide_gamut_peak`, `feat_wide_gamut_fraction`, `feat_effective_bit_depth`, `feat_hdr_present`, `feat_gamut_coverage_srgb`, `feat_gamut_coverage_p3`
- **zenavif-rav1e-v0.1.1-shipped** (5): `feat_text_likelihood`, `feat_screen_content_likelihood`, `feat_natural_likelihood`, `feat_log_min_dim`, `feat_log_max_dim`
- **zenjpeg-v0.5-modesfull** (0): —
- **zenjpeg-lossy-ssim2-v0.1** (0): —
- **zenavif-lossy-zensim-v0.1** (0): —
- **zenjxl-lossy-ssim2-v0.1** (0): —
- **zenjpeg-v2.1-full** (0): —
- **zenjxl-v0.7b** (1): `feat_log_dist`
- **meta-v0.5-5codec** (1): `feat_target_band`

## Never consumed (9 of 117 in the universe)

Supported by this zenanalyze build, requested by none of the bakes above.

- `feat_xyb444_color_loss`
- `feat_xyb_bquarter_chroma_loss`
- `feat_chroma_subsample_dct_loss`
- `feat_highlight_luma_mean`
- `feat_highlight_luma_std`
- `feat_highlight_chroma_mean`
- `feat_highlight_chroma_std`
- `feat_highlight_edge_count`
- `feat_highlight_orientation_ratio`

## Consumed but not in the universe (13)

Bake columns this build's analyzer does not produce. Either caller-supplied inputs that ride in the `feat_` namespace (content-class one-hots `feat_cclass_*`, `feat_log_dist`, `feat_target_band`) — fine — or analyzer features that were retired, renamed, or gated off by a cargo feature, in which case the bake cannot be served by this build as-is (it pins the zenanalyze it was extracted with; see its `zentrain.analyzer_version`).

- `feat_cclass_document` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_lineart` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_photo` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_screen` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_cclass_synthetic` — zenjxl-v0.7b, meta-v0.5-5codec
- `feat_line_art_score` — zenavif-rav1e-v0.1.1-shipped, zenjpeg-v2.1-full
- `feat_log_dist` — zenjxl-v0.7b
- `feat_log_max_dim` — zenavif-rav1e-v0.1.1-shipped
- `feat_log_min_dim` — zenavif-rav1e-v0.1.1-shipped
- `feat_natural_likelihood` — zenavif-rav1e-v0.1.1-shipped
- `feat_screen_content_likelihood` — zenavif-rav1e-v0.1.1-shipped
- `feat_target_band` — meta-v0.5-5codec
- `feat_text_likelihood` — zenavif-rav1e-v0.1.1-shipped

