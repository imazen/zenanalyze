# Feature consumption inventory

Which zenanalyze features each downstream bake consumes, aggregated from the bake artifacts by `tools/feature_inventory.py` (zenanalyze#41). Regenerate with `just feature-inventory`; do not edit by hand.

- generated: 2026-08-28
- command: `tools/feature_inventory.py --universe target/inventory/universe.txt --cost benchmarks/per_feature_cost_grid_2026-08-28.tsv --cost-class photo --cost-side 2048 --inspect-bin target/release/zenpredict-inspect --label zenjpeg-a-v3-shipped=../zenjpeg/zenjpeg/src/encode/picker_data/feature_order.txt --label zenavif-rav1e-v0.1.1-shipped=../zenavif/src/models/rav1e_picker_v0_1_1.bin --label zenjpeg-v0.5-modesfull=../zenjpeg/benchmarks/zenjpeg_picker_v0.5_modesfull-tiled-evenodd_2026-06-28.manifest.json --label zenjpeg-lossy-ssim2-v0.1=../zenjpeg/benchmarks/zenjpeg_lossy_ssim2_picker_v0.1_K3_cleansplit_2026-06-29.manifest.json --label zenavif-lossy-zensim-v0.1=../zenavif/benchmarks/pickers/zenavif_lossy_mlp_zensim_v0.1_2026-06-28.manifest.json --label zenjxl-lossy-ssim2-v0.1=../zenjxl/benchmarks/zenjxl_lossy_ssim2_picker_v0.1_cleansplit_2026-06-29.manifest.json --label zenjpeg-v2.1-full=zentrain/testdata/zenjpeg_picker_v2.1_full.manifest.json --label zenjxl-v0.7b=benchmarks/zenjxl_picker_v0.7b_2026-05-06.manifest.json --label meta-v0.5-5codec=benchmarks/zenpicker_meta_v0.5_5codec_2026-05-06.manifest.json --out docs/feature-consumption.md`

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

Sorted by number of consuming bakes, then name. `solo` / `LOO` = µs at the reference cell (photo 2048², baseline `SUPPORTED` = 8492 µs) from the cost grid; see *Cost vs use*.

| feature | n | solo µs | LOO µs | zenjpeg-a-v3-shipped | zenavif-rav1e-v0.1.1-shipped | zenjpeg-v0.5-modesfull | zenjpeg-lossy-ssim2-v0.1 | zenavif-lossy-zensim-v0.1 | zenjxl-lossy-ssim2-v0.1 | zenjpeg-v2.1-full | zenjxl-v0.7b | meta-v0.5-5codec |
|---|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `feat_chroma_complexity` | 9 | 1992 | 11 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_colourfulness` | 9 | 2785 | -121 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_dct_compressibility_uv` | 9 | 4263 | -15 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_dct_compressibility_y` | 9 | 4274 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_edge_density` | 9 | 1999 | -10 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_flat_color_block_ratio` | 9 | 2012 | -150 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_gradient_fraction` | 9 | 4244 | 21 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_high_freq_energy_ratio` | 9 | 4260 | 15 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_laplacian_variance` | 9 | 3306 | -24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_luma_histogram_entropy` | 9 | 3143 | -7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_uniformity` | 9 | 2017 | -128 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_variance` | 9 | 2750 | -21 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `feat_aq_map_mean` | 8 | 4276 | -21 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_aq_map_std` | 8 | 4255 | 40 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_aspect_min_over_max` | 8 | 1994 | -21 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| `feat_cb_horiz_sharpness` | 8 | 2504 | 13 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cb_peak_sharpness` | 8 | 2504 | -26 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cb_sharpness` | 8 | 1995 | -22 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cb_vert_sharpness` | 8 | 2496 | 6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_horiz_sharpness` | 8 | 2498 | -21 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_peak_sharpness` | 8 | 2492 | -13 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_sharpness` | 8 | 2000 | -27 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_cr_vert_sharpness` | 8 | 2503 | -28 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_edge_slope_stdev` | 8 | 2767 | -19 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_grayscale_score` | 8 | 3695 | 456 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_log_pixels` | 8 | 1994 | -24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ |
| `feat_noise_floor_uv` | 8 | 4265 | -16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_noise_floor_y` | 8 | 4265 | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_patch_fraction_fast` | 8 | 4247 | 24 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_quant_survival_uv` | 8 | 4244 | 13 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_quant_survival_y` | 8 | 4260 | -14 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_variance_spread` | 8 | 1998 | -23 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_alpha_bimodal_score` | 7 | 1999 | -17 | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_alpha_used_fraction` | 7 | 1994 | -5 | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| `feat_channel_count` | 6 | 2002 | -11 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_distinct_color_bins` | 6 | 3257 | 12 | ✓ | ✓ |  | ✓ |  | ✓ | ✓ | ✓ |  |
| `feat_gradient_fraction_smooth` | 6 | 2002 | -6 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p50` | 6 | 2541 | -14 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p75` | 6 | 2521 | -44 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p90` | 6 | 2532 | -57 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p99` | 6 | 2542 | -16 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_laplacian_variance_peak` | 6 | 2524 | -15 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_luma_kurtosis` | 6 | 2008 | -28 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_pixel_count` | 6 | 1999 | 1 | ✓ |  | ✓ | ✓ | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p75` | 5 | 4268 | -12 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p90` | 5 | 4270 | -26 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p95` | 5 | 4246 | 7 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_aq_map_p99` | 5 | 4264 | -13 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_log_aspect_abs` | 5 | 2005 | -19 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p50` | 5 | 4259 | 5 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p90` | 5 | 4266 | 2 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_patch_fraction` | 5 | 4302 | -30 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p10` | 5 | 4253 | -17 | ✓ |  | ✓ |  | ✓ | ✓ |  | ✓ |  |
| `feat_skin_tone_fraction` | 5 | 2174 | 169 | ✓ | ✓ |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_alpha_present` | 4 | 2007 | -20 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_bitmap_bytes` | 4 | 2000 | -18 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_block_misalignment_32` | 4 | 1991 | -40 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_block_misalignment_8` | 4 | 1997 | -14 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_is_grayscale` | 4 | 1993 | -27 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p1` | 4 | 1998 | -31 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p10` | 4 | 2000 | -34 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_laplacian_variance_p5` | 4 | 1996 | -21 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_log_padded_pixels_16` | 4 | 2000 | -66 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_log_padded_pixels_32` | 4 | 1992 | -16 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_log_padded_pixels_8` | 4 | 1992 | -11 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_max_dim` | 4 | 2005 | -30 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_min_dim` | 4 | 2004 | -13 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_palette_density` | 4 | 3260 | 33 | ✓ | ✓ |  |  |  |  | ✓ | ✓ |  |
| `feat_palette_fits_in_256` | 4 | 2006 | -44 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_palette_log2_size` | 4 | 3263 | -39 | ✓ |  |  | ✓ |  | ✓ |  | ✓ |  |
| `feat_aq_map_p1` | 3 | 1998 | 1 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_aq_map_p10` | 3 | 2000 | 3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_aq_map_p5` | 3 | 1997 | -58 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_aq_map_p50` | 3 | 4247 | 30 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_chroma_luma_covariance_cb` | 3 | 2765 | -33 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_chroma_luma_covariance_cr` | 3 | 2754 | -19 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_info_weight_mean` | 3 | 4261 | -19 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_noise_floor_uv_p25` | 3 | 4271 | -17 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_uv_p50` | 3 | 4272 | -17 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_uv_p75` | 3 | 4265 | -3 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_uv_p90` | 3 | 4274 | 1 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p1` | 3 | 1994 | -20 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p10` | 3 | 2000 | -72 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p25` | 3 | 4258 | -20 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p5` | 3 | 1990 | -43 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_noise_floor_y_p75` | 3 | 4278 | 5 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_orientation_energy_ratio` | 3 | 2764 | -55 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_quant_survival_uv_p10` | 3 | 4258 | -1 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_uv_p25` | 3 | 4272 | -31 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_uv_p50` | 3 | 4249 | -8 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_uv_p75` | 3 | 4280 | 0 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p1` | 3 | 2005 | -25 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p25` | 3 | 4251 | -20 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p5` | 3 | 1989 | 15 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p50` | 3 | 4262 | -24 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_quant_survival_y_p75` | 3 | 4272 | 0 | ✓ |  |  |  |  | ✓ |  | ✓ |  |
| `feat_spectral_slope_y` | 3 | 4267 | -2 | ✓ |  |  | ✓ |  | ✓ |  |  |  |
| `feat_cclass_document` | 2 |  |  |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_lineart` | 2 |  |  |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_photo` | 2 |  |  |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_screen` | 2 |  |  |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_cclass_synthetic` | 2 |  |  |  |  |  |  |  |  |  | ✓ | ✓ |
| `feat_info_weight_p90` | 2 | 4253 | 5 | ✓ |  |  |  |  | ✓ |  |  |  |
| `feat_line_art_score` | 2 |  |  |  | ✓ |  |  |  |  | ✓ |  |  |
| `feat_effective_bit_depth` | 1 | 1994 | -14 | ✓ |  |  |  |  |  |  |  |  |
| `feat_gamut_coverage_p3` | 1 | 2005 | -3 | ✓ |  |  |  |  |  |  |  |  |
| `feat_gamut_coverage_srgb` | 1 | 1981 | -94 | ✓ |  |  |  |  |  |  |  |  |
| `feat_hdr_headroom_stops` | 1 | 1991 | -34 | ✓ |  |  |  |  |  |  |  |  |
| `feat_hdr_pixel_fraction` | 1 | 1994 | -2 | ✓ |  |  |  |  |  |  |  |  |
| `feat_hdr_present` | 1 | 2002 | -43 | ✓ |  |  |  |  |  |  |  |  |
| `feat_log_dist` | 1 |  |  |  |  |  |  |  |  |  | ✓ |  |
| `feat_log_max_dim` | 1 |  |  |  | ✓ |  |  |  |  |  |  |  |
| `feat_log_min_dim` | 1 |  |  |  | ✓ |  |  |  |  |  |  |  |
| `feat_natural_likelihood` | 1 |  |  |  | ✓ |  |  |  |  |  |  |  |
| `feat_p99_luminance_nits` | 1 | 1995 | -20 | ✓ |  |  |  |  |  |  |  |  |
| `feat_peak_luminance_nits` | 1 | 1999 | -10 | ✓ |  |  |  |  |  |  |  |  |
| `feat_screen_content_likelihood` | 1 |  |  |  | ✓ |  |  |  |  |  |  |  |
| `feat_target_band` | 1 |  |  |  |  |  |  |  |  |  |  | ✓ |
| `feat_text_likelihood` | 1 |  |  |  | ✓ |  |  |  |  |  |  |  |
| `feat_wide_gamut_fraction` | 1 | 1994 | 19 | ✓ |  |  |  |  |  |  |  |  |
| `feat_wide_gamut_peak` | 1 | 1995 | -24 | ✓ |  |  |  |  |  |  |  |  |

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

## Cost vs use

Per-feature wall-clock from `examples/per_feature_cost_grid.rs` (real photo / screen crops, sweep-discipline sizes; medians over crops) joined to the consumption counts above — the #41 cross-reference and the #50 Sub-A cost map. **LOO** (leave-one-out) is what the feature adds when everything else is already computed; **solo** is what it costs alone, dependencies included. LOO ≤ 0 means the feature shares a pass (noise-level).

    # zenanalyze per-feature cost grid — analyze_features RGB8_SRGB, real content
    # git=823f605 host=mac arch=aarch64 date=2026-08-28 corpus=../codec-corpus crops/cell=2 features=117 (FeatureSet::SUPPORTED of this build)
    # sides=[64, 256, 1024, 2048, 4096] classes=["photo", "screen"]; photo=CID22 512² (64/256) + clic2025 1024² centre crops (1024+, 2×2/4×4 mosaics); screen=gb82 512² centre crops (mosaicked 2×2..8×8 for 1024+)
    # baseline_ns = median of SUPPORTED; solo_ns = median with only this feature requested; loo_ns = baseline − median(SUPPORTED \ feature) (≤ 0 → shares a pass / noise)
    # median over ≥ 5 runs and ≥ 100 ms per cell; tiles = number of distinct source crops mosaicked

Reference cell: **photo 2048²**, baseline `SUPPORTED` = **8492 µs**. Baseline by side (photo): 64² = 617 µs, 256² = 1598 µs, 1024² = 5857 µs, 2048² = 8492 µs, 4096² = 18596 µs. Fit baseline = 2703 µs + 0.9799 ns/px.

> **Read the per-side baselines, not the fit.** The fitted α is dominated by the two largest sides and is NOT a per-call floor — measured 2026-08-28, the 64² call is ~0.63 ms against an α of ~2.7 ms. The cost is not affine in pixels because the sampling budgets cap several passes, so marginal cost per pixel falls ~26× across the sweep. Full analysis: `benchmarks/perf_2026-08-28.md`.

> **Any grid whose header says `screen=gb82` measured PHOTOS.** `gb82` is the photographic set; the screen-content set is `gb82-sc`. Fixed in `examples/common/mod.rs` on 2026-08-28 — grids produced after that read `screen=gb82-sc` and also carry `photohard` (gb82, correctly named) and `mixed`. The `screen` columns below therefore describe photographic content and will change when this file is next regenerated.

> **The baselines above predate the 2026-08-28 optimizations** (`103c5b1b`, `cca3c625`): whole-pass cost is now ~1.04× lower at 1 MP, ~1.11× at 4 MP and up to ~1.35× at 16 MP, so the absolute µs here read high. Relative ranking is unaffected.

### Supported but consumed by no listed bake — ranked by LOO (9)

Optimization fodder per #41: every µs here is paid by a `SUPPORTED` request and read by nobody. Candidates for opt-in / `experimental` gating or a cheaper implementation.

| feature | n | solo µs | LOO µs | LOO µs photo | LOO µs screen | LOO fit α µs | LOO fit β ns/px | consumers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `feat_chroma_subsample_dct_loss` | 0 | 2310 | 240 | 240 | 262 | 98.9 | 0.0249 | — |
| `feat_xyb_bquarter_chroma_loss` | 0 | 2220 | 207 | 207 | 210 | 34.9 | 0.0438 | — |
| `feat_xyb444_color_loss` | 0 | 2073 | 71 | 71 | 76 | 17.5 | 0.0202 | — |
| `feat_highlight_luma_std` | 0 | 1999 | 0 | 0 | -25 | 2.7 | -0.0008 | — |
| `feat_highlight_luma_mean` | 0 | 2002 | -15 | -15 | 2 | 3.6 | -0.0064 | — |
| `feat_highlight_chroma_mean` | 0 | 1993 | -29 | -29 | -16 | -8.4 | 0.0035 | — |
| `feat_highlight_orientation_ratio` | 0 | 2006 | -33 | -33 | -30 | 9.7 | -0.0128 | — |
| `feat_highlight_edge_count` | 0 | 2002 | -34 | -34 | -33 | 6.6 | -0.0118 | — |
| `feat_highlight_chroma_std` | 0 | 2008 | -52 | -52 | -4 | -7.3 | -0.0049 | — |

### Consumed by exactly one bake — ranked by LOO (10)

Codec-specific cost: only that bake's request should pay for it (request narrowing, #50 Sub-B).

| feature | n | solo µs | LOO µs | LOO µs photo | LOO µs screen | LOO fit α µs | LOO fit β ns/px | consumers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `feat_wide_gamut_fraction` | 1 | 1994 | 19 | 19 | -24 | 0.4 | -0.0067 | zenjpeg-a-v3-shipped |
| `feat_hdr_pixel_fraction` | 1 | 1994 | -2 | -2 | 11 | -1.2 | -0.0069 | zenjpeg-a-v3-shipped |
| `feat_gamut_coverage_p3` | 1 | 2005 | -3 | -3 | -175 | -2.0 | -0.0103 | zenjpeg-a-v3-shipped |
| `feat_peak_luminance_nits` | 1 | 1999 | -10 | -10 | -9 | -0.2 | -0.0085 | zenjpeg-a-v3-shipped |
| `feat_effective_bit_depth` | 1 | 1994 | -14 | -14 | -3 | -19.3 | -0.0055 | zenjpeg-a-v3-shipped |
| `feat_p99_luminance_nits` | 1 | 1995 | -20 | -20 | -27 | -5.1 | -0.0072 | zenjpeg-a-v3-shipped |
| `feat_wide_gamut_peak` | 1 | 1995 | -24 | -24 | -17 | -5.3 | -0.0077 | zenjpeg-a-v3-shipped |
| `feat_hdr_headroom_stops` | 1 | 1991 | -34 | -34 | -25 | -9.0 | -0.0065 | zenjpeg-a-v3-shipped |
| `feat_hdr_present` | 1 | 2002 | -43 | -43 | -25 | -7.6 | -0.0125 | zenjpeg-a-v3-shipped |
| `feat_gamut_coverage_srgb` | 1 | 1981 | -94 | -94 | -148 | -22.0 | -0.0065 | zenjpeg-a-v3-shipped |

### Shared by ≥ 2 bakes — ranked by LOO (98)

The hot path: these are what every picker asks for, so they are where a SIMD / pass-sharing win pays off across codecs.

| feature | n | solo µs | LOO µs | LOO µs photo | LOO µs screen | LOO fit α µs | LOO fit β ns/px | consumers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `feat_grayscale_score` | 8 | 3695 | 456 | 456 | 415 | 1.7 | 0.0951 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_skin_tone_fraction` | 5 | 2174 | 169 | 169 | 100 | 83.7 | -0.0036 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_aq_map_std` | 8 | 4255 | 40 | 40 | -20 | 4.8 | -0.0078 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_palette_density` | 4 | 3260 | 33 | 33 | 1 | 26.7 | -0.0163 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_aq_map_p50` | 3 | 4247 | 30 | 30 | -83 | 8.2 | -0.0063 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_patch_fraction_fast` | 8 | 4247 | 24 | 24 | -119 | 1.1 | -0.0078 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_gradient_fraction` | 9 | 4244 | 21 | 21 | -68 | -1.6 | -0.0061 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_high_freq_energy_ratio` | 9 | 4260 | 15 | 15 | 23 | 19.9 | -0.0120 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_quant_survival_y_p5` | 3 | 1989 | 15 | 15 | -39 | 6.0 | -0.0021 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_uv` | 8 | 4244 | 13 | 13 | -65 | -0.1 | -0.0064 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_cb_horiz_sharpness` | 8 | 2504 | 13 | 13 | 2 | 16.6 | -0.0109 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_distinct_color_bins` | 6 | 3257 | 12 | 12 | -12 | 21.9 | -0.0124 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_chroma_complexity` | 9 | 1992 | 11 | 11 | -25 | -7.0 | 0.0074 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_aq_map_p95` | 5 | 4246 | 7 | 7 | -120 | -8.9 | 0.0011 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cb_vert_sharpness` | 8 | 2496 | 6 | 6 | 13 | 9.4 | -0.0086 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_noise_floor_y_p75` | 3 | 4278 | 5 | 5 | -69 | -5.2 | -0.0022 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_dct_compressibility_y` | 9 | 4274 | 5 | 5 | 1 | 6.0 | -0.0098 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_noise_floor_y_p50` | 5 | 4259 | 5 | 5 | -62 | 0.1 | -0.0067 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_info_weight_p90` | 2 | 4253 | 5 | 5 | 2 | 1.6 | 0.0036 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1 |
| `feat_aq_map_p10` | 3 | 2000 | 3 | 3 | -202 | -2.3 | 0.0019 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_y_p90` | 5 | 4266 | 2 | 2 | -43 | -2.7 | -0.0022 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_y` | 8 | 4265 | 2 | 2 | -68 | 0.6 | -0.0112 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_noise_floor_uv_p90` | 3 | 4274 | 1 | 1 | -116 | 7.5 | -0.0047 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_pixel_count` | 6 | 1999 | 1 | 1 | -117 | 1.1 | -0.0092 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_aq_map_p1` | 3 | 1998 | 1 | 1 | -111 | 0.7 | -0.0026 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_y_p75` | 3 | 4272 | 0 | 0 | -68 | 11.7 | -0.0051 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_uv_p75` | 3 | 4280 | 0 | 0 | -57 | 5.2 | -0.0025 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_uv_p10` | 3 | 4258 | -1 | -1 | -47 | 1.6 | 0.0012 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_spectral_slope_y` | 3 | 4267 | -2 | -2 | -12 | 3.5 | -0.0009 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1 |
| `feat_noise_floor_uv_p75` | 3 | 4265 | -3 | -3 | -219 | 2.0 | -0.0020 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_alpha_used_fraction` | 7 | 1994 | -5 | -5 | -10 | 4.4 | -0.0096 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_gradient_fraction_smooth` | 6 | 2002 | -6 | -6 | -18 | -1.3 | -0.0028 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_luma_histogram_entropy` | 9 | 3143 | -7 | -7 | 7 | 18.6 | -0.0124 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_quant_survival_uv_p50` | 3 | 4249 | -8 | -8 | -50 | 3.5 | -0.0011 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_edge_density` | 9 | 1999 | -10 | -10 | -7 | 7.0 | -0.0009 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_channel_count` | 6 | 2002 | -11 | -11 | -101 | 6.0 | -0.0104 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_log_padded_pixels_8` | 4 | 1992 | -11 | -11 | -85 | -6.1 | 0.0038 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_aq_map_p75` | 5 | 4268 | -12 | -12 | -60 | 4.1 | -0.0078 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_aq_map_p99` | 5 | 4264 | -13 | -13 | -99 | -12.6 | 0.0035 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cr_peak_sharpness` | 8 | 2492 | -13 | -13 | -11 | 12.7 | -0.0086 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_min_dim` | 4 | 2004 | -13 | -13 | -102 | -4.5 | -0.0064 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_block_misalignment_8` | 4 | 1997 | -14 | -14 | -95 | -1.0 | -0.0091 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_laplacian_variance_p50` | 6 | 2541 | -14 | -14 | -173 | 8.7 | -0.0029 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_y` | 8 | 4260 | -14 | -14 | -81 | -3.1 | -0.0079 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_laplacian_variance_peak` | 6 | 2524 | -15 | -15 | -98 | 5.1 | -0.0028 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_dct_compressibility_uv` | 9 | 4263 | -15 | -15 | 6 | 5.7 | -0.0063 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_noise_floor_uv` | 8 | 4265 | -16 | -16 | -57 | -2.8 | -0.0101 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_laplacian_variance_p99` | 6 | 2542 | -16 | -16 | -174 | 1.7 | 0.0006 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_log_padded_pixels_32` | 4 | 1992 | -16 | -16 | -70 | -3.6 | -0.0009 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_alpha_bimodal_score` | 7 | 1999 | -17 | -17 | -22 | -4.5 | -0.0091 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_noise_floor_uv_p25` | 3 | 4271 | -17 | -17 | -46 | -13.4 | -0.0008 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_y_p10` | 5 | 4253 | -17 | -17 | -65 | -5.5 | -0.0017 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_uv_p50` | 3 | 4272 | -17 | -17 | -57 | -14.5 | -0.0014 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_bitmap_bytes` | 4 | 2000 | -18 | -18 | -104 | -1.2 | -0.0106 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_chroma_luma_covariance_cr` | 3 | 2754 | -19 | -19 | -38 | 2.3 | -0.0064 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1 |
| `feat_edge_slope_stdev` | 8 | 2767 | -19 | -19 | -126 | -9.3 | -0.0066 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_log_aspect_abs` | 5 | 2005 | -19 | -19 | -104 | -3.4 | -0.0066 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_info_weight_mean` | 3 | 4261 | -19 | -19 | -7 | -8.0 | 0.0021 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1 |
| `feat_quant_survival_y_p25` | 3 | 4251 | -20 | -20 | -56 | -9.2 | 0.0039 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_y_p25` | 3 | 4258 | -20 | -20 | -47 | -12.7 | 0.0019 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_y_p1` | 3 | 1994 | -20 | -20 | -37 | -9.4 | 0.0003 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_alpha_present` | 4 | 2007 | -20 | -20 | -38 | -8.2 | -0.0078 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cr_horiz_sharpness` | 8 | 2498 | -21 | -21 | -3 | 9.7 | -0.0098 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_variance` | 9 | 2750 | -21 | -21 | -21 | 5.7 | 0.0014 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_aspect_min_over_max` | 8 | 1994 | -21 | -21 | -88 | 7.6 | -0.0113 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_aq_map_mean` | 8 | 4276 | -21 | -21 | 9 | -14.0 | -0.0070 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_laplacian_variance_p5` | 4 | 1996 | -21 | -21 | -74 | -1.4 | 0.0008 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cb_sharpness` | 8 | 1995 | -22 | -22 | 6 | -0.0 | 0.0042 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_variance_spread` | 8 | 1998 | -23 | -23 | -18 | 6.6 | -0.0056 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_laplacian_variance` | 9 | 3306 | -24 | -24 | 5 | 3.0 | -0.0025 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_log_pixels` | 8 | 1994 | -24 | -24 | -149 | 6.9 | -0.0129 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_quant_survival_y_p50` | 3 | 4262 | -24 | -24 | -55 | -5.5 | 0.0003 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_y_p1` | 3 | 2005 | -25 | -25 | -12 | -6.3 | 0.0004 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cb_peak_sharpness` | 8 | 2504 | -26 | -26 | -13 | 3.5 | -0.0099 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_aq_map_p90` | 5 | 4270 | -26 | -26 | -62 | 17.1 | -0.0233 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cr_sharpness` | 8 | 2000 | -27 | -27 | -20 | -2.8 | -0.0000 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_is_grayscale` | 4 | 1993 | -27 | -27 | -117 | -9.5 | -0.0064 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_cr_vert_sharpness` | 8 | 2503 | -28 | -28 | 4 | 7.7 | -0.0084 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b |
| `feat_luma_kurtosis` | 6 | 2008 | -28 | -28 | -30 | -14.6 | -0.0010 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_patch_fraction` | 5 | 4302 | -30 | -30 | 25 | 13.9 | -0.0128 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_max_dim` | 4 | 2005 | -30 | -30 | -89 | -7.9 | -0.0064 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_laplacian_variance_p1` | 4 | 1998 | -31 | -31 | -105 | -6.7 | -0.0029 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_quant_survival_uv_p25` | 3 | 4272 | -31 | -31 | -55 | -8.2 | 0.0008 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_chroma_luma_covariance_cb` | 3 | 2765 | -33 | -33 | -15 | -7.6 | 0.0012 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1 |
| `feat_laplacian_variance_p10` | 4 | 2000 | -34 | -34 | -117 | -8.9 | -0.0002 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_palette_log2_size` | 4 | 3263 | -39 | -39 | -12 | -12.0 | 0.0030 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_block_misalignment_32` | 4 | 1991 | -40 | -40 | -102 | 3.3 | -0.0149 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_y_p5` | 3 | 1990 | -43 | -43 | -27 | -11.2 | -0.0006 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_laplacian_variance_p75` | 6 | 2521 | -44 | -44 | -154 | -5.5 | -0.0039 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_palette_fits_in_256` | 4 | 2006 | -44 | -44 | -7 | -12.1 | -0.0091 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_orientation_energy_ratio` | 3 | 2764 | -55 | -55 | -35 | -14.2 | 0.0000 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1 |
| `feat_laplacian_variance_p90` | 6 | 2532 | -57 | -57 | -130 | -12.9 | 0.0014 | zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_aq_map_p5` | 3 | 1997 | -58 | -58 | -154 | -11.2 | -0.0007 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_log_padded_pixels_16` | 4 | 2000 | -66 | -66 | -70 | -15.1 | 0.0016 | zenjpeg-a-v3-shipped, zenjpeg-lossy-ssim2-v0.1, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_noise_floor_y_p10` | 3 | 2000 | -72 | -72 | -22 | -10.4 | -0.0009 | zenjpeg-a-v3-shipped, zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b |
| `feat_colourfulness` | 9 | 2785 | -121 | -121 | -24 | -26.3 | 0.0033 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_uniformity` | 9 | 2017 | -128 | -128 | -22 | -18.4 | -0.0008 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |
| `feat_flat_color_block_ratio` | 9 | 2012 | -150 | -150 | -25 | -31.8 | 0.0034 | zenjpeg-a-v3-shipped, zenavif-rav1e-v0.1.1-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenavif-lossy-zensim-v0.1, zenjxl-lossy-ssim2-v0.1, zenjpeg-v2.1-full, zenjxl-v0.7b, meta-v0.5-5codec |

### Consumed but not in the cost grid

Bake columns the grid's build did not produce (caller-supplied `feat_*` inputs or features gated off / retired):

`feat_cclass_document`, `feat_cclass_lineart`, `feat_cclass_photo`, `feat_cclass_screen`, `feat_cclass_synthetic`, `feat_line_art_score`, `feat_log_dist`, `feat_log_max_dim`, `feat_log_min_dim`, `feat_natural_likelihood`, `feat_screen_content_likelihood`, `feat_target_band`, `feat_text_likelihood`

## Family presets (proposed `FeatureSet` constants)

Per codec family, the union of every listed bake's features (analyzer-produced only, in `SUPPORTED` order) — the `JPEG_FAMILY` / `WEBP_FAMILY` / … presets #41 asks for, derived from the artifacts instead of hand-written. **Proposals only:** adding them to `zenanalyze::feature::FeatureSet` (next to `ZENJPEG_PICKER_V1_1`) is a public-API addition that needs sign-off, and a preset pins a *union* — a codec that requests it pays for every bake's features, so narrow to the shipped bake's `feat_cols` when only one bake is live.

### jpeg — 108 features across 4 bakes

Bakes: zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenjpeg-v2.1-full

Dropped (not produced by this build): `feat_line_art_score`

```rust
/// Union of the features every listed jpeg bake consumes (108 features across 4 bakes: zenjpeg-a-v3-shipped, zenjpeg-v0.5-modesfull, zenjpeg-lossy-ssim2-v0.1, zenjpeg-v2.1-full).
/// Generated by `tools/feature_inventory.py` — a PROPOSAL, not shipped API.
pub const JPEG_FAMILY: Self = Self::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::EdgeDensity)
    .with(AnalysisFeature::ChromaComplexity)
    .with(AnalysisFeature::CbSharpness)
    .with(AnalysisFeature::CrSharpness)
    .with(AnalysisFeature::Uniformity)
    .with(AnalysisFeature::FlatColorBlockRatio)
    .with(AnalysisFeature::Colourfulness)
    .with(AnalysisFeature::LaplacianVariance)
    .with(AnalysisFeature::VarianceSpread)
    .with(AnalysisFeature::DistinctColorBins)
    .with(AnalysisFeature::PaletteDensity)
    .with(AnalysisFeature::CbHorizSharpness)
    .with(AnalysisFeature::CbVertSharpness)
    .with(AnalysisFeature::CbPeakSharpness)
    .with(AnalysisFeature::CrHorizSharpness)
    .with(AnalysisFeature::CrVertSharpness)
    .with(AnalysisFeature::CrPeakSharpness)
    .with(AnalysisFeature::HighFreqEnergyRatio)
    .with(AnalysisFeature::LumaHistogramEntropy)
    .with(AnalysisFeature::DctCompressibilityY)
    .with(AnalysisFeature::DctCompressibilityUV)
    .with(AnalysisFeature::PatchFraction)
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction)
    .with(AnalysisFeature::AlphaBimodalScore)
    .with(AnalysisFeature::PaletteFitsIn256)
    .with(AnalysisFeature::PeakLuminanceNits)
    .with(AnalysisFeature::P99LuminanceNits)
    .with(AnalysisFeature::HdrHeadroomStops)
    .with(AnalysisFeature::HdrPixelFraction)
    .with(AnalysisFeature::WideGamutPeak)
    .with(AnalysisFeature::WideGamutFraction)
    .with(AnalysisFeature::EffectiveBitDepth)
    .with(AnalysisFeature::HdrPresent)
    .with(AnalysisFeature::GrayscaleScore)
    .with(AnalysisFeature::AqMapMean)
    .with(AnalysisFeature::AqMapStd)
    .with(AnalysisFeature::NoiseFloorY)
    .with(AnalysisFeature::NoiseFloorUV)
    .with(AnalysisFeature::GamutCoverageSrgb)
    .with(AnalysisFeature::GamutCoverageP3)
    .with(AnalysisFeature::GradientFraction)
    .with(AnalysisFeature::SkinToneFraction)
    .with(AnalysisFeature::EdgeSlopeStdev)
    .with(AnalysisFeature::PatchFractionFast)
    .with(AnalysisFeature::QuantSurvivalY)
    .with(AnalysisFeature::QuantSurvivalUv)
    .with(AnalysisFeature::IsGrayscale)
    .with(AnalysisFeature::PixelCount)
    .with(AnalysisFeature::LogPixels)
    .with(AnalysisFeature::MinDim)
    .with(AnalysisFeature::MaxDim)
    .with(AnalysisFeature::BitmapBytes)
    .with(AnalysisFeature::AspectMinOverMax)
    .with(AnalysisFeature::LogAspectAbs)
    .with(AnalysisFeature::BlockMisalignment8)
    .with(AnalysisFeature::BlockMisalignment32)
    .with(AnalysisFeature::ChannelCount)
    .with(AnalysisFeature::AqMapP50)
    .with(AnalysisFeature::AqMapP75)
    .with(AnalysisFeature::AqMapP90)
    .with(AnalysisFeature::AqMapP95)
    .with(AnalysisFeature::AqMapP99)
    .with(AnalysisFeature::NoiseFloorYP25)
    .with(AnalysisFeature::NoiseFloorYP50)
    .with(AnalysisFeature::NoiseFloorYP75)
    .with(AnalysisFeature::NoiseFloorYP90)
    .with(AnalysisFeature::NoiseFloorUvP25)
    .with(AnalysisFeature::NoiseFloorUvP50)
    .with(AnalysisFeature::NoiseFloorUvP75)
    .with(AnalysisFeature::NoiseFloorUvP90)
    .with(AnalysisFeature::LaplacianVarianceP50)
    .with(AnalysisFeature::LaplacianVarianceP75)
    .with(AnalysisFeature::LaplacianVarianceP90)
    .with(AnalysisFeature::LaplacianVarianceP99)
    .with(AnalysisFeature::LaplacianVariancePeak)
    .with(AnalysisFeature::QuantSurvivalYP10)
    .with(AnalysisFeature::QuantSurvivalYP25)
    .with(AnalysisFeature::QuantSurvivalYP50)
    .with(AnalysisFeature::QuantSurvivalYP75)
    .with(AnalysisFeature::QuantSurvivalUvP10)
    .with(AnalysisFeature::QuantSurvivalUvP25)
    .with(AnalysisFeature::QuantSurvivalUvP50)
    .with(AnalysisFeature::QuantSurvivalUvP75)
    .with(AnalysisFeature::LogPaddedPixels8)
    .with(AnalysisFeature::LogPaddedPixels16)
    .with(AnalysisFeature::LogPaddedPixels32)
    .with(AnalysisFeature::LaplacianVarianceP1)
    .with(AnalysisFeature::LaplacianVarianceP5)
    .with(AnalysisFeature::LaplacianVarianceP10)
    .with(AnalysisFeature::AqMapP1)
    .with(AnalysisFeature::AqMapP5)
    .with(AnalysisFeature::AqMapP10)
    .with(AnalysisFeature::NoiseFloorYP1)
    .with(AnalysisFeature::NoiseFloorYP5)
    .with(AnalysisFeature::NoiseFloorYP10)
    .with(AnalysisFeature::QuantSurvivalYP1)
    .with(AnalysisFeature::QuantSurvivalYP5)
    .with(AnalysisFeature::LumaKurtosis)
    .with(AnalysisFeature::GradientFractionSmooth)
    .with(AnalysisFeature::PaletteLog2Size)
    .with(AnalysisFeature::ChromaLumaCovarianceCb)
    .with(AnalysisFeature::ChromaLumaCovarianceCr)
    .with(AnalysisFeature::InfoWeightMean)
    .with(AnalysisFeature::InfoWeightP90)
    .with(AnalysisFeature::OrientationEnergyRatio)
    .with(AnalysisFeature::SpectralSlopeY);
```

### avif — 55 features across 2 bakes

Bakes: zenavif-rav1e-v0.1.1-shipped, zenavif-lossy-zensim-v0.1

Dropped (not produced by this build): `feat_line_art_score`, `feat_log_max_dim`, `feat_log_min_dim`, `feat_natural_likelihood`, `feat_screen_content_likelihood`, `feat_text_likelihood`

```rust
/// Union of the features every listed avif bake consumes (55 features across 2 bakes: zenavif-rav1e-v0.1.1-shipped, zenavif-lossy-zensim-v0.1).
/// Generated by `tools/feature_inventory.py` — a PROPOSAL, not shipped API.
pub const AVIF_FAMILY: Self = Self::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::EdgeDensity)
    .with(AnalysisFeature::ChromaComplexity)
    .with(AnalysisFeature::CbSharpness)
    .with(AnalysisFeature::CrSharpness)
    .with(AnalysisFeature::Uniformity)
    .with(AnalysisFeature::FlatColorBlockRatio)
    .with(AnalysisFeature::Colourfulness)
    .with(AnalysisFeature::LaplacianVariance)
    .with(AnalysisFeature::VarianceSpread)
    .with(AnalysisFeature::DistinctColorBins)
    .with(AnalysisFeature::PaletteDensity)
    .with(AnalysisFeature::CbHorizSharpness)
    .with(AnalysisFeature::CbVertSharpness)
    .with(AnalysisFeature::CbPeakSharpness)
    .with(AnalysisFeature::CrHorizSharpness)
    .with(AnalysisFeature::CrVertSharpness)
    .with(AnalysisFeature::CrPeakSharpness)
    .with(AnalysisFeature::HighFreqEnergyRatio)
    .with(AnalysisFeature::LumaHistogramEntropy)
    .with(AnalysisFeature::DctCompressibilityY)
    .with(AnalysisFeature::DctCompressibilityUV)
    .with(AnalysisFeature::PatchFraction)
    .with(AnalysisFeature::AlphaUsedFraction)
    .with(AnalysisFeature::AlphaBimodalScore)
    .with(AnalysisFeature::GrayscaleScore)
    .with(AnalysisFeature::AqMapMean)
    .with(AnalysisFeature::AqMapStd)
    .with(AnalysisFeature::NoiseFloorY)
    .with(AnalysisFeature::NoiseFloorUV)
    .with(AnalysisFeature::GradientFraction)
    .with(AnalysisFeature::SkinToneFraction)
    .with(AnalysisFeature::EdgeSlopeStdev)
    .with(AnalysisFeature::PatchFractionFast)
    .with(AnalysisFeature::QuantSurvivalY)
    .with(AnalysisFeature::QuantSurvivalUv)
    .with(AnalysisFeature::PixelCount)
    .with(AnalysisFeature::LogPixels)
    .with(AnalysisFeature::AspectMinOverMax)
    .with(AnalysisFeature::LogAspectAbs)
    .with(AnalysisFeature::ChannelCount)
    .with(AnalysisFeature::AqMapP75)
    .with(AnalysisFeature::AqMapP90)
    .with(AnalysisFeature::AqMapP95)
    .with(AnalysisFeature::AqMapP99)
    .with(AnalysisFeature::NoiseFloorYP50)
    .with(AnalysisFeature::NoiseFloorYP90)
    .with(AnalysisFeature::LaplacianVarianceP50)
    .with(AnalysisFeature::LaplacianVarianceP75)
    .with(AnalysisFeature::LaplacianVarianceP90)
    .with(AnalysisFeature::LaplacianVarianceP99)
    .with(AnalysisFeature::LaplacianVariancePeak)
    .with(AnalysisFeature::QuantSurvivalYP10)
    .with(AnalysisFeature::LumaKurtosis)
    .with(AnalysisFeature::GradientFractionSmooth);
```

### jxl — 98 features across 2 bakes

Bakes: zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b

Dropped (not produced by this build): `feat_cclass_document`, `feat_cclass_lineart`, `feat_cclass_photo`, `feat_cclass_screen`, `feat_cclass_synthetic`, `feat_log_dist`

```rust
/// Union of the features every listed jxl bake consumes (98 features across 2 bakes: zenjxl-lossy-ssim2-v0.1, zenjxl-v0.7b).
/// Generated by `tools/feature_inventory.py` — a PROPOSAL, not shipped API.
pub const JXL_FAMILY: Self = Self::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::EdgeDensity)
    .with(AnalysisFeature::ChromaComplexity)
    .with(AnalysisFeature::CbSharpness)
    .with(AnalysisFeature::CrSharpness)
    .with(AnalysisFeature::Uniformity)
    .with(AnalysisFeature::FlatColorBlockRatio)
    .with(AnalysisFeature::Colourfulness)
    .with(AnalysisFeature::LaplacianVariance)
    .with(AnalysisFeature::VarianceSpread)
    .with(AnalysisFeature::DistinctColorBins)
    .with(AnalysisFeature::PaletteDensity)
    .with(AnalysisFeature::CbHorizSharpness)
    .with(AnalysisFeature::CbVertSharpness)
    .with(AnalysisFeature::CbPeakSharpness)
    .with(AnalysisFeature::CrHorizSharpness)
    .with(AnalysisFeature::CrVertSharpness)
    .with(AnalysisFeature::CrPeakSharpness)
    .with(AnalysisFeature::HighFreqEnergyRatio)
    .with(AnalysisFeature::LumaHistogramEntropy)
    .with(AnalysisFeature::DctCompressibilityY)
    .with(AnalysisFeature::DctCompressibilityUV)
    .with(AnalysisFeature::PatchFraction)
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction)
    .with(AnalysisFeature::AlphaBimodalScore)
    .with(AnalysisFeature::PaletteFitsIn256)
    .with(AnalysisFeature::GrayscaleScore)
    .with(AnalysisFeature::AqMapMean)
    .with(AnalysisFeature::AqMapStd)
    .with(AnalysisFeature::NoiseFloorY)
    .with(AnalysisFeature::NoiseFloorUV)
    .with(AnalysisFeature::GradientFraction)
    .with(AnalysisFeature::SkinToneFraction)
    .with(AnalysisFeature::EdgeSlopeStdev)
    .with(AnalysisFeature::PatchFractionFast)
    .with(AnalysisFeature::QuantSurvivalY)
    .with(AnalysisFeature::QuantSurvivalUv)
    .with(AnalysisFeature::IsGrayscale)
    .with(AnalysisFeature::PixelCount)
    .with(AnalysisFeature::LogPixels)
    .with(AnalysisFeature::MinDim)
    .with(AnalysisFeature::MaxDim)
    .with(AnalysisFeature::BitmapBytes)
    .with(AnalysisFeature::AspectMinOverMax)
    .with(AnalysisFeature::LogAspectAbs)
    .with(AnalysisFeature::BlockMisalignment8)
    .with(AnalysisFeature::BlockMisalignment32)
    .with(AnalysisFeature::ChannelCount)
    .with(AnalysisFeature::AqMapP50)
    .with(AnalysisFeature::AqMapP75)
    .with(AnalysisFeature::AqMapP90)
    .with(AnalysisFeature::AqMapP95)
    .with(AnalysisFeature::AqMapP99)
    .with(AnalysisFeature::NoiseFloorYP25)
    .with(AnalysisFeature::NoiseFloorYP50)
    .with(AnalysisFeature::NoiseFloorYP75)
    .with(AnalysisFeature::NoiseFloorYP90)
    .with(AnalysisFeature::NoiseFloorUvP25)
    .with(AnalysisFeature::NoiseFloorUvP50)
    .with(AnalysisFeature::NoiseFloorUvP75)
    .with(AnalysisFeature::NoiseFloorUvP90)
    .with(AnalysisFeature::LaplacianVarianceP50)
    .with(AnalysisFeature::LaplacianVarianceP75)
    .with(AnalysisFeature::LaplacianVarianceP90)
    .with(AnalysisFeature::LaplacianVarianceP99)
    .with(AnalysisFeature::LaplacianVariancePeak)
    .with(AnalysisFeature::QuantSurvivalYP10)
    .with(AnalysisFeature::QuantSurvivalYP25)
    .with(AnalysisFeature::QuantSurvivalYP50)
    .with(AnalysisFeature::QuantSurvivalYP75)
    .with(AnalysisFeature::QuantSurvivalUvP10)
    .with(AnalysisFeature::QuantSurvivalUvP25)
    .with(AnalysisFeature::QuantSurvivalUvP50)
    .with(AnalysisFeature::QuantSurvivalUvP75)
    .with(AnalysisFeature::LogPaddedPixels8)
    .with(AnalysisFeature::LogPaddedPixels16)
    .with(AnalysisFeature::LogPaddedPixels32)
    .with(AnalysisFeature::LaplacianVarianceP1)
    .with(AnalysisFeature::LaplacianVarianceP5)
    .with(AnalysisFeature::LaplacianVarianceP10)
    .with(AnalysisFeature::AqMapP1)
    .with(AnalysisFeature::AqMapP5)
    .with(AnalysisFeature::AqMapP10)
    .with(AnalysisFeature::NoiseFloorYP1)
    .with(AnalysisFeature::NoiseFloorYP5)
    .with(AnalysisFeature::NoiseFloorYP10)
    .with(AnalysisFeature::QuantSurvivalYP1)
    .with(AnalysisFeature::QuantSurvivalYP5)
    .with(AnalysisFeature::LumaKurtosis)
    .with(AnalysisFeature::GradientFractionSmooth)
    .with(AnalysisFeature::PaletteLog2Size)
    .with(AnalysisFeature::ChromaLumaCovarianceCb)
    .with(AnalysisFeature::ChromaLumaCovarianceCr)
    .with(AnalysisFeature::InfoWeightMean)
    .with(AnalysisFeature::InfoWeightP90)
    .with(AnalysisFeature::OrientationEnergyRatio)
    .with(AnalysisFeature::SpectralSlopeY);
```

### meta — 14 features across 1 bake

Bakes: meta-v0.5-5codec

Dropped (not produced by this build): `feat_cclass_document`, `feat_cclass_lineart`, `feat_cclass_photo`, `feat_cclass_screen`, `feat_cclass_synthetic`, `feat_target_band`

```rust
/// Union of the features every listed meta bake consumes (14 features across 1 bakes: meta-v0.5-5codec).
/// Generated by `tools/feature_inventory.py` — a PROPOSAL, not shipped API.
pub const META_FAMILY: Self = Self::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::EdgeDensity)
    .with(AnalysisFeature::ChromaComplexity)
    .with(AnalysisFeature::Uniformity)
    .with(AnalysisFeature::FlatColorBlockRatio)
    .with(AnalysisFeature::Colourfulness)
    .with(AnalysisFeature::LaplacianVariance)
    .with(AnalysisFeature::HighFreqEnergyRatio)
    .with(AnalysisFeature::LumaHistogramEntropy)
    .with(AnalysisFeature::DctCompressibilityY)
    .with(AnalysisFeature::DctCompressibilityUV)
    .with(AnalysisFeature::GradientFraction)
    .with(AnalysisFeature::LogPixels)
    .with(AnalysisFeature::AspectMinOverMax);
```

