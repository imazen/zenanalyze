# Cross-codec multi-seed LOO consensus, 2026-05-03

5 seeds × paired with/without retrains across all 4 zen codec pickers, sourced from per-codec full-active-feature LOO sweeps:

- `zenwebp`: 34 features, sourced from `benchmarks/loo_zenwebp_multiseed_2026-05-03.tsv`
- `zenjpeg`: 51 features, sourced from `benchmarks/loo_zenjpeg_multiseed_2026-05-03.tsv`
- `zenavif`: 67 features, sourced from `benchmarks/loo_zenavif_multiseed_2026-05-03.tsv`
- `zenjxl`: 67 features, sourced from `benchmarks/loo_zenjxl_multiseed_2026-05-03.tsv`

**Vote scheme** (per codec, per feature, on the multi-seed mean ΔAC):

- `drop`  — mean ΔAC > +1σ above zero (removing increases argmin acc → cull candidate)
- `keep`  — mean ΔAC < −1σ below zero (removing decreases argmin acc → keep)
- `flat`  — |mean ΔAC| ≤ 1σ (within noise floor, no signal)
- `absent` — feature not in this codec's active KEEP_FEATURES

Cross-codec consensus uses the 4 votes per feature.

## Consensus CULL (0 features)

Features where ≥2 codecs vote `drop` AND no codec votes `keep`. Removing these is likely safe across the codec family.

_None._

## Consensus KEEP (0 features)

Features where ≥2 codecs vote `keep` AND no codec votes `drop`. These should remain in active KEEP_FEATURES.

_None._

## Codec-specific signal (30 features)

Features where exactly one codec shows signal (drop or keep) and all others are flat or absent. Treat these as codec-local — act on them only in that codec's KEEP_FEATURES.

| Feature | Verdict | Codec | ΔAC | Other codecs |
|---|---|---|---|---|
| `feat_alpha_bimodal_score` | **drop** | `zenavif` | +2.70±2.15 | zenwebp:absent, zenjpeg:flat, zenjxl:flat |
| `feat_alpha_present` | **drop** | `zenavif` | +2.70±2.15 | zenwebp:absent, zenjpeg:absent, zenjxl:flat |
| `feat_alpha_used_fraction` | **drop** | `zenavif` | +2.70±2.15 | zenwebp:absent, zenjpeg:flat, zenjxl:flat |
| `feat_aq_map_mean` | **keep** | `zenjpeg` | -3.00±2.78 | zenwebp:flat, zenavif:flat, zenjxl:flat |
| `feat_aq_map_p95` | **keep** | `zenjpeg` | -1.48±1.45 | zenwebp:absent, zenavif:flat, zenjxl:flat |
| `feat_aq_map_p99` | **keep** | `zenjpeg` | -1.96±1.37 | zenwebp:absent, zenavif:flat, zenjxl:flat |
| `feat_cb_horiz_sharpness` | **keep** | `zenjxl` | -1.60±1.06 | zenwebp:flat, zenjpeg:flat, zenavif:flat |
| `feat_cb_vert_sharpness` | **keep** | `zenjxl` | -0.78±0.78 | zenwebp:absent, zenjpeg:flat, zenavif:flat |
| `feat_chroma_complexity` | **drop** | `zenavif` | +3.72±2.78 | zenwebp:flat, zenjpeg:flat, zenjxl:flat |
| `feat_dct_compressibility_y` | **keep** | `zenjxl` | -0.82±0.70 | zenwebp:absent, zenjpeg:flat, zenavif:flat |
| `feat_flat_color_block_ratio` | **drop** | `zenjxl` | +1.56±1.29 | zenwebp:absent, zenjpeg:flat, zenavif:flat |
| `feat_high_freq_energy_ratio` | **drop** | `zenavif` | +4.54±3.79 | zenwebp:flat, zenjpeg:flat, zenjxl:flat |
| `feat_laplacian_variance_p50` | **drop** | `zenjxl` | +1.40±1.20 | zenwebp:flat, zenjpeg:flat, zenavif:flat |
| `feat_laplacian_variance_p75` | **keep** | `zenjpeg` | -3.74±3.29 | zenwebp:flat, zenavif:flat, zenjxl:flat |
| `feat_laplacian_variance_p99` | **drop** | `zenavif` | +2.62±2.31 | zenwebp:absent, zenjpeg:flat, zenjxl:flat |
| `feat_laplacian_variance_peak` | **keep** | `zenjpeg` | -0.90±0.86 | zenwebp:absent, zenavif:flat, zenjxl:flat |
| `feat_luma_kurtosis` | **keep** | `zenjpeg` | -1.60±1.24 | zenwebp:absent, zenavif:flat, zenjxl:flat |
| `feat_min_dim` | **drop** | `zenwebp` | +3.10±1.16 | zenjpeg:absent, zenavif:flat, zenjxl:flat |
| `feat_noise_floor_uv_p25` | **drop** | `zenavif` | +5.76±5.10 | zenwebp:absent, zenjpeg:absent, zenjxl:flat |
| `feat_noise_floor_uv_p50` | **drop** | `zenavif` | +2.70±2.56 | zenwebp:flat, zenjpeg:absent, zenjxl:flat |
| `feat_noise_floor_uv_p90` | **drop** | `zenjxl` | +0.96±0.81 | zenwebp:absent, zenjpeg:absent, zenavif:flat |
| `feat_noise_floor_y` | **drop** | `zenavif` | +4.56±3.89 | zenwebp:absent, zenjpeg:flat, zenjxl:flat |
| `feat_noise_floor_y_p25` | **drop** | `zenavif` | +4.92±3.93 | zenwebp:flat, zenjpeg:absent, zenjxl:flat |
| `feat_noise_floor_y_p50` | **drop** | `zenavif` | +5.48±2.77 | zenwebp:flat, zenjpeg:flat, zenjxl:flat |
| `feat_noise_floor_y_p75` | **drop** | `zenavif` | +6.02±3.94 | zenwebp:flat, zenjpeg:absent, zenjxl:flat |
| `feat_palette_density` | **keep** | `zenjpeg` | -3.32±2.40 | zenwebp:absent, zenavif:flat, zenjxl:flat |
| `feat_patch_fraction` | **drop** | `zenjxl` | +1.24±0.99 | zenwebp:flat, zenjpeg:absent, zenavif:flat |
| `feat_quant_survival_uv_p10` | **drop** | `zenavif` | +6.30±2.87 | zenwebp:absent, zenjpeg:absent, zenjxl:flat |
| `feat_variance` | **drop** | `zenjxl` | +1.12±1.03 | zenwebp:absent, zenjpeg:flat, zenavif:flat |
| `feat_variance_spread` | **drop** | `zenjxl` | +1.58±1.52 | zenwebp:absent, zenjpeg:flat, zenavif:flat |

## Surprises vs handoff "all-time best 15" (13 hits)

Features the 2026-05-02 handoff identifies as top-tier but which show flat or drop signal in this multi-seed sweep. Worth a closer look.

| Feature | zenwebp | zenjpeg | zenavif | zenjxl |
|---|---|---|---|---|
| `feat_laplacian_variance_p50` | flat ΔAC=-1.96±7.39 | flat ΔAC=-2.12±2.96 | flat ΔAC=+2.48±2.61 | drop ΔAC=+1.40±1.20 |
| `feat_laplacian_variance` | flat ΔAC=+0.80±7.93 | flat ΔAC=+4.40±7.60 | flat ΔAC=+3.64±7.54 | flat ΔAC=+1.00±1.66 |
| `feat_quant_survival_y` | flat ΔAC=+1.98±2.11 | flat ΔAC=-2.02±2.14 | flat ΔAC=+2.18±4.56 | flat ΔAC=+0.08±1.67 |
| `feat_cb_sharpness` | flat ΔAC=+5.26±6.77 | flat ΔAC=+0.70±6.58 | flat ΔAC=+5.78±6.71 | flat ΔAC=+0.52±1.92 |
| `feat_pixel_count` | flat ΔAC=+0.14±2.02 | flat ΔAC=-1.06±1.36 | flat ΔAC=+1.66±3.36 | flat ΔAC=+0.40±1.79 |
| `feat_uniformity` | flat ΔAC=+2.56±2.96 | flat ΔAC=+1.18±8.69 | flat ΔAC=+0.36±4.09 | flat ΔAC=+1.16±1.19 |
| `feat_distinct_color_bins` | flat ΔAC=+1.36±4.57 | absent | flat ΔAC=+0.66±2.27 | flat ΔAC=+0.14±1.01 |
| `feat_cr_sharpness` | flat ΔAC=-3.80±5.30 | flat ΔAC=+0.98±8.52 | flat ΔAC=+4.70±6.96 | flat ΔAC=+0.20±0.90 |
| `feat_edge_density` | flat ΔAC=+0.36±2.88 | flat ΔAC=+2.62±5.35 | flat ΔAC=+3.38±6.67 | flat ΔAC=-0.02±1.34 |
| `feat_noise_floor_y_p50` | flat ΔAC=-0.84±3.15 | flat ΔAC=-0.52±1.97 | drop ΔAC=+5.48±2.77 | flat ΔAC=+0.60±0.99 |
| `feat_luma_histogram_entropy` | flat ΔAC=+0.54±4.93 | flat ΔAC=-2.54±4.00 | flat ΔAC=-0.78±3.54 | flat ΔAC=-0.12±1.63 |
| `feat_quant_survival_y_p50` | flat ΔAC=+1.94±3.05 | absent | flat ΔAC=+3.94±4.84 | flat ΔAC=+1.26±2.07 |
| `feat_noise_floor_uv_p50` | flat ΔAC=+0.66±2.86 | absent | drop ΔAC=+2.70±2.56 | flat ΔAC=+0.98±2.27 |

## Per-codec action items

Concrete next-step recommendations per codec config. Apply only with user review — analysis is suggestive, not authoritative.

### `zenwebp`

**Cull candidates** (mean ΔAC > +1σ, removing helps argmin):

- `feat_aq_map_std` ΔAC=+5.18±0.81, ΔOH=-0.40±0.09
- `feat_min_dim` ΔAC=+3.10±1.16, ΔOH=-0.04±0.15 (codec-specific)

### `zenjpeg`

No cull candidates beyond noise floor.

**Keep (high-confidence)** (mean ΔAC < −1σ):

- `feat_laplacian_variance_p75` ΔAC=-3.74±3.29
- `feat_palette_density` ΔAC=-3.32±2.40
- `feat_aq_map_mean` ΔAC=-3.00±2.78
- `feat_edge_slope_stdev` ΔAC=-2.88±1.64
- `feat_aq_map_std` ΔAC=-2.02±1.97
- `feat_aq_map_p99` ΔAC=-1.96±1.37
- `feat_luma_kurtosis` ΔAC=-1.60±1.24
- `feat_aq_map_p95` ΔAC=-1.48±1.45
- `feat_laplacian_variance_peak` ΔAC=-0.90±0.86

### `zenavif`

**Cull candidates** (mean ΔAC > +1σ, removing helps argmin):

- `feat_quant_survival_uv_p10` ΔAC=+6.30±2.87, ΔOH=-0.08±0.26 (codec-specific)
- `feat_noise_floor_y_p75` ΔAC=+6.02±3.94, ΔOH=+0.09±0.36 (codec-specific)
- `feat_noise_floor_uv_p25` ΔAC=+5.76±5.10, ΔOH=+0.04±0.17 (codec-specific)
- `feat_noise_floor_y_p50` ΔAC=+5.48±2.77, ΔOH=-0.04±0.19 (codec-specific)
- `feat_aq_map_std` ΔAC=+5.30±3.59, ΔOH=+0.05±0.19
- `feat_edge_slope_stdev` ΔAC=+4.92±4.67, ΔOH=-0.05±0.27
- `feat_noise_floor_y_p25` ΔAC=+4.92±3.93, ΔOH=-0.06±0.30 (codec-specific)
- `feat_noise_floor_y` ΔAC=+4.56±3.89, ΔOH=+0.09±0.27 (codec-specific)
- `feat_high_freq_energy_ratio` ΔAC=+4.54±3.79, ΔOH=-0.04±0.29 (codec-specific)
- `feat_chroma_complexity` ΔAC=+3.72±2.78, ΔOH=-0.02±0.19 (codec-specific)
- `feat_alpha_bimodal_score` ΔAC=+2.70±2.15, ΔOH=-0.04±0.31 (codec-specific)
- `feat_alpha_present` ΔAC=+2.70±2.15, ΔOH=-0.04±0.31 (codec-specific)
- `feat_alpha_used_fraction` ΔAC=+2.70±2.15, ΔOH=-0.04±0.31 (codec-specific)
- `feat_noise_floor_uv_p50` ΔAC=+2.70±2.56, ΔOH=-0.03±0.27 (codec-specific)
- `feat_laplacian_variance_p99` ΔAC=+2.62±2.31, ΔOH=+0.08±0.23 (codec-specific)

### `zenjxl`

**Cull candidates** (mean ΔAC > +1σ, removing helps argmin):

- `feat_variance_spread` ΔAC=+1.58±1.52, ΔOH=-0.75±1.30 (codec-specific)
- `feat_flat_color_block_ratio` ΔAC=+1.56±1.29, ΔOH=-0.57±1.34 (codec-specific)
- `feat_laplacian_variance_p50` ΔAC=+1.40±1.20, ΔOH=-0.26±0.34 (codec-specific)
- `feat_patch_fraction` ΔAC=+1.24±0.99, ΔOH=-0.11±0.68 (codec-specific)
- `feat_variance` ΔAC=+1.12±1.03, ΔOH=-0.83±1.42 (codec-specific)
- `feat_noise_floor_uv_p90` ΔAC=+0.96±0.81, ΔOH=-0.05±0.56 (codec-specific)

**Keep (high-confidence)** (mean ΔAC < −1σ):

- `feat_cb_horiz_sharpness` ΔAC=-1.60±1.06
- `feat_dct_compressibility_y` ΔAC=-0.82±0.70
- `feat_cb_vert_sharpness` ΔAC=-0.78±0.78


---

*Generated by* `benchmarks/loo_cross_codec_consensus_2026-05-03.py`. Driver: `benchmarks/loo_driver_multiseed_2026-05-03.py`. Per-codec inputs: `benchmarks/loo_<codec>_multiseed_2026-05-03.tsv`.
