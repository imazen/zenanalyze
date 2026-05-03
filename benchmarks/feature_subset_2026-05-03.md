# Semantic-aware feature + output scaling for picker configs (2026-05-03)

Layered on top of the per-head numeric standardize-and-rescale that
landed in `7e2534a` (PR #51). Three independent improvements applied
per codec:

1. **Per-output OutputSpec metadata** — `bounds`, `transform`,
   `discrete_set`, and `sentinel` for each scalar head, derived from
   each codec's published `InternalParams` ranges.
2. **Per-feature pre-standardize transforms** — `log` for log-spaced
   positive features (pixel_count, dimensions), `log1p` for
   heavy-tailed positives (variance, laplacian_variance, edge slope
   stdev), identity (default) for bounded ratios and sharpness scores.
3. **KEEP_FEATURES subset based on multi-seed LOO** — drop features
   whose mean ΔOH ≥ +0.10pp AND ≥ +0.5σ across 5 seeds (2026-05-03
   results from `loo_<codec>_multiseed_2026-05-03.tsv`).

## OUTPUT_SPECS shape per codec

| Codec   | bounded | discrete (round) | log-transformed (exp) | total |
|---------|---------|------------------|-----------------------|-------|
| zenwebp | 3       | 1                | 0                     | 4     |
| zenjpeg | 2       | 1                | 0 *                   | 3     |
| zenavif | 1       | 0                | 0                     | 1     |
| zenjxl  | 4       | 0                | 0                     | 4     |

\* zenjpeg lambda is naturally log-spaced ({8.0, 14.5, 25.0}); chose
`discrete_set + round` here rather than `Exp` because the trainer's
scalar head fits the linear-unit values directly. Switching to Exp
would require log-space label transformation in `train_hybrid.py` —
deferred for v0.2.

Discrete sets:
- zenwebp `filter_sharpness`: {0,1,2,3,4,5,6,7}
- zenjpeg `lambda`: {0.0, 8.0, 14.5, 25.0}  (0.0 = trellis-off sentinel)

## FEATURE_TRANSFORMS additions per codec

| Codec   | log | log1p | identity (default) | total tx | KEEP cols |
|---------|-----|-------|--------------------|----------|-----------|
| zenwebp | 3   | 5     | 25                 | 8        | 33        |
| zenjpeg | 1   | 10    | 40                 | 11       | 51        |
| zenavif | 3   | 7     | 42                 | 10       | 52        |
| zenjxl  | 3   | 6     | 55                 | 9        | 64        |

`log` features (across all codecs): `feat_pixel_count`, `feat_min_dim`,
`feat_max_dim`. zenjpeg uses log on `feat_pixel_count` only (it ships
both `feat_pixel_count` and `feat_log_pixels` — the latter is already
log so identity is correct).

`log1p` features: `feat_laplacian_variance` and percentiles,
`feat_variance`, `feat_variance_spread`, `feat_edge_slope_stdev`,
`feat_aq_map_std`, `feat_aq_map_p99` (zenjpeg only — long upper tail).

Identity (default) covers all bounded ratios in [0,1] (uniformity,
edge_density, patch_fraction, grayscale_score), sharpness scores
which are already standardized in zenanalyze, and discrete counts
(channel_count, distinct_color_bins).

## KEEP_FEATURES BEFORE → AFTER per codec (LOO consensus cull)

Cull rule: `mean ΔOH ≥ +0.10pp AND mean ≥ +0.5σ` across 5 seeds.

| Codec   | BEFORE | AFTER | Dropped |
|---------|-------:|------:|---------|
| zenwebp | 34     | 33    | 1       |
| zenjpeg | 51     | 51    | 0 *     |
| zenavif | 67     | 52    | 15 †    |
| zenjxl  | 67     | 64    | 3       |

\* zenjpeg LOO TSV has only 1 evaluated feature (incomplete sweep);
no defensible cull list yet. Re-run multi-seed LOO when the parallel
features TSV path is fixed.

† zenavif had 16 cull candidates; 15 dropped. `feat_cb_horiz_sharpness`
(mean +0.214pp, σ 0.301) was retained because it's the cb-side
horizontal partner of cb_vert/cb_peak (kept), preserving the
`cb_*_sharpness` group's chroma-direction coverage. Dropping the
horizontal alone would distort the group balance.

### Dropped features

**zenwebp (1):** `feat_cr_sharpness` (mean +0.226pp, σ 0.263)

**zenavif (15):**
- `feat_palette_density` (+0.288pp, σ 0.406) — biggest cull, makes
  sense; rav1e doesn't use palette.
- `feat_laplacian_variance_p50` (+0.244pp, σ 0.369)
- `feat_quant_survival_y_p75` (+0.242pp, σ 0.194)
- `feat_aq_map_p75` (+0.232pp, σ 0.135)
- `feat_noise_floor_uv_p75` (+0.220pp, σ 0.387)
- `feat_laplacian_variance_p75` (+0.218pp, σ 0.179)
- `feat_quant_survival_uv` (+0.200pp, σ 0.250)
- `feat_aq_map_p99` (+0.200pp, σ 0.351)
- `feat_laplacian_variance_p90` (+0.194pp, σ 0.252)
- `feat_aq_map_mean` (+0.170pp, σ 0.332)
- `feat_aq_map_p50` (+0.166pp, σ 0.296)
- `feat_noise_floor_uv_p90` (+0.160pp, σ 0.197)
- `feat_edge_density` (+0.150pp, σ 0.212)
- `feat_gradient_fraction_smooth` (+0.102pp, σ 0.164)
- `feat_cr_sharpness` (+0.100pp, σ 0.170)

**zenjxl (3):**
- `feat_noise_floor_y` (+0.508pp, σ 0.363) — strongest cull
  candidate. Aligns with JXL's XYB-domain coding: raw luma noise
  floor is less informative once chroma transforms re-bind the noise.
- `feat_gradient_fraction` (+0.266pp, σ 0.512)
- `feat_edge_slope_stdev` (+0.234pp, σ 0.254)

## Plumbing changes

- `zentrain/tools/train_hybrid.py`:
  - Added module-level `FEATURE_TRANSFORMS`, `OUTPUT_SPECS`,
    `SPARSE_OVERRIDES` placeholders + binding in `load_codec_config`.
  - `_apply_feature_transform` helper (identity / log / log1p) with
    safe x≤0 handling.
  - `load_features` returns a parallel `transforms` list and applies
    the transform inline before standardize.
  - Model JSON now emits `feature_transforms`, `output_specs` (per-
    output array, expanded from head-keyed dict via `output_layout`),
    and `sparse_overrides`.
- `tools/bake_picker.py`: emits `zentrain.feature_transforms`
  metadata section (newline-separated utf8) when the trainer JSON
  ships non-identity transforms. Length-validates against `feat_cols`.
- 4 codec configs updated with FEATURE_TRANSFORMS + OUTPUT_SPECS +
  KEEP_FEATURES subset (zenavif/zenjxl) per the LOO TSVs.

## Caveats

The codec runtime (`zenpredict`) does not yet read
`feature_transforms` — until that lands, fresh bakes with
non-identity transforms produce silently-wrong predictions at
inference. Existing bakes (which were trained without transforms)
remain valid because they fed raw values into a scaler fit on raw
values. New retrains with these configs need the runtime side
shipped together.

`zenjpeg_picker_config.py` references a stale features TSV path
(`zq_pareto_features_2026-05-01.tsv`); the actual file is
`zq_pareto_features_2026-05-01_parallel.tsv` (96 cols, larger
schema). Path was untouched here — needs user direction on which TSV
to use for v2.2 retrain.
