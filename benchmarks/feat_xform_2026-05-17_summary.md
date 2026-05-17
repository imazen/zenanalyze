# Feature-transform sweep — 3-codec results (2026-05-17)

**Methodology.** zensim-style two-tier screen, implemented in
`zentrain/tools/feature_transform_sweep.py`:

1. **Fast Pearson screen** — for each `(feat × transform × params)`,
   aggregate `|Pearson(transform(feat), bytes_log[:, c])|` over
   reachable cells, lift vs `identity` baseline.
2. **End-to-end confirmation** — train two `(192,192,192)` leakyrelu
   student MLPs at 60 epochs: baseline applies the codec's CURRENT
   `FEATURE_TRANSFORMS`; recommended applies the screen winners. Same
   `HISTGB_FAST` teacher, same val split.

Vocabulary: all 9 `zenpredict::FeatureTransform` variants with the
zensim v0_20 parameter sweep schedules.

## Headline — all 3 codecs improve

| Codec | Baseline argmin | Recommended argmin | Δ argmin (pp) | Δ mean overhead (pp) | Δ p99 overhead |
|---|--:|--:|--:|--:|--:|
| **zenjpeg** | 29.5 % | 32.6 % | **+3.14** | **−2.45** | **−59.7** |
| **zenwebp** | 20.8 % | 38.8 % | **+18.03** | **−2.10** | −3.2 |
| **zenavif** | 19.2 % | 21.0 % | +1.84 | −0.17 | +11.6 |

Note: absolute accuracies here (29.5 % / 20.8 % / 19.2 %) are below
the production v2.1 figures because this A/B uses `HISTGB_FAST`
teachers and 60-epoch training. The **deltas are apples-to-apples** —
same teacher, same student arch, same split, same epochs, only the
input transforms differ.

## Per-feature winner distribution (across all 3 codecs)

Out of 122 feature-codec entries (51 jpeg + 33 webp + 38 avif) where
the screen found lift ≥ 0.005:

| Best transform | Wins | Notes |
|---|--:|---|
| `winsor_p99` | 51 | Clip outlier tails to `[p1, p99]`. Dominates heavy-tailed gradient features (cb/cr sharpness, peak sharpness, aq_map percentiles). |
| `clip_then_log1p` | 37 | Subtract noise floor (`ε ∈ [p5, p10, p25, p50, p75]`), then `log1p`. Wins on aq_map_mean, laplacian_variance percentiles, dct_compressibility. |
| `signed_cbrt` | 16 | `sign(x)·∛|x|`. Mildly variance-stabilizing, no parameters. Wins on distinct_color_bins, chroma_complexity, laplacian_variance. |
| `quantile_bins` | 12 | 8-bin rank-order bucketization. Wins on power-law features (high_freq_energy_ratio, patch_fraction, gradient_fraction). |
| `log` | 4 | Plain `log`. Only chosen when feature is strictly positive AND distribution is exactly log-uniform. |
| `signed_log1p` | 2 | Tail-compression both directions. Rare wins. |

**Plain `log` and `log1p` rarely win** — the parameterized variants
beat them almost everywhere. The codec configs' current
`FEATURE_TRANSFORMS = {feat_X: "log1p"}` map is a coarse approximation
to what the sweep recommends.

## Notable per-feature findings

### zenwebp `feat_high_freq_energy_ratio`: +0.123 Pearson lift

- Baseline Pearson: **0.0024** (no signal under identity)
- `quantile_bins` Pearson: **0.1253**
- The raw feature has near-zero linear correlation with `bytes_log`
  but a strong rank-order relationship. `quantile_bins` turns the
  feature into its empirical rank, which the MLP can use directly.

### zenjpeg `feat_aq_map_p75`: +0.034 Pearson lift via `winsor_p99 [3.00, 4.86]`

- Baseline Pearson: 0.044
- Transformed: 0.078
- The feature has heavy outliers above the p99 mark; clipping to
  `[p1, p99]` brings the distribution into a range the MLP can fit
  without the outliers dominating.

### zenwebp `feat_pixel_count`: `clip_then_log1p ε=10.7` beats plain `log`

- Plain `log` baseline: 0.635
- `clip_then_log1p ε ≈ 10.7`: 0.673 (**+0.038 lift over `log`**)
- Subtracting `e^10.7 ≈ 44000` (≈ tiny-image floor) before `log1p`
  produces a cleaner distribution than `log(pixel_count)` directly.
  This contradicts the existing `FEATURE_TRANSFORMS = {"feat_pixel_count": "log"}`
  pattern that every codec uses today — the screen says `clip_then_log1p`
  is uniformly better.

### Same feature, different transform across codecs

`feat_laplacian_variance_p50`:
- zenjpeg: `clip_then_log1p ε=0.69`
- zenwebp: `signed_cbrt`
- zenavif: (not in KEEP_FEATURES)

The codecs' per-cell `bytes_log` distributions differ enough that
the optimal compress-the-tail transform isn't universal. **Tier-3
codec-specific transforms beat a "universal" 0.3 zenanalyze default
on this kind of feature.**

`feat_aq_map_mean`:
- zenjpeg: `clip_then_log1p ε=2.68`
- zenwebp: `log`
- zenavif: (not in winners)

Same conclusion — codec-specific transforms via the bake's metadata
section are more flexible than zenanalyze-native log transforms
applied uniformly.

## Implications for the zenanalyze 0.3 transform-defaults plan

The plan in the earlier review (move `log`/`log1p` defaults into
zenanalyze for universally-log'd features) is **superseded** by the
sweep results:

1. **`log` and `log1p` are wrong for most features.** The screen
   picks them only 6/122 times. The parameterized variants
   (`winsor_p99`, `clip_then_log1p`, `quantile_bins`) win 100/122.

2. **The optimal transform depends on the per-cell `bytes_log`
   distribution — codec-specific.** Same feature gets different
   winners across codecs. Moving transforms into zenanalyze would
   force a one-size-fits-all that's measurably worse than per-codec
   per-feature winners.

3. **The right architectural move is the opposite of the original
   plan**: keep zenanalyze emitting raw values; teach codec configs
   to use the FULL `FeatureTransform` vocabulary (including
   parameterized variants); have the sweep harness auto-derive the
   recommended `FEATURE_TRANSFORMS` + `FEATURE_TRANSFORM_PARAMS`
   maps from training data.

4. The runtime infrastructure to apply these (parameterized
   transforms with per-feature params via the bake's
   `FEATURE_TRANSFORM_PARAMS` metadata key) already exists in
   `zenpredict::feature_transform`. The wiring is done; the python
   trainer just needs to start using it.

## What's needed to adopt this

1. **Extend `train_hybrid.py:_apply_feature_transform`** to support
   all 9 transforms + per-feature params (currently only handles
   identity / log / log1p without params). ~50 lines of Python.

2. **Extend each codec_config.py** to optionally declare
   `FEATURE_TRANSFORM_PARAMS = {feat_name: [...], ...}` alongside the
   existing `FEATURE_TRANSFORMS` map. Drop-in from the sweep's
   `recommended_transforms.py` output.

3. **Bake side** (already done): `bake_picker.py` + `zenpredict-bake`
   already serialize `FEATURE_TRANSFORM_PARAMS`; the runtime
   `feature_transform.rs::apply_with_params` already applies them.

4. **One re-bake per codec.** Schema-hash changes; old bakes incompatible.

Estimated total work: ~1 day (small Python changes + 4 re-bakes).

## Caveats

- **HISTGB_FAST teacher in this A/B.** Production runs use HISTGB_FULL;
  the per-feature winners may shift slightly under a stronger teacher.
  Re-run the screen with `--use-full-teacher` (queued, not implemented
  yet) before shipping the production transforms.
- **Greedy per-feature.** The screen scores transforms one feature at
  a time; cross-feature interactions are NOT modeled in the Pearson
  aggregate. The end-to-end confirmation catches catastrophic
  interactions; subtler ones might still slip through.
- **Param sweeps are coarse.** 5 candidates per parameterized
  variant. Production tuning of the top-5 candidate features may
  want denser sweeps.

## Cross-references

- `zentrain/tools/feature_transform_sweep.py` — the harness
- `benchmarks/feat_xform_zenjpeg_2026-05-17/` — per-codec
  detailed results (screen_results.tsv,
  recommended_transforms.py, confirmation_summary.json, summary.md)
- `benchmarks/feat_xform_zenwebp_2026-05-17/`
- `benchmarks/feat_xform_zenavif_2026-05-17/`
- `zenpredict/src/feature_transform.rs` — runtime API
- `zenpredict-bake/src/composer.rs` — bake-time encoder
- `zensim/scripts/v_next/v0_20_feature_transform_greedy_screen.py` —
  original methodology
