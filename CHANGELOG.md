# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> **Note on cross-crate releases:** the workspace publishes its
> member crates independently. The zenpredict 0.1.0 entries that
> were under `[Unreleased]` during development are now in the dated
> [`## [zenpredict 0.1.0] - 2026-05-01`](#zenpredict-010---2026-05-01)
> section below. Items remaining under `[Unreleased]` belong to the
> next zenanalyze release (size-invariance discipline, patch
> fingerprint, threshold recalibration).

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release.
     Add items here as you discover them. Do NOT ship these piecemeal — batch them. -->

<!-- 13-redundant-pixel-count-transforms cull is COMPLETE — moved to
     "Removed" section below. -->

- **`feat_indexed_palette_width` (id 30) replaced by
  `feat_palette_log2_size` (id 121).** The 2026-05-02 cross-codec
  ablation flagged `IndexedPaletteWidth` as Tier 0 redundant against
  `PaletteFitsIn256` on all 4 codecs (min |r| 0.9680–0.9972,
  `n_unique = 4` everywhere) — but the n_unique=4 was the feature's
  full codomain, not a corpus gap. The old codomain `{0, 2, 4, 8}`
  also lacked the 1-BPP case for binary content (PNG-1 saves a
  measurable bit/pixel vs PNG-2 on monochrome scans) and didn't
  surface JXL Modular palette breakpoints at 9..15 (≤512..32768
  colours). The replacement emits `ceil(log2(distinct))` clamped to
  `[1, 15]` with **`24` as the truecolor saturation sentinel**
  (replaced the original `0` after picker-training analysis: a
  discontinuous `..., 14, 15, 0` jump fights the trained MLP's
  gradient signal; `24` keeps the scale monotonic with a meaningful
  9-unit gap that itself encodes "we saturated the 5-bit binning;
  this is unambiguously truecolor"). 24 is the bit-width the source
  would need with no palette compression at all (8 bits × 3
  channels), so it's a defensible upper bound. The empty-image edge
  case folds to `1` rather than producing a separate sentinel.
  `PaletteLog2Size` is in `PALETTE_FULL_FEATURES`, so requesting it
  forces the full-scan path and the full 1..15 ∪ {24} resolution is
  always available — callers do NOT need to manually co-request
  `DistinctColorBins`. Both old and new variants are
  `#[cfg(feature = "experimental")]` with no in-tree consumer at
  the time of the swap; stable feature id 30 is reserved retired.
- **5-bit-per-channel bin storage bias documented on
  `feat_distinct_color_bins` and `feat_palette_fits_in_256`.** The
  32 KB / 32 768-cell bin array (chosen so it fits in one L1D way)
  under-counts the true 8-bit distinct-colour count whenever colours
  fold into the same 5-bit cell. Bias is always toward undercount,
  never over. Quantitative collision rate at uniform distribution:
  ~1 % at 256 colours, ~12 % at 4 096, saturates at 32 768.
  Consequence for `feat_palette_fits_in_256`: false-positive rate at
  C ∈ [257, ~300] true 8-bit colours where collisions pull the
  binned count back under 257. Encoders consuming the boolean would
  attempt indexed mode and fall back on the failed fit — correctness
  preserved, one wasted attempt per false positive. Callers who
  can't tolerate the false-positive rate should run their own exact
  8-bit pass on borderline cases.
- **Remove 3 redundant new shape/smoothness features**
  (`AnalysisFeature::ChromaKurtosis` = 117,
  `AnalysisFeature::UniformitySmooth` = 118,
  `AnalysisFeature::FlatColorSmooth` = 119). Cross-codec ablation
  (zenjpeg + zenwebp + zenavif + zenjxl, 2026-05-01) showed all three
  Tier-0 redundant on ≥3/4 codecs against existing features (`Uniformity`,
  `FlatColorBlockRatio`, the chroma_grad_sum-derived sharpness signals).
  The two new features that EARNED their keep — `LumaKurtosis` (116)
  and `GradientFractionSmooth` (120) — remain. Stable feature ids 117,
  118, 119 stay reserved (never recycled).
- **Remove the `composites` cargo feature and its 4 enum variants**
  (`AnalysisFeature::TextLikelihood` = 27,
  `AnalysisFeature::ScreenContentLikelihood` = 28,
  `AnalysisFeature::NaturalLikelihood` = 29,
  `AnalysisFeature::LineArtScore` = 45). The composite scores are
  hand-tuned weighted sums of stable raw signals; their coefficients
  drift faster than the API can usefully expose. Empirically
  `PatchFraction` (AUC = 0.88) outperforms `ScreenContentLikelihood`
  as a single discriminator anyway. Stable feature ids 27, 28, 29, 45
  stay reserved (never recycled). Migration: consume raw signals
  directly (LumaHistogramEntropy, EdgeDensity, ChromaComplexity,
  PatchFraction, FlatColorBlockRatio, DistinctColorBins) or train a
  small zenpredict model on a labeled corpus and ship the `.bin`.
  `tier3::compute_derived_likelihoods` and the four variants now
  emit `#[deprecated]` warnings in 0.1.x to surface the migration.

### Removed — 13 mathematical transforms of `feat_pixel_count` (cull complete 2026-05-02)

The full list of redundant pixel-count transforms originally queued
for cull (`log_pixels`, `log2_pixels`, `log10_pixels`,
`log_pixels_rounded`, `sqrt_pixels`, `bitmap_bytes`,
`log_bitmap_bytes`, `log_min_dim`, `log_max_dim`,
`log_padded_pixels_{8,16,32,64}`) are now removed:

- ids 94..104 (the 11 secondary transforms) retired in commit
  `e5c3c39` (2026-05-01).
- ids 57 (`LogPixels`) and 60 (`BitmapBytes`) retired in this
  commit (2026-05-02), after the 4-codec cross-codec ablation on
  the expanded multi-axis corpus confirmed Spearman 1.0 with
  `feat_pixel_count` on all four codecs at min |r| 0.9656–1.0000
  and no LOO impact. Tier 1.5 caveat surfaced during the run:
  `feat_log_pixels` showed +0.596 Δpp on zenjpeg permutation
  importance despite Spearman 1.0 with `feat_pixel_count` —
  permutation importance mis-attributes signal across redundant
  pairs, so **Tier 0 correlation is the load-bearing test, not
  Tier 1.5**.

All retired ids are added to `RESERVED_RETIRED_IDS` and never
recycled. **Kept** (still load-bearing in zenwebp at LOO Δ
+0.10–0.14pp on m4/m5/m6 method choice): `feat_min_dim` (58),
`feat_max_dim` (59), `feat_aspect_min_over_max` (61),
`feat_log_aspect_abs` (62).

Picker configs and `zentrain/FOR_NEW_CODECS.md` updated to drop the
removed feature names from `KEEP_FEATURES`.

### Added — picker training principles + cross-codec defaults (will ship with next zenanalyze release)

- **`zentrain/PRINCIPLES.md`** — single source of truth for what's
  invariant across codec pickers (zenjpeg / zenwebp / zenavif /
  zenjxl / zenpng / zengif / zenpicker / zensim). Covers the data
  discipline (sweep four dimensions, per-(image, size) zensim
  ceiling, sample-count floors, OOD bounds), argmin score
  composition (default size-optimal, RD-vs-time, hard time cap,
  time-to-percent-saved, multi-metric bakes for ssim2 / butter /
  zensim), per-codec adoption notes (each codec's categorical /
  scalar axes + rescue strategy default), default settings cheat
  sheet (trainer / bake / sweep / runtime), validation gates that
  block release, and known landmines (re-bake triggers, OOD
  necessity, composite-features stability, metadata necessity).
  Cross-linked from `zentrain/README.md`, `zentrain/FOR_NEW_CODECS.md`
  (now points at PRINCIPLES first), and the top-level `README.md`.

### Added — zenpredict 0.1.0 (new Rust runtime crate)

- **`zenpredict/`** — zero-copy MLP runtime: ZNPR v2 binary format
  parser, forward pass (f32 / f16 / i8 weights, Identity / ReLU /
  LeakyReLU activations), `Predictor` scratch wrapper, masked argmin
  with `ScoreTransform` + `ArgminOffsets`, top-K with confidence,
  `FeatureBound` + `first_out_of_distribution`, typed-TLV metadata
  blob, `RescuePolicy` / `should_rescue` two-shot decision logic,
  Rust-side `bake::bake_v2` composer + JSON-driven `bake_from_json`
  + `zenpredict-bake` CLI binary. 80 Rust tests (63 unit + 7
  lifecycle + 8 json_bake + 5 doc tests). Cargo features: `std`
  (default), `bake` (default).
- **ZNPR v2 binary format** — fixed `#[repr(C)]` 128-byte header,
  packed `[LayerEntry; n_layers]` table, aligned data blobs at
  Section-described offsets, optional zero-copy `feature_bounds`
  Section, typed-TLV metadata blob with `bytes` / `utf8` / `numeric`
  value types. Magic changed from `ZNPK` to `ZNPR`; v1 bakes do not
  load (rebake required).

### Added — zenpicker 0.1.0 (codec-family meta-picker)

- **`zenpicker/`** — Rust crate. `MetaPicker` wraps a
  `zenpredict::Predictor` whose output dimension equals the
  `CodecFamily` count. Given features + target quality + an
  `AllowedFamilies` mask, picks one of `{jpeg, webp, jxl, avif, png,
  gif}`. Per-codec config pickers run after, resolving the family
  into a concrete encoder config. Bake declares the family order via
  `zenpicker.family_order` metadata key; runtime validates via
  `MetaPicker::validate_family_order`.
- **Layered picker architecture**: `zenpicker::MetaPicker` chooses
  the family → per-codec `zenpredict::Predictor` resolves the family
  into a concrete config. Both layers use the same ZNPR v2 format
  and the same `zenpredict` runtime.

### Renamed — three-crate cleanup

- **The unpublished `zenpicker` Rust shell** (the placeholder runtime
  that path-dep'd from `zenwebp`) → split into `zenpredict` (generic
  Rust runtime) and `zenpicker` (codec-family meta-picker; was
  briefly named `zenpickerchoose` in earlier drafts). The old shell's
  `Cargo.toml` and `src/` are deleted.
- **`zenpicker/tools/`, `zenpicker/examples/`** (the Python training
  pipeline) → moved to **[`zentrain/`](zentrain/)**. The Python
  pipeline is unambiguously named after what it does (train); the
  `zenpicker` name now belongs to the meta-picker call site.
- **Trainer-emitted metadata-key namespace** → moved from
  `zenpicker.*` (in v2 design drafts) to **`zentrain.*`**. Producer
  matches the namespace. Affected keys (all in
  `zenpredict::keys::*`):
  `zenpicker.profile` → `zentrain.profile`,
  `zenpicker.schema_version_tag` → `zentrain.schema_version_tag`,
  `zenpicker.feature_columns` → `zentrain.feature_columns`,
  `zenpicker.hybrid_heads_layout` → `zentrain.hybrid_heads_layout`,
  `zenpicker.provenance` → `zentrain.provenance`,
  `zenpicker.calibration_metrics` → `zentrain.calibration_metrics`,
  `zenpicker.safety_report` → `zentrain.safety_report`,
  `zenpicker.bake_name` → `zentrain.bake_name`,
  `zenpicker.reach_rates` → `zentrain.reach_rates`,
  `zenpicker.reach_zq_targets` → `zentrain.reach_zq_targets`.
- **Constant rename**: `zenpredict::keys::PICKER_PROFILE` →
  `zenpredict::keys::PROFILE`. The `PICKER_` prefix was redundant
  given the producer-namespace rename.
- The `zenpicker.*` namespace is reserved for the **meta-picker**
  (currently just `zenpicker.family_order`); codec-private keys live
  under `<codec>.*` (e.g. `zenjpeg.cell_config`).

### Changed — Python training pipeline

- **`tools/bake_picker.py` rewritten** to emit a portable
  `BakeRequestJson` and shell out to `zenpredict-bake`. Byte-packing
  (struct.pack, magic constants, i8 quantization, alignment padding)
  all moved to Rust. Drops ~180 lines.
- **`tools/bake_roundtrip_check.py`** runs against the new
  `zenpredict load_baked_model` example, handles all three
  activations (`relu` / `leakyrelu` / `identity`) in the numpy
  reference forward pass.
- **`tools/test_bake_roundtrip.py`** (new) — regression test that
  exercises the full bake → CLI → load → forward chain across
  3 activations × 3 dtypes = 9 combinations on synthetic models.
  All pass with f32 max-abs-diff ≤ 6e-8, f16 ≤ 2.4e-7, i8 ≤ 6e-8.

### Documentation

- **`MIGRATION.md`** — full walkthrough at the repo root: rename
  table, Cargo dep diff, source diffs (argmin / hybrid-heads /
  top-K / reach-rate gate / rescue / bounds / metadata),
  alignment note, layered-picker architecture, timeline.
- **`README.md`** — "Companion crates in this repo" table now lists
  `zenpredict`, `zenpicker`, and `zentrain`.

### Added — zenpicker size-invariance discipline (legacy training-pipeline notes; will ship with next zenanalyze release)

- **Size invariance is a safety property.** The picker is now
  *required* to be near-optimal at every image `(width, height)`, not
  just at the four sample sizes (`tiny / small / medium / large`)
  sampled at training time. This is now structural, not a "be
  thorough" guideline.
- **`tools/train_hybrid.py`: new safety-gate violations.** Two new
  thresholds in `DEFAULT_SAFETY_THRESHOLDS`:
  - `max_per_size_p99_overhead_pct` (default 80) — `PER_SIZE_TAIL`
    fires when any single `size_class`'s p99 overhead exceeds the
    ceiling.
  - `min_train_rows_per_size_zq` (default 50) — `DATA_STARVED_SIZE`
    fires when any `(size_class, target_zq)` training cell has fewer
    rows than the floor. Catches sweep harnesses that silently skip a
    size class for a chunk of the corpus.
- **`safety_report.diagnostics.train_rows_by_size_zq`**: new dict in
  the diagnostics block exposing per-`(size, zq)` training-row counts.
  Forwarded into the bake manifest by `bake_picker.py` (passes through
  the existing `safety_report` plumbing — no bake-tool change needed).
- **`zenpicker/tools/size_invariance_probe.py`** — post-bake
  generalization gate. Resizes a fixture corpus to each of `tiny /
  small / medium / large` and asserts the picker's argmin cell stays
  stable across sizes per `(image, target_zq)`. `--strict` exits 1
  when stability < threshold (default 90 %). Counterpart to the
  in-trainer `PER_SIZE_TAIL` / `DATA_STARVED_SIZE` violations.
- **Documentation updates.** `SAFETY_PLANE.md` gains a "Size
  invariance is a safety property" section; `FOR_NEW_CODECS.md` gains
  Step 1.5 ("sweep at all four size classes") between Steps 1 and 2;
  `tools/README.md` extends the canonical command sequence with the
  size-invariance probe; `README.md` marks the size-invariance gate
  ✅ v0.1; `lib.rs` docstring documents that size invariance is
  enforced at the codec edge, not in the picker runtime.

### Changed — zenpicker

- **`tools/train_hybrid.py`: codec-agnostic `CATEGORICAL_AXES` + `SCALAR_AXES`.**
  The trainer was documented as codec-agnostic but had `(color, sub,
  trellis_on, sa)` and `(chroma_scale, lambda)` hardcoded. New behavior
  reads `CATEGORICAL_AXES` and `SCALAR_AXES` (and optional
  `SCALAR_SENTINELS`, `SCALAR_DISPLAY_RANGES`) from the codec config
  module. zenjpeg metrics reproduce bit-exactly when its config
  declares the prior shape (now done in
  `examples/zenjpeg_picker_config.py`). New consumers like zenwebp
  declare their own (e.g. `["method", "segments"]` ×
  `["sns_strength", "filter_strength", "filter_sharpness"]`) without
  forking the trainer. Manifest gains `categorical_axes`,
  `scalar_axes`, `scalar_sentinels` fields; `lambda_notrellis_sentinel`
  is preserved as a back-compat alias. Closes the gap surfaced when
  wiring zenwebp through FOR_NEW_CODECS.md.

### Added — patch fingerprint cost-efficient sibling + quant-survival signals

- **`PatchFractionFast`** (`patch_fraction_fast`, id 52, experimental).
  Cost-efficient sibling of `PatchFraction`: same sort-and-sweep
  collision-fraction construction, but the per-block fingerprint is a
  64-bit dHash of raw 8×8 luma folded to 32 bits. **~10× cheaper per
  block** than the DCT-based `patch_fraction`. AUC 0.852 (DCT 0.880);
  peak F1 **0.779** (DCT 0.763) on the 219-image labeled corpus.
  Pearson correlation with `patch_fraction`: 0.99 — same content
  signal at lower cost. Pick on cost.
- **`QuantSurvivalY` / `QuantSurvivalUv`** (id 53 / 54, experimental).
  Mean fraction of luma / chroma AC coefficients surviving jpegli-
  default quantization at d=2.0 (q≈75). Approximates per-block JPEG
  file-size cost. Direction is **inverted** vs initial hypothesis on
  this corpus: photos preserve slightly MORE coefficients than screens
  (photo sensor noise gives every block some survivable ACs; true-flat
  screen regions hash to 0% survival). Standalone AUC vs is_screen:
  Y = 0.413 (|−0.5| = 0.087, weak), Uv = 0.336 (|−0.5| = 0.164,
  decent in inverted direction). Both genuinely orthogonal to the
  patch_fraction family (correlation −0.23 to −0.32) — useful as
  inputs to multi-feature classifiers even if standalone AUC is
  modest. See imazen/zenanalyze#1 for the ablation.

### Reserved feature IDs

- **id 51** reserved (was `PatchFractionWht`, removed pre-stabilization).
  WHT-based variant correlated 0.997 with `patch_fraction`, AUC 0.864
  vs DCT's 0.880, ~2.5× cheaper than DCT but ~4× more expensive than
  dHash with no AUC win over dHash. Cut in favor of the cheaper
  `patch_fraction_fast`. ID reserved so future wire-format compat
  isn't broken if a different WHT-shaped feature wants the slot.

### Changed (operating thresholds — read this before upgrading consumers)

- **Calibrated `text_likelihood` / `screen_content_likelihood` / `natural_likelihood`
  to saturate at 1.0 on real content** (previously capped at 0.71 / 0.70 / 0.69 on
  the 219-image labeled corpus because the sub-components don't co-fire to their
  individual maxima). Each composite now divides its raw value by its empirical
  saturation point, then re-clamps to `[0, 1]`. **AUC is preserved** (rank order
  unchanged) — only the scale stretches.
- **Reformulated `screen_content_likelihood`** when `experimental` is enabled:
  the old `palette_small`-based formula collapsed to 0 on real screens (charts /
  anti-aliased UIs routinely exceed 4000 distinct color bins). Replaced with
  `0.6 * patch_fraction + 0.4 * flat_high`, which lifts AUC from 0.831 to **0.845**
  and peak F1 from 0.59 to **0.78** on the same labeled corpus. When `experimental`
  is off, the legacy formula remains (still stretched by the same divisor).
- **Operating thresholds shifted** — divide old thresholds by the per-composite
  saturation point to translate, or use the recommended new thresholds:

  | Composite | Old threshold | **New threshold** | New F1 | New AUC |
  |---|---:|---:|---:|---:|
  | `text_likelihood` | 0.30 | **0.35** | 0.585 | 0.713 |
  | `screen_content_likelihood` | 0.60 | **0.80** | **0.779** | **0.845** |
  | `natural_likelihood` | 0.06 | **0.10** | **0.923** | 0.814 |

  The `screen_content_likelihood >= 0.80` threshold reflects both the formula
  reshape AND the saturation stretch.
- **Dependency change:** `screen_content_likelihood` now requires Tier 3 to run
  (it reads `patch_fraction` when `experimental` is enabled). Previously it only
  required the palette pass. `T3_NEEDED_BY` updated to include
  `ScreenContentLikelihood` so the dispatcher activates Tier 3 automatically.
  Callers requesting only `ScreenContentLikelihood` will see the analysis pay an
  extra Tier 3 pass; callers already requesting any other Tier 3 feature pay
  nothing extra.

## [zenpredict 0.1.0] - 2026-05-01

First crates.io release of `zenpredict`. The runtime, format, and
public API were described above under `[Unreleased]` (kept there for
narrative continuity with the rest of the workspace); this section
enumerates the prepublish hardening on top of that work.

### Hardened (prepublish audit)

- **Format-parse safety.** `Model::from_bytes` now uses `checked_mul`
  + a typed `PredictError::DimensionOverflow` for every dimension
  multiplication that previously could wrap on i686 / wasm32 (n_inputs
  * 2 for feature_bounds, expected_count * 4 for f32 sections,
  expected_count * 2 for f16). `from_bytes_with_schema` checks the
  header's `schema_hash` BEFORE walking the layer table — adversarial
  bakes with a wrong schema and giant `n_layers` bail in O(1) instead
  of allocating.
- **Scaler /0 guard.** `inference::forward` mirrors sklearn's
  `_handle_zeros_in_scale`: a `scaler_scale[i] == 0.0` (zero-variance
  column) is treated as `1.0`, so the column passes through as
  `(x - mean)` instead of producing NaN/inf. Cost is one branch per
  input dim per predict.
- **argmin contract tightened.** `argmin_masked` /
  `argmin_masked_top_k` / `_with_scorer` / `threshold_mask` now
  **panic on `mask.len() < predictions.len()`** in both debug AND
  release. Short masks used to silently deny high-index cells, which
  hid real bugs. Documented NaN-skip semantics, lowest-index tie-break,
  and the `ScoreTransform::Exp` no_std fallthrough behavior.
- **`scratch buffer` reuse documented.** `Predictor::predict` does
  not zero scratch between calls; every layer's matmul writes biases
  before accumulating, so stale data never leaks. Calling `predict`
  twice with the same features is deterministic.
- **Reserved fields documented.** Header `_pad0` / `reserved[14]` /
  `flags`, LayerEntry `reserved[3]` / `flags` are ignored on read
  (forward-compat) and bakers MUST zero them. `value_type=3..=15` in
  the metadata TLV is reserved for future compressed payloads.

### CI / packaging

- **CI matrix expanded** to build + test zenpredict on
  `ubuntu-latest`, `windows-11-arm`, `macos-15-intel`, `macos-latest`
  with three feature combos (`""`, `"std"`, `"std,bake"`). Cross job
  also runs zenpredict on `i686-unknown-linux-gnu` (32-bit overflow
  guards needed live-fire validation) and `aarch64-unknown-linux-gnu`.
- **`zenpredict/deny.toml`** — cargo-deny configuration enforcing
  permissive licenses on transitive deps (MIT / Apache / BSD / ISC /
  Zlib / Unicode / 0BSD / CC0). First-party AGPL workspace crates
  are exempted via per-name exception. Yanked crates and unknown
  registries / git sources are denied.
- **Dependency versions pinned** to full triplets per CLAUDE.md:
  bytemuck 1.25.0, serde 1.0.228, serde_json 1.0.149, rand 0.9.2.

### Tests

- 91 lib tests (was 80 at original landing), +5 NaN/tie/short-mask
  argmin regressions, +3 schema-early-bail / scaler /0 / from_bytes
  agreement tests.
- 8 integration tests (json_bake), 7 lifecycle tests, 7 doc tests.

### Bake side

- **`tools/bake_picker.py --dtype` default flipped from `f32` to
  `i8`**. Typical bake size: ~30 KB f32 → ~15 KB f16 → ~8 KB i8;
  most consuming binaries (codec wasm, mobile builds) don't want
  > 80 KB of weights. Held-out argmin-acc delta from f32→i8 is < 0.5
  pp on every bake shipped, well inside calibration-noise.
- **`zentrain/examples/zenwebp_picker_config.py`** — reference codec
  config for hybrid-heads training (PR #57 follow-up).
- Docs lead with `--activation leakyrelu` as the fast trainer path
  (10–20× wall-clock vs sklearn-relu default).

## [0.1.0] - 2026-04-28

First public release. Published to crates.io as
[zenanalyze 0.1.0](https://crates.io/crates/zenanalyze/0.1.0); GitHub
release [zenanalyze-v0.1.0](https://github.com/imazen/zenjpeg/releases/tag/zenanalyze-v0.1.0).

### Cargo features

- **default** — stable raw signals (variance, edge density, chroma
  sharpness, DCT energy, alpha, palette, distinct-color bins). Numeric
  drift in 0.1.x bounded by the threshold contract; signatures frozen.
- **`experimental`** — research-stage signals (PatchFraction,
  AqMapMean / AqMapStd, NoiseFloorY / UV, GradientFraction, source-direct
  HDR / wide-gamut / bit-depth tier). Metric definition or scale may
  change in 0.1.x patches.
- **`composites`** — classifier-style scores (TextLikelihood,
  ScreenContentLikelihood, NaturalLikelihood, LineArtScore). Hand-tuned
  weighted combinators of stable raw signals; the *combinator
  coefficients* drift as the corpus calibration matures. The raw signals
  these consume are stable.

### Added (since the rest of this section)

- **`composites` cargo feature** gating `TextLikelihood` /
  `ScreenContentLikelihood` / `NaturalLikelihood` / `LineArtScore`. The
  three likelihood enum variants and `LineArtScore` now require this
  cargo feature. Variants stay defined either way; without the flag
  their `RawAnalysis` fields, `into_results` writes, and SUPPORTED
  entries cfg out together. `compute_derived_likelihoods()` body is
  cfg-gated with a no-op stub when the flag is off.
- **Tier 1 SKIN const-bool axis.** `accumulate_row_simd` was
  `<BT601, FULL>`; now `<BT601, FULL, SKIN>`. Splitting `SkinToneFraction`
  off the `FULL` accumulators frees AVX2 register pressure (joint kernel
  spilled 12 vmovups + 13 vbroadcastss because all accumulators —
  luma stats, Hasler M3, BT.601 chroma matrix, Chai-Ngan thresholds —
  were live at once). Callers can now request just Variance /
  Colourfulness / EdgeSlope or just SkinToneFraction without paying for
  the other half. New `feature::TIER1_FULL_FEATURES` and
  `feature::TIER1_SKIN_FEATURES` sets drive an 8-arm match on
  (BT601, FULL, SKIN). 1 MP Tier 1 Variance only: 3.15 ms → 3.01 ms
  (-4 %).

### Added

- Initial release. Image content analyzers extracted from `zenjpeg::analyze`
  so other zen codecs can share the same oracle-trained feature pipeline.
- `analyze(PixelSlice)` and `analyze_rgb8(&[u8], w, h)` entry points returning
  `AnalyzerOutput`.
- **Alpha analysis** (`alpha_present`, `alpha_used_fraction`, `alpha_bimodal_score`).
  Mirrors zenwebp's classifier alpha-histogram bimodality detector — bypasses
  the RowStream RGB8 conversion and reads the source `PixelSlice` alpha
  channel directly via `PixelFormat`-keyed byte offsets. Supports RGBA8/BGRA8/
  RGBA16/RGBAF32/GrayA8/GrayA16/GrayAF32; premultiplied alpha is treated as
  opaque (`alpha_present = false`) per coefficient's convention. Stride-sampled
  to share the `pixel_budget`.
- **Cross-codec features** added per the prior-art audit (Hasler-Süsstrunk 2003,
  Pech-Pacheco, libwebp `analysis_enc.c`, libjxl modular):
  - `colourfulness` — Hasler-Süsstrunk M3 colourfulness metric
    (σ_rg² + σ_yb² and μ_rg + μ_yb in opponent space). Drives JXL VarDCT
    chroma quant scale, GIF palette size, JPEG chroma subsampling.
  - `laplacian_variance` — variance of discrete 5-tap Laplacian over
    sampled luma. Classical Pech-Pacheco blur/sharpness axis. Drives JXL
    noise synthesis, AVIF film-grain, JPEG trellis effort.
  - `variance_spread` — `(max − min) / (max + 1)` of per-block luma
    variance, free piggyback on the existing tier-1 SIMD block-stats
    kernel. Captures spatial heterogeneity. Drives JPEG AQ, WebP SNS,
    AVIF cdef strength.
  - `dct_compressibility_y` — **libwebp `GetAlpha` exact algorithm** now:
    per-block 64-bin histogram of `|AC|/16`, α = `256 * last_non_zero / max_count`,
    averaged across sampled blocks. Folded into tier-3's existing DCT pass.
  - `dct_compressibility_uv` — **real chroma DCT pass** now: extends the
    block-extract loop to compute Cb/Cr per pixel (BT.601 integer-quantized,
    matching the luma scale), runs the same separable 1D-DCT + transpose,
    computes per-block α, takes `max(α_cb, α_cr)` per block. Reuses
    `dct1d_8` and `dct2d_8` so it's still LLVM-autovec-friendly. **Histogram
    bin divisor = 8 for chroma vs 16 for luma** because chroma DCT
    coefficients run ~half luma's; finer binning lifts chroma α from p50≈1
    into the same dynamic range as luma α (p50≈5).
  - `palette_density` — `distinct_color_bins / min(pixel_count, 32 768)`,
    a **scale-aware** fraction of populated 5-bit-per-channel bin space.
    Avoids the absolute-count saturation that makes raw `distinct_color_bins`
    drift with image resolution. Real-corpus discrimination: photos p50 ≈
    0.06–0.09, screen content p50 ≈ 0.009 — clean ~7× separation.
  - `distinct_color_bins_chao1` — **Chao1 estimator** (Chao 1984) of the
    true full-image distinct-bin count, derived from singletons and
    doubletons in the budget-sampled histogram. Reduces sampling error
    on multi-megapixel images at default budget: corpus convergence
    study (139 images, 1-50% sampling rates) shows raw count p50 error
    27%/13%/4% at 25%/50%/100% sampling vs Chao1 p50 error 12%/4%/0% —
    **2-3× improvement**. At full sampling, Chao1 = raw count.
    Implementation: 32 KB per-bin u8 saturating-counter array tracked
    alongside the existing presence-bitset, no extra row fetches.
  - `patch_fraction` — **real perceptual-hash patch detection** now:
    each sampled block produces a 32-bit DCT signature from the lowest
    16 zigzag-positioned AC coefficients (sign bit + above-threshold
    bit per coefficient). Sort-and-sweep across the small sampled set
    (≤256 by default) gives the fraction of blocks whose signature
    matches at least one other block. Real perceptual matching, ~10 µs
    overhead total.

  Implementation notes: colourfulness and Laplacian fold into tier-1's
  per-row SIMD pass (4 + 3 new f64 accumulators). The Laplacian
  precomputes BT.601 luma into 3 row-scratch buffers in a single linear
  pass before the 5-tap stencil — avoids computing luma 5× per pixel.
  variance_spread piggybacks on the existing tier-1 block-stats kernel.
  dct_compressibility_y reuses tier-3's already-extracted DCT
  coefficients. Net cost: +1.2 ms on a 4 MP image versus the pre-feature
  baseline.
- `analyze_with(slice, &AnalyzerConfig)` and `analyze_rgb8_with(...)` entry
  points accepting an explicit budget config. Mirrors the
  `coefficient::analysis::feature_extract::ExtractConfig` surface so callers
  migrating from coefficient see the same shape.
- `AnalyzerConfig` (`#[non_exhaustive]`) with `pixel_budget` (default
  500_000, the trained value) and `hf_max_blocks` (default 256, the trained
  value). `AnalyzerConfig::full()` disables sampling for reference scans.
- `AnalyzerOutput` is now `#[non_exhaustive]` so future tier additions
  (HDR luminance histogram, Oklab chroma stats, etc.) can land without a
  0.x-major bump.
- Pixel-format coverage tests across the descriptor matrix every zen* codec
  uses at its encoder boundary: RGB8/RGBA8/BGRA8, RGB16/RGBA16, RGBF32/RGBAF32,
  GRAY8/GRAY16/GRAYF32 — all sRGB and linear transfer paths exercised through
  `RowConverter`.
- CI now runs `cargo test -p zenanalyze` on every Test platform (ubuntu,
  ubuntu-arm, macos, macos-intel, windows, windows-arm), `cross test` on
  i686, and `cargo build` on wasm32-wasip1 (scalar + SIMD128).
- Sanity tests across image-size regimes (`tiny_*`, `medium_*`, `large_*`):
  defends well-formedness invariants (geometry, range bounds on every
  feature) at 1×1, 2×2, 3×3, 7×7, 8×8, 8×16, 16×8, 256×256, 512×256,
  2048×2048. Includes a determinism test (same input → bit-identical
  output) and a timing tripwire on 2048×2048 default-budget analyze
  (must complete in <1 s — flags accidentally O(N²) regressions or
  budget-plumbing breakage).

### Fixed (breaking — feature numerics shift)

The 0.1.x line is explicitly *not* a frozen contract; downstream
consumers that compile-in fitted models (oracle decision trees,
selectors) must pin to a specific zenanalyze patch version and
re-validate / retrain when bumping. The fixes below all change Tier 2
or Tier 3 feature outputs.

- **Tier 2 Cb/Cr asymmetry repaired.** `rgb_to_ycbcr_q` no longer
  halves Cb/Cr when `cr < 0`. Previously, gradient energies for
  cool/green-dominant edges were systematically half those of
  warm/red-dominant edges of equal magnitude. `cb_peak_sharpness`
  and `cr_peak_sharpness` may now be up to 2× higher on average for
  images whose dominant color edges fall in the `cr < 0` half-plane.
- **Tier 2 odd-width edge-column drop fixed.** `span = (width − 2) / 2`
  → `(width − 1) / 2`. Pre-fix, odd widths (3, 5, 7, …) systematically
  dropped the rightmost triplet column from horizontal chroma
  sampling. Most affected: tiny images and any image whose right
  edge holds the dominant chroma transition.
- **Tier 2 trailing-fragment drop fixed.** Trailing partial fragments
  are now flushed if they have any rows (`> 0`), not only if they
  exceed 16 triplet-rows (`> 16`). Pre-fix, **any image with
  `height ≤ 32`** silently emitted zero for every Tier 2 field, and
  taller images undersampled the bottom by up to ~32 pixel rows.
- **Tier 3 high-frequency split changed from raster `k ≥ 16` to
  zigzag `k ≥ 16`.** Previously the metric used JPEG raster order,
  which puts `(u=4, v=0)` ("highest horizontal frequency, zero
  vertical") on the *low* side and `(u=0, v=2)` ("low horizontal,
  low-mid vertical") on the *high* side — biasing toward vertical
  detail. The new zigzag split matches JPEG's natural "drop high AC
  first" scan order and is symmetric in horizontal/vertical detail.
- **Threshold contract relaxed during 0.1.x.** `lib.rs` doc no longer
  calls the contract "frozen". It now states the contract is
  iterating and downstream consumers must pin patch versions.

### Documentation

- Rewrote the misleading "the codegen pass can emit `if features.<name>`"
  claim in `AnalyzerOutput` doc. There is no codegen today: zenjpeg's
  `encode::adaptive::infer_bucket` hand-distills the oracle into
  thresholds. Doc now says so explicitly.

### Downstream impact

- `zenjpeg::encode::adaptive::infer_bucket` carries hand-derived
  cutoffs against the *old* analyzer outputs (`cb_peak_sharpness > 5.0`,
  `> 8.0`; `high_freq_energy_ratio > 0.30`). These are flagged
  `CALIBRATION-PENDING` in source — neither bucket choice is
  catastrophic (PhotoDetailed and PhotoNatural both ship sensible
  configs), but a corpus sweep should tighten the boundary before the
  next adaptive-quality recalibration pass.
- coefficient continues to vendor its own `evalchroma_ext.rs` copy.
  When it migrates to depend on zenanalyze, it picks up these fixes
  automatically.

### BREAKING — opaque feature-set API is now the only API

Legacy `AnalyzerOutput` / `AnalyzerConfig` / `analyze` / `analyze_with`
/ `analyze_rgb8` / `analyze_rgb8_with` are deleted from the public
surface. The only public entries are:

- `analyze_features(slice, &AnalysisQuery) -> Result<AnalysisResults, _>`
- `analyze_features_rgb8(rgb, w, h, &AnalysisQuery) -> AnalysisResults`

All public types are opaque (no `pub` fields anywhere on
[`feature::AnalysisQuery`] / [`feature::AnalysisResults`] /
[`feature::ImageGeometry`]). Sampling budgets are crate invariants;
the `#[doc(hidden)]` `__internal_with_overrides` backdoor exists for
oracle re-extraction tests but is not public-stable.

SIMD tier signatures changed from `&mut AnalyzerOutput` to
`&mut feature::RawAnalysis`. The dense flat record is now
`pub(crate)` and macro-generated alongside the rest of the feature
table — adding a feature is a single row at `features_table!`.

Side-effect-free property guaranteed: requesting feature *F* alone
produces the same numeric value as requesting *{F, …}*. Verified by
the `requesting_more_features_does_not_change_existing_values` test,
which runs every supported feature alone and against
`FeatureSet::SUPPORTED` and asserts bit-equality.

`compute_derived_likelihoods` is const-bool gated on `<T3, PAL>`
axes — likelihoods whose dependencies didn't run are left at default
and dropped by `into_results`. Dispatch axes use precomputed
[`feature::T3_NEEDED_BY`] / [`feature::PAL_NEEDED_BY`] supersets so
asking for a derived feature gates the right tier on. Layered defense:
even if dispatch is wrong, the gate prevents garbage; even if the
gate were bypassed, `into_results` only emits requested features.
**Caller never sees garbage — only `Some(real_value)` or `None`.**

### Opaque feature-set API (initial draft)

New public types (in module `feature`, draft — not yet promoted to
the crate root):

- `AnalysisFeature` — `#[non_exhaustive] #[repr(u16)]` enum with
  sequential discriminants `0..30`. Discriminants are **immutable
  once shipped**; retired variants keep their slot. `name()` is the
  only public accessor (matches the legacy `AnalyzerOutput` field
  names for JSON-sidecar continuity); `id` / `from_u16` /
  `is_active` are `pub(crate)`.
- `FeatureValue` — `#[non_exhaustive]` `Copy` enum (`F32` / `U32` /
  `Bool`) with type-checked `as_*` accessors and a lossless
  `to_f32()` coercion.
- `FeatureSet` — opaque `[u64; 4]` bitset, full `const fn` set math
  (`union` / `intersect` / `difference` / `intersects` / `contains`
  / `contains_all` / `is_empty` / `len`). The only "all features"
  entry point is `FeatureSet::SUPPORTED` (which intentionally may
  shrink with future cargo-feature gating); there is no `all()`.
- `AnalysisQuery` — opaque request handle. **Only public knob is
  the feature set** — sampling budgets are crate invariants
  calibrated against shipped oracle / threshold tables, exposed
  only via a `#[doc(hidden)]` `__internal_with_overrides` backdoor
  for tests / oracle re-extraction.
- `ImageGeometry` — opaque, `width()` / `height()` / `pixels()` /
  `megapixels()` / `aspect_ratio()`.
- `AnalysisResults` — opaque container; sparse
  `Vec<(AnalysisFeature, FeatureValue)>` storage sized to the
  requested set, no over-allocation. Public API is
  `get(feature) -> Option<FeatureValue>`, `get_f32` convenience,
  `requested()`, `geometry()`, `Debug`.

New entry points (additive — legacy `analyze` / `analyze_with` /
`AnalyzerOutput` / `AnalyzerConfig` retained for now):

- `analyze_features(slice, &AnalysisQuery) -> Result<AnalysisResults, _>`
- `analyze_features_rgb8(rgb, w, h, &AnalysisQuery) -> AnalysisResults`

Single-source-of-truth `features_table!` macro: one invocation in
`feature.rs` generates the `AnalysisFeature` enum, its
`id`/`from_u16`/`is_active`/`name` impls, the `pub(crate)` dense
`RawAnalysis` SIMD-target struct, the `RawAnalysis::into_results`
sparse-translator, `FeatureSet::SUPPORTED`, and
`From<&AnalyzerOutput> for RawAnalysis` — all in lockstep. Adding a
feature is one row at the table.

`analyze_features` dispatches via four `const bool` axes
(`PAL`/`T2`/`T3`/`ALPHA`). Each axis becomes straight-line
const-eval inside the monomorphized `analyze_specialized` body —
unrequested tiers are dead-code, no runtime branch, no tier-stats
computation. Sixteen monomorphizations cover all combinations.
Tier 1 is always-on (cheap, almost every caller wants something
from it). Measured at 8 MP, default target-cpu, runtime archmage
dispatch:

| Query | Time | ms/MP | vs legacy |
|-------|-----:|------:|----------:|
| Legacy (all features always) | 14.10 ms | 1.68 | 1.00× |
| **`Variance` only** | **3.35 ms** | **0.40** | **4.2×** |
| `DistinctColorBins` only | 6.55 ms | 0.78 | 2.2× |
| `DctCompressibilityY` only | 7.46 ms | 0.89 | 1.9× |
| `FeatureSet::SUPPORTED` (all) | 12.77 ms | 1.52 | 1.10× |

`AnalyzerConfig::pixel_budget` and `hf_max_blocks` demoted from
`pub` to `pub(crate)`. The legacy struct is now `Default`-only from
outside the crate; arbitrary budget overrides were silently
producing features that drifted from oracle-trained thresholds.

### Default — `hf_max_blocks` back to 1024 (accurate DCT features)

The cross-block f32x8 DCT cut Tier 3's per-block cost ~8-15× vs the
prior scalar separable DCT, so the convergence-elbow setting (1024
blocks, 4-6 % p50 error on DCT-energy features) now costs only ~+1 ms
at 8 MP over the 256-block sloppy default (11-16 % p50 error). The
earlier session reverted the default to 256 *because* of the scalar
DCT's perf cliff at 1024 (~+3 ms); with SIMD that pressure is gone,
and accurate features are unconditionally the right default.

| `hf_max_blocks` | DCT-energy p50 err | `patch_fraction` p50 err | 8 MP cost |
|---:|---:|---:|---:|
| 256 (old default) | 11-16 % | ~100 % | 0.95 ms |
| **1024 (new default)** | **4-6 %** | ~57 % | **1.95 ms** |
| 4096 | converged | structural limit | ~4 ms |

`patch_fraction` is still sample-limited at any block cap below
dense-scan and stays at 57 % p50 error at 1024 — acceptable for
content classification when combined with the other DCT features;
bump to 4096+ via `AnalyzerConfig::hf_max_blocks` if that single
feature dominates a downstream decision.

Total 8 MP runtime moves 9.40 → 11.38 ms (1.36 ms/MP) under the
accurate default. Non-Tier-3 work is still 1.08 ms/MP — the Tier 3
DCT pass at 1024 blocks accounts for the entire delta.

### Performance — palette flag-array + chunked autoversion

Replaced the 4 KB `[u64; 512]` presence bitset with a 32 KB `[u8; 32_768]`
flag array indexed by the 15-bit RGB-bin id. Each pixel does one
unconditional `flags[idx] = 1` byte store instead of the bitset's
read-modify-write `bits[i>>6] |= 1<<(i&63)` chain. Wrapped the entire
scan-and-count in a single `#[autoversion(v4x, v4, v3, neon, scalar)]`
function — one runtime dispatch per `scan_palette` call covering the
row loop, chunked-by-24-byte index compute, all the byte stores, and
the final `iter().sum()` reduction together.

The chunk pattern is a fixed-size `[u8; 24]` view with 8 unrolled
index computes per chunk. That fixed-size proof + the lack of
inter-iteration dependencies lets the v3/v4 autovectorizer issue the
shifts and ORs as a SIMD batch and emit the 8 byte stores together.
The 32 KB-byte reduction at the end SIMD-sums on the same v4 / v4x
path.

Standalone palette-only bench (4096×2048, 50-iter median):

| variant | photo input | saturated input | ratio |
|---------|-------------|-----------------|-------|
| bitset RMW (scalar) | 6.05 ms | 5.65 ms | 1.00× |
| bitset SIMD index | 6.08 ms | 5.69 ms | 1.00× *(no win — RMW dominates)* |
| flag array (scalar) | 4.12 ms | 4.82 ms | **1.47× / 1.17×** |
| flag array SIMD index | 4.59 ms | 5.20 ms | 1.32× / 1.09× *(slower than scalar)* |

Integrated zenanalyze 8 MP RGB8 (Ryzen 9 7950X, default target-cpu,
runtime archmage dispatch):

- Before this change: 12.13 ms (1.45 ms/MP)
- After flag array, no autoversion: 10.80 ms (1.29 ms/MP)
- After autoversion + chunks-of-24: **9.40 ms (1.12 ms/MP)**

Tier 3 contributes ~0.95 ms; the non-Tier-3 work runs at exactly
1.00 ms/MP. We're now at the pre-stated 1 ms/MP target for all
non-DCT analyzer work, and within ~12 % overall. **`-C target-cpu=
native` is now 10.86 ms — *slower* than the default-target build at
9.40 ms** because autoversion specializes the palette path to v4x
while letting the rest of the binary keep its smaller code-cache
footprint at v3. Confirmed bench, not measurement noise.

Three findings from this round, validated by bench:

- **SIMD index-compute is *slower* than scalar with the flag array.**
  An earlier magetypes `u32x8` shift+OR path that fed 8 stores per
  chunk ran ~10–15 % slower than the plain scalar loop. Once the
  bitset RMW dependency was gone, LLVM's autovec on the scalar path
  was already near-optimal — the deinterleave to arrays + vector
  load + store back round-trip costs more than just letting the
  autovectorizer see the linear pattern.
- **`#[autoversion]` per-row dispatch regressed.** Putting autoversion
  on a per-row helper paid the dispatch cost `height` times and went
  *slower* than plain scalar (10.80 → 12.72 ms at 8 MP). Wrapping the
  whole row+count loop reverses that — one dispatch amortizes over
  the entire scan and inside the v4x boundary the autovec is real.
- **`chunks_exact(3)` is too narrow for LLVM.** Switching to
  `chunks_exact(24)` with a `[u8; 24]` fixed-size view + 8 unrolled
  pixel-index computes was the unlock that let the autovectorizer
  see structure. Without that the bare 3-byte loop body looked too
  small to vectorize even under v4x.

The 32 KB working set is the only real cost: 8× larger than the
bitset, but still fits in L1D on every modern x86-64 / ARM core. One
`Box<[u8; 32_768]>` allocation per call, no per-row heap traffic.

### Performance — Tier 3 DCT + palette SIMD round

Tier 3's `dct2d_8` rewritten as a magetypes f32x8 kernel — three planes
batched per `incant!` call so dispatch cost amortizes 3× and the eight
column-of-D vectors stay hot in YMM registers. Each plane is now
`Y_row[v] = Σ_n splat(X[v][n]) * d_col[n]` (8 fmas per row) for the row
pass and `Z_row[v] = Σ_w splat(D[v][w]) * Y_row[w]` for the column pass —
no transpose, no horizontal reduction, no scalar dot products. ~128 fmas
per plane vs ~1920 scalar ops in the old separable-scalar version.

Palette tier moved from a per-pixel scalar loop to an `incant!`-per-row
magetypes u32x8 kernel that computes eight 15-bit
`((r>>3)<<10)|((g>>3)<<5)|(b>>3)` indices in parallel via lane-wise
shifts and ORs, then does eight scalar bitset ORs (the bitset-OR half
stays scalar — random scatter into a 4 KB table doesn't vectorize).

Measured on a Ryzen 9 7950X (release build, default target-cpu —
runtime archmage dispatch, no `-C target-cpu=native`):

| Image | Pre-DCT/palette | Post | Speedup | ms/MP |
|-------|-----------------|------|---------|-------|
| 1024×1024 | 4.49 ms | 4.45 ms | 1.01× | 4.24 |
| 2048×2048 | 7.93 ms | 7.79 ms | 1.02× | 1.86 |
| 4096×2048 | 12.67 ms | 12.13 ms | 1.04× | **1.45** |

Tier 3 alone dropped from ~1.0 ms to ~0.95 ms at 8 MP — small absolute
movement because Tier 3 was already well-bounded by the 256-block cap.
The DCT cleanup is mostly preparation for raising `hf_max_blocks` in a
future revision; at 1024 blocks the kernel cost would otherwise quadruple.

Palette improvement was modest (~5–8 %) because the bitset OR — eight
random 64-bit scatters per chunk — dominates over the index
computation. The SIMD index pipeline does help, but the OR latency is
the structural floor on this tier.

To reach the 1 ms/MP target, the remaining levers from the prior
analysis still apply: rayon parallelism across tiers, or a smarter
palette algorithm that batches the bitset writes (sort-and-sweep, or a
larger flag array with a final pop-count). With `-C target-cpu=native`
the same 8 MP run hits 1.33 ms/MP; that's the LLVM autovec ceiling
without further algorithmic work.

### Performance — full magetypes SIMD pass

Measured on a Ryzen 9 7950X (release build). Adds explicit
`#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`-driven
SIMD on Tier 1's `accumulate_row` and Laplacian passes; reverts
`hf_max_blocks` back to 256 (1024 didn't fix `patch_fraction`'s
structural sample-sensitivity, just made it ~57 % vs ~100 %); drops
Chao1's redundant 32 KB counter array (raw count IS truth at full
scan).

| Image | Original | Final | Speedup | ms/MP |
|-------|----------|-------|---------|-------|
| 512×512 | 3.31 ms | 2.07 ms | **1.6×** | 7.9 |
| 1024×1024 | 6.82 ms | 4.46 ms | **1.5×** | 4.3 |
| 2048×2048 | 14.1 ms | 7.82 ms | **1.8×** | 1.9 |
| 4096×2048 | 25.3 ms | 12.6 ms | **2.0×** | **1.50** |

**2× faster overall** at 4 MP+ vs the original. ms/MP improves
super-linearly because per-image fixed overheads (alpha scan
allocations, `RowStream` setup) amortize across more pixels.

To reach the 1 ms/MP target, the remaining levers are all bigger:
- **Tier 3 cross-block DCT** (8 blocks per f32x8 batch): est. 3-4 ms
  savings at 4 MP+. Multi-day rewrite.
- **Rayon parallelism** across tiers: est. 2× via multi-core.
  Architectural change.
- **Skip palette full-pass for big images** (use Chao1 + budget at
  >2 MP): saves 4-5 ms but reintroduces palette sampling error.

The full-pass palette tier (~5 ms at 4 MP+) and the bumped block cap
(~3 ms) replace the earlier budget-sampled implementations whose corpus
convergence study showed **never-converging** behaviour for palette
features and 11-16 % p50 error on Tier 3 features. The new pipeline is
slower than the mid-stage 7.93 ms but produces correct outputs:
- exact `distinct_color_bins` and `palette_density` (was 27 % p50 error)
- DCT energy features at 4-6 % p50 error (was 11-16 %)
- usable `patch_fraction` (was 100 % p50 error at 256 blocks)

Tier-by-tier:

- **Tier 2 stride sampling (`pixel_budget` plumbing).** Same
  `compute_stripe_step` pattern Tier 1 uses, now applied to Tier 2.
  Caps Tier 2 at ~0.5 ms (with SIMD) regardless of image size. At
  4096×2048, Tier 2 went from 17.4 ms → 0.48 ms (36×). Pass
  `usize::MAX` as `pixel_budget` to disable.
- **Tier 2 magetypes f32x8 SIMD inner loop.** Replaced the scalar
  every-other-column triplet loop with explicit f32x8 vectorization
  via `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`.
  Processes 8 column triplets per iter, deinterleaves each row once
  into per-channel f32 scratch (10 pixels for row0, 8 each for
  row1/row2), then takes overlapping `f32x8::load` slices for the
  a/b/c triplet positions. Periodic reduce to f64 every 32 iters
  bounds precision loss. **2× over scalar** at every size; 4096×2048
  full scan dropped 16.5 ms → 8.5 ms. Also processes every column
  triplet (was every other) — same data, denser sampling, no extra
  cost.
- **Tier 3 fast DCT.** Replaced the per-iteration `.cos()` calls in
  the naive 8×8 DCT with a precomputed `DCT_COEF[8][8]` constant
  matrix and a clean `dct1d_8` matrix-vector helper. LLVM
  auto-vectorizes the 8-element inner products into AVX2 mul-add
  chains. Tier 3 dropped 9-10× at small images and 1.4× at 4 MP+
  (where the entropy-histogram pass dominates over DCT).

- **Tier 1 magetypes f32x8 block-stats kernel.** The 8×8 block stats
  loop (luma uniformity + per-channel flat-color detection) was fully
  scalar. Replaced with a `#[magetypes(define(f32x8), v4, v3, neon,
  wasm128, scalar)]` kernel that processes one block-row (8 pixels)
  per iter using lane-wise `min` / `max` / `mul_add` accumulators
  reduced to scalar at end-of-block. ~10-15% on Tier 1.

Tier 1's `accumulate_row` (luma/chroma/edges) is already
`#[archmage::autoversion]`-vectorized and dominates the remaining ~3.6
ms. Further wins require switching the seven f64 accumulators to
f32x8 with periodic reductions — non-trivial precision work, filed as
follow-up. The color-bins update (gather-bound histogram) is the
other significant remaining cost; SIMD-ing the index calculation
without a scatter primitive is also follow-up.

### Tests

- 5 new regression tests lock the FIXED behavior:
  `tier2_cb_cr_no_longer_asymmetric_under_cr_sign_flip`,
  `tier2_odd_width_5_includes_right_edge_triplet`,
  `tier2_small_height_no_longer_silently_zeroes`,
  `tier2_bottom_edge_partial_fragment_is_counted`,
  `tier3_zigzag_split_is_symmetric_in_horiz_vs_vert_detail`.
  Any regression that re-introduces one of the four bugs fails the
  matching test.
- Tier 1: stripe-sampled variance, edge density, chroma complexity,
  uniformity, flat-color blocks, 5-bit palette bins. `archmage::autoversion`
  on the inner row scan (v3 / NEON / WASM128 / scalar).
- Tier 2: per-channel per-axis chroma sharpness (`cb_horiz`/`cb_vert`/`cb_peak`
  and `cr_*`). Three-row sliding-window port of evalchroma 1.0.3
  `image_sharpness` with fragment-based normalization.
- Tier 3: 32-bin luma histogram entropy, naive 8×8 DCT high-freq energy ratio
  (capped at 256 sampled blocks), derived text/screen/natural likelihoods.
- `RowStream`: pulls RGB8 rows on demand from any `zenpixels::PixelSlice`,
  zero-copy on RGB8/RGB8_SRGB inputs and one-row scratch on every other
  format via `zenpixels-convert::RowConverter`.
