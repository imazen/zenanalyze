# zenanalyze [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenanalyze/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenanalyze?style=flat-square)](https://crates.io/crates/zenanalyze) [![lib.rs](https://img.shields.io/crates/v/zenanalyze?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenanalyze) [![docs.rs](https://img.shields.io/docsrs/zenanalyze?style=flat-square)](https://docs.rs/zenanalyze) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

Streaming image content analyzer for adaptive codec pipelines. One pass over a
`zenpixels::PixelSlice` extracts the numeric features that decision trees,
selectors, and per-image encoder configurators consume — variance, edge density,
chroma sharpness, palette population, DCT energy, alpha statistics, AQ-map and
noise-floor estimates, and (behind the opt-in `hdr` cargo feature) source-direct
HDR / wide-gamut / bit-depth signals that codecs use to **detect when a
descriptor over-promises and the actual pixel content is encodable in something
smaller**.

```toml
[dependencies]
zenanalyze = "0.2.0"
# Naming the source PixelDescriptor / ColorPrimaries off a result needs zenpixels too:
zenpixels = { version = "0.2.14", default-features = false }
# Opt in to the still-settling XYB color-loss + deprecated palette-density signals:
# zenanalyze = { version = "0.2.0", features = ["experimental"] }
# Opt in to source-direct HDR / wide-gamut / bit-depth signals (the depth tier):
# zenanalyze = { version = "0.2.0", features = ["experimental", "hdr"] }
```

> **Versioning:** zenanalyze is on the **0.2.x** line (standard 0.x semver —
> breaking changes bump the minor `0.2 → 0.3`, additive changes bump the patch
> `0.2.0 → 0.2.1`). The Rust library surface in this source tree is `0.2.0`.
> The convenience entries `analyze_features_rgb8` and its fallible parallel
> `try_analyze_features_rgb8` both exist on this surface. (An earlier "0.1.x
> forever, additive-only" policy was retired; if you still see that claim
> anywhere, it is stale — trust the version in `Cargo.toml`.)

## Cargo features

| feature | what it gates | count | stability |
|---|---|---|---|
| _(default)_ | The full mature surface: luma stats, edges, chroma sharpness, DCT energy, alpha, palette, distinct-color bins, AQ-map / noise-floor / quant-survival / Laplacian-variance families, gradient & patch fractions, grayscale & skin-tone scores, geometry, and the HVS/spectral pack. | **97** | Numeric drift bounded by the threshold contract; signatures semver-governed |
| `experimental` | Two still-settling definitions: the XYB color-loss pair (`Xyb444ColorLoss`, `XybBquarterChromaLoss`) plus the deprecated `PaletteDensity`. | +3 → 100 | Metric definition or scale may still change; opt in only if you re-validate per patch |
| `hdr` | Source-direct HDR / wide-gamut / bit-depth signals + the clip-and-separate `highlight_*` descriptors — 16 features, the depth tier (ids 32–39, 46, 47, 212–217). | +16 → **116** | Off by default (SDR hot path skips the tier); definitions may change per patch |

> **As of the 0.2.x line, `experimental` is narrow.** ~58 features that used to
> sit behind it (the `AqMap*`, `NoiseFloor*`, `QuantSurvival*`,
> `LaplacianVariance*` families, `GradientFraction`, `PatchFraction`,
> `GrayscaleScore`, `SkinToneFraction`, `EdgeSlopeStdev`, the HVS/spectral pack,
> etc.) were **promoted to the default surface** once their definitions pinned —
> they need no feature flag now. The gate now scopes to only the three signals
> above, whose structural definition is still settling.
>
> **Retired likelihoods:** the four composite likelihoods `TextLikelihood`,
> `ScreenContentLikelihood`, `NaturalLikelihood`, `LineArtScore` were removed;
> their ids (27 / 28 / 29 / 45) stay reserved and are never re-used, so the wire
> format (`AnalysisFeature::id`) is stable. The "Empirical operating thresholds"
> section below still references them as historical calibration data — those
> rows describe a prior surface and no longer map to live variants.

## Why

Modern adaptive codecs (zenjpeg, zenwebp, zenpng, zenavif, zenjxl) all want the
same handful of cheap content features to pick a quality knob — *is this a
photograph or a screenshot? does it have alpha? is it palette-friendly? is the
HDR flag stale? does this Rec.2020 file actually use the wider gamut?* — before
they run their expensive encoder. Re-deriving that from scratch in each codec
means three copies of the same Tier 1 SIMD scan, three slightly different
threshold contracts, and three independent oracle retrains every time the math
moves.

zenanalyze is the shared single-pass scanner: codecs ask for the feature set
they care about, the orchestrator unions the requests, and one walk over the
image returns every signal. Tiers gate themselves out when their outputs aren't
needed.

## Quick start

```rust
use zenanalyze::{
    analyze_features,
    feature::{AnalysisFeature, AnalysisQuery, FeatureSet},
};

const JPEG_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::EdgeDensity)
    .with(AnalysisFeature::HighFreqEnergyRatio);

const WEBP_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::Variance)
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction);

let needed = JPEG_FEATURES.union(WEBP_FEATURES);
let results = analyze_features(slice, &AnalysisQuery::new(needed))?;

let variance = results.get_f32(AnalysisFeature::Variance);
let alpha    = results.get(AnalysisFeature::AlphaPresent)
                      .and_then(|v| v.as_bool());
```

`AnalysisFeature` is `#[non_exhaustive]` with stable `u16` discriminants —
retired ids stay reserved (the id sequence has gaps where features were removed),
new features get fresh ids, and an id is never re-used, so the `id()`-keyed wire
format stays stable across versions. `FeatureSet` has full `const fn` set math
(`union`, `intersect`, `difference`) so per-codec presets compose at compile
time. `AnalysisQuery` is intentionally opaque: sampling budgets are crate
invariants, not per-call knobs.

For a packed RGB8 buffer the convenience entry skips the `PixelSlice` ceremony:

```rust
// Panicking — for known-good inputs (freshly decoded buffers).
let r = zenanalyze::analyze_features_rgb8(&rgb_bytes, w, h, &q);

// Fallible — for untrusted input. Returns AnalyzeError::InvalidInput on
// length / stride mismatch and AnalyzeError::OutOfMemory on (future)
// fallible-allocation paths.
let r = zenanalyze::try_analyze_features_rgb8(&rgb_bytes, w, h, &q)?;
```

Codecs read the source `PixelDescriptor` directly off the result for encode-side
metadata decisions:

```rust
let descriptor = results.source_descriptor();
match descriptor.primaries {
    zenpixels::ColorPrimaries::Bt709     => /* sRGB-class */,
    zenpixels::ColorPrimaries::DisplayP3 => /* P3 wide-gamut */,
    zenpixels::ColorPrimaries::Bt2020    => /* Rec.2020 / HDR */,
    zenpixels::ColorPrimaries::AdobeRgb  => /* AdobeRGB */,
    _ => /* unknown / future */,
}
```

`FeatureSet::iter()` walks the contained features in `AnalysisFeature::id()`
order — convenient for sidecars, harnesses, and Python fitters that need to
enumerate the surface without hand-listing variants.

## The picker feature vector

A codec picker (decision tree, MLP, GBDT) consumes a **fixed-shape feature
vector** — a known count of features in a known column order. zenanalyze's job
is to produce exactly that vector. The canonical "all features" set is the
`FeatureSet::SUPPORTED` const, and the column order is `AnalysisFeature::id()`
ascending (the same order `FeatureSet::iter()` and `AnalysisResults::pack()`
emit). Recipe:

```rust
use zenanalyze::{analyze_features, feature::{AnalysisFeature, AnalysisQuery, FeatureSet}};

// SUPPORTED is the full set THIS BUILD can compute. Always public — it just
// shrinks if you disable a cargo feature. `experimental` is ON by default, so
// the default build has 101 features (`--no-default-features` drops to 97);
// add `hdr` for 117 (113 without `experimental`).
let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
let results = analyze_features(slice, &q)?;

// `pack()` is the canonical emit: id-sorted `(u16 stable_id, f32 value)` pairs.
// This IS the training/inference vector — column i is the i-th feature in
// id() order, value already coerced to f32. Integral/bool features round-trip
// losslessly (Bool(true) -> 1.0, U32(n) -> n as f32).
let vector: Vec<(u16, f32)> = results.pack();

// To name the columns (for a sidecar header, a fitter, a schema):
let column_order: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
//   column_order[i].id()  is the stable id; .name() is the snake_case string.
```

### Low-coupling picker boundary

Use `analyze_features` / `AnalysisResults::pack()` when the caller is already a
zenanalyze-aware tool, trainer, or test harness. Use the `feature_*` facade when
the caller is a codec crate that wants to ship a baked `zenpredict` model without
leaking zenanalyze's typed API through its own public surface.

That facade keeps the cross-crate contract to primitive schema data:

- `u16` feature ids define the model input columns.
- `f32` values fill the input vector.
- Unknown, retired, cfg-disabled, or image-undefined ids become `f32::NAN`.
- The codec still owns its model `schema_hash` check; `feature_count()` is only a
  sizing helper for the current build.

```rust
// Stored next to the baked model, or derived from model metadata at load time.
const MODEL_FEATURE_IDS: &[u16] = &[0, 1, 2, 19, 20];

let mut input = vec![0.0f32; MODEL_FEATURE_IDS.len()];
if !zenanalyze::feature_vector(slice, MODEL_FEATURE_IDS, &mut input) {
    // invalid pixel slice / conversion failure / output buffer too small
    return Err(MyPickerError::FeatureExtraction);
}

let decision = predictor.predict(&input)?;
```

For the common packed-8-bit case, `feature_vector_packed8` avoids naming
`zenpixels::PixelSlice` at the codec picker boundary:

```rust
let mut input = vec![0.0f32; MODEL_FEATURE_IDS.len()];
let ok = zenanalyze::feature_vector_packed8(
    rgb_or_rgba_bytes,
    width,
    height,
    row_stride_bytes, // 0 means tightly packed
    channels,         // 1 gray, 3 RGB, 4 RGBA
    MODEL_FEATURE_IDS,
    &mut input,
);
```

For higher bit depth, non-sRGB transfer functions, wide-gamut descriptors, or
HDR feature ids, build a `zenpixels::PixelSlice` and call `feature_vector`. The
typed zenanalyze API remains available, but codec crates do not need to re-export
`AnalysisFeature`, `FeatureSet`, `AnalysisQuery`, or `AnalysisResults` just to
run inference.

The emitted order is **stable across versions**: ids follow a
retired-keeps-its-slot rule (a removed feature's id is never re-used, new
features get fresh ids), so a vector packed by one zenanalyze version is
re-readable by another via `AnalysisResults::from_packed`, and
`AnalysisResults::require(set)` asserts a fixed input set is present (returns
the missing ids rather than silently zero-filling). **Pin to a specific patch
when you compile-in a fitted model** — feature *values* drift within a minor
per the threshold contract, even though the id order doesn't.

Because `SUPPORTED` membership depends on enabled cargo features, a model
trained against the `experimental + hdr` 116-feature surface must be consumed by
a build with the same features on. If you want a feature-flag-independent vector,
request an explicit named set instead of `SUPPORTED`.

**Per-codec subsets are cheaper.** A picker rarely needs all 97. Request only
the features the model uses and the analyzer skips whole passes. The crate ships
one such const — `FeatureSet::ZENJPEG_PICKER_V1_1` (8 features: `Variance`,
`EdgeDensity`, `Uniformity`, `ChromaComplexity`, `CbSharpness`, `CrSharpness`,
`HighFreqEnergyRatio`, `LumaHistogramEntropy`, ids `[0,1,2,3,4,5,19,20]`), which
lets the analyzer skip the Tier 2, Palette, and Alpha passes entirely.

### Reading individual feature values

| accessor | behavior |
|---|---|
| `results.get(f)` | `Option<FeatureValue>` — `None` if `f` wasn't requested or its computation failed. The typed enum (`F32` / `U32` / `U64` / `Bool`). |
| `results.get_f32(f)` | `Option<f32>` — **coerces** any type: `Bool(false) → 0.0`, `Bool(true) → 1.0`, `U32(n) → n as f32`. Never panics, never returns `NaN` for a present integral/bool value. `None` only when absent. This is what you want when building a flat vector. |
| `FeatureValue::as_f32()` / `as_u32()` / `as_bool()` / `as_u64()` | strict typed access — `Some` **only** if the value is that exact variant, else `None`. Use when you know the underlying type and want a type mismatch to surface. |

So for the non-f32 default features there is no guesswork: `get_f32(DistinctColorBins)`
returns the count as an `f32` (e.g. `Some(412.0)`), and `get_f32(AlphaPresent)`
returns `Some(0.0)` / `Some(1.0)`. For their native types use
`get(DistinctColorBins).and_then(FeatureValue::as_u32)` (a `u32`) and
`get(AlphaPresent).and_then(FeatureValue::as_bool)` (a `bool`).

## What it computes

The default surface (`experimental` on by default) is 101 features
(97 with `--no-default-features`; 117 with `hdr` added). The table below names
the most commonly consumed ones with their **real `AnalysisFeature` variant
identifiers** (no globs — every name is a literal variant). Enumerate the
complete set in id order with `FeatureSet::SUPPORTED.iter()`.

| Feature(s) | Type | Description |
|---|---|---|
| `Variance` | f32 | Luma variance on the BT.601 [0, 255] scale. |
| `EdgeDensity` | f32 | Fraction of sampled interior pixels with `\|∇L\| > 20`. |
| `ChromaComplexity` | f32 | `√(Var(Cb) + Var(Cr))` over sampled pixels. |
| `CbSharpness`, `CrSharpness` | f32 | Mean `\|∇Cb\|` / `\|∇Cr\|` over horizontally-paired sampled pixels. |
| `CbHorizSharpness`, `CbVertSharpness`, `CbPeakSharpness` | f32 | Per-axis Cb chroma sharpness (horizontal / vertical / peak). |
| `CrHorizSharpness`, `CrVertSharpness`, `CrPeakSharpness` | f32 | Per-axis Cr chroma sharpness (horizontal / vertical / peak). |
| `Uniformity` | f32 | Fraction of 8×8 blocks with luma variance < 25. |
| `FlatColorBlockRatio` | f32 | Fraction of 8×8 blocks with R/G/B ranges all ≤ 4. |
| `DistinctColorBins` | u32 | Distinct 5-bit-per-channel RGB bins observed. |
| `PaletteFitsIn256`, `PaletteLog2Size` | bool, u32 | Palette-size signals (≤256 colours; log2 of the distinct count). |
| `HighFreqEnergyRatio` | f32 | DCT AC energy ratio over sampled 8×8 luma blocks. |
| `LumaHistogramEntropy` | f32 | Shannon entropy of a 32-bin luma histogram (bits). |
| `GrayscaleScore` | f32 | Fraction of pixels with R≈G≈B (grayscale gap-filler; near-binary on true grayscale). |
| `AlphaPresent`, `AlphaUsedFraction`, `AlphaBimodalScore` | bool, f32, f32 | Straight-alpha statistics. |
| `AqMapMean`, `AqMapStd`, `AqMapP1/P5/P10/P50/P75/P90/P95/P99` | f32 | Adaptive-quant map statistics + percentiles. |
| `NoiseFloorY`, `NoiseFloorUV` (+ Y/Uv percentile families) | f32 | Per-channel noise-floor estimates. |
| `GradientFraction`, `GradientFractionSmooth`, `PatchFraction`, `PatchFractionFast` | f32 | Smooth-region / flat-patch fractions (large-DCT & screen-vs-photo signals). |
| `SkinToneFraction`, `EdgeSlopeStdev` | f32 | Photo-vs-screen dispatch signals. |
| `IsGrayscale` | bool | Hard grayscale flag. |
| `PixelCount`, `MinDim`, `MaxDim`, `ChannelCount` | u32 | Geometry. `LogPixels`, `AspectMinOverMax`, … (f32) round these out. |

(The full id table — including the `QuantSurvival*`, `LaplacianVariance*`,
`LogPaddedPixels*`, `ChromaLumaCovariance*`, `InfoWeight*`, `SpectralSlopeY`,
`OrientationEnergyRatio`, `BlockMisalignment*` families — is enumerable from
`FeatureSet::SUPPORTED`; each variant carries a one-line docstring on
`AnalysisFeature`.)

**Behind the `experimental` cargo feature** are only `Xyb444ColorLoss` /
`XybBquarterChromaLoss` (XYB-vs-YCbCr and XYB chroma-subsampling discriminants)
and the deprecated `PaletteDensity`. The rest of this section's
formerly-experimental families are on the **default** surface now.

The signals below are grouped by what they drive (all on the default surface
unless noted):

### Codec-orchestrator gap-fillers

| Feature | Drives |
|---|---|
| `GrayscaleScore` | zenjpeg `ColorMode::Grayscale`, AVIF `Yuv400`, png/jxl gray paths (~30–40% smaller for B&W) |
| `AqMapMean` / `AqMapStd` | zenjpeg hybrid trellis λ, webp segments + sns_strength, avif vaq |
| `NoiseFloorY` / `NoiseFloorUV` | zenjpeg `pre_blur`, jxl `noise/denoise`, webp `sns_strength`, zenrav1e `film_grain` |
| `GradientFraction` | jxl `with_force_strategy` (DCT16 / DCT32 selection), zenrav1e deblock strength |
| `SkinToneFraction` | photo-vs-other dispatch (one-direction signal, AUC 0.80) — webp `Preset::Photo`, jxl perceptual presets, jpeg chroma-aware quant |
| `EdgeSlopeStdev` | screen-vs-photo dispatch (AUC 0.84, second only to `PatchFraction`) — webp `Preset::Drawing` vs `Photo`, jxl modular vs VarDCT |

### Source-direct HDR / wide-gamut / bit-depth tier (`hdr` feature)

These 16 features are gated behind the `hdr` cargo feature (ids 32–39, 46, 47,
212–217). They read source samples without going through `RowConverter`, since
`RowConverter` doesn't tonemap — a 4000-nit PQ source and a 100-nit-clipped
SDR source would otherwise produce byte-identical RGB8 streams.

| Feature | Drives |
|---|---|
| `PeakLuminanceNits` / `P99LuminanceNits` | AVIF `clli`, JXL `intensity_target`, HDR encoder peak |
| `HdrHeadroomStops` / `HdrPixelFraction` | HDR vs SDR encode-mode selection |
| `WideGamutPeak` / `WideGamutFraction` | "Linear value > 1.0" detection |
| `GamutCoverageSrgb` / `GamutCoverageP3` | **Descriptor-gap signal** — if a Rec.2020 source's pixels all live in the sRGB sub-gamut, encode at sRGB primaries and save the wide-gamut metadata + encoder modes |
| `EffectiveBitDepth` | AVIF / JXL `bit_depth`, png `near_lossless_bits` (catches u8-promoted u16) |
| `HdrPresent` | Composite "transfer claims HDR AND pixels are actually bright" — catches stale HDR flags |
| `HighlightLumaMean` / `_Std`, `HighlightChromaMean` / `_Std`, `HighlightEdgeCount`, `HighlightOrientationRatio` | **Clip-and-separate** — the bounded HDR signal a picker reads alongside `with_diffuse_white_clip` content features (recovers the highlight extension at R²≈0.95) |

### Still-settling (`experimental` feature)

Only three signals remain gated behind `experimental` because their structural
definition is still being refined: `Xyb444ColorLoss` and `XybBquarterChromaLoss`
(the XYB-vs-YCbCr and XYB chroma-subsampling discriminants for the zenjpeg colour
picker) and `PaletteDensity` (deprecated, being retired). Numeric scale or
definition of these may change between 0.2.x patches; opt in only if you
re-validate. Everything else that used to live here — `Colourfulness`,
`LaplacianVariance` (+ percentiles), `VarianceSpread`, `DctCompressibilityY/UV`,
`PatchFraction`, `PaletteLog2Size`, `PaletteFitsIn256` — is now on the default
surface.

## Descriptor-gap detection

The analyzer's job is to spot the gap between what the descriptor *promises*
and what the data *actually carries*, so encoders don't bloat files paying
for capacity the source doesn't need.

| Gap | Signal |
|---|---|
| RGB declared, content is grayscale | `GrayscaleScore ≥ 0.99` |
| Wider primaries declared, content fits sRGB | `GamutCoverageSrgb ≥ 0.99` |
| Rec.2020 declared, content fits Display P3 | `GamutCoverageP3 ≥ 0.99` |
| HDR transfer declared, content is SDR | `HdrPresent == false` |
| u16 declared, content is u8-promoted | `EffectiveBitDepth == 8` |
| RGBA declared, alpha is constant 1.0 | `AlphaUsedFraction == 0` |
| Standard 8×8 transforms, content is smooth | `GradientFraction ≥ 0.5` (use larger DCTs) |
| HDR flag set, peak is dim | `PeakLuminanceNits < 200 && HdrPresent` (mis-tagged HDR) |

Each one is a place where a codec orchestrator can downcast metadata + encoder
modes before encoding, saving real bytes on real corpora.

## Wide gamut, HDR, and bit depth

`analyze_features` accepts every layout `zenpixels-convert::RowConverter` can
ingest — RGB8 / RGBA8 / BGRA8, RGB16 / RGBA16, RGB-F32 / RGBA-F32 (linear, sRGB,
PQ, HLG), grayscale variants, all primaries (sRGB / Display P3 / Rec.2020 /
AdobeRGB). One entry, no opt-in step. The principle: per-image codec decisions
don't usually break on a few LSBs of luma drift, they break on the analyzer
refusing to run.

**u8-promotion invariance is locked by tests.** An RGB8 image promoted to
RGB16 via the standard `u8 * 257` doubling, or to RGBF32 via `u8 / 255.0`,
produces *bit-identical* features to the original RGB8 source. Codecs that
upgrade from u8 to wider formats internally don't see different analyzer
answers. (Verified by garb's exact-identity narrowing
`(u16 * 255 + 32768) >> 16` for `u16 = u8 * 257`.)

**Wide gamut adapts the values, not the API.** RGB8 with Display P3 / Rec.2020
/ AdobeRGB primaries passes through the zero-copy `Native` row path with its
bytes intact. The standard tiers pick the **right luma matrix per source
primaries**: BT.601 weights for sRGB / BT.709 (preserving the trained-threshold
baseline — coefficient's existing thresholds were calibrated against this
matrix on sRGB content), BT.2020 weights for Rec.2020 sources, the Y row of
each primary set's RGB→XYZ matrix for Display P3 / AdobeRGB. Fixed-point
integer-luma scales are normalised to the same sum-220 libwebp baseline so a
pure-white pixel hits the same histogram bin regardless of source primaries —
what differs is the per-channel weight that lands it there. No conversion, no
clipping, just the right matrix. See `src/luma.rs`.

**HDR f32 / linear inputs.** Standard tiers see what an SDR display would
show — `RowConverter` clips out-of-[0, 1] linear values, applies the sRGB
OETF, and narrows to u8. That's the legitimate input for SDR-calibrated
thresholds; tonemapping a 4 000-nit highlight into a visible mid-tone
before measuring "high-frequency-energy ratio" would just lie about what's
there. The above-clip signal lives in `tier_depth`, which reads the source
samples directly via `PixelSlice::row` (bypassing `RowConverter` entirely)
and decodes through the descriptor's transfer function — sRGB / BT.709 /
Gamma 2.2 / Linear / PQ / HLG — to linear nits. Two views of the same
source: the SDR-display view for trained thresholds, the source-direct view
for HDR / wide-gamut signal.

**Opt-in linear-light content analysis (`with_linear_light`).** Those two
views are the *default*. For HDR-correct *content* features — not just the
depth-tier signals — set `AnalysisQuery::with_linear_light(true)`. The row
stream decodes to linear, applies the diffuse-white exposure **anchor** (a
×scale, not a tonemap), re-encodes through the sRGB OETF, and feeds the
content tiers in f32. Below diffuse white an HDR scene then scores like the
equivalent SDR scene — the displayable range is the same perceptual sRGB the
default path produces — and super-white above the anchor survives past the u8
ceiling instead of clipping to a flat plateau, so the content tiers measure
the real envelope. The default gamma narrowing stays the zero-copy fast path
for SDR-calibrated thresholds; reach for linear-light when the depth tier
flags HDR and you want the content features to follow it into the highlights.

**Clip-and-separate, for a downstream picker (`with_diffuse_white_clip`).**
Letting the content features *extend* with super-white entangles the HDR signal
into every feature — and an A/B measurement showed that's *provably lossy* for the
high-frequency chroma-sharpness family (no regime scalar can reconstruct it) and
forces a model to learn a feature×headroom surface from scarce HDR data. The
better-conditioned representation: pair `with_linear_light(true)` with
`with_diffuse_white_clip(true)` so the content tiers clamp at diffuse white and
stay **SDR-invariant**, then read the HDR signal from six bounded, additive
`highlight_*` depth descriptors (`highlight_luma_mean`/`_std`,
`highlight_chroma_mean`/`_std`, `highlight_edge_count`,
`highlight_orientation_ratio`) — which recover the extension at median R²≈0.95.
A plain model then works without a learned per-regime modulation. Extend stays the
default (zero migration); clip-and-separate is the opt-in picker-facing form.

**Normalizing for a model (`recommended_transform`).** Each feature carries its
structural pre-standardization transform via
`AnalysisFeature::recommended_transform() -> TransformHint` (`as_str` matches the
common `FEATURE_TRANSFORMS` vocabulary): dimensions → `log`, the heavy-tailed
variance / laplacian / edge-slope family → `log1p`, chroma-luma covariances →
`signed_cbrt`, everything else `identity`. Apply it before z-scoring so
heavy-tailed features don't dominate the gradient; override per-corpus if your own
ablation finds better.

The `tier_depth` reference convention is stable across the 0.2.x line:

| Transfer | Linear 1.0 maps to | Convention |
|---|---|---|
| `Srgb` / `Bt709` / `Gamma22` / `Linear` | 80 nits | sRGB display reference (IEC 61966-2-1) |
| `Pq` | 10 000 nits | SMPTE ST 2084 absolute |
| `Hlg` | 1 000 nits | nominal HLG broadcast |

The standard tiers' threshold contract is calibrated on display-space RGB8
bytes; the depth tier surfaces the additional metadata-gap and HDR signals
that the RGB8 narrowing destroys.

## Errors

```rust
#[non_exhaustive]
pub enum AnalyzeError {
    Convert(String),                                  // RowConverter setup failed
    InvalidInput(String),                             // user-supplied bad layout / length
    OutOfMemory { bytes_requested: Option<usize> },   // future fallible-alloc path
    Internal(String),                                 // unexpected
}
```

Production code handling untrusted images should pattern-match on
`InvalidInput` / `OutOfMemory` explicitly. Today every internal allocation is
infallible (so `OutOfMemory` is reserved, never returned by current builds);
the variant is part of the public surface so a future minor that flips
internals to `Vec::try_reserve` doesn't break anyone's `match`.

## How it's organised

Five passes, each gated by what the requested `FeatureSet` actually needs:

| Pass | Iterates over | Reads | Cost (4 MP) | Drives |
|---|---|---|---|---|
| Tier 1 | Stripe-sampled rows | RGB8 | ~1 ms | luma stats, edges, chroma, uniformity, grayscale |
| Tier 2 | 3-row sliding window | RGB8 | ~2 ms | per-axis Cb/Cr sharpness |
| Tier 3 | Sampled 8×8 DCT blocks | RGB8 | ~3 ms | DCT energy, entropy, AQ map, noise floor, line-art, gradient, patch fraction |
| Palette | Full image | RGB8 | ~1 ms | distinct color bins |
| Alpha | Stride-sampled rows | **Source bytes** | ~0.3 ms | alpha presence / used / bimodal |
| `tier_depth` (experimental) | Stride-sampled rows | **Source bytes** | ~0.5 ms HDR, ~0 SDR-fast-path | HDR / wide-gamut / bit-depth / gamut-coverage |

Tier 1/2/3 + Palette read RGB8 via `RowStream`, which has three internal paths:

- **Native** (zero-copy) — RGB8-byte-layout-compatible inputs. Sub-slice straight from the source.
- **StripAlpha8** (zero RowConverter, scratch-only) — RGBA8 / BGRA8 / Rgbx8 / Bgrx8. Tight strip-and-maybe-swap into the row scratch. Skips the RowConverter alloc + plan + per-row CPU work.
- **Convert** — everything else (16-bit, f32, grayscale, CMYK, …) goes through `RowConverter` row-by-row.

Alpha and `tier_depth` always read source bytes directly, never through
`RowStream` — a load-bearing detail for HDR (RowConverter doesn't tonemap;
its narrowing clips PQ / HLG into sRGB-display).

## Performance

Release build, AVX2, no `target-cpu=native`, full `FeatureSet::SUPPORTED`:

| Input | 4 MP | RowStream path |
|---|---|---|
| RGB8 / Rgbx8 with sRGB / wide-gamut primaries | 9.5 ms | `Native` (zero-copy slice subindex) |
| RGBA8 | 10.9 ms | `StripAlpha8` (garb SIMD strip) |
| BGRA8 | 12.0 ms | `StripAlpha8` (garb SIMD strip + swap) |
| RGB16 | 24.7 ms | `Convert` (zenpixels-convert RowConverter) |
| RGBA16 | 28.6 ms | `Convert` (zenpixels-convert handles strip + narrow) |

The RGBA8 strip uses `garb::bytes::rgba_to_rgb` / `bgra_to_rgb`
(SIMD-dispatched via `archmage::incant!`) — measured 7× faster than
the previous in-tree scalar strip on a 2048-px row, dropping the
RGBA8 overhead vs RGB8 baseline from +5.8 ms to +1.4 ms. The 16-bit
input paths cost more because RowConverter does transfer-function-
aware narrowing — that's the correct tool for genuinely
heterogeneous input.

Per-call working-set memory is ~265 KB across ~7 allocations (largest single
chunk is the Tier 1 stripe scratch at 9 × width × 3 = 108 KB at 4 K). All
allocations are infallible today; the `OutOfMemory` variant exists so a
future minor can flip them without API breakage.

## Empirical operating thresholds

Picked on a 219-image labeled corpus from coefficient
(`benchmarks/classifier-eval/labels.tsv`, spanning cid22-train/val,
clic2025-1024, gb82, gb82-sc, imageflow, kadid10k, qoi-benchmark). 174
photo, 36 screen, 9 illustration, 44 marked synthetic. F1 / AUC are
for binary screen-vs-photo classification.

**For codec-orchestrator dispatch ("is this a screen or a photo?"):**

| signal | threshold | F1 | AUC | notes |
|---|---|---|---|---|
| `line_art_score > 0` | any nonzero | 0.978 | 0.750 | near-deterministic — line art ⇒ screen-like |
| `natural_likelihood >= 0.06` | photo detection | 0.924 | 0.814 | high precision photo classifier |
| `patch_fraction >= 0.27` | screen detection | **0.769** | **0.880** | **strongest single screen discriminator** |
| `edge_slope_stdev >= 35` | screen detection | — | **0.844** | **second-strongest screen discriminator** — photos cluster 15–32, screens 32–58 |
| `screen_content_likelihood >= 0.60` | screen detection | 0.750 | 0.831 | derived from flat blocks + palette + chroma |
| `flat_color_block_ratio >= 0.53` | screen detection | 0.750 | 0.838 | raw — same F1 as the derived `_likelihood` |
| `skin_tone_fraction >= 0.05` | photo detection | 0.824 | 0.799 | one-direction (presence ⇒ photo); pigmentation-invariant Chai-Ngan YCbCr |
| `text_likelihood >= 0.30` | text detection | 0.682 | 0.774 | weaker but real |
| `grayscale_score >= 0.99` | grayscale dispatch | — | — | encoder gap-filler, near-binary on real grayscale |

**Note:** the three `*_likelihood` features empirically saturate at
~0.70 (not 1.0) on real content, because each is a weighted sum of
clamped sub-components that don't simultaneously max on real images.
**Don't threshold them at `>= 0.8` — nothing will fire.** Operating
points are in the 0.3–0.6 band. The exact corpus maxes are:

- `text_likelihood` max **0.71**
- `screen_content_likelihood` max **0.70**
- `natural_likelihood` max **0.69**

**For descriptor-gap detection** the thresholds are content-physical
(see the "Descriptor-gap detection" table above): `GrayscaleScore >= 0.99`,
`GamutCoverageSrgb >= 0.99`, etc. Those are spec-driven, not corpus-fit.

The full per-class distributions, ROC-AUC ranking for every feature,
Spearman redundancy matrix, and the recalibration findings that were
considered and rejected are recorded in
[`docs/calibration-corpus-2026-04-27.md`](https://github.com/imazen/zenanalyze/blob/main/docs/calibration-corpus-2026-04-27.md).
That file is the original pre-ship empirical baseline (it predates several
features now on the default surface, and the four composite `*_likelihood`
signals referenced in the table above were since retired); patches that drift
numerics should compare against it.

## Threshold contract

Numeric thresholds and normalisation scales drift between patch releases.
Downstream consumers that compile-in fitted models (oracle decision trees,
content selectors, MLPs) must pin to a specific zenanalyze patch version and
re-validate when they bump it.

**Versioning is standard 0.x semver on the 0.2.x line.** A *breaking* change to
the library API (renaming/removing an item, changing a signature) bumps the
minor (`0.2 → 0.3`); *additive* changes (new `AnalysisFeature` variants — the
enum is `#[non_exhaustive]` — new functions, new consts) bump the patch (`0.2.0
→ 0.2.1`); numeric/behavioural drift is allowed within a minor and is governed
by this threshold contract. The wire format is independent of the semver
version: `AnalysisFeature::id()` follows a retired-keeps-its-slot rule (ids are
never re-used), so `pack()`/`from_packed` output round-trips across versions
regardless of minor bumps.

## Test surface

130+ tests covering math invariants on synthetic inputs (solid colours,
horizontal bands, uniform luma distribution, palette-locked images, two-tone
line drawings, smooth gradients, pure noise), the full 16-arm dispatch matrix,
every supported pixel format (3 channel-types × 6 transfers × 4 primaries × 2
alpha = 144 sanity-matrix combinations), tier sizes from 1×1 to 4096×4096,
deterministic-input bit-equality (catches accumulator non-determinism),
u8-promotion bit-equality across u16 / f32 sources, HDR-survival (PQ ~1000-nit
content preserved end-to-end where standard tiers would have clipped to SDR),
gamut-coverage projections (saturated Rec.2020 green correctly fails sRGB
coverage), and `AnalyzeError` Display / source coverage. Math locks use
absolute tolerances chosen to clear ULP-level f32 noise from SIMD tree
reductions but catch any genuine architecture divergence.

> Note on coverage tooling: the SIMD kernels in `tier1.rs`, `palette.rs`,
> `tier2_chroma.rs`, and `tier3.rs` use `#[magetypes(... v4, v3, neon,
> wasm128, scalar)]` to generate one source-level monomorphisation per
> architecture tier. At runtime archmage's `incant!` dispatches to whichever
> the CPU supports; the other variants stay compiled but unreachable. Line-
> coverage tools count each variant separately, so the raw percentage on
> these files looks ≈30 % on x86_64. Real coverage of executable code paths
> (counted on the dispatched variant only) is ≥95 % across every module.

## Companion crates in this repo

The repository hosts three sibling pieces that the codecs in
[`imazen/zenjpeg`](https://github.com/imazen/zenjpeg),
[`imazen/zenwebp`](https://github.com/imazen/zenwebp), etc. compose with
zenanalyze:

| Path | Identity | Status |
|---|---|---|
| [`zenpredict/`](https://github.com/imazen/zenanalyze/tree/main/zenpredict) | **Rust runtime** — zero-copy MLP loader (ZNPR v3 binary format), forward pass, masked argmin, typed metadata, feature-space OOD bounds (output-space OOD + two-shot rescue behind the `advanced` feature). Used by codec pickers (`zenjpeg`/`zenwebp`/`zenavif`/`zenjxl`) and by `zensim` V0.4 perceptual scoring | Crate, `0.2.x` |
| [`zenpicker/`](https://github.com/imazen/zenanalyze/tree/main/zenpicker) | **Codec-family meta-picker** — given features + target quality + an allowed-family mask, picks `{jpeg, webp, jxl, avif, png, gif}`; per-codec pickers then resolve the family into a concrete encoder config. Wraps `zenpredict::Predictor` | Crate, `0.1.x` |
| [`zentrain/`](https://github.com/imazen/zenanalyze/tree/main/zentrain) | **Python training pipeline** — pareto sweep harness, teacher fit, distill, ablation, holdout probes, safety reports, bake to ZNPR v3 (via `tools/bake_picker.py` → `zenpredict-bake`). Produces both meta-picker and per-codec bakes | Tooling, in-repo |

The runtime crates (`zenpredict`, `zenpicker`) and the trainer (`zentrain`) version
independently; the binary format (`ZNPR v3`) is the contract between them.
See [`MIGRATION.md`](https://github.com/imazen/zenanalyze/blob/main/MIGRATION.md) for the path from the previous
(unpublished) `zenpicker` Rust shell to the current layout.

**Cross-codec defaults + data discipline**: read
[`zentrain/PRINCIPLES.md`](https://github.com/imazen/zenanalyze/blob/main/zentrain/PRINCIPLES.md) before adopting or
re-baking a picker for any codec (zenjpeg / zenwebp / zenavif /
zenjxl / zenpng / zengif / zenpicker / zensim). It's the source of
truth for what's invariant — corpus shape, argmin objectives, time-
budget patterns, multi-metric bakes, OOD / reach gates, validation
gates that block release.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. Commercial licensing available
from imazen — contact `lilith@imazen.io`.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | **zenanalyze** · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
