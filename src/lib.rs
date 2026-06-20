//! Image content analyzers for adaptive codec decisions.
//!
//! Computes the numeric features used by oracle-trained decision trees
//! (originally `coefficient::scripts/fit_oracle_tree.py`) to drive
//! per-image encoder configuration. Every shipped feature is wired
//! into at least one fitted tree's splits per the 2026-04-25 audit.
//!
//! # Public API
//!
//! All public types are opaque. Construction goes through the
//! [`feature`] module:
//!
//! - [`feature::AnalysisFeature`]: stable identifier per feature
//!   (`#[non_exhaustive]`, `#[repr(u16)]`, sequential discriminants —
//!   ids are immutable once shipped).
//! - [`feature::FeatureValue`]: `F32` / `U32` / `Bool` tagged value
//!   with `to_f32` lossless coercion.
//! - [`feature::FeatureSet`]: opaque bitset with full `const fn` set
//!   math. The only "all features" entry is
//!   [`feature::FeatureSet::SUPPORTED`] — there is intentionally no
//!   `all()`. Production callers enumerate what they need.
//! - [`feature::AnalysisQuery`]: opaque request handle. Sampling
//!   budgets are crate invariants, not per-call knobs.
//! - [`feature::ImageGeometry`]: opaque `width`/`height`/
//!   `pixels`/`megapixels`/`aspect_ratio` accessors.
//! - [`feature::AnalysisResults`]: opaque queryable container.
//!
//! # Entry points
//!
//! - [`analyze_features`] — preferred. Takes any [`PixelSlice`] and
//!   an [`feature::AnalysisQuery`]. Native zero-copy on RGB8/RGB8_SRGB
//!   inputs; one-row scratch + `RowConverter` on anything else (every
//!   `zenpixels` descriptor: RGBA/BGRA u8/u16, GRAY u8/u16, RGB/RGBA
//!   f32 linear, PQ/HLG/Bt709, BT.709/DisplayP3/BT.2020/AdobeRGB).
//! - [`analyze_features_rgb8`] — convenience for packed RGB8 buffers.
//! - [`feature_vector`] / [`feature_vector_all`] and
//!   [`feature_vector_packed8`] / [`feature_vector_packed8_all`] — low-coupling
//!   picker-facing facade. These functions use stable `u16` feature ids and
//!   caller-owned `&mut [f32]` buffers instead of exposing
//!   [`feature::AnalysisQuery`], [`feature::AnalysisResults`],
//!   [`feature::FeatureSet`], or [`feature::AnalysisFeature`] in downstream
//!   codec signatures. Use them when a codec wants to ship a baked
//!   `zenpredict` model without re-exporting zenanalyze's typed analysis API
//!   through its own public surface.
//!
//! # Composition pattern
//!
//! ```ignore
//! use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
//!
//! const JPEG_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::EdgeDensity)
//!     .with(AnalysisFeature::DctCompressibilityY);
//! const WEBP_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::AlphaPresent);
//!
//! // Orchestrator unions, runs once, hands results down.
//! let needed = JPEG_FEATURES.union(WEBP_FEATURES);
//! let r = zenanalyze::analyze_features(slice, &AnalysisQuery::new(needed))?;
//! let var = r.get_f32(AnalysisFeature::Variance);
//! ```
//!
//! # Threshold contract — iterating during 0.1.x
//!
//! Numeric thresholds and normalization scales in this crate are
//! converging during the 0.1.x line. Downstream consumers that
//! compile-in fitted models (oracle decision trees, content
//! selectors) must pin to a specific zenanalyze patch version and
//! re-validate / retrain whenever they bump it — feature outputs
//! change between patches whenever a real bug is fixed.
//!
//! Every breaking numeric change ships with a CHANGELOG entry under
//! the version it landed in. When the algorithm stabilizes (1.0),
//! the contract will freeze. Sampling budgets (Tier 1/2 stripe step,
//! Tier 3 block cap) are part of that contract — not exposed as
//! per-call knobs.
//!
//! # Tier architecture
//!
//! Five passes, each gated by what the requested [`feature::FeatureSet`]
//! actually needs. None of them materialize a full RGB8 buffer:
//!
//! | Pass        | Iterates over            | Reads               | Cost (4 MP) | Drives                                    |
//! |-------------|--------------------------|---------------------|-------------|-------------------------------------------|
//! | Tier 1      | Stripe-sampled rows      | RGB8 (via RowStream) | ~1 ms      | luma stats, edges, chroma, uniformity, grayscale_score |
//! | Tier 2      | 3-row sliding window     | RGB8                 | ~2 ms      | per-axis Cb/Cr sharpness                  |
//! | Tier 3      | Sampled 8×8 DCT blocks   | RGB8                 | ~3 ms      | DCT energy, entropy, AQ map, noise floor, line-art, patch fraction |
//! | Palette     | Full-image (every pixel) | RGB8                 | ~1 ms      | distinct_color_bins                       |
//! | Alpha       | Stride-sampled rows      | **Source bytes**     | ~0.3 ms    | alpha presence / used / bimodal           |
//! | tier_depth  | Stride-sampled rows      | **Source bytes**     | ~0.5 ms HDR / ~0 ms SDR | peak nits, headroom, bit depth, HDR-present |
//!
//! Tier 1/2/3 + Palette read RGB8, going through `RowStream`'s `Native`
//! zero-copy path on RGB8 layouts and `RowConverter` row-by-row on
//! everything else. Alpha and `tier_depth` read source samples
//! directly — that's load-bearing for HDR (RowConverter doesn't
//! tonemap PQ/HLG; its narrowing clips to [0, 1] sRGB-display) and
//! for the alpha pass (avoids a precision-losing pre-multiply round-
//! trip on u16/f32 alpha).
//!
//! ## Why a separate `tier_depth`
//!
//! The standard tiers' threshold contract is calibrated on display-
//! space RGB8 bytes. HDR content's source samples encode dynamic
//! range that the RGB8 narrowing destroys; a 4000-nit PQ source and
//! a 100-nit-clipped SDR source produce byte-identical RGB8 streams.
//! Routing the depth tier through RowStream would have made it
//! literally impossible to surface HDR-aware features.
//!
//! Adding `tier_depth` as a fifth `const bool` axis to the dispatch
//! table would have doubled it from 16 to 32 monomorphizations for
//! near-zero benefit — the depth tier reads source bytes, not RGB8
//! rows, and shares no inner loop with T1/T2/T3 to specialize. So
//! it's wired as a runtime branch outside the const-bool dispatch.
//!
//! ## Empirical calibration (corpus-eval 2026-04-27)
//!
//! Pre-0.1.0-ship calibration baseline measured on a 219-image
//! labeled corpus from `coefficient/benchmarks/classifier-eval/labels.tsv`
//! (174 photo, 36 screen, 9 illustration, 44 marked synthetic;
//! pooled from cid22-train/val, clic2025-1024, gb82, gb82-sc,
//! imageflow, kadid10k, qoi-benchmark). Full per-class
//! distributions, ROC-AUC ranking for every feature, recommended
//! operating thresholds, and the recalibration candidates that
//! were considered and rejected are recorded in
//! `docs/calibration-corpus-2026-04-27.md`.
//!
//! Top-line empirical findings:
//!
//! - **Strongest single screen-vs-photo discriminator**:
//!   [`feature::AnalysisFeature::PatchFraction`] (AUC = 0.880,
//!   F1 = 0.769 at `>= 0.27`).
//! - **Strongest photo classifier** was the (now-retired in 0.2.0)
//!   `NaturalLikelihood` composite (F1 = 0.924 at `>= 0.06`).
//! - **Near-deterministic line-art signal** was the (now-retired in 0.2.0)
//!   `LineArtScore` composite (`> 0`, F1 = 0.978).
//! - **Derived likelihoods empirically saturate at ~0.70**, not
//!   1.0 — operating thresholds live in the 0.3–0.6 band, not 0.8+.
//!
//! Spearman ρ |≥ 0.85| pairs (mostly structural, all kept):
//!
//! - `distinct_color_bins` ↔ `palette_density`: ρ = +1.00 — derived.
//! - `flat_color_block_ratio` ↔ `screen_content_likelihood`: ρ =
//!   +0.99 — structural, the former is the latter's primary input.
//! - `uniformity` ↔ `aq_map_mean`: ρ = −0.97. Both measure block
//!   flatness; drive different knobs (T1 fast-path vs T3 AQ-driven
//!   trellis-λ).
//! - `chroma_complexity` ↔ `colourfulness`: ρ = +0.97. Both
//!   quantify chroma spread; one is normalised, the other is the
//!   raw Hasler-Süsstrunk M3 published scale — keep both for
//!   ergonomics.
//! - `aq_map_mean` ↔ `noise_floor_y`: ρ = +0.91. Both fall when
//!   blocks are flat. Different downstream knobs (zenjpeg trellis-λ
//!   vs `pre_blur`), keep both.
//! - `noise_floor_y` ↔ `screen_content_likelihood`: ρ = −0.89.
//!   Screen content is clean-by-construction; photos carry sensor
//!   noise. Different passes, different knobs.
//! - `cb_sharpness` ↔ `cr_sharpness`: ρ = +0.89. Co-vary by
//!   construction in natural content.
//!
//! No deletion candidates were found. Numeric drift in 0.1.x
//! patches is permitted by the threshold contract above; the
//! committed `docs/calibration-corpus-2026-04-27.md` is the
//! pre-ship baseline against which patch drift can be compared.

// `#[archmage::autoversion]` generates dispatch trampolines that
// don't always reference the unversioned base function.
#![allow(dead_code)]

mod alpha;
mod dimensions;
pub mod feature;
mod grayscale;
#[cfg(test)]
mod linear_tier;
pub(crate) mod luma;
mod palette;
pub(crate) mod row_stream;
pub(crate) mod tier1;
pub(crate) mod tier2_chroma;
pub(crate) mod tier3;
#[cfg(feature = "hdr")]
pub(crate) mod tier_depth;
/// XYB color-loss features ([`feature::AnalysisFeature::Xyb444ColorLoss`] /
/// [`feature::AnalysisFeature::XybBquarterChromaLoss`]). Computed from the
/// tier [`row_stream::RowStream`] in [`analyze_features`]; see the module docs.
#[cfg(feature = "experimental")]
pub(crate) mod xyb_color_loss;

use core::fmt;

use zenpixels::{PixelDescriptor, PixelSlice};

use row_stream::RowStream;

/// Errors returned from the public analyzer entries.
///
/// Variants are stable; new variants will be added behind
/// `#[non_exhaustive]` so a `match` on this type must include a `_`
/// arm. The `Display` form is suitable for surfacing to end users
/// (concise, no internal type names); the `source()` chain points at
/// the underlying cause for `Convert` and `Internal` variants.
#[non_exhaustive]
#[derive(Debug)]
pub enum AnalyzeError {
    /// `zenpixels_convert::RowConverter` couldn't build a converter
    /// for the source descriptor (e.g. CMYK without a CMS plugin
    /// loaded, or an unsupported alpha mode). The analyzer accepts
    /// every layout `RowConverter` can ingest, so this variant is
    /// reserved for genuinely-unsupported descriptors.
    Convert(String),
    /// Caller-supplied raw bytes don't form a valid `PixelSlice` for
    /// the declared dimensions / descriptor (e.g. the buffer is
    /// too short, the stride is below the per-row byte minimum, or
    /// alignment doesn't match the channel type). Used by the
    /// `try_analyze_features_rgb8` and any future `try_*` entry that
    /// doesn't accept a pre-validated [`PixelSlice`]. Production
    /// callers can pattern-match on this variant to distinguish
    /// "user input was malformed" from "system ran out of resources"
    /// without parsing the message.
    InvalidInput(String),
    /// Heap allocation refused — the analyzer's working buffers are
    /// proportional to image width × tier (largest single chunk is
    /// `8 × width × 3` for the Tier 3 block-row scratch on a 4 K
    /// image, so ~96 KiB; total per-call working set runs ~265 KiB
    /// at 4 K).
    ///
    /// Currently **never returned** by the public APIs because every
    /// allocation is infallible (`vec![]` / `Box::new`); the variant
    /// exists so that a future `try_*` entry that does fallible
    /// allocation (`Vec::try_reserve`, etc.) has a stable, pattern-
    /// matchable home for the OOM signal. Production callers handling
    /// untrusted images should match this variant explicitly so the
    /// switch from infallible to fallible internals is a non-event.
    ///
    /// `bytes_requested` is the size of the rejected allocation when
    /// known; `None` means the failing path didn't carry the size to
    /// the error site.
    OutOfMemory {
        /// Size of the rejected allocation, in bytes. `None` when
        /// the failing path didn't carry the size into the error.
        bytes_requested: Option<usize>,
    },
    /// An unexpected failure that doesn't fit the other variants.
    /// Strings are kept opaque on purpose — production code should
    /// not pattern-match on the message.
    Internal(String),
}

impl fmt::Display for AnalyzeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalyzeError::Convert(msg) => write!(f, "row conversion setup failed: {msg}"),
            AnalyzeError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            AnalyzeError::OutOfMemory { bytes_requested } => match bytes_requested {
                Some(n) => write!(f, "out of memory: requested {n} bytes"),
                None => write!(f, "out of memory"),
            },
            AnalyzeError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl core::error::Error for AnalyzeError {}

/// Run the analyzer over any [`PixelSlice`] for the requested feature
/// set. Returns an opaque [`feature::AnalysisResults`] holding only
/// the features the query asked for.
///
/// `query` is built from a [`feature::FeatureSet`] composed via
/// `const fn` set math:
///
/// ```ignore
/// use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
/// const MY_FEATURES: FeatureSet = FeatureSet::new()
///     .with(AnalysisFeature::Variance)
///     .with(AnalysisFeature::EdgeDensity);
/// let results = zenanalyze::analyze_features(slice, &AnalysisQuery::new(MY_FEATURES))?;
/// let var = results.get_f32(AnalysisFeature::Variance);
/// ```
///
/// Sampling budgets (Tier 1/2 stripe sampling, Tier 3 block cap) are
/// crate invariants — see [`feature::AnalysisQuery`] for the rationale
/// and the `#[doc(hidden)]` `__internal_with_overrides` backdoor for
/// tests / oracle re-extraction.
///
/// # Dispatch
///
/// Inspects the requested [`feature::FeatureSet`] and computes four
/// `const bool` axes (`PAL`/`T2`/`T3`/`ALPHA`). Sixteen
/// monomorphized [`analyze_specialized`] instantiations cover every
/// combination; inside each, unrequested tiers become straight-line
/// const-eval'd dead code. A caller asking for only `Variance` runs
/// Tier 1 and nothing else.
///
/// # Layered defense against garbage outputs
///
/// 1. Dispatch axes are unioned with derived-feature dependency
///    closures ([`feature::T3_NEEDED_BY`], [`feature::PAL_NEEDED_BY`])
///    so e.g. asking for `ScreenContentLikelihood` correctly gates
///    the palette pass on (it reads `distinct_color_bins`).
/// 2. [`tier3::compute_derived_likelihoods`] is itself const-bool
///    gated — even if the dispatch axes drift, a likelihood whose
///    deps weren't computed is left at default and never written.
/// 3. [`feature::AnalysisResults::get`] only returns values that
///    were both requested and written. Unrequested or unwritten
///    features come back as `None`. **Caller never sees garbage.**
///
/// # Wide gamut, HDR, and bit-depth policy
///
/// **The analyzer accepts every layout `zenpixels_convert::RowConverter`
/// can ingest.** Specifically:
///
/// - **24-bpp packed RGB** (RGB8 / RGB8_SRGB / RGB8 with Display P3 /
///   Rec.2020 / AdobeRGB primaries / non-sRGB transfer) — passes through
///   the zero-copy [`row_stream::RowStream`] `Native` path, byte-for-byte.
///   The analyzer's BT.601 luma weights produce slightly-different
///   numbers for non-sRGB primaries, and that's the principled outcome
///   — wide-gamut content has more saturated colour, so chroma /
///   colourfulness signals legitimately read higher.
/// - **RGBA8 / BGRA8 / Rgbx8 / Bgrx8** — alpha is analysed by the
///   separate alpha pass that reads the source descriptor directly.
///   The RGB tiers see a row-converted RGB8 view; for opaque pixels
///   that's identity, for translucent pixels it depends on the
///   converter's straight-alpha policy.
/// - **RGB16 / RGBA16** — converted to RGB8 by taking the high byte of
///   each channel. Crucially: an 8-bit image promoted to 16 bits via
///   the standard `u8 * 257` (or `u8 << 8`) doubling produces the same
///   high byte back, so analyzer features on a u8-promoted u16 source
///   are *bit-identical* to the original u8 (locked by tests). Genuine
///   16-bit content's extra precision feeds derived signals where
///   it's algorithmically meaningful (palette bins are 5-bit-per-
///   channel; sub-LSB drift in the bottom 8 bits doesn't change
///   uniformity / variance categorically).
/// - **f32 RGB / RGBA (linear or HDR)** — converted to display-space
///   RGB8 via the descriptor's transfer function. For SDR f32 content
///   normalized to `[0, 1]`, the round-trip from a u8 promotion is
///   bit-identical. For PQ / HLG / linear-light HDR, the analyzer
///   measures the SDR rendition — that's the principled choice for
///   features whose threshold contract is calibrated on display-space
///   bytes; HDR-aware analysis (peak luminance, HDR pixel fraction,
///   wide-gamut peak) lives in the future `tier_depth` work
///   (issue #120) and is computed directly on the source samples.
/// - **Gray8 / Gray16 / GrayF32 / GrayA{8,16,F32}** — converted to RGB8
///   by replicating the luma channel. Chroma signals are then trivially
///   zero, which is correct.
///
/// The point: per-image codec decisions don't usually break on a few
/// LSBs of luma drift. They break on the analyzer refusing to run.
/// The previous "opt-in via `analyze_features_lossy_convert`" gate was
/// over-defensive — every supported wide-gamut / HDR / 16-bit source
/// either round-trips through conversion as identity (u8-promoted) or
/// produces a rendering that is the legitimate signal for SDR-
/// calibrated features. We accept everything, document the cases
/// where the rendering differs from a byte-for-byte input, and give
/// codecs a stable feature surface.
///
/// # Errors
///
/// - [`AnalyzeError::Convert`] if `RowConverter` doesn't support the
///   source descriptor (e.g. CMYK without a CMS plugin loaded). The
///   set of accepted descriptors is `zenpixels-convert`'s problem,
///   not the analyzer's.
/// - [`AnalyzeError::Internal`] for unexpected failures (e.g. a
///   malformed `PixelSlice` whose row stride doesn't match its
///   declared format).
pub fn analyze_features(
    slice: PixelSlice<'_>,
    query: &feature::AnalysisQuery,
) -> Result<feature::AnalysisResults, AnalyzeError> {
    let features = query.features();
    let pal = features.intersects(feature::PAL_NEEDED_BY);
    let t2 = features.intersects(feature::TIER2_FEATURES);
    let t3 = features.intersects(feature::T3_NEEDED_BY);
    // Sub-tier gate: run the per-block DCT pass only when at least
    // one DCT-derived feature was requested. When false but `t3` is
    // true the cheap luma-histogram pass still runs (entropy /
    // line-art) but the ~0.97 ms-per-Mpx DCT walk is skipped.
    let dct = features.intersects(feature::DCT_NEEDED_BY);
    let alpha = features.intersects(feature::ALPHA_FEATURES);
    // Strict-equality grayscale classifier (R == G == B for every
    // pixel). Runtime axis like `run_depth` — independent of the
    // const-bool tier dispatch. Walks rows with early exit on the
    // first non-gray row; sub-microsecond on colored images.
    let run_strict_gray = features.contains(feature::AnalysisFeature::IsGrayscale);

    // RGB8-derived XYB color-loss features (experimental). Computed from the
    // tier `RowStream` (works for any pixel format), not a post-step — so the
    // generic `analyze_features` path produces them too, not just the RGB8
    // convenience wrapper. Runtime axes like `run_strict_gray`.
    #[cfg(feature = "experimental")]
    let run_xyb444 = features.contains(feature::AnalysisFeature::Xyb444ColorLoss);
    #[cfg(not(feature = "experimental"))]
    let run_xyb444 = false;
    #[cfg(feature = "experimental")]
    let run_xyb_bq = features.contains(feature::AnalysisFeature::XybBquarterChromaLoss);
    #[cfg(not(feature = "experimental"))]
    let run_xyb_bq = false;

    // Pick the palette path: full-precision scan if any "exact count"
    // palette feature was requested; otherwise the early-exit scan
    // (much faster on photographic content). Computed once per call;
    // analyze_specialized_raw const-folds nothing on this axis.
    let palette_full_required = features.intersects(feature::PALETTE_FULL_FEATURES);
    // GrayscaleScore lives on the palette tier (full-scan, 100 %
    // coverage) but its per-pixel max/min gate isn't free — only run
    // the gate when the caller actually requested it.
    let palette_wants_grayscale = features.contains(feature::AnalysisFeature::GrayscaleScore);

    // Tier 1 dispatch: skip the separate Laplacian SIMD row pass
    // when LaplacianVariance isn't requested. Load-bearing for
    // orchestrator-style callers like zenjpeg's ADAPTIVE_FEATURES,
    // which doesn't ask for LaplacianVariance and currently pays
    // the full cost of the separate SIMD row walk anyway.
    // Trigger the laplacian SIMD pass on the variance feature OR
    // any of the percentile variants (#42 / #49). Without the P*
    // additions, requesting only `LaplacianVariancePeak` would skip
    // the whole histogram pass and leave RawAnalysis at zeros,
    // bypassing the per-feature minimum-sample-count floor (#49).
    let tier1_wants_laplacian = features.intersects(
        feature::FeatureSet::new()
            .with(feature::AnalysisFeature::LaplacianVariance)
            .with(feature::AnalysisFeature::LaplacianVarianceP50)
            .with(feature::AnalysisFeature::LaplacianVarianceP75)
            .with(feature::AnalysisFeature::LaplacianVarianceP90)
            .with(feature::AnalysisFeature::LaplacianVarianceP99)
            .with(feature::AnalysisFeature::LaplacianVariancePeak),
    );

    // Tier 1 full-kernel gate: flip on for `Variance` /
    // `Colourfulness` / `EdgeSlopeStdev` / `LaplacianVariance` — the
    // accumulators inside `accumulate_row_simd`'s `if FULL` block.
    // See `feature::TIER1_FULL_FEATURES`.
    let tier1_full_kernel = features.intersects(feature::TIER1_FULL_FEATURES);
    // Tier 1 skin-gate: peeled off `wants_full_kernel` so callers
    // that only want `SkinToneFraction` don't pay for Variance /
    // Colourfulness / edge-slope, and vice versa. The BT.601 chroma
    // matrix + Chai-Ngan thresholds spent 12 vmovups + 13 broadcasts
    // of register-spill traffic when bundled — see `cargo asm` audit.
    let tier1_wants_skin = features.intersects(feature::TIER1_SKIN_FEATURES);

    // Depth tier is a runtime axis (not const-bool) — adding it to
    // the 16-arm dispatch would double to 32 arms for one extra
    // pass with negligible LLVM monomorphization wins. The depth
    // tier reads source bytes directly, no shared inner loop with
    // T1/T2/T3 to specialize, so a runtime branch costs nothing.
    let run_depth = features.intersects(feature::DEPTH_FEATURES);
    // Linear-light opt-in. When on, the RowStream emits diffuse-white-normalized
    // linear RGB8 (decode → ×anchor → display range), so EVERY content tier
    // reads envelope-normalized bytes in its existing combined pass and an HDR
    // envelope carrying SDR content yields the same features as plain SDR. Off
    // by default — the default path keeps its zero-copy fast paths and pays
    // nothing.
    let run_linear_light = query.linear_light();
    // Source descriptor — captured up front so we can hand it back to
    // codecs verbatim via `AnalysisResults::source_descriptor()` even
    // after the analyzer's RowConverter / RowStream consumes the slice.
    let source_descriptor = slice.descriptor();

    macro_rules! dispatch {
        ($pal:literal, $t2:literal, $t3:literal, $a:literal) => {{
            let (raw, geometry) = analyze_specialized_raw::<$pal, $t2, $t3, $a>(
                slice,
                feature::DEFAULT_PIXEL_BUDGET,
                feature::DEFAULT_HF_MAX_BLOCKS,
                palette_full_required,
                palette_wants_grayscale,
                tier1_wants_laplacian,
                tier1_full_kernel,
                tier1_wants_skin,
                run_depth,
                dct,
                run_strict_gray,
                run_xyb444,
                run_xyb_bq,
                run_linear_light,
            )?;
            Ok(raw.into_results(features, geometry, source_descriptor))
        }};
    }
    match (pal, t2, t3, alpha) {
        (false, false, false, false) => dispatch!(false, false, false, false),
        (false, false, false, true) => dispatch!(false, false, false, true),
        (false, false, true, false) => dispatch!(false, false, true, false),
        (false, false, true, true) => dispatch!(false, false, true, true),
        (false, true, false, false) => dispatch!(false, true, false, false),
        (false, true, false, true) => dispatch!(false, true, false, true),
        (false, true, true, false) => dispatch!(false, true, true, false),
        (false, true, true, true) => dispatch!(false, true, true, true),
        (true, false, false, false) => dispatch!(true, false, false, false),
        (true, false, false, true) => dispatch!(true, false, false, true),
        (true, false, true, false) => dispatch!(true, false, true, false),
        (true, false, true, true) => dispatch!(true, false, true, true),
        (true, true, false, false) => dispatch!(true, true, false, false),
        (true, true, false, true) => dispatch!(true, true, false, true),
        (true, true, true, false) => dispatch!(true, true, true, false),
        (true, true, true, true) => dispatch!(true, true, true, true),
    }
}

/// Const-bool-monomorphized analyzer body. Returns the dense
/// [`feature::RawAnalysis`] + [`feature::ImageGeometry`] — the public
/// [`analyze_features`] entry post-converts via
/// [`feature::RawAnalysis::into_results`] using the caller's
/// [`feature::FeatureSet`]. Each `if PAL { … }` arm is dead code in
/// instantiations where the axis is `false`; LLVM const-evals the
/// predicate and DCEs the branch entirely.
///
/// Tier 1 is always-on: if a caller dispatches through here they
/// almost always want at least one Tier 1 feature (variance /
/// edge_density / colourfulness / etc.). Adding a fifth axis to skip
/// it doubles the dispatch table for marginal benefit; defer until
/// there's data showing real callers asking for zero T1 features.
///
/// Sampling budgets `pixel_budget` / `hf_max_blocks` are runtime
/// values for the test / oracle override path. `analyze_features`
/// always passes the canonical [`feature::DEFAULT_PIXEL_BUDGET`] /
/// [`feature::DEFAULT_HF_MAX_BLOCKS`] constants.
#[allow(clippy::too_many_arguments)] // monomorphization dispatcher: 4 const-bool tier gates + 5 runtime sub-knobs all live in one specialization site
fn analyze_specialized_raw<const PAL: bool, const T2: bool, const T3: bool, const ALPHA: bool>(
    slice: PixelSlice<'_>,
    pixel_budget: usize,
    hf_max_blocks: usize,
    palette_full_required: bool,
    palette_wants_grayscale: bool,
    tier1_wants_laplacian: bool,
    tier1_full_kernel: bool,
    tier1_wants_skin: bool,
    run_depth: bool,
    run_dct: bool,
    run_strict_gray: bool,
    run_xyb444: bool,
    run_xyb_bq: bool,
    run_linear_light: bool,
) -> Result<(feature::RawAnalysis, feature::ImageGeometry), AnalyzeError> {
    #[cfg(not(feature = "experimental"))]
    let _ = (run_xyb444, run_xyb_bq);
    let width = slice.width();
    let height = slice.rows();
    let geometry = feature::ImageGeometry::new(width, height);
    // Snapshot descriptor before `slice` is moved into RowStream.
    let descriptor = slice.descriptor();

    let alpha_stats = if ALPHA {
        alpha::scan_alpha(&slice, pixel_budget)
    } else {
        Default::default()
    };

    // Source-direct depth tier — no RowStream / RowConverter; reads
    // descriptor samples and decodes through the transfer function.
    // Behind the `hdr` cargo feature; gated off entirely when the
    // feature is disabled (then `run_depth` is dead-code-eliminated
    // since DEPTH_FEATURES is empty in that build).
    #[cfg(feature = "hdr")]
    let depth_stats = if run_depth {
        tier_depth::scan_depth(&slice, pixel_budget)
    } else {
        Default::default()
    };
    // Suppress unused-var warning when hdr is off.
    let _ = run_depth;

    // Build the row stream. Linear-light routes every content tier through one
    // diffuse-white-normalized linear RGB8 stream (so SDR-in-HDR ≡ SDR for all
    // features, in the shared passes); the default keeps the zero-copy fast
    // paths.
    let mut stream = if run_linear_light {
        RowStream::new_normalized_linear(slice).map_err(AnalyzeError::Convert)?
    } else {
        RowStream::new(slice).map_err(AnalyzeError::Convert)?
    };

    // Palette routing: if any "full-precision" palette feature was
    // requested (DistinctColorBins / Chao1 / PaletteDensity), run
    // the full scan. Otherwise — when only quick-path features
    // (PaletteFitsIn256 / PaletteLog2Size) are asked — use the
    // early-exit scan that bails as soon as the running count
    // exceeds 256. On photos this typically returns within ~10
    // image rows.
    let palette_stats = if PAL {
        if palette_full_required {
            palette::scan_palette(&mut stream, palette_wants_grayscale)
        } else {
            palette::scan_palette_quick(&mut stream)
        }
    } else {
        Default::default()
    };

    let mut raw = feature::RawAnalysis::default();

    // Dimension features — pure descriptor math, no per-pixel work,
    // always populated. The `into_results` filter drops them if the
    // caller didn't ask. Costs ~10 ns per call.
    dimensions::populate_dimensions(&mut raw, width, height, descriptor);

    if width >= 2 && height >= 2 {
        // Tier 1 dispatch knobs — currently just `wants_laplacian`,
        // which lets orchestrator-style callers (e.g. zenjpeg's
        // `ADAPTIVE_FEATURES`) skip the separate Laplacian SIMD row
        // pass when `LaplacianVariance` isn't requested.
        // `TIER1_EXTRAS_FEATURES` is the union of all features whose
        // accumulators add cost beyond the Tier 1 baseline; for now
        // only LaplacianVariance gates a runtime branch, but the
        // dispatch struct is laid out to grow more knobs (skin /
        // edge-slope / colourfulness const-fold) without churning
        // call sites.
        let t1_dispatch = tier1::Tier1Dispatch {
            wants_laplacian: tier1_wants_laplacian,
            wants_full_kernel: tier1_full_kernel,
            wants_skin: tier1_wants_skin,
        };
        tier1::extract_tier1_into_dispatch(&mut raw, &mut stream, pixel_budget, t1_dispatch);
        if T2 && width >= 3 && height >= 3 {
            tier2_chroma::populate_tier2(&mut raw, &mut stream, pixel_budget);
        }
        if T3 && width >= 8 && height >= 8 {
            tier3::populate_tier3(&mut raw, &mut stream, hf_max_blocks, run_dct);
        }
        // Strict-equality grayscale classifier — runtime axis. Walks
        // every row with early exit at the first non-gray pixel. Uses
        // the same RowStream the tier passes do; on Native RGB8
        // sources this is zero-copy. Independent of the palette tier;
        // costs ~6 µs on colored content (exits row 1) and a few ms
        // on truly grayscale 4 MP images.
        if run_strict_gray {
            raw.is_grayscale = grayscale::scan_strict_grayscale(&mut stream);
        }
        // XYB color-loss features: re-walk the RowStream (random-access, same
        // rows the tiers read — native zero-copy on RGB8). Strided sampling
        // matches the contiguous reference exactly.
        #[cfg(feature = "experimental")]
        {
            if run_xyb444 {
                raw.xyb444_color_loss = xyb_color_loss::xyb444_color_loss_from_stream(&mut stream);
            }
            if run_xyb_bq {
                raw.xyb_bquarter_chroma_loss =
                    xyb_color_loss::xyb_bquarter_chroma_loss_from_stream(&mut stream);
            }
        }
        // Layered defense: const-bool gated. Refuses to write a
        // likelihood whose deps weren't computed, regardless of what
        // the dispatch axes decided.
        tier3::compute_derived_likelihoods::<T3, PAL>(&mut raw);
    }

    if ALPHA {
        raw.alpha_present = alpha_stats.present;
        raw.alpha_used_fraction = alpha_stats.used_fraction;
        raw.alpha_bimodal_score = alpha_stats.bimodal_score;
    }

    if PAL {
        // Quick-path signals — populated by both scan paths.
        raw.palette_log2_size = palette_stats.palette_log2_size;
        raw.palette_fits_in_256 = palette_stats.fits_in_256;
        // Full-path-only signals — quick-path leaves these at 0; the
        // `into_results` filter drops them so callers who didn't ask
        // for them get `None` instead of a misleading 0.
        if palette_full_required {
            raw.distinct_color_bins = palette_stats.distinct;
            // GrayscaleScore is computed on the same full-scan walk
            // as the distinct-bin histogram. 100 % coverage is
            // load-bearing — the score is used downstream as a
            // binary classifier (`>= 0.99` ⇒ encode as grayscale),
            // and stripe-sampling at ~5 % budget would let one
            // colour pixel slip past the gate ~95 % of the time.
            raw.grayscale_score = if palette_stats.total_pixels > 0 {
                let gray = palette_stats
                    .total_pixels
                    .saturating_sub(palette_stats.non_grayscale);
                (gray as f64 / palette_stats.total_pixels as f64) as f32
            } else {
                0.0
            };
            // PaletteDensity is `#[deprecated]` + experimental — still
            // computed during the deprecation window for un-migrated
            // pickers; derived from the always-on DistinctColorBins.
            #[cfg(feature = "experimental")]
            {
                let pixel_count = (width as f64) * (height as f64);
                let denom = pixel_count.clamp(1.0, 32_768.0);
                raw.palette_density =
                    (raw.distinct_color_bins as f64 / denom).clamp(0.0, 1.0) as f32;
            }
        }
        let _ = palette_stats; // silence unused on the all-experimental-off path
    }

    // Depth tier writeback. Fields + `tier_depth` module live behind
    // `cfg(feature = "hdr")` (HDR / wide-gamut / depth features —
    // off by default, opt-in for codecs that handle PQ / HLG /
    // 10-bit / wide-gamut content).
    #[cfg(feature = "hdr")]
    if run_depth {
        raw.peak_luminance_nits = depth_stats.peak_nits;
        raw.p99_luminance_nits = depth_stats.p99_nits;
        raw.hdr_headroom_stops = depth_stats.headroom_stops;
        raw.hdr_pixel_fraction = depth_stats.hdr_pixel_fraction;
        raw.wide_gamut_peak = depth_stats.wide_gamut_peak;
        raw.wide_gamut_fraction = depth_stats.wide_gamut_fraction;
        raw.effective_bit_depth = depth_stats.effective_bit_depth;
        raw.hdr_present = depth_stats.hdr_present;
        raw.gamut_coverage_srgb = depth_stats.gamut_coverage_srgb;
        raw.gamut_coverage_p3 = depth_stats.gamut_coverage_p3;
    }

    Ok((raw, geometry))
}

#[doc(hidden)]
/// **Unstable. Tests / oracle re-extraction only.** Run every tier
/// with caller-supplied sampling budgets, returning the dense
/// internal record. No stable API, no public path.
pub fn __analyze_internal(
    slice: PixelSlice<'_>,
    query: &feature::InternalQuery,
) -> Result<feature::AnalysisResults, AnalyzeError> {
    let run_depth = query.features.intersects(feature::DEPTH_FEATURES);
    let source_descriptor = slice.descriptor();
    let (raw, geometry) = analyze_specialized_raw::<true, true, true, true>(
        slice,
        query.pixel_budget,
        query.hf_max_blocks,
        true, // override path always uses full palette scan
        true, // and includes the grayscale gate (test path wants every signal)
        true, // and the Laplacian SIMD pass
        true, // and the Tier 1 full kernel (luma stats / Hasler M3 / edge slope)
        true, // and the Tier 1 skin gate
        run_depth,
        true,  // override path always runs the DCT pass (test/oracle wants every signal)
        true,  // and the strict-grayscale classifier
        true,  // xyb444_color_loss
        true,  // xyb_bquarter_chroma_loss
        false, // run_linear_light — oracle / dense extraction stays gamma
    )?;
    Ok(raw.into_results(query.features, geometry, source_descriptor))
}

#[cfg(test)]
/// Crate-internal full-fat analysis for tests: always runs every
/// tier so test assertions can read any field. Returns the dense
/// record + geometry; tests wrap it in `tests::TestOutput`.
pub(crate) fn analyze_full_raw_for_test(
    slice: PixelSlice<'_>,
    full_budgets: bool,
) -> Result<(feature::RawAnalysis, feature::ImageGeometry), AnalyzeError> {
    let pb = if full_budgets {
        usize::MAX
    } else {
        feature::DEFAULT_PIXEL_BUDGET
    };
    let hf = if full_budgets {
        4096
    } else {
        feature::DEFAULT_HF_MAX_BLOCKS
    };
    // Tests want every signal — always run the full palette path,
    // and the depth tier when it's compiled in.
    let run_depth = cfg!(feature = "experimental");
    analyze_specialized_raw::<true, true, true, true>(
        slice, pb, hf, true, true, true, true, true, run_depth, true, true, true, true, false,
    )
}

/// Convenience entry for callers holding a packed RGB8 buffer plus a
/// query. **Panics** on length mismatch or stride-construction failure.
///
/// Use this when the input is known-good (e.g. coming from a freshly
/// decoded image where length and stride are guaranteed by
/// construction). For untrusted input — anything where the buffer
/// length or dimensions came in over a wire / from disk / from an
/// FFI boundary — prefer [`try_analyze_features_rgb8`], which surfaces
/// the same problems through [`AnalyzeError::InvalidInput`] /
/// [`AnalyzeError::OutOfMemory`] instead of unwinding.
pub fn analyze_features_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    query: &feature::AnalysisQuery,
) -> feature::AnalysisResults {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(
        rgb.len(),
        w * h * 3,
        "analyze_features_rgb8: RGB8 buffer size mismatch"
    );
    let stride = w * 3;
    let slice = PixelSlice::new(rgb, width, height, stride, PixelDescriptor::RGB8_SRGB)
        .expect("RGB8 PixelSlice from packed buffer");
    // `Xyb444ColorLoss` / `XybBquarterChromaLoss` (experimental) are computed
    // inside `analyze_features` from the tier RowStream now, so this wrapper
    // needs no special handling — it just forwards.
    analyze_features(slice, query).expect("analyze never fails on RGB8")
}

/// Fallible parallel of [`analyze_features_rgb8`]. Returns
/// [`AnalyzeError::InvalidInput`] when the buffer length doesn't
/// match `width * height * 3` or the resulting stride is invalid;
/// returns [`AnalyzeError::OutOfMemory`] when (in a future fallible-
/// allocation build) the analyzer's working buffers can't be reserved.
///
/// **Recommended for production code that handles untrusted images.**
/// The internals share an analyzer pass with [`analyze_features_rgb8`];
/// the only difference is the error-reporting contract.
///
/// Today this fn never returns `OutOfMemory` because every internal
/// allocation is infallible — but the variant is part of the
/// [`AnalyzeError`] surface (it's `#[non_exhaustive]`) so that a
/// future minor release can tighten the internals to use
/// `Vec::try_reserve` / `Box::try_new` without breaking the public
/// signature. Production callers should match it explicitly today.
///
/// # Errors
///
/// - [`AnalyzeError::InvalidInput`] for buffer-length mismatches or
///   slice-construction failures.
/// - [`AnalyzeError::OutOfMemory`] from future fallible-alloc paths.
/// - [`AnalyzeError::Convert`] / [`AnalyzeError::Internal`] are not
///   reachable on the RGB8 fast path today, but listed here so a
///   `match` over the enum stays complete.
pub fn try_analyze_features_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    query: &feature::AnalysisQuery,
) -> Result<feature::AnalysisResults, AnalyzeError> {
    let w = width as usize;
    let h = height as usize;
    let need = w
        .checked_mul(h)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or_else(|| {
            AnalyzeError::InvalidInput(format!(
                "{width}×{height} pixels overflows usize when computing buffer length"
            ))
        })?;
    if rgb.len() != need {
        return Err(AnalyzeError::InvalidInput(format!(
            "RGB8 buffer length mismatch: expected {need} bytes for {width}×{height}, got {}",
            rgb.len()
        )));
    }
    let stride = w * 3;
    let slice = PixelSlice::new(rgb, width, height, stride, PixelDescriptor::RGB8_SRGB)
        .map_err(|e| AnalyzeError::InvalidInput(format!("PixelSlice::new failed: {e:?}")))?;
    analyze_features(slice, query)
}

// ============================================================================
// Feature-vector API for the codec-picking crate tree
// ============================================================================
//
// This facade is deliberately lower-coupling than `analyze_features`. Codec
// crates can keep their public picker/config APIs in terms of primitive schema
// data (`u16` feature ids + `f32` vectors) instead of re-exporting or naming
// zenanalyze's typed analysis surface. That avoids making codec semver depend
// on `AnalysisQuery`, `AnalysisResults`, `FeatureSet`, or `AnalysisFeature`
// appearing in their public signatures.
//
// The general entry points still take the shared `PixelSlice` type because
// zenpixels is the common pixel currency across the zen tree. That preserves
// the full image contract: u8/u16/f32, RGB/RGBA/BGRA/gray, explicit stride,
// transfer/primaries, and HDR. The packed8 helpers are raw-byte conveniences for
// the common std-only buffer case.

/// Number of features the canonical vector carries in this build — every feature
/// [`feature::FeatureSet::SUPPORTED`] computes.
///
/// This count is build-feature dependent (`experimental` / `hdr` add columns)
/// but stable within a build and equal to the number of ids [`feature_ids`]
/// writes. Codecs that compile in a baked model should store/check the model's
/// schema hash separately; this count is a sizing helper, not a compatibility
/// proof.
pub fn feature_count() -> usize {
    feature::FeatureSet::SUPPORTED.iter().count()
}

/// Write the canonical feature ids — the order [`feature_vector_packed8_all`]
/// fills — into `out`, returning how many were written. Align a model's input
/// schema to this order. If `out` is shorter than [`feature_count`] it's filled
/// to its length and the truncated count returned.
///
/// The ids are the low-coupling schema contract. Downstream crates can persist
/// or embed these ids without naming [`feature::AnalysisFeature`] in their own
/// public API.
pub fn feature_ids(out: &mut [u16]) -> usize {
    let mut n = 0;
    for f in feature::FeatureSet::SUPPORTED.iter() {
        if n >= out.len() {
            break;
        }
        out[n] = f.id();
        n += 1;
    }
    n
}

/// Stable snake_case name for a feature id, or `None` if no feature in this build
/// has that id (retired / cfg-disabled / unknown).
pub fn feature_name(id: u16) -> Option<&'static str> {
    feature::AnalysisFeature::from_u16(id).map(|f| f.name())
}

/// Whether this build computes the feature with the given stable id.
pub fn feature_id_supported(id: u16) -> bool {
    match feature::AnalysisFeature::from_u16(id) {
        Some(f) => feature::FeatureSet::SUPPORTED.contains(f),
        None => false,
    }
}

/// Resolve a snake_case feature name (optionally `feat_`-prefixed, as training
/// columns are written) to its stable id, or `None` if unknown in this build.
///
/// The reverse of [`feature_name`]. This is the primitive a codec needs to map
/// a baked model's feature-column NAMES onto ids — every wired picker currently
/// re-implements it by hand-building a name→variant map. Prefer
/// [`resolve_feature_ids`] when resolving a whole column list.
pub fn feature_id_by_name(name: &str) -> Option<u16> {
    feature::AnalysisFeature::from_name(name).map(|f| f.id())
}

/// Bump on ANY feature numeric-definition change (see [`feature_defs_version`]).
const FEATURE_DEFS_VERSION: u32 = 1;

/// Monotonic version of the feature **definitions** — the numeric algorithms,
/// thresholds, and normalization scales that decide a feature's computed VALUE.
/// Independent of the crate version and of the id/name set.
///
/// The schema hash a baked model carries (in `zenpredict`) protects the feature
/// *names + order*, but NOT their numeric meaning: a feature can keep its
/// id/name while its definition drifts (which the 0.x threshold contract
/// explicitly permits), silently feeding a stale model wrong values. Bake this
/// version next to the model and compare it at load — a mismatch means "the
/// analyzer's feature math moved since training; re-validate / retrain". It is
/// bumped in lockstep with the threshold-contract CHANGELOG entries; it freezes
/// at 1.0.
pub const fn feature_defs_version() -> u32 {
    FEATURE_DEFS_VERSION
}

/// This crate's version string (`CARGO_PKG_VERSION`, e.g. `"0.2.0"`).
///
/// The producer side of the **offered-results** contract stamps this together
/// with [`feature_defs_version`] onto a self-describing feature offer, so a
/// downstream codec picker can decide whether the offer was produced by the
/// same feature *definitions* its model was trained on — `(major.minor,
/// defs_version)` is the reuse key (a `0.2` offer can't satisfy a `1.0`-trained
/// model; `0.2.3` vs `0.2.7` is caught by `defs_version`). The negotiation logic
/// itself is version-agnostic and lives downstream, not here.
pub const fn analyzer_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Resolve a model's feature-column NAME list (each optionally `feat_`-prefixed)
/// to this build's stable ids, **preserving order**. Returns a plain `Vec<u16>`
/// — a core type the caller owns; nothing from this crate's typed surface is
/// held or crosses a boundary. `None` if ANY name is unknown to this build
/// (retired / cfg-off): the caller falls back rather than feed the model a zero
/// where it expected signal.
///
/// This is the blessed bind step. The resolved ids feed [`feature_vector`]; the
/// output `&[f32]` is the only thing that flows on to the predictor. Because the
/// ids are version-LOCAL (a 0.2 id need not equal a 1.0 id), the resolution must
/// run against the *same* zenanalyze version the model was baked with — pin it
/// in `Cargo.toml`. See `docs/feature-contract-pr-2026-06-19.md`.
pub fn resolve_feature_ids<S: AsRef<str>>(names: &[S]) -> Option<Vec<u16>> {
    names
        .iter()
        .map(|n| feature_id_by_name(n.as_ref()))
        .collect()
}

/// Extract a **model-aligned** feature vector from any [`PixelSlice`] into `out`.
/// `out[i]` receives the feature whose stable id is `ids[i]` (`f32::NAN` for an
/// id unknown in this build or undefined for the image). Only the analyzer
/// passes the requested ids need are run.
///
/// Works on **any bit depth / format the slice carries** — u8 / u16 / f32,
/// RGB / RGBA / BGRA / gray, sRGB / linear / PQ / HLG, SDR or HDR. HDR / wide-
/// gamut / bit-depth features (ids 32–39, 46, 47) are computed from the source
/// samples and require the `hdr` cargo feature; without it those ids read back
/// `NaN`. Returns `false` (leaving `out` untouched) when `out.len() < ids.len()`
/// or the analyzer's row conversion fails.
///
/// This is the preferred entry point for codec crates that already traffic in
/// `zenpixels` image descriptors. It keeps the codec's picker boundary to
/// `&[u16]` + `&mut [f32]`: callers do not need to import, re-export, or expose
/// zenanalyze's typed feature/query/result API.
///
/// Feed `out[..ids.len()]` straight to `zenpredict::Predictor::predict`.
pub fn feature_vector(slice: PixelSlice<'_>, ids: &[u16], out: &mut [f32]) -> bool {
    if out.len() < ids.len() {
        return false;
    }
    // Minimal FeatureSet covering the requested ids — gates the passes.
    let mut fs = feature::FeatureSet::new();
    for &id in ids {
        if let Some(f) = feature::AnalysisFeature::from_u16(id) {
            fs = fs.with(f);
        }
    }
    let query = feature::AnalysisQuery::new(fs);
    let Ok(results) = analyze_features(slice, &query) else {
        return false;
    };
    for (i, &id) in ids.iter().enumerate() {
        out[i] = feature::AnalysisFeature::from_u16(id)
            .and_then(|f| results.get_f32(f))
            .unwrap_or(f32::NAN);
    }
    true
}

/// Extract the **full canonical** feature vector (every supported feature, in
/// [`feature_ids`] order) from any [`PixelSlice`]. `out.len()` must be `>=`
/// [`feature_count`]. Same bit-depth generality and return-value contract as
/// [`feature_vector`].
pub fn feature_vector_all(slice: PixelSlice<'_>, out: &mut [f32]) -> bool {
    if out.len() < feature_count() {
        return false;
    }
    let query = feature::AnalysisQuery::new(feature::FeatureSet::SUPPORTED);
    let Ok(results) = analyze_features(slice, &query) else {
        return false;
    };
    for (i, f) in feature::FeatureSet::SUPPORTED.iter().enumerate() {
        out[i] = results.get_f32(f).unwrap_or(f32::NAN);
    }
    true
}

/// Packed 8-bit descriptor for a channel count: 1 = gray, 3 = RGB, 4 = RGBA.
fn descriptor_for_channels(channels: u8) -> Option<PixelDescriptor> {
    match channels {
        1 => Some(PixelDescriptor::GRAY8_SRGB),
        3 => Some(PixelDescriptor::RGB8_SRGB),
        4 => Some(PixelDescriptor::RGBA8_SRGB),
        _ => None,
    }
}

/// Build a `PixelSlice` over packed 8-bit `bytes`. `row_stride == 0` ⇒ tightly
/// packed (`width * channels`); a non-zero stride must be pixel-aligned.
fn packed8_slice(
    bytes: &[u8],
    width: u32,
    height: u32,
    row_stride: usize,
    channels: u8,
) -> Option<PixelSlice<'_>> {
    let desc = descriptor_for_channels(channels)?;
    let stride = if row_stride == 0 {
        width as usize * channels as usize
    } else {
        row_stride
    };
    PixelSlice::new(bytes, width, height, stride, desc).ok()
}

/// Raw-bytes convenience for [`feature_vector`] on packed 8-bit pixels.
///
/// This is the std-only-buffer shape for callers that do not want `PixelSlice`
/// in their own picker plumbing. It builds a `PixelSlice` internally:
/// `channels` 1 (gray) / 3 (RGB) / 4 (RGBA),
/// `row_stride` bytes per row (`0` ⇒ packed). Returns `false` on an unsupported
/// channel count, a misaligned/too-small buffer, or `out.len() < ids.len()`.
/// **8-bit only** — for higher bit depth (u16 / f32) or other formats, build a
/// [`PixelSlice`] and use [`feature_vector`].
pub fn feature_vector_packed8(
    bytes: &[u8],
    width: u32,
    height: u32,
    row_stride: usize,
    channels: u8,
    ids: &[u16],
    out: &mut [f32],
) -> bool {
    match packed8_slice(bytes, width, height, row_stride, channels) {
        Some(slice) => feature_vector(slice, ids, out),
        None => false,
    }
}

/// Raw-bytes convenience for [`feature_vector_all`] on packed 8-bit pixels.
///
/// Same `channels` / `row_stride` contract as [`feature_vector_packed8`];
/// **8-bit only** (use [`feature_vector`] with a [`PixelSlice`] for higher bit
/// depth).
pub fn feature_vector_packed8_all(
    bytes: &[u8],
    width: u32,
    height: u32,
    row_stride: usize,
    channels: u8,
    out: &mut [f32],
) -> bool {
    match packed8_slice(bytes, width, height, row_stride, channels) {
        Some(slice) => feature_vector_all(slice, out),
        None => false,
    }
}

#[cfg(test)]
#[allow(deprecated)] // tests cover composites variants slated for removal next major
mod tests;

#[cfg(test)]
mod feature_vector_tests {
    use super::*;
    use feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    #[test]
    fn from_name_round_trips_and_strips_prefix() {
        // every supported feature's name resolves back to its id
        let mut ids = [0u16; 256];
        let n = feature_ids(&mut ids);
        for &id in &ids[..n] {
            let name = feature_name(id).unwrap();
            assert_eq!(feature_id_by_name(name), Some(id), "name {name}");
            // the feat_ prefix (training-column form) is accepted too
            let prefixed = std::format!("feat_{name}");
            assert_eq!(feature_id_by_name(&prefixed), Some(id));
        }
        assert_eq!(feature_id_by_name("not_a_real_feature"), None);
        assert_eq!(
            AnalysisFeature::from_name("variance"),
            Some(AnalysisFeature::Variance)
        );
    }

    #[test]
    fn resolve_feature_ids_preserves_order_and_extracts() {
        // A model's column list (out of id-order, feat_-prefixed) resolves to a
        // core Vec<u16> in THAT order — the caller holds no zenanalyze type, so
        // the resolved ids survive even a multi-zenanalyze-version build.
        let cols = ["feat_edge_density", "feat_variance", "feat_uniformity"];
        let ids: Vec<u16> = resolve_feature_ids(&cols).expect("known features");
        assert_eq!(ids.len(), 3);
        assert_eq!(feature_name(ids[0]), Some("edge_density")); // order preserved
        assert_eq!(feature_name(ids[1]), Some("variance"));

        let buf = rgb8_image(96, 64);
        let slice = PixelSlice::new(&buf, 96, 64, 96 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let mut feats = [0.0f32; 3];
        assert!(feature_vector(slice, &ids, &mut feats)); // PixelSlice in, &[f32] out

        // defs version is a plain u32 to bake next to the model
        assert_eq!(feature_defs_version(), 1);

        // an unknown column => None (caller falls back), never a silent zero
        assert!(resolve_feature_ids(&["feat_variance", "feat_does_not_exist"]).is_none());
    }

    fn rgb8_image(w: u32, h: u32) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let p = ((y * w + x) * 3) as usize;
                v[p] = (x % 256) as u8;
                v[p + 1] = (y % 256) as u8;
                v[p + 2] = ((x + y) % 256) as u8;
            }
        }
        v
    }

    /// The full canonical vector equals the typed `get_f32` extraction, in the
    /// same order — the core API is a faithful facade, not a reimplementation.
    #[test]
    fn full_vector_matches_typed_extraction() {
        let (w, h) = (96u32, 64u32);
        let img = rgb8_image(w, h);
        let mut vec = vec![0.0f32; feature_count()];
        assert!(feature_vector_packed8_all(&img, w, h, 0, 3, &mut vec));

        let results = analyze_features_rgb8(&img, w, h, &AnalysisQuery::new(FeatureSet::SUPPORTED));
        for (i, f) in FeatureSet::SUPPORTED.iter().enumerate() {
            let typed = results.get_f32(f).unwrap_or(f32::NAN);
            let a = vec[i];
            assert!(
                (a.is_nan() && typed.is_nan()) || a == typed,
                "feature {} ({}): vector {a} != typed {typed}",
                f.id(),
                f.name()
            );
        }
    }

    /// Model-aligned extraction returns exactly the requested ids, in order.
    #[test]
    fn ids_aligned_returns_requested_order() {
        let (w, h) = (80u32, 48u32);
        let img = rgb8_image(w, h);
        let ids = [
            AnalysisFeature::EdgeDensity.id(),
            AnalysisFeature::Variance.id(),
            AnalysisFeature::Uniformity.id(),
        ];
        let mut out = [0.0f32; 3];
        assert!(feature_vector_packed8(&img, w, h, 0, 3, &ids, &mut out));

        let results = analyze_features_rgb8(&img, w, h, &AnalysisQuery::new(FeatureSet::SUPPORTED));
        for (i, &id) in ids.iter().enumerate() {
            let f = AnalysisFeature::from_u16(id).unwrap();
            assert_eq!(out[i], results.get_f32(f).unwrap(), "{}", f.name());
        }
    }

    /// Unknown ids → NaN (not an error); the rest still fill.
    #[test]
    fn unknown_id_is_nan() {
        let (w, h) = (32u32, 32u32);
        let img = rgb8_image(w, h);
        let ids = [AnalysisFeature::Variance.id(), 60000u16];
        let mut out = [1.0f32; 2];
        assert!(feature_vector_packed8(&img, w, h, 0, 3, &ids, &mut out));
        assert!(out[0].is_finite());
        assert!(out[1].is_nan());
    }

    /// Invalid input → `false`, never panics.
    #[test]
    fn invalid_input_returns_false() {
        let img = rgb8_image(16, 16);
        let mut out = vec![0.0f32; feature_count()];
        assert!(!feature_vector_packed8_all(&img, 16, 16, 0, 2, &mut out)); // bad channels
        assert!(!feature_vector_packed8_all(
            &img,
            16,
            16,
            0,
            3,
            &mut out[..1]
        )); // out too small
        assert!(!feature_vector_packed8_all(&img, 999, 999, 0, 3, &mut out)); // buffer too small for dims
        let ids = [0u16];
        assert!(!feature_vector_packed8(&img, 16, 16, 0, 3, &ids, &mut [])); // out < ids
    }

    /// RGBA8 input is accepted (alpha-aware features populate, no panic).
    #[test]
    fn rgba8_accepted() {
        let (w, h) = (40u32, 40u32);
        let mut img = vec![0u8; (w * h * 4) as usize];
        for (i, px) in img.chunks_exact_mut(4).enumerate() {
            px[0] = (i % 256) as u8;
            px[1] = 128;
            px[2] = 64;
            px[3] = if i % 2 == 0 { 255 } else { 100 }; // varying alpha
        }
        let mut out = vec![0.0f32; feature_count()];
        assert!(feature_vector_packed8_all(&img, w, h, 0, 4, &mut out));
    }

    /// Metadata is self-consistent and aligns the vector.
    #[test]
    fn metadata_consistency() {
        let n = feature_count();
        let mut ids = vec![0u16; n];
        assert_eq!(feature_ids(&mut ids), n);
        // ids match the SUPPORTED iteration order, names resolve, all supported.
        for (i, f) in FeatureSet::SUPPORTED.iter().enumerate() {
            assert_eq!(ids[i], f.id());
            assert_eq!(feature_name(ids[i]), Some(f.name()));
            assert!(feature_id_supported(ids[i]));
        }
        assert!(!feature_id_supported(60000));
        assert_eq!(feature_name(60000), None);
    }

    /// Explicit row stride (padded rows) gives the same result as packed.
    #[test]
    fn explicit_stride_matches_packed() {
        let (w, h) = (50u32, 30u32);
        let packed = rgb8_image(w, h);
        let pad = 12usize; // must keep stride pixel-aligned (multiple of bpp=3)
        let stride = w as usize * 3 + pad;
        let mut padded = vec![7u8; stride * h as usize];
        for y in 0..h as usize {
            let src = &packed[y * w as usize * 3..(y + 1) * w as usize * 3];
            padded[y * stride..y * stride + w as usize * 3].copy_from_slice(src);
        }
        let mut a = vec![0.0f32; feature_count()];
        let mut b = vec![0.0f32; feature_count()];
        assert!(feature_vector_packed8_all(&packed, w, h, 0, 3, &mut a));
        assert!(feature_vector_packed8_all(&padded, w, h, stride, 3, &mut b));
        for i in 0..a.len() {
            assert!(
                (a[i].is_nan() && b[i].is_nan()) || a[i] == b[i],
                "feature {i}: packed {} != strided {}",
                a[i],
                b[i]
            );
        }
    }

    /// The `PixelSlice` primary path equals the packed-8-bit convenience for
    /// the same RGB8 bytes (the convenience is a thin wrapper).
    #[test]
    fn pixelslice_path_matches_packed8() {
        let (w, h) = (72u32, 56u32);
        let img = rgb8_image(w, h);
        let slice =
            PixelSlice::new(&img, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
        let mut a = vec![0.0f32; feature_count()];
        let mut b = vec![0.0f32; feature_count()];
        assert!(feature_vector_all(slice, &mut a));
        assert!(feature_vector_packed8_all(&img, w, h, 0, 3, &mut b));
        for i in 0..a.len() {
            assert!(
                (a[i].is_nan() && b[i].is_nan()) || a[i] == b[i],
                "feature {i}"
            );
        }
    }

    /// Higher bit depth: a u16 RGB16 source is accepted (proving the API isn't
    /// 8-bit-bound — it inherits PixelSlice's bit-depth generality), and a
    /// lossless 8→16 promotion `(c<<8)|c` reads back the same SDR features as
    /// the 8-bit original (the converter narrows both to the same RGB8).
    #[test]
    fn u16_source_matches_8bit_promotion() {
        use feature::{AnalysisFeature, FeatureSet};
        let (w, h) = (64u32, 48u32);
        let img8 = rgb8_image(w, h);
        let u16s: Vec<u16> = img8.iter().map(|&c| ((c as u16) << 8) | c as u16).collect();
        let bytes16: Vec<u8> = u16s.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let s16 = PixelSlice::new(
            &bytes16,
            w,
            h,
            (w * 6) as usize,
            PixelDescriptor::RGB16_SRGB,
        )
        .unwrap();
        let s8 =
            PixelSlice::new(&img8, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();

        // SDR *content* features only. Exclude the depth tier (needs the `hdr`
        // feature) and `BitmapBytes` (source byte count — legitimately 2× for a
        // 16-bit source; it's format metadata, not content).
        let ids: Vec<u16> = FeatureSet::SUPPORTED
            .iter()
            .filter(|f| !feature::DEPTH_FEATURES.contains(*f) && *f != AnalysisFeature::BitmapBytes)
            .map(AnalysisFeature::id)
            .collect();
        let mut a = vec![0.0f32; ids.len()];
        let mut b = vec![0.0f32; ids.len()];
        assert!(feature_vector(s16, &ids, &mut a), "u16 RGB16 accepted");
        assert!(feature_vector(s8, &ids, &mut b));
        for i in 0..ids.len() {
            assert!(
                (a[i].is_nan() && b[i].is_nan())
                    || (a[i] - b[i]).abs() <= 1e-4 * a[i].abs().max(1.0),
                "feature id {}: u16 {} vs u8 {}",
                ids[i],
                a[i],
                b[i]
            );
        }
    }

    /// SDR content carried inside an HDR (PQ) envelope ≡ SDR-normal, through the
    /// **diffuse-white-normalized linear path** (`with_linear_light`). Same scene,
    /// SDR-white at the BT.2408 diffuse-white level (203 nits) in PQ. The
    /// normalization (anchor → 1.0, no tone curve) makes the Variance match by
    /// construction. The default gamma path does NOT hold — it treats the PQ peak
    /// (10000 nits) as display white and crushes the content to near-black; the
    /// second half of this test pins that contrast. (Other content features join
    /// as their linear kernels land; today the linear path normalizes Variance.)
    #[test]
    fn sdr_in_hdr_envelope_matches_sdr_normal() {
        use feature::{AnalysisFeature as AF, AnalysisQuery, FeatureSet};
        use linear_srgb::tf::{linear_to_pq, srgb_to_linear};
        use zenpixels::TransferFunction;

        let (w, h) = (96u32, 64u32);
        let sdr = rgb8_image(w, h);

        // The SAME content as PQ u16, SDR-white anchored at diffuse white.
        const DIFFUSE_NITS: f32 = 203.0;
        const PQ_PEAK_NITS: f32 = 10000.0;
        let pq16: Vec<u16> = sdr
            .iter()
            .map(|&c| {
                let lin = srgb_to_linear(c as f32 / 255.0); // SDR white → 1.0
                let pq = linear_to_pq(lin * DIFFUSE_NITS / PQ_PEAK_NITS);
                (pq.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
            })
            .collect();
        let pq_bytes: Vec<u8> = pq16.iter().flat_map(|&v| v.to_ne_bytes()).collect();

        let mk_sdr =
            || PixelSlice::new(&sdr, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
        let hdr_desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
        let mk_hdr = || PixelSlice::new(&pq_bytes, w, h, (w * 6) as usize, hdr_desc).unwrap();
        let var = |slice, ll: bool| {
            let q = AnalysisQuery::new(FeatureSet::just(AF::Variance));
            let q = if ll { q.with_linear_light(true) } else { q };
            analyze_features(slice, &q)
                .unwrap()
                .get_f32(AF::Variance)
                .unwrap()
        };

        // Normalized linear light: SDR-in-HDR ≡ SDR-normal.
        let (ls, lh) = (var(mk_sdr(), true), var(mk_hdr(), true));
        assert!(
            (ls - lh).abs() <= 0.02 * ls.abs().max(1.0),
            "linear-light: SDR-in-HDR Variance {lh} must match SDR-normal {ls}"
        );
        // Default gamma path: the PQ envelope collapses to near-black — the bug
        // the normalization fixes.
        let (gs, gh) = (var(mk_sdr(), false), var(mk_hdr(), false));
        assert!(
            gh < 0.2 * gs,
            "gamma path should crush the PQ envelope: hdr {gh} vs sdr {gs}"
        );
    }
}
