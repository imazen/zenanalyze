//! Public API surface: a stable, opaque, set-composable feature
//! interface that lets multiple codecs share one analysis pass.
//!
//! **Stability contract: there is no 0.2.x.** Every change in 0.1.x
//! is additive — new variants on `#[non_exhaustive]` enums, new
//! parallel functions, never a signature change to a shipped item.
//! See `CLAUDE.md`.
//!
//! Stability contract:
//! - [`AnalysisFeature`] discriminants are **immutable once shipped**.
//!   A retired variant keeps its `u16` slot forever; new variants get
//!   the next sequential number. Never reuse a number — that would
//!   silently break callers persisting [`FeatureSet`] bits to disk or
//!   wire.
//! - [`FeatureSet`] storage is opaque. Future versions may grow the
//!   backing bitset; the public ops are `const fn` set math.
//! - [`FeatureValue`] is `#[non_exhaustive]` — future structured
//!   feature types (small histograms, vectors) can land via additional
//!   variants without a major bump.
//!
//! Composition pattern:
//! ```ignore
//! // Each codec exposes the features it cares about.
//! const JPEG_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::EdgeDensity)
//!     .with(AnalysisFeature::DctCompressibilityY);
//! const WEBP_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::AlphaPresent)
//!     .with(AnalysisFeature::AlphaBimodalScore);
//!
//! // Orchestrator unions and runs once.
//! let needed = JPEG_FEATURES.union(WEBP_FEATURES);
//! // analyze(slice, &AnalysisQuery::new(needed)) → AnalysisResults …
//! ```
//!
//! Design note — typed-feature path (deferred):
//!
//! A separate sealed-trait `Feature` + `IsActive` system was
//! prototyped alongside this enum, with one zero-sized struct per
//! feature (`pub struct Variance(_)`). It would let callers write
//! `results.get_typed::<Variance>() -> Option<f32>` (no runtime
//! variant match) and would make `FeatureSet::just_typed::<F>()`
//! refuse to compile when `F` is a retired feature. Deferred to a
//! later round so we can settle the dynamic surface first; the macro
//! that generates the witnesses is straightforward when the enum
//! variants are stable. **Action item before adding it:** decide
//! whether the 30 extra public type names are worth the compile-time
//! gating, or whether `#[deprecated]` warnings on the enum variant +
//! `None` at runtime is enough.

// -------------------- features_table! macro --------------------------
//
// Single-source-of-truth for every analysis feature. This macro call
// generates the `AnalysisFeature` enum, the per-feature `id()` /
// `from_u16()` / `name()` impls, the internal `RawAnalysis` record,
// the `RawAnalysis::into_results` translator, and
// `FeatureSet::SUPPORTED` — all in lockstep.
//
// Editing this table updates the AnalysisFeature enum, the
// `id`/`from_u16`/`is_active`/`name` impls, the RawAnalysis dense
// record, the `into_results` translator, and FeatureSet::SUPPORTED —
// all in lockstep. Per-row format:
//   /// docstring
//   Variant = id : type => raw_field_name,
//
// The Rust `$ty` and the snake-case field name double as the
// AnalysisResults value type and the `name()` string. Adding a
// feature here is the only edit needed inside feature.rs.

macro_rules! features_table {
    (
        $(
            $(#[$variant_attr:meta])*
            $(@decl[$($decl_attr:meta),* $(,)?])?
            $variant:ident = $id:literal : $ty:ty => $field:ident
        ),* $(,)?
    ) => {
        /// Stable identifier for every feature zenanalyze can compute.
        ///
        /// Discriminants are explicit sequential `u16` values that
        /// **must never change** once shipped. Adding a feature uses
        /// the next free number; retiring a feature keeps its number
        /// reserved (the variant stays for ABI stability). Persisted
        /// [`FeatureSet`]s round-trip through major versions correctly.
        ///
        /// `#[non_exhaustive]` so adding variants is not a breaking
        /// change; callers must use a `_` arm in any `match`.
        #[non_exhaustive]
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
        #[repr(u16)]
        pub enum AnalysisFeature {
            $(
                $(#[$variant_attr])*
                $($(#[$decl_attr])*)?
                $variant = $id,
            )*
        }

        impl AnalysisFeature {
            /// The feature's stable `u16` discriminant.
            #[inline]
            pub const fn id(self) -> u16 { self as u16 }

            /// Reconstruct from its `u16` id.
            #[inline]
            pub const fn from_u16(v: u16) -> Option<Self> {
                match v {
                    $( $id => Some(Self::$variant), )*
                    _ => None,
                }
            }

            /// Snake-case name string.
            #[inline]
            pub const fn name(self) -> &'static str {
                match self {
                    $( Self::$variant => stringify!($field), )*
                    #[allow(unreachable_patterns)]
                    _ => "<unknown>",
                }
            }
        }

        /// Raw, dense record written by the SIMD tiers.
        #[allow(dead_code)]
        pub(crate) struct RawAnalysis {
            $( pub(crate) $field: $ty, )*
        }

        impl Default for RawAnalysis {
            fn default() -> Self {
                Self { $( $field: Default::default(), )* }
            }
        }

        impl RawAnalysis {
            pub(crate) fn into_results(
                self,
                requested: FeatureSet,
                geometry: ImageGeometry,
                source_descriptor: zenpixels::PixelDescriptor,
            ) -> AnalysisResults {
                let mut r = AnalysisResults::new(requested, geometry, source_descriptor);
                $(
                    if requested.contains(AnalysisFeature::$variant) {
                        r.set(AnalysisFeature::$variant, self.$field);
                    }
                )*
                r
            }
        }

        impl FeatureSet {
            pub const SUPPORTED: FeatureSet = {
                let mut s = FeatureSet::new();
                $( s = s.with(AnalysisFeature::$variant); )*
                s
            };
        }
    };
}

features_table! {
    // ---------------- Tier 1: sparse stripe scan (pixel_budget) ------
    /// `f32`. Luma variance on the BT.601 [0, 255] scale.
    Variance = 0 : f32 => variance,
    /// `f32`. Fraction of sampled interior pixels with `|∇L| > 20`.
    EdgeDensity = 1 : f32 => edge_density,
    /// `f32`. `√(Var(Cb) + Var(Cr))` over sampled pixels.
    ChromaComplexity = 2 : f32 => chroma_complexity,
    /// `f32`. Mean `|∇Cb|` over horizontally-paired sampled pixels.
    CbSharpness = 3 : f32 => cb_sharpness,
    /// `f32`. Mean `|∇Cr|` over horizontally-paired sampled pixels.
    CrSharpness = 4 : f32 => cr_sharpness,
    /// `f32`. Fraction of 8×8 blocks with luma variance < 25.
    Uniformity = 5 : f32 => uniformity,
    /// `f32`. Fraction of 8×8 blocks with R, G, B ranges all ≤ 4.
    FlatColorBlockRatio = 6 : f32 => flat_color_block_ratio,
    /// `f32`. Hasler-Süsstrunk M3 colourfulness.
    #[cfg(feature = "experimental")]
    Colourfulness = 7 : f32 => colourfulness,
    /// `f32`. Variance of the 5-tap Laplacian over sampled luma.
    #[cfg(feature = "experimental")]
    LaplacianVariance = 8 : f32 => laplacian_variance,
    /// `f32`. `log10(1 + max_var / max(1, mean_var))` over 8×8 blocks.
    #[cfg(feature = "experimental")]
    VarianceSpread = 9 : f32 => variance_spread,

    // ---------------- Palette: always full-scan ----------------------
    DistinctColorBins = 10 : u32 => distinct_color_bins,
    // id 11 reserved (was `DistinctColorBinsChao1`, removed pre-0.1.0).
    #[cfg(feature = "experimental")]
    @decl[deprecated(
        since = "0.1.0",
        note = "use DistinctColorBins or PaletteLog2Size"
    )]
    PaletteDensity = 12 : f32 => palette_density,

    // ---------------- Tier 2: per-channel per-axis chroma ------------
    CbHorizSharpness = 13 : f32 => cb_horiz_sharpness,
    CbVertSharpness = 14 : f32 => cb_vert_sharpness,
    CbPeakSharpness = 15 : f32 => cb_peak_sharpness,
    CrHorizSharpness = 16 : f32 => cr_horiz_sharpness,
    CrVertSharpness = 17 : f32 => cr_vert_sharpness,
    CrPeakSharpness = 18 : f32 => cr_peak_sharpness,

    // ---------------- Tier 3: DCT energy + entropy -------------------
    HighFreqEnergyRatio = 19 : f32 => high_freq_energy_ratio,
    LumaHistogramEntropy = 20 : f32 => luma_histogram_entropy,
    #[cfg(feature = "experimental")]
    DctCompressibilityY = 21 : f32 => dct_compressibility_y,
    #[cfg(feature = "experimental")]
    DctCompressibilityUV = 22 : f32 => dct_compressibility_uv,
    #[cfg(feature = "experimental")]
    PatchFraction = 23 : f32 => patch_fraction,

    // ---------------- Alpha ------------------------------------------
    AlphaPresent = 24 : bool => alpha_present,
    AlphaUsedFraction = 25 : f32 => alpha_used_fraction,
    AlphaBimodalScore = 26 : f32 => alpha_bimodal_score,

    // ids 27, 28, 29 reserved (TextLikelihood/ScreenContentLikelihood/NaturalLikelihood).
    // id 30 reserved (IndexedPaletteWidth, replaced by PaletteLog2Size=121).

    /// `bool`. Source fits in 256 distinct 5-bit RGB bins.
    #[cfg(feature = "experimental")]
    PaletteFitsIn256 = 31 : bool => palette_fits_in_256,

    // ---------------- Depth tier (HDR / bit-depth) -------------------
    #[cfg(feature = "hdr")]
    PeakLuminanceNits = 32 : f32 => peak_luminance_nits,
    #[cfg(feature = "hdr")]
    P99LuminanceNits = 33 : f32 => p99_luminance_nits,
    #[cfg(feature = "hdr")]
    HdrHeadroomStops = 34 : f32 => hdr_headroom_stops,
    #[cfg(feature = "hdr")]
    HdrPixelFraction = 35 : f32 => hdr_pixel_fraction,
    #[cfg(feature = "hdr")]
    WideGamutPeak = 36 : f32 => wide_gamut_peak,
    #[cfg(feature = "hdr")]
    WideGamutFraction = 37 : f32 => wide_gamut_fraction,
    #[cfg(feature = "hdr")]
    EffectiveBitDepth = 38 : u32 => effective_bit_depth,
    #[cfg(feature = "hdr")]
    HdrPresent = 39 : bool => hdr_present,
    /// `f32`. Fraction of pixels within sRGB gamut.
    #[cfg(feature = "experimental")]
    GrayscaleScore = 40 : f32 => grayscale_score,
    /// `f32`. Mean log10 AC energy over 8×8 blocks.
    #[cfg(feature = "experimental")]
    AqMapMean = 41 : f32 => aq_map_mean,
    /// `f32`. Stddev of per-block log10 AC energy.
    #[cfg(feature = "experimental")]
    AqMapStd = 42 : f32 => aq_map_std,
    /// `f32`. Luma noise floor estimate.
    #[cfg(feature = "experimental")]
    NoiseFloorY = 43 : f32 => noise_floor_y,
    /// `f32`. Chroma noise floor estimate.
    #[cfg(feature = "experimental")]
    NoiseFloorUV = 44 : f32 => noise_floor_uv,
    // id 45 reserved (LineArtScore, composite deleted).
    #[cfg(feature = "hdr")]
    GamutCoverageSrgb = 46 : f32 => gamut_coverage_srgb,
    #[cfg(feature = "hdr")]
    GamutCoverageP3 = 47 : f32 => gamut_coverage_p3,
    #[cfg(feature = "experimental")]
    GradientFraction = 48 : f32 => gradient_fraction,
    #[cfg(feature = "experimental")]
    SkinToneFraction = 49 : f32 => skin_tone_fraction,
    #[cfg(feature = "experimental")]
    EdgeSlopeStdev = 50 : f32 => edge_slope_stdev,
    // id 51 reserved (PatchFractionWht, removed pre-stabilization).
    #[cfg(feature = "experimental")]
    PatchFractionFast = 52 : f32 => patch_fraction_fast,
    #[cfg(feature = "experimental")]
    QuantSurvivalY = 53 : f32 => quant_survival_y,
    #[cfg(feature = "experimental")]
    QuantSurvivalUv = 54 : f32 => quant_survival_uv,
    IsGrayscale = 55 : bool => is_grayscale,
    PixelCount = 56 : u32 => pixel_count,
    LogPixels = 57 : f32 => log_pixels,
    MinDim = 58 : u32 => min_dim,
    MaxDim = 59 : u32 => max_dim,
    BitmapBytes = 60 : f32 => bitmap_bytes,
    AspectMinOverMax = 61 : f32 => aspect_min_over_max,
    @decl[deprecated(
        since = "0.1.0",
        note = "use AspectMinOverMax"
    )]
    LogAspectAbs = 62 : f32 => log_aspect_abs,
    BlockMisalignment8 = 63 : f32 => block_misalignment_8,
    // id 64 reserved (BlockMisalignment16).
    BlockMisalignment32 = 65 : f32 => block_misalignment_32,
    // id 66 reserved (BlockMisalignment64).
    ChannelCount = 67 : u32 => channel_count,

    // ---------------- AqMap percentiles -----------------------------
    #[cfg(feature = "experimental")]
    AqMapP50 = 68 : f32 => aq_map_p50,
    #[cfg(feature = "experimental")]
    AqMapP75 = 69 : f32 => aq_map_p75,
    #[cfg(feature = "experimental")]
    AqMapP90 = 70 : f32 => aq_map_p90,
    #[cfg(feature = "experimental")]
    AqMapP95 = 71 : f32 => aq_map_p95,
    #[cfg(feature = "experimental")]
    AqMapP99 = 72 : f32 => aq_map_p99,

    // ---------------- NoiseFloor percentiles ------------------------
    #[cfg(feature = "experimental")]
    NoiseFloorYP25 = 73 : f32 => noise_floor_y_p25,
    #[cfg(feature = "experimental")]
    NoiseFloorYP50 = 74 : f32 => noise_floor_y_p50,
    #[cfg(feature = "experimental")]
    NoiseFloorYP75 = 75 : f32 => noise_floor_y_p75,
    #[cfg(feature = "experimental")]
    NoiseFloorYP90 = 76 : f32 => noise_floor_y_p90,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP25 = 77 : f32 => noise_floor_uv_p25,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP50 = 78 : f32 => noise_floor_uv_p50,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP75 = 79 : f32 => noise_floor_uv_p75,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP90 = 80 : f32 => noise_floor_uv_p90,

    // ---------------- LaplacianVariance percentiles -----------------
    #[cfg(feature = "experimental")]
    LaplacianVarianceP50 = 81 : f32 => laplacian_variance_p50,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP75 = 82 : f32 => laplacian_variance_p75,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP90 = 83 : f32 => laplacian_variance_p90,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP99 = 84 : f32 => laplacian_variance_p99,
    #[cfg(feature = "experimental")]
    LaplacianVariancePeak = 85 : f32 => laplacian_variance_peak,

    // ---------------- QuantSurvival percentiles ---------------------
    #[cfg(feature = "experimental")]
    QuantSurvivalYP10 = 86 : f32 => quant_survival_y_p10,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP25 = 87 : f32 => quant_survival_y_p25,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP50 = 88 : f32 => quant_survival_y_p50,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP75 = 89 : f32 => quant_survival_y_p75,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP10 = 90 : f32 => quant_survival_uv_p10,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP25 = 91 : f32 => quant_survival_uv_p25,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP50 = 92 : f32 => quant_survival_uv_p50,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP75 = 93 : f32 => quant_survival_uv_p75,

    // ids 94..100 retired (log2/log10/sqrt pixel transforms).

    LogPaddedPixels8 = 101 : f32 => log_padded_pixels_8,
    LogPaddedPixels16 = 102 : f32 => log_padded_pixels_16,
    LogPaddedPixels32 = 103 : f32 => log_padded_pixels_32,
    // id 104 retired (LogPaddedPixels64).

    // ---------------- Low-tail percentile companions ----------------
    #[cfg(feature = "experimental")]
    LaplacianVarianceP1 = 105 : f32 => laplacian_variance_p1,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP5 = 106 : f32 => laplacian_variance_p5,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP10 = 107 : f32 => laplacian_variance_p10,
    #[cfg(feature = "experimental")]
    AqMapP1 = 108 : f32 => aq_map_p1,
    #[cfg(feature = "experimental")]
    AqMapP5 = 109 : f32 => aq_map_p5,
    #[cfg(feature = "experimental")]
    AqMapP10 = 110 : f32 => aq_map_p10,
    #[cfg(feature = "experimental")]
    NoiseFloorYP1 = 111 : f32 => noise_floor_y_p1,
    #[cfg(feature = "experimental")]
    NoiseFloorYP5 = 112 : f32 => noise_floor_y_p5,
    #[cfg(feature = "experimental")]
    NoiseFloorYP10 = 113 : f32 => noise_floor_y_p10,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP1 = 114 : f32 => quant_survival_y_p1,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP5 = 115 : f32 => quant_survival_y_p5,

    // ---------------- Distribution-shape (kurtosis) -----------------
    #[cfg(feature = "experimental")]
    LumaKurtosis = 116 : f32 => luma_kurtosis,

    // ids 117, 118, 119 retired (ChromaKurtosis/UniformitySmooth/FlatColorSmooth).

    #[cfg(feature = "experimental")]
    GradientFractionSmooth = 120 : f32 => gradient_fraction_smooth,
    #[cfg(feature = "experimental")]
    PaletteLog2Size = 121 : u32 => palette_log2_size,

    // ---------------- Dense percentile sweep (2026-05-07) -----------
    // IDs 122-211: 5-percentile-step grid for LaplacianVariance, AqMap,
    // NoiseFloorY/Uv, QuantSurvivalY/Uv. All experimental-gated.

    // --- LaplacianVariance dense percentiles (122-135) ---
    #[cfg(feature = "experimental")]
    LaplacianVarianceP15 = 122 : f32 => laplacian_variance_p15,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP20 = 123 : f32 => laplacian_variance_p20,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP25 = 124 : f32 => laplacian_variance_p25,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP30 = 125 : f32 => laplacian_variance_p30,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP35 = 126 : f32 => laplacian_variance_p35,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP40 = 127 : f32 => laplacian_variance_p40,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP45 = 128 : f32 => laplacian_variance_p45,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP55 = 129 : f32 => laplacian_variance_p55,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP60 = 130 : f32 => laplacian_variance_p60,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP65 = 131 : f32 => laplacian_variance_p65,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP70 = 132 : f32 => laplacian_variance_p70,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP80 = 133 : f32 => laplacian_variance_p80,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP85 = 134 : f32 => laplacian_variance_p85,
    #[cfg(feature = "experimental")]
    LaplacianVarianceP95 = 135 : f32 => laplacian_variance_p95,

    // --- AqMap dense percentiles (136-148) ---
    #[cfg(feature = "experimental")]
    AqMapP15 = 136 : f32 => aq_map_p15,
    #[cfg(feature = "experimental")]
    AqMapP20 = 137 : f32 => aq_map_p20,
    #[cfg(feature = "experimental")]
    AqMapP25 = 138 : f32 => aq_map_p25,
    #[cfg(feature = "experimental")]
    AqMapP30 = 139 : f32 => aq_map_p30,
    #[cfg(feature = "experimental")]
    AqMapP35 = 140 : f32 => aq_map_p35,
    #[cfg(feature = "experimental")]
    AqMapP40 = 141 : f32 => aq_map_p40,
    #[cfg(feature = "experimental")]
    AqMapP45 = 142 : f32 => aq_map_p45,
    #[cfg(feature = "experimental")]
    AqMapP55 = 143 : f32 => aq_map_p55,
    #[cfg(feature = "experimental")]
    AqMapP60 = 144 : f32 => aq_map_p60,
    #[cfg(feature = "experimental")]
    AqMapP65 = 145 : f32 => aq_map_p65,
    #[cfg(feature = "experimental")]
    AqMapP70 = 146 : f32 => aq_map_p70,
    #[cfg(feature = "experimental")]
    AqMapP80 = 147 : f32 => aq_map_p80,
    #[cfg(feature = "experimental")]
    AqMapP85 = 148 : f32 => aq_map_p85,

    // --- NoiseFloorY dense percentiles (149-162) ---
    #[cfg(feature = "experimental")]
    NoiseFloorYP15 = 149 : f32 => noise_floor_y_p15,
    #[cfg(feature = "experimental")]
    NoiseFloorYP20 = 150 : f32 => noise_floor_y_p20,
    #[cfg(feature = "experimental")]
    NoiseFloorYP30 = 151 : f32 => noise_floor_y_p30,
    #[cfg(feature = "experimental")]
    NoiseFloorYP35 = 152 : f32 => noise_floor_y_p35,
    #[cfg(feature = "experimental")]
    NoiseFloorYP40 = 153 : f32 => noise_floor_y_p40,
    #[cfg(feature = "experimental")]
    NoiseFloorYP45 = 154 : f32 => noise_floor_y_p45,
    #[cfg(feature = "experimental")]
    NoiseFloorYP55 = 155 : f32 => noise_floor_y_p55,
    #[cfg(feature = "experimental")]
    NoiseFloorYP60 = 156 : f32 => noise_floor_y_p60,
    #[cfg(feature = "experimental")]
    NoiseFloorYP65 = 157 : f32 => noise_floor_y_p65,
    #[cfg(feature = "experimental")]
    NoiseFloorYP70 = 158 : f32 => noise_floor_y_p70,
    #[cfg(feature = "experimental")]
    NoiseFloorYP80 = 159 : f32 => noise_floor_y_p80,
    #[cfg(feature = "experimental")]
    NoiseFloorYP85 = 160 : f32 => noise_floor_y_p85,
    #[cfg(feature = "experimental")]
    NoiseFloorYP95 = 161 : f32 => noise_floor_y_p95,
    #[cfg(feature = "experimental")]
    NoiseFloorYP99 = 162 : f32 => noise_floor_y_p99,

    // --- NoiseFloorUv dense percentiles (163-179) ---
    #[cfg(feature = "experimental")]
    NoiseFloorUvP1 = 163 : f32 => noise_floor_uv_p1,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP5 = 164 : f32 => noise_floor_uv_p5,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP10 = 165 : f32 => noise_floor_uv_p10,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP15 = 166 : f32 => noise_floor_uv_p15,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP20 = 167 : f32 => noise_floor_uv_p20,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP30 = 168 : f32 => noise_floor_uv_p30,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP35 = 169 : f32 => noise_floor_uv_p35,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP40 = 170 : f32 => noise_floor_uv_p40,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP45 = 171 : f32 => noise_floor_uv_p45,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP55 = 172 : f32 => noise_floor_uv_p55,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP60 = 173 : f32 => noise_floor_uv_p60,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP65 = 174 : f32 => noise_floor_uv_p65,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP70 = 175 : f32 => noise_floor_uv_p70,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP80 = 176 : f32 => noise_floor_uv_p80,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP85 = 177 : f32 => noise_floor_uv_p85,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP95 = 178 : f32 => noise_floor_uv_p95,
    #[cfg(feature = "experimental")]
    NoiseFloorUvP99 = 179 : f32 => noise_floor_uv_p99,

    // --- QuantSurvivalY dense percentiles (180-194) ---
    #[cfg(feature = "experimental")]
    QuantSurvivalYP15 = 180 : f32 => quant_survival_y_p15,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP20 = 181 : f32 => quant_survival_y_p20,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP30 = 182 : f32 => quant_survival_y_p30,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP35 = 183 : f32 => quant_survival_y_p35,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP40 = 184 : f32 => quant_survival_y_p40,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP45 = 185 : f32 => quant_survival_y_p45,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP55 = 186 : f32 => quant_survival_y_p55,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP60 = 187 : f32 => quant_survival_y_p60,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP65 = 188 : f32 => quant_survival_y_p65,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP70 = 189 : f32 => quant_survival_y_p70,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP80 = 190 : f32 => quant_survival_y_p80,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP85 = 191 : f32 => quant_survival_y_p85,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP90 = 192 : f32 => quant_survival_y_p90,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP95 = 193 : f32 => quant_survival_y_p95,
    #[cfg(feature = "experimental")]
    QuantSurvivalYP99 = 194 : f32 => quant_survival_y_p99,

    // --- QuantSurvivalUv dense percentiles (195-211) ---
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP1 = 195 : f32 => quant_survival_uv_p1,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP5 = 196 : f32 => quant_survival_uv_p5,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP15 = 197 : f32 => quant_survival_uv_p15,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP20 = 198 : f32 => quant_survival_uv_p20,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP30 = 199 : f32 => quant_survival_uv_p30,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP35 = 200 : f32 => quant_survival_uv_p35,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP40 = 201 : f32 => quant_survival_uv_p40,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP45 = 202 : f32 => quant_survival_uv_p45,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP55 = 203 : f32 => quant_survival_uv_p55,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP60 = 204 : f32 => quant_survival_uv_p60,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP65 = 205 : f32 => quant_survival_uv_p65,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP70 = 206 : f32 => quant_survival_uv_p70,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP80 = 207 : f32 => quant_survival_uv_p80,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP85 = 208 : f32 => quant_survival_uv_p85,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP90 = 209 : f32 => quant_survival_uv_p90,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP95 = 210 : f32 => quant_survival_uv_p95,
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP99 = 211 : f32 => quant_survival_uv_p99,
}

/// A scalar feature value.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FeatureValue {
    F32(f32),
    U32(u32),
    U64(u64),
    Bool(bool),
}

impl FeatureValue {
    #[inline]
    pub const fn as_f32(self) -> Option<f32> {
        match self { Self::F32(x) => Some(x), _ => None }
    }
    #[inline]
    pub const fn as_u32(self) -> Option<u32> {
        match self { Self::U32(x) => Some(x), _ => None }
    }
    #[inline]
    pub const fn as_u64(self) -> Option<u64> {
        match self { Self::U64(x) => Some(x), _ => None }
    }
    #[inline]
    pub const fn as_bool(self) -> Option<bool> {
        match self { Self::Bool(x) => Some(x), _ => None }
    }
    #[inline]
    pub fn to_f32(self) -> f32 {
        match self {
            Self::F32(x) => x,
            Self::U32(x) => x as f32,
            Self::U64(x) => x as f64 as f32,
            Self::Bool(false) => 0.0,
            Self::Bool(true) => 1.0,
        }
    }
}

impl From<f32> for FeatureValue { fn from(x: f32) -> Self { Self::F32(x) } }
impl From<u32> for FeatureValue { fn from(x: u32) -> Self { Self::U32(x) } }
impl From<u64> for FeatureValue { fn from(x: u64) -> Self { Self::U64(x) } }
impl From<bool> for FeatureValue { fn from(x: bool) -> Self { Self::Bool(x) } }

/// Opaque set of [`AnalysisFeature`]s, supporting `const fn` set math.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FeatureSet {
    bits: [u64; 4],
}

impl FeatureSet {
    pub const fn new() -> Self { Self { bits: [0; 4] } }
    pub const fn just(f: AnalysisFeature) -> Self { Self::new().with(f) }
    pub const fn with(mut self, f: AnalysisFeature) -> Self {
        let id = f as u16 as usize;
        self.bits[id >> 6] |= 1u64 << (id & 63);
        self
    }
    pub const fn without(mut self, f: AnalysisFeature) -> Self {
        let id = f as u16 as usize;
        self.bits[id >> 6] &= !(1u64 << (id & 63));
        self
    }
    pub const fn union(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 { self.bits[i] |= other.bits[i]; i += 1; }
        self
    }
    pub const fn intersect(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 { self.bits[i] &= other.bits[i]; i += 1; }
        self
    }
    pub const fn difference(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 { self.bits[i] &= !other.bits[i]; i += 1; }
        self
    }
    pub const fn intersects(self, other: Self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] & other.bits[i] != 0 { return true; }
            i += 1;
        }
        false
    }
    pub const fn contains_all(self, other: Self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] & other.bits[i] != other.bits[i] { return false; }
            i += 1;
        }
        true
    }
    pub const fn contains(self, f: AnalysisFeature) -> bool {
        let id = f as u16 as usize;
        (self.bits[id >> 6] >> (id & 63)) & 1 != 0
    }
    pub const fn is_empty(self) -> bool {
        let mut i = 0;
        while i < 4 { if self.bits[i] != 0 { return false; } i += 1; }
        true
    }
    pub const fn len(self) -> u32 {
        let mut total = 0;
        let mut i = 0;
        while i < 4 { total += self.bits[i].count_ones(); i += 1; }
        total
    }
    pub fn iter(self) -> FeatureSetIter {
        FeatureSetIter { bits: self.bits, next_id: 0 }
    }
    pub const ZENJPEG_PICKER_V1_1: Self = Self::new()
        .with(AnalysisFeature::Variance)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::Uniformity)
        .with(AnalysisFeature::ChromaComplexity)
        .with(AnalysisFeature::CbSharpness)
        .with(AnalysisFeature::CrSharpness)
        .with(AnalysisFeature::HighFreqEnergyRatio)
        .with(AnalysisFeature::LumaHistogramEntropy);
}

pub struct FeatureSetIter { bits: [u64; 4], next_id: u16 }

impl Iterator for FeatureSetIter {
    type Item = AnalysisFeature;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let id = self.next_id as usize;
            if id >= 256 { return None; }
            let bit = (self.bits[id >> 6] >> (id & 63)) & 1;
            self.next_id += 1;
            if bit == 1 && let Some(f) = AnalysisFeature::from_u16(id as u16) {
                return Some(f);
            }
        }
    }
}

impl IntoIterator for FeatureSet {
    type Item = AnalysisFeature;
    type IntoIter = FeatureSetIter;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl Default for FeatureSet { fn default() -> Self { Self::new() } }

// --- Tier-membership constants ---

#[allow(deprecated)]
pub(crate) const PALETTE_FULL_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::DistinctColorBins);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::PaletteDensity);
        s = s.with(AnalysisFeature::GrayscaleScore);
        s = s.with(AnalysisFeature::PaletteLog2Size);
    }
    s
};

#[allow(unused_mut, unused_assignments)]
pub(crate) const PALETTE_QUICK_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    { s = s.with(AnalysisFeature::PaletteFitsIn256); }
    s
};

pub(crate) const PALETTE_FEATURES: FeatureSet = PALETTE_FULL_FEATURES.union(PALETTE_QUICK_FEATURES);

#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_EXTRAS_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::Variance);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::Colourfulness);
        s = s.with(AnalysisFeature::LaplacianVariance);
        s = s.with(AnalysisFeature::SkinToneFraction);
        s = s.with(AnalysisFeature::EdgeSlopeStdev);
    }
    s
};

#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_FULL_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::Variance);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::Colourfulness);
        s = s.with(AnalysisFeature::LaplacianVariance);
        s = s.with(AnalysisFeature::EdgeSlopeStdev);
    }
    s
};

#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_SKIN_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    { s = s.with(AnalysisFeature::SkinToneFraction); }
    s
};

pub(crate) const TIER2_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::CbHorizSharpness)
    .with(AnalysisFeature::CbVertSharpness)
    .with(AnalysisFeature::CbPeakSharpness)
    .with(AnalysisFeature::CrHorizSharpness)
    .with(AnalysisFeature::CrVertSharpness)
    .with(AnalysisFeature::CrPeakSharpness);

#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER3_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::HighFreqEnergyRatio);
    s = s.with(AnalysisFeature::LumaHistogramEntropy);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::DctCompressibilityY);
        s = s.with(AnalysisFeature::DctCompressibilityUV);
        s = s.with(AnalysisFeature::PatchFraction);
        s = s.with(AnalysisFeature::AqMapMean);
        s = s.with(AnalysisFeature::AqMapStd);
        s = s.with(AnalysisFeature::NoiseFloorY);
        s = s.with(AnalysisFeature::NoiseFloorUV);
        s = s.with(AnalysisFeature::GradientFraction);
        s = s.with(AnalysisFeature::PatchFractionFast);
        s = s.with(AnalysisFeature::QuantSurvivalY);
        s = s.with(AnalysisFeature::QuantSurvivalUv);
        s = s.with(AnalysisFeature::AqMapP50);
        s = s.with(AnalysisFeature::AqMapP75);
        s = s.with(AnalysisFeature::AqMapP90);
        s = s.with(AnalysisFeature::AqMapP95);
        s = s.with(AnalysisFeature::AqMapP99);
        s = s.with(AnalysisFeature::NoiseFloorYP25);
        s = s.with(AnalysisFeature::NoiseFloorYP50);
        s = s.with(AnalysisFeature::NoiseFloorYP75);
        s = s.with(AnalysisFeature::NoiseFloorYP90);
        s = s.with(AnalysisFeature::NoiseFloorUvP25);
        s = s.with(AnalysisFeature::NoiseFloorUvP50);
        s = s.with(AnalysisFeature::NoiseFloorUvP75);
        s = s.with(AnalysisFeature::NoiseFloorUvP90);
        s = s.with(AnalysisFeature::QuantSurvivalYP10);
        s = s.with(AnalysisFeature::QuantSurvivalYP25);
        s = s.with(AnalysisFeature::QuantSurvivalYP50);
        s = s.with(AnalysisFeature::QuantSurvivalYP75);
        s = s.with(AnalysisFeature::QuantSurvivalUvP10);
        s = s.with(AnalysisFeature::QuantSurvivalUvP25);
        s = s.with(AnalysisFeature::QuantSurvivalUvP50);
        s = s.with(AnalysisFeature::QuantSurvivalUvP75);
        // Dense sweep variants (IDs 122-211)
        s = s.with(AnalysisFeature::AqMapP1);
        s = s.with(AnalysisFeature::AqMapP5);
        s = s.with(AnalysisFeature::AqMapP10);
        s = s.with(AnalysisFeature::AqMapP15);
        s = s.with(AnalysisFeature::AqMapP20);
        s = s.with(AnalysisFeature::AqMapP25);
        s = s.with(AnalysisFeature::AqMapP30);
        s = s.with(AnalysisFeature::AqMapP35);
        s = s.with(AnalysisFeature::AqMapP40);
        s = s.with(AnalysisFeature::AqMapP45);
        s = s.with(AnalysisFeature::AqMapP55);
        s = s.with(AnalysisFeature::AqMapP60);
        s = s.with(AnalysisFeature::AqMapP65);
        s = s.with(AnalysisFeature::AqMapP70);
        s = s.with(AnalysisFeature::AqMapP80);
        s = s.with(AnalysisFeature::AqMapP85);
        s = s.with(AnalysisFeature::NoiseFloorYP1);
        s = s.with(AnalysisFeature::NoiseFloorYP5);
        s = s.with(AnalysisFeature::NoiseFloorYP10);
        s = s.with(AnalysisFeature::NoiseFloorYP15);
        s = s.with(AnalysisFeature::NoiseFloorYP20);
        s = s.with(AnalysisFeature::NoiseFloorYP30);
        s = s.with(AnalysisFeature::NoiseFloorYP35);
        s = s.with(AnalysisFeature::NoiseFloorYP40);
        s = s.with(AnalysisFeature::NoiseFloorYP45);
        s = s.with(AnalysisFeature::NoiseFloorYP55);
        s = s.with(AnalysisFeature::NoiseFloorYP60);
        s = s.with(AnalysisFeature::NoiseFloorYP65);
        s = s.with(AnalysisFeature::NoiseFloorYP70);
        s = s.with(AnalysisFeature::NoiseFloorYP80);
        s = s.with(AnalysisFeature::NoiseFloorYP85);
        s = s.with(AnalysisFeature::NoiseFloorYP95);
        s = s.with(AnalysisFeature::NoiseFloorYP99);
        s = s.with(AnalysisFeature::NoiseFloorUvP1);
        s = s.with(AnalysisFeature::NoiseFloorUvP5);
        s = s.with(AnalysisFeature::NoiseFloorUvP10);
        s = s.with(AnalysisFeature::NoiseFloorUvP15);
        s = s.with(AnalysisFeature::NoiseFloorUvP20);
        s = s.with(AnalysisFeature::NoiseFloorUvP30);
        s = s.with(AnalysisFeature::NoiseFloorUvP35);
        s = s.with(AnalysisFeature::NoiseFloorUvP40);
        s = s.with(AnalysisFeature::NoiseFloorUvP45);
        s = s.with(AnalysisFeature::NoiseFloorUvP55);
        s = s.with(AnalysisFeature::NoiseFloorUvP60);
        s = s.with(AnalysisFeature::NoiseFloorUvP65);
        s = s.with(AnalysisFeature::NoiseFloorUvP70);
        s = s.with(AnalysisFeature::NoiseFloorUvP80);
        s = s.with(AnalysisFeature::NoiseFloorUvP85);
        s = s.with(AnalysisFeature::NoiseFloorUvP95);
        s = s.with(AnalysisFeature::NoiseFloorUvP99);
        s = s.with(AnalysisFeature::QuantSurvivalYP1);
        s = s.with(AnalysisFeature::QuantSurvivalYP5);
        s = s.with(AnalysisFeature::QuantSurvivalYP15);
        s = s.with(AnalysisFeature::QuantSurvivalYP20);
        s = s.with(AnalysisFeature::QuantSurvivalYP30);
        s = s.with(AnalysisFeature::QuantSurvivalYP35);
        s = s.with(AnalysisFeature::QuantSurvivalYP40);
        s = s.with(AnalysisFeature::QuantSurvivalYP45);
        s = s.with(AnalysisFeature::QuantSurvivalYP55);
        s = s.with(AnalysisFeature::QuantSurvivalYP60);
        s = s.with(AnalysisFeature::QuantSurvivalYP65);
        s = s.with(AnalysisFeature::QuantSurvivalYP70);
        s = s.with(AnalysisFeature::QuantSurvivalYP80);
        s = s.with(AnalysisFeature::QuantSurvivalYP85);
        s = s.with(AnalysisFeature::QuantSurvivalYP90);
        s = s.with(AnalysisFeature::QuantSurvivalYP95);
        s = s.with(AnalysisFeature::QuantSurvivalYP99);
        s = s.with(AnalysisFeature::QuantSurvivalUvP1);
        s = s.with(AnalysisFeature::QuantSurvivalUvP5);
        s = s.with(AnalysisFeature::QuantSurvivalUvP15);
        s = s.with(AnalysisFeature::QuantSurvivalUvP20);
        s = s.with(AnalysisFeature::QuantSurvivalUvP30);
        s = s.with(AnalysisFeature::QuantSurvivalUvP35);
        s = s.with(AnalysisFeature::QuantSurvivalUvP40);
        s = s.with(AnalysisFeature::QuantSurvivalUvP45);
        s = s.with(AnalysisFeature::QuantSurvivalUvP55);
        s = s.with(AnalysisFeature::QuantSurvivalUvP60);
        s = s.with(AnalysisFeature::QuantSurvivalUvP65);
        s = s.with(AnalysisFeature::QuantSurvivalUvP70);
        s = s.with(AnalysisFeature::QuantSurvivalUvP80);
        s = s.with(AnalysisFeature::QuantSurvivalUvP85);
        s = s.with(AnalysisFeature::QuantSurvivalUvP90);
        s = s.with(AnalysisFeature::QuantSurvivalUvP95);
        s = s.with(AnalysisFeature::QuantSurvivalUvP99);
    }
    s
};

#[allow(unused_mut, unused_assignments)]
pub(crate) const DCT_NEEDED_BY: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::HighFreqEnergyRatio);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::DctCompressibilityY);
        s = s.with(AnalysisFeature::DctCompressibilityUV);
        s = s.with(AnalysisFeature::PatchFraction);
        s = s.with(AnalysisFeature::AqMapMean);
        s = s.with(AnalysisFeature::AqMapStd);
        s = s.with(AnalysisFeature::NoiseFloorY);
        s = s.with(AnalysisFeature::NoiseFloorUV);
        s = s.with(AnalysisFeature::GradientFraction);
        s = s.with(AnalysisFeature::PatchFractionFast);
        s = s.with(AnalysisFeature::QuantSurvivalY);
        s = s.with(AnalysisFeature::QuantSurvivalUv);
        s = s.with(AnalysisFeature::AqMapP50);
        s = s.with(AnalysisFeature::AqMapP75);
        s = s.with(AnalysisFeature::AqMapP90);
        s = s.with(AnalysisFeature::AqMapP95);
        s = s.with(AnalysisFeature::AqMapP99);
        s = s.with(AnalysisFeature::NoiseFloorYP25);
        s = s.with(AnalysisFeature::NoiseFloorYP50);
        s = s.with(AnalysisFeature::NoiseFloorYP75);
        s = s.with(AnalysisFeature::NoiseFloorYP90);
        s = s.with(AnalysisFeature::NoiseFloorUvP25);
        s = s.with(AnalysisFeature::NoiseFloorUvP50);
        s = s.with(AnalysisFeature::NoiseFloorUvP75);
        s = s.with(AnalysisFeature::NoiseFloorUvP90);
        s = s.with(AnalysisFeature::QuantSurvivalYP10);
        s = s.with(AnalysisFeature::QuantSurvivalYP25);
        s = s.with(AnalysisFeature::QuantSurvivalYP50);
        s = s.with(AnalysisFeature::QuantSurvivalYP75);
        s = s.with(AnalysisFeature::QuantSurvivalUvP10);
        s = s.with(AnalysisFeature::QuantSurvivalUvP25);
        s = s.with(AnalysisFeature::QuantSurvivalUvP50);
        s = s.with(AnalysisFeature::QuantSurvivalUvP75);
        // Dense sweep variants
        s = s.with(AnalysisFeature::AqMapP1);
        s = s.with(AnalysisFeature::AqMapP5);
        s = s.with(AnalysisFeature::AqMapP10);
        s = s.with(AnalysisFeature::AqMapP15);
        s = s.with(AnalysisFeature::AqMapP20);
        s = s.with(AnalysisFeature::AqMapP25);
        s = s.with(AnalysisFeature::AqMapP30);
        s = s.with(AnalysisFeature::AqMapP35);
        s = s.with(AnalysisFeature::AqMapP40);
        s = s.with(AnalysisFeature::AqMapP45);
        s = s.with(AnalysisFeature::AqMapP55);
        s = s.with(AnalysisFeature::AqMapP60);
        s = s.with(AnalysisFeature::AqMapP65);
        s = s.with(AnalysisFeature::AqMapP70);
        s = s.with(AnalysisFeature::AqMapP80);
        s = s.with(AnalysisFeature::AqMapP85);
        s = s.with(AnalysisFeature::NoiseFloorYP1);
        s = s.with(AnalysisFeature::NoiseFloorYP5);
        s = s.with(AnalysisFeature::NoiseFloorYP10);
        s = s.with(AnalysisFeature::NoiseFloorYP15);
        s = s.with(AnalysisFeature::NoiseFloorYP20);
        s = s.with(AnalysisFeature::NoiseFloorYP30);
        s = s.with(AnalysisFeature::NoiseFloorYP35);
        s = s.with(AnalysisFeature::NoiseFloorYP40);
        s = s.with(AnalysisFeature::NoiseFloorYP45);
        s = s.with(AnalysisFeature::NoiseFloorYP55);
        s = s.with(AnalysisFeature::NoiseFloorYP60);
        s = s.with(AnalysisFeature::NoiseFloorYP65);
        s = s.with(AnalysisFeature::NoiseFloorYP70);
        s = s.with(AnalysisFeature::NoiseFloorYP80);
        s = s.with(AnalysisFeature::NoiseFloorYP85);
        s = s.with(AnalysisFeature::NoiseFloorYP95);
        s = s.with(AnalysisFeature::NoiseFloorYP99);
        s = s.with(AnalysisFeature::NoiseFloorUvP1);
        s = s.with(AnalysisFeature::NoiseFloorUvP5);
        s = s.with(AnalysisFeature::NoiseFloorUvP10);
        s = s.with(AnalysisFeature::NoiseFloorUvP15);
        s = s.with(AnalysisFeature::NoiseFloorUvP20);
        s = s.with(AnalysisFeature::NoiseFloorUvP30);
        s = s.with(AnalysisFeature::NoiseFloorUvP35);
        s = s.with(AnalysisFeature::NoiseFloorUvP40);
        s = s.with(AnalysisFeature::NoiseFloorUvP45);
        s = s.with(AnalysisFeature::NoiseFloorUvP55);
        s = s.with(AnalysisFeature::NoiseFloorUvP60);
        s = s.with(AnalysisFeature::NoiseFloorUvP65);
        s = s.with(AnalysisFeature::NoiseFloorUvP70);
        s = s.with(AnalysisFeature::NoiseFloorUvP80);
        s = s.with(AnalysisFeature::NoiseFloorUvP85);
        s = s.with(AnalysisFeature::NoiseFloorUvP95);
        s = s.with(AnalysisFeature::NoiseFloorUvP99);
        s = s.with(AnalysisFeature::QuantSurvivalYP1);
        s = s.with(AnalysisFeature::QuantSurvivalYP5);
        s = s.with(AnalysisFeature::QuantSurvivalYP15);
        s = s.with(AnalysisFeature::QuantSurvivalYP20);
        s = s.with(AnalysisFeature::QuantSurvivalYP30);
        s = s.with(AnalysisFeature::QuantSurvivalYP35);
        s = s.with(AnalysisFeature::QuantSurvivalYP40);
        s = s.with(AnalysisFeature::QuantSurvivalYP45);
        s = s.with(AnalysisFeature::QuantSurvivalYP55);
        s = s.with(AnalysisFeature::QuantSurvivalYP60);
        s = s.with(AnalysisFeature::QuantSurvivalYP65);
        s = s.with(AnalysisFeature::QuantSurvivalYP70);
        s = s.with(AnalysisFeature::QuantSurvivalYP80);
        s = s.with(AnalysisFeature::QuantSurvivalYP85);
        s = s.with(AnalysisFeature::QuantSurvivalYP90);
        s = s.with(AnalysisFeature::QuantSurvivalYP95);
        s = s.with(AnalysisFeature::QuantSurvivalYP99);
        s = s.with(AnalysisFeature::QuantSurvivalUvP1);
        s = s.with(AnalysisFeature::QuantSurvivalUvP5);
        s = s.with(AnalysisFeature::QuantSurvivalUvP15);
        s = s.with(AnalysisFeature::QuantSurvivalUvP20);
        s = s.with(AnalysisFeature::QuantSurvivalUvP30);
        s = s.with(AnalysisFeature::QuantSurvivalUvP35);
        s = s.with(AnalysisFeature::QuantSurvivalUvP40);
        s = s.with(AnalysisFeature::QuantSurvivalUvP45);
        s = s.with(AnalysisFeature::QuantSurvivalUvP55);
        s = s.with(AnalysisFeature::QuantSurvivalUvP60);
        s = s.with(AnalysisFeature::QuantSurvivalUvP65);
        s = s.with(AnalysisFeature::QuantSurvivalUvP70);
        s = s.with(AnalysisFeature::QuantSurvivalUvP80);
        s = s.with(AnalysisFeature::QuantSurvivalUvP85);
        s = s.with(AnalysisFeature::QuantSurvivalUvP90);
        s = s.with(AnalysisFeature::QuantSurvivalUvP95);
        s = s.with(AnalysisFeature::QuantSurvivalUvP99);
    }
    s
};

pub(crate) const ALPHA_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction)
    .with(AnalysisFeature::AlphaBimodalScore);

#[allow(unused_mut, unused_assignments)]
pub(crate) const DEPTH_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "hdr")]
    {
        s = s.with(AnalysisFeature::PeakLuminanceNits);
        s = s.with(AnalysisFeature::P99LuminanceNits);
        s = s.with(AnalysisFeature::HdrHeadroomStops);
        s = s.with(AnalysisFeature::HdrPixelFraction);
        s = s.with(AnalysisFeature::WideGamutPeak);
        s = s.with(AnalysisFeature::WideGamutFraction);
        s = s.with(AnalysisFeature::EffectiveBitDepth);
        s = s.with(AnalysisFeature::HdrPresent);
        s = s.with(AnalysisFeature::GamutCoverageSrgb);
        s = s.with(AnalysisFeature::GamutCoverageP3);
    }
    s
};

#[allow(unused_mut, unused_assignments)]
pub(crate) const DERIVED_FEATURES: FeatureSet = FeatureSet::new();

pub(crate) const T3_NEEDED_BY: FeatureSet = TIER3_FEATURES;
pub(crate) const PAL_NEEDED_BY: FeatureSet = PALETTE_FEATURES;

// --- AnalysisQuery ---

/// A request to the analyzer: what features to compute.
#[derive(Clone, Debug)]
pub struct AnalysisQuery {
    features: FeatureSet,
}

impl AnalysisQuery {
    pub const fn new(features: FeatureSet) -> Self { Self { features } }
    pub const fn features(&self) -> FeatureSet { self.features }
}

pub(crate) const DEFAULT_PIXEL_BUDGET: usize = 500_000;
pub(crate) const DEFAULT_HF_MAX_BLOCKS: usize = 1024;

#[doc(hidden)]
impl AnalysisQuery {
    pub fn __internal_with_overrides(
        features: FeatureSet,
        pixel_budget: usize,
        hf_max_blocks: usize,
    ) -> InternalQuery {
        InternalQuery { features, pixel_budget, hf_max_blocks }
    }
}

#[doc(hidden)]
pub struct InternalQuery {
    pub(crate) features: FeatureSet,
    pub(crate) pixel_budget: usize,
    pub(crate) hf_max_blocks: usize,
}

// --- ImageGeometry ---

#[derive(Copy, Clone, Debug)]
pub struct ImageGeometry { width: u32, height: u32 }

impl ImageGeometry {
    pub(crate) const fn new(width: u32, height: u32) -> Self { Self { width, height } }
    pub const fn width(self) -> u32 { self.width }
    pub const fn height(self) -> u32 { self.height }
    pub const fn pixels(self) -> u64 { self.width as u64 * self.height as u64 }
    pub fn megapixels(self) -> f32 { self.pixels() as f32 / 1_000_000.0 }
    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 { 0.0 } else { self.width as f32 / self.height as f32 }
    }
}

// --- AnalysisResults ---

/// Opaque container for one analysis pass's outputs.
pub struct AnalysisResults {
    requested: FeatureSet,
    geometry: ImageGeometry,
    source_descriptor: zenpixels::PixelDescriptor,
    values: Vec<(AnalysisFeature, FeatureValue)>,
}

impl AnalysisResults {
    pub(crate) fn new(
        requested: FeatureSet,
        geometry: ImageGeometry,
        source_descriptor: zenpixels::PixelDescriptor,
    ) -> Self {
        Self {
            requested,
            geometry,
            source_descriptor,
            values: Vec::with_capacity(requested.len() as usize),
        }
    }

    pub(crate) fn set(&mut self, f: AnalysisFeature, v: impl Into<FeatureValue>) {
        debug_assert!(
            self.requested.contains(f),
            "analyzer wrote unrequested feature {:?} (id={}) — dispatcher gating is broken",
            f, f.id()
        );
        if !self.requested.contains(f) { return; }
        let v = v.into();
        if let FeatureValue::F32(x) = v && x.is_nan() { return; }
        let mut i = 0;
        while i < self.values.len() {
            match self.values[i].0.id().cmp(&f.id()) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Equal => { self.values[i].1 = v; return; }
                core::cmp::Ordering::Greater => break,
            }
        }
        self.values.insert(i, (f, v));
    }

    pub const fn requested(&self) -> FeatureSet { self.requested }
    pub const fn geometry(&self) -> ImageGeometry { self.geometry }
    #[inline]
    pub const fn source_descriptor(&self) -> zenpixels::PixelDescriptor { self.source_descriptor }

    #[inline]
    pub fn get(&self, f: AnalysisFeature) -> Option<FeatureValue> {
        self.values.iter().find(|(k, _)| *k == f).map(|(_, v)| *v)
    }

    #[inline]
    pub fn get_f32(&self, f: AnalysisFeature) -> Option<f32> {
        self.get(f).map(FeatureValue::to_f32)
    }
}

impl core::fmt::Debug for AnalysisResults {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut d = f.debug_struct("AnalysisResults");
        d.field("requested", &self.requested);
        d.field("geometry", &self.geometry);
        for (feature, v) in &self.values { d.field(feature.name(), v); }
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESERVED_RETIRED_IDS: &[u16] = &[
        11, 27, 28, 29, 45, 64, 66, 94, 95, 96, 97, 98, 99, 100, 104, 117, 118, 119, 30,
    ];

    #[test]
    fn feature_set_basic_ops() {
        let a = FeatureSet::just(AnalysisFeature::Variance);
        let b = FeatureSet::just(AnalysisFeature::EdgeDensity);
        let u = a.union(b);
        assert!(u.contains(AnalysisFeature::Variance));
        assert!(u.contains(AnalysisFeature::EdgeDensity));
        assert!(!u.contains(AnalysisFeature::ChromaComplexity));
        assert_eq!(u.len(), 2);
        assert!(a.intersects(u));
        assert!(u.contains_all(a));
        assert!(!a.contains_all(u));
    }

    #[test]
    fn feature_value_roundtrip() {
        let v: FeatureValue = 1.5f32.into();
        assert_eq!(v.as_f32(), Some(1.5));
        assert_eq!(v.as_u32(), None);
        assert_eq!(v.to_f32(), 1.5);
        let v: FeatureValue = 7u32.into();
        assert_eq!(v.as_u32(), Some(7));
        assert_eq!(v.to_f32(), 7.0);
        let v: FeatureValue = true.into();
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(v.to_f32(), 1.0);
    }

    #[test]
    fn discriminants_round_trip() {
        for id in 0..64u16 {
            if RESERVED_RETIRED_IDS.contains(&id) {
                assert!(AnalysisFeature::from_u16(id).is_none());
                continue;
            }
            if let Some(f) = AnalysisFeature::from_u16(id) {
                assert_eq!(f.id(), id);
            }
        }
        #[cfg(not(feature = "experimental"))]
        assert!(AnalysisFeature::from_u16(122).is_none());
        #[cfg(feature = "experimental")]
        assert!(AnalysisFeature::from_u16(122).is_some());
        assert!(AnalysisFeature::from_u16(255).is_none());
    }

    #[test]
    fn analysis_query_constructor_only() {
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
        assert!(q.features().contains(AnalysisFeature::Variance));
        assert!(!q.features().contains(AnalysisFeature::EdgeDensity));
    }

    #[test]
    fn raw_analysis_round_trip() {
        let raw = RawAnalysis {
            variance: 12.5,
            distinct_color_bins: 4096,
            alpha_present: true,
            ..Default::default()
        };
        let requested = FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::AlphaPresent);
        let r = raw.into_results(
            requested,
            ImageGeometry::new(64, 64),
            zenpixels::PixelDescriptor::RGB8_SRGB,
        );
        assert_eq!(r.get_f32(AnalysisFeature::Variance), Some(12.5));
        assert_eq!(r.get(AnalysisFeature::DistinctColorBins), Some(FeatureValue::U32(4096)));
        assert_eq!(r.get(AnalysisFeature::AlphaPresent), Some(FeatureValue::Bool(true)));
    }

    #[test]
    fn supported_set_covers_all_active_variants() {
        let mut active = 0u32;
        for id in 0..256u16 {
            if RESERVED_RETIRED_IDS.contains(&id) { continue; }
            let Some(f) = AnalysisFeature::from_u16(id) else { continue; };
            assert!(
                FeatureSet::SUPPORTED.contains(f),
                "{:?} (id={}) is missing from FeatureSet::SUPPORTED", f, id
            );
            active += 1;
        }
        assert_eq!(FeatureSet::SUPPORTED.len(), active);
    }

    #[test]
    fn analysis_results_get_and_set() {
        let requested = FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::AlphaPresent);
        let mut r = AnalysisResults::new(
            requested,
            ImageGeometry::new(1920, 1080),
            zenpixels::PixelDescriptor::RGB8_SRGB,
        );
        r.set(AnalysisFeature::AlphaPresent, true);
        r.set(AnalysisFeature::Variance, 42.0_f32);
        r.set(AnalysisFeature::DistinctColorBins, 1234_u32);
        r.set(AnalysisFeature::Variance, 43.0_f32);
        assert_eq!(r.get(AnalysisFeature::Variance), Some(FeatureValue::F32(43.0)));
        assert_eq!(r.get(AnalysisFeature::DistinctColorBins), Some(FeatureValue::U32(1234)));
        assert_eq!(r.get(AnalysisFeature::AlphaPresent), Some(FeatureValue::Bool(true)));
        assert_eq!(r.get(AnalysisFeature::ChromaComplexity), None);
        assert_eq!(r.get(AnalysisFeature::EdgeDensity), None);
        assert_eq!(r.get_f32(AnalysisFeature::Variance), Some(43.0));
        assert_eq!(r.get_f32(AnalysisFeature::DistinctColorBins), Some(1234.0));
        assert_eq!(r.get_f32(AnalysisFeature::AlphaPresent), Some(1.0));
        assert_eq!(r.geometry().width(), 1920);
        assert_eq!(r.geometry().pixels(), 1920 * 1080);
        assert!((r.geometry().aspect_ratio() - 1920.0 / 1080.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "analyzer wrote unrequested feature")]
    fn set_unrequested_feature_panics_in_debug() {
        let mut r = AnalysisResults::new(
            FeatureSet::just(AnalysisFeature::Variance),
            ImageGeometry::new(64, 64),
            zenpixels::PixelDescriptor::RGB8_SRGB,
        );
        r.set(AnalysisFeature::EdgeDensity, 0.0_f32);
    }

    #[test]
    fn feature_set_iter_visits_every_set_member_in_id_order() {
        let s = FeatureSet::just(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity);
        let collected: Vec<_> = s.iter().collect();
        assert_eq!(
            collected,
            vec![AnalysisFeature::Variance, AnalysisFeature::EdgeDensity, AnalysisFeature::DistinctColorBins]
        );
        let n = FeatureSet::SUPPORTED.iter().count();
        assert_eq!(n as u32, FeatureSet::SUPPORTED.len());
        assert_eq!(FeatureSet::new().iter().count(), 0);
        let s2 = FeatureSet::just(AnalysisFeature::Variance);
        let v: Vec<_> = s2.into_iter().collect();
        assert_eq!(v, vec![AnalysisFeature::Variance]);
    }

    #[test]
    fn analysis_feature_id_is_public_and_stable() {
        assert_eq!(AnalysisFeature::Variance.id(), 0);
        assert_eq!(AnalysisFeature::EdgeDensity.id(), 1);
        assert_eq!(AnalysisFeature::DistinctColorBins.id(), 10);
        assert_eq!(AnalysisFeature::AlphaPresent.id(), 24);
        assert_eq!(AnalysisFeature::from_u16(0), Some(AnalysisFeature::Variance));
        assert_eq!(AnalysisFeature::from_u16(11), None);
        assert_eq!(AnalysisFeature::from_u16(9999), None);
    }

    #[test]
    fn tier_bundles_are_disjoint() {
        let bundles = [
            PALETTE_FEATURES,
            TIER2_FEATURES,
            TIER3_FEATURES,
            ALPHA_FEATURES,
            DERIVED_FEATURES,
        ];
        for (i, a) in bundles.iter().enumerate() {
            for b in bundles.iter().skip(i + 1) {
                assert!(!a.intersects(*b), "tier bundles overlap (this is a bug)");
            }
        }
    }
}
