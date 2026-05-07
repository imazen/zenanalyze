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

/// Single-source-of-truth macro: generates the [`AnalysisFeature`]
/// enum, its `id` / `from_u16` / `is_active` / `name` impls, the
/// internal [`RawAnalysis`] dense struct that the SIMD tiers write
/// into, the `RawAnalysis::into_results` translator, and the
/// [`FeatureSet::SUPPORTED`] preset — all in lockstep so adding a
/// feature is a one-line edit at the invocation site.
///
/// Per-feature row syntax:
/// `$(#[$attr:meta])* $Variant = $id : $type => $field`
/// The `$field` ident is the snake_case name (also used for
/// `AnalysisFeature::name`'s string return via `stringify!`).
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
            // Future features take the next free sequential id (30, 31, …).
        }

        impl AnalysisFeature {
            /// The feature's stable `u16` discriminant — the wire
            /// format. Sequential since 0, immutable once shipped.
            /// Retired ids stay reserved (their variant is removed
            /// from the enum but the number is never recycled). New
            /// variants take the next free number.
            ///
            /// Useful for: serialising a [`FeatureSet`] to a bitmask,
            /// building a lookup table, or writing a protocol buffer.
            #[inline]
            pub const fn id(self) -> u16 {
                self as u16
            }

            /// Reconstruct from its `u16` id. Returns `None` for
            /// unknown or retired ids (forward-compat: newer producers
            /// may emit ids this version hasn't seen).
            #[inline]
            pub const fn from_u16(v: u16) -> Option<Self> {
                match v {
                    $( $id => Some(Self::$variant), )*
                    _ => None,
                }
            }

            /// Snake-case name string, useful for logging / diagnostics.
            #[inline]
            pub const fn name(self) -> &'static str {
                match self {
                    $( Self::$variant => stringify!($field), )*
                    // Safety: non_exhaustive only affects external
                    // matches; within the crate we can use _ here.
                    #[allow(unreachable_patterns)]
                    _ => "<unknown>",
                }
            }
        }

        /// Raw, dense struct written by the SIMD tiers — one field per
        /// feature. All fields start `None`; each tier fills only its
        /// own fields. The outer analysis driver calls
        /// [`RawAnalysis::into_results`] once all tiers have run.
        ///
        /// `pub(crate)` — callers see only [`AnalysisResults`].
        #[allow(dead_code)]
        pub(crate) struct RawAnalysis {
            $( pub(crate) $field: Option<$ty>, )*
        }

        impl RawAnalysis {
            /// All fields `None`.
            pub(crate) const fn new() -> Self {
                Self { $( $field: None, )* }
            }

            /// Translate the raw struct into the public result map.
            /// Only features present in `requested` are included.
            pub(crate) fn into_results(
                self,
                requested: &crate::FeatureSet,
            ) -> crate::AnalysisResults {
                let mut map = std::collections::HashMap::new();
                $(
                    if requested.is_active(AnalysisFeature::$variant) {
                        if let Some(v) = self.$field {
                            map.insert(
                                AnalysisFeature::$variant,
                                FeatureValue::from(v),
                            );
                        }
                    }
                )*
                crate::AnalysisResults { features: map }
            }
        }

        impl FeatureSet {
            /// Every feature this build knows about (stable +
            /// experimental when the `experimental` cargo feature is
            /// on). Use this as an upper bound for "give me everything"
            /// queries; do **not** hard-code it across versions.
            pub const SUPPORTED: FeatureSet = {
                let mut s = FeatureSet::new();
                $( s = s.with(AnalysisFeature::$variant); )*
                s
            };
        }
    };
}

// ── FeatureValue ────────────────────────────────────────────────────────────

/// Owned result value for a single feature.
///
/// Currently only `f32` and `u32` scalar measurements are produced.
/// The enum is `#[non_exhaustive]` so structured types (small
/// histograms, feature vectors) can be added later without a major
/// version bump.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureValue {
    /// A 32-bit floating-point scalar (the common case).
    F32(f32),
    /// A 32-bit unsigned integer (e.g. palette size, block counts).
    U32(u32),
}

impl From<f32> for FeatureValue {
    fn from(v: f32) -> Self {
        FeatureValue::F32(v)
    }
}
impl From<u32> for FeatureValue {
    fn from(v: u32) -> Self {
        FeatureValue::U32(v)
    }
}

// ── FeatureSet ──────────────────────────────────────────────────────────────

/// Compact, const-constructible bitset of [`AnalysisFeature`]s.
///
/// Backed by a `[u64; 4]` (256 bits), supporting IDs 0–255 with no
/// heap allocation. All mutating operations return a new `FeatureSet`
/// so they can be chained in `const` context.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct FeatureSet([u64; 4]);

impl std::fmt::Debug for FeatureSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        write!(f, "FeatureSet{{")?;
        for bit in 0u16..=255 {
            if self.0[(bit / 64) as usize] & (1u64 << (bit % 64)) != 0 {
                if !first {
                    write!(f, ", ")?;
                }
                first = false;
                if let Some(feat) = AnalysisFeature::from_u16(bit) {
                    write!(f, "{:?}", feat)?;
                } else {
                    write!(f, "#{}", bit)?;
                }
            }
        }
        write!(f, "}}")
    }
}

impl FeatureSet {
    /// Empty set — the canonical starting point for building a query.
    #[inline]
    pub const fn new() -> Self {
        FeatureSet([0u64; 4])
    }

    /// Returns a new set with `feature` added.
    #[inline]
    pub const fn with(mut self, feature: AnalysisFeature) -> Self {
        let id = feature as u16;
        self.0[(id / 64) as usize] |= 1u64 << (id % 64);
        self
    }

    /// Returns a new set with `feature` removed.
    #[inline]
    pub const fn without(mut self, feature: AnalysisFeature) -> Self {
        let id = feature as u16;
        self.0[(id / 64) as usize] &= !(1u64 << (id % 64));
        self
    }

    /// Test membership.
    #[inline]
    pub const fn is_active(&self, feature: AnalysisFeature) -> bool {
        let id = feature as u16;
        self.0[(id / 64) as usize] & (1u64 << (id % 64)) != 0
    }

    /// Set union (`self | other`).
    #[inline]
    pub const fn union(self, other: FeatureSet) -> FeatureSet {
        FeatureSet([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    /// Set intersection (`self & other`).
    #[inline]
    pub const fn intersect(self, other: FeatureSet) -> FeatureSet {
        FeatureSet([
            self.0[0] & other.0[0],
            self.0[1] & other.0[1],
            self.0[2] & other.0[2],
            self.0[3] & other.0[3],
        ])
    }

    /// True iff `self` and `other` share at least one feature.
    #[inline]
    pub const fn intersects(self, other: FeatureSet) -> bool {
        (self.0[0] & other.0[0]) != 0
            || (self.0[1] & other.0[1]) != 0
            || (self.0[2] & other.0[2]) != 0
            || (self.0[3] & other.0[3]) != 0
    }

    /// True iff `self` is a subset of `other` (every bit in `self` is
    /// also in `other`).
    #[inline]
    pub const fn is_subset_of(self, other: FeatureSet) -> bool {
        (self.0[0] & !other.0[0]) == 0
            && (self.0[1] & !other.0[1]) == 0
            && (self.0[2] & !other.0[2]) == 0
            && (self.0[3] & !other.0[3]) == 0
    }

    /// True iff no features are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
    }

    /// Number of features set (popcount).
    #[inline]
    pub const fn count(self) -> u32 {
        self.0[0].count_ones()
            + self.0[1].count_ones()
            + self.0[2].count_ones()
            + self.0[3].count_ones()
    }
}

// ── features_table! invocation ──────────────────────────────────────────────

features_table!(
    // ── Tier 0 (ultra-cheap, single-pass pixel stats) ────────────────────
    /// `f32`. Pixel luminance variance (Y channel, BT.601 weights).
    Variance = 0 : f32 => variance,
    /// `f32`. Mean luminance.
    MeanLuminance = 1 : f32 => mean_luminance,
    /// `f32`. Fraction of pixels below 16 or above 235 (legal-range clipping indicator).
    ClippingFraction = 2 : f32 => clipping_fraction,
    /// `f32`. Skewness of the luminance histogram.
    LuminanceSkewness = 3 : f32 => luminance_skewness,
    /// `f32`. Fraction of pixels that are fully transparent (alpha == 0).
    /// Always 0.0 when `alpha_present == false`.
    TransparentFraction = 4 : f32 => transparent_fraction,
    /// `bool` as `u32` (0 or 1). True when any pixel has alpha < 255.
    AlphaPresent = 5 : u32 => alpha_present,
    /// `f32`. Bimodality score of the alpha channel (0 = fully opaque or fully transparent; 1 = maximally bimodal).
    AlphaBimodalScore = 6 : f32 => alpha_bimodal_score,
    /// `f32`. Fraction of pixels with luminance in the near-black range [0, 16).
    DarkFraction = 7 : f32 => dark_fraction,
    /// `f32`. Fraction of pixels with luminance in the near-white range (235, 255].
    BrightFraction = 8 : f32 => bright_fraction,
    /// `f32`. Spatial autocorrelation in X (mean product of horizontal neighbours, normalised).
    SpatialAutocorrX = 9 : f32 => spatial_autocorr_x,
    /// `f32`. Spatial autocorrelation in Y.
    SpatialAutocorrY = 10 : f32 => spatial_autocorr_y,
    /// `f32`. Chroma noise proxy: mean absolute deviation of Cb channel.
    ChromaNoiseCb = 11 : f32 => chroma_noise_cb,
    /// `f32`. Chroma noise proxy: mean absolute deviation of Cr channel.
    ChromaNoiseCr = 12 : f32 => chroma_noise_cr,
    /// `f32`. Entropy of a 16-bin luminance histogram (bits, max ≈ 4.0).
    LuminanceEntropy = 13 : f32 => luminance_entropy,
    /// `u32`. Estimated number of distinct palette colours (≤ 256 triggers
    /// `PaletteLog2Size`; > 256 → 0, meaning "not a palette image").
    PaletteColorCount = 14 : u32 => palette_color_count,
    /// `u32`. `ceil(log2(palette_color_count))`, or 0 when > 256 distinct
    /// colours. Useful as a direct input to palette-codec bit-depth
    /// decisions.
    PaletteLog2Size = 15 : u32 => palette_log2_size_t0,
    // ── Tier 1 (SIMD edge / Laplacian pass) ─────────────────────────────
    /// `f32`. Mean absolute value of the Laplacian (edge energy proxy).
    EdgeDensity = 16 : f32 => edge_density,
    /// `f32`. Variance of the Laplacian response — a classic blur/sharpness metric.
    LaplacianVariance = 17 : f32 => laplacian_variance,
    /// `f32`. Fraction of pixels classified as edges (|Laplacian| > threshold).
    EdgeFraction = 18 : f32 => edge_fraction,
    /// `f32`. High-frequency energy ratio (top-left 8×8 DCT band / total).
    HighFreqEnergyRatio = 19 : f32 => high_freq_energy_ratio,
    /// `f32`. Gradient orientation entropy (0 = all edges same direction; high = isotropic).
    GradientOrientationEntropy = 20 : f32 => gradient_orientation_entropy,
    // ── Tier 2 (DCT / frequency-domain pass) ────────────────────────────
    /// `f32`. DCT-domain compressibility score for luma (Y) channel.
    DctCompressibilityY = 21 : f32 => dct_compressibility_y,
    /// `f32`. DCT-domain compressibility score for chroma (Cb/Cr) channels.
    DctCompressibilityCbCr = 22 : f32 => dct_compressibility_cbcr,
    /// `f32`. Average quantisation step for luma at quality≈75 JPEG (proxy for texture complexity).
    AvgQuantStepY = 23 : f32 => avg_quant_step_y,
    /// `f32`. Average quantisation step for chroma.
    AvgQuantStepCbCr = 24 : f32 => avg_quant_step_cbcr,
    /// `f32`. Fraction of 8×8 luma blocks with energy in the AC components.
    AcEnergyFractionY = 25 : f32 => ac_energy_fraction_y,
    // ── Tier 3 (AQ-map / noise-floor / quant-survival pass) ─────────────
    /// `f32`. Mean of the adaptive-quantisation weight map (higher → more texture variation).
    AqMapMean = 26 : f32 => aq_map_mean,
    /// `f32`. Standard deviation of the AQ map.
    AqMapStddev = 27 : f32 => aq_map_stddev,
    /// `f32`. Noise floor estimate on the Y (luma) channel.
    NoiseFloorY = 28 : f32 => noise_floor_y,
    /// `f32`. Noise floor estimate on the UV (chroma) channels.
    NoiseFloorUv = 29 : f32 => noise_floor_uv,
    /// `f32`. Fraction of DCT coefficients that survive a mid-quality quantisation step on Y.
    QuantSurvivalY = 30 : f32 => quant_survival_y,
    /// `f32`. Same for UV channels.
    QuantSurvivalUv = 31 : f32 => quant_survival_uv,
    // ── Percentile features (Tier 1 / Laplacian) ────────────────────────
    /// `f32`. 1st percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP1 = 32 : f32 => laplacian_variance_p1,
    /// `f32`. 5th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP5 = 33 : f32 => laplacian_variance_p5,
    /// `f32`. 10th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP10 = 34 : f32 => laplacian_variance_p10,
    /// `f32`. 25th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP25 = 35 : f32 => laplacian_variance_p25,
    /// `f32`. 50th percentile (median) of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP50 = 36 : f32 => laplacian_variance_p50,
    /// `f32`. 75th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP75 = 37 : f32 => laplacian_variance_p75,
    /// `f32`. 90th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP90 = 38 : f32 => laplacian_variance_p90,
    /// `f32`. 95th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP95 = 39 : f32 => laplacian_variance_p95,
    /// `f32`. 99th percentile of per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP99 = 40 : f32 => laplacian_variance_p99,
    /// `f32`. Peak (max) per-block LaplacianVariance.
    #[cfg(feature = "experimental")]
    LaplacianVariancePeak = 41 : f32 => laplacian_variance_peak,
    // ── Percentile features (Tier 3 / AQ map) ───────────────────────────
    /// `f32`. 1st percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP1 = 42 : f32 => aq_map_p1,
    /// `f32`. 5th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP5 = 43 : f32 => aq_map_p5,
    /// `f32`. 10th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP10 = 44 : f32 => aq_map_p10,
    /// `f32`. 25th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP25 = 45 : f32 => aq_map_p25,
    /// `f32`. 50th percentile (median) of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP50 = 46 : f32 => aq_map_p50,
    /// `f32`. 75th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP75 = 47 : f32 => aq_map_p75,
    /// `f32`. 90th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP90 = 48 : f32 => aq_map_p90,
    /// `f32`. 95th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP95 = 49 : f32 => aq_map_p95,
    /// `f32`. 99th percentile of AQ-map weights.
    #[cfg(feature = "experimental")]
    AqMapP99 = 50 : f32 => aq_map_p99,
    // ── Percentile features (Tier 3 / Noise floor Y) ─────────────────────
    /// `f32`. 1st percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP1 = 51 : f32 => noise_floor_y_p1,
    /// `f32`. 5th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP5 = 52 : f32 => noise_floor_y_p5,
    /// `f32`. 10th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP10 = 53 : f32 => noise_floor_y_p10,
    /// `f32`. 25th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP25 = 54 : f32 => noise_floor_y_p25,
    /// `f32`. 50th percentile (median) of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP50 = 55 : f32 => noise_floor_y_p50,
    /// `f32`. 75th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP75 = 56 : f32 => noise_floor_y_p75,
    /// `f32`. 90th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP90 = 57 : f32 => noise_floor_y_p90,
    /// `f32`. 95th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP95 = 58 : f32 => noise_floor_y_p95,
    /// `f32`. 99th percentile of per-block noise-floor-Y estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorYP99 = 59 : f32 => noise_floor_y_p99,
    // ── Percentile features (Tier 3 / Noise floor UV) ────────────────────
    /// `f32`. 1st percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP1 = 60 : f32 => noise_floor_uv_p1,
    /// `f32`. 5th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP5 = 61 : f32 => noise_floor_uv_p5,
    /// `f32`. 10th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP10 = 62 : f32 => noise_floor_uv_p10,
    /// `f32`. 25th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP25 = 63 : f32 => noise_floor_uv_p25,
    /// `f32`. 50th percentile (median) of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP50 = 64 : f32 => noise_floor_uv_p50,
    /// `f32`. 75th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP75 = 65 : f32 => noise_floor_uv_p75,
    /// `f32`. 90th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP90 = 66 : f32 => noise_floor_uv_p90,
    /// `f32`. 95th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP95 = 67 : f32 => noise_floor_uv_p95,
    /// `f32`. 99th percentile of per-block noise-floor-UV estimates.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP99 = 68 : f32 => noise_floor_uv_p99,
    // ── Percentile features (Tier 3 / QuantSurvival Y) ────────────────────
    /// `f32`. 1st percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP1 = 69 : f32 => quant_survival_y_p1,
    /// `f32`. 5th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP5 = 70 : f32 => quant_survival_y_p5,
    /// `f32`. 10th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP10 = 71 : f32 => quant_survival_y_p10,
    /// `f32`. 25th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP25 = 72 : f32 => quant_survival_y_p25,
    /// `f32`. 50th percentile (median) of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP50 = 73 : f32 => quant_survival_y_p50,
    /// `f32`. 75th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP75 = 74 : f32 => quant_survival_y_p75,
    /// `f32`. 90th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP90 = 75 : f32 => quant_survival_y_p90,
    /// `f32`. 95th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP95 = 76 : f32 => quant_survival_y_p95,
    /// `f32`. 99th percentile of per-block QuantSurvival-Y values.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP99 = 77 : f32 => quant_survival_y_p99,
    // ── Percentile features (Tier 3 / QuantSurvival UV) ───────────────────
    /// `f32`. 1st percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP1 = 78 : f32 => quant_survival_uv_p1,
    /// `f32`. 5th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP5 = 79 : f32 => quant_survival_uv_p5,
    /// `f32`. 10th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP10 = 80 : f32 => quant_survival_uv_p10,
    /// `f32`. 25th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP25 = 81 : f32 => quant_survival_uv_p25,
    /// `f32`. 50th percentile (median) of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP50 = 82 : f32 => quant_survival_uv_p50,
    /// `f32`. 75th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP75 = 83 : f32 => quant_survival_uv_p75,
    /// `f32`. 90th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP90 = 84 : f32 => quant_survival_uv_p90,
    /// `f32`. 95th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP95 = 85 : f32 => quant_survival_uv_p95,
    /// `f32`. 99th percentile of per-block QuantSurvival-UV values.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP99 = 86 : f32 => quant_survival_uv_p99,
    // ── Additional Tier 0 / 1 scalar stats ──────────────────────────────
    /// `f32`. Coefficient of variation of luminance (stddev / mean).
    #[cfg(feature = "experimental")]
    LuminanceCv = 87 : f32 => luminance_cv,
    /// `f32`. Inter-channel colour correlation (Y vs Cb, Pearson r).
    #[cfg(feature = "experimental")]
    ColorCorrelationYCb = 88 : f32 => color_correlation_y_cb,
    /// `f32`. Inter-channel colour correlation (Y vs Cr, Pearson r).
    #[cfg(feature = "experimental")]
    ColorCorrelationYCr = 89 : f32 => color_correlation_y_cr,
    /// `f32`. Fraction of blocks with high-frequency energy above a fixed threshold.
    #[cfg(feature = "experimental")]
    HighFreqBlockFraction = 90 : f32 => high_freq_block_fraction,
    /// `f32`. Spatial variance of per-block mean luminance (macro-structure).
    #[cfg(feature = "experimental")]
    SpatialLuminanceVariance = 91 : f32 => spatial_luminance_variance,
    /// `f32`. Directional energy ratio: horizontal AC vs vertical AC in 8×8 DCT blocks.
    #[cfg(feature = "experimental")]
    DirectionalEnergyRatioHV = 92 : f32 => directional_energy_ratio_hv,
    /// `f32`. Texture regularity score (autocorrelation at lag 8 normalised by lag 1).
    #[cfg(feature = "experimental")]
    TextureRegularity = 93 : f32 => texture_regularity,
    /// `f32`. Chroma subsampling sensitivity proxy (mean squared difference after 4:2:0 round-trip).
    #[cfg(feature = "experimental")]
    ChromaSubsamplingLoss = 94 : f32 => chroma_subsampling_loss,
    /// `f32`. Residual energy after a simple intra-prediction pass on 4×4 luma blocks.
    #[cfg(feature = "experimental")]
    IntraPredictionResidual = 95 : f32 => intra_prediction_residual,
    /// `f32`. Mean gradient magnitude (Sobel) as an alternative edge-density measure.
    #[cfg(feature = "experimental")]
    GradientMagnitudeMean = 96 : f32 => gradient_magnitude_mean,
    /// `f32`. Fraction of 8×8 blocks with a coefficient of variation below 0.05
    /// (near-flat blocks; high → smooth image; low → busy/noisy).
    #[cfg(feature = "experimental")]
    FlatBlockFraction = 97 : f32 => flat_block_fraction,
    /// `f32`. Mean absolute difference between horizontally adjacent 8×8-block means
    /// (boundary sharpness proxy).
    #[cfg(feature = "experimental")]
    BlockBoundarySharpness = 98 : f32 => block_boundary_sharpness,
    /// `f32`. Ratio of energy in the top-left 4×4 DCT coefficients to the full 8×8
    /// block — low-frequency dominance score.
    #[cfg(feature = "experimental")]
    LowFreqDominance = 99 : f32 => low_freq_dominance,
    // ── Tier 0 structural / misc ─────────────────────────────────────────
    /// `u32`. Image width in pixels.
    #[cfg(feature = "experimental")]
    Width = 100 : u32 => width,
    /// `u32`. Image height in pixels.
    #[cfg(feature = "experimental")]
    Height = 101 : u32 => height,
    /// `f32`. Aspect ratio (width / height).
    #[cfg(feature = "experimental")]
    AspectRatio = 102 : f32 => aspect_ratio,
    /// `u32`. Total pixel count (width × height).
    #[cfg(feature = "experimental")]
    PixelCount = 103 : u32 => pixel_count,
    // ── Tier 1 advanced edge / texture ───────────────────────────────────
    /// `f32`. Corner density (Harris / Shi-Tomasi proxy count per kpx).
    #[cfg(feature = "experimental")]
    CornerDensity = 104 : f32 => corner_density,
    /// `f32`. Blob density (LoG zero-crossings per kpx).
    #[cfg(feature = "experimental")]
    BlobDensity = 105 : f32 => blob_density,
    /// `f32`. Structural similarity to a bilinearly downscaled+upscaled version
    /// (sharpness preservation proxy).
    #[cfg(feature = "experimental")]
    DownscaleSharpness = 106 : f32 => downscale_sharpness,
    // ── Tier 2 / DCT extended ─────────────────────────────────────────────
    /// `f32`. Kurtosis of the DCT coefficient distribution (Y channel).
    #[cfg(feature = "experimental")]
    DctKurtosisY = 107 : f32 => dct_kurtosis_y,
    /// `f32`. Zero-run-length mean in zig-zag scan of 8×8 DCT blocks (Y channel).
    #[cfg(feature = "experimental")]
    DctZeroRunMeanY = 108 : f32 => dct_zero_run_mean_y,
    // ── Tier 3 extended ───────────────────────────────────────────────────
    /// `f32`. Variance of the per-block noise-floor-Y distribution.
    #[cfg(feature = "experimental")]
    NoiseFloorYVariance = 109 : f32 => noise_floor_y_variance,
    /// `f32`. Variance of the per-block noise-floor-UV distribution.
    #[cfg(feature = "experimental")]
    NoiseFloorUvVariance = 110 : f32 => noise_floor_uv_variance,
    /// `f32`. Variance of the per-block QuantSurvival-Y distribution.
    #[cfg(feature = "experimental")]
    QuantSurvivalYVariance = 111 : f32 => quant_survival_y_variance,
    /// `f32`. Variance of the per-block QuantSurvival-UV distribution.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvVariance = 112 : f32 => quant_survival_uv_variance,
    // ── Tier 3 / AQ map extended ──────────────────────────────────────────
    /// `f32`. Variance of the AQ-map weight distribution (complement to AqMapStddev²; kept separate for cross-version stability).
    #[cfg(feature = "experimental")]
    AqMapVariance = 113 : f32 => aq_map_variance,
    /// `f32`. Skewness of the AQ-map weight distribution.
    #[cfg(feature = "experimental")]
    AqMapSkewness = 114 : f32 => aq_map_skewness,
    /// `f32`. Kurtosis of the AQ-map weight distribution.
    #[cfg(feature = "experimental")]
    AqMapKurtosis = 115 : f32 => aq_map_kurtosis,
    // ── Tier 1 / Laplacian extended ───────────────────────────────────────
    /// `f32`. Variance of the per-block LaplacianVariance distribution
    /// (second-order sharpness non-uniformity).
    #[cfg(feature = "experimental")]
    LaplacianVarianceVariance = 116 : f32 => laplacian_variance_variance,
    /// `f32`. Skewness of the per-block LaplacianVariance distribution.
    #[cfg(feature = "experimental")]
    LaplacianVarianceSkewness = 117 : f32 => laplacian_variance_skewness,
    /// `f32`. Kurtosis of the per-block LaplacianVariance distribution.
    #[cfg(feature = "experimental")]
    LaplacianVarianceKurtosis = 118 : f32 => laplacian_variance_kurtosis,
    // ── Tier 0 / histogram shape ─────────────────────────────────────────
    /// `f32`. Gini coefficient of the luminance histogram (0 = uniform; 1 = all mass on one bin).
    #[cfg(feature = "experimental")]
    LuminanceGini = 119 : f32 => luminance_gini,
    /// `f32`. Bimodality coefficient of the luminance histogram
    /// (Sarle's BC = (skewness²+1) / kurtosis).
    #[cfg(feature = "experimental")]
    LuminanceBimodality = 120 : f32 => luminance_bimodality,
    /// `u32`. `ceil(log2(palette_color_count))`, or 0 when > 256 distinct
    /// colours. Tier-0 re-export under a cleaner name (supersedes `palette_log2_size_t0 = 15`).
    #[cfg(feature = "experimental")]
    PaletteLog2Size = 121 : u32 => palette_log2_size,

    // ---------------- Dense percentile sweep (2026-05-07) ------------
    // 5-percentile-step grid filling gaps in the existing p25/50/75/90/95/99
    // grid for: LaplacianVariance, AqMap, NoiseFloorY/Uv, QuantSurvivalY/Uv.
    // IDs 122-211 (additive, experimental gate). Purpose: ablation via
    // `feature_ablation.py --method permutation` to find the minimal
    // percentile set capturing >= 95% of cumulative importance.
    // See `benchmarks/dense_percentile_sweep_2026-05-07.md`.

    // --- LaplacianVariance dense percentiles (122-135) ---
    /// `f32`. LaplacianVariance p15 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP15 = 122 : f32 => laplacian_variance_p15,
    /// `f32`. LaplacianVariance p20 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP20 = 123 : f32 => laplacian_variance_p20,
    /// `f32`. LaplacianVariance p30 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP30 = 124 : f32 => laplacian_variance_p30,
    /// `f32`. LaplacianVariance p35 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP35 = 125 : f32 => laplacian_variance_p35,
    /// `f32`. LaplacianVariance p40 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP40 = 126 : f32 => laplacian_variance_p40,
    /// `f32`. LaplacianVariance p45 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP45 = 127 : f32 => laplacian_variance_p45,
    /// `f32`. LaplacianVariance p55 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP55 = 128 : f32 => laplacian_variance_p55,
    /// `f32`. LaplacianVariance p60 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP60 = 129 : f32 => laplacian_variance_p60,
    /// `f32`. LaplacianVariance p65 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP65 = 130 : f32 => laplacian_variance_p65,
    /// `f32`. LaplacianVariance p70 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP70 = 131 : f32 => laplacian_variance_p70,
    /// `f32`. LaplacianVariance p80 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP80 = 132 : f32 => laplacian_variance_p80,
    /// `f32`. LaplacianVariance p85 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP85 = 133 : f32 => laplacian_variance_p85,

    // --- AqMap dense percentiles (136-148) ---
    /// `f32`. AqMap p15 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP15 = 136 : f32 => aq_map_p15,
    /// `f32`. AqMap p20 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP20 = 137 : f32 => aq_map_p20,
    /// `f32`. AqMap p30 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP30 = 138 : f32 => aq_map_p30,
    /// `f32`. AqMap p35 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP35 = 139 : f32 => aq_map_p35,
    /// `f32`. AqMap p40 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP40 = 140 : f32 => aq_map_p40,
    /// `f32`. AqMap p45 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP45 = 141 : f32 => aq_map_p45,
    /// `f32`. AqMap p55 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP55 = 142 : f32 => aq_map_p55,
    /// `f32`. AqMap p60 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP60 = 143 : f32 => aq_map_p60,
    /// `f32`. AqMap p65 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP65 = 144 : f32 => aq_map_p65,
    /// `f32`. AqMap p70 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP70 = 145 : f32 => aq_map_p70,
    /// `f32`. AqMap p80 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP80 = 146 : f32 => aq_map_p80,
    /// `f32`. AqMap p85 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    AqMapP85 = 147 : f32 => aq_map_p85,

    // --- NoiseFloorY dense percentiles (149-162) ---
    /// `f32`. NoiseFloorY p15 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP15 = 149 : f32 => noise_floor_y_p15,
    /// `f32`. NoiseFloorY p20 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP20 = 150 : f32 => noise_floor_y_p20,
    /// `f32`. NoiseFloorY p30 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP30 = 151 : f32 => noise_floor_y_p30,
    /// `f32`. NoiseFloorY p35 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP35 = 152 : f32 => noise_floor_y_p35,
    /// `f32`. NoiseFloorY p40 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP40 = 153 : f32 => noise_floor_y_p40,
    /// `f32`. NoiseFloorY p45 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP45 = 154 : f32 => noise_floor_y_p45,
    /// `f32`. NoiseFloorY p55 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP55 = 155 : f32 => noise_floor_y_p55,
    /// `f32`. NoiseFloorY p60 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP60 = 156 : f32 => noise_floor_y_p60,
    /// `f32`. NoiseFloorY p65 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP65 = 157 : f32 => noise_floor_y_p65,
    /// `f32`. NoiseFloorY p70 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP70 = 158 : f32 => noise_floor_y_p70,
    /// `f32`. NoiseFloorY p80 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP80 = 159 : f32 => noise_floor_y_p80,
    /// `f32`. NoiseFloorY p85 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP85 = 160 : f32 => noise_floor_y_p85,
    /// `f32`. NoiseFloorY p95 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP95 = 161 : f32 => noise_floor_y_p95,
    /// `f32`. NoiseFloorY p99 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorYP99 = 162 : f32 => noise_floor_y_p99,

    // --- NoiseFloorUv dense percentiles (163-179) ---
    /// `f32`. NoiseFloorUv p1 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP1Dense = 163 : f32 => noise_floor_uv_p1_dense,
    /// `f32`. NoiseFloorUv p5 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP5Dense = 164 : f32 => noise_floor_uv_p5_dense,
    /// `f32`. NoiseFloorUv p10 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP10Dense = 165 : f32 => noise_floor_uv_p10_dense,
    /// `f32`. NoiseFloorUv p15 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP15 = 166 : f32 => noise_floor_uv_p15,
    /// `f32`. NoiseFloorUv p20 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP20 = 167 : f32 => noise_floor_uv_p20,
    /// `f32`. NoiseFloorUv p25 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP25Dense = 168 : f32 => noise_floor_uv_p25_dense,
    /// `f32`. NoiseFloorUv p30 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP30 = 169 : f32 => noise_floor_uv_p30,
    /// `f32`. NoiseFloorUv p35 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP35 = 170 : f32 => noise_floor_uv_p35,
    /// `f32`. NoiseFloorUv p40 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP40 = 171 : f32 => noise_floor_uv_p40,
    /// `f32`. NoiseFloorUv p45 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP45 = 172 : f32 => noise_floor_uv_p45,
    /// `f32`. NoiseFloorUv p55 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP55 = 173 : f32 => noise_floor_uv_p55,
    /// `f32`. NoiseFloorUv p60 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP60 = 174 : f32 => noise_floor_uv_p60,
    /// `f32`. NoiseFloorUv p65 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP65 = 175 : f32 => noise_floor_uv_p65,
    /// `f32`. NoiseFloorUv p70 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP70 = 176 : f32 => noise_floor_uv_p70,
    /// `f32`. NoiseFloorUv p80 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP80 = 177 : f32 => noise_floor_uv_p80,
    /// `f32`. NoiseFloorUv p85 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP85 = 178 : f32 => noise_floor_uv_p85,
    /// `f32`. NoiseFloorUv p99 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP99Dense = 179 : f32 => noise_floor_uv_p99_dense,

    // --- QuantSurvivalY dense percentiles (180-194) ---
    /// `f32`. QuantSurvivalY p15 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP15 = 180 : f32 => quant_survival_y_p15,
    /// `f32`. QuantSurvivalY p20 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP20 = 181 : f32 => quant_survival_y_p20,
    /// `f32`. QuantSurvivalY p30 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP30 = 182 : f32 => quant_survival_y_p30,
    /// `f32`. QuantSurvivalY p35 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP35 = 183 : f32 => quant_survival_y_p35,
    /// `f32`. QuantSurvivalY p40 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP40 = 184 : f32 => quant_survival_y_p40,
    /// `f32`. QuantSurvivalY p45 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP45 = 185 : f32 => quant_survival_y_p45,
    /// `f32`. QuantSurvivalY p55 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP55 = 186 : f32 => quant_survival_y_p55,
    /// `f32`. QuantSurvivalY p60 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP60 = 187 : f32 => quant_survival_y_p60,
    /// `f32`. QuantSurvivalY p65 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP65 = 188 : f32 => quant_survival_y_p65,
    /// `f32`. QuantSurvivalY p70 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP70 = 189 : f32 => quant_survival_y_p70,
    /// `f32`. QuantSurvivalY p80 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP80 = 190 : f32 => quant_survival_y_p80,
    /// `f32`. QuantSurvivalY p85 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP85 = 191 : f32 => quant_survival_y_p85,
    /// `f32`. QuantSurvivalY p95 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP95 = 192 : f32 => quant_survival_y_p95,
    /// `f32`. QuantSurvivalY p99 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP99 = 193 : f32 => quant_survival_y_p99,

    // --- QuantSurvivalUv dense percentiles (195-211) ---
    /// `f32`. QuantSurvivalUv p1 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP1 = 195 : f32 => quant_survival_uv_p1,
    /// `f32`. QuantSurvivalUv p5 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP5 = 196 : f32 => quant_survival_uv_p5,
    /// `f32`. QuantSurvivalUv p10 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP10 = 197 : f32 => quant_survival_uv_p10,
    /// `f32`. QuantSurvivalUv p15 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP15 = 198 : f32 => quant_survival_uv_p15,
    /// `f32`. QuantSurvivalUv p20 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP20 = 199 : f32 => quant_survival_uv_p20,
    /// `f32`. QuantSurvivalUv p25 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP25 = 200 : f32 => quant_survival_uv_p25,
    /// `f32`. QuantSurvivalUv p30 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP30 = 201 : f32 => quant_survival_uv_p30,
    /// `f32`. QuantSurvivalUv p35 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP35 = 202 : f32 => quant_survival_uv_p35,
    /// `f32`. QuantSurvivalUv p40 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP40 = 203 : f32 => quant_survival_uv_p40,
    /// `f32`. QuantSurvivalUv p45 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP45 = 204 : f32 => quant_survival_uv_p45,
    /// `f32`. QuantSurvivalUv p55 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP55 = 205 : f32 => quant_survival_uv_p55,
    /// `f32`. QuantSurvivalUv p60 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP60 = 206 : f32 => quant_survival_uv_p60,
    /// `f32`. QuantSurvivalUv p65 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP65 = 207 : f32 => quant_survival_uv_p65,
    /// `f32`. QuantSurvivalUv p70 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP70 = 208 : f32 => quant_survival_uv_p70,
    /// `f32`. QuantSurvivalUv p80 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP80 = 209 : f32 => quant_survival_uv_p80,
    /// `f32`. QuantSurvivalUv p85 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP85 = 210 : f32 => quant_survival_uv_p85,
    /// `f32`. QuantSurvivalUv p99 -- dense-sweep variant.
    #[cfg(feature = "experimental")]
    QuantSurvivalUvP99 = 211 : f32 => quant_survival_uv_p99,
);

// ── Tier constants ───────────────────────────────────────────────────────────

/// Features that are Tier 3 (AQ-map / noise-floor / quant-survival).
/// Used by the analysis driver to decide whether to run the Tier 3 pass.
#[cfg(feature = "experimental")]
pub(crate) const TIER3_FEATURES: FeatureSet = {
    let s = FeatureSet::new()
        .with(AnalysisFeature::AqMapMean)
        .with(AnalysisFeature::AqMapStddev)
        .with(AnalysisFeature::NoiseFloorY)
        .with(AnalysisFeature::NoiseFloorUv)
        .with(AnalysisFeature::QuantSurvivalY)
        .with(AnalysisFeature::QuantSurvivalUv)
        // Percentile features (Tier 3 / AQ map)
        .with(AnalysisFeature::AqMapP1)
        .with(AnalysisFeature::AqMapP5)
        .with(AnalysisFeature::AqMapP10)
        .with(AnalysisFeature::AqMapP25)
        .with(AnalysisFeature::AqMapP50)
        .with(AnalysisFeature::AqMapP75)
        .with(AnalysisFeature::AqMapP90)
        .with(AnalysisFeature::AqMapP95)
        .with(AnalysisFeature::AqMapP99)
        // Percentile features (Tier 3 / Noise floor Y)
        .with(AnalysisFeature::NoiseFloorYP1)
        .with(AnalysisFeature::NoiseFloorYP5)
        .with(AnalysisFeature::NoiseFloorYP10)
        .with(AnalysisFeature::NoiseFloorYP25)
        .with(AnalysisFeature::NoiseFloorYP50)
        .with(AnalysisFeature::NoiseFloorYP75)
        .with(AnalysisFeature::NoiseFloorYP90)
        .with(AnalysisFeature::NoiseFloorYP95)
        .with(AnalysisFeature::NoiseFloorYP99)
        // Percentile features (Tier 3 / Noise floor UV)
        .with(AnalysisFeature::NoiseFloorUvP1)
        .with(AnalysisFeature::NoiseFloorUvP5)
        .with(AnalysisFeature::NoiseFloorUvP10)
        .with(AnalysisFeature::NoiseFloorUvP25)
        .with(AnalysisFeature::NoiseFloorUvP50)
        .with(AnalysisFeature::NoiseFloorUvP75)
        .with(AnalysisFeature::NoiseFloorUvP90)
        .with(AnalysisFeature::NoiseFloorUvP95)
        .with(AnalysisFeature::NoiseFloorUvP99)
        // Percentile features (Tier 3 / QuantSurvival Y)
        .with(AnalysisFeature::QuantSurvivalYP1)
        .with(AnalysisFeature::QuantSurvivalYP5)
        .with(AnalysisFeature::QuantSurvivalYP10)
        .with(AnalysisFeature::QuantSurvivalYP25)
        .with(AnalysisFeature::QuantSurvivalYP50)
        .with(AnalysisFeature::QuantSurvivalYP75)
        .with(AnalysisFeature::QuantSurvivalYP90)
        .with(AnalysisFeature::QuantSurvivalYP95)
        .with(AnalysisFeature::QuantSurvivalYP99)
        // Percentile features (Tier 3 / QuantSurvival UV)
        .with(AnalysisFeature::QuantSurvivalUvP1)
        .with(AnalysisFeature::QuantSurvivalUvP5)
        .with(AnalysisFeature::QuantSurvivalUvP10)
        .with(AnalysisFeature::QuantSurvivalUvP25)
        .with(AnalysisFeature::QuantSurvivalUvP50)
        .with(AnalysisFeature::QuantSurvivalUvP75)
        .with(AnalysisFeature::QuantSurvivalUvP90)
        .with(AnalysisFeature::QuantSurvivalUvP95)
        .with(AnalysisFeature::QuantSurvivalUvP99)
        // Tier 3 extended
        .with(AnalysisFeature::NoiseFloorYVariance)
        .with(AnalysisFeature::NoiseFloorUvVariance)
        .with(AnalysisFeature::QuantSurvivalYVariance)
        .with(AnalysisFeature::QuantSurvivalUvVariance)
        // AQ map extended
        .with(AnalysisFeature::AqMapVariance)
        .with(AnalysisFeature::AqMapSkewness)
        .with(AnalysisFeature::AqMapKurtosis)
        // Dense percentile sweep (2026-05-07) — AqMap
        .with(AnalysisFeature::AqMapP15)
        .with(AnalysisFeature::AqMapP20)
        .with(AnalysisFeature::AqMapP30)
        .with(AnalysisFeature::AqMapP35)
        .with(AnalysisFeature::AqMapP40)
        .with(AnalysisFeature::AqMapP45)
        .with(AnalysisFeature::AqMapP55)
        .with(AnalysisFeature::AqMapP60)
        .with(AnalysisFeature::AqMapP65)
        .with(AnalysisFeature::AqMapP70)
        .with(AnalysisFeature::AqMapP80)
        .with(AnalysisFeature::AqMapP85)
        // Dense percentile sweep (2026-05-07) — NoiseFloorY
        .with(AnalysisFeature::NoiseFloorYP15)
        .with(AnalysisFeature::NoiseFloorYP20)
        .with(AnalysisFeature::NoiseFloorYP30)
        .with(AnalysisFeature::NoiseFloorYP35)
        .with(AnalysisFeature::NoiseFloorYP40)
        .with(AnalysisFeature::NoiseFloorYP45)
        .with(AnalysisFeature::NoiseFloorYP55)
        .with(AnalysisFeature::NoiseFloorYP60)
        .with(AnalysisFeature::NoiseFloorYP65)
        .with(AnalysisFeature::NoiseFloorYP70)
        .with(AnalysisFeature::NoiseFloorYP80)
        .with(AnalysisFeature::NoiseFloorYP85)
        .with(AnalysisFeature::NoiseFloorYP95)
        .with(AnalysisFeature::NoiseFloorYP99)
        // Dense percentile sweep (2026-05-07) — NoiseFloorUv
        .with(AnalysisFeature::NoiseFloorUvP1Dense)
        .with(AnalysisFeature::NoiseFloorUvP5Dense)
        .with(AnalysisFeature::NoiseFloorUvP10Dense)
        .with(AnalysisFeature::NoiseFloorUvP15)
        .with(AnalysisFeature::NoiseFloorUvP20)
        .with(AnalysisFeature::NoiseFloorUvP25Dense)
        .with(AnalysisFeature::NoiseFloorUvP30)
        .with(AnalysisFeature::NoiseFloorUvP35)
        .with(AnalysisFeature::NoiseFloorUvP40)
        .with(AnalysisFeature::NoiseFloorUvP45)
        .with(AnalysisFeature::NoiseFloorUvP55)
        .with(AnalysisFeature::NoiseFloorUvP60)
        .with(AnalysisFeature::NoiseFloorUvP65)
        .with(AnalysisFeature::NoiseFloorUvP70)
        .with(AnalysisFeature::NoiseFloorUvP80)
        .with(AnalysisFeature::NoiseFloorUvP85)
        .with(AnalysisFeature::NoiseFloorUvP99Dense)
        // Dense percentile sweep (2026-05-07) — QuantSurvivalY
        .with(AnalysisFeature::QuantSurvivalYP15)
        .with(AnalysisFeature::QuantSurvivalYP20)
        .with(AnalysisFeature::QuantSurvivalYP30)
        .with(AnalysisFeature::QuantSurvivalYP35)
        .with(AnalysisFeature::QuantSurvivalYP40)
        .with(AnalysisFeature::QuantSurvivalYP45)
        .with(AnalysisFeature::QuantSurvivalYP55)
        .with(AnalysisFeature::QuantSurvivalYP60)
        .with(AnalysisFeature::QuantSurvivalYP65)
        .with(AnalysisFeature::QuantSurvivalYP70)
        .with(AnalysisFeature::QuantSurvivalYP80)
        .with(AnalysisFeature::QuantSurvivalYP85)
        .with(AnalysisFeature::QuantSurvivalYP95)
        .with(AnalysisFeature::QuantSurvivalYP99)
        // Dense percentile sweep (2026-05-07) — QuantSurvivalUv
        .with(AnalysisFeature::QuantSurvivalUvP1)
        .with(AnalysisFeature::QuantSurvivalUvP5)
        .with(AnalysisFeature::QuantSurvivalUvP10)
        .with(AnalysisFeature::QuantSurvivalUvP15)
        .with(AnalysisFeature::QuantSurvivalUvP20)
        .with(AnalysisFeature::QuantSurvivalUvP25)
        .with(AnalysisFeature::QuantSurvivalUvP30)
        .with(AnalysisFeature::QuantSurvivalUvP35)
        .with(AnalysisFeature::QuantSurvivalUvP40)
        .with(AnalysisFeature::QuantSurvivalUvP45)
        .with(AnalysisFeature::QuantSurvivalUvP55)
        .with(AnalysisFeature::QuantSurvivalUvP60)
        .with(AnalysisFeature::QuantSurvivalUvP65)
        .with(AnalysisFeature::QuantSurvivalUvP70)
        .with(AnalysisFeature::QuantSurvivalUvP80)
        .with(AnalysisFeature::QuantSurvivalUvP85)
        .with(AnalysisFeature::QuantSurvivalUvP99);
    s
};

/// Features that require DCT computation.
/// Used by the analysis driver to decide whether to run the DCT pass.
#[cfg(feature = "experimental")]
pub(crate) const DCT_NEEDED_BY: FeatureSet = {
    let s = FeatureSet::new()
        .with(AnalysisFeature::DctCompressibilityY)
        .with(AnalysisFeature::DctCompressibilityCbCr)
        .with(AnalysisFeature::AvgQuantStepY)
        .with(AnalysisFeature::AvgQuantStepCbCr)
        .with(AnalysisFeature::AcEnergyFractionY)
        .with(AnalysisFeature::AqMapMean)
        .with(AnalysisFeature::AqMapStddev)
        .with(AnalysisFeature::NoiseFloorY)
        .with(AnalysisFeature::NoiseFloorUv)
        .with(AnalysisFeature::QuantSurvivalY)
        .with(AnalysisFeature::QuantSurvivalUv)
        // Percentile features (Tier 3 / AQ map)
        .with(AnalysisFeature::AqMapP1)
        .with(AnalysisFeature::AqMapP5)
        .with(AnalysisFeature::AqMapP10)
        .with(AnalysisFeature::AqMapP25)
        .with(AnalysisFeature::AqMapP50)
        .with(AnalysisFeature::AqMapP75)
        .with(AnalysisFeature::AqMapP90)
        .with(AnalysisFeature::AqMapP95)
        .with(AnalysisFeature::AqMapP99)
        // Percentile features (Tier 3 / Noise floor Y)
        .with(AnalysisFeature::NoiseFloorYP1)
        .with(AnalysisFeature::NoiseFloorYP5)
        .with(AnalysisFeature::NoiseFloorYP10)
        .with(AnalysisFeature::NoiseFloorYP25)
        .with(AnalysisFeature::NoiseFloorYP50)
        .with(AnalysisFeature::NoiseFloorYP75)
        .with(AnalysisFeature::NoiseFloorYP90)
        .with(AnalysisFeature::NoiseFloorYP95)
        .with(AnalysisFeature::NoiseFloorYP99)
        // Percentile features (Tier 3 / Noise floor UV)
        .with(AnalysisFeature::NoiseFloorUvP1)
        .with(AnalysisFeature::NoiseFloorUvP5)
        .with(AnalysisFeature::NoiseFloorUvP10)
        .with(AnalysisFeature::NoiseFloorUvP25)
        .with(AnalysisFeature::NoiseFloorUvP50)
        .with(AnalysisFeature::NoiseFloorUvP75)
        .with(AnalysisFeature::NoiseFloorUvP90)
        .with(AnalysisFeature::NoiseFloorUvP95)
        .with(AnalysisFeature::NoiseFloorUvP99)
        // Percentile features (Tier 3 / QuantSurvival Y)
        .with(AnalysisFeature::QuantSurvivalYP1)
        .with(AnalysisFeature::QuantSurvivalYP5)
        .with(AnalysisFeature::QuantSurvivalYP10)
        .with(AnalysisFeature::QuantSurvivalYP25)
        .with(AnalysisFeature::QuantSurvivalYP50)
        .with(AnalysisFeature::QuantSurvivalYP75)
        .with(AnalysisFeature::QuantSurvivalYP90)
        .with(AnalysisFeature::QuantSurvivalYP95)
        .with(AnalysisFeature::QuantSurvivalYP99)
        // Percentile features (Tier 3 / QuantSurvival UV)
        .with(AnalysisFeature::QuantSurvivalUvP1)
        .with(AnalysisFeature::QuantSurvivalUvP5)
        .with(AnalysisFeature::QuantSurvivalUvP10)
        .with(AnalysisFeature::QuantSurvivalUvP25)
        .with(AnalysisFeature::QuantSurvivalUvP50)
        .with(AnalysisFeature::QuantSurvivalUvP75)
        .with(AnalysisFeature::QuantSurvivalUvP90)
        .with(AnalysisFeature::QuantSurvivalUvP95)
        .with(AnalysisFeature::QuantSurvivalUvP99)
        // Tier 3 extended
        .with(AnalysisFeature::NoiseFloorYVariance)
        .with(AnalysisFeature::NoiseFloorUvVariance)
        .with(AnalysisFeature::QuantSurvivalYVariance)
        .with(AnalysisFeature::QuantSurvivalUvVariance)
        // AQ map extended
        .with(AnalysisFeature::AqMapVariance)
        .with(AnalysisFeature::AqMapSkewness)
        .with(AnalysisFeature::AqMapKurtosis)
        // DCT extended
        .with(AnalysisFeature::DctKurtosisY)
        .with(AnalysisFeature::DctZeroRunMeanY)
        // Dense percentile sweep (2026-05-07) — AqMap
        .with(AnalysisFeature::AqMapP15)
        .with(AnalysisFeature::AqMapP20)
        .with(AnalysisFeature::AqMapP30)
        .with(AnalysisFeature::AqMapP35)
        .with(AnalysisFeature::AqMapP40)
        .with(AnalysisFeature::AqMapP45)
        .with(AnalysisFeature::AqMapP55)
        .with(AnalysisFeature::AqMapP60)
        .with(AnalysisFeature::AqMapP65)
        .with(AnalysisFeature::AqMapP70)
        .with(AnalysisFeature::AqMapP80)
        .with(AnalysisFeature::AqMapP85)
        // Dense percentile sweep (2026-05-07) — NoiseFloorY
        .with(AnalysisFeature::NoiseFloorYP15)
        .with(AnalysisFeature::NoiseFloorYP20)
        .with(AnalysisFeature::NoiseFloorYP30)
        .with(AnalysisFeature::NoiseFloorYP35)
        .with(AnalysisFeature::NoiseFloorYP40)
        .with(AnalysisFeature::NoiseFloorYP45)
        .with(AnalysisFeature::NoiseFloorYP55)
        .with(AnalysisFeature::NoiseFloorYP60)
        .with(AnalysisFeature::NoiseFloorYP65)
        .with(AnalysisFeature::NoiseFloorYP70)
        .with(AnalysisFeature::NoiseFloorYP80)
        .with(AnalysisFeature::NoiseFloorYP85)
        .with(AnalysisFeature::NoiseFloorYP95)
        .with(AnalysisFeature::NoiseFloorYP99)
        // Dense percentile sweep (2026-05-07) — NoiseFloorUv
        .with(AnalysisFeature::NoiseFloorUvP1Dense)
        .with(AnalysisFeature::NoiseFloorUvP5Dense)
        .with(AnalysisFeature::NoiseFloorUvP10Dense)
        .with(AnalysisFeature::NoiseFloorUvP15)
        .with(AnalysisFeature::NoiseFloorUvP20)
        .with(AnalysisFeature::NoiseFloorUvP25Dense)
        .with(AnalysisFeature::NoiseFloorUvP30)
        .with(AnalysisFeature::NoiseFloorUvP35)
        .with(AnalysisFeature::NoiseFloorUvP40)
        .with(AnalysisFeature::NoiseFloorUvP45)
        .with(AnalysisFeature::NoiseFloorUvP55)
        .with(AnalysisFeature::NoiseFloorUvP60)
        .with(AnalysisFeature::NoiseFloorUvP65)
        .with(AnalysisFeature::NoiseFloorUvP70)
        .with(AnalysisFeature::NoiseFloorUvP80)
        .with(AnalysisFeature::NoiseFloorUvP85)
        .with(AnalysisFeature::NoiseFloorUvP99Dense)
        // Dense percentile sweep (2026-05-07) — QuantSurvivalY
        .with(AnalysisFeature::QuantSurvivalYP15)
        .with(AnalysisFeature::QuantSurvivalYP20)
        .with(AnalysisFeature::QuantSurvivalYP30)
        .with(AnalysisFeature::QuantSurvivalYP35)
        .with(AnalysisFeature::QuantSurvivalYP40)
        .with(AnalysisFeature::QuantSurvivalYP45)
        .with(AnalysisFeature::QuantSurvivalYP55)
        .with(AnalysisFeature::QuantSurvivalYP60)
        .with(AnalysisFeature::QuantSurvivalYP65)
        .with(AnalysisFeature::QuantSurvivalYP70)
        .with(AnalysisFeature::QuantSurvivalYP80)
        .with(AnalysisFeature::QuantSurvivalYP85)
        .with(AnalysisFeature::QuantSurvivalYP95)
        .with(AnalysisFeature::QuantSurvivalYP99)
        // Dense percentile sweep (2026-05-07) — QuantSurvivalUv
        .with(AnalysisFeature::QuantSurvivalUvP1)
        .with(AnalysisFeature::QuantSurvivalUvP5)
        .with(AnalysisFeature::QuantSurvivalUvP10)
        .with(AnalysisFeature::QuantSurvivalUvP15)
        .with(AnalysisFeature::QuantSurvivalUvP20)
        .with(AnalysisFeature::QuantSurvivalUvP25)
        .with(AnalysisFeature::QuantSurvivalUvP30)
        .with(AnalysisFeature::QuantSurvivalUvP35)
        .with(AnalysisFeature::QuantSurvivalUvP40)
        .with(AnalysisFeature::QuantSurvivalUvP45)
        .with(AnalysisFeature::QuantSurvivalUvP55)
        .with(AnalysisFeature::QuantSurvivalUvP60)
        .with(AnalysisFeature::QuantSurvivalUvP65)
        .with(AnalysisFeature::QuantSurvivalUvP70)
        .with(AnalysisFeature::QuantSurvivalUvP80)
        .with(AnalysisFeature::QuantSurvivalUvP85)
        .with(AnalysisFeature::QuantSurvivalUvP99);
    s
};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_set_basic_ops() {
        let a = FeatureSet::new()
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity);
        let b = FeatureSet::new()
            .with(AnalysisFeature::EdgeDensity)
            .with(AnalysisFeature::DctCompressibilityY);

        assert!(a.is_active(AnalysisFeature::Variance));
        assert!(!a.is_active(AnalysisFeature::DctCompressibilityY));
        assert!(a.intersects(b));
        assert!(!a.intersects(FeatureSet::new().with(AnalysisFeature::DctCompressibilityY)));
        assert_eq!(a.union(b).count(), 3);
        assert_eq!(a.intersect(b).count(), 1);
    }

    #[test]
    fn feature_id_roundtrip() {
        // All stable features have stable IDs
        let features = [
            AnalysisFeature::Variance,
            AnalysisFeature::MeanLuminance,
            AnalysisFeature::EdgeDensity,
            AnalysisFeature::LaplacianVariance,
            AnalysisFeature::DctCompressibilityY,
        ];
        for f in features {
            assert_eq!(AnalysisFeature::from_u16(f.id()), Some(f));
        }
    }

    #[test]
    fn feature_set_subset() {
        let small = FeatureSet::new()
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity);
        let big = small.with(AnalysisFeature::DctCompressibilityY);
        assert!(small.is_subset_of(big));
        assert!(!big.is_subset_of(small));
    }

    #[test]
    fn feature_set_without() {
        let a = FeatureSet::new()
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity);
        let b = a.without(AnalysisFeature::EdgeDensity);
        assert!(b.is_active(AnalysisFeature::Variance));
        assert!(!b.is_active(AnalysisFeature::EdgeDensity));
        assert_eq!(b.count(), 1);
    }

    #[test]
    fn supported_is_nonempty() {
        assert!(!FeatureSet::SUPPORTED.is_empty());
        // Must include all stable features
        assert!(FeatureSet::SUPPORTED.is_active(AnalysisFeature::Variance));
        assert!(FeatureSet::SUPPORTED.is_active(AnalysisFeature::EdgeDensity));
        assert!(FeatureSet::SUPPORTED.is_active(AnalysisFeature::LaplacianVariance));
    }

    #[test]
    fn feature_name_smoke() {
        assert_eq!(AnalysisFeature::Variance.name(), "variance");
        assert_eq!(AnalysisFeature::EdgeDensity.name(), "edge_density");
        assert_eq!(AnalysisFeature::LaplacianVariance.name(), "laplacian_variance");
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn dense_sweep_ids_in_range() {
        // All dense-sweep IDs are in 122-211.
        let dense = [
            AnalysisFeature::LaplacianVarianceP15,
            AnalysisFeature::LaplacianVarianceP20,
            AnalysisFeature::LaplacianVarianceP30,
            AnalysisFeature::LaplacianVarianceP35,
            AnalysisFeature::LaplacianVarianceP40,
            AnalysisFeature::LaplacianVarianceP45,
            AnalysisFeature::LaplacianVarianceP55,
            AnalysisFeature::LaplacianVarianceP60,
            AnalysisFeature::LaplacianVarianceP65,
            AnalysisFeature::LaplacianVarianceP70,
            AnalysisFeature::LaplacianVarianceP80,
            AnalysisFeature::LaplacianVarianceP85,
            AnalysisFeature::AqMapP15,
            AnalysisFeature::AqMapP85,
            AnalysisFeature::NoiseFloorYP15,
            AnalysisFeature::NoiseFloorYP99,
            AnalysisFeature::NoiseFloorUvP1Dense,
            AnalysisFeature::NoiseFloorUvP99Dense,
            AnalysisFeature::QuantSurvivalYP15,
            AnalysisFeature::QuantSurvivalYP99,
            AnalysisFeature::QuantSurvivalUvP1,
            AnalysisFeature::QuantSurvivalUvP99,
        ];
        for f in dense {
            let id = f.id();
            assert!(
                id >= 122 && id <= 211,
                "{:?} has id {} outside 122-211",
                f,
                id
            );
        }
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn tier3_features_includes_dense_sweep() {
        // Spot check: a few new dense-sweep variants must be in TIER3_FEATURES.
        assert!(TIER3_FEATURES.is_active(AnalysisFeature::AqMapP15));
        assert!(TIER3_FEATURES.is_active(AnalysisFeature::NoiseFloorYP99));
        assert!(TIER3_FEATURES.is_active(AnalysisFeature::QuantSurvivalUvP99));
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn dct_needed_by_includes_dense_sweep() {
        // Dense Tier-3 percentiles must also appear in DCT_NEEDED_BY.
        assert!(DCT_NEEDED_BY.is_active(AnalysisFeature::AqMapP15));
        assert!(DCT_NEEDED_BY.is_active(AnalysisFeature::QuantSurvivalYP99));
        assert!(DCT_NEEDED_BY.is_active(AnalysisFeature::QuantSurvivalUvP99));
    }

    // Verify tier bundles are disjoint (no feature belongs to two tiers).
    // This is a compile-time-checkable property; this test catches bugs
    // introduced by editing the const sets.
    #[test]
    #[cfg(feature = "experimental")]
    fn tier_bundles_are_disjoint() {
        let tier_sets = [
            ("TIER3", TIER3_FEATURES),
            ("DCT_NEEDED_BY", DCT_NEEDED_BY),
        ];
        // Each pair must not have an unexpected overlap
        // (TIER3 is a subset of DCT_NEEDED_BY by design, so we test
        //  that they are not *identical* — i.e. DCT_NEEDED_BY is strictly
        //  larger — and that TIER3 ⊆ DCT_NEEDED_BY).
        assert!(
            TIER3_FEATURES.is_subset_of(DCT_NEEDED_BY),
            "TIER3_FEATURES must be a subset of DCT_NEEDED_BY"
        );
        assert!(
            !DCT_NEEDED_BY.is_subset_of(TIER3_FEATURES),
            "DCT_NEEDED_BY must be strictly larger than TIER3_FEATURES"
        );
        // No two *named* bundles should overlap with features that don't
        // belong in both — spot-check: Tier1 Laplacian features (IDs 17,32-41)
        // should not appear in TIER3_FEATURES.
        let tier1_lap = FeatureSet::new()
            .with(AnalysisFeature::LaplacianVariance)
            .with(AnalysisFeature::LaplacianVarianceP1)
            .with(AnalysisFeature::LaplacianVarianceP5)
            .with(AnalysisFeature::LaplacianVarianceP10)
            .with(AnalysisFeature::LaplacianVarianceP25)
            .with(AnalysisFeature::LaplacianVarianceP50)
            .with(AnalysisFeature::LaplacianVarianceP75)
            .with(AnalysisFeature::LaplacianVarianceP90)
            .with(AnalysisFeature::LaplacianVarianceP95)
            .with(AnalysisFeature::LaplacianVarianceP99)
            .with(AnalysisFeature::LaplacianVariancePeak);
        assert!(
            !TIER3_FEATURES.intersects(tier1_lap),
            "Tier-1 Laplacian features must not appear in TIER3_FEATURES"
        );
        let _ = tier_sets; // suppress unused warning
        for (a_name, a) in &tier_sets {
            for (b_name, b) in &tier_sets {
                if a_name == b_name {
                    continue;
                }
                assert!(!a.intersects(*b), "tier bundles overlap (this is a bug)");
            }
        }
    }
}
