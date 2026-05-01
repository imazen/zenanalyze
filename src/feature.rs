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
            /// Sidecars / Python fitters / JSON dumps that key
            /// features by stable id should read this.
            #[inline]
            pub const fn id(self) -> u16 {
                self as u16
            }

            /// Inverse of [`Self::id`]. Returns `None` for unknown
            /// numbers — including ids whose variants are retired
            /// or gated behind a cargo feature that isn't enabled
            /// in this build. Sidecars deserializing a wire-format
            /// id must accept the `None` arm gracefully (older /
            /// newer / cfg-disabled builds).
            #[inline]
            #[allow(unused_doc_comments, deprecated)]
            pub const fn from_u16(n: u16) -> Option<Self> {
                // `unused_doc_comments` allow: variants in the
                // table carry `///` docstrings that the macro forwards
                // here so that `#[cfg(...)]` attrs stay in lockstep;
                // doc comments don't render on match arms but are
                // valid syntax there.
                match n {
                    $(
                        $(#[$variant_attr])*
                        $id => Some(Self::$variant),
                    )*
                    _ => None,
                }
            }

            /// Run-time check: is this feature *currently* computable
            /// in this build? Crate-internal — public callers learn
            /// "feature is missing" by getting `None` from
            /// [`AnalysisResults::get`].
            #[inline]
            pub(crate) const fn is_active(self) -> bool {
                // Today every variant in this build is active. The
                // experimental cargo feature gates whole variants out
                // at compile time, so a variant existing implies it's
                // active. Variant-level deprecation that keeps the
                // variant in the enum but marks it inactive would
                // wire its arm here.
                match self {
                    _ => true,
                }
            }

            /// Stable, machine-readable name (snake_case). Matches
            /// the field names in the internal [`RawAnalysis`] so
            /// JSON sidecars and downstream Python fitters keep
            /// working. Generated from the field-name token via
            /// `stringify!`.
            #[allow(unused_doc_comments, deprecated)]
            pub const fn name(self) -> &'static str {
                match self {
                    $(
                        $(#[$variant_attr])*
                        Self::$variant => stringify!($field),
                    )*
                }
            }
        }

        impl FeatureSet {
            /// The set of [`AnalysisFeature`]s this build can compute.
            ///
            /// Built by walking the `features_table!` rows in order;
            /// per-row `#[cfg(feature = "...")]` propagates here, so
            /// disabling a cargo feature shrinks `SUPPORTED`
            /// automatically. Always intersect a caller's wish-list
            /// against `FeatureSet::SUPPORTED` before passing to
            /// [`AnalysisQuery`] — asking for an unsupported feature
            /// isn't an error, just yields `None` from
            /// [`AnalysisResults::get`].
            ///
            /// Use this rather than enumerating every variant by hand.
            /// It's the only "all features" entry point in the public
            /// API; there is intentionally no `FeatureSet::all()`
            /// because production callers should request only what
            /// they need.
            #[allow(unused_doc_comments, unused_mut, unused_assignments, deprecated)]
            pub const SUPPORTED: Self = {
                // Const block with `let mut` so individual `with`
                // calls can be cfg-gated. The chain form
                // `Self::new().with(X).with(Y)` doesn't allow
                // attributes on individual method calls; this form
                // does. Per-row doc comments forward here too — they
                // don't render on a block expression but are valid
                // syntax. `unused_mut` covers the
                // `experimental`-fully-disabled build where every
                // gated arm vanishes.
                let mut s = Self::new();
                $(
                    $(#[$variant_attr])*
                    {
                        s = s.with(AnalysisFeature::$variant);
                    }
                )*
                s
            };
        }

        /// Dense flat record the SIMD tiers write to with zero
        /// overhead. Each SIMD inner loop already produces a single
        /// `f32`/`u32`/`bool` per feature; storing those into named
        /// struct fields is one mov per feature and lets LLVM keep
        /// accumulators in registers across the loop boundary.
        ///
        /// `pub(crate)` only — never crosses the public API.
        /// Translated once at the end of `analyze_features` into the
        /// sparse [`AnalysisResults`] based on the caller's requested
        /// [`FeatureSet`] (only requested features are copied).
        /// Default = zero / `false` for every field. Cfg-gated rows in
        /// the table become cfg-gated fields here.
        ///
        /// Generated from the [`features_table!`] invocation in
        /// lockstep with [`AnalysisFeature`] and
        /// [`FeatureSet::SUPPORTED`].
        #[derive(Default, Debug, Clone, Copy)]
        pub(crate) struct RawAnalysis {
            $(
                $(#[$variant_attr])*
                pub $field: $ty,
            )*
        }

        impl RawAnalysis {
            /// Translate this dense struct into the sparse public
            /// results, copying **only** the features the caller
            /// asked for. Iteration is in id order, so
            /// [`AnalysisResults`]' insertion-sort hits the
            /// append-at-end fast path on every call. Cfg-gated rows
            /// drop out of the copy list when their feature is
            /// disabled.
            #[allow(unused_doc_comments, deprecated)]
            pub(crate) fn into_results(
                self,
                requested: FeatureSet,
                geometry: ImageGeometry,
                source_descriptor: zenpixels::PixelDescriptor,
            ) -> AnalysisResults {
                let mut r = AnalysisResults::new(requested, geometry, source_descriptor);
                $(
                    $(#[$variant_attr])*
                    #[allow(deprecated)]
                    {
                        if requested.contains(AnalysisFeature::$variant) {
                            r.set(AnalysisFeature::$variant, self.$field);
                        }
                    }
                )*
                r
            }
        }

    };
}

// ---------------------- Single source of truth -----------------------
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
    /// `u32`. Distinct 5-bit-per-channel RGB bins observed in
    /// `[0, 32_768]` (the bin storage caps at 32³ = 32 768 cells, one
    /// per L1D way). **This under-counts the true 8-bit distinct
    /// colour count whenever two 8-bit colours fold into the same
    /// 5-bit cell.** Bias is always toward undercounting, never over.
    ///
    /// Quantitative collision rate (uniformly-distributed colours):
    /// - true count 256 → expected bins ≈ 253 (~1 % undercount)
    /// - true count 1 024 → ≈ 1 009 (~1.5 %)
    /// - true count 4 096 → ≈ 3 611 (~12 %)
    /// - true count 8 192 → ≈ 7 308 (~11 %)
    /// - true count > 32 768 → saturates at 32 768
    ///
    /// Consequence for downstream features:
    /// - [`Self::PaletteFitsIn256`] can return `true` for sources with
    ///   257–~300 true 8-bit colours when collisions pull the binned
    ///   count back under 257. Encoders that try indexed mode based
    ///   on this signal would fail to fit the actual colour set and
    ///   fall back to truecolor — correctness preserved, one wasted
    ///   attempt per false positive.
    /// - [`Self::PaletteLog2Size`] (which buckets `ceil(log2)` of this
    ///   value) tends to land in the same bucket as the true count
    ///   except at exact-power-of-2 boundaries, where the undercount
    ///   can shift one bucket smaller.
    ///
    /// 5-bit was chosen so the 32 KB flag array fits in one L1D way;
    /// 6-bit (256 KB) would cache-thrash the entire scan. The
    /// accuracy tradeoff was paid intentionally.
    DistinctColorBins = 10 : u32 => distinct_color_bins,
    // id 11 reserved (was `DistinctColorBinsChao1`, removed pre-0.1.0).
    // Today's full-pass scan made the Chao1 correction term degenerate
    // — it always returned the same value as `DistinctColorBins`, with
    // no path to diverge until a budget-sampled palette variant is
    // implemented. Keeping a permanently-equal field invited callers
    // to write `if chao1 > distinct` checks that would silently break
    // when the divergence-producing path lands. The id is reserved so
    // the future budget-sampled variant can re-introduce the variant
    // at the same number for FeatureSet wire-format compatibility.
    /// `f32`. `DistinctColorBins / min(pixel_count, 32 768)`.
    #[cfg(feature = "experimental")]
    PaletteDensity = 12 : f32 => palette_density,

    // ---------------- Tier 2: per-channel per-axis chroma ------------
    /// `f32`. Cb horizontal gradient energy / 1e5.
    CbHorizSharpness = 13 : f32 => cb_horiz_sharpness,
    /// `f32`. Cb vertical gradient energy / 1e5.
    CbVertSharpness = 14 : f32 => cb_vert_sharpness,
    /// `f32`. Cb peak gradient magnitude. Calibrated so natural
    /// photographic content lands `< 100`; saturated synthetic
    /// inputs (alternating-channel chroma stripes) can reach
    /// `~666`. Renormalising onto a stable `[0, 100]` ceiling is a
    /// follow-up calibration task — until then, code that wants
    /// "is this peak unusually high?" should compare against a
    /// content-class-appropriate threshold rather than clamping.
    CbPeakSharpness = 15 : f32 => cb_peak_sharpness,
    /// `f32`. Cr horizontal gradient energy / 1e5.
    CrHorizSharpness = 16 : f32 => cr_horiz_sharpness,
    /// `f32`. Cr vertical gradient energy / 1e5.
    CrVertSharpness = 17 : f32 => cr_vert_sharpness,
    /// `f32`. Cr peak gradient magnitude. Same calibration story as
    /// [`Self::CbPeakSharpness`] — natural photographs `< 100`,
    /// saturated synthetic content up to `~666`.
    CrPeakSharpness = 18 : f32 => cr_peak_sharpness,

    // ---------------- Tier 3: DCT energy + entropy -------------------
    /// `f32`. `Σ AC[k≥16] / Σ AC[k∈1..16]` over sampled 8×8 luma blocks.
    HighFreqEnergyRatio = 19 : f32 => high_freq_energy_ratio,
    /// `f32`. Shannon entropy of a 32-bin luma histogram, in bits.
    LumaHistogramEntropy = 20 : f32 => luma_histogram_entropy,
    /// `f32`. Mean libwebp α on sampled luma 8×8 DCT blocks. Higher
    /// = harder to compress (more spread AC, fewer near-zero coefs).
    /// **Range:** theoretical `[0, ~8064]` from `256 * last_non_zero
    /// / max_count`; on real photo corpora the median sits at ~16,
    /// p90 at ~30. Downstream calibration must NOT clamp / normalise
    /// against 255 (earlier docs were wrong about that).
    #[cfg(feature = "experimental")]
    DctCompressibilityY = 21 : f32 => dct_compressibility_y,
    /// `f32`. Same shape on chroma DCT blocks, `max(α_cb, α_cr)` per
    /// block. Same `[0, ~8064]` theoretical range; chroma values run
    /// lower in practice (median ~5 on photos). Scale matches
    /// [`Self::DctCompressibilityY`].
    #[cfg(feature = "experimental")]
    DctCompressibilityUV = 22 : f32 => dct_compressibility_uv,
    /// `f32`. Fraction `[0, 1]` of sampled blocks matching another.
    /// Strongest single photo-vs-screen-content discriminator on
    /// real corpora.
    ///
    /// **Empirical AUC = 0.880** for screen-vs-photo on a 219-image
    /// labeled corpus (cid22 + clic2025 + gb82 + gb82-sc + imageflow
    /// + kadid10k + qoi-benchmark) — higher than every other shipped
    /// feature, including the derived [`Self::ScreenContentLikelihood`]
    /// (AUC 0.83). Photos: p50 = 0.002, p90 = 0.037. Screens: p50 =
    /// 0.726, p90 = 0.906. **Recommended operating threshold:
    /// `patch_fraction >= 0.27`** (F1 = 0.769, P = 0.91, R = 0.67)
    /// for screen-like classification.
    ///
    /// Originally validated on a smaller 50-photo / 10-screen pilot
    /// corpus where the default-budget classifier scored ROC-AUC =
    /// 0.978 (vs 0.980 at dense scan) and Spearman ρ between
    /// default and dense = 0.887. The 219-image labeled-corpus AUC
    /// is lower because that corpus includes screen-like
    /// illustrations and edge-case "uniform photos" that genuinely
    /// sit closer to the boundary; the rank order remains correct
    /// and the 1024-block default budget is fit for codec dispatch
    /// as shipped.
    ///
    /// Bumping `hf_max_blocks` to 4096+ tightens absolute error
    /// further but doesn't materially improve the classifier — see
    /// `docs/calibration-corpus-2026-04-27.md` for the full
    /// per-class distribution and AUC ranking.
    ///
    /// Gated behind `experimental` for 0.1.0 because no in-tree
    /// codec consumes it yet; promote when a consumer wires up.
    #[cfg(feature = "experimental")]
    PatchFraction = 23 : f32 => patch_fraction,

    // ---------------- Alpha (always full-scan when present) ----------
    /// `bool`. True iff the source has straight (unassociated) alpha.
    AlphaPresent = 24 : bool => alpha_present,
    /// `f32`. Fraction of sampled pixels with alpha < 255.
    AlphaUsedFraction = 25 : f32 => alpha_used_fraction,
    /// `f32`. Bimodal-ness of the alpha histogram, `[0, 0.5]`.
    AlphaBimodalScore = 26 : f32 => alpha_bimodal_score,

    // ---------------- Derived likelihoods ----------------------------

    // ---------------- Quick-path palette signals --------------------
    /// `bool`. Source fits in 256 distinct 5-bit-per-channel RGB
    /// bins — an indexed-mode codec can represent it without further
    /// colour quantization at this resolution. Drives the binary
    /// "encode as indexed?" decision when the caller doesn't care
    /// about palette size. Computed via an early-exit scan that bails
    /// at the 257th distinct bin (~half-image walk on truecolor, full
    /// walk on indexed-class content). Gated behind `experimental`.
    ///
    /// **Bin-storage bias.** Inherits the under-count bias documented
    /// on [`Self::DistinctColorBins`]: this can return `true` for
    /// sources with 257–~300 true 8-bit colours when 5-bit-cell
    /// collisions pull the binned count back under 257. An encoder
    /// consuming this would attempt indexed mode and fall back to
    /// truecolor on the failed fit — correct, but slightly wasteful.
    /// Callers who can't tolerate the false-positive rate should
    /// run their own exact-8-bit pass on the borderline cases.
    #[cfg(feature = "experimental")]
    PaletteFitsIn256 = 31 : bool => palette_fits_in_256,

    // ---------------- Depth tier (source-direct HDR / bit-depth) -----
    /// `f32`. Peak luminance over sampled pixels in nits. Computed
    /// against source samples directly (no `RowConverter` tonemap),
    /// honoring the descriptor's transfer function. SDR sources hit
    /// ~80 nits; PQ ~10 000; HLG ~1 000.
    #[cfg(feature = "hdr")]
    PeakLuminanceNits = 32 : f32 => peak_luminance_nits,
    /// `f32`. 99th-percentile luminance in nits (robust against single
    /// hot pixels).
    #[cfg(feature = "hdr")]
    P99LuminanceNits = 33 : f32 => p99_luminance_nits,
    /// `f32`. HDR headroom in stops: `log2(peak_nits / 80)`. SDR ⇒ 0;
    /// 1 000-nit HLG ⇒ ~3.6; 10 000-nit PQ ⇒ ~6.97.
    #[cfg(feature = "hdr")]
    HdrHeadroomStops = 34 : f32 => hdr_headroom_stops,
    /// `f32`. Fraction `[0, 1]` of sampled pixels above 100 nits.
    #[cfg(feature = "hdr")]
    HdrPixelFraction = 35 : f32 => hdr_pixel_fraction,
    /// `f32`. Largest single-channel linear value across sampled
    /// pixels. `> 1.0` ⇒ source carries above-sRGB values that would
    /// clip if narrowed to sRGB primaries.
    #[cfg(feature = "hdr")]
    WideGamutPeak = 36 : f32 => wide_gamut_peak,
    /// `f32`. Fraction `[0, 1]` of sampled pixels with at least one
    /// channel above 1.0 in linear light.
    #[cfg(feature = "hdr")]
    WideGamutFraction = 37 : f32 => wide_gamut_fraction,
    /// `u32`. Effective bit depth: smallest power-of-2 quantization
    /// grid the sampled values populate. `{8, 10, 12, 14, 16, 32}`.
    /// For u8 sources always 8; u8-promoted u16 detected as 8 via the
    /// low-byte distinct-count probe.
    #[cfg(feature = "hdr")]
    EffectiveBitDepth = 38 : u32 => effective_bit_depth,
    /// `bool`. `true` iff peak luminance well exceeds the SDR threshold
    /// AND the source transfer function can carry HDR
    /// (`Pq` / `Hlg` / `Linear`). Catches the hard case the standard
    /// tiers miss: PQ-encoded content whose tonemapped rendition looks
    /// like SDR but whose source carries far more dynamic range.
    #[cfg(feature = "hdr")]
    HdrPresent = 39 : bool => hdr_present,
    /// `f32`. Fraction `[0, 1]` of sampled pixels whose linear-RGB,
    /// projected from the source primaries into BT.709 / sRGB, has
    /// every channel within `[-ε, 1 + ε]`. **Descriptor-gap signal:**
    /// `1.0` ⇒ the source declares wider primaries (P3 / Rec.2020 /
    /// AdobeRGB) but its pixels actually live in the sRGB sub-gamut,
    /// so codecs can encode it with sRGB primaries and save bits on
    /// the colour-metadata + drop the gamut-extended encoder modes.
    /// For sRGB-declared sources this is trivially `1.0`. Threshold
    /// of `≥ 0.99` is a reasonable "downcast safe" cutoff.
    #[cfg(feature = "hdr")]
    GamutCoverageSrgb = 46 : f32 => gamut_coverage_srgb,
    /// `f32`. Same shape, projecting into Display P3. **Descriptor-
    /// gap signal:** for a Rec.2020-declared source, `1.0` here means
    /// the content is encodable in P3 (smaller container than
    /// Rec.2020). Useful as a middle tier when the image isn't sRGB-
    /// safe but doesn't actually use the full Rec.2020 gamut either.
    #[cfg(feature = "hdr")]
    GamutCoverageP3 = 47 : f32 => gamut_coverage_p3,
    /// `f32`. Fraction `[0, 1]` of sampled luma 8×8 blocks where
    /// ≥ 90 % of AC energy lives in the lowest-zigzag positions —
    /// **smooth-content / gradient signal**. Drives JXL
    /// `with_force_strategy` (DCT16 / DCT32 selection — large
    /// transforms pay off when most energy is in the lowest
    /// frequencies) and zenrav1e deblock-strength scaling. Distinct
    /// from `high_freq_energy_ratio` (global mean): this is per-
    /// block-thresholded, robust to a few high-detail blocks dragging
    /// the mean.
    #[cfg(feature = "experimental")]
    GradientFraction = 48 : f32 => gradient_fraction,

    // ---------------- Tier 1 piggyback: cheap secondary signals -----
    /// `f32`. Fraction `[0, 1]` of sampled pixels whose
    /// `max(|R-G|, |G-B|, |R-B|) ≤ 4`. ≥ 0.99 ⇒ effectively grayscale.
    /// Drives zenjpeg `ColorMode::Grayscale`, png/avif/jxl single-
    /// channel encode paths.
    #[cfg(feature = "experimental")]
    GrayscaleScore = 40 : f32 => grayscale_score,

    // ---------------- Tier 3 piggyback: AQ-map signals --------------
    /// `f32`. Mean of `log10(1 + Σ AC²)` over sampled luma 8×8 blocks.
    /// Image-average busyness signal; AQ orchestrators read this for
    /// the global "how textured is the image" baseline.
    #[cfg(feature = "experimental")]
    AqMapMean = 41 : f32 => aq_map_mean,
    /// `f32`. Standard deviation of the same per-block log-AC-energy.
    /// Drives zenjpeg hybrid trellis lambda scaling, webp
    /// segments+sns_strength, avif vaq_strength — high std ⇒
    /// heterogeneous content where AQ pays off.
    #[cfg(feature = "experimental")]
    AqMapStd = 42 : f32 => aq_map_std,
    /// `f32`. Robust luma noise floor estimate, normalized to
    /// `[0, 1]`. 10th percentile of √(low-AC-energy / 15) across
    /// sampled luma 8×8 blocks ÷ 32. Drives zenjpeg `pre_blur`,
    /// jxl `noise/denoise`, webp `sns_strength`.
    #[cfg(feature = "experimental")]
    NoiseFloorY = 43 : f32 => noise_floor_y,
    /// `f32`. Same on chroma — `max(p10_cb, p10_cr)`. Drives chroma-
    /// channel denoise scheduling.
    #[cfg(feature = "experimental")]
    NoiseFloorUV = 44 : f32 => noise_floor_uv,

    /// `f32`. Fraction `[0, 1]` of sampled pixels in the canonical
    /// chrominance-only skin-tone region. The chroma gates are
    /// **invariant to skin pigmentation** (Cb / Cr quantify hue, not
    /// brightness), so the same thresholds work across light, medium,
    /// and dark skin. Luma covers a wide range to span every tone:
    ///
    /// - `Y  ∈ [40,  240]` — spans deep shadow on dark skin to bright
    ///   highlight on light skin without rejecting either end
    /// - `Cb ∈ [77,  127]` — Chai & Ngan (1999) chrominance bound
    /// - `Cr ∈ [133, 173]` — Chai & Ngan (1999) chrominance bound
    ///
    /// Computed per-pixel in Tier 1 alongside the existing grayscale
    /// counter, reusing the BT.601 fixed-point YCbCr conversion. Zero
    /// added allocations; ~2 ns/pixel on a 7950X.
    ///
    /// **One-direction signal.** Non-zero fraction is strong evidence
    /// of a natural photograph (humans, animals, food). Zero fraction
    /// is **not** evidence against a photograph — landscapes,
    /// architecture, and macro shots without skin tones all score
    /// zero. Use as a positive-only confirmation, never as a negative
    /// classifier.
    ///
    /// **Why YCbCr instead of CIELAB.** CIELAB skin classifiers
    /// (Garcia & Tziritas 1999) outperform YCbCr by ~2 percentage
    /// points on standard skin-detection benchmarks but cost a
    /// non-linear sRGB → XYZ → LAB conversion per pixel. The Chai-Ngan
    /// YCbCr classifier is within ~5 % of LAB at zero extra
    /// arithmetic — already paid for by `chroma_complexity`,
    /// `cb_sharpness`, and the BT.601 luma the analyzer needs anyway.
    ///
    /// **Empirical ranges** (from a 219-image labeled corpus —
    /// `docs/calibration-corpus-2026-04-27.md`):
    ///
    /// - `photo_natural`:   p10 = 0.009, p50 = 0.130, p90 = 0.543
    /// - `photo_portrait`:  p10 = 0.047, p50 = 0.241, p90 = 0.541 — 94 % > 1 %
    /// - `photo_detailed`:  p10 = 0.021, p50 = 0.300, p90 = 0.470
    /// - `screen_document`: p10 = 0.000, p50 = 0.006, p90 = 0.077
    /// - `screen_ui`:       p10 = 0.000, p50 = 0.027, p90 = 0.127
    /// - `illustration`:    p10 = 0.001, p50 = 0.081, p90 = 0.323
    ///
    /// AUC = `0.799` for photo-vs-other classification — comparable
    /// to [`Self::NaturalLikelihood`] (0.814).
    ///
    /// **Operating threshold:** `skin_tone_fraction >= 0.05` gives
    /// `P = 0.89, R = 0.76, F1 = 0.82` for photo classification.
    /// Lower thresholds (`> 0`) maximize recall (`F1 = 0.882`).
    ///
    /// References: Chai & Ngan, "Face segmentation using skin-color
    /// map in videophone applications", IEEE TCSVT 1999;
    /// Vezhnevets et al., "A Survey on Pixel-Based Skin Color
    /// Detection Techniques", Graphicon 2003.
    #[cfg(feature = "experimental")]
    SkinToneFraction = 49 : f32 => skin_tone_fraction,

    /// `f32`. Standard deviation of luma gradient magnitudes across
    /// pixels that crossed the [`Self::EdgeDensity`] threshold
    /// (`|∇L|² > 400`, i.e. `|∇L| > 20`). Range `[0, ~150]` on the
    /// 0–255 luma scale.
    ///
    /// Tier 1 piggyback: the same SIMD edge sweep that produces
    /// `edge_density` accumulates `Σ g` and `Σ g²` over the threshold-
    /// crossing subset. Stddev is computed at row close from those
    /// running sums. Returns `0.0` if zero edges crossed (smooth
    /// image or below-threshold-only gradients).
    ///
    /// **Physical signal.** Natural photographs have edges anti-
    /// aliased by lens MTF + sensor pixel pitch — gradient magnitudes
    /// cluster tightly around the optical cutoff (typical stddev
    /// ~`8–18` on 0–255 luma). Digital artwork has either no edges
    /// (smooth gradients), single-pixel edges (line art — already
    /// caught by [`Self::LineArtScore`]), or **bimodal** gradients
    /// from variable-pressure brushwork or stylization (typical
    /// stddev `> 25`). JPEG-roundtripped artwork's blocking artifacts
    /// also widen this stddev relative to a pristine PNG photograph.
    ///
    /// **Empirical ranges** (from a 219-image labeled corpus —
    /// `docs/calibration-corpus-2026-04-27.md`):
    ///
    /// - `photo_natural`:   p10 = 15.9, p50 = 24.2, p90 = 31.8
    /// - `photo_portrait`:  p10 = 15.3, p50 = 20.7, p90 = 27.0
    /// - `photo_detailed`:  p10 = 18.2, p50 = 23.1, p90 = 32.0
    /// - `illustration`:    p10 = 12.9, p50 = 20.9, p90 = 26.7
    /// - `screen_document`: p10 = 42.0, p50 = 55.3, p90 = 57.5
    /// - `screen_ui`:       p10 = 31.6, p50 = 42.1, p90 = 54.4
    ///
    /// AUC = `0.843` for screen-vs-photo classification (high values →
    /// screen content). The strongest single screen-content signal
    /// after [`Self::PatchFraction`].
    ///
    /// **Operating thresholds:**
    /// - `edge_slope_stdev > 35` ⇒ very likely screen / chart / UI
    /// - `15 ≤ edge_slope_stdev ≤ 32` ⇒ photographic-edge distribution
    /// - `< 15` with low [`Self::EdgeDensity`] ⇒ smooth content
    ///   (illustrations or low-detail photos overlap here, ~13–27)
    ///
    /// Tracks the issue #123 proposal `EdgeSlopeStdev`.
    #[cfg(feature = "experimental")]
    EdgeSlopeStdev = 50 : f32 => edge_slope_stdev,

    // ---------------- Tier 3 patch-fingerprint experiments ----------
    // id 51 reserved (was `PatchFractionWht`, removed pre-stabilization
    // because dHash dominated it on cost without AUC loss; ablation
    // results in imazen/zenanalyze#1).
    /// `f32`. **Experimental.** Cost-efficient sibling of
    /// [`Self::PatchFraction`]: same sort-and-sweep collision-fraction
    /// construction, but the per-block fingerprint is the 64-bit dHash
    /// of raw 8×8 luma (`bit[i*8+j] = pixels[i][j+1] > pixels[i][j]`)
    /// folded to 32 bits via XOR of high/low halves. Pure pixel
    /// comparisons — **~10× cheaper per block** than the DCT-based
    /// `patch_fraction`. Brightness-invariant; captures gradient
    /// direction and edge layout.
    ///
    /// **Validated AUC** on the 219-image labeled corpus (vs is_screen):
    /// 0.852 (DCT version: 0.880). **Peak F1: 0.779** at threshold ≥ 0.40
    /// (DCT version: 0.763 at ≥ 0.40 — `patch_fraction_fast` wins).
    /// Pearson correlation with `patch_fraction`: 0.99 — same content
    /// signal, different cost / noise-floor profile. See zenanalyze#1
    /// for the full ablation.
    #[cfg(feature = "experimental")]
    PatchFractionFast = 52 : f32 => patch_fraction_fast,

    // ---------------- Tier 3 quant-survival compressibility ---------
    /// `f32`. **Experimental.** Mean fraction of luma AC coefficients
    /// (zigzag 1..63) that survive jpegli-default quantization at
    /// distance 2.0 (q=75 area). Approximates per-block JPEG file-size
    /// cost. Photos: ~0.10–0.25; UI / text edges: ~0.30–0.50; flat
    /// regions: ~0.0. Drives codec dispatch decisions about whether
    /// JPEG-style quantization will preserve content (high survival ⇒
    /// JPEG keeps the detail) vs other codecs.
    #[cfg(feature = "experimental")]
    QuantSurvivalY = 53 : f32 => quant_survival_y,
    /// `f32`. **Experimental.** Same for chroma — `max(survival_cb,
    /// survival_cr)` per block, mean across blocks. Useful for
    /// separating low-chroma photos (clear sky) from high-chroma
    /// screens (UI accent colors).
    #[cfg(feature = "experimental")]
    QuantSurvivalUv = 54 : f32 => quant_survival_uv,

    // ---------------- Strict-equality grayscale classifier ----------
    /// `bool`. **True iff every pixel in the image has `R == G == B`**
    /// (no tolerance). Distinct from [`Self::GrayscaleScore`] which
    /// surfaces a 4-unit-tolerance fraction; this is the binary signal
    /// codec selectors use to decide whether to encode without chroma
    /// planes (YUV400 JPEG / monochrome JXL / single-plane WebP).
    ///
    /// Computed by [`crate::grayscale`] as a row-by-row OR-reduction
    /// of `(r ^ g) | (g ^ b)` with **early exit** on the first
    /// non-gray row. Typical photo: bails on row 1, ~6 µs at 4 MP.
    /// Truly grayscale 4 MP image: walks every row, ~3-5 ms. Mean
    /// across a mixed corpus: < 100 µs at any size.
    ///
    /// Independent of the palette tier — does not require
    /// [`Self::GrayscaleScore`] / [`Self::DistinctColorBins`] to be
    /// requested. Set both if you want both the binary classifier and
    /// the tolerance fraction.
    IsGrayscale = 55 : bool => is_grayscale,

    // ---------------- Dimension features ----------------------------
    // Pure descriptor math — no per-pixel work. Computed once per
    // call from `(width, height, descriptor)`. Always-on, no cargo
    // feature gate. Issue #42.
    /// `u32`. Total pixel count `w * h`. Saturates at `u32::MAX` for
    /// images larger than ~4.29 gigapixels (rare; we don't support
    /// images that big elsewhere).
    PixelCount = 56 : u32 => pixel_count,
    /// `f32`. `ln(w * h)`, natural log. Useful as a smooth
    /// resolution axis for predictors (vs `size_class` one-hot).
    LogPixels = 57 : f32 => log_pixels,
    /// `u32`. `min(w, h)`. Catches strips and thumbnails directly —
    /// a 1024×1 image and a 32×32 image have very different
    /// per-pixel codec costs but the same `PixelCount`.
    MinDim = 58 : u32 => min_dim,
    /// `u32`. `max(w, h)`. Pairs with `MinDim` for shape-aware
    /// reasoning.
    MaxDim = 59 : u32 => max_dim,
    /// `f32`. Uncompressed bitmap byte count: `w * h * channels *
    /// bytes_per_sample`, cast to f32. The natural reference for
    /// "how much could compression possibly save" — predictors
    /// regress against `log(bitmap_bytes)`. f32 is exact for
    /// values up to 2²⁴ ≈ 16 MB; larger images lose ~ULP-level
    /// precision per byte but stay correct in log space (≤ 1 ULP
    /// drift in log10 ≈ 1e-7) — fine for ML features.
    BitmapBytes = 60 : f32 => bitmap_bytes,
    /// `f32`. `min(w, h) / max(w, h)` ∈ `(0, 1]`. Square = `1.0`,
    /// extreme strip → 0. Bounded and smooth — well-conditioned for
    /// MLPs and tree models alike.
    AspectMinOverMax = 61 : f32 => aspect_min_over_max,
    /// `f32`. `|ln(w / h)|` ∈ `[0, ∞)`. Square = `0`, larger =
    /// more extreme. Symmetric (no sign ambiguity between landscape
    /// and portrait) and unbounded above — sensitive to very
    /// extreme ratios where `AspectMinOverMax` saturates.
    LogAspectAbs = 62 : f32 => log_aspect_abs,
    /// `f32`. Fraction of padding pixels needed to round the
    /// image up to a complete 8×8 grid. `0.0` for images whose
    /// dimensions are both multiples of 8; positive otherwise. The
    /// codec's per-block overhead (DCT, prediction, signaling)
    /// applies to padded blocks too — this captures that "wasted"
    /// fraction. Hits JPEG 8×8 DCT and WebP/AVIF 8×8 partitions.
    BlockMisalignment8 = 63 : f32 => block_misalignment_8,
    /// `f32`. Same as `BlockMisalignment8` but for 32×32 blocks.
    /// Hits JXL DCT32 and AV1 32×32 partitions.
    BlockMisalignment32 = 65 : f32 => block_misalignment_32,
    /// `u32`. Number of color channels in the source descriptor:
    /// 1 (grayscale), 3 (RGB), or 4 (RGBA). Pure descriptor lookup.
    ChannelCount = 67 : u32 => channel_count,

    // ---------------- AqMap percentiles -----------------------------
    // [`Self::AqMapMean`] / [`Self::AqMapStd`] already buffer
    // per-block AC energy in `block_acs`. These percentiles sort the
    // same buffer and read at the listed quantile, in log10 space
    // matching `aq_map_mean`. Free incremental compute. Issue #42 →
    // distributional features analysis 2026-04-30.
    /// `f32`. log10 AC energy at the 50th percentile (median) over
    /// 8×8 blocks. See [`Self::AqMapMean`] for the underlying
    /// per-block accumulator.
    #[cfg(feature = "experimental")]
    AqMapP50 = 68 : f32 => aq_map_p50,
    /// `f32`. log10 AC energy p75. Detail-floor signal —
    /// distinguishes "uniformly busy" (high p75) from "mostly flat
    /// with a few hard blocks" (low p75) where mean alone collapses
    /// both into one number.
    #[cfg(feature = "experimental")]
    AqMapP75 = 69 : f32 => aq_map_p75,
    /// `f32`. log10 AC energy p90.
    #[cfg(feature = "experimental")]
    AqMapP90 = 70 : f32 => aq_map_p90,
    /// `f32`. log10 AC energy p95.
    #[cfg(feature = "experimental")]
    AqMapP95 = 71 : f32 => aq_map_p95,
    /// `f32`. log10 AC energy p99 — peak-block detail. Picks up
    /// localized hard blocks that drive the worst-case JPEG cost.
    #[cfg(feature = "experimental")]
    AqMapP99 = 72 : f32 => aq_map_p99,

    // ---------------- NoiseFloor percentiles ------------------------
    // [`Self::NoiseFloorY`] / [`Self::NoiseFloorUV`] already sort
    // `block_low_*` buffers and read at p10. Same scaling
    // (`sqrt(arr[idx]/15) / 32`, clamped to [0,1]) at additional
    // quantiles surfaces noise *texture* (uniform vs streaky) to
    // the picker. Zero added compute beyond per-quantile array
    // index reads.
    /// `f32`. Noise floor (Y) at the 25th percentile of per-block
    /// low-AC energy. Same `[0, 1]` scaling as
    /// [`Self::NoiseFloorY`] (which is p10).
    #[cfg(feature = "experimental")]
    NoiseFloorYP25 = 73 : f32 => noise_floor_y_p25,
    /// `f32`. Noise floor (Y) at the 50th percentile (median).
    #[cfg(feature = "experimental")]
    NoiseFloorYP50 = 74 : f32 => noise_floor_y_p50,
    /// `f32`. Noise floor (Y) at the 75th percentile.
    #[cfg(feature = "experimental")]
    NoiseFloorYP75 = 75 : f32 => noise_floor_y_p75,
    /// `f32`. Noise floor (Y) at the 90th percentile — top-quartile
    /// busy blocks. Higher = noisier in the dirtiest regions of the
    /// image.
    #[cfg(feature = "experimental")]
    NoiseFloorYP90 = 76 : f32 => noise_floor_y_p90,
    /// `f32`. Noise floor (UV) at p25 — `max(noise_floor_cb_p25,
    /// noise_floor_cr_p25)`, same shape as [`Self::NoiseFloorUV`].
    #[cfg(feature = "experimental")]
    NoiseFloorUvP25 = 77 : f32 => noise_floor_uv_p25,
    /// `f32`. Noise floor (UV) at p50.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP50 = 78 : f32 => noise_floor_uv_p50,
    /// `f32`. Noise floor (UV) at p75.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP75 = 79 : f32 => noise_floor_uv_p75,
    /// `f32`. Noise floor (UV) at p90.
    #[cfg(feature = "experimental")]
    NoiseFloorUvP90 = 80 : f32 => noise_floor_uv_p90,

    // ---------------- LaplacianVariance percentiles -----------------
    // Tier 1 piggyback. Driven by a 256-bin histogram over `|∇²L|`
    // (clamped to `[0, 255]`) accumulated alongside the existing
    // Laplacian variance pass — see
    // `src/tier1.rs::accumulate_laplacian_simd`. The histogram lives
    // in `PixelStats` and adds 8 scalar adds per SIMD iter (lane
    // scatter); no per-pixel branching, no allocation.
    /// `f32`. `|∇²L|` at the 50th percentile (median) over interior
    /// pixels. Range `[0, 255]` (saturated at the ceiling).
    #[cfg(feature = "experimental")]
    LaplacianVarianceP50 = 81 : f32 => laplacian_variance_p50,
    /// `f32`. `|∇²L|` p75. Sharpness floor — distinguishes "single
    /// sharp edge in a smooth image" (low p75 + high p99) from
    /// "uniformly textured" (high p75) where the variance alone
    /// can't tell them apart.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP75 = 82 : f32 => laplacian_variance_p75,
    /// `f32`. `|∇²L|` p90.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP90 = 83 : f32 => laplacian_variance_p90,
    /// `f32`. `|∇²L|` p99 — peak-edge magnitude. Picks up the
    /// rare-but-loud sharpness events that drive the codec's
    /// per-block trellis decision.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP99 = 84 : f32 => laplacian_variance_p99,
    /// `f32`. Highest histogram bin (`0`–`255`) with at least one
    /// observation. Saturates at `255` when any pixel hit the
    /// histogram-clamp ceiling — flag value for "extreme edge
    /// present somewhere in the image". Size-dependent (larger
    /// images roll more chances at extremes); see issue #42 size
    /// features for cross-term cushioning.
    #[cfg(feature = "experimental")]
    LaplacianVariancePeak = 85 : f32 => laplacian_variance_peak,

    // ---------------- QuantSurvival percentiles ---------------------
    // Per-block buffer of `quant_survival(...)` values is collected
    // alongside the existing streaming mean (one f32 per block, ≤4096
    // entries on a 4 MP image, ≤16 KB transient per channel). Sorted
    // once at end of pass, indexed at fixed quantiles. p10 ⇒
    // worst-block survival ⇒ trellis ROI proxy; p75 ⇒ best-block
    // survival ⇒ compression ceiling.
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

    // ---------------- Dimension log/derivative variants -------------
    // Companions to PixelCount/LogPixels (#42). Different bases give
    // the network different numerical handles for the same underlying
    // resolution signal — and let ablation tell us whether
    // PixelCount's dominance is real signal or memorization. The
    // block-padded variants reflect the *encoded* surface area: for
    // a 257×257 image the codec actually encodes 264×264 (block-8
    // grid), 272×272 (block-16), etc. These correlate with bytes
    // spent more directly than the visible pixel count does.

    // ---------------- Low-tail percentile companions ---------------
    // Adds P1/P5/P10 to LaplacianVariance / AqMap / NoiseFloorY (and
    // P1/P5 to QuantSurvivalY which already ships P10). Same
    // histograms / sorted block buffers already computed for the
    // upper-tail percentiles — the additions are pure index lookups
    // (or one extra prefix-sum threshold for Laplacian's histogram).
    // Provisional: ablation on subset100 will prune the redundant
    // ones via Spearman ρ > 0.95 with adjacent percentiles.
    /// `f32`. `|∇²L|` p1 — fraction of *very* flat blocks. Different
    /// signal from `Uniformity` (mean): captures presence of large
    /// smooth regions even when the rest of the image is busy.
    /// Codec-relevant — flat blocks compress to ~few bits via DC +
    /// early-zero AC.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP1 = 105 : f32 => laplacian_variance_p1,
    /// `f32`. `|∇²L|` p5.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP5 = 106 : f32 => laplacian_variance_p5,
    /// `f32`. `|∇²L|` p10.
    #[cfg(feature = "experimental")]
    LaplacianVarianceP10 = 107 : f32 => laplacian_variance_p10,
    /// `f32`. AC-energy log10 p1. Encoder-floor signal — blocks the
    /// encoder treats as easiest. May saturate at the AQ floor on
    /// most images (hypothesis to validate via ablation).
    #[cfg(feature = "experimental")]
    AqMapP1 = 108 : f32 => aq_map_p1,
    /// `f32`. AC-energy log10 p5.
    #[cfg(feature = "experimental")]
    AqMapP5 = 109 : f32 => aq_map_p5,
    /// `f32`. AC-energy log10 p10.
    #[cfg(feature = "experimental")]
    AqMapP10 = 110 : f32 => aq_map_p10,
    /// `f32`. Y noise floor at p1.
    #[cfg(feature = "experimental")]
    NoiseFloorYP1 = 111 : f32 => noise_floor_y_p1,
    /// `f32`. Y noise floor at p5.
    #[cfg(feature = "experimental")]
    NoiseFloorYP5 = 112 : f32 => noise_floor_y_p5,
    /// `f32`. Y noise floor at p10. Same scaling as parent
    /// [`Self::NoiseFloorY`] (which is also p10) — reserved as a
    /// distinct slot in case the parent's scaling drifts.
    #[cfg(feature = "experimental")]
    NoiseFloorYP10 = 113 : f32 => noise_floor_y_p10,
    /// `f32`. Y quant-survival at p1 — the worst-block tail. Sharper
    /// trellis-ROI signal than [`Self::QuantSurvivalYP10`].
    #[cfg(feature = "experimental")]
    QuantSurvivalYP1 = 114 : f32 => quant_survival_y_p1,
    /// `f32`. Y quant-survival at p5.
    #[cfg(feature = "experimental")]
    QuantSurvivalYP5 = 115 : f32 => quant_survival_y_p5,

    // ---------------- Distribution-shape (kurtosis) signals ----------
    // L4-moment-based features. Capture "how peaked" a distribution
    // is — a single scalar that distinguishes Gaussian-ish (~ 3) from
    // heavy-tailed (≫ 3, e.g. screen content with bimodal AC) without
    // needing a sort. Differentiable; smoother than percentile features
    // for MLP picker training.
    /// `f32`. Excess kurtosis of `|∇²L|` over the 256-bin Laplacian
    /// histogram already computed for `LaplacianVariance*P*` features.
    /// Computed post-hoc from the histogram with no per-pixel cost —
    /// `(E[X⁴] / E[X²]²) − 3` where `X = |∇²L|` clamped to `[0, 255]`.
    /// Captures texture-energy distribution shape: ~ 0 for natural
    /// photos (Gaussian-ish gradient distribution); ≫ 0 for screen
    /// content (heavy tail from sharp transitions on flat backgrounds).
    /// Complementary to `LaplacianVariance` (which measures spread, not
    /// shape).
    #[cfg(feature = "experimental")]
    LumaKurtosis = 116 : f32 => luma_kurtosis,

    // ---------------- Smooth replacements for hard thresholds --------
    // Threshold-counting features (var < 25 / range ≤ 4 / etc.) have
    // piecewise-constant gradients that fight MLP training. Smooth
    // counterparts using exp-decay or direct ratios are differentiable
    // everywhere and capture the same "fraction-of-mass" signal.
    /// `f32`. Smooth analog of `GradientFraction`: mean of
    /// `block_low_y_ac / max(block_ac, 16)` over sampled 8×8 DCT blocks.
    /// The hard `GradientFraction` only counts blocks where the ratio
    /// is `≥ 0.9` AND `block_ac > 16`; this smooth version returns the
    /// continuous ratio for every block. Captures "what fraction of
    /// luma AC energy lives in the low-zigzag positions" as a smooth
    /// signal — drives JXL DCT16/DCT32 enable decisions.
    #[cfg(feature = "experimental")]
    GradientFractionSmooth = 120 : f32 => gradient_fraction_smooth,

    // ---------------- Palette: ceiling-log2 of distinct-bin count ----
    /// `u32`. Smallest indexed-palette bits-per-pixel that
    /// representationally contains the source's distinct-colour set,
    /// computed from `ceil(log2(distinct_color_bins))` over the
    /// analyzer's 5-bit-per-channel bin storage. **Codomain:
    /// `[1, 15] ∪ {24}`.** Value 24 is the truecolor saturation
    /// sentinel — see below.
    ///
    /// Bucket meanings:
    ///
    /// - `1`  ⇒ ≤ 2 colours          (PNG-1, true monochrome)
    /// - `2`  ⇒ ≤ 4 colours          (PNG-2)
    /// - `3`  ⇒ ≤ 8 colours          (GIF BPP=3 — PNG rounds to 4)
    /// - `4`  ⇒ ≤ 16 colours         (PNG-4)
    /// - `5..7` ⇒ ≤ 32..128 colours  (GIF BPP, PNG rounds to 8)
    /// - `8`  ⇒ ≤ 256 colours        (PNG-8, GIF, WebP-lossless palette)
    /// - `9..15` ⇒ ≤ 512..32768      (JXL Modular palette transform tiers)
    /// - `24` ⇒ bin array saturated  (truecolor; true 8-bit count is
    ///                                somewhere in `[32 768, 2²⁴]`)
    ///
    /// **Why 24 and not 0 for truecolor.** A trained-MLP picker treats
    /// this as a numeric input; a discontinuous `..., 14, 15, 0` jump
    /// fights the gradient signal. Emitting `24` keeps the scale
    /// monotonic — `..., 14, 15, [9-unit gap], 24` — and the gap
    /// itself is meaningful ("we ran out of measurement resolution;
    /// this is unambiguously truecolor"). 24 is the bit-width the
    /// source would need with no palette compression at all (8 bits
    /// per channel × 3 channels), so it's a defensible upper bound.
    ///
    /// **Bin-storage bias.** The 5-bit-per-channel quantization
    /// (32 768 cells) under-counts the true 8-bit distinct-colour
    /// count when colours collapse into the same 5-bit cell. Bias is
    /// always toward undercounting, so `palette_log2_size` is biased
    /// toward smaller buckets — encoders consuming the value will
    /// pick a slightly smaller indexed BPP than ideal, and an indexed
    /// encoder building its own palette will quickly discover the
    /// extra colours and either expand the palette or fall back to
    /// truecolor. Self-correcting.
    ///
    /// **Cost.** Always full-scan: this feature is in
    /// `PALETTE_FULL_FEATURES`, so requesting it forces the full
    /// per-pixel walk that builds the bin array. The reduction itself
    /// is one `leading_zeros` instruction. Callers who only need the
    /// cheap "fits in 256?" signal should request `PaletteFitsIn256`
    /// instead, which uses the early-exit quick-path scan.
    ///
    /// Drives PNG-indexed bit-width choice (round up to {1, 2, 4, 8}),
    /// GIF LZW initial-code-size (matches BPP exactly), JXL Modular
    /// palette/delta-palette enable decisions. Gated behind
    /// `experimental` for 0.1.0; promote when a consumer wires up.
    ///
    /// Replaces the pre-2026-05-02 `IndexedPaletteWidth` (id 30,
    /// codomain `{0, 2, 4, 8}`) which lacked the 1-BPP case for
    /// binary content and didn't surface JXL's high-cap breakpoints.
    /// That id is reserved retired.
    #[cfg(feature = "experimental")]
    PaletteLog2Size = 121 : u32 => palette_log2_size,
}

/// A scalar feature value — discriminated by the value type, not by
/// the feature it came from. Most callers want [`Self::to_f32`] (the
/// lossless coercion that maps `Bool(false) → 0.0`, `Bool(true) →
/// 1.0`, and `U32(n) → n as f32`).
///
/// `#[non_exhaustive]` so future structured feature types (small
/// histograms, vectors) can land via additional variants without a
/// major bump. Today the variants cover every scalar field on the
/// legacy `AnalyzerOutput`.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FeatureValue {
    F32(f32),
    U32(u32),
    /// `u64` for features whose natural domain exceeds `u32::MAX` —
    /// notably [`AnalysisFeature::BitmapBytes`] which can exceed 4 GB
    /// on 16K+ HDR (RGBAF32) images. Lossless `to_f32` coercion is
    /// only exact for `n ≤ 2⁵³`; values beyond are rounded to the
    /// nearest f64-representable then cast.
    U64(u64),
    Bool(bool),
}

impl FeatureValue {
    /// Type-checked accessor: returns `Some(x)` only if the value is
    /// `F32(_)`. Use when the caller knows the underlying type.
    #[inline]
    pub const fn as_f32(self) -> Option<f32> {
        match self {
            Self::F32(x) => Some(x),
            _ => None,
        }
    }

    /// Type-checked accessor for `U32`.
    #[inline]
    pub const fn as_u32(self) -> Option<u32> {
        match self {
            Self::U32(x) => Some(x),
            _ => None,
        }
    }

    /// Type-checked accessor for `U64`.
    #[inline]
    pub const fn as_u64(self) -> Option<u64> {
        match self {
            Self::U64(x) => Some(x),
            _ => None,
        }
    }

    /// Type-checked accessor for `Bool`.
    #[inline]
    pub const fn as_bool(self) -> Option<bool> {
        match self {
            Self::Bool(x) => Some(x),
            _ => None,
        }
    }

    /// Lossless coercion to `f32`. `Bool(false) → 0.0`, `Bool(true) →
    /// 1.0`, `U32(n) → n as f32` (lossless for `n ≤ 2²⁴`, which
    /// covers every current and foreseeable feature). Convenience for
    /// ML pipelines / threshold comparisons that don't care about the
    /// underlying type.
    #[inline]
    pub fn to_f32(self) -> f32 {
        match self {
            Self::F32(x) => x,
            Self::U32(x) => x as f32,
            // u64 → f32 via f64 to keep precision near 2^53 boundary.
            Self::U64(x) => x as f64 as f32,
            Self::Bool(false) => 0.0,
            Self::Bool(true) => 1.0,
        }
    }
}

impl From<f32> for FeatureValue {
    fn from(x: f32) -> Self {
        Self::F32(x)
    }
}
impl From<u32> for FeatureValue {
    fn from(x: u32) -> Self {
        Self::U32(x)
    }
}
impl From<u64> for FeatureValue {
    fn from(x: u64) -> Self {
        Self::U64(x)
    }
}
impl From<bool> for FeatureValue {
    fn from(x: bool) -> Self {
        Self::Bool(x)
    }
}

/// Opaque set of [`AnalysisFeature`]s, supporting `const fn` set math.
///
/// Backed by a 256-bit (4 × `u64`) presence bitmap indexed by the
/// feature's `u16` discriminant. The size is an internal detail —
/// future versions may grow it transparently. Public callers only see
/// the set ops and [`Self::contains`].
///
/// Deliberately no `all()` constructor: callers must enumerate the
/// features they actually need. "Compute everything" is rarely the
/// right choice and disables the runtime dispatch optimisation that
/// skips entire passes when none of their outputs were requested.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FeatureSet {
    bits: [u64; 4],
}

impl FeatureSet {
    /// Empty set.
    pub const fn new() -> Self {
        Self { bits: [0; 4] }
    }

    /// Singleton set containing exactly one feature.
    pub const fn just(f: AnalysisFeature) -> Self {
        Self::new().with(f)
    }

    /// Return a copy with `f` inserted. `const fn` so it composes in
    /// `const`-context preset definitions.
    pub const fn with(mut self, f: AnalysisFeature) -> Self {
        let id = f as u16 as usize;
        self.bits[id >> 6] |= 1u64 << (id & 63);
        self
    }

    /// Return a copy with `f` removed.
    pub const fn without(mut self, f: AnalysisFeature) -> Self {
        let id = f as u16 as usize;
        self.bits[id >> 6] &= !(1u64 << (id & 63));
        self
    }

    /// Set union (`A ∪ B`).
    pub const fn union(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 {
            self.bits[i] |= other.bits[i];
            i += 1;
        }
        self
    }

    /// Set intersection (`A ∩ B`).
    pub const fn intersect(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 {
            self.bits[i] &= other.bits[i];
            i += 1;
        }
        self
    }

    /// Set difference (`A − B`).
    pub const fn difference(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < 4 {
            self.bits[i] &= !other.bits[i];
            i += 1;
        }
        self
    }

    /// `true` iff `self` and `other` share any feature.
    pub const fn intersects(self, other: Self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] & other.bits[i] != 0 {
                return true;
            }
            i += 1;
        }
        false
    }

    /// `true` iff `self` contains every feature in `other`.
    pub const fn contains_all(self, other: Self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] & other.bits[i] != other.bits[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// `true` iff the set contains `f`.
    pub const fn contains(self, f: AnalysisFeature) -> bool {
        let id = f as u16 as usize;
        (self.bits[id >> 6] >> (id & 63)) & 1 != 0
    }

    /// `true` iff the set has no features.
    pub const fn is_empty(self) -> bool {
        let mut i = 0;
        while i < 4 {
            if self.bits[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Number of features in the set.
    pub const fn len(self) -> u32 {
        let mut total = 0;
        let mut i = 0;
        while i < 4 {
            total += self.bits[i].count_ones();
            i += 1;
        }
        total
    }

    /// Iterate the contained features in ascending [`AnalysisFeature::id`]
    /// order.
    ///
    /// Skips ids that don't correspond to a variant in this build —
    /// retired discriminants are never re-mapped, and cfg-disabled
    /// experimental variants legitimately return `None` from
    /// [`AnalysisFeature::from_u16`]. This means the iterator length
    /// is **at most** [`Self::len`]; on a cross-feature build it
    /// equals it. Sidecars / Python fitters / harness code can use
    /// this to walk every supported feature without hand-listing the
    /// enum.
    pub fn iter(self) -> FeatureSetIter {
        FeatureSetIter {
            bits: self.bits,
            next_id: 0,
        }
    }
    // `SUPPORTED` is generated by the [`features_table!`] invocation
    // above so it stays in lockstep with the enum.

    /// Reduced 8-feature schema used by the zenjpeg Zq picker (v1.1).
    ///
    /// An ablation study against the picker's RD oracle (PR
    /// `imazen/zenjpeg#129`) showed that 11 of the 19 features the
    /// picker originally consumed are droppable without measurable
    /// quality loss. The 8 retained features all live in Tier 1 +
    /// Tier 3, so requesting this set lets the analyzer's existing
    /// tier-gating skip three full passes:
    ///
    /// - **Tier 2** (per-axis Cb/Cr horiz/vert/peak sharpness — the
    ///   most expensive zenanalyze pass, full-image 3-row sliding
    ///   window).
    /// - **Palette** (full-image distinct-color scan).
    /// - **Alpha** (stride-sampled alpha pass).
    ///
    /// Composition (all 8 are non-experimental; const-buildable on
    /// default features):
    ///
    /// **Tier 1 — basic stripe-sampled signals**
    /// - [`AnalysisFeature::Variance`]
    /// - [`AnalysisFeature::EdgeDensity`]
    /// - [`AnalysisFeature::Uniformity`]
    /// - [`AnalysisFeature::ChromaComplexity`]
    ///
    /// **Tier 1 — chroma sharpness (mean `|∇Cb|` / `|∇Cr|`)**
    /// - [`AnalysisFeature::CbSharpness`]
    /// - [`AnalysisFeature::CrSharpness`]
    ///
    /// **Tier 3 — DCT energy + entropy (sampled 8×8 blocks)**
    /// - [`AnalysisFeature::HighFreqEnergyRatio`]
    /// - [`AnalysisFeature::LumaHistogramEntropy`]
    ///
    /// Pin to a specific patch when compiling-in fitted models —
    /// numeric thresholds drift across 0.1.x per the crate-level
    /// threshold contract. Re-validate downstream when bumping.
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

/// Iterator over the [`AnalysisFeature`]s in a [`FeatureSet`], in
/// ascending [`AnalysisFeature::id`] order. Built by
/// [`FeatureSet::iter`].
pub struct FeatureSetIter {
    bits: [u64; 4],
    next_id: u16,
}

impl Iterator for FeatureSetIter {
    type Item = AnalysisFeature;
    fn next(&mut self) -> Option<Self::Item> {
        // Walk bits in id-ascending order, skipping ids the bitset
        // doesn't have set, AND ids whose variants aren't in this
        // build (cfg-gated experimentals on default features).
        loop {
            let id = self.next_id as usize;
            if id >= 256 {
                return None;
            }
            let bit = (self.bits[id >> 6] >> (id & 63)) & 1;
            self.next_id += 1;
            if bit == 1
                && let Some(f) = AnalysisFeature::from_u16(id as u16)
            {
                return Some(f);
            }
        }
    }
}

impl IntoIterator for FeatureSet {
    type Item = AnalysisFeature;
    type IntoIter = FeatureSetIter;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self::new()
    }
}

// --- Internal: tier-membership constants used by the runtime const-bool
// dispatch in `analyze_with`. Not public — codecs name their features
// individually, not via tier bundles, so the tier split stays an
// implementation detail and can be refactored without breaking callers.

/// Palette features whose computation requires an **exact** distinct-
/// colour count. Asking for any of these forces the full-scan path
/// (every pixel walked). Const-block style so individual `.with()`
/// calls can be cfg-gated when their variant is experimental.
pub(crate) const PALETTE_FULL_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::DistinctColorBins);
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::PaletteDensity);
        // GrayscaleScore is computed on the same full-scan walk that
        // builds the palette histogram. It needs 100 % coverage —
        // stripe-sampling at ~5 % budget would let a single colour
        // pixel slip past the gate ~95 % of the time and produce a
        // false-positive grayscale classification.
        s = s.with(AnalysisFeature::GrayscaleScore);
        // PaletteLog2Size's full codomain (1..15 ∪ 24) requires an
        // exact distinct count — the quick-path scan saturates at 8
        // (early-exit at 257 bins) and can't surface JXL palette
        // breakpoints at 9..15 or the truecolor saturation sentinel
        // at 24. Including it here forces the full-scan path so
        // callers get the resolution the feature promises without
        // having to manually co-request `DistinctColorBins`.
        s = s.with(AnalysisFeature::PaletteLog2Size);
    }
    s
};

/// Palette features that only need the running count, with an early-
/// exit at 256. On photographic content these typically resolve in
/// ~10 rows; on indexed content they walk to the end. Both members
/// are experimental in 0.1.0; this set is empty when the feature is
/// off, which makes the quick-path dispatch never fire.
#[allow(unused_mut, unused_assignments)]
pub(crate) const PALETTE_QUICK_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::PaletteFitsIn256);
    }
    s
};

/// Union: any palette feature triggers the palette pass. The quick
/// path runs when the request is exclusively in `PALETTE_QUICK_FEATURES`;
/// any overlap with `PALETTE_FULL_FEATURES` forces the full path
/// (which produces both signal classes).
pub(crate) const PALETTE_FEATURES: FeatureSet = PALETTE_FULL_FEATURES.union(PALETTE_QUICK_FEATURES);

/// Tier 1 "extras" — the optional accumulators that elevate the
/// SIMD kernel from `Minimal` to `Full`. When the requested
/// `FeatureSet` doesn't intersect this set, `accumulate_row_simd`
/// is dispatched as `<FULL = false>` and skips the per-chunk
/// luma_sum / Hasler-Süsstrunk M3 (rg/yb) / skin-tone / edge-slope
/// accumulators — and `extract_tier1_into` skips the separate
/// Laplacian SIMD row pass entirely. Drops ~10 lane-wise f32x8
/// accumulators on AVX2, freeing register pressure on the Tier 1
/// hot path.
///
/// Driven by zenjpeg's actual `ADAPTIVE_FEATURES` query — neither
/// `Variance`, `Colourfulness`, `LaplacianVariance`,
/// `SkinToneFraction`, nor `EdgeSlopeStdev` is in that set, so
/// every zenjpeg analyze call lands in the `Minimal` bucket.
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

/// Subset of [`TIER1_EXTRAS_FEATURES`] gated by the
/// `accumulate_row_simd` `FULL` const-bool: luma stats (Variance) +
/// Hasler M3 (Colourfulness) + edge-slope batching
/// (EdgeSlopeStdev) + the separate Laplacian SIMD pass
/// (LaplacianVariance). `SkinToneFraction` is peeled off into
/// [`TIER1_SKIN_FEATURES`] so the two halves share register
/// pressure on AVX2 only when both are requested.
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

/// Subset of [`TIER1_EXTRAS_FEATURES`] gated by the
/// `accumulate_row_simd` `SKIN` const-bool: BT.601 chroma matrix
/// (2 fma chains) + 6 Chai-Ngan threshold compares + 5 mask
/// AND-chain + masked counter. Independent of `FULL` — a caller
/// that only wants `SkinToneFraction` dispatches with
/// `<*, false, true>` and skips luma stats / Hasler M3 entirely.
#[allow(unused_mut, unused_assignments)]
pub(crate) const TIER1_SKIN_FEATURES: FeatureSet = {
    let mut s = FeatureSet::new();
    #[cfg(feature = "experimental")]
    {
        s = s.with(AnalysisFeature::SkinToneFraction);
    }
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
        // Percentile variants (#42 / #49). Same per-block buffers
        // as the means; dispatch must drag Tier 3 in when only a
        // percentile is requested, otherwise the per-feature
        // sample-count floor in `tier3.rs` never fires and the
        // result-map gets the default 0.0.
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
    }
    s
};

/// Subset of [`TIER3_FEATURES`] whose computation requires the full
/// per-block DCT pass (`tier3::dct_stats`). Excludes
/// [`AnalysisFeature::LumaHistogramEntropy`] and
/// [`AnalysisFeature::LineArtScore`] which both come from the cheap
/// luma-histogram pass alone.
///
/// The dispatcher uses this to set the `DCT` const-bool gate on
/// `populate_tier3<const DCT: bool>` — when false, the
/// ~0.97 ms-per-Mpx DCT pass is skipped entirely, leaving callers who
/// asked for just histogram entropy / line-art on the cheap path.
/// Compile-time const-folding ensures the unused branch contributes
/// zero code in the monomorphized variant.
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
        // Percentile variants (#49) — same per-block source data
        // as the means; gate the DCT pass on them too.
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
    }
    s
};

pub(crate) const ALPHA_FEATURES: FeatureSet = FeatureSet::new()
    .with(AnalysisFeature::AlphaPresent)
    .with(AnalysisFeature::AlphaUsedFraction)
    .with(AnalysisFeature::AlphaBimodalScore);

/// Source-direct depth tier. Reads source samples without going
/// through `RowConverter` so HDR / wide-gamut / high-bit-depth signals
/// survive the analysis. All members are experimental in 0.1.0; the
/// set is empty when the cargo feature is off, which makes the depth-
/// tier dispatch never fire in default builds.
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
// DERIVED_FEATURES is empty after the composite-likelihood deletion
// (#6, #59). Kept for now so the dispatch wiring compiles; eligible
// for full removal once the derived-likelihood call site in
// `tier3::compute_derived_likelihoods` is collapsed away.
pub(crate) const DERIVED_FEATURES: FeatureSet = FeatureSet::new();

// --- Derived-feature dependency closures ----------------------------
//
// After the composite-likelihood deletion these collapse to their
// underlying tier supersets. Kept as named aliases so the dispatch
// axes don't need a second sweep.

pub(crate) const T3_NEEDED_BY: FeatureSet = TIER3_FEATURES;
pub(crate) const PAL_NEEDED_BY: FeatureSet = PALETTE_FEATURES;

// `RawAnalysis` and `into_results` are generated by the
// `features_table!` invocation at the top of this file.

// ---------------------- AnalysisQuery ----------------------------

/// A request to the analyzer: what features to compute.
///
/// Opaque — the only public knob is the [`FeatureSet`]. Tier sampling
/// budgets (`pixel_budget`, `hf_max_blocks`) are **invariants of the
/// crate**, not per-call knobs: the convergence-trained defaults are
/// the only values that ship, and shipped oracle / threshold tables
/// are calibrated against them. Refining them is a release-level
/// decision, not a caller decision.
///
/// Use [`Self::new`] to build. There is intentionally no `Default`,
/// no fluent budget setters, and no `full()` shortcut — callers must
/// enumerate the features they need, and "compute everything" must be
/// expressed by enumerating every feature explicitly. The runtime
/// dispatcher uses the requested set to skip entire passes whose
/// outputs aren't needed.
#[derive(Clone, Debug)]
pub struct AnalysisQuery {
    features: FeatureSet,
}

impl AnalysisQuery {
    /// Build a query for `features`.
    ///
    /// `features` typically comes from `const`-context union of each
    /// codec's preset, e.g. `JPEG_FEATURES.union(WEBP_FEATURES)`.
    pub const fn new(features: FeatureSet) -> Self {
        Self { features }
    }

    /// The feature set this query asks for.
    pub const fn features(&self) -> FeatureSet {
        self.features
    }
}

/// Crate-internal: the canonical sampling budgets.
///
/// These are **not** exposed on [`AnalysisQuery`]. They're invariants
/// the crate maintains so shipped oracle / threshold tables stay
/// calibrated. Updating them is a release-level decision (with a
/// retrain or recalibration of every downstream consumer); callers
/// don't get to override per-call.
///
/// Tests / oracle re-extraction that genuinely need different
/// sampling go through the `__internal_with_overrides` ctor —
/// double-underscored, `#[doc(hidden)]`, **not** a stable API.
pub(crate) const DEFAULT_PIXEL_BUDGET: usize = 500_000;
pub(crate) const DEFAULT_HF_MAX_BLOCKS: usize = 1024;

#[doc(hidden)]
impl AnalysisQuery {
    /// **Unstable. Tests / oracle re-extraction only.** Lets the
    /// caller override the otherwise-invariant sampling budgets. No
    /// stability guarantee — may be removed or renamed at any time.
    /// Production code that calls this is wrong; if you think you
    /// need it, file an issue describing the use case.
    pub fn __internal_with_overrides(
        features: FeatureSet,
        pixel_budget: usize,
        hf_max_blocks: usize,
    ) -> InternalQuery {
        InternalQuery {
            features,
            pixel_budget,
            hf_max_blocks,
        }
    }
}

/// **Unstable.** Backdoor for oracle / convergence-study work that
/// needs non-default sampling budgets. Constructed only via
/// [`AnalysisQuery::__internal_with_overrides`]; consumed by an
/// internal entry point that public callers don't see.
#[doc(hidden)]
pub struct InternalQuery {
    pub(crate) features: FeatureSet,
    pub(crate) pixel_budget: usize,
    pub(crate) hf_max_blocks: usize,
}

// ---------------------- AnalysisResults ---------------------------

/// Image geometry — width / height / megapixels / aspect ratio.
/// Returned alongside [`AnalysisResults`] regardless of which features
/// were requested (it's a property of the input, not the analysis).
#[derive(Copy, Clone, Debug)]
pub struct ImageGeometry {
    width: u32,
    height: u32,
}

impl ImageGeometry {
    /// Construct from raw width/height. Crate-internal entry; public
    /// callers receive `ImageGeometry` from [`AnalysisResults::geometry`]
    /// rather than building it themselves.
    pub(crate) const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Image width in pixels.
    pub const fn width(self) -> u32 {
        self.width
    }
    /// Image height in pixels.
    pub const fn height(self) -> u32 {
        self.height
    }
    /// `width × height` as `u64` to avoid overflow on giant images.
    pub const fn pixels(self) -> u64 {
        self.width as u64 * self.height as u64
    }
    /// `pixels / 1e6` as f32. Lossy for >2²⁴ MP (>16 trillion pixels).
    pub fn megapixels(self) -> f32 {
        self.pixels() as f32 / 1_000_000.0
    }
    /// `width / max(1, height)`. Returns 0 if height is 0.
    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

/// Opaque container for one analysis pass's outputs.
///
/// Query individual features with [`Self::get`], passing the
/// [`AnalysisFeature`] you want. Returns `None` if:
/// - The feature wasn't in the requested set, or
/// - The feature is retired ([`AnalysisFeature::is_active`] = false), or
/// - The pass that produces it failed (e.g. image too small).
///
/// Storage is opaque — internally a sparse `Vec<(AnalysisFeature,
/// FeatureValue)>` sized to the requested set, **no over-allocation**.
/// Future versions may switch the backing layout (packed parallel
/// arrays, sorted slice, etc.) without breaking the public API.
pub struct AnalysisResults {
    requested: FeatureSet,
    geometry: ImageGeometry,
    source_descriptor: zenpixels::PixelDescriptor,
    /// One entry per *populated* feature, sorted ascending by
    /// `AnalysisFeature::id`. Vec capacity is preallocated to
    /// `requested.len()` so the analyzer's `set` calls never realloc.
    /// Lookup by linear scan over up to ~30 entries (one cache line).
    values: Vec<(AnalysisFeature, FeatureValue)>,
}

impl AnalysisResults {
    /// Build an empty `AnalysisResults` for the given `requested` set,
    /// image geometry, and source descriptor. Crate-internal — public
    /// callers receive results from `analyze_features`, never
    /// construct them directly.
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

    /// Crate-internal: write a value.
    ///
    /// `debug_assert!`s that `f` is in the requested set — analyzer
    /// code is wrong if it tries to store unrequested features (the
    /// dispatcher should have skipped that work entirely). In release
    /// the call silently no-ops to avoid storing data the caller
    /// didn't ask for.
    ///
    /// If `f` is already present (e.g. a tier wrote it twice), the
    /// later value wins; the entry stays sorted by id.
    pub(crate) fn set(&mut self, f: AnalysisFeature, v: impl Into<FeatureValue>) {
        debug_assert!(
            self.requested.contains(f),
            "analyzer wrote unrequested feature {:?} (id={}) — dispatcher gating is broken",
            f,
            f.id()
        );
        if !self.requested.contains(f) {
            return;
        }
        let v = v.into();
        // Skip the `f32::NAN` sentinel. Tiers emit NaN to signal
        // "this feature is unreliable for this image" (e.g. percentile
        // features below their per-feature minimum-sample-count floor;
        // see issue #49). Keeping NaN out of the public results forces
        // the codec's OOD-fallback path on those inputs rather than
        // silently propagating noisy point estimates as if they were
        // real signal. Picker-side `first_out_of_distribution(...)`
        // would also catch the NaN, but skipping at the writer level
        // is the cleaner safety contract: the absence is the signal.
        if let FeatureValue::F32(x) = v
            && x.is_nan()
        {
            return;
        }
        // Insertion-sort by id. Up to ~30 entries; binary search +
        // memmove would be the same cost as a linear walk here.
        let mut i = 0;
        while i < self.values.len() {
            match self.values[i].0.id().cmp(&f.id()) {
                core::cmp::Ordering::Less => i += 1,
                core::cmp::Ordering::Equal => {
                    self.values[i].1 = v;
                    return;
                }
                core::cmp::Ordering::Greater => break,
            }
        }
        self.values.insert(i, (f, v));
    }

    /// The feature set the caller asked for. Useful for asserting
    /// "did the analyzer compute what I asked for" or for joining
    /// results from multiple analyses.
    pub const fn requested(&self) -> FeatureSet {
        self.requested
    }

    /// Image geometry for the analysed input.
    pub const fn geometry(&self) -> ImageGeometry {
        self.geometry
    }

    /// The source [`zenpixels::PixelDescriptor`] the analyzer ingested.
    ///
    /// Codecs (zenavif / zenjxl / zenwebp / zenjpeg) read this to
    /// drive bit-depth, primaries, transfer-function, color-model,
    /// alpha-mode, and signal-range encode decisions — the analyzer
    /// doesn't surface every descriptor field as a separate feature
    /// because the descriptor is already a small `Copy` value with a
    /// stable shape; callers can pull whatever they need with one
    /// accessor instead of querying ten boolean features.
    ///
    /// For "is the image actually using the gamut?" / "does the bit
    /// depth carry information?" decisions, pair this with the
    /// `tier_depth` features ([`AnalysisFeature::EffectiveBitDepth`],
    /// [`AnalysisFeature::HdrPresent`],
    /// [`AnalysisFeature::WideGamutFraction`], …) which read the
    /// pixel data, not just the metadata.
    #[inline]
    pub const fn source_descriptor(&self) -> zenpixels::PixelDescriptor {
        self.source_descriptor
    }

    /// Look up one feature's value. `None` if not requested, retired,
    /// or computation failed.
    #[inline]
    pub fn get(&self, f: AnalysisFeature) -> Option<FeatureValue> {
        // Linear scan — Vec is sorted by id but at ≤ 30 entries the
        // branch-predictor handles a linear walk faster than a
        // binary search.
        self.values.iter().find(|(k, _)| *k == f).map(|(_, v)| *v)
    }

    /// Convenience: get and coerce to `f32`. Returns `None` if the
    /// feature isn't present, or `Some(0.0)` for `Bool(false)`,
    /// `Some(1.0)` for `Bool(true)`, `Some(n as f32)` for `U32(n)`.
    /// Callers that want strict typed access should use
    /// [`Self::get`] + [`FeatureValue::as_f32`] (or `as_u32` /
    /// `as_bool`).
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
        for (feature, v) in &self.values {
            d.field(feature.name(), v);
        }
        d.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_set_basic_ops() {
        let a = FeatureSet::just(AnalysisFeature::Variance);
        let b = FeatureSet::just(AnalysisFeature::EdgeDensity);
        let u = a.union(b);
        assert!(u.contains(AnalysisFeature::Variance));
        assert!(u.contains(AnalysisFeature::EdgeDensity));
        // Pick a third unrelated variant to spot-check non-membership.
        // ChromaComplexity is unflagged so the test compiles regardless
        // of the experimental feature gate.
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

    /// Discriminant ids that have been retired and reserved — `from_u16`
    /// must return `None` for these so callers don't get a value at a
    /// slot whose meaning has changed.
    const RESERVED_RETIRED_IDS: &[u16] = &[
        // id 11 was `DistinctColorBinsChao1`, removed pre-0.1.0.
        11,
        // ids 27, 28, 29, 45 were `TextLikelihood` /
        // `ScreenContentLikelihood` / `NaturalLikelihood` /
        // `LineArtScore` — composites flag deleted; stable ids reserved.
        27, 28, 29, 45,
        // ids 64, 66 were `BlockMisalignment16` / `BlockMisalignment64`
        // — collapsed into `_8` / `_32` anchors at Spearman 0.96 / 0.998
        // on zenavif's non-power-of-2 corpus.
        64, 66,
        // ids 94..104 were the 11 mathematical transforms of
        // `feat_pixel_count` (log2/log10/log_padded/sqrt/etc.) — pure
        // collinearity per #59.
        94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
        // ids 117, 118, 119 were `ChromaKurtosis` /
        // `UniformitySmooth` / `FlatColorSmooth` — Tier-0 redundant
        // with existing features on ≥3/4 codecs in the 2026-05-01
        // cross-codec ablation. Stable ids reserved.
        117, 118, 119,
        // id 30 was `IndexedPaletteWidth` (codomain {0, 2, 4, 8}).
        // Replaced 2026-05-02 by `PaletteLog2Size` (id 121, codomain
        // {0, 1..15}) which adds the 1-BPP case for binary content
        // and surfaces JXL palette breakpoints at 9..15.
        30,
    ];

    #[test]
    fn discriminants_round_trip() {
        // Sequential 0..32 with retired-id and cfg-disabled holes.
        // Active variants round-trip through id() / from_u16; retired
        // and cfg-disabled ids return None. Retired ids must always
        // return None (asserted explicitly); cfg-disabled ones either
        // return Some (when the cargo feature is on) or None (when off)
        // — both legal.
        for id in 0..64u16 {
            if RESERVED_RETIRED_IDS.contains(&id) {
                assert!(
                    AnalysisFeature::from_u16(id).is_none(),
                    "id {id} is retired but from_u16 returned Some — \
                     don't recycle retired discriminants"
                );
                continue;
            }
            // cfg-disabled experimental ids legitimately return None.
            if let Some(f) = AnalysisFeature::from_u16(id) {
                assert_eq!(f.id(), id);
            }
        }
        // First unused id past the palette redefinition (id 121
        // is now `PaletteLog2Size`).
        assert!(AnalysisFeature::from_u16(122).is_none());
        assert!(AnalysisFeature::from_u16(255).is_none());
    }

    #[test]
    fn analysis_query_constructor_only() {
        // `AnalysisQuery` exposes only the features it was constructed
        // with; sampling budgets are crate invariants, not knobs.
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
        assert!(q.features().contains(AnalysisFeature::Variance));
        assert!(!q.features().contains(AnalysisFeature::EdgeDensity));
    }

    #[test]
    fn raw_analysis_round_trip() {
        // Sanity: filling RawAnalysis and translating drops the
        // unrequested fields and keeps the requested ones, with the
        // right typing per feature.
        let raw = RawAnalysis {
            variance: 12.5,
            distinct_color_bins: 4096,
            alpha_present: true,
            edge_density: 0.5, // not requested below
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
        assert_eq!(
            r.get(AnalysisFeature::DistinctColorBins),
            Some(FeatureValue::U32(4096))
        );
        assert_eq!(
            r.get(AnalysisFeature::AlphaPresent),
            Some(FeatureValue::Bool(true))
        );
        // Was zeroed by Default and never requested → None.
        assert_eq!(r.get(AnalysisFeature::EdgeDensity), None);
    }

    #[test]
    fn supported_set_covers_all_active_variants() {
        // Every active variant must round-trip through SUPPORTED.
        // Skip both retired ids (RESERVED_RETIRED_IDS) and ids whose
        // variants are cfg-gated out of this build — `from_u16` already
        // returns `None` for both kinds of holes, so a single
        // `if let Some(f) = …` walk handles them uniformly.
        let mut active = 0u32;
        // Iterate past the dimension features (max id 67 today). Bump
        // the upper bound when new ids land — `assert_eq!` below
        // catches drift between SUPPORTED.len() and this loop's
        // walked range.
        for id in 0..128u16 {
            if RESERVED_RETIRED_IDS.contains(&id) {
                continue;
            }
            let Some(f) = AnalysisFeature::from_u16(id) else {
                // cfg-disabled experimental variant — legitimately
                // absent in this build.
                continue;
            };
            assert!(
                FeatureSet::SUPPORTED.contains(f),
                "{:?} (id={}) is missing from FeatureSet::SUPPORTED",
                f,
                id
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
        // Set values for everything in the requested set, deliberately
        // out of id-order to exercise the insertion-sort.
        r.set(AnalysisFeature::AlphaPresent, true);
        r.set(AnalysisFeature::Variance, 42.0_f32);
        r.set(AnalysisFeature::DistinctColorBins, 1234_u32);
        // Overwrite (later wins)
        r.set(AnalysisFeature::Variance, 43.0_f32);

        assert_eq!(
            r.get(AnalysisFeature::Variance),
            Some(FeatureValue::F32(43.0))
        );
        assert_eq!(
            r.get(AnalysisFeature::DistinctColorBins),
            Some(FeatureValue::U32(1234))
        );
        assert_eq!(
            r.get(AnalysisFeature::AlphaPresent),
            Some(FeatureValue::Bool(true))
        );
        // Requested but never set (no analysis ran) → None.
        // ChromaComplexity is unflagged so the assertion compiles
        // regardless of the experimental cargo feature.
        assert_eq!(r.get(AnalysisFeature::ChromaComplexity), None);
        // Not requested at all → None
        assert_eq!(r.get(AnalysisFeature::EdgeDensity), None);

        // get_f32 coercion
        assert_eq!(r.get_f32(AnalysisFeature::Variance), Some(43.0));
        assert_eq!(r.get_f32(AnalysisFeature::DistinctColorBins), Some(1234.0));
        assert_eq!(r.get_f32(AnalysisFeature::AlphaPresent), Some(1.0));

        // Geometry round-trips
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
        // EdgeDensity wasn't requested — debug_assert! must fire.
        r.set(AnalysisFeature::EdgeDensity, 0.0_f32);
    }

    #[test]
    fn feature_set_iter_visits_every_set_member_in_id_order() {
        let s = FeatureSet::just(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity);
        let collected: Vec<_> = s.iter().collect();
        // Iter is ascending by id; Variance=0, EdgeDensity=1, DistinctColorBins=10.
        assert_eq!(
            collected,
            vec![
                AnalysisFeature::Variance,
                AnalysisFeature::EdgeDensity,
                AnalysisFeature::DistinctColorBins,
            ]
        );

        // SUPPORTED iterates with len == count.
        let n = FeatureSet::SUPPORTED.iter().count();
        assert_eq!(n as u32, FeatureSet::SUPPORTED.len());

        // Empty set yields nothing.
        assert_eq!(FeatureSet::new().iter().count(), 0);

        // IntoIterator trait impl mirrors iter().
        let s2 = FeatureSet::just(AnalysisFeature::Variance);
        let v: Vec<_> = s2.into_iter().collect();
        assert_eq!(v, vec![AnalysisFeature::Variance]);
    }

    #[test]
    fn analysis_feature_id_is_public_and_stable() {
        // Wire-format guarantee: ids are sequential u16, immutable
        // once shipped. Public callers (Python fitters, sidecars)
        // must be able to read this.
        assert_eq!(AnalysisFeature::Variance.id(), 0);
        assert_eq!(AnalysisFeature::EdgeDensity.id(), 1);
        assert_eq!(AnalysisFeature::DistinctColorBins.id(), 10);
        assert_eq!(AnalysisFeature::AlphaPresent.id(), 24);

        // from_u16 is the public inverse.
        assert_eq!(
            AnalysisFeature::from_u16(0),
            Some(AnalysisFeature::Variance)
        );
        assert_eq!(AnalysisFeature::from_u16(11), None); // retired
        assert_eq!(AnalysisFeature::from_u16(9999), None);
    }

    #[test]
    fn tier_bundles_are_disjoint() {
        // Each feature should belong to at most one tier bundle.
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
