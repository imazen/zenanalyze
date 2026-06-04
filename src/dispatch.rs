//! Adaptive dispatch tree for [`crate::analyze_with_dispatch_plan`].
//!
//! Stages 0 + 1.5 + 2 of the issue-#53 plan. Layered on top of the
//! existing tier code — no changes to tier internals. The dispatch
//! tree inlines the same pass calls [`crate::analyze_specialized_raw`]
//! makes, in the same order against the same random-access
//! [`RowStream`], so every feature it *keeps* is bit-identical to
//! [`crate::analyze_features`]. The only divergences are deliberate:
//! features it *drops* (Stage 1.5) come back as `None`, and the three
//! budget-sensitive Tier 3 features it *re-samples* (Stage 2) come back
//! more accurate than the default-budget value on ≥ 8 MP images.
//!
//! ## Stages
//!
//! - **Stage 0** (free, runs in <1 µs): inspects [`feature::AnalysisQuery`]
//!   and image dimensions before any scan. Empty-feature requests
//!   short-circuit; ≤ 64K-pixel images get the budget bumped to
//!   exhaustive; ≥ 8 MP images get [`DispatchPlan::flag_extended_pass`]
//!   set for the Stage 2 retry.
//! - **Stage 1**: Tier 1 + the strict-grayscale classifier + alpha +
//!   dimension + depth features run as today — same code path, same
//!   `RowStream`, structured so the partial [`feature::RawAnalysis`]
//!   (`is_grayscale`, `uniformity`) is available before the rest of
//!   the dispatcher decides what to do with it.
//! - **Stage 1.5** (content-class gating): narrows the remaining query
//!   by subtracting [`feature::CHROMA_DROP_FEATURES`] (when
//!   `is_grayscale = true`) and [`feature::SATURATING_DROP_FEATURES`]
//!   (when `uniformity > 0.95`), then runs Tier 2 / Tier 3 / palette
//!   gated on the narrowed set. Dropped features come back as `None`
//!   from [`feature::AnalysisResults::get`] — the layered defense in
//!   [`feature::RawAnalysis::into_results`] guarantees no caller ever
//!   sees garbage for a dropped feature.
//! - **Stage 2** (extended-budget retry): when `flag_extended_pass`
//!   was set (≥ 8 MP) and the narrowed query still asks for a
//!   budget-sensitive Tier 3 feature, re-run the Tier 3 DCT pass at a
//!   larger `hf_max_blocks` cap and overwrite only the
//!   [`feature::EXTENDED_PASS_FEATURES`] columns
//!   (`patch_fraction_fast`, `aq_map_p99`, `noise_floor_y_p90`) with
//!   the finer-sampled values. Skipped when the image is already
//!   sampled densely enough (default cap covers every block) — see
//!   [`DispatchPlan::extended_hf_max_blocks`].
//!
//! ## Parity contract
//!
//! For every feature *not* dropped by Stage 1.5 and *not* one of the
//! three [`feature::EXTENDED_PASS_FEATURES`] on a ≥ 8 MP image, the
//! value equals [`crate::analyze_features`] bit-for-bit (the passes,
//! budgets, and gate booleans match exactly, and `RowStream` is
//! random-access so reordering tier1/grayscale relative to tier2/3
//! changes nothing). The Stage 2 features diverge *by design* — a
//! strictly finer block sample is the whole point of the retry.

use zenpixels::PixelSlice;

use crate::feature::{
    self, AnalysisQuery, AnalysisResults, FeatureSet, ImageGeometry, RawAnalysis,
};
use crate::row_stream::RowStream;
use crate::{AnalyzeError, alpha, dimensions, grayscale, palette, tier1, tier2_chroma, tier3};

/// Optional caller hints for [`crate::analyze_with_dispatch_plan`].
///
/// Empty today — Stages 0 / 1.5 / 2 derive every decision from image
/// dimensions and Tier 1 output, so no hint is consumed. The
/// `#[non_exhaustive]` attribute is the seat: future stages add fields
/// additively under 0.2.x without a public signature change to the
/// entry function. Callers that hand a `&DispatchHints` today keep
/// compiling when fields land, since they construct via
/// [`DispatchHints::empty`] / [`DispatchHints::default`] rather than
/// struct-literal syntax. See issue imazen/zenanalyze#53.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
pub struct DispatchHints {}

impl DispatchHints {
    /// Construct an empty hint bag. Equivalent to passing `None` to
    /// [`crate::analyze_with_dispatch_plan`].
    pub const fn empty() -> Self {
        Self {}
    }
}

/// Pixel-count threshold below which the dispatch plan promotes the
/// scan to exhaustive (Tier 1 / 3 sample every pixel / block). At
/// 256 × 256 = 65 536 pixels the per-call fixed overhead dominates
/// the per-pixel work, and the
/// [`feature::DEFAULT_PIXEL_BUDGET`]-sampled stripe walk produces
/// noisier features than scanning the whole image. Sub-microsecond
/// either way.
pub(crate) const MIN_EXHAUSTIVE_THRESHOLD: u64 = 64_000;

/// Pixel-count threshold above which the dispatch plan flags the
/// image for a Stage 2 extended-budget retry. 8 MP ≈ a 24 MP
/// camera frame downsampled by 3, the size at which budget-sensitive
/// features (`patch_fraction_fast`, `aq_map_p99`, `noise_floor_y_p90`)
/// start showing the most variance vs an exhaustive scan on the
/// pre-0.1.0 calibration corpus. Stage 2 acts on the flag set here.
pub(crate) const LARGE_THRESHOLD: u64 = 8_000_000;

/// Internal record of the Stage 0 decisions for an image. Held by
/// [`run`] across stages; not exposed publicly.
#[derive(Copy, Clone, Debug)]
struct DispatchPlan {
    /// Sampling budget for Tier 1 / Tier 2 stripe scans and the
    /// alpha pass. Either [`feature::DEFAULT_PIXEL_BUDGET`] or the
    /// total pixel count when the image is below
    /// [`MIN_EXHAUSTIVE_THRESHOLD`].
    pixel_budget: usize,
    /// HF block cap for Tier 3's DCT pass on the first pass. Stage 0
    /// uses the canonical default; Stage 2 re-runs with
    /// [`Self::extended_hf_max_blocks`].
    hf_max_blocks: usize,
    /// Set when `pixel_count >= LARGE_THRESHOLD`. Stage 2 consumes
    /// this to decide whether to re-sample the budget-sensitive Tier 3
    /// features at a finer block stride.
    flag_extended_pass: bool,
}

impl DispatchPlan {
    /// Compute the Stage 0 plan from image dimensions alone.
    fn from_geometry(geometry: ImageGeometry) -> Self {
        let pixels = geometry.pixels();
        let pixel_budget = if pixels <= MIN_EXHAUSTIVE_THRESHOLD {
            // Exhaustive override: cap at usize so we never panic
            // even on a hypothetical 32-bit target with a degenerate
            // image. usize::MAX would also work but pixel_count is a
            // sharper "match every pixel" intent.
            pixels.try_into().unwrap_or(usize::MAX)
        } else {
            feature::DEFAULT_PIXEL_BUDGET
        };
        let flag_extended_pass = pixels >= LARGE_THRESHOLD;
        Self {
            pixel_budget,
            hf_max_blocks: feature::DEFAULT_HF_MAX_BLOCKS,
            flag_extended_pass,
        }
    }

    /// Block cap for the Stage 2 extended Tier 3 DCT pass: 2× the
    /// default cap, so the DCT walk samples roughly twice as many
    /// 8×8 blocks (finer stride). Sub-quadratic — the per-block work
    /// is fixed, only the count doubles.
    fn extended_hf_max_blocks(&self) -> usize {
        self.hf_max_blocks.saturating_mul(2)
    }
}

/// Public-entry implementation. Returns the populated
/// [`AnalysisResults`].
pub(crate) fn run(
    slice: PixelSlice<'_>,
    query: &AnalysisQuery,
    _hints: Option<&DispatchHints>,
) -> Result<AnalysisResults, AnalyzeError> {
    let requested = query.features();
    let width = slice.width();
    let height = slice.rows();
    let geometry = ImageGeometry::new(width, height);
    let source_descriptor = slice.descriptor();

    // Stage 0: empty request → no work.
    if requested.is_empty() {
        return Ok(RawAnalysis::default().into_results(requested, geometry, source_descriptor));
    }

    // Stage 0: budget + extended-pass flag from dimensions alone.
    let plan = DispatchPlan::from_geometry(geometry);

    // Re-derive the per-feature gates `analyze_features` uses, but
    // split them so we can apply the Stage 1.5 narrowing *between*
    // Tier 1 and Tier 2/3. Palette + alpha + depth are decided from
    // the *initial* query (Stage 1.5 never drops those — they're
    // content-class signals for the codec, not chroma signals).
    let pal_initial = requested.intersects(feature::PAL_NEEDED_BY);
    let alpha = requested.intersects(feature::ALPHA_FEATURES);
    let run_depth = requested.intersects(feature::DEPTH_FEATURES);
    // Always run the strict-grayscale classifier — Stage 1.5 needs its
    // output to decide whether to drop the chroma tiers, and the cost
    // is sub-microsecond on coloured content (early-exit at the first
    // non-gray row). This is the one place the dispatch plan does more
    // work than `analyze_features` would for a query that didn't ask
    // for `IsGrayscale`; the result is only surfaced when requested.
    let want_is_grayscale = requested.contains(feature::AnalysisFeature::IsGrayscale);

    // ----- Alpha + depth: source-direct passes, order-independent ---
    let alpha_stats = if alpha {
        alpha::scan_alpha(&slice, plan.pixel_budget)
    } else {
        Default::default()
    };

    #[cfg(feature = "hdr")]
    let depth_stats = if run_depth {
        crate::tier_depth::scan_depth(&slice, plan.pixel_budget)
    } else {
        Default::default()
    };
    let _ = run_depth;

    let mut stream = RowStream::new(slice).map_err(AnalyzeError::Convert)?;

    // Palette routing uses the *initial* query, mirroring
    // `analyze_specialized_raw` exactly.
    let palette_full_required = requested.intersects(feature::PALETTE_FULL_FEATURES);
    let palette_wants_grayscale = palette_wants_grayscale(requested);
    let palette_stats = if pal_initial {
        if palette_full_required {
            palette::scan_palette(&mut stream, palette_wants_grayscale)
        } else {
            palette::scan_palette_quick(&mut stream)
        }
    } else {
        Default::default()
    };

    let mut raw = RawAnalysis::default();

    // Always-on dimension features (pure descriptor math).
    dimensions::populate_dimensions(&mut raw, width, height, source_descriptor);

    // ----- Stage 1: Tier 1 + strict-grayscale classifier ------------
    if width >= 2 && height >= 2 {
        let t1_dispatch = tier1::Tier1Dispatch {
            wants_laplacian: tier1_wants_laplacian(requested),
            wants_full_kernel: requested.intersects(feature::TIER1_FULL_FEATURES),
            wants_skin: requested.intersects(feature::TIER1_SKIN_FEATURES),
        };
        tier1::extract_tier1_into_dispatch(&mut raw, &mut stream, plan.pixel_budget, t1_dispatch);

        // Always classify grayscale for the Stage 1.5 gate. `raw`'s
        // `is_grayscale` field is only emitted later if the caller
        // requested `IsGrayscale` (it's part of `requested`), so this
        // never leaks an unrequested value into the result map.
        raw.is_grayscale = grayscale::scan_strict_grayscale(&mut stream);
    }
    let _ = want_is_grayscale;

    // ----- Stage 1.5: narrow the remaining query --------------------
    let mut narrowed = requested;

    // GRAYSCALE gate — VALIDATED SAFE against the imazen-26 corpus
    // (benchmarks/dispatch_gate_validation_2026-06-04.{tsv,meta}). On
    // strict R==G==B grayscale input every CHROMA_DROP feature is
    // *bit-exactly* its default (chroma is definitionally zero), and
    // the strict-equality classifier never fired on an image carrying
    // real chroma (0 misfires across the sample, cross-checked against
    // a ground-truth max-channel-spread measure). Safe to drop.
    if raw.is_grayscale {
        narrowed = narrowed.difference(feature::CHROMA_DROP_FEATURES);
    }

    // UNIFORMITY gate — DISABLED PENDING DATA. The issue-#53 spec
    // assumed `uniformity > 0.95` ⇒ the Laplacian percentiles pin to
    // ~0 and `patch_fraction_fast` won't vary. That holds for
    // *photographic* flat content but is VIOLATED on text / line-art /
    // document / diagram screen content: a mostly-white page with
    // sparse-but-maximally-sharp black text reports `uniformity > 0.95`
    // yet has `laplacian_variance_peak = 255`, `laplacian_variance_p99`
    // up to 38, `patch_fraction_fast ≈ 0.99`, `aq_map_p99` up to 5.9.
    // The imazen-26 validation (benchmarks/dispatch_gate_validation_*)
    // measured 34 meaningful-drop events across 10 gated images — every
    // dropped saturating feature carried real signal on the screen-
    // content class the picker most needs to distinguish. Dropping them
    // is information loss, so the gate stays OFF until a content-aware
    // threshold (e.g. uniformity-AND-low-edge-peak) is calibrated.
    // `SATURATING_DROP_FEATURES` is retained for that follow-up.
    const ENABLE_UNIFORMITY_GATE: bool = false;
    #[allow(clippy::overly_complex_bool_expr)]
    if ENABLE_UNIFORMITY_GATE && raw.uniformity > 0.95 {
        narrowed = narrowed.difference(feature::SATURATING_DROP_FEATURES);
    }

    // ----- Tier 2 / Tier 3 on the narrowed query --------------------
    let t2 = narrowed.intersects(feature::TIER2_FEATURES);
    let t3 = narrowed.intersects(feature::T3_NEEDED_BY);
    let dct = narrowed.intersects(feature::DCT_NEEDED_BY);

    if width >= 2 && height >= 2 {
        if t2 && width >= 3 && height >= 3 {
            tier2_chroma::populate_tier2(&mut raw, &mut stream, plan.pixel_budget);
        }
        if t3 && width >= 8 && height >= 8 {
            tier3::populate_tier3(&mut raw, &mut stream, plan.hf_max_blocks, dct);

            // ----- Stage 2: extended-budget retry -------------------
            // On a ≥ 8 MP image, the budget-sensitive Tier 3 DCT
            // features are the noisiest vs an exhaustive scan because
            // the default block cap is spread thin. If the narrowed
            // query still wants one of them, re-run the DCT pass at a
            // finer block stride and overwrite *only* those columns.
            //
            // `dct` is already true here (EXTENDED_PASS_FEATURES ⊂
            // DCT_NEEDED_BY, and they all survived Stage 1.5 gating
            // to be in `narrowed`). Skip when the extended cap can't
            // sample more blocks than the first pass already did
            // (`extended_hf_max_blocks <= hf_max_blocks` is impossible
            // here, but the saturating-mul guard keeps it honest).
            if plan.flag_extended_pass
                && narrowed.intersects(feature::EXTENDED_PASS_FEATURES)
                && plan.extended_hf_max_blocks() > plan.hf_max_blocks
            {
                run_extended_pass(&mut raw, &mut stream, &plan);
            }
        }
        // Const-bool derived likelihoods: same maximum-instantiation
        // discipline as the main dispatcher — the layered defense in
        // `compute_derived_likelihoods` refuses to write a likelihood
        // whose deps weren't computed.
        if t3 && pal_initial {
            tier3::compute_derived_likelihoods::<true, true>(&mut raw);
        } else if t3 {
            tier3::compute_derived_likelihoods::<true, false>(&mut raw);
        } else if pal_initial {
            tier3::compute_derived_likelihoods::<false, true>(&mut raw);
        } else {
            tier3::compute_derived_likelihoods::<false, false>(&mut raw);
        }
    }

    // ----- Alpha + palette + depth writeback ------------------------
    if alpha {
        raw.alpha_present = alpha_stats.present;
        raw.alpha_used_fraction = alpha_stats.used_fraction;
        raw.alpha_bimodal_score = alpha_stats.bimodal_score;
    }
    if pal_initial {
        #[cfg(feature = "experimental")]
        {
            raw.palette_log2_size = palette_stats.palette_log2_size;
            raw.palette_fits_in_256 = palette_stats.fits_in_256;
        }
        if palette_full_required {
            raw.distinct_color_bins = palette_stats.distinct;
            #[cfg(feature = "experimental")]
            {
                let pixel_count = (width as f64) * (height as f64);
                let denom = pixel_count.clamp(1.0, 32_768.0);
                raw.palette_density =
                    (raw.distinct_color_bins as f64 / denom).clamp(0.0, 1.0) as f32;
                raw.grayscale_score = if palette_stats.total_pixels > 0 {
                    let gray = palette_stats
                        .total_pixels
                        .saturating_sub(palette_stats.non_grayscale);
                    (gray as f64 / palette_stats.total_pixels as f64) as f32
                } else {
                    0.0
                };
            }
        }
        let _ = palette_stats;
    }

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

    // Emit only the *narrowed* set so dropped features come back as
    // `None` to the caller (the original `requested` would re-include
    // them and yield zeroed garbage from the default-initialized
    // RawAnalysis fields).
    Ok(raw.into_results(narrowed, geometry, source_descriptor))
}

/// Stage 2 extended-budget retry. Re-runs the Tier 3 DCT pass at a
/// finer block stride into a scratch [`RawAnalysis`], then copies the
/// three budget-sensitive [`feature::EXTENDED_PASS_FEATURES`] columns
/// back into `raw`. Every other Tier 3 field stays at its first-pass
/// (default-budget) value, so only the three deliberately-refined
/// features diverge from `analyze_features`.
///
/// `run_dct = true` is hard-coded — the caller only invokes this when
/// the narrowed query still wants a DCT-derived extended feature, so
/// the DCT pass must run. On the non-experimental build the three
/// fields are cfg-gated out and this becomes a no-op recompute we
/// never reach (EXTENDED_PASS_FEATURES is empty), but keep the cfg
/// guards on the field copies for clarity.
fn run_extended_pass(raw: &mut RawAnalysis, stream: &mut RowStream<'_>, plan: &DispatchPlan) {
    let mut extended = RawAnalysis::default();
    tier3::populate_tier3(&mut extended, stream, plan.extended_hf_max_blocks(), true);
    #[cfg(feature = "experimental")]
    {
        raw.patch_fraction_fast = extended.patch_fraction_fast;
        raw.aq_map_p99 = extended.aq_map_p99;
        raw.noise_floor_y_p90 = extended.noise_floor_y_p90;
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = (raw, extended);
    }
}

/// Mirror of the gate in `analyze_features`. Crate-internal helper so
/// we don't drift from the canonical dispatch decision. The percentile
/// variants (#42 / #49) gate the laplacian SIMD pass too — requesting
/// only `LaplacianVarianceP90` must still run the histogram pass, or
/// the dispatch plan would leave those features at zeros where
/// `analyze_features` would populate them.
#[inline]
fn tier1_wants_laplacian(features: FeatureSet) -> bool {
    #[cfg(feature = "experimental")]
    {
        features.intersects(
            FeatureSet::new()
                .with(feature::AnalysisFeature::LaplacianVariance)
                .with(feature::AnalysisFeature::LaplacianVarianceP50)
                .with(feature::AnalysisFeature::LaplacianVarianceP75)
                .with(feature::AnalysisFeature::LaplacianVarianceP90)
                .with(feature::AnalysisFeature::LaplacianVarianceP99)
                .with(feature::AnalysisFeature::LaplacianVariancePeak),
        )
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = features;
        false
    }
}

/// Mirror of the `palette_wants_grayscale` gate in `analyze_features`.
#[inline]
fn palette_wants_grayscale(features: FeatureSet) -> bool {
    #[cfg(feature = "experimental")]
    {
        features.contains(feature::AnalysisFeature::GrayscaleScore)
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = features;
        false
    }
}
