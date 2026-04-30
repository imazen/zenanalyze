//! Adaptive dispatch tree for [`crate::analyze_with_dispatch_plan`].
//!
//! Stages 0 + 1.5 of the issue-#53 plan: cheap per-image pre-tier1
//! decisions and post-tier1 query narrowing based on `is_grayscale` /
//! `uniformity`. Layered on top of the existing tier code — no
//! changes to tier internals.
//!
//! ## Stages
//!
//! - **Stage 0** (free, runs in <1 µs): inspects [`feature::AnalysisQuery`]
//!   and image dimensions before any scan. Empty-feature requests
//!   short-circuit; ≤ 64K-pixel images get the budget bumped to
//!   exhaustive; ≥ 8 MP images get [`DispatchPlan::flag_extended_pass`]
//!   set for a future Stage 2 follow-up (see issue #53 / #46 / corpus
//!   #47).
//! - **Stage 1**: runs Tier 1 + the strict-grayscale classifier +
//!   alpha + dimension + depth features. Same code path as today,
//!   just structured to expose the partial [`RawAnalysis`] before
//!   the rest of the dispatcher decides what to do with it.
//! - **Stage 1.5**: narrows the remaining query by subtracting
//!   [`feature::CHROMA_DROP_FEATURES`] (when `is_grayscale = true`)
//!   and [`feature::SATURATING_DROP_FEATURES`] (when `uniformity >
//!   0.95`), then runs Tier 2 / Tier 3 / palette gated on the
//!   narrowed set. Dropped features come back as `None` from
//!   [`feature::AnalysisResults::get`] — the layered defense in
//!   [`feature::RawAnalysis::into_results`] guarantees no caller
//!   ever sees garbage for a dropped feature.
//!
//! Stage 2 (extended-budget retry on the budget-sensitive features
//! when `flag_extended_pass = true`) is **deferred** — its threshold
//! values need empirical validation against the imazen/zenanalyze#47
//! corpus sweep before they can ship.

use zenpixels::PixelSlice;

#[cfg(feature = "experimental")]
use crate::feature::AnalysisFeature;
use crate::feature::{
    self, AnalysisQuery, AnalysisResults, FeatureSet, ImageGeometry, RawAnalysis,
};
use crate::row_stream::RowStream;
use crate::{AnalyzeError, alpha, dimensions, grayscale, palette, tier1, tier2_chroma, tier3};

/// Optional caller hints for [`crate::analyze_with_dispatch_plan`].
///
/// Advisory only — every field is optional and the analyzer falls
/// back to safe defaults when [`crate::analyze_with_dispatch_plan`]
/// receives `None`. Stages 0 and 1.5 do not act on any hint yet; the
/// fields exist so future stages (corpus#47-validated Stage 2,
/// content-hash result caching) can consume them without a
/// signature change to the public entry. See issue
/// imazen/zenanalyze#53.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
pub struct DispatchHints {
    /// Picker's target perceptual quality, in zq units. High targets
    /// (lossless / near-lossless) are budget-sensitive on
    /// [`AnalysisFeature::PatchFractionFast`]; low targets aren't.
    /// Reserved for Stage 2 once the corpus sweep validates the
    /// thresholds.
    pub target_zq: Option<f32>,
    /// Caller-supplied content hash for cross-call result caching.
    /// Reserved for a future cache layer; not consumed today.
    pub content_hash: Option<u64>,
}

impl DispatchHints {
    /// Empty hints — every field `None`. Equivalent to passing
    /// `None` to [`crate::analyze_with_dispatch_plan`].
    pub const fn empty() -> Self {
        Self {
            target_zq: None,
            content_hash: None,
        }
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
/// pre-0.1.0 calibration corpus. The flag is **not acted on today** —
/// Stage 2 is deferred until imazen/zenanalyze#47's corpus sweep
/// validates the threshold values.
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
    /// HF block cap for Tier 3's DCT pass. Stage 0 doesn't tune
    /// this knob today (the canonical default is fine for tiny
    /// images — they hit the per-image block count well before the
    /// cap matters).
    hf_max_blocks: usize,
    /// Set when `pixel_count >= LARGE_THRESHOLD`. Stage 2 will
    /// consume this on the follow-up PR after corpus#47 sweep
    /// validates the budget-sensitive feature thresholds; for now
    /// this PR only stores the flag and never acts on it.
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
    // `flag_extended_pass` is recorded but not consumed in this PR
    // (Stage 2 deferred to corpus#47-gated follow-up). Bind to `_`
    // so future readers see the deliberate no-op.
    let _ = plan.flag_extended_pass;

    // Re-derive the same per-feature gates `analyze_features` uses,
    // but split them so we can apply the Stage 1.5 narrowing
    // *between* Tier 1 and Tier 2/3.
    let pal_initial = requested.intersects(feature::PAL_NEEDED_BY);
    let alpha = requested.intersects(feature::ALPHA_FEATURES);
    let run_depth = requested.intersects(feature::DEPTH_FEATURES);
    // Always run the strict-grayscale classifier — Stage 1.5 needs
    // its output to decide whether to drop the chroma tiers, and
    // the cost is sub-microsecond on coloured content (early-exit
    // at the first non-gray row).
    let run_strict_gray_for_plan = true;

    let alpha_stats = if alpha {
        alpha::scan_alpha(&slice, plan.pixel_budget)
    } else {
        Default::default()
    };

    #[cfg(feature = "experimental")]
    let depth_stats = if run_depth {
        crate::tier_depth::scan_depth(&slice, plan.pixel_budget)
    } else {
        Default::default()
    };
    let _ = run_depth;

    let mut stream = RowStream::new(slice).map_err(AnalyzeError::Convert)?;

    // Palette path needs the *initial* query to decide between
    // full-scan / quick-scan / no-scan — Stage 1.5 doesn't drop any
    // palette features (palette outputs are content-class signals
    // for the codec, not chroma signals).
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

    // ----- Stage 1: Tier 1 + strict-grayscale gate ------------------
    if width >= 2 && height >= 2 {
        let t1_dispatch = tier1::Tier1Dispatch {
            wants_laplacian: tier1_wants_laplacian(requested),
            wants_full_kernel: requested.intersects(feature::TIER1_FULL_FEATURES),
            wants_skin: requested.intersects(feature::TIER1_SKIN_FEATURES),
        };
        tier1::extract_tier1_into_dispatch(&mut raw, &mut stream, plan.pixel_budget, t1_dispatch);

        if run_strict_gray_for_plan {
            raw.is_grayscale = grayscale::scan_strict_grayscale(&mut stream);
        }
    }

    // ----- Stage 1.5: narrow remaining query ------------------------
    let mut narrowed = requested;
    if raw.is_grayscale {
        narrowed = narrowed.difference(feature::CHROMA_DROP_FEATURES);
    }
    // `uniformity` is a Tier 1 output written above. The threshold is
    // the issue-#53 spec value (0.95 — flat enough that Laplacian
    // percentiles will pin to ~0 and `patch_fraction_fast` won't
    // vary). Stage 2 will revisit this with corpus-validated
    // numbers; the hard-coded 0.95 is fine for stages 0 + 1.5.
    if raw.uniformity > 0.95 {
        narrowed = narrowed.difference(feature::SATURATING_DROP_FEATURES);
    }

    // ----- Stage 2: run Tier 2 + Tier 3 with the narrowed query -----
    let t2 = narrowed.intersects(feature::TIER2_FEATURES);
    let t3 = narrowed.intersects(feature::T3_NEEDED_BY);
    let dct = narrowed.intersects(feature::DCT_NEEDED_BY);

    if width >= 2 && height >= 2 {
        if t2 && width >= 3 && height >= 3 {
            tier2_chroma::populate_tier2(&mut raw, &mut stream, plan.pixel_budget);
        }
        if t3 && width >= 8 && height >= 8 {
            tier3::populate_tier3(&mut raw, &mut stream, plan.hf_max_blocks, dct);
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
            raw.indexed_palette_width = palette_stats.indexed_width;
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

    #[cfg(feature = "experimental")]
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

/// Mirror of the gate in `analyze_features`. Crate-internal helper
/// so we don't drift from the canonical dispatch decision.
#[inline]
fn tier1_wants_laplacian(features: FeatureSet) -> bool {
    #[cfg(feature = "experimental")]
    {
        features.contains(AnalysisFeature::LaplacianVariance)
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = features;
        false
    }
}

/// Mirror of the gate in `analyze_features` for `palette_wants_grayscale`.
#[inline]
fn palette_wants_grayscale(features: FeatureSet) -> bool {
    #[cfg(feature = "experimental")]
    {
        features.contains(AnalysisFeature::GrayscaleScore)
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = features;
        false
    }
}
