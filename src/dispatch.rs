//! Adaptive dispatch tree for [`crate::analyze_with_dispatch_plan`].
//!
//! Stage 0 of the issue-#53 plan: cheap per-image pre-tier1 decisions
//! that adjust **how much budget** to spend, never **which features
//! to skip**. The dispatch tree is purely additive — it can grow the
//! sampling budget on tiny images and (eventually, in Stage 2) on
//! large ones, but it never drops a feature the caller asked for.
//! Returning a populated value for every requested feature keeps the
//! contract intact for consumers that fill fixed-shape input vectors
//! (e.g. zenpicker's MLP) and were trained on the full feature
//! distribution including grayscale / flat content.
//!
//! ## Stages
//!
//! - **Stage 0** (free, runs in <1 µs): inspects [`feature::AnalysisQuery`]
//!   and image dimensions before any scan. Empty-feature requests
//!   short-circuit; ≤ 64K-pixel images get the budget bumped to
//!   exhaustive; ≥ 8 MP images get [`DispatchPlan::flag_extended_pass`]
//!   set for a future Stage 2 follow-up (see issue #53 / #46 / corpus
//!   #47). The flag is **not acted on today** — Stage 2 lands in a
//!   separate PR after corpus #47 validates the threshold values.
//! - Stages 1+ run the same code path as [`crate::analyze_features`],
//!   just with the Stage 0 budget overrides applied.
//!
//! Stages 2 (extended-budget retry on the budget-sensitive features
//! when `flag_extended_pass = true`), 3, and 4 are deferred to
//! follow-up PRs. None of them add or remove features from the output
//! either — the dispatch tree's job is to *spend more compute when
//! it'll help*, not to skip work the caller asked for.

use zenpixels::PixelSlice;

use crate::feature::{self, AnalysisQuery, AnalysisResults, ImageGeometry};
use crate::{AnalyzeError, analyze_specialized_raw};

/// Optional caller hints for [`crate::analyze_with_dispatch_plan`].
///
/// Advisory only — every field is optional and the analyzer falls
/// back to safe defaults when [`crate::analyze_with_dispatch_plan`]
/// receives `None`. Stage 0 does not act on any hint yet; the fields
/// exist so future stages (corpus#47-validated Stage 2,
/// content-hash result caching) can consume them without a
/// signature change to the public entry. See issue
/// imazen/zenanalyze#53.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, Default)]
pub struct DispatchHints {
    /// Picker's target perceptual quality, in zq units. High targets
    /// (lossless / near-lossless) are budget-sensitive on the
    /// patch / percentile features; low targets aren't. Reserved for
    /// Stage 2 once the corpus sweep validates the thresholds.
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
        return Ok(feature::RawAnalysis::default().into_results(
            requested,
            geometry,
            source_descriptor,
        ));
    }

    // Stage 0: budget + extended-pass flag from dimensions alone.
    let plan = DispatchPlan::from_geometry(geometry);
    // `flag_extended_pass` is recorded but not consumed in this PR
    // (Stage 2 deferred to corpus#47-gated follow-up). Bind to `_`
    // so future readers see the deliberate no-op.
    let _ = plan.flag_extended_pass;

    // Mirror the per-feature gates `analyze_features` uses. Same
    // canonical decisions (no narrowing, no synthesised values) — the
    // dispatch plan only changes Tier 1/2/3/alpha sampling budgets.
    let pal = requested.intersects(feature::PAL_NEEDED_BY);
    let t2 = requested.intersects(feature::TIER2_FEATURES);
    let t3 = requested.intersects(feature::T3_NEEDED_BY);
    let alpha = requested.intersects(feature::ALPHA_FEATURES);

    let palette_full_required = requested.intersects(feature::PALETTE_FULL_FEATURES);
    let palette_wants_grayscale = palette_wants_grayscale(requested);
    let run_depth = requested.intersects(feature::DEPTH_FEATURES);
    let run_strict_gray = requested.contains(feature::AnalysisFeature::IsGrayscale);
    let wants_laplacian = tier1_wants_laplacian(requested);
    let wants_full_kernel = requested.intersects(feature::TIER1_FULL_FEATURES);
    let wants_skin = requested.intersects(feature::TIER1_SKIN_FEATURES);
    let dct = requested.intersects(feature::DCT_NEEDED_BY);

    // Mirror the 16-arm const-bool dispatch from `analyze_features`,
    // just with the Stage 0 budget overrides. Keeps the same LLVM
    // monomorphisation table — no extra code-bloat, no divergence in
    // tier internals.
    macro_rules! dispatch {
        ($pal:literal, $t2:literal, $t3:literal, $a:literal) => {{
            let (raw, geometry) = analyze_specialized_raw::<$pal, $t2, $t3, $a>(
                slice,
                plan.pixel_budget,
                plan.hf_max_blocks,
                palette_full_required,
                palette_wants_grayscale,
                wants_laplacian,
                wants_full_kernel,
                wants_skin,
                run_depth,
                dct,
                run_strict_gray,
            )?;
            Ok(raw.into_results(requested, geometry, source_descriptor))
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

/// Mirror of the gate in `analyze_features`. Crate-internal helper
/// so we don't drift from the canonical dispatch decision.
#[inline]
fn tier1_wants_laplacian(features: feature::FeatureSet) -> bool {
    #[cfg(feature = "experimental")]
    {
        features.contains(feature::AnalysisFeature::LaplacianVariance)
    }
    #[cfg(not(feature = "experimental"))]
    {
        let _ = features;
        false
    }
}

/// Mirror of the `palette_wants_grayscale` gate in `analyze_features`.
#[inline]
fn palette_wants_grayscale(features: feature::FeatureSet) -> bool {
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
