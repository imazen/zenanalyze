//! Quality-aware cross-codec routing — the pieces [`MetaPicker::route`](crate::MetaPicker)
//! composes:
//!
//! - [`QualityTarget`] — what the caller asks for (the **primary** routing axis): a
//!   perceptual target (`zq`/`ssim2`) or mathematically lossless.
//! - [`RouteDecision`] — what the router returns: the chosen family, the lossy-vs-lossless
//!   call, and the full ranked list (for a queued multi-shot family-verify).
//! - [`content_capability`] — the families the IMAGE CONTENT permits, read from a
//!   zenanalyze-api [`Offer`](zenanalyze_api::Offer) (alpha → no JPEG; HDR/deep → JXL/AVIF/PNG
//!   only). Rules, not learned.
//!
//! The route pipeline narrows in this order: caller allowlist ∩ [`content_capability`] ∩
//! [`AllowedFamilies::viable`] (latency) ∩ the gated branch set
//! ([`AllowedFamilies::LOSSY`]/[`LOSSLESS`](AllowedFamilies::LOSSLESS)), then a masked argmin
//! over the surviving families via [`RouteDecision::resolve`].

use crate::{AllowedFamilies, CodecFamily};
use alloc::vec::Vec;

/// The quality a caller targets — the **primary** codec-routing axis (the best family shifts
/// with it). Either a perceptual score (0..100, higher = closer to source) or fully lossless.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum QualityTarget {
    /// Target a zensim score (`zq`, 0..100). The metric the routers were trained on.
    Zq(f32),
    /// Target an SSIMULACRA2 score (0..100). Same scale; the caller must pair it with an
    /// ssim2-trained router (the bake records its metric).
    Ssim2(f32),
    /// Mathematically lossless — the auto-gate is bypassed straight to the lossless router.
    Lossless,
}

impl QualityTarget {
    /// Whether the caller demanded mathematically-lossless output (forces the lossless router,
    /// skipping the auto-gate).
    #[must_use]
    pub const fn is_lossless(self) -> bool {
        matches!(self, Self::Lossless)
    }

    /// The scalar quality value the lossy router / auto-gate model consumes as its
    /// target-quality input. `Lossless` projects to `100.0` (the ceiling).
    #[must_use]
    pub const fn score_input(self) -> f32 {
        match self {
            Self::Zq(x) | Self::Ssim2(x) => x,
            Self::Lossless => 100.0,
        }
    }
}

/// The router's decision: the chosen codec family, whether to encode losslessly (the
/// auto-gate's call), and the full ranked family list (best→worst among the families that
/// survived every mask). `ranked` feeds a queued **meta multi-shot family-verify** — encode
/// the top-K families and keep the smallest at the target — the family-level analogue of the
/// within-codec knob-check.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteDecision {
    family: CodecFamily,
    lossless: bool,
    ranked: Vec<CodecFamily>,
}

impl RouteDecision {
    /// The single best family to encode with.
    #[must_use]
    pub fn family(&self) -> CodecFamily {
        self.family
    }
    /// Whether to encode losslessly (the auto-gate's call). Orthogonal to `family` —
    /// `family == Jxl` with `lossless == true` means JXL-modular.
    #[must_use]
    pub fn lossless(&self) -> bool {
        self.lossless
    }
    /// Families best→worst among the survivors — `ranked[0] == family`. For a queued
    /// multi-shot family-verify, encode the first K and keep the best.
    #[must_use]
    pub fn ranked(&self) -> &[CodecFamily] {
        &self.ranked
    }

    /// Compose a decision from a router model's output: per-family scores (**lower = better**,
    /// matching [`zenpredict`](zenpredict)'s argmin convention) masked to the already-narrowed
    /// `allowed` set (caller allowlist ∩ content-capability ∩ latency-viable ∩ branch set).
    /// Non-finite scores are dropped. `None` if no family survives. Ties break by family
    /// discriminant (stable).
    #[must_use]
    pub fn resolve(
        lossless: bool,
        family_scores: &[f32; CodecFamily::COUNT],
        allowed: AllowedFamilies,
    ) -> Option<Self> {
        let mut surviving: Vec<(CodecFamily, f32)> = CodecFamily::ALL
            .iter()
            .copied()
            .filter(|f| allowed.is_allowed(*f))
            .map(|f| (f, family_scores[f.index()]))
            .filter(|(_, s)| s.is_finite())
            .collect();
        if surviving.is_empty() {
            return None;
        }
        surviving.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.0.index().cmp(&b.0.index()))
        });
        let ranked: Vec<CodecFamily> = surviving.iter().map(|(f, _)| *f).collect();
        Some(Self {
            family: ranked[0],
            lossless,
            ranked,
        })
    }
}

/// The families the IMAGE CONTENT can actually use, read from a zenanalyze-api
/// [`Offer`](zenanalyze_api::Offer) — capability **rules**, not a learned model:
///
/// - straight alpha (`alpha_present`) → drop **JPEG** (no alpha channel).
/// - HDR or > 8-bit (`hdr_present`, or `effective_bit_depth > 8`) → drop **JPEG / WebP / GIF**
///   (8-bit only); keep **JXL / AVIF / PNG** (carry ≥ 10/16-bit).
///
/// Features absent from the offer (e.g. an analyzer built without the `hdr` feature omits the
/// depth columns) are treated as **no restriction**. The result is the content-permitted set;
/// the caller intersects it with its own allowlist and the latency-viable set
/// ([`AllowedFamilies::intersect`] / [`AllowedFamilies::viable`]).
#[cfg(feature = "api")]
#[must_use]
pub fn content_capability(offer: &zenanalyze_api::Offer) -> AllowedFamilies {
    let feat = |name: &str| offer.get(name).map(zenanalyze_api::FeatureResult::float);
    let mut out = AllowedFamilies::all();
    if feat("alpha_present").is_some_and(|v| v > 0.5) {
        out = out.deny(CodecFamily::Jpeg);
    }
    let hdr = feat("hdr_present").is_some_and(|v| v > 0.5);
    let deep = feat("effective_bit_depth").is_some_and(|v| v > 8.5);
    if hdr || deep {
        out = out
            .deny(CodecFamily::Jpeg)
            .deny(CodecFamily::Webp)
            .deny(CodecFamily::Gif);
    }
    out
}

/// Lossy codec preference, best first — the **fixed-prior fallback** (what `family_rule` returns
/// when no zenanalyze features are available; the image-AWARE projection is the intended default —
/// see below). JXL and AVIF are the modern high-efficiency codecs; WebP is the older ubiquitous
/// fallback; JPEG is the compatibility floor (never RD-optimal); GIF last (palette, niche).
///
/// This order is **confirmed by the corrected data — no re-sweep** (correcting an earlier doc that
/// claimed the data couldn't order AVIF vs WebP). With a clean RD measure (cheapest encode reaching
/// the target), AVIF's best swept speed (the data spans **s2–s8**, not just s4; the per-step byte
/// effect is only ~3%, so s2 is near RD-optimal), PAIRED comparison (only images where both reach
/// the target), and size-reweighting (the corpus skews small), the order is **AVIF ≈ JXL ≫ WebP >
/// JPEG** — and strongly content-dependent (large→AVIF 73–85%, tiny→WebP at low-q / JXL at high-q).
/// The earlier "WebP > AVIF" was a coarse-measure + low-q + small-corpus artifact, not reality.
/// An image-AWARE *linear projection* of zenanalyze fit on this corrected data routes one-shot at
/// **3.85%** vs the perfect oracle (any fixed order is 22–30%) — the intended default; this prior
/// is the no-features fallback. Analysis: zenmetrics
/// `scripts/picker/{corrected_ranking,linear_projection_order}.py`.
pub static LOSSY_PREFERENCE: [CodecFamily; 5] = [
    CodecFamily::Jxl,
    CodecFamily::Avif,
    CodecFamily::Webp,
    CodecFamily::Jpeg,
    CodecFamily::Gif,
];
/// Lossless preference, best first — JXL > WebP > PNG > GIF (genuinely ordered: JXL-modular beats
/// WebP-lossless beats PNG; matches both codec reality and the measured win-rate). AVIF/JPEG omit
/// (no true-lossless path here).
pub static LOSSLESS_PREFERENCE: [CodecFamily; 4] = [
    CodecFamily::Jxl,
    CodecFamily::Webp,
    CodecFamily::Png,
    CodecFamily::Gif,
];
/// Target quality at/above which to store losslessly rather than push a lossy encoder to its
/// limit (the sharp RD crossover, ~zq96). The one numeric threshold in the rule.
pub const LOSSLESS_QUALITY: f32 = 96.0;

/// Pick a codec family with an **obviously-correct, format-set-robust** rule — no model, no
/// fitted thresholds beyond the single documented quality crossover. The whole decision is the
/// two preference lists + the capability rules; verify it by reading:
///
/// 1. **lossless?** — yes if the caller asked, or the quality target is near-perfect
///    (≥ [`LOSSLESS_QUALITY`]); pick the matching preference list.
/// 2. **viable** = families the caller allowed, that can represent the image (alpha / HDR /
///    depth — [`content_capability`], format-spec facts), in the list (the list encodes the
///    lossy/lossless mode).
/// 3. return the **highest-preference viable** family.
///
/// Works for ANY subset of `allowed` — one format, several, or none — because it just takes the
/// best available. `None` only when nothing allowed can encode the image (e.g. a lossy target
/// with only PNG allowed). The preference is the data-confirmed prior ([`LOSSY_PREFERENCE`] /
/// [`LOSSLESS_PREFERENCE`]). Content-adaptive reordering is now **viable on the corrected data** —
/// an image-AWARE linear projection of zenanalyze routes one-shot at 3.85% vs the perfect oracle
/// (vs 22–30% for any fixed order); it's being baked as the lossy router. This `family_rule`
/// remains the obviously-correct, no-model, no-features **fallback / audit path**.
#[cfg(feature = "api")]
pub fn family_rule(
    offer: &zenanalyze_api::Offer<'_>,
    target: QualityTarget,
    allowed: AllowedFamilies,
    mode: zenpredict::EncodeMode,
    latency_ms: Option<u32>,
    per_family_est_ms: &[u32; CodecFamily::COUNT],
) -> Option<CodecFamily> {
    let lossless = target.is_lossless() || target.score_input() >= LOSSLESS_QUALITY;
    // viable = allowed ∩ can-represent-the-image (capability) ∩ fits-the-latency-budget.
    // The budget gate is [`AllowedFamilies::viable`]: real-time modes drop any codec whose own
    // per-image encode estimate exceeds `latency_ms` (so a tight RealtimeFastest budget falls
    // through the slow codecs JXL/AVIF to fast WebP/JPEG); queued modes keep every codec.
    let viable =
        allowed
            .intersect(content_capability(offer))
            .viable(mode, latency_ms, per_family_est_ms);
    let order: &[CodecFamily] = if lossless {
        &LOSSLESS_PREFERENCE
    } else {
        &LOSSY_PREFERENCE
    };
    order.iter().copied().find(|&f| viable.is_allowed(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_target_lossless_and_score() {
        assert!(QualityTarget::Lossless.is_lossless());
        assert!(!QualityTarget::Zq(90.0).is_lossless());
        assert_eq!(QualityTarget::Zq(85.0).score_input(), 85.0);
        assert_eq!(QualityTarget::Ssim2(72.0).score_input(), 72.0);
        assert_eq!(QualityTarget::Lossless.score_input(), 100.0);
    }

    // scores: lower = better. jxl best, then avif, webp, jpeg.
    const LOSSY_SCORES: [f32; CodecFamily::COUNT] = [5.0, 3.0, 1.0, 2.0, 9.9, 9.9];

    #[test]
    fn resolve_picks_min_and_ranks() {
        let d = RouteDecision::resolve(false, &LOSSY_SCORES, AllowedFamilies::LOSSY).unwrap();
        assert_eq!(d.family(), CodecFamily::Jxl);
        assert!(!d.lossless());
        assert_eq!(
            d.ranked(),
            &[
                CodecFamily::Jxl,
                CodecFamily::Avif,
                CodecFamily::Webp,
                CodecFamily::Jpeg
            ]
        );
    }

    #[test]
    fn resolve_respects_mask() {
        // deny jxl+avif: best surviving lossy family is webp
        let allowed = AllowedFamilies::LOSSY
            .deny(CodecFamily::Jxl)
            .deny(CodecFamily::Avif);
        let d = RouteDecision::resolve(false, &LOSSY_SCORES, allowed).unwrap();
        assert_eq!(d.family(), CodecFamily::Webp);
        assert_eq!(d.ranked(), &[CodecFamily::Webp, CodecFamily::Jpeg]);
    }

    #[test]
    fn resolve_none_when_empty() {
        assert!(RouteDecision::resolve(false, &LOSSY_SCORES, AllowedFamilies::none()).is_none());
    }

    #[test]
    fn resolve_drops_nonfinite() {
        let mut s = LOSSY_SCORES;
        s[CodecFamily::Jxl.index()] = f32::NAN; // best is NaN -> skip to avif
        let d = RouteDecision::resolve(false, &s, AllowedFamilies::LOSSY).unwrap();
        assert_eq!(d.family(), CodecFamily::Avif);
        assert!(!d.ranked().contains(&CodecFamily::Jxl));
    }

    #[test]
    fn resolve_lossless_branch() {
        // lossless families webp/jxl/png; scores favor jxl
        let s: [f32; CodecFamily::COUNT] = [9.9, 4.0, 1.0, 9.9, 6.0, 9.9];
        let d = RouteDecision::resolve(true, &s, AllowedFamilies::LOSSLESS).unwrap();
        assert_eq!(d.family(), CodecFamily::Jxl);
        assert!(d.lossless());
        assert_eq!(
            d.ranked(),
            &[CodecFamily::Jxl, CodecFamily::Webp, CodecFamily::Png]
        );
    }

    #[cfg(feature = "api")]
    mod capability {
        use super::*;
        use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance};

        fn offer<'a>(cells: &'a [FeatureResult<'a>]) -> Offer<'a> {
            Offer::new(cells, Provenance::new("test"))
        }
        fn cell(name: &str, v: impl Into<zenanalyze_api::Value>) -> FeatureResult<'_> {
            FeatureResult::new(NamedFeature::from_qualified(name), v)
        }

        #[test]
        fn no_signals_allows_all() {
            let cap = content_capability(&offer(&[cell("variance@00000000", 0.5f32)]));
            assert_eq!(cap, AllowedFamilies::all());
        }

        #[test]
        fn alpha_denies_jpeg_only() {
            let cells = [cell("alpha_present@00000000", true)];
            let cap = content_capability(&offer(&cells));
            assert!(!cap.is_allowed(CodecFamily::Jpeg));
            for f in [
                CodecFamily::Webp,
                CodecFamily::Jxl,
                CodecFamily::Avif,
                CodecFamily::Png,
                CodecFamily::Gif,
            ] {
                assert!(cap.is_allowed(f));
            }
        }

        #[test]
        fn hdr_keeps_only_jxl_avif_png() {
            let cells = [cell("hdr_present@00000000", true)];
            let cap = content_capability(&offer(&cells));
            for f in [CodecFamily::Jxl, CodecFamily::Avif, CodecFamily::Png] {
                assert!(cap.is_allowed(f));
            }
            for f in [CodecFamily::Jpeg, CodecFamily::Webp, CodecFamily::Gif] {
                assert!(!cap.is_allowed(f));
            }
        }

        #[test]
        fn deep_bit_depth_keeps_only_jxl_avif_png() {
            let cells = [cell("effective_bit_depth@00000000", 10u32)];
            let cap = content_capability(&offer(&cells));
            assert!(!cap.is_allowed(CodecFamily::Webp));
            assert!(cap.is_allowed(CodecFamily::Jxl));
            assert!(cap.is_allowed(CodecFamily::Png));
        }

        // family_rule: the obviously-correct, any-subset prior + budget gate.
        const NO_BUDGET: [u32; CodecFamily::COUNT] = [0; CodecFamily::COUNT];
        // queued + no latency cap = no budget gating, so `pick` tests the capability + prior only.
        fn pick(o: &Offer, t: QualityTarget, a: AllowedFamilies) -> Option<CodecFamily> {
            family_rule(
                o,
                t,
                a,
                zenpredict::EncodeMode::QueuedBalanced,
                None,
                &NO_BUDGET,
            )
        }

        #[test]
        fn rule_prior_order_and_any_subset() {
            let cells = [cell("variance@00000000", 0.5f32)]; // plain image, no alpha/HDR
            let o = offer(&cells);
            let all = AllowedFamilies::all();
            assert_eq!(
                pick(&o, QualityTarget::Zq(80.0), all),
                Some(CodecFamily::Jxl)
            );
            // JXL denied -> AVIF (the corrected prior: AVIF before WebP)
            assert_eq!(
                pick(&o, QualityTarget::Zq(80.0), all.deny(CodecFamily::Jxl)),
                Some(CodecFamily::Avif)
            );
            // JXL + AVIF denied -> WebP
            assert_eq!(
                pick(
                    &o,
                    QualityTarget::Zq(80.0),
                    all.deny(CodecFamily::Jxl).deny(CodecFamily::Avif)
                ),
                Some(CodecFamily::Webp)
            );
            // any subset works, even GIF-only
            assert_eq!(
                pick(
                    &o,
                    QualityTarget::Zq(80.0),
                    AllowedFamilies::from_allowed([CodecFamily::Gif])
                ),
                Some(CodecFamily::Gif)
            );
        }

        #[test]
        fn rule_capability_mode_and_empty() {
            let alpha = [cell("alpha_present@00000000", true)];
            let oa = offer(&alpha);
            // alpha + only JPEG -> None (JPEG can't represent alpha)
            assert_eq!(
                pick(
                    &oa,
                    QualityTarget::Zq(80.0),
                    AllowedFamilies::from_allowed([CodecFamily::Jpeg])
                ),
                None
            );
            assert_eq!(
                pick(&oa, QualityTarget::Zq(80.0), AllowedFamilies::all()),
                Some(CodecFamily::Jxl)
            );
            let plain = [cell("variance@00000000", 0.5f32)];
            let op = offer(&plain);
            // lossy target but only PNG allowed -> None (PNG has no lossy mode in the list)
            assert_eq!(
                pick(
                    &op,
                    QualityTarget::Zq(80.0),
                    AllowedFamilies::from_allowed([CodecFamily::Png])
                ),
                None
            );
            // nothing allowed -> None
            assert_eq!(
                pick(&op, QualityTarget::Zq(80.0), AllowedFamilies::none()),
                None
            );
        }

        #[test]
        fn rule_lossless_and_quality_threshold() {
            let plain = [cell("variance@00000000", 0.5f32)];
            let o = offer(&plain);
            // explicit Lossless, JXL absent -> WebP before PNG
            assert_eq!(
                pick(
                    &o,
                    QualityTarget::Lossless,
                    AllowedFamilies::from_allowed([CodecFamily::Png, CodecFamily::Webp])
                ),
                Some(CodecFamily::Webp)
            );
            assert_eq!(
                pick(
                    &o,
                    QualityTarget::Lossless,
                    AllowedFamilies::from_allowed([CodecFamily::Png])
                ),
                Some(CodecFamily::Png)
            );
            // near-lossless quality auto-routes to the lossless list -> JXL
            assert_eq!(
                pick(&o, QualityTarget::Zq(98.0), AllowedFamilies::all()),
                Some(CodecFamily::Jxl)
            );
        }

        #[test]
        fn rule_realtime_budget_drops_slow_codecs() {
            use zenpredict::EncodeMode;
            let cells = [cell("variance@00000000", 0.5f32)];
            let o = offer(&cells);
            // per-family encode estimate (ms) for this image; index order jpeg,webp,jxl,avif,png,gif
            let est = [10u32, 20, 300, 400, 8, 30]; // jxl/avif slow, webp/jpeg fast
            // queued ignores latency -> best-RD survivor JXL even though it's slow
            assert_eq!(
                family_rule(
                    &o,
                    QualityTarget::Zq(80.0),
                    AllowedFamilies::all(),
                    EncodeMode::QueuedBalanced,
                    Some(100),
                    &est
                ),
                Some(CodecFamily::Jxl)
            );
            // RealtimeFastest @100ms -> JXL(300)/AVIF(400) too slow -> fall through to WebP
            assert_eq!(
                family_rule(
                    &o,
                    QualityTarget::Zq(80.0),
                    AllowedFamilies::all(),
                    EncodeMode::RealtimeFastest,
                    Some(100),
                    &est
                ),
                Some(CodecFamily::Webp)
            );
        }
    }
}
