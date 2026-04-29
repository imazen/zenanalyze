//! Codec-agnostic rescue plumbing for the two-shot pass framework.
//!
//! Encode + verify themselves stay codec-side (zenpicker has no
//! decoder, no zensim, no awareness of jpegli-q / lambda / chroma
//! scaling). What zenpicker contributes is the decision logic and
//! shared types so every codec wrapper writes the same orchestration
//! instead of inventing its own threshold rule:
//!
//! ```ignore
//! let top2 = picker.argmin_masked_top_k::<2>(&features, &mask, None)?;
//! let first = top2[0].expect("mask non-empty");
//! let bytes_0 = codec::encode(first);
//! let achieved = codec::verify(&bytes_0, &source);
//!
//! match zenpicker::rescue::should_rescue(achieved, target_zq, &policy) {
//!     RescueDecision::Ship => bytes_0,
//!     RescueDecision::Rescue => {
//!         let strategy = policy.strategy;
//!         let bytes_1 = match strategy {
//!             RescueStrategy::SecondBestPick => codec::encode(top2[1].unwrap_or(first)),
//!             RescueStrategy::ConservativeBump => codec::encode_bumped(first, target_zq, achieved),
//!             RescueStrategy::KnownGoodFallback => codec::encode_known_good(target_zq),
//!         };
//!         codec::ship_best_of(bytes_0, bytes_1)
//!     }
//! }
//! ```
//!
//! The two-shot bound is enforced by the codec — this module does
//! not loop. See `zenpicker/SAFETY_PLANE.md` for the full design.

/// Threshold + strategy the codec uses to decide whether to rescue
/// after pass 0.
///
/// `#[non_exhaustive]` so future bake-side metadata (per-cell
/// confidence quantile, `VerifyMode::OnUndershootRisk` hints) can be
/// added additively under 0.1.x.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct RescuePolicy {
    /// Trigger rescue when `achieved_zq < target_zq − rescue_threshold_pp`.
    ///
    /// Default `3.0` is a placeholder pending held-out p99 zensim
    /// shortfall calibration (SAFETY_PLANE.md "Open questions" §1).
    pub rescue_threshold_pp: f32,
    /// Which rescue strategy the codec should follow if pass 0
    /// undershot the target.
    pub strategy: RescueStrategy,
}

impl Default for RescuePolicy {
    fn default() -> Self {
        Self {
            rescue_threshold_pp: 3.0,
            strategy: RescueStrategy::ConservativeBump,
        }
    }
}

/// What the codec should do for the rescue (pass 1) encode.
///
/// `zenpicker` does not implement these — they're a codec-side
/// vocabulary so we report the chosen strategy uniformly in
/// `EncodeMetrics` across codecs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RescueStrategy {
    /// Bump quality (codec-defined: jpegli-q, lambda, distance) and
    /// disable lossy chroma options. Single deterministic transform
    /// of the first-pick `EncoderConfig`.
    ConservativeBump,
    /// Use the cached second-best argmin from
    /// [`Picker::argmin_masked_top_k`]. Codec falls through to
    /// `ConservativeBump` if the second-best's predicted bytes are
    /// too close to the first-best (no margin → unlikely to help).
    SecondBestPick,
    /// Hard-coded "always-safe" config for the target band — used
    /// when the picker's confidence is too low to trust either pick.
    KnownGoodFallback,
}

/// What the codec should do after observing pass-0 verify result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RescueDecision {
    /// Pass 0 met the target within the rescue threshold — ship it.
    Ship,
    /// Pass 0 undershot — codec should run pass 1 using the
    /// configured [`RescueStrategy`] and ship the higher-zq result.
    Rescue,
}

/// Predicate the codec calls after pass-0 verify to decide whether
/// to run a rescue pass.
///
/// `achieved_zq` and `target_zq` are both on the codec's quality
/// scale (zensim points). NaN / non-finite `achieved_zq` is treated
/// as a verify-failed signal and triggers rescue.
pub fn should_rescue(achieved_zq: f32, target_zq: f32, policy: &RescuePolicy) -> RescueDecision {
    if !achieved_zq.is_finite() {
        return RescueDecision::Rescue;
    }
    if achieved_zq < target_zq - policy.rescue_threshold_pp {
        RescueDecision::Rescue
    } else {
        RescueDecision::Ship
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ships_when_within_threshold() {
        let policy = RescuePolicy::default();
        // achieved 83 vs target 85 (gap 2pp), threshold 3pp → ship.
        assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Ship);
        // exact target → ship.
        assert_eq!(should_rescue(85.0, 85.0, &policy), RescueDecision::Ship);
        // overshoot → ship.
        assert_eq!(should_rescue(90.0, 85.0, &policy), RescueDecision::Ship);
    }

    #[test]
    fn rescues_on_threshold_breach() {
        let policy = RescuePolicy::default();
        // achieved 81 vs target 85 (gap 4pp), threshold 3pp → rescue.
        assert_eq!(should_rescue(81.0, 85.0, &policy), RescueDecision::Rescue);
    }

    #[test]
    fn rescues_on_nonfinite_achieved() {
        let policy = RescuePolicy::default();
        assert_eq!(
            should_rescue(f32::NAN, 85.0, &policy),
            RescueDecision::Rescue,
        );
        assert_eq!(
            should_rescue(f32::INFINITY, 85.0, &policy),
            RescueDecision::Rescue,
        );
        assert_eq!(
            should_rescue(f32::NEG_INFINITY, 85.0, &policy),
            RescueDecision::Rescue,
        );
    }

    #[test]
    fn custom_threshold_overrides_default() {
        let policy = RescuePolicy {
            rescue_threshold_pp: 1.0,
            strategy: RescueStrategy::SecondBestPick,
        };
        assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Rescue);
        assert_eq!(should_rescue(84.5, 85.0, &policy), RescueDecision::Ship);
    }

    #[test]
    fn default_policy_is_conservative_bump() {
        let policy = RescuePolicy::default();
        assert_eq!(policy.strategy, RescueStrategy::ConservativeBump);
        assert_eq!(policy.rescue_threshold_pp, 3.0);
    }
}
