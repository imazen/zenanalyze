//! Codec-side rescue plumbing for the two-shot pass framework.
//!
//! Encode + verify themselves stay codec-side (zenpredict has no
//! decoder, no zensim, no awareness of jpegli-q / lambda / chroma
//! scaling). What zenpredict contributes is the decision logic and
//! shared types so every codec wrapper writes the same orchestration
//! instead of inventing its own threshold rule:
//!
//! ```ignore
//! let top2 = predictor.argmin_masked_top_k::<2>(&features, &mask, transform, offsets)?;
//! let first = top2[0].expect("mask non-empty");
//! let bytes_0 = codec::encode(first);
//! let achieved = codec::verify(&bytes_0, &source);
//!
//! match zenpredict::should_rescue(achieved, target_zq, &policy) {
//!     RescueDecision::Ship => bytes_0,
//!     RescueDecision::Rescue => {
//!         let bytes_1 = match policy.strategy {
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
//! not loop.

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct RescuePolicy {
    /// Trigger rescue when `achieved < target − rescue_threshold`.
    /// Units match the codec's quality scale (zensim points). Default
    /// `3.0` is a placeholder pending held-out p99 zensim shortfall
    /// calibration.
    pub rescue_threshold: f32,
    pub strategy: RescueStrategy,
}

impl Default for RescuePolicy {
    fn default() -> Self {
        Self {
            rescue_threshold: 3.0,
            strategy: RescueStrategy::ConservativeBump,
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RescueStrategy {
    /// Bump quality (codec-defined: jpegli-q, lambda, distance) and
    /// disable lossy chroma options. Single deterministic transform
    /// of the first-pick `EncoderConfig`.
    ConservativeBump,
    /// Use the cached second-best argmin. Codec falls through to
    /// `ConservativeBump` if the second-best's predicted bytes are
    /// too close to the first-best (no margin → unlikely to help).
    SecondBestPick,
    /// Hard-coded "always-safe" config for the target band — used
    /// when the picker's confidence is too low to trust either pick.
    KnownGoodFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RescueDecision {
    Ship,
    Rescue,
}

/// Predicate the codec calls after pass-0 verify to decide whether
/// to run a rescue pass. Non-finite `achieved` is treated as
/// verify-failed and forces rescue.
pub fn should_rescue(achieved: f32, target: f32, policy: &RescuePolicy) -> RescueDecision {
    if !achieved.is_finite() {
        return RescueDecision::Rescue;
    }
    if achieved < target - policy.rescue_threshold {
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
        assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Ship);
        assert_eq!(should_rescue(85.0, 85.0, &policy), RescueDecision::Ship);
        assert_eq!(should_rescue(90.0, 85.0, &policy), RescueDecision::Ship);
    }

    #[test]
    fn rescues_on_threshold_breach() {
        let policy = RescuePolicy::default();
        assert_eq!(should_rescue(81.0, 85.0, &policy), RescueDecision::Rescue);
    }

    #[test]
    fn rescues_on_nonfinite_achieved() {
        let policy = RescuePolicy::default();
        assert_eq!(
            should_rescue(f32::NAN, 85.0, &policy),
            RescueDecision::Rescue
        );
        assert_eq!(
            should_rescue(f32::INFINITY, 85.0, &policy),
            RescueDecision::Rescue
        );
        assert_eq!(
            should_rescue(f32::NEG_INFINITY, 85.0, &policy),
            RescueDecision::Rescue
        );
    }

    #[test]
    fn custom_threshold_overrides_default() {
        let policy = RescuePolicy {
            rescue_threshold: 1.0,
            strategy: RescueStrategy::SecondBestPick,
        };
        assert_eq!(should_rescue(83.0, 85.0, &policy), RescueDecision::Rescue);
        assert_eq!(should_rescue(84.5, 85.0, &policy), RescueDecision::Ship);
    }
}
