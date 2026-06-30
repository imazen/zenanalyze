//! Trial-encode strategy + budget for multi-cell picker disambiguation.
//!
//! A K=1 picker's top-1 prediction is free (no encode), but for multi-cell codecs
//! (webp/avif) it can mis-pick a content-underdetermined knob — webp's 5-way `filter`
//! is only ~32% predictable from content, vs ~94% for `method` and ~62% for
//! `sharp_yuv` (held-out, 2026-06-29). A *multi-shot* pass trial-encodes the picker's
//! top-K ranked candidates (which differ only in the ambiguous knob) and keeps the
//! best by the codec's metric — accurate, but it costs extra encodes whose cost scales
//! with image size.
//!
//! Two orthogonal controls:
//! - [`PickerStrategy`] — the codec's *mode*: trust the pick, force trials, or adapt.
//! - [`EncodeBudget`] — the *resource ceiling* for trials, in the unit the caller
//!   controls (a hard pass count, a pixel ceiling, or a wall-time ceiling).
//!
//! [`EncodeBudget::passes`] combines them into a candidate count the codec trial-
//! encodes. This is the cheap, real-time-capable counterpart to a full offline
//! metric-K-verify (which would encode every cell).

/// The codec's trial-encode mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerStrategy {
    /// Trust the picker's top-1: a single encode, no trials (strict real-time).
    OneShot,
    /// Always trial-encode per the budget (quality-first).
    MultiShot,
    /// Let the budget decide — size/time-adaptive. The recommended default with a
    /// `TrialPixels` / `Milliseconds` budget: a large image naturally collapses to
    /// one-shot (no trial fits), a small one trials up to the budget.
    Auto,
}

/// The resource ceiling for trial encodes, in the unit the caller controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeBudget {
    /// Hard count of trial encodes (the picker's top-K).
    Passes(u16),
    /// Pixel ceiling across trial encodes — size-adaptive: a large image fits fewer
    /// passes (each costs `image_pixels`), a small one more.
    TrialPixels(u64),
    /// Wall-time ceiling, with the codec's estimated ms-per-encode for this image.
    Milliseconds { budget_ms: u32, est_ms_per_encode: u32 },
}

impl EncodeBudget {
    /// Trial passes this budget alone affords for an `image_pixels`-sized image,
    /// before the strategy and candidate-count clamps. `usize::MAX` for a degenerate
    /// budget (no size / no estimate), so the candidate count becomes the only cap.
    fn affordable(&self, image_pixels: u64) -> usize {
        match *self {
            EncodeBudget::Passes(p) => p as usize,
            EncodeBudget::TrialPixels(px) => {
                if image_pixels == 0 {
                    usize::MAX
                } else {
                    (px / image_pixels) as usize
                }
            }
            EncodeBudget::Milliseconds { budget_ms, est_ms_per_encode } => {
                if est_ms_per_encode == 0 {
                    usize::MAX
                } else {
                    (budget_ms / est_ms_per_encode) as usize
                }
            }
        }
    }

    /// How many of the picker's `n_candidates` ranked cells to trial-encode under
    /// `strategy` for an `image_pixels`-sized image. Always `>= 1` (the top pick is
    /// always encoded) and never exceeds `n_candidates`. `OneShot` is always 1;
    /// `MultiShot` / `Auto` spend this budget.
    pub fn passes(&self, strategy: PickerStrategy, n_candidates: usize, image_pixels: u64) -> usize {
        let avail = n_candidates.max(1);
        match strategy {
            PickerStrategy::OneShot => 1,
            PickerStrategy::MultiShot | PickerStrategy::Auto => {
                self.affordable(image_pixels).max(1).min(avail)
            }
        }
    }

    /// Whether the codec should run the multi-shot trial loop (`passes > 1`).
    pub fn is_multishot(&self, strategy: PickerStrategy, n_candidates: usize, image_pixels: u64) -> bool {
        self.passes(strategy, n_candidates, image_pixels) > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PickerStrategy::*;

    #[test]
    fn one_shot_ignores_budget() {
        assert_eq!(EncodeBudget::Passes(5).passes(OneShot, 5, 1_000), 1);
        assert_eq!(EncodeBudget::TrialPixels(99_999_999).passes(OneShot, 5, 1), 1);
    }

    #[test]
    fn passes_budget_clamps_to_cap_and_candidates() {
        assert_eq!(EncodeBudget::Passes(3).passes(MultiShot, 5, 0), 3); // capped by budget
        assert_eq!(EncodeBudget::Passes(3).passes(MultiShot, 2, 0), 2); // capped by candidates
    }

    #[test]
    fn trial_pixels_is_size_adaptive() {
        let small = 256 * 256;
        let large = 4096 * 4096;
        // budget = 3 small images -> 3 passes on a small image (capped at candidates)
        assert_eq!(EncodeBudget::TrialPixels(3 * small).passes(Auto, 4, small), 3);
        // same budget on a large image -> < 1 image fits -> one-shot
        assert_eq!(EncodeBudget::TrialPixels(3 * small).passes(Auto, 4, large), 1);
    }

    #[test]
    fn milliseconds_carries_its_estimate() {
        let b = EncodeBudget::Milliseconds { budget_ms: 300, est_ms_per_encode: 100 };
        assert_eq!(b.passes(Auto, 5, 0), 3); // 300/100
        let tight = EncodeBudget::Milliseconds { budget_ms: 50, est_ms_per_encode: 100 };
        assert_eq!(tight.passes(Auto, 5, 0), 1); // < 1 -> one-shot
    }

    #[test]
    fn degenerate_inputs_stay_valid() {
        assert_eq!(EncodeBudget::Passes(5).passes(MultiShot, 0, 0), 1); // no candidates -> 1
        assert_eq!(EncodeBudget::Passes(0).passes(MultiShot, 5, 0), 1); // zero budget -> 1
        // no size info for TrialPixels -> candidate count is the only cap
        assert_eq!(EncodeBudget::TrialPixels(9).passes(Auto, 4, 0), 4);
    }
}
