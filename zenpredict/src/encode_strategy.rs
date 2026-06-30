//! Encode-budget-driven trial-encode strategy for multi-cell picker disambiguation.
//!
//! A K=1 picker's top-1 prediction is free (no encode), but for multi-cell codecs
//! (webp/avif) it can mis-pick a content-underdetermined knob — webp's 5-way `filter`
//! is only ~32% predictable from content, vs ~94% for `method` and ~62% for
//! `sharp_yuv` (held-out, 2026-06-29). A *multi-shot* pass trial-encodes the picker's
//! top-K ranked candidates (which differ only in the ambiguous knob) and keeps the
//! best by the codec's metric — accurate, but it costs extra encodes, and encode cost
//! scales with image size. So the budget is expressed in whichever unit the caller
//! controls and [`EncodeBudget::passes`] resolves it to a candidate count:
//!
//! - [`EncodeBudget::Passes`] — a hard count of trial encodes. `Passes(1)` = one-shot.
//! - [`EncodeBudget::TrialPixels`] — a pixel ceiling across trial encodes. This is the
//!   natural size discriminant: a large image fits fewer passes (each costs more
//!   pixels), a small image more, and below one image's worth it collapses to one-shot.
//! - [`EncodeBudget::Milliseconds`] — a wall-time ceiling, divided by the codec's
//!   estimated ms-per-encode for this image.
//!
//! This is the cheap, real-time-capable counterpart to a full offline metric-K-verify
//! (which would encode every cell). The picker emits ranked candidates; the codec runs
//! `passes()` trial encodes and keeps the best.

/// The resource the codec will spend trial-encoding the picker's ranked candidates to
/// disambiguate a content-underdetermined knob. Resolve to a candidate count with
/// [`EncodeBudget::passes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeBudget {
    /// A hard count of trial encodes (the picker's top-K). `Passes(1)` = one-shot.
    Passes(u16),
    /// Total pixels across trial encodes — size-adaptive: a large image gets fewer
    /// passes (each costs `image_pixels`), a small image more; below one image's worth
    /// it is one-shot.
    TrialPixels(u64),
    /// Wall-time ceiling (ms) for trial encodes, divided by the codec's estimated
    /// ms-per-encode for this image.
    Milliseconds(u32),
}

impl EncodeBudget {
    /// Strict real-time: a single encode, no trials.
    pub const ONE_SHOT: Self = EncodeBudget::Passes(1);

    /// How many of the picker's `n_candidates` ranked cells to trial-encode, given the
    /// image size (`image_pixels`, used by `TrialPixels`) and the codec's
    /// `est_ms_per_encode` for this image (used by `Milliseconds`). Always returns
    /// `>= 1` (the top pick is always encoded) and never exceeds `n_candidates`.
    pub fn passes(&self, n_candidates: usize, image_pixels: u64, est_ms_per_encode: u32) -> usize {
        let avail = n_candidates.max(1);
        let raw = match *self {
            EncodeBudget::Passes(p) => p as usize,
            EncodeBudget::TrialPixels(px) => {
                // Degenerate (no size info): don't let the budget constrain.
                if image_pixels == 0 {
                    avail
                } else {
                    (px / image_pixels) as usize
                }
            }
            EncodeBudget::Milliseconds(ms) => {
                if est_ms_per_encode == 0 {
                    avail
                } else {
                    (ms / est_ms_per_encode) as usize
                }
            }
        };
        raw.max(1).min(avail)
    }

    /// Whether the codec should run the multi-shot trial loop (`passes > 1`).
    pub fn is_multishot(&self, n_candidates: usize, image_pixels: u64, est_ms_per_encode: u32) -> bool {
        self.passes(n_candidates, image_pixels, est_ms_per_encode) > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // image_pixels / est irrelevant to Passes -> pass zeros.
    fn p(b: EncodeBudget, n: usize) -> usize {
        b.passes(n, 0, 0)
    }

    #[test]
    fn one_shot_is_single_pass() {
        assert_eq!(p(EncodeBudget::ONE_SHOT, 5), 1);
        assert_eq!(p(EncodeBudget::Passes(1), 5), 1);
        assert!(!EncodeBudget::ONE_SHOT.is_multishot(5, 0, 0));
    }

    #[test]
    fn passes_clamps_to_cap_and_candidates() {
        assert_eq!(EncodeBudget::Passes(3).passes(5, 9_999, 9), 3); // capped by count
        assert_eq!(EncodeBudget::Passes(3).passes(2, 9_999, 9), 2); // capped by candidates
    }

    #[test]
    fn trial_pixels_is_size_adaptive() {
        let small = 256 * 256;
        let large = 4096 * 4096;
        // budget = 3 small images -> 3 passes on a small image (capped at candidates)
        assert_eq!(EncodeBudget::TrialPixels(3 * small).passes(4, small, 0), 3);
        // same pixel budget on a large image -> fits < 1 image -> one-shot
        assert_eq!(EncodeBudget::TrialPixels(3 * small).passes(4, large, 0), 1);
        // exactly 2 images' worth -> 2 passes
        assert_eq!(EncodeBudget::TrialPixels(2 * small).passes(4, small, 0), 2);
    }

    #[test]
    fn milliseconds_divides_by_estimate() {
        // 300ms budget, 100ms/encode -> 3 passes
        assert_eq!(EncodeBudget::Milliseconds(300).passes(5, 0, 100), 3);
        // 50ms budget, 100ms/encode -> < 1 -> one-shot
        assert_eq!(EncodeBudget::Milliseconds(50).passes(5, 0, 100), 1);
    }

    #[test]
    fn degenerate_inputs_stay_valid() {
        assert_eq!(EncodeBudget::Passes(5).passes(0, 0, 0), 1); // zero candidates -> 1
        assert_eq!(EncodeBudget::Passes(0).passes(5, 0, 0), 1); // zero passes -> 1
        assert_eq!(EncodeBudget::TrialPixels(9).passes(4, 0, 0), 4); // no size -> all
    }
}
