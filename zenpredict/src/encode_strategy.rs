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
//! - [`EncodeBudget`] — a *multi-axis resource ceiling*. Any subset of axes (pass
//!   count, trial pixels, wall-time) may be set; the most restrictive one binds.
//!
//! [`EncodeBudget::resolve`] combines them into a candidate count the codec trial-
//! encodes. This is the cheap, real-time-capable counterpart to a full offline
//! metric-K-verify (which would encode every cell).

/// The codec's trial-encode mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerStrategy {
    /// Trust the picker's top-1: a single encode, no trials (strict real-time).
    OneShot,
    /// Always trial-encode up to the budget (quality-first).
    MultiShot,
    /// Let the budget decide — size/time-adaptive. The recommended default: with a
    /// `max_trial_pixels` / `max_ms` axis a large image collapses to one-shot (no trial
    /// fits), a small one trials up to the budget.
    Auto,
}

/// Multi-axis resource ceiling for trial encodes. Any subset of axes may be set; the
/// **most restrictive (smallest) binds**. All-unset = unbounded (capped only by the
/// candidate count). [`Self::est_ms_per_encode`] feeds the [`Self::max_ms`] axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeBudget {
    /// Hard cap on trial encodes. `None` = not bounded by pass count.
    pub max_passes: Option<u16>,
    /// Pixel ceiling across trial encodes (size-adaptive: a large image fits fewer
    /// passes). `None` = not bounded by pixels.
    pub max_trial_pixels: Option<u64>,
    /// Wall-time ceiling (ms) for trials. `None` = not bounded by time. Applies only
    /// when `est_ms_per_encode > 0`.
    pub max_ms: Option<u32>,
    /// The codec's estimated ms per trial encode for this image (for `max_ms`).
    /// `0` = unknown (the `max_ms` axis is then skipped).
    pub est_ms_per_encode: u32,
}

impl EncodeBudget {
    /// Bound by a hard pass count only.
    pub const fn passes(n: u16) -> Self {
        Self { max_passes: Some(n), max_trial_pixels: None, max_ms: None, est_ms_per_encode: 0 }
    }
    /// Bound by a trial-pixel ceiling only (size-adaptive).
    pub const fn trial_pixels(px: u64) -> Self {
        Self { max_passes: None, max_trial_pixels: Some(px), max_ms: None, est_ms_per_encode: 0 }
    }
    /// Bound by a wall-time ceiling only, with the codec's per-encode estimate.
    pub const fn milliseconds(budget_ms: u32, est_ms_per_encode: u32) -> Self {
        Self { max_passes: None, max_trial_pixels: None, max_ms: Some(budget_ms), est_ms_per_encode }
    }

    /// How many of the picker's `n_candidates` ranked cells to trial-encode under
    /// `strategy` for an `image_pixels`-sized image. The min across every set axis,
    /// floored at 1 (the top pick is always encoded) and capped at `n_candidates`.
    /// `OneShot` is always 1; `MultiShot` / `Auto` spend the budget.
    pub fn resolve(&self, strategy: PickerStrategy, n_candidates: usize, image_pixels: u64) -> usize {
        let avail = n_candidates.max(1);
        match strategy {
            PickerStrategy::OneShot => 1,
            PickerStrategy::MultiShot | PickerStrategy::Auto => {
                let mut p = avail; // unbounded start; every set axis only tightens it
                if let Some(mp) = self.max_passes {
                    p = p.min(mp as usize);
                }
                if let Some(px) = self.max_trial_pixels {
                    if image_pixels > 0 {
                        p = p.min((px / image_pixels) as usize);
                    }
                }
                if let Some(ms) = self.max_ms {
                    if self.est_ms_per_encode > 0 {
                        p = p.min((ms / self.est_ms_per_encode) as usize);
                    }
                }
                p.max(1).min(avail)
            }
        }
    }

    /// Whether the codec should run the multi-shot trial loop (`resolve > 1`).
    pub fn is_multishot(&self, strategy: PickerStrategy, n_candidates: usize, image_pixels: u64) -> bool {
        self.resolve(strategy, n_candidates, image_pixels) > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PickerStrategy::*;

    #[test]
    fn one_shot_ignores_budget() {
        assert_eq!(EncodeBudget::passes(5).resolve(OneShot, 5, 1_000), 1);
        assert_eq!(EncodeBudget::trial_pixels(u64::MAX).resolve(OneShot, 5, 1), 1);
    }

    #[test]
    fn single_axis_clamps_to_cap_and_candidates() {
        assert_eq!(EncodeBudget::passes(3).resolve(MultiShot, 5, 0), 3); // budget binds
        assert_eq!(EncodeBudget::passes(3).resolve(MultiShot, 2, 0), 2); // candidates bind
    }

    #[test]
    fn trial_pixels_is_size_adaptive() {
        let (small, large) = (256 * 256, 4096u64 * 4096);
        assert_eq!(EncodeBudget::trial_pixels(3 * small).resolve(Auto, 4, small), 3);
        assert_eq!(EncodeBudget::trial_pixels(3 * small).resolve(Auto, 4, large), 1); // <1 fits
    }

    #[test]
    fn most_restrictive_axis_binds() {
        // passes allows 5, time allows 2 -> time binds at 2
        let b = EncodeBudget { max_passes: Some(5), max_ms: Some(200), est_ms_per_encode: 100, ..Default::default() };
        assert_eq!(b.resolve(MultiShot, 8, 0), 2);
        // passes allows 5, pixels allow 3 (on this image) -> pixels bind at 3
        let b2 = EncodeBudget { max_passes: Some(5), max_trial_pixels: Some(3 * 65536), ..Default::default() };
        assert_eq!(b2.resolve(Auto, 8, 65536), 3);
    }

    #[test]
    fn unset_axes_dont_constrain() {
        // only the candidate count caps an all-unset budget
        assert_eq!(EncodeBudget::default().resolve(MultiShot, 4, 9_999), 4);
        // max_ms set but no estimate -> that axis is skipped
        let b = EncodeBudget { max_ms: Some(10), est_ms_per_encode: 0, ..Default::default() };
        assert_eq!(b.resolve(MultiShot, 4, 0), 4);
    }

    #[test]
    fn degenerate_inputs_stay_valid() {
        assert_eq!(EncodeBudget::passes(5).resolve(MultiShot, 0, 0), 1); // no candidates -> 1
        assert_eq!(EncodeBudget::passes(0).resolve(MultiShot, 5, 0), 1); // zero budget -> 1
    }
}
