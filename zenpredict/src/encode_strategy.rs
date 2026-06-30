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
//! - [`EncodeBudget`] — a *multi-axis resource ceiling*. Any subset of axes may be set;
//!   the most restrictive binds.
//!
//! The count axes ([`EncodeBudget::max_passes`], [`EncodeBudget::max_trial_pixels`])
//! are static — [`EncodeBudget::resolve`] turns them into an upper bound on trial
//! encodes the codec pre-slices to. The time axis ([`EncodeBudget::max_ms`]) is
//! enforced at **runtime** by [`EncodeBudget::time_exhausted`]: the encoder is the only
//! thing that knows the real per-encode cost (it just timed one), so the loop checks
//! its own elapsed clock rather than asking the caller for a fragile estimate.

/// The codec's trial-encode mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerStrategy {
    /// Trust the picker's top-1: a single encode, no trials (strict real-time).
    OneShot,
    /// Always trial-encode up to the budget (quality-first).
    MultiShot,
    /// Let the budget decide — size/time-adaptive. The recommended default: with a
    /// `max_trial_pixels` axis a large image collapses to one-shot (no trial fits), a
    /// small one trials up to the budget; `max_ms` then stops the loop on the clock.
    Auto,
}

/// Multi-axis resource ceiling for trial encodes. Any subset of axes may be set; the
/// **most restrictive binds**. All-unset = unbounded (capped only by the candidate
/// count). No per-encode-time estimate is required — see [`Self::time_exhausted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeBudget {
    /// Hard cap on trial encodes. `None` = not bounded by pass count.
    pub max_passes: Option<u16>,
    /// Pixel ceiling across trial encodes (size-adaptive: a large image fits fewer
    /// passes). `None` = not bounded by pixels.
    pub max_trial_pixels: Option<u64>,
    /// Wall-time ceiling (ms) for trials, enforced at runtime by the codec's loop via
    /// [`Self::time_exhausted`]. `None` = not bounded by time.
    pub max_ms: Option<u32>,
}

impl EncodeBudget {
    /// Bound by a hard pass count only.
    pub const fn passes(n: u16) -> Self {
        Self { max_passes: Some(n), max_trial_pixels: None, max_ms: None }
    }
    /// Bound by a trial-pixel ceiling only (size-adaptive).
    pub const fn trial_pixels(px: u64) -> Self {
        Self { max_passes: None, max_trial_pixels: Some(px), max_ms: None }
    }
    /// Bound by a wall-time ceiling only (runtime-enforced).
    pub const fn milliseconds(ms: u32) -> Self {
        Self { max_passes: None, max_trial_pixels: None, max_ms: Some(ms) }
    }

    /// Static upper bound on trial encodes from the count + pixel axes (the codec
    /// pre-slices its candidate list to this many). Floored at 1 (the top pick always
    /// encodes) and capped at `n_candidates`. `OneShot` is always 1. The time axis is
    /// NOT applied here — the loop stops early via [`Self::time_exhausted`].
    pub fn resolve(&self, strategy: PickerStrategy, n_candidates: usize, image_pixels: u64) -> usize {
        let avail = n_candidates.max(1);
        match strategy {
            PickerStrategy::OneShot => 1,
            PickerStrategy::MultiShot | PickerStrategy::Auto => {
                let mut p = avail; // unbounded start; every set static axis tightens it
                if let Some(mp) = self.max_passes {
                    p = p.min(mp as usize);
                }
                if let Some(px) = self.max_trial_pixels {
                    if image_pixels > 0 {
                        p = p.min((px / image_pixels) as usize);
                    }
                }
                p.max(1).min(avail)
            }
        }
    }

    /// Runtime time-stop: `true` once the wall-time budget is spent. The codec calls
    /// this in its trial loop with the elapsed time since it began encoding this image,
    /// so the real (measured) per-encode cost governs the cutoff — no estimate needed.
    /// Always `false` when `max_ms` is unset.
    pub fn time_exhausted(&self, elapsed_ms: u32) -> bool {
        self.max_ms.is_some_and(|ms| elapsed_ms >= ms)
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
        assert_eq!(EncodeBudget::trial_pixels(3 * small).resolve(Auto, 4, large), 1);
    }

    #[test]
    fn most_restrictive_static_axis_binds() {
        // passes allows 5, pixels allow 3 on this image -> pixels bind at 3
        let b = EncodeBudget { max_passes: Some(5), max_trial_pixels: Some(3 * 65536), max_ms: None };
        assert_eq!(b.resolve(Auto, 8, 65536), 3);
    }

    #[test]
    fn unset_static_axes_dont_constrain() {
        assert_eq!(EncodeBudget::default().resolve(MultiShot, 4, 9_999), 4);
        // a time-only budget has no static cap -> resolve = all candidates; the loop
        // trims them on the clock via time_exhausted.
        assert_eq!(EncodeBudget::milliseconds(200).resolve(MultiShot, 4, 9_999), 4);
    }

    #[test]
    fn time_exhausted_uses_real_elapsed() {
        let b = EncodeBudget::milliseconds(200);
        assert!(!b.time_exhausted(100));
        assert!(b.time_exhausted(200));
        assert!(b.time_exhausted(350));
        // no time axis -> never exhausted
        assert!(!EncodeBudget::passes(3).time_exhausted(99_999));
    }

    #[test]
    fn degenerate_inputs_stay_valid() {
        assert_eq!(EncodeBudget::passes(5).resolve(MultiShot, 0, 0), 1); // no candidates -> 1
        assert_eq!(EncodeBudget::passes(0).resolve(MultiShot, 5, 0), 1); // zero budget -> 1
    }
}
