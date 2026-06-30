//! Encode-budget-driven one-shot vs multi-shot picker strategy.
//!
//! A K=1 picker's top-1 prediction is free (no encode), but for multi-cell codecs
//! (webp/avif) it can mis-pick a content-underdetermined knob — webp's 5-way `filter`
//! is only ~32% predictable from content, vs ~94% for `method` and ~62% for
//! `sharp_yuv` (held-out, 2026-06-29). A *multi-shot* pass encodes the picker's top-K
//! ranked candidates and keeps the best by the codec's own metric — accurate, but it
//! costs K encodes. Encode cost scales with image size, so the right mode is
//! size/budget-adaptive: **one-shot for large images** (encode expensive — trust the
//! cheap pick), **multi-shot for small images** (encode cheap — verify the ambiguous
//! knob). The codec supplies a budget + a strategy; this resolves how many of the
//! ranked candidates to actually encode.
//!
//! This is the cheap, real-time-capable counterpart to a full offline metric-K-verify
//! (which would encode every cell). The picker emits ranked candidates that differ
//! only in the ambiguous knob; the codec runs `passes()` encodes and keeps the best.

/// How the codec turns the picker's ranked candidates into encode passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerStrategy {
    /// Trust the picker's top-1: a single encode, no verification (real-time).
    OneShot,
    /// Encode the top-K candidates and keep the best by the codec's metric.
    MultiShot,
    /// Decide per-image from the budget: multi-shot while the size fits, else one-shot.
    Auto,
}

/// The resource budget the codec gives the strategy. `pixel_count` is the size
/// discriminant; `max_encode_passes` is the hard ceiling on encodes the codec will
/// spend on one image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeBudget {
    /// One-shot / multi-shot / auto.
    pub strategy: PickerStrategy,
    /// Hard cap on encode passes (clamped to `>= 1`). `1` forces one-shot regardless
    /// of `strategy` — the encoder will never spend more than this many encodes.
    pub max_encode_passes: u16,
    /// Image pixels (`width * height`) — the `Auto` size discriminant.
    pub pixel_count: u64,
    /// `Auto` only: permit multi-shot at or below this many pixels (where encode stays
    /// cheap). Above it, `Auto` collapses to one-shot.
    pub auto_multishot_max_pixels: u64,
}

impl EncodeBudget {
    /// Strict real-time default: one-shot, a single encode pass.
    pub const fn one_shot(pixel_count: u64) -> Self {
        Self {
            strategy: PickerStrategy::OneShot,
            max_encode_passes: 1,
            pixel_count,
            auto_multishot_max_pixels: 0,
        }
    }

    /// Size-adaptive default: multi-shot up to `auto_multishot_max_pixels`, capped at
    /// `max_encode_passes` encodes, else one-shot.
    pub const fn auto(pixel_count: u64, max_encode_passes: u16, auto_multishot_max_pixels: u64) -> Self {
        Self {
            strategy: PickerStrategy::Auto,
            max_encode_passes,
            pixel_count,
            auto_multishot_max_pixels,
        }
    }

    /// How many of the picker's `n_candidates` ranked cells to encode + verify.
    ///
    /// Always returns `>= 1` (a degenerate empty candidate list still encodes the
    /// picker's single best). Never exceeds `n_candidates` or `max_encode_passes`.
    /// `Auto` returns the multi-shot cap when `pixel_count <= auto_multishot_max_pixels`,
    /// otherwise `1`.
    pub fn passes(&self, n_candidates: usize) -> usize {
        let cap = (self.max_encode_passes as usize).max(1).min(n_candidates.max(1));
        match self.strategy {
            PickerStrategy::OneShot => 1,
            PickerStrategy::MultiShot => cap,
            PickerStrategy::Auto => {
                if self.pixel_count <= self.auto_multishot_max_pixels {
                    cap
                } else {
                    1
                }
            }
        }
    }

    /// Whether the codec should run the multi-shot encode+verify loop (`passes > 1`).
    pub fn is_multishot(&self, n_candidates: usize) -> bool {
        self.passes(n_candidates) > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_is_always_single_pass() {
        let b = EncodeBudget::one_shot(8_000_000);
        assert_eq!(b.passes(5), 1);
        assert!(!b.is_multishot(5));
    }

    #[test]
    fn multishot_clamps_to_cap_and_candidates() {
        let b = EncodeBudget { strategy: PickerStrategy::MultiShot, max_encode_passes: 3, pixel_count: 0, auto_multishot_max_pixels: 0 };
        assert_eq!(b.passes(5), 3); // capped by max_encode_passes
        assert_eq!(b.passes(2), 2); // capped by n_candidates
        assert!(b.is_multishot(3));
    }

    #[test]
    fn auto_picks_multishot_for_small_one_shot_for_large() {
        // small image (<= threshold) -> multi-shot up to the cap
        let small = EncodeBudget::auto(256 * 256, 3, 1_000_000);
        assert_eq!(small.passes(4), 3);
        assert!(small.is_multishot(4));
        // large image (> threshold) -> one-shot
        let large = EncodeBudget::auto(4096 * 4096, 3, 1_000_000);
        assert_eq!(large.passes(4), 1);
        assert!(!large.is_multishot(4));
        // exactly at the threshold -> still multi-shot
        let edge = EncodeBudget::auto(1_000_000, 3, 1_000_000);
        assert_eq!(edge.passes(4), 3);
    }

    #[test]
    fn degenerate_inputs_collapse_to_one() {
        // zero candidates -> still 1 (encode the single best)
        let b = EncodeBudget { strategy: PickerStrategy::MultiShot, max_encode_passes: 5, pixel_count: 0, auto_multishot_max_pixels: 0 };
        assert_eq!(b.passes(0), 1);
        // max_encode_passes 0 -> clamped to 1
        let z = EncodeBudget { strategy: PickerStrategy::MultiShot, max_encode_passes: 0, pixel_count: 0, auto_multishot_max_pixels: 0 };
        assert_eq!(z.passes(5), 1);
    }
}
