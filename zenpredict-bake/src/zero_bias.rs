//! Pre-quantization zero-bias for ZNPR weight tensors.
//!
//! Applied to the raw `f32` weights before constructing a [`BakeLayer`].
//! Doesn't change the wire format — the biased weights still bake as
//! F32, F16, or I8 — but it creates long runs of exact zeros that the
//! (forthcoming) `compressed-weights` feature in `zenpredict` can
//! exploit. Without zero-bias the i8 LSBs look uniform-random and
//! lz4/zstd/deflate all plateau at compression ratio ~0.93.
//!
//! ## Semantics
//!
//! Per-output-column thresholding: for each output `o`, compute
//! `max_col = max(|W[:, o]|)`. Weights satisfying
//! `|W[i, o]| < threshold * max_col` become exactly `0.0`. The
//! per-output structure matches the per-output i8 scale.
//!
//! ## Calibrated thresholds (from
//! [`zensim/benchmarks/zenpredict_rle_zerobias_eval_2026-05-13.md`])
//!
//! | τ | i8-byte zero density | CID22 SROCC vs V0_18 |
//! |--:|--:|--:|
//! | 0     | 1.4 %  | 0      |
//! | 0.005 | 87.5 % | -0.0001 |
//! | 0.02  | 90.2 % | -0.0003 |
//! | 0.05  | 92.8 % | -0.0014 |
//!
//! `τ = 0.005` is the sweet spot: 87 percentage-point increase in zero
//! density at SROCC cost within sampling noise.
//!
//! ## Methodology requirement
//!
//! Per `zensim/CLAUDE.md`, every shipped bake using a non-zero τ must
//! land alongside a methodology doc that records τ + the post-bias
//! SROCC numbers on the 5 canonical corpora.
//!
//! [`BakeLayer`]: crate::BakeLayer

extern crate alloc;
use alloc::vec::Vec;

/// In-place per-output-column zero-bias. Mutates `weights` directly.
///
/// `weights` is row-major `[in_dim * out_dim]` (input-major; matches
/// the `BakeLayer.weights` layout). Threshold `tau` is a fraction of
/// each output column's max-abs weight; values strictly below
/// `tau * max_col` are set to `0.0`. `tau <= 0.0` is a no-op.
///
/// ```
/// // tau=0.5: anything below half the column max becomes 0.
/// let mut w = vec![1.0_f32, 0.1, 0.4, 2.0];  // in_dim=2, out_dim=2
/// zenpredict_bake::apply_zero_bias_in_place(&mut w, 2, 0.5);
/// // out=0 column: max=1.0, cut=0.5 → 0.4 zeros out (idx 2)
/// // out=1 column: max=2.0, cut=1.0 → 0.1 zeros out (idx 1)
/// assert_eq!(w, vec![1.0, 0.0, 0.0, 2.0]);
/// ```
pub fn apply_zero_bias_in_place(weights: &mut [f32], out_dim: usize, tau: f32) {
    if !(tau > 0.0) || out_dim == 0 {
        return;
    }
    assert_eq!(
        weights.len() % out_dim,
        0,
        "weights.len() ({}) must be divisible by out_dim ({})",
        weights.len(),
        out_dim
    );
    // Pass 1: per-output max abs.
    let mut max_col = alloc::vec![0.0f32; out_dim];
    for (idx, &w) in weights.iter().enumerate() {
        let o = idx % out_dim;
        let abs = w.abs();
        if abs > max_col[o] {
            max_col[o] = abs;
        }
    }
    // Pass 2: threshold.
    for (idx, w) in weights.iter_mut().enumerate() {
        let o = idx % out_dim;
        let cut = tau * max_col[o];
        if w.abs() < cut {
            *w = 0.0;
        }
    }
}

/// Allocating variant: copy `weights` and apply the same per-output
/// zero-bias. The original buffer is left untouched. Use when the
/// caller wants both the original and biased versions (e.g. to
/// validate SROCC delta before shipping a biased bake).
pub fn apply_zero_bias(weights: &[f32], out_dim: usize, tau: f32) -> Vec<f32> {
    let mut out = weights.to_vec();
    apply_zero_bias_in_place(&mut out, out_dim, tau);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_bias_zero_threshold_is_noop() {
        let mut w = alloc::vec![1.0_f32, 0.1, 0.4, 2.0];
        apply_zero_bias_in_place(&mut w, 2, 0.0);
        assert_eq!(w, alloc::vec![1.0, 0.1, 0.4, 2.0]);
    }

    #[test]
    fn zero_bias_negative_threshold_is_noop() {
        let mut w = alloc::vec![1.0_f32, 0.1, 0.4, 2.0];
        apply_zero_bias_in_place(&mut w, 2, -0.5);
        assert_eq!(w, alloc::vec![1.0, 0.1, 0.4, 2.0]);
    }

    #[test]
    fn zero_bias_per_column_max_is_independent() {
        // out_dim=2:
        //   col 0 weights: [1.0, 0.4]   max=1.0
        //   col 1 weights: [0.1, 2.0]   max=2.0
        // tau=0.5 cuts at 0.5 / 1.0
        let mut w = alloc::vec![1.0_f32, 0.1, 0.4, 2.0];
        apply_zero_bias_in_place(&mut w, 2, 0.5);
        assert_eq!(w, alloc::vec![1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn zero_bias_uniform_column_keeps_everything() {
        // When all weights in a column equal max_col, none satisfy
        // |w| < tau * max_col (the strict-less-than). Everything kept.
        let mut w = alloc::vec![5.0_f32, 5.0, 5.0, 5.0];
        apply_zero_bias_in_place(&mut w, 1, 0.999);
        assert_eq!(w, alloc::vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn zero_bias_zero_column_stays_zero() {
        let mut w = alloc::vec![0.0_f32, 0.0, 0.0];
        apply_zero_bias_in_place(&mut w, 1, 0.005);
        assert_eq!(w, alloc::vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn allocating_variant_does_not_touch_source() {
        let w_src = alloc::vec![1.0_f32, 0.1, 0.4, 2.0];
        let biased = apply_zero_bias(&w_src, 2, 0.5);
        assert_eq!(w_src, alloc::vec![1.0, 0.1, 0.4, 2.0]);
        assert_eq!(biased, alloc::vec![1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn calibrated_threshold_0_005_clears_small_weights() {
        // Spot-check the documented τ=0.005 behavior: any weight under
        // 0.5 % of column max → 0.
        let mut w = alloc::vec![100.0_f32, 0.4, 0.6, 0.51, 50.0, 50.0, 0.0, 100.0];
        // out_dim=4 → col maxes [100, 50, 50, 100]; cuts [0.5, 0.25, 0.25, 0.5]
        apply_zero_bias_in_place(&mut w, 4, 0.005);
        assert_eq!(w, alloc::vec![100.0, 0.4, 0.6, 0.51, 50.0, 50.0, 0.0, 100.0]);
        //                          ^col0 ^col1 ^col2 ^col3  ^col0 ^col1 ^col2 ^col3
        // col 0: cut=0.5; 100, 50 kept.
        // col 1: cut=0.25; 0.4, 50 kept (0.4 > 0.25).
        // col 2: cut=0.25; 0.6 > 0.25 kept; 0.0 already zero.
        // col 3: cut=0.5; 0.51 > 0.5 kept; 100 kept.

        // Now τ=0.02 cuts at 2/1/0.012/2 → idx-1 (col 1) 0.4 < 1 → 0;
        //                                  idx-3 (col 3) 0.51 < 2 → 0;
        //                                  idx-2 (col 2) 0.6 stays (> 0.012).
        let mut w2 = alloc::vec![100.0_f32, 0.4, 0.6, 0.51, 50.0, 50.0, 0.0, 100.0];
        apply_zero_bias_in_place(&mut w2, 4, 0.02);
        assert_eq!(w2, alloc::vec![100.0, 0.0, 0.6, 0.0, 50.0, 50.0, 0.0, 100.0]);
    }
}
