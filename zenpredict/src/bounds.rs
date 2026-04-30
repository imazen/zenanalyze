//! Per-feature numeric bounds + out-of-distribution detection.

use bytemuck::{Pod, Zeroable};

/// Inclusive low/high bounds for one input feature.
///
/// Bakes ship a `[FeatureBound; n_inputs]` table (typically the
/// `p01` / `p99` quantiles of each feature on the training corpus)
/// and consumers call [`first_out_of_distribution`] before scoring
/// to detect inputs the model wasn't trained on. Codecs typically
/// fall through to a known-good rescue strategy on a hit; scoring
/// crates may surface the OOD index as a metric warning.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct FeatureBound {
    pub low: f32,
    pub high: f32,
}

impl FeatureBound {
    pub const fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }

    /// `true` when `value` is finite and inside `[low, high]`.
    #[inline]
    pub fn contains(&self, value: f32) -> bool {
        value.is_finite() && value >= self.low && value <= self.high
    }
}

/// Index of the first feature outside its bounds, or `None` when
/// the entire vector is in-distribution.
///
/// Treats NaN / Inf as out-of-distribution — those inputs can't
/// be modelled and should always force fallback. `bounds.len()`
/// must equal `features.len()`; mismatched lengths panic in debug
/// builds and short-circuit at the shorter length in release.
pub fn first_out_of_distribution(features: &[f32], bounds: &[FeatureBound]) -> Option<usize> {
    debug_assert_eq!(
        features.len(),
        bounds.len(),
        "feature vector length must match bounds length",
    );
    for (i, (f, b)) in features.iter().zip(bounds.iter()).enumerate() {
        if !b.contains(*f) {
            return Some(i);
        }
    }
    None
}
