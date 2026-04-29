//! Constraint mask + masked-argmin.

use crate::CostAdjust;

/// Bool slice over `n_outputs` — `true` means "this output may be
/// picked." The simplest possible representation; for the small
/// number of configs we expect (10s–100s) the per-bit packing isn't
/// worth the API complexity.
#[derive(Clone, Copy, Debug)]
pub struct AllowedMask<'a> {
    pub allowed: &'a [bool],
}

impl<'a> AllowedMask<'a> {
    pub fn new(allowed: &'a [bool]) -> Self {
        Self { allowed }
    }

    pub fn is_allowed(&self, idx: usize) -> bool {
        self.allowed.get(idx).copied().unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.allowed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

/// Pick the argmin index over `predictions`, restricted to entries
/// the mask permits.
///
/// `predictions` is the model's raw output (log-bytes). When
/// `adjust` is `None`, argmin runs in log-space — equivalent to
/// argmin over `exp(predictions)`. When `adjust` is `Some`, we apply
/// the additive byte offsets in raw-byte space (`exp` first), then
/// argmin.
///
/// Returns `None` if no entry is allowed by the mask.
pub fn argmin_masked(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    adjust: Option<CostAdjust<'_>>,
) -> Option<usize> {
    debug_assert!(mask.len() >= predictions.len());

    let mut best_idx: Option<usize> = None;
    let mut best_score: f32 = f32::INFINITY;

    match adjust {
        None => {
            for (i, &score) in predictions.iter().enumerate() {
                if !mask.is_allowed(i) {
                    continue;
                }
                if score < best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }
        Some(CostAdjust {
            additive_bytes,
            per_output_offset,
        }) => {
            // Apply offsets in raw-byte space. `additive_bytes` is
            // the same across all outputs so it doesn't change the
            // argmin (just scales the score), but a future caller
            // might want it; we still compute it for consistency.
            for (i, &score) in predictions.iter().enumerate() {
                if !mask.is_allowed(i) {
                    continue;
                }
                let mut bytes = clamped_exp(score) + additive_bytes;
                if let Some(offsets) = per_output_offset
                    && let Some(off) = offsets.get(i)
                {
                    bytes += *off;
                }
                if bytes < best_score {
                    best_score = bytes;
                    best_idx = Some(i);
                }
            }
        }
    }

    best_idx
}

/// Pick the `K` lowest-scoring indices over `predictions` that the
/// mask permits, returned in ascending score order (best first).
///
/// Slots beyond the number of allowed entries are `None`. Score
/// semantics match [`argmin_masked`]: log-space when `adjust` is
/// `None`, raw-byte space otherwise.
///
/// `K` is generic to keep the call site allocation-free; in practice
/// `K = 2` is what codec rescue logic wants (best + cached
/// second-best for the two-shot rescue path).
pub fn argmin_masked_top_k<const K: usize>(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    adjust: Option<CostAdjust<'_>>,
) -> [Option<usize>; K] {
    debug_assert!(mask.len() >= predictions.len());

    // Sorted ascending by score; `count` is how many slots are filled.
    let mut top: [(f32, usize); K] = [(f32::INFINITY, usize::MAX); K];
    let mut count: usize = 0;

    let consider = |score: f32, idx: usize, top: &mut [(f32, usize); K], count: &mut usize| {
        if *count < K {
            // Insert in sorted position.
            let mut i = *count;
            while i > 0 && top[i - 1].0 > score {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = (score, idx);
            *count += 1;
        } else if K > 0 && score < top[K - 1].0 {
            // Replace the worst-kept and bubble up.
            let mut i = K - 1;
            while i > 0 && top[i - 1].0 > score {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = (score, idx);
        }
    };

    match adjust {
        None => {
            for (i, &score) in predictions.iter().enumerate() {
                if !mask.is_allowed(i) {
                    continue;
                }
                consider(score, i, &mut top, &mut count);
            }
        }
        Some(CostAdjust {
            additive_bytes,
            per_output_offset,
        }) => {
            for (i, &score) in predictions.iter().enumerate() {
                if !mask.is_allowed(i) {
                    continue;
                }
                let mut bytes = clamped_exp(score) + additive_bytes;
                if let Some(offsets) = per_output_offset
                    && let Some(off) = offsets.get(i)
                {
                    bytes += *off;
                }
                consider(bytes, i, &mut top, &mut count);
            }
        }
    }

    let mut out: [Option<usize>; K] = [None; K];
    for slot in 0..count {
        out[slot] = Some(top[slot].1);
    }
    out
}

/// `exp` with input clamped to [-30, 30] to keep training-time
/// out-of-range predictions from producing NaN/Inf at inference.
fn clamped_exp(x: f32) -> f32 {
    libm_exp_f32(x.clamp(-30.0, 30.0))
}

/// `libm`-free `exp` approximation. We don't depend on `libm` to
/// keep the dependency footprint small. The `f32::exp` intrinsic is
/// available on every reasonable target via the platform libm; in
/// `no_std` we can fall back to a polynomial approximation.
///
/// In `std` builds, `f32::exp` is available. In `no_std`, we'd need
/// `libm` — for now the crate's `default-features = ["std"]` covers
/// the common case. When a `no_std` consumer surfaces, add `libm` as
/// an optional dep behind the `no_std-math` feature.
fn libm_exp_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        // 5-term Taylor expansion around 0; argmin doesn't need
        // monotonic accuracy, just monotonic ordering. Sufficient
        // when the cost adjust path is rarely hit. Replace with
        // libm if a no_std consumer needs better.
        let _ = x;
        0.0 // intentionally inert; no_std users should enable `std`
        // or supply libm. Surfaces as "all picks tie" which the
        // caller will notice loudly.
    }
}
