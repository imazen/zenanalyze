//! Generic decision math: masked argmin, top-K, confidence,
//! threshold mask, score transforms, and additive offsets.
//!
//! None of this is codec-specific — `argmin_masked` is just "argmin
//! over a slice with a boolean filter," `ArgminOffsets` is just
//! "uniform additive plus per-output additive in the post-transform
//! score space." The names that *were* picker-flavored in the
//! pre-zenpredict codebase (`CostAdjust`, `additive_bytes`,
//! `per_output_offset`, `reach_gate_mask`) generalize cleanly here.

use crate::error::PredictError;

/// Boolean filter over a score slice. `true` means "this index may
/// be picked." Bit-packing isn't worth the API complexity for the
/// 10s–100s of outputs real bakes have.
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

/// Score-domain transform applied before offsets are added and
/// before argmin runs. Default is `Identity`.
///
/// Codecs whose model emits log-bytes (the typical zenjpeg shape)
/// pass `Exp` — argmin then runs in raw-byte space, which is what
/// matters when an offsets table mixes per-output overhead in
/// linear bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScoreTransform {
    /// Argmin over the raw model outputs (no transform). Use when
    /// outputs are already in the argmin-target space — perceptual
    /// distances, raw counts, etc.
    #[default]
    Identity,
    /// Apply `exp` (clamped to [-30, 30] input range to keep
    /// numerics finite) before adding offsets and running argmin.
    /// Lets log-domain regressors mix with linear-domain offsets.
    /// `no_std` builds without `f32::exp` produce ties; enable the
    /// `std` feature for correct linear-space argmin.
    Exp,
}

impl ScoreTransform {
    /// Apply the transform to a single score. Used by the argmin
    /// implementation; not typically called by consumers.
    #[inline]
    pub fn apply(self, score: f32) -> f32 {
        match self {
            Self::Identity => score,
            Self::Exp => clamped_exp(score),
        }
    }
}

/// Caller-supplied additive cost adjustments applied in the
/// post-transform score space, before argmin.
///
/// `uniform` is added to every output's score (e.g. caller's ICC /
/// EXIF / XMP overhead in a log-bytes-via-Exp picker). It's the
/// same across all outputs so it doesn't change argmin on its own
/// — but combined with `per_output` (e.g. XYB intrinsic ICC vs
/// YCbCr-no-ICC) it can shift the pick.
#[derive(Clone, Copy, Debug, Default)]
pub struct ArgminOffsets<'a> {
    /// Added to every output's score after the transform.
    pub uniform: f32,
    /// Optional per-output additive — when present must equal the
    /// argmin's working slice length (full `n_outputs` for
    /// `argmin_masked`, the sub-range length for `*_in_range`).
    pub per_output: Option<&'a [f32]>,
}

impl<'a> ArgminOffsets<'a> {
    pub fn uniform(uniform: f32) -> Self {
        Self {
            uniform,
            per_output: None,
        }
    }

    /// Validate the per-output length against the argmin's working
    /// slice length. Returns Ok when no per-output table is set.
    pub(crate) fn validate(&self, expected: usize) -> Result<(), PredictError> {
        if let Some(po) = self.per_output
            && po.len() != expected
        {
            return Err(PredictError::OffsetsLenMismatch {
                expected,
                got: po.len(),
            });
        }
        Ok(())
    }
}

/// Argmin over `predictions`, restricted by `mask`. Score for index
/// `i` is `transform(predictions[i]) + uniform + per_output[i]`.
///
/// Returns `None` when no entry is allowed by the mask.
///
/// Argmin in `Identity` space without offsets reduces to a simple
/// `f32::min` linear scan — the offsets / transform branches are
/// only walked when the caller actually opts in.
pub fn argmin_masked(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<usize> {
    debug_assert!(mask.len() >= predictions.len());
    let mut best_idx: Option<usize> = None;
    let mut best_score: f32 = f32::INFINITY;

    for (i, &raw) in predictions.iter().enumerate() {
        if !mask.is_allowed(i) {
            continue;
        }
        let score = score_at(raw, i, transform, offsets);
        if score < best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Top-`K` lowest-scoring indices over `predictions` that the mask
/// permits, ascending (best first). Slots beyond the number of
/// allowed entries are `None`. `K` is generic to keep the call
/// site allocation-free; in practice `K = 2` is what codec rescue
/// logic wants (cached second-best).
pub fn argmin_masked_top_k<const K: usize>(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> [Option<usize>; K] {
    debug_assert!(mask.len() >= predictions.len());

    let mut top: [(f32, usize); K] = [(f32::INFINITY, usize::MAX); K];
    let mut count: usize = 0;

    let mut consider = |score: f32, idx: usize| {
        if count < K {
            let mut i = count;
            while i > 0 && top[i - 1].0 > score {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = (score, idx);
            count += 1;
        } else if K > 0 && score < top[K - 1].0 {
            let mut i = K - 1;
            while i > 0 && top[i - 1].0 > score {
                top[i] = top[i - 1];
                i -= 1;
            }
            top[i] = (score, idx);
        }
    };

    for (i, &raw) in predictions.iter().enumerate() {
        if !mask.is_allowed(i) {
            continue;
        }
        let score = score_at(raw, i, transform, offsets);
        consider(score, i);
    }

    let mut out: [Option<usize>; K] = [None; K];
    for slot in 0..count {
        out[slot] = Some(top[slot].1);
    }
    out
}

/// Argmin over `predictions[range.0..range.1]`, masked by `mask`
/// (whose `len()` must equal `range.1 - range.0`). Returned index
/// is *within the sub-range* (0..(range.1 - range.0)).
pub fn argmin_masked_in_range(
    predictions: &[f32],
    range: (usize, usize),
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<usize> {
    let (start, end) = range;
    let slice = predictions.get(start..end)?;
    argmin_masked(slice, mask, transform, offsets)
}

pub fn argmin_masked_top_k_in_range<const K: usize>(
    predictions: &[f32],
    range: (usize, usize),
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> [Option<usize>; K] {
    let (start, end) = range;
    if end > predictions.len() || start > end {
        return [None; K];
    }
    argmin_masked_top_k::<K>(&predictions[start..end], mask, transform, offsets)
}

/// Pick the argmin and report a confidence signal: the score gap
/// to the second-best mask-allowed entry. Returns `(best_idx, gap)`
/// where `gap` is in the same score units argmin used (post-
/// transform, post-offsets).
///
/// `gap = +∞` when only one mask entry is allowed; `0.0` if every
/// score ties at the top. Returns `None` when the mask permits
/// zero entries.
pub fn pick_with_confidence(
    predictions: &[f32],
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<(usize, f32)> {
    let top = argmin_masked_top_k::<2>(predictions, mask, transform, offsets);
    pick_confidence_from_top_k(predictions, transform, offsets, top)
}

pub fn pick_with_confidence_in_range(
    predictions: &[f32],
    range: (usize, usize),
    mask: &AllowedMask<'_>,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> Option<(usize, f32)> {
    let (start, end) = range;
    let slice = predictions.get(start..end)?;
    let top = argmin_masked_top_k::<2>(slice, mask, transform, offsets);
    pick_confidence_from_top_k(slice, transform, offsets, top)
}

pub(crate) fn pick_confidence_from_top_k(
    predictions: &[f32],
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
    top: [Option<usize>; 2],
) -> Option<(usize, f32)> {
    let best = top[0]?;
    let Some(second) = top[1] else {
        return Some((best, f32::INFINITY));
    };
    let s_best = score_at(predictions[best], best, transform, offsets);
    let s_second = score_at(predictions[second], second, transform, offsets);
    Some((best, (s_second - s_best).max(0.0)))
}

/// Fill `out[i] = rates[i].is_finite() && rates[i] >= threshold`.
/// Generic version of the picker's old `reach_gate_mask` —
/// thresholding a per-output score against a runtime-chosen value.
///
/// Codec consumers AND this against their constraint mask before
/// calling [`argmin_masked`].
///
/// `out.len()` must equal `rates.len()`.
pub fn threshold_mask(rates: &[f32], threshold: f32, out: &mut [bool]) {
    debug_assert_eq!(rates.len(), out.len());
    for (i, &r) in rates.iter().enumerate() {
        out[i] = r.is_finite() && r >= threshold;
    }
}

#[inline]
fn score_at(
    raw: f32,
    idx: usize,
    transform: ScoreTransform,
    offsets: Option<&ArgminOffsets<'_>>,
) -> f32 {
    let mut s = transform.apply(raw);
    if let Some(o) = offsets {
        s += o.uniform;
        if let Some(po) = o.per_output
            && let Some(&v) = po.get(idx)
        {
            s += v;
        }
    }
    s
}

fn clamped_exp(x: f32) -> f32 {
    let x = x.clamp(-30.0, 30.0);
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        // no_std + alloc has no `f32::exp`. Returning `x` keeps the
        // ordering monotonic so argmin still picks the smallest log
        // value, but the magnitudes won't reflect linear-space
        // mixing with offsets. Enable `std` for correctness.
        x
    }
}
