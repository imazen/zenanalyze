//! Scratch-owning forward-pass wrapper.
//!
//! Owns a [`Model`] plus the scratch buffers that [`forward`] needs,
//! exposes one allocation-free `predict` entry. Both downstream
//! shapes — codec picker, perceptual scorer — wrap a `Predictor`
//! and add their own typed front door.

use crate::argmin::{self, AllowedMask, ArgminOffsets, ScoreTransform, pick_confidence_from_top_k};
use crate::error::PredictError;
use crate::inference::forward;
use crate::model::Model;
use crate::output_spec::{OutputValue, apply_spec};

/// Scratch-owning forward-pass wrapper. Allocations happen in
/// [`Predictor::new`]; `predict` and the `argmin_*` family are
/// allocation-free hot paths.
pub struct Predictor<'a> {
    model: Model<'a>,
    scratch_a: alloc::vec::Vec<f32>,
    scratch_b: alloc::vec::Vec<f32>,
    output: alloc::vec::Vec<f32>,
    /// Post-processed output buffer for [`Self::predict_with_specs`].
    /// Sized to `n_outputs` at construction; reused across calls.
    spec_output: alloc::vec::Vec<OutputValue>,
}

impl<'a> Predictor<'a> {
    pub fn new(model: Model<'a>) -> Self {
        let need = model.scratch_len();
        let n_out = model.n_outputs();
        Self {
            model,
            scratch_a: alloc::vec![0.0; need],
            scratch_b: alloc::vec![0.0; need],
            output: alloc::vec![0.0; n_out],
            spec_output: alloc::vec![OutputValue::Default; n_out],
        }
    }

    pub fn n_inputs(&self) -> usize {
        self.model.n_inputs()
    }

    pub fn n_outputs(&self) -> usize {
        self.model.n_outputs()
    }

    pub fn schema_hash(&self) -> u64 {
        self.model.schema_hash()
    }

    pub fn model(&self) -> &Model<'a> {
        &self.model
    }

    /// Run forward pass. Returns the output vector — for log-bytes
    /// regressors, this is `n_outputs` log-bytes-per-config; for
    /// zensim's V0_4 scorer, this is `[distance]`.
    ///
    /// Scratch buffers (`scratch_a`, `scratch_b`) are reused across
    /// calls without zeroing — every layer's matmul writes biases
    /// into the destination buffer **before** accumulating, so stale
    /// data from a prior call never leaks into the result. Calling
    /// `predict` twice with the same `features` is deterministic.
    pub fn predict(&mut self, features: &[f32]) -> Result<&[f32], PredictError> {
        forward(
            &self.model,
            features,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.output,
        )?;
        Ok(&self.output)
    }

    /// Run forward pass and apply the bake's per-output specs and
    /// sparse overrides. Returns one [`OutputValue`] per model
    /// output: either an `Override(f32)` carrying the post-processed
    /// value, or `Default` indicating "use the codec's built-in
    /// default for this parameter".
    ///
    /// Pipeline per output `i`:
    ///
    /// 1. forward pass yields `raw[i]`
    /// 2. `output_specs[i]` (if present) applies activation, clamp,
    ///    snap-to-discrete, then sentinel match (see
    ///    [`crate::output_spec::apply_spec`])
    /// 3. sparse overrides: any `(i, value)` entry replaces the
    ///    pipeline result with `Override(value)`, or with `Default`
    ///    when `value.is_nan()`
    ///
    /// When the bake has no `output_specs` section (raw-passthrough
    /// bake), every output is returned as `Override(raw[i])` —
    /// equivalent to [`Self::predict`] wrapped in `Override`. Sparse
    /// overrides still apply.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`]: feature-length mismatch, etc. The
    /// spec pipeline itself is infallible — invalid (offset, len)
    /// pairs were rejected at load time.
    pub fn predict_with_specs(
        &mut self,
        features: &[f32],
    ) -> Result<&[OutputValue], PredictError> {
        forward(
            &self.model,
            features,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.output,
        )?;
        let specs = self.model.output_specs();
        let pool = self.model.discrete_sets();
        // Two paths: with-specs (apply per-output pipeline) vs.
        // without (raw passthrough → Override). We keep the loop
        // tight in either case.
        if specs.is_empty() {
            for (slot, &raw) in self.spec_output.iter_mut().zip(self.output.iter()) {
                *slot = OutputValue::Override(raw);
            }
        } else {
            for (i, slot) in self.spec_output.iter_mut().enumerate() {
                *slot = apply_spec(&specs[i], self.output[i], pool);
            }
        }
        // Sparse overrides land last so a maintainer can force a
        // specific output even when the spec pipeline produced
        // something different.
        for entry in self.model.sparse_overrides() {
            let i = entry.idx as usize;
            // Bounds were validated at load time but defend in
            // depth.
            if i < self.spec_output.len() {
                self.spec_output[i] = if entry.value.is_nan() {
                    OutputValue::Default
                } else {
                    OutputValue::Override(entry.value)
                };
            }
        }
        Ok(&self.spec_output)
    }

    /// Pick the argmin output index over the masked set.
    pub fn argmin_masked(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<usize>, PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.output.len())?;
        }
        Ok(argmin::argmin_masked(
            &self.output,
            mask,
            transform,
            offsets,
        ))
    }

    /// Argmin over a sub-range of the output vector. Hybrid-heads
    /// pickers lay outputs out as
    /// `[bytes[0..n_cells], scalar1[0..n_cells], scalar2[0..n_cells], …]`;
    /// the argmin of interest runs over `bytes` only.
    pub fn argmin_masked_in_range(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<usize>, PredictError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        Ok(argmin::argmin_masked(
            &self.output[start..end],
            mask,
            transform,
            offsets,
        ))
    }

    pub fn argmin_masked_top_k<const K: usize>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<[Option<usize>; K], PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.output.len())?;
        }
        Ok(argmin::argmin_masked_top_k::<K>(
            &self.output,
            mask,
            transform,
            offsets,
        ))
    }

    pub fn argmin_masked_top_k_in_range<const K: usize>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<[Option<usize>; K], PredictError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        Ok(argmin::argmin_masked_top_k::<K>(
            &self.output[start..end],
            mask,
            transform,
            offsets,
        ))
    }

    /// Pick + log-domain confidence (gap to second-best). Returns
    /// `None` when the mask permits zero entries.
    pub fn pick_with_confidence(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<(usize, f32)>, PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.output.len())?;
        }
        let top = argmin::argmin_masked_top_k::<2>(&self.output, mask, transform, offsets);
        Ok(pick_confidence_from_top_k(
            &self.output,
            transform,
            offsets,
            top,
        ))
    }

    pub fn pick_with_confidence_in_range(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<(usize, f32)>, PredictError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        let slice = &self.output[start..end];
        let top = argmin::argmin_masked_top_k::<2>(slice, mask, transform, offsets);
        Ok(pick_confidence_from_top_k(slice, transform, offsets, top))
    }

    /// Run forward pass, then argmin under a caller-supplied score
    /// function (#55). The closure is called for each `i` where
    /// `mask.is_allowed(i)` and reads from the model's output slice
    /// `out` to compute a scalar score; smallest score wins.
    ///
    /// Use this when a `ScoreTransform + ArgminOffsets` adjustment
    /// can't express the desired reduction — e.g. RD-vs-time
    /// (`bytes + μ·ms` reading from two hybrid heads), multi-metric
    /// pickers (selecting one metric's sub-range with a runtime
    /// index), or codec-specific saturating clamps.
    ///
    /// Returns `None` when no entry is allowed by the mask.
    /// `mask.len()` must be ≥ `n_outputs`.
    pub fn argmin_masked_with_scorer<F>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<Option<usize>, PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let out = &self.output[..];
        Ok(argmin::argmin_masked_with_scorer(out.len(), mask, |i| {
            scorer(out, i)
        }))
    }

    /// Top-`K` argmin under a caller-supplied score function.
    /// Slots beyond the number of mask-allowed entries are `None`.
    pub fn argmin_masked_top_k_with_scorer<const K: usize, F>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<[Option<usize>; K], PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let out = &self.output[..];
        Ok(argmin::argmin_masked_top_k_with_scorer::<K, _>(
            out.len(),
            mask,
            |i| scorer(out, i),
        ))
    }

    /// Top-`K` argmin with scorer over a sub-range of the output
    /// vector. Slots beyond the number of mask-allowed entries are
    /// `None`. The scorer's `out` is the full predict output (so
    /// the closure can read any head); `i` is the cell index within
    /// the sub-range.
    pub fn argmin_masked_top_k_with_scorer_in_range<const K: usize, F>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<[Option<usize>; K], PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        let out = &self.output[..];
        Ok(argmin::argmin_masked_top_k_with_scorer::<K, _>(
            end - start,
            mask,
            |i| scorer(out, i),
        ))
    }

    /// Argmin with scorer over a sub-range of the output vector.
    /// Hybrid-heads pickers expose multiple per-cell heads; this
    /// lets the caller restrict the argmin to e.g. the bytes head
    /// while the closure freely reads from any head. The closure's
    /// `i` is **within the sub-range** (`0..(range.1 - range.0)`);
    /// the scorer's `out` slice is the full predictor output, so
    /// the closure can compute `out[N_CELLS + i]` for the time head
    /// while ranking by cell index `i`.
    pub fn argmin_masked_with_scorer_in_range<F>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        scorer: F,
    ) -> Result<Option<usize>, PredictError>
    where
        F: Fn(&[f32], usize) -> f32,
    {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        let out = &self.output[..];
        Ok(argmin::argmin_masked_with_scorer(end - start, mask, |i| {
            scorer(out, i)
        }))
    }
}
