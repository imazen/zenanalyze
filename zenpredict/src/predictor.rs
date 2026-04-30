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

/// Scratch-owning forward-pass wrapper. Allocations happen in
/// [`Predictor::new`]; `predict` and the `argmin_*` family are
/// allocation-free hot paths.
pub struct Predictor<'a> {
    model: Model<'a>,
    scratch_a: alloc::vec::Vec<f32>,
    scratch_b: alloc::vec::Vec<f32>,
    output: alloc::vec::Vec<f32>,
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
}
