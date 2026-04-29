//! # zenpicker — codec-agnostic picker runtime
//!
//! Loads a packed MLP from bytes, runs SIMD inference, returns argmin
//! under caller-supplied constraints.
//!
//! The crate has no codec knowledge. Each codec crate (zenjpeg,
//! zenwebp, …) ships its own baked `.bin` model, declares its own
//! feature schema, and asks `zenpicker` to pick from a constraint
//! mask. Two-level use is just two models — outer model picks the
//! codec, inner model picks that codec's config.
//!
//! ## Lifecycle
//!
//! ```ignore
//! let bytes: &'static [u8] = include_bytes!("zenjpeg_picker_v1.bin");
//! let model = zenpicker::Model::from_bytes(bytes)?;
//! let mut picker = zenpicker::Picker::new(model);
//!
//! let features = my_codec::extract_features(&analysis, target_zq);
//! let mask = my_codec::allowed_configs(&caller_constraints);
//! let pick = picker.argmin_masked(&features, &mask, None)
//!     .expect("at least one config must be allowed");
//! ```
//!
//! ## Format stability
//!
//! The binary format is versioned (header.version). v1 is the only
//! version supported today. New versions add fields after the
//! existing header (header.header_size advertises the actual size,
//! so old loaders can skip over future extensions).
//!
//! ## Storage
//!
//! Weights are stored as f32 or f16. f16 is built in (no feature
//! gate, no `half` dep) — the conversion is ~15 lines of integer bit
//! math, see `inference::f16_bits_to_f32`. f16 halves the model size
//! at ~no accuracy cost; SIMD f16→f32 (F16C / FCVT) is the v0.2
//! work, scalar today.
//!
//! ## no_std
//!
//! `default-features = false` keeps the crate `no_std + alloc`. The
//! `std` feature adds `std::error::Error` impls.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod error;
mod inference;
mod mask;
mod model;

pub use error::PickerError;
pub use mask::{AllowedMask, argmin_masked, argmin_masked_top_k};
pub use model::{Activation, LayerView, Model, WeightDtype};

/// Caller-supplied additive cost adjustments applied to the model's
/// raw byte predictions before argmin.
///
/// `additive_bytes` is the size of metadata the caller plans to embed
/// (ICC, EXIF, XMP). It's the same across all configs so it doesn't
/// affect `argmin` on its own — but combined with `per_output_offset`
/// (e.g., XYB intrinsic ICC vs YCbCr-no-ICC) it can shift the pick.
///
/// `per_output_offset[i]` is added to the model's predicted
/// log-bytes-via-`exp` for output `i`. Use this when the format has
/// fixed per-config overhead the model couldn't learn (e.g., a new
/// caller constraint that disables a feature whose tax was baked
/// into training data).
#[derive(Clone, Copy, Debug, Default)]
pub struct CostAdjust<'a> {
    /// Added to all predicted byte counts (ICC, EXIF, …).
    pub additive_bytes: f32,
    /// Optional per-output additive bytes (length must equal `n_outputs`).
    pub per_output_offset: Option<&'a [f32]>,
}

/// Picker — wraps a [`Model`] with reusable scratch buffers for
/// repeated inference.
///
/// Allocations happen in `new`. `predict` and `argmin_masked` are
/// allocation-free hot paths.
pub struct Picker<'a> {
    model: Model<'a>,
    scratch_a: alloc::vec::Vec<f32>,
    scratch_b: alloc::vec::Vec<f32>,
    /// Last-prediction output buffer; sized to `n_outputs`.
    output: alloc::vec::Vec<f32>,
}

impl<'a> Picker<'a> {
    /// Create a picker over `model`. Pre-allocates scratch buffers.
    pub fn new(model: Model<'a>) -> Self {
        let max_hidden = model
            .layers()
            .iter()
            .map(|l| l.out_dim)
            .max()
            .unwrap_or(0)
            .max(model.n_inputs());
        let n_out = model.n_outputs();
        Self {
            model,
            scratch_a: alloc::vec![0.0; max_hidden],
            scratch_b: alloc::vec![0.0; max_hidden],
            output: alloc::vec![0.0; n_out],
        }
    }

    /// Number of input features the model expects.
    pub fn n_inputs(&self) -> usize {
        self.model.n_inputs()
    }

    /// Number of output values the model produces (one per config /
    /// per codec / per whatever the bake target was).
    pub fn n_outputs(&self) -> usize {
        self.model.n_outputs()
    }

    /// Schema hash baked into the model. Codec consumers should
    /// compare this to their compiled-in schema hash on load and
    /// fail loudly on mismatch.
    pub fn schema_hash(&self) -> u64 {
        self.model.schema_hash()
    }

    /// Run forward pass. Returns the raw output vector (for
    /// regressors, this is log-bytes-per-config).
    ///
    /// `features.len()` must equal `n_inputs()`. Returns a slice of
    /// length `n_outputs()`.
    pub fn predict(&mut self, features: &[f32]) -> Result<&[f32], PickerError> {
        if features.len() != self.model.n_inputs() {
            return Err(PickerError::FeatureLenMismatch {
                expected: self.model.n_inputs(),
                got: features.len(),
            });
        }
        inference::forward(
            &self.model,
            features,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.output,
        );
        Ok(&self.output)
    }

    /// Pick the argmin output index over the masked set, optionally
    /// applying additive cost adjustments.
    ///
    /// Returns `None` when no output is allowed by the mask. For a
    /// log-bytes regressor, the argmin in log-space is the same as
    /// argmin in raw bytes (`exp` is monotonic), but `CostAdjust`
    /// applies in raw-byte space — so when adjustments are non-None
    /// we materialize bytes via `exp` first.
    pub fn argmin_masked(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        adjust: Option<CostAdjust<'_>>,
    ) -> Result<Option<usize>, PickerError> {
        self.predict(features)?;
        Ok(mask::argmin_masked(&self.output, mask, adjust))
    }

    /// Pick the argmin within a sub-range of the output vector.
    ///
    /// Hybrid-heads picker layout (v0.2): the model's output vector
    /// is laid out as
    /// `[bytes_log[0..n_cells], scalar1[0..n_cells], scalar2[0..n_cells], …]`.
    /// The argmin of interest is over `bytes_log` only — not the
    /// scalar prediction heads. Codec consumers slice the output
    /// vector and feed the bytes-only sub-range here.
    ///
    /// `range.0..range.1` indexes into the output vector. `mask.len()`
    /// must equal `range.1 - range.0`. Returns the masked-argmin
    /// index *within the sub-range* (0..(range.1 - range.0)) — add
    /// `range.0` if you need the absolute index in the full output.
    pub fn argmin_masked_in_range(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        adjust: Option<CostAdjust<'_>>,
    ) -> Result<Option<usize>, PickerError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PickerError::FeatureLenMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        Ok(mask::argmin_masked(&self.output[start..end], mask, adjust))
    }

    /// Pick the top-`K` argmin output indices over the masked set in
    /// ascending score order (best first).
    ///
    /// Slots beyond the number of mask-allowed entries are `None`.
    /// Codec rescue logic (two-shot pass framework) uses `K = 2` —
    /// the second-best is the fallback when the first pick fails
    /// post-encode verification.
    pub fn argmin_masked_top_k<const K: usize>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        adjust: Option<CostAdjust<'_>>,
    ) -> Result<[Option<usize>; K], PickerError> {
        self.predict(features)?;
        Ok(mask::argmin_masked_top_k::<K>(&self.output, mask, adjust))
    }

    /// `argmin_masked_top_k` over a sub-range, in line with
    /// [`argmin_masked_in_range`]. Returns indices *within the
    /// sub-range* (0..(range.1 - range.0)).
    pub fn argmin_masked_top_k_in_range<const K: usize>(
        &mut self,
        features: &[f32],
        range: (usize, usize),
        mask: &AllowedMask<'_>,
        adjust: Option<CostAdjust<'_>>,
    ) -> Result<[Option<usize>; K], PickerError> {
        self.predict(features)?;
        let (start, end) = range;
        if end > self.output.len() || start > end {
            return Err(PickerError::FeatureLenMismatch {
                expected: self.output.len(),
                got: end,
            });
        }
        Ok(mask::argmin_masked_top_k::<K>(
            &self.output[start..end],
            mask,
            adjust,
        ))
    }
}

#[cfg(test)]
mod tests;
