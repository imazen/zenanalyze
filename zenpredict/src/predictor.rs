//! Scratch-owning forward-pass wrapper.
//!
//! Borrows a [`Model`] and owns the scratch buffers that [`forward`] needs,
//! exposes one allocation-free `predict` entry. Both downstream
//! shapes — codec picker, perceptual scorer — wrap a `Predictor`
//! and add their own typed front door.

#[cfg(feature = "advanced")]
use crate::argmin::pick_confidence_from_top_k;
use crate::argmin::{self, AllowedMask, ArgminOffsets, ScoreTransform};
use crate::error::PredictError;
use crate::feature_transform::{FeatureTransform, apply_feature_transforms};
use crate::inference::forward;
use crate::model::Model;
#[cfg(feature = "advanced")]
use crate::output_spec::{OutputValue, apply_spec};

/// Scratch-owning forward-pass wrapper. Allocations happen in
/// [`Predictor::new`]; `predict` and the `argmin_*` family are
/// allocation-free hot paths.
///
/// The selection wrappers accept the same caller-prepared model inputs as
/// [`Self::predict`]; they do not apply baked feature transforms. For raw
/// features, call [`Self::predict_transformed`] then a free selection function
/// such as [`crate::argmin_masked`]. This keeps input transformation explicit.
///
/// `Predictor` **borrows** the `Model` rather than owning it. This
/// lets a single `Model` live in `static OnceLock<Model>` and be
/// shared by per-thread `Predictor` instances without a `Mutex` —
/// `Model` is `Sync + Send` (read-only owned bytes), while
/// `Predictor` owns the mutable scratch buffers per-thread.
///
/// Single lifetime `'a`: the lifetime of the model borrow. For
/// `static OnceLock<Model>` it's `'static`.
pub struct Predictor<'a> {
    model: &'a Model,
    scratch: Scratch,
    #[cfg(feature = "advanced")]
    spec_output: alloc::vec::Vec<OutputValue>,
}
struct Scratch {
    values: alloc::boxed::Box<[f32]>,
    need: usize,
    n_out: usize,
    n_feat: usize,
    n_program: usize,
    n_compact: usize,
}
struct Parts<'a> {
    a: &'a mut [f32],
    b: &'a mut [f32],
    output: &'a mut [f32],
    feature: &'a mut [f32],
    program: &'a mut [f32],
    compact: &'a mut [f32],
}
impl Scratch {
    fn new(model: &Model) -> Result<Self, PredictError> {
        let need = model.scratch_len();
        let n_out = model.n_outputs();
        let n_feat = if model.feature_transforms().is_some() {
            model.n_inputs()
        } else {
            0
        };
        let n_program = model.input_program().map_or(0, |p| p.scratch_len());
        let n_compact = model.input_program().map_or(0, |p| p.outputs().len());
        let len = 2 * need + n_out + n_feat + n_program + n_compact;
        let values = bytemuck::allocation::try_zeroed_slice_box(len).map_err(|_| {
            PredictError::AllocationFailed {
                what: "predictor scratch",
            }
        })?;
        Ok(Self {
            values,
            need,
            n_out,
            n_feat,
            n_program,
            n_compact,
        })
    }
    fn parts(&mut self) -> Parts<'_> {
        let (a, rest) = self.values.split_at_mut(self.need);
        let (b, rest) = rest.split_at_mut(self.need);
        let (output, rest) = rest.split_at_mut(self.n_out);
        let (feature, rest) = rest.split_at_mut(self.n_feat);
        let (program, compact) = rest.split_at_mut(self.n_program);
        debug_assert_eq!(compact.len(), self.n_compact);
        Parts {
            a,
            b,
            output,
            feature,
            program,
            compact,
        }
    }
    fn output(&self) -> &[f32] {
        &self.values[2 * self.need..2 * self.need + self.n_out]
    }
}

impl<'a> Predictor<'a> {
    /// Construct a predictor borrowing `model`. The model lives at
    /// the caller's choosing — typically inside a
    /// `static OnceLock<Model>` so the same parsed bake is shared
    /// across many `Predictor` instances (one per thread for the
    /// lock-free path, or one per call site for ad-hoc use).
    ///
    /// Example (static singleton, lock-free per-thread):
    /// ```ignore
    /// use std::cell::RefCell;
    /// use std::sync::OnceLock;
    /// use zenpredict::{Model, Predictor};
    ///
    /// static BAKE: &[u8] = include_bytes!("picker.bin");
    /// static MODEL: OnceLock<Model> = OnceLock::new();
    ///
    /// thread_local! {
    ///     static PREDICTOR: RefCell<Predictor<'static>> =
    ///         RefCell::new(Predictor::new(MODEL.get_or_init(
    ///             || Model::from_bytes(BAKE).unwrap()
    ///         )).unwrap());
    /// }
    /// ```
    pub fn new(model: &'a Model) -> Result<Self, PredictError> {
        #[cfg(feature = "advanced")]
        let spec_output = {
            let mut v = alloc::vec::Vec::new();
            v.try_reserve_exact(model.n_outputs())
                .map_err(|_| PredictError::AllocationFailed {
                    what: "output specifications",
                })?;
            v.resize(model.n_outputs(), OutputValue::Default);
            v
        };
        Ok(Self {
            model,
            scratch: Scratch::new(model)?,
            #[cfg(feature = "advanced")]
            spec_output,
        })
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

    pub fn model(&self) -> &Model {
        self.model
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
    ///
    /// This does not execute an input program. For source columns of a
    /// composed bake use [`Self::predict_from_source`].
    /// This applies the embedded scaler, but does **not** apply the bake's
    /// nonlinear feature transforms. For raw features from extraction, use
    /// [`Self::predict_transformed`] when the bake declares those transforms.
    /// The `argmin_*` convenience methods use this same scaler-only contract;
    /// compose transformed prediction with the free selection functions when
    /// selecting directly from raw extracted features.
    pub fn predict(&mut self, features: &[f32]) -> Result<&[f32], PredictError> {
        let Parts { a, b, output, .. } = self.scratch.parts();
        forward(self.model, features, a, b, output)?;
        Ok(output)
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
    #[cfg(feature = "advanced")]
    pub fn predict_with_specs(&mut self, features: &[f32]) -> Result<&[OutputValue], PredictError> {
        self.predict(features)?;
        Ok(self.finish_output_specs())
    }

    #[cfg(feature = "advanced")]
    fn finish_output_specs(&mut self) -> &[OutputValue] {
        let specs = self.model.output_specs();
        let pool = self.model.discrete_sets();
        // Two paths: with-specs (apply per-output pipeline) vs.
        // without (raw passthrough → Override). We keep the loop
        // tight in either case.
        if specs.is_empty() {
            for (slot, &raw) in self
                .spec_output
                .iter_mut()
                .zip(self.scratch.output().iter())
            {
                *slot = OutputValue::Override(raw);
            }
        } else {
            for (i, slot) in self.spec_output.iter_mut().enumerate() {
                *slot = apply_spec(&specs[i], self.scratch.output()[i], pool);
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
        &self.spec_output
    }

    /// Starts from compact program outputs, without executing an input program.
    /// For source columns use [`Self::predict_from_source`].
    /// Like [`Self::predict`], but applies the bake's
    /// `zentrain.feature_transforms` (issue #52) to each feature
    /// element before standardize + forward pass. **This is the
    /// recommended entry point for codec runtimes** — calling
    /// [`Self::predict`] on a bake whose `FEATURE_TRANSFORMS`
    /// declares any non-identity transform produces silently-wrong
    /// predictions because the trainer's scaler stats already
    /// reflect the post-transform distribution.
    ///
    /// When the bake omits `feature_transforms` (or every entry is
    /// `Identity`), this method forwards the caller's slice without
    /// copying — same allocation profile as [`Self::predict`].
    pub fn predict_transformed(&mut self, features: &[f32]) -> Result<&[f32], PredictError> {
        let Parts {
            a,
            b,
            output,
            feature,
            ..
        } = self.scratch.parts();
        forward_transformed(self.model, features, feature, a, b, output)?;
        Ok(output)
    }

    /// Compose raw source columns, then apply baked feature transforms, scaler
    /// and network. Without an input program this equals predict_transformed.
    /// No allocations occur after construction. Selection uses the free argmin
    /// functions on the returned scores, keeping preprocessing explicit.
    pub fn predict_from_source(&mut self, features: &[f32]) -> Result<&[f32], PredictError> {
        let Some(program) = self.model.input_program() else {
            return self.predict_transformed(features);
        };
        let Parts {
            a,
            b,
            output,
            feature,
            program: values,
            compact,
        } = self.scratch.parts();
        program.apply(features, values, compact)?;
        forward_transformed(self.model, compact, feature, a, b, output)?;
        Ok(output)
    }

    /// Like [`Self::predict_with_specs`], but applies the bake's
    /// `zentrain.feature_transforms` (issue #52) to each feature
    /// element before forward pass. See
    /// [`Self::predict_transformed`] for the rationale on which
    /// entry point codec runtimes should call.
    #[cfg(feature = "advanced")]
    pub fn predict_with_specs_transformed(
        &mut self,
        features: &[f32],
    ) -> Result<&[OutputValue], PredictError> {
        self.predict_transformed(features)?;
        Ok(self.finish_output_specs())
    }

    /// Source composition followed by the same output specs and overrides as
    /// predict_with_specs. This includes all baked preprocessing stages.
    #[cfg(feature = "advanced")]
    pub fn predict_with_specs_from_source(
        &mut self,
        features: &[f32],
    ) -> Result<&[OutputValue], PredictError> {
        self.predict_from_source(features)?;
        Ok(self.finish_output_specs())
    }

    /// Per-feature transforms declared by the bake. Convenience
    /// re-export of [`Model::feature_transforms`] so callers that
    /// already hold a `Predictor` don't need to thread the model
    /// through.
    ///
    /// [`Model::feature_transforms`]: crate::Model::feature_transforms
    pub fn feature_transforms(&self) -> Option<&[FeatureTransform]> {
        self.model.feature_transforms()
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
            o.validate(self.scratch.output().len())?;
        }
        Ok(argmin::argmin_masked(
            self.scratch.output(),
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
        if end > self.scratch.output().len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.scratch.output().len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        Ok(argmin::argmin_masked(
            &self.scratch.output()[start..end],
            mask,
            transform,
            offsets,
        ))
    }

    /// Run the forward pass, then return the top-`K` lowest-scoring
    /// output indices the `mask` permits, ascending (best first).
    /// Slots beyond the number of mask-allowed entries are `None`.
    ///
    /// On the default surface, so a per-codec picker or `zenpicker` can
    /// run the proven "predict-top-K then encode-verify" path (`K = 3`
    /// typically) at runtime without re-implementing the ranking
    /// contract. Same masking + score-transform + offsets options as
    /// [`Self::argmin_masked`]; same NaN / tie-break / mask-length
    /// contract as [`argmin::argmin_masked_top_k`].
    pub fn argmin_masked_top_k<const K: usize>(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<[Option<usize>; K], PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.scratch.output().len())?;
        }
        Ok(argmin::argmin_masked_top_k::<K>(
            self.scratch.output(),
            mask,
            transform,
            offsets,
        ))
    }

    /// Top-`K` over a sub-range of the output vector (hybrid-heads
    /// layout — see [`Self::argmin_masked_in_range`]). Returned
    /// indices are within the sub-range. Same support guarantee as
    /// [`Self::argmin_masked_top_k`].
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
        if end > self.scratch.output().len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.scratch.output().len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        Ok(argmin::argmin_masked_top_k::<K>(
            &self.scratch.output()[start..end],
            mask,
            transform,
            offsets,
        ))
    }

    /// Pick + log-domain confidence (gap to second-best). Returns
    /// `None` when the mask permits zero entries.
    #[cfg(feature = "advanced")]
    pub fn pick_with_confidence(
        &mut self,
        features: &[f32],
        mask: &AllowedMask<'_>,
        transform: ScoreTransform,
        offsets: Option<&ArgminOffsets<'_>>,
    ) -> Result<Option<(usize, f32)>, PredictError> {
        self.predict(features)?;
        if let Some(o) = offsets {
            o.validate(self.scratch.output().len())?;
        }
        let top = argmin::argmin_masked_top_k::<2>(self.scratch.output(), mask, transform, offsets);
        Ok(pick_confidence_from_top_k(
            self.scratch.output(),
            transform,
            offsets,
            top,
        ))
    }

    #[cfg(feature = "advanced")]
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
        if end > self.scratch.output().len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.scratch.output().len(),
                got: end,
            });
        }
        if let Some(o) = offsets {
            o.validate(end - start)?;
        }
        let slice = &self.scratch.output()[start..end];
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
    #[cfg(feature = "advanced")]
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
        let out = self.scratch.output();
        Ok(argmin::argmin_masked_with_scorer(out.len(), mask, |i| {
            scorer(out, i)
        }))
    }

    /// Top-`K` argmin under a caller-supplied score function.
    /// Slots beyond the number of mask-allowed entries are `None`.
    #[cfg(feature = "advanced")]
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
        let out = self.scratch.output();
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
    #[cfg(feature = "advanced")]
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
        if end > self.scratch.output().len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.scratch.output().len(),
                got: end,
            });
        }
        let out = self.scratch.output();
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
    #[cfg(feature = "advanced")]
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
        if end > self.scratch.output().len() || start > end {
            return Err(PredictError::OutputDimMismatch {
                expected: self.scratch.output().len(),
                got: end,
            });
        }
        let out = self.scratch.output();
        Ok(argmin::argmin_masked_with_scorer(end - start, mask, |i| {
            scorer(out, i)
        }))
    }
}

// One transform implementation for compact and source-input serving paths.
fn forward_transformed(
    model: &Model,
    features: &[f32],
    feat_scratch: &mut [f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    output: &mut [f32],
) -> Result<(), PredictError> {
    let Some(transforms) = model.feature_transforms() else {
        return forward(model, features, scratch_a, scratch_b, output);
    };
    if features.len() != transforms.len() {
        return Err(PredictError::FeatureLenMismatch {
            expected: transforms.len(),
            got: features.len(),
        });
    }
    let dst = feat_scratch;
    if model.has_expander_feature_transforms() {
        let params = model.feature_transform_params();
        if params.is_none() && transforms.contains(&FeatureTransform::Sinusoidal) {
            return Err(PredictError::UnexpectedExpanderInScalarPipeline { feature_index: 0 });
        }
        crate::feature_transform::expand_model_inputs(transforms, params, features, dst)?;
    } else {
        match model.feature_transform_params() {
            Some(params) => {
                for i in 0..features.len() {
                    dst[i] = transforms[i].apply_with_params(features[i], &params[i]);
                }
            }
            None => apply_feature_transforms(transforms, features, dst)?,
        }
    }
    forward(model, dst, scratch_a, scratch_b, output)
}
