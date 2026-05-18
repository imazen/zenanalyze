//! ZNPR v3 byte-stream composer.
//!
//! Emits the v3 wire format described in zenpredict's `model` module
//! (see [`zenpredict::Model`] and [`zenpredict::Header`]). Earlier
//! formats (v1, v2) are not producible from this crate; older bakes
//! must be migrated via `zentrain/tools/migrate_znpr_v2_to_v3.py`.

use core::fmt;

use zenpredict::{
    Activation, MetadataType, OutputSpec, OutputTransform, Section, SparseOverride, WeightDtype,
};

/// Bake-side input for a single codec's slot in a multi-codec joint
/// Errors raised by [`bake`]. Distinct from `PredictError` —
/// these are bake-side validation failures, not runtime decode
/// issues.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BakeError {
    EmptyLayers,
    LayerDimMismatch {
        layer: usize,
        expected_in: usize,
        got_in: usize,
    },
    WeightLengthMismatch {
        layer: usize,
        expected: usize,
        got: usize,
    },
    BiasLengthMismatch {
        layer: usize,
        expected: usize,
        got: usize,
    },
    ScalesLengthMismatch {
        layer: usize,
        expected: usize,
        got: usize,
    },
    ScalerLengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    FeatureBoundsLengthMismatch {
        expected: usize,
        got: usize,
    },
    MetadataKeyTooLong {
        len: usize,
    },
    MetadataKeyEmpty,
    /// `output_specs.len()` was non-zero but didn't equal `n_outputs`.
    OutputSpecsLengthMismatch {
        expected: usize,
        got: usize,
    },
    /// An [`OutputSpec`]'s `(discrete_set_offset, discrete_set_len)`
    /// addresses bytes outside the `discrete_sets` pool.
    OutputSpecDiscreteOutOfRange {
        output_index: usize,
        offset: u32,
        len: u32,
        pool_len: usize,
    },
    /// An [`OutputSpec`]'s transform byte wasn't a recognized
    /// [`OutputTransform`] variant. (Bake-side rejects unknowns to
    /// avoid silent misbehaviour at load time.)
    UnknownOutputTransform {
        output_index: usize,
        byte: u8,
    },
    /// A [`SparseOverride`]'s `idx` was `>= n_outputs`.
    SparseOverrideIndexOutOfRange {
        idx: u32,
        n_outputs: usize,
    },
    /// `feature_order` length didn't equal `n_inputs`.
    FeatureOrderLengthMismatch {
        expected: usize,
        got: usize,
    },
    /// `output_order` length didn't equal `n_outputs`.
    OutputOrderLengthMismatch {
        expected: usize,
        got: usize,
    },
    /// A permutation index was out of range (`>= n`) or duplicate.
    InvalidPermutation {
        what: &'static str,
    },
    /// Per-feature `FeatureTransform` declared in
    /// `zentrain.feature_transforms` metadata was unrecognized at
    /// bake time. Mirrors the runtime parser's
    /// `PredictError::UnknownFeatureTransform`. Bake-side rejection
    /// stops a misspelled token from shipping in a wire-format bake.
    UnknownFeatureTransformToken {
        feature_index: usize,
    },
    /// `feature_transforms` and `feature_transform_params` line counts
    /// didn't agree (both must equal `n_inputs` when present).
    FeatureTransformsLenMismatch {
        expected: usize,
        got: usize,
    },
    /// A parameterized [`zenpredict::FeatureTransform`] entry carried
    /// fewer params than its variant requires. Two-step stacked
    /// variants need 2 params (`p1, p99` or `q1, q99`);
    /// `ClipThenLog1pThenWinsor` needs 3 (`eps, q1, q99`);
    /// `ClipThenLog1p` needs 1.
    FeatureTransformParamArityMismatch {
        feature_index: usize,
        transform: &'static str,
        expected: usize,
        got: usize,
    },
    /// A `WinsorThenLog` / `WinsorThenLog1p` entry carried a `p1`
    /// outside the variant's domain (`p1 <= 0` for `WinsorThenLog`;
    /// `p1 < -1` for `WinsorThenLog1p`). Shipping would produce
    /// `NaN` / `-Inf` at runtime, so the composer refuses.
    FeatureTransformParamInvalid {
        feature_index: usize,
        transform: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLayers => write!(f, "bake: layers list is empty"),
            Self::LayerDimMismatch {
                layer,
                expected_in,
                got_in,
            } => write!(
                f,
                "bake: layer {layer} expected in_dim {expected_in}, got {got_in}"
            ),
            Self::WeightLengthMismatch {
                layer,
                expected,
                got,
            } => write!(
                f,
                "bake: layer {layer} weights length {got} != expected {expected}"
            ),
            Self::BiasLengthMismatch {
                layer,
                expected,
                got,
            } => write!(
                f,
                "bake: layer {layer} biases length {got} != expected {expected}"
            ),
            Self::ScalesLengthMismatch {
                layer,
                expected,
                got,
            } => write!(
                f,
                "bake: layer {layer} I8 scales length {got} != expected {expected}"
            ),
            Self::ScalerLengthMismatch {
                what,
                expected,
                got,
            } => write!(f, "bake: {what} length {got} != expected {expected}"),
            Self::FeatureBoundsLengthMismatch { expected, got } => write!(
                f,
                "bake: feature_bounds length {got} != expected {expected} (n_inputs * 2)"
            ),
            Self::MetadataKeyTooLong { len } => {
                write!(f, "bake: metadata key length {len} exceeds u8 max (255)")
            }
            Self::MetadataKeyEmpty => write!(f, "bake: metadata key is empty"),
            Self::OutputSpecsLengthMismatch { expected, got } => write!(
                f,
                "bake: output_specs length {got} != expected {expected} (n_outputs)"
            ),
            Self::OutputSpecDiscreteOutOfRange {
                output_index,
                offset,
                len,
                pool_len,
            } => write!(
                f,
                "bake: output_spec[{output_index}].discrete_set range \
                 (offset={offset}, len={len}) outside discrete_sets pool of {pool_len} f32s"
            ),
            Self::UnknownOutputTransform { output_index, byte } => write!(
                f,
                "bake: output_spec[{output_index}].transform byte {byte:#x} is not a known OutputTransform variant"
            ),
            Self::SparseOverrideIndexOutOfRange { idx, n_outputs } => write!(
                f,
                "bake: sparse_override idx {idx} >= n_outputs {n_outputs}"
            ),
            Self::FeatureOrderLengthMismatch { expected, got } => write!(
                f,
                "bake: feature_order length {got} != expected {expected} (n_inputs)"
            ),
            Self::OutputOrderLengthMismatch { expected, got } => write!(
                f,
                "bake: output_order length {got} != expected {expected} (n_outputs)"
            ),
            Self::InvalidPermutation { what } => {
                write!(
                    f,
                    "bake: {what} is not a valid permutation (out of range or duplicate)"
                )
            }
            Self::UnknownFeatureTransformToken { feature_index } => write!(
                f,
                "bake: feature_transforms[{feature_index}] is not a known token"
            ),
            Self::FeatureTransformsLenMismatch { expected, got } => write!(
                f,
                "bake: feature_transforms / feature_transform_params length {got} != expected {expected} (n_inputs)"
            ),
            Self::FeatureTransformParamArityMismatch {
                feature_index,
                transform,
                expected,
                got,
            } => write!(
                f,
                "bake: feature_transform_params[{feature_index}] for {transform} expected {expected} param(s), got {got}"
            ),
            Self::FeatureTransformParamInvalid {
                feature_index,
                transform,
                reason,
            } => write!(
                f,
                "bake: feature_transform_params[{feature_index}] for {transform}: {reason}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BakeError {}

/// One layer's data, as Rust-native `f32`. The composer converts to
/// the requested storage dtype (f32 / f16 / i8) at write time.
pub struct BakeLayer<'a> {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Activation,
    pub dtype: WeightDtype,
    pub weights: &'a [f32],
    pub biases: &'a [f32],
}

/// One metadata key/value entry. Borrows from the caller.
pub struct BakeMetadataEntry<'a> {
    pub key: &'a str,
    pub kind: MetadataType,
    pub value: &'a [u8],
}

/// All inputs to a v3 bake. Construct directly via struct literal,
/// or use [`BakeRequest::new`] / [`BakeRequest::builder`] for the
/// common case of "required fields plus empty optional sections."
///
/// Future v3.x section additions will require either a minor version
/// bump on this crate (since this struct is not `#[non_exhaustive]`)
/// or a separate `BakeRequestV2`-style sibling — we accept the
/// breaking-change cost in exchange for the construction ergonomics.
pub struct BakeRequest<'a> {
    pub schema_hash: u64,
    /// Header flags. Bits 0..=3 are managed by the composer based on
    /// `compressed` / `compression_algo` — caller should leave them
    /// zero. Bits 4..=15 are reserved.
    pub flags: u16,
    pub scaler_mean: &'a [f32],
    pub scaler_scale: &'a [f32],
    pub layers: &'a [BakeLayer<'a>],
    /// Length must equal `n_inputs` (one [low, high] pair per
    /// input). Pass an empty slice to omit the section.
    pub feature_bounds: &'a [zenpredict::FeatureBound],
    pub metadata: &'a [BakeMetadataEntry<'a>],
    /// Per-output specs. Length must equal `n_outputs`. Pass an
    /// empty slice to omit the section (raw passthrough on decode).
    pub output_specs: &'a [OutputSpec],
    /// Pool of f32 values referenced by `OutputSpec::{discrete_set_offset,
    /// discrete_set_len}`. Each spec slices an inclusive set out
    /// of this pool. Pass empty if no spec snaps to a discrete set.
    pub discrete_sets: &'a [f32],
    /// Sparse hand-tune overrides applied AFTER the per-output spec
    /// pipeline. Each entry's `idx` must be `< n_outputs`.
    pub sparse_overrides: &'a [SparseOverride],
    /// Optional input-feature permutation: `feature_order[bake_pos] =
    /// caller_idx`. When `Some`, the composer reorders all
    /// input-indexed data (scaler_mean, scaler_scale, feature_bounds,
    /// `layer[0]` weight rows) into bake (permuted) order at write
    /// time, and stores the permutation in the `feature_order`
    /// header section so the loader can apply the inverse at load.
    /// Index width on disk is auto-sized to the smallest u8/u16/u32
    /// that fits `n_inputs`. None = identity (no reorder).
    pub feature_order: Option<&'a [u32]>,
    /// Optional output permutation: `output_order[bake_pos] =
    /// caller_idx`. Symmetric to `feature_order`. Affects
    /// `layer[last]` weight cols + biases (and I8 scales), output_specs,
    /// sparse_overrides indices. Metadata entries (cell_rescue_hints,
    /// output_bounds) are NOT permuted — the trainer's contract is
    /// to emit metadata in caller-natural order regardless.
    pub output_order: Option<&'a [u32]>,
    /// When true, the composer wraps bytes `[128..end]` in LZ4 block
    /// compression and sets three related header fields: the
    /// `flags.compressed` bit, the algo nibble, and
    /// `decompressed_payload_len`. The loader transparently
    /// decompresses at load time. Default false.
    pub compressed: bool,
    /// Optional caller-supplied hidden-unit permutations, one entry
    /// per interior dimension (length = `layers.len() - 1`). Each
    /// entry is a permutation of `0..interior_dim`. When `None`,
    /// the composer applies the default L2-norm-ascending HU
    /// reorder per the `hu_reorder` module. When `Some`, the
    /// composer uses these permutations verbatim — useful for the
    /// auto-optimizer (`bake_optimized`) which tries several
    /// candidate HU layouts. Since HU labels are arbitrary internal
    /// nodes, no metadata is emitted to the wire format; the
    /// resulting bake is byte-shape-identical to a default-HU one
    /// modulo the weight content.
    pub hu_permutations: Option<&'a [&'a [u32]]>,
}

impl<'a> BakeRequest<'a> {
    /// Build a bake request with the v3 mandatory fields set and
    /// every optional section empty (no feature bounds, no metadata,
    /// no output specs, no discrete sets, no sparse overrides). Set
    /// the optional fields by direct assignment after construction —
    /// the struct is `#[non_exhaustive]`, which blocks struct-literal
    /// construction outside this crate, but field assignment on a
    /// `mut` binding works.
    ///
    /// Prefer [`Self::builder`] when chaining several optional
    /// sections; it returns a fluent [`BakeRequestBuilder`] with one
    /// setter per optional field.
    pub fn new(
        schema_hash: u64,
        flags: u16,
        scaler_mean: &'a [f32],
        scaler_scale: &'a [f32],
        layers: &'a [BakeLayer<'a>],
    ) -> Self {
        Self {
            schema_hash,
            flags,
            scaler_mean,
            scaler_scale,
            layers,
            feature_bounds: &[],
            metadata: &[],
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        }
    }

    /// Start a fluent builder. The five required arguments mirror
    /// [`Self::new`]; chain `.feature_bounds(...)`, `.metadata(...)`,
    /// `.output_specs(...)`, `.discrete_sets(...)`,
    /// `.sparse_overrides(...)` to populate optional sections, then
    /// `.build()` to return the `BakeRequest` (or pass the builder
    /// directly to [`bake`] via `.bake()`).
    ///
    /// ```ignore
    /// let bytes = BakeRequest::builder(0, 0, &mean, &scale, &layers)
    ///     .metadata(&entries)
    ///     .output_specs(&specs)
    ///     .discrete_sets(&pool)
    ///     .bake()?;
    /// ```
    pub fn builder(
        schema_hash: u64,
        flags: u16,
        scaler_mean: &'a [f32],
        scaler_scale: &'a [f32],
        layers: &'a [BakeLayer<'a>],
    ) -> BakeRequestBuilder<'a> {
        BakeRequestBuilder {
            inner: Self::new(schema_hash, flags, scaler_mean, scaler_scale, layers),
        }
    }
}

/// Fluent builder for [`BakeRequest`]. Construct via
/// [`BakeRequest::builder`]; finalize with [`Self::build`] or
/// [`Self::bake`].
pub struct BakeRequestBuilder<'a> {
    inner: BakeRequest<'a>,
}

impl<'a> BakeRequestBuilder<'a> {
    /// Per-input `[low, high]` pairs. Pass an empty slice (the
    /// default) to omit the section.
    pub fn feature_bounds(mut self, bounds: &'a [zenpredict::FeatureBound]) -> Self {
        self.inner.feature_bounds = bounds;
        self
    }

    /// Typed-TLV metadata entries.
    pub fn metadata(mut self, entries: &'a [BakeMetadataEntry<'a>]) -> Self {
        self.inner.metadata = entries;
        self
    }

    /// Per-output `OutputSpec` table. Length must equal `n_outputs`
    /// (validated at bake time).
    pub fn output_specs(mut self, specs: &'a [OutputSpec]) -> Self {
        self.inner.output_specs = specs;
        self
    }

    /// f32 pool referenced by `OutputSpec::{discrete_set_offset,
    /// discrete_set_len}`.
    pub fn discrete_sets(mut self, pool: &'a [f32]) -> Self {
        self.inner.discrete_sets = pool;
        self
    }

    /// Sparse `(idx, value)` overrides applied after the per-output
    /// spec pipeline.
    pub fn sparse_overrides(mut self, overrides: &'a [SparseOverride]) -> Self {
        self.inner.sparse_overrides = overrides;
        self
    }

    /// Input-feature permutation: `perm[bake_pos] = caller_idx`. The
    /// composer reorders all input-indexed data into bake order and
    /// writes the permutation as the `feature_order` header section.
    /// Length must equal `n_inputs`.
    pub fn feature_order(mut self, perm: &'a [u32]) -> Self {
        self.inner.feature_order = Some(perm);
        self
    }

    /// Output permutation: `perm[bake_pos] = caller_idx`. The composer
    /// reorders all output-indexed data into bake order. Length must
    /// equal `n_outputs`.
    pub fn output_order(mut self, perm: &'a [u32]) -> Self {
        self.inner.output_order = Some(perm);
        self
    }

    /// Enable bake-level LZ4 compression. Bytes [128..end] are LZ4-
    /// block-compressed at write time; the loader transparently
    /// decompresses at load. ~58 % shrink on HU-reordered V0_18.
    pub fn compressed(mut self, enabled: bool) -> Self {
        self.inner.compressed = enabled;
        self
    }

    /// Finalize the builder and return the underlying `BakeRequest`.
    pub fn build(self) -> BakeRequest<'a> {
        self.inner
    }

    /// Convenience: finalize and bake in one call. Equivalent to
    /// `bake(&builder.build())`.
    pub fn bake(self) -> Result<alloc::vec::Vec<u8>, BakeError> {
        bake(&self.inner)
    }
}

use zenpredict::wire::{
    HEADER_SIZE, LAYER_ENTRY_SIZE, SECTION_OFF_DISCRETE_SETS, SECTION_OFF_FEATURE_BOUNDS,
    SECTION_OFF_LAYER_TABLE, SECTION_OFF_METADATA, SECTION_OFF_OUTPUT_SPECS,
    SECTION_OFF_SCALER_MEAN, SECTION_OFF_SCALER_SCALE, SECTION_OFF_SPARSE_OVERRIDES,
};

/// Compose a v3 ZNPR byte stream. Output round-trips through
/// [`Model::from_bytes`](zenpredict::Model::from_bytes).
pub fn bake(req: &BakeRequest<'_>) -> Result<alloc::vec::Vec<u8>, BakeError> {
    if req.layers.is_empty() {
        return Err(BakeError::EmptyLayers);
    }
    let n_inputs = req.layers[0].in_dim;
    let n_outputs = req.layers.last().unwrap().out_dim;
    let n_layers = req.layers.len();

    if req.scaler_mean.len() != n_inputs {
        return Err(BakeError::ScalerLengthMismatch {
            what: "scaler_mean",
            expected: n_inputs,
            got: req.scaler_mean.len(),
        });
    }
    if req.scaler_scale.len() != n_inputs {
        return Err(BakeError::ScalerLengthMismatch {
            what: "scaler_scale",
            expected: n_inputs,
            got: req.scaler_scale.len(),
        });
    }
    if !req.feature_bounds.is_empty() && req.feature_bounds.len() != n_inputs {
        return Err(BakeError::FeatureBoundsLengthMismatch {
            expected: n_inputs,
            got: req.feature_bounds.len(),
        });
    }

    if !req.output_specs.is_empty() && req.output_specs.len() != n_outputs {
        return Err(BakeError::OutputSpecsLengthMismatch {
            expected: n_outputs,
            got: req.output_specs.len(),
        });
    }
    for (output_index, spec) in req.output_specs.iter().enumerate() {
        if OutputTransform::from_byte(spec.transform).is_none() {
            return Err(BakeError::UnknownOutputTransform {
                output_index,
                byte: spec.transform,
            });
        }
        if spec.discrete_set_len > 0 {
            let off = spec.discrete_set_offset as usize;
            let len = spec.discrete_set_len as usize;
            let end = off
                .checked_add(len)
                .ok_or(BakeError::OutputSpecDiscreteOutOfRange {
                    output_index,
                    offset: spec.discrete_set_offset,
                    len: spec.discrete_set_len,
                    pool_len: req.discrete_sets.len(),
                })?;
            if end > req.discrete_sets.len() {
                return Err(BakeError::OutputSpecDiscreteOutOfRange {
                    output_index,
                    offset: spec.discrete_set_offset,
                    len: spec.discrete_set_len,
                    pool_len: req.discrete_sets.len(),
                });
            }
        }
    }
    for entry in req.sparse_overrides {
        if (entry.idx as usize) >= n_outputs {
            return Err(BakeError::SparseOverrideIndexOutOfRange {
                idx: entry.idx,
                n_outputs,
            });
        }
    }

    // Validate layer chain.
    let mut prev_out = n_inputs;
    for (i, layer) in req.layers.iter().enumerate() {
        if layer.in_dim != prev_out {
            return Err(BakeError::LayerDimMismatch {
                layer: i,
                expected_in: prev_out,
                got_in: layer.in_dim,
            });
        }
        let expect_w = layer.in_dim * layer.out_dim;
        if layer.weights.len() != expect_w {
            return Err(BakeError::WeightLengthMismatch {
                layer: i,
                expected: expect_w,
                got: layer.weights.len(),
            });
        }
        if layer.biases.len() != layer.out_dim {
            return Err(BakeError::BiasLengthMismatch {
                layer: i,
                expected: layer.out_dim,
                got: layer.biases.len(),
            });
        }
        prev_out = layer.out_dim;
    }
    let _ = n_outputs;

    // Hidden-unit reorder: always-on, sorts each interior dim's
    // hidden units by L2 norm ascending. Mathematically identical
    // to the un-permuted bake (HU labels are arbitrary internal
    // nodes) but row-major LZ4/zstd find longer zero runs when
    // dead units cluster contiguously. Saves ~58 % on layer-0
    // weights for V0_18-shape bakes (228 → 384 → 1, 74 % dead HUs).
    // No-op on fully-live matrices; never regresses size.
    //
    // See `hu_reorder` module for measured numbers and references.
    let permuted_owned = crate::hu_reorder::apply_hu_reorder(req.layers, req.hu_permutations);
    let permuted_layers: alloc::vec::Vec<BakeLayer<'_>> = permuted_owned
        .iter()
        .map(crate::hu_reorder::OwnedBakeLayer::as_borrowed)
        .collect();
    let layers: &[BakeLayer<'_>] = &permuted_layers;

    // Validate metadata keys up front.
    for entry in req.metadata {
        if entry.key.is_empty() {
            return Err(BakeError::MetadataKeyEmpty);
        }
        if entry.key.len() > 255 {
            return Err(BakeError::MetadataKeyTooLong {
                len: entry.key.len(),
            });
        }
    }

    // Validate `zentrain.feature_transforms` + `feature_transform_params`
    // pair (if present). Catches misspelled tokens and per-variant
    // arity / domain errors at bake time so they never ship in a
    // wire bake. See `validate_feature_transforms` for the full set
    // of checks.
    validate_feature_transforms(req.metadata, n_inputs)?;

    // Layout: Header at 0, LayerEntry table at HEADER_SIZE, then
    // aligned data sections in this order:
    //   scaler_mean, scaler_scale,
    //   for each layer: weights, scales (if I8), biases,
    //   feature_bounds (if non-empty),
    //   metadata blob (if any entries).
    let mut buf = alloc::vec::Vec::with_capacity(HEADER_SIZE + n_layers * LAYER_ENTRY_SIZE + 4096);

    // Reserve header + layer table; we'll patch offsets after data
    // is written.
    buf.resize(HEADER_SIZE + n_layers * LAYER_ENTRY_SIZE, 0);

    // Magic + version + flags + dims + schema_hash.
    buf[0..4].copy_from_slice(b"ZNPR");
    buf[4..6].copy_from_slice(&zenpredict::FORMAT_VERSION.to_le_bytes());
    buf[6..8].copy_from_slice(&req.flags.to_le_bytes());
    buf[8..12].copy_from_slice(&(n_inputs as u32).to_le_bytes());
    buf[12..16].copy_from_slice(&(n_outputs as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(n_layers as u32).to_le_bytes());
    // [20..24] reserved padding for u64 alignment, already zeroed.
    buf[24..32].copy_from_slice(&req.schema_hash.to_le_bytes());

    // layer_table Section is fixed: starts at HEADER_SIZE, length =
    // n_layers * LAYER_ENTRY_SIZE.
    write_section(
        &mut buf,
        SECTION_OFF_LAYER_TABLE,
        Section::new(HEADER_SIZE as u32, (n_layers * LAYER_ENTRY_SIZE) as u32),
    );

    // Scaler sections.
    pad_to(&mut buf, 4);
    let scaler_mean_section = append_f32(&mut buf, req.scaler_mean);
    write_section(&mut buf, SECTION_OFF_SCALER_MEAN, scaler_mean_section);

    pad_to(&mut buf, 4);
    let scaler_scale_section = append_f32(&mut buf, req.scaler_scale);
    write_section(&mut buf, SECTION_OFF_SCALER_SCALE, scaler_scale_section);

    // Per-layer payloads — iterate the post-HU-reorder permuted
    // layers, NOT `req.layers` (the latter holds caller-order input;
    // the bake's on-disk layout is the L2-asc-sorted permutation).
    for (i, layer) in layers.iter().enumerate() {
        let layer_entry_off = HEADER_SIZE + i * LAYER_ENTRY_SIZE;

        let weights_section = match layer.dtype {
            WeightDtype::F32 => {
                pad_to(&mut buf, 4);
                append_f32(&mut buf, layer.weights)
            }
            WeightDtype::F16 => {
                pad_to(&mut buf, 2);
                let start = buf.len() as u32;
                for &w in layer.weights {
                    let bits = f32_to_f16_bits(w);
                    buf.extend_from_slice(&bits.to_le_bytes());
                }
                Section::new(start, (layer.weights.len() * 2) as u32)
            }
            WeightDtype::I8 => {
                let start = buf.len() as u32;
                let scales = compute_i8_scales_per_output(layer);
                for (idx, &w) in layer.weights.iter().enumerate() {
                    let o = idx % layer.out_dim;
                    let q = if scales[o] == 0.0 {
                        0
                    } else {
                        (w / scales[o]).round().clamp(-128.0, 127.0) as i8
                    };
                    buf.push(q as u8);
                }
                Section::new(start, layer.weights.len() as u32)
            }
        };

        let scales_section = match layer.dtype {
            WeightDtype::I8 => {
                pad_to(&mut buf, 4);
                append_f32(&mut buf, &compute_i8_scales_per_output(layer))
            }
            _ => Section::empty(),
        };

        pad_to(&mut buf, 4);
        let biases_section = append_f32(&mut buf, layer.biases);

        // LayerEntry header (in-place patch).
        let entry = &mut buf[layer_entry_off..layer_entry_off + LAYER_ENTRY_SIZE];
        entry[0..4].copy_from_slice(&(layer.in_dim as u32).to_le_bytes());
        entry[4..8].copy_from_slice(&(layer.out_dim as u32).to_le_bytes());
        entry[8] = layer.activation as u8;
        entry[9] = layer.dtype as u8;
        // [10..12] flags, zero.
        write_section_inline(entry, 12, weights_section);
        write_section_inline(entry, 20, scales_section);
        write_section_inline(entry, 28, biases_section);
        // [36..48] reserved, zero.
    }

    // Feature bounds (optional).
    let feature_bounds_section = if req.feature_bounds.is_empty() {
        Section::empty()
    } else {
        pad_to(&mut buf, 4);
        let start = buf.len() as u32;
        for fb in req.feature_bounds {
            buf.extend_from_slice(&fb.low.to_le_bytes());
            buf.extend_from_slice(&fb.high.to_le_bytes());
        }
        Section::new(start, (req.feature_bounds.len() * 8) as u32)
    };
    write_section(&mut buf, SECTION_OFF_FEATURE_BOUNDS, feature_bounds_section);

    // Metadata blob (optional).
    let metadata_section = if req.metadata.is_empty() {
        Section::empty()
    } else {
        let start = buf.len() as u32;
        for entry in req.metadata {
            // [1] key_len
            buf.push(entry.key.len() as u8);
            // [...] key bytes
            buf.extend_from_slice(entry.key.as_bytes());
            // [1] value_type
            let type_byte = match entry.kind {
                MetadataType::Bytes => 0u8,
                MetadataType::Utf8 => 1,
                MetadataType::Numeric => 2,
                MetadataType::Reserved(b) => b,
            };
            buf.push(type_byte);
            // [4] value_len LE
            buf.extend_from_slice(&(entry.value.len() as u32).to_le_bytes());
            // [...] value
            buf.extend_from_slice(entry.value);
        }
        Section::new(start, (buf.len() as u32) - start)
    };
    write_section(&mut buf, SECTION_OFF_METADATA, metadata_section);

    // Output specs (optional). 32 bytes per entry, must align to 4
    // (largest f32 field).
    let output_specs_section = if req.output_specs.is_empty() {
        Section::empty()
    } else {
        pad_to(&mut buf, 4);
        let start = buf.len() as u32;
        let bytes: &[u8] = bytemuck::cast_slice(req.output_specs);
        buf.extend_from_slice(bytes);
        Section::new(start, bytes.len() as u32)
    };
    write_section(&mut buf, SECTION_OFF_OUTPUT_SPECS, output_specs_section);

    // Discrete-sets pool (optional). f32 array.
    let discrete_sets_section = if req.discrete_sets.is_empty() {
        Section::empty()
    } else {
        pad_to(&mut buf, 4);
        append_f32(&mut buf, req.discrete_sets)
    };
    write_section(&mut buf, SECTION_OFF_DISCRETE_SETS, discrete_sets_section);

    // Sparse overrides (optional). 8 bytes per entry; align to 4.
    let sparse_overrides_section = if req.sparse_overrides.is_empty() {
        Section::empty()
    } else {
        pad_to(&mut buf, 4);
        let start = buf.len() as u32;
        let bytes: &[u8] = bytemuck::cast_slice(req.sparse_overrides);
        buf.extend_from_slice(bytes);
        Section::new(start, bytes.len() as u32)
    };
    write_section(
        &mut buf,
        SECTION_OFF_SPARSE_OVERRIDES,
        sparse_overrides_section,
    );

    // ─── Forward permutation + permutation-table emission ──────────
    //
    // The bake's on-disk layout SHOULD be in compression-optimal
    // (bake-natural) order. We just wrote everything in caller order
    // — now we apply forward permutations in-place AND emit the
    // permutation tables so the loader can invert at load time.
    if let Some(perm) = req.feature_order {
        if perm.len() != n_inputs {
            return Err(BakeError::FeatureOrderLengthMismatch {
                expected: n_inputs,
                got: perm.len(),
            });
        }
        validate_permutation(perm, n_inputs, "feature_order")?;
        // Read layer[0]'s weight section from the layer table BEFORE
        // we mut-borrow buf for the permutation pass.
        let layer0_weights = read_section_inline(&buf, HEADER_SIZE, 12);
        forward_permute_inputs(
            &mut buf,
            perm,
            scaler_mean_section,
            scaler_scale_section,
            feature_bounds_section,
            layer0_weights,
            layers[0].out_dim,
            layers[0].dtype,
        );
        // Emit feature_order section with auto-sized index width.
        pad_to(&mut buf, 4);
        let start = buf.len() as u32;
        let written = write_permutation_indices(&mut buf, perm);
        write_section(
            &mut buf,
            zenpredict::wire::SECTION_OFF_FEATURE_ORDER,
            Section::new(start, written as u32),
        );
    }
    if let Some(perm) = req.output_order {
        if perm.len() != n_outputs {
            return Err(BakeError::OutputOrderLengthMismatch {
                expected: n_outputs,
                got: perm.len(),
            });
        }
        validate_permutation(perm, n_outputs, "output_order")?;
        let last_idx = n_layers - 1;
        let last_layer = &layers[last_idx];
        let last_entry_off = HEADER_SIZE + last_idx * LAYER_ENTRY_SIZE;
        let last_weights = read_section_inline(&buf, last_entry_off, 12);
        let last_scales = read_section_inline(&buf, last_entry_off, 20);
        let last_biases = read_section_inline(&buf, last_entry_off, 28);
        forward_permute_outputs(
            &mut buf,
            perm,
            last_weights,
            last_scales,
            last_biases,
            last_layer.in_dim,
            last_layer.out_dim,
            last_layer.dtype,
            output_specs_section,
            sparse_overrides_section,
        );
        pad_to(&mut buf, 4);
        let start = buf.len() as u32;
        let written = write_permutation_indices(&mut buf, perm);
        write_section(
            &mut buf,
            zenpredict::wire::SECTION_OFF_OUTPUT_ORDER,
            Section::new(start, written as u32),
        );
    }

    // ─── Whole-bake LZ4 compression (optional) ─────────────────────
    //
    // Compress bytes [128..end] in place. Set the header's
    // compressed flag, algo nibble, and decompressed_payload_len.
    if req.compressed {
        let payload_len = (buf.len() - HEADER_SIZE) as u32;
        let compressed = lz4_flex::block::compress(&buf[HEADER_SIZE..]);
        // Replace bytes [HEADER_SIZE..] with the compressed blob.
        buf.truncate(HEADER_SIZE);
        buf.extend_from_slice(&compressed);
        // Set flags: bit 0 (compressed) + algo nibble (LZ4 = 1).
        let mut flags = u16::from_le_bytes([buf[6], buf[7]]);
        flags |= zenpredict::wire::FLAG_COMPRESSED;
        // Clear algo nibble first, then set LZ4 (1 << 1 == 0x02).
        flags &= !zenpredict::wire::FLAGS_COMPRESSION_ALGO_MASK;
        flags |= (zenpredict::wire::COMPRESSION_ALGO_LZ4 as u16) << 1;
        buf[6..8].copy_from_slice(&flags.to_le_bytes());
        // Write decompressed_payload_len.
        buf[zenpredict::wire::OFF_DECOMPRESSED_PAYLOAD_LEN
            ..zenpredict::wire::OFF_DECOMPRESSED_PAYLOAD_LEN + 4]
            .copy_from_slice(&payload_len.to_le_bytes());
    }

    Ok(buf)
}

// ───── Permutation + compression helpers ─────────────────────────

fn read_section_inline(buf: &[u8], entry_off: usize, field_off: usize) -> Section {
    let off = u32::from_le_bytes(
        buf[entry_off + field_off..entry_off + field_off + 4]
            .try_into()
            .unwrap(),
    );
    let len = u32::from_le_bytes(
        buf[entry_off + field_off + 4..entry_off + field_off + 8]
            .try_into()
            .unwrap(),
    );
    Section::new(off, len)
}

fn validate_permutation(perm: &[u32], n: usize, what: &'static str) -> Result<(), BakeError> {
    let mut seen = alloc::vec![false; n];
    for &v in perm {
        let i = v as usize;
        if i >= n || seen[i] {
            return Err(BakeError::InvalidPermutation { what });
        }
        seen[i] = true;
    }
    Ok(())
}

/// Emit a permutation array with auto-sized width (u8/u16/u32) chosen
/// to minimize on-disk size while addressing all indices.
fn write_permutation_indices(buf: &mut alloc::vec::Vec<u8>, perm: &[u32]) -> usize {
    let max_val = perm.iter().copied().max().unwrap_or(0);
    if max_val <= 255 {
        for &v in perm {
            buf.push(v as u8);
        }
        perm.len()
    } else if max_val <= 65_535 {
        for &v in perm {
            buf.extend_from_slice(&(v as u16).to_le_bytes());
        }
        perm.len() * 2
    } else {
        for &v in perm {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        perm.len() * 4
    }
}

/// Forward permutation: `bake_data[bake_pos] = caller_data[perm[bake_pos]]`.
#[allow(clippy::too_many_arguments)] // internal helper; refactoring to a
// struct adds plumbing without clarity
fn forward_permute_inputs(
    buf: &mut [u8],
    perm: &[u32],
    scaler_mean_section: Section,
    scaler_scale_section: Section,
    feature_bounds_section: Section,
    layer0_weights_section: Section,
    layer0_out_dim: usize,
    layer0_dtype: WeightDtype,
) {
    forward_permute_f32(buf, scaler_mean_section, perm);
    forward_permute_f32(buf, scaler_scale_section, perm);
    if !feature_bounds_section.is_empty() {
        forward_permute_pod(buf, feature_bounds_section, perm, 8);
    }
    let elem_bytes = match layer0_dtype {
        WeightDtype::F32 => 4,
        WeightDtype::F16 => 2,
        WeightDtype::I8 => 1,
    };
    let row_bytes = layer0_out_dim * elem_bytes;
    forward_permute_rows(buf, layer0_weights_section, perm, row_bytes);
}

#[allow(clippy::too_many_arguments)] // internal helper; refactoring to a
// struct adds plumbing without clarity
fn forward_permute_outputs(
    buf: &mut [u8],
    perm: &[u32],
    last_weights: Section,
    last_scales: Section,
    last_biases: Section,
    last_in_dim: usize,
    last_out_dim: usize,
    last_dtype: WeightDtype,
    output_specs_section: Section,
    sparse_overrides_section: Section,
) {
    let elem_bytes = match last_dtype {
        WeightDtype::F32 => 4,
        WeightDtype::F16 => 2,
        WeightDtype::I8 => 1,
    };
    // Permute COLS of last-layer weights.
    forward_permute_cols(
        buf,
        last_weights,
        perm,
        last_in_dim,
        last_out_dim,
        elem_bytes,
    );
    forward_permute_f32(buf, last_biases, perm);
    if !last_scales.is_empty() {
        forward_permute_f32(buf, last_scales, perm);
    }
    if !output_specs_section.is_empty() {
        let spec_size = core::mem::size_of::<OutputSpec>();
        forward_permute_pod(buf, output_specs_section, perm, spec_size);
    }
    if !sparse_overrides_section.is_empty() {
        // Remap idx of each entry: idx = inverse_perm[old_idx].
        // For sparse_overrides, the caller wrote entries with idx
        // pointing into CALLER-natural output positions. After bake
        // reorder, idx should point into BAKE positions, so we need
        // the inverse mapping: caller_idx → bake_pos.
        let mut inv = alloc::vec![0u32; perm.len()];
        for (bake_pos, &caller_idx) in perm.iter().enumerate() {
            inv[caller_idx as usize] = bake_pos as u32;
        }
        forward_remap_sparse_indices(buf, sparse_overrides_section, &inv);
    }
}

fn forward_permute_f32(buf: &mut [u8], section: Section, perm: &[u32]) {
    forward_permute_pod(buf, section, perm, 4);
}

fn forward_permute_pod(buf: &mut [u8], section: Section, perm: &[u32], elem_bytes: usize) {
    let off = section.offset() as usize;
    let n = perm.len();
    let mut new_bytes = alloc::vec![0u8; n * elem_bytes];
    for bake_pos in 0..n {
        let caller_idx = perm[bake_pos] as usize;
        let src = &buf[off + caller_idx * elem_bytes..off + (caller_idx + 1) * elem_bytes];
        new_bytes[bake_pos * elem_bytes..(bake_pos + 1) * elem_bytes].copy_from_slice(src);
    }
    buf[off..off + n * elem_bytes].copy_from_slice(&new_bytes);
}

fn forward_permute_rows(buf: &mut [u8], section: Section, perm: &[u32], row_bytes: usize) {
    let off = section.offset() as usize;
    let n = perm.len();
    let mut new_bytes = alloc::vec![0u8; n * row_bytes];
    for bake_pos in 0..n {
        let caller_idx = perm[bake_pos] as usize;
        let src = &buf[off + caller_idx * row_bytes..off + (caller_idx + 1) * row_bytes];
        new_bytes[bake_pos * row_bytes..(bake_pos + 1) * row_bytes].copy_from_slice(src);
    }
    buf[off..off + n * row_bytes].copy_from_slice(&new_bytes);
}

fn forward_permute_cols(
    buf: &mut [u8],
    section: Section,
    perm: &[u32],
    n_rows: usize,
    n_cols: usize,
    elem_bytes: usize,
) {
    let off = section.offset() as usize;
    let row_stride = n_cols * elem_bytes;
    let total = n_rows * row_stride;
    let mut new_bytes = alloc::vec![0u8; total];
    for r in 0..n_rows {
        let row_off = r * row_stride;
        for bake_pos in 0..n_cols {
            let caller_idx = perm[bake_pos] as usize;
            let src = &buf[off + row_off + caller_idx * elem_bytes
                ..off + row_off + (caller_idx + 1) * elem_bytes];
            new_bytes[row_off + bake_pos * elem_bytes..row_off + (bake_pos + 1) * elem_bytes]
                .copy_from_slice(src);
        }
    }
    buf[off..off + total].copy_from_slice(&new_bytes);
}

fn forward_remap_sparse_indices(buf: &mut [u8], section: Section, inv: &[u32]) {
    let off = section.offset() as usize;
    let len = section.len_bytes() as usize;
    let entry_size = core::mem::size_of::<SparseOverride>();
    let region = &mut buf[off..off + len];
    for chunk in region.chunks_exact_mut(entry_size) {
        let old = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
        let new_idx = inv[old];
        chunk[0..4].copy_from_slice(&new_idx.to_le_bytes());
    }
}

fn pad_to(buf: &mut alloc::vec::Vec<u8>, alignment: usize) {
    let rem = buf.len() % alignment;
    if rem != 0 {
        let pad = alignment - rem;
        for _ in 0..pad {
            buf.push(0);
        }
    }
}

fn append_f32(buf: &mut alloc::vec::Vec<u8>, values: &[f32]) -> Section {
    let start = buf.len() as u32;
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Section::new(start, (values.len() * 4) as u32)
}

fn write_section(buf: &mut [u8], at: usize, s: Section) {
    buf[at..at + 4].copy_from_slice(&s.offset().to_le_bytes());
    buf[at + 4..at + 8].copy_from_slice(&s.len_bytes().to_le_bytes());
}

fn write_section_inline(entry: &mut [u8], at: usize, s: Section) {
    entry[at..at + 4].copy_from_slice(&s.offset().to_le_bytes());
    entry[at + 4..at + 8].copy_from_slice(&s.len_bytes().to_le_bytes());
}

/// Validate the optional `zentrain.feature_transforms` and
/// `zentrain.feature_transform_params` metadata entries.
///
/// Both keys are optional. When present, the lines (split on `\n`)
/// must equal `n_inputs`, and each parameterized variant's param row
/// must match its required arity. Domain checks reject parameters
/// that would produce `NaN` / `-Inf` at runtime
/// (`WinsorThenLog` with `p1 ≤ 0`, `WinsorThenLog1p` with `p1 < -1`).
///
/// Returns `Ok(())` when neither key is present, when both are well
/// formed, or when only `feature_transforms` is present (every
/// transform is then non-parameterized — runtime fallback handles
/// missing params).
//
// `collapsible_match` would have us hoist each `if` into a match
// guard. Keeping them as `match arm + if` keeps the per-variant
// reason strings co-located with the rule that produces them, which
// outweighs the lint's terseness here.
#[allow(clippy::collapsible_match, clippy::collapsible_if)]
fn validate_feature_transforms(
    metadata: &[BakeMetadataEntry<'_>],
    n_inputs: usize,
) -> Result<(), BakeError> {
    use zenpredict::FeatureTransform;
    use zenpredict::keys;
    let transforms_entry = metadata.iter().find(|m| m.key == keys::FEATURE_TRANSFORMS);
    let params_entry = metadata
        .iter()
        .find(|m| m.key == keys::FEATURE_TRANSFORM_PARAMS);

    let Some(t_entry) = transforms_entry else {
        // No transforms declared → nothing to validate. Stray
        // `feature_transform_params` without `feature_transforms` is
        // legal at the wire level (runtime ignores it); accept it
        // here too rather than imposing a new constraint.
        return Ok(());
    };
    let Ok(t_text) = core::str::from_utf8(t_entry.value) else {
        // Non-UTF-8 transforms blob — caller error, but the existing
        // `MetadataKeyTooLong` etc. errors don't cover this. Skip
        // here; the runtime parser will reject at load time.
        return Ok(());
    };
    // Parse transforms (lines).
    let mut transforms: alloc::vec::Vec<FeatureTransform> = alloc::vec::Vec::new();
    for (i, tok) in t_text.split('\n').enumerate() {
        match FeatureTransform::from_token(tok) {
            Ok(v) => transforms.push(v),
            Err(_) => {
                return Err(BakeError::UnknownFeatureTransformToken { feature_index: i });
            }
        }
    }
    let has_expander = transforms.iter().any(|t| t.is_expander());

    // For scalar-only pipelines the wire invariant is
    // `transforms.len() == n_inputs` (n_inputs is the first layer's
    // in_dim, which equals the raw input count). For pipelines
    // containing an expander variant (today: only Sinusoidal),
    // transforms.len() is the **raw** input count and the
    // post-expansion sum must equal n_inputs. The expansion math
    // needs the params, so the expander-path check happens below
    // after we've parsed params.
    if !has_expander && transforms.len() != n_inputs {
        return Err(BakeError::FeatureTransformsLenMismatch {
            expected: n_inputs,
            got: transforms.len(),
        });
    }

    // If params are missing, parameterized variants will fall back
    // to their no-op behaviour at runtime. That's a soft caller
    // error, not a bake-time hard failure (matches the runtime's
    // `Identity` / plain-Log degradation). We do NOT reject here
    // so the existing "feature_transforms only" workflow keeps
    // baking — UNLESS an expander variant is declared, in which
    // case params are required (frequencies for Sinusoidal).
    let Some(p_entry) = params_entry else {
        if has_expander {
            return Err(BakeError::FeatureTransformsLenMismatch {
                expected: transforms.len(),
                got: 0,
            });
        }
        return Ok(());
    };
    let Ok(p_text) = core::str::from_utf8(p_entry.value) else {
        return Ok(());
    };
    let mut params: alloc::vec::Vec<alloc::vec::Vec<f32>> = alloc::vec::Vec::new();
    for line in p_text.split('\n') {
        let mut row: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
        if !line.is_empty() {
            for tok in line.split(',') {
                let v: f32 = match tok.trim().parse() {
                    Ok(v) => v,
                    Err(_) => return Ok(()), // runtime will reject
                };
                row.push(v);
            }
        }
        params.push(row);
    }
    if params.len() != transforms.len() {
        return Err(BakeError::FeatureTransformsLenMismatch {
            expected: transforms.len(),
            got: params.len(),
        });
    }

    // Expander-aware length check: post-expansion sum must equal
    // the first-layer in_dim (n_inputs).
    if has_expander {
        let expanded: usize = transforms
            .iter()
            .zip(params.iter())
            .map(|(t, p)| t.output_arity(p))
            .sum();
        if expanded != n_inputs {
            return Err(BakeError::FeatureTransformsLenMismatch {
                expected: n_inputs,
                got: expanded,
            });
        }
    }

    // Per-variant arity + domain checks.
    for (i, (&t, p)) in transforms.iter().zip(params.iter()).enumerate() {
        // QuantileBins has variable arity (N edges) — validate
        // separately from the fixed-arity variants.
        if matches!(t, FeatureTransform::QuantileBins) {
            for &edge in p {
                if !edge.is_finite() {
                    return Err(BakeError::FeatureTransformParamInvalid {
                        feature_index: i,
                        transform: t.as_token(),
                        reason: "all bin edges must be finite",
                    });
                }
            }
            continue;
        }
        let needed = required_param_arity(t);
        if let Some(expected) = needed {
            if p.len() != expected {
                return Err(BakeError::FeatureTransformParamArityMismatch {
                    feature_index: i,
                    transform: t.as_token(),
                    expected,
                    got: p.len(),
                });
            }
            // Domain checks for the variants whose math breaks on
            // out-of-domain inputs.
            match t {
                FeatureTransform::WinsorThenLog => {
                    // ln(p1) is undefined for p1 <= 0.
                    if p[0] <= 0.0 || !p[0].is_finite() {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "p1 must be > 0 (ln domain)",
                        });
                    }
                    if !p[1].is_finite() || p[1] < p[0] {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "p99 must be finite and >= p1",
                        });
                    }
                }
                FeatureTransform::WinsorThenLog1p => {
                    // log1p(1 + p1) requires p1 > -1 (else ln(<=0)).
                    if !p[0].is_finite() || p[0] <= -1.0 {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "p1 must be > -1 (log1p domain)",
                        });
                    }
                    if !p[1].is_finite() || p[1] < p[0] {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "p99 must be finite and >= p1",
                        });
                    }
                }
                FeatureTransform::WinsorThenSignedCbrt
                | FeatureTransform::SignedCbrtThenWinsor
                | FeatureTransform::WinsorP99 => {
                    if !p[0].is_finite() || !p[1].is_finite() || p[1] < p[0] {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "bounds must be finite and p99 >= p1",
                        });
                    }
                }
                FeatureTransform::ClipThenLog1pThenWinsor => {
                    // p = [eps, q1, q99]. eps >= 0 (negative noise floor
                    // would shift the inner output negative, which the
                    // max(0, .) clamp eats — caller probably meant 0).
                    // q1 <= q99, both finite. q1 < 0 is rejected too:
                    // the inner stage produces non-negative output, so
                    // a negative lower bound is dead code (and likely
                    // a transcription bug).
                    if !p[0].is_finite() || p[0] < 0.0 {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "eps must be finite and >= 0",
                        });
                    }
                    if !p[1].is_finite() || !p[2].is_finite() || p[2] < p[1] {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "q1, q99 must be finite and q99 >= q1",
                        });
                    }
                    if p[1] < 0.0 {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "q1 must be >= 0 (inner stage output is non-negative)",
                        });
                    }
                }
                FeatureTransform::ClipThenLog1p => {
                    // eps >= 0; non-finite eps would yield NaN inputs
                    // to log1p. Negative eps is legal mathematically
                    // (it lowers the noise floor below zero, harmless
                    // after the max(0, .) clamp), but reject anyway
                    // to surface a likely transcription bug.
                    if !p[0].is_finite() {
                        return Err(BakeError::FeatureTransformParamInvalid {
                            feature_index: i,
                            transform: t.as_token(),
                            reason: "eps must be finite",
                        });
                    }
                }
                _ => {}
            }
        }
        // Non-parameterized variants may carry an empty params row,
        // or no row at all — both are valid (runtime ignores params).
    }
    Ok(())
}

/// Per-variant required param count. `None` for variants whose
/// arity is either variable (`QuantileBins`, handled separately) or
/// who don't consume params at all (the non-parameterized variants).
/// Used by the bake-side validator.
fn required_param_arity(t: zenpredict::FeatureTransform) -> Option<usize> {
    use zenpredict::FeatureTransform;
    match t {
        FeatureTransform::ClipThenLog1p => Some(1),
        FeatureTransform::WinsorP99
        | FeatureTransform::WinsorThenLog
        | FeatureTransform::WinsorThenLog1p
        | FeatureTransform::WinsorThenSignedCbrt
        | FeatureTransform::SignedCbrtThenWinsor => Some(2),
        FeatureTransform::ClipThenLog1pThenWinsor => Some(3),
        _ => None,
    }
}

/// Per-output max-abs scale: `scales[o] = max_i |W[i, o]| / 127.0`.
fn compute_i8_scales_per_output(layer: &BakeLayer<'_>) -> alloc::vec::Vec<f32> {
    let mut scales = alloc::vec![0.0f32; layer.out_dim];
    for (idx, &w) in layer.weights.iter().enumerate() {
        let o = idx % layer.out_dim;
        let abs = w.abs();
        if abs > scales[o] {
            scales[o] = abs;
        }
    }
    for s in scales.iter_mut() {
        if *s == 0.0 {
            *s = 1.0;
        } else {
            *s /= 127.0;
        }
    }
    scales
}

/// IEEE-754 binary32 → binary16 bit conversion (round-to-nearest-even,
/// flush-to-zero subnormals on f32 side first via standard rules).
/// Pure integer bit math; mirrors what F16C / VCVTPS2PH does in
/// hardware. Matches `zenpredict::f16_bits_to_f32` round-trip
/// for representable values.
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        // Inf / NaN.
        let m16 = (mant >> 13) as u16;
        return sign | 0x7c00 | if mant != 0 && m16 == 0 { 1 } else { m16 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow → ±Inf.
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        // Subnormal or zero.
        if unbiased < -25 {
            // Underflow → ±0.
            return sign;
        }
        // Convert to subnormal: shift `1.mant` right by appropriate
        // amount.
        let mant_with_implicit = mant | 0x800000;
        let shift = (-14 - unbiased + 13) as u32; // total right shift to get into 10-bit range
        let half = 1u32 << (shift - 1);
        let rounded = (mant_with_implicit + half) >> shift;
        return sign | (rounded as u16);
    }
    // Normal.
    let exp16 = (unbiased + 15) as u16;
    let mant16_full = mant + (1 << 12); // round-bit
    // Banker's-ish rounding via carry into exponent if mantissa
    // overflows.
    let mant16 = (mant16_full >> 13) as u16 & 0x3ff;
    let carry = (mant16_full >> 13) >> 10;
    sign | ((exp16 + carry as u16) << 10) | mant16
}
