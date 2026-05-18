//! JSON input schema for [`bake`].
//!
//! Lets language-agnostic toolchains (the Python training pipeline at
//! `zenanalyze/zenpicker/tools/`, ad-hoc baking scripts, etc.) drive
//! a v3 bake without re-implementing the byte-level format. The
//! Python side dumps a `BakeRequestJson`, then shells out to the
//! `zenpredict-bake` binary which calls [`bake`] on the
//! deserialized request.
//!
//! ## Schema (top level)
//!
//! ```json
//! {
//!   "schema_hash": 18446744073709551615,        // u64
//!   "flags": 0,                                  // u16, optional (default 0)
//!   "scaler_mean":  [0.0, 0.0, ...],             // f32[n_inputs]
//!   "scaler_scale": [1.0, 1.0, ...],             // f32[n_inputs]
//!   "layers": [ /* BakeLayerJson, see below */ ],
//!   "feature_bounds": [ {"low": -1.0, "high": 1.0}, ... ],  // optional
//!   "metadata": [ /* MetadataEntryJson, see below */ ],     // optional
//!   "zerobias_tau": 0.005,                       // optional, default 0.0
//!   "compressed": true,                          // optional, default false
//!   "optimize": true                             // optional, default false
//! }
//! ```
//!
//! ### Bake-time compression knobs
//!
//! - `zerobias_tau` — per-layer zero threshold (`τ * max|W_layer|`)
//!   applied BEFORE i8/f16 quantization. `0.005` is the calibrated
//!   sweet spot from `zensim/benchmarks/zenpredict_rle_zerobias_eval_2026-05-13.md`
//!   (87.5 % i8 zero density, -0.0001 SROCC on V0_18). Default `0.0`.
//! - `compressed` — wrap post-header payload in LZ4 block compression;
//!   loader transparently decompresses. Pair with `zerobias_tau` to
//!   monetize the zeros. Default `false`.
//! - `optimize` — run [`bake_optimized`] (permutation + compressed-flag
//!   search + bounded hillclimb) instead of [`bake`]. ~1-2 s budget on
//!   V_X-shape models, mathematically identical predict output.
//!   Default `false`.
//!
//! ## `BakeLayerJson`
//!
//! ```json
//! {
//!   "in_dim": 5,
//!   "out_dim": 32,
//!   "activation": "leakyrelu",        // "identity" | "relu" | "leakyrelu"
//!   "dtype": "f16",                   // "f32" | "f16" | "i8"
//!   "weights": [...],                 // f32 row-major, in_dim * out_dim
//!   "biases":  [...]                  // f32 length out_dim
//! }
//! ```
//!
//! ## `MetadataEntryJson`
//!
//! ```json
//! { "key": "zentrain.bake_name",
//!   "type": "utf8",
//!   "text": "v2.1_full" }
//! ```
//!
//! ```json
//! { "key": "zentrain.calibration_metrics",
//!   "type": "numeric",
//!   "f32": [0.0233, 0.0512, 0.563] }
//! ```
//!
//! ```json
//! { "key": "zenjpeg.cell_config",
//!   "type": "bytes",
//!   "hex": "deadbeef" }
//! ```
//!
//! Three value-shape keys are accepted: `text` (UTF-8 string,
//! preferred for `type: "utf8"`), `f32` (f32 array, convenience for
//! `type: "numeric"`), and `hex` (lowercase hex string, accepted for
//! any type and used for `bytes` payloads). Picking the right
//! `type` is the caller's responsibility — the loader uses it as
//! the wire byte without further interpretation.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use serde::Deserialize;

use crate::composer::{
    BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, MultiCodecSchemaInput, PerCodecMapInput,
    bake,
};
use crate::optimize::bake_optimized;
use crate::zero_bias::apply_zero_bias_per_layer_in_place;
use zenpredict::{
    Activation, FeatureBound, MetadataType, OutputSpec, OutputTransform, SparseOverride,
    WeightDtype,
};

/// Errors specific to JSON-driven baking. Thin wrapper over
/// [`BakeError`] adding the JSON-side validation cases.
#[derive(Debug)]
pub enum BakeJsonError {
    /// Underlying bake failed.
    Bake(BakeError),
    /// Hex decode failed.
    BadHex(String),
    /// Metadata entry didn't carry exactly one of {text, f32, hex}.
    MetadataValueMissing {
        key: String,
    },
    MetadataValueAmbiguous {
        key: String,
    },
    /// `text` value used with non-`utf8` type, or `f32` with non-numeric.
    MetadataValueWrongType {
        key: String,
        wire: MetadataType,
        repr: &'static str,
    },
}

impl core::fmt::Display for BakeJsonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bake(e) => write!(f, "{e}"),
            Self::BadHex(s) => write!(f, "bake_json: invalid hex string: {s:?}"),
            Self::MetadataValueMissing { key } => write!(
                f,
                "bake_json: metadata entry {key:?} carried no value (need one of text/f32/hex)"
            ),
            Self::MetadataValueAmbiguous { key } => write!(
                f,
                "bake_json: metadata entry {key:?} carried multiple value reprs (text/f32/hex)"
            ),
            Self::MetadataValueWrongType { key, wire, repr } => write!(
                f,
                "bake_json: metadata entry {key:?} declared type={wire:?} but value uses {repr} repr"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BakeJsonError {}

impl From<BakeError> for BakeJsonError {
    fn from(e: BakeError) -> Self {
        Self::Bake(e)
    }
}

/// Bake input deserialized from the language-agnostic JSON envelope.
///
/// **`#[non_exhaustive]` since 0.1.1.** Adding fields to this struct
/// is non-breaking from the JSON side (every new field has
/// `#[serde(default)]`) and now also non-breaking on the Rust side —
/// callers cannot construct this via struct literal, so an
/// `#[serde(default)]` field added in a future version doesn't break
/// existing in-tree Rust code. Build one via
/// `serde_json::from_str(json)` / `serde_json::from_slice(bytes)` /
/// the `bake_from_json_str` convenience.
#[derive(Deserialize, Debug)]
#[non_exhaustive]
pub struct BakeRequestJson {
    pub schema_hash: u64,
    #[serde(default)]
    pub flags: u16,
    pub scaler_mean: Vec<f32>,
    pub scaler_scale: Vec<f32>,
    pub layers: Vec<BakeLayerJson>,
    #[serde(default)]
    pub feature_bounds: Vec<FeatureBoundJson>,
    #[serde(default)]
    pub metadata: Vec<MetadataEntryJson>,
    /// Per-output specs: bounds clamp, activation, snap-to-discrete,
    /// and optional sentinel. Length, when present, must equal
    /// `n_outputs` (i.e. the last layer's `out_dim`). Pass an empty
    /// array to omit.
    ///
    /// JSON shape:
    /// ```json
    /// [
    ///   {"bounds": [0, 100], "transform": "sigmoid_scaled", "params": [0, 100]},
    ///   {"bounds": [0, 7], "transform": "round", "discrete_set": [0,1,2,3,4,5,6,7], "sentinel": -1}
    /// ]
    /// ```
    #[serde(default)]
    pub output_specs: Vec<OutputSpecJson>,
    /// Sparse hand-tune overrides, applied AFTER the per-output spec
    /// pipeline. Each entry's `idx` must be `< n_outputs`.
    /// `"value": null` (or omitted) emits `f32::NAN`, which triggers
    /// `OutputValue::Default` at runtime.
    ///
    /// JSON shape:
    /// ```json
    /// [
    ///   {"idx": 3, "value": 0.0},
    ///   {"idx": 5, "value": null}
    /// ]
    /// ```
    #[serde(default)]
    pub sparse_overrides: Vec<SparseOverrideJson>,
    /// Multi-codec joint-picker schema. When present, emits the v3.2
    /// `multi_codec_schema` section so the runtime can dispatch
    /// [`zenpredict::Predictor::predict_multi_codec`]. Absent (or
    /// `null` in JSON) → single-codec bake.
    ///
    /// JSON shape:
    /// ```json
    /// {
    ///   "union_feat_count": 64,
    ///   "per_codec": [
    ///     {
    ///       "codec_name": "zenjpeg",
    ///       "union_slot_for_codec_feat": [0, 1, 4, ...],
    ///       "output_range": [0, 48],
    ///       "head_n_cells": 12,
    ///       "head_n_heads": 4
    ///     },
    ///     ...
    ///   ]
    /// }
    /// ```
    #[serde(default)]
    pub multi_codec_schema: Option<MultiCodecSchemaJson>,
    /// Optional pre-quantization per-layer zero-bias threshold. When
    /// `> 0.0`, weights whose magnitude is below `tau * max|W_layer|`
    /// are zeroed BEFORE the layer's declared `dtype` quantization
    /// runs. Per-layer (single threshold per layer) — matches the
    /// 2026-05-13 `zensim/benchmarks/zenpredict_rle_zerobias_eval_*.md`
    /// methodology and the `zenpredict repack --zerobias <τ>` CLI.
    ///
    /// Recommended value: `0.005` (87.5 % i8 zero density, SROCC cost
    /// within sampling noise on V0_18 / CID22). Pair with `compressed:
    /// true` to monetize the zeros; raw i8 streams alone are near
    /// incompressible.
    ///
    /// Default `0.0` (disabled — bake bytes match the legacy JSON
    /// behavior).
    #[serde(default)]
    pub zerobias_tau: f32,
    /// When true, wrap the post-header payload in LZ4 block
    /// compression at write time. Loader transparently decompresses
    /// at `Model::from_bytes`. Equivalent to setting
    /// `BakeRequest.compressed = true` in the Rust API or passing
    /// `zenpredict repack --compress` on a pre-baked `.bin`. Default
    /// `false`.
    #[serde(default)]
    pub compressed: bool,
    /// When true, run [`bake_optimized`] instead of [`bake`]: sweep
    /// candidate (`feature_order`, `output_order`, hidden-unit
    /// permutation, compressed-flag) combinations + a bounded
    /// pairwise-swap hillclimb, and return the smallest output. ~1-2
    /// seconds per bake on V_X-shape models; mathematically identical
    /// predict output to the un-optimized path (load-time permutation
    /// inverses + decompression are lossless). Default `false`.
    ///
    /// When `compressed` is also set, the optimizer evaluates both
    /// `compressed=true` and `compressed=false` variants and picks
    /// whichever produces fewer total bytes — set `compressed: true`
    /// only when you specifically want to force compression even if
    /// the uncompressed variant happens to be smaller.
    #[serde(default)]
    pub optimize: bool,
}

/// Multi-codec joint-picker schema, JSON-side.
#[derive(Deserialize, Debug, Clone)]
pub struct MultiCodecSchemaJson {
    /// `U` — number of union image features across all codecs.
    pub union_feat_count: u32,
    /// Per-codec entries. Index = codec id at runtime.
    pub per_codec: Vec<PerCodecMapJson>,
}

/// Per-codec map entry, JSON-side. Mirrors
/// [`crate::composer::PerCodecMapInput`] but owns its data so the
/// JSON layer doesn't pin lifetimes to the request.
#[derive(Deserialize, Debug, Clone)]
pub struct PerCodecMapJson {
    /// Stable codec name (`"zenjpeg"`, `"zenwebp"`, …).
    pub codec_name: String,
    /// For each of this codec's natural feat_cols, the union slot
    /// index `[0..union_feat_count)` to scatter that value into.
    pub union_slot_for_codec_feat: Vec<u32>,
    /// `[lo, hi]` — half-open range into the trunk's flat output
    /// vector that belongs to this codec.
    pub output_range: [u32; 2],
    /// Number of config cells in this codec's argmin space.
    pub head_n_cells: u32,
    /// Number of heads per cell (bytes + scalar heads).
    pub head_n_heads: u32,
}

#[derive(Deserialize, Debug)]
pub struct BakeLayerJson {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: ActivationJson,
    pub dtype: DtypeJson,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ActivationJson {
    Identity,
    Relu,
    LeakyRelu,
}

impl From<ActivationJson> for Activation {
    fn from(a: ActivationJson) -> Self {
        match a {
            ActivationJson::Identity => Activation::Identity,
            ActivationJson::Relu => Activation::Relu,
            ActivationJson::LeakyRelu => Activation::LeakyRelu,
        }
    }
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DtypeJson {
    F32,
    F16,
    I8,
}

impl From<DtypeJson> for WeightDtype {
    fn from(d: DtypeJson) -> Self {
        match d {
            DtypeJson::F32 => WeightDtype::F32,
            DtypeJson::F16 => WeightDtype::F16,
            DtypeJson::I8 => WeightDtype::I8,
        }
    }
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct FeatureBoundJson {
    pub low: f32,
    pub high: f32,
}

/// Per-output post-processing config, JSON-side.
///
/// All fields are optional. Missing `bounds` means
/// `[-inf, +inf]` (no clamp). Missing `transform` means
/// [`OutputTransform::Identity`]. Missing `discrete_set` means no
/// snap. `sentinel: null` (or omitted) means no sentinel match.
#[derive(Deserialize, Debug, Clone)]
pub struct OutputSpecJson {
    /// Inclusive `[low, high]` clamp. Two-element JSON array.
    #[serde(default)]
    pub bounds: Option<[f32; 2]>,
    /// One of `"identity"`, `"sigmoid"`, `"sigmoid_scaled"`, `"exp"`,
    /// `"round"`. Default `"identity"`.
    #[serde(default)]
    pub transform: Option<OutputTransformJson>,
    /// Two f32 parameters interpreted by the transform. For
    /// `sigmoid_scaled` this is `[low, high]`; for the others, unused.
    #[serde(default)]
    pub params: Option<[f32; 2]>,
    /// Snap to nearest value in this set. Empty / null = no snap.
    #[serde(default)]
    pub discrete_set: Option<Vec<f32>>,
    /// Output value that should surface as
    /// `zenpredict::OutputValue::Default` (re-exported when the
    /// `advanced` feature is on). `null` = no sentinel.
    #[serde(default)]
    pub sentinel: Option<f32>,
}

#[derive(Deserialize, Debug, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum OutputTransformJson {
    #[default]
    Identity,
    Sigmoid,
    SigmoidScaled,
    Exp,
    Round,
}

impl OutputTransformJson {
    pub(crate) fn to_byte(self) -> u8 {
        OutputTransform::from(self) as u8
    }
}

impl From<OutputTransformJson> for OutputTransform {
    fn from(t: OutputTransformJson) -> Self {
        match t {
            OutputTransformJson::Identity => OutputTransform::Identity,
            OutputTransformJson::Sigmoid => OutputTransform::Sigmoid,
            OutputTransformJson::SigmoidScaled => OutputTransform::SigmoidScaled,
            OutputTransformJson::Exp => OutputTransform::Exp,
            OutputTransformJson::Round => OutputTransform::Round,
        }
    }
}

/// Sparse hand-tune override, JSON-side. `value: null` (or omitted)
/// emits `f32::NAN`, which surfaces as
/// `zenpredict::OutputValue::Default` (re-exported when the
/// `advanced` feature is on) at runtime.
#[derive(Deserialize, Debug, Clone, Copy)]
pub struct SparseOverrideJson {
    pub idx: u32,
    #[serde(default)]
    pub value: Option<f32>,
}

impl From<FeatureBoundJson> for FeatureBound {
    fn from(b: FeatureBoundJson) -> Self {
        FeatureBound::new(b.low, b.high)
    }
}

#[derive(Deserialize, Debug)]
pub struct MetadataEntryJson {
    pub key: String,
    #[serde(rename = "type")]
    pub kind: MetadataKindJson,
    /// UTF-8 string. Preferred for `type: "utf8"`; rejected for
    /// `type: "bytes"` (use `hex` instead) and for `type: "numeric"`
    /// (use `f32` or `hex`).
    #[serde(default)]
    pub text: Option<String>,
    /// f32 array, encoded as little-endian f32 bytes when written.
    /// Convenience for `type: "numeric"` payloads (calibration
    /// metrics, reach rates, etc.).
    #[serde(rename = "f32", default)]
    pub f32_values: Option<Vec<f32>>,
    /// Lowercase hex string. Universal — works for any type. Used by
    /// the Python training pipeline for codec-private opaque payloads
    /// or for non-f32 numeric data (e.g., a single u8 profile flag,
    /// reach_zq_targets stored as u8 array).
    #[serde(default)]
    pub hex: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum MetadataKindJson {
    Bytes,
    Utf8,
    Numeric,
}

impl From<MetadataKindJson> for MetadataType {
    fn from(k: MetadataKindJson) -> Self {
        match k {
            MetadataKindJson::Bytes => MetadataType::Bytes,
            MetadataKindJson::Utf8 => MetadataType::Utf8,
            MetadataKindJson::Numeric => MetadataType::Numeric,
        }
    }
}

/// Decode a metadata entry into the byte payload that the baker
/// writes verbatim into the metadata blob's value field.
fn decode_metadata_value(entry: &MetadataEntryJson) -> Result<Vec<u8>, BakeJsonError> {
    let wire = MetadataType::from(entry.kind);
    let mut count = 0;
    if entry.text.is_some() {
        count += 1;
    }
    if entry.f32_values.is_some() {
        count += 1;
    }
    if entry.hex.is_some() {
        count += 1;
    }
    if count == 0 {
        return Err(BakeJsonError::MetadataValueMissing {
            key: entry.key.clone(),
        });
    }
    if count > 1 {
        return Err(BakeJsonError::MetadataValueAmbiguous {
            key: entry.key.clone(),
        });
    }

    if let Some(s) = &entry.text {
        if !matches!(wire, MetadataType::Utf8) {
            return Err(BakeJsonError::MetadataValueWrongType {
                key: entry.key.clone(),
                wire,
                repr: "text",
            });
        }
        return Ok(s.as_bytes().to_vec());
    }
    if let Some(values) = &entry.f32_values {
        if !matches!(wire, MetadataType::Numeric) {
            return Err(BakeJsonError::MetadataValueWrongType {
                key: entry.key.clone(),
                wire,
                repr: "f32",
            });
        }
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        return Ok(out);
    }
    let hex = entry.hex.as_deref().expect("count != 0 implies one branch");
    decode_hex(hex).ok_or_else(|| BakeJsonError::BadHex(hex.into()))
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Bake a `BakeRequestJson` into ZNPR v3 bytes. Performs all input
/// validation that [`bake`] does plus the JSON-side type-vs-repr
/// checks for metadata entries.
///
/// Honors the three optional bake-time knobs on `BakeRequestJson`:
/// `zerobias_tau` (pre-quant per-layer thresholding), `compressed`
/// (LZ4 payload wrap), and `optimize` (run [`bake_optimized`] to
/// search permutation + compression candidates). All three default
/// to off / 0.0 / false, so existing JSON callers see no behavior
/// change.
pub fn bake_from_json(req: &BakeRequestJson) -> Result<Vec<u8>, BakeJsonError> {
    // Decode metadata values up front so the byte buffers outlive
    // the borrow into BakeMetadataEntry.
    let decoded_values: Vec<Vec<u8>> = req
        .metadata
        .iter()
        .map(decode_metadata_value)
        .collect::<Result<_, _>>()?;

    // Apply per-layer zero-bias when requested. We own the weight
    // vectors only when zerobias is active; otherwise borrow from
    // `req` directly to avoid the clone on the no-op path. Both
    // branches yield `&[f32]` slices that outlive the BakeLayer set.
    let zerobiased_weights: Option<Vec<Vec<f32>>> = if req.zerobias_tau > 0.0 {
        Some(
            req.layers
                .iter()
                .map(|l| {
                    let mut w = l.weights.clone();
                    apply_zero_bias_per_layer_in_place(&mut w, req.zerobias_tau);
                    w
                })
                .collect(),
        )
    } else {
        None
    };

    // Convert layers (owned Vec<f32> → borrowed slices via reuse).
    let layers: Vec<BakeLayer<'_>> = req
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| BakeLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation.into(),
            dtype: l.dtype.into(),
            weights: zerobiased_weights
                .as_ref()
                .map(|v| v[i].as_slice())
                .unwrap_or(&l.weights),
            biases: &l.biases,
        })
        .collect();

    let feature_bounds: Vec<FeatureBound> = req
        .feature_bounds
        .iter()
        .copied()
        .map(FeatureBound::from)
        .collect();

    let metadata: Vec<BakeMetadataEntry<'_>> = req
        .metadata
        .iter()
        .zip(decoded_values.iter())
        .map(|(entry, bytes)| BakeMetadataEntry {
            key: &entry.key,
            kind: entry.kind.into(),
            value: bytes,
        })
        .collect();

    // Build the v3 OutputSpec table + flat discrete-sets pool from
    // the JSON. The pool is grown as each spec's discrete set is
    // appended; specs reference (offset, len) into the pool.
    let mut discrete_sets_pool: Vec<f32> = Vec::new();
    let output_specs: Vec<OutputSpec> = req
        .output_specs
        .iter()
        .map(|s| {
            let (off, len) = if let Some(values) = &s.discrete_set {
                let off = discrete_sets_pool.len() as u32;
                discrete_sets_pool.extend_from_slice(values);
                (off, values.len() as u32)
            } else {
                (0, 0)
            };
            OutputSpec {
                bounds: FeatureBound::new(
                    s.bounds.map(|b| b[0]).unwrap_or(f32::NEG_INFINITY),
                    s.bounds.map(|b| b[1]).unwrap_or(f32::INFINITY),
                ),
                transform: s.transform.unwrap_or_default().to_byte(),
                _pad: [0; 3],
                transform_params: s.params.unwrap_or([0.0, 0.0]),
                discrete_set_offset: off,
                discrete_set_len: len,
                sentinel: s.sentinel.unwrap_or(f32::NAN),
            }
        })
        .collect();

    let sparse_overrides: Vec<SparseOverride> = req
        .sparse_overrides
        .iter()
        .map(|o| SparseOverride {
            idx: o.idx,
            value: o.value.unwrap_or(f32::NAN),
        })
        .collect();

    // The multi-codec section, when present, needs borrowed slices
    // backed by allocations that outlive the BakeRequest. Build them
    // here so the lifetimes are clean.
    let per_codec_owned: Vec<PerCodecMapJson> = req
        .multi_codec_schema
        .as_ref()
        .map(|s| s.per_codec.clone())
        .unwrap_or_default();
    let per_codec_inputs: Vec<PerCodecMapInput<'_>> = per_codec_owned
        .iter()
        .map(|m| PerCodecMapInput {
            codec_name: &m.codec_name,
            union_slot_for_codec_feat: &m.union_slot_for_codec_feat,
            output_range: (m.output_range[0], m.output_range[1]),
            head_n_cells: m.head_n_cells,
            head_n_heads: m.head_n_heads,
        })
        .collect();
    let multi_codec_input: Option<MultiCodecSchemaInput<'_>> =
        req.multi_codec_schema
            .as_ref()
            .map(|s| MultiCodecSchemaInput {
                union_feat_count: s.union_feat_count,
                per_codec: &per_codec_inputs,
            });

    let request = BakeRequest {
        schema_hash: req.schema_hash,
        flags: req.flags,
        scaler_mean: &req.scaler_mean,
        scaler_scale: &req.scaler_scale,
        layers: &layers,
        feature_bounds: &feature_bounds,
        metadata: &metadata,
        output_specs: &output_specs,
        discrete_sets: &discrete_sets_pool,
        sparse_overrides: &sparse_overrides,
        feature_order: None,
        output_order: None,
        compressed: req.compressed,
        hu_permutations: None,
        multi_codec_schema: multi_codec_input,
    };
    let bytes = if req.optimize {
        bake_optimized(&request)?
    } else {
        bake(&request)?
    };
    Ok(bytes)
}

/// Convenience: parse a JSON string and bake.
pub fn bake_from_json_str(s: &str) -> Result<Vec<u8>, BakeJsonError> {
    let req: BakeRequestJson = serde_json::from_str(s)
        .map_err(|e| BakeJsonError::BadHex(alloc::format!("json parse: {e}")))?;
    bake_from_json(&req)
}
