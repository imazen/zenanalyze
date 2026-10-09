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
//!   "layers": [ /* BakeLayerJson, see below */ ],   // v3 chain, or
//!   "graph":  [ /* BakeNodeJson — a v4 op graph */ ],
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
//!
//! ## Feature transforms (incl. Sinusoidal expander)
//!
//! Per-feature transforms ride on two metadata entries:
//!
//! ```json
//! [
//!   { "key": "zentrain.feature_transforms",
//!     "type": "utf8",
//!     "text": "identity\nsinusoidal\nidentity\nidentity\nidentity" },
//!   { "key": "zentrain.feature_transform_params",
//!     "type": "utf8",
//!     "text": "\n1,2,4,8,16,32,64,128,256,512,1024,2048\n\n\n" }
//! ]
//! ```
//!
//! The two strings are line-aligned: line `i` of `feature_transforms`
//! declares the variant for raw input `i`, and line `i` of
//! `feature_transform_params` is the comma-separated parameter list
//! for that transform. Empty lines mean "no params".
//!
//! Scalar variants (everything but `sinusoidal`) preserve the one-input
//! → one-output contract. The **`sinusoidal`** token marks a
//! scalar-to-vector expander: each input feature is expanded to
//! `2 * num_freqs` outputs (sin + cos at each frequency). In the
//! example above, raw input 1 carries 12 frequencies, so it expands
//! to 24 features. The total post-transform layer-1 input width is
//! `sum(output_arity)` over all features, not the raw input count;
//! the bake-side composer validates `layers[0].in_dim` matches.
//!
//! Frequencies are in cycles per unit input — i.e. the multiplier
//! inside `sin(2π·f·x)` before `2π` scaling. NeRF-style `2^k`
//! schedules and learned schedules are both fine.
//!
//! See `zenpredict::FeatureTransform::Sinusoidal` for the variant's
//! full math + reference, and `apply_feature_pipeline_expanding` for
//! the runtime that consumes the wire format.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use serde::Deserialize;

use crate::composer::{BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, bake};
use crate::graph::{BakeNode, GraphBakeRequest, bake_graph};
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
    /// v3 layer chain. Leave empty (or omit) when `graph` is set.
    #[serde(default)]
    pub layers: Vec<BakeLayerJson>,
    /// ZNPR v4 op graph (see [`BakeNodeJson`]). When non-empty the
    /// baker writes a v4 graph via [`bake_graph`] instead of a v3 chain;
    /// `layers` must then be empty and `optimize` false. `zerobias_tau`
    /// and `compressed` apply as for chains (zero-bias runs per Dense
    /// node). Default empty: chain bakes are unchanged.
    #[serde(default)]
    pub graph: Vec<BakeNodeJson>,
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
    /// The `zenanalyze` crate version the features were extracted with
    /// (e.g. `"0.2.7"`). When set, the baker writes it verbatim to the
    /// [`ANALYZER_VERSION`](zenpredict::keys::ANALYZER_VERSION) UTF-8
    /// metadata key — the analyzer half of the `zenanalyze-api`
    /// offer/reuse key. Prefer this typed field over a hand-rolled
    /// `metadata` entry. Default `None` (key omitted). The value can
    /// only be known by whoever ran the extraction, so the baker can't
    /// synthesize it — pass it through from the training pipeline.
    #[serde(default)]
    pub analyzer_version: Option<String>,
    /// The `zenanalyze::feature_defs_version()` the features were
    /// extracted with. When set, the baker writes it as a 4-byte
    /// little-endian `u32` to the
    /// [`FEATURE_DEFS_VERSION`](zenpredict::keys::FEATURE_DEFS_VERSION)
    /// numeric metadata key — the within-major drift half of the reuse
    /// key. Prefer this typed field over a hand-rolled `metadata` entry:
    /// a `u32` can't ride the `f32` repr (different bytes), so hand
    /// encoding would mean emitting LE hex by hand. Default `None`.
    #[serde(default)]
    pub feature_defs_version: Option<u32>,
    /// The `zenanalyze::AnalysisQuery::config_hash()` the features were
    /// extracted under (`0` = canonical default / gamma). When set, the
    /// baker writes it as an 8-byte little-endian `u64` to the
    /// [`FEATURE_CONFIG_HASH`](zenpredict::keys::FEATURE_CONFIG_HASH)
    /// numeric metadata key — the analysis-config third of the
    /// `zenanalyze-api` reuse key, so a codec won't reuse e.g.
    /// linear-light features against a gamma-trained model. Prefer this
    /// typed field over a hand-rolled `metadata` entry. Default `None`.
    #[serde(default)]
    pub feature_config_hash: Option<u64>,
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

/// Activation names accepted by v4 graph nodes ([`BakeNodeJson`]): the
/// three chain activations plus the graph-only `exp` and `softplus`.
/// Separate from [`ActivationJson`] (v3 layers) so chain specs can't name
/// an activation a v3 file can't carry.
#[derive(Deserialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum GraphActivationJson {
    Identity,
    Relu,
    LeakyRelu,
    Exp,
    Softplus,
}

impl From<GraphActivationJson> for Activation {
    fn from(a: GraphActivationJson) -> Self {
        match a {
            GraphActivationJson::Identity => Activation::Identity,
            GraphActivationJson::Relu => Activation::Relu,
            GraphActivationJson::LeakyRelu => Activation::LeakyRelu,
            GraphActivationJson::Exp => Activation::Exp,
            GraphActivationJson::Softplus => Activation::Softplus,
        }
    }
}

/// One node of a v4 graph, JSON-side. Tagged by `"op"`; mirrors
/// [`BakeNode`]. Input indices name earlier nodes.
///
/// ```json
/// { "op": "input", "width": 4 }
/// { "op": "dense", "input": 1, "out_dim": 3, "activation": "relu",
///   "dtype": "f16", "weights": [...], "biases": [...] }   // biases optional
/// { "op": "activation", "input": 3, "activation": "exp" }
/// { "op": "gather", "input": 0, "indices": [2, 3] }
/// { "op": "add", "a": 3, "b": 4 }
/// { "op": "mul", "a": 3, "b": 4 }
/// { "op": "concat", "inputs": [3, 4] }
/// ```
#[derive(Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase", deny_unknown_fields)]
#[non_exhaustive]
pub enum BakeNodeJson {
    Input {
        width: usize,
    },
    Dense {
        input: u32,
        out_dim: usize,
        activation: GraphActivationJson,
        dtype: DtypeJson,
        weights: Vec<f32>,
        #[serde(default)]
        biases: Option<Vec<f32>>,
    },
    Activation {
        input: u32,
        activation: GraphActivationJson,
    },
    Gather {
        input: u32,
        indices: Vec<u32>,
    },
    Add {
        a: u32,
        b: u32,
    },
    Mul {
        a: u32,
        b: u32,
    },
    Concat {
        inputs: Vec<u32>,
    },
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
    for &[hi, lo] in bytes.as_chunks::<2>().0 {
        let hi = hex_nibble(hi)?;
        let lo = hex_nibble(lo)?;
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
///
/// Also honors the optional `analyzer_version` / `feature_defs_version`
/// / `feature_config_hash` stamps, writing them to the `zenanalyze-api`
/// reuse-key metadata (UTF-8 / LE-u32 / LE-u64). All default to `None`
/// (key omitted); an explicit `metadata` entry for the same key takes
/// precedence.
pub fn bake_from_json(req: &BakeRequestJson) -> Result<Vec<u8>, BakeJsonError> {
    // Decode metadata values up front so the byte buffers outlive
    // the borrow into BakeMetadataEntry.
    let decoded_values: Vec<Vec<u8>> = req
        .metadata
        .iter()
        .map(decode_metadata_value)
        .collect::<Result<_, _>>()?;

    // First-class version stamps → owned bytes that outlive the
    // metadata borrow. defs_version is a u32, config_hash a u64, both
    // written LE so they decode identically on any endianness (cf.
    // Model::feature_defs_version / feature_config_hash).
    let defs_version_bytes: Option<[u8; 4]> = req.feature_defs_version.map(u32::to_le_bytes);
    let config_hash_bytes: Option<[u8; 8]> = req.feature_config_hash.map(u64::to_le_bytes);

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

    let mut metadata: Vec<BakeMetadataEntry<'_>> = req
        .metadata
        .iter()
        .zip(decoded_values.iter())
        .map(|(entry, bytes)| BakeMetadataEntry {
            key: &entry.key,
            kind: entry.kind.into(),
            value: bytes,
        })
        .collect();

    // Append the first-class version stamps, unless the caller already
    // hand-rolled a `metadata` entry for the same key (that explicit
    // entry wins — we never emit a duplicate key into the blob).
    let has_key = |k: &str| req.metadata.iter().any(|e| e.key == k);
    if let Some(v) = &req.analyzer_version
        && !has_key(zenpredict::keys::ANALYZER_VERSION)
    {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::ANALYZER_VERSION,
            kind: MetadataType::Utf8,
            value: v.as_bytes(),
        });
    }
    if let Some(bytes) = &defs_version_bytes
        && !has_key(zenpredict::keys::FEATURE_DEFS_VERSION)
    {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_DEFS_VERSION,
            kind: MetadataType::Numeric,
            value: &bytes[..],
        });
    }
    if let Some(bytes) = &config_hash_bytes
        && !has_key(zenpredict::keys::FEATURE_CONFIG_HASH)
    {
        metadata.push(BakeMetadataEntry {
            key: zenpredict::keys::FEATURE_CONFIG_HASH,
            kind: MetadataType::Numeric,
            value: &bytes[..],
        });
    }

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
    };
    if !req.graph.is_empty() {
        if !req.layers.is_empty() {
            return Err(BakeError::GraphInvalid {
                node: 0,
                what: "json: set either `layers` (v3 chain) or `graph` (v4), not both",
            }
            .into());
        }
        if req.optimize {
            return Err(BakeError::GraphInvalid {
                node: 0,
                what: "json: optimize applies to layer chains only",
            }
            .into());
        }
        // Owned zero-biased Dense weights, indexed by node.
        let graph_weights: Vec<Option<Vec<f32>>> = req
            .graph
            .iter()
            .map(|n| match n {
                BakeNodeJson::Dense { weights, .. } if req.zerobias_tau > 0.0 => {
                    let mut w = weights.clone();
                    apply_zero_bias_per_layer_in_place(&mut w, req.zerobias_tau);
                    Some(w)
                }
                _ => None,
            })
            .collect();
        let nodes: Vec<BakeNode<'_>> = req
            .graph
            .iter()
            .zip(&graph_weights)
            .map(|(n, zb)| match n {
                BakeNodeJson::Input { width } => BakeNode::Input { width: *width },
                BakeNodeJson::Dense {
                    input,
                    out_dim,
                    activation,
                    dtype,
                    weights,
                    biases,
                } => BakeNode::Dense {
                    input: *input,
                    out_dim: *out_dim,
                    activation: (*activation).into(),
                    dtype: (*dtype).into(),
                    weights: zb.as_deref().unwrap_or(weights),
                    biases: biases.as_deref(),
                },
                BakeNodeJson::Activation { input, activation } => BakeNode::Activation {
                    input: *input,
                    activation: (*activation).into(),
                },
                BakeNodeJson::Gather { input, indices } => BakeNode::Gather {
                    input: *input,
                    indices,
                },
                BakeNodeJson::Add { a, b } => BakeNode::Add { a: *a, b: *b },
                BakeNodeJson::Mul { a, b } => BakeNode::Mul { a: *a, b: *b },
                BakeNodeJson::Concat { inputs } => BakeNode::Concat { inputs },
            })
            .collect();
        let graph =
            GraphBakeRequest::new(req.schema_hash, &req.scaler_mean, &req.scaler_scale, &nodes)
                .flags(req.flags)
                .feature_bounds(&feature_bounds)
                .metadata(&metadata)
                .output_specs(&output_specs)
                .discrete_sets(&discrete_sets_pool)
                .sparse_overrides(&sparse_overrides)
                .compressed(req.compressed);
        return Ok(bake_graph(&graph)?);
    }
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
