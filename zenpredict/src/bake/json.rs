//! JSON input schema for [`bake_v2`].
//!
//! Lets language-agnostic toolchains (the Python training pipeline at
//! `zenanalyze/zenpicker/tools/`, ad-hoc baking scripts, etc.) drive
//! a v2 bake without re-implementing the byte-level format. The
//! Python side dumps a `BakeRequestJson`, then shells out to the
//! `zenpredict-bake` binary which calls [`bake_v2`] on the
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
//!   "metadata": [ /* MetadataEntryJson, see below */ ]      // optional
//! }
//! ```
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
//! { "key": "zenpicker.bake_name",
//!   "type": "utf8",
//!   "text": "v2.1_full" }
//! ```
//!
//! ```json
//! { "key": "zenpicker.calibration_metrics",
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

use super::v2::{BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, bake_v2};
use crate::bounds::FeatureBound;
use crate::metadata::MetadataType;
use crate::model::{Activation, WeightDtype};

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

#[derive(Deserialize, Debug)]
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

/// Decode a metadata entry into the byte payload that the v2 baker
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

/// Bake a `BakeRequestJson` into ZNPR v2 bytes. Performs all input
/// validation that [`bake_v2`] does plus the JSON-side type-vs-repr
/// checks for metadata entries.
pub fn bake_from_json(req: &BakeRequestJson) -> Result<Vec<u8>, BakeJsonError> {
    // Decode metadata values up front so the byte buffers outlive
    // the borrow into BakeMetadataEntry.
    let decoded_values: Vec<Vec<u8>> = req
        .metadata
        .iter()
        .map(decode_metadata_value)
        .collect::<Result<_, _>>()?;

    // Convert layers (owned Vec<f32> → borrowed slices via reuse).
    let layers: Vec<BakeLayer<'_>> = req
        .layers
        .iter()
        .map(|l| BakeLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation.into(),
            dtype: l.dtype.into(),
            weights: &l.weights,
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

    let bake = BakeRequest {
        schema_hash: req.schema_hash,
        flags: req.flags,
        scaler_mean: &req.scaler_mean,
        scaler_scale: &req.scaler_scale,
        layers: &layers,
        feature_bounds: &feature_bounds,
        metadata: &metadata,
    };
    Ok(bake_v2(&bake)?)
}

/// Convenience: parse a JSON string and bake.
pub fn bake_from_json_str(s: &str) -> Result<Vec<u8>, BakeJsonError> {
    let req: BakeRequestJson = serde_json::from_str(s)
        .map_err(|e| BakeJsonError::BadHex(alloc::format!("json parse: {e}")))?;
    bake_from_json(&req)
}
