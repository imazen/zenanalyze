//! ZNPR v2 byte-stream composer.

use core::fmt;

use crate::metadata::MetadataType;
use crate::model::{Activation, Section, WeightDtype};

/// Errors raised by [`bake_v2`]. Distinct from `PredictError` —
/// these are bake-side validation failures, not runtime decode
/// issues.
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

/// All inputs to a v2 bake.
pub struct BakeRequest<'a> {
    pub schema_hash: u64,
    pub flags: u16,
    pub scaler_mean: &'a [f32],
    pub scaler_scale: &'a [f32],
    pub layers: &'a [BakeLayer<'a>],
    /// Length must equal `n_inputs` (one [low, high] pair per
    /// input). Pass an empty slice to omit the section.
    pub feature_bounds: &'a [crate::bounds::FeatureBound],
    pub metadata: &'a [BakeMetadataEntry<'a>],
}

const HEADER_SIZE: usize = 128;
const LAYER_ENTRY_SIZE: usize = 48;
const SECTION_OFF_SCALER_MEAN: usize = 32;
const SECTION_OFF_SCALER_SCALE: usize = 40;
const SECTION_OFF_LAYER_TABLE: usize = 48;
const SECTION_OFF_FEATURE_BOUNDS: usize = 56;
const SECTION_OFF_METADATA: usize = 64;

/// Compose a v2 ZNPR byte stream. Output round-trips through
/// [`crate::Model::from_bytes`].
pub fn bake_v2(req: &BakeRequest<'_>) -> Result<alloc::vec::Vec<u8>, BakeError> {
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
    buf[4..6].copy_from_slice(&2u16.to_le_bytes());
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
        Section {
            offset: HEADER_SIZE as u32,
            len: (n_layers * LAYER_ENTRY_SIZE) as u32,
        },
    );

    // Scaler sections.
    pad_to(&mut buf, 4);
    let scaler_mean_section = append_f32(&mut buf, req.scaler_mean);
    write_section(&mut buf, SECTION_OFF_SCALER_MEAN, scaler_mean_section);

    pad_to(&mut buf, 4);
    let scaler_scale_section = append_f32(&mut buf, req.scaler_scale);
    write_section(&mut buf, SECTION_OFF_SCALER_SCALE, scaler_scale_section);

    // Per-layer payloads.
    for (i, layer) in req.layers.iter().enumerate() {
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
                Section {
                    offset: start,
                    len: (layer.weights.len() * 2) as u32,
                }
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
                Section {
                    offset: start,
                    len: layer.weights.len() as u32,
                }
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
        Section {
            offset: start,
            len: (req.feature_bounds.len() * 8) as u32,
        }
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
        Section {
            offset: start,
            len: (buf.len() as u32) - start,
        }
    };
    write_section(&mut buf, SECTION_OFF_METADATA, metadata_section);

    Ok(buf)
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
    Section {
        offset: start,
        len: (values.len() * 4) as u32,
    }
}

fn write_section(buf: &mut [u8], at: usize, s: Section) {
    buf[at..at + 4].copy_from_slice(&s.offset.to_le_bytes());
    buf[at + 4..at + 8].copy_from_slice(&s.len.to_le_bytes());
}

fn write_section_inline(entry: &mut [u8], at: usize, s: Section) {
    entry[at..at + 4].copy_from_slice(&s.offset.to_le_bytes());
    entry[at + 4..at + 8].copy_from_slice(&s.len.to_le_bytes());
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
/// hardware. Matches `crate::inference::f16_bits_to_f32` round-trip
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
