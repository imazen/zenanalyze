//! ZNPR v2 binary model format — parser and typed views.
//!
//! ## Layout
//!
//! Little-endian throughout. Wrap `include_bytes!` output in
//! `#[repr(C, align(16))]` so the byte slice's start address is
//! 16-aligned; every section's offset is a multiple of 8, so all
//! typed slices land at native alignment without copies.
//!
//! ```text
//! 0..128                  Header (#[repr(C)], pod, fixed size)
//! 128..(128 + n_layers*48) LayerEntry[n_layers] (#[repr(C)], pod)
//! (rest of file)          Aligned data blobs at offsets named by
//!                         the Sections in Header / LayerEntry:
//!                           - scaler_mean, scaler_scale (f32)
//!                           - per-layer weights (f32 / f16 / i8)
//!                           - per-layer scales (f32, only for I8 layers)
//!                           - per-layer biases (f32)
//!                           - feature_bounds (FeatureBound = (f32, f32))
//!                           - metadata blob (TLV; see [`crate::metadata`])
//! ```
//!
//! ### Header (128 bytes)
//!
//! ```text
//! 0..4    magic = b"ZNPR"
//! 4..6    version: u16 = 2
//! 6..8    flags: u16  (reserved, 0)
//! 8..12   n_inputs: u32
//! 12..16  n_outputs: u32
//! 16..20  n_layers: u32
//! 20..24  _pad0: u32  (alignment for u64)
//! 24..32  schema_hash: u64
//! 32..40  scaler_mean: Section
//! 40..48  scaler_scale: Section
//! 48..56  layer_table: Section
//! 56..64  feature_bounds: Section   (len=0 when absent)
//! 64..72  metadata: Section         (len=0 when absent)
//! 72..128 reserved: [u32; 14]
//! ```
//!
//! ### LayerEntry (48 bytes)
//!
//! ```text
//! 0..4    in_dim: u32
//! 4..8    out_dim: u32
//! 8..9    activation: u8       (0=Identity, 1=Relu, 2=LeakyRelu)
//! 9..10   weight_dtype: u8     (0=F32, 1=F16, 2=I8)
//! 10..12  flags: u16           (reserved)
//! 12..20  weights: Section
//! 20..28  scales: Section      (len=0 unless weight_dtype == I8)
//! 28..36  biases: Section
//! 36..48  reserved: [u32; 3]
//! ```
//!
//! ### Section
//!
//! ```text
//! 0..4   offset: u32   absolute byte offset into the file
//! 4..8   len: u32      byte length
//! ```
//!
//! `len = 0` means the section is absent. Both `feature_bounds` and
//! `metadata` are optional in this sense.
//!
//! ## Weight storage
//!
//! - **F32** — `weights.len / 4` f32 values, row-major (input-major).
//!   `W[i * out_dim + o]` is the contribution from input `i` to
//!   output `o`.
//! - **F16** — `weights.len / 2` raw IEEE-754 binary16 bit patterns
//!   stored as `u16`. Converted to f32 per element in the inner loop
//!   (see [`crate::inference::f16_bits_to_f32`]).
//! - **I8** — `weights.len` signed 8-bit values, row-major. Per-output
//!   f32 scales in the `scales` Section. Dequantized weight is
//!   `W[i, o] = weights[i * out_dim + o] as f32 * scales[o]`.
//!
//! ## I8 quantization scheme (bake-side, not enforced at runtime)
//!
//! ```text
//! scales[o]    = max_i |W_f32[i, o]| / 127.0   (or 1.0 if the column is zero)
//! i8_weights[i, o] = round(W_f32[i, o] / scales[o]).clamp(-128, 127)
//! ```

use bytemuck::{Pod, Zeroable, pod_read_unaligned};

use crate::error::PredictError;
use crate::metadata::Metadata;

pub const FORMAT_VERSION: u16 = 2;
pub const LEAKY_RELU_ALPHA: f32 = 0.01;
const MAGIC: [u8; 4] = *b"ZNPR";
const HEADER_SIZE: usize = 128;
const LAYER_ENTRY_SIZE: usize = 48;

/// `(offset, len)` pair addressing a byte range in the file. `len = 0`
/// is reserved for "section absent".
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct Section {
    pub offset: u32,
    pub len: u32,
}

impl Section {
    pub const fn empty() -> Self {
        Self { offset: 0, len: 0 }
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Slice into `bytes`, validating bounds. Returns the empty
    /// slice when `len == 0` (section absent), without consulting
    /// `offset`.
    pub fn slice<'a>(self, what: &'static str, bytes: &'a [u8]) -> Result<&'a [u8], PredictError> {
        if self.len == 0 {
            return Ok(&[]);
        }
        let start = self.offset as usize;
        let end = start
            .checked_add(self.len as usize)
            .ok_or(PredictError::SectionOutOfRange {
                what,
                offset: self.offset,
                len: self.len,
                file_len: bytes.len(),
            })?;
        if end > bytes.len() {
            return Err(PredictError::SectionOutOfRange {
                what,
                offset: self.offset,
                len: self.len,
                file_len: bytes.len(),
            });
        }
        Ok(&bytes[start..end])
    }
}

/// Fixed `#[repr(C)]` file header. Loader does one
/// `bytemuck::pod_read_unaligned::<Header>(&bytes[..128])` to copy
/// it out — never borrows the in-file bytes for the header itself,
/// so the `Header` value is owned and the loader can validate
/// without holding a `&Header` into possibly-misaligned input.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u16,
    pub flags: u16,
    pub n_inputs: u32,
    pub n_outputs: u32,
    pub n_layers: u32,
    pub _pad0: u32,
    pub schema_hash: u64,
    pub scaler_mean: Section,
    pub scaler_scale: Section,
    pub layer_table: Section,
    pub feature_bounds: Section,
    pub metadata: Section,
    pub reserved: [u32; 14],
}

const _: () = assert!(core::mem::size_of::<Header>() == HEADER_SIZE);
const _: () = assert!(core::mem::size_of::<Section>() == 8);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LayerEntry {
    pub in_dim: u32,
    pub out_dim: u32,
    pub activation: u8,
    pub weight_dtype: u8,
    pub flags: u16,
    pub weights: Section,
    pub scales: Section,
    pub biases: Section,
    pub reserved: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<LayerEntry>() == LAYER_ENTRY_SIZE);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Identity = 0,
    Relu = 1,
    /// `LeakyRelu(alpha = 0.01)` — fixed slope; matches PyTorch's
    /// `nn.LeakyReLU` default and what `train_hybrid.py
    /// --activation leakyrelu` emits.
    LeakyRelu = 2,
}

impl Activation {
    pub(crate) fn from_byte(b: u8) -> Result<Self, PredictError> {
        match b {
            0 => Ok(Self::Identity),
            1 => Ok(Self::Relu),
            2 => Ok(Self::LeakyRelu),
            other => Err(PredictError::UnknownActivation { byte: other }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightDtype {
    F32 = 0,
    F16 = 1,
    I8 = 2,
}

impl WeightDtype {
    pub(crate) fn from_byte(b: u8) -> Result<Self, PredictError> {
        match b {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::I8),
            other => Err(PredictError::UnknownWeightDtype { byte: other }),
        }
    }

    pub fn storage_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }
}

/// Layered weight + bias view, borrowed from the file bytes.
#[derive(Debug)]
pub struct LayerView<'a> {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Activation,
    pub weights: WeightStorage<'a>,
    pub biases: &'a [f32],
}

#[derive(Debug)]
pub enum WeightStorage<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    I8 {
        weights: &'a [i8],
        scales: &'a [f32],
    },
}

/// Parsed view over a ZNPR v2 model. All slices borrow from the
/// caller's `&[u8]`. No ownership transfer, no copies of the weight
/// data — the model lives as long as the bytes do.
#[derive(Debug)]
pub struct Model<'a> {
    bytes: &'a [u8],
    header: Header,
    scaler_mean: &'a [f32],
    scaler_scale: &'a [f32],
    layers: alloc::vec::Vec<LayerView<'a>>,
    feature_bounds: &'a [crate::bounds::FeatureBound],
    metadata: Metadata<'a>,
}

impl<'a> Model<'a> {
    /// Parse a v2 model from raw bytes. The returned `Model` borrows
    /// from `bytes` for `'a`.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, PredictError> {
        if bytes.len() < HEADER_SIZE {
            return Err(PredictError::Truncated {
                offset: 0,
                want: HEADER_SIZE,
                have: bytes.len(),
            });
        }
        let header: Header = pod_read_unaligned(&bytes[..HEADER_SIZE]);

        if header.magic != MAGIC {
            return Err(PredictError::BadMagic {
                found: header.magic,
            });
        }
        if header.version != FORMAT_VERSION {
            return Err(PredictError::UnsupportedVersion {
                version: header.version,
                expected: FORMAT_VERSION,
            });
        }

        let n_inputs = header.n_inputs as usize;
        let n_outputs = header.n_outputs as usize;
        let n_layers = header.n_layers as usize;
        if n_inputs == 0 {
            return Err(PredictError::ZeroDimension { what: "n_inputs" });
        }
        if n_outputs == 0 {
            return Err(PredictError::ZeroDimension { what: "n_outputs" });
        }
        if n_layers == 0 {
            return Err(PredictError::ZeroDimension { what: "n_layers" });
        }

        // Scaler.
        let scaler_mean = cast_f32_section("scaler_mean", header.scaler_mean, bytes, n_inputs)?;
        let scaler_scale = cast_f32_section("scaler_scale", header.scaler_scale, bytes, n_inputs)?;

        // Layer table.
        let layer_bytes = header.layer_table.slice("layer_table", bytes)?;
        let expected_layer_bytes =
            n_layers
                .checked_mul(LAYER_ENTRY_SIZE)
                .ok_or(PredictError::ZeroDimension {
                    what: "layer_table size overflow",
                })?;
        if layer_bytes.len() != expected_layer_bytes {
            return Err(PredictError::SectionOutOfRange {
                what: "layer_table",
                offset: header.layer_table.offset,
                len: header.layer_table.len,
                file_len: bytes.len(),
            });
        }
        // Layer table is a packed `[LayerEntry]`; cast via
        // `bytemuck::try_cast_slice` (returns Err on alignment
        // failure rather than panicking).
        let layer_entries: &[LayerEntry] =
            bytemuck::try_cast_slice(layer_bytes).map_err(|_| PredictError::SectionMisaligned {
                what: "layer_table",
                offset: header.layer_table.offset,
                required_align: core::mem::align_of::<LayerEntry>(),
            })?;

        let mut layers = alloc::vec::Vec::with_capacity(n_layers);
        let mut prev_out = n_inputs;
        for (layer_idx, entry) in layer_entries.iter().enumerate() {
            let in_dim = entry.in_dim as usize;
            let out_dim = entry.out_dim as usize;
            if in_dim == 0 {
                return Err(PredictError::ZeroDimension {
                    what: "layer.in_dim",
                });
            }
            if out_dim == 0 {
                return Err(PredictError::ZeroDimension {
                    what: "layer.out_dim",
                });
            }
            if in_dim != prev_out {
                return Err(PredictError::LayerDimMismatch {
                    layer: layer_idx,
                    expected_in: prev_out,
                    got_in: in_dim,
                });
            }
            let activation = Activation::from_byte(entry.activation)?;
            let weight_dtype = WeightDtype::from_byte(entry.weight_dtype)?;

            let n_weights = in_dim
                .checked_mul(out_dim)
                .ok_or(PredictError::ZeroDimension {
                    what: "layer.weights overflow",
                })?;
            let weights = match weight_dtype {
                WeightDtype::F32 => WeightStorage::F32(cast_f32_section(
                    "layer.weights[f32]",
                    entry.weights,
                    bytes,
                    n_weights,
                )?),
                WeightDtype::F16 => WeightStorage::F16(cast_u16_section(
                    "layer.weights[f16]",
                    entry.weights,
                    bytes,
                    n_weights,
                )?),
                WeightDtype::I8 => {
                    let w = cast_i8_section("layer.weights[i8]", entry.weights, bytes, n_weights)?;
                    let s = cast_f32_section("layer.scales", entry.scales, bytes, out_dim)?;
                    WeightStorage::I8 {
                        weights: w,
                        scales: s,
                    }
                }
            };
            // For non-I8 layers the scales section must be empty.
            if !matches!(weight_dtype, WeightDtype::I8) && !entry.scales.is_empty() {
                return Err(PredictError::SectionOutOfRange {
                    what: "layer.scales (must be empty for non-I8 layer)",
                    offset: entry.scales.offset,
                    len: entry.scales.len,
                    file_len: bytes.len(),
                });
            }
            let biases = cast_f32_section("layer.biases", entry.biases, bytes, out_dim)?;

            layers.push(LayerView {
                in_dim,
                out_dim,
                activation,
                weights,
                biases,
            });
            prev_out = out_dim;
        }

        if prev_out != n_outputs {
            return Err(PredictError::OutputDimMismatch {
                expected: n_outputs,
                got: prev_out,
            });
        }

        // Feature bounds (optional). Length, when present, must be
        // 2 * n_inputs f32s.
        let feature_bounds = if header.feature_bounds.is_empty() {
            &[][..]
        } else {
            let raw =
                cast_f32_section("feature_bounds", header.feature_bounds, bytes, n_inputs * 2)?;
            // Reinterpret pairs of f32 as `[FeatureBound]`. They are
            // `#[repr(C)] { low: f32, high: f32 }` so the layout is
            // identical.
            let bound_bytes = bytemuck::cast_slice::<f32, u8>(raw);
            bytemuck::try_cast_slice::<u8, crate::bounds::FeatureBound>(bound_bytes).map_err(
                |_| PredictError::SectionMisaligned {
                    what: "feature_bounds",
                    offset: header.feature_bounds.offset,
                    required_align: core::mem::align_of::<crate::bounds::FeatureBound>(),
                },
            )?
        };

        // Metadata blob (optional).
        let metadata_bytes = header.metadata.slice("metadata", bytes)?;
        let metadata = Metadata::parse(metadata_bytes)?;

        Ok(Self {
            bytes,
            header,
            scaler_mean,
            scaler_scale,
            layers,
            feature_bounds,
            metadata,
        })
    }

    /// Parse and verify the schema_hash matches `expected`. Convenience
    /// for codecs that compile-in a schema hash and want to fail loudly
    /// at load when a stale bake gets shipped.
    pub fn from_bytes_with_schema(bytes: &'a [u8], expected: u64) -> Result<Self, PredictError> {
        let model = Self::from_bytes(bytes)?;
        if model.header.schema_hash != expected {
            return Err(PredictError::SchemaHashMismatch {
                expected,
                got: model.header.schema_hash,
            });
        }
        Ok(model)
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn version(&self) -> u16 {
        self.header.version
    }

    pub fn flags(&self) -> u16 {
        self.header.flags
    }

    pub fn n_inputs(&self) -> usize {
        self.header.n_inputs as usize
    }

    pub fn n_outputs(&self) -> usize {
        self.header.n_outputs as usize
    }

    pub fn schema_hash(&self) -> u64 {
        self.header.schema_hash
    }

    pub fn scaler_mean(&self) -> &'a [f32] {
        self.scaler_mean
    }

    pub fn scaler_scale(&self) -> &'a [f32] {
        self.scaler_scale
    }

    pub fn layers(&self) -> &[LayerView<'a>] {
        &self.layers
    }

    pub fn feature_bounds(&self) -> &'a [crate::bounds::FeatureBound] {
        self.feature_bounds
    }

    pub fn metadata(&self) -> &Metadata<'a> {
        &self.metadata
    }

    /// Length of the scratch buffers (`scratch_a`, `scratch_b`,
    /// `output`) needed to call [`crate::inference::forward`] —
    /// `max(n_inputs, max_layer_out_dim)`.
    pub fn scratch_len(&self) -> usize {
        let max_out = self.layers.iter().map(|l| l.out_dim).max().unwrap_or(0);
        max_out.max(self.n_inputs())
    }

    /// Raw bytes the model was parsed from. Useful for re-serializing
    /// or hashing a known-good blob.
    pub fn raw_bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

fn cast_f32_section<'a>(
    what: &'static str,
    section: Section,
    bytes: &'a [u8],
    expected_count: usize,
) -> Result<&'a [f32], PredictError> {
    let raw = section.slice(what, bytes)?;
    let expected_bytes = expected_count * 4;
    if raw.len() != expected_bytes {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    if expected_count == 0 {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice::<u8, f32>(raw).map_err(|_| PredictError::SectionMisaligned {
        what,
        offset: section.offset,
        required_align: core::mem::align_of::<f32>(),
    })
}

fn cast_u16_section<'a>(
    what: &'static str,
    section: Section,
    bytes: &'a [u8],
    expected_count: usize,
) -> Result<&'a [u16], PredictError> {
    let raw = section.slice(what, bytes)?;
    let expected_bytes = expected_count * 2;
    if raw.len() != expected_bytes {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    if expected_count == 0 {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice::<u8, u16>(raw).map_err(|_| PredictError::SectionMisaligned {
        what,
        offset: section.offset,
        required_align: core::mem::align_of::<u16>(),
    })
}

fn cast_i8_section<'a>(
    what: &'static str,
    section: Section,
    bytes: &'a [u8],
    expected_count: usize,
) -> Result<&'a [i8], PredictError> {
    let raw = section.slice(what, bytes)?;
    if raw.len() != expected_count {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    if expected_count == 0 {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice::<u8, i8>(raw).map_err(|_| PredictError::SectionMisaligned {
        what,
        offset: section.offset,
        required_align: 1,
    })
}
