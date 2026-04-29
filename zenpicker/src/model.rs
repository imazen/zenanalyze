//! Binary model format parser.
//!
//! Layout (little-endian throughout):
//!
//! ```text
//! Header (32 bytes, v1):
//!   [0..4]   magic: b"ZNPK"
//!   [4..6]   version: u16 = 1
//!   [6..8]   header_size: u16 = 32
//!   [8..12]  n_inputs: u32
//!   [12..16] n_outputs: u32
//!   [16..20] n_layers: u32
//!   [20..28] schema_hash: u64
//!   [28..32] flags: u32 (reserved)
//!
//! Scaler section (8 * n_inputs bytes):
//!   scaler_mean: [f32; n_inputs]
//!   scaler_scale: [f32; n_inputs]
//!
//! Per-layer section (× n_layers):
//!   in_dim: u32
//!   out_dim: u32
//!   activation: u8        (0=Identity, 1=Relu)
//!   weight_dtype: u8      (0=F32, 1=F16)
//!   reserved: [u8; 2]
//!   weights: [W_dtype; in_dim * out_dim]   row-major (in_dim major)
//!   biases: [f32; out_dim]
//! ```
//!
//! Weights are stored input-major: `W[i * out_dim + o]` is the
//! contribution from input `i` to output `o`. This layout lets the
//! matmul stream `out_dim` outputs in chunks of 8 across each input
//! row, which is what `magetypes::f32x8` wants.

use crate::error::PickerError;

const MAGIC: [u8; 4] = *b"ZNPK";
pub const FORMAT_VERSION: u16 = 1;
const MIN_HEADER_SIZE: u16 = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Identity = 0,
    Relu = 1,
}

impl Activation {
    fn from_byte(b: u8) -> Result<Self, PickerError> {
        match b {
            0 => Ok(Self::Identity),
            1 => Ok(Self::Relu),
            other => Err(PickerError::UnknownActivation { byte: other }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightDtype {
    F32 = 0,
    F16 = 1,
}

impl WeightDtype {
    fn from_byte(b: u8) -> Result<Self, PickerError> {
        match b {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            other => Err(PickerError::UnknownWeightDtype { byte: other }),
        }
    }

    /// Bytes per weight in storage (compute is always f32).
    pub fn storage_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// A parsed view over the binary model. All slices borrow from the
/// caller's `&[u8]`. No ownership transfer, no copies — the model
/// lives as long as the bytes do.
#[derive(Debug)]
pub struct Model<'a> {
    n_inputs: usize,
    n_outputs: usize,
    schema_hash: u64,
    scaler_mean: &'a [f32],
    scaler_scale: &'a [f32],
    layers: alloc::vec::Vec<LayerView<'a>>,
}

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
    #[cfg(feature = "f16")]
    F16(&'a [half::f16]),
}

impl<'a> Model<'a> {
    /// Parse a model from raw bytes. The returned model borrows from
    /// `bytes` for its lifetime.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, PickerError> {
        let mut cur = Cursor::new(bytes);

        // Header.
        let magic = cur.take_array::<4>()?;
        if magic != MAGIC {
            return Err(PickerError::BadMagic { found: magic });
        }
        let version = cur.take_u16()?;
        if version > FORMAT_VERSION {
            return Err(PickerError::UnsupportedVersion {
                version,
                max_supported: FORMAT_VERSION,
            });
        }
        let header_size = cur.take_u16()?;
        if header_size < MIN_HEADER_SIZE {
            return Err(PickerError::HeaderTooSmall {
                advertised: header_size,
                min: MIN_HEADER_SIZE,
            });
        }

        let n_inputs = cur.take_u32()? as usize;
        let n_outputs = cur.take_u32()? as usize;
        let n_layers = cur.take_u32()? as usize;
        let schema_hash = cur.take_u64()?;
        let _flags = cur.take_u32()?;

        if n_inputs == 0 {
            return Err(PickerError::ZeroDimension { what: "n_inputs" });
        }
        if n_outputs == 0 {
            return Err(PickerError::ZeroDimension { what: "n_outputs" });
        }
        if n_layers == 0 {
            return Err(PickerError::ZeroDimension { what: "n_layers" });
        }

        // Skip any header extension fields a future version added.
        let header_consumed = MIN_HEADER_SIZE as usize;
        if header_size as usize > header_consumed {
            cur.skip(header_size as usize - header_consumed)?;
        }

        // Scaler section (always f32 — it's tiny).
        let scaler_mean = cur.take_f32_slice(n_inputs)?;
        let scaler_scale = cur.take_f32_slice(n_inputs)?;

        // Layers.
        let mut layers = alloc::vec::Vec::with_capacity(n_layers);
        let mut prev_out = n_inputs;
        for layer_idx in 0..n_layers {
            let in_dim = cur.take_u32()? as usize;
            let out_dim = cur.take_u32()? as usize;
            let activation_byte = cur.take_u8()?;
            let weight_dtype_byte = cur.take_u8()?;
            let _reserved = cur.take_array::<2>()?;

            if in_dim == 0 {
                return Err(PickerError::ZeroDimension { what: "layer.in_dim" });
            }
            if out_dim == 0 {
                return Err(PickerError::ZeroDimension { what: "layer.out_dim" });
            }
            if in_dim != prev_out {
                return Err(PickerError::LayerDimMismatch {
                    layer: layer_idx,
                    expected_in: prev_out,
                    got_in: in_dim,
                });
            }

            let activation = Activation::from_byte(activation_byte)?;
            let weight_dtype = WeightDtype::from_byte(weight_dtype_byte)?;
            let n_weights = in_dim
                .checked_mul(out_dim)
                .ok_or(PickerError::ZeroDimension {
                    what: "layer.weights overflow",
                })?;

            let weights = match weight_dtype {
                WeightDtype::F32 => WeightStorage::F32(cur.take_f32_slice(n_weights)?),
                WeightDtype::F16 => {
                    #[cfg(feature = "f16")]
                    {
                        WeightStorage::F16(cur.take_f16_slice(n_weights)?)
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(PickerError::F16Disabled);
                    }
                }
            };
            let biases = cur.take_f32_slice(out_dim)?;

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
            return Err(PickerError::OutputDimMismatch {
                expected: n_outputs,
                got: prev_out,
            });
        }

        Ok(Self {
            n_inputs,
            n_outputs,
            schema_hash,
            scaler_mean,
            scaler_scale,
            layers,
        })
    }

    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn schema_hash(&self) -> u64 {
        self.schema_hash
    }

    pub fn scaler_mean(&self) -> &[f32] {
        self.scaler_mean
    }

    pub fn scaler_scale(&self) -> &[f32] {
        self.scaler_scale
    }

    pub fn layers(&self) -> &[LayerView<'a>] {
        &self.layers
    }
}

/// Tiny zero-copy parser. Bounds-checks every read, returns
/// little-endian primitives, and slices f32 arrays by reading them
/// into a fresh `Vec` (no `unsafe`-cast since the input may be
/// unaligned).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N], PickerError> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or(PickerError::Truncated {
                offset: self.pos,
                want: N,
                have: self.bytes.len().saturating_sub(self.pos),
            })?;
        if end > self.bytes.len() {
            return Err(PickerError::Truncated {
                offset: self.pos,
                want: N,
                have: self.bytes.len() - self.pos,
            });
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn take_u8(&mut self) -> Result<u8, PickerError> {
        Ok(self.take_array::<1>()?[0])
    }

    fn take_u16(&mut self) -> Result<u16, PickerError> {
        Ok(u16::from_le_bytes(self.take_array::<2>()?))
    }

    fn take_u32(&mut self) -> Result<u32, PickerError> {
        Ok(u32::from_le_bytes(self.take_array::<4>()?))
    }

    fn take_u64(&mut self) -> Result<u64, PickerError> {
        Ok(u64::from_le_bytes(self.take_array::<8>()?))
    }

    fn skip(&mut self, n: usize) -> Result<(), PickerError> {
        let end = self.pos.checked_add(n).ok_or(PickerError::Truncated {
            offset: self.pos,
            want: n,
            have: self.bytes.len().saturating_sub(self.pos),
        })?;
        if end > self.bytes.len() {
            return Err(PickerError::Truncated {
                offset: self.pos,
                want: n,
                have: self.bytes.len() - self.pos,
            });
        }
        self.pos = end;
        Ok(())
    }

    /// Take a slice of `n` little-endian f32s, borrowed from the
    /// underlying bytes. Requires the bytes at `self.pos` to be
    /// 4-aligned in memory — the bake tool emits 4-aligned f32
    /// sections by construction (header is 32 bytes, scaler section
    /// follows at offset 32, layer headers are 12 bytes which keeps
    /// weights at 4-aligned offsets, biases are likewise 4-aligned
    /// because the f32 weights end at a 4-aligned boundary; in the
    /// f16 case we pad the weight count to even before the f32
    /// biases). Misaligned bytes (rare with `include_bytes!` because
    /// ELF rodata is page-aligned) are surfaced as a parse error
    /// rather than silently allocating to copy.
    fn take_f32_slice(&mut self, n: usize) -> Result<&'a [f32], PickerError> {
        let byte_len = n.checked_mul(4).ok_or(PickerError::Truncated {
            offset: self.pos,
            want: usize::MAX,
            have: 0,
        })?;
        let end = self
            .pos
            .checked_add(byte_len)
            .ok_or(PickerError::Truncated {
                offset: self.pos,
                want: byte_len,
                have: self.bytes.len().saturating_sub(self.pos),
            })?;
        if end > self.bytes.len() {
            return Err(PickerError::Truncated {
                offset: self.pos,
                want: byte_len,
                have: self.bytes.len() - self.pos,
            });
        }
        let raw = &self.bytes[self.pos..end];
        self.pos = end;
        if n == 0 {
            return Ok(&[]);
        }
        if !(raw.as_ptr() as usize).is_multiple_of(core::mem::align_of::<f32>()) {
            return Err(PickerError::Truncated {
                offset: self.pos - byte_len,
                want: byte_len,
                have: 0,
            });
        }
        bytemuck_cast_f32_slice(raw).ok_or(PickerError::Truncated {
            offset: self.pos - byte_len,
            want: byte_len,
            have: 0,
        })
    }

    #[cfg(feature = "f16")]
    fn take_f16_slice(&mut self, n: usize) -> Result<&'a [half::f16], PickerError> {
        let byte_len = n
            .checked_mul(2)
            .ok_or(PickerError::Truncated {
                offset: self.pos,
                want: usize::MAX,
                have: 0,
            })?;
        let end = self.pos.checked_add(byte_len).ok_or(PickerError::Truncated {
            offset: self.pos,
            want: byte_len,
            have: self.bytes.len().saturating_sub(self.pos),
        })?;
        if end > self.bytes.len() {
            return Err(PickerError::Truncated {
                offset: self.pos,
                want: byte_len,
                have: self.bytes.len() - self.pos,
            });
        }
        let raw = &self.bytes[self.pos..end];
        self.pos = end;
        if n == 0 {
            return Ok(&[]);
        }
        if !(raw.as_ptr() as usize).is_multiple_of(core::mem::align_of::<half::f16>()) {
            return Err(PickerError::Truncated {
                offset: self.pos - byte_len,
                want: byte_len,
                have: 0,
            });
        }
        bytemuck_cast_f16_slice(raw).ok_or(PickerError::Truncated {
            offset: self.pos - byte_len,
            want: byte_len,
            have: 0,
        })
    }
}

/// Lifetime-preserving zero-copy reinterpret of a byte slice as a
/// f32 slice. Returns `None` if alignment or length doesn't match.
///
/// Implemented via `bytemuck` in a wrapper module so the rest of
/// `zenpicker` can stay `unsafe = "forbid"`.
fn bytemuck_cast_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    use bytemuck::try_cast_slice;
    try_cast_slice(bytes).ok()
}

#[cfg(feature = "f16")]
fn bytemuck_cast_f16_slice(bytes: &[u8]) -> Option<&[half::f16]> {
    use bytemuck::try_cast_slice;
    try_cast_slice(bytes).ok()
}
