//! Shared wire-format constants for the ZNPR v3 binary format.
//!
//! These offsets and sizes are byte-exact and used by both the parser
//! ([`crate::Model::from_bytes`]) and the composer in the sibling
//! `zenpredict-bake` crate. Keeping them in one module guarantees the
//! two sides can't drift — a parser-side offset change must update
//! the composer in lockstep and vice versa.
//!
//! See [`crate::Header`] for the full byte layout documentation.

/// Size of the fixed-format header in bytes.
pub const HEADER_SIZE: usize = 128;

/// Size of one [`crate::LayerEntry`] in bytes.
pub const LAYER_ENTRY_SIZE: usize = 48;

/// Header byte offset of the `scaler_mean` [`crate::Section`].
pub const SECTION_OFF_SCALER_MEAN: usize = 32;

/// Header byte offset of the `scaler_scale` [`crate::Section`].
pub const SECTION_OFF_SCALER_SCALE: usize = 40;

/// Header byte offset of the `layer_table` [`crate::Section`].
pub const SECTION_OFF_LAYER_TABLE: usize = 48;

/// Header byte offset of the `feature_bounds` [`crate::Section`].
pub const SECTION_OFF_FEATURE_BOUNDS: usize = 56;

/// Header byte offset of the `metadata` [`crate::Section`].
pub const SECTION_OFF_METADATA: usize = 64;

/// Header byte offset of the `output_specs` [`crate::Section`] (v3+).
pub const SECTION_OFF_OUTPUT_SPECS: usize = 72;

/// Header byte offset of the `discrete_sets` [`crate::Section`] (v3+).
pub const SECTION_OFF_DISCRETE_SETS: usize = 80;

/// Header byte offset of the `sparse_overrides` [`crate::Section`] (v3+).
pub const SECTION_OFF_SPARSE_OVERRIDES: usize = 88;

/// Header byte offset of `decompressed_payload_len: u32` (v3.1+).
/// Meaningful when `flags` bit 0 is set; zero otherwise.
pub const OFF_DECOMPRESSED_PAYLOAD_LEN: usize = 96;

/// Header byte offset of the `feature_order` [`crate::Section`] (v3.1+).
/// Empty (len=0) when the bake's inputs are in caller-natural order.
pub const SECTION_OFF_FEATURE_ORDER: usize = 100;

/// Header byte offset of the `output_order` [`crate::Section`] (v3.1+).
/// Empty (len=0) when the bake's outputs are in caller-natural order.
pub const SECTION_OFF_OUTPUT_ORDER: usize = 108;

/// Header byte offset of the `multi_codec_schema` [`crate::Section`]
/// (v3.2+). Empty (`len == 0`) for single-codec bakes. When present,
/// the bake is a multi-codec shared-trunk picker whose inputs are
/// the union of all participating codecs' image features (scattered
/// by per-codec maps), plus a presence mask, size onehot,
/// log-pixels, zq_norm, and a codec onehot. The runtime composes
/// the input vector via [`crate::Predictor::predict_multi_codec`].
///
/// Section payload layout (after section start, all little-endian):
///
/// ```text
/// 0..4    union_feat_count: u32       n_union image features
/// 4..8    n_codecs: u32
/// 8..(8 + n_codecs*32)  PerCodecMapEntry[n_codecs]
///   Each entry (32 bytes):
///     0..4   name_offset: u32         offset relative to section start
///     4..8   name_len: u32
///     8..12  slots_offset: u32        offset relative to section start, 4-aligned
///     12..16 slots_count: u32         number of u32 entries
///     16..20 output_range_lo: u32     inclusive lo into the trunk output
///     20..24 output_range_hi: u32     exclusive hi into the trunk output
///     24..28 head_n_cells: u32        per-codec head metadata (informational)
///     28..32 head_n_heads: u32        bytes-head + scalar-head count
/// (then) name pool (utf8) + slot tables (u32 arrays), each
///        padded to 4-byte alignment.
/// ```
pub const SECTION_OFF_MULTI_CODEC_SCHEMA: usize = 116;

// ── Header flags bits (offset 6..8) ──────────────────────────────

/// Mask for `flags` bit 0: when set, payload bytes [128..end] are
/// compressed. The algorithm is encoded in `FLAGS_COMPRESSION_ALGO_MASK`.
pub const FLAG_COMPRESSED: u16 = 0x0001;

/// Mask for `flags` bits 1..3: 3-bit compression algorithm field.
/// Shift right by 1 after masking to get the algo enum.
pub const FLAGS_COMPRESSION_ALGO_MASK: u16 = 0x000E;

/// Compression algorithm: no compression.
pub const COMPRESSION_ALGO_NONE: u8 = 0;

/// Compression algorithm: LZ4 block format (`lz4_flex::block`).
pub const COMPRESSION_ALGO_LZ4: u8 = 1;
