//! Metadata-append primitive: splice one UTF-8 metadata entry into an
//! already-serialized ZNPR v3 bake **without re-encoding any model
//! content**.
//!
//! [`crate::cli`]'s `repack` path parses a bake into f32 and re-bakes
//! it — score-neutral, but not byte-exact for f16/i8 layers (scales
//! get recomputed). Tooling that only wants to attach provenance
//! (e.g. zensim's trainer stamping `zentrain.repro` on finished
//! bakes) needs a hard guarantee that scoring behavior is untouched.
//! [`append_metadata_utf8`] provides it: the operation is a
//! section-level splice — every section other than the metadata blob
//! is copied **byte-verbatim** (weights, I8 scales, biases, scaler,
//! bounds, output_specs, discrete_sets, sparse_overrides,
//! feature_order / output_order permutations), only the metadata
//! section is re-serialized, subsequent section offsets are shifted
//! with the composer's own alignment rules, and the header's section
//! table is patched. For a composer-produced input, the output is
//! byte-identical to what [`crate::bake`] would have emitted with the
//! extended metadata list (locked by tests below).
//!
//! Compressed bakes (header flag bit 0) are handled transparently:
//! payload is LZ4-decompressed, spliced, recompressed, and
//! `decompressed_payload_len` is updated; the output stays compressed.

use alloc::vec::Vec;
use core::fmt;

use zenpredict::wire::{
    COMPRESSION_ALGO_LZ4, FLAG_COMPRESSED, FLAGS_COMPRESSION_ALGO_MASK, HEADER_SIZE,
    LAYER_ENTRY_SIZE, OFF_DECOMPRESSED_PAYLOAD_LEN, SECTION_OFF_DISCRETE_SETS,
    SECTION_OFF_FEATURE_BOUNDS, SECTION_OFF_FEATURE_ORDER, SECTION_OFF_LAYER_TABLE,
    SECTION_OFF_METADATA, SECTION_OFF_OUTPUT_ORDER, SECTION_OFF_OUTPUT_SPECS,
    SECTION_OFF_SCALER_MEAN, SECTION_OFF_SCALER_SCALE, SECTION_OFF_SPARSE_OVERRIDES,
};
use zenpredict::{FORMAT_VERSION, Metadata, MetadataType, PredictError};

/// Errors raised by [`append_metadata_utf8`]. Distinct from
/// [`crate::BakeError`] (bake-side input validation) — these are
/// splice-side failures on an existing byte stream.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AppendError {
    /// The new entry's key is empty (mirrors `BakeError::MetadataKeyEmpty`).
    KeyEmpty,
    /// The new entry's key exceeds the wire format's u8 length prefix.
    KeyTooLong { len: usize },
    /// The new entry's value exceeds the wire format's u32 length prefix.
    ValueTooLong { len: usize },
    /// Input shorter than the fixed 128-byte header, or larger than
    /// the loader's [`zenpredict::limits::MAX_BAKE_BYTES`] cap.
    Truncated { want: usize, have: usize },
    /// Input doesn't start with the `ZNPR` magic.
    BadMagic { found: [u8; 4] },
    /// Input is a ZNPR bake of a version this splice doesn't speak
    /// (v1/v2 bakes must be migrated to v3 first; future versions
    /// need a matching zenpredict-bake).
    UnsupportedVersion { version: u16, expected: u16 },
    /// Compressed flag set with an algorithm nibble other than LZ4.
    UnsupportedCompressionAlgo { algo: u8 },
    /// LZ4 payload failed to decompress to the header-declared
    /// `decompressed_payload_len`.
    DecompressFailed,
    /// A structural header/section-table field is inconsistent
    /// (e.g. layer-table length disagrees with `n_layers`, zero
    /// `decompressed_payload_len` on a compressed bake).
    Malformed { what: &'static str },
    /// A section addresses bytes outside the (decompressed) file.
    SectionOutOfRange { what: &'static str },
    /// Sections overlap each other, the fixed header, or the
    /// metadata blob in a way the splice can't preserve. Composer-
    /// produced bakes never trip this; a foreign emitter with an
    /// exotic physical layout might.
    UnsupportedLayout { what: &'static str },
    /// The existing metadata blob failed to parse under the runtime's
    /// [`zenpredict::Metadata::parse`] contract.
    InvalidMetadata(PredictError),
    /// An existing metadata entry carries a [`MetadataType`] variant
    /// this build can't map back to a wire byte (future zenpredict).
    UnsupportedMetadataKind,
    /// Result would exceed u32 section addressing.
    TooLarge { len: usize },
    /// Post-splice self-check failed: the output did not round-trip
    /// through [`zenpredict::Model::from_bytes`]. Indicates the input
    /// was not a loadable bake to begin with (the splice validates
    /// its output, not its input, beyond the section table).
    OutputValidation(PredictError),
}

impl fmt::Display for AppendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyEmpty => write!(f, "append: metadata key is empty"),
            Self::KeyTooLong { len } => {
                write!(f, "append: metadata key length {len} exceeds u8 max (255)")
            }
            Self::ValueTooLong { len } => {
                write!(f, "append: metadata value length {len} exceeds u32 max")
            }
            Self::Truncated { want, have } => {
                write!(f, "append: input length {have} outside expected {want}")
            }
            Self::BadMagic { found } => write!(f, "append: bad magic {found:02x?}"),
            Self::UnsupportedVersion { version, expected } => write!(
                f,
                "append: ZNPR version {version} unsupported (expected {expected}; \
                 migrate v1/v2 bakes via zentrain/tools/migrate_znpr_v2_to_v3.py)"
            ),
            Self::UnsupportedCompressionAlgo { algo } => {
                write!(f, "append: unsupported compression algo nibble {algo}")
            }
            Self::DecompressFailed => write!(
                f,
                "append: LZ4 payload did not decompress to the declared length"
            ),
            Self::Malformed { what } => write!(f, "append: malformed bake: {what}"),
            Self::SectionOutOfRange { what } => {
                write!(f, "append: section out of range: {what}")
            }
            Self::UnsupportedLayout { what } => {
                write!(f, "append: unsupported physical layout: {what}")
            }
            Self::InvalidMetadata(e) => write!(f, "append: metadata blob unparseable: {e:?}"),
            Self::UnsupportedMetadataKind => write!(
                f,
                "append: existing metadata entry has a kind this build can't re-serialize"
            ),
            Self::TooLarge { len } => {
                write!(f, "append: result length {len} exceeds u32 addressing")
            }
            Self::OutputValidation(e) => {
                write!(
                    f,
                    "append: output failed Model::from_bytes self-check: {e:?}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AppendError {}

/// Where a region's `(offset, len)` pair lives, so the splice can
/// patch it after shifting the region.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionId {
    /// One of the fixed header-table sections; `table_off` is the
    /// byte offset of its `Section` entry inside the 128-byte header.
    Header { table_off: usize },
    /// A per-layer payload; the `Section` entry lives at
    /// `layer_table.offset + idx * LAYER_ENTRY_SIZE + field_off`.
    Layer { idx: usize, field_off: usize },
}

/// A non-empty data section: physical placement + the alignment the
/// composer would give it + where to patch its table entry.
#[derive(Clone, Copy, Debug)]
struct Region {
    off: usize,
    len: usize,
    align: usize,
    id: RegionId,
}

impl Region {
    fn end(&self) -> usize {
        self.off + self.len
    }
}

/// Append (or replace, if the key exists) one UTF-8 metadata entry on
/// an already-serialized ZNPR v3 bake, leaving every other byte of
/// model content semantically untouched (weights / I8 scales / biases
/// / scaler / feature_bounds / feature_transforms / output_specs /
/// discrete_sets / sparse_overrides / feature_order / output_order
/// preserved EXACTLY; no re-quantization, no HU re-reorder).
///
/// Semantics:
/// - Key absent → the new entry is appended after all existing
///   entries, preserving their order.
/// - Key present → the entry is replaced **in place** (same position,
///   kind forced to [`MetadataType::Utf8`], new value). At most one
///   entry per key results even if the input carried duplicates.
/// - Compressed bakes stay compressed (payload is decompressed,
///   spliced, recompressed; `decompressed_payload_len` updated).
/// - v1/v2 bakes and non-ZNPR inputs return a clean `Err` — nothing
///   is ever silently rewritten across versions.
///
/// For inputs produced by this crate's [`crate::bake`], the output is
/// byte-identical to re-baking with the extended metadata list; for
/// foreign v3 emitters with a valid section table, all non-metadata
/// sections are still copied verbatim (their offsets may shift). The
/// output is self-checked through [`zenpredict::Model::from_bytes`]
/// before it is returned, so a successful return is guaranteed
/// loadable.
pub fn append_metadata_utf8(bytes: &[u8], key: &str, value: &str) -> Result<Vec<u8>, AppendError> {
    // ── New-entry validation (mirrors the composer's rules). ──────
    if key.is_empty() {
        return Err(AppendError::KeyEmpty);
    }
    if key.len() > 255 {
        return Err(AppendError::KeyTooLong { len: key.len() });
    }
    if u32::try_from(value.len()).is_err() {
        return Err(AppendError::ValueTooLong { len: value.len() });
    }

    // ── Header sanity. ─────────────────────────────────────────────
    if bytes.len() < HEADER_SIZE || bytes.len() > zenpredict::limits::MAX_BAKE_BYTES {
        return Err(AppendError::Truncated {
            want: HEADER_SIZE,
            have: bytes.len(),
        });
    }
    if &bytes[0..4] != b"ZNPR" {
        return Err(AppendError::BadMagic {
            found: [bytes[0], bytes[1], bytes[2], bytes[3]],
        });
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != FORMAT_VERSION {
        return Err(AppendError::UnsupportedVersion {
            version,
            expected: FORMAT_VERSION,
        });
    }
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    let compressed = (flags & FLAG_COMPRESSED) != 0;

    // ── Materialize the uncompressed working image. ────────────────
    let working: Vec<u8> = if compressed {
        let algo = ((flags & FLAGS_COMPRESSION_ALGO_MASK) >> 1) as u8;
        if algo != COMPRESSION_ALGO_LZ4 {
            return Err(AppendError::UnsupportedCompressionAlgo { algo });
        }
        let payload_len = read_u32(bytes, OFF_DECOMPRESSED_PAYLOAD_LEN) as usize;
        if payload_len == 0 {
            return Err(AppendError::Malformed {
                what: "decompressed_payload_len is zero on a compressed bake",
            });
        }
        let total = HEADER_SIZE
            .checked_add(payload_len)
            .filter(|&t| t <= zenpredict::limits::MAX_BAKE_BYTES)
            .ok_or(AppendError::Truncated {
                want: zenpredict::limits::MAX_BAKE_BYTES,
                have: payload_len,
            })?;
        let mut w = alloc::vec![0u8; total];
        w[..HEADER_SIZE].copy_from_slice(&bytes[..HEADER_SIZE]);
        let written =
            lz4_flex::block::decompress_into(&bytes[HEADER_SIZE..], &mut w[HEADER_SIZE..])
                .map_err(|_| AppendError::DecompressFailed)?;
        if written != payload_len {
            return Err(AppendError::DecompressFailed);
        }
        w
    } else {
        bytes.to_vec()
    };

    // ── Section-table walk. ────────────────────────────────────────
    let n_layers = read_u32(&working, 16) as usize;
    let (lt_off, lt_len) = read_section(&working, SECTION_OFF_LAYER_TABLE);
    if lt_len
        != n_layers
            .checked_mul(LAYER_ENTRY_SIZE)
            .ok_or(AppendError::Malformed {
                what: "n_layers overflows the layer table",
            })?
    {
        return Err(AppendError::Malformed {
            what: "layer_table length disagrees with n_layers",
        });
    }
    check_in_bounds(&working, lt_off, lt_len, "layer_table")?;

    // Fixed header-table sections (metadata excluded — it's the one
    // being replaced). All are 4-aligned in the composer's layout.
    const HEADER_DATA_SECTIONS: &[(usize, &str)] = &[
        (SECTION_OFF_SCALER_MEAN, "scaler_mean"),
        (SECTION_OFF_SCALER_SCALE, "scaler_scale"),
        (SECTION_OFF_LAYER_TABLE, "layer_table"),
        (SECTION_OFF_FEATURE_BOUNDS, "feature_bounds"),
        (SECTION_OFF_OUTPUT_SPECS, "output_specs"),
        (SECTION_OFF_DISCRETE_SETS, "discrete_sets"),
        (SECTION_OFF_SPARSE_OVERRIDES, "sparse_overrides"),
        (SECTION_OFF_FEATURE_ORDER, "feature_order"),
        (SECTION_OFF_OUTPUT_ORDER, "output_order"),
    ];

    let mut regions: Vec<Region> = Vec::new();
    for &(table_off, what) in HEADER_DATA_SECTIONS {
        let (off, len) = read_section(&working, table_off);
        if len == 0 {
            continue;
        }
        check_in_bounds(&working, off, len, what)?;
        regions.push(Region {
            off,
            len,
            align: 4,
            id: RegionId::Header { table_off },
        });
    }
    for idx in 0..n_layers {
        let entry_off = lt_off + idx * LAYER_ENTRY_SIZE;
        let dtype = working[entry_off + 9];
        // Weight alignment matches the composer's `pad_to` per dtype.
        let weights_align = match dtype {
            0 => 4, // f32
            1 => 2, // f16
            2 => 1, // i8 (composer writes unpadded)
            _ => {
                return Err(AppendError::Malformed {
                    what: "unknown layer weight dtype",
                });
            }
        };
        for (field_off, align, what) in [
            (12usize, weights_align, "layer weights"),
            (20, 4, "layer scales"),
            (28, 4, "layer biases"),
        ] {
            let (off, len) = read_section(&working, entry_off + field_off);
            if len == 0 {
                continue;
            }
            check_in_bounds(&working, off, len, what)?;
            regions.push(Region {
                off,
                len,
                align,
                id: RegionId::Layer { idx, field_off },
            });
        }
    }

    // No region may reach into the fixed header, and regions must be
    // pairwise disjoint — the splice copies them verbatim and cannot
    // preserve aliasing.
    for r in &regions {
        if r.off < HEADER_SIZE {
            return Err(AppendError::UnsupportedLayout {
                what: "section overlaps the fixed header",
            });
        }
    }
    let mut by_off: Vec<&Region> = regions.iter().collect();
    by_off.sort_by_key(|r| r.off);
    for pair in by_off.windows(2) {
        if pair[0].end() > pair[1].off {
            return Err(AppendError::UnsupportedLayout {
                what: "sections overlap",
            });
        }
    }

    // ── Re-serialize the metadata blob. ────────────────────────────
    let (meta_off, meta_len) = read_section(&working, SECTION_OFF_METADATA);
    let old_blob: &[u8] = if meta_len == 0 {
        &[]
    } else {
        check_in_bounds(&working, meta_off, meta_len, "metadata")?;
        &working[meta_off..meta_off + meta_len]
    };
    let parsed = Metadata::parse(old_blob).map_err(AppendError::InvalidMetadata)?;
    let mut new_blob: Vec<u8> = Vec::with_capacity(meta_len + 1 + key.len() + 1 + 4 + value.len());
    let mut replaced = false;
    for entry in parsed.iter() {
        if entry.key == key {
            // Replace in place; drop any duplicate occurrences.
            if !replaced {
                write_meta_entry(&mut new_blob, key, 1, value.as_bytes());
                replaced = true;
            }
        } else {
            let type_byte = kind_to_byte(entry.kind)?;
            write_meta_entry(&mut new_blob, entry.key, type_byte, entry.value);
        }
    }
    if !replaced {
        write_meta_entry(&mut new_blob, key, 1, value.as_bytes());
    }

    // ── Splice point + tail set. ───────────────────────────────────
    //
    // Non-empty metadata: splice where the old blob sat; the tail is
    // every region at/after its end. Empty metadata: insert at the
    // composer-canonical position — right after the last payload
    // section, before the post-metadata sections (output_specs,
    // discrete_sets, sparse_overrides, feature_order, output_order).
    const POST_METADATA_TABLE_OFFS: [usize; 5] = [
        SECTION_OFF_OUTPUT_SPECS,
        SECTION_OFF_DISCRETE_SETS,
        SECTION_OFF_SPARSE_OVERRIDES,
        SECTION_OFF_FEATURE_ORDER,
        SECTION_OFF_OUTPUT_ORDER,
    ];
    let (splice_at, mut tail): (usize, Vec<Region>) = if meta_len > 0 {
        let meta_end = meta_off + meta_len;
        for r in &regions {
            let disjoint = r.end() <= meta_off || r.off >= meta_end;
            if !disjoint {
                return Err(AppendError::UnsupportedLayout {
                    what: "a section overlaps the metadata blob",
                });
            }
        }
        let tail = regions
            .iter()
            .filter(|r| r.off >= meta_end)
            .copied()
            .collect();
        (meta_off, tail)
    } else {
        let is_post_meta = |r: &Region| {
            matches!(r.id, RegionId::Header { table_off }
                if POST_METADATA_TABLE_OFFS.contains(&table_off))
        };
        let head_end = regions
            .iter()
            .filter(|r| !is_post_meta(r))
            .map(Region::end)
            .max()
            .unwrap_or(HEADER_SIZE)
            .max(HEADER_SIZE);
        let tail: Vec<Region> = regions
            .iter()
            .filter(|r| is_post_meta(r))
            .copied()
            .collect();
        if tail.iter().any(|r| r.off < head_end) {
            return Err(AppendError::UnsupportedLayout {
                what: "a post-metadata section precedes a payload section",
            });
        }
        (head_end, tail)
    };
    tail.sort_by_key(|r| r.off);

    // ── Rebuild: verbatim prefix + new blob + realigned tail. ──────
    let mut out: Vec<u8> = Vec::with_capacity(working.len() + new_blob.len() + 8);
    out.extend_from_slice(&working[..splice_at]);
    let meta_new_off = out.len();
    out.extend_from_slice(&new_blob);
    let mut moved: Vec<(Region, usize)> = Vec::with_capacity(tail.len());
    for r in &tail {
        pad_to(&mut out, r.align);
        let new_off = out.len();
        out.extend_from_slice(&working[r.off..r.end()]);
        moved.push((*r, new_off));
    }
    if u32::try_from(out.len()).is_err() {
        return Err(AppendError::TooLarge { len: out.len() });
    }

    // ── Patch the tables. ──────────────────────────────────────────
    write_section_raw(
        &mut out,
        SECTION_OFF_METADATA,
        meta_new_off as u32,
        new_blob.len() as u32,
    );
    // Resolve the layer table's (possibly shifted) offset before
    // patching any layer entry through it.
    let mut lt_new_off = lt_off;
    for (r, new_off) in &moved {
        if r.id
            == (RegionId::Header {
                table_off: SECTION_OFF_LAYER_TABLE,
            })
        {
            lt_new_off = *new_off;
        }
    }
    for (r, new_off) in &moved {
        match r.id {
            RegionId::Header { table_off } => {
                write_section_raw(&mut out, table_off, *new_off as u32, r.len as u32);
            }
            RegionId::Layer { idx, field_off } => {
                let at = lt_new_off + idx * LAYER_ENTRY_SIZE + field_off;
                write_section_raw(&mut out, at, *new_off as u32, r.len as u32);
            }
        }
    }

    // ── Recompress when the input was compressed. ──────────────────
    let final_out = if compressed {
        let payload_len = out.len() - HEADER_SIZE;
        let compressed_payload = lz4_flex::block::compress(&out[HEADER_SIZE..]);
        let mut f = Vec::with_capacity(HEADER_SIZE + compressed_payload.len());
        f.extend_from_slice(&out[..HEADER_SIZE]);
        f.extend_from_slice(&compressed_payload);
        // Flags (compressed bit + algo nibble) came through verbatim
        // from the input header; only the payload length changes.
        f[OFF_DECOMPRESSED_PAYLOAD_LEN..OFF_DECOMPRESSED_PAYLOAD_LEN + 4]
            .copy_from_slice(&(payload_len as u32).to_le_bytes());
        f
    } else {
        out
    };

    // ── Self-check: a successful return is guaranteed loadable. ───
    zenpredict::Model::from_bytes(&final_out).map_err(AppendError::OutputValidation)?;
    Ok(final_out)
}

// ───── Byte helpers ────────────────────────────────────────────────

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_section(bytes: &[u8], table_off: usize) -> (usize, usize) {
    (
        read_u32(bytes, table_off) as usize,
        read_u32(bytes, table_off + 4) as usize,
    )
}

fn write_section_raw(buf: &mut [u8], at: usize, off: u32, len: u32) {
    buf[at..at + 4].copy_from_slice(&off.to_le_bytes());
    buf[at + 4..at + 8].copy_from_slice(&len.to_le_bytes());
}

fn check_in_bounds(
    bytes: &[u8],
    off: usize,
    len: usize,
    what: &'static str,
) -> Result<(), AppendError> {
    off.checked_add(len)
        .filter(|&end| end <= bytes.len())
        .map(|_| ())
        .ok_or(AppendError::SectionOutOfRange { what })
}

/// Map a parsed [`MetadataType`] back to its wire byte. Total for
/// every kind [`Metadata::parse`] can currently produce.
fn kind_to_byte(kind: MetadataType) -> Result<u8, AppendError> {
    match kind {
        MetadataType::Bytes => Ok(0),
        MetadataType::Utf8 => Ok(1),
        MetadataType::Numeric => Ok(2),
        MetadataType::Reserved(b) => Ok(b),
        // MetadataType is #[non_exhaustive]: a future zenpredict
        // variant without a Reserved round-trip must fail loudly
        // rather than ship a mis-typed entry.
        _ => Err(AppendError::UnsupportedMetadataKind),
    }
}

/// Serialize one entry with the composer's encoding:
/// `[1] key_len, [key_len] key, [1] value_type, [4] value_len LE, [value_len] value`.
/// Length invariants hold by construction: existing entries came from
/// a parse whose prefixes fit, and the new entry is validated up front.
fn write_meta_entry(buf: &mut Vec<u8>, key: &str, type_byte: u8, value: &[u8]) {
    buf.push(key.len() as u8);
    buf.extend_from_slice(key.as_bytes());
    buf.push(type_byte);
    buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
    buf.extend_from_slice(value);
}

/// Composer-identical zero padding.
fn pad_to(buf: &mut Vec<u8>, alignment: usize) {
    let rem = buf.len() % alignment;
    if rem != 0 {
        for _ in 0..(alignment - rem) {
            buf.push(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composer::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};
    use zenpredict::{
        Activation, FeatureBound, Model, OutputSpec, Predictor, SparseOverride, WeightDtype,
    };

    const N_IN: usize = 8;
    const HID: usize = 6;

    fn weights0() -> Vec<f32> {
        (0..N_IN * HID)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect()
    }
    fn biases0() -> Vec<f32> {
        (0..HID).map(|i| (i as f32 * 0.11).cos() * 0.1).collect()
    }
    fn weights1() -> Vec<f32> {
        (0..HID * 2)
            .map(|i| (i as f32 * 0.73).cos() * 0.4)
            .collect()
    }
    fn biases1() -> Vec<f32> {
        alloc::vec![0.05f32, -0.02]
    }

    struct Fixture {
        w0: Vec<f32>,
        b0: Vec<f32>,
        w1: Vec<f32>,
        b1: Vec<f32>,
        mean: Vec<f32>,
        scale: Vec<f32>,
        bounds: Vec<FeatureBound>,
        specs: Vec<OutputSpec>,
        pool: Vec<f32>,
        overrides: Vec<SparseOverride>,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                w0: weights0(),
                b0: biases0(),
                w1: weights1(),
                b1: biases1(),
                mean: (0..N_IN).map(|i| i as f32 * 0.01).collect(),
                scale: (0..N_IN).map(|i| 1.0 + i as f32 * 0.1).collect(),
                bounds: (0..N_IN)
                    .map(|i| FeatureBound::new(-1.0 - i as f32, 1.0 + i as f32))
                    .collect(),
                specs: alloc::vec![OutputSpec::passthrough(), OutputSpec::passthrough()],
                pool: alloc::vec![0.0f32, 0.5, 1.0],
                overrides: alloc::vec![SparseOverride::new(1, 42.0)],
            }
        }

        fn layers(&self, dtype: WeightDtype) -> [BakeLayer<'_>; 2] {
            [
                BakeLayer {
                    in_dim: N_IN,
                    out_dim: HID,
                    activation: Activation::LeakyRelu,
                    dtype,
                    weights: &self.w0,
                    biases: &self.b0,
                },
                BakeLayer {
                    in_dim: HID,
                    out_dim: 2,
                    activation: Activation::Identity,
                    dtype,
                    weights: &self.w1,
                    biases: &self.b1,
                },
            ]
        }
    }

    /// Standard pre-existing metadata: utf8 + bytes + numeric kinds.
    fn base_metadata() -> Vec<(&'static str, MetadataType, Vec<u8>)> {
        alloc::vec![
            (
                "zentrain.bake_name",
                MetadataType::Utf8,
                b"append_fixture_v1".to_vec(),
            ),
            (
                "zenjpeg.blob",
                MetadataType::Bytes,
                alloc::vec![1u8, 2, 3, 4, 5]
            ),
            (
                "zentrain.feature_defs_version",
                MetadataType::Numeric,
                7u32.to_le_bytes().to_vec(),
            ),
            (
                zenpredict::keys::FEATURE_TRANSFORMS,
                MetadataType::Utf8,
                core::iter::repeat_n("identity", N_IN)
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes(),
            ),
        ]
    }

    fn as_entries<'a>(
        meta: &'a [(&'static str, MetadataType, Vec<u8>)],
    ) -> Vec<BakeMetadataEntry<'a>> {
        meta.iter()
            .map(|(key, kind, value)| BakeMetadataEntry {
                key,
                kind: *kind,
                value,
            })
            .collect()
    }

    /// Build the "rich" fixture bake: every optional section present.
    fn bake_rich(
        fx: &Fixture,
        dtype: WeightDtype,
        meta: &[(&'static str, MetadataType, Vec<u8>)],
        compressed: bool,
        with_orders: bool,
    ) -> Vec<u8> {
        let layers = fx.layers(dtype);
        let entries = as_entries(meta);
        let feature_order: Vec<u32> = (0..N_IN as u32).rev().collect();
        let output_order: Vec<u32> = alloc::vec![1, 0];
        let mut b = BakeRequest::builder(0xfeed_beef, 0, &fx.mean, &fx.scale, &layers)
            .feature_bounds(&fx.bounds)
            .metadata(&entries)
            .output_specs(&fx.specs)
            .discrete_sets(&fx.pool)
            .sparse_overrides(&fx.overrides)
            .compressed(compressed);
        if with_orders {
            b = b.feature_order(&feature_order).output_order(&output_order);
        }
        b.bake().expect("fixture bake")
    }

    fn feature_vectors() -> Vec<[f32; N_IN]> {
        (0..10)
            .map(|k| {
                let mut v = [0.0f32; N_IN];
                for (i, x) in v.iter_mut().enumerate() {
                    *x = ((k * N_IN + i) as f32 * 0.619).sin() * 2.0;
                }
                v
            })
            .collect()
    }

    fn predict_all(bytes: &[u8]) -> Vec<Vec<u32>> {
        let model = Model::from_bytes(bytes).expect("model parses");
        let mut p = Predictor::new(&model);
        feature_vectors()
            .iter()
            .map(|v| p.predict(v).unwrap().iter().map(|f| f.to_bits()).collect())
            .collect()
    }

    /// Compare every section EXCEPT metadata byte-for-byte between two
    /// (uncompressed) bakes, resolving each side through its own
    /// section table. This is the "no requantization" proof.
    fn assert_non_metadata_sections_identical(a: &[u8], b: &[u8]) {
        for &(table_off, what) in &[
            (SECTION_OFF_SCALER_MEAN, "scaler_mean"),
            (SECTION_OFF_SCALER_SCALE, "scaler_scale"),
            (SECTION_OFF_LAYER_TABLE, "layer_table_dims"),
            (SECTION_OFF_FEATURE_BOUNDS, "feature_bounds"),
            (SECTION_OFF_OUTPUT_SPECS, "output_specs"),
            (SECTION_OFF_DISCRETE_SETS, "discrete_sets"),
            (SECTION_OFF_SPARSE_OVERRIDES, "sparse_overrides"),
            (SECTION_OFF_FEATURE_ORDER, "feature_order"),
            (SECTION_OFF_OUTPUT_ORDER, "output_order"),
        ] {
            let (a_off, a_len) = read_section(a, table_off);
            let (b_off, b_len) = read_section(b, table_off);
            assert_eq!(a_len, b_len, "{what}: length changed");
            if what == "layer_table_dims" {
                // The layer table embeds weights/scales/biases offsets
                // which legitimately shift; compare the non-Section
                // fields (dims, activation, dtype, flags, reserved)
                // and the payload bytes below.
                let n = a_len / LAYER_ENTRY_SIZE;
                for i in 0..n {
                    let ea = &a[a_off + i * LAYER_ENTRY_SIZE..a_off + (i + 1) * LAYER_ENTRY_SIZE];
                    let eb = &b[b_off + i * LAYER_ENTRY_SIZE..b_off + (i + 1) * LAYER_ENTRY_SIZE];
                    assert_eq!(&ea[0..12], &eb[0..12], "layer {i} dims/activation/dtype");
                    assert_eq!(&ea[36..48], &eb[36..48], "layer {i} reserved");
                    for (field_off, name) in [(12usize, "weights"), (20, "scales"), (28, "biases")]
                    {
                        let (wa_off, wa_len) = read_section(ea, field_off);
                        let (wb_off, wb_len) = read_section(eb, field_off);
                        assert_eq!(wa_len, wb_len, "layer {i} {name}: length changed");
                        assert_eq!(
                            &a[wa_off..wa_off + wa_len],
                            &b[wb_off..wb_off + wb_len],
                            "layer {i} {name}: bytes changed"
                        );
                    }
                }
            } else {
                assert_eq!(
                    &a[a_off..a_off + a_len],
                    &b[b_off..b_off + b_len],
                    "{what}: bytes changed"
                );
            }
        }
    }

    #[test]
    fn append_then_parse_reads_new_key_and_preserves_existing() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F32, &meta, false, false);
        let after = append_metadata_utf8(&before, "zentrain.repro", "cmd=train --seed 7").unwrap();

        let m_before = Model::from_bytes(&before).unwrap();
        let m_after = Model::from_bytes(&after).unwrap();

        // New key readable, typed utf8.
        assert_eq!(
            m_after.metadata().get_utf8("zentrain.repro").unwrap(),
            "cmd=train --seed 7"
        );

        // Pre-existing entries present, same order, same kind + bytes;
        // the new entry is last.
        let before_entries: Vec<_> = m_before
            .metadata()
            .iter()
            .map(|e| (alloc::string::String::from(e.key), e.kind, e.value.to_vec()))
            .collect();
        let after_entries: Vec<_> = m_after
            .metadata()
            .iter()
            .map(|e| (alloc::string::String::from(e.key), e.kind, e.value.to_vec()))
            .collect();
        assert_eq!(after_entries.len(), before_entries.len() + 1);
        assert_eq!(&after_entries[..before_entries.len()], &before_entries[..]);
        assert_eq!(after_entries.last().unwrap().0, "zentrain.repro");
        assert_eq!(after_entries.last().unwrap().1, MetadataType::Utf8);

        // Model shape + auxiliary sections unchanged.
        assert_eq!(m_after.n_inputs(), m_before.n_inputs());
        assert_eq!(m_after.n_outputs(), m_before.n_outputs());
        assert_eq!(m_after.n_layers(), m_before.n_layers());
        assert_eq!(m_after.schema_hash(), m_before.schema_hash());
        assert_eq!(m_after.scaler_mean(), m_before.scaler_mean());
        assert_eq!(m_after.scaler_scale(), m_before.scaler_scale());
        assert_eq!(m_after.feature_bounds(), m_before.feature_bounds());
        assert_eq!(m_after.feature_transforms(), m_before.feature_transforms());
        assert_eq!(m_after.output_specs().len(), m_before.output_specs().len());
        assert_eq!(m_after.discrete_sets(), m_before.discrete_sets());
        assert_eq!(
            m_after.sparse_overrides().len(),
            m_before.sparse_overrides().len()
        );

        assert_non_metadata_sections_identical(&before, &after);
    }

    #[test]
    fn score_identity_f32_bitwise() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F32, &meta, false, false);
        let after = append_metadata_utf8(&before, "zentrain.repro", "run-42").unwrap();
        assert_eq!(predict_all(&before), predict_all(&after));
    }

    #[test]
    fn section_bytes_identical_f16() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F16, &meta, false, false);
        let after = append_metadata_utf8(&before, "zentrain.repro", "f16 case").unwrap();
        assert_non_metadata_sections_identical(&before, &after);
        assert_eq!(predict_all(&before), predict_all(&after));
    }

    #[test]
    fn section_bytes_identical_i8_no_requantization() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::I8, &meta, false, false);
        let after = append_metadata_utf8(&before, "zentrain.repro", "i8 case").unwrap();
        // Covers the I8 scales section explicitly via the layer walk.
        assert_non_metadata_sections_identical(&before, &after);
        assert_eq!(predict_all(&before), predict_all(&after));
    }

    #[test]
    fn replace_existing_key_keeps_one_entry_latest_value_in_place() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F32, &meta, false, false);
        let once = append_metadata_utf8(&before, "zentrain.repro", "first").unwrap();
        let twice = append_metadata_utf8(&once, "zentrain.repro", "second").unwrap();

        let model = Model::from_bytes(&twice).unwrap();
        let hits: Vec<_> = model
            .metadata()
            .iter()
            .filter(|e| e.key == "zentrain.repro")
            .map(|e| e.value.to_vec())
            .collect();
        assert_eq!(hits.len(), 1, "exactly one entry after re-append");
        assert_eq!(hits[0], b"second");
        // In-place: position (index) preserved from the first append.
        let pos = model
            .metadata()
            .iter()
            .position(|e| e.key == "zentrain.repro")
            .unwrap();
        assert_eq!(pos, base_metadata().len(), "entry stays at its slot");
        assert_non_metadata_sections_identical(&before, &twice);
        assert_eq!(predict_all(&before), predict_all(&twice));
    }

    #[test]
    fn compressed_bake_appends_and_stays_compressed() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F32, &meta, true, false);
        let after = append_metadata_utf8(&before, "zentrain.repro", "lz4 case").unwrap();

        let flags = u16::from_le_bytes([after[6], after[7]]);
        assert_ne!(
            flags & FLAG_COMPRESSED,
            0,
            "output still flagged compressed"
        );
        assert_eq!(
            ((flags & FLAGS_COMPRESSION_ALGO_MASK) >> 1) as u8,
            COMPRESSION_ALGO_LZ4
        );

        let model = Model::from_bytes(&after).unwrap();
        assert_eq!(
            model.metadata().get_utf8("zentrain.repro").unwrap(),
            "lz4 case"
        );
        assert_eq!(predict_all(&before), predict_all(&after));

        // Section-level identity on the DECOMPRESSED images.
        let decompress = |bytes: &[u8]| -> Vec<u8> {
            let payload_len = read_u32(bytes, OFF_DECOMPRESSED_PAYLOAD_LEN) as usize;
            let mut w = alloc::vec![0u8; HEADER_SIZE + payload_len];
            w[..HEADER_SIZE].copy_from_slice(&bytes[..HEADER_SIZE]);
            lz4_flex::block::decompress_into(&bytes[HEADER_SIZE..], &mut w[HEADER_SIZE..]).unwrap();
            w
        };
        assert_non_metadata_sections_identical(&decompress(&before), &decompress(&after));
    }

    /// The gold gate: for composer-produced inputs, append must emit
    /// EXACTLY the bytes `bake()` would have produced with the
    /// extended metadata list — alignment, padding, offsets and all.
    #[test]
    fn matches_composer_rebake_byte_for_byte() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let key = "zentrain.repro";
        let value = "grid=q0..q100 corpus=canonical-2026-07-15";

        let mut extended = base_metadata();
        extended.push((key, MetadataType::Utf8, value.as_bytes().to_vec()));

        for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::I8] {
            for compressed in [false, true] {
                for with_orders in [false, true] {
                    let before = bake_rich(&fx, dtype, &meta, compressed, with_orders);
                    let spliced = append_metadata_utf8(&before, key, value).unwrap();
                    let rebaked = bake_rich(&fx, dtype, &extended, compressed, with_orders);
                    assert_eq!(
                        spliced, rebaked,
                        "splice != composer rebake for {dtype:?} compressed={compressed} \
                         with_orders={with_orders}"
                    );
                }
            }
        }
    }

    /// Replace path is also composer-exact: swapping a value in the
    /// metadata list and rebaking equals splicing onto the old bake.
    #[test]
    fn replace_matches_composer_rebake_byte_for_byte() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F32, &meta, false, false);
        let spliced =
            append_metadata_utf8(&before, "zentrain.bake_name", "renamed_fixture").unwrap();

        let mut swapped = base_metadata();
        swapped[0].2 = b"renamed_fixture".to_vec();
        let rebaked = bake_rich(&fx, WeightDtype::F32, &swapped, false, false);
        assert_eq!(spliced, rebaked);
    }

    /// A bake with NO metadata section at all gets its first entry at
    /// the composer-canonical position (before output_specs et al.).
    #[test]
    fn empty_metadata_bake_gets_first_entry_composer_exact() {
        let fx = Fixture::new();
        let empty: Vec<(&'static str, MetadataType, Vec<u8>)> = Vec::new();
        let only_new = alloc::vec![(
            "zentrain.repro",
            MetadataType::Utf8,
            b"first entry".to_vec()
        )];
        for with_orders in [false, true] {
            let before = bake_rich(&fx, WeightDtype::F32, &empty, false, with_orders);
            let (off, len) = read_section(&before, SECTION_OFF_METADATA);
            assert_eq!((off, len), (0, 0), "fixture really has no metadata");
            let spliced = append_metadata_utf8(&before, "zentrain.repro", "first entry").unwrap();
            let rebaked = bake_rich(&fx, WeightDtype::F32, &only_new, false, with_orders);
            assert_eq!(spliced, rebaked, "with_orders={with_orders}");
            let model = Model::from_bytes(&spliced).unwrap();
            assert_eq!(
                model.metadata().get_utf8("zentrain.repro").unwrap(),
                "first entry"
            );
        }
    }

    /// Minimal bake: no optional sections at all — metadata lands at
    /// the end of the payload.
    #[test]
    fn minimal_bake_append_matches_rebake() {
        let fx = Fixture::new();
        let layers = fx.layers(WeightDtype::F32);
        let before = bake(&BakeRequest::new(1, 0, &fx.mean, &fx.scale, &layers)).unwrap();
        let spliced = append_metadata_utf8(&before, "k", "v").unwrap();

        let entries = [BakeMetadataEntry {
            key: "k",
            kind: MetadataType::Utf8,
            value: b"v",
        }];
        let layers2 = fx.layers(WeightDtype::F32);
        let rebaked = BakeRequest::builder(1, 0, &fx.mean, &fx.scale, &layers2)
            .metadata(&entries)
            .bake()
            .unwrap();
        assert_eq!(spliced, rebaked);
    }

    #[test]
    fn v2_bake_is_a_clean_error() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let mut v2 = bake_rich(&fx, WeightDtype::F32, &meta, false, false);
        v2[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            append_metadata_utf8(&v2, "k", "v"),
            Err(AppendError::UnsupportedVersion {
                version: 2,
                expected: FORMAT_VERSION
            })
        );
    }

    #[test]
    fn bad_magic_and_truncated_are_clean_errors() {
        assert!(matches!(
            append_metadata_utf8(b"nope", "k", "v"),
            Err(AppendError::Truncated { .. })
        ));
        let mut junk = alloc::vec![0u8; HEADER_SIZE];
        junk[0..4].copy_from_slice(b"JUNK");
        assert!(matches!(
            append_metadata_utf8(&junk, "k", "v"),
            Err(AppendError::BadMagic { .. })
        ));
    }

    #[test]
    fn key_validation_mirrors_composer() {
        let fx = Fixture::new();
        let meta = base_metadata();
        let before = bake_rich(&fx, WeightDtype::F32, &meta, false, false);
        assert_eq!(
            append_metadata_utf8(&before, "", "v"),
            Err(AppendError::KeyEmpty)
        );
        let long = "k".repeat(256);
        assert_eq!(
            append_metadata_utf8(&before, &long, "v"),
            Err(AppendError::KeyTooLong { len: 256 })
        );
    }
}
