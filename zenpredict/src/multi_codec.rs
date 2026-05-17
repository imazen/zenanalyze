//! Multi-codec shared-trunk picker schema (ZNPR v3.2+).
//!
//! A multi-codec bake is one shared MLP trunk trained jointly over
//! the **union** of all participating codecs' image features. Each
//! codec contributes its natural feature vector at inference; the
//! runtime scatters that vector into the union slots, builds a
//! presence mask, appends `(size_onehot, log_pixels, zq_norm,
//! codec_onehot)`, and runs the trunk forward pass once. The output
//! has separate ranges (heads) per codec — argmin within the
//! caller's range selects that codec's pick.
//!
//! ## When this exists
//!
//! - **Multi-codec bake**: `Header.multi_codec_schema.len_bytes() > 0`.
//!   [`crate::Model::multi_codec_schema`] returns `Some(_)`.
//!   [`crate::Predictor::predict_multi_codec`] is callable.
//! - **Single-codec bake (default)**: the section is absent. All
//!   existing single-codec entries ([`crate::Predictor::predict`],
//!   `argmin_masked`, etc.) work unchanged.
//!
//! ## Input vector layout (composed at predict-time)
//!
//! For a bake with `union_feat_count = U`, `n_codecs = C`:
//!
//! ```text
//! [0..U)              union feature values (0.0 where this codec
//!                      doesn't have the feature)
//! [U..2U)             presence mask (1.0 in this codec's slots,
//!                      0.0 elsewhere)
//! [2U..2U+4)          size_onehot[0..4] (tiny/small/medium/large)
//! [2U+4]              log_pixels
//! [2U+5]              zq_norm
//! [2U+6..2U+6+C)      codec_onehot[0..C] (1.0 at codec_id)
//! ```
//!
//! The bake's `n_inputs` MUST equal `2*U + 6 + C` — load-time
//! validation enforces this.

use crate::error::PredictError;

/// Per-codec map: how a single codec's natural feature vector lands
/// in the union, where its head sits in the trunk's output vector,
/// and a small header describing the head shape (informational).
#[derive(Clone, Debug)]
pub struct PerCodecMap<'a> {
    /// Stable codec name — `"zenjpeg"` / `"zenwebp"` / `"zenavif"` /
    /// `"zenjxl"`. Used by debug / dump tools; runtime keys on
    /// `codec_id` (slice index in the schema's `per_codec` vector).
    pub codec_name: &'a str,
    /// For each of this codec's natural feat_cols (length =
    /// `n_codec_feats`), the union slot index `[0..union_feat_count)`
    /// to scatter that value into.
    pub union_slot_for_codec_feat: &'a [u32],
    /// `(lo, hi)` — half-open range into the trunk's flat output
    /// vector that belongs to this codec. `lo..hi` is the slice
    /// [`crate::Predictor::predict_multi_codec`] returns.
    pub output_range: (u32, u32),
    /// Informational head metadata. `n_cells` and `n_heads` describe
    /// the per-codec head shape (typically `n_cells` configs × `1`
    /// bytes head, optionally extra scalar heads). Not consumed by
    /// inference itself — surfaced for callers that want to
    /// `argmin_masked` over the bytes head only.
    pub head_meta: HeadMeta,
}

/// Per-codec head shape descriptor. Informational — `n_cells *
/// n_heads` SHOULD equal `output_range.1 - output_range.0`, but the
/// loader doesn't enforce that equality (the trunk may legitimately
/// expose extra slots, e.g. a softmax classifier head).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HeadMeta {
    /// Number of config cells in this codec's argmin space.
    pub n_cells: u32,
    /// Number of heads per cell (bytes head + scalar heads).
    pub n_heads: u32,
}

/// Parsed multi-codec schema. Borrows from the model's owned bytes
/// (the `'a` lifetime). Cheap to materialize on demand from
/// `Model.bytes` via [`parse`].
#[derive(Clone, Debug)]
pub struct MultiCodecSchema<'a> {
    pub union_feat_count: u32,
    pub n_codecs: u32,
    pub per_codec: alloc::vec::Vec<PerCodecMap<'a>>,
}

impl<'a> MultiCodecSchema<'a> {
    /// Looked-up codec map. Returns `None` if `codec_id >= n_codecs`.
    pub fn per_codec(&self, codec_id: u32) -> Option<&PerCodecMap<'a>> {
        self.per_codec.get(codec_id as usize)
    }

    /// All codec names, in order. Same order as
    /// [`Self::per_codec`] indexing.
    pub fn codec_names(&self) -> alloc::vec::Vec<&'a str> {
        self.per_codec.iter().map(|m| m.codec_name).collect()
    }
}

/// Per-codec map entry on the wire. 32 bytes; matches
/// [`crate::wire::SECTION_OFF_MULTI_CODEC_SCHEMA`].
pub const PER_CODEC_ENTRY_SIZE: usize = 32;

/// Parse a `MultiCodecSchema` from the bytes of the
/// `multi_codec_schema` header section. `section_bytes` must be the
/// exact bytes of the section (e.g. the result of
/// `header.multi_codec_schema.slice(...)`); empty input is invalid
/// here — callers that handle the absent case should check the
/// `Section::is_empty()` themselves.
pub fn parse(section_bytes: &[u8]) -> Result<MultiCodecSchema<'_>, PredictError> {
    if section_bytes.len() < 8 {
        return Err(PredictError::Truncated {
            offset: 0,
            want: 8,
            have: section_bytes.len(),
        });
    }
    let union_feat_count = u32::from_le_bytes(section_bytes[0..4].try_into().unwrap());
    let n_codecs = u32::from_le_bytes(section_bytes[4..8].try_into().unwrap());

    if n_codecs == 0 {
        return Err(PredictError::ZeroDimension {
            what: "multi_codec_schema.n_codecs",
        });
    }
    if (n_codecs as usize) > crate::limits::MAX_DIM {
        return Err(PredictError::DimensionOverflow {
            what: "multi_codec_schema.n_codecs",
        });
    }
    if (union_feat_count as usize) > crate::limits::MAX_DIM {
        return Err(PredictError::DimensionOverflow {
            what: "multi_codec_schema.union_feat_count",
        });
    }

    let table_bytes = (n_codecs as usize)
        .checked_mul(PER_CODEC_ENTRY_SIZE)
        .ok_or(PredictError::DimensionOverflow {
            what: "multi_codec_schema: n_codecs * PER_CODEC_ENTRY_SIZE",
        })?;
    let table_end = 8usize
        .checked_add(table_bytes)
        .ok_or(PredictError::DimensionOverflow {
            what: "multi_codec_schema: header + table size",
        })?;
    if table_end > section_bytes.len() {
        return Err(PredictError::Truncated {
            offset: 8,
            want: table_bytes,
            have: section_bytes.len() - 8,
        });
    }

    let mut per_codec: alloc::vec::Vec<PerCodecMap<'_>> =
        alloc::vec::Vec::with_capacity(n_codecs as usize);
    for i in 0..n_codecs as usize {
        let entry_off = 8 + i * PER_CODEC_ENTRY_SIZE;
        let entry: &[u8; PER_CODEC_ENTRY_SIZE] = section_bytes
            [entry_off..entry_off + PER_CODEC_ENTRY_SIZE]
            .try_into()
            .map_err(|_| PredictError::Truncated {
                offset: entry_off,
                want: PER_CODEC_ENTRY_SIZE,
                have: 0,
            })?;
        let name_offset = u32::from_le_bytes(entry[0..4].try_into().unwrap()) as usize;
        let name_len = u32::from_le_bytes(entry[4..8].try_into().unwrap()) as usize;
        let slots_offset = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
        let slots_count = u32::from_le_bytes(entry[12..16].try_into().unwrap()) as usize;
        let output_range_lo = u32::from_le_bytes(entry[16..20].try_into().unwrap());
        let output_range_hi = u32::from_le_bytes(entry[20..24].try_into().unwrap());
        let head_n_cells = u32::from_le_bytes(entry[24..28].try_into().unwrap());
        let head_n_heads = u32::from_le_bytes(entry[28..32].try_into().unwrap());

        // Validate name range.
        let name_end =
            name_offset
                .checked_add(name_len)
                .ok_or(PredictError::DimensionOverflow {
                    what: "multi_codec_schema: name_offset + name_len",
                })?;
        if name_end > section_bytes.len() {
            return Err(PredictError::SectionOutOfRange {
                what: "multi_codec_schema: codec_name range",
                offset: name_offset as u32,
                len: name_len as u32,
                file_len: section_bytes.len(),
            });
        }
        let codec_name =
            core::str::from_utf8(&section_bytes[name_offset..name_end]).map_err(|_| {
                PredictError::MetadataKeyNotUtf8 {
                    offset: name_offset,
                }
            })?;

        // Validate slots range.
        if !slots_offset.is_multiple_of(4) {
            return Err(PredictError::SectionMisaligned {
                what: "multi_codec_schema: slot table",
                offset: slots_offset as u32,
                required_align: 4,
            });
        }
        let slots_bytes = slots_count
            .checked_mul(4)
            .ok_or(PredictError::DimensionOverflow {
                what: "multi_codec_schema: slots_count * 4",
            })?;
        let slots_end =
            slots_offset
                .checked_add(slots_bytes)
                .ok_or(PredictError::DimensionOverflow {
                    what: "multi_codec_schema: slots_offset + slots_bytes",
                })?;
        if slots_end > section_bytes.len() {
            return Err(PredictError::SectionOutOfRange {
                what: "multi_codec_schema: slot table range",
                offset: slots_offset as u32,
                len: slots_bytes as u32,
                file_len: section_bytes.len(),
            });
        }
        let slots_raw = &section_bytes[slots_offset..slots_end];
        let slots: &[u32] =
            bytemuck::try_cast_slice(slots_raw).map_err(|_| PredictError::SectionMisaligned {
                what: "multi_codec_schema: slot table u32 cast",
                offset: slots_offset as u32,
                required_align: 4,
            })?;
        // Validate every slot < union_feat_count.
        for &s in slots {
            if s >= union_feat_count {
                return Err(PredictError::OutputDimMismatch {
                    expected: union_feat_count as usize,
                    got: s as usize,
                });
            }
        }

        // Validate output_range.
        if output_range_lo > output_range_hi {
            return Err(PredictError::OutputDimMismatch {
                expected: output_range_hi as usize,
                got: output_range_lo as usize,
            });
        }

        per_codec.push(PerCodecMap {
            codec_name,
            union_slot_for_codec_feat: slots,
            output_range: (output_range_lo, output_range_hi),
            head_meta: HeadMeta {
                n_cells: head_n_cells,
                n_heads: head_n_heads,
            },
        });
    }

    Ok(MultiCodecSchema {
        union_feat_count,
        n_codecs,
        per_codec,
    })
}

/// Expected `n_inputs` for a multi-codec trunk:
/// `2 * union_feat_count + 6 + n_codecs`. Used by
/// [`crate::Predictor::predict_multi_codec`] and the load-time
/// validator to size scratch and verify the bake.
pub fn expected_n_inputs(union_feat_count: u32, n_codecs: u32) -> usize {
    2 * (union_feat_count as usize) + 6 + (n_codecs as usize)
}
