//! ZNPR v2 metadata blob — typed, length-prefixed key/value entries.
//!
//! Read once at load, kept on the [`crate::Model`] as a borrowed
//! `Vec<MetadataEntry>` over the original bytes. Per-entry value
//! decoding happens at the call site that wants the value (typically
//! the codec or scoring crate, knowing what shape it expects from
//! the bake).
//!
//! ## Wire format
//!
//! ```text
//! while offset < section_end:
//!   [1]    key_len: u8                 (255-byte cap)
//!   [...]  key:     [u8; key_len]      UTF-8, validated on parse
//!   [1]    value_type: u8              0=bytes, 1=utf8, 2=numeric, 3..=255 reserved
//!   [4]    value_len: u32 (LE, unaligned read)
//!   [...]  value:    [u8; value_len]
//! ```
//!
//! Entries are packed back-to-back; no padding. `value_len` is read
//! unaligned via `u32::from_le_bytes`. Values that are themselves
//! `#[repr(C)]` structs (e.g. `zentrain.calibration_metrics`'s
//! three-`f32` payload) are read with `bytemuck::pod_read_unaligned`
//! at the call site — metadata is not zero-copy by design.
//!
//! ## Namespace convention
//!
//! - `zentrain.*` — defined by the Python training pipeline. Loader
//!   exposes typed accessors for the standard ones via well-known
//!   constants in [`keys`].
//! - `zenpicker.*` — reserved for the Rust meta-picker
//!   ([`zenpicker::FAMILY_ORDER_KEY`] etc.).
//! - `zensim.*` — reserved for the zensim scorer pipeline.
//! - `<codec>.*` (e.g. `zenjpeg.*`, `zenwebp.*`) — codec-private,
//!   opaque to zenpredict.
//! - No prefix — reserved.
//!
//! [`zenpicker::FAMILY_ORDER_KEY`]: https://docs.rs/zenpicker
//!
//! Unknown keys are not removed; iteration via [`Metadata::iter`]
//! still surfaces them, so debug/dump tools see everything.

use bytemuck::pod_read_unaligned;

use crate::error::PredictError;

/// What shape the value bytes should be interpreted as. The actual
/// per-key decoding is the caller's responsibility — this byte just
/// gates a generic dump tool's render and lets typed accessors
/// (`get_utf8`, `get_numeric`, `get_bytes`) fail loudly on mismatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetadataType {
    /// Opaque bytes (wire type 0). Could be a `#[repr(C)]` struct,
    /// codec-private payload, packed length-prefixed string array,
    /// anything.
    Bytes,
    /// UTF-8 text (wire type 1). Validated on parse — the borrowed
    /// `&str` is known-valid by the time the caller gets it.
    Utf8,
    /// Fixed-width LE numeric (wire type 2). Width is `value_len`
    /// (1=u8/i8, 2=u16/i16, 4=u32/i32/f32, 8=u64/i64/f64,
    /// 4·N = f32 array, 8·N = f64 array). The per-key loader knows
    /// which shape — the type byte is for tooling and validation,
    /// not dispatch.
    Numeric,
    /// A future type code this build doesn't recognize. Loader
    /// preserves the raw bytes and surfaces the original code to
    /// dump tools.
    Reserved(u8),
}

impl MetadataType {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Bytes,
            1 => Self::Utf8,
            2 => Self::Numeric,
            other => Self::Reserved(other),
        }
    }
}

/// Borrowed view of a single metadata entry. `key` is validated as
/// UTF-8 on parse; `value` is raw bytes regardless of `kind`.
#[derive(Clone, Copy, Debug)]
pub struct MetadataEntry<'a> {
    pub key: &'a str,
    pub kind: MetadataType,
    pub value: &'a [u8],
}

/// Parsed metadata blob. Entries kept in their original order.
/// Linear scan on `get` is fine at the small numbers of keys
/// real bakes carry (≤ 30).
#[derive(Clone, Debug, Default)]
pub struct Metadata<'a> {
    entries: alloc::vec::Vec<MetadataEntry<'a>>,
}

impl<'a> Metadata<'a> {
    /// Parse a metadata blob. An empty blob (`bytes.len() == 0`) is
    /// the "no metadata" case and yields an empty `Metadata`.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, PredictError> {
        let mut entries = alloc::vec::Vec::new();
        let mut pos = 0usize;
        while pos < bytes.len() {
            // [1] key_len
            let key_len = *bytes.get(pos).ok_or(PredictError::Truncated {
                offset: pos,
                want: 1,
                have: 0,
            })? as usize;
            pos += 1;

            // [key_len] key
            let key_end = pos.checked_add(key_len).ok_or(PredictError::Truncated {
                offset: pos,
                want: key_len,
                have: bytes.len().saturating_sub(pos),
            })?;
            if key_end > bytes.len() {
                return Err(PredictError::Truncated {
                    offset: pos,
                    want: key_len,
                    have: bytes.len() - pos,
                });
            }
            let key_bytes = &bytes[pos..key_end];
            let key = core::str::from_utf8(key_bytes)
                .map_err(|_| PredictError::MetadataKeyNotUtf8 { offset: pos })?;
            pos = key_end;

            // [1] value_type
            let value_type_byte = *bytes.get(pos).ok_or(PredictError::Truncated {
                offset: pos,
                want: 1,
                have: 0,
            })?;
            pos += 1;

            // [4] value_len (unaligned LE)
            if pos + 4 > bytes.len() {
                return Err(PredictError::Truncated {
                    offset: pos,
                    want: 4,
                    have: bytes.len() - pos,
                });
            }
            let value_len_bytes: [u8; 4] = bytes[pos..pos + 4]
                .try_into()
                .expect("4-byte slice → [u8; 4] is infallible");
            let value_len = u32::from_le_bytes(value_len_bytes) as usize;
            pos += 4;

            // [value_len] value
            let value_end = pos.checked_add(value_len).ok_or(PredictError::Truncated {
                offset: pos,
                want: value_len,
                have: bytes.len().saturating_sub(pos),
            })?;
            if value_end > bytes.len() {
                return Err(PredictError::Truncated {
                    offset: pos,
                    want: value_len,
                    have: bytes.len() - pos,
                });
            }
            let value = &bytes[pos..value_end];
            pos = value_end;

            entries.push(MetadataEntry {
                key,
                kind: MetadataType::from_byte(value_type_byte),
                value,
            });
        }
        Ok(Self { entries })
    }

    /// Lookup by exact key. Linear scan — fine at ≤ 30 entries.
    pub fn get(&self, key: &str) -> Option<&MetadataEntry<'a>> {
        self.entries.iter().find(|e| e.key == key)
    }

    /// Lookup + type-check + UTF-8 borrow. Use for keys whose value
    /// is a single string.
    pub fn get_utf8(&self, key: &str) -> Result<&'a str, PredictError> {
        let entry = self.get(key).ok_or(PredictError::MetadataTypeMismatch {
            key_len: key.len(),
            expected: MetadataType::Utf8,
            got: MetadataType::Bytes,
        })?;
        if entry.kind != MetadataType::Utf8 {
            return Err(PredictError::MetadataTypeMismatch {
                key_len: key.len(),
                expected: MetadataType::Utf8,
                got: entry.kind,
            });
        }
        core::str::from_utf8(entry.value)
            .map_err(|_| PredictError::MetadataValueNotUtf8 { key_len: key.len() })
    }

    /// Lookup + type-check, returning the raw bytes (no further
    /// decoding). Per-key loader interprets the bytes against the
    /// known shape (e.g. `bytemuck::pod_read_unaligned::<Metrics>`).
    pub fn get_bytes(&self, key: &str) -> Result<&'a [u8], PredictError> {
        let entry = self.get(key).ok_or(PredictError::MetadataTypeMismatch {
            key_len: key.len(),
            expected: MetadataType::Bytes,
            got: MetadataType::Bytes,
        })?;
        if entry.kind != MetadataType::Bytes {
            return Err(PredictError::MetadataTypeMismatch {
                key_len: key.len(),
                expected: MetadataType::Bytes,
                got: entry.kind,
            });
        }
        Ok(entry.value)
    }

    /// Lookup + type-check, returning the raw numeric bytes. Caller
    /// width-decodes via `u32::from_le_bytes`/`f32::from_le_bytes`/
    /// `bytemuck::cast_slice`/`pod_read_unaligned` per the per-key
    /// shape contract.
    pub fn get_numeric(&self, key: &str) -> Result<&'a [u8], PredictError> {
        let entry = self.get(key).ok_or(PredictError::MetadataTypeMismatch {
            key_len: key.len(),
            expected: MetadataType::Numeric,
            got: MetadataType::Bytes,
        })?;
        if entry.kind != MetadataType::Numeric {
            return Err(PredictError::MetadataTypeMismatch {
                key_len: key.len(),
                expected: MetadataType::Numeric,
                got: entry.kind,
            });
        }
        Ok(entry.value)
    }

    /// Decode a single numeric value at a known fixed width. Returns
    /// `None` if the entry is missing, has the wrong type, or has
    /// the wrong byte length.
    pub fn get_pod<T: bytemuck::Pod>(&self, key: &str) -> Option<T> {
        let entry = self.get(key)?;
        if entry.kind != MetadataType::Numeric && entry.kind != MetadataType::Bytes {
            return None;
        }
        if entry.value.len() != core::mem::size_of::<T>() {
            return None;
        }
        Some(pod_read_unaligned(entry.value))
    }

    pub fn iter(&self) -> impl Iterator<Item = &MetadataEntry<'a>> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Well-known metadata key constants. Bakes are encouraged to use
/// these for consistency; loaders can match against them without
/// allocating a string.
///
/// Naming convention: keys land under the namespace of the **producer**.
/// The training pipeline lives in [`zentrain`] (Python) and emits these
/// keys at bake time, so they're under `zentrain.*`. The
/// `zenpicker.*` namespace is reserved for the Rust meta-picker
/// (`zenpicker::FAMILY_ORDER_KEY`, etc.); codec-private keys land
/// under `<codec>.*` (e.g. `zenjpeg.cell_config`).
///
/// [`zentrain`]: https://github.com/imazen/zenanalyze/tree/main/zentrain
pub mod keys {
    /// `u8` — 0=size_optimal, 1=zensim_strict.
    pub const PROFILE: &str = "zentrain.profile";
    /// utf8 — `zentrain.v1.shared-mlp.distill+icc` etc.
    pub const SCHEMA_VERSION_TAG: &str = "zentrain.schema_version_tag";
    /// bytes — `[u32 count][count × (u16 len + utf8 bytes)]`.
    pub const FEATURE_COLUMNS: &str = "zentrain.feature_columns";
    /// bytes — `#[repr(C)] { n_cells: u32, n_heads: u32, head_kinds: [u8; n_heads] }`.
    pub const HYBRID_HEADS_LAYOUT: &str = "zentrain.hybrid_heads_layout";
    /// utf8 — free-form provenance (git, corpus, sklearn, ts, host).
    pub const PROVENANCE: &str = "zentrain.provenance";
    /// numeric (12 bytes) — `#[repr(C)] { mean_overhead, p99_shortfall, argmin_acc: f32 }`.
    pub const CALIBRATION_METRICS: &str = "zentrain.calibration_metrics";
    /// bytes — `[u8 passed]` then optional utf8 violation list.
    pub const SAFETY_REPORT: &str = "zentrain.safety_report";
    /// utf8 — friendly name (`zenjpeg_picker_v2.1_full`).
    pub const BAKE_NAME: &str = "zentrain.bake_name";
    /// bytes (numeric) — `f32[n_zq * n_cells]`. Read by codecs
    /// applying the strict reach-rate gate. Copied out at startup.
    pub const REACH_RATES: &str = "zentrain.reach_rates";
    /// bytes (numeric) — `u8[n_zq]`. Paired with [`REACH_RATES`].
    pub const REACH_ZQ_TARGETS: &str = "zentrain.reach_zq_targets";
}
