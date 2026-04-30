//! Codec-family meta-picker — given image features, a quality
//! target, and the caller's allowed-family mask, choose a codec
//! family. Per-codec pickers (separate ZNPR v2 bakes shipped by the
//! codec crate) then resolve the family into a concrete encoder
//! config.
//!
//! ## Where it sits
//!
//! ```text
//!     features (zenanalyze) + target_zq + caller constraints
//!                            │
//!                            ▼
//!                ┌──────────────────────┐
//!                │ zenpicker::MetaPicker│   one ZNPR v2 model;
//!                │  argmin over family  │   N_outputs = N families
//!                └──────────┬───────────┘
//!                           │ chosen family
//!                           ▼
//!                ┌──────────────────────┐
//!                │ Per-codec picker     │   one .bin per family,
//!                │  (zenpredict model)  │   shipped from the codec
//!                │  → cell + scalars    │   crate
//!                └──────────┬───────────┘
//!                           ▼
//!                  concrete EncoderConfig
//! ```
//!
//! The meta-picker emits a [`CodecFamily`]; it does **not** know how
//! to resolve a family into a concrete encoder config. That's the
//! job of the family's per-codec picker (a separate ZNPR v2 bake
//! shipped by the codec crate, also loaded via [`zenpredict`]).
//!
//! ## Wire format
//!
//! Internally a [`MetaPicker`] is just a [`zenpredict::Predictor`]
//! whose `n_outputs` equals [`CodecFamily::COUNT`]. The output index
//! is the family enum's discriminant; bake-time and runtime must
//! agree on the order via the model's metadata
//! ([`FAMILY_ORDER_KEY`] = `zenpicker.family_order`, UTF-8,
//! comma-separated lower-case labels — same order as
//! [`CodecFamily::ALL`]).
//!
//! ## Crate boundary
//!
//! - [`zenpredict`] — the runtime this crate composes on. Owns the
//!   ZNPR v2 binary format, the parser, the forward pass, the
//!   masked-argmin math, the metadata blob, and the `Predictor`.
//!   `zenpicker` adds: family enum + family-order validation +
//!   `AllowedFamilies` mask sugar.
//! - [`zentrain`](https://github.com/imazen/zenanalyze/tree/main/zentrain)
//!   — Python training pipeline that produces the `.bin` artifact a
//!   meta-picker (or a per-codec picker) loads. Train with
//!   `cells = families` and `output_layout = bytes_log` only
//!   (purely categorical, no scalar heads).
//! - [`zenanalyze`](https://crates.io/crates/zenanalyze) — feature
//!   extractor that produces the input vector both this meta-picker
//!   and the per-codec pickers consume.
//!
//! ## Status
//!
//! v0.1 establishes the crate boundary and the API shape. Baking an
//! actual cross-codec meta-picker model is downstream work — once a
//! labelled training set exists where each row maps `(image
//! features, target_zq) → best family`, run zentrain's
//! `train_hybrid.py` with `cells = families` and `output_layout` of
//! `bytes_log` only.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};

use zenpredict::{AllowedMask, ArgminOffsets, Model, PredictError, Predictor, ScoreTransform};

/// Codec families the meta-picker can choose between.
///
/// **Important — order matters.** The discriminants here must match
/// the order in the baked meta-picker model's output vector. Bakes
/// declare the order via the model metadata ([`FAMILY_ORDER_KEY`]).
/// Runtime checks this at load via [`MetaPicker::validate_family_order`].
///
/// Adding a new family is a breaking change for any baked model that
/// existed before — the schema_hash will mismatch and the runtime
/// will refuse to load the old model. Bake a fresh meta-picker that
/// includes the new family before deploying the codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum CodecFamily {
    Jpeg = 0,
    Webp = 1,
    Jxl = 2,
    Avif = 3,
    Png = 4,
    Gif = 5,
}

impl CodecFamily {
    /// Number of variants currently defined. Used to size masks /
    /// allocate output buffers. Bump this when adding a variant.
    pub const COUNT: usize = 6;

    /// All variants in declared order — same order the bake's
    /// `output_layout` must use.
    pub const ALL: [CodecFamily; Self::COUNT] = [
        Self::Jpeg,
        Self::Webp,
        Self::Jxl,
        Self::Avif,
        Self::Png,
        Self::Gif,
    ];

    /// Discriminant as `usize` for indexing into mask / output arrays.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Stable string label.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Jxl => "jxl",
            Self::Avif => "avif",
            Self::Png => "png",
            Self::Gif => "gif",
        }
    }
}

/// Caller-supplied filter over which families are acceptable for a
/// given encode. Wraps a fixed-size `[bool; CodecFamily::COUNT]` so
/// the runtime can build a [`zenpredict::AllowedMask`] without
/// allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowedFamilies {
    flags: [bool; CodecFamily::COUNT],
}

impl AllowedFamilies {
    pub const fn none() -> Self {
        Self {
            flags: [false; CodecFamily::COUNT],
        }
    }

    pub const fn all() -> Self {
        Self {
            flags: [true; CodecFamily::COUNT],
        }
    }

    /// Build an `AllowedFamilies` from an iterator over allowed
    /// families (everything else denied). Named `from` rather than
    /// `from_iter` to keep clippy's `should_implement_trait` lint
    /// happy without claiming the full `FromIterator` contract
    /// (which would force `Self::from_iter(empty()) == none()`,
    /// matching what we do — but spelling out the trait pulls in
    /// blanket impls we don't need).
    pub fn from_allowed<I: IntoIterator<Item = CodecFamily>>(iter: I) -> Self {
        let mut me = Self::none();
        for fam in iter {
            me.flags[fam.index()] = true;
        }
        me
    }

    pub fn allow(mut self, fam: CodecFamily) -> Self {
        self.flags[fam.index()] = true;
        self
    }

    pub fn deny(mut self, fam: CodecFamily) -> Self {
        self.flags[fam.index()] = false;
        self
    }

    pub fn is_allowed(self, fam: CodecFamily) -> bool {
        self.flags[fam.index()]
    }

    pub const fn as_slice(&self) -> &[bool] {
        &self.flags
    }

    pub fn any(self) -> bool {
        self.flags.iter().any(|f| *f)
    }
}

/// Metadata key the bake declares to assert family-order agreement
/// between trainer and runtime.
pub const FAMILY_ORDER_KEY: &str = "zenpicker.family_order";

/// Expected value of [`FAMILY_ORDER_KEY`] for the current
/// [`CodecFamily::ALL`] layout.
pub const ALL_LABELS_CSV: &str = "jpeg,webp,jxl,avif,png,gif";

/// One thin meta-picker.
///
/// Owns a [`zenpredict::Predictor`] (mutable scratch for one forward
/// pass at a time). Construction is cheap; reuse a single instance
/// across many encode requests.
pub struct MetaPicker<'b> {
    predictor: Predictor<'b>,
}

impl<'b> MetaPicker<'b> {
    /// Wrap a parsed [`zenpredict::Model`]. Caller is expected to
    /// have validated the schema hash via
    /// [`zenpredict::Model::from_bytes_with_schema`] or by reading
    /// the model's metadata.
    ///
    /// Call [`MetaPicker::validate_family_order`] right after
    /// construction to confirm bake-time and runtime agree on the
    /// family enum layout.
    pub fn new(model: Model<'b>) -> Self {
        Self {
            predictor: Predictor::new(model),
        }
    }

    /// Borrow the underlying predictor — useful when the caller
    /// wants to read model metadata or run `predict` for diagnostics.
    pub fn predictor(&mut self) -> &mut Predictor<'b> {
        &mut self.predictor
    }

    /// Run argmin over the family dimension under the caller's
    /// allowed-family filter.
    ///
    /// `features` is the same feature vector the per-codec pickers
    /// consume — the bake declares which feature columns it uses
    /// (`feat_cols` in the manifest).
    ///
    /// Returns `Ok(None)` when every family is masked out (caller
    /// constraints unsatisfiable) and `Err` only on a runtime error
    /// (shape mismatch, NaN, …).
    pub fn pick(
        &mut self,
        features: &[f32],
        allowed: &AllowedFamilies,
    ) -> Result<Option<CodecFamily>, MetaPickerError> {
        if !allowed.any() {
            return Ok(None);
        }
        let mask = AllowedMask::new(allowed.as_slice());
        let pick = self
            .predictor
            .argmin_masked(
                features,
                &mask,
                ScoreTransform::Identity,
                None::<&ArgminOffsets>,
            )
            .map_err(MetaPickerError::Predict)?;
        Ok(pick.map(|idx| CodecFamily::ALL[idx]))
    }

    /// Read the [`FAMILY_ORDER_KEY`] (`zenpicker.family_order`)
    /// metadata key from the bake and confirm it matches
    /// [`ALL_LABELS_CSV`]. Returns `Ok(())` if the order matches,
    /// `Err` on mismatch (caller should refuse to use the picker —
    /// the bake was made against a different enum layout).
    ///
    /// Best practice: call once at startup, fail loudly on mismatch.
    pub fn validate_family_order(&mut self) -> Result<(), MetaPickerError> {
        let raw = self
            .predictor
            .model()
            .metadata()
            .get_utf8(FAMILY_ORDER_KEY)
            .map_err(|e| MetaPickerError::Metadata(format!("metadata: {:?}", e)))?;

        if raw == ALL_LABELS_CSV {
            Ok(())
        } else {
            Err(MetaPickerError::FamilyOrderMismatch {
                expected: ALL_LABELS_CSV.to_string(),
                actual: raw.to_string(),
            })
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum MetaPickerError {
    Predict(PredictError),
    Metadata(String),
    FamilyOrderMismatch { expected: String, actual: String },
}

#[cfg(feature = "std")]
impl core::fmt::Display for MetaPickerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Predict(e) => write!(f, "predict: {:?}", e),
            Self::Metadata(s) => write!(f, "metadata: {}", s),
            Self::FamilyOrderMismatch { expected, actual } => write!(
                f,
                "family order mismatch: bake declares {:?}, runtime expects {:?}",
                actual, expected
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MetaPickerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn family_order_csv_matches_all() {
        let computed = CodecFamily::ALL
            .iter()
            .map(|f| f.label())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(computed, ALL_LABELS_CSV);
    }

    #[test]
    fn allowed_families_basic_ops() {
        let af = AllowedFamilies::none()
            .allow(CodecFamily::Jpeg)
            .allow(CodecFamily::Webp);
        assert!(af.is_allowed(CodecFamily::Jpeg));
        assert!(af.is_allowed(CodecFamily::Webp));
        assert!(!af.is_allowed(CodecFamily::Avif));
        assert!(af.any());

        let none = AllowedFamilies::none();
        assert!(!none.any());

        let all = AllowedFamilies::all();
        for fam in CodecFamily::ALL {
            assert!(all.is_allowed(fam));
        }
    }

    #[test]
    fn family_indexing_is_dense_and_zero_based() {
        for (i, fam) in CodecFamily::ALL.iter().enumerate() {
            assert_eq!(fam.index(), i);
        }
    }

    #[test]
    fn family_order_constants_are_consistent() {
        assert_eq!(CodecFamily::ALL.len(), CodecFamily::COUNT);
        assert_eq!(ALL_LABELS_CSV.split(',').count(), CodecFamily::COUNT);
    }

    #[test]
    fn family_order_key_is_zenpicker_namespaced() {
        // Reflects the rename: the meta-picker IS zenpicker now.
        // Keep this test so a future rename doesn't quietly drift.
        assert_eq!(FAMILY_ORDER_KEY, "zenpicker.family_order");
    }
}
