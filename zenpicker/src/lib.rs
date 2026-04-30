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
use alloc::string::String;

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
///
/// At construction the picker reads the bake's
/// [`FAMILY_ORDER_KEY`] metadata and caches the resulting
/// `output_index → CodecFamily` mapping. Bakes that cover only a
/// subset of [`CodecFamily::ALL`] (e.g. an early bake without any
/// gif training data) work transparently: [`MetaPicker::pick`] uses
/// the bake's order directly when masking and decoding the argmin
/// result. If the metadata is missing or unparseable the cache is
/// `None` and the picker assumes [`CodecFamily::ALL`] order — call
/// [`MetaPicker::validate_family_order`] at startup to fail loudly
/// in that case.
pub struct MetaPicker<'b> {
    predictor: Predictor<'b>,
    /// Parsed mapping from bake output index to CodecFamily.
    /// `None` when the bake is missing or has unparseable
    /// `zenpicker.family_order` metadata.
    family_at_output: Option<alloc::vec::Vec<CodecFamily>>,
}

impl<'b> MetaPicker<'b> {
    /// Wrap a parsed [`zenpredict::Model`]. Caller is expected to
    /// have validated the schema hash via
    /// [`zenpredict::Model::from_bytes_with_schema`] or by reading
    /// the model's metadata.
    ///
    /// Reads the bake's [`FAMILY_ORDER_KEY`] once and caches the
    /// `output_index → CodecFamily` mapping. Call
    /// [`MetaPicker::validate_family_order`] right after
    /// construction to confirm the metadata was present and
    /// well-formed (subset-of-[`CodecFamily::ALL`] is OK, but any
    /// label not in the enum or a length mismatch with `n_outputs`
    /// will surface there).
    ///
    /// Takes `&'b Model` — the Model is the long-lived parsed bake
    /// (typically inside a `static OnceLock<Model>`); the borrow
    /// lifetime `'b` flows through into `Predictor<'b>` and then
    /// `MetaPicker<'b>`. zenpredict 0.2.0+ made Model own its
    /// reference-data rather than carry a lifetime parameter.
    pub fn new(model: &'b Model) -> Self {
        let family_at_output = model
            .metadata()
            .get_utf8(FAMILY_ORDER_KEY)
            .ok()
            .and_then(parse_family_csv)
            .filter(|v| v.len() == model.n_outputs());
        Self {
            predictor: Predictor::new(model),
            family_at_output,
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
        // Build the mask in the bake's output order — *not*
        // `allowed.as_slice()`. The bake may cover a subset of
        // [`CodecFamily::ALL`] in any order, so the i-th bake output
        // doesn't necessarily map to the i-th `AllowedFamilies` slot.
        let order: &[CodecFamily] = self
            .family_at_output
            .as_deref()
            .unwrap_or(&CodecFamily::ALL[..]);
        let bake_mask: alloc::vec::Vec<bool> =
            order.iter().map(|f| allowed.is_allowed(*f)).collect();
        let mask = AllowedMask::new(&bake_mask);
        let pick = self
            .predictor
            .argmin_masked(
                features,
                &mask,
                ScoreTransform::Identity,
                None::<&ArgminOffsets>,
            )
            .map_err(MetaPickerError::Predict)?;
        Ok(pick.map(|idx| order[idx]))
    }

    /// Pick under a runtime size/speed tradeoff.
    ///
    /// `predicted_encode_ms_per_family` is a [`CodecFamily::COUNT`]-
    /// sized table of caller-provided per-family encode-time
    /// estimates **at the source's resolution** (typically computed
    /// from `α + β·MPx` models calibrated from a sweep). The caller's
    /// `bytes_per_ms` weight expresses willingness to pay file-size
    /// for encode time:
    ///
    /// - `0.0` → pure size-optimal (equivalent to [`Self::pick`])
    /// - `1.0` → 1 ms saved is worth 1 byte (almost always
    ///   size-optimal in practice)
    /// - `100.0` → 1 ms saved is worth 100 bytes (typical for
    ///   client-side encodes where time is precious)
    /// - `f32::INFINITY` → pure time-optimal (always picks the
    ///   fastest family in the allowed set)
    ///
    /// Internally this builds an [`ArgminOffsets`] in linear-byte
    /// space (`time_cost = predicted_ms × bytes_per_ms`), maps it
    /// into the bake's output order, and runs argmin under
    /// [`ScoreTransform::Exp`] (which converts the trainer's
    /// log-bytes outputs to the linear-bytes the offset is in).
    ///
    /// Behavioural contract: identical to [`Self::pick`] when
    /// `bytes_per_ms == 0.0`. Different when `> 0`.
    pub fn pick_with_time_cost(
        &mut self,
        features: &[f32],
        allowed: &AllowedFamilies,
        predicted_encode_ms_per_family: &[f32; CodecFamily::COUNT],
        bytes_per_ms: f32,
    ) -> Result<Option<CodecFamily>, MetaPickerError> {
        if !allowed.any() {
            return Ok(None);
        }
        let order: &[CodecFamily] = self
            .family_at_output
            .as_deref()
            .unwrap_or(&CodecFamily::ALL[..]);

        let bake_mask: alloc::vec::Vec<bool> =
            order.iter().map(|f| allowed.is_allowed(*f)).collect();
        let mask = AllowedMask::new(&bake_mask);

        // Re-index the per-family ms table into the bake's actual
        // output order. Multiply by bytes_per_ms to produce the
        // additive byte-equivalent offset.
        let per_output_offset: alloc::vec::Vec<f32> = order
            .iter()
            .map(|f| predicted_encode_ms_per_family[f.index()] * bytes_per_ms)
            .collect();
        let offsets = ArgminOffsets {
            uniform: 0.0,
            per_output: Some(&per_output_offset),
        };

        let pick = self
            .predictor
            .argmin_masked(features, &mask, ScoreTransform::Exp, Some(&offsets))
            .map_err(MetaPickerError::Predict)?;
        Ok(pick.map(|idx| order[idx]))
    }

    /// Confirm the bake's [`FAMILY_ORDER_KEY`] metadata was present,
    /// parseable as a CSV of [`CodecFamily`] labels, and length-
    /// equal to `n_outputs`.
    ///
    /// **Subset is OK.** A bake whose order is `"avif,jpeg,jxl,png,webp"`
    /// (no gif — substrate had no gif data) passes this check. The
    /// picker's [`pick`](Self::pick) method uses the bake's actual
    /// order to map output indices to [`CodecFamily`]; callers can
    /// inspect [`Self::family_at_output`] to learn which families
    /// the bake can pick.
    ///
    /// Returns `Err(MetaPickerError::Metadata(_))` when the metadata
    /// was missing, malformed, contains an unrecognised label, or
    /// the parsed length disagrees with `n_outputs`. Best practice:
    /// call once at startup, fail loudly on mismatch.
    pub fn validate_family_order(&mut self) -> Result<(), MetaPickerError> {
        let raw = self
            .predictor
            .model()
            .metadata()
            .get_utf8(FAMILY_ORDER_KEY)
            .map_err(|e| MetaPickerError::Metadata(format!("metadata: {:?}", e)))?;

        let parsed = parse_family_csv(raw).ok_or_else(|| {
            MetaPickerError::Metadata(format!("unparseable family_order CSV: {:?}", raw))
        })?;

        let n_out = self.predictor.n_outputs();
        if parsed.len() != n_out {
            return Err(MetaPickerError::Metadata(format!(
                "family_order length {} != n_outputs {}",
                parsed.len(),
                n_out
            )));
        }
        // Cache (replacing whatever was set at construction).
        self.family_at_output = Some(parsed);
        Ok(())
    }

    /// The bake's `output_index → CodecFamily` mapping, parsed once
    /// from the [`FAMILY_ORDER_KEY`] metadata at construction.
    /// `None` when the metadata was missing or unparseable. Callers
    /// can use this to enumerate which families this bake can pick
    /// (e.g. for surfacing UI / honouring caller-supplied
    /// allow-lists).
    pub fn family_at_output(&self) -> Option<&[CodecFamily]> {
        self.family_at_output.as_deref()
    }
}

/// Parse `"jpeg,webp,avif,…"` into a vector of [`CodecFamily`].
/// Returns `None` if any label isn't recognised.
fn parse_family_csv(csv: &str) -> Option<alloc::vec::Vec<CodecFamily>> {
    csv.split(',')
        .map(|tok| match tok.trim() {
            "jpeg" => Some(CodecFamily::Jpeg),
            "webp" => Some(CodecFamily::Webp),
            "jxl" => Some(CodecFamily::Jxl),
            "avif" => Some(CodecFamily::Avif),
            "png" => Some(CodecFamily::Png),
            "gif" => Some(CodecFamily::Gif),
            _ => None,
        })
        .collect::<Option<alloc::vec::Vec<_>>>()
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
    fn parse_family_csv_full() {
        let parsed = parse_family_csv(ALL_LABELS_CSV).unwrap();
        assert_eq!(parsed, CodecFamily::ALL.to_vec());
    }

    #[test]
    fn parse_family_csv_subset_no_gif() {
        let parsed = parse_family_csv("avif,jpeg,jxl,png,webp").unwrap();
        assert_eq!(
            parsed,
            vec![
                CodecFamily::Avif,
                CodecFamily::Jpeg,
                CodecFamily::Jxl,
                CodecFamily::Png,
                CodecFamily::Webp,
            ]
        );
    }

    #[test]
    fn parse_family_csv_unknown_label_returns_none() {
        assert!(parse_family_csv("jpeg,unknown,webp").is_none());
    }

    #[test]
    fn parse_family_csv_handles_whitespace() {
        let parsed = parse_family_csv(" jpeg , webp ").unwrap();
        assert_eq!(parsed, vec![CodecFamily::Jpeg, CodecFamily::Webp]);
    }

    #[test]
    fn time_cost_table_uses_codec_family_index() {
        // Assert the synthetic ms-table layout my docstring claims:
        // index = CodecFamily::ALL position, NOT bake order. The
        // `pick_with_time_cost` impl re-indexes into bake order
        // internally — this test fixes the contract.
        let mut table = [0.0_f32; CodecFamily::COUNT];
        table[CodecFamily::Jpeg.index()] = 50.0;
        table[CodecFamily::Webp.index()] = 80.0;
        table[CodecFamily::Jxl.index()] = 800.0;

        // CodecFamily::Jpeg is index 0, Webp is 1, Jxl is 2.
        assert_eq!(table[0], 50.0);
        assert_eq!(table[1], 80.0);
        assert_eq!(table[2], 800.0);

        // A bake-order CSV like "avif,jpeg,jxl" should re-index to
        // [Avif, Jpeg, Jxl] — Avif is index 3, so table[3]=0.0.
        let bake_order = parse_family_csv("avif,jpeg,jxl").unwrap();
        let reindexed: alloc::vec::Vec<f32> = bake_order.iter().map(|f| table[f.index()]).collect();
        assert_eq!(reindexed, vec![0.0, 50.0, 800.0]);
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
