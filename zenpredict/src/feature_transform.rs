//! Per-feature pre-standardize transforms (issue #52).
//!
//! Some features (encoded byte counts, pixel counts, edge densities)
//! have power-law-ish distributions where a `log` or `log1p` transform
//! gives the trainer's standardize step a much tighter range to work
//! with. The trainer applies the transform BEFORE fitting
//! `scaler_mean` / `scaler_scale`, so the scaler stats encode the
//! post-transform distribution. Codec runtimes that consume the bake
//! MUST apply the same transform before forward-pass — skipping it
//! feeds the network features whose distribution doesn't match what
//! it was trained on, producing silently-wrong predictions.
//!
//! ## Wire format
//!
//! The bake stores transforms under the metadata key
//! [`crate::keys::FEATURE_TRANSFORMS`] as a UTF-8, newline-separated
//! list parallel to [`crate::keys::FEATURE_COLUMNS`]. Each line is
//! one of `identity`, `log`, or `log1p`. The key is **omitted
//! entirely** when every feature is `identity`; consumers MUST treat
//! absence as "all-identity".
//!
//! Per the bake-side convention (`tools/bake_picker.py`), the line
//! count must equal `n_inputs` when present. This crate enforces the
//! same on parse — a mismatched length is a hard error rather than
//! silent truncation.

use crate::error::PredictError;
use crate::metadata::{Metadata, MetadataType};

/// Pre-standardize transform applied to one feature column before the
/// trainer's scaler runs (and therefore before the runtime's forward
/// pass).
///
/// The `Identity` variant is the no-op — feature flows through
/// unchanged. `Log` uses [`f32::ln`] and is only valid for strictly
/// positive features; pre-clamping is the caller's responsibility (the
/// trainer guarantees this for the columns it ships under `log`).
/// `Log1p` uses [`f32::ln_1p`] for non-negative features that can hit
/// zero — `log1p(0) == 0`, no clamping needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum FeatureTransform {
    /// Pass-through. Same as not declaring a transform.
    #[default]
    Identity,
    /// Natural log. Valid only for strictly positive inputs; the
    /// trainer's `feature_transforms` config is responsible for only
    /// applying this to columns whose corpus distribution is bounded
    /// below by some positive ε (or pre-clamps before transform).
    Log,
    /// `ln(1 + x)`. Valid for `x ≥ 0`. The standard choice for
    /// counts and any feature that can be exactly zero.
    Log1p,
}

impl FeatureTransform {
    /// Apply the transform to a single value.
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Self::Identity => x,
            // `f32::ln` and `f32::ln_1p` are core fns in std builds.
            // For `no_std` we go through `libm` to avoid pulling
            // `intrinsics::logf` (not available without std).
            #[cfg(feature = "std")]
            Self::Log => x.ln(),
            #[cfg(feature = "std")]
            Self::Log1p => x.ln_1p(),
            #[cfg(not(feature = "std"))]
            Self::Log => libm::logf(x),
            #[cfg(not(feature = "std"))]
            Self::Log1p => libm::log1pf(x),
        }
    }

    /// Parse one of the wire-format token strings. Unknown tokens
    /// produce `PredictError::MetadataValueNotUtf8` — a tighter error
    /// variant would be welcome but adding one would be a breaking
    /// change to a recently-published 0.1.x crate; the caller path
    /// from [`crate::Model`] already routes parse failures through
    /// the same generic-metadata error class.
    pub(crate) fn from_token(s: &str) -> Result<Self, PredictError> {
        match s {
            "identity" => Ok(Self::Identity),
            "log" => Ok(Self::Log),
            "log1p" => Ok(Self::Log1p),
            _ => Err(PredictError::UnknownFeatureTransform),
        }
    }

    /// Wire-format string — round-trips with [`Self::from_token`].
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Log => "log",
            Self::Log1p => "log1p",
        }
    }
}

/// Read the `zentrain.feature_transforms` metadata into a typed list
/// of length `n_inputs`. Returns `Ok(None)` when the key is absent
/// (consumers should then treat every feature as `Identity`); returns
/// `Ok(Some(_))` with `len == n_inputs` when present and well-formed.
///
/// Error cases (all hard-fail):
/// - The key is present but its value type is not UTF-8.
/// - The value contains a token that isn't `identity`/`log`/`log1p`.
/// - The line count doesn't equal `n_inputs`.
pub(crate) fn parse_feature_transforms(
    metadata: &Metadata<'_>,
    n_inputs: usize,
) -> Result<Option<alloc::vec::Vec<FeatureTransform>>, PredictError> {
    let Some(entry) = metadata.get(crate::keys::FEATURE_TRANSFORMS) else {
        return Ok(None);
    };
    if entry.kind != MetadataType::Utf8 {
        return Err(PredictError::MetadataTypeMismatch {
            key_len: crate::keys::FEATURE_TRANSFORMS.len(),
            expected: MetadataType::Utf8,
            got: entry.kind,
        });
    }
    // Validated as UTF-8 at parse time, but go through `from_utf8`
    // again so a future refactor can't regress the invariant.
    let text = core::str::from_utf8(entry.value).map_err(|_| PredictError::MetadataValueNotUtf8 {
        key_len: crate::keys::FEATURE_TRANSFORMS.len(),
    })?;
    // Split on '\n' only — bake_picker.py emits newline-separated
    // exactly. An empty trailing line (text ending with '\n') would
    // produce a stray empty entry; reject that as malformed rather
    // than silently dropping it.
    let mut transforms = alloc::vec::Vec::with_capacity(n_inputs);
    for token in text.split('\n') {
        transforms.push(FeatureTransform::from_token(token)?);
    }
    if transforms.len() != n_inputs {
        return Err(PredictError::FeatureTransformsLenMismatch {
            expected: n_inputs,
            got: transforms.len(),
        });
    }
    Ok(Some(transforms))
}

/// Apply each transform to the matching feature, writing into `dst`.
/// `transforms.len()` must equal `src.len()`; `dst.len()` must equal
/// `src.len()`.
#[inline]
pub fn apply_feature_transforms(
    transforms: &[FeatureTransform],
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), PredictError> {
    if transforms.len() != src.len() || dst.len() != src.len() {
        return Err(PredictError::FeatureTransformsLenMismatch {
            expected: src.len(),
            got: transforms.len(),
        });
    }
    for (i, &t) in transforms.iter().enumerate() {
        dst[i] = t.apply(src[i]);
    }
    Ok(())
}
