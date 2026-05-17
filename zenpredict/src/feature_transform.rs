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
    /// `sign(x) · ln(1 + |x|)` — bidirectional log. Useful for
    /// centered-zero features (sign-balance metrics, residuals)
    /// that can be either positive or negative and need
    /// tail-compression in both directions. Added 2026-05-14 for
    /// V0_20 input-shaping research.
    SignedLog1p,
    /// `sign(x) · sqrt(|x|)` — variance-stabilizing, milder than
    /// log. Good for moderately-skewed features that don't quite
    /// warrant log. Added 2026-05-14.
    SignedSqrt,
    /// `sign(x) · cbrt(|x|)` — even milder than sqrt. Preserves
    /// direction with reduced tail compression. Added 2026-05-14.
    SignedCbrt,
    /// `ln(1 + max(0, x − ε))` — clip-then-log1p. Useful for features
    /// with a known noise floor `ε` below which the value is
    /// statistically meaningless. Requires **1 parameter** (`ε`) in
    /// the [`crate::keys::FEATURE_TRANSFORM_PARAMS`] metadata. With
    /// `params = []` falls back to plain `Log1p`. Added 2026-05-14
    /// for V0_20.
    ClipThenLog1p,
    /// Clip to `[p1, p99]` of the training distribution before
    /// passing through. Robust to outliers; preserves rank order
    /// within `[p1, p99]`. Requires **2 parameters** (`p1`, `p99`)
    /// in [`crate::keys::FEATURE_TRANSFORM_PARAMS`]. With
    /// `params = []` falls back to `Identity`. Added 2026-05-14.
    WinsorP99,
    /// Bucket into N quantile bins; output is `bin_index / N` in
    /// `[0, 1]`. Captures arbitrary monotone non-linearity at the
    /// cost of a step-function output. Requires **N parameters**
    /// (the edges, sorted ascending) in
    /// [`crate::keys::FEATURE_TRANSFORM_PARAMS`]. With `params = []`
    /// falls back to `Identity`. Added 2026-05-14.
    QuantileBins,
    /// `ln(clip(x, p1, p99))` — winsorize first, then natural log.
    /// The dominant high-win stack across zenjpeg / zenwebp / zenavif
    /// in the 2026-05-17 stacks sweep (12+10+13 wins). Use for
    /// strictly positive features with heavy upper tails: winsor
    /// caps both tails, the log compresses what's left.
    ///
    /// Requires **2 parameters** (`p1`, `p99`) in
    /// [`crate::keys::FEATURE_TRANSFORM_PARAMS`]. The bake-side
    /// validator MUST reject `p1 <= 0` since `ln(0)` is undefined;
    /// see `zenpredict-bake`'s composer validation. With
    /// `params = []` falls back to plain `Log` on `x` (caller error
    /// — model expects params to be present at bake time).
    /// Added 2026-05-17.
    WinsorThenLog,
    /// `ln(1 + clip(x, p1, p99))` — winsorize first, then log1p.
    /// Secondary stack winner. Use for non-negative-with-zero features
    /// (counts, densities) where the upper tail dominates: winsor
    /// caps the high end, log1p compresses without barfing at zero.
    ///
    /// Requires **2 parameters** (`p1`, `p99`). The bake-side
    /// validator MUST reject `p1 < -1` since `ln(1 + (-1)) = ln(0)`
    /// is undefined. With `params = []` falls back to plain `Log1p`.
    /// Added 2026-05-17.
    WinsorThenLog1p,
    /// `sign(y) · cbrt(|y|)` where `y = clip(x, p1, p99)` —
    /// winsorize, then signed-cbrt. Scattered wins (0+1+1) but
    /// included for completeness of the WinsorThen* family. Use
    /// for centered-zero features that benefit from both
    /// outlier-clipping and mild tail compression.
    ///
    /// Requires **2 parameters** (`p1`, `p99`). With `params = []`
    /// falls back to plain `SignedCbrt` on `x`. Added 2026-05-17.
    WinsorThenSignedCbrt,
    /// `clip(sign(x) · cbrt(|x|), q1, q99)` — signed-cbrt first,
    /// then winsorize the cbrt-domain result. Scattered wins.
    /// `q1` / `q99` are bounds in cbrt-transformed space, not raw
    /// feature space — different semantics from `WinsorThenSignedCbrt`.
    ///
    /// Requires **2 parameters** (`q1`, `q99` — cbrt-domain bounds).
    /// With `params = []` falls back to plain `SignedCbrt`.
    /// Added 2026-05-17.
    SignedCbrtThenWinsor,
    /// `clip(ln(1 + max(0, x − ε)), q1, q99)` — clip-then-log1p
    /// first, then winsorize the log-domain result. Second most
    /// common stack winner (7+8+8). Use for noise-floor-bounded
    /// features whose log-domain distribution still has heavy
    /// upper tails.
    ///
    /// Requires **3 parameters** (`ε`, `q1`, `q99`). Bake-side
    /// validator SHOULD reject `ε < 0` (negative noise floor
    /// doesn't subtract anything below zero), `q1 > q99` (inverted
    /// bounds), and may warn on `q1 < 0` since the inner stage
    /// produces non-negative output for `ε ≥ 0`. With `params = []`
    /// falls back to plain `Log1p` on `max(0, x)`. Added 2026-05-17.
    ClipThenLog1pThenWinsor,
}

/// Branchless inclusive clamp that tolerates `lo > hi` (returns `lo`)
/// and `NaN` (returns `NaN`). Mirrors the WinsorP99 arm's behaviour
/// exactly so the new stacked variants stay byte-identical to a
/// `winsor → outer` composition.
#[inline]
fn clamp_inclusive(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// `sign(x) · cbrt(|x|)` — shared helper for the cbrt-family
/// variants. Bit-equivalent across std and no_std builds because
/// both go through `cbrtf`'s monotone f32 path.
#[inline]
fn signed_cbrt(x: f32) -> f32 {
    let s = if x >= 0.0 { 1.0 } else { -1.0 };
    #[cfg(feature = "std")]
    {
        s * x.abs().cbrt()
    }
    #[cfg(not(feature = "std"))]
    {
        s * libm::cbrtf(libm::fabsf(x))
    }
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
            #[cfg(feature = "std")]
            Self::SignedLog1p => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * x.abs().ln_1p()
            }
            #[cfg(feature = "std")]
            Self::SignedSqrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * x.abs().sqrt()
            }
            #[cfg(feature = "std")]
            Self::SignedCbrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * x.abs().cbrt()
            }
            #[cfg(not(feature = "std"))]
            Self::SignedLog1p => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * libm::log1pf(libm::fabsf(x))
            }
            #[cfg(not(feature = "std"))]
            Self::SignedSqrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * libm::sqrtf(libm::fabsf(x))
            }
            #[cfg(not(feature = "std"))]
            Self::SignedCbrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * libm::cbrtf(libm::fabsf(x))
            }
            // Parameterized variants without params degrade to a sane
            // no-op fallback. Callers that have params should use
            // `apply_with_params` instead. ClipThenLog1p with ε = 0
            // reduces to plain Log1p; WinsorP99 with no clamp range
            // and QuantileBins with no edges fall back to Identity.
            #[cfg(feature = "std")]
            Self::ClipThenLog1p => x.max(0.0).ln_1p(),
            #[cfg(not(feature = "std"))]
            Self::ClipThenLog1p => {
                let y = if x > 0.0 { x } else { 0.0 };
                libm::log1pf(y)
            }
            // Stacked variants without params degrade to their outer
            // step's no-param behaviour (the inner winsor/clip has
            // no clamp range, so it's an identity pass). The signed
            // variants apply the unparameterized signed transform.
            // `WinsorThenLog` falls back to plain `Log` only when x
            // is strictly positive — `ln(0)` / `ln(-)` produce NaN /
            // -Inf which matches the Python `t_log` semantics (NaN
            // for non-positive inputs).
            #[cfg(feature = "std")]
            Self::WinsorThenLog => x.ln(),
            #[cfg(not(feature = "std"))]
            Self::WinsorThenLog => libm::logf(x),
            #[cfg(feature = "std")]
            Self::WinsorThenLog1p => x.ln_1p(),
            #[cfg(not(feature = "std"))]
            Self::WinsorThenLog1p => libm::log1pf(x),
            #[cfg(feature = "std")]
            Self::WinsorThenSignedCbrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * x.abs().cbrt()
            }
            #[cfg(not(feature = "std"))]
            Self::WinsorThenSignedCbrt => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * libm::cbrtf(libm::fabsf(x))
            }
            #[cfg(feature = "std")]
            Self::SignedCbrtThenWinsor => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * x.abs().cbrt()
            }
            #[cfg(not(feature = "std"))]
            Self::SignedCbrtThenWinsor => {
                let s = if x >= 0.0 { 1.0 } else { -1.0 };
                s * libm::cbrtf(libm::fabsf(x))
            }
            // `ClipThenLog1pThenWinsor` with no params: ε = 0, no
            // outer clamp → reduces to `log1p(max(0, x))`.
            #[cfg(feature = "std")]
            Self::ClipThenLog1pThenWinsor => x.max(0.0).ln_1p(),
            #[cfg(not(feature = "std"))]
            Self::ClipThenLog1pThenWinsor => {
                let y = if x > 0.0 { x } else { 0.0 };
                libm::log1pf(y)
            }
            Self::WinsorP99 | Self::QuantileBins => x,
        }
    }

    /// Apply with optional per-feature params from
    /// [`crate::keys::FEATURE_TRANSFORM_PARAMS`]. Non-parameterized
    /// variants ignore `params`. Parameterized variants:
    ///
    /// - `ClipThenLog1p`: `params = [epsilon]` → `ln(1 + max(0, x − ε))`.
    ///   Empty params reduces to `Log1p` on `max(0, x)`.
    /// - `WinsorP99`: `params = [p1, p99]` → `x.clamp(p1, p99)`.
    ///   Empty params reduces to `Identity`.
    /// - `QuantileBins`: `params = [e0, e1, …, e_{N-1}]` (edges, sorted
    ///   ascending) → number of edges `x ≥ ei` divided by `N`. Empty
    ///   params reduces to `Identity`.
    /// - `WinsorThenLog`: `params = [p1, p99]` → `ln(clamp(x, p1, p99))`.
    ///   Empty / single-element params fall back to plain `Log`.
    /// - `WinsorThenLog1p`: `params = [p1, p99]` →
    ///   `ln(1 + clamp(x, p1, p99))`. Empty / single-element params
    ///   fall back to plain `Log1p`.
    /// - `WinsorThenSignedCbrt`: `params = [p1, p99]` →
    ///   `signed_cbrt(clamp(x, p1, p99))`. Falls back to plain
    ///   `SignedCbrt`.
    /// - `SignedCbrtThenWinsor`: `params = [q1, q99]` →
    ///   `clamp(signed_cbrt(x), q1, q99)` (`q1`, `q99` are bounds in
    ///   cbrt-space). Falls back to plain `SignedCbrt`.
    /// - `ClipThenLog1pThenWinsor`: `params = [ε, q1, q99]` →
    ///   `clamp(ln(1 + max(0, x − ε)), q1, q99)`. Falls back to
    ///   `Log1p` on `max(0, x)` when fewer than 3 params are given.
    ///
    /// Edges for `QuantileBins` are sorted at training time; the
    /// runtime walks them linearly (N ≤ 32 in practice, so a
    /// branchless linear scan is faster than binary search on small N).
    #[inline]
    pub fn apply_with_params(self, x: f32, params: &[f32]) -> f32 {
        match self {
            Self::ClipThenLog1p => {
                let eps = params.first().copied().unwrap_or(0.0);
                let y = x - eps;
                let y_clip = if y > 0.0 { y } else { 0.0 };
                #[cfg(feature = "std")]
                {
                    y_clip.ln_1p()
                }
                #[cfg(not(feature = "std"))]
                {
                    libm::log1pf(y_clip)
                }
            }
            Self::WinsorP99 => {
                let lo = params.first().copied().unwrap_or(f32::NEG_INFINITY);
                let hi = params.get(1).copied().unwrap_or(f32::INFINITY);
                if x < lo {
                    lo
                } else if x > hi {
                    hi
                } else {
                    x
                }
            }
            Self::QuantileBins => {
                let n = params.len();
                if n == 0 {
                    return x;
                }
                let mut idx = 0.0f32;
                for &edge in params {
                    if x >= edge {
                        idx += 1.0;
                    }
                }
                idx / (n as f32)
            }
            // Stacked variants. Math mirrors
            // `zentrain/tools/feature_transform_sweep.py` exactly:
            // inner first (clip/winsor), then outer (log family).
            Self::WinsorThenLog => {
                if params.len() < 2 {
                    return self.apply(x);
                }
                let lo = params[0];
                let hi = params[1];
                let y = clamp_inclusive(x, lo, hi);
                #[cfg(feature = "std")]
                {
                    y.ln()
                }
                #[cfg(not(feature = "std"))]
                {
                    libm::logf(y)
                }
            }
            Self::WinsorThenLog1p => {
                if params.len() < 2 {
                    return self.apply(x);
                }
                let lo = params[0];
                let hi = params[1];
                let y = clamp_inclusive(x, lo, hi);
                #[cfg(feature = "std")]
                {
                    y.ln_1p()
                }
                #[cfg(not(feature = "std"))]
                {
                    libm::log1pf(y)
                }
            }
            Self::WinsorThenSignedCbrt => {
                if params.len() < 2 {
                    return self.apply(x);
                }
                let lo = params[0];
                let hi = params[1];
                let y = clamp_inclusive(x, lo, hi);
                signed_cbrt(y)
            }
            Self::SignedCbrtThenWinsor => {
                if params.len() < 2 {
                    return self.apply(x);
                }
                let q_lo = params[0];
                let q_hi = params[1];
                let y = signed_cbrt(x);
                clamp_inclusive(y, q_lo, q_hi)
            }
            Self::ClipThenLog1pThenWinsor => {
                if params.len() < 3 {
                    return self.apply(x);
                }
                let eps = params[0];
                let q_lo = params[1];
                let q_hi = params[2];
                let shifted = x - eps;
                let clipped = if shifted > 0.0 { shifted } else { 0.0 };
                #[cfg(feature = "std")]
                let y = clipped.ln_1p();
                #[cfg(not(feature = "std"))]
                let y = libm::log1pf(clipped);
                clamp_inclusive(y, q_lo, q_hi)
            }
            _ => self.apply(x),
        }
    }

    /// Returns true for variants that consume per-feature params from
    /// [`crate::keys::FEATURE_TRANSFORM_PARAMS`]. Used by metadata
    /// validation to flag inconsistencies (e.g., a `WinsorP99` feature
    /// with no params is silently degraded to `Identity` — usually
    /// caller error).
    #[inline]
    pub fn requires_params(self) -> bool {
        matches!(
            self,
            Self::ClipThenLog1p
                | Self::WinsorP99
                | Self::QuantileBins
                | Self::WinsorThenLog
                | Self::WinsorThenLog1p
                | Self::WinsorThenSignedCbrt
                | Self::SignedCbrtThenWinsor
                | Self::ClipThenLog1pThenWinsor
        )
    }

    /// Parse one of the wire-format token strings. Unknown tokens
    /// produce `PredictError::MetadataValueNotUtf8` — a tighter error
    /// variant would be welcome but adding one would be a breaking
    /// change to a recently-published 0.1.x crate; the caller path
    /// from [`crate::Model`] already routes parse failures through
    /// the same generic-metadata error class.
    /// Parse a wire-format token (e.g. `"log1p"`) into a variant. Round-trips
    /// with [`Self::as_token`]. Unknown tokens return
    /// [`PredictError::UnknownFeatureTransform`].
    pub fn from_token(s: &str) -> Result<Self, PredictError> {
        match s {
            "identity" => Ok(Self::Identity),
            "log" => Ok(Self::Log),
            "log1p" => Ok(Self::Log1p),
            "signed_log1p" => Ok(Self::SignedLog1p),
            "signed_sqrt" => Ok(Self::SignedSqrt),
            "signed_cbrt" => Ok(Self::SignedCbrt),
            "clip_then_log1p" => Ok(Self::ClipThenLog1p),
            "winsor_p99" => Ok(Self::WinsorP99),
            "quantile_bins" => Ok(Self::QuantileBins),
            "winsor_then_log" => Ok(Self::WinsorThenLog),
            "winsor_then_log1p" => Ok(Self::WinsorThenLog1p),
            "winsor_then_signed_cbrt" => Ok(Self::WinsorThenSignedCbrt),
            "signed_cbrt_then_winsor" => Ok(Self::SignedCbrtThenWinsor),
            "clip_then_log1p_then_winsor" => Ok(Self::ClipThenLog1pThenWinsor),
            _ => Err(PredictError::UnknownFeatureTransform),
        }
    }

    /// Wire-format string — round-trips with [`Self::from_token`].
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Log => "log",
            Self::Log1p => "log1p",
            Self::SignedLog1p => "signed_log1p",
            Self::SignedSqrt => "signed_sqrt",
            Self::SignedCbrt => "signed_cbrt",
            Self::ClipThenLog1p => "clip_then_log1p",
            Self::WinsorP99 => "winsor_p99",
            Self::QuantileBins => "quantile_bins",
            Self::WinsorThenLog => "winsor_then_log",
            Self::WinsorThenLog1p => "winsor_then_log1p",
            Self::WinsorThenSignedCbrt => "winsor_then_signed_cbrt",
            Self::SignedCbrtThenWinsor => "signed_cbrt_then_winsor",
            Self::ClipThenLog1pThenWinsor => "clip_then_log1p_then_winsor",
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
    let text =
        core::str::from_utf8(entry.value).map_err(|_| PredictError::MetadataValueNotUtf8 {
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

/// Read the `zentrain.feature_transform_params` metadata into a typed
/// list of per-feature `Vec<f32>` of length `n_inputs`. The wire
/// format is UTF-8, newline-separated, each line a comma-separated
/// list of f32 values (an empty line means "no params for this
/// feature"). Returns `Ok(None)` when the key is absent.
///
/// Per the bake convention, the line count MUST equal `n_inputs`
/// when present. Empty trailing lines (text ending with `\n`) produce
/// a stray entry — rejected as malformed.
///
/// Numeric tokens are parsed with [`f32::from_str`]; non-finite values
/// (`NaN` / `Inf`) round-trip as the matching IEEE-754 bit pattern,
/// which is what the trainer emits for guard rails on
/// [`FeatureTransform::WinsorP99`].
pub(crate) fn parse_feature_transform_params(
    metadata: &Metadata<'_>,
    n_inputs: usize,
) -> Result<Option<alloc::vec::Vec<alloc::vec::Vec<f32>>>, PredictError> {
    let Some(entry) = metadata.get(crate::keys::FEATURE_TRANSFORM_PARAMS) else {
        return Ok(None);
    };
    if entry.kind != MetadataType::Utf8 {
        return Err(PredictError::MetadataTypeMismatch {
            key_len: crate::keys::FEATURE_TRANSFORM_PARAMS.len(),
            expected: MetadataType::Utf8,
            got: entry.kind,
        });
    }
    let text =
        core::str::from_utf8(entry.value).map_err(|_| PredictError::MetadataValueNotUtf8 {
            key_len: crate::keys::FEATURE_TRANSFORM_PARAMS.len(),
        })?;
    let mut params = alloc::vec::Vec::with_capacity(n_inputs);
    for line in text.split('\n') {
        let mut row: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
        if !line.is_empty() {
            for tok in line.split(',') {
                let v: f32 =
                    tok.trim()
                        .parse()
                        .map_err(|_| PredictError::MetadataValueNotUtf8 {
                            key_len: crate::keys::FEATURE_TRANSFORM_PARAMS.len(),
                        })?;
                row.push(v);
            }
        }
        params.push(row);
    }
    if params.len() != n_inputs {
        return Err(PredictError::FeatureTransformsLenMismatch {
            expected: n_inputs,
            got: params.len(),
        });
    }
    Ok(Some(params))
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

/// Apply each transform with optional per-feature params, writing into
/// `dst`. `transforms.len()`, `params.len()`, and `dst.len()` must all
/// equal `src.len()`.
///
/// Each `params[i]` is a `&[f32]` slice carrying the variant-specific
/// parameters. Empty slices fall back to the no-param variant of
/// `apply` (e.g., `WinsorP99` with no params → `Identity`).
///
/// **API-stability note**: this fn is currently unused inside
/// zenpredict — `Predictor::predict_transformed` calls
/// `apply_feature_transforms` directly with the per-feature param
/// slices already resolved. The `_with_params` parallel exists for
/// downstream consumers that want to apply transforms without
/// owning a `Predictor` (i.e., feature-preprocessing pipelines).
/// Kept per the 0.1.x API-stability contract (see
/// zenanalyze/CLAUDE.md "There will never be a 0.2.x").
#[allow(dead_code)]
#[inline]
pub fn apply_feature_transforms_with_params(
    transforms: &[FeatureTransform],
    params: &[&[f32]],
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), PredictError> {
    if transforms.len() != src.len() || dst.len() != src.len() || params.len() != src.len() {
        return Err(PredictError::FeatureTransformsLenMismatch {
            expected: src.len(),
            got: transforms.len(),
        });
    }
    for i in 0..src.len() {
        dst[i] = transforms[i].apply_with_params(src[i], params[i]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_token_round_trips_parameterized_variants() {
        for tok in [
            "clip_then_log1p",
            "winsor_p99",
            "quantile_bins",
            "winsor_then_log",
            "winsor_then_log1p",
            "winsor_then_signed_cbrt",
            "signed_cbrt_then_winsor",
            "clip_then_log1p_then_winsor",
        ] {
            let v = FeatureTransform::from_token(tok).expect("parse");
            assert_eq!(v.as_token(), tok);
            assert!(v.requires_params(), "{tok} should require params");
        }
    }

    #[test]
    fn non_parameterized_variants_do_not_require_params() {
        for v in [
            FeatureTransform::Identity,
            FeatureTransform::Log,
            FeatureTransform::Log1p,
            FeatureTransform::SignedLog1p,
            FeatureTransform::SignedSqrt,
            FeatureTransform::SignedCbrt,
        ] {
            assert!(!v.requires_params(), "{:?} should not require params", v);
        }
    }

    #[test]
    fn clip_then_log1p_with_epsilon() {
        let t = FeatureTransform::ClipThenLog1p;
        // ln(1 + max(0, 10 - 2)) = ln(1 + 8) = 8f32.ln_1p() ≈ 2.1972
        let y = t.apply_with_params(10.0, &[2.0]);
        assert!((y - 8f32.ln_1p()).abs() < 1e-6);
        // below ε clips to 0 → ln(1 + 0) = 0
        let y_low = t.apply_with_params(1.0, &[2.0]);
        assert_eq!(y_low, 0.0);
        // empty params reduces to plain log1p on max(0, x)
        let y_no_params = t.apply_with_params(3.0, &[]);
        assert!((y_no_params - 3f32.ln_1p()).abs() < 1e-6);
    }

    #[test]
    fn winsor_p99_clips_to_range() {
        let t = FeatureTransform::WinsorP99;
        assert_eq!(t.apply_with_params(0.5, &[1.0, 99.0]), 1.0); // below p1
        assert_eq!(t.apply_with_params(50.0, &[1.0, 99.0]), 50.0); // in range
        assert_eq!(t.apply_with_params(200.0, &[1.0, 99.0]), 99.0); // above p99
        // empty params reduces to Identity
        assert_eq!(t.apply_with_params(42.0, &[]), 42.0);
    }

    #[test]
    fn quantile_bins_buckets_input() {
        let t = FeatureTransform::QuantileBins;
        let edges = [0.1, 0.3, 0.5, 0.7, 0.9];
        // x < first edge: 0 / 5
        assert!((t.apply_with_params(0.05, &edges) - 0.0).abs() < 1e-6);
        // x between edge 0 and 1: 1 / 5
        assert!((t.apply_with_params(0.2, &edges) - 0.2).abs() < 1e-6);
        // x ≥ all edges: 5 / 5 = 1.0
        assert!((t.apply_with_params(1.0, &edges) - 1.0).abs() < 1e-6);
        // empty params reduces to Identity
        assert_eq!(t.apply_with_params(0.42, &[]), 0.42);
    }

    #[test]
    fn apply_without_params_is_no_op_for_parameterized_variants() {
        // apply(self, x) -> f32 — old code path — must produce sane
        // no-op fallback for the new variants.
        assert_eq!(FeatureTransform::WinsorP99.apply(42.0), 42.0);
        assert_eq!(FeatureTransform::QuantileBins.apply(0.5), 0.5);
        // ClipThenLog1p with no ε reduces to log1p(max(0, x))
        let y = FeatureTransform::ClipThenLog1p.apply(3.0);
        assert!((y - 3f32.ln_1p()).abs() < 1e-6);
    }

    // ───── Stacked-variant tests (2026-05-17) ─────

    #[test]
    fn winsor_then_log_applies_winsor_then_log() {
        let t = FeatureTransform::WinsorThenLog;
        // In-range: clip noop, log applies.
        let y = t.apply_with_params(10.0, &[1.0, 99.0]);
        assert!((y - 10f32.ln()).abs() < 1e-6);
        // Below p1: clip up, then log.
        let y_lo = t.apply_with_params(0.5, &[1.0, 99.0]);
        assert!((y_lo - 1f32.ln()).abs() < 1e-6);
        assert_eq!(y_lo, 0.0);
        // Above p99: clip down, then log.
        let y_hi = t.apply_with_params(500.0, &[1.0, 99.0]);
        assert!((y_hi - 99f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn winsor_then_log_no_params_falls_back_to_plain_log() {
        // apply() is the no-params path. Plain Log on strictly
        // positive input.
        let y = FeatureTransform::WinsorThenLog.apply(7.0);
        assert!((y - 7f32.ln()).abs() < 1e-6);
        // apply_with_params with empty params also falls back.
        let y2 = FeatureTransform::WinsorThenLog.apply_with_params(7.0, &[]);
        assert!((y2 - 7f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn winsor_then_log1p_applies_winsor_then_log1p() {
        let t = FeatureTransform::WinsorThenLog1p;
        // In-range: clip noop, log1p applies.
        let y = t.apply_with_params(10.0, &[0.0, 99.0]);
        assert!((y - 10f32.ln_1p()).abs() < 1e-6);
        // Below p1=0 (winsor allows non-negative lower bound): clip to 0.
        let y_lo = t.apply_with_params(-5.0, &[0.0, 99.0]);
        assert_eq!(y_lo, 0.0); // ln_1p(0) = 0
        // Above p99: clip down.
        let y_hi = t.apply_with_params(200.0, &[0.0, 99.0]);
        assert!((y_hi - 99f32.ln_1p()).abs() < 1e-6);
    }

    #[test]
    fn winsor_then_signed_cbrt_applies_in_order() {
        let t = FeatureTransform::WinsorThenSignedCbrt;
        // Inside range: signed_cbrt of the value.
        let y = t.apply_with_params(8.0, &[-100.0, 100.0]);
        assert!((y - 8f32.cbrt()).abs() < 1e-6);
        // Below p1=-100: clip to -100, then signed_cbrt(-100) = -cbrt(100).
        let y_lo = t.apply_with_params(-500.0, &[-100.0, 100.0]);
        assert!((y_lo - (-100f32).abs().cbrt() * -1.0).abs() < 1e-5);
        // Above p99=100: clip to 100, then cbrt(100).
        let y_hi = t.apply_with_params(500.0, &[-100.0, 100.0]);
        assert!((y_hi - 100f32.cbrt()).abs() < 1e-5);
    }

    #[test]
    fn signed_cbrt_then_winsor_clips_in_cbrt_space() {
        let t = FeatureTransform::SignedCbrtThenWinsor;
        // signed_cbrt(8) = 2; bounds [-1.5, 1.5] → clip to 1.5.
        let y = t.apply_with_params(8.0, &[-1.5, 1.5]);
        assert!((y - 1.5).abs() < 1e-6);
        // signed_cbrt(-8) = -2; clip up to -1.5.
        let y_neg = t.apply_with_params(-8.0, &[-1.5, 1.5]);
        assert!((y_neg - (-1.5)).abs() < 1e-6);
        // signed_cbrt(0.001) ≈ 0.1, within range → unchanged.
        let y_in = t.apply_with_params(0.001, &[-1.5, 1.5]);
        let expected = 0.001f32.cbrt();
        assert!((y_in - expected).abs() < 1e-5);
    }

    #[test]
    fn signed_cbrt_then_winsor_no_params_falls_back() {
        // No params → plain signed_cbrt.
        let y = FeatureTransform::SignedCbrtThenWinsor.apply_with_params(8.0, &[]);
        assert!((y - 8f32.cbrt()).abs() < 1e-6);
        let y_neg = FeatureTransform::SignedCbrtThenWinsor.apply_with_params(-27.0, &[]);
        assert!((y_neg - -3.0f32).abs() < 1e-5);
    }

    #[test]
    fn clip_then_log1p_then_winsor_full_pipeline() {
        let t = FeatureTransform::ClipThenLog1pThenWinsor;
        // ε=2, q range [0.0, 3.0] (log-domain bounds).
        // For x=10: shifted=8, log1p(8) ≈ 2.197 — within [0, 3] → unchanged.
        let y = t.apply_with_params(10.0, &[2.0, 0.0, 3.0]);
        assert!((y - 8f32.ln_1p()).abs() < 1e-5);
        // For x=1000: shifted=998, log1p(998) ≈ 6.906 — clipped to 3.0.
        let y_hi = t.apply_with_params(1000.0, &[2.0, 0.0, 3.0]);
        assert!((y_hi - 3.0).abs() < 1e-6);
        // For x=1: shifted=-1 → max(0, -1)=0 → log1p(0)=0 — within range.
        let y_lo = t.apply_with_params(1.0, &[2.0, 0.0, 3.0]);
        assert_eq!(y_lo, 0.0);
    }

    #[test]
    fn clip_then_log1p_then_winsor_partial_params_falls_back() {
        // Fewer than 3 params → fall back to apply() (log1p of max(0, x)).
        let t = FeatureTransform::ClipThenLog1pThenWinsor;
        let y_no = t.apply_with_params(5.0, &[]);
        assert!((y_no - 5f32.ln_1p()).abs() < 1e-6);
        let y_one = t.apply_with_params(5.0, &[1.0]);
        assert!((y_one - 5f32.ln_1p()).abs() < 1e-6);
        let y_two = t.apply_with_params(5.0, &[1.0, 0.0]);
        assert!((y_two - 5f32.ln_1p()).abs() < 1e-6);
    }

    #[test]
    fn stacked_variants_no_params_fallback_is_documented() {
        // apply(x) on each stacked variant: ensure each maps to its
        // documented degenerate behaviour and doesn't panic.
        let x = 4.0f32;
        assert!((FeatureTransform::WinsorThenLog.apply(x) - x.ln()).abs() < 1e-6);
        assert!((FeatureTransform::WinsorThenLog1p.apply(x) - x.ln_1p()).abs() < 1e-6);
        assert!((FeatureTransform::WinsorThenSignedCbrt.apply(x) - x.cbrt()).abs() < 1e-6);
        assert!((FeatureTransform::SignedCbrtThenWinsor.apply(x) - x.cbrt()).abs() < 1e-6);
        assert!((FeatureTransform::ClipThenLog1pThenWinsor.apply(x) - x.ln_1p()).abs() < 1e-6);
    }

    #[test]
    fn stacked_variants_round_trip_through_tokens() {
        let variants = [
            FeatureTransform::WinsorThenLog,
            FeatureTransform::WinsorThenLog1p,
            FeatureTransform::WinsorThenSignedCbrt,
            FeatureTransform::SignedCbrtThenWinsor,
            FeatureTransform::ClipThenLog1pThenWinsor,
        ];
        for v in variants {
            let tok = v.as_token();
            let parsed = FeatureTransform::from_token(tok).expect("parse");
            assert_eq!(parsed, v);
            assert!(parsed.requires_params(), "{tok} should require params");
        }
    }
}
