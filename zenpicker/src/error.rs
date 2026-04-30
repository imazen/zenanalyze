//! Errors raised by model loading and inference.

use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickerError {
    /// Header magic bytes don't match `ZNPK`.
    BadMagic { found: [u8; 4] },
    /// Format version not supported by this build.
    UnsupportedVersion { version: u16, max_supported: u16 },
    /// Header advertises a `header_size` smaller than v1 minimum.
    HeaderTooSmall { advertised: u16, min: u16 },
    /// Bytes ran out before the model was fully parsed.
    Truncated {
        offset: usize,
        want: usize,
        have: usize,
    },
    /// Reserved error variant — was raised in pre-0.1 builds when
    /// f16 weights weren't yet supported. Kept for ABI stability;
    /// never produced by the current parser.
    F16NotSupported,
    /// `weight_dtype` byte was not 0 (f32), 1 (f16), or 2 (i8).
    UnknownWeightDtype { byte: u8 },
    /// `activation` byte was not a recognized variant.
    UnknownActivation { byte: u8 },
    /// Layer's `in_dim` doesn't match the prior layer's `out_dim`
    /// (or, for layer 0, doesn't match the model's `n_inputs`).
    LayerDimMismatch {
        layer: usize,
        expected_in: usize,
        got_in: usize,
    },
    /// The final layer's `out_dim` doesn't match the header's
    /// `n_outputs`.
    OutputDimMismatch { expected: usize, got: usize },
    /// A header dimension was zero where it must be positive.
    ZeroDimension { what: &'static str },
    /// Caller passed a feature vector of the wrong length.
    FeatureLenMismatch { expected: usize, got: usize },
    /// `CostAdjust::per_output_offset` length didn't match
    /// `n_outputs`.
    AdjustLenMismatch { expected: usize, got: usize },
}

impl fmt::Display for PickerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BadMagic { found } => {
                write!(f, "zenpicker: bad magic, expected ZNPK, found {found:?}")
            }
            Self::UnsupportedVersion {
                version,
                max_supported,
            } => write!(
                f,
                "zenpicker: format version {version} not supported (max supported: {max_supported})"
            ),
            Self::HeaderTooSmall { advertised, min } => {
                write!(f, "zenpicker: header_size {advertised} < minimum {min}")
            }
            Self::Truncated { offset, want, have } => write!(
                f,
                "zenpicker: truncated at offset {offset}, wanted {want} bytes, have {have}"
            ),
            Self::F16NotSupported => write!(
                f,
                "zenpicker: f16 weights flagged as unsupported (legacy error; current parser handles f16 directly)"
            ),
            Self::UnknownWeightDtype { byte } => {
                write!(f, "zenpicker: unknown weight dtype byte {byte:#x}")
            }
            Self::UnknownActivation { byte } => {
                write!(f, "zenpicker: unknown activation byte {byte:#x}")
            }
            Self::LayerDimMismatch {
                layer,
                expected_in,
                got_in,
            } => write!(
                f,
                "zenpicker: layer {layer} expected in_dim {expected_in}, got {got_in}"
            ),
            Self::OutputDimMismatch { expected, got } => write!(
                f,
                "zenpicker: final layer out_dim {got} != header n_outputs {expected}"
            ),
            Self::ZeroDimension { what } => {
                write!(f, "zenpicker: zero dimension in `{what}`")
            }
            Self::FeatureLenMismatch { expected, got } => write!(
                f,
                "zenpicker: feature vector length {got} != n_inputs {expected}"
            ),
            Self::AdjustLenMismatch { expected, got } => write!(
                f,
                "zenpicker: per_output_offset length {got} != n_outputs {expected}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PickerError {}
