//! # zenpicker-train — per-codec quality picker trainer (skeleton)
//!
//! This is the **bounded first chunk** of the `zenpicker-train`
//! binary described in spec §4 of
//! `zenmetrics/docs/ZEN_CLOUD_AND_CONSOLIDATION_SPEC_2026-05-26.md`.
//! It establishes the END-TO-END pipeline — load a unified sweep
//! parquet, filter to one codec, build `(features → target-quality)`
//! rows, train a minimal baseline, emit a **ZNPR v3** bake via the
//! `zenpredict-bake` JSON pipeline + a sibling TOML reproduce-this
//! manifest, and report the held-out [`zenstats`] panel.
//!
//! It is deliberately NOT the full picker trainer. See
//! [`README`](https://github.com/imazen/zenanalyze) and the "Follow-ons"
//! section there for what's intentionally out of scope (the
//! scikit-learn-parity hyperparameter search, CubeCL GPU
//! acceleration, cross-codec `MetaPicker` auto-regeneration, dense
//! size/quality sampling discipline).
//!
//! ## What the skeleton learns
//!
//! A per-codec quality picker maps `(image features, target knobs)
//! → achieved quality`. The codec binary-searches encode params to
//! hit a target score; the picker is what makes that search a single
//! forward pass instead of an encode sweep. This skeleton trains a
//! ridge-regularized linear baseline (one F32 ZNPR layer, identity
//! activation) on `feat_* (+ q)` → `--target-column` and bakes it.
//! A linear model is an honest baseline, not SOTA — the mature
//! non-linear search is a documented follow-on.
//!
//! ## Bake shape
//!
//! The output is a single-output ZNPR v3 model loadable by
//! [`zenpredict::Model::from_bytes`] and (for the regression-head
//! case) by [`zenpicker::MetaPicker::predictor`]. The standardizing
//! scaler is folded into the model's `scaler_mean` / `scaler_scale`;
//! the linear weights live in one identity-activation layer.

#![forbid(unsafe_code)]

mod bake;
mod eval;
mod model;
mod parquet_input;

pub use bake::{
    BakeOutcome, ModelManifest, PickerManifest, PickerManifestInputs, bake_picker,
    bake_ridge_to_znpr_v3, file_sha256,
};
pub use eval::{EvalReport, evaluate};
pub use model::{RidgeModel, Standardizer, train_ridge};
pub use parquet_input::{CodecFilter, TrainingData, grouped_split, load_training_rows};

/// Errors surfaced by the training pipeline.
#[derive(Debug)]
pub enum TrainError {
    /// Parquet / arrow read failure.
    Parquet(String),
    /// No rows matched the requested codec filter.
    EmptyAfterFilter { codec: String },
    /// The requested target column was absent or non-numeric.
    MissingTargetColumn { column: String },
    /// No `feat_*` columns were present in the input.
    NoFeatureColumns,
    /// Bake construction / serialization failed.
    Bake(String),
    /// IO failure writing the bake or manifest.
    Io(String),
    /// A numeric degeneracy made the closed-form fit impossible.
    Degenerate(String),
}

impl core::fmt::Display for TrainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parquet(s) => write!(f, "parquet read failed: {s}"),
            Self::EmptyAfterFilter { codec } => {
                write!(f, "no rows matched codec filter {codec:?}")
            }
            Self::MissingTargetColumn { column } => {
                write!(f, "target column {column:?} missing or non-numeric")
            }
            Self::NoFeatureColumns => write!(f, "no feat_* columns found in input parquet"),
            Self::Bake(s) => write!(f, "bake failed: {s}"),
            Self::Io(s) => write!(f, "io error: {s}"),
            Self::Degenerate(s) => write!(f, "degenerate fit: {s}"),
        }
    }
}

impl std::error::Error for TrainError {}
