//! # zenpicker-train — per-codec quality picker trainer
//!
//! Trains a per-codec quality picker by porting zentrain's established
//! within-cell-optimal formulation (`zentrain/tools/train_hybrid.py`)
//! to Rust: load a unified sweep parquet, factor each codec config into
//! a **categorical cell**, and for every `(image, target_zq)` compute
//! the within-cell-optimal `bytes_log[cell]` (= ln of the min encoded
//! bytes over configs in the cell that reach the requested quality).
//! An MLP maps `(image features, zq_norm)` → `bytes_log[0..n_cells]`;
//! the codec picks via `argmin(bytes_log, mask=reach)`.
//!
//! ## NO q-leakage
//!
//! The codec's per-encode `q` is **never** an input feature. `q` is the
//! decision the picker exists to make (and it is monotone with achieved
//! score, so feeding it in trivially inflates any "predict the score"
//! task — that was the prior skeleton's bug). The only inputs are
//! IMAGE features (`feat_*`) + `zq_norm` (the user's REQUESTED target
//! quality / 100). The supervised target is per-cell `bytes_log`, not
//! the achieved score.
//!
//! ## Models
//!
//! - [`Mlp`] — the real picker: a LeakyReLU MLP (matching zentrain's
//!   student topology), N-cell output, trained via [`train_mlp`] with
//!   a bounded hyperparameter [`search`].
//! - [`RidgeModel`] — a legacy single-layer linear baseline kept for
//!   the cheap-reference path.
//!
//! Both bake to **ZNPR v3** via the `zenpredict-bake` JSON pipeline
//! (no hand-rolled wire format) and load through
//! [`zenpredict::Model::from_bytes`] / [`zenpicker::MetaPicker`].
//!
//! ## Data-coverage caveat
//!
//! The available `unified_v13_zenjpeg_cvvdp.parquet` sweeps only 5 `q`
//! levels {10,30,60,80,90}, so the "reaches target_zq" ladder is
//! COARSE — sparse on quality per zensim/CLAUDE.md "Dense sampling for
//! trained models". This validates the FORMULATION and the
//! Rust-vs-zentrain port; a dense q+size sweep is a documented
//! follow-on, not part of this bounded chunk.

#![forbid(unsafe_code)]

mod bake;
mod distill;
mod eval;
mod input_shaping;
mod mlp;
mod model;
mod pareto_dataset;
mod parquet_input;
mod picker_eval;
mod search;

pub use bake::{
    BakeOutcome, DistillManifest, HeldoutManifest, MlpModelManifest, MlpPickerManifest,
    MlpPickerManifestInputs, ModelManifest, PickerManifest, PickerManifestInputs, SearchCandidate,
    SearchManifest, bake_mlp_picker, bake_mlp_picker_to_znpr_v3, bake_picker,
    bake_ridge_to_znpr_v3, file_sha256,
};
pub use distill::{
    SoftTargets, TeacherParams, export_teacher_dataset, load_soft_targets,
    teacher_params_fingerprint,
};
pub use eval::{EvalReport, evaluate};
pub use input_shaping::{FittedTransforms, ShapingMode, apply_inplace, fit_transforms};
pub use mlp::{Mlp, MlpConfig, train_mlp};
pub use model::{RidgeModel, Standardizer, train_ridge};
pub use pareto_dataset::{
    PickerDataset, ScalarAxisSpec, build_picker_dataset, build_picker_dataset_with,
    default_zq_targets, fit_standardizer, grouped_split_picker, standardize_all,
};
pub use parquet_input::{CodecFilter, TrainingData, grouped_split, load_training_rows};
pub use picker_eval::{PickerEval, evaluate_picker, evaluate_picker_bake};
pub use search::{GridPoint, SearchResult, default_grid, run_search, run_search_distill};

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
