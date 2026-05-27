//! Teacher → student distillation — the part of zentrain's recipe the
//! direct-hard-target port skipped.
//!
//! ## What zentrain actually does (faithful replication)
//!
//! `zentrain/tools/train_hybrid.py` does NOT train its MLP student on
//! the sparse hard `bytes_log` targets directly. It:
//!
//! 1. **Trains a per-cell HistGradientBoosting teacher.** For each
//!    categorical cell `c`, one `sklearn.ensemble.
//!    HistGradientBoostingRegressor` is fit on `(Xs_tr,
//!    bytes_log_tr[:, c])` over the rows where cell `c` reached the
//!    target (the `reach` mask; sklearn drops the NaN rows). Production
//!    params = `HISTGB_FULL` (`max_iter=400, max_depth=8,
//!    learning_rate=0.05, l2_regularization=0.5`). Cells with < 50
//!    reaching train rows get no teacher (`None`) and fall back to the
//!    per-cell `nanmean(bytes_log_tr)`.
//!    (`train_teacher_per_cell` + `_fit_one_cell`.)
//!
//! 2. **Generates DENSE soft targets.** `teacher_predict_all` runs every
//!    per-cell teacher over ALL train rows, producing a dense
//!    `bytes_pred_tr[n_rows, n_cells]` with NO NaN holes — the teacher
//!    interpolates a smooth cost surface even for (row, cell) pairs that
//!    were unreachable in the sparse sweep. This is the distillation
//!    signal.
//!
//! 3. **Distills the MLP student against the soft targets.** The student
//!    (`_train_torch_leakyrelu_student`) trains on `soft_tr =
//!    bytes_pred_tr` — pure soft-target MSE, NO blend with the hard
//!    `bytes_log`, NO temperature, NO sample weighting (the default
//!    `hard_example_mode = "none"`; hard-example reweighting is an
//!    OFF-BY-DEFAULT diagnostic option, not the recipe). For our
//!    bytes-only picker (no scalar/time/metric heads) `soft_tr` is
//!    exactly the teacher's dense bytes_log predictions, and the
//!    per-head scalar-block standardization is a no-op (single log-bytes
//!    head, left untouched per zentrain).
//!
//! 4. **Evaluates against the HARD oracle.** Both teacher and student
//!    are scored by `argmin(prediction, mask=reach)` vs the true
//!    within-cell-optimal `bytes_log` — never vs the soft targets. So
//!    distillation changes the *training target* only; the held-out
//!    decision-quality metric is unchanged.
//!
//! ## Why the teacher is an offline Python step
//!
//! `HistGradientBoostingRegressor` is a substantial gradient-boosted
//! tree implementation; re-porting it to Rust would be a large,
//! error-prone effort orthogonal to the goal (which is the *student* +
//! the *runtime*). zentrain's parquet/feature pipeline is already Python,
//! so generating the teacher's soft targets ONCE, offline, and persisting
//! them as a content-addressed sidecar parquet is the right call. The
//! Rust RUNTIME gains NO Python dependency — only this one-time
//! target-generation step shells to `scripts/teacher_soft_targets.py`.
//!
//! This module owns the Rust ↔ Python contract:
//!
//! - [`export_teacher_dataset`] writes the exact Rust-built dataset
//!   (raw teacher-input features, hard `bytes_log`, `reach` mask, the
//!   grouped train/val split, row identity) to a parquet keyed by
//!   `row_idx`, so the Python teacher fits on precisely the Rust
//!   trainer's train rows and emits soft targets aligned 1:1.
//! - [`load_soft_targets`] reads the teacher's `soft_targets.parquet`
//!   back, validates the row count + `n_cells` + the source-export
//!   sha256, and returns a dense `n_rows × n_cells` matrix in row order.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use sha2::{Digest, Sha256};

use crate::TrainError;
use crate::bake::file_sha256;
use crate::pareto_dataset::PickerDataset;

/// HistGB teacher hyperparameters recorded in the export so the Python
/// step and the manifest agree on a single source of truth. Mirrors
/// zentrain's `HISTGB_FULL`.
#[derive(Debug, Clone, Copy)]
pub struct TeacherParams {
    pub max_iter: u32,
    pub max_depth: u32,
    pub learning_rate: f64,
    pub l2_regularization: f64,
    /// Minimum reaching train rows for a cell to get its own teacher
    /// (else fall back to the per-cell nanmean). zentrain uses 50.
    pub min_cell_rows: u32,
    pub random_state: u64,
}

impl Default for TeacherParams {
    fn default() -> Self {
        // zentrain HISTGB_FULL + the < 50 cell-row floor.
        Self {
            max_iter: 400,
            max_depth: 8,
            learning_rate: 0.05,
            l2_regularization: 0.5,
            min_cell_rows: 50,
            random_state: 0xCAFE,
        }
    }
}

/// Write the Rust-built picker dataset to a parquet the Python teacher
/// consumes. Columns:
///
/// - `row_idx: i64` — the dataset row index (0..n_rows), the join key.
/// - `image_id: utf8` — for diagnostics / provenance.
/// - `target_zq: i64` — the requested-quality band.
/// - `split: i64` — 0 = train, 1 = val (the grouped-by-image split). The
///   teacher fits on `split == 0` ONLY; it predicts soft targets for ALL
///   rows.
/// - `reach_{c}: i64` (0/1) for `c` in `0..n_cells` — per-cell reach mask.
/// - `bytes_log_{c}: f32` for `c` in `0..n_cells` — hard target (NaN
///   where unreachable). The teacher fits each cell on the non-NaN rows.
/// - `f_{j}: f32` for `j` in `0..n_in` — the RAW (un-standardized)
///   teacher input vector (image features + `zq_norm` as the last
///   column). Trees are invariant to per-feature monotone scaling, so
///   the raw features match zentrain's raw `Xs` teacher inputs.
///
/// Returns the sha256 of the written file (carried into the soft-target
/// sidecar + the manifest so re-runs are verifiable).
pub fn export_teacher_dataset(
    ds: &PickerDataset,
    train_rows: &[usize],
    val_rows: &[usize],
    out_path: &Path,
) -> Result<String, TrainError> {
    let n_rows = ds.n_rows();
    let n_cells = ds.n_cells;
    let n_in = ds.n_in;

    let mut split = vec![1i64; n_rows]; // default val; overwrite train below
    for &r in train_rows {
        split[r] = 0;
    }
    // Rows that are neither in train nor val should not exist, but guard:
    // mark any untouched as val (split stays 1) — harmless for the teacher
    // (it only fits on split==0).
    let _ = val_rows;

    let mut fields: Vec<Field> = vec![
        Field::new("row_idx", DataType::Int64, false),
        Field::new("image_id", DataType::Utf8, false),
        Field::new("target_zq", DataType::Int64, false),
        Field::new("split", DataType::Int64, false),
    ];
    for c in 0..n_cells {
        fields.push(Field::new(format!("reach_{c}"), DataType::Int64, false));
    }
    for c in 0..n_cells {
        fields.push(Field::new(
            format!("bytes_log_{c}"),
            DataType::Float32,
            true,
        ));
    }
    for j in 0..n_in {
        fields.push(Field::new(format!("f_{j}"), DataType::Float32, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let row_idx: Vec<i64> = (0..n_rows as i64).collect();
    let image_id = StringArray::from(ds.image_ids.clone());
    let target_zq = Int64Array::from(ds.target_zq.clone());
    let split_arr = Int64Array::from(split);

    let mut columns: Vec<Arc<dyn Array>> = vec![
        Arc::new(Int64Array::from(row_idx)),
        Arc::new(image_id),
        Arc::new(target_zq),
        Arc::new(split_arr),
    ];
    // reach_{c}
    for c in 0..n_cells {
        let col: Vec<i64> = (0..n_rows)
            .map(|r| if ds.reach[r * n_cells + c] { 1 } else { 0 })
            .collect();
        columns.push(Arc::new(Int64Array::from(col)));
    }
    // bytes_log_{c} (nullable; NaN -> null)
    for c in 0..n_cells {
        let col: Vec<Option<f32>> = (0..n_rows)
            .map(|r| {
                let v = ds.bytes_log[r * n_cells + c];
                if v.is_finite() { Some(v as f32) } else { None }
            })
            .collect();
        columns.push(Arc::new(Float32Array::from(col)));
    }
    // f_{j} raw teacher inputs
    for j in 0..n_in {
        let col: Vec<f32> = (0..n_rows)
            .map(|r| ds.features[r * n_in + j] as f32)
            .collect();
        columns.push(Arc::new(Float32Array::from(col)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| TrainError::Parquet(format!("build export batch: {e}")))?;

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| TrainError::Io(format!("mkdir {}: {e}", parent.display())))?;
    }
    let file = File::create(out_path)
        .map_err(|e| TrainError::Io(format!("create {}: {e}", out_path.display())))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| TrainError::Parquet(format!("arrow writer: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| TrainError::Parquet(format!("write export: {e}")))?;
    writer
        .close()
        .map_err(|e| TrainError::Parquet(format!("close export: {e}")))?;

    file_sha256(out_path)
}

/// The teacher's soft targets, loaded back from the Python sidecar
/// parquet, in dataset row order.
pub struct SoftTargets {
    /// Row-major `n_rows × n_cells` dense soft `bytes_log` predictions
    /// (no NaN — the teacher fills the whole surface).
    pub soft: Vec<f64>,
    pub n_rows: usize,
    pub n_cells: usize,
    /// Number of cells that got a real per-cell teacher (vs the nanmean
    /// fallback) — recorded for the manifest.
    pub n_cells_with_teacher: usize,
    /// sha256 of the export the teacher was fit on (must match the export
    /// we just wrote — the integrity gate).
    pub source_export_sha256: String,
    pub sha256: String,
}

/// Read the teacher's `soft_targets.parquet`. Expects:
/// - `row_idx: i64`
/// - `soft_{c}: f32` for `c` in `0..n_cells`
/// - parquet key-value metadata `source_export_sha256`, optional
///   `n_cells_with_teacher`.
///
/// Validates the row count, cell count, and that `row_idx` is the dense
/// `0..n_rows` permutation, reordering into row order if needed.
pub fn load_soft_targets(
    path: &Path,
    expect_rows: usize,
    expect_cells: usize,
) -> Result<SoftTargets, TrainError> {
    let file = File::open(path).map_err(|e| TrainError::Io(format!("{}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TrainError::Parquet(e.to_string()))?;
    let schema = builder.schema().clone();

    // Pull provenance metadata. PyArrow's `replace_schema_metadata` and
    // arrow-rs's `Schema::with_metadata` both surface custom keys in the
    // ARROW schema metadata (`builder.schema().metadata()`); the raw
    // parquet file-level KV is the fallback for writers that set it
    // directly.
    let mut source_export_sha256 = String::new();
    let mut n_cells_with_teacher = expect_cells;
    for (k, v) in schema.metadata() {
        match k.as_str() {
            "source_export_sha256" => source_export_sha256 = v.clone(),
            "n_cells_with_teacher" => {
                n_cells_with_teacher = v.parse().unwrap_or(expect_cells);
            }
            _ => {}
        }
    }
    if let Some(entries) = builder.metadata().file_metadata().key_value_metadata() {
        for e in entries {
            match e.key.as_str() {
                "source_export_sha256" if source_export_sha256.is_empty() => {
                    source_export_sha256 = e.value.clone().unwrap_or_default();
                }
                "n_cells_with_teacher" => {
                    if let Some(v) = &e.value {
                        n_cells_with_teacher = v.parse().unwrap_or(n_cells_with_teacher);
                    }
                }
                _ => {}
            }
        }
    }

    let col_idx =
        |name: &str| -> Option<usize> { schema.fields().iter().position(|f| f.name() == name) };
    let ridx = col_idx("row_idx").ok_or_else(|| TrainError::MissingTargetColumn {
        column: "row_idx".into(),
    })?;
    let mut soft_idx: Vec<usize> = Vec::with_capacity(expect_cells);
    for c in 0..expect_cells {
        let name = format!("soft_{c}");
        let idx = col_idx(&name).ok_or(TrainError::MissingTargetColumn { column: name })?;
        soft_idx.push(idx);
    }

    let reader = builder
        .build()
        .map_err(|e| TrainError::Parquet(e.to_string()))?;

    let mut row_idx_all: Vec<i64> = Vec::with_capacity(expect_rows);
    let mut soft_cols: Vec<Vec<f64>> = vec![Vec::with_capacity(expect_rows); expect_cells];
    for batch in reader {
        let batch = batch.map_err(|e| TrainError::Parquet(e.to_string()))?;
        let ri = batch
            .column(ridx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| TrainError::Parquet("row_idx not i64".into()))?;
        for i in 0..ri.len() {
            row_idx_all.push(ri.value(i));
        }
        for (c, &ci) in soft_idx.iter().enumerate() {
            let arr = batch.column(ci);
            if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                for i in 0..a.len() {
                    soft_cols[c].push(if a.is_null(i) {
                        f64::NAN
                    } else {
                        a.value(i) as f64
                    });
                }
            } else {
                return Err(TrainError::Parquet(format!("soft_{c} not f32")));
            }
        }
    }

    let got = row_idx_all.len();
    if got != expect_rows {
        return Err(TrainError::Degenerate(format!(
            "soft-target row count {got} != expected {expect_rows}"
        )));
    }

    // Reorder into dense row order by row_idx.
    let mut soft = vec![f64::NAN; expect_rows * expect_cells];
    let mut seen = vec![false; expect_rows];
    for (k, &ri) in row_idx_all.iter().enumerate() {
        if ri < 0 || (ri as usize) >= expect_rows {
            return Err(TrainError::Degenerate(format!(
                "soft-target row_idx {ri} out of range 0..{expect_rows}"
            )));
        }
        let r = ri as usize;
        if seen[r] {
            return Err(TrainError::Degenerate(format!(
                "soft-target row_idx {r} appears twice"
            )));
        }
        seen[r] = true;
        for c in 0..expect_cells {
            let v = soft_cols[c][k];
            if !v.is_finite() {
                return Err(TrainError::Degenerate(format!(
                    "soft-target (row {r}, cell {c}) is non-finite — teacher must emit dense targets"
                )));
            }
            soft[r * expect_cells + c] = v;
        }
    }
    if !seen.iter().all(|&s| s) {
        return Err(TrainError::Degenerate(
            "soft-target file is missing some row indices".into(),
        ));
    }

    let sha256 = file_sha256(path)?;
    Ok(SoftTargets {
        soft,
        n_rows: expect_rows,
        n_cells: expect_cells,
        n_cells_with_teacher,
        source_export_sha256,
        sha256,
    })
}

/// Stable sha256 of the [`TeacherParams`] so the manifest records exactly
/// which teacher config produced the soft targets.
pub fn teacher_params_fingerprint(p: &TeacherParams) -> String {
    let mut h = Sha256::new();
    h.update(
        format!(
            "histgb|max_iter={}|max_depth={}|lr={}|l2={}|min_cell_rows={}|seed={}",
            p.max_iter,
            p.max_depth,
            p.learning_rate,
            p.l2_regularization,
            p.min_cell_rows,
            p.random_state
        )
        .as_bytes(),
    );
    let d = h.finalize();
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
