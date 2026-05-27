//! Load `(features, target)` training rows from a unified sweep
//! parquet.
//!
//! Expected columns (the canonical unified-sweep schema described in
//! the spec): `image_path: utf8`, `codec: utf8`, `q: int64`,
//! `knob_tuple_json: utf8`, one or more `score_*` metric columns, and
//! `feat_0..feat_N: float`. This loader pulls the `feat_*` block
//! (optionally appending `q` as a feature), the requested target
//! column, and `image_path` (for the grouped held-out split).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float32Array, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::TrainError;

/// Which codec rows to keep.
#[derive(Debug, Clone)]
pub struct CodecFilter {
    /// Lower-case codec-family / codec-name substring to match against
    /// the `codec` column. `None` keeps every row (used when the
    /// parquet is already a single-codec cut).
    pub codec: Option<String>,
}

impl CodecFilter {
    pub fn new(codec: Option<String>) -> Self {
        Self { codec }
    }

    fn label(&self) -> String {
        self.codec.clone().unwrap_or_else(|| "<all>".to_string())
    }
}

/// Materialized training rows: a dense `n_rows × n_features` feature
/// matrix (row-major), the parallel target vector, and the per-row
/// image identifier used to group the held-out split.
#[derive(Debug)]
pub struct TrainingData {
    /// Row-major `n_rows * n_features` feature values.
    pub features: Vec<f32>,
    /// One target value per row.
    pub targets: Vec<f64>,
    /// One image identifier per row (for grouped train/val split).
    pub image_ids: Vec<String>,
    /// Names of the feature columns, in matrix-column order.
    pub feature_names: Vec<String>,
    /// Number of feature columns.
    pub n_features: usize,
}

impl TrainingData {
    pub fn n_rows(&self) -> usize {
        self.targets.len()
    }
}

/// Read a numeric column (Float32 / Float64 / Int64) into f64,
/// pushing onto `out`. Nulls become `f64::NAN`.
fn push_numeric(batch: &RecordBatch, col: usize, out: &mut Vec<f64>) -> Option<()> {
    let arr = batch.column(col);
    if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        for i in 0..a.len() {
            out.push(if a.is_null(i) {
                f64::NAN
            } else {
                a.value(i) as f64
            });
        }
        Some(())
    } else if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        for i in 0..a.len() {
            out.push(if a.is_null(i) { f64::NAN } else { a.value(i) });
        }
        Some(())
    } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        for i in 0..a.len() {
            out.push(if a.is_null(i) {
                f64::NAN
            } else {
                a.value(i) as f64
            });
        }
        Some(())
    } else {
        None
    }
}

/// Read a utf8 column into owned strings, pushing onto `out`. Nulls
/// become the empty string.
fn push_strings(batch: &RecordBatch, col: usize, out: &mut Vec<String>) -> Option<()> {
    let a = batch.column(col).as_any().downcast_ref::<StringArray>()?;
    for i in 0..a.len() {
        out.push(if a.is_null(i) {
            String::new()
        } else {
            a.value(i).to_string()
        });
    }
    Some(())
}

/// Load `(features, target)` rows from `path`, filtering to the
/// requested codec and reading `target_column` as the supervised
/// target. When `include_q` is true, the integer `q` column is
/// appended as a trailing feature (named `q`) so the picker can
/// condition on the encode quality knob.
pub fn load_training_rows(
    path: &Path,
    filter: &CodecFilter,
    target_column: &str,
    include_q: bool,
) -> Result<TrainingData, TrainError> {
    let file = File::open(path).map_err(|e| TrainError::Io(format!("{}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TrainError::Parquet(e.to_string()))?;
    let schema = builder.schema().clone();

    // Discover the feature columns in declared order (feat_*), the
    // codec / image / q / target columns by name.
    let mut feature_names: Vec<String> = Vec::new();
    for f in schema.fields() {
        if f.name().starts_with("feat_") {
            feature_names.push(f.name().clone());
        }
    }
    if feature_names.is_empty() {
        return Err(TrainError::NoFeatureColumns);
    }
    // Stable numeric ordering of feat_N so feat_2 < feat_10.
    feature_names.sort_by_key(|n| {
        n.strip_prefix("feat_")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(u64::MAX)
    });

    let col_idx =
        |name: &str| -> Option<usize> { schema.fields().iter().position(|f| f.name() == name) };

    let target_idx = col_idx(target_column).ok_or_else(|| TrainError::MissingTargetColumn {
        column: target_column.to_string(),
    })?;
    let codec_idx = col_idx("codec");
    let image_idx = col_idx("image_path");
    let q_idx = col_idx("q");

    let mut feat_idx: Vec<usize> = Vec::with_capacity(feature_names.len());
    for n in &feature_names {
        feat_idx.push(col_idx(n).expect("feature col discovered from same schema"));
    }

    let reader = builder
        .build()
        .map_err(|e| TrainError::Parquet(e.to_string()))?;

    // Per-column accumulators. We read whole columns then assemble
    // the row-major matrix once (clearer than interleaving).
    let mut codecs: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    let mut targets_all: Vec<f64> = Vec::new();
    let mut q_all: Vec<f64> = Vec::new();
    // feature column -> all values
    let mut feat_cols: Vec<Vec<f64>> = vec![Vec::new(); feat_idx.len()];

    for batch in reader {
        let batch = batch.map_err(|e| TrainError::Parquet(e.to_string()))?;
        if let Some(ci) = codec_idx {
            push_strings(&batch, ci, &mut codecs)
                .ok_or_else(|| TrainError::Parquet("codec column is not utf8".into()))?;
        }
        if let Some(ii) = image_idx {
            push_strings(&batch, ii, &mut images)
                .ok_or_else(|| TrainError::Parquet("image_path column is not utf8".into()))?;
        }
        push_numeric(&batch, target_idx, &mut targets_all).ok_or_else(|| {
            TrainError::MissingTargetColumn {
                column: target_column.to_string(),
            }
        })?;
        if include_q && let Some(qi) = q_idx {
            push_numeric(&batch, qi, &mut q_all)
                .ok_or_else(|| TrainError::Parquet("q column is not numeric".into()))?;
        }
        for (slot, &ci) in feat_idx.iter().enumerate() {
            push_numeric(&batch, ci, &mut feat_cols[slot])
                .ok_or_else(|| TrainError::Parquet(format!("feature col {ci} not numeric")))?;
        }
    }

    let total = targets_all.len();
    let have_q = include_q && q_idx.is_some() && q_all.len() == total;
    let n_features = feature_names.len() + usize::from(have_q);

    // Assemble row-major, applying the codec filter and dropping rows
    // with a non-finite target or any non-finite feature.
    let mut features: Vec<f32> = Vec::with_capacity(total * n_features);
    let mut targets: Vec<f64> = Vec::with_capacity(total);
    let mut image_ids: Vec<String> = Vec::with_capacity(total);

    let want_codec = filter.codec.as_deref();
    for r in 0..total {
        if let (Some(want), true) = (want_codec, codec_idx.is_some()) {
            let c = codecs.get(r).map(String::as_str).unwrap_or("");
            if !c.to_ascii_lowercase().contains(&want.to_ascii_lowercase()) {
                continue;
            }
        }
        let t = targets_all[r];
        if !t.is_finite() {
            continue;
        }
        // Gather this row's features; skip if any is non-finite.
        let mut row: Vec<f32> = Vec::with_capacity(n_features);
        let mut ok = true;
        for col in &feat_cols {
            let v = col[r];
            if !v.is_finite() {
                ok = false;
                break;
            }
            row.push(v as f32);
        }
        if ok && have_q {
            let qv = q_all[r];
            if qv.is_finite() {
                row.push(qv as f32);
            } else {
                ok = false;
            }
        }
        if !ok {
            continue;
        }
        features.extend_from_slice(&row);
        targets.push(t);
        image_ids.push(
            image_idx
                .and_then(|_| images.get(r).cloned())
                .unwrap_or_else(|| format!("row{r}")),
        );
    }

    if targets.is_empty() {
        return Err(TrainError::EmptyAfterFilter {
            codec: filter.label(),
        });
    }

    let mut final_feature_names = feature_names;
    if have_q {
        final_feature_names.push("q".to_string());
    }

    Ok(TrainingData {
        features,
        targets,
        image_ids,
        feature_names: final_feature_names,
        n_features,
    })
}

/// Deterministic grouped train/val split: every distinct image goes
/// wholly to train or val (no leakage of an image's rows across the
/// split). `val_frac` ∈ (0, 1) is the target fraction of *images*
/// (not rows) held out. Returns `(train_row_indices, val_row_indices)`.
pub fn grouped_split(data: &TrainingData, val_frac: f64) -> (Vec<usize>, Vec<usize>) {
    // Collect distinct images in first-seen order for determinism.
    let mut order: Vec<&str> = Vec::new();
    let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
    for id in &data.image_ids {
        if seen.insert(id.as_str(), ()).is_none() {
            order.push(id.as_str());
        }
    }
    // Sort image ids so the split is reproducible regardless of row
    // order in the parquet.
    order.sort_unstable();
    let n_val_groups = ((order.len() as f64) * val_frac).round() as usize;
    let n_val_groups = n_val_groups.clamp(if order.len() > 1 { 1 } else { 0 }, order.len());
    let val_set: std::collections::HashSet<&str> =
        order.iter().take(n_val_groups).copied().collect();

    let mut train = Vec::new();
    let mut val = Vec::new();
    for (i, id) in data.image_ids.iter().enumerate() {
        if val_set.contains(id.as_str()) {
            val.push(i);
        } else {
            train.push(i);
        }
    }
    (train, val)
}
