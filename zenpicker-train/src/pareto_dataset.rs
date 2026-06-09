//! Build the per-codec picker dataset in zentrain's
//! within-cell-optimal formulation — the FAITHFUL port of
//! `build_dataset` in `zentrain/tools/train_hybrid.py`.
//!
//! ## What the picker predicts (and why `q` is NOT an input feature)
//!
//! The unified sweep parquet has one row per `(image, codec-config,
//! q)` with the achieved `score_zensim` and `encoded_bytes`. A picker
//! must NOT see the codec's chosen `q` as an input: `q` is the
//! DECISION the picker exists to make (it's monotone with achieved
//! score, so feeding it in trivially inflates any "predict the score"
//! task — that was the skeleton's q-leakage bug).
//!
//! Instead, following zentrain, we:
//!
//! 1. Factor each config into a **categorical cell** (the discrete knob
//!    combination, e.g. `subsampling × progressive × sharp_yuv`).
//!    zenjpeg's `knob_tuple_json` supplies these.
//! 2. For every `(image, target_zq)` pair (over the `ZQ_TARGETS` grid —
//!    the *requested* quality, an INPUT), compute the within-cell
//!    optimal: `bytes_log[cell] = ln(min encoded_bytes over rows in this
//!    cell whose score_zensim >= target_zq)`, and `reach[cell] = any
//!    such row exists`.
//! 3. Train an MLP `(image_features, zq_norm) -> bytes_log[0..N]`.
//! 4. At inference: `cell = argmin(pred_bytes_log, mask=reach)` — the
//!    smallest-bytes config that reaches the requested quality.
//!
//! So the inputs are IMAGE FEATURES + the user's TARGET quality
//! (`zq_norm = target_zq / 100`). The codec's per-encode `q` never
//! enters the feature vector. The supervised target is `bytes_log` per
//! cell, not the achieved score.
//!
//! ## Data-coverage caveat
//!
//! The available `unified_v13_zenjpeg_cvvdp.parquet` sweeps only 5 `q`
//! levels {10, 30, 60, 80, 90} per image. The within-cell "reaches
//! target_zq" test therefore sees a coarse score ladder — a row
//! "reaches" `target_zq` only if one of those 5 encodes happened to
//! land at-or-above it. That is SPARSE on quality (per zensim/CLAUDE.md
//! "Dense sampling for trained models" a production picker needs
//! ~30 q points). We flag it in the manifest; this port validates the
//! *formulation*, not a production bake.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use arrow::array::{Array, Float32Array, Float64Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::TrainError;

/// One raw sweep row pulled from the parquet.
struct RawRow {
    image: String,
    cell_key: String,
    score: f64,
    bytes: f64,
    feat: Vec<f32>,
    /// This config's value for each requested scalar axis, parallel to
    /// the caller's `scalar_axes` order. `NaN` where the knob is absent
    /// or equals the axis sentinel (so it never becomes a target).
    scalar_vals: Vec<f64>,
}

/// The materialized picker dataset: per `(image, target_zq)` rows with
/// a dense feature matrix, a per-row × per-cell `bytes_log` target
/// matrix (NaN = unreachable), and a parallel `reach` mask.
pub struct PickerDataset {
    /// Row-major `n_rows × n_in` standardized-later feature matrix
    /// (raw image features + `zq_norm` appended as the LAST column).
    pub features: Vec<f64>,
    pub n_in: usize,
    /// Row-major `n_rows × n_cells` `bytes_log` targets; NaN where the
    /// cell did not reach the target.
    pub bytes_log: Vec<f64>,
    /// Row-major `n_rows × n_cells` reach mask (`true` = reachable).
    pub reach: Vec<bool>,
    pub n_cells: usize,
    /// Cell labels, index = cell id.
    pub cell_labels: Vec<String>,
    /// Per-row image identifier (for the grouped split).
    pub image_ids: Vec<String>,
    /// Per-row `target_zq` (for diagnostics / per-band reporting).
    pub target_zq: Vec<i64>,
    /// Image feature column names (NOT including the appended `zq_norm`).
    pub feature_names: Vec<String>,
    /// The `target_zq` grid actually used.
    pub zq_targets: Vec<i64>,
    /// Scalar prediction axes (empty = bytes-only categorical picker).
    pub scalar_axes: Vec<String>,
    /// Per-axis sentinel value (`NaN` = none), parallel to `scalar_axes`.
    pub scalar_sentinels: Vec<f64>,
    /// Per-axis row-major `n_rows × n_cells` within-cell-optimal scalar
    /// target. `scalars[a]` has length `n_rows * n_cells`; entries are
    /// `NaN` where the cell was unreachable OR the optimal config's value
    /// was the axis sentinel (those are masked out of the head's loss).
    pub scalars: Vec<Vec<f64>>,
}

impl PickerDataset {
    pub fn n_rows(&self) -> usize {
        self.target_zq.len()
    }
}

/// One scalar prediction axis: the `knob_tuple_json` key to read plus an
/// optional `sentinel` (e.g. `lambda = 0.0` for trellis-off configs). A
/// row whose value equals the sentinel contributes a `NaN` target so the
/// head never trains on the placeholder, mirroring zentrain's
/// `SCALAR_SENTINELS` masking.
#[derive(Clone, Debug)]
pub struct ScalarAxisSpec {
    pub name: String,
    pub sentinel: Option<f64>,
}

impl ScalarAxisSpec {
    pub fn new(name: impl Into<String>, sentinel: Option<f64>) -> Self {
        Self {
            name: name.into(),
            sentinel,
        }
    }
}

/// Decode a JSON scalar (number / bool / numeric-string) to `f64`.
fn json_as_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Pull this row's scalar-axis values from its `knob_tuple_json`. A
/// missing key, an unparseable value, or a value equal to the axis
/// sentinel all map to `NaN` (a masked target).
fn scalars_from_knob(knob_json: &str, axes: &[ScalarAxisSpec]) -> Vec<f64> {
    if axes.is_empty() {
        return Vec::new();
    }
    let obj = serde_json::from_str::<serde_json::Value>(knob_json)
        .ok()
        .and_then(|v| v.as_object().cloned());
    axes.iter()
        .map(|ax| {
            let raw = obj
                .as_ref()
                .and_then(|o| o.get(&ax.name))
                .and_then(json_as_f64);
            match raw {
                Some(v) => match ax.sentinel {
                    Some(s) if (v - s).abs() <= f64::EPSILON => f64::NAN,
                    _ => v,
                },
                None => f64::NAN,
            }
        })
        .collect()
}

/// zentrain's `ZQ_TARGETS` for zenjpeg: step 5 from 0..70 then step 2
/// from 70..100. The *requested* quality grid (an input axis).
pub fn default_zq_targets() -> Vec<i64> {
    let mut v: Vec<i64> = (0..70).step_by(5).collect();
    v.extend((70..=100).step_by(2));
    v
}

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

/// Derive the categorical cell key from a zenjpeg `knob_tuple_json`.
///
/// The discrete knob axes are the categorical cell; `effort` is treated
/// as categorical too (it's a small enum). Any key named in
/// `scalar_axes` is EXCLUDED from the cell key — those are continuous
/// Pareto axes (chroma_scale, lambda) the scalar heads regress WITHIN a
/// cell, so folding them into the key would collapse the within-cell
/// variation the hybrid-heads formulation depends on. This mirrors
/// zentrain's `categorical_key(parse_config_name(name))`, where the cell
/// is `CATEGORICAL_AXES` only and `SCALAR_AXES` are separate.
fn cell_key_from_knob(knob_json: &str, scalar_axes: &[ScalarAxisSpec]) -> String {
    // Parse the small flat JSON object without a serde struct so any
    // knob ordering / subset works. We extract the recognized keys in a
    // canonical order so the cell label is stable.
    let v: serde_json::Value = match serde_json::from_str(knob_json) {
        Ok(v) => v,
        Err(_) => return format!("raw:{knob_json}"),
    };
    let obj = match v.as_object() {
        Some(o) => o,
        None => return format!("raw:{knob_json}"),
    };
    let is_scalar = |k: &str| scalar_axes.iter().any(|a| a.name == k);
    let mut parts: Vec<String> = Vec::new();
    // Canonical order: subsampling, progressive, sharp_yuv, effort,
    // then any remaining non-scalar keys sorted.
    let canonical = ["subsampling", "progressive", "sharp_yuv", "effort"];
    for k in canonical {
        if is_scalar(k) {
            continue;
        }
        if let Some(val) = obj.get(k) {
            parts.push(format!("{k}={}", render_json_scalar(val)));
        }
    }
    let mut rest: Vec<&String> = obj
        .keys()
        .filter(|k| !canonical.contains(&k.as_str()) && !is_scalar(k))
        .collect();
    rest.sort();
    for k in rest {
        parts.push(format!("{k}={}", render_json_scalar(&obj[k])));
    }
    parts.join("|")
}

fn render_json_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Load + build the picker dataset from a unified sweep parquet.
///
/// `codec_filter`: lower-cased substring matched against the `codec`
/// column (None keeps all rows — used when the parquet is single-codec).
/// `zq_targets`: the requested-quality grid.
pub fn build_picker_dataset(
    path: &Path,
    codec_filter: Option<&str>,
    zq_targets: &[i64],
) -> Result<PickerDataset, TrainError> {
    build_picker_dataset_with(path, codec_filter, zq_targets, &[])
}

/// Like [`build_picker_dataset`] but ALSO captures per-cell scalar
/// prediction targets for each axis in `scalar_axes`: the
/// within-cell-optimal config's value for that knob (sentinel-masked to
/// `NaN`). Passing an empty `scalar_axes` reproduces the bytes-only
/// categorical picker exactly. The scalar values are read from each
/// row's `knob_tuple_json`.
pub fn build_picker_dataset_with(
    path: &Path,
    codec_filter: Option<&str>,
    zq_targets: &[i64],
    scalar_axes: &[ScalarAxisSpec],
) -> Result<PickerDataset, TrainError> {
    let file = File::open(path).map_err(|e| TrainError::Io(format!("{}: {e}", path.display())))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TrainError::Parquet(e.to_string()))?;
    let schema = builder.schema().clone();

    let mut feature_names: Vec<String> = Vec::new();
    for f in schema.fields() {
        if f.name().starts_with("feat_") {
            feature_names.push(f.name().clone());
        }
    }
    if feature_names.is_empty() {
        return Err(TrainError::NoFeatureColumns);
    }
    feature_names.sort_by_key(|n| {
        n.strip_prefix("feat_")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(u64::MAX)
    });

    let col_idx =
        |name: &str| -> Option<usize> { schema.fields().iter().position(|f| f.name() == name) };

    let image_idx = col_idx("image_basename").or_else(|| col_idx("image_path"));
    let codec_idx = col_idx("codec");
    let q_idx = col_idx("q");
    let knob_idx = col_idx("knob_tuple_json");
    let score_idx = col_idx("score_zensim").ok_or_else(|| TrainError::MissingTargetColumn {
        column: "score_zensim".into(),
    })?;
    let bytes_idx = col_idx("encoded_bytes").ok_or_else(|| TrainError::MissingTargetColumn {
        column: "encoded_bytes".into(),
    })?;
    let knob_idx = knob_idx.ok_or_else(|| TrainError::MissingTargetColumn {
        column: "knob_tuple_json".into(),
    })?;
    let q_idx = q_idx.ok_or_else(|| TrainError::MissingTargetColumn { column: "q".into() })?;
    let image_idx = image_idx.ok_or_else(|| TrainError::MissingTargetColumn {
        column: "image_basename|image_path".into(),
    })?;

    let mut feat_idx: Vec<usize> = Vec::with_capacity(feature_names.len());
    for n in &feature_names {
        feat_idx.push(col_idx(n).expect("feature col from same schema"));
    }

    let reader = builder
        .build()
        .map_err(|e| TrainError::Parquet(e.to_string()))?;

    let mut images: Vec<String> = Vec::new();
    let mut codecs: Vec<String> = Vec::new();
    let mut knobs: Vec<String> = Vec::new();
    let mut qs: Vec<f64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    let mut bytes_v: Vec<f64> = Vec::new();
    let mut feat_cols: Vec<Vec<f64>> = vec![Vec::new(); feat_idx.len()];

    for batch in reader {
        let batch = batch.map_err(|e| TrainError::Parquet(e.to_string()))?;
        push_strings(&batch, image_idx, &mut images)
            .ok_or_else(|| TrainError::Parquet("image column not utf8".into()))?;
        if let Some(ci) = codec_idx {
            push_strings(&batch, ci, &mut codecs)
                .ok_or_else(|| TrainError::Parquet("codec column not utf8".into()))?;
        }
        push_strings(&batch, knob_idx, &mut knobs)
            .ok_or_else(|| TrainError::Parquet("knob_tuple_json not utf8".into()))?;
        push_numeric(&batch, q_idx, &mut qs)
            .ok_or_else(|| TrainError::Parquet("q not numeric".into()))?;
        push_numeric(&batch, score_idx, &mut scores)
            .ok_or_else(|| TrainError::Parquet("score_zensim not numeric".into()))?;
        push_numeric(&batch, bytes_idx, &mut bytes_v)
            .ok_or_else(|| TrainError::Parquet("encoded_bytes not numeric".into()))?;
        for (slot, &ci) in feat_idx.iter().enumerate() {
            push_numeric(&batch, ci, &mut feat_cols[slot])
                .ok_or_else(|| TrainError::Parquet(format!("feature col {ci} not numeric")))?;
        }
    }

    let total = scores.len();

    // Assemble raw rows, applying the codec filter + dropping non-finite.
    let mut raw: Vec<RawRow> = Vec::with_capacity(total);
    // Collect cell labels in first-seen order, then sort for a stable
    // cell index independent of row order.
    let mut cell_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let want = codec_filter.map(|c| c.to_ascii_lowercase());
    for r in 0..total {
        if let (Some(w), Some(ci)) = (&want, codec_idx) {
            let _ = ci;
            let c = codecs.get(r).map(String::as_str).unwrap_or("");
            if !c.to_ascii_lowercase().contains(w.as_str()) {
                continue;
            }
        }
        let s = scores[r];
        let b = bytes_v[r];
        if !s.is_finite() || !b.is_finite() || b <= 0.0 {
            continue;
        }
        let mut feat: Vec<f32> = Vec::with_capacity(feat_idx.len());
        let mut ok = true;
        for col in &feat_cols {
            let v = col[r];
            if !v.is_finite() {
                ok = false;
                break;
            }
            feat.push(v as f32);
        }
        if !ok {
            continue;
        }
        let cell_key = cell_key_from_knob(&knobs[r], scalar_axes);
        cell_set.insert(cell_key.clone());
        let _ = qs[r]; // q is intentionally NOT used as a feature (no leakage)
        raw.push(RawRow {
            image: images[r].clone(),
            cell_key,
            score: s,
            bytes: b,
            feat,
            scalar_vals: scalars_from_knob(&knobs[r], scalar_axes),
        });
    }

    if raw.is_empty() {
        return Err(TrainError::EmptyAfterFilter {
            codec: codec_filter.unwrap_or("<all>").to_string(),
        });
    }

    let cell_labels: Vec<String> = cell_set.into_iter().collect();
    let n_cells = cell_labels.len();
    let cell_index: BTreeMap<&str, usize> = cell_labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();

    // Group rows by image. Each image carries its per-image feature
    // vector (features are per-image-constant across q/config in this
    // schema; we take the first row's features per image).
    let mut by_image: BTreeMap<&str, Vec<&RawRow>> = BTreeMap::new();
    for rr in &raw {
        by_image.entry(rr.image.as_str()).or_default().push(rr);
    }

    let n_in = feature_names.len() + 1; // + zq_norm
    let mut features: Vec<f64> = Vec::new();
    let mut bytes_log: Vec<f64> = Vec::new();
    let mut reach: Vec<bool> = Vec::new();
    let mut image_ids: Vec<String> = Vec::new();
    let mut target_zq: Vec<i64> = Vec::new();
    let n_axes = scalar_axes.len();
    // Per-axis accumulator, row-major `n_rows × n_cells`, parallel to
    // `bytes_log`/`reach`.
    let mut scalars: Vec<Vec<f64>> = vec![Vec::new(); n_axes];

    for (image, rows) in &by_image {
        // Per-image feature vector (constant across configs).
        let f = &rows[0].feat;
        for &zq in zq_targets {
            // Within-cell optimal: min bytes over rows in the cell with
            // score >= zq. `cell_scalar[a][c]` tracks the scalar value of
            // that same min-bytes winner, so the scalar head learns the
            // Pareto-optimal knob per cell (NaN if the winner's value was
            // the axis sentinel).
            let mut cell_min_bytes = vec![f64::INFINITY; n_cells];
            let mut cell_reach = vec![false; n_cells];
            let mut cell_scalar = vec![vec![f64::NAN; n_cells]; n_axes];
            for rr in rows {
                if rr.score >= zq as f64 {
                    let c = cell_index[rr.cell_key.as_str()];
                    if rr.bytes < cell_min_bytes[c] {
                        cell_min_bytes[c] = rr.bytes;
                        cell_reach[c] = true;
                        for (cell_a, &val) in cell_scalar.iter_mut().zip(&rr.scalar_vals) {
                            cell_a[c] = val;
                        }
                    }
                }
            }
            // Skip rows where NO cell reached the target — physically
            // unreachable for this image at the sampled q ladder (the
            // ceiling-aware skip in zentrain's build_dataset).
            if !cell_reach.iter().any(|&r| r) {
                continue;
            }
            // Emit the row.
            for &x in f {
                features.push(x as f64);
            }
            features.push(zq as f64 / 100.0); // zq_norm — REQUESTED quality
            for c in 0..n_cells {
                if cell_reach[c] {
                    bytes_log.push(cell_min_bytes[c].ln());
                } else {
                    bytes_log.push(f64::NAN);
                }
                reach.push(cell_reach[c]);
                for a in 0..n_axes {
                    scalars[a].push(if cell_reach[c] {
                        cell_scalar[a][c]
                    } else {
                        f64::NAN
                    });
                }
            }
            image_ids.push((*image).to_string());
            target_zq.push(zq);
        }
    }

    if target_zq.is_empty() {
        return Err(TrainError::Degenerate(
            "no reachable (image, target_zq) rows built".into(),
        ));
    }

    Ok(PickerDataset {
        features,
        n_in,
        bytes_log,
        reach,
        n_cells,
        cell_labels,
        image_ids,
        target_zq,
        feature_names,
        zq_targets: zq_targets.to_vec(),
        scalar_axes: scalar_axes.iter().map(|a| a.name.clone()).collect(),
        scalar_sentinels: scalar_axes
            .iter()
            .map(|a| a.sentinel.unwrap_or(f64::NAN))
            .collect(),
        scalars,
    })
}

/// Deterministic grouped-by-image split. Returns `(train_rows,
/// val_rows)`. `val_frac` ∈ (0,1) is the fraction of distinct IMAGES
/// held out; no image appears in both sides.
pub fn grouped_split_picker(ds: &PickerDataset, val_frac: f64) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<&str> = Vec::new();
    let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
    for id in &ds.image_ids {
        if seen.insert(id.as_str(), ()).is_none() {
            order.push(id.as_str());
        }
    }
    order.sort_unstable();
    let n_val_groups = ((order.len() as f64) * val_frac).round() as usize;
    let n_val_groups = n_val_groups.clamp(if order.len() > 1 { 1 } else { 0 }, order.len());
    let val_set: std::collections::HashSet<&str> =
        order.iter().take(n_val_groups).copied().collect();
    let mut train = Vec::new();
    let mut val = Vec::new();
    for (i, id) in ds.image_ids.iter().enumerate() {
        if val_set.contains(id.as_str()) {
            val.push(i);
        } else {
            train.push(i);
        }
    }
    (train, val)
}

/// Fit a per-feature standardizer (mean/std) over `rows`. Constant
/// columns get scale 1.0 (pass-through). Returns `(mean, scale)`.
pub fn fit_standardizer(features: &[f64], n_in: usize, rows: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len().max(1) as f64;
    let mut mean = vec![0.0f64; n_in];
    for &r in rows {
        let base = r * n_in;
        for j in 0..n_in {
            mean[j] += features[base + j];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0f64; n_in];
    for &r in rows {
        let base = r * n_in;
        for j in 0..n_in {
            let d = features[base + j] - mean[j];
            var[j] += d * d;
        }
    }
    let mut scale = vec![1.0f64; n_in];
    for j in 0..n_in {
        let std = (var[j] / n).sqrt();
        scale[j] = if std > 1e-9 { std } else { 1.0 };
    }
    (mean, scale)
}

/// Apply a standardizer to the whole matrix, returning a new row-major
/// matrix of the same shape.
pub fn standardize_all(features: &[f64], n_in: usize, mean: &[f64], scale: &[f64]) -> Vec<f64> {
    let n_rows = features.len() / n_in;
    let mut out = vec![0.0f64; features.len()];
    for r in 0..n_rows {
        let base = r * n_in;
        for j in 0..n_in {
            out[base + j] = (features[base + j] - mean[j]) / scale[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_from_knob_parses_decodes_and_masks_sentinel() {
        let axes = vec![
            ScalarAxisSpec::new("chroma_scale", None),
            ScalarAxisSpec::new("lambda", Some(0.0)),
        ];
        // Both present, lambda non-sentinel → both pass through.
        let v = scalars_from_knob(r#"{"chroma_scale":0.8,"lambda":14.5}"#, &axes);
        assert_eq!(v.len(), 2);
        assert!((v[0] - 0.8).abs() < 1e-12);
        assert!((v[1] - 14.5).abs() < 1e-12);
        // lambda == sentinel (trellis-off) → masked to NaN.
        let v = scalars_from_knob(r#"{"chroma_scale":1.0,"lambda":0.0}"#, &axes);
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert!(v[1].is_nan());
        // Missing key → NaN.
        let v = scalars_from_knob(r#"{"lambda":8.0}"#, &axes);
        assert!(v[0].is_nan());
        assert!((v[1] - 8.0).abs() < 1e-12);
        // Numeric-string + bool values still decode.
        let axes2 = vec![
            ScalarAxisSpec::new("x", None),
            ScalarAxisSpec::new("flag", None),
        ];
        let v = scalars_from_knob(r#"{"x":"2.5","flag":true}"#, &axes2);
        assert!((v[0] - 2.5).abs() < 1e-12);
        assert!((v[1] - 1.0).abs() < 1e-12);
        // No axes requested → empty (bytes-only path).
        assert!(scalars_from_knob("{}", &[]).is_empty());
    }
}
