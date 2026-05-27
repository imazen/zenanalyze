//! Distillation path tests (Rust side — no Python required).
//!
//! Covers the parts of zentrain's teacher → student recipe that live in
//! Rust:
//!   1. `export_teacher_dataset` writes the exact dataset (raw teacher
//!      inputs + hard `bytes_log` + `reach` + the train/val split) keyed
//!      by `row_idx`, and the schema round-trips.
//!   2. A hand-written soft-target parquet loads back via
//!      `load_soft_targets` in dataset row order, validating the
//!      provenance + dense-target invariants.
//!   3. `run_search_distill` trains the student on DENSE soft targets and
//!      evaluates against the HARD oracle, producing a finite picker
//!      panel and a ZNPR-v3-bakeable model.
//!
//! The teacher itself (per-cell HistGB) is an offline Python step; here
//! we synthesize a smooth soft-target surface directly so the Rust
//! distillation path is exercised deterministically in CI without a
//! Python dependency.

use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;

use zenpicker_train::{
    DistillManifest, HeldoutManifest, MlpConfig, MlpPickerManifestInputs, SearchManifest,
    TeacherParams, bake_mlp_picker, build_picker_dataset, default_grid, export_teacher_dataset,
    fit_standardizer, grouped_split_picker, load_soft_targets, run_search_distill, standardize_all,
    teacher_params_fingerprint,
};

/// Minimal synthetic sweep: n images, 2 features, 2 cells, q ladder.
fn write_synthetic_sweep(path: &std::path::Path, n_images: usize) {
    let qs = [10i64, 30, 60, 90];
    let cells = [
        ("{\"subsampling\":\"420\"}", 0.0f64),
        ("{\"subsampling\":\"444\"}", 1.0f64),
    ];
    let (mut img, mut codec, mut q, mut knob, mut eb, mut score, mut f0, mut f1) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for i in 0..n_images {
        let a = (i as f32) * 0.06 - 0.4;
        let b = ((i * 5) % 13) as f32 * 0.03;
        for &qq in &qs {
            for (kj, chroma) in cells {
                img.push(format!("img_{i:03}.png"));
                codec.push("zenjpeg".to_string());
                q.push(qq);
                knob.push(kj.to_string());
                f0.push(a);
                f1.push(b);
                score.push(20.0 + 0.7 * qq as f64 + 3.0 * chroma + 8.0 * a as f64);
                eb.push(
                    (1000.0 + 40.0 * qq as f64) * (1.0 + 0.25 * chroma) * (1.0 + 0.3 * b as f64),
                );
            }
        }
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("image_basename", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::Int64, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new("encoded_bytes", DataType::Float64, false),
        Field::new("score_zensim", DataType::Float64, false),
        Field::new("feat_0", DataType::Float32, false),
        Field::new("feat_1", DataType::Float32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(img)),
            Arc::new(StringArray::from(codec)),
            Arc::new(Int64Array::from(q)),
            Arc::new(StringArray::from(knob)),
            Arc::new(Float64Array::from(eb)),
            Arc::new(Float64Array::from(score)),
            Arc::new(Float32Array::from(f0)),
            Arc::new(Float32Array::from(f1)),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn teacher_export_roundtrips_and_soft_targets_distill() {
    let dir = std::env::temp_dir().join(format!("zenpicker_distill_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let parquet_path = dir.join("sweep.parquet");
    write_synthetic_sweep(&parquet_path, 48);

    let zq_targets: Vec<i64> = vec![20, 30, 40, 50, 60];
    let ds =
        build_picker_dataset(&parquet_path, Some("zenjpeg"), &zq_targets).expect("build dataset");
    assert_eq!(ds.n_cells, 2);
    assert_eq!(ds.n_in, 3); // feat_0, feat_1, zq_norm

    let (train_rows, val_rows) = grouped_split_picker(&ds, 0.25);
    assert!(!train_rows.is_empty() && !val_rows.is_empty());

    // --- 1. Export round-trips with the expected schema.
    let export_path = dir.join("teacher_export.parquet");
    let export_sha =
        export_teacher_dataset(&ds, &train_rows, &val_rows, &export_path).expect("export");
    assert_eq!(export_sha.len(), 64, "sha256 hex");

    let file = std::fs::File::open(&export_path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let names: Vec<String> = builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    for must in ["row_idx", "image_id", "target_zq", "split"] {
        assert!(names.iter().any(|n| n == must), "export missing {must}");
    }
    for c in 0..ds.n_cells {
        assert!(names.iter().any(|n| *n == format!("reach_{c}")));
        assert!(names.iter().any(|n| *n == format!("bytes_log_{c}")));
    }
    for j in 0..ds.n_in {
        assert!(names.iter().any(|n| *n == format!("f_{j}")));
    }

    // --- 2. Synthesize a DENSE soft-target surface (the teacher's job)
    // and write it as the soft-target parquet, then load it back.
    let n_rows = ds.n_rows();
    let n_cells = ds.n_cells;
    let mut soft = vec![0.0f32; n_rows * n_cells];
    for r in 0..n_rows {
        for c in 0..n_cells {
            // Smooth fill: real bytes_log where reachable, a plausible
            // interpolant (row mean of reachable cells) elsewhere — no NaN.
            let v = ds.bytes_log[r * n_cells + c];
            soft[r * n_cells + c] = if v.is_finite() {
                v as f32
            } else {
                let mut sum = 0.0;
                let mut cnt = 0;
                for k in 0..n_cells {
                    let w = ds.bytes_log[r * n_cells + k];
                    if w.is_finite() {
                        sum += w;
                        cnt += 1;
                    }
                }
                (if cnt > 0 { sum / cnt as f64 } else { 0.0 }) as f32
            };
        }
    }
    let soft_path = dir.join("soft_targets.parquet");
    write_soft_targets(&soft_path, &soft, n_rows, n_cells, &export_sha, n_cells);

    let loaded = load_soft_targets(&soft_path, n_rows, n_cells).expect("load soft targets");
    assert_eq!(loaded.n_rows, n_rows);
    assert_eq!(loaded.n_cells, n_cells);
    assert_eq!(loaded.source_export_sha256, export_sha);
    // Dense: every value finite, matches what we wrote (row order).
    for r in 0..n_rows {
        for c in 0..n_cells {
            assert!(loaded.soft[r * n_cells + c].is_finite());
            assert!(
                (loaded.soft[r * n_cells + c] - soft[r * n_cells + c] as f64).abs() < 1e-4,
                "soft target round-trip mismatch at ({r},{c})"
            );
        }
    }

    // --- 3. Distill: student trains on dense soft targets, eval vs HARD oracle.
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train_rows);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);
    let base = MlpConfig {
        hidden: vec![16, 16],
        max_iter: 120,
        ..MlpConfig::default()
    };
    let grid = default_grid();
    let res = run_search_distill(
        &ds,
        &x_std,
        &loaded.soft,
        &train_rows,
        &val_rows,
        &grid,
        &base,
        1.0,
        |_m| {},
    )
    .expect("distill search produced a model");
    assert!(res.best_eval.bytes_panel.srocc.is_finite());
    assert!(res.best_eval.argmin_acc >= 0.0 && res.best_eval.argmin_acc <= 1.0);
    assert_eq!(res.best_model.n_out, ds.n_cells);

    // --- 4. Bake the distilled student → ZNPR v3 with distill provenance.
    let eval = &res.best_eval;
    let heldout = HeldoutManifest {
        bytes_srocc: eval.bytes_panel.srocc,
        bytes_plcc: eval.bytes_panel.plcc,
        bytes_krocc: eval.bytes_panel.krocc,
        bytes_pwrc: eval.bytes_panel.pwrc,
        bytes_z_rmse: eval.bytes_panel.z_rmse,
        bytes_or_ratio: eval.bytes_panel.or_ratio,
        argmin_acc: eval.argmin_acc,
        overhead_mean: eval.overhead_mean,
        overhead_p50: eval.overhead_p50,
        overhead_p90: eval.overhead_p90,
        n_rows: eval.n_rows,
        n_pairs: eval.n_pairs,
    };
    let tp = TeacherParams::default();
    let distillation = Some(DistillManifest {
        recipe: "histgb_per_cell_soft_target_mse".to_string(),
        recipe_note: "test".to_string(),
        teacher_kind: "HistGradientBoostingRegressor (per cell)".to_string(),
        teacher_max_iter: tp.max_iter,
        teacher_max_depth: tp.max_depth,
        teacher_learning_rate: tp.learning_rate,
        teacher_l2_regularization: tp.l2_regularization,
        teacher_min_cell_rows: tp.min_cell_rows,
        teacher_random_state: tp.random_state,
        teacher_params_fingerprint: teacher_params_fingerprint(&tp),
        n_cells_with_teacher: loaded.n_cells_with_teacher,
        student_loss: "soft_target_mse".to_string(),
        soft_weight: 1.0,
        teacher_dataset_export_path: export_path.display().to_string(),
        teacher_dataset_export_sha256: export_sha.clone(),
        soft_targets_path: soft_path.display().to_string(),
        soft_targets_sha256: loaded.sha256.clone(),
        teacher_script: "scripts/teacher_soft_targets.py".to_string(),
        teacher_heldout_argmin_acc: Some(0.5),
        teacher_heldout_overhead_mean: Some(0.05),
    });
    let bake_path = dir.join("distilled.bin");
    let cfg = res.best_cfg.clone();
    let search = SearchManifest {
        kind: "bounded_grid".to_string(),
        candidates: vec![],
        selected_index: res.selected_index,
        selection_metric: "heldout_argmin_accuracy".to_string(),
    };
    let outcome = bake_mlp_picker(
        &res.best_model,
        &mean,
        &scale,
        &bake_path,
        MlpPickerManifestInputs {
            codec_family: "zenjpeg",
            input_parquet: parquet_path.to_str().unwrap(),
            input_sha256: "test",
            input_rows_total: 0,
            picker_rows_total: ds.n_rows(),
            train_rows: train_rows.len(),
            val_rows: val_rows.len(),
            n_image_features: ds.feature_names.len(),
            feature_names: &ds.feature_names,
            cell_labels: &ds.cell_labels,
            zq_targets: &ds.zq_targets,
            cfg: &cfg,
            search,
            heldout,
            distillation,
        },
    )
    .expect("bake distilled picker");

    let bytes = &outcome.bake_bytes;
    assert_eq!(&bytes[0..4], b"ZNPR", "ZNPR magic");
    assert_eq!(bytes[4], 0x03, "ZNPR v3");

    // Manifest records the distillation recipe + the no-q-leak contract.
    let manifest_txt = std::fs::read_to_string(format!("{}.toml", bake_path.display())).unwrap();
    assert!(manifest_txt.contains("q_is_input = false"));
    assert!(manifest_txt.contains("histgb_per_cell_soft_target_mse"));
    assert!(manifest_txt.contains("[distillation]") || manifest_txt.contains("distillation"));

    // Loads through the runtime consumers: zenpredict::Model AND
    // zenpicker::MetaPicker (the same Predictor wrapper a per-codec
    // picker uses). Confirm a finite forward pass over n_cells outputs.
    let zmodel = zenpredict::Model::from_bytes(bytes).expect("load distilled bake");
    assert_eq!(zmodel.n_inputs(), ds.n_in);
    assert_eq!(zmodel.n_outputs(), ds.n_cells);

    let mut mp = zenpicker::MetaPicker::new(&zmodel);
    let r = val_rows[0];
    let row: Vec<f32> = ds.features[r * ds.n_in..r * ds.n_in + ds.n_in]
        .iter()
        .map(|&v| v as f32)
        .collect();
    let out = mp
        .predictor()
        .predict(&row)
        .expect("distilled bake predicts via MetaPicker");
    assert_eq!(out.len(), ds.n_cells, "picker emits one bytes_log per cell");
    assert!(out.iter().all(|v| v.is_finite()), "predictions finite");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Write a soft-target parquet matching what the Python teacher emits:
/// `row_idx:i64`, `soft_{c}:f32`, + KV metadata.
fn write_soft_targets(
    path: &std::path::Path,
    soft: &[f32],
    n_rows: usize,
    n_cells: usize,
    source_export_sha256: &str,
    n_cells_with_teacher: usize,
) {
    let mut fields = vec![Field::new("row_idx", DataType::Int64, false)];
    for c in 0..n_cells {
        fields.push(Field::new(format!("soft_{c}"), DataType::Float32, false));
    }
    let mut md = std::collections::HashMap::new();
    md.insert(
        "source_export_sha256".to_string(),
        source_export_sha256.to_string(),
    );
    md.insert(
        "n_cells_with_teacher".to_string(),
        n_cells_with_teacher.to_string(),
    );
    let schema = Arc::new(Schema::new(fields).with_metadata(md));

    let row_idx: Vec<i64> = (0..n_rows as i64).collect();
    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![Arc::new(Int64Array::from(row_idx))];
    for c in 0..n_cells {
        let col: Vec<f32> = (0..n_rows).map(|r| soft[r * n_cells + c]).collect();
        columns.push(Arc::new(Float32Array::from(col)));
    }
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}
