//! End-to-end: mint a tiny synthetic unified-sweep parquet → train →
//! bake → load the bake via `zenpredict::Model::from_bytes` → assert
//! ZNPR v3 (header byte 0x03) + finite predictions.
//!
//! The synthetic target is a deterministic linear function of two
//! features plus a small `q`-dependent term, so the ridge baseline
//! should recover a strong held-out rank correlation — but the test
//! gate is correctness (loads, finite, v3), not a SROCC threshold,
//! since this is a skeleton baseline.

use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;

use zenpicker_train::{
    CodecFilter, PickerManifestInputs, bake_picker, evaluate, file_sha256, grouped_split,
    load_training_rows, train_ridge,
};

/// Write a synthetic parquet shaped like the unified sweep cut:
/// image_path, codec, q, knob_tuple_json, score_zensim, feat_0,
/// feat_1. 12 images × 4 q values = 48 rows, all codec=zenjpeg.
fn write_synthetic_parquet(path: &std::path::Path) {
    let n_images = 12;
    let qs = [10i64, 30, 60, 90];

    let mut image_path = Vec::new();
    let mut codec = Vec::new();
    let mut q = Vec::new();
    let mut knob = Vec::new();
    let mut score = Vec::new();
    let mut f0 = Vec::new();
    let mut f1 = Vec::new();

    for img in 0..n_images {
        // Per-image feature values (deterministic).
        let a = (img as f32) * 0.1 - 0.5;
        let b = ((img * 7) % 11) as f32 * 0.05;
        for &qq in &qs {
            image_path.push(format!("/synthetic/img_{img:02}.png"));
            codec.push("zenjpeg".to_string());
            q.push(qq);
            knob.push(format!("{{\"effort\":{}}}", img % 3));
            f0.push(a);
            f1.push(b);
            // Target: clear linear signal in (a, b, q).
            let target = 50.0 + 12.0 * a as f64 - 7.0 * b as f64 + 0.2 * qq as f64;
            score.push(target);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("image_path", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::Int64, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new("score_zensim", DataType::Float64, false),
        Field::new("feat_0", DataType::Float32, false),
        Field::new("feat_1", DataType::Float32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(image_path)),
            Arc::new(StringArray::from(codec)),
            Arc::new(Int64Array::from(q)),
            Arc::new(StringArray::from(knob)),
            Arc::new(Float64Array::from(score)),
            Arc::new(Float32Array::from(f0)),
            Arc::new(Float32Array::from(f1)),
        ],
    )
    .unwrap();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn train_bake_load_predict_finite_znpr_v3() {
    let dir = std::env::temp_dir().join(format!("zenpicker_train_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let parquet_path = dir.join("synthetic.parquet");
    let bake_path = dir.join("picker.bin");

    write_synthetic_parquet(&parquet_path);

    // Load + grouped split.
    let filter = CodecFilter::new(Some("zenjpeg".to_string()));
    let data = load_training_rows(&parquet_path, &filter, "score_zensim", true)
        .expect("load synthetic parquet");
    assert_eq!(data.n_rows(), 48, "12 images × 4 q values");
    // feat_0, feat_1 + appended q = 3 features.
    assert_eq!(data.n_features, 3);
    assert_eq!(data.feature_names, vec!["feat_0", "feat_1", "q"]);

    let (train_rows, val_rows) = grouped_split(&data, 0.25);
    assert!(!train_rows.is_empty());
    assert!(!val_rows.is_empty());
    // Grouped split: no image appears in both train and val.
    let train_imgs: std::collections::HashSet<&str> = train_rows
        .iter()
        .map(|&r| data.image_ids[r].as_str())
        .collect();
    for &r in &val_rows {
        assert!(
            !train_imgs.contains(data.image_ids[r].as_str()),
            "image leaked across the grouped split"
        );
    }

    let model = train_ridge(
        &data.features,
        &data.targets,
        data.n_features,
        &train_rows,
        0.1,
    )
    .expect("ridge fit");
    assert!(model.weights.iter().all(|w| w.is_finite()));

    // Held-out panel — strong linear signal, expect high SROCC, but
    // the gate is finiteness, not a threshold (skeleton baseline).
    let report = evaluate(&model, &data, &val_rows).expect("held-out rows");
    assert!(report.panel.srocc.is_finite());
    assert!(report.panel.n > 0);
    // Sanity: a clean linear target should rank well.
    assert!(
        report.panel.srocc > 0.5,
        "synthetic linear target should rank-correlate; got {}",
        report.panel.srocc
    );

    // Bake.
    let input_sha = file_sha256(&parquet_path).unwrap();
    let outcome = bake_picker(
        &model,
        &data.feature_names,
        "zenjpeg",
        "score_zensim",
        &bake_path,
        PickerManifestInputs {
            input_parquet: parquet_path.to_str().unwrap(),
            input_sha256: &input_sha,
            input_rows_total: data.n_rows(),
            train_rows: train_rows.len(),
            val_rows: val_rows.len(),
            heldout_srocc: report.panel.srocc,
            heldout_plcc: report.panel.plcc,
            heldout_krocc: report.panel.krocc,
            heldout_n: report.panel.n,
        },
    )
    .expect("bake");

    // ZNPR v3 header asserts (v2 is banned).
    let bytes = &outcome.bake_bytes;
    assert_eq!(&bytes[0..4], b"ZNPR", "bake must carry the ZNPR magic");
    assert_eq!(bytes[4], 0x03, "bake must be ZNPR v3");

    // The sibling manifest must exist.
    let manifest_path = std::path::PathBuf::from(format!("{}.toml", bake_path.display()));
    assert!(manifest_path.exists(), "manifest TOML must be written");
    let manifest_txt = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest_txt.contains("zenpicker-train"));
    assert!(manifest_txt.contains("ridge_linear_baseline"));
    assert!(manifest_txt.contains("follow_ons"));

    // Load via zenpredict::Model and predict — must be finite.
    let zmodel = zenpredict::Model::from_bytes(bytes).expect("load bake via zenpredict");
    assert_eq!(zmodel.n_inputs(), data.n_features);
    assert_eq!(zmodel.n_outputs(), 1);
    let mut predictor = zenpredict::Predictor::new(&zmodel);

    // Predict on every val row through the loaded runtime; compare to
    // the in-process model (they share the same scaler + weights, so
    // they should agree to f32 precision).
    let p = data.n_features;
    for &r in &val_rows {
        let base = r * p;
        let row = &data.features[base..base + p];
        let out = predictor.predict(row).expect("predict");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_finite(), "loaded-bake prediction must be finite");

        let in_proc = model.predict_raw(row) as f32;
        let delta = (out[0] - in_proc).abs();
        assert!(
            delta < 1e-2,
            "loaded bake ({}) should match in-proc model ({}) within f32 tolerance, delta={}",
            out[0],
            in_proc,
            delta
        );
    }

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

/// The per-codec quality bake is a 1-output regression head, not a
/// 6-family categorical meta-picker. It is nonetheless loadable by the
/// same consumer (`zenpicker::MetaPicker` wraps a
/// `zenpredict::Predictor`), and its predictor produces finite output.
/// This confirms the "loadable by zenpicker/MetaPicker" requirement
/// without claiming it IS a family meta-picker (it isn't — see README).
#[test]
fn bake_loads_through_zenpicker_meta_picker_predictor() {
    let dir = std::env::temp_dir().join(format!("zenpicker_train_mp_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let parquet_path = dir.join("synthetic.parquet");
    let bake_path = dir.join("picker.bin");
    write_synthetic_parquet(&parquet_path);

    let filter = CodecFilter::new(Some("zenjpeg".to_string()));
    let data =
        load_training_rows(&parquet_path, &filter, "score_zensim", true).expect("load synthetic");
    let (train_rows, val_rows) = grouped_split(&data, 0.25);
    let model = train_ridge(
        &data.features,
        &data.targets,
        data.n_features,
        &train_rows,
        0.1,
    )
    .expect("ridge fit");
    let report = evaluate(&model, &data, &val_rows).expect("held-out rows");
    let input_sha = file_sha256(&parquet_path).unwrap();
    let outcome = bake_picker(
        &model,
        &data.feature_names,
        "zenjpeg",
        "score_zensim",
        &bake_path,
        PickerManifestInputs {
            input_parquet: parquet_path.to_str().unwrap(),
            input_sha256: &input_sha,
            input_rows_total: data.n_rows(),
            train_rows: train_rows.len(),
            val_rows: val_rows.len(),
            heldout_srocc: report.panel.srocc,
            heldout_plcc: report.panel.plcc,
            heldout_krocc: report.panel.krocc,
            heldout_n: report.panel.n,
        },
    )
    .expect("bake");

    // Load through zenpicker — the meta-picker constructor wraps any
    // zenpredict::Model; the predictor runs the forward pass.
    let zmodel = zenpredict::Model::from_bytes(&outcome.bake_bytes).expect("load");
    let mut mp = zenpicker::MetaPicker::new(&zmodel);
    let p = data.n_features;
    let r = val_rows[0];
    let row = &data.features[r * p..r * p + p];
    let out = mp
        .predictor()
        .predict(row)
        .expect("predict via meta-picker");
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());

    let _ = std::fs::remove_dir_all(&dir);
}
