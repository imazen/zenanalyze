//! End-to-end for the within-cell-optimal MLP picker (the real
//! formulation). Mint a synthetic unified-sweep parquet with multiple
//! knob configs per image, build the picker dataset, confirm:
//!   - the dataset inputs are image features + `zq_norm` ONLY (q is
//!     not a feature),
//!   - the within-cell `bytes_log` targets respect the reach mask,
//!   - the MLP trains, the held-out picker eval is finite,
//!   - the bake is ZNPR v3 and loads via `zenpredict::Model`,
//!   - a loaded-bake forward pass matches the in-process MLP.

use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;

use zenpicker_train::{
    HeldoutManifest, MlpConfig, MlpPickerManifestInputs, SearchManifest, bake_mlp_picker,
    build_picker_dataset, evaluate_picker, fit_standardizer, grouped_split_picker, standardize_all,
    train_mlp,
};

/// Synthetic unified-sweep parquet:
///   - `n_images` images, each with 2 deterministic features,
///   - 2 categorical knob cells (`subsampling=420` vs `444`),
///   - q ∈ {10,30,60,90}; achieved `score_zensim` rises with q and is
///     slightly higher for 444; `encoded_bytes` rises with q and 444
///     costs more bytes than 420 (so the within-cell-optimal pick at a
///     given target is content-dependent).
fn write_synthetic_sweep(path: &std::path::Path, n_images: usize) {
    let qs = [10i64, 30, 60, 90];
    let cells = [
        ("{\"subsampling\":\"420\"}", 0.0f64),
        ("{\"subsampling\":\"444\"}", 1.0f64),
    ];

    let mut image_basename = Vec::new();
    let mut codec = Vec::new();
    let mut q = Vec::new();
    let mut knob = Vec::new();
    let mut enc_bytes = Vec::new();
    let mut score = Vec::new();
    let mut f0 = Vec::new();
    let mut f1 = Vec::new();

    for img in 0..n_images {
        let a = (img as f32) * 0.07 - 0.4; // feature 0
        let b = ((img * 5) % 13) as f32 * 0.03; // feature 1
        for &qq in &qs {
            for (kj, chroma) in cells {
                image_basename.push(format!("img_{img:03}.png"));
                codec.push("zenjpeg".to_string());
                q.push(qq);
                knob.push(kj.to_string());
                f0.push(a);
                f1.push(b);
                // Achieved score: rises with q, +a couple points for 444,
                // content-modulated by feature a.
                let s = 20.0 + 0.7 * qq as f64 + 3.0 * chroma + 8.0 * a as f64;
                score.push(s);
                // Bytes: rise with q; 444 ~25% larger; content modulated.
                let by =
                    (1000.0 + 40.0 * qq as f64) * (1.0 + 0.25 * chroma) * (1.0 + 0.3 * b as f64);
                enc_bytes.push(by);
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
            Arc::new(StringArray::from(image_basename)),
            Arc::new(StringArray::from(codec)),
            Arc::new(Int64Array::from(q)),
            Arc::new(StringArray::from(knob)),
            Arc::new(Float64Array::from(enc_bytes)),
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
fn picker_no_q_leak_train_bake_load() {
    let dir = std::env::temp_dir().join(format!("zenpicker_mlp_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let parquet_path = dir.join("sweep.parquet");
    let bake_path = dir.join("picker.bin");
    write_synthetic_sweep(&parquet_path, 40);

    // A small zq grid so the synthetic ladder reaches several targets.
    let zq_targets: Vec<i64> = vec![20, 30, 40, 50, 60];
    let ds = build_picker_dataset(&parquet_path, Some("zenjpeg"), &zq_targets)
        .expect("build picker dataset");

    // 2 cells (420, 444).
    assert_eq!(ds.n_cells, 2, "two subsampling cells");
    // Inputs = 2 image features + zq_norm.
    assert_eq!(ds.n_in, 3, "feat_0 + feat_1 + zq_norm");
    assert_eq!(ds.feature_names, vec!["feat_0", "feat_1"]);
    // CRITICAL: no feature named "q" and no codec-q axis. The only
    // non-image input is zq_norm, the REQUESTED quality.
    assert!(
        !ds.feature_names.iter().any(|n| n == "q"),
        "codec q must NOT be a feature (no leakage)"
    );

    // Reach mask sanity: every emitted row has at least one reachable cell.
    for r in 0..ds.n_rows() {
        let reach = &ds.reach[r * ds.n_cells..(r + 1) * ds.n_cells];
        assert!(
            reach.iter().any(|&x| x),
            "every row must have a reachable cell"
        );
        // bytes_log is NaN exactly where unreachable.
        let bl = &ds.bytes_log[r * ds.n_cells..(r + 1) * ds.n_cells];
        for c in 0..ds.n_cells {
            assert_eq!(
                reach[c],
                bl[c].is_finite(),
                "reach mask must match finite bytes_log"
            );
        }
    }

    // Grouped-by-image split.
    let (train_rows, val_rows) = grouped_split_picker(&ds, 0.25);
    assert!(!train_rows.is_empty());
    assert!(!val_rows.is_empty());
    let train_imgs: std::collections::HashSet<&str> = train_rows
        .iter()
        .map(|&r| ds.image_ids[r].as_str())
        .collect();
    for &r in &val_rows {
        assert!(
            !train_imgs.contains(ds.image_ids[r].as_str()),
            "image leaked across the grouped split"
        );
    }

    // Standardize on train, train a small MLP fast.
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train_rows);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);

    // Pack train rows contiguously for the MLP trainer.
    let mut x_tr = Vec::new();
    let mut y_tr = Vec::new();
    for &r in &train_rows {
        x_tr.extend_from_slice(&x_std[r * ds.n_in..(r + 1) * ds.n_in]);
        y_tr.extend_from_slice(&ds.bytes_log[r * ds.n_cells..(r + 1) * ds.n_cells]);
    }
    let cfg = MlpConfig {
        hidden: vec![16, 16],
        max_iter: 120,
        ..MlpConfig::default()
    };
    let model = train_mlp(&x_tr, &y_tr, train_rows.len(), ds.n_in, ds.n_cells, &cfg);
    assert_eq!(model.n_out, ds.n_cells);
    assert_eq!(model.n_in, ds.n_in);

    // Held-out picker eval — finite panel.
    let eval = evaluate_picker(&model, &ds, &x_std, &val_rows).expect("held-out eval");
    assert!(eval.bytes_panel.srocc.is_finite());
    assert!(eval.argmin_acc >= 0.0 && eval.argmin_acc <= 1.0);
    assert!(eval.n_pairs > 0);

    // Bake → ZNPR v3 → load.
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
    let search = SearchManifest {
        kind: "single_fit".to_string(),
        candidates: vec![],
        selected_index: 0,
        selection_metric: "heldout_bytes_log_srocc".to_string(),
    };
    let outcome = bake_mlp_picker(
        &model,
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
            distillation: None,
        },
    )
    .expect("bake mlp picker");

    let bytes = &outcome.bake_bytes;
    assert_eq!(&bytes[0..4], b"ZNPR", "ZNPR magic");
    assert_eq!(bytes[4], 0x03, "ZNPR v3");

    // Manifest confirms the no-q-leak contract.
    let manifest_txt = std::fs::read_to_string(format!("{}.toml", bake_path.display())).unwrap();
    assert!(manifest_txt.contains("q_is_input = false"));
    assert!(manifest_txt.contains("within_cell_optimal_bytes_argmin"));
    assert!(manifest_txt.contains("data_coverage_caveat"));

    // Load via zenpredict and confirm the forward pass matches.
    let zmodel = zenpredict::Model::from_bytes(bytes).expect("load bake");
    assert_eq!(zmodel.n_inputs(), ds.n_in);
    assert_eq!(zmodel.n_outputs(), ds.n_cells);
    let mut predictor = zenpredict::Predictor::new(&zmodel);

    // zenpredict applies the folded scaler to RAW inputs; the in-proc
    // model takes already-standardized inputs. Feed the runtime the RAW
    // feature row, the in-proc model the standardized row, and compare.
    for &r in val_rows.iter().take(8) {
        let raw = &ds.features[r * ds.n_in..(r + 1) * ds.n_in];
        let std_row = &x_std[r * ds.n_in..(r + 1) * ds.n_in];
        let raw_f32: Vec<f32> = raw.iter().map(|&x| x as f32).collect();
        let out = predictor.predict(&raw_f32).expect("predict");
        assert_eq!(out.len(), ds.n_cells);
        let in_proc = model.predict(std_row);
        for c in 0..ds.n_cells {
            assert!(out[c].is_finite());
            let delta = (out[c] as f64 - in_proc[c]).abs();
            // f32 runtime vs f64 in-proc; bytes_log ~ 7..12, allow a
            // modest absolute tolerance.
            assert!(
                delta < 0.05,
                "loaded bake cell {c} ({}) vs in-proc ({}) delta {delta}",
                out[c],
                in_proc[c]
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}
