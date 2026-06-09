//! Chunk-1 of the scalar hybrid-heads restore: within-cell-optimal
//! SCALAR target capture.
//!
//! Mints a tiny single-cell sweep that varies two scalar knobs
//! (`chroma_scale`, `lambda`) across configs with hand-chosen
//! `(score, bytes)` so the within-cell-optimal pick at each target is
//! known by inspection, then asserts `build_picker_dataset_with` captures
//! the optimal config's scalar value per cell — including:
//!   - scalar knobs do NOT split the categorical cell,
//!   - the non-monotonic case (a higher-`chroma_scale` config that is
//!     nonetheless cheaper wins),
//!   - sentinel masking (`lambda = 0.0` trellis-off → NaN target).

use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;

use zenpicker_train::{
    MlpConfig, ScalarAxisSpec, ScalarHeadSpec, bake_mlp_picker_to_znpr_v3,
    build_picker_dataset_with, default_grid, evaluate_scalar_heads, fit_standardizer,
    grouped_split_picker, run_search, standardize_all,
};

/// One image, one categorical cell (`subsampling=420`). Four configs,
/// each `(chroma_scale, q, lambda, score_zensim, encoded_bytes)`:
///   A: cs=0.6 q=40 λ=0.0  s=40 b=1000
///   B: cs=1.0 q=40 λ=8.0  s=45 b=1100
///   C: cs=0.6 q=80 λ=14.5 s=80 b=2000
///   D: cs=1.0 q=80 λ=25.0 s=85 b=1900   (cheaper than C despite higher cs)
fn write_scalar_sweep(path: &std::path::Path) {
    let configs = [
        (0.6f64, 40i64, 0.0f64, 40.0f64, 1000.0f64),
        (1.0, 40, 8.0, 45.0, 1100.0),
        (0.6, 80, 14.5, 80.0, 2000.0),
        (1.0, 80, 25.0, 85.0, 1900.0),
    ];

    let mut image_basename = Vec::new();
    let mut codec = Vec::new();
    let mut q = Vec::new();
    let mut knob = Vec::new();
    let mut enc_bytes = Vec::new();
    let mut score = Vec::new();
    let mut f0 = Vec::new();

    for (cs, qq, lam, sc, by) in configs {
        image_basename.push("img_000.png".to_string());
        codec.push("zenjpeg".to_string());
        q.push(qq);
        knob.push(format!(
            "{{\"subsampling\":\"420\",\"chroma_scale\":{cs},\"lambda\":{lam}}}"
        ));
        enc_bytes.push(by);
        score.push(sc);
        f0.push(0.5f32);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("image_basename", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::Int64, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new("encoded_bytes", DataType::Float64, false),
        Field::new("score_zensim", DataType::Float64, false),
        Field::new("feat_0", DataType::Float32, false),
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
        ],
    )
    .unwrap();

    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn within_cell_optimal_scalar_capture_and_sentinel_mask() {
    let dir = std::env::temp_dir().join(format!("zpt_scalar_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pq = dir.join("scalar_sweep.parquet");
    write_scalar_sweep(&pq);

    let zq_targets = vec![40i64, 45, 80, 85];
    let axes = vec![
        ScalarAxisSpec::new("chroma_scale", None),
        ScalarAxisSpec::new("lambda", Some(0.0)),
    ];
    let ds = build_picker_dataset_with(&pq, Some("zenjpeg"), &zq_targets, &axes)
        .expect("build scalar dataset");

    // Scalar knobs must NOT split the categorical cell.
    assert_eq!(ds.n_cells, 1, "chroma_scale/lambda must not form cells");
    assert_eq!(ds.scalar_axes, vec!["chroma_scale", "lambda"]);
    assert_eq!(ds.scalar_sentinels.len(), 2);
    assert!(
        ds.scalar_sentinels[0].is_nan(),
        "chroma_scale has no sentinel"
    );
    assert_eq!(ds.scalar_sentinels[1], 0.0, "lambda sentinel = 0.0");

    // 4 reachable targets × 1 cell.
    assert_eq!(ds.n_rows(), 4);
    assert_eq!(ds.scalars.len(), 2);
    assert_eq!(ds.scalars[0].len(), 4);
    assert_eq!(ds.scalars[1].len(), 4);
    assert!(ds.reach.iter().all(|&r| r), "all four targets reachable");

    // Rows follow `zq_targets` order for the single image. Expected
    // within-cell-optimal config per target:
    //   T=40 → A (cs=0.6, λ sentinel → NaN)
    //   T=45 → B (cs=1.0, λ=8.0)
    //   T=80 → D (cs=1.0, λ=25.0)  [D beats C on bytes despite higher cs]
    //   T=85 → D (cs=1.0, λ=25.0)
    let cs = &ds.scalars[0];
    let lam = &ds.scalars[1];
    let near = |a: f64, b: f64| (a - b).abs() < 1e-9;

    assert!(near(cs[0], 0.6), "T=40 chroma_scale (config A)");
    assert!(
        lam[0].is_nan(),
        "T=40 lambda sentinel-masked (config A λ=0.0)"
    );

    assert!(near(cs[1], 1.0), "T=45 chroma_scale (config B)");
    assert!(near(lam[1], 8.0), "T=45 lambda (config B)");

    assert!(near(cs[2], 1.0), "T=80 chroma_scale (config D)");
    assert!(
        near(lam[2], 25.0),
        "T=80 lambda (config D beats C on bytes)"
    );

    assert!(near(cs[3], 1.0), "T=85 chroma_scale (config D)");
    assert!(near(lam[3], 25.0), "T=85 lambda (config D)");

    // bytes_log stays parallel + correct (min-bytes winner per target).
    assert_eq!(ds.bytes_log.len(), 4);
    assert!(near(ds.bytes_log[0], 1000.0_f64.ln()), "T=40 min bytes = A");
    assert!(near(ds.bytes_log[1], 1100.0_f64.ln()), "T=45 min bytes = B");
    assert!(near(ds.bytes_log[2], 1900.0_f64.ln()), "T=80 min bytes = D");
}

/// Multi-image sweep: 2 cells (`subsampling` 420/444), `chroma_scale` ∈
/// {0.6,1.0,1.5} × q ∈ {30,60,90} per cell, per image. Score/bytes rise
/// with q + chroma_scale so the within-cell-optimal scalar varies.
fn write_multi_scalar_sweep(path: &std::path::Path, n_images: usize) {
    let cells = [("420", 0.0f64), ("444", 1.0f64)];
    let css = [0.6f64, 1.0, 1.5];
    let qs = [30i64, 60, 90];

    let mut image_basename = Vec::new();
    let mut codec = Vec::new();
    let mut q = Vec::new();
    let mut knob = Vec::new();
    let mut enc_bytes = Vec::new();
    let mut score = Vec::new();
    let mut f0 = Vec::new();

    for img in 0..n_images {
        let a = (img as f32) * 0.05 - 0.5;
        for (sub, cbias) in cells {
            for &cs in &css {
                for &qq in &qs {
                    image_basename.push(format!("img_{img:03}.png"));
                    codec.push("zenjpeg".to_string());
                    q.push(qq);
                    knob.push(format!(
                        "{{\"subsampling\":\"{sub}\",\"chroma_scale\":{cs}}}"
                    ));
                    f0.push(a);
                    score.push(20.0 + 0.7 * qq as f64 + 6.0 * cs + 2.0 * cbias + 5.0 * a as f64);
                    enc_bytes.push(
                        (800.0 + 30.0 * qq as f64) * (1.0 + 0.4 * (cs - 0.6)) * (1.0 + 0.2 * cbias),
                    );
                }
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
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn scalar_heads_train_and_emit_natural_units() {
    let dir = std::env::temp_dir().join(format!("zpt_scalar_train_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pq = dir.join("multi_scalar.parquet");
    write_multi_scalar_sweep(&pq, 40);

    let zq_targets = vec![30i64, 40, 50, 60, 70];
    let axes = vec![ScalarAxisSpec::new("chroma_scale", None)];
    let ds = build_picker_dataset_with(&pq, Some("zenjpeg"), &zq_targets, &axes)
        .expect("build scalar dataset");
    assert_eq!(ds.n_cells, 2, "two subsampling cells");
    assert_eq!(ds.scalar_axes, vec!["chroma_scale"]);

    let (train, val) = grouped_split_picker(&ds, 0.25);
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);

    let grid = default_grid();
    let base = MlpConfig {
        max_iter: 60,
        ..Default::default()
    };
    let res = run_search(&ds, &x_std, &train, &val, &grid, &base, |_| {}).expect("search result");
    let m = &res.best_model;

    // Output widened: bytes_log block + one scalar block.
    assert_eq!(
        m.n_out,
        ds.n_cells * 2,
        "n_out = n_cells*(1 + n_scalar_axes)"
    );

    // Predict over held-out rows; the scalar block must be finite AND in
    // natural units (data is 0.6..1.5). A non-rescaled standardized head
    // would center near 0.0 — the per-head rescale folds μ/σ back so the
    // head centers near the data mean (~1.0).
    let mut chroma_sum = 0.0f64;
    let mut cnt = 0usize;
    for &r in val.iter().take(30) {
        let x = &x_std[r * ds.n_in..(r + 1) * ds.n_in];
        let p = m.predict(x);
        assert_eq!(p.len(), ds.n_cells * 2);
        assert!(p.iter().all(|v| v.is_finite()), "all predictions finite");
        for c in 0..ds.n_cells {
            chroma_sum += p[ds.n_cells + c]; // scalar block follows bytes_log block
            cnt += 1;
        }
    }
    let chroma_mean = chroma_sum / cnt as f64;
    assert!(
        chroma_mean > 0.3 && chroma_mean < 1.8,
        "rescaled chroma_scale head emits natural units (got mean {chroma_mean})"
    );

    // Held-out per-axis MAE is finite, scored over real targets, and in a
    // sane natural-unit range (chroma_scale spans 0.6..1.5, so MAE < 1).
    let evals = evaluate_scalar_heads(m, &ds, &x_std, &val);
    assert_eq!(evals.len(), 1);
    assert_eq!(evals[0].axis, "chroma_scale");
    assert!(evals[0].n > 0, "scalar MAE scored some held-out targets");
    assert!(
        evals[0].mae.is_finite() && evals[0].mae < 1.0,
        "chroma_scale held-out MAE finite + sane (got {})",
        evals[0].mae
    );
}

/// Two scalar axes in `knob_tuple_json` (chroma_scale + lambda), 2 cells.
fn write_two_axis_sweep(path: &std::path::Path, n_images: usize) {
    let cells = [("420", 0.0f64), ("444", 1.0f64)];
    let css = [0.6f64, 1.0, 1.5];
    let lams = [8.0f64, 14.5, 25.0];
    let qs = [30i64, 60, 90];

    let mut image_basename = Vec::new();
    let mut codec = Vec::new();
    let mut q = Vec::new();
    let mut knob = Vec::new();
    let mut enc_bytes = Vec::new();
    let mut score = Vec::new();
    let mut f0 = Vec::new();

    for img in 0..n_images {
        let a = (img as f32) * 0.05 - 0.5;
        for (sub, cbias) in cells {
            for &cs in &css {
                for &lam in &lams {
                    for &qq in &qs {
                        image_basename.push(format!("img_{img:03}.png"));
                        codec.push("zenjpeg".to_string());
                        q.push(qq);
                        knob.push(format!(
                            "{{\"subsampling\":\"{sub}\",\"chroma_scale\":{cs},\"lambda\":{lam}}}"
                        ));
                        f0.push(a);
                        score.push(
                            20.0 + 0.6 * qq as f64
                                + 5.0 * cs
                                + 0.3 * lam
                                + 2.0 * cbias
                                + 4.0 * a as f64,
                        );
                        enc_bytes.push(
                            (700.0 + 25.0 * qq as f64)
                                * (1.0 + 0.3 * (cs - 0.6))
                                * (1.0 + 0.01 * (lam - 8.0))
                                * (1.0 + 0.2 * cbias),
                        );
                    }
                }
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
        ],
    )
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn scalar_bake_roundtrips_through_zenpredict() {
    let dir = std::env::temp_dir().join(format!("zpt_scalar_bake_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pq = dir.join("two_axis.parquet");
    write_two_axis_sweep(&pq, 36);

    let zq_targets = vec![30i64, 40, 50, 60, 70];
    let axes = vec![
        ScalarAxisSpec::new("chroma_scale", None),
        ScalarAxisSpec::new("lambda", Some(0.0)),
    ];
    let ds = build_picker_dataset_with(&pq, Some("zenjpeg"), &zq_targets, &axes).unwrap();
    assert_eq!(ds.n_cells, 2, "subsampling cells only");
    assert_eq!(ds.scalar_axes, vec!["chroma_scale", "lambda"]);

    let (train, val) = grouped_split_picker(&ds, 0.25);
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);
    let base = MlpConfig {
        max_iter: 50,
        ..Default::default()
    };
    let res = run_search(&ds, &x_std, &train, &val, &default_grid(), &base, |_| {}).unwrap();
    let model = &res.best_model;
    assert_eq!(model.n_out, ds.n_cells * 3, "bytes + 2 scalar blocks");

    // Bake with zenjpeg's standard output specs (chroma identity [0.6,1.5];
    // lambda round + discrete {0,8,14.5,25} + sentinel 0).
    let heads = vec![
        ScalarHeadSpec::zenjpeg("chroma_scale").unwrap(),
        ScalarHeadSpec::zenjpeg("lambda").unwrap(),
    ];
    let bytes = bake_mlp_picker_to_znpr_v3(
        model,
        &mean,
        &scale,
        &ds.feature_names,
        &ds.cell_labels,
        "zenjpeg",
        &ds.zq_targets,
        false,
        None,
        &heads,
    )
    .expect("hybrid bake");
    assert_eq!(&bytes[0..4], b"ZNPR");
    assert_eq!(bytes[4], 0x03, "ZNPR v3");

    let zmodel = zenpredict::Model::from_bytes(&bytes).expect("load bake");
    assert_eq!(
        zmodel.n_outputs(),
        ds.n_cells * 3,
        "bake carries the wide output"
    );
    let mut predictor = zenpredict::Predictor::new(&zmodel);

    let n_cells = ds.n_cells;
    let n_in = ds.n_in;
    let mut pick_matches = 0usize;
    let mut checked = 0usize;
    for &r in val.iter().take(20) {
        // The bake folds the input standardizer, so it takes RAW features
        // and must reproduce the in-process (standardized-input) forward.
        let raw: Vec<f32> = ds.features[r * n_in..(r + 1) * n_in]
            .iter()
            .map(|&v| v as f32)
            .collect();
        let in_proc = model.predict(&x_std[r * n_in..(r + 1) * n_in]);

        // RAW predict: the bytes-block argmin pick must survive the bake.
        let loaded = predictor.predict(&raw).expect("predict").to_vec();
        assert_eq!(loaded.len(), n_cells * 3);
        let argmin = |v: &[f64]| (0..n_cells).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        let argmin_f32 = |v: &[f32]| (0..n_cells).min_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        if argmin(&in_proc) == argmin_f32(&loaded) {
            pick_matches += 1;
        }
        checked += 1;

        // SPEC predict: chroma_scale clamped to [0.6,1.5]; lambda snapped
        // to the trellis grid (or Default on the 0.0 sentinel).
        let ov = predictor
            .predict_with_specs(&raw)
            .expect("predict_with_specs");
        for c in 0..n_cells {
            if let Some(v) = ov[n_cells + c].value() {
                assert!(
                    (0.6..=1.5).contains(&v),
                    "chroma_scale clamped to bounds, got {v}"
                );
            }
            if let Some(v) = ov[2 * n_cells + c].value() {
                assert!(
                    [0.0f32, 8.0, 14.5, 25.0]
                        .iter()
                        .any(|g| (v - g).abs() < 1e-4),
                    "lambda snapped to the discrete grid, got {v}"
                );
            }
        }
    }
    assert_eq!(
        pick_matches, checked,
        "loaded-bake argmin pick must match the in-process pick on every probe row"
    );
}
