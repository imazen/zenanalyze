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

use zenpicker_train::{ScalarAxisSpec, build_picker_dataset_with};

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
    assert!(ds.scalar_sentinels[0].is_nan(), "chroma_scale has no sentinel");
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
    assert!(lam[0].is_nan(), "T=40 lambda sentinel-masked (config A λ=0.0)");

    assert!(near(cs[1], 1.0), "T=45 chroma_scale (config B)");
    assert!(near(lam[1], 8.0), "T=45 lambda (config B)");

    assert!(near(cs[2], 1.0), "T=80 chroma_scale (config D)");
    assert!(near(lam[2], 25.0), "T=80 lambda (config D beats C on bytes)");

    assert!(near(cs[3], 1.0), "T=85 chroma_scale (config D)");
    assert!(near(lam[3], 25.0), "T=85 lambda (config D)");

    // bytes_log stays parallel + correct (min-bytes winner per target).
    assert_eq!(ds.bytes_log.len(), 4);
    assert!(near(ds.bytes_log[0], 1000.0_f64.ln()), "T=40 min bytes = A");
    assert!(near(ds.bytes_log[1], 1100.0_f64.ln()), "T=45 min bytes = B");
    assert!(near(ds.bytes_log[2], 1900.0_f64.ln()), "T=80 min bytes = D");
}
