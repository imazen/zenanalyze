//! Dataset policy tests use small tabular fixtures, not codec quality claims.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use zenpicker_train::{
    PickerDataset, ScalarAxisSpec, build_picker_dataset, build_picker_dataset_with,
    build_picker_dataset_with_time_budget,
};

#[derive(Clone)]
struct Row {
    image: &'static str,
    codec: &'static str,
    knob: &'static str,
    bytes: f64,
    score: f64,
    time: Option<f64>,
    budget: Option<f64>,
}

fn rows() -> Vec<Row> {
    [
        ("{\"cell\":\"fast\",\"lambda\":25}", 100.0, 100.0, 20.0),
        ("{\"cell\":\"fast\",\"lambda\":8}", 120.0, 100.0, 10.0),
        ("{\"cell\":\"slow\",\"lambda\":14.5}", 80.0, 100.0, 11.0),
        ("{\"cell\":\"low\",\"lambda\":0}", 60.0, 40.0, 5.0),
    ]
    .into_iter()
    .map(|(knob, bytes, score, time)| Row {
        image: "a_64.png",
        codec: "zenjxl",
        knob,
        bytes,
        score,
        time: Some(time),
        budget: Some(10.0),
    })
    .collect()
}

fn write_sweep(
    label: &str,
    rows: &[Row],
    time_type: Option<DataType>,
    budget_type: Option<DataType>,
) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("picker_time_{}_{}", std::process::id(), label));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sweep.parquet");
    let mut fields = vec![
        Field::new("image_basename", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::Int64, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new("encoded_bytes", DataType::Float64, false),
        Field::new("score_zensim", DataType::Float64, false),
        Field::new("feat_0", DataType::Float32, false),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.image).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.codec).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(vec![100; rows.len()])),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.knob).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|r| r.bytes).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|r| r.score).collect::<Vec<_>>(),
        )),
        Arc::new(Float32Array::from(vec![0.5; rows.len()])),
    ];
    for (name, dtype, values) in [
        (
            "encode_ms",
            time_type,
            rows.iter().map(|r| r.time).collect::<Vec<_>>(),
        ),
        (
            "budget_ms",
            budget_type,
            rows.iter().map(|r| r.budget).collect::<Vec<_>>(),
        ),
    ] {
        if let Some(dtype) = dtype {
            fields.push(Field::new(name, dtype.clone(), true));
            columns.push(match dtype {
                DataType::Float64 => Arc::new(Float64Array::from(values)),
                DataType::Float32 => Arc::new(Float32Array::from(
                    values
                        .into_iter()
                        .map(|v| v.map(|v| v as f32))
                        .collect::<Vec<_>>(),
                )),
                _ => panic!("test type"),
            });
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

fn load(path: &Path, targets: &[i64]) -> Result<PickerDataset, zenpicker_train::TrainError> {
    build_picker_dataset_with_time_budget(
        path,
        Some("zenjxl"),
        targets,
        &[ScalarAxisSpec::new("lambda", Some(0.0))],
        "encode_ms",
        "budget_ms",
    )
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn quality_and_time_are_both_required_and_scalars_follow_the_winner() {
    let mut data = rows();
    let mut larger = rows();
    for r in &mut larger {
        r.image = "a_256.png";
        r.budget = Some(30.0);
    }
    data.extend(larger);
    let path = write_sweep(
        "winners",
        &data,
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    let ds = load(&path, &[40, 100]).unwrap();
    assert_eq!(ds.cell_labels, ["fast", "low", "slow"]);
    assert_eq!(
        ds.image_ids,
        ["a_256.png", "a_256.png", "a_64.png", "a_64.png"]
    );
    assert_eq!(
        ds.reach,
        [
            true, true, true, true, false, true, true, true, false, true, false, false
        ]
    );
    assert_eq!(
        bits(&ds.bytes_log),
        bits(&[
            100_f64.ln(),
            60_f64.ln(),
            80_f64.ln(),
            100_f64.ln(),
            f64::NAN,
            80_f64.ln(),
            120_f64.ln(),
            60_f64.ln(),
            f64::NAN,
            120_f64.ln(),
            f64::NAN,
            f64::NAN,
        ])
    );
    assert_eq!(
        bits(&ds.scalars[0]),
        bits(&[
            25.0,
            f64::NAN,
            14.5,
            25.0,
            f64::NAN,
            14.5,
            8.0,
            f64::NAN,
            f64::NAN,
            8.0,
            f64::NAN,
            f64::NAN,
        ])
    );
}

#[test]
fn unconstrained_builders_keep_the_frozen_dataset_and_ignore_time_columns() {
    let mut data = rows();
    for r in &mut data {
        r.time = None;
        r.budget = Some(f64::NAN);
    }
    let path = write_sweep(
        "unconstrained",
        &data,
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    let ds = build_picker_dataset_with(
        &path,
        None,
        &[40, 100, 101],
        &[ScalarAxisSpec::new("lambda", Some(0.0))],
    )
    .unwrap();
    assert_eq!(ds.n_in, 2);
    assert_eq!(ds.n_cells, 3);
    assert_eq!(ds.image_ids, ["a_64.png", "a_64.png"]);
    assert_eq!(ds.target_zq, [40, 100]);
    assert_eq!(ds.zq_targets, [40, 100, 101]);
    assert_eq!(ds.feature_names, ["feat_0"]);
    assert_eq!(ds.features, [0.5, 0.4, 0.5, 1.0]);
    assert_eq!(ds.cell_labels, ["fast", "low", "slow"]);
    assert_eq!(ds.reach, [true, true, true, true, false, true]);
    assert_eq!(
        bits(&ds.bytes_log),
        bits(&[
            100_f64.ln(),
            60_f64.ln(),
            80_f64.ln(),
            100_f64.ln(),
            f64::NAN,
            80_f64.ln()
        ])
    );
    assert_eq!(ds.scalar_axes, ["lambda"]);
    assert_eq!(ds.scalar_sentinels, [0.0]);
    assert_eq!(
        bits(&ds.scalars[0]),
        bits(&[25.0, f64::NAN, 14.5, 25.0, f64::NAN, 14.5])
    );
    let base = build_picker_dataset(&path, None, &[40, 100, 101]).unwrap();
    assert_eq!(bits(&base.bytes_log), bits(&ds.bytes_log));
    assert_eq!(base.features, ds.features);
    assert_eq!(base.reach, ds.reach);
    assert_eq!(base.cell_labels, ds.cell_labels);
    assert!(base.scalar_axes.is_empty() && base.scalars.is_empty());
}

#[test]
fn invalid_or_missing_timing_data_is_refused() {
    for column in ["encode_ms", "budget_ms"] {
        for (case, value) in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(0.0),
            Some(-1.0),
        ]
        .into_iter()
        .enumerate()
        {
            let mut data = rows();
            if column == "encode_ms" {
                data[0].time = value;
            } else {
                data[0].budget = value;
            }
            let path = write_sweep(
                &format!("invalid_{column}_{case}"),
                &data,
                Some(DataType::Float64),
                Some(DataType::Float64),
            );
            let error = load(&path, &[100]).err().unwrap().to_string();
            assert!(error.contains(column) && error.contains("row 0"), "{error}");
        }
        for dtype in [None, Some(DataType::Float32)] {
            let (t, b) = if column == "encode_ms" {
                (dtype.clone(), Some(DataType::Float64))
            } else {
                (Some(DataType::Float64), dtype.clone())
            };
            let path = write_sweep(&format!("schema_{column}_{dtype:?}"), &rows(), t, b);
            let error = load(&path, &[100]).err().unwrap().to_string();
            assert!(error.contains(column), "{error}");
        }
    }
}

#[test]
fn inconsistent_budgets_and_any_unreachable_image_target_are_errors() {
    let mut data = rows();
    data[1].budget = Some(9.0);
    let path = write_sweep(
        "inconsistent",
        &data,
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    assert!(
        load(&path, &[100])
            .err()
            .unwrap()
            .to_string()
            .contains("inconsistent time budget")
    );
    let mut data = rows();
    let mut failed = rows();
    for r in &mut failed {
        r.image = "failed.png";
        r.time = Some(50.0);
    }
    data.extend(failed);
    let path = write_sweep(
        "unreachable",
        &data,
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    let error = load(&path, &[100]).err().unwrap().to_string();
    assert!(
        error.contains("failed.png") && error.contains("target 100"),
        "{error}"
    );
    let path = write_sweep(
        "quality",
        &rows(),
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    assert!(
        load(&path, &[101])
            .err()
            .unwrap()
            .to_string()
            .contains("target 101")
    );
}

#[test]
fn codec_filter_applies_before_time_value_validation() {
    let mut data = rows();
    let mut other = data[0].clone();
    other.codec = "unselected";
    other.time = None;
    other.budget = None;
    data.push(other);
    let path = write_sweep(
        "filter",
        &data,
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    assert_eq!(load(&path, &[100]).unwrap().n_rows(), 1);
}
