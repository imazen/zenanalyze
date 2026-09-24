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

fn cli(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_zenpicker-train"))
        .args(args)
        .output()
        .unwrap()
}

fn success(output: std::process::Output) {
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn manifest(path: &Path) -> toml::Table {
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn check_policy(value: &toml::Table) {
    assert_eq!(
        value["formulation"].as_str(),
        Some("min_encoded_bytes_subject_to_quality_and_encode_time_budget")
    );
    assert_eq!(
        value["dataset_selection"]["encode_time_column"].as_str(),
        Some("encode_ms")
    );
    assert_eq!(
        value["dataset_selection"]["time_budget_column"].as_str(),
        Some("budget_ms")
    );
}

#[test]
fn cli_requires_paired_nonempty_columns_and_mlp_mode() {
    for args in [
        vec!["--encode-time-column", "encode_ms"],
        vec!["--time-budget-column", "budget_ms"],
        vec![
            "--encode-time-column",
            "",
            "--time-budget-column",
            "budget_ms",
        ],
    ] {
        let result = cli(&args);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("must be supplied together"));
    }
    let result = cli(&[
        "--mode",
        "ridge",
        "--encode-time-column",
        "encode_ms",
        "--time-budget-column",
        "budget_ms",
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("requires --mode mlp"));
}

#[test]
fn cli_export_train_and_eval_apply_and_record_the_same_constraint() {
    use arrow::array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let mut data = rows();
    let mut second = rows();
    for r in &mut second {
        r.image = "b_64.png";
    }
    data.extend(second);
    let path = write_sweep(
        "cli",
        &data,
        Some(DataType::Float64),
        Some(DataType::Float64),
    );
    let dir = path.parent().unwrap();
    let export = dir.join("teacher.parquet");
    let bake = dir.join("budget.bin");
    let plain = dir.join("plain.bin");
    let recipe = dir.join("recipe.toml");
    // Exercise TOML defaults and overriding one column via the CLI.
    std::fs::write(
        &recipe,
        "encode_time_column = 'wrong'\ntime_budget_column = 'budget_ms'\n",
    )
    .unwrap();
    let input = path.to_str().unwrap();
    let out = bake.to_str().unwrap();
    success(cli(&[
        "--input",
        input,
        "--codec",
        "zenjxl",
        "--out",
        out,
        "--export-dataset",
        export.to_str().unwrap(),
        "--manifest",
        recipe.to_str().unwrap(),
        "--encode-time-column",
        "encode_ms",
    ]));
    let exported = manifest(&dir.join("teacher.parquet.toml"));
    check_policy(&exported);
    assert_eq!(exported["codec_filter"].as_str(), Some("zenjxl"));
    assert_eq!(
        exported["artifact_sha256"].as_str(),
        Some(zenpicker_train::file_sha256(&export).unwrap().as_str())
    );
    let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&export).unwrap())
        .unwrap()
        .build()
        .unwrap();
    let mut n = 0;
    for batch in reader {
        let batch = batch.unwrap();
        let bytes = batch
            .column_by_name("bytes_log_0")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            assert!(!bytes.is_null(i));
            assert_eq!(bytes.value(i).to_bits(), (120_f64.ln() as f32).to_bits());
            n += 1;
        }
    }
    assert_eq!(n, 2 * zenpicker_train::default_zq_targets().len());

    success(cli(&[
        "--input",
        input,
        "--codec",
        "zenjxl",
        "--out",
        out,
        "--hidden",
        "2",
        "--encode-time-column",
        "encode_ms",
        "--time-budget-column",
        "budget_ms",
    ]));
    let trained = manifest(&dir.join("budget.bin.toml"));
    check_policy(&trained);
    assert_eq!(
        trained["input_sha256"].as_str(),
        Some(zenpicker_train::file_sha256(&path).unwrap().as_str())
    );
    assert_eq!(
        trained["bake_sha256"].as_str(),
        Some(zenpicker_train::file_sha256(&bake).unwrap().as_str())
    );

    success(cli(&[
        "--input",
        input,
        "--codec",
        "zenjxl",
        "--eval-bake",
        out,
        "--val-frac",
        "1.0",
        "--baselines",
        "--encode-time-column",
        "encode_ms",
        "--time-budget-column",
        "budget_ms",
    ]));
    let evaluated = manifest(&dir.join("budget.bin.eval.toml"));
    check_policy(&evaluated);
    assert_eq!(evaluated["codec_filter"].as_str(), Some("zenjxl"));
    assert_eq!(evaluated["phase"].as_str(), Some("evaluation"));
    assert_eq!(evaluated["heldout"]["n_rows"].as_integer(), Some(n as i64));
    assert_eq!(
        evaluated["artifact_sha256"].as_str(),
        trained["bake_sha256"].as_str()
    );

    success(cli(&[
        "--input",
        input,
        "--codec",
        "zenjxl",
        "--out",
        plain.to_str().unwrap(),
        "--hidden",
        "2",
    ]));
    let unbudgeted = manifest(&dir.join("plain.bin.toml"));
    assert_eq!(
        unbudgeted["formulation"].as_str(),
        Some("within_cell_optimal_bytes_argmin")
    );
    assert!(!unbudgeted.contains_key("dataset_selection"));
}
