//! JSON → bake_v2 round-trip tests. Exercises the public
//! `BakeRequestJson` schema that `zenpicker/tools/bake_picker.py`
//! emits, then loads the resulting bytes through the runtime to
//! confirm round-trip equivalence.

#![cfg(feature = "bake")]

use zenpredict::bake::{BakeJsonError, BakeRequestJson, bake_from_json, bake_from_json_str};
use zenpredict::keys;
use zenpredict::{MetadataType, Model, OutputValue, Predictor};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

#[test]
fn json_round_trip_minimal() {
    let json = r#"{
        "schema_hash": 12345,
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [
            {
                "in_dim": 2,
                "out_dim": 2,
                "activation": "identity",
                "dtype": "f32",
                "weights": [1.0, 0.0, 0.0, 1.0],
                "biases": [0.5, 1.0]
            }
        ]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert_eq!(model.schema_hash(), 12345);
    let mut p = Predictor::new(model);
    let out = p.predict(&[1.0, 1.0]).unwrap();
    assert_eq!(out, &[1.5, 2.0]);
}

#[test]
fn json_round_trip_with_metadata_and_bounds() {
    let json = r#"{
        "schema_hash": 9999,
        "flags": 0,
        "scaler_mean":  [0.0, 0.0, 0.0],
        "scaler_scale": [1.0, 1.0, 1.0],
        "layers": [
            {
                "in_dim": 3,
                "out_dim": 4,
                "activation": "leakyrelu",
                "dtype": "f16",
                "weights": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0
                ],
                "biases": [0.0, 0.0, 0.0, 0.0]
            },
            {
                "in_dim": 4,
                "out_dim": 2,
                "activation": "identity",
                "dtype": "f32",
                "weights": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "biases": [10.0, 20.0]
            }
        ],
        "feature_bounds": [
            {"low": -1.0, "high": 1.0},
            {"low":  0.0, "high": 100.0},
            {"low": -2.0, "high": 2.0}
        ],
        "metadata": [
            {"key": "zentrain.profile", "type": "numeric", "hex": "00"},
            {"key": "zentrain.bake_name", "type": "utf8", "text": "json_round_trip"},
            {"key": "zentrain.calibration_metrics", "type": "numeric",
             "f32": [0.0233, 0.0512, 0.563]},
            {"key": "zenjpeg.cell_config", "type": "bytes", "hex": "deadbeef"}
        ]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert_eq!(model.n_inputs(), 3);
    assert_eq!(model.n_outputs(), 2);

    let bounds = model.feature_bounds();
    assert_eq!(bounds.len(), 3);
    assert_eq!(bounds[1].high, 100.0);

    let md = model.metadata();
    assert_eq!(md.len(), 4);
    let prof: u8 = md.get_pod(keys::PROFILE).unwrap();
    assert_eq!(prof, 0);
    assert_eq!(md.get_utf8(keys::BAKE_NAME).unwrap(), "json_round_trip");
    let metrics: [f32; 3] = md.get_pod(keys::CALIBRATION_METRICS).unwrap();
    assert!((metrics[0] - 0.0233).abs() < 1e-6);
    assert_eq!(
        md.get_bytes("zenjpeg.cell_config").unwrap(),
        &[0xde, 0xad, 0xbe, 0xef]
    );
}

#[test]
fn json_rejects_text_on_numeric_type() {
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1.0],"biases":[0.0]}],
        "metadata": [
            {"key": "x", "type": "numeric", "text": "not allowed"}
        ]
    }"#;
    let req: BakeRequestJson = serde_json::from_str(json).unwrap();
    let err = bake_from_json(&req).unwrap_err();
    assert!(matches!(err, BakeJsonError::MetadataValueWrongType { .. }));
}

#[test]
fn json_rejects_missing_metadata_value() {
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1.0],"biases":[0.0]}],
        "metadata": [
            {"key": "x", "type": "bytes"}
        ]
    }"#;
    let req: BakeRequestJson = serde_json::from_str(json).unwrap();
    let err = bake_from_json(&req).unwrap_err();
    assert!(matches!(err, BakeJsonError::MetadataValueMissing { .. }));
}

#[test]
fn json_rejects_ambiguous_metadata_value() {
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1.0],"biases":[0.0]}],
        "metadata": [
            {"key": "x", "type": "numeric", "f32": [1.0], "hex": "deadbeef"}
        ]
    }"#;
    let req: BakeRequestJson = serde_json::from_str(json).unwrap();
    let err = bake_from_json(&req).unwrap_err();
    assert!(matches!(err, BakeJsonError::MetadataValueAmbiguous { .. }));
}

#[test]
fn json_rejects_bad_hex() {
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1.0],"biases":[0.0]}],
        "metadata": [
            {"key": "x", "type": "bytes", "hex": "deadbee"}
        ]
    }"#;
    let req: BakeRequestJson = serde_json::from_str(json).unwrap();
    let err = bake_from_json(&req).unwrap_err();
    assert!(matches!(err, BakeJsonError::BadHex(_)));
}

#[test]
fn json_metadata_kind_round_trips_to_wire_byte() {
    // Verify that `type: "bytes"` sets wire 0, "utf8" sets 1,
    // "numeric" sets 2 (not just maps to MetadataType variants).
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "layers": [{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1.0],"biases":[0.0]}],
        "metadata": [
            {"key": "b", "type": "bytes",   "hex": ""},
            {"key": "u", "type": "utf8",    "text": "hi"},
            {"key": "n", "type": "numeric", "hex": "01"}
        ]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let md = model.metadata();
    assert_eq!(md.get("b").unwrap().kind, MetadataType::Bytes);
    assert_eq!(md.get("u").unwrap().kind, MetadataType::Utf8);
    assert_eq!(md.get("n").unwrap().kind, MetadataType::Numeric);
}

/// Integration test that runs the actual `zenpredict-bake` binary
/// against a JSON file on disk, mirroring what the Python tooling
/// does. Skipped when the binary hasn't been built yet (cargo will
/// build it on demand the first time the test runs under
/// `cargo test`).
#[test]
fn cli_binary_round_trip() {
    use std::io::Write;
    use std::process::Command;

    let dir = tempdir();
    let json_path = dir.join("model.json");
    let bin_path = dir.join("model.bin");

    let json = r#"{
        "schema_hash": 42,
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [
            {"in_dim": 2, "out_dim": 2, "activation": "identity", "dtype": "f32",
             "weights": [1.0, 0.0, 0.0, 1.0], "biases": [3.0, 4.0]}
        ],
        "metadata": [
            {"key": "zentrain.bake_name", "type": "utf8", "text": "cli_test"}
        ]
    }"#;
    let mut f = std::fs::File::create(&json_path).expect("write json");
    f.write_all(json.as_bytes()).unwrap();
    drop(f);

    let exe = cargo_bin("zenpredict-bake");
    let status = Command::new(&exe)
        .arg(&json_path)
        .arg(&bin_path)
        .status()
        .expect("spawn zenpredict-bake");
    assert!(status.success(), "zenpredict-bake failed: {status:?}");

    let raw = std::fs::read(&bin_path).expect("read bin");
    let aligned = Aligned(raw);
    let model = Model::from_bytes(&aligned.0).expect("load v2 bin");
    assert_eq!(model.schema_hash(), 42);
    assert_eq!(
        model.metadata().get_utf8(keys::BAKE_NAME).unwrap(),
        "cli_test"
    );
}

#[test]
fn json_round_trip_with_output_specs() {
    // 2-input → 4-output identity-ish model:
    //   out0 = in0      → bounded to [0, 5]
    //   out1 = in1      → rounded to {0..7} with sentinel 0
    //   out2 = in0+in1  → passthrough
    //   out3 = in0-in1  → bounded to [-1, 1] with sentinel -1
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [
            {
                "in_dim": 2,
                "out_dim": 4,
                "activation": "identity",
                "dtype": "f32",
                "weights": [1.0, 0.0, 1.0, 1.0,  0.0, 1.0, 1.0, -1.0],
                "biases":  [0.0, 0.0, 0.0, 0.0]
            }
        ],
        "output_specs": [
            {"bounds": [0.0, 5.0]},
            {"bounds": [0.0, 7.0], "transform": "round",
             "discrete_set": [0,1,2,3,4,5,6,7], "sentinel": 0.0},
            {},
            {"bounds": [-1.0, 1.0], "sentinel": -1.0}
        ]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert_eq!(model.version(), 3);
    assert!(model.has_output_specs());
    assert_eq!(model.output_specs().len(), 4);
    assert_eq!(model.discrete_sets().len(), 8);
    let mut p = Predictor::new(model);
    // Use 0.4 so `f32::round` (round-half-away-from-zero in Rust)
    // floors it to 0 — proves the round → snap → sentinel chain.
    let r = p.predict_with_specs(&[3.0, 0.4]).unwrap();
    // raw = [3, 0.4, 3.4, 2.6]
    // out0 clamp 3 → 3
    assert_eq!(r[0], OutputValue::Override(3.0));
    // out1 round 0.4 → 0; snap → 0; sentinel 0 → Default
    assert_eq!(r[1], OutputValue::Default);
    // out2 passthrough
    if let OutputValue::Override(v) = r[2] {
        assert!((v - 3.4).abs() < 1e-6);
    } else {
        panic!("expected Override");
    }
    // out3 clamp 2.6 → 1.0
    assert_eq!(r[3], OutputValue::Override(1.0));
}

#[test]
fn json_round_trip_with_sparse_overrides() {
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [
            {
                "in_dim": 2,
                "out_dim": 4,
                "activation": "identity",
                "dtype": "f32",
                "weights": [1.0, 0.0, 1.0, 1.0,  0.0, 1.0, 1.0, -1.0],
                "biases":  [0.0, 0.0, 0.0, 0.0]
            }
        ],
        "sparse_overrides": [
            {"idx": 0, "value": 99.0},
            {"idx": 2, "value": null}
        ]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert_eq!(model.sparse_overrides().len(), 2);
    let mut p = Predictor::new(model);
    let r = p.predict_with_specs(&[3.0, 4.0]).unwrap();
    assert_eq!(r[0], OutputValue::Override(99.0));
    assert_eq!(r[1], OutputValue::Override(4.0));
    assert_eq!(r[2], OutputValue::Default);
    assert_eq!(r[3], OutputValue::Override(-1.0));
}

#[test]
fn json_sigmoid_scaled_transform() {
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "layers": [
            {
                "in_dim": 1,
                "out_dim": 1,
                "activation": "identity",
                "dtype": "f32",
                "weights": [1.0],
                "biases": [0.0]
            }
        ],
        "output_specs": [
            {"transform": "sigmoid_scaled", "params": [0.0, 100.0],
             "bounds": [0.0, 100.0]}
        ]
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(model);
    // raw 0 → sigmoid 0.5 → scaled to 50
    let r = p.predict_with_specs(&[0.0]).unwrap();
    if let OutputValue::Override(v) = r[0] {
        assert!((v - 50.0).abs() < 1e-3);
    } else {
        panic!("expected Override");
    }
}

fn tempdir() -> std::path::PathBuf {
    let base = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let p = base.join(format!("zenpredict-bake-test-{pid}-{nanos}"));
    std::fs::create_dir_all(&p).expect("create tempdir");
    p
}

fn cargo_bin(name: &str) -> std::path::PathBuf {
    // CARGO_BIN_EXE_<name> is set by cargo when running tests against
    // a binary in the same crate. See the Cargo book.
    std::env::var_os(format!("CARGO_BIN_EXE_{name}"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("CARGO_BIN_EXE_{name} not set; run via `cargo test`"))
}
