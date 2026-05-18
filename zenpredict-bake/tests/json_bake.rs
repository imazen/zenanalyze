//! JSON → bake round-trip tests. Exercises the public
//! `BakeRequestJson` schema that `zenpicker/tools/bake_picker.py`
//! emits, then loads the resulting bytes through the runtime to
//! confirm round-trip equivalence.

use zenpredict::keys;
use zenpredict::{MetadataType, Model, OutputValue, Predictor};
use zenpredict_bake::{BakeJsonError, BakeRequestJson, bake_from_json, bake_from_json_str};

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
    let mut p = Predictor::new(&model);
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
    let mut p = Predictor::new(&model);
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
    let mut p = Predictor::new(&model);
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
    let mut p = Predictor::new(&model);
    // raw 0 → sigmoid 0.5 → scaled to 50
    let r = p.predict_with_specs(&[0.0]).unwrap();
    if let OutputValue::Override(v) = r[0] {
        assert!((v - 50.0).abs() < 1e-3);
    } else {
        panic!("expected Override");
    }
}

#[test]
fn json_round_trip_with_multi_codec_schema() {
    // Two codecs, 4 union features, codec 0 owns slots [0, 1, 2], codec
    // 1 owns slots [3, 1] (overlap on slot 1 — they share the same
    // image feature). Trunk inputs = 2*4 + 4 + 2 + 2 = 16. Output dim 4:
    // codec 0 owns [0, 2), codec 1 owns [2, 4).
    let json = r#"{
        "schema_hash": 7777,
        "scaler_mean":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "scaler_scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "layers": [
            {
                "in_dim": 16,
                "out_dim": 4,
                "activation": "identity",
                "dtype": "f32",
                "weights": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0
                ],
                "biases": [0.0, 0.0, 0.0, 0.0]
            }
        ],
        "multi_codec_schema": {
            "union_feat_count": 4,
            "per_codec": [
                {
                    "codec_name": "zenjpeg",
                    "union_slot_for_codec_feat": [0, 1, 2],
                    "output_range": [0, 2],
                    "head_n_cells": 2,
                    "head_n_heads": 1
                },
                {
                    "codec_name": "zenwebp",
                    "union_slot_for_codec_feat": [3, 1],
                    "output_range": [2, 4],
                    "head_n_cells": 2,
                    "head_n_heads": 1
                }
            ]
        }
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();

    // The multi-codec schema must round-trip and the runtime must
    // dispatch through predict_multi_codec.
    assert!(model.has_multi_codec_schema());
    let schema = model.multi_codec_schema().expect("multi-codec schema");
    assert_eq!(schema.union_feat_count, 4);
    assert_eq!(schema.n_codecs, 2);
    assert_eq!(schema.per_codec(0).unwrap().codec_name, "zenjpeg");
    assert_eq!(schema.per_codec(1).unwrap().codec_name, "zenwebp");

    let mut p = Predictor::new(&model);

    // Codec 0 (zenjpeg): feed feature values [10, 20, 30] into the
    // codec's 3 natural feat slots. The identity trunk passes the first
    // 4 union slots straight through; codec 0's output_range is [0, 2),
    // so we should see slots 0 and 1 (= 10, 20) in the returned slice.
    let out0 = p
        .predict_multi_codec(0, &[10.0, 20.0, 30.0], 0, 0.0, 0.0)
        .unwrap();
    assert_eq!(out0, &[10.0, 20.0]);

    // Codec 1 (zenwebp): natural features map to union slots [3, 1].
    // With identity trunk, output_range [2, 4) reads union slots 2 and
    // 3. Codec 1 doesn't write slot 2, so it remains 0.0 from the zero-
    // initialized scatter; slot 3 is the FIRST codec_1 feat (= 100).
    let out1 = p
        .predict_multi_codec(1, &[100.0, 200.0], 1, 1.0, 0.5)
        .unwrap();
    assert_eq!(out1, &[0.0, 100.0]);
}

/// `zerobias_tau` on the JSON layer zeros below-threshold weights
/// before quantization. With `dtype: f32` the biased zeros survive to
/// the wire; loading + predicting reflects the zeroed contribution.
#[test]
fn json_zerobias_tau_zeros_small_weights() {
    // Two-output single-layer net: row-major weights are
    // [w00, w01, w10, w11] = [100.0, 0.01, 50.0, 0.001].
    // Layer max-abs = 100.0; τ=0.005 → cut = 0.5.
    // Zerobiased: [100.0, 0.0, 50.0, 0.0] (the two tiny entries fall).
    let json = r#"{
        "schema_hash": 0,
        "scaler_mean":  [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [{
            "in_dim": 2,
            "out_dim": 2,
            "activation": "identity",
            "dtype": "f32",
            "weights": [100.0, 0.01, 50.0, 0.001],
            "biases":  [0.0, 0.0]
        }],
        "zerobias_tau": 0.005
    }"#;
    let bytes = bake_from_json_str(json).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(&model);
    // With biased weights [100, 0, 50, 0], predict([1, 1]) =
    //   [1*100 + 1*50, 1*0 + 1*0] = [150, 0].
    let out = p.predict(&[1.0, 1.0]).unwrap();
    assert_eq!(out, &[150.0, 0.0]);
}

/// `zerobias_tau: 0.0` (default) is a no-op — the bake matches the
/// pre-knob behavior byte-for-byte.
#[test]
fn json_zerobias_default_is_noop() {
    let base = r#"{
        "schema_hash": 0,
        "scaler_mean":  [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [{
            "in_dim": 2,
            "out_dim": 2,
            "activation": "identity",
            "dtype": "f32",
            "weights": [1.0, 0.0, 0.0, 1.0],
            "biases":  [0.0, 0.0]
        }]
    }"#;
    let with_default = r#"{
        "schema_hash": 0,
        "scaler_mean":  [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "layers": [{
            "in_dim": 2,
            "out_dim": 2,
            "activation": "identity",
            "dtype": "f32",
            "weights": [1.0, 0.0, 0.0, 1.0],
            "biases":  [0.0, 0.0]
        }],
        "zerobias_tau": 0.0,
        "compressed": false,
        "optimize": false
    }"#;
    let a = bake_from_json_str(base).unwrap();
    let b = bake_from_json_str(with_default).unwrap();
    assert_eq!(a, b, "default knob values must produce identical bytes");
}

/// `compressed: true` on the JSON layer produces a bake with the
/// compressed flag set; the loader transparently decompresses.
#[test]
fn json_compressed_roundtrip() {
    // Larger zero-rich payload so LZ4 actually shrinks it.
    let mut weights = vec![0.0_f32; 256 * 32];
    weights[0] = 1.0;
    weights[31] = 2.0;
    let weights_json = serde_json::to_string(&weights).unwrap();
    let biases_json = serde_json::to_string(&vec![0.0_f32; 32]).unwrap();
    let scaler_mean = serde_json::to_string(&vec![0.0_f32; 256]).unwrap();
    let scaler_scale = serde_json::to_string(&vec![1.0_f32; 256]).unwrap();

    let json_uncompressed = format!(
        "{{ \"schema_hash\": 0, \"scaler_mean\": {scaler_mean}, \"scaler_scale\": {scaler_scale}, \
         \"layers\": [{{ \"in_dim\": 256, \"out_dim\": 32, \"activation\": \"identity\", \
         \"dtype\": \"f32\", \"weights\": {weights_json}, \"biases\": {biases_json} }}] }}"
    );
    let json_compressed = format!(
        "{{ \"schema_hash\": 0, \"scaler_mean\": {scaler_mean}, \"scaler_scale\": {scaler_scale}, \
         \"layers\": [{{ \"in_dim\": 256, \"out_dim\": 32, \"activation\": \"identity\", \
         \"dtype\": \"f32\", \"weights\": {weights_json}, \"biases\": {biases_json} }}], \
         \"compressed\": true }}"
    );
    let uncompressed = bake_from_json_str(&json_uncompressed).unwrap();
    let compressed = bake_from_json_str(&json_compressed).unwrap();
    assert!(
        compressed.len() < uncompressed.len(),
        "compressed bake ({}) should be smaller than uncompressed ({})",
        compressed.len(),
        uncompressed.len(),
    );
    // Round-trip the compressed bytes through the loader.
    let aligned = Aligned(compressed);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(&model);
    let mut x = vec![0.0_f32; 256];
    x[0] = 1.0;
    x[31] = 1.0;
    let out = p.predict(&x).unwrap();
    // weights[0] = 1 maps input[0] → output[0]; weights[31] = 2 means
    // row=0, col=31 (input 0 contributes 2 to output 31). With input
    // x[31]=1, w[31*32+31]=0 → no contribution. So out[0]=1, others=0.
    assert_eq!(out[0], 1.0);
}

/// `optimize: true` invokes `bake_optimized` and produces a bake that
/// is no larger than the un-optimized version. On a zero-heavy payload
/// the optimizer typically produces a strictly smaller bake.
#[test]
fn json_optimize_roundtrip_smaller_or_equal() {
    let mut weights = vec![0.0_f32; 64 * 16];
    weights[0] = 1.0;
    weights[1] = 2.0;
    let weights_json = serde_json::to_string(&weights).unwrap();
    let biases_json = serde_json::to_string(&vec![0.0_f32; 16]).unwrap();
    let scaler_mean = serde_json::to_string(&vec![0.0_f32; 64]).unwrap();
    let scaler_scale = serde_json::to_string(&vec![1.0_f32; 64]).unwrap();

    let json_plain = format!(
        "{{ \"schema_hash\": 0, \"scaler_mean\": {scaler_mean}, \"scaler_scale\": {scaler_scale}, \
         \"layers\": [{{ \"in_dim\": 64, \"out_dim\": 16, \"activation\": \"identity\", \
         \"dtype\": \"f32\", \"weights\": {weights_json}, \"biases\": {biases_json} }}] }}"
    );
    let json_opt = format!(
        "{{ \"schema_hash\": 0, \"scaler_mean\": {scaler_mean}, \"scaler_scale\": {scaler_scale}, \
         \"layers\": [{{ \"in_dim\": 64, \"out_dim\": 16, \"activation\": \"identity\", \
         \"dtype\": \"f32\", \"weights\": {weights_json}, \"biases\": {biases_json} }}], \
         \"optimize\": true }}"
    );
    let plain = bake_from_json_str(&json_plain).unwrap();
    let optimized = bake_from_json_str(&json_opt).unwrap();
    assert!(
        optimized.len() <= plain.len(),
        "optimized bake ({}) must not exceed plain bake ({})",
        optimized.len(),
        plain.len(),
    );
    // Round-trip the optimized bytes — predict output must be lossless.
    let aligned = Aligned(optimized);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(&model);
    let mut x = vec![0.0_f32; 64];
    x[0] = 1.0;
    let out = p.predict(&x).unwrap();
    // weights[0] = 1, weights[1] = 2 — these are row=0, col=0 / col=1.
    // input[0]=1 → output[0] += 1, output[1] += 2. All other rows are
    // zero.
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], 2.0);
    for v in &out[2..] {
        assert_eq!(*v, 0.0);
    }
}

/// All three knobs composed: zerobias 0.005 + compressed + optimize.
/// Verifies the pipeline works end-to-end on a realistic shape and
/// produces a strictly smaller bake than the unoptimized baseline.
#[test]
fn json_zerobias_compressed_optimize_composed() {
    // 64 → 32 → 1 net with mostly-small weights — zerobias should
    // zero ~88 % at τ=0.005 (per the V0_18 eval), compression then
    // collapses the zero runs.
    let mut layer0_w = Vec::with_capacity(64 * 32);
    for i in 0..64 * 32 {
        layer0_w.push(if i % 16 == 0 { 1.0_f32 } else { 1e-6_f32 });
    }
    let layer0_b = vec![0.0_f32; 32];
    let mut layer1_w = vec![0.0_f32; 32];
    layer1_w[0] = 2.0;
    let layer1_b = vec![1.0_f32];

    let layer0_w_json = serde_json::to_string(&layer0_w).unwrap();
    let layer0_b_json = serde_json::to_string(&layer0_b).unwrap();
    let layer1_w_json = serde_json::to_string(&layer1_w).unwrap();
    let layer1_b_json = serde_json::to_string(&layer1_b).unwrap();
    let scaler_mean = serde_json::to_string(&vec![0.0_f32; 64]).unwrap();
    let scaler_scale = serde_json::to_string(&vec![1.0_f32; 64]).unwrap();

    let layers = format!(
        "[\
         {{ \"in_dim\": 64, \"out_dim\": 32, \"activation\": \"identity\", \
            \"dtype\": \"f32\", \"weights\": {layer0_w_json}, \"biases\": {layer0_b_json} }},\
         {{ \"in_dim\": 32, \"out_dim\": 1, \"activation\": \"identity\", \
            \"dtype\": \"f32\", \"weights\": {layer1_w_json}, \"biases\": {layer1_b_json} }}\
         ]"
    );

    let json_plain = format!(
        "{{ \"schema_hash\": 0, \"scaler_mean\": {scaler_mean}, \"scaler_scale\": {scaler_scale}, \
         \"layers\": {layers} }}"
    );
    let json_all = format!(
        "{{ \"schema_hash\": 0, \"scaler_mean\": {scaler_mean}, \"scaler_scale\": {scaler_scale}, \
         \"layers\": {layers}, \
         \"zerobias_tau\": 0.005, \"compressed\": true, \"optimize\": true }}"
    );

    let plain = bake_from_json_str(&json_plain).unwrap();
    let all = bake_from_json_str(&json_all).unwrap();
    assert!(
        all.len() < plain.len(),
        "composed knobs ({}) should shrink vs plain ({})",
        all.len(),
        plain.len(),
    );

    // Load the all-knobs bake and confirm predict still works.
    let aligned = Aligned(all);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(&model);
    let mut x = vec![0.0_f32; 64];
    x[0] = 1.0;
    // Forward pass (post-zerobias, weights ≤ 1e-6 are 0):
    //   layer0 out[0] = 1.0 * w[0]  = 1.0; all other out[j] = 0
    //   layer1 out    = 2 * 1.0 + 1 = 3.0
    let out = p.predict(&x).unwrap();
    assert_eq!(out, &[3.0]);
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
