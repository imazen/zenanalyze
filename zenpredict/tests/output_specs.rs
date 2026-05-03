//! Integration tests for ZNPR v3 output specs + sparse overrides.

use zenpredict::bake::{BakeLayer, BakeRequest, bake_v2};
use zenpredict::{
    Activation, FORMAT_VERSION, FeatureBound, Model, OutputSpec, OutputTransform, OutputValue,
    Predictor, SparseOverride, WeightDtype,
};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// Build a tiny identity-ish model: 2 inputs → 4 outputs. Forward
/// pass yields `[features[0], features[1], features[0]+features[1],
/// features[0]-features[1]]` plus a per-output bias of 0.
fn build_identity_model() -> Vec<u8> {
    let scaler_mean = [0.0f32, 0.0];
    let scaler_scale = [1.0f32, 1.0];
    // weights row-major: weights[i, o] is the weight from input i to
    // output o.  in_dim=2, out_dim=4, so 8 weights:
    //   out0: [1, 0]      (= input0)
    //   out1: [0, 1]      (= input1)
    //   out2: [1, 1]      (= input0 + input1)
    //   out3: [1, -1]     (= input0 - input1)
    let weights = [
        // input0 contributions to out0..out3
        1.0f32, 0.0, 1.0, 1.0, // input0
        0.0, 1.0, 1.0, -1.0, // input1
    ];
    let biases = [0.0f32, 0.0, 0.0, 0.0];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    bake_v2(&BakeRequest {
        schema_hash: 0xfeed,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
    })
    .unwrap()
}

#[test]
fn version_is_v3() {
    assert_eq!(FORMAT_VERSION, 3);
    let bytes = Aligned(build_identity_model());
    let model = Model::from_bytes(&bytes.0).unwrap();
    assert_eq!(model.version(), 3);
    assert!(!model.has_output_specs());
    assert!(model.output_specs().is_empty());
    assert!(model.discrete_sets().is_empty());
    assert!(model.sparse_overrides().is_empty());
}

#[test]
fn predict_with_specs_passthrough_when_no_specs() {
    // A bake without an output_specs section should make
    // predict_with_specs return Override(raw[i]) for every output.
    let bytes = Aligned(build_identity_model());
    let model = Model::from_bytes(&bytes.0).unwrap();
    let mut p = Predictor::new(model);
    let raw = p.predict(&[3.0, 4.0]).unwrap().to_vec();
    assert_eq!(raw, vec![3.0, 4.0, 7.0, -1.0]);
    let with_specs = p.predict_with_specs(&[3.0, 4.0]).unwrap();
    assert_eq!(
        with_specs,
        &[
            OutputValue::Override(3.0),
            OutputValue::Override(4.0),
            OutputValue::Override(7.0),
            OutputValue::Override(-1.0),
        ]
    );
}

fn build_full_spec_model(
    specs: &[OutputSpec],
    discrete_sets: &[f32],
    overrides: &[SparseOverride],
) -> Vec<u8> {
    let scaler_mean = [0.0f32, 0.0];
    let scaler_scale = [1.0f32, 1.0];
    let weights = [
        1.0f32, 0.0, 1.0, 1.0, // input0
        0.0, 1.0, 1.0, -1.0, // input1
    ];
    let biases = [0.0f32, 0.0, 0.0, 0.0];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    bake_v2(&BakeRequest {
        schema_hash: 0xfeed,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: specs,
        discrete_sets,
        sparse_overrides: overrides,
    })
    .unwrap()
}

#[test]
fn full_pipeline_round_trip() {
    // Output 0: clamp into [0, 5]
    // Output 1: round-to-int, snap to {0..7}, sentinel = 0
    // Output 2: identity passthrough
    // Output 3: clamp into [-1, 1] with sentinel = -1
    let specs = [
        OutputSpec {
            bounds: FeatureBound::new(0.0, 5.0),
            transform: OutputTransform::Identity as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 0,
            sentinel: f32::NAN,
        },
        OutputSpec {
            bounds: FeatureBound::new(0.0, 7.0),
            transform: OutputTransform::Round as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 8,
            sentinel: 0.0,
        },
        OutputSpec::passthrough(),
        OutputSpec {
            bounds: FeatureBound::new(-1.0, 1.0),
            transform: OutputTransform::Identity as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 0,
            sentinel: -1.0,
        },
    ];
    let pool = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let bytes = Aligned(build_full_spec_model(&specs, &pool, &[]));
    let model = Model::from_bytes(&bytes.0).unwrap();
    assert!(model.has_output_specs());
    assert_eq!(model.output_specs().len(), 4);
    assert_eq!(model.discrete_sets(), &pool);
    let mut p = Predictor::new(model);

    // features = [3.0, 0.6]
    // raw = [3.0, 0.6, 3.6, 2.4]
    let r = p.predict_with_specs(&[3.0, 0.6]).unwrap();
    assert_eq!(r.len(), 4);
    // out0: clamp 3.0 → 3.0
    assert_eq!(r[0], OutputValue::Override(3.0));
    // out1: round 0.6 → 1.0; clamp to [0,7] → 1.0; snap → 1.0; sentinel 0 ≠ 1
    assert_eq!(r[1], OutputValue::Override(1.0));
    // out2: passthrough
    assert!(matches!(r[2], OutputValue::Override(v) if (v - 3.6).abs() < 1e-6));
    // out3: clamp 2.4 → 1.0
    assert_eq!(r[3], OutputValue::Override(1.0));

    // features = [-2, -2] → raw = [-2, -2, -4, 0]
    let r = p.predict_with_specs(&[-2.0, -2.0]).unwrap();
    // out0: clamp -2 → 0
    assert_eq!(r[0], OutputValue::Override(0.0));
    // out1: round -2 → -2; clamp to [0,7] → 0; snap → 0; sentinel 0 == 0 → Default
    assert_eq!(r[1], OutputValue::Default);
    // out2: passthrough -4
    assert_eq!(r[2], OutputValue::Override(-4.0));
    // out3: clamp 0 → 0
    assert_eq!(r[3], OutputValue::Override(0.0));

    // features = [-1, 0] → raw = [-1, 0, -1, -1]
    let r = p.predict_with_specs(&[-1.0, 0.0]).unwrap();
    // out3: clamp -1 → -1; sentinel -1 → Default
    assert_eq!(r[3], OutputValue::Default);
}

#[test]
fn sparse_overrides_apply_after_pipeline() {
    let specs = [OutputSpec::passthrough(); 4];
    let overrides = [
        SparseOverride::new(0, 99.0),
        SparseOverride::new(2, f32::NAN), // forces Default
    ];
    let bytes = Aligned(build_full_spec_model(&specs, &[], &overrides));
    let model = Model::from_bytes(&bytes.0).unwrap();
    let mut p = Predictor::new(model);
    let r = p.predict_with_specs(&[3.0, 4.0]).unwrap().to_vec();
    // out0: forward yields 3, override → 99
    assert_eq!(r[0], OutputValue::Override(99.0));
    // out1: forward yields 4, no override
    assert_eq!(r[1], OutputValue::Override(4.0));
    // out2: forward yields 7, override is NaN → Default
    assert_eq!(r[2], OutputValue::Default);
    // out3: forward yields -1, no override
    assert_eq!(r[3], OutputValue::Override(-1.0));
}

#[test]
fn sparse_override_overrules_sentinel() {
    // A sentinel hit produces Default; a sparse override with a
    // concrete value pulls it back to Override.
    let specs = [
        OutputSpec {
            bounds: FeatureBound::new(-10.0, 10.0),
            transform: OutputTransform::Identity as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 0,
            sentinel: 3.0,
        },
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
    ];
    let overrides = [SparseOverride::new(0, 42.0)];
    let bytes = Aligned(build_full_spec_model(&specs, &[], &overrides));
    let model = Model::from_bytes(&bytes.0).unwrap();
    let mut p = Predictor::new(model);
    // raw out0 = 3.0; sentinel matches → Default; override → 42.0
    let r = p.predict_with_specs(&[3.0, 0.0]).unwrap();
    assert_eq!(r[0], OutputValue::Override(42.0));
}

#[test]
fn discrete_snap_at_midpoint() {
    let specs = [
        OutputSpec {
            bounds: FeatureBound::new(0.0, 100.0),
            transform: OutputTransform::Identity as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 3,
            sentinel: f32::NAN,
        },
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
    ];
    let pool = [0.0f32, 50.0, 100.0];
    let bytes = Aligned(build_full_spec_model(&specs, &pool, &[]));
    let model = Model::from_bytes(&bytes.0).unwrap();
    let mut p = Predictor::new(model);
    // raw out0 = 24, closer to 0 than 50
    let r = p.predict_with_specs(&[24.0, 0.0]).unwrap();
    assert_eq!(r[0], OutputValue::Override(0.0));
    // raw out0 = 26, closer to 50
    let r = p.predict_with_specs(&[26.0, 0.0]).unwrap();
    assert_eq!(r[0], OutputValue::Override(50.0));
}

#[test]
fn unknown_output_transform_byte_rejected_at_bake() {
    // Build a spec with a transform byte 0x99 — not a valid variant.
    let bad_spec = OutputSpec {
        bounds: FeatureBound::new(-1.0, 1.0),
        transform: 0x99,
        _pad: [0; 3],
        transform_params: [0.0, 0.0],
        discrete_set_offset: 0,
        discrete_set_len: 0,
        sentinel: f32::NAN,
    };
    let specs = [
        bad_spec,
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
    ];
    let scaler_mean = [0.0f32, 0.0];
    let scaler_scale = [1.0f32, 1.0];
    let weights = [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0];
    let biases = [0.0f32, 0.0, 0.0, 0.0];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    let err = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &specs,
        discrete_sets: &[],
        sparse_overrides: &[],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        zenpredict::bake::BakeError::UnknownOutputTransform { .. }
    ));
}

#[test]
fn output_specs_length_mismatch_rejected_at_bake() {
    // n_outputs=4 but only 2 specs.
    let specs = [OutputSpec::passthrough(), OutputSpec::passthrough()];
    let scaler_mean = [0.0f32, 0.0];
    let scaler_scale = [1.0f32, 1.0];
    let weights = [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0];
    let biases = [0.0f32, 0.0, 0.0, 0.0];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    let err = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &specs,
        discrete_sets: &[],
        sparse_overrides: &[],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        zenpredict::bake::BakeError::OutputSpecsLengthMismatch {
            expected: 4,
            got: 2
        }
    ));
}

#[test]
fn discrete_set_out_of_range_rejected_at_bake() {
    let specs = [
        OutputSpec {
            bounds: FeatureBound::new(0.0, 10.0),
            transform: OutputTransform::Identity as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 5,
            discrete_set_len: 100, // pool only has 3
            sentinel: f32::NAN,
        },
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
    ];
    let pool = [0.0f32, 1.0, 2.0];
    let scaler_mean = [0.0f32, 0.0];
    let scaler_scale = [1.0f32, 1.0];
    let weights = [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0];
    let biases = [0.0f32, 0.0, 0.0, 0.0];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    let err = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &specs,
        discrete_sets: &pool,
        sparse_overrides: &[],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        zenpredict::bake::BakeError::OutputSpecDiscreteOutOfRange { .. }
    ));
}

#[test]
fn sparse_override_index_out_of_range_rejected_at_bake() {
    let overrides = [SparseOverride::new(99, 1.0)];
    let scaler_mean = [0.0f32, 0.0];
    let scaler_scale = [1.0f32, 1.0];
    let weights = [1.0f32, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0];
    let biases = [0.0f32, 0.0, 0.0, 0.0];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    let err = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &overrides,
    })
    .unwrap_err();
    assert!(matches!(
        err,
        zenpredict::bake::BakeError::SparseOverrideIndexOutOfRange { idx: 99, .. }
    ));
}

#[test]
fn v2_bin_rejected_with_unsupported_version() {
    // Build a v3 bin, mutate the version bytes back to 2, expect
    // Model::from_bytes to fail with UnsupportedVersion.
    let mut bytes = build_identity_model();
    bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
    let aligned = Aligned(bytes);
    let err = Model::from_bytes(&aligned.0).unwrap_err();
    assert!(matches!(
        err,
        zenpredict::PredictError::UnsupportedVersion {
            version: 2,
            expected: 3
        }
    ));
}

#[test]
fn raw_predict_unchanged_by_specs() {
    // Specs only matter for predict_with_specs; raw predict() must
    // continue to return the unprocessed forward-pass output.
    let specs = [
        OutputSpec {
            bounds: FeatureBound::new(0.0, 1.0),
            transform: OutputTransform::Sigmoid as u8,
            _pad: [0; 3],
            transform_params: [0.0, 0.0],
            discrete_set_offset: 0,
            discrete_set_len: 0,
            sentinel: f32::NAN,
        },
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
        OutputSpec::passthrough(),
    ];
    let bytes = Aligned(build_full_spec_model(&specs, &[], &[]));
    let model = Model::from_bytes(&bytes.0).unwrap();
    let mut p = Predictor::new(model);
    let raw = p.predict(&[3.0, 4.0]).unwrap().to_vec();
    // raw forward should be untransformed; sigmoid would have squashed
    // the 3.0 to ~0.95.
    assert_eq!(raw, vec![3.0, 4.0, 7.0, -1.0]);
}

#[test]
fn nan_sparse_override_forces_default_even_when_no_specs() {
    // Bake without output_specs but with a NaN sparse override on
    // output 1.
    let overrides = [SparseOverride::new(1, f32::NAN)];
    let bytes = Aligned(build_full_spec_model(&[], &[], &overrides));
    let model = Model::from_bytes(&bytes.0).unwrap();
    let mut p = Predictor::new(model);
    let r = p.predict_with_specs(&[3.0, 4.0]).unwrap();
    assert_eq!(r[0], OutputValue::Override(3.0));
    assert_eq!(r[1], OutputValue::Default);
    assert_eq!(r[2], OutputValue::Override(7.0));
    assert_eq!(r[3], OutputValue::Override(-1.0));
}
