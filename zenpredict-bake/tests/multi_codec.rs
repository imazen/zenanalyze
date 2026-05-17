//! Multi-codec joint-picker bake + predict round-trip tests.
//!
//! Exercise the ZNPR v3.2 `multi_codec_schema` section: build a
//! small synthetic shared-trunk bake (2 codecs, 4 union features,
//! 3 per-codec features each, 2 cells per codec), parse it, run
//! [`zenpredict::Predictor::predict_multi_codec`] against expected
//! output ranges, and verify backward-compatibility with
//! single-codec bakes.

use zenpredict::{
    Activation, Model, MultiCodecSchema, PerCodecMap, PredictError, Predictor, WeightDtype,
};
use zenpredict_bake::{BakeLayer, BakeRequest, MultiCodecSchemaInput, PerCodecMapInput, bake};

/// Wrapper that guarantees 16-byte alignment of an in-memory model blob —
/// what `include_bytes!` consumers do via `#[repr(C, align(16))]`.
#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// Synthetic multi-codec setup:
/// - `union_feat_count = 4`
/// - `n_codecs = 2` (codec 0 has slots [0,1,2], codec 1 has slots [1,2,3])
/// - 2 cells per codec → trunk n_outputs = 4
/// - trunk n_inputs = 2*4 + 6 + 2 = 16
const UNION: u32 = 4;
const N_CODECS: u32 = 2;
const N_CELLS_PER_CODEC: u32 = 2;
const N_INPUTS: usize = 16;
const N_OUTPUTS: usize = 4;

fn codec0_slots() -> [u32; 3] {
    [0, 1, 2]
}

fn codec1_slots() -> [u32; 3] {
    [1, 2, 3]
}

fn make_multi_codec_bake(custom_weights: Option<[f32; N_INPUTS * N_OUTPUTS]>) -> Vec<u8> {
    // Identity-like trunk: a single linear layer with weights chosen
    // so we can derive ground-truth outputs by hand.
    //
    // For each output `o` (cell), bias[o] = 100*o, and weights are
    // 1.0 at one selected input slot and 0.0 elsewhere — picking the
    // input slot for each output cell makes the round-trip easy to
    // verify.
    let weights = custom_weights.unwrap_or_else(|| {
        let mut w = [0.0f32; N_INPUTS * N_OUTPUTS];
        // weight[i * N_OUTPUTS + o]: input i contributes to output o.
        // Output 0 reads codec0_feat[0] (= union[0])  → w[0*4 + 0] = 1.0
        // Output 1 reads codec0_feat[1] (= union[1])  → w[1*4 + 1] = 1.0
        // Output 2 reads codec1_feat[0] (= union[1])  → w[1*4 + 2] = 1.0
        //   Note: codec1's first natural feat ALSO lands at union[1].
        // Output 3 reads codec1_feat[2] (= union[3])  → w[3*4 + 3] = 1.0
        // Index pattern: w[input_idx * N_OUTPUTS + output_idx] = 1.0
        // Explicit numbers (clippy's identity_op/erasing_op object
        // to `0 * 4 + 0`); equivalent layout, easier to read.
        w[0] = 1.0; // input 0 → out 0
        w[5] = 1.0; // input 1 → out 1   (1*4 + 1)
        w[6] = 1.0; // input 1 → out 2   (1*4 + 2)
        w[15] = 1.0; // input 3 → out 3  (3*4 + 3)
        w
    });
    let biases = [10.0f32, 20.0, 30.0, 40.0];
    let layer = BakeLayer {
        in_dim: N_INPUTS,
        out_dim: N_OUTPUTS,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    };
    let layers = [layer];

    let codec0_slots = codec0_slots();
    let codec1_slots = codec1_slots();
    let per_codec = [
        PerCodecMapInput {
            codec_name: "zenjpeg",
            union_slot_for_codec_feat: &codec0_slots,
            output_range: (0, N_CELLS_PER_CODEC),
            head_n_cells: N_CELLS_PER_CODEC,
            head_n_heads: 1,
        },
        PerCodecMapInput {
            codec_name: "zenwebp",
            union_slot_for_codec_feat: &codec1_slots,
            output_range: (N_CELLS_PER_CODEC, 2 * N_CELLS_PER_CODEC),
            head_n_cells: N_CELLS_PER_CODEC,
            head_n_heads: 1,
        },
    ];
    let schema = MultiCodecSchemaInput {
        union_feat_count: UNION,
        per_codec: &per_codec,
    };

    let scaler_mean = [0.0f32; N_INPUTS];
    let scaler_scale = [1.0f32; N_INPUTS];

    let req = BakeRequest {
        schema_hash: 0xfeedface_deadbeef,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
        multi_codec_schema: Some(schema),
    };
    bake(&req).unwrap()
}

#[test]
fn schema_round_trips_unchanged() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();

    assert_eq!(model.n_inputs(), N_INPUTS);
    assert_eq!(model.n_outputs(), N_OUTPUTS);
    assert!(model.has_multi_codec_schema());

    let schema: MultiCodecSchema<'_> = model.multi_codec_schema().unwrap();
    assert_eq!(schema.union_feat_count, UNION);
    assert_eq!(schema.n_codecs, N_CODECS);
    assert_eq!(schema.per_codec.len(), 2);

    let m0: &PerCodecMap<'_> = &schema.per_codec[0];
    assert_eq!(m0.codec_name, "zenjpeg");
    assert_eq!(m0.union_slot_for_codec_feat, codec0_slots());
    assert_eq!(m0.output_range, (0, 2));
    assert_eq!(m0.head_meta.n_cells, 2);
    assert_eq!(m0.head_meta.n_heads, 1);

    let m1 = &schema.per_codec[1];
    assert_eq!(m1.codec_name, "zenwebp");
    assert_eq!(m1.union_slot_for_codec_feat, codec1_slots());
    assert_eq!(m1.output_range, (2, 4));
}

#[test]
fn predict_codec_0_returns_first_two_outputs() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);

    // codec 0: features [7.0, 11.0, 13.0] scatter to union [0,1,2].
    // After scatter, union = [7.0, 11.0, 13.0, 0.0]; presence at
    // [0,1,2] = 1.0, [3] = 0.0; size onehot small (idx=1) → onehot
    // = [0,1,0,0]; log_pixels = 5.0; zq_norm = 0.3; codec onehot
    // [1,0].
    //
    // Output 0: bias 10 + weight[0*4+0]*union[0] = 10 + 7 = 17
    // Output 1: bias 20 + weight[1*4+1]*union[1] = 20 + 11 = 31
    // (outputs 2,3 are computed too but not returned for codec 0)
    let codec_features = [7.0f32, 11.0, 13.0];
    let out = predictor
        .predict_multi_codec(0, &codec_features, 1, 5.0, 0.3)
        .unwrap();
    assert_eq!(out.len(), 2, "codec 0's output_range is (0, 2)");
    assert!(
        (out[0] - 17.0).abs() < 1e-4,
        "expected 17.0 (10+7), got {}",
        out[0]
    );
    assert!(
        (out[1] - 31.0).abs() < 1e-4,
        "expected 31.0 (20+11), got {}",
        out[1]
    );
}

#[test]
fn predict_codec_1_returns_last_two_outputs() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);

    // codec 1: features [101.0, 102.0, 103.0] scatter to union [1,2,3].
    // After scatter, union = [0.0, 101.0, 102.0, 103.0]; presence at
    // [1,2,3] = 1.0, [0] = 0.0; size onehot tiny (idx=0) → [1,0,0,0].
    //
    // Output 2: bias 30 + weight[1*4+2]*union[1] = 30 + 101 = 131
    // Output 3: bias 40 + weight[3*4+3]*union[3] = 40 + 103 = 143
    let codec_features = [101.0f32, 102.0, 103.0];
    let out = predictor
        .predict_multi_codec(1, &codec_features, 0, 4.0, 0.5)
        .unwrap();
    assert_eq!(out.len(), 2, "codec 1's output_range is (2, 4)");
    assert!(
        (out[0] - 131.0).abs() < 1e-4,
        "expected 131.0 (30+101), got {}",
        out[0]
    );
    assert!(
        (out[1] - 143.0).abs() < 1e-4,
        "expected 143.0 (40+103), got {}",
        out[1]
    );
}

#[test]
fn predicts_are_deterministic_across_calls() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);

    // Same codec_id called twice should yield identical output.
    let cf = [7.0f32, 11.0, 13.0];
    let first: Vec<f32> = predictor
        .predict_multi_codec(0, &cf, 1, 5.0, 0.3)
        .unwrap()
        .to_vec();
    // Call a different codec in between to dirty the scratch.
    let _ = predictor
        .predict_multi_codec(1, &[1.0, 2.0, 3.0], 0, 4.0, 0.5)
        .unwrap();
    let second: Vec<f32> = predictor
        .predict_multi_codec(0, &cf, 1, 5.0, 0.3)
        .unwrap()
        .to_vec();
    assert_eq!(
        first, second,
        "predict_multi_codec must be deterministic — codec 0 first call vs after dirtying"
    );
}

#[test]
fn single_codec_bake_still_loads_and_predicts() {
    // No `multi_codec_schema` field set — backward compatibility.
    let scaler_mean = [0.0f32, 0.0, 0.0];
    let scaler_scale = [1.0f32, 1.0, 1.0];
    let weights = [
        1.0f32, 0.0, // input 0 → outs
        0.0, 1.0, // input 1 → outs
        0.0, 0.0, // input 2 → outs
    ];
    let biases = [5.0f32, 7.0];
    let layer = BakeLayer {
        in_dim: 3,
        out_dim: 2,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    };
    let layers = [layer];
    let req = BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
        multi_codec_schema: None,
    };
    let bytes = bake(&req).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert!(
        !model.has_multi_codec_schema(),
        "single-codec bake should not carry a schema"
    );
    assert!(model.multi_codec_schema().is_none());

    let mut predictor = Predictor::new(&model);
    let out = predictor.predict(&[2.0, 3.0, 99.0]).unwrap();
    // Output 0: 5 + 1*2 = 7. Output 1: 7 + 1*3 = 10.
    assert!((out[0] - 7.0).abs() < 1e-4);
    assert!((out[1] - 10.0).abs() < 1e-4);
}

#[test]
fn predict_multi_codec_on_single_codec_bake_errors() {
    let scaler_mean = [0.0f32];
    let scaler_scale = [1.0f32];
    let weights = [1.0f32];
    let biases = [0.0f32];
    let layer = BakeLayer {
        in_dim: 1,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    };
    let layers = [layer];
    let req = BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
        multi_codec_schema: None,
    };
    let bytes = bake(&req).unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);

    let err = predictor
        .predict_multi_codec(0, &[1.0], 0, 0.0, 0.0)
        .unwrap_err();
    assert!(
        matches!(err, PredictError::MultiCodecNotSupported),
        "expected MultiCodecNotSupported, got {err:?}"
    );
}

#[test]
fn unknown_codec_id_errors() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);

    // n_codecs = 2 → codec_id 2 (and beyond) is invalid.
    let err = predictor
        .predict_multi_codec(2, &[1.0, 2.0, 3.0], 0, 0.0, 0.0)
        .unwrap_err();
    match err {
        PredictError::UnknownCodecId { codec_id, n_codecs } => {
            assert_eq!(codec_id, 2);
            assert_eq!(n_codecs, N_CODECS);
        }
        other => panic!("expected UnknownCodecId, got {other:?}"),
    }
}

#[test]
fn codec_feature_len_mismatch_errors() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut predictor = Predictor::new(&model);

    // codec 0 expects 3 features; pass 4 (wrong).
    let err = predictor
        .predict_multi_codec(0, &[1.0, 2.0, 3.0, 4.0], 0, 0.0, 0.0)
        .unwrap_err();
    match err {
        PredictError::CodecFeatureLenMismatch {
            codec_id,
            expected,
            got,
        } => {
            assert_eq!(codec_id, 0);
            assert_eq!(expected, 3);
            assert_eq!(got, 4);
        }
        other => panic!("expected CodecFeatureLenMismatch, got {other:?}"),
    }
}

#[test]
fn bake_rejects_input_dim_mismatch() {
    // Build a bake whose layer[0].in_dim doesn't match the
    // multi-codec schema's expected `2*U + 6 + C`. The composer
    // should reject this rather than emit a bake the loader would
    // refuse.
    let scaler_mean = [0.0f32; 8];
    let scaler_scale = [1.0f32; 8];
    // Wrong: trunk n_inputs = 8, but schema expects 2*4 + 6 + 2 = 16.
    let weights = [0.0f32; 8 * 4];
    let biases = [0.0f32; 4];
    let layer = BakeLayer {
        in_dim: 8,
        out_dim: 4,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    };
    let layers = [layer];

    let c0 = [0u32, 1, 2];
    let c1 = [1u32, 2, 3];
    let per_codec = [
        PerCodecMapInput {
            codec_name: "zenjpeg",
            union_slot_for_codec_feat: &c0,
            output_range: (0, 2),
            head_n_cells: 2,
            head_n_heads: 1,
        },
        PerCodecMapInput {
            codec_name: "zenwebp",
            union_slot_for_codec_feat: &c1,
            output_range: (2, 4),
            head_n_cells: 2,
            head_n_heads: 1,
        },
    ];
    let schema = MultiCodecSchemaInput {
        union_feat_count: 4,
        per_codec: &per_codec,
    };
    let req = BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
        multi_codec_schema: Some(schema),
    };
    let err = bake(&req).unwrap_err();
    use zenpredict_bake::BakeError;
    match err {
        BakeError::MultiCodecInputDimMismatch { expected, got } => {
            assert_eq!(expected, 16);
            assert_eq!(got, 8);
        }
        other => panic!("expected MultiCodecInputDimMismatch, got {other:?}"),
    }
}

#[test]
fn codec_names_in_schema() {
    let bytes = make_multi_codec_bake(None);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let schema = model.multi_codec_schema().unwrap();
    let names = schema.codec_names();
    assert_eq!(names, vec!["zenjpeg", "zenwebp"]);
}
