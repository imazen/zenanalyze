//! Integration test exercising the full bake → load → predict → argmin
//! lifecycle from a downstream consumer's perspective. Uses only the
//! public crate API; fails to compile if a public item gets renamed
//! or hidden behind a feature flag without a migration plan.

#![cfg(feature = "bake")]

use zenpredict::bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake_v2};
use zenpredict::keys;
use zenpredict::{
    Activation, AllowedMask, ArgminOffsets, FeatureBound, Metadata, MetadataType, Model, Predictor,
    RescueDecision, RescuePolicy, RescueStrategy, ScoreTransform, WeightDtype,
    first_out_of_distribution, should_rescue, threshold_mask,
};

/// Aligns an in-memory blob to 16 bytes — what `include_bytes!`
/// consumers do via `#[repr(C, align(16))]` wrapping.
#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// Bake a small "codec picker" — 5 features, 12 categorical cells
/// (cell-bytes head, no scalar heads to keep the test focused), with
/// metadata mimicking what a real bake would carry.
fn bake_codec_picker() -> Vec<u8> {
    let n_inputs = 5;
    let n_hidden = 32;
    let n_outputs = 12;

    // Diagonal-ish weights so we get distinct outputs from distinct
    // inputs — no degenerate ties.
    let scaler_mean = vec![0.0f32; n_inputs];
    let scaler_scale = vec![1.0f32; n_inputs];

    let mut w0 = vec![0.0f32; n_inputs * n_hidden];
    for i in 0..n_inputs {
        for h in 0..n_hidden {
            w0[i * n_hidden + h] = ((i + h) as f32) * 0.1;
        }
    }
    let b0 = vec![0.1f32; n_hidden];

    let mut w1 = vec![0.0f32; n_hidden * n_outputs];
    for h in 0..n_hidden {
        for o in 0..n_outputs {
            w1[h * n_outputs + o] = ((h + o + 1) as f32) * 0.05;
        }
    }
    // Biases vary per output — argmin should land on the lowest.
    let b1: Vec<f32> = (0..n_outputs).map(|i| 5.0 - i as f32 * 0.1).collect();

    let layers = vec![
        BakeLayer {
            in_dim: n_inputs,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F16,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_outputs,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w1,
            biases: &b1,
        },
    ];

    // Feature bounds spanning a generous range — within the calibration
    // p01..p99 envelope.
    let feature_bounds: Vec<FeatureBound> = (0..n_inputs)
        .map(|_| FeatureBound::new(-10.0, 10.0))
        .collect();

    // Metadata: profile flag, schema_version_tag, calibration metrics,
    // bake name, codec-private cell config.
    let prof = [0u8]; // size_optimal
    let metrics_struct = [0.0233f32, 0.0512, 0.563];
    let mut metrics_bytes = [0u8; 12];
    metrics_bytes[0..4].copy_from_slice(&metrics_struct[0].to_le_bytes());
    metrics_bytes[4..8].copy_from_slice(&metrics_struct[1].to_le_bytes());
    metrics_bytes[8..12].copy_from_slice(&metrics_struct[2].to_le_bytes());
    let cell_config_blob = b"opaque-codec-private-cell-table";

    let metadata = vec![
        BakeMetadataEntry {
            key: keys::PROFILE,
            kind: MetadataType::Numeric,
            value: &prof,
        },
        BakeMetadataEntry {
            key: keys::SCHEMA_VERSION_TAG,
            kind: MetadataType::Utf8,
            value: b"zenpredict.v1.test-fixture",
        },
        BakeMetadataEntry {
            key: keys::CALIBRATION_METRICS,
            kind: MetadataType::Numeric,
            value: &metrics_bytes,
        },
        BakeMetadataEntry {
            key: keys::BAKE_NAME,
            kind: MetadataType::Utf8,
            value: b"lifecycle_test_v1",
        },
        BakeMetadataEntry {
            key: "zenjpeg.cell_config",
            kind: MetadataType::Bytes,
            value: cell_config_blob,
        },
    ];

    bake_v2(&BakeRequest {
        schema_hash: 0xfeedf00d_deadbeef,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &feature_bounds,
        metadata: &metadata,
    })
    .unwrap()
}

#[test]
fn end_to_end_codec_picker_lifecycle() {
    let raw = bake_codec_picker();
    let aligned = Aligned(raw);

    // Load with schema_hash gate.
    let model = Model::from_bytes_with_schema(&aligned.0, 0xfeedf00d_deadbeef).unwrap();
    assert_eq!(model.n_inputs(), 5);
    assert_eq!(model.n_outputs(), 12);
    assert_eq!(model.layers().len(), 2);

    // Verify metadata.
    let md = model.metadata();
    assert_eq!(md.len(), 5);
    let prof: u8 = md.get_pod(keys::PROFILE).unwrap();
    assert_eq!(prof, 0);
    assert_eq!(
        md.get_utf8(keys::SCHEMA_VERSION_TAG).unwrap(),
        "zenpredict.v1.test-fixture"
    );
    assert_eq!(md.get_utf8(keys::BAKE_NAME).unwrap(), "lifecycle_test_v1");
    let metrics: [f32; 3] = md.get_pod(keys::CALIBRATION_METRICS).unwrap();
    assert!((metrics[0] - 0.0233).abs() < 1e-6);
    let codec_blob = md.get_bytes("zenjpeg.cell_config").unwrap();
    assert_eq!(codec_blob, b"opaque-codec-private-cell-table");

    // Verify feature_bounds.
    let bounds = model.feature_bounds();
    assert_eq!(bounds.len(), 5);

    // OOD detection.
    assert_eq!(
        first_out_of_distribution(&[0.0, 1.0, -1.0, 5.0, 9.9], bounds),
        None
    );
    assert_eq!(
        first_out_of_distribution(&[0.0, 1.0, -1.0, 100.0, 9.9], bounds),
        Some(3)
    );

    // Build a Predictor and run a pick.
    let mut predictor = Predictor::new(model);
    let features = [0.5f32, -0.3, 1.2, 0.8, -0.1];
    let mask_data = [true; 12];
    let mask = AllowedMask::new(&mask_data);

    let pick = predictor
        .argmin_masked(&features, &mask, ScoreTransform::Identity, None)
        .unwrap();
    assert!(pick.is_some());

    // Same pick with Exp transform — for log-bytes shape regressors
    // the ordering is preserved (exp is monotonic), so picks match
    // when no offsets are present.
    let exp_pick = predictor
        .argmin_masked(&features, &mask, ScoreTransform::Exp, None)
        .unwrap();
    assert_eq!(exp_pick, pick);

    // With per-output offsets that strongly disfavor the chosen cell,
    // pick shifts.
    let mut offsets_arr = [0.0f32; 12];
    offsets_arr[pick.unwrap()] = 1e6;
    let offsets = ArgminOffsets {
        uniform: 0.0,
        per_output: Some(&offsets_arr),
    };
    let shifted_pick = predictor
        .argmin_masked(&features, &mask, ScoreTransform::Identity, Some(&offsets))
        .unwrap();
    assert_ne!(shifted_pick, pick);

    // Top-2 with confidence — the gap is in score space.
    let (best, gap) = predictor
        .pick_with_confidence(&features, &mask, ScoreTransform::Identity, None)
        .unwrap()
        .unwrap();
    assert_eq!(Some(best), pick);
    assert!(
        gap >= 0.0 && gap.is_finite(),
        "gap must be a finite non-negative number, got {gap}"
    );

    // Constraint mask narrows the legal set.
    let mut narrow_mask_data = [false; 12];
    narrow_mask_data[2] = true;
    narrow_mask_data[5] = true;
    let narrow = AllowedMask::new(&narrow_mask_data);
    let narrow_pick = predictor
        .argmin_masked(&features, &narrow, ScoreTransform::Identity, None)
        .unwrap();
    assert!(matches!(narrow_pick, Some(2) | Some(5)));
}

#[test]
fn end_to_end_perceptual_scorer_lifecycle() {
    // Mimic zensim's V0_4 scorer: many inputs (228), one output, one
    // hidden layer. F16 storage to match what real bakes use.
    let n_in = 228;
    let n_h = 64;
    let scaler_mean = vec![0.0f32; n_in];
    let scaler_scale = vec![1.0f32; n_in];

    let w0: Vec<f32> = (0..n_in * n_h).map(|i| (i as f32 * 0.001).sin()).collect();
    let b0 = vec![0.0f32; n_h];
    let w1: Vec<f32> = (0..n_h).map(|i| (i as f32 * 0.01).cos()).collect();
    let b1 = vec![0.5f32];

    let layers = vec![
        BakeLayer {
            in_dim: n_in,
            out_dim: n_h,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F16,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: n_h,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w1,
            biases: &b1,
        },
    ];
    let bytes = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
    })
    .unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(model);
    let features: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.01).sin()).collect();
    let out = p.predict(&features).unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].is_finite());
}

#[test]
fn rescue_workflow_two_shot() {
    // RescuePolicy is `#[non_exhaustive]`, so external code constructs
    // it via `Default` + field updates.
    let mut policy = RescuePolicy::default();
    policy.rescue_threshold = 2.0;
    policy.strategy = RescueStrategy::SecondBestPick;
    // Pass-0 verify hit zq=78 against target 80 → within threshold → ship.
    assert_eq!(should_rescue(78.0, 80.0, &policy), RescueDecision::Ship);
    // Same zq=78 against target 81 (gap 3) → exceeds 2pp threshold → rescue.
    assert_eq!(should_rescue(78.0, 81.0, &policy), RescueDecision::Rescue);
}

#[test]
fn threshold_mask_combines_with_constraint_mask() {
    // Reach rates: most cells safe at 0.99, one cell unsafe at 0.6,
    // one cell missing data (NaN).
    let rates = [0.99, 0.99, 0.6, f32::NAN, 0.95, 0.99];
    let constraint_mask = [true, true, true, true, false, true];
    let mut reach_gate = [false; 6];
    threshold_mask(&rates, 0.95, &mut reach_gate);
    let combined: Vec<bool> = constraint_mask
        .iter()
        .zip(&reach_gate)
        .map(|(&c, &r)| c && r)
        .collect();
    // Allowed = constraint AND reach_gated:
    //   0: true & true  → true
    //   1: true & true  → true
    //   2: true & false → false (rate below threshold)
    //   3: true & false → false (NaN)
    //   4: false & true → false (constraint denied)
    //   5: true & true  → true
    assert_eq!(combined, [true, true, false, false, false, true]);
}

#[test]
fn metadata_namespace_convention_works() {
    // Bake several entries from different namespaces; ensure all
    // round-trip cleanly.
    let scaler_mean = [0.0f32];
    let scaler_scale = [1.0f32];
    let w = [1.0f32];
    let b = [0.0f32];
    let layers = [BakeLayer {
        in_dim: 1,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w,
        biases: &b,
    }];
    let entries = [
        BakeMetadataEntry {
            key: "zentrain.profile",
            kind: MetadataType::Numeric,
            value: &[0u8],
        },
        BakeMetadataEntry {
            key: "zensim.calibration",
            kind: MetadataType::Utf8,
            value: b"v04",
        },
        BakeMetadataEntry {
            key: "zenjpeg.cell_config",
            kind: MetadataType::Bytes,
            value: b"opaque",
        },
        BakeMetadataEntry {
            key: "zenwebp.method_grid",
            kind: MetadataType::Bytes,
            value: b"opaque",
        },
    ];
    let bytes = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &entries,
    })
    .unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let md = model.metadata();
    assert_eq!(md.len(), 4);
    // Each namespace's keys are visible.
    assert!(md.get("zentrain.profile").is_some());
    assert!(md.get("zensim.calibration").is_some());
    assert!(md.get("zenjpeg.cell_config").is_some());
    assert!(md.get("zenwebp.method_grid").is_some());
}

#[test]
fn empty_metadata_does_not_break_load() {
    let scaler_mean = [0.0f32];
    let scaler_scale = [1.0f32];
    let w = [1.0f32];
    let b = [0.0f32];
    let layers = [BakeLayer {
        in_dim: 1,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w,
        biases: &b,
    }];
    let bytes = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
    })
    .unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    assert!(model.metadata().is_empty());
    assert!(model.feature_bounds().is_empty());
}

#[test]
fn metadata_iteration_returns_all_entries() {
    // Quick sanity check of iter, since the unit tests check
    // ordering and the public-API integration uses iter for dump
    // tooling shapes.
    let scaler_mean = [0.0f32];
    let scaler_scale = [1.0f32];
    let w = [1.0f32];
    let b = [0.0f32];
    let layers = [BakeLayer {
        in_dim: 1,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w,
        biases: &b,
    }];
    let entries = [
        BakeMetadataEntry {
            key: "a",
            kind: MetadataType::Bytes,
            value: b"",
        },
        BakeMetadataEntry {
            key: "b",
            kind: MetadataType::Utf8,
            value: b"yes",
        },
    ];
    let bytes = bake_v2(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata: &entries,
    })
    .unwrap();
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).unwrap();
    let md: &Metadata<'_> = model.metadata();
    let collected: Vec<(&str, MetadataType)> = md.iter().map(|e| (e.key, e.kind)).collect();
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0].0, "a");
    assert_eq!(collected[1].0, "b");
}
