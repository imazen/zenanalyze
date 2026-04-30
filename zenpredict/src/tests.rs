//! Crate-internal sanity checks — model parser, metadata parser,
//! argmin math. Round-trip tests that exercise bake → load → forward
//! live in `tests/roundtrip.rs`.

use crate::*;

#[test]
fn argmin_identity_no_offsets() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true, true, true, true, true];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(1));
}

#[test]
fn argmin_respects_mask() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true, false, true, true, true];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(3));
}

#[test]
fn argmin_empty_mask_returns_none() {
    let pred = [3.0f32, 1.0, 4.0];
    let mask = [false; 3];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, None);
}

#[test]
fn argmin_with_per_output_offsets_shifts_pick() {
    // Without offsets the lowest score is index 1 (=1.0).
    // With per_output adding +5 to index 1, the new lowest is 3 (=1.5).
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let offsets = ArgminOffsets {
        uniform: 0.0,
        per_output: Some(&[0.0, 5.0, 0.0, 0.0, 0.0]),
    };
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, Some(&offsets));
    assert_eq!(pick, Some(3));
}

#[test]
fn argmin_top_k_returns_sorted_indices() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(top, [Some(1), Some(3), Some(0)]);
}

#[test]
fn pick_with_confidence_reports_gap() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let pick = argmin::pick_with_confidence(&pred, &m, ScoreTransform::Identity, None);
    let (idx, gap) = pick.unwrap();
    assert_eq!(idx, 1);
    assert!((gap - 0.5).abs() < 1e-6);
}

#[test]
fn pick_with_confidence_inf_when_only_one_allowed() {
    let pred = [3.0f32, 1.0, 4.0];
    let mask = [false, true, false];
    let m = AllowedMask::new(&mask);
    let (idx, gap) =
        argmin::pick_with_confidence(&pred, &m, ScoreTransform::Identity, None).unwrap();
    assert_eq!(idx, 1);
    assert!(gap.is_infinite());
}

#[test]
fn threshold_mask_finite_gate() {
    // INFINITY is non-finite → treated like NaN, gated out. The
    // mask gate is intentionally finite-only because `INFINITY` in
    // a reach-rate table conventionally means "missing data" not
    // "always reached."
    let rates = [0.99, 0.5, f32::NAN, 0.95, f32::INFINITY];
    let mut out = [false; 5];
    argmin::threshold_mask(&rates, 0.95, &mut out);
    assert_eq!(out, [true, false, false, true, false]);
}

#[test]
fn first_out_of_distribution_finds_first() {
    let bounds = [
        FeatureBound::new(0.0, 1.0),
        FeatureBound::new(-1.0, 1.0),
        FeatureBound::new(0.0, 100.0),
    ];
    assert_eq!(first_out_of_distribution(&[0.5, 0.0, 50.0], &bounds), None);
    assert_eq!(
        first_out_of_distribution(&[2.0, 0.0, 50.0], &bounds),
        Some(0)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, 0.0, f32::NAN], &bounds),
        Some(2)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, f32::INFINITY, 50.0], &bounds),
        Some(1)
    );
}

#[test]
fn metadata_empty_blob_yields_empty() {
    let m = Metadata::parse(&[]).unwrap();
    assert!(m.is_empty());
}

#[test]
fn rescue_default_threshold_three_pp() {
    let policy = RescuePolicy::default();
    assert!((policy.rescue_threshold - 3.0).abs() < f32::EPSILON);
}

#[cfg(feature = "bake")]
mod bake_roundtrip {
    use crate::bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake_v2};
    use crate::metadata::MetadataType;
    use crate::*;

    /// Wrapper that guarantees 16-byte alignment of an in-memory
    /// model blob — what `include_bytes!` consumers do via
    /// `#[repr(C, align(16))]`.
    #[repr(C, align(16))]
    struct Aligned(Vec<u8>);

    fn make_simple_model() -> Vec<u8> {
        // 3 inputs → 4 hidden (LeakyReLU, F32) → 2 outputs (Identity, F32)
        let scaler_mean = [0.0f32, 0.0, 0.0];
        let scaler_scale = [1.0f32, 1.0, 1.0];
        let w0 = [
            // layer 0: 3 × 4 row-major (input-major)
            1.0, 0.0, 0.0, 0.0, // input 0 → outs
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let b0 = [0.0f32, 0.0, 0.0, 0.0];
        let w1 = [
            // layer 1: 4 × 2 row-major
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let b1 = [10.0f32, 20.0];
        let layers = [
            BakeLayer {
                in_dim: 3,
                out_dim: 4,
                activation: Activation::LeakyRelu,
                dtype: WeightDtype::F32,
                weights: &w0,
                biases: &b0,
            },
            BakeLayer {
                in_dim: 4,
                out_dim: 2,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w1,
                biases: &b1,
            },
        ];
        let req = BakeRequest {
            schema_hash: 0xdeadbeef_cafebabe,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
        };
        bake_v2(&req).unwrap()
    }

    #[test]
    fn round_trip_f32_basic() {
        let bytes = make_simple_model();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert_eq!(model.n_inputs(), 3);
        assert_eq!(model.n_outputs(), 2);
        assert_eq!(model.schema_hash(), 0xdeadbeef_cafebabe);

        let mut predictor = Predictor::new(model);
        let out = predictor.predict(&[1.0, 1.0, 1.0]).unwrap();
        // Layer 0 produces [1,1,1,0] (LeakyReLU on non-negatives is
        // identity). Layer 1 produces [w[0,0]*1 + b0=10, w[1,1]*1 + b1=20]
        // = [11, 21].
        assert!((out[0] - 11.0).abs() < 1e-5, "got {out:?}");
        assert!((out[1] - 21.0).abs() < 1e-5, "got {out:?}");
    }

    #[test]
    fn round_trip_with_metadata() {
        let scaler_mean = [0.0f32, 0.0];
        let scaler_scale = [1.0f32, 1.0];
        let w0 = [1.0f32, 0.0, 0.0, 1.0];
        let b0 = [0.0f32, 0.0];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 2,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        }];
        let prof_value = [1u8];
        let metrics = [0.0233f32, 0.0512, 0.563];
        let metrics_bytes: [u8; 12] = unsafe_to_bytes(&metrics);
        let metadata = [
            BakeMetadataEntry {
                key: "zenpicker.profile",
                kind: MetadataType::Numeric,
                value: &prof_value,
            },
            BakeMetadataEntry {
                key: "zenpicker.bake_name",
                kind: MetadataType::Utf8,
                value: b"test_bake_v0",
            },
            BakeMetadataEntry {
                key: "zenpicker.calibration_metrics",
                kind: MetadataType::Numeric,
                value: &metrics_bytes,
            },
        ];
        let req = BakeRequest {
            schema_hash: 0xaaaa_bbbb_cccc_dddd,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &metadata,
        };
        let bytes = bake_v2(&req).unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let md = model.metadata();

        assert_eq!(md.len(), 3);
        assert_eq!(md.get_utf8("zenpicker.bake_name").unwrap(), "test_bake_v0");
        assert_eq!(md.get_numeric("zenpicker.profile").unwrap(), &[1u8]);

        // Wrong-type lookup fails.
        let err = md.get_utf8("zenpicker.profile").unwrap_err();
        assert!(matches!(err, PredictError::MetadataTypeMismatch { .. }));

        // pod_read_unaligned of the calibration metrics struct.
        let metrics_back: [f32; 3] = md.get_pod("zenpicker.calibration_metrics").unwrap();
        assert!((metrics_back[0] - 0.0233).abs() < 1e-6);
        assert!((metrics_back[2] - 0.563).abs() < 1e-6);
    }

    #[test]
    fn round_trip_f16_storage() {
        let scaler_mean = [0.0f32, 0.0];
        let scaler_scale = [1.0f32, 1.0];
        let w0 = [0.5f32, -0.25, 1.0, 2.0];
        let b0 = [0.0f32, 1.0];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 2,
            activation: Activation::Identity,
            dtype: WeightDtype::F16,
            weights: &w0,
            biases: &b0,
        }];
        let req = BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
        };
        let bytes = bake_v2(&req).unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);
        let out = predictor.predict(&[1.0, 1.0]).unwrap();
        // f16 round-trip is exact for these values:
        //   y0 = 0.5*1 + 1.0*1 + 0.0 = 1.5
        //   y1 = -0.25*1 + 2.0*1 + 1.0 = 2.75
        assert!((out[0] - 1.5).abs() < 1e-3, "got {out:?}");
        assert!((out[1] - 2.75).abs() < 1e-3, "got {out:?}");
    }

    #[test]
    fn round_trip_i8_storage() {
        let scaler_mean = [0.0f32, 0.0];
        let scaler_scale = [1.0f32, 1.0];
        // Use values well-quantizable to i8 (max abs / 127 → small step).
        let w0 = [1.0f32, -0.5, 0.5, 1.0];
        let b0 = [0.0f32, 0.0];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 2,
            activation: Activation::Identity,
            dtype: WeightDtype::I8,
            weights: &w0,
            biases: &b0,
        }];
        let req = BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
        };
        let bytes = bake_v2(&req).unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);
        let out = predictor.predict(&[1.0, 1.0]).unwrap();
        // Row-major weights [1.0, -0.5, 0.5, 1.0]:
        //   y0 = 1*1.0 + 1*0.5 = 1.5
        //   y1 = 1*(-0.5) + 1*1.0 = 0.5
        // I8 quant noise per-output is ~max_col_abs/127 ≈ 0.008 →
        // round-trip should be within ~1% of the f32 reference.
        assert!((out[0] - 1.5).abs() < 0.05, "got {out:?}");
        assert!((out[1] - 0.5).abs() < 0.05, "got {out:?}");
    }

    #[test]
    fn round_trip_with_feature_bounds() {
        let scaler_mean = [0.0f32, 0.0];
        let scaler_scale = [1.0f32, 1.0];
        let w0 = [1.0f32, 0.0, 0.0, 1.0];
        let b0 = [0.0f32, 0.0];
        let bounds = [FeatureBound::new(-1.0, 1.0), FeatureBound::new(0.0, 100.0)];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 2,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        }];
        let req = BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &bounds,
            metadata: &[],
        };
        let bytes = bake_v2(&req).unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let fb = model.feature_bounds();
        assert_eq!(fb.len(), 2);
        assert_eq!(fb[0], FeatureBound::new(-1.0, 1.0));
        assert_eq!(fb[1], FeatureBound::new(0.0, 100.0));
        assert_eq!(first_out_of_distribution(&[0.0, 50.0], fb), None);
        assert_eq!(first_out_of_distribution(&[2.0, 50.0], fb), Some(0));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = make_simple_model();
        bytes[0] = b'X';
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(matches!(err, PredictError::BadMagic { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = make_simple_model();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(
            matches!(err, PredictError::UnsupportedVersion { version: 99, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn schema_hash_check_succeeds_when_match() {
        let bytes = make_simple_model();
        let aligned = Aligned(bytes);
        let m = Model::from_bytes_with_schema(&aligned.0, 0xdeadbeef_cafebabe).unwrap();
        assert_eq!(m.schema_hash(), 0xdeadbeef_cafebabe);
    }

    #[test]
    fn schema_hash_check_fails_on_mismatch() {
        let bytes = make_simple_model();
        let aligned = Aligned(bytes);
        let err = Model::from_bytes_with_schema(&aligned.0, 0).unwrap_err();
        assert!(matches!(err, PredictError::SchemaHashMismatch { .. }));
    }

    /// Tiny helper to convert `&[f32; 3]` → `[u8; 12]` for metadata
    /// payloads. Not in the public crate API.
    fn unsafe_to_bytes(arr: &[f32; 3]) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&arr[0].to_le_bytes());
        out[4..8].copy_from_slice(&arr[1].to_le_bytes());
        out[8..12].copy_from_slice(&arr[2].to_le_bytes());
        out
    }
}
