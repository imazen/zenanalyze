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
                key: "zentrain.profile",
                kind: MetadataType::Numeric,
                value: &prof_value,
            },
            BakeMetadataEntry {
                key: "zentrain.bake_name",
                kind: MetadataType::Utf8,
                value: b"test_bake_v0",
            },
            BakeMetadataEntry {
                key: "zentrain.calibration_metrics",
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
        assert_eq!(md.get_utf8("zentrain.bake_name").unwrap(), "test_bake_v0");
        assert_eq!(md.get_numeric("zentrain.profile").unwrap(), &[1u8]);

        // Wrong-type lookup fails.
        let err = md.get_utf8("zentrain.profile").unwrap_err();
        assert!(matches!(err, PredictError::MetadataTypeMismatch { .. }));

        // pod_read_unaligned of the calibration metrics struct.
        let metrics_back: [f32; 3] = md.get_pod("zentrain.calibration_metrics").unwrap();
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

    // ----------------------------------------------------------------
    // Depth / width / shape flexibility.
    //
    // ZNPR puts no upper bound on n_layers, in_dim, or out_dim other
    // than u32::MAX (and a successful `n_inputs * n_hidden` non-overflowing
    // multiply). These tests pin that flexibility against representative
    // shapes consumers might actually ship.
    // ----------------------------------------------------------------

    fn build_layer<'a>(
        in_dim: usize,
        out_dim: usize,
        activation: Activation,
        dtype: WeightDtype,
        weights: &'a Vec<f32>,
        biases: &'a Vec<f32>,
    ) -> BakeLayer<'a> {
        BakeLayer {
            in_dim,
            out_dim,
            activation,
            dtype,
            weights,
            biases,
        }
    }

    fn ones(n: usize) -> Vec<f32> {
        alloc::vec![1.0f32; n]
    }
    fn zeros(n: usize) -> Vec<f32> {
        alloc::vec![0.0f32; n]
    }

    #[test]
    fn single_layer_model_works() {
        // Just a linear projection, no hidden layer.
        let scaler_mean = [0.0f32; 4];
        let scaler_scale = [1.0f32; 4];
        let w = [
            1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let b = [10.0f32, 20.0, 30.0];
        let layers = [BakeLayer {
            in_dim: 4,
            out_dim: 3,
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
        assert_eq!(model.layers().len(), 1);
        let mut p = Predictor::new(model);
        let out = p.predict(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(out, &[11.0, 21.0, 31.0]);
    }

    #[test]
    fn ten_layer_deep_model_works() {
        // Identity-ish chain: 4 → 4 → 4 → ... → 4 (10 layers).
        // Each layer's weight matrix is the identity, biases zero.
        // Output should equal scaled input.
        let n = 4;
        let scaler_mean = zeros(n);
        let scaler_scale = ones(n);
        let mut id = zeros(n * n);
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let b = zeros(n);
        // Need to keep `id` and `b` alive across 10 BakeLayer entries.
        let layers: Vec<BakeLayer<'_>> = (0..10)
            .map(|_| BakeLayer {
                in_dim: n,
                out_dim: n,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &id,
                biases: &b,
            })
            .collect();
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
        assert_eq!(model.layers().len(), 10);
        let mut p = Predictor::new(model);
        let out = p.predict(&[2.0, 3.0, 5.0, 7.0]).unwrap();
        assert_eq!(out, &[2.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn wide_then_narrow_bottleneck() {
        // 4 → 64 → 4 → 2. Tests that scratch_len picks the widest layer.
        let scaler_mean = zeros(4);
        let scaler_scale = ones(4);
        let w0 = ones(4 * 64);
        let b0 = zeros(64);
        let w1 = ones(64 * 4);
        let b1 = zeros(4);
        let w2 = ones(4 * 2);
        let b2 = zeros(2);
        let layers = [
            build_layer(4, 64, Activation::Relu, WeightDtype::F32, &w0, &b0),
            build_layer(64, 4, Activation::Identity, WeightDtype::F32, &w1, &b1),
            build_layer(4, 2, Activation::Identity, WeightDtype::F32, &w2, &b2),
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
        // scratch_len must be at least 64 (widest layer) for forward to succeed.
        assert_eq!(model.scratch_len(), 64);
        let mut p = Predictor::new(model);
        let out = p.predict(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        // h0[i] = sum_j 1 * 1 = 4 (per chunk). 64 hidden units, all 4.
        // h1[i] = sum 1*4 = 256. 4 outputs, all 256.
        // out[i] = sum 1*256 = 1024. 2 outputs, both 1024.
        assert_eq!(out, &[1024.0, 1024.0]);
    }

    #[test]
    fn mixed_dtypes_per_layer() {
        // Layer 0 = i8, layer 1 = f16, layer 2 = f32. All identity-ish.
        let scaler_mean = zeros(3);
        let scaler_scale = ones(3);
        let mut w0 = zeros(3 * 3);
        for i in 0..3 {
            w0[i * 3 + i] = 1.0;
        }
        let b0 = zeros(3);
        let mut w1 = zeros(3 * 3);
        for i in 0..3 {
            w1[i * 3 + i] = 1.0;
        }
        let b1 = zeros(3);
        let mut w2 = zeros(3 * 3);
        for i in 0..3 {
            w2[i * 3 + i] = 1.0;
        }
        let b2 = zeros(3);
        let layers = [
            build_layer(3, 3, Activation::Identity, WeightDtype::I8, &w0, &b0),
            build_layer(3, 3, Activation::Identity, WeightDtype::F16, &w1, &b1),
            build_layer(3, 3, Activation::Identity, WeightDtype::F32, &w2, &b2),
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
        let out = p.predict(&[1.0, 2.0, 3.0]).unwrap();
        // i8 quantization is lossy; f16 and f32 paths are exact at these
        // magnitudes. Identity chain should round-trip within ~1%.
        assert!((out[0] - 1.0).abs() < 0.05, "got {out:?}");
        assert!((out[1] - 2.0).abs() < 0.05, "got {out:?}");
        assert!((out[2] - 3.0).abs() < 0.05, "got {out:?}");
    }

    #[test]
    fn each_activation_works() {
        // Two-layer model: 1 input, 4 hidden, 1 output.
        // Hidden weights = [1, -1, 0.5, -0.5], biases = 0. Activation
        // varies. Final layer collapses with [1,1,1,1] ones identity.
        for act in [
            Activation::Identity,
            Activation::Relu,
            Activation::LeakyRelu,
        ] {
            let scaler_mean = [0.0f32];
            let scaler_scale = [1.0f32];
            let w0 = alloc::vec![1.0f32, -1.0, 0.5, -0.5];
            let b0 = alloc::vec![0.0f32; 4];
            let w1 = alloc::vec![1.0f32, 1.0, 1.0, 1.0];
            let b1 = alloc::vec![0.0f32];
            let layers = [
                build_layer(1, 4, act, WeightDtype::F32, &w0, &b0),
                build_layer(4, 1, Activation::Identity, WeightDtype::F32, &w1, &b1),
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
            let out = p.predict(&[2.0]).unwrap();
            // Pre-activation hidden: [2, -2, 1, -1].
            //   Identity   → sum = 0
            //   ReLU       → [2, 0, 1, 0] sum = 3
            //   LeakyReLU  → [2, -0.02, 1, -0.01] sum = 2.97
            let expected = match act {
                Activation::Identity => 0.0,
                Activation::Relu => 3.0,
                Activation::LeakyRelu => 2.97,
            };
            assert!((out[0] - expected).abs() < 1e-5, "act={act:?} got {out:?}");
        }
    }

    #[test]
    fn very_wide_layer_works() {
        // 8 → 1024 → 1. ~12K weights, well within bake/parse budget.
        let scaler_mean = zeros(8);
        let scaler_scale = ones(8);
        let w0 = ones(8 * 1024);
        let b0 = zeros(1024);
        let w1 = ones(1024);
        let b1 = zeros(1);
        let layers = [
            build_layer(8, 1024, Activation::Relu, WeightDtype::F16, &w0, &b0),
            build_layer(1024, 1, Activation::Identity, WeightDtype::F32, &w1, &b1),
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
        assert_eq!(model.scratch_len(), 1024);
        let mut p = Predictor::new(model);
        let out = p.predict(&[1.0; 8]).unwrap();
        // h[i] = ReLU(sum_j 1*1) = 8 (positive). All 1024 hidden = 8.
        // y = sum_i 1*8 = 8 * 1024 = 8192.
        assert!((out[0] - 8192.0).abs() < 0.5, "got {out:?}");
    }

    // ----------------------------------------------------------------
    // Argmin family — ranges, top-K, transforms, offset validation.
    // ----------------------------------------------------------------

    #[test]
    fn argmin_in_range_returns_local_index() {
        // Hybrid-heads layout: [bytes[0..3], scalar1[3..6], scalar2[6..9]].
        let pred = [3.0f32, 1.0, 4.0, 999.0, 0.0, -100.0, 50.0, 51.0, 52.0];
        let mask = [true, true, true]; // 3-cell mask
        let m = AllowedMask::new(&mask);
        let pick =
            argmin::argmin_masked_in_range(&pred, (0, 3), &m, ScoreTransform::Identity, None);
        // Sub-range argmin should ignore the scalar heads.
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn argmin_in_range_top_k_returns_local_indices() {
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0, 999.0, 0.0, -100.0];
        let mask = [true; 5];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k_in_range::<3>(
            &pred,
            (0, 5),
            &m,
            ScoreTransform::Identity,
            None,
        );
        // Sub-range top-3: index 1 (1.0), index 3 (1.5), index 0 (3.0).
        assert_eq!(top, [Some(1), Some(3), Some(0)]);
    }

    #[test]
    fn argmin_top_k_with_few_allowed() {
        // Only two entries allowed → top-3 returns 2 Some + 1 None.
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let mask = [false, true, false, true, false];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(1), Some(3), None]);
    }

    #[test]
    fn argmin_top_k_with_no_allowed() {
        let pred = [3.0f32, 1.0, 4.0];
        let mask = [false; 3];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k::<2>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [None, None]);
    }

    #[test]
    fn argmin_top_k_k_one_works() {
        let pred = [3.0f32, 1.0, 4.0];
        let mask = [true; 3];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k::<1>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(1)]);
    }

    #[test]
    fn score_transform_exp_changes_pick_with_offsets() {
        // Identity-domain: scores = [log(1000), log(2000)] ≈ [6.9, 7.6].
        // Argmin in identity space → index 0 (small log = small bytes).
        // With per_output offset of [+0, +5000] in linear space, raw bytes
        // become [1000, 2000+5000=7000] → still index 0.
        // But if we set offsets to [+5000, +0]: [1000+5000=6000, 2000] →
        // index 1.
        let pred = [(1000f32).ln(), (2000f32).ln()];
        let mask = [true; 2];
        let m = AllowedMask::new(&mask);
        // Without offsets, identity argmin picks 0.
        assert_eq!(
            argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None),
            Some(0)
        );
        // With Exp + per_output [5000, 0]: index 0 has 6000 vs index 1
        // has 2000 → pick 1.
        let off = ArgminOffsets {
            uniform: 0.0,
            per_output: Some(&[5000.0, 0.0]),
        };
        assert_eq!(
            argmin::argmin_masked(&pred, &m, ScoreTransform::Exp, Some(&off)),
            Some(1)
        );
    }

    #[test]
    fn pick_with_confidence_in_range_works() {
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0, 999.0, 0.0];
        let mask = [true; 5];
        let m = AllowedMask::new(&mask);
        let result = argmin::pick_with_confidence_in_range(
            &pred,
            (0, 5),
            &m,
            ScoreTransform::Identity,
            None,
        );
        let (idx, gap) = result.unwrap();
        assert_eq!(idx, 1);
        assert!((gap - 0.5).abs() < 1e-6, "got gap {gap}");
    }

    #[test]
    fn argmin_offsets_validation_rejects_wrong_length() {
        let scaler_mean = [0.0f32; 2];
        let scaler_scale = [1.0f32; 2];
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let b = [0.0f32; 2];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 2,
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
        let mut p = Predictor::new(model);
        // n_outputs = 2 but we pass per_output of length 5.
        let bad_offsets = ArgminOffsets {
            uniform: 0.0,
            per_output: Some(&[0.0, 0.0, 0.0, 0.0, 0.0]),
        };
        let mask = [true; 2];
        let m = AllowedMask::new(&mask);
        let err = p
            .argmin_masked(
                &[1.0, 1.0],
                &m,
                ScoreTransform::Identity,
                Some(&bad_offsets),
            )
            .unwrap_err();
        assert!(matches!(err, PredictError::OffsetsLenMismatch { .. }));
    }

    #[test]
    fn forward_rejects_wrong_feature_length() {
        let scaler_mean = [0.0f32; 3];
        let scaler_scale = [1.0f32; 3];
        let w = [1.0f32; 3 * 2];
        let b = [0.0f32; 2];
        let layers = [BakeLayer {
            in_dim: 3,
            out_dim: 2,
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
        let mut p = Predictor::new(model);
        let err = p.predict(&[1.0, 1.0]).unwrap_err(); // 2 vs expected 3
        assert!(matches!(
            err,
            PredictError::FeatureLenMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    // ----------------------------------------------------------------
    // Format negative tests — truncation, dim mismatch, alignment.
    // ----------------------------------------------------------------

    #[test]
    fn empty_bytes_rejected() {
        let err = Model::from_bytes(&[]).unwrap_err();
        assert!(matches!(err, PredictError::Truncated { .. }));
    }

    #[test]
    fn header_truncated_rejected() {
        let bytes = [0u8; 64]; // less than 128
        let err = Model::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, PredictError::Truncated { .. }));
    }

    #[test]
    fn truncated_after_header_rejected() {
        let bytes = make_simple_model();
        let truncated = &bytes[..bytes.len() - 8]; // drop last 8 bytes
        let aligned = Aligned(truncated.to_vec());
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(matches!(err, PredictError::SectionOutOfRange { .. }));
    }

    #[test]
    fn corrupt_layer_in_dim_rejected() {
        // Write an invalid in_dim into layer 0's LayerEntry.
        // Layer table starts at offset 128 in our bake; layer 0's in_dim
        // is at offset 128 (bytes [128..132]).
        let mut bytes = make_simple_model();
        // Original in_dim was 3 (matches n_inputs). Set it to 99.
        bytes[128..132].copy_from_slice(&99u32.to_le_bytes());
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(matches!(err, PredictError::LayerDimMismatch { .. }));
    }

    #[test]
    fn unknown_activation_byte_rejected() {
        let mut bytes = make_simple_model();
        // Layer 0 activation byte is at offset 128 + 8 = 136.
        bytes[136] = 99;
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(matches!(err, PredictError::UnknownActivation { byte: 99 }));
    }

    #[test]
    fn unknown_weight_dtype_rejected() {
        let mut bytes = make_simple_model();
        // Layer 0 weight_dtype byte is at offset 128 + 9 = 137.
        bytes[137] = 99;
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(matches!(err, PredictError::UnknownWeightDtype { byte: 99 }));
    }

    #[test]
    fn zero_n_inputs_rejected() {
        let mut bytes = make_simple_model();
        // n_inputs is at offset 8.
        bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(matches!(err, PredictError::ZeroDimension { .. }));
    }

    #[test]
    fn raw_bytes_round_trips_blob() {
        let bytes = make_simple_model();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert_eq!(model.raw_bytes(), &aligned.0[..]);
    }

    // ----------------------------------------------------------------
    // Metadata edge cases.
    // ----------------------------------------------------------------

    #[test]
    fn metadata_get_missing_key_returns_none() {
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
            metadata: &[BakeMetadataEntry {
                key: "foo",
                kind: MetadataType::Utf8,
                value: b"bar",
            }],
        })
        .unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert!(model.metadata().get("nope").is_none());
        assert_eq!(model.metadata().get("foo").unwrap().key, "foo");
    }

    #[test]
    fn metadata_get_pod_wrong_size_returns_none() {
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
            metadata: &[BakeMetadataEntry {
                key: "k",
                kind: MetadataType::Numeric,
                value: &[1, 2, 3, 4], // 4 bytes
            }],
        })
        .unwrap();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        // Asking for a u64 (8 bytes) when the value is 4 bytes → None.
        assert!(model.metadata().get_pod::<u64>("k").is_none());
        // Right-sized read works.
        let got: u32 = model.metadata().get_pod("k").unwrap();
        assert_eq!(got, u32::from_le_bytes([1, 2, 3, 4]));
    }

    #[test]
    fn metadata_truncated_blob_rejected() {
        // Hand-craft a metadata blob that's truncated mid-value.
        let mut blob = alloc::vec![3u8]; // key_len = 3
        blob.extend_from_slice(b"key"); // key
        blob.push(1); // value_type = utf8
        blob.extend_from_slice(&100u32.to_le_bytes()); // value_len = 100
        blob.extend_from_slice(b"short"); // only 5 bytes of value
        let err = Metadata::parse(&blob).unwrap_err();
        assert!(matches!(err, PredictError::Truncated { .. }));
    }

    #[test]
    fn metadata_invalid_utf8_key_rejected() {
        // [u8 key_len][key][u8 type][u32 value_len]
        let mut blob = alloc::vec![
            2,    // key_len = 2
            0xff, // invalid UTF-8 lead byte
            0xff, // continuation
            0,    // value_type = bytes
        ];
        blob.extend_from_slice(&0u32.to_le_bytes()); // value_len = 0
        let err = Metadata::parse(&blob).unwrap_err();
        assert!(matches!(err, PredictError::MetadataKeyNotUtf8 { .. }));
    }

    #[test]
    fn metadata_reserved_type_preserved() {
        let mut blob = alloc::vec![
            1u8,  // key_len = 1
            b'x', // key bytes
            99,   // reserved type
        ];
        blob.extend_from_slice(&3u32.to_le_bytes());
        blob.extend_from_slice(&[1, 2, 3]);
        let m = Metadata::parse(&blob).unwrap();
        let entry = m.get("x").unwrap();
        assert!(matches!(entry.kind, MetadataType::Reserved(99)));
        assert_eq!(entry.value, &[1, 2, 3]);
    }

    #[test]
    fn metadata_iter_preserves_order() {
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
                key: "first",
                kind: MetadataType::Bytes,
                value: b"",
            },
            BakeMetadataEntry {
                key: "second",
                kind: MetadataType::Utf8,
                value: b"hello",
            },
            BakeMetadataEntry {
                key: "third",
                kind: MetadataType::Numeric,
                value: &[42],
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
        let keys: Vec<&str> = model.metadata().iter().map(|e| e.key).collect();
        assert_eq!(keys, ["first", "second", "third"]);
    }

    // ----------------------------------------------------------------
    // Bake-time validation errors.
    // ----------------------------------------------------------------

    #[test]
    fn bake_rejects_layer_dim_chain_break() {
        let scaler_mean = [0.0f32; 2];
        let scaler_scale = [1.0f32; 2];
        let w0 = [1.0f32; 2 * 3];
        let b0 = [0.0f32; 3];
        // Layer 1 declares in_dim = 99 instead of 3.
        let w1 = [1.0f32; 99 * 4];
        let b1 = [0.0f32; 4];
        let layers = [
            BakeLayer {
                in_dim: 2,
                out_dim: 3,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w0,
                biases: &b0,
            },
            BakeLayer {
                in_dim: 99,
                out_dim: 4,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w1,
                biases: &b1,
            },
        ];
        let err = bake_v2(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
        })
        .unwrap_err();
        assert!(matches!(
            err,
            crate::bake::BakeError::LayerDimMismatch { .. }
        ));
    }

    #[test]
    fn bake_rejects_empty_metadata_key() {
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
        let err = bake_v2(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[BakeMetadataEntry {
                key: "",
                kind: MetadataType::Bytes,
                value: b"x",
            }],
        })
        .unwrap_err();
        assert!(matches!(err, crate::bake::BakeError::MetadataKeyEmpty));
    }

    #[test]
    fn bake_rejects_metadata_key_too_long() {
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
        let long_key: alloc::string::String = "a".repeat(300);
        let err = bake_v2(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[BakeMetadataEntry {
                key: &long_key,
                kind: MetadataType::Bytes,
                value: b"x",
            }],
        })
        .unwrap_err();
        assert!(matches!(
            err,
            crate::bake::BakeError::MetadataKeyTooLong { len: 300 }
        ));
    }

    #[test]
    fn bake_rejects_empty_layers() {
        let err = bake_v2(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &[],
            scaler_scale: &[],
            layers: &[],
            feature_bounds: &[],
            metadata: &[],
        })
        .unwrap_err();
        assert!(matches!(err, crate::bake::BakeError::EmptyLayers));
    }
}

// =====================================================================
// Issue #55: argmin_masked_with_scorer for caller-composed score
// functions. Tests cover the standalone helper, predictor wrappers,
// equivalence to ScoreTransform::Identity, and the motivating
// RD-vs-time pattern.
// =====================================================================

#[cfg(test)]
mod scorer_tests {
    use super::*;

    #[test]
    fn standalone_scorer_finds_argmin_under_mask() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, true, true, true, true];
        let mask = AllowedMask::new(&mask_data);
        let pick = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| scores[i]);
        assert_eq!(pick, Some(1));
    }

    #[test]
    fn standalone_scorer_respects_mask() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, false, true, true, true];
        let mask = AllowedMask::new(&mask_data);
        let pick = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| scores[i]);
        assert_eq!(pick, Some(3));
    }

    #[test]
    fn standalone_scorer_empty_mask_returns_none() {
        let mask_data = [false; 5];
        let mask = AllowedMask::new(&mask_data);
        let pick = argmin::argmin_masked_with_scorer(5, &mask, |_| 0.0);
        assert_eq!(pick, None);
    }

    #[test]
    fn scorer_call_count_matches_allowed_indices() {
        // Closure mutates a side-effect counter; assert it fires
        // exactly once per allowed index.
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, false, true, false, true];
        let mask = AllowedMask::new(&mask_data);
        use core::cell::Cell;
        let count = Cell::new(0);
        let _ = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| {
            count.set(count.get() + 1);
            scores[i]
        });
        assert_eq!(count.get(), 3, "scorer fired once per allowed index");
    }

    #[test]
    fn scorer_top_k_returns_sorted() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true; 5];
        let mask = AllowedMask::new(&mask_data);
        let top =
            argmin::argmin_masked_top_k_with_scorer::<3, _>(scores.len(), &mask, |i| scores[i]);
        assert_eq!(top, [Some(1), Some(3), Some(0)]);
    }

    #[test]
    fn scorer_top_k_with_fewer_allowed_pads_none() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [false, true, false, true, false];
        let mask = AllowedMask::new(&mask_data);
        let top =
            argmin::argmin_masked_top_k_with_scorer::<3, _>(scores.len(), &mask, |i| scores[i]);
        assert_eq!(top, [Some(1), Some(3), None]);
    }

    /// Identity-scorer should match `argmin_masked` with
    /// `ScoreTransform::Identity` and no offsets — sanity that the
    /// new path doesn't subtly diverge for the trivial case.
    #[test]
    fn scorer_identity_equivalent_to_argmin_masked_identity() {
        let scores = [3.0_f32, 1.0, 4.0, 1.5, 9.0];
        let mask_data = [true, true, false, true, true];
        let mask = AllowedMask::new(&mask_data);
        let scorer_pick = argmin::argmin_masked_with_scorer(scores.len(), &mask, |i| scores[i]);
        let transform_pick = argmin::argmin_masked(&scores, &mask, ScoreTransform::Identity, None);
        assert_eq!(scorer_pick, transform_pick);
    }

    #[cfg(feature = "bake")]
    fn make_two_head_model() -> Vec<u8> {
        // Hybrid-heads layout: 3 cells, [bytes_log[0..3], time[3..6]].
        // Identity weights so the model passes its inputs straight
        // through to outputs (lets the test put exact known values
        // in the output without needing real training).
        use crate::bake::{BakeLayer, BakeRequest, bake_v2};
        let scaler_mean = [0.0_f32; 6];
        let scaler_scale = [1.0_f32; 6];
        let mut w = vec![0.0_f32; 6 * 6];
        for i in 0..6 {
            w[i * 6 + i] = 1.0;
        }
        let b = vec![0.0_f32; 6];
        let layers = [BakeLayer {
            in_dim: 6,
            out_dim: 6,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w,
            biases: &b,
        }];
        bake_v2(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
        })
        .unwrap()
    }

    /// Motivating case: RD-vs-time argmin with hybrid-heads outputs.
    /// `score = bytes + μ·ms`; closure reads from both heads.
    #[cfg(feature = "bake")]
    #[test]
    fn predictor_scorer_rd_vs_time() {
        #[repr(C, align(16))]
        struct Aligned(Vec<u8>);
        let bytes = make_two_head_model();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);

        // bytes_log = [10.0, 11.0, 12.0]    → bytes ≈ [22 026, 59 874, 162 754]
        // time      = [5.0, 8.0, 12.0]
        let features = [10.0_f32, 11.0, 12.0, 5.0, 8.0, 12.0];

        // μ = 0 → cell 0 (smallest bytes).
        let mask_data = [true; 3];
        let mask = AllowedMask::new(&mask_data);
        let pick = predictor
            .argmin_masked_with_scorer_in_range(&features, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 0.0 * ms
            })
            .unwrap();
        assert_eq!(pick, Some(0));

        // μ = 1e6 → time dominates; cell 0 still wins (5 ms vs 8/12).
        let pick = predictor
            .argmin_masked_with_scorer_in_range(&features, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 1.0e6 * ms
            })
            .unwrap();
        assert_eq!(pick, Some(0));

        // Skew the time axis: now cell 2 has the smallest time.
        let features2 = [10.0_f32, 11.0, 12.0, 100.0, 50.0, 1.0];
        // μ = 1e6 → time dominates → cell 2.
        let pick = predictor
            .argmin_masked_with_scorer_in_range(&features2, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 1.0e6 * ms
            })
            .unwrap();
        assert_eq!(pick, Some(2));
    }

    #[cfg(feature = "bake")]
    #[test]
    fn predictor_scorer_top_k_rd_vs_time() {
        #[repr(C, align(16))]
        struct Aligned(Vec<u8>);
        let bytes = make_two_head_model();
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);

        // bytes_log[0..3] = [10.0, 11.0, 12.0]
        // time[3..6]       = [100.0, 50.0, 1.0]
        let features = [10.0_f32, 11.0, 12.0, 100.0, 50.0, 1.0];

        let mask_data = [true; 3];
        let mask = AllowedMask::new(&mask_data);

        // μ = 1e6 → time dominates: cell 2 (time=1) < cell 1 (50) < cell 0 (100).
        let top = predictor
            .argmin_masked_top_k_with_scorer_in_range::<3, _>(&features, (0, 3), &mask, |out, i| {
                let bytes = out[i].exp();
                let ms = out[3 + i];
                bytes + 1.0e6 * ms
            })
            .unwrap();
        // First slot is cell 2; second is cell 1.
        assert_eq!(top[0], Some(2));
        assert_eq!(top[1], Some(1));
    }
}
