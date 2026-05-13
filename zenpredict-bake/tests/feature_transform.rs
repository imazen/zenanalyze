use zenpredict::*;
use zenpredict_bake::*;


mod feature_transform_tests {
    //! Tests for issue #52 — `zentrain.feature_transforms` metadata
    //! key + `Predictor::predict_*_transformed` runtime helpers.
    //!
    //! Coverage:
    //! - Round-trip a bake with `[Identity, Log, Log1p, Identity]`
    //!   transforms; verify `predict_transformed` applies them and
    //!   `predict` does NOT (the train/serve skew that #52 closes).
    //! - Bake without the key parses with `feature_transforms() ==
    //!   None` and `predict_transformed` is identical to `predict`
    //!   (no allocation, no copy).
    //! - Length mismatch and unknown-token bakes hard-fail at load.
    //! - `predict_with_specs_transformed` applies transforms AND the
    //!   spec pipeline.
    //! - Numeric agreement: applying the transform manually before
    //!   `predict()` matches `predict_transformed()` byte-for-byte.
    //!   This is the small synthetic train/serve parity check the
    //!   #52 plan asks for.
    use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake_v2};
    use zenpredict::MetadataType;
    use zenpredict::*;

    #[repr(C, align(16))]
    struct Aligned(Vec<u8>);

    /// Identity-mapping 4-input → 4-output single-layer model: each
    /// output `i` simply forwards `features[i]` (after the scaler,
    /// which is mean=0/scale=1, also identity). Useful for asserting
    /// that the transform fired by inspecting the output directly.
    fn make_identity_passthrough(metadata: &[BakeMetadataEntry<'_>]) -> Vec<u8> {
        let scaler_mean = [0.0f32; 4];
        let scaler_scale = [1.0f32; 4];
        // 4 × 4 row-major identity weights.
        let w0 = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b0 = [0.0f32; 4];
        let layers = [BakeLayer {
            in_dim: 4,
            out_dim: 4,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        }];
        bake_v2(&BakeRequest {
            schema_hash: 0xfeed_face_dead_beef,
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &layers,
            feature_bounds: &[],
            metadata,
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
        })
        .unwrap()
    }

    #[test]
    fn feature_transforms_absent_means_none() {
        // No metadata at all → feature_transforms() returns None.
        let bytes = make_identity_passthrough(&[]);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        assert!(model.feature_transforms().is_none());
        assert!(!model.has_nontrivial_feature_transforms());
    }

    #[test]
    fn feature_transforms_round_trip() {
        let txt = b"identity\nlog\nlog1p\nidentity";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let ts = model.feature_transforms().expect("present");
        assert_eq!(
            ts,
            &[
                FeatureTransform::Identity,
                FeatureTransform::Log,
                FeatureTransform::Log1p,
                FeatureTransform::Identity,
            ]
        );
        assert!(model.has_nontrivial_feature_transforms());
    }

    #[test]
    fn predict_transformed_applies_log_and_log1p() {
        // [identity, log, log1p, identity] — feed [1.0, e, 0.0, 7.0].
        // Pass-through model means output[i] = transform(features[i]).
        let txt = b"identity\nlog\nlog1p\nidentity";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);
        let e = core::f32::consts::E;
        let raw = [1.0f32, e, 0.0, 7.0];
        let out = predictor.predict_transformed(&raw).unwrap().to_vec();
        assert!((out[0] - 1.0).abs() < 1e-6, "identity passthrough");
        assert!((out[1] - 1.0).abs() < 1e-5, "ln(e) ≈ 1.0, got {}", out[1]);
        assert!((out[2] - 0.0).abs() < 1e-6, "ln(1+0) = 0");
        assert!((out[3] - 7.0).abs() < 1e-6, "identity passthrough");
    }

    #[test]
    fn predict_does_not_apply_transforms() {
        // Confirm the OLD path is left untouched — calling `predict`
        // on a transforms-bearing bake gives the raw (untransformed)
        // outputs. This is the train/serve skew #52 closes; calling
        // `predict` instead of `predict_transformed` is now an
        // explicit caller bug.
        let txt = b"identity\nlog\nlog1p\nidentity";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);
        let raw = [1.0f32, core::f32::consts::E, 0.0, 7.0];
        let out = predictor.predict(&raw).unwrap().to_vec();
        // Pass-through model + no transform → output equals input.
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - core::f32::consts::E).abs() < 1e-5);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn manual_transform_then_predict_matches_predict_transformed() {
        // Train/serve parity check (#52): the value the runtime
        // produces via `predict_transformed(raw)` must equal the
        // value it would produce via `predict(transform(raw))`. If
        // these diverge, the trainer's scaler stats are being
        // applied to a different distribution than the one the
        // network saw at fit time.
        let txt = b"log\nlog1p\nidentity\nlog";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let model_a = Model::from_bytes(&aligned.0).unwrap();
        let model_b = Model::from_bytes(&aligned.0).unwrap();
        let mut p_auto = Predictor::new(model_a);
        let mut p_manual = Predictor::new(model_b);

        let raw = [3.5f32, 12.0, -2.7, 0.5];
        // Auto path.
        let auto = p_auto.predict_transformed(&raw).unwrap().to_vec();
        // Manual path: pre-transform features, call plain predict.
        let transforms = [
            FeatureTransform::Log,
            FeatureTransform::Log1p,
            FeatureTransform::Identity,
            FeatureTransform::Log,
        ];
        let mut transformed = [0.0f32; 4];
        apply_feature_transforms(&transforms, &raw, &mut transformed).unwrap();
        let manual = p_manual.predict(&transformed).unwrap().to_vec();

        assert_eq!(auto.len(), manual.len());
        for (a, b) in auto.iter().zip(manual.iter()) {
            // Both paths run the same forward pass on the same
            // post-transform inputs → bit-exact equality, not just
            // approximate.
            assert_eq!(a.to_bits(), b.to_bits(), "auto={a} manual={b}");
        }
    }

    #[test]
    fn predict_transformed_no_transforms_matches_predict() {
        // Bakes without `feature_transforms` MUST treat
        // `predict_transformed` as a synonym for `predict`, with
        // bit-exact output.
        let bytes = make_identity_passthrough(&[]);
        let aligned = Aligned(bytes);
        let model_a = Model::from_bytes(&aligned.0).unwrap();
        let model_b = Model::from_bytes(&aligned.0).unwrap();
        let mut p_a = Predictor::new(model_a);
        let mut p_b = Predictor::new(model_b);
        let raw = [1.5f32, -3.2, 4.0, 0.25];
        let a = p_a.predict_transformed(&raw).unwrap().to_vec();
        let b = p_b.predict(&raw).unwrap().to_vec();
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn predict_with_specs_transformed_runs_full_pipeline() {
        // Smoke-test that the spec-passthrough variant also fires
        // the transform. The bake here has no `output_specs` so
        // the result should be `Override(transformed_value)` for
        // each output.
        let txt = b"identity\nlog\nidentity\nidentity";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(model);
        let e = core::f32::consts::E;
        let raw = [1.0f32, e, 2.0, 3.0];
        let out = predictor.predict_with_specs_transformed(&raw).unwrap();
        match out[1] {
            OutputValue::Override(v) => {
                assert!((v - 1.0).abs() < 1e-5, "ln(e) ≈ 1.0, got {v}");
            }
            OutputValue::Default => panic!("expected Override"),
        }
    }

    #[test]
    fn unknown_token_rejected_at_load() {
        let txt = b"identity\nlog\nbogus\nidentity";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(
            matches!(err, PredictError::UnknownFeatureTransform),
            "got {err:?}"
        );
    }

    #[test]
    fn length_mismatch_rejected_at_load() {
        // 4 inputs but 3 transforms.
        let txt = b"identity\nlog\nidentity";
        let metadata = [BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: txt,
        }];
        let bytes = make_identity_passthrough(&metadata);
        let aligned = Aligned(bytes);
        let err = Model::from_bytes(&aligned.0).unwrap_err();
        assert!(
            matches!(
                err,
                PredictError::FeatureTransformsLenMismatch {
                    expected: 4,
                    got: 3
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn token_round_trip() {
        for &t in &[
            FeatureTransform::Identity,
            FeatureTransform::Log,
            FeatureTransform::Log1p,
        ] {
            let s = t.as_token();
            // No public `from_token` — exercise via parse_feature_transforms.
            let txt = format!("{s}\n{s}\n{s}\n{s}");
            let metadata = [BakeMetadataEntry {
                key: keys::FEATURE_TRANSFORMS,
                kind: MetadataType::Utf8,
                value: txt.as_bytes(),
            }];
            let bytes = make_identity_passthrough(&metadata);
            let aligned = Aligned(bytes);
            let model = Model::from_bytes(&aligned.0).unwrap();
            // Skip when `t == Identity` — bake_picker.py omits the
            // key when every transform is identity, but our raw
            // baker doesn't do that filtering, so we get the full
            // round-trip regardless of variant.
            let parsed = model.feature_transforms().expect("present");
            assert_eq!(parsed, &[t; 4]);
        }
    }
}