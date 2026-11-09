use zenpredict::input_program::{DamageShape, InputOp, InputProgram, METADATA_KEY, ReferenceGate};
use zenpredict::{Activation, MetadataType, Model, PredictError, Predictor, WeightDtype};
use zenpredict_bake::input_program::bake_with_input_program;
use zenpredict_bake::{
    BakeLayer, BakeMetadataEntry, BakeRequest, append_metadata_utf8, bake, bake_from_json_str,
};
fn program() -> InputProgram {
    InputProgram::new(
        3,
        vec![
            InputOp::Source { index: 0 },
            InputOp::Source { index: 2 },
            InputOp::Product { left: 0, right: 1 },
            InputOp::Shape {
                input: 2,
                shape: DamageShape::new([0.25, 1.], [2., 0.5, 3.]).unwrap(),
            },
            InputOp::Shape {
                input: 0,
                shape: DamageShape::new([0.25, 1.], [2., 0.5, 3.]).unwrap(),
            },
            InputOp::Gate {
                input: 1,
                gate: ReferenceGate::new([1., 0.25, 2.]).unwrap(),
            },
            InputOp::Product { left: 4, right: 5 },
        ],
        vec![3, 6],
    )
    .unwrap()
}
fn fixture(compressed: bool, permuted: bool) -> Vec<u8> {
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 2,
        activation: Activation::Softplus,
        dtype: WeightDtype::F32,
        weights: &[1., 2., 3., 4.],
        biases: &[0.1, -0.25],
    }];
    let entries = [BakeMetadataEntry {
        key: zenpredict::keys::FEATURE_TRANSFORMS,
        kind: MetadataType::Utf8,
        value: b"log1p\nidentity",
    }];
    let mut req = BakeRequest::new(123, 0, &[0.1, 0.2], &[2., 3.], &layers);
    req.compressed = compressed;
    req.metadata = &entries;
    if permuted {
        req.feature_order = Some(&[1, 0]);
        req.output_order = Some(&[1, 0]);
    }
    bake_with_input_program(&req, &program()).unwrap()
}
#[test]
fn public_source_path_preserves_composition_transform_scaler_softplus_and_selection() {
    for compressed in [false, true] {
        for permuted in [false, true] {
            let bytes = fixture(compressed, permuted);
            let model = Model::from_bytes_with_schema(&bytes, 123).unwrap();
            assert_eq!(model.version(), 4);
            assert_eq!(model.n_source_inputs(), 3);
            assert_eq!(model.n_inputs(), 2);
            assert_eq!(model.input_program(), Some(&program()));
            let mut predictor = Predictor::new(&model);
            let mut prepared = Predictor::new(&model);
            for source in [
                [2., 99., 0.25],
                [0., -9., 0.],
                [0.125, 0., 1.],
                [0.5, 7., 0.75],
            ] {
                let p = program();
                let mut compact = [0.; 2];
                p.apply(&source, &mut vec![0.; p.scratch_len()], &mut compact)
                    .unwrap();
                let expected = prepared.predict_transformed(&compact).unwrap().to_vec();
                assert_eq!(
                    predictor
                        .predict_from_source(&source)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                let actual = predictor.predict_from_source(&source).unwrap();
                let a = zenpredict::argmin_masked(
                    actual,
                    &zenpredict::AllowedMask::new(&[true, true]),
                    zenpredict::ScoreTransform::Identity,
                    None,
                );
                let b = zenpredict::argmin_masked(
                    &expected,
                    &zenpredict::AllowedMask::new(&[true, true]),
                    zenpredict::ScoreTransform::Identity,
                    None,
                );
                assert_eq!(a, b);
            }
            assert!(predictor.predict_from_source(&[1., 2.]).is_err());
            let changed = append_metadata_utf8(&bytes, "test.provenance", "roundtrip").unwrap();
            let reparsed = Model::from_bytes(&changed).unwrap();
            assert_eq!(reparsed.input_program(), model.input_program());
            let mut other = Predictor::new(&reparsed);
            assert_eq!(
                other.predict_from_source(&[2., 0., 0.25]).unwrap(),
                predictor.predict_from_source(&[2., 0., 0.25]).unwrap()
            );
        }
    }
}
#[test]
fn joint_product_differs_from_separate_shape_and_preserves_zero_anchor() {
    let p = program();
    let mut scratch = vec![0.; p.scratch_len()];
    let mut out = [0.; 2];
    p.apply(&[2., 99., 0.25], &mut scratch, &mut out).unwrap();
    assert_eq!(out, [0.625, 2.421875]);
    assert_ne!(out[0], out[1]);
    p.apply(&[0., 99., 0.25], &mut scratch, &mut out).unwrap();
    assert_eq!(out, [0., 0.]);
}
#[test]
fn v3_is_unchanged_and_required_programs_cannot_be_silently_ignored() {
    let layers = [BakeLayer {
        in_dim: 1,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &[2.],
        biases: &[1.],
    }];
    let plain = bake(&BakeRequest::new(0, 0, &[0.], &[1.], &layers)).unwrap();
    let old = Model::from_bytes(&plain).unwrap();
    assert_eq!(old.version(), 3);
    assert!(old.input_program().is_none());
    let mut predictor = Predictor::new(&old);
    assert_eq!(predictor.predict_from_source(&[2.]).unwrap(), &[5.]);
    let mut missing = plain.clone();
    missing[4..6].copy_from_slice(&4u16.to_le_bytes());
    assert!(matches!(
        Model::from_bytes(&missing),
        Err(PredictError::InvalidInputProgram { .. })
    ));
    let mut downgrade = fixture(false, false);
    downgrade[4..6].copy_from_slice(&3u16.to_le_bytes());
    assert!(matches!(
        Model::from_bytes(&downgrade),
        Err(PredictError::InvalidInputProgram { .. })
    ));
    let encoded = program().to_bytes();
    let entries = [BakeMetadataEntry {
        key: METADATA_KEY,
        kind: MetadataType::Bytes,
        value: &encoded,
    }];
    let mut req = BakeRequest::new(0, 0, &[0.], &[1.], &layers);
    req.metadata = &entries;
    assert!(bake(&req).is_err()); // two program outputs cannot feed one input
    assert!(bake_with_input_program(&req, &program()).is_err()); // duplicate mandatory key
}
#[test]
fn json_program_is_typed_validated_and_available_to_cli() {
    let valid = r#"{"schema_hash":0,"scaler_mean":[0],"scaler_scale":[1],"input_program":{"source_count":2,"nodes":[{"op":"source","index":0},{"op":"source","index":1},{"op":"product","left":0,"right":1}],"outputs":[2]},"layers":[{"in_dim":1,"out_dim":1,"activation":"softplus","dtype":"f32","weights":[1],"biases":[0]}]}"#;
    let bytes = bake_from_json_str(valid).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    assert!((p.predict_from_source(&[2., 3.]).unwrap()[0] - (1. + 6f32.exp()).ln()).abs() < 1e-6);
    for invalid in [
        valid.replace("\"left\":0", "\"left\":2"),
        valid.replace("\"right\":1", "\"rgiht\":1"),
        valid.replace("\"product\"", "\"unknown\""),
    ] {
        assert!(bake_from_json_str(&invalid).is_err());
    }
}
#[test]
fn source_specs_and_overrides_use_the_same_forward_path() {
    let bytes = fixture(false, false);
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    let expected = p.predict_from_source(&[2., 0., 0.25]).unwrap().to_vec();
    let specs = p.predict_with_specs_from_source(&[2., 0., 0.25]).unwrap();
    for (a, b) in specs.iter().zip(expected) {
        assert!(matches!(a,zenpredict::OutputValue::Override(x) if *x==b));
    }
}

#[test]
fn attaching_program_preserves_existing_network_sections_and_scores() {
    use zenpredict_bake::input_program::attach_input_program;
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 2,
        activation: Activation::Softplus,
        dtype: WeightDtype::F16,
        weights: &[0.3333, 2.1, 3.7, 0.004],
        biases: &[0.1, -0.25],
    }];
    for compressed in [false, true] {
        let mut req = BakeRequest::new(234, 0, &[0.1, 0.2], &[2., 3.], &layers);
        req.compressed = compressed;
        req.feature_order = Some(&[1, 0]);
        let original = bake(&req).unwrap();
        let old = Model::from_bytes(&original).unwrap();
        let attached = attach_input_program(&original, &program()).unwrap();
        let new = Model::from_bytes(&attached).unwrap();
        assert_eq!(old.scaler_mean(), new.scaler_mean());
        assert_eq!(old.scaler_scale(), new.scaler_scale());
        let a = old.layer(0);
        let b = new.layer(0);
        assert_eq!(a.biases, b.biases);
        match (a.weights, b.weights) {
            (zenpredict::WeightStorage::F16(a), zenpredict::WeightStorage::F16(b)) => {
                assert_eq!(a, b)
            }
            _ => panic!("dtype changed"),
        }
        let mut left = Predictor::new(&old);
        let mut right = Predictor::new(&new);
        for source in [[2., 0., 0.25], [0., 1., 1.], [0.25, 2., 0.5]] {
            let mut compact = [0.; 2];
            program()
                .apply(&source, &mut [0.; 7], &mut compact)
                .unwrap();
            assert_eq!(
                left.predict_transformed(&compact).unwrap(),
                right.predict_from_source(&source).unwrap()
            );
        }
    }
}
#[test]
fn hidden_permutations_preserve_semantic_max_pair_references() {
    let layers = [
        BakeLayer {
            in_dim: 1,
            out_dim: 2,
            activation: Activation::ResearchReluMaxPairs,
            dtype: WeightDtype::F32,
            weights: &[3., -2.],
            biases: &[0., 0.],
        },
        BakeLayer {
            in_dim: 2,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &[1., 1.],
            biases: &[0.],
        },
    ];
    let md = [BakeMetadataEntry {
        key: "zenpredict.research_max_pairs",
        kind: MetadataType::Utf8,
        value: b"relu-max-pairs-v1\n0 1\n",
    }];
    let p = InputProgram::new(1, vec![InputOp::Source { index: 0 }], vec![0]).unwrap();
    for perm in [None, Some(&[1u32, 0][..]), Some(&[0u32, 1][..])] {
        let slots = perm.map(|p| [p]);
        let mut req = BakeRequest::new(0, 0, &[0.], &[1.], &layers);
        req.metadata = &md;
        req.hu_permutations = slots.as_ref().map(|p| p.as_slice());
        let bytes = bake_with_input_program(&req, &p).unwrap();
        let m = Model::from_bytes(&bytes).unwrap();
        let mut f = Predictor::new(&m);
        assert_eq!(f.predict_from_source(&[2.]).unwrap(), &[6.]);
    }
}
#[test]
fn invalid_builder_inputs_return_errors_instead_of_panicking() {
    let empty = BakeRequest::new(0, 0, &[], &[], &[]);
    assert!(zenpredict_bake::bake_optimized(&empty).is_err());
    let layers = [
        BakeLayer {
            in_dim: 1,
            out_dim: 2,
            activation: Activation::Relu,
            dtype: WeightDtype::F32,
            weights: &[1., 2.],
            biases: &[0., 0.],
        },
        BakeLayer {
            in_dim: 2,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &[1., 2.],
            biases: &[0.],
        },
    ];
    for permutations in [
        vec![],
        vec![&[0u32][..]],
        vec![&[0u32, 0][..]],
        vec![&[0u32, 2][..]],
    ] {
        let mut req = BakeRequest::new(0, 0, &[0.], &[1.], &layers);
        req.hu_permutations = Some(&permutations);
        assert!(bake(&req).is_err());
    }
}

#[test]
fn source_program_composes_with_existing_expanders_and_real_output_overrides() {
    let json = r#"{"schema_hash":0,"scaler_mean":[0,0],"scaler_scale":[1,1],"input_program":{"source_count":2,"nodes":[{"op":"source","index":1}],"outputs":[0]},"metadata":[{"key":"zentrain.feature_transforms","type":"utf8","text":"sinusoidal"},{"key":"zentrain.feature_transform_params","type":"utf8","text":"1"}],"layers":[{"in_dim":2,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1,0],"biases":[0]}],"output_specs":[{"bounds":[-2,2],"transform":"round"}],"sparse_overrides":[{"idx":0,"value":7}]}"#;
    let bytes = bake_from_json_str(json).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    assert_eq!(model.n_source_inputs(), 2);
    assert_eq!(model.input_program().unwrap().outputs().len(), 1);
    for x in [0., 0.125, 0.25, 0.5] {
        let expected = p.predict_transformed(&[x]).unwrap()[0];
        assert_eq!(
            p.predict_from_source(&[99., x]).unwrap()[0].to_bits(),
            expected.to_bits()
        );
        assert!(matches!(
            p.predict_with_specs_from_source(&[99., x]).unwrap()[0],
            zenpredict::OutputValue::Override(7.)
        ));
    }
}

#[test]
fn drop_only_pipelines_need_no_parameter_metadata() {
    let json = r#"{"schema_hash":0,"scaler_mean":[0],"scaler_scale":[1],"input_program":{"source_count":2,"nodes":[{"op":"source","index":0},{"op":"source","index":1}],"outputs":[0,1]},"metadata":[{"key":"zentrain.feature_transforms","type":"utf8","text":"drop\nidentity"}],"layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[2],"biases":[1]}]}"#;
    let bytes = bake_from_json_str(json).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    for x in [0., 1., 2.] {
        assert_eq!(p.predict_from_source(&[999., x]).unwrap(), &[2. * x + 1.]);
        assert_eq!(p.predict_transformed(&[999., x]).unwrap(), &[2. * x + 1.]);
    }
}

#[test]
fn excessive_or_zero_dimensions_are_errors_before_optimizer_indexing() {
    for (input, output) in [(0, 1), (1, 0), (usize::MAX, 2), (2, usize::MAX), (65537, 1)] {
        let layers = [BakeLayer {
            in_dim: input,
            out_dim: output,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &[],
            biases: &[],
        }];
        let req = BakeRequest::new(0, 0, &[], &[], &layers);
        assert!(bake(&req).is_err());
        assert!(zenpredict_bake::bake_optimized(&req).is_err());
    }
}
