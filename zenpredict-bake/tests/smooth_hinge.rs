use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake_from_json_str};

#[test]
fn smooth_hinge_packed_roundtrip_and_pin() {
    for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::I8] {
        let layers = [
            BakeLayer {
                in_dim: 1,
                out_dim: 1,
                activation: Activation::SmoothHinge,
                dtype,
                weights: &[1.],
                biases: &[0.],
            },
            BakeLayer {
                in_dim: 1,
                out_dim: 1,
                activation: Activation::Identity,
                dtype,
                weights: &[-1.],
                biases: &[100.],
            },
        ];
        let bytes = BakeRequest::builder(0, 0, &[0.], &[1.], &layers)
            .bake()
            .unwrap();
        let model = Model::from_bytes(&bytes).unwrap();
        assert_eq!(
            model.layers().next().unwrap().activation,
            Activation::SmoothHinge
        );
        let mut predictor = Predictor::new(&model);
        for (z, h) in [
            (-1., 0.),
            (0., 0.),
            (0.0625, 0.015625),
            (0.125, 0.0625),
            (1., 0.9375),
        ] {
            assert_eq!(
                predictor.predict(&[z]).unwrap()[0].to_bits(),
                (100.0f32 - h).to_bits()
            );
        }
    }
}

#[test]
fn smooth_hinge_json_wire_code_and_unknown_refusal() {
    let bytes = bake_from_json_str(r#"{"schema_hash":0,"scaler_mean":[0.0],"scaler_scale":[1.0],"layers":[{"in_dim":1,"out_dim":1,"activation":"smooth_hinge","dtype":"f32","weights":[1.0],"biases":[0.0]}]}"#).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    assert_eq!(model.layers().next().unwrap().activation as u8, 3);
    let invalid = r#"{"schema_hash":0,"scaler_mean":[0.0],"scaler_scale":[1.0],"layers":[{"in_dim":1,"out_dim":1,"activation":"smooth_hinge_tau2","dtype":"f32","weights":[1.0],"biases":[0.0]}]}"#;
    let error = bake_from_json_str(invalid).unwrap_err().to_string();
    assert!(error.contains("unknown variant"), "{error}");
}
