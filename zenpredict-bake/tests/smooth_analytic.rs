use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake_from_json_str};

fn reference(x: f64, activation: Activation) -> f64 {
    match activation {
        Activation::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        Activation::Silu => x / (1.0 + (-x).exp()),
        _ => unreachable!(),
    }
}

#[test]
fn analytic_bake_matches_double_reference_including_extremes() {
    for (activation, name) in [
        (Activation::Softplus, "softplus"),
        (Activation::Silu, "silu"),
    ] {
        let json = format!(
            r#"{{"schema_hash":0,"scaler_mean":[0.0],"scaler_scale":[1.0],"layers":[{{"in_dim":1,"out_dim":1,"activation":"{name}","dtype":"f32","weights":[1.0],"biases":[0.0]}}]}}"#
        );
        let bytes = bake_from_json_str(&json).unwrap();
        let model = Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.layers().next().unwrap().activation, activation);
        let mut predictor = Predictor::new(&model);
        let inputs = [
            -f32::MAX,
            -100.0,
            -20.0,
            -2.0,
            -1.0,
            -0.001,
            -0.0,
            0.0,
            0.001,
            1.0,
            20.0,
            100.0,
            f32::MAX,
        ];
        for x in inputs {
            let actual = predictor.predict(&[x]).unwrap()[0];
            let expected = reference(f64::from(x), activation) as f32;
            assert!(actual.is_finite(), "{activation:?}({x})={actual}");
            assert!(
                (actual - expected).abs() <= 3e-7 * expected.abs().max(1e-30),
                "{activation:?}({x}): {actual} != {expected}"
            );
        }
    }
}

#[test]
fn explicit_anchor_subtraction_preserves_identity_and_monotone_domain() {
    for activation in [Activation::Softplus, Activation::Silu] {
        for bias in [0.0, 0.01, 1.0, 20.0] {
            for temperature in [0.125, 1.0, 8.0] {
                let weights = [1.0 / temperature, 0.0];
                let biases = [bias / temperature; 2];
                let subtraction = [temperature, -temperature];
                let layers = [
                    BakeLayer {
                        in_dim: 1,
                        out_dim: 2,
                        activation,
                        dtype: WeightDtype::F32,
                        weights: &weights,
                        biases: &biases,
                    },
                    // Subtract branches before adding the identity score.
                    BakeLayer {
                        in_dim: 2,
                        out_dim: 1,
                        activation: Activation::Identity,
                        dtype: WeightDtype::F32,
                        weights: &subtraction,
                        biases: &[0.0],
                    },
                    BakeLayer {
                        in_dim: 1,
                        out_dim: 1,
                        activation: Activation::Identity,
                        dtype: WeightDtype::F32,
                        weights: &[-1.0],
                        biases: &[100.0],
                    },
                ];
                let bytes = BakeRequest::builder(0, 0, &[0.0], &[1.0], &layers)
                    .bake()
                    .unwrap();
                let model = Model::from_bytes(&bytes).unwrap();
                let mut predictor = Predictor::new(&model);
                assert_eq!(
                    predictor.predict(&[0.0]).unwrap()[0].to_bits(),
                    100.0f32.to_bits()
                );
                let mut last = 100.0;
                for i in 0..1024 {
                    let x = i as f32 / 128.0;
                    let score = predictor.predict(&[x]).unwrap()[0];
                    assert!(
                        score <= last,
                        "{activation:?} bias={bias} temperature={temperature}: {score}>{last}"
                    );
                    last = score;
                }
            }
        }
    }
}

#[test]
fn ordinary_silu_is_not_monotone() {
    assert!(reference(-2.0, Activation::Silu) < reference(-3.0, Activation::Silu));
}
