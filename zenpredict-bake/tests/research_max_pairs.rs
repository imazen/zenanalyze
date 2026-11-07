use zenpredict::{Activation, MetadataType, Model, PredictError, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

fn fixture(declaration: Option<&str>, tied: bool) -> Vec<u8> {
    let weights = [
        37.33797, -18.191578, 0.80901235, 80.138084, 0.39394888, 64.71758,
    ];
    let middle = [1., if tied { 1. } else { 2. }];
    let layers = [
        BakeLayer {
            in_dim: 3,
            out_dim: 2,
            activation: if declaration.is_some() {
                Activation::ResearchReluMaxPairs
            } else {
                Activation::Relu
            },
            dtype: WeightDtype::F32,
            weights: &weights,
            biases: &[0., 0.],
        },
        BakeLayer {
            in_dim: 2,
            out_dim: 1,
            activation: Activation::Relu,
            dtype: WeightDtype::F32,
            weights: &middle,
            biases: &[0.],
        },
        BakeLayer {
            in_dim: 1,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &[-1.],
            biases: &[100.],
        },
    ];
    let metadata = declaration.map(|s| BakeMetadataEntry {
        key: "zenpredict.research_max_pairs",
        kind: MetadataType::Utf8,
        value: s.as_bytes(),
    });
    bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &[0.; 3],
        scaler_scale: &[1.; 3],
        layers: &layers,
        feature_bounds: &[],
        metadata: metadata.as_slice(),
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: Some(&[&[0, 1], &[0]]),
    })
    .unwrap()
}

#[test]
fn stable_maximum_repairs_a_real_float32_order_reversal() {
    let low = [16.118246, 80.85465, 69.74723];
    let high = [16.118256, 80.85465, 69.74723];
    let legacy = Model::from_bytes(&fixture(None, true)).unwrap();
    let mut old = Predictor::new(&legacy);
    let a = old.predict(&low).unwrap()[0];
    let b = old.predict(&high).unwrap()[0];
    assert!(
        b > a,
        "fixture must expose the original rounded-dense reversal: {a}, {b}"
    );
    let stable = Model::from_bytes(&fixture(Some("relu-max-pairs-v1\n0 1\n"), true)).unwrap();
    let mut new = Predictor::new(&stable);
    assert_eq!(new.predict(&[0.; 3]).unwrap()[0], 100.);
    let a = new.predict(&low).unwrap()[0];
    let b = new.predict(&high).unwrap()[0];
    assert!(b <= a);
    // Every adjacent f32 value in a neighbourhood of the failing coordinate.
    let mut last = f32::INFINITY;
    for bits in low[0].to_bits() - 1024..=high[0].to_bits() + 1024 {
        let score = new
            .predict(&[f32::from_bits(bits), low[1], low[2]])
            .unwrap()[0];
        assert!(score <= last);
        last = score;
    }
    assert!(matches!(
        new.predict(&[-1., 0., 0.]),
        Err(PredictError::InvalidResearchMaxPair { .. })
    ));
}

#[test]
fn invalid_pair_claims_are_refused_and_unmarked_models_are_unchanged() {
    for text in [
        "wrong\n0 1",
        "relu-max-pairs-v1",
        "relu-max-pairs-v1\n0 2",
        "relu-max-pairs-v1\n0 1\n0 1",
        "relu-max-pairs-v1\n1 0",
    ] {
        assert!(matches!(
            Model::from_bytes(&fixture(Some(text), true)),
            Err(PredictError::InvalidResearchMaxPair { .. })
        ));
    }
    assert!(matches!(
        Model::from_bytes(&fixture(Some("relu-max-pairs-v1\n0 1"), false)),
        Err(PredictError::InvalidResearchMaxPair { .. })
    ));
    let a = fixture(None, true);
    let b = fixture(None, true);
    assert_eq!(a, b);
}
