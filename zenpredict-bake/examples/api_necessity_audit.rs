//! Observations for the pre-publication API audit, not normative regression expectations.
//! Run with `just api-necessity-probe`; see docs/API_NECESSITY_2026-10-28.md.

use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance, Request, Select};
use zenpredict::{
    Activation, AllowedMask, EncodeBudget, MetadataType, Model, PickerStrategy, PreArgminDecision,
    Predictor, QualityTarget, ScoreTransform, Trial, UnachievableZone, UnachievableZones,
    WeightDtype, argmin_masked, best_trial, keys, resolve_pre_argmin,
};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mean = [0.0];
    let scale = [1.0];
    let weights = [1.0, -1.0];
    let biases = [0.0, 4.0];
    let layers = [BakeLayer {
        in_dim: 1,
        out_dim: 2,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &weights,
        biases: &biases,
    }];
    let metadata = [BakeMetadataEntry {
        key: keys::FEATURE_TRANSFORMS,
        kind: MetadataType::Utf8,
        value: b"log1p",
    }];
    let bytes = BakeRequest::builder(0, 0, &mean, &scale, &layers)
        .metadata(&metadata)
        .bake()?;
    let model = Model::from_bytes(&bytes)?;
    let mut predictor = Predictor::new(&model);
    let mask = AllowedMask::new(&[true, true]);
    let wrapper_pick = predictor.argmin_masked(&[3.0], &mask, ScoreTransform::Identity, None)?;
    let transformed = predictor.predict_transformed(&[3.0])?;
    let transformed_pick = argmin_masked(transformed, &mask, ScoreTransform::Identity, None);
    println!("transform: wrapper={wrapper_pick:?}, transformed={transformed_pick:?}");

    let zones = UnachievableZones {
        pixels_feat_idx: 0,
        zones: vec![UnachievableZone {
            pixel_upper: f32::INFINITY,
            ceiling_zq: 90.0,
            fallback_cell: 1,
            fallback_scalar: f32::NAN,
        }],
    };
    let mut allowed = [true, false];
    if let PreArgminDecision::ZoneFallback(fallback) =
        resolve_pre_argmin(&[100.0], 95.0, &zones, &[], &mut allowed)
    {
        println!(
            "zone: fallback={}, allowed={}",
            fallback.cell, allowed[fallback.cell]
        );
    }

    let done = [
        Trial {
            candidate: 0,
            achieved_zq: 70.0,
            bytes: 100,
        },
        Trial {
            candidate: 1,
            achieved_zq: f32::NAN,
            bytes: 200,
        },
    ];
    println!(
        "trial: best={:?}",
        best_trial(&done, QualityTarget::Quality { target_zq: 80.0 })
    );
    println!(
        "budget: zero_passes={}, zero_candidates={}",
        EncodeBudget::passes(0).resolve(PickerStrategy::Auto, 2, 100, 1),
        EncodeBudget::default().resolve(PickerStrategy::Auto, 0, 100, 1)
    );

    let identity = NamedFeature::parse("variance@12345678").ok_or("invalid probe identity")?;
    let cells = [FeatureResult::new(identity, 1.0f32)];
    let wants = [identity];
    let request = Request::new(Select::Features(&wants));
    let gamma = Offer::new(&cells, Provenance::new("probe").with_config(0));
    let linear = Offer::new(&cells, Provenance::new("probe").with_config(1));
    println!(
        "config: gamma_reuses={}, linear_reuses={}, schema_equal={}",
        gamma.satisfies(&request),
        linear.satisfies(&request),
        gamma.schema_hash() == linear.schema_hash()
    );
    Ok(())
}
