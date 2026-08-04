//! `FeatureTransform::Drop` — the arity-0 sink that makes dead-column
//! pruning expressible without changing the caller's feature width.
//!
//! A bake with `drop` on line `k` of `zentrain.feature_transforms`
//! still takes `caller_input_width()` features from the caller, but
//! stores no `W0` row, no `scaler_mean[k]` / `scaler_scale[k]`, and no
//! `feature_bounds[k]` for that input. The tests below pin the two
//! properties the pruner depends on:
//!
//! 1. **Weight-dead pruning is bit-identical.** Dropping an input whose
//!    `W0` row is exactly zero produces the same `f32` bits as leaving
//!    it in. This is the class-1 guarantee in
//!    `zensim/zensim-validate/src/prune.rs`.
//! 2. **Constant-forced pruning is exact in real arithmetic.** Dropping
//!    an input whose transform forces a constant, with the constant's
//!    contribution folded into the layer-0 bias, reproduces the full
//!    model to fp tolerance (the fold changes the summation order, so
//!    it is *not* promised bit-identical). Class 2.
//!
//! Plus the bake-time guards: `drop` takes no params, and Sinusoidal is
//! no longer allowed to spell arity 0 by having none.

use zenpredict::MetadataType;
use zenpredict::*;
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

/// A 2-layer regressor: `n_in → 2 (leaky-relu) → 1 (identity)`.
/// `w0` is row-major `n_in × 2`.
#[allow(clippy::too_many_arguments)]
fn make_two_layer(
    scaler_mean: &[f32],
    scaler_scale: &[f32],
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    metadata: &[BakeMetadataEntry<'_>],
) -> Vec<u8> {
    let n_in = scaler_mean.len();
    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: 2,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: w0,
            biases: b0,
        },
        BakeLayer {
            in_dim: 2,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: w1,
            biases: b1,
        },
    ];
    bake(&BakeRequest {
        schema_hash: 0x0bad_c0de_0bad_c0de,
        flags: 0,
        scaler_mean,
        scaler_scale,
        layers: &layers,
        feature_bounds: &[],
        metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("bake")
}

fn transforms_md<'a>(t: &'a [u8], p: &'a [u8]) -> [BakeMetadataEntry<'a>; 2] {
    [
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORMS,
            kind: MetadataType::Utf8,
            value: t,
        },
        BakeMetadataEntry {
            key: keys::FEATURE_TRANSFORM_PARAMS,
            kind: MetadataType::Utf8,
            value: p,
        },
    ]
}

const PROBES: [[f32; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0],
    [-4.5, 17.25, 0.125],
    [1e6, -1e-6, 42.0],
    [-0.25, -0.5, -0.75],
    [3.3, 9.9, -12.5],
];

#[test]
fn drop_token_round_trips_and_splits_caller_width_from_n_inputs() {
    let w0 = [1.0f32, -2.0, 0.0, 0.0, 0.5, 0.25];
    let b0 = [0.1f32, -0.2];
    let w1 = [1.5f32, -0.5];
    let b1 = [0.75f32];
    // Pruned twin: raw width 3, input 1 dropped ⇒ layer-0 in_dim 2.
    let pw0 = [1.0f32, -2.0, 0.5, 0.25];
    let md = transforms_md(b"identity\ndrop\nidentity", b"\n\n");
    let bytes = make_two_layer(&[0.0, 0.0], &[1.0, 1.0], &pw0, &b0, &w1, &b1, &md);
    let aligned = Aligned(bytes);
    let model = Model::from_bytes(&aligned.0).expect("load pruned bake");

    assert_eq!(model.n_inputs(), 2, "post-transform layer-0 width");
    assert_eq!(model.caller_input_width(), 3, "raw width the caller feeds");
    assert_eq!(
        model.feature_transforms().expect("present"),
        &[
            FeatureTransform::Identity,
            FeatureTransform::Drop,
            FeatureTransform::Identity
        ]
    );
    assert!(model.has_expander_feature_transforms());
    assert_eq!(model.expanded_input_dim(), 2);
    // Sanity: `w0` is only referenced by the full-model tests below.
    let _ = w0;
}

#[test]
fn weight_dead_prune_is_bit_identical() {
    // CLASS 1. Full model: 3 inputs, `W0` row 1 is exactly zero, so
    // input 1 contributes nothing for ANY value. Pruned twin drops it.
    let mean = [0.5f32, -3.0, 2.0];
    let scale = [2.0f32, 0.25, 1.5];
    let w0_full = [
        1.0f32, -2.0, // input 0
        0.0, 0.0, // input 1 — dead
        0.5, 0.25, // input 2
    ];
    let b0 = [0.1f32, -0.2];
    let w1 = [1.5f32, -0.5];
    let b1 = [0.75f32];

    // The full twin declares all-identity transforms so the byte
    // comparison below is apples-to-apples (both bakes carry the
    // transforms metadata; only the dropped row/scaler/token differ).
    let full_md = transforms_md(b"identity\nidentity\nidentity", b"\n\n");
    let full = Aligned(make_two_layer(
        &mean, &scale, &w0_full, &b0, &w1, &b1, &full_md,
    ));
    let full_model = Model::from_bytes(&full.0).expect("load full");
    let mut full_pred = Predictor::new(&full_model);

    // Pruned: rows 0 and 2 of W0, scaler minus index 1, `drop` on 1.
    let w0_pruned = [1.0f32, -2.0, 0.5, 0.25];
    let md = transforms_md(b"identity\ndrop\nidentity", b"\n\n");
    let pruned = Aligned(make_two_layer(
        &[mean[0], mean[2]],
        &[scale[0], scale[2]],
        &w0_pruned,
        &b0,
        &w1,
        &b1,
        &md,
    ));
    let pruned_model = Model::from_bytes(&pruned.0).expect("load pruned");
    let mut pruned_pred = Predictor::new(&pruned_model);

    assert!(
        pruned.0.len() < full.0.len(),
        "pruned bake must be smaller: {} vs {}",
        pruned.0.len(),
        full.0.len()
    );

    for probe in PROBES {
        let a = full_pred.predict_transformed(&probe).expect("full predict")[0];
        let b = pruned_pred
            .predict_transformed(&probe)
            .expect("pruned predict")[0];
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "weight-dead prune must be BIT-identical at {probe:?}: {a} vs {b}"
        );
    }
}

#[test]
fn transform_forced_constant_prune_folds_into_bias() {
    // CLASS 2. Input 1 carries `winsor_p99` with p_lo == p_hi == 4.0,
    // so its post-transform value is the constant 4.0 no matter what
    // the caller passes. Its layer-0 contribution is therefore the
    // constant `((4 - mean1)/scale1) * W0[1,:]`, which the pruner folds
    // into `b0` before dropping the column.
    let mean = [0.5f32, -3.0, 2.0];
    let scale = [2.0f32, 0.25, 1.5];
    let w0_full = [1.0f32, -2.0, 0.75, 1.25, 0.5, 0.25];
    let b0 = [0.1f32, -0.2];
    let w1 = [1.5f32, -0.5];
    let b1 = [0.75f32];
    const C: f32 = 4.0;

    let full_md = transforms_md(b"identity\nwinsor_p99\nidentity", b"\n4,4\n");
    let full = Aligned(make_two_layer(
        &mean, &scale, &w0_full, &b0, &w1, &b1, &full_md,
    ));
    let full_model = Model::from_bytes(&full.0).expect("load full");
    let mut full_pred = Predictor::new(&full_model);

    // Fold: b0' = b0 + x̃₁ · W0[1,:] where x̃₁ = (C - mean1)/scale1.
    let xt = (C - mean[1]) / scale[1];
    let b0_folded = [b0[0] + xt * w0_full[2], b0[1] + xt * w0_full[3]];
    let w0_pruned = [1.0f32, -2.0, 0.5, 0.25];
    let md = transforms_md(b"identity\ndrop\nidentity", b"\n\n");
    let pruned = Aligned(make_two_layer(
        &[mean[0], mean[2]],
        &[scale[0], scale[2]],
        &w0_pruned,
        &b0_folded,
        &w1,
        &b1,
        &md,
    ));
    let pruned_model = Model::from_bytes(&pruned.0).expect("load pruned");
    let mut pruned_pred = Predictor::new(&pruned_model);

    for probe in PROBES {
        let a = full_pred.predict_transformed(&probe).expect("full predict")[0];
        let b = pruned_pred
            .predict_transformed(&probe)
            .expect("pruned predict")[0];
        // Real-arithmetic exact; the fold reorders one f32 sum, so
        // compare to fp tolerance rather than bit-equality.
        assert!(
            (a - b).abs() <= 1e-5 * a.abs().max(1.0),
            "constant-fold prune diverged at {probe:?}: {a} vs {b}"
        );
    }
}

#[test]
fn pruned_bake_rejects_a_vector_sized_by_n_inputs() {
    // The failure mode we care most about: a caller that sizes by
    // `n_inputs()` instead of `caller_input_width()` must FAIL, never
    // silently score a prefix.
    let w0 = [1.0f32, -2.0, 0.5, 0.25];
    let b0 = [0.1f32, -0.2];
    let w1 = [1.5f32, -0.5];
    let b1 = [0.75f32];
    let md = transforms_md(b"identity\ndrop\nidentity", b"\n\n");
    let bytes = Aligned(make_two_layer(
        &[0.0, 0.0],
        &[1.0, 1.0],
        &w0,
        &b0,
        &w1,
        &b1,
        &md,
    ));
    let model = Model::from_bytes(&bytes.0).expect("load");
    let mut pred = Predictor::new(&model);
    assert_eq!(model.n_inputs(), 2);
    let err = pred.predict_transformed(&[1.0, 2.0]).unwrap_err();
    assert!(
        matches!(
            err,
            PredictError::FeatureLenMismatch {
                expected: 3,
                got: 2
            }
        ),
        "expected a loud length mismatch, got {err:?}"
    );
    // The right width works.
    assert!(pred.predict_transformed(&[1.0, 2.0, 3.0]).is_ok());
}

#[test]
fn drop_with_params_is_rejected_at_bake_time() {
    let w0 = [1.0f32, -2.0, 0.5, 0.25];
    let b0 = [0.1f32, -0.2];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 2,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w0,
        biases: &b0,
    }];
    let md = transforms_md(b"identity\ndrop\nidentity", b"\n1.5\n");
    let err = bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &[0.0, 0.0],
        scaler_scale: &[1.0, 1.0],
        layers: &layers,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect_err("drop must take no params");
    let msg = alloc_fmt(&err);
    assert!(msg.contains("drop"), "unexpected error: {msg}");
}

#[test]
fn sinusoidal_with_no_frequencies_is_rejected_and_points_at_drop() {
    // Arity 0 used to be spellable as "sinusoidal with zero
    // frequencies". That is now a bake-time error naming `drop`.
    let w0 = [1.0f32, -2.0, 0.5, 0.25];
    let b0 = [0.1f32, -0.2];
    let layers = [BakeLayer {
        in_dim: 2,
        out_dim: 2,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &w0,
        biases: &b0,
    }];
    let md = transforms_md(b"identity\nsinusoidal\nidentity", b"\n\n");
    let err = bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &[0.0, 0.0],
        scaler_scale: &[1.0, 1.0],
        layers: &layers,
        feature_bounds: &[],
        metadata: &md,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect_err("empty-param sinusoidal must be rejected");
    let msg = alloc_fmt(&err);
    assert!(
        msg.contains("drop") && msg.contains("frequency"),
        "error should point at `drop`: {msg}"
    );
}

#[test]
fn drop_arity_and_token_round_trip() {
    let t = FeatureTransform::from_token("drop").expect("known token");
    assert_eq!(t, FeatureTransform::Drop);
    assert_eq!(t.as_token(), "drop");
    assert_eq!(t.output_arity(&[]), 0);
    assert_eq!(t.output_arity(&[1.0, 2.0]), 0);
    assert!(t.is_expander(), "Drop needs the variable-arity pipeline");
    assert!(!t.requires_params());
    let mut dst: [f32; 0] = [];
    assert_eq!(t.apply_expanding(123.0, &[], &mut dst), 0);
}

#[test]
#[should_panic(expected = "FeatureTransform::Drop cannot be applied via scalar `apply`")]
fn drop_panics_in_the_scalar_pipeline() {
    let _ = FeatureTransform::Drop.apply(1.0);
}

fn alloc_fmt<E: core::fmt::Display>(e: &E) -> String {
    format!("{e}")
}
