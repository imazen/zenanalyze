//! Parity of `zenpredict_viz::forward_with_taps_native` with the
//! zenpredict runtime, on bakes built IN the test from the composer
//! (`zenpredict_bake`) — every dtype, every activation, with and
//! without `feature_transforms` / `output_specs` — asserted
//! BIT-EXACT (`f32::to_bits`), which is stricter than the 1-ULP bar in
//! zenanalyze#79. The shipped-bake check (`shipped_bakes_parity`) reads
//! `ZENPREDICT_VIZ_BAKES` and is deliberately loud when it is unset:
//! the caller (justfile / CI) decides whether to skip it, never the test
//! body.

mod common;

use common::{HIDDEN, N_IN, N_OUT, Synth, all_activations, all_dtypes, probe_features, synth_bake};
use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zenpredict_viz::{forward_with_taps_native, parse_bake_native};

fn assert_bits_eq(label: &str, reference: &[f32], viz: &[f32]) {
    assert_eq!(reference.len(), viz.len(), "{label}: output dim");
    for (i, (r, v)) in reference.iter().zip(viz).enumerate() {
        assert_eq!(
            r.to_bits(),
            v.to_bits(),
            "{label}: output[{i}] reference={r} viz={v} (not bit-identical)"
        );
    }
}

#[test]
fn forward_taps_are_bit_identical_to_predict_for_every_dtype_and_activation() {
    let synth = Synth::new(0xC0FFEE);
    for dtype in all_dtypes() {
        for activation in all_activations() {
            let bytes = synth_bake(&synth, dtype, activation, false, false);
            let model = Model::from_bytes(&bytes).expect("parse");
            let mut predictor = Predictor::new(&model);
            for seed in [1u64, 2, 3] {
                let features = probe_features(seed);
                let reference = predictor.predict(&features).expect("predict").to_vec();
                let taps = forward_with_taps_native(&bytes, &features).expect("taps");
                let label = format!("{dtype:?}/{activation:?}/probe{seed}");
                assert_bits_eq(&label, &reference, &taps.output);
                assert!(
                    taps.transformed.is_none(),
                    "{label}: no transforms expected"
                );
                assert!(taps.specs_applied.is_none(), "{label}: no specs expected");
                assert_eq!(taps.layer_stages.len(), 2);
                // The waterfall's last stage IS the output (no hidden drift).
                assert_bits_eq(
                    &format!("{label}/last-stage"),
                    &taps.layer_stages[1].post_activation,
                    &taps.output,
                );
                // Hidden activation actually applied: relu zeroes negatives,
                // leaky scales them, identity leaves them.
                let l0 = &taps.layer_stages[0];
                let neg = l0
                    .pre_activation
                    .iter()
                    .zip(&l0.post_activation)
                    .filter(|(pre, _)| **pre < 0.0)
                    .collect::<Vec<_>>();
                assert!(
                    !neg.is_empty(),
                    "{label}: probe should hit a negative pre-activation"
                );
                for (pre, post) in neg {
                    let expect = match activation {
                        Activation::Relu => 0.0,
                        Activation::LeakyRelu => pre * zenpredict::LEAKY_RELU_ALPHA,
                        _ => *pre,
                    };
                    assert_eq!(post.to_bits(), expect.to_bits(), "{label}: activation");
                }
            }
        }
    }
}

#[test]
fn feature_transforms_stage_matches_predict_transformed() {
    let synth = Synth::new(0xBEEF);
    for dtype in all_dtypes() {
        let bytes = synth_bake(&synth, dtype, Activation::LeakyRelu, true, false);
        let model = Model::from_bytes(&bytes).expect("parse");
        assert!(model.has_nontrivial_feature_transforms());
        let mut predictor = Predictor::new(&model);
        let features = probe_features(7);
        let reference = predictor
            .predict_transformed(&features)
            .expect("predict_transformed")
            .to_vec();
        let taps = forward_with_taps_native(&bytes, &features).expect("taps");
        assert_bits_eq(&format!("{dtype:?}/transformed"), &reference, &taps.output);
        let t = taps
            .transformed
            .as_ref()
            .expect("transformed stage present");
        assert_eq!(t.len(), N_IN);
        for i in 0..N_IN {
            let expect = if i % 2 == 0 {
                features[i].ln_1p()
            } else {
                features[i]
            };
            assert_eq!(t[i].to_bits(), expect.to_bits(), "transform[{i}]");
        }
        // And it is NOT what the untransformed pass gives (the stage matters).
        let raw = predictor.predict(&features).expect("predict").to_vec();
        assert!(
            raw.iter()
                .zip(&taps.output)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "transforms changed nothing — probe is degenerate"
        );
    }
    // Wrong caller width is an error, not a silent truncation.
    let bytes = synth_bake(&synth, WeightDtype::F32, Activation::Relu, true, false);
    assert!(forward_with_taps_native(&bytes, &probe_features(1)[..N_IN - 1]).is_err());
}

#[test]
fn output_specs_stage_matches_predict_with_specs() {
    let synth = Synth::new(0xD00D);
    let bytes = synth_bake(&synth, WeightDtype::F32, Activation::Relu, false, true);
    let model = Model::from_bytes(&bytes).expect("parse");
    assert!(model.has_output_specs());
    let mut predictor = Predictor::new(&model);
    let features = probe_features(11);
    let reference: Vec<Option<f32>> = predictor
        .predict_with_specs(&features)
        .expect("predict_with_specs")
        .iter()
        .map(|v| v.value())
        .collect();
    let taps = forward_with_taps_native(&bytes, &features).expect("taps");
    let applied = taps.specs_applied.expect("specs stage present");
    assert_eq!(applied.len(), N_OUT);
    for (i, (r, v)) in reference.iter().zip(&applied).enumerate() {
        match (r, v) {
            (Some(r), Some(v)) => assert_eq!(r.to_bits(), v.to_bits(), "specs[{i}]"),
            (None, None) => {}
            _ => panic!("specs[{i}]: reference={r:?} viz={v:?}"),
        }
    }
    // The clamp on output 0 and the sigmoid on output 1 are visible.
    let c = applied[0].unwrap();
    assert!((-0.25..=0.25).contains(&c), "clamped output 0 = {c}");
    let s = applied[1].unwrap();
    assert!((0.0..=1.0).contains(&s) && s.to_bits() != taps.output[1].to_bits());
    // Raw output stays raw.
    assert_bits_eq("raw", predictor.predict(&features).unwrap(), &taps.output);
}

#[test]
fn summary_reports_names_widths_and_importance() {
    let synth = Synth::new(0xA11CE);
    let bytes = synth_bake(&synth, WeightDtype::I8, Activation::Relu, true, true);
    let s = parse_bake_native(&bytes).expect("summary");
    assert_eq!(s.n_inputs, N_IN);
    assert_eq!(s.n_outputs, N_OUT);
    assert_eq!(s.n_layers, 2);
    assert_eq!(s.caller_input_width, N_IN);
    assert!(s.has_feature_transforms && s.has_output_specs);
    assert_eq!(s.feature_names.len(), N_IN);
    assert_eq!(s.feature_names[3], "feat_synth_3");
    assert_eq!(s.l0_importance.len(), N_IN);
    // Zero-variance column ⇒ scaler_scale 0 ⇒ importance 0 by definition.
    assert_eq!(s.l0_importance[4], 0.0);
    assert!(s.l0_importance.iter().any(|&v| v > 0.0));
    assert_eq!(s.layers[0].dtype, "i8");
    assert!(
        s.layers[0]
            .i8_scales
            .as_ref()
            .is_some_and(|v| v.len() == HIDDEN)
    );
    assert_eq!(s.layers[1].activation, "identity");
    let keys: Vec<&str> = s.metadata_keys.iter().map(|k| k.key.as_str()).collect();
    assert!(keys.contains(&zenpredict::keys::FEATURE_COLUMNS));
    assert!(keys.contains(&zenpredict::keys::FEATURE_TRANSFORMS));
    // A bake without transforms reports no tokens (UI hides the toggle).
    let plain = synth_bake(&synth, WeightDtype::F32, Activation::Relu, false, false);
    assert!(
        zenpredict_viz::feature_transform_tokens_native(&plain)
            .unwrap()
            .is_empty()
    );
    let toks = zenpredict_viz::feature_transform_tokens_native(&bytes).unwrap();
    assert_eq!(toks.len(), N_IN);
    assert_eq!(toks[0], "log1p");
    assert_eq!(toks[1], "identity");
}

#[test]
fn layer_weights_dequantize_f32_exactly() {
    // The composer always reorders hidden units (by L2 norm, for
    // compressibility), so layer 0's COLUMNS come back permuted: compare
    // the dequantized matrix as a multiset of bit patterns, which a
    // column permutation preserves and any value change breaks.
    let synth = Synth::new(0x5EED);
    let bytes = synth_bake(&synth, WeightDtype::F32, Activation::Relu, false, false);
    let mut w0 = zenpredict_viz::layer_weights_native(&bytes, 0).expect("layer 0");
    assert_eq!(w0.len(), N_IN * HIDDEN);
    let mut expect = synth.w0.clone();
    w0.sort_by_key(|v| v.to_bits());
    expect.sort_by_key(|v| v.to_bits());
    for (a, b) in w0.iter().zip(&expect) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let mut w1 = zenpredict_viz::layer_weights_native(&bytes, 1).expect("layer 1");
    let mut expect1 = synth.w1.clone();
    w1.sort_by_key(|v| v.to_bits());
    expect1.sort_by_key(|v| v.to_bits());
    assert_eq!(w1.len(), HIDDEN * N_OUT);
    for (a, b) in w1.iter().zip(&expect1) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert!(zenpredict_viz::layer_weights_native(&bytes, 2).is_err());
}

/// Parity on real shipped bakes. `ZENPREDICT_VIZ_BAKES` is a
/// `:`-separated list of `.bin` files and/or directories (every `*.bin`
/// inside a directory is checked). Unset ⇒ FAIL with instructions —
/// skipping is the caller's decision (`cargo test -- --skip
/// shipped_bakes_parity`, as the justfile and CI do), never a silent
/// pass inside the test.
#[test]
fn shipped_bakes_parity() {
    let spec = std::env::var("ZENPREDICT_VIZ_BAKES").unwrap_or_else(|_| {
        panic!(
            "ZENPREDICT_VIZ_BAKES is not set. Point it at shipped .bin bakes \
             (`:`-separated files or directories), e.g. \
             ZENPREDICT_VIZ_BAKES=../zensim/zensim/weights, or skip this test \
             explicitly with `-- --skip shipped_bakes_parity`."
        )
    });
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    for item in spec.split(':').filter(|s| !s.is_empty()) {
        let p = std::path::PathBuf::from(item);
        if p.is_dir() {
            let mut found: Vec<_> = std::fs::read_dir(&p)
                .expect("read dir")
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "bin"))
                .collect();
            found.sort();
            paths.extend(found);
        } else {
            paths.push(p);
        }
    }
    assert!(
        !paths.is_empty(),
        "ZENPREDICT_VIZ_BAKES={spec}: no .bin files found"
    );
    for path in &paths {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let model = match Model::from_bytes(&bytes) {
            Ok(m) => m,
            Err(e) => panic!("{}: not a parseable ZNPR bake: {e}", path.display()),
        };
        let mut predictor = Predictor::new(&model);
        let width = model.caller_input_width();
        // Bounded positive probe — safe for log-family transforms.
        let features: Vec<f32> = (0..width)
            .map(|i| ((0.1 * i as f32).sin() + 1.5) * 2.0)
            .collect();
        let reference = predictor
            .predict_transformed(&features)
            .unwrap_or_else(|e| panic!("{}: predict_transformed: {e}", path.display()))
            .to_vec();
        let taps = forward_with_taps_native(&bytes, &features)
            .unwrap_or_else(|e| panic!("{}: forward_with_taps: {e}", path.display()));
        assert_bits_eq(&path.display().to_string(), &reference, &taps.output);
        println!(
            "✓ {}: {} outputs bit-identical",
            path.display(),
            reference.len()
        );
    }
}
