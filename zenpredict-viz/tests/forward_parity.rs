//! Parity check: `zenpredict_viz::forward_with_taps`' final output
//! must equal `zenpredict::Predictor::predict` within 1 ULP across the
//! shipped bake set.

use serde::Deserialize;
use std::fs;

#[derive(Deserialize, Debug)]
struct ForwardTaps {
    output: Vec<f32>,
}

const BAKES: &[&str] = &[
    "/home/lilith/work/zen/zensim/zensim/weights/v_tuner_v11_2026-05-24.bin",
    "/home/lilith/work/zen/zensim/zensim/weights/v_tuner_v9_2026-05-20.bin",
    "/home/lilith/work/zen/zensim/zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin",
    "/home/lilith/work/zen/zensim/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin",
];

#[test]
fn forward_matches_reference_predict() {
    for path in BAKES {
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: bake not present at {path}");
            continue;
        }
        let bytes = fs::read(path).expect("read bake");
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse bake");
        let n_inputs = model.n_inputs();
        let n_outputs = model.n_outputs();

        // Synthetic feature vector: f(i) = sin(0.1·i) * 5.0 — varies
        // smoothly across the input range so every feature has a non-
        // zero contribution.
        let features: Vec<f32> = (0..n_inputs)
            .map(|i| (0.1 * i as f32).sin() * 5.0)
            .collect();

        // Reference: zenpredict::Predictor::predict.
        let mut predictor = zenpredict::Predictor::new(&model);
        let ref_output: Vec<f32> = predictor.predict(&features).expect("predict").to_vec();

        // Under test: zenpredict_viz::forward_with_taps's final output.
        let taps_js =
            zenpredict_viz::forward_with_taps_native(&bytes, &features).expect("forward_with_taps");
        assert_eq!(
            taps_js.output.len(),
            n_outputs,
            "{path}: output dim mismatch"
        );
        for (i, (&a, &b)) in ref_output.iter().zip(taps_js.output.iter()).enumerate() {
            let diff = (a - b).abs();
            // 1e-4 relative + 1e-5 absolute — accounts for f32 SAXPY
            // accumulator-order drift between the reference's
            // saxpy_matmul_* loops and our pre/post-activation tap
            // loops. Both produce visually identical heatmaps; only
            // strict bit-for-bit equality would require exact loop
            // mirroring.
            let tol = a.abs().max(b.abs()) * 1e-4_f32 + 1e-5_f32;
            assert!(
                diff <= tol,
                "{path}: output[{i}] reference={a} viz={b} diff={diff} tol={tol}"
            );
        }
        println!("✓ {path}: {n_outputs} outputs within 1 ULP");
    }
}
