//! `WeightDtype::I8Lz4` end-to-end: bake → load → predict → verify.
//!
//! Reuses the V_X-shape sample (228 → 64 → 1 LeakyReLU) and proves
//! that the I8Lz4 path produces the same predictions as plain I8
//! to within i8 round-trip precision. Also asserts size shrinkage
//! when the weights are zero-biased before quantization.

#![cfg(feature = "lz4")]

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, apply_zero_bias, bake};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

fn random_weights(n_in: usize, n_hidden: usize, n_out: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let w0: Vec<f32> = (0..n_in * n_hidden).map(|_| rng.random_range(-0.3f32..0.3)).collect();
    let b0: Vec<f32> = (0..n_hidden).map(|_| 0.0).collect();
    let w1: Vec<f32> = (0..n_hidden * n_out).map(|_| rng.random_range(-0.3f32..0.3)).collect();
    let b1: Vec<f32> = (0..n_out).map(|_| 0.0).collect();
    (w0, b0, w1, b1)
}

/// Trained-MLP-like distribution: most weights tiny (Gaussian σ=0.02)
/// + a few large ones (σ=0.3). Mirrors the long-tail shape that
/// production V_X weights have, so the per-layer max-abs is dominated
/// by a few outliers — that's the regime where τ-zerobias finds a
/// lot of zeros.
fn long_tail_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            // Box-Muller approximation: sum of 4 uniforms - 2 ≈ N(0, sqrt(4/12))
            let u: f32 = (0..4).map(|_| rng.random_range(0.0f32..1.0)).sum();
            let z = u - 2.0; // ~ N(0, 0.577)
            // 99 % of weights at σ=0.02 (tiny). 1 % outliers at σ=0.3
            // (the few load-bearing weights). Matches Lottery Ticket
            // / pruning literature's reported MLP weight distributions.
            if rng.random::<f32>() < 0.99 {
                z * 0.02
            } else {
                z * 0.3
            }
        })
        .collect()
}

fn build_bake(
    dtype: WeightDtype,
    w0: &[f32],
    b0: &[f32],
    w1: &[f32],
    b1: &[f32],
    n_in: usize,
    n_hidden: usize,
    n_out: usize,
) -> Vec<u8> {
    let scaler_mean = vec![0.0f32; n_in];
    let scaler_scale = vec![1.0f32; n_in];
    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype,
            weights: w0,
            biases: b0,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_out,
            activation: Activation::Identity,
            dtype,
            weights: w1,
            biases: b1,
        },
    ];
    let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);
    bake(&req).expect("bake")
}

#[test]
fn lz4_roundtrips_for_random_weights() {
    let (n_in, n_hidden, n_out) = (228, 64, 1);
    let (w0, b0, w1, b1) = random_weights(n_in, n_hidden, n_out, 0xfeed);

    // Bake the same weights twice — once as I8, once as I8Lz4.
    let bake_i8 = Aligned(build_bake(WeightDtype::I8, &w0, &b0, &w1, &b1, n_in, n_hidden, n_out));
    let bake_lz4 = Aligned(build_bake(WeightDtype::I8Lz4, &w0, &b0, &w1, &b1, n_in, n_hidden, n_out));

    let model_i8 = Model::from_bytes(&bake_i8.0).expect("parse i8");
    let model_lz4 = Model::from_bytes(&bake_lz4.0).expect("parse i8_lz4");
    let mut p_i8 = Predictor::new(&model_i8);
    let mut p_lz4 = Predictor::new(&model_lz4);

    // Predict on 8 random feature vectors — outputs MUST be
    // bit-identical between I8 and I8Lz4 (same quantized weights,
    // same matmul; lz4 just changes byte storage).
    let mut rng = SmallRng::seed_from_u64(0xc0ffee);
    for _ in 0..8 {
        let features: Vec<f32> = (0..n_in).map(|_| rng.random_range(-3.0f32..3.0)).collect();
        let y_i8 = p_i8.predict(&features).unwrap();
        let y_lz4 = p_lz4.predict(&features).unwrap();
        assert_eq!(y_i8.len(), y_lz4.len());
        for k in 0..y_i8.len() {
            assert!(
                (y_i8[k] - y_lz4[k]).abs() < 1e-6,
                "predict mismatch at output {k}: i8={} lz4={}",
                y_i8[k],
                y_lz4[k]
            );
        }
    }
}

#[test]
fn lz4_shrinks_after_zerobias() {
    // Uniform-random weights are near-incompressible. Production V_X
    // weights have a long-tail distribution (most weights tiny,
    // a few large) — that's the regime where τ-zerobias finds runs
    // of zeros that lz4 can squeeze. Mirror that distribution here.
    let (n_in, n_hidden, n_out) = (228, 384, 1);
    let w0 = long_tail_weights(n_in * n_hidden, 0x1234);
    let b0 = vec![0.0f32; n_hidden];
    let w1 = long_tail_weights(n_hidden * n_out, 0x5678);
    let b1 = vec![0.0f32; n_out];

    // Aggressive zero-bias (τ=0.2) on synthetic weights to demonstrate
    // the compression effect. Production V_X uses τ=0.005 but its
    // weight distribution is much heavier-tailed than this synthetic
    // approximation; the wire path is the same regardless.
    let w0_zb = apply_zero_bias_layer(&w0, n_hidden, 0.2);
    let w1_zb = apply_zero_bias_layer(&w1, n_out, 0.2);
    let zb_zero_frac = w0_zb.iter().filter(|w| **w == 0.0).count() as f32 / w0_zb.len() as f32;
    println!("post-zerobias zero fraction in layer 0: {:.1}%", zb_zero_frac * 100.0);

    let bake_lz4_raw = build_bake(WeightDtype::I8Lz4, &w0, &b0, &w1, &b1, n_in, n_hidden, n_out);
    let bake_lz4_zb = build_bake(WeightDtype::I8Lz4, &w0_zb, &b0, &w1_zb, &b1, n_in, n_hidden, n_out);

    println!(
        "lz4 raw weights: {} bytes  vs  zero-biased: {} bytes",
        bake_lz4_raw.len(),
        bake_lz4_zb.len()
    );
    // Sanity: zero-biased version is meaningfully smaller. Exact
    // shrink depends on the (synthetic) weight distribution; we just
    // assert > 30 % shrink as a wire-path correctness check.
    let shrink = 1.0 - (bake_lz4_zb.len() as f32 / bake_lz4_raw.len() as f32);
    println!("shrink = {:.1} %", shrink * 100.0);
    assert!(
        bake_lz4_zb.len() < (bake_lz4_raw.len() * 70) / 100,
        "expected lz4-on-zerobiased < 70 % of lz4-on-raw, got {} vs {}",
        bake_lz4_zb.len(),
        bake_lz4_raw.len(),
    );
}

/// Per-layer zerobias (single global max). Mirrors what
/// `zenpredict_bake::apply_zero_bias_in_place` does for per-column,
/// but in tests we want the more aggressive per-layer threshold
/// to demonstrate the size-shrink effect.
fn apply_zero_bias_layer(weights: &[f32], _out_dim: usize, tau: f32) -> Vec<f32> {
    let max = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let cut = tau * max;
    weights
        .iter()
        .map(|w| if w.abs() < cut { 0.0 } else { *w })
        .collect()
}
