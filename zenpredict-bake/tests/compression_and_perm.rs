//! Whole-bake compression + feature_order / output_order round-trip tests.
//!
//! Verifies that:
//! - A compressed bake produces byte-equal predict output to its
//!   uncompressed counterpart.
//! - A bake with feature_order produces byte-equal predict output
//!   when called in the same caller-order as the un-permuted bake.
//! - Same for output_order.
//! - All three (compression + feature_order + output_order) compose.

use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

fn build_layers(n_in: usize, n_hidden: usize, n_out: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // Deterministic-ish small weights.
    let mut w0 = vec![0.0f32; n_in * n_hidden];
    for (i, w) in w0.iter_mut().enumerate() {
        *w = ((i as f32 * 0.123).sin() - 0.5) * 0.3;
    }
    let b0 = vec![0.05f32; n_hidden];
    let mut w1 = vec![0.0f32; n_hidden * n_out];
    for (i, w) in w1.iter_mut().enumerate() {
        *w = ((i as f32 * 0.789).cos() - 0.5) * 0.4;
    }
    let b1 = vec![0.01f32; n_out];
    (w0, b0, w1, b1)
}

fn make_request<'a>(
    scaler_mean: &'a [f32],
    scaler_scale: &'a [f32],
    layers: &'a [BakeLayer<'a>],
) -> BakeRequest<'a> {
    BakeRequest::new(0, 0, scaler_mean, scaler_scale, layers)
}

fn build_bake(
    n_in: usize,
    n_hidden: usize,
    n_out: usize,
    feature_order: Option<&[u32]>,
    output_order: Option<&[u32]>,
    compressed: bool,
) -> Vec<u8> {
    let (w0, b0, w1, b1) = build_layers(n_in, n_hidden, n_out);
    let scaler_mean = vec![0.0f32; n_in];
    let scaler_scale = vec![1.0f32; n_in];
    let layers = vec![
        BakeLayer {
            in_dim: n_in,
            out_dim: n_hidden,
            activation: Activation::Relu,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_out,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w1,
            biases: &b1,
        },
    ];
    let mut req = make_request(&scaler_mean, &scaler_scale, &layers);
    req.feature_order = feature_order;
    req.output_order = output_order;
    req.compressed = compressed;
    bake(&req).unwrap()
}

fn predict_features(bake_bytes: &[u8], features: &[f32]) -> Vec<f32> {
    let aligned = Aligned(bake_bytes.to_vec());
    let model = Model::from_bytes(&aligned.0).unwrap();
    let mut p = Predictor::new(&model);
    p.predict(features).unwrap().to_vec()
}

#[test]
fn compression_round_trip_matches_uncompressed() {
    let plain = build_bake(16, 24, 4, None, None, false);
    let compressed = build_bake(16, 24, 4, None, None, true);

    // Compressed bake must be strictly smaller for non-trivial data
    // (and ours has enough structure to compress meaningfully). We
    // tolerate edge cases where lz4 might not shrink very small bakes,
    // so just check it's not catastrophic.
    println!("plain = {} B, compressed = {} B", plain.len(), compressed.len());
    assert!(compressed.len() <= plain.len() + 64);

    let features: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let out_plain = predict_features(&plain, &features);
    let out_compressed = predict_features(&compressed, &features);

    assert_eq!(out_plain, out_compressed, "compressed predict output != plain");
}

#[test]
fn feature_order_round_trip() {
    let n_in = 16;
    // Reverse permutation: bake[bake_pos] = caller[n_in - 1 - bake_pos]
    let perm: Vec<u32> = (0..n_in as u32).rev().collect();

    let plain = build_bake(n_in, 24, 4, None, None, false);
    let permuted = build_bake(n_in, 24, 4, Some(&perm), None, false);

    // The permutation must actually change the layer-0 weights on disk.
    // Layer 0's weights section offset lives at byte 140..144 in the
    // first layer entry (entry starts at byte 128, weights Section
    // starts at offset 12 within the entry).
    let l0_weights_off = u32::from_le_bytes([plain[140], plain[141], plain[142], plain[143]]) as usize;
    let l0_weights_len = u32::from_le_bytes([plain[144], plain[145], plain[146], plain[147]]) as usize;
    assert_ne!(
        &plain[l0_weights_off..l0_weights_off + l0_weights_len],
        &permuted[l0_weights_off..l0_weights_off + l0_weights_len],
        "feature_order didn't permute layer-0 weight rows"
    );

    let features: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let out_plain = predict_features(&plain, &features);
    let out_permuted = predict_features(&permuted, &features);

    assert_eq!(out_plain, out_permuted, "feature_order round-trip mismatch");
}

#[test]
fn output_order_round_trip() {
    let n_out = 4;
    let perm: Vec<u32> = vec![3, 0, 2, 1]; // arbitrary

    let plain = build_bake(16, 24, n_out, None, None, false);
    let permuted = build_bake(16, 24, n_out, None, Some(&perm), false);

    let features: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let out_plain = predict_features(&plain, &features);
    let out_permuted = predict_features(&permuted, &features);

    assert_eq!(out_plain, out_permuted, "output_order round-trip mismatch");
}

#[test]
fn all_three_compose() {
    let n_in = 16;
    let n_out = 4;
    let f_perm: Vec<u32> = (0..n_in as u32).rev().collect();
    let o_perm: Vec<u32> = vec![2, 3, 0, 1];

    let plain = build_bake(n_in, 24, n_out, None, None, false);
    let combined = build_bake(n_in, 24, n_out, Some(&f_perm), Some(&o_perm), true);

    let features: Vec<f32> = (0..n_in).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let out_plain = predict_features(&plain, &features);
    let out_combined = predict_features(&combined, &features);

    assert_eq!(out_plain, out_combined, "feature_order + output_order + compressed mismatch");
}

#[test]
fn feature_order_u8_indices_when_fits() {
    // n_in = 16 → u8 fits, permutation should serialize at 1 byte per index.
    let n_in = 16;
    let perm: Vec<u32> = (0..n_in as u32).rev().collect();
    let bake = build_bake(n_in, 8, 2, Some(&perm), None, false);

    // Find header.feature_order section: offset 100, len 8 bytes.
    let off = u32::from_le_bytes([bake[100], bake[101], bake[102], bake[103]]) as usize;
    let len = u32::from_le_bytes([bake[104], bake[105], bake[106], bake[107]]) as usize;
    assert_eq!(len, n_in, "u8 width expected (1 byte per index = n_in bytes)");
    // Verify content: bake[off + bake_pos] == n_in - 1 - bake_pos.
    for bake_pos in 0..n_in {
        assert_eq!(bake[off + bake_pos] as u32, (n_in - 1 - bake_pos) as u32);
    }
}

#[test]
fn feature_order_u16_indices_when_needed() {
    // n_in = 300 → u8 doesn't fit; u16 should be picked.
    let n_in = 300;
    let perm: Vec<u32> = (0..n_in as u32).rev().collect();
    let bake = build_bake(n_in, 8, 2, Some(&perm), None, false);

    let len = u32::from_le_bytes([bake[104], bake[105], bake[106], bake[107]]) as usize;
    assert_eq!(len, n_in * 2, "u16 width expected (2 bytes per index)");
}

#[test]
fn invalid_permutation_rejected() {
    let n_in = 16;
    // Duplicate index 0.
    let bad: Vec<u32> = (0..n_in as u32).map(|i| if i == 5 { 0 } else { i }).collect();

    let (w0, b0, w1, b1) = build_layers(n_in, 8, 2);
    let scaler_mean = vec![0.0f32; n_in];
    let scaler_scale = vec![1.0f32; n_in];
    let layers = vec![
        BakeLayer {
            in_dim: n_in,
            out_dim: 8,
            activation: Activation::Relu,
            dtype: WeightDtype::F32,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: 8,
            out_dim: 2,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w1,
            biases: &b1,
        },
    ];
    let mut req = make_request(&scaler_mean, &scaler_scale, &layers);
    req.feature_order = Some(&bad);
    assert!(bake(&req).is_err(), "bad permutation should be rejected");
}
