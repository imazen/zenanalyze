//! Compare two ZNPR bakes by running predict on random inputs.
//! Usage: compare_bakes <bake_a.bin> <bake_b.bin>

use std::env;
use std::fs;

use zenpredict::{Model, Predictor};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: compare_bakes <a.bin> <b.bin>");
        std::process::exit(1);
    }
    let bytes_a = fs::read(&args[1]).unwrap();
    let bytes_b = fs::read(&args[2]).unwrap();
    let model_a = Model::from_bytes(&bytes_a).unwrap();
    let model_b = Model::from_bytes(&bytes_b).unwrap();
    let n_in = model_a.n_inputs();
    assert_eq!(n_in, model_b.n_inputs(), "n_inputs mismatch");
    assert_eq!(model_a.n_outputs(), model_b.n_outputs(), "n_outputs mismatch");
    let mut p_a = Predictor::new(&model_a);
    let mut p_b = Predictor::new(&model_b);
    // Try a small grid of inputs covering reasonable feature ranges.
    let mut max_abs_diff: f32 = 0.0;
    let mut sum_sq_diff: f32 = 0.0;
    let mut n_samples = 0u64;
    for seed in 0..50u64 {
        let features: Vec<f32> = (0..n_in)
            .map(|i| {
                let v = (seed.wrapping_mul(31) ^ i as u64) as u32;
                ((v % 1000) as f32 / 1000.0 - 0.5) * 6.0
            })
            .collect();
        let out_a = p_a.predict(&features).unwrap();
        let out_b = p_b.predict(&features).unwrap();
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
            sum_sq_diff += d * d;
            n_samples += 1;
        }
    }
    let rmse = (sum_sq_diff / n_samples as f32).sqrt();
    println!("{} samples, max|Δ| = {}, RMSE = {}", n_samples, max_abs_diff, rmse);
    println!("a: {} bytes, b: {} bytes", bytes_a.len(), bytes_b.len());
}
