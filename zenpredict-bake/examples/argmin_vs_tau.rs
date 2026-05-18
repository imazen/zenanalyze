//! Sweep `zerobias_tau` and measure argmin-agreement vs the
//! tau=0.0 baseline. Pulls a single source JSON, bakes it at each
//! tau, runs predict on N=2000 deterministic synthetic feature
//! vectors, and reports:
//!   - argmin-agreement %    (fraction of inputs where argmin is unchanged)
//!   - mean output Δ         (mean |out_baseline - out_tau| per output cell)
//!   - max output Δ          (worst-case absolute output shift)
//!   - .bin size in bytes
//!
//! Synthetic input: feature[i] = clip(sin(seed * 0.71 + i * 0.13) * 1.5, -3, 3)
//! That covers the typical post-standardize z-range the picker
//! actually sees (most features end up within ±2σ after the
//! pareto-statistics-fitted scaler in train_hybrid.py).
//!
//! Usage:
//!   cargo run --release -p zenpredict-bake --example argmin_vs_tau -- <source.json>

use std::env;
use std::fs;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};
use zenpredict_bake::{BakeRequestJson, bake_from_json};

const N_SAMPLES: usize = 2000;

fn synth_features(n_in: usize, seed: u64) -> Vec<f32> {
    (0..n_in)
        .map(|i| {
            let v = (seed as f32 * 0.71 + i as f32 * 0.13).sin() * 1.5;
            v.clamp(-3.0, 3.0)
        })
        .collect()
}

fn predict_all(bytes: &[u8], inputs: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<usize>) {
    // Reasonable alignment for the loader. Production consumers wrap
    // include_bytes! in #[repr(C, align(16))]; we just copy into a
    // u64-backed Vec.
    let n_u64 = bytes.len().div_ceil(8);
    let mut storage: Vec<u64> = vec![0; n_u64];
    let view: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
    view[..bytes.len()].copy_from_slice(bytes);
    let aligned: &[u8] = &bytemuck::cast_slice::<u64, u8>(&storage)[..bytes.len()];
    let model = Model::from_bytes(aligned).expect("parse");
    let mut p = Predictor::new(&model);
    let mut outs = Vec::with_capacity(inputs.len());
    let mut argmins = Vec::with_capacity(inputs.len());
    for f in inputs {
        let o = p.predict(f).expect("predict");
        let am = o
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        argmins.push(am);
        // `predict` returns &[f32] borrowed from the predictor — copy
        // before the next call drops the borrow.
        outs.push(o.to_vec());
    }
    (outs, argmins)
}

fn bake_at(json_src: &str, tau: f32, compress: bool, optimize: bool) -> Vec<u8> {
    // Parse the source JSON, mutate the bake-time knobs in-place, and
    // bake. BakeRequestJson is Deserialize-only (#[non_exhaustive] on
    // the public surface) so we don't round-trip through serde_json.
    let mut req: BakeRequestJson = serde_json::from_str(json_src).expect("parse JSON");
    req.zerobias_tau = tau;
    req.compressed = compress;
    req.optimize = optimize;
    bake_from_json(&req).expect("bake")
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: argmin_vs_tau <source.json>");
        return ExitCode::from(2);
    }
    let json_src = fs::read_to_string(&args[1]).expect("read source");

    // Bake the baseline at tau=0, i8, no compress no optimize. This is
    // the canonical f32 → i8 round-trip without any lossy zerobias.
    let bake_baseline = bake_at(&json_src, 0.0, false, false);
    let model = Model::from_bytes(&bake_baseline).expect("parse baseline");
    let n_in = model.n_inputs();
    let n_out = model.n_outputs();
    eprintln!("baseline: n_in={n_in} n_out={n_out} bytes={}", bake_baseline.len());

    let inputs: Vec<Vec<f32>> = (0..N_SAMPLES as u64).map(|s| synth_features(n_in, s)).collect();
    let (out_base, argmin_base) = predict_all(&bake_baseline, &inputs);

    println!(
        "{:>16} {:>10} {:>10} {:>14} {:>14} {:>14}",
        "config", "bytes", "Δbytes", "argmin agree", "mean |Δout|", "max |Δout|"
    );
    println!("{}", "-".repeat(82));

    let configs: &[(f32, bool, bool, &str)] = &[
        (0.0, false, false, "tau=0.0 baseline"),
        (0.005, false, false, "tau=0.005"),
        (0.01, false, false, "tau=0.01"),
        (0.02, false, false, "tau=0.02"),
        (0.05, false, false, "tau=0.05"),
        (0.1, false, false, "tau=0.1"),
        (0.005, true, true, "tau=0.005+cmp+opt"),
        (0.02, true, true, "tau=0.02+cmp+opt"),
    ];

    for &(tau, cmp, opt, label) in configs {
        let bake = bake_at(&json_src, tau, cmp, opt);
        let (out_v, argmin_v) = predict_all(&bake, &inputs);
        let n_agree = argmin_v
            .iter()
            .zip(argmin_base.iter())
            .filter(|(a, b)| a == b)
            .count();
        let mut sum_abs = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut total = 0u64;
        for (oa, ob) in out_base.iter().zip(out_v.iter()) {
            for (a, b) in oa.iter().zip(ob.iter()) {
                let d = (a - b).abs();
                sum_abs += d as f64;
                if d > max_abs {
                    max_abs = d;
                }
                total += 1;
            }
        }
        let mean_abs = (sum_abs / total as f64) as f32;
        println!(
            "{:>16} {:>10} {:>+10} {:>13.2}% {:>14.5} {:>14.5}",
            label,
            bake.len(),
            bake.len() as i64 - bake_baseline.len() as i64,
            100.0 * n_agree as f32 / N_SAMPLES as f32,
            mean_abs,
            max_abs,
        );
    }

    ExitCode::SUCCESS
}
