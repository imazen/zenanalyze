//! First-parse + per-predict latency for V0_18-shape bakes.
//!
//! Reads the archived V0_17 F32 bake (the source V0_18 was quantized
//! from), then re-bakes four variants:
//!
//! 1. I8        raw weights
//! 2. I8        τ=0.005 per-layer zerobias
//! 3. I8Lz4     raw weights
//! 4. I8Lz4     τ=0.005 per-layer zerobias
//!
//! For each variant, measure:
//!
//! - **Bake size**: bytes on disk.
//! - **First parse**: `Model::from_bytes(bytes)` + `Predictor::new(&model)`
//!   in a single tight loop. This is what a runtime pays on its
//!   first `predict()` (the parser does zero-copy slicing for non-LZ4
//!   dtypes; for LZ4 the per-Predictor scratch is allocated here).
//! - **Per-predict**: warm steady-state `predict()` over deterministic
//!   feature vectors. Median of N runs.
//!
//! Usage:
//!     cargo run --release -p zenpredict-bake --example lz4_first_parse_bench --features lz4
//!         -- <path to V0_17 F32 bake .bin>
//!
//! Defaults to the V0_17 archive in zensim's repo when no arg is
//! given. Emits results as a markdown table on stdout.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use zenpredict::{Activation, Model, Predictor, WeightDtype, WeightStorage};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

fn parse_source_bake(path: &PathBuf) -> SourceLayers {
    let bytes = std::fs::read(path).expect("read source bake");
    let model = Model::from_bytes(&bytes).expect("parse source");
    let scaler_mean = model.scaler_mean().to_vec();
    let scaler_scale = model.scaler_scale().to_vec();
    let mut layers = Vec::new();
    for l in model.layers() {
        let w_f32: Vec<f32> = match &l.weights {
            WeightStorage::F32(w) => w.to_vec(),
            WeightStorage::F16(w) => w
                .iter()
                .map(|b| zenpredict::f16_bits_to_f32(*b))
                .collect(),
            WeightStorage::I8 { weights, scales } => weights
                .iter()
                .enumerate()
                .map(|(idx, &q)| (q as f32) * scales[idx % l.out_dim])
                .collect(),
            #[cfg(feature = "lz4")]
            WeightStorage::I8Lz4 { .. } => {
                panic!("source bake unexpectedly already compressed");
            }
        };
        layers.push(LayerSpec {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation,
            weights: w_f32,
            biases: l.biases.to_vec(),
        });
    }
    SourceLayers {
        scaler_mean,
        scaler_scale,
        layers,
    }
}

struct SourceLayers {
    scaler_mean: Vec<f32>,
    scaler_scale: Vec<f32>,
    layers: Vec<LayerSpec>,
}

struct LayerSpec {
    in_dim: usize,
    out_dim: usize,
    activation: Activation,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

/// Per-layer global-max zerobias (matches the 2026-05-13 RLE/zerobias eval).
fn zero_bias_per_layer(weights: &[f32], tau: f32) -> Vec<f32> {
    let max = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let cut = tau * max;
    weights
        .iter()
        .map(|&w| if w.abs() < cut { 0.0 } else { w })
        .collect()
}

fn build_bake(src: &SourceLayers, dtype: WeightDtype, tau: f32) -> Vec<u8> {
    let bias_layers: Vec<(Vec<f32>, Vec<f32>)> = src
        .layers
        .iter()
        .map(|l| {
            let w = if tau > 0.0 {
                zero_bias_per_layer(&l.weights, tau)
            } else {
                l.weights.clone()
            };
            (w, l.biases.clone())
        })
        .collect();
    let bake_layers: Vec<BakeLayer<'_>> = src
        .layers
        .iter()
        .zip(bias_layers.iter())
        .map(|(l, (w, b))| BakeLayer {
            in_dim: l.in_dim,
            out_dim: l.out_dim,
            activation: l.activation,
            dtype,
            weights: w,
            biases: b,
        })
        .collect();
    let req = BakeRequest::new(0, 0, &src.scaler_mean, &src.scaler_scale, &bake_layers);
    bake(&req).expect("bake")
}

fn median_us<F: FnMut()>(mut f: F, n_iter: usize) -> f64 {
    let mut samples = Vec::with_capacity(n_iter);
    // Warmup
    for _ in 0..n_iter.min(50) {
        f();
    }
    for _ in 0..n_iter {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

fn main() {
    let default_path =
        PathBuf::from("/home/lilith/work/zen/zensim/zensim/weights/archive/v0_17_2026-05-13.bin");
    let path = env::args().nth(1).map(PathBuf::from).unwrap_or(default_path);
    eprintln!("source: {}", path.display());

    let src = parse_source_bake(&path);
    let n_in = src.scaler_mean.len();
    eprintln!("  n_in={n_in}  layers={}", src.layers.len());

    // Build four variants.
    let variants: Vec<(&str, Vec<u8>)> = vec![
        ("I8 raw          ", build_bake(&src, WeightDtype::I8, 0.0)),
        ("I8 zerobias@005 ", build_bake(&src, WeightDtype::I8, 0.005)),
        #[cfg(feature = "lz4")]
        ("I8Lz4 raw       ", build_bake(&src, WeightDtype::I8Lz4, 0.0)),
        #[cfg(feature = "lz4")]
        (
            "I8Lz4 zerobias  ",
            build_bake(&src, WeightDtype::I8Lz4, 0.005),
        ),
    ];

    // Random feature vector for predict timing.
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    let mut rng = SmallRng::seed_from_u64(0xc0ffee);
    let features: Vec<f32> = (0..n_in).map(|_| rng.random_range(-3.0f32..3.0)).collect();

    println!("# Bake size + first-parse + per-predict latency");
    println!();
    println!("Source: `{}` ({}→{}→1)", path.display(),
        src.layers[0].in_dim, src.layers[0].out_dim);
    println!();
    println!("## Per-layer LZ4 compression (zenpredict's compressed-weights path)");
    println!();
    println!("| variant | bake bytes | shrink vs I8 raw | first parse (µs, median) | per-predict (µs, median) |");
    println!("|---|--:|--:|--:|--:|");

    let baseline_bytes = variants[0].1.len();
    for (label, bytes) in &variants {
        let aligned = Aligned(bytes.clone());
        // First parse: Model::from_bytes + Predictor::new. We use a
        // fresh local block per iter so the optimizer can't fuse
        // across calls.
        let mk_predictor = || {
            let model = Model::from_bytes(&aligned.0).unwrap();
            let _ = Predictor::new(&model);
        };
        let t_parse = median_us(mk_predictor, 200);

        // Per-predict: parse once, time the warm hot path.
        let model = Model::from_bytes(&aligned.0).unwrap();
        let mut predictor = Predictor::new(&model);
        let t_predict = median_us(|| {
            let _ = predictor.predict(&features).unwrap();
        }, 1000);

        let shrink = 1.0 - (bytes.len() as f32 / baseline_bytes as f32);
        println!(
            "| {} | {:>8} | {:>+5.1} % | {:>10.1} | {:>10.2} |",
            label.trim(),
            bytes.len(),
            shrink * 100.0,
            t_parse,
            t_predict,
        );
    }

    // -------- Full-bin compression (consumer-side decode) --------
    //
    // Compress the entire 93 KB .bin (including headers, scaler arrays,
    // biases, scales) as one blob. Decompress at consumer load time,
    // then parse normally. Per-predict cost is ZERO (decompression
    // happened once at startup) — but the consumer pays the
    // decompression once instead of zenpredict paying per-predict.
    //
    // Practical relevance: this is what you'd do if you wanted a
    // smaller embed without zenpredict's compressed-weights feature
    // active. The trade-off is decompression-on-load vs per-predict,
    // not the size shrink itself.
    #[cfg(feature = "lz4")]
    {
        use lz4_flex::block::compress as lz4_compress;
        println!();
        println!("## Full-bin LZ4 compression (consumer decompresses entire .bin once at load)");
        println!();
        println!("Decompression cost is paid ONCE at consumer load, NOT per-predict.");
        println!("Per-predict latency is identical to the uncompressed `I8 raw` baseline.");
        println!();
        println!("| variant | full-bin bytes | shrink vs I8 raw | full-bin decode µs (median) |");
        println!("|---|--:|--:|--:|");
        for (label, bytes) in &variants {
            let compressed = lz4_compress(bytes);
            let shrink = 1.0 - (compressed.len() as f32 / baseline_bytes as f32);
            let decompressed_len = bytes.len();
            let t_decode = median_us(
                || {
                    let _ = lz4_flex::block::decompress(&compressed, decompressed_len).unwrap();
                },
                500,
            );
            println!(
                "| {} | {:>8} | {:>+5.1} % | {:>10.1} |",
                label.trim(),
                compressed.len(),
                shrink * 100.0,
                t_decode,
            );
        }
    }
}
