//! `Predictor::predict` throughput on two production shapes.
//!
//! - **zensim** — V0_18 scorer shape: 228 → 384 → 1, I8 weights with
//!   per-output f32 scales (concat MLP).
//! - **zenwebp picker** — production picker shape: 51 inputs → 64
//!   hidden → 24 outputs, F16 weights. Wired in
//!   `zenwebp/src/encoder/picker/runtime.rs`.
//!
//! Both shapes bake fresh in-memory, then run `Predictor::predict` in
//! a tight loop with deterministic feature vectors. Measures the
//! actual hot path codecs and zensim run — not a synthetic microbench.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use zenbench::black_box;
use zenpredict::{Activation, Model, Predictor, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};

fn bake_shape(
    n_in: usize,
    n_hidden: usize,
    n_out: usize,
    dtype: WeightDtype,
    seed: u64,
) -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let scaler_mean: Vec<f32> = (0..n_in).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let scaler_scale: Vec<f32> = (0..n_in).map(|_| rng.gen_range(0.5..1.5)).collect();
    let w0: Vec<f32> = (0..n_in * n_hidden)
        .map(|_| rng.gen_range(-0.3f32..0.3))
        .collect();
    let b0: Vec<f32> = vec![0.0; n_hidden];
    let w1: Vec<f32> = (0..n_hidden * n_out)
        .map(|_| rng.gen_range(-0.3f32..0.3))
        .collect();
    let b1: Vec<f32> = vec![0.0; n_out];
    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_out,
            activation: Activation::Identity,
            dtype,
            weights: &w1,
            biases: &b1,
        },
    ];
    let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);
    bake(&req).expect("bake shape")
}

fn random_inputs(n_in: usize, n_samples: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_samples)
        .map(|_| (0..n_in).map(|_| rng.gen_range(-3.0f32..3.0)).collect())
        .collect()
}

zenbench::main!(|suite| {
    // Leak the bake bytes + input vectors so their borrowed references
    // satisfy the `'static` requirement of the bench closures.
    let zensim_bytes: &'static [u8] =
        Box::leak(bake_shape(228, 384, 1, WeightDtype::I8, 0xfeed).into_boxed_slice());
    let webp_bytes: &'static [u8] =
        Box::leak(bake_shape(51, 64, 24, WeightDtype::F16, 0xb33f).into_boxed_slice());
    let zensim_inputs: &'static [Vec<f32>] =
        Box::leak(random_inputs(228, 256, 0xc0ffee).into_boxed_slice());
    let webp_inputs: &'static [Vec<f32>] =
        Box::leak(random_inputs(51, 256, 0xdead).into_boxed_slice());

    // Align via a leaked Box<[u8; N]>-equivalent buffer. The fresh
    // `bake` output's heap allocation is already at least 8-aligned;
    // bytemuck::pod_read_unaligned in Model::from_bytes tolerates the
    // residual misalignment for the Header read. For weight slices, we
    // ensure 16-byte alignment by re-copying into an aligned scratch.
    let zensim_bytes = align_to_16(zensim_bytes);
    let webp_bytes = align_to_16(webp_bytes);

    suite.group("zensim_v018_228_384_1_i8", |g| {
        g.bench("predict", move |b| {
            let model = Model::from_bytes(zensim_bytes).expect("parse zensim");
            let mut predictor = Predictor::new(&model);
            let mut i = 0usize;
            b.iter(move || {
                let out = predictor
                    .predict(&zensim_inputs[i % zensim_inputs.len()])
                    .unwrap();
                i = i.wrapping_add(1);
                black_box(out[0])
            })
        });
    });

    suite.group("zenwebp_picker_51_64_24_f16", |g| {
        g.bench("predict", move |b| {
            let model = Model::from_bytes(webp_bytes).expect("parse webp");
            let mut predictor = Predictor::new(&model);
            let mut i = 0usize;
            b.iter(move || {
                let out = predictor
                    .predict(&webp_inputs[i % webp_inputs.len()])
                    .unwrap();
                i = i.wrapping_add(1);
                black_box(out[0])
            })
        });
    });
});

fn align_to_16(bytes: &'static [u8]) -> &'static [u8] {
    #[repr(C, align(16))]
    struct Aligned16(Vec<u8>);
    let aligned = Aligned16(bytes.to_vec());
    let leaked: &'static Aligned16 = Box::leak(Box::new(aligned));
    leaked.0.as_slice()
}
