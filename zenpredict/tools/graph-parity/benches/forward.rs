//! Paired forward-pass benchmark: pre-graph zenpredict (`zp_old`, git
//! `e6c72f99`) vs the graph runtime (`zp_new`, this tree), same bytes,
//! same feature vectors, interleaved by zenbench.
//!
//! Bakes: every path in `ZPGRAPH_BENCH_BAKES` (semicolon-separated — paths may contain ':'; the
//! justfile passes the rev4 production bake), plus three in-memory shapes
//! baked once with the old composer (944→128→1 f32, 228→384→1 i8,
//! 51→64→24 f16 — the zenpredict-bake `predict` bench shapes).
//!
//! ```text
//! cargo bench --bench forward
//! ```

use zenbench::black_box;
use zenpredict_graph_parity::make_vectors;

fn shape(n_in: usize, n_hidden: usize, n_out: usize, dtype: u8, seed: u64) -> Vec<u8> {
    use zp_old::{Activation, WeightDtype};
    use zpb_old::{BakeLayer, BakeRequest, bake};
    let mut rng = zenpredict_graph_parity::Rng(seed | 1);
    let dt = match dtype {
        0 => WeightDtype::F32,
        1 => WeightDtype::F16,
        _ => WeightDtype::I8,
    };
    let mean: Vec<f32> = (0..n_in).map(|_| rng.range(-1.0, 1.0)).collect();
    let scale: Vec<f32> = (0..n_in).map(|_| rng.range(0.5, 1.5)).collect();
    let w0: Vec<f32> = (0..n_in * n_hidden).map(|_| rng.range(-0.3, 0.3)).collect();
    let w1: Vec<f32> = (0..n_hidden * n_out).map(|_| rng.range(-0.3, 0.3)).collect();
    let (b0, b1) = (vec![0.0; n_hidden], vec![0.0; n_out]);
    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype: dt,
            weights: &w0,
            biases: &b0,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_out,
            activation: Activation::Identity,
            dtype: dt,
            weights: &w1,
            biases: &b1,
        },
    ];
    bake(&BakeRequest::new(0, 0, &mean, &scale, &layers)).expect("bake shape")
}

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

zenbench::main!(|suite| {
    let mut bakes: Vec<(String, Vec<u8>)> = vec![
        ("sota944_944x128x1_f32".into(), shape(944, 128, 1, 0, 0x944f)),
        ("v018_228x384x1_i8".into(), shape(228, 384, 1, 2, 0xfeed)),
        ("webp_51x64x24_f16".into(), shape(51, 64, 24, 1, 0xb33f)),
    ];
    if let Ok(list) = std::env::var("ZPGRAPH_BENCH_BAKES") {
        for p in list.split(';').filter(|p| !p.is_empty()) {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
            let name = std::path::Path::new(p)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.to_string());
            bakes.push((name, bytes));
        }
    }
    for (name, bytes) in bakes {
        let old: &'static zp_old::Model = leak(zp_old::Model::from_bytes(&bytes).expect("old load"));
        let new: &'static zp_new::Model = leak(zp_new::Model::from_bytes(&bytes).expect("new load"));
        let v = make_vectors(
            old.n_inputs(),
            old.caller_input_width(),
            old.scaler_mean(),
            old.scaler_scale(),
            256,
            0xbe7c,
        );
        // Realistic rows only (the first 256 are mean ± 3·scale / wide).
        let inputs: &'static [Vec<f32>] = leak(v.features[..256].to_vec());
        let n_in = old.n_inputs();
        let inputs: &'static [Vec<f32>] = leak(
            inputs
                .iter()
                .filter(|r| r.len() == n_in)
                .cloned()
                .collect::<Vec<_>>(),
        );
        suite.group(format!("predict_{name}"), move |g| {
            g.baseline("old (e6c72f99 chain)");
            g.bench("old (e6c72f99 chain)", move |b| {
                let mut p = zp_old::Predictor::new(old);
                let mut i = 0usize;
                b.iter(move || {
                    let out = p.predict(&inputs[i % inputs.len()]).unwrap();
                    i = i.wrapping_add(1);
                    black_box(out[0])
                })
            });
            g.bench("new (graph)", move |b| {
                let mut p = zp_new::Predictor::new(new);
                let mut i = 0usize;
                b.iter(move || {
                    let out = p.predict(&inputs[i % inputs.len()]).unwrap();
                    i = i.wrapping_add(1);
                    black_box(out[0])
                })
            });
        });
    }
});
