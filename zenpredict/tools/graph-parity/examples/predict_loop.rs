//! Fixed-count predict loop for `perf stat` A/B (instructions are
//! deterministic, unlike wall clock on a shared box).
//!
//! ```text
//! predict_loop old|new sota944|v018|webp|<bake.bin> <iters>
//! ```

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (side, what, iters) = (&a[0], &a[1], a[2].parse::<usize>().unwrap());
    let bytes = match what.as_str() {
        "sota944" => zenpredict_graph_parity::bench_shape(944, 128, 1, 0, 0x944f),
        "v018" => zenpredict_graph_parity::bench_shape(228, 384, 1, 2, 0xfeed),
        "webp" => zenpredict_graph_parity::bench_shape(51, 64, 24, 1, 0xb33f),
        p => std::fs::read(p).unwrap(),
    };
    let old = zp_old::Model::from_bytes(&bytes).unwrap();
    let new = zp_new::Model::from_bytes(&bytes).unwrap();
    let v = zenpredict_graph_parity::make_vectors(
        old.n_inputs(),
        old.caller_input_width(),
        old.scaler_mean(),
        old.scaler_scale(),
        256,
        0xbe7c,
    );
    let rows: Vec<&Vec<f32>> = v.features[..256]
        .iter()
        .filter(|r| r.len() == old.n_inputs())
        .collect();
    let mut acc = 0f32;
    if side == "old" {
        let mut p = zp_old::Predictor::new(&old);
        for i in 0..iters {
            acc += p.predict(rows[i % rows.len()]).unwrap()[0];
        }
    } else {
        let mut p = zp_new::Predictor::new(&new);
        for i in 0..iters {
            acc += p.predict(rows[i % rows.len()]).unwrap()[0];
        }
    }
    println!("{acc}");
}
