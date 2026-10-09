// Shared exercise core for the ZNPR v4 graph fuzz targets AND
// `tests/fuzz_regression.rs` (pulled in with `include!`), so a replayed
// seed runs exactly what the fuzzer ran. Must never panic.

/// Load `bytes`; on success walk every node view, build a predictor and
/// run the forward pass on a few feature vectors.
#[allow(dead_code)]
fn run_model_bytes(bytes: &[u8]) {
    let Ok(model) = zenpredict::Model::from_bytes(bytes) else {
        return;
    };
    let mut touched = 0usize;
    for node in model.nodes() {
        touched += match node {
            zenpredict::NodeView::Input { width } => width,
            zenpredict::NodeView::Dense { input, layer } => input + layer.biases.len() + layer.out_dim,
            zenpredict::NodeView::Activation { input, width, .. } => input + width,
            zenpredict::NodeView::Gather { input, indices } => input + indices.len(),
            zenpredict::NodeView::Add { a, b, width } | zenpredict::NodeView::Mul { a, b, width } => {
                a + b + width
            }
            zenpredict::NodeView::Concat { inputs, width } => inputs.len() + width,
            _ => 0,
        };
    }
    for layer in model.layers() {
        touched += layer.in_dim;
    }
    std::hint::black_box(touched);
    let n_in = model.n_inputs();
    if n_in > zenpredict::limits::MAX_DIM {
        return;
    }
    let Ok(mut p) = zenpredict::Predictor::try_new(&model) else {
        return;
    };
    let _ = p.predict(&vec![0.0f32; n_in]);
    let ramp: Vec<f32> = (0..n_in).map(|i| i as f32 * 0.37 - 3.0).collect();
    let _ = p.predict(&ramp);
    let wide: Vec<f32> = (0..n_in)
        .map(|i| if i % 3 == 0 { 1.0e30 } else { -45.0 })
        .collect();
    let _ = p.predict(&wide);
    let w = model.caller_input_width();
    if w <= zenpredict::limits::MAX_DIM {
        let _ = p.predict_transformed(&vec![0.5f32; w]);
    }
}
