// Shared core for the ZNPR v4 graph fuzz targets, `tests/fuzz_regression.rs`
// and zenpredict-bake's `tests/graph.rs` (each pulls it in with `include!`),
// so a replayed seed runs exactly what the fuzzer ran and the tests use the
// same reference evaluator the differential fuzz target uses.
//
// Needs `zenpredict` and `libm` in scope as crates.

/// Activation math as the spec states it (`docs/ZNPR_V4_GRAPH.md`).
#[allow(dead_code)]
fn ref_act(v: f32, a: zenpredict::Activation) -> f32 {
    use zenpredict::Activation;
    match a {
        Activation::Identity => v,
        Activation::Relu => {
            if v < 0.0 {
                0.0
            } else {
                v
            }
        }
        Activation::LeakyRelu => {
            if v < 0.0 {
                v * 0.01
            } else {
                v
            }
        }
        Activation::Exp => libm::expf(v.clamp(-30.0, 30.0)),
        Activation::Softplus => {
            if v > 20.0 {
                v
            } else {
                libm::log1pf(libm::expf(v))
            }
        }
        other => panic!("reference evaluator: unknown activation {other:?}"),
    }
}

/// `act(b + x·W)` element by element, accumulating inputs in index order
/// and skipping exact-zero inputs — the order the spec fixes for Dense.
/// `fused`: a `std` zenpredict build accumulates with fused multiply-add
/// (`f32::mul_add`); a `no_std` build uses `a * b + c` (two roundings, see
/// `inference::fma`). The reference models whichever build it checks.
#[allow(dead_code)]
fn ref_dense(l: &zenpredict::LayerView<'_>, x: &[f32], fused: bool) -> Vec<f32> {
    use zenpredict::WeightStorage;
    let out = l.out_dim;
    let is_i8 = matches!(l.weights, WeightStorage::I8 { .. });
    let mut acc = if is_i8 || l.biases.is_empty() {
        vec![0.0f32; out]
    } else {
        l.biases.to_vec()
    };
    for (i, &s) in x.iter().enumerate() {
        if s == 0.0 {
            continue;
        }
        for (o, a) in acc.iter_mut().enumerate() {
            let w = match &l.weights {
                WeightStorage::F32(w) => w[i * out + o],
                WeightStorage::F16(w) => zenpredict::f16_bits_to_f32(w[i * out + o]),
                WeightStorage::I8 { weights, .. } => weights[i * out + o] as f32,
            };
            *a = if fused { s.mul_add(w, *a) } else { s * w + *a };
        }
    }
    if let WeightStorage::I8 { scales, .. } = &l.weights {
        for o in 0..out {
            let b = if l.biases.is_empty() {
                0.0
            } else {
                l.biases[o]
            };
            acc[o] = b + scales[o] * acc[o];
        }
    }
    acc.iter().map(|&v| ref_act(v, l.activation)).collect()
}

/// Evaluate every node into its own vector — no arena, no slot reuse, no
/// SIMD dispatch — modelling a `std` (fused multiply-add) zenpredict build.
/// `features` must be `n_inputs` long.
#[allow(dead_code)]
fn reference_eval(model: &zenpredict::Model, features: &[f32]) -> Vec<f32> {
    reference_eval_with(model, features, true)
}

/// [`reference_eval`] with the multiply-add rule chosen explicitly
/// (`fused = false` models a `no_std` zenpredict build).
#[allow(dead_code)]
fn reference_eval_with(model: &zenpredict::Model, features: &[f32], fused: bool) -> Vec<f32> {
    use zenpredict::NodeView;
    let mut vals: Vec<Vec<f32>> = Vec::new();
    for node in model.nodes() {
        let v = match node {
            NodeView::Input { .. } => features
                .iter()
                .zip(model.scaler_mean().iter().zip(model.scaler_scale()))
                .map(|(&x, (&m, &s))| (x - m) / if s == 0.0 { 1.0 } else { s })
                .collect(),
            NodeView::Dense { input, layer, .. } => {
                ref_dense(&layer, &vals[input as usize], fused)
            }
            NodeView::Activation {
                input, activation, ..
            } => vals[input as usize]
                .iter()
                .map(|&v| ref_act(v, activation))
                .collect(),
            NodeView::Gather { input, indices, .. } => indices
                .iter()
                .map(|&k| vals[input as usize][k as usize])
                .collect(),
            NodeView::Add { a, b, .. } => vals[a as usize]
                .iter()
                .zip(&vals[b as usize])
                .map(|(x, y)| x + y)
                .collect(),
            NodeView::Mul { a, b, .. } => vals[a as usize]
                .iter()
                .zip(&vals[b as usize])
                .map(|(x, y)| x * y)
                .collect(),
            NodeView::Concat { inputs, .. } => inputs
                .iter()
                .flat_map(|&j| vals[j as usize].iter().copied())
                .collect(),
            other => panic!("reference evaluator: unknown node {other:?}"),
        };
        vals.push(v);
    }
    vals.pop().unwrap()
}

/// Same value for the differential check: identical bits, or both NaN
/// (NaN payloads may legitimately differ between a software `fmaf` in the
/// reference and a hardware FMA in the executor).
#[allow(dead_code)]
fn same_value(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The fixed probe vectors every exercise runs.
#[allow(dead_code)]
fn probe_vectors(n_in: usize) -> [Vec<f32>; 3] {
    [
        vec![0.0f32; n_in],
        (0..n_in).map(|i| i as f32 * 0.37 - 3.0).collect(),
        (0..n_in)
            .map(|i| if i % 3 == 0 { 1.0e30 } else { -45.0 })
            .collect(),
    ]
}

/// Load `bytes`; on success walk every node view, build a predictor and
/// run the forward pass on the probe vectors. With `differential:
/// Some(fused)`, also require every output to equal
/// [`reference_eval_with`]`(.., fused)` (panics otherwise, which a fuzzer
/// reports as a finding). `fused` must match the zenpredict build under
/// test: `true` with `std`, `false` without.
#[allow(dead_code)]
fn exercise_model_bytes(bytes: &[u8], differential: Option<bool>) {
    let Ok(model) = zenpredict::Model::from_bytes(bytes) else {
        return;
    };
    let mut touched = 0usize;
    for node in model.nodes() {
        touched += node.width();
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
    for x in probe_vectors(n_in) {
        let Ok(got) = p.predict(&x) else {
            continue;
        };
        if let Some(fused) = differential {
            let want = reference_eval_with(&model, &x, fused);
            assert_eq!(got.len(), want.len(), "output width");
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    same_value(g, w),
                    "executor vs reference differ at output {i}: {g} ({:#010x}) vs {w} ({:#010x})",
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
    }
    let w = model.caller_input_width();
    if w <= zenpredict::limits::MAX_DIM {
        let _ = p.predict_transformed(&vec![0.5f32; w]);
    }
}

/// Crash-only exercise (the `graph_from_bytes` contract).
#[allow(dead_code)]
fn run_model_bytes(bytes: &[u8]) {
    exercise_model_bytes(bytes, None);
}
