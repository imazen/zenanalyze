//! Bake the hand-built gated-head example graph (`gated_head.json`, the
//! E33 arm-B shape `g = Σ_k v_k·ReLU(w_k·d)·exp(u_k·r)`) and run it.
//!
//! ```text
//! cargo run -p zenpredict-bake --example gated_head [-- out.bin]
//! ```
//!
//! Prints the graph and a few predictions, including the gate property:
//! with the difference inputs `d` at zero the output is exactly `+0.0`,
//! whatever the reference inputs `r` are.

use zenpredict::{Model, NodeView, Predictor};

fn main() {
    let bytes = zenpredict_bake::bake_from_json_str(include_str!("gated_head.json"))
        .expect("bake gated_head.json");
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(&path, &bytes).expect("write bake");
        println!("wrote {path} ({} bytes)", bytes.len());
    }
    let model = Model::from_bytes(&bytes).expect("load");
    println!(
        "ZNPR v{}: {} nodes, {} inputs -> {} output(s), layer chain: {}",
        model.version(),
        model.n_nodes(),
        model.n_inputs(),
        model.n_outputs(),
        model.is_layer_chain()
    );
    for (i, node) in model.nodes().enumerate() {
        let desc = match node {
            NodeView::Input { width, .. } => format!("Input width={width}"),
            NodeView::Dense { input, layer, .. } => format!(
                "Dense(node {input}) {}x{} {:?} bias={}",
                layer.in_dim,
                layer.out_dim,
                layer.activation,
                !layer.biases.is_empty()
            ),
            NodeView::Gather { input, indices, .. } => format!("Gather(node {input}) {indices:?}"),
            NodeView::Mul { a, b, width, .. } => format!("Mul(node {a}, node {b}) width={width}"),
            other => format!("{other:?}"),
        };
        println!("  node {i}: {desc}");
    }
    let mut p = Predictor::new(&model);
    for x in [
        [0.5f32, -0.25, 1.0, 0.5, -1.0, 2.0],
        [1.0, 1.0, 1.0, 10.0, 3.0, -4.0],
        [0.0, 0.0, 0.0, 1.0e30, -7.5, 123.0],
        [0.0, 0.0, 0.0, -3.0, 0.25, 9.0],
    ] {
        let g = p.predict(&x).expect("predict")[0];
        println!("  g({x:?}) = {g:e} (bits {:#010x})", g.to_bits());
    }
}
