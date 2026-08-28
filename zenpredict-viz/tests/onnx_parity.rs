//! `znpr2onnx` parity (zenanalyze#79 Track A): run the converter binary
//! on bakes composed in-test, decode the ONNX with the same `onnx-pb`
//! types the converter emits, check the graph shape, and evaluate the
//! exported graph here (Sub → Mul → Gemm → activation …) against
//! `zenpredict::Predictor` — no onnxruntime needed. Compiled only under
//! the `onnx-export` feature (which is what builds the binary); the
//! feature is the caller's explicit switch, there is no in-test skip.
#![cfg(feature = "onnx-export")]

mod common;

use common::{HIDDEN, N_IN, N_OUT, Synth, all_activations, all_dtypes, probe_features, synth_bake};
use onnx_pb::{ModelProto, TensorProto};
use prost::Message;
use std::path::PathBuf;
use std::process::Command;
use zenpredict::{Activation, Model, Predictor, WeightDtype};

fn tmp_dir(tag: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("onnx_parity_{tag}"));
    std::fs::create_dir_all(&d).expect("tmpdir");
    d
}

fn tensor_f32(t: &TensorProto) -> Vec<f32> {
    if !t.float_data.is_empty() {
        return t.float_data.clone();
    }
    t.raw_data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn convert(bytes: &[u8], tag: &str) -> ModelProto {
    let dir = tmp_dir(tag);
    let bin = dir.join("bake.bin");
    let onnx = dir.join("bake.onnx");
    std::fs::write(&bin, bytes).expect("write bake");
    let out = Command::new(env!("CARGO_BIN_EXE_znpr2onnx"))
        .arg(&bin)
        .arg(&onnx)
        .arg("--name")
        .arg(tag)
        .output()
        .expect("run znpr2onnx");
    assert!(
        out.status.success(),
        "znpr2onnx failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let onnx_bytes = std::fs::read(&onnx).expect("read onnx");
    ModelProto::decode(onnx_bytes.as_slice()).expect("decode ModelProto")
}

/// Evaluate the exported graph the way onnxruntime would, node by node.
fn eval_graph(m: &ModelProto, features: &[f32]) -> Vec<f32> {
    let g = m.graph.as_ref().expect("graph");
    let init: std::collections::HashMap<&str, Vec<f32>> = g
        .initializer
        .iter()
        .map(|t| (t.name.as_str(), tensor_f32(t)))
        .collect();
    let dims: std::collections::HashMap<&str, Vec<i64>> = g
        .initializer
        .iter()
        .map(|t| (t.name.as_str(), t.dims.clone()))
        .collect();
    let mut env: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    env.insert("features".into(), features.to_vec());
    for node in &g.node {
        let a = |name: &str| -> Vec<f32> {
            env.get(name)
                .cloned()
                .or_else(|| init.get(name).cloned())
                .unwrap_or_else(|| panic!("unknown tensor {name}"))
        };
        let out = match node.op_type.as_str() {
            "Sub" => {
                let (x, y) = (a(&node.input[0]), a(&node.input[1]));
                x.iter().zip(&y).map(|(p, q)| p - q).collect()
            }
            "Mul" => {
                let (x, y) = (a(&node.input[0]), a(&node.input[1]));
                x.iter().zip(&y).map(|(p, q)| p * q).collect()
            }
            "Gemm" => {
                let x = a(&node.input[0]);
                let w = a(&node.input[1]);
                let b = a(&node.input[2]);
                let d = &dims[node.input[1].as_str()];
                let (in_dim, out_dim) = (d[0] as usize, d[1] as usize);
                assert_eq!(x.len(), in_dim);
                (0..out_dim)
                    .map(|o| b[o] + (0..in_dim).map(|i| x[i] * w[i * out_dim + o]).sum::<f32>())
                    .collect()
            }
            "Relu" => a(&node.input[0]).iter().map(|v| v.max(0.0)).collect(),
            "LeakyRelu" => {
                let alpha = node
                    .attribute
                    .iter()
                    .find(|at| at.name == "alpha")
                    .map(|at| at.f)
                    .expect("alpha");
                a(&node.input[0])
                    .iter()
                    .map(|&v| if v < 0.0 { v * alpha } else { v })
                    .collect()
            }
            "Identity" => a(&node.input[0]),
            other => panic!("unexpected op {other}"),
        };
        env.insert(node.output[0].clone(), out);
    }
    env.remove("output").expect("graph output")
}

#[test]
fn exported_graph_matches_zenpredict_for_every_dtype_and_activation() {
    let synth = Synth::new(0x0DD);
    for dtype in all_dtypes() {
        for activation in all_activations() {
            let bytes = synth_bake(&synth, dtype, activation, false, false);
            let tag = format!("{dtype:?}_{activation:?}").to_lowercase();
            let m = convert(&bytes, &tag);
            assert_eq!(m.ir_version, 8);
            assert_eq!(m.opset_import[0].version, 13);
            let g = m.graph.as_ref().unwrap();
            let ops: Vec<&str> = g.node.iter().map(|n| n.op_type.as_str()).collect();
            let hidden_op = match activation {
                Activation::Relu => "Relu",
                Activation::LeakyRelu => "LeakyRelu",
                _ => "Identity",
            };
            assert_eq!(
                ops,
                ["Sub", "Mul", "Gemm", hidden_op, "Gemm", "Identity"],
                "{tag}"
            );
            assert_eq!(g.input[0].name, "features");
            assert_eq!(g.output[0].name, "output");
            let w0 = g.initializer.iter().find(|t| t.name == "W_0").unwrap();
            assert_eq!(w0.dims, [N_IN as i64, HIDDEN as i64]);
            // Dequantized weights in the graph == what the viz shows.
            let viz_w0 = zenpredict_viz::layer_weights_native(&bytes, 0).unwrap();
            for (a, b) in tensor_f32(w0).iter().zip(&viz_w0) {
                assert_eq!(a.to_bits(), b.to_bits(), "{tag}: W_0 dequantization");
            }

            let model = Model::from_bytes(&bytes).unwrap();
            let mut p = Predictor::new(&model);
            for seed in [1u64, 5] {
                let f = probe_features(seed);
                let reference = p.predict(&f).unwrap().to_vec();
                let got = eval_graph(&m, &f);
                assert_eq!(got.len(), N_OUT);
                for (i, (r, v)) in reference.iter().zip(&got).enumerate() {
                    // The graph standardizes with `× (1/scale)` and sums
                    // without FMA: f32 rounding only, no structural drift.
                    let tol = r.abs().max(v.abs()) * 2e-5 + 1e-6;
                    assert!(
                        (r - v).abs() <= tol,
                        "{tag}: output[{i}] zenpredict={r} onnx={v}"
                    );
                }
            }
        }
    }
}

#[test]
fn converter_reports_dropped_calibration_stages_in_doc_string() {
    // A bake with output_specs: the converter exports the MLP only and
    // must say what it left out rather than silently emitting a graph
    // that disagrees with `predict_with_specs`.
    let synth = Synth::new(0xF00D);
    let bytes = synth_bake(&synth, WeightDtype::F32, Activation::Relu, true, true);
    let m = convert(&bytes, "specs");
    let doc = &m.graph.as_ref().unwrap().doc_string;
    assert!(doc.contains("n_inputs=11"), "{doc}");
    // feature_transforms are not part of the exported graph; the runtime
    // applies them before standardize, so the ONNX output equals
    // `predict` (untransformed), not `predict_transformed`.
    let model = Model::from_bytes(&bytes).unwrap();
    let mut p = Predictor::new(&model);
    let f = probe_features(3);
    let raw = p.predict(&f).unwrap().to_vec();
    let got = eval_graph(&m, &f);
    for (r, v) in raw.iter().zip(&got) {
        assert!((r - v).abs() <= r.abs().max(v.abs()) * 2e-5 + 1e-6);
    }
    assert!(
        doc.contains("NOT included")
            && doc.contains("feature_transforms")
            && doc.contains("output_specs"),
        "doc string must name the dropped stages: {doc}"
    );
}
