//! ZNPR v3 → ONNX converter.
//!
//! Reads a ZNPR v3 bake and emits a valid ONNX file that Netron and
//! ort-style runtimes can load. The exported graph covers the
//! standardize step + every MLP layer; post-MLP zensim calibration
//! stages (`zentrain.tanh_output_head`, `zentrain.output_calibration_spline`,
//! `zentrain.per_codec_calibration`, `zentrain.per_sample_alpha_head`)
//! are NOT exported — they have no clean ONNX-native equivalents and
//! tooling round-trip is more valuable for the MLP itself. The
//! converter prints a notice when those stages are present so the user
//! knows the ONNX is the rank head only.
//!
//! Graph shape:
//!   Input  : `features` [1, N]
//!   Initializers: `scaler_mean` [N], `scaler_scale_inv` [N]
//!   Op     : Sub(features, scaler_mean) → s0
//!   Op     : Mul(s0, scaler_scale_inv) → s1   (standardized)
//!   For each layer k:
//!     Initializers: `W_k` [in_dim, out_dim], `B_k` [out_dim]
//!     Op   : Gemm(s_{2k+1}, W_k, B_k) → pre_k
//!     Op   : <activation>(pre_k) → s_{2(k+1)+1}
//!   Output : final post-activation [1, out_dim_last]
//!
//! Activations: Identity → no op, Relu → `Relu`, LeakyRelu → `LeakyRelu(alpha=0.01)`.
//! I8 weights are dequantized to F32 before serialization (ONNX has
//! no compact per-output-column quant scheme that maps 1:1 to ZNPR's).

use onnx_pb::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TypeProto,
    ValueInfoProto, attribute_proto::AttributeType, tensor_proto::DataType,
    tensor_shape_proto::Dimension, type_proto::Value as TypeProtoValue,
};
use prost::Message;
use std::env;
use std::path::Path;
use std::process::ExitCode;
use zenpredict::{Activation, Model, WeightStorage, f16_bits_to_f32};

const USAGE: &str = "\
znpr2onnx — ZNPR v3 → ONNX converter

  usage: znpr2onnx <input.bin> <output.onnx> [--name NAME]

  Exports the standardize step + MLP layers of a ZNPR v3 bake as an
  ONNX file Netron can render. Calibration stages (tanh_output_head,
  output_calibration_spline, per_codec_calibration, per_sample_alpha_head)
  are NOT exported — the ONNX represents the MLP rank head only.
  Post-MLP calibration is a few lines of Rust / Python at most; if you
  need round-trip fidelity for those, run the production zenpredict
  runtime instead.

  Output uses opset 13 (Gemm + LeakyRelu + Relu + Sub + Mul covers it)
  with IR version 8 — broadly compatible with Netron, ONNX Runtime ≥ 1.13.
";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if args.len() < 2 {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let input_path = &args[0];
    let output_path = &args[1];
    let mut name = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("znpr_model")
        .to_string();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                if i + 1 >= args.len() {
                    eprintln!("--name requires a value");
                    return ExitCode::from(2);
                }
                name = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}\n\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let bytes = match std::fs::read(input_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {input_path}: {e}");
            return ExitCode::from(1);
        }
    };

    let model = match Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse {input_path}: {e}");
            return ExitCode::from(1);
        }
    };

    let calibration_warnings = collect_calibration_warnings(&model);
    if !calibration_warnings.is_empty() {
        eprintln!("note: bake carries post-MLP calibration stages NOT exported to ONNX:");
        for w in &calibration_warnings {
            eprintln!("  - {w}");
        }
        eprintln!("      (the ONNX represents the standardize + MLP rank head only.)");
    }

    let onnx = match build_onnx_model(&model, &name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("build onnx: {e}");
            return ExitCode::from(1);
        }
    };

    let mut buf = Vec::with_capacity(8192);
    if let Err(e) = onnx.encode(&mut buf) {
        eprintln!("encode onnx: {e}");
        return ExitCode::from(1);
    }
    if let Err(e) = std::fs::write(output_path, &buf) {
        eprintln!("write {output_path}: {e}");
        return ExitCode::from(1);
    }

    eprintln!(
        "✓ wrote {} ({} bytes; inputs={}, outputs={}, layers={})",
        output_path,
        buf.len(),
        model.n_inputs(),
        model.n_outputs(),
        model.n_layers()
    );
    ExitCode::SUCCESS
}

fn collect_calibration_warnings(model: &Model) -> Vec<&'static str> {
    let mut keys: Vec<&'static str> = Vec::new();
    let md = model.metadata();
    for k in [
        "zentrain.tanh_output_head",
        "zentrain.output_calibration_spline",
        "zentrain.per_codec_calibration",
        "zentrain.per_sample_alpha_head",
        "zentrain.feature_transforms",
    ] {
        if md.get(k).is_some() {
            keys.push(k);
        }
    }
    keys
}

fn build_onnx_model(model: &Model, name: &str) -> Result<ModelProto, String> {
    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    let n_layers = model.n_layers();
    let scaler_mean: Vec<f32> = model.scaler_mean().to_vec();
    let scaler_scale: Vec<f32> = model.scaler_scale().to_vec();
    let scaler_scale_inv: Vec<f32> = scaler_scale
        .iter()
        .map(|&s| if s == 0.0 { 1.0 } else { 1.0 / s })
        .collect();

    let mut initializers: Vec<TensorProto> = Vec::new();
    let mut nodes: Vec<NodeProto> = Vec::new();

    // Standardize: (features - mean) * (1/scale).
    initializers.push(float_tensor(
        "scaler_mean",
        &scaler_mean,
        &[n_inputs as i64],
    ));
    initializers.push(float_tensor(
        "scaler_scale_inv",
        &scaler_scale_inv,
        &[n_inputs as i64],
    ));
    nodes.push(make_node(
        "Sub",
        "Sub_standardize",
        &["features", "scaler_mean"],
        &["__pre_scale"],
        &[],
    ));
    nodes.push(make_node(
        "Mul",
        "Mul_standardize",
        &["__pre_scale", "scaler_scale_inv"],
        &["__standardized"],
        &[],
    ));

    let mut current_out = "__standardized".to_string();

    for (idx, layer) in model.layers().enumerate() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;

        // Materialize weights as f32 [in_dim, out_dim] row-major.
        let mut w_dense: Vec<f32> = Vec::with_capacity(in_dim * out_dim);
        match &layer.weights {
            WeightStorage::F32(w) => w_dense.extend_from_slice(w),
            WeightStorage::F16(w) => {
                w_dense.extend(w.iter().map(|&h| f16_bits_to_f32(h)));
            }
            WeightStorage::I8 { weights, scales } => {
                for i in 0..in_dim {
                    let base = i * out_dim;
                    for o in 0..out_dim {
                        w_dense.push(weights[base + o] as f32 * scales[o]);
                    }
                }
            }
        }

        let w_name = format!("W_{idx}");
        let b_name = format!("B_{idx}");
        initializers.push(float_tensor(
            &w_name,
            &w_dense,
            &[in_dim as i64, out_dim as i64],
        ));
        initializers.push(float_tensor(&b_name, layer.biases, &[out_dim as i64]));

        let pre_name = format!("pre_{idx}");
        let post_name = if idx + 1 == n_layers {
            "output".to_string()
        } else {
            format!("post_{idx}")
        };

        // Gemm: Y = alpha * (A x B) + beta * C  with transA/transB = 0,
        // alpha=beta=1. C is the bias and broadcasts across the batch.
        nodes.push(make_node(
            "Gemm",
            &format!("Gemm_{idx}"),
            &[current_out.as_str(), w_name.as_str(), b_name.as_str()],
            &[pre_name.as_str()],
            &[
                make_attr_float("alpha", 1.0),
                make_attr_float("beta", 1.0),
                make_attr_int("transA", 0),
                make_attr_int("transB", 0),
            ],
        ));

        match layer.activation {
            Activation::Identity => {
                if idx + 1 == n_layers {
                    // Need to rename pre_{last} → "output". Use an Identity op.
                    nodes.push(make_node(
                        "Identity",
                        &format!("Identity_{idx}"),
                        &[pre_name.as_str()],
                        &[post_name.as_str()],
                        &[],
                    ));
                } else {
                    nodes.push(make_node(
                        "Identity",
                        &format!("Identity_{idx}"),
                        &[pre_name.as_str()],
                        &[post_name.as_str()],
                        &[],
                    ));
                }
            }
            Activation::Relu => {
                nodes.push(make_node(
                    "Relu",
                    &format!("Relu_{idx}"),
                    &[pre_name.as_str()],
                    &[post_name.as_str()],
                    &[],
                ));
            }
            Activation::LeakyRelu => {
                nodes.push(make_node(
                    "LeakyRelu",
                    &format!("LeakyRelu_{idx}"),
                    &[pre_name.as_str()],
                    &[post_name.as_str()],
                    &[make_attr_float("alpha", 0.01)],
                ));
            }
            _ => {
                return Err(format!(
                    "znpr2onnx: cannot export Activation {:?} to ONNX; \
                     this converter build does not support it",
                    layer.activation
                ));
            }
        }

        current_out = post_name;
    }

    let input_info = make_value_info("features", &[1, n_inputs as i64]);
    let output_info = make_value_info("output", &[1, n_outputs as i64]);

    let mut doc_parts = vec![format!(
        "Generated from ZNPR v3 bake by znpr2onnx. n_inputs={n_inputs}, n_outputs={n_outputs}, n_layers={n_layers}.",
    )];
    let extra = collect_calibration_warnings(model);
    if !extra.is_empty() {
        doc_parts.push(format!(
            "Post-MLP calibration stages NOT included: {}.",
            extra.join(", ")
        ));
    }

    let graph = GraphProto {
        node: nodes,
        name: format!("{name}_graph"),
        initializer: initializers,
        input: vec![input_info],
        output: vec![output_info],
        doc_string: doc_parts.join(" "),
        ..GraphProto::default()
    };

    Ok(ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "znpr2onnx".to_string(),
        producer_version: env!("CARGO_PKG_VERSION").to_string(),
        graph: Some(graph),
        ..ModelProto::default()
    })
}

fn float_tensor(name: &str, data: &[f32], dims: &[i64]) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: DataType::Float as i32,
        float_data: data.to_vec(),
        name: name.to_string(),
        ..TensorProto::default()
    }
}

fn make_node(
    op_type: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: &[AttributeProto],
) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        attribute: attrs.to_vec(),
        ..NodeProto::default()
    }
}

fn make_attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Float as i32,
        f: value,
        ..AttributeProto::default()
    }
}

fn make_attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Int as i32,
        i: value,
        ..AttributeProto::default()
    }
}

fn make_value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
    let shape = onnx_pb::TensorShapeProto {
        dim: dims
            .iter()
            .map(|&d| Dimension {
                value: Some(onnx_pb::tensor_shape_proto::dimension::Value::DimValue(d)),
                ..Dimension::default()
            })
            .collect(),
    };
    let tensor_type = onnx_pb::type_proto::Tensor {
        elem_type: DataType::Float as i32,
        shape: Some(shape),
    };
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            denotation: String::new(),
            value: Some(TypeProtoValue::TensorType(tensor_type)),
        }),
        ..ValueInfoProto::default()
    }
}
