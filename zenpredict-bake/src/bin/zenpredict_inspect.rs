//! `zenpredict-inspect` — load a ZNPR v2 `.bin` and dump everything
//! the loader sees as a single JSON document on stdout. Used by
//! `zentrain/tools/inspect_picker.py` for tree-approximation,
//! pick-distribution, pathology / boundary-stress, and confidence
//! analyses.
//!
//! ```text
//! zenpredict-inspect <model.bin>            # dumps full JSON to stdout
//! zenpredict-inspect <model.bin> --weights  # also include weights/biases (large)
//! ```
//!
//! Without `--weights` the output omits the per-layer weight arrays
//! (still hundreds of KB on a typical bake) and keeps just the
//! shape: dims, activations, scaler, feature_bounds, metadata. With
//! `--weights` the dump is sufficient to reconstruct the forward
//! pass numerically in any language.
//!
//! Exit codes:
//!   0  success
//!   1  IO error reading input
//!   2  parse error from `Model::from_bytes`

use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{Map, Value, json};
use zenpredict::{Activation, Metadata, MetadataType, Model, WeightStorage};

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let input = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: zenpredict-inspect <model.bin> [--weights]");
            return ExitCode::from(1);
        }
    };
    let want_weights = args.any(|a| a == "--weights");

    let bytes = match std::fs::read(&input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("zenpredict-inspect: read {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };
    // Re-align into a u64-backed buffer (mirrors what the
    // load_baked_model example does) so cast_slice succeeds.
    let n_u64 = bytes.len().div_ceil(8);
    let mut storage: Vec<u64> = vec![0; n_u64];
    let view: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
    view[..bytes.len()].copy_from_slice(&bytes);
    let aligned: &[u8] = &bytemuck::cast_slice::<u64, u8>(&storage)[..bytes.len()];

    let model = match Model::from_bytes(aligned) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("zenpredict-inspect: parse: {e}");
            return ExitCode::from(2);
        }
    };

    let mut out = Map::new();
    out.insert("file".into(), json!(input.display().to_string()));
    out.insert("file_bytes".into(), json!(bytes.len()));
    out.insert("version".into(), json!(model.version()));
    out.insert("flags".into(), json!(model.flags()));
    out.insert("n_inputs".into(), json!(model.n_inputs()));
    out.insert("n_outputs".into(), json!(model.n_outputs()));
    out.insert("n_layers".into(), json!(model.n_layers()));
    out.insert(
        "schema_hash".into(),
        json!(format!("0x{:016x}", model.schema_hash())),
    );

    out.insert("scaler_mean".into(), json!(model.scaler_mean()));
    out.insert("scaler_scale".into(), json!(model.scaler_scale()));

    let bounds: Vec<Value> = model
        .feature_bounds()
        .iter()
        .map(|b| json!({ "low": b.low, "high": b.high }))
        .collect();
    out.insert("feature_bounds".into(), json!(bounds));

    let layers: Vec<Value> = model
        .layers()
        .iter()
        .map(|layer| {
            let activation = match layer.activation {
                Activation::Identity => "identity",
                Activation::Relu => "relu",
                Activation::LeakyRelu => "leakyrelu",
            };
            let (dtype, n_weights, scales): (&str, usize, Option<&[f32]>) = match &layer.weights {
                WeightStorage::F32(w) => ("f32", w.len(), None),
                WeightStorage::F16(w) => ("f16", w.len(), None),
                WeightStorage::I8 { weights, scales } => ("i8", weights.len(), Some(scales)),
            };
            let mut obj = json!({
                "in_dim": layer.in_dim,
                "out_dim": layer.out_dim,
                "activation": activation,
                "dtype": dtype,
                "n_weights": n_weights,
            });
            obj["biases_len"] = json!(layer.biases.len());
            // Weight magnitude summary statistics. Cheap to compute,
            // catches dead-layer / saturated-layer bakes without
            // shipping the full weight array.
            let mag = weight_magnitude_stats(layer);
            obj["weight_stats"] = mag;
            if let Some(s) = scales {
                obj["i8_scales"] = json!(s);
            }
            if want_weights {
                let weights_arr: Vec<f32> = match &layer.weights {
                    WeightStorage::F32(w) => w.to_vec(),
                    WeightStorage::F16(w) => {
                        w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect()
                    }
                    WeightStorage::I8 { weights, scales } => {
                        let mut out = Vec::with_capacity(weights.len());
                        for (idx, w) in weights.iter().enumerate() {
                            let o = idx % layer.out_dim;
                            out.push(*w as f32 * scales[o]);
                        }
                        out
                    }
                };
                obj["biases"] = json!(layer.biases);
                obj["weights"] = json!(weights_arr);
            }
            obj
        })
        .collect();
    out.insert("layers".into(), json!(layers));

    out.insert("metadata".into(), metadata_to_json(model.metadata()));

    let serialized = serde_json::to_string_pretty(&Value::Object(out)).unwrap();
    println!("{serialized}");
    ExitCode::SUCCESS
}

fn weight_magnitude_stats(layer: &zenpredict::LayerView<'_>) -> Value {
    let weights: Vec<f32> = match &layer.weights {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect(),
        WeightStorage::I8 { weights, scales } => weights
            .iter()
            .enumerate()
            .map(|(idx, w)| (*w as f32) * scales[idx % layer.out_dim])
            .collect(),
    };
    if weights.is_empty() {
        return json!({});
    }
    let mut abs: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    abs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = abs.len();
    let abs_max = abs[n - 1];
    let abs_p50 = abs[n / 2];
    let abs_p99 = abs[(n * 99 / 100).min(n - 1)];
    let near_zero = abs.iter().filter(|x| **x < 1e-4).count();
    let mean_abs: f32 = abs.iter().sum::<f32>() / n as f32;
    json!({
        "abs_max": abs_max,
        "abs_p50": abs_p50,
        "abs_p99": abs_p99,
        "abs_mean": mean_abs,
        "near_zero_fraction": near_zero as f32 / n as f32,
    })
}

fn metadata_to_json(md: &Metadata<'_>) -> Value {
    let entries: Vec<Value> = md
        .iter()
        .map(|e| {
            let kind = match e.kind {
                MetadataType::Bytes => "bytes",
                MetadataType::Utf8 => "utf8",
                MetadataType::Numeric => "numeric",
                MetadataType::Reserved(b) => {
                    return json!({
                        "key": e.key,
                        "kind": format!("reserved({b})"),
                        "value_len": e.value.len(),
                        "value_hex": hex_of(e.value),
                    });
                }
            };
            // Surface text inline for utf8; hex for everything else.
            let mut obj = json!({
                "key": e.key,
                "kind": kind,
                "value_len": e.value.len(),
            });
            match e.kind {
                MetadataType::Utf8 => {
                    obj["value_text"] = match core::str::from_utf8(e.value) {
                        Ok(s) => json!(s),
                        Err(_) => json!(null),
                    };
                }
                _ => {
                    obj["value_hex"] = json!(hex_of(e.value));
                    // Best-effort numeric decode for short payloads —
                    // helpers for human reading. Not authoritative.
                    if matches!(e.kind, MetadataType::Numeric) {
                        if e.value.len() == 1 {
                            obj["value_u8"] = json!(e.value[0]);
                        }
                        if e.value.len() == 4 {
                            let bytes: [u8; 4] = e.value.try_into().unwrap();
                            obj["value_f32"] = json!(f32::from_le_bytes(bytes));
                            obj["value_u32"] = json!(u32::from_le_bytes(bytes));
                        }
                        if e.value.len() == 12 && e.value.len() % 4 == 0 {
                            let f0 = f32::from_le_bytes(e.value[0..4].try_into().unwrap());
                            let f1 = f32::from_le_bytes(e.value[4..8].try_into().unwrap());
                            let f2 = f32::from_le_bytes(e.value[8..12].try_into().unwrap());
                            obj["value_f32_3"] = json!([f0, f1, f2]);
                        }
                        if e.value.len() % 4 == 0 && e.value.len() >= 4 && e.value.len() <= 8192 {
                            let n = e.value.len() / 4;
                            let arr: Vec<f32> = (0..n)
                                .map(|i| {
                                    f32::from_le_bytes(
                                        e.value[i * 4..(i + 1) * 4].try_into().unwrap(),
                                    )
                                })
                                .collect();
                            obj["value_f32_array"] = json!(arr);
                        }
                    }
                }
            }
            obj
        })
        .collect();
    json!(entries)
}

fn hex_of(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}
