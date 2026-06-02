//! Shared CLI entry points for the `zenpredict-bake`,
//! `zenpredict-inspect`, and unified `zenpredict` binaries.
//!
//! Each `run_*_cli` function takes an argv slice (positional
//! arguments only — `argv[0]` is NOT included) and returns an
//! `ExitCode`. Output formatting, exit codes, and stderr messages
//! match the legacy single-purpose binaries byte-for-byte so any
//! existing caller of `zenpredict-bake` / `zenpredict-inspect`
//! continues to work unchanged.
//!
//! This module is `std`-only; it pulls in file IO, `ExitCode`, and
//! `serde_json` printing. The library's `no_std + alloc` surface
//! lives in `composer`, `json`, `optimize`, and `zero_bias`.

use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{Map, Value, json};
use zenpredict::{
    Activation, FeatureBound, MetadataType, Model, OutputSpec, SparseOverride, WeightDtype,
    WeightStorage, f16_bits_to_f32,
};

use crate::{
    BakeJsonError, BakeLayer, BakeMetadataEntry, BakeRequest, BakeRequestJson,
    apply_zero_bias_per_layer_in_place, bake, bake_from_json, bake_optimized,
};

/// `zenpredict-bake` / `zenpredict bake` body. `argv` is positional
/// arguments (no program name).
///
/// Exit codes:
///   0  success
///   1  IO error reading input or writing output
///   2  JSON parse error
///   3  bake validation failure (BakeJsonError / BakeError)
pub fn run_bake_cli(argv: &[String]) -> ExitCode {
    let mut iter = argv.iter();
    let input = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: zenpredict-bake <input.json> <output.bin>");
            return ExitCode::from(1);
        }
    };
    let output = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: zenpredict-bake <input.json> <output.bin>");
            return ExitCode::from(1);
        }
    };
    if iter.next().is_some() {
        eprintln!("zenpredict-bake: too many arguments");
        return ExitCode::from(1);
    }

    let json_bytes = match std::fs::read(&input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("zenpredict-bake: failed to read {}: {e}", input.display());
            return ExitCode::from(1);
        }
    };

    let req: BakeRequestJson = match serde_json::from_slice(&json_bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "zenpredict-bake: JSON parse error in {}: {e}",
                input.display()
            );
            return ExitCode::from(2);
        }
    };

    let bytes = match bake_from_json(&req) {
        Ok(b) => b,
        Err(BakeJsonError::Bake(e)) => {
            eprintln!("zenpredict-bake: bake error: {e}");
            return ExitCode::from(3);
        }
        Err(e) => {
            eprintln!("zenpredict-bake: input error: {e}");
            return ExitCode::from(3);
        }
    };

    if let Err(e) = std::fs::write(&output, &bytes) {
        eprintln!("zenpredict-bake: failed to write {}: {e}", output.display());
        return ExitCode::from(1);
    }

    eprintln!(
        "zenpredict-bake: wrote {} ({} bytes) — n_inputs={} n_outputs={} n_layers={} schema_hash=0x{:016x} metadata_entries={}",
        output.display(),
        bytes.len(),
        req.scaler_mean.len(),
        req.layers.last().map(|l| l.out_dim).unwrap_or(0),
        req.layers.len(),
        req.schema_hash,
        req.metadata.len(),
    );
    ExitCode::SUCCESS
}

/// `zenpredict-inspect` / `zenpredict inspect` body. `argv` is
/// positional arguments (no program name).
///
/// Exit codes:
///   0  success
///   1  IO error reading input
///   2  parse error from `Model::from_bytes`
pub fn run_inspect_cli(argv: &[String]) -> ExitCode {
    let mut iter = argv.iter();
    let input = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: zenpredict-inspect <model.bin> [--weights]");
            return ExitCode::from(1);
        }
    };
    let want_weights = iter.any(|a| a == "--weights");

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
        .map(|layer| {
            let activation = match layer.activation {
                Activation::Identity => "identity",
                Activation::Relu => "relu",
                Activation::LeakyRelu => "leakyrelu",
                _ => "unknown",
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
            let mag = weight_magnitude_stats(&layer);
            obj["weight_stats"] = mag;
            if let Some(s) = scales {
                obj["i8_scales"] = json!(s);
            }
            if want_weights {
                let weights_arr: Vec<f32> = match &layer.weights {
                    WeightStorage::F32(w) => w.to_vec(),
                    WeightStorage::F16(w) => w.iter().map(|b| f16_bits_to_f32(*b)).collect(),
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

    let md = model.metadata();
    out.insert("metadata".into(), metadata_to_json(&md));

    let serialized = serde_json::to_string_pretty(&Value::Object(out)).unwrap();
    println!("{serialized}");
    ExitCode::SUCCESS
}

/// `zenpredict repack` body. Mirrors `examples/rebake_v3_1.rs`.
/// `argv` is positional arguments (no program name).
///
/// Usage:
///   zenpredict repack <input.bin> <output.bin> [--dtype f32|f16|i8] \
///                     [--zerobias <tau>] [--compress] [--optimize]
///
/// Exit codes:
///   0  success
///   1  IO error / argument error / bake failure
pub fn run_repack_cli(argv: &[String]) -> ExitCode {
    if argv.len() < 2 {
        eprintln!(
            "usage: zenpredict repack <input.bin> <output.bin> [--dtype f32|f16|i8] [--zerobias <tau>] [--compress] [--optimize]"
        );
        return ExitCode::from(1);
    }
    let in_path = PathBuf::from(&argv[0]);
    let out_path = PathBuf::from(&argv[1]);
    let compress = argv.iter().any(|a| a == "--compress");
    let optimize = argv.iter().any(|a| a == "--optimize");

    let zerobias_tau: Option<f32> = match argv.windows(2).find(|w| w[0] == "--zerobias") {
        Some(w) => match w[1].parse() {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("zenpredict repack: --zerobias requires a float: {e}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    let force_dtype: Option<WeightDtype> = match argv.windows(2).find(|w| w[0] == "--dtype") {
        Some(w) => match w[1].as_str() {
            "f32" => Some(WeightDtype::F32),
            "f16" => Some(WeightDtype::F16),
            "i8" => Some(WeightDtype::I8),
            other => {
                eprintln!("zenpredict repack: unknown --dtype: {other}");
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    let input_bytes = match std::fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {}: {}", in_path.display(), e);
            return ExitCode::from(1);
        }
    };

    let model = match Model::from_bytes(&input_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse: {:?}", e);
            return ExitCode::from(1);
        }
    };

    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    let n_layers = model.n_layers();
    eprintln!(
        "loaded: {} inputs, {} outputs, {} layers, {} bytes",
        n_inputs,
        n_outputs,
        n_layers,
        input_bytes.len()
    );

    let scaler_mean: Vec<f32> = model.scaler_mean().to_vec();
    let scaler_scale: Vec<f32> = model.scaler_scale().to_vec();
    let feature_bounds: Vec<FeatureBound> = model.feature_bounds().to_vec();
    let output_specs: Vec<OutputSpec> = model.output_specs().to_vec();
    let discrete_sets: Vec<f32> = model.discrete_sets().to_vec();
    let sparse_overrides: Vec<SparseOverride> = model.sparse_overrides().to_vec();

    // Materialize each layer's weights as f32 + biases as f32, preserving
    // the original dtype choice unless --dtype overrides.
    let mut owned_layers: Vec<(usize, usize, Activation, WeightDtype, Vec<f32>, Vec<f32>)> =
        Vec::with_capacity(n_layers);
    for layer in model.layers() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        let activation = layer.activation;
        let (dtype, weights_f32) = match &layer.weights {
            WeightStorage::F32(w) => (WeightDtype::F32, w.to_vec()),
            WeightStorage::F16(w) => (
                WeightDtype::F16,
                w.iter().map(|b| f16_bits_to_f32(*b)).collect(),
            ),
            WeightStorage::I8 { weights, scales } => {
                let mut out = Vec::with_capacity(weights.len());
                for (idx, w) in weights.iter().enumerate() {
                    let o = idx % out_dim;
                    out.push(*w as f32 * scales[o]);
                }
                (WeightDtype::I8, out)
            }
        };
        let final_dtype = force_dtype.unwrap_or(dtype);
        owned_layers.push((
            in_dim,
            out_dim,
            activation,
            final_dtype,
            weights_f32,
            layer.biases.to_vec(),
        ));
    }

    // Optional zerobias pass: zero out weights with magnitude below
    // tau × per-layer max. Run on f32 BEFORE re-quantization so the
    // i8 path actually carries zeros into the wire bytes.
    if let Some(tau) = zerobias_tau {
        let mut zero_count = 0usize;
        let mut total = 0usize;
        for (_, _out_dim, _, _, weights, _) in owned_layers.iter_mut() {
            apply_zero_bias_per_layer_in_place(weights, tau);
            for w in weights.iter() {
                total += 1;
                if *w == 0.0 {
                    zero_count += 1;
                }
            }
        }
        eprintln!(
            "zerobias τ={}: {} of {} weights zeroed ({:.1}%)",
            tau,
            zero_count,
            total,
            zero_count as f64 / total as f64 * 100.0
        );
    }

    // Materialize metadata entries (raw byte values).
    let md = model.metadata();
    let metadata_owned: Vec<(String, MetadataType, Vec<u8>)> = md
        .iter()
        .map(|e| (e.key.to_string(), e.kind, e.value.to_vec()))
        .collect();
    let metadata_borrowed: Vec<BakeMetadataEntry<'_>> = metadata_owned
        .iter()
        .map(|(k, kind, v)| BakeMetadataEntry {
            key: k.as_str(),
            kind: *kind,
            value: v.as_slice(),
        })
        .collect();

    let layers_borrowed: Vec<BakeLayer<'_>> = owned_layers
        .iter()
        .map(
            |(in_dim, out_dim, activation, dtype, weights, biases)| BakeLayer {
                in_dim: *in_dim,
                out_dim: *out_dim,
                activation: *activation,
                dtype: *dtype,
                weights: weights.as_slice(),
                biases: biases.as_slice(),
            },
        )
        .collect();

    let req = BakeRequest {
        schema_hash: model.schema_hash(),
        flags: 0,
        scaler_mean: &scaler_mean,
        scaler_scale: &scaler_scale,
        layers: &layers_borrowed,
        feature_bounds: &feature_bounds,
        metadata: &metadata_borrowed,
        output_specs: &output_specs,
        discrete_sets: &discrete_sets,
        sparse_overrides: &sparse_overrides,
        feature_order: None,
        output_order: None,
        compressed: compress,
        hu_permutations: None,
    };

    let new_bytes = if optimize {
        match bake_optimized(&req) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("bake_optimized: {:?}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        match bake(&req) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("bake: {:?}", e);
                return ExitCode::from(1);
            }
        }
    };

    if let Err(e) = std::fs::write(&out_path, &new_bytes) {
        eprintln!("write {}: {}", out_path.display(), e);
        return ExitCode::from(1);
    }
    eprintln!(
        "wrote {}: {} bytes ({:.1}% of input)",
        out_path.display(),
        new_bytes.len(),
        new_bytes.len() as f64 / input_bytes.len() as f64 * 100.0
    );

    // Verify round-trip: load the new bake, run a quick predict on
    // a uniform-0.5 input, compare to the original model's predict.
    let new_model = match Model::from_bytes(&new_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("re-parse: {:?}", e);
            return ExitCode::from(1);
        }
    };
    let mut p_old = zenpredict::Predictor::new(&model);
    let mut p_new = zenpredict::Predictor::new(&new_model);
    let features = vec![0.5f32; n_inputs];
    let out_old = p_old.predict(&features).unwrap();
    let out_new = p_new.predict(&features).unwrap();
    let max_diff = out_old
        .iter()
        .zip(out_new.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "round-trip max|Δ| on uniform-0.5 input: {} (n_outputs={})",
        max_diff,
        out_old.len()
    );
    if max_diff > 1e-3 {
        eprintln!(
            "WARNING: round-trip differs by > 1e-3 — likely I8 quantization noise from re-baking"
        );
    }
    ExitCode::SUCCESS
}

fn weight_magnitude_stats(layer: &zenpredict::LayerView<'_>) -> Value {
    let weights: Vec<f32> = match &layer.weights {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|b| f16_bits_to_f32(*b)).collect(),
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

fn metadata_to_json(md: &zenpredict::Metadata<'_>) -> Value {
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
                _ => "unknown",
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
