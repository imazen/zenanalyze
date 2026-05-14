//! Re-bake an existing ZNPR v3 bake through the new (v3.1) bake()
//! to pick up HU reorder (always-on) + whole-bake LZ4 compression.
//!
//! Usage:
//!   cargo run --release -p zenpredict-bake --example rebake_v3_1 -- \
//!     <input.bin> <output.bin> [--compress]
//!
//! Loads the input, dequantizes every layer's weights to f32,
//! reconstructs a BakeRequest with the same metadata, and writes
//! out through `bake()`. The HU reorder applies automatically.
//! Pass `--compress` to wrap the payload in LZ4.

use std::env;
use std::fs;
use std::path::PathBuf;

use zenpredict::{Activation, Model, WeightDtype, WeightStorage, f16_bits_to_f32};
use zenpredict_bake::{
    BakeLayer, BakeMetadataEntry, BakeRequest, apply_zero_bias_per_layer_in_place, bake,
    bake_optimized,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: rebake_v3_1 <input.bin> <output.bin> [--compress] [--zerobias <tau>]"
        );
        std::process::exit(1);
    }
    let in_path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);
    let compress = args.iter().any(|a| a == "--compress");
    let zerobias_tau: Option<f32> = args
        .windows(2)
        .find(|w| w[0] == "--zerobias")
        .map(|w| w[1].parse().expect("zerobias tau must be a float"));
    let force_dtype: Option<WeightDtype> = args
        .windows(2)
        .find(|w| w[0] == "--dtype")
        .map(|w| match w[1].as_str() {
            "f32" => WeightDtype::F32,
            "f16" => WeightDtype::F16,
            "i8" => WeightDtype::I8,
            other => panic!("unknown dtype: {}", other),
        });
    let optimize = args.iter().any(|a| a == "--optimize");

    let input_bytes = fs::read(&in_path).unwrap_or_else(|e| {
        eprintln!("read {}: {}", in_path.display(), e);
        std::process::exit(1);
    });

    let model = Model::from_bytes(&input_bytes).unwrap_or_else(|e| {
        eprintln!("parse: {:?}", e);
        std::process::exit(1);
    });

    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    let n_layers = model.n_layers();
    eprintln!(
        "loaded: {} inputs, {} outputs, {} layers, {} bytes",
        n_inputs, n_outputs, n_layers, input_bytes.len()
    );

    let scaler_mean: Vec<f32> = model.scaler_mean().to_vec();
    let scaler_scale: Vec<f32> = model.scaler_scale().to_vec();
    let feature_bounds: Vec<zenpredict::FeatureBound> = model.feature_bounds().to_vec();
    let output_specs: Vec<zenpredict::OutputSpec> = model.output_specs().to_vec();
    let discrete_sets: Vec<f32> = model.discrete_sets().to_vec();
    let sparse_overrides: Vec<zenpredict::SparseOverride> = model.sparse_overrides().to_vec();

    // Materialize each layer's weights as f32 + biases as f32, preserving
    // the original dtype choice.
    let mut owned_layers: Vec<(usize, usize, Activation, WeightDtype, Vec<f32>, Vec<f32>)> =
        Vec::with_capacity(n_layers);
    for layer in model.layers() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;
        let activation = layer.activation;
        let (dtype, weights_f32) = match &layer.weights {
            WeightStorage::F32(w) => (WeightDtype::F32, w.to_vec()),
            WeightStorage::F16(w) => {
                (WeightDtype::F16, w.iter().map(|b| f16_bits_to_f32(*b)).collect())
            }
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
        owned_layers.push((in_dim, out_dim, activation, final_dtype, weights_f32, layer.biases.to_vec()));
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
    let metadata_owned: Vec<(String, zenpredict::MetadataType, Vec<u8>)> = md
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
        .map(|(in_dim, out_dim, activation, dtype, weights, biases)| BakeLayer {
            in_dim: *in_dim,
            out_dim: *out_dim,
            activation: *activation,
            dtype: *dtype,
            weights: weights.as_slice(),
            biases: biases.as_slice(),
        })
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
    };

    let new_bytes = if optimize {
        bake_optimized(&req).unwrap_or_else(|e| {
            eprintln!("bake_optimized: {:?}", e);
            std::process::exit(1);
        })
    } else {
        bake(&req).unwrap_or_else(|e| {
            eprintln!("bake: {:?}", e);
            std::process::exit(1);
        })
    };

    fs::write(&out_path, &new_bytes).unwrap_or_else(|e| {
        eprintln!("write {}: {}", out_path.display(), e);
        std::process::exit(1);
    });
    eprintln!(
        "wrote {}: {} bytes ({:.1}% of input)",
        out_path.display(),
        new_bytes.len(),
        new_bytes.len() as f64 / input_bytes.len() as f64 * 100.0
    );

    // Verify round-trip: load the new bake, run a quick predict on
    // zeros, compare to the original model's predict on zeros.
    let new_model = Model::from_bytes(&new_bytes).unwrap_or_else(|e| {
        eprintln!("re-parse: {:?}", e);
        std::process::exit(1);
    });
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
        max_diff, out_old.len()
    );
    if max_diff > 1e-3 {
        eprintln!("WARNING: round-trip differs by > 1e-3 — likely I8 quantization noise from re-baking");
    }
}
