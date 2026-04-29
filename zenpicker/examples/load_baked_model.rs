//! Loads a `.bin` model produced by `tools/bake_picker.py`, runs
//! inference against a deterministic input, and writes the output
//! row to stdout. The companion `tools/bake_roundtrip_check.py`
//! script generates the same input through scikit-learn and compares
//! — round-trip correctness check for the binary format.
//!
//! Usage:
//!   cargo run --release --example load_baked_model -- <model.bin>
//!
//! Reads `bin_path` from argv[1]. Prints each output value as a
//! whitespace-separated f32 on stdout.

use std::env;
use std::fs;
use std::process::ExitCode;

use bytemuck;

#[repr(C, align(8))]
struct AlignedBytes<const N: usize>([u8; N]);

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: load_baked_model <model.bin>");
        return ExitCode::from(2);
    }
    let bytes = match fs::read(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("read {}: {e}", args[1]);
            return ExitCode::from(1);
        }
    };

    // Re-align into a u64-backed buffer so the loader can zero-copy
    // borrow the f32 sections.
    let n_u64 = bytes.len().div_ceil(8);
    let mut storage: Vec<u64> = vec![0; n_u64];
    let view: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
    view[..bytes.len()].copy_from_slice(&bytes);
    let aligned: &[u8] = &bytemuck::cast_slice::<u64, u8>(&storage)[..bytes.len()];

    let model = match zenpicker::Model::from_bytes(aligned) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse: {e}");
            return ExitCode::from(1);
        }
    };
    eprintln!(
        "loaded: n_inputs={} n_outputs={} schema_hash=0x{:016x}",
        model.n_inputs(),
        model.n_outputs(),
        model.schema_hash()
    );

    // Deterministic input: feature_i = sin(i * 0.1) so the round-trip
    // check has something content-rich to compare against.
    let n_in = model.n_inputs();
    let features: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.1).sin()).collect();

    let mut picker = zenpicker::Picker::new(model);
    let out = match picker.predict(&features) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("predict: {e}");
            return ExitCode::from(1);
        }
    };

    // One value per line, full f32 precision via debug formatter.
    for v in out {
        println!("{v}");
    }
    ExitCode::SUCCESS
}
