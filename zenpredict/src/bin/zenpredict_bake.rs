//! `zenpredict-bake` — CLI that consumes a `BakeRequestJson`
//! description on disk and writes a ZNPR v2 `.bin`.
//!
//! Used by the Python training pipeline at
//! `zenanalyze/zenpicker/tools/` so the byte-level format stays
//! defined exclusively in Rust. Python emits the JSON, this binary
//! produces the bin.
//!
//! ```text
//! zenpredict-bake <input.json> <output.bin>
//! ```
//!
//! Exit codes:
//!   0  success
//!   1  IO error reading input or writing output
//!   2  JSON parse error
//!   3  bake validation failure (BakeJsonError / BakeError)
//!
//! On success, prints a one-line summary of `(n_inputs, n_outputs,
//! n_layers, schema_hash, n_metadata_entries, output bytes)` to
//! stderr.

use std::path::PathBuf;
use std::process::ExitCode;

use zenpredict::bake::{BakeJsonError, BakeRequestJson, bake_from_json};

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let input = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: zenpredict-bake <input.json> <output.bin>");
            return ExitCode::from(1);
        }
    };
    let output = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: zenpredict-bake <input.json> <output.bin>");
            return ExitCode::from(1);
        }
    };
    if args.next().is_some() {
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
