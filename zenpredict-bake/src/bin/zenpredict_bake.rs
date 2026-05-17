//! `zenpredict-bake` — CLI that consumes a `BakeRequestJson`
//! description on disk and writes a ZNPR v3 `.bin`.
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
//!
//! The body of this binary delegates to
//! [`zenpredict_bake::cli::run_bake_cli`] so the unified `zenpredict`
//! binary (subcommand `zenpredict bake ...`) shares the same code
//! path and behaviour. Argument order, exit codes, and stderr output
//! are identical.

use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    zenpredict_bake::cli::run_bake_cli(&argv)
}
