//! `zenpredict-inspect` — load a ZNPR v3 `.bin` and dump everything
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
//!
//! The body of this binary delegates to
//! [`zenpredict_bake::cli::run_inspect_cli`] so the unified
//! `zenpredict` binary (subcommand `zenpredict inspect ...`) shares
//! the same code path and behaviour. Argument order, exit codes, and
//! stdout/stderr output are identical.

use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    zenpredict_bake::cli::run_inspect_cli(&argv)
}
