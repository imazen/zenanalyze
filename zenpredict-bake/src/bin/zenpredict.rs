//! `zenpredict` — unified CLI entry point for ZNPR v3 bake tooling.
//!
//! Three subcommands wrap the existing single-purpose binaries:
//!
//! ```text
//! zenpredict bake    <input.json> <output.bin>
//! zenpredict inspect <bake.bin>            [--weights]
//! zenpredict repack  <in.bin> <out.bin> [--dtype f32|f16|i8]
//!                                       [--zerobias <tau>]
//!                                       [--compress]
//!                                       [--optimize]
//! ```
//!
//! `bake` and `inspect` delegate to the same shared functions used
//! by the legacy `zenpredict-bake` and `zenpredict-inspect` binaries
//! (preserved unchanged for backwards compatibility). `repack` is
//! a new front-end for the logic previously only exposed as the
//! `rebake_v3_1` example — i8/f16 quantization, zero-bias
//! preprocessing, LZ4 whole-bake compression, and the Hu-reorder
//! pipeline via `bake_optimized`.
//!
//! Exit codes match the underlying subcommand bodies; see
//! [`zenpredict_bake::cli`] module docs.

use std::process::ExitCode;

use zenpredict_bake::cli::{run_bake_cli, run_inspect_cli, run_repack_cli};

const HELP: &str = "\
zenpredict <subcommand> [args...]

Subcommands:
  bake    Convert a BakeRequestJson to a ZNPR v3 .bin
  inspect Show structure + metadata of a ZNPR v3 bake
  repack  Re-quantize / compress / optimize an existing bake

Run `zenpredict <subcommand> --help` for subcommand-specific help.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }

    let sub = args[0].as_str();
    let rest = &args[1..];

    match sub {
        "bake" => {
            if rest.iter().any(|a| a == "--help" || a == "-h") {
                println!("usage: zenpredict bake <input.json> <output.bin>");
                return ExitCode::SUCCESS;
            }
            run_bake_cli(rest)
        }
        "inspect" => {
            if rest.iter().any(|a| a == "--help" || a == "-h") {
                println!("usage: zenpredict inspect <model.bin> [--weights]");
                return ExitCode::SUCCESS;
            }
            run_inspect_cli(rest)
        }
        "repack" => {
            if rest.iter().any(|a| a == "--help" || a == "-h") {
                println!(
                    "usage: zenpredict repack <input.bin> <output.bin> [--dtype f32|f16|i8] [--zerobias <tau>] [--compress] [--optimize]"
                );
                return ExitCode::SUCCESS;
            }
            run_repack_cli(rest)
        }
        other => {
            eprintln!("zenpredict: unknown subcommand '{other}'");
            eprintln!();
            eprintln!("{HELP}");
            ExitCode::from(1)
        }
    }
}
