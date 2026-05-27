//! `zenpicker-train` — per-codec quality picker trainer (skeleton).
//!
//! ```text
//! zenpicker-train --input <parquet> --codec <family> \
//!     --target-column <name> --out <bake.bin> [options]
//! ```
//!
//! Loads a unified sweep parquet, filters to one codec, builds
//! `(feat_* [+ q] → target-quality)` rows, trains a ridge linear
//! baseline, emits a ZNPR v3 bake (via the zenpredict-bake JSON
//! pipeline) plus a sibling `<out>.toml` reproduce-this manifest, and
//! prints the held-out zenstats panel.
//!
//! This is the bounded first chunk of spec §4. The full
//! hyperparameter search, CubeCL acceleration, and cross-codec
//! MetaPicker auto-regeneration are documented follow-ons (see the
//! crate README).
//!
//! Exit codes: 0 success, 1 usage error, 2 training/IO error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Deserialize;

use zenpicker_train::{
    CodecFilter, TrainError, bake_picker, evaluate, load_training_rows, train_ridge,
};

const USAGE: &str = "\
zenpicker-train — per-codec quality picker trainer (skeleton, spec §4)

USAGE:
    zenpicker-train --input <parquet> --target-column <name> --out <bake.bin> [OPTIONS]

OPTIONS:
    --input <PATH>          Unified sweep parquet (image_path, codec, q,
                            knob_tuple_json, score_*, feat_0..feat_N).
    --codec <FAMILY>        Codec-name substring to filter rows by (e.g.
                            zenjpeg). Omit when the parquet is a single-codec cut.
    --target-column <NAME>  Supervised target column (zenstats nomenclature —
                            e.g. score_zensim, score_ssim2).
    --out <PATH>            Output ZNPR v3 bake path. A sibling
                            <PATH>.toml manifest is written alongside.
    --manifest <PATH>       Optional TOML recipe supplying defaults for the
                            above flags (reproduce-this input pattern). CLI
                            flags override manifest values.
    --lambda <F>            Ridge L2 penalty (default 1.0).
    --val-frac <F>          Held-out fraction of IMAGES, grouped split
                            (default 0.2).
    --include-q             Append the integer `q` column as a feature.
    -h, --help              Show this help.
";

/// TOML manifest-input recipe (reproduce-this). Every field optional;
/// CLI flags override.
#[derive(Debug, Default, Deserialize)]
struct RecipeToml {
    input: Option<String>,
    codec: Option<String>,
    target_column: Option<String>,
    out: Option<String>,
    lambda: Option<f64>,
    val_frac: Option<f64>,
    include_q: Option<bool>,
}

struct Args {
    input: Option<String>,
    codec: Option<String>,
    target_column: Option<String>,
    out: Option<String>,
    manifest: Option<String>,
    lambda: Option<f64>,
    val_frac: Option<f64>,
    include_q: Option<bool>,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args {
        input: None,
        codec: None,
        target_column: None,
        out: None,
        manifest: None,
        lambda: None,
        val_frac: None,
        include_q: None,
    };
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--input" => a.input = Some(next_val(&mut it, "--input")?),
            "--codec" => a.codec = Some(next_val(&mut it, "--codec")?),
            "--target-column" => a.target_column = Some(next_val(&mut it, "--target-column")?),
            "--out" => a.out = Some(next_val(&mut it, "--out")?),
            "--manifest" => a.manifest = Some(next_val(&mut it, "--manifest")?),
            "--lambda" => {
                a.lambda = Some(
                    next_val(&mut it, "--lambda")?
                        .parse()
                        .map_err(|e| format!("--lambda: {e}"))?,
                )
            }
            "--val-frac" => {
                a.val_frac = Some(
                    next_val(&mut it, "--val-frac")?
                        .parse()
                        .map_err(|e| format!("--val-frac: {e}"))?,
                )
            }
            "--include-q" => a.include_q = Some(true),
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Some(a))
}

fn next_val(it: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    it.next()
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn run(argv: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let parsed = parse_args(argv).map_err(|e| -> Box<dyn std::error::Error> {
        eprintln!("{USAGE}");
        e.into()
    })?;
    let Some(args) = parsed else {
        print!("{USAGE}");
        return Ok(());
    };

    // Merge the optional TOML recipe: manifest provides defaults, CLI
    // flags override.
    let recipe: RecipeToml = if let Some(mpath) = &args.manifest {
        let txt =
            std::fs::read_to_string(mpath).map_err(|e| format!("read manifest {mpath}: {e}"))?;
        toml::from_str(&txt).map_err(|e| format!("parse manifest {mpath}: {e}"))?
    } else {
        RecipeToml::default()
    };

    let input = args
        .input
        .or(recipe.input)
        .ok_or("--input (or manifest `input`) is required")?;
    let target_column = args
        .target_column
        .or(recipe.target_column)
        .ok_or("--target-column (or manifest `target_column`) is required")?;
    let out = args
        .out
        .or(recipe.out)
        .ok_or("--out (or manifest `out`) is required")?;
    let codec = args.codec.or(recipe.codec);
    let lambda = args.lambda.or(recipe.lambda).unwrap_or(1.0);
    let val_frac = args.val_frac.or(recipe.val_frac).unwrap_or(0.2);
    let include_q = args.include_q.or(recipe.include_q).unwrap_or(false);

    let input_path = Path::new(&input);
    let out_path = PathBuf::from(&out);

    eprintln!(
        "[zenpicker-train] input={input} codec={} target={target_column} lambda={lambda} val_frac={val_frac} include_q={include_q}",
        codec.as_deref().unwrap_or("<all>")
    );

    let filter = CodecFilter::new(codec.clone());
    let data = load_training_rows(input_path, &filter, &target_column, include_q)?;
    let codec_family = codec.clone().unwrap_or_else(|| "unknown".to_string());

    let total = data.n_rows();
    eprintln!(
        "[zenpicker-train] loaded {total} rows × {} features ({} distinct columns)",
        data.n_features,
        data.feature_names.len()
    );

    let (train_rows, val_rows) = zenpicker_train::grouped_split(&data, val_frac);
    eprintln!(
        "[zenpicker-train] grouped split: {} train rows / {} val rows (>= {:.0}% held out by image)",
        train_rows.len(),
        val_rows.len(),
        val_frac * 100.0
    );
    if train_rows.is_empty() {
        return Err(Box::new(TrainError::Degenerate(
            "train split is empty — lower --val-frac or supply more images".into(),
        )));
    }

    let model = train_ridge(
        &data.features,
        &data.targets,
        data.n_features,
        &train_rows,
        lambda,
    )?;
    eprintln!(
        "[zenpicker-train] trained ridge baseline: {} weights, intercept={:.4}, lambda={}",
        model.n_features(),
        model.intercept,
        model.lambda
    );

    // Held-out panel.
    let report = evaluate(&model, &data, &val_rows);
    let (srocc, plcc, krocc, n) = match report {
        Some(r) => (r.panel.srocc, r.panel.plcc, r.panel.krocc, r.panel.n),
        None => (f64::NAN, f64::NAN, f64::NAN, 0),
    };
    if let Some(r) = report {
        let p = r.panel;
        eprintln!("[zenpicker-train] HELD-OUT PANEL (zenstats / Mohammadi 2025):");
        eprintln!("    n      = {}", p.n);
        eprintln!("    SROCC  = {:.4}", p.srocc);
        eprintln!("    PLCC   = {:.4}", p.plcc);
        eprintln!("    KROCC  = {:.4}", p.krocc);
        eprintln!("    OR     = {:.4}", p.or_ratio);
        eprintln!("    PWRC   = {:.4}", p.pwrc);
        eprintln!("    Z-RMSE = {:.4}", p.z_rmse);
    } else {
        eprintln!("[zenpicker-train] WARNING: no held-out rows — panel skipped");
    }

    let input_sha = zenpicker_train::file_sha256(input_path)?;
    let outcome = bake_picker(
        &model,
        &data.feature_names,
        &codec_family,
        &target_column,
        &out_path,
        zenpicker_train::PickerManifestInputs {
            input_parquet: &input,
            input_sha256: &input_sha,
            input_rows_total: total,
            train_rows: train_rows.len(),
            val_rows: val_rows.len(),
            heldout_srocc: srocc,
            heldout_plcc: plcc,
            heldout_krocc: krocc,
            heldout_n: n,
        },
    )?;

    eprintln!(
        "[zenpicker-train] wrote ZNPR v3 bake: {} ({} bytes)",
        outcome.bake_path,
        outcome.bake_bytes.len()
    );
    eprintln!(
        "[zenpicker-train] wrote manifest: {}",
        outcome.manifest_path
    );
    eprintln!("[zenpicker-train] done.");
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[zenpicker-train] error: {e}");
            ExitCode::from(2)
        }
    }
}
