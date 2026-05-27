//! `zenpicker-train` — per-codec quality picker trainer.
//!
//! ```text
//! zenpicker-train --input <parquet> [--codec <family>] --out <bake.bin> [options]
//! ```
//!
//! Default mode (`--mode mlp`) ports zentrain's within-cell-optimal
//! picker formulation to Rust: factor each codec config into a
//! categorical cell, build per-`(image, target_zq)` `bytes_log[cell]`
//! targets, train a LeakyReLU MLP `(image_features, zq_norm) ->
//! bytes_log[0..N]` with a bounded hyperparameter search, emit a ZNPR
//! v3 bake (via the zenpredict-bake JSON pipeline) + a sibling
//! `<out>.toml` manifest, and print the HONEST held-out picker panel
//! (no q-leakage).
//!
//! `--mode ridge` keeps the legacy single-layer linear baseline
//! (predicts `feat_* [+q] -> --target-column`) for the cheap reference
//! path. NOTE: `ridge --include-q` IS q-leakage by construction and is
//! only for the legacy baseline comparison — the MLP mode never sees q.
//!
//! Exit codes: 0 success, 1 usage error, 2 training/IO error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Deserialize;

use zenpicker_train::{
    CodecFilter, GridPoint, MlpConfig, MlpPickerManifestInputs, SearchCandidate, SearchManifest,
    TrainError, bake_mlp_picker, bake_picker, build_picker_dataset, default_grid,
    default_zq_targets, evaluate, fit_standardizer, grouped_split_picker, load_training_rows,
    run_search, standardize_all, train_ridge,
};

const USAGE: &str = "\
zenpicker-train — per-codec quality picker trainer

USAGE:
    zenpicker-train --input <parquet> --out <bake.bin> [OPTIONS]

OPTIONS:
    --input <PATH>          Unified sweep parquet (image_*, codec, q,
                            knob_tuple_json, encoded_bytes, score_zensim, feat_*).
    --codec <FAMILY>        Codec-name substring filter (e.g. zenjpeg). Omit when
                            the parquet is a single-codec cut.
    --out <PATH>            Output ZNPR v3 bake path. A sibling <PATH>.toml
                            manifest is written alongside.
    --mode <mlp|ridge>      mlp (default): within-cell-optimal MLP picker,
                            image features + zq_norm only (NO q-leakage).
                            ridge: legacy single-layer linear baseline.
    --val-frac <F>          Held-out fraction of IMAGES, grouped split (default 0.2).
    --hidden <CSV>          (mlp) Override hidden widths, e.g. 128,128. When set,
                            disables the search and trains exactly this topology.
    --seed <N>              (mlp) Seed override (default 0). With --hidden, a single
                            deterministic fit.

  ridge-mode only:
    --target-column <NAME>  Supervised target column (e.g. score_zensim).
    --lambda <F>            Ridge L2 penalty (default 1.0).
    --include-q             Append integer q as a feature (LEAKY — baseline only).

    --manifest <PATH>       Optional TOML recipe supplying defaults. CLI overrides.
    -h, --help              Show this help.
";

#[derive(Debug, Default, Deserialize)]
struct RecipeToml {
    input: Option<String>,
    codec: Option<String>,
    target_column: Option<String>,
    out: Option<String>,
    mode: Option<String>,
    lambda: Option<f64>,
    val_frac: Option<f64>,
    include_q: Option<bool>,
    hidden: Option<String>,
    seed: Option<u64>,
}

struct Args {
    input: Option<String>,
    codec: Option<String>,
    target_column: Option<String>,
    out: Option<String>,
    mode: Option<String>,
    manifest: Option<String>,
    lambda: Option<f64>,
    val_frac: Option<f64>,
    include_q: Option<bool>,
    hidden: Option<String>,
    seed: Option<u64>,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args {
        input: None,
        codec: None,
        target_column: None,
        out: None,
        mode: None,
        manifest: None,
        lambda: None,
        val_frac: None,
        include_q: None,
        hidden: None,
        seed: None,
    };
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--input" => a.input = Some(next_val(&mut it, "--input")?),
            "--codec" => a.codec = Some(next_val(&mut it, "--codec")?),
            "--target-column" => a.target_column = Some(next_val(&mut it, "--target-column")?),
            "--out" => a.out = Some(next_val(&mut it, "--out")?),
            "--mode" => a.mode = Some(next_val(&mut it, "--mode")?),
            "--manifest" => a.manifest = Some(next_val(&mut it, "--manifest")?),
            "--hidden" => a.hidden = Some(next_val(&mut it, "--hidden")?),
            "--seed" => {
                a.seed = Some(
                    next_val(&mut it, "--seed")?
                        .parse()
                        .map_err(|e| format!("--seed: {e}"))?,
                )
            }
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

fn parse_hidden(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<usize>()
                .map_err(|e| format!("--hidden: {e}"))
        })
        .collect()
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
    let out = args
        .out
        .or(recipe.out)
        .ok_or("--out (or manifest `out`) is required")?;
    let codec = args.codec.or(recipe.codec);
    let mode = args
        .mode
        .or(recipe.mode)
        .unwrap_or_else(|| "mlp".to_string());
    let val_frac = args.val_frac.or(recipe.val_frac).unwrap_or(0.2);

    let input_path = Path::new(&input);
    let out_path = PathBuf::from(&out);

    match mode.as_str() {
        "mlp" => run_mlp(
            &input,
            input_path,
            &out_path,
            codec,
            val_frac,
            args.hidden.or(recipe.hidden),
            args.seed.or(recipe.seed),
        ),
        "ridge" => run_ridge(
            &input,
            input_path,
            &out_path,
            codec,
            val_frac,
            args.target_column.or(recipe.target_column),
            args.lambda.or(recipe.lambda).unwrap_or(1.0),
            args.include_q.or(recipe.include_q).unwrap_or(false),
        ),
        other => Err(format!("--mode must be 'mlp' or 'ridge', got {other:?}").into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_mlp(
    input: &str,
    input_path: &Path,
    out_path: &Path,
    codec: Option<String>,
    val_frac: f64,
    hidden_override: Option<String>,
    seed_override: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let codec_family = codec.clone().unwrap_or_else(|| "unknown".to_string());
    eprintln!(
        "[zenpicker-train] mode=mlp input={input} codec={} val_frac={val_frac}",
        codec.as_deref().unwrap_or("<all>")
    );
    eprintln!(
        "[zenpicker-train] formulation: within-cell-optimal bytes_log argmin (port of zentrain train_hybrid)"
    );
    eprintln!("[zenpicker-train] inputs = image feat_* + zq_norm; q is NOT an input (no leakage)");

    let zq_targets = default_zq_targets();
    let ds = build_picker_dataset(input_path, codec.as_deref(), &zq_targets)?;
    eprintln!(
        "[zenpicker-train] built picker dataset: {} (image,target_zq) rows | {} image features (+zq_norm = {} inputs) | {} categorical cells",
        ds.n_rows(),
        ds.feature_names.len(),
        ds.n_in,
        ds.n_cells
    );
    eprintln!("[zenpicker-train] cells: {}", ds.cell_labels.join("  ::  "));

    let (train_rows, val_rows) = grouped_split_picker(&ds, val_frac);
    eprintln!(
        "[zenpicker-train] grouped-by-image split: {} train rows / {} val rows ({:.0}% images held out)",
        train_rows.len(),
        val_rows.len(),
        val_frac * 100.0
    );
    if train_rows.is_empty() || val_rows.is_empty() {
        return Err(Box::new(TrainError::Degenerate(
            "empty train or val split — adjust --val-frac or supply more images".into(),
        )));
    }

    // Standardizer fit on TRAIN rows only, applied to the full matrix.
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train_rows);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);

    let base = MlpConfig::default();

    // Either a single explicit topology or the bounded grid search.
    let (model, cfg, eval, search_manifest) = if let Some(h) = hidden_override {
        let hidden = parse_hidden(&h)?;
        let grid = vec![GridPoint {
            hidden: hidden.clone(),
            lr: base.lr,
            seed: seed_override.unwrap_or(0),
        }];
        eprintln!("[zenpicker-train] single fit: hidden={hidden:?} (search disabled)");
        let res = run_search(&ds, &x_std, &train_rows, &val_rows, &grid, &base, |m| {
            eprintln!("{m}")
        })
        .ok_or("MLP fit produced no evaluable model")?;
        let sm = build_search_manifest(&res, "single_fit");
        (res.best_model, res.best_cfg, res.best_eval, sm)
    } else {
        let grid = default_grid();
        eprintln!(
            "[zenpicker-train] bounded grid search: {} candidates (hidden × lr × seed), ranked by held-out argmin accuracy",
            grid.len()
        );
        let res = run_search(&ds, &x_std, &train_rows, &val_rows, &grid, &base, |m| {
            eprintln!("{m}")
        })
        .ok_or("grid search produced no evaluable model")?;
        eprintln!(
            "[zenpicker-train] selected candidate #{}: hidden={:?} lr={} seed={}",
            res.selected_index, res.best_cfg.hidden, res.best_cfg.lr, res.best_cfg.seed
        );
        let sm = build_search_manifest(&res, "bounded_grid");
        (res.best_model, res.best_cfg, res.best_eval, sm)
    };

    eprintln!("[zenpicker-train] HELD-OUT PICKER PANEL (honest — no q in inputs):");
    eprintln!("    rows scored   = {}", eval.n_rows);
    eprintln!("    cell pairs    = {}", eval.n_pairs);
    eprintln!("    bytes_log SROCC = {:.4}", eval.bytes_panel.srocc);
    eprintln!("    bytes_log PLCC  = {:.4}", eval.bytes_panel.plcc);
    eprintln!("    bytes_log KROCC = {:.4}", eval.bytes_panel.krocc);
    eprintln!("    bytes_log PWRC  = {:.4}", eval.bytes_panel.pwrc);
    eprintln!("    bytes_log Z-RMSE= {:.4}", eval.bytes_panel.z_rmse);
    eprintln!("    bytes_log OR    = {:.4}", eval.bytes_panel.or_ratio);
    eprintln!("    argmin accuracy = {:.4}", eval.argmin_acc);
    eprintln!(
        "    byte overhead   = mean {:.3} | p50 {:.3} | p90 {:.3}",
        eval.overhead_mean, eval.overhead_p50, eval.overhead_p90
    );

    let input_sha = zenpicker_train::file_sha256(input_path)?;
    let heldout = zenpicker_train::HeldoutManifest {
        bytes_srocc: eval.bytes_panel.srocc,
        bytes_plcc: eval.bytes_panel.plcc,
        bytes_krocc: eval.bytes_panel.krocc,
        bytes_pwrc: eval.bytes_panel.pwrc,
        bytes_z_rmse: eval.bytes_panel.z_rmse,
        bytes_or_ratio: eval.bytes_panel.or_ratio,
        argmin_acc: eval.argmin_acc,
        overhead_mean: eval.overhead_mean,
        overhead_p50: eval.overhead_p50,
        overhead_p90: eval.overhead_p90,
        n_rows: eval.n_rows,
        n_pairs: eval.n_pairs,
    };
    let outcome = bake_mlp_picker(
        &model,
        &mean,
        &scale,
        out_path,
        MlpPickerManifestInputs {
            codec_family: &codec_family,
            input_parquet: input,
            input_sha256: &input_sha,
            // Raw sweep row count is not retained past dataset build;
            // picker_rows_total (the (image,target_zq) rows) is the
            // meaningful training-set size for this formulation.
            input_rows_total: 0,
            picker_rows_total: ds.n_rows(),
            train_rows: train_rows.len(),
            val_rows: val_rows.len(),
            n_image_features: ds.feature_names.len(),
            feature_names: &ds.feature_names,
            cell_labels: &ds.cell_labels,
            zq_targets: &ds.zq_targets,
            cfg: &cfg,
            search: search_manifest,
            heldout,
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

fn build_search_manifest(res: &zenpicker_train::SearchResult, kind: &str) -> SearchManifest {
    let candidates: Vec<SearchCandidate> = res
        .trail
        .iter()
        .map(|(gp, srocc, argmin)| SearchCandidate {
            hidden: gp.hidden.clone(),
            lr: gp.lr,
            seed: gp.seed,
            heldout_bytes_srocc: *srocc,
            argmin_acc: *argmin,
        })
        .collect();
    SearchManifest {
        kind: kind.to_string(),
        candidates,
        selected_index: res.selected_index,
        selection_metric: "heldout_argmin_accuracy".to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ridge(
    input: &str,
    input_path: &Path,
    out_path: &Path,
    codec: Option<String>,
    val_frac: f64,
    target_column: Option<String>,
    lambda: f64,
    include_q: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let target_column = target_column
        .ok_or("--target-column (or manifest `target_column`) is required for ridge mode")?;
    if include_q {
        eprintln!(
            "[zenpicker-train] WARNING: ridge --include-q appends q as a feature; this is \
             q-leakage by construction and exists ONLY for the legacy baseline comparison."
        );
    }
    eprintln!(
        "[zenpicker-train] mode=ridge input={input} codec={} target={target_column} lambda={lambda} val_frac={val_frac} include_q={include_q}",
        codec.as_deref().unwrap_or("<all>")
    );

    let filter = CodecFilter::new(codec.clone());
    let data = load_training_rows(input_path, &filter, &target_column, include_q)?;
    let codec_family = codec.clone().unwrap_or_else(|| "unknown".to_string());
    let total = data.n_rows();
    eprintln!(
        "[zenpicker-train] loaded {total} rows × {} features",
        data.n_features
    );

    let (train_rows, val_rows) = zenpicker_train::grouped_split(&data, val_frac);
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
    let report = evaluate(&model, &data, &val_rows);
    let (srocc, plcc, krocc, n) = match report {
        Some(r) => (r.panel.srocc, r.panel.plcc, r.panel.krocc, r.panel.n),
        None => (f64::NAN, f64::NAN, f64::NAN, 0),
    };
    if let Some(r) = report {
        let p = r.panel;
        eprintln!("[zenpicker-train] HELD-OUT PANEL (ridge baseline):");
        eprintln!(
            "    n={} SROCC={:.4} PLCC={:.4} KROCC={:.4}",
            p.n, p.srocc, p.plcc, p.krocc
        );
    }

    let input_sha = zenpicker_train::file_sha256(input_path)?;
    let outcome = bake_picker(
        &model,
        &data.feature_names,
        &codec_family,
        &target_column,
        out_path,
        zenpicker_train::PickerManifestInputs {
            input_parquet: input,
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
        "[zenpicker-train] wrote ZNPR v3 bake: {} ({} bytes) + manifest {}",
        outcome.bake_path,
        outcome.bake_bytes.len(),
        outcome.manifest_path
    );
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
