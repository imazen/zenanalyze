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
    CodecFilter, DistillManifest, GridPoint, MlpConfig, MlpPickerManifestInputs, ScalarAxisSpec,
    ScalarHeadSpec, SearchCandidate, SearchManifest, ShapingMode, TeacherParams, TrainError,
    apply_inplace, bake_mlp_picker, bake_picker, build_picker_dataset, build_picker_dataset_with,
    default_grid, default_zq_targets, evaluate, evaluate_picker_bake, export_teacher_dataset,
    fit_standardizer, fit_transforms, grouped_split_picker, load_soft_targets, load_training_rows,
    run_search, run_search_distill, standardize_all, teacher_params_fingerprint, train_ridge,
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
    --scalar-axes <CSV>     (mlp) Hybrid scalar prediction heads, e.g.
                            'chroma_scale,lambda'. Each adds a per-cell regression
                            head predicting the within-cell-optimal knob value in
                            natural units (bounds/snap baked as output_specs). The
                            knob values are read from knob_tuple_json. Omit for the
                            bytes-only categorical picker. (zenjpeg axes only today.)

  mlp distillation (zentrain teacher -> student recipe):
    --distill               Distill the MLP student against a per-cell HistGB
                            teacher's DENSE soft bytes_log targets (zentrain's
                            full recipe), instead of the direct hard-target fit.
                            Orchestrates: export dataset -> run teacher script ->
                            load soft targets -> distill. q stays OUT of inputs.
    --teacher-script <PATH> Python teacher (default: alongside this crate's
                            scripts/teacher_soft_targets.py). One-time OFFLINE
                            target-gen step; the Rust runtime gains no Python dep.
    --export-dataset <PATH> Just export the Rust-built dataset parquet (the
                            teacher's exact train rows + hard targets) and exit.
                            For running the teacher step manually.
    --soft-targets <PATH>   Distill against a pre-computed soft-target parquet
                            (skips the teacher-script shell-out; pairs with a
                            prior --export-dataset + manual teacher run).
    --python <BIN>          Python interpreter for --distill (default: python3).
    --soft-weight <F>       Distillation blend (default 1.0 = zentrain's pure soft-target
                            MSE). < 1.0 mixes in the hard within-cell-optimal target where
                            reachable: soft_weight*soft + (1-soft_weight)*hard. A principled
                            extension to probe when pure soft distillation doesn't close the gap.

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
    distill: Option<bool>,
    teacher_script: Option<String>,
    export_dataset: Option<String>,
    soft_targets: Option<String>,
    python: Option<String>,
    soft_weight: Option<f64>,
    input_shaping: Option<String>,
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
    distill: Option<bool>,
    teacher_script: Option<String>,
    export_dataset: Option<String>,
    soft_targets: Option<String>,
    python: Option<String>,
    soft_weight: Option<f64>,
    input_shaping: Option<String>,
    eval_bake: Option<String>,
    /// `--scalar-axes` CSV (e.g. "chroma_scale,lambda") → hybrid scalar
    /// prediction heads. Empty = bytes-only categorical picker.
    scalar_axes: Option<String>,
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
        distill: None,
        teacher_script: None,
        export_dataset: None,
        soft_targets: None,
        python: None,
        soft_weight: None,
        input_shaping: None,
        eval_bake: None,
        scalar_axes: None,
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
            "--distill" => a.distill = Some(true),
            "--teacher-script" => a.teacher_script = Some(next_val(&mut it, "--teacher-script")?),
            "--export-dataset" => a.export_dataset = Some(next_val(&mut it, "--export-dataset")?),
            "--soft-targets" => a.soft_targets = Some(next_val(&mut it, "--soft-targets")?),
            "--python" => a.python = Some(next_val(&mut it, "--python")?),
            "--soft-weight" => {
                a.soft_weight = Some(
                    next_val(&mut it, "--soft-weight")?
                        .parse()
                        .map_err(|e| format!("--soft-weight: {e}"))?,
                )
            }
            "--input-shaping" => a.input_shaping = Some(next_val(&mut it, "--input-shaping")?),
            "--scalar-axes" => a.scalar_axes = Some(next_val(&mut it, "--scalar-axes")?),
            "--eval-bake" => a.eval_bake = Some(next_val(&mut it, "--eval-bake")?),
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

    // --eval-bake: score an EXTERNAL ZNPR v3 bake on the held-out split
    // via the deployed runtime path (predict_transformed → argmin) and
    // exit. No training, no --out — used to measure quantized / bounded
    // / shaped bake variants against the identical held-out split.
    if let Some(bake_path) = args.eval_bake {
        let codec = args.codec.or(recipe.codec);
        let val_frac = args.val_frac.or(recipe.val_frac).unwrap_or(0.2);
        return run_eval_bake(Path::new(&input), codec, val_frac, &bake_path);
    }

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
            DistillOpts {
                distill: args.distill.or(recipe.distill).unwrap_or(false),
                teacher_script: args.teacher_script.or(recipe.teacher_script),
                export_dataset: args.export_dataset.or(recipe.export_dataset),
                soft_targets: args.soft_targets.or(recipe.soft_targets),
                python: args
                    .python
                    .or(recipe.python)
                    .unwrap_or_else(|| "python3".to_string()),
                soft_weight: args.soft_weight.or(recipe.soft_weight).unwrap_or(1.0),
            },
            args.input_shaping
                .or(recipe.input_shaping)
                .unwrap_or_else(|| "none".to_string()),
            args.scalar_axes,
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

/// Distillation orchestration options (zentrain teacher → student).
struct DistillOpts {
    distill: bool,
    teacher_script: Option<String>,
    export_dataset: Option<String>,
    soft_targets: Option<String>,
    python: String,
    /// Distillation blend (1.0 = pure soft-target MSE, zentrain's recipe).
    soft_weight: f64,
}

/// Default location of the bundled Python teacher script — sibling to
/// this crate's `Cargo.toml`. Resolved at build time via `CARGO_MANIFEST_DIR`.
fn default_teacher_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("teacher_soft_targets.py")
}

/// `--eval-bake`: load an external ZNPR v3 picker bake and score it on
/// the held-out split via the deployed runtime path. Measures any
/// variant (quantized / feature-bounded / output-spec'd / shaped)
/// against the identical val split the trainer used.
fn run_eval_bake(
    input_path: &Path,
    codec: Option<String>,
    val_frac: f64,
    bake_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let zq_targets = default_zq_targets();
    let ds = build_picker_dataset(input_path, codec.as_deref(), &zq_targets)?;
    let (_train_rows, val_rows) = grouped_split_picker(&ds, val_frac);
    let bytes =
        std::fs::read(bake_path).map_err(|e| TrainError::Io(format!("read {bake_path}: {e}")))?;
    eprintln!(
        "[zenpicker-train] eval-bake: {bake_path} ({} bytes) on {} held-out rows (val_frac={val_frac})",
        bytes.len(),
        val_rows.len()
    );
    match evaluate_picker_bake(&bytes, &ds, &val_rows).map_err(TrainError::Bake)? {
        Some(e) => {
            println!(
                "bake={bake_path}\nargmin_acc={:.4} overhead_mean={:.4} overhead_p50={:.4} \
                 overhead_p90={:.4} bytes_srocc={:.4} n_rows={} n_pairs={}",
                e.argmin_acc,
                e.overhead_mean,
                e.overhead_p50,
                e.overhead_p90,
                e.bytes_panel.srocc,
                e.n_rows,
                e.n_pairs
            );
        }
        None => eprintln!("[zenpicker-train] eval-bake: no scorable held-out rows"),
    }
    Ok(())
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
    distill_opts: DistillOpts,
    shaping_mode_str: String,
    scalar_axes_csv: Option<String>,
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

    // Scalar prediction heads (hybrid-heads). `--scalar-axes` parses into
    // per-axis natural-unit output specs + dataset axes; empty = the
    // bytes-only categorical picker. Only zenjpeg axes are defined today.
    let scalar_heads: Vec<ScalarHeadSpec> = match scalar_axes_csv {
        Some(csv) => csv
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|name| {
                ScalarHeadSpec::zenjpeg(name).ok_or_else(|| {
                    format!("--scalar-axes: unknown axis '{name}' (known: chroma_scale, lambda)")
                })
            })
            .collect::<Result<_, String>>()?,
        None => Vec::new(),
    };
    let dataset_axes: Vec<ScalarAxisSpec> = scalar_heads
        .iter()
        .map(|h| ScalarAxisSpec::new(h.name.clone(), h.sentinel.map(|s| s as f64)))
        .collect();
    if !scalar_heads.is_empty() {
        eprintln!(
            "[zenpicker-train] hybrid heads: [{}] (within-cell-optimal scalar regression, natural-unit output_specs)",
            scalar_heads
                .iter()
                .map(|h| h.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    let zq_targets = default_zq_targets();
    let mut ds = if dataset_axes.is_empty() {
        build_picker_dataset(input_path, codec.as_deref(), &zq_targets)?
    } else {
        build_picker_dataset_with(input_path, codec.as_deref(), &zq_targets, &dataset_axes)?
    };
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

    // Input shaping (zenpredict `feature_transforms`): fit a PER-FEATURE
    // transform on TRAIN rows and apply it to the full matrix BEFORE the
    // standardizer. Each image feature independently picks the transform
    // that best Gaussianises it (the dial column stays Identity). Applied
    // through zenpredict's own `apply_with_params`, so train-time shaping
    // is bit-identical to the runtime's `predict_transformed`. The bake
    // emits the `feature_transforms` (+ params) metadata.
    let shaping_mode = ShapingMode::parse(&shaping_mode_str).ok_or_else(|| {
        TrainError::Degenerate(format!(
            "unknown --input-shaping '{shaping_mode_str}' (expected none|auto|yeo)"
        ))
    })?;
    let fitted = fit_transforms(&ds.features, ds.n_in, &train_rows, shaping_mode);
    if !fitted.is_all_identity() {
        eprintln!(
            "[zenpicker-train] input shaping ({shaping_mode_str}): shaped {}/{} image features (per-feature transforms; dial stays identity)",
            fitted.n_shaped(),
            ds.n_in - 1
        );
        apply_inplace(&mut ds.features, ds.n_in, &fitted);
    } else {
        eprintln!("[zenpicker-train] input shaping: none (raw features)");
    }

    // Standardizer fit on TRAIN rows only, applied to the full (shaped) matrix.
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train_rows);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);

    // --export-dataset: write the teacher's exact dataset parquet and exit.
    if let Some(export_path) = &distill_opts.export_dataset {
        let sha = export_teacher_dataset(&ds, &train_rows, &val_rows, Path::new(export_path))?;
        eprintln!(
            "[zenpicker-train] exported teacher dataset: {export_path} (sha256 {sha}); \
             n_rows={} n_cells={} n_in={} (raw teacher inputs incl. zq_norm). \
             Run the teacher script next to produce soft_targets.parquet.",
            ds.n_rows(),
            ds.n_cells,
            ds.n_in
        );
        return Ok(());
    }

    // Distillation: obtain the teacher's DENSE soft targets, either by
    // orchestrating the offline teacher step (--distill) or from a
    // pre-computed sidecar (--soft-targets).
    let teacher_params = TeacherParams::default();
    let distill_state: Option<DistillState> = if distill_opts.distill {
        Some(orchestrate_teacher(
            &ds,
            &train_rows,
            &val_rows,
            out_path,
            &teacher_params,
            &distill_opts,
        )?)
    } else if let Some(sp) = &distill_opts.soft_targets {
        eprintln!("[zenpicker-train] distilling against pre-computed soft targets: {sp}");
        let soft = load_soft_targets(Path::new(sp), ds.n_rows(), ds.n_cells)?;
        Some(DistillState {
            soft_path: sp.clone(),
            soft_sha256: soft.sha256.clone(),
            export_path: String::new(),
            export_sha256: soft.source_export_sha256.clone(),
            n_cells_with_teacher: soft.n_cells_with_teacher,
            teacher_argmin: None,
            teacher_overhead: None,
            soft: soft.soft,
        })
    } else {
        None
    };

    let base = MlpConfig::default();
    let distilling = distill_state.is_some();
    if distilling {
        eprintln!(
            "[zenpicker-train] DISTILLATION: training student on the per-cell HistGB teacher's \
             dense soft bytes_log (zentrain recipe); held-out eval is vs the HARD oracle."
        );
    }

    // The closure picks hard vs soft training per the distillation state.
    let do_search = |grid: &[GridPoint]| {
        if let Some(st) = &distill_state {
            run_search_distill(
                &ds,
                &x_std,
                &st.soft,
                &train_rows,
                &val_rows,
                grid,
                &base,
                distill_opts.soft_weight,
                |m| eprintln!("{m}"),
            )
        } else {
            run_search(&ds, &x_std, &train_rows, &val_rows, grid, &base, |m| {
                eprintln!("{m}")
            })
        }
    };

    // Either a single explicit topology or the bounded grid search.
    let (model, cfg, eval, search_manifest) = if let Some(h) = hidden_override {
        let hidden = parse_hidden(&h)?;
        let grid = vec![GridPoint {
            hidden: hidden.clone(),
            lr: base.lr,
            seed: seed_override.unwrap_or(0),
        }];
        eprintln!("[zenpicker-train] single fit: hidden={hidden:?} (search disabled)");
        let res = do_search(&grid).ok_or("MLP fit produced no evaluable model")?;
        let sm = build_search_manifest(&res, "single_fit");
        (res.best_model, res.best_cfg, res.best_eval, sm)
    } else {
        let grid = default_grid();
        eprintln!(
            "[zenpicker-train] bounded grid search: {} candidates (hidden × lr × seed), ranked by held-out argmin accuracy",
            grid.len()
        );
        let res = do_search(&grid).ok_or("grid search produced no evaluable model")?;
        eprintln!(
            "[zenpicker-train] selected candidate #{}: hidden={:?} lr={} seed={}",
            res.selected_index, res.best_cfg.hidden, res.best_cfg.lr, res.best_cfg.seed
        );
        let sm = build_search_manifest(&res, "bounded_grid");
        (res.best_model, res.best_cfg, res.best_eval, sm)
    };

    // Build the distillation manifest block, if distilling.
    let distillation: Option<DistillManifest> = distill_state.as_ref().map(|st| DistillManifest {
        recipe: "histgb_per_cell_soft_target_mse".to_string(),
        recipe_note: "zentrain/tools/train_hybrid.py: per-cell HistGradientBoostingRegressor \
            teacher (one tree ensemble per categorical cell, fit on the reaching train rows for \
            that cell), then teacher_predict_all produces DENSE per-(row,cell) soft bytes_log; \
            the LeakyReLU MLP student trains on those soft targets via pure MSE (no hard-target \
            blend, no temperature, no sample weighting). Held-out eval is argmin(prediction, \
            mask=reach) vs the true within-cell-optimal oracle — distillation changes the \
            training target only."
            .to_string(),
        teacher_kind: "sklearn.ensemble.HistGradientBoostingRegressor (per cell)".to_string(),
        teacher_max_iter: teacher_params.max_iter,
        teacher_max_depth: teacher_params.max_depth,
        teacher_learning_rate: teacher_params.learning_rate,
        teacher_l2_regularization: teacher_params.l2_regularization,
        teacher_min_cell_rows: teacher_params.min_cell_rows,
        teacher_random_state: teacher_params.random_state,
        teacher_params_fingerprint: teacher_params_fingerprint(&teacher_params),
        n_cells_with_teacher: st.n_cells_with_teacher,
        student_loss: if distill_opts.soft_weight >= 1.0 {
            "soft_target_mse (dense, unmasked)".to_string()
        } else {
            format!(
                "blended_mse (soft_weight={:.3}*soft + {:.3}*hard where reachable)",
                distill_opts.soft_weight,
                1.0 - distill_opts.soft_weight
            )
        },
        soft_weight: distill_opts.soft_weight,
        teacher_dataset_export_path: st.export_path.clone(),
        teacher_dataset_export_sha256: st.export_sha256.clone(),
        soft_targets_path: st.soft_path.clone(),
        soft_targets_sha256: st.soft_sha256.clone(),
        teacher_script: distill_opts
            .teacher_script
            .clone()
            .unwrap_or_else(|| default_teacher_script().display().to_string()),
        teacher_heldout_argmin_acc: st.teacher_argmin,
        teacher_heldout_overhead_mean: st.teacher_overhead,
    });

    if let Some(d) = &distillation {
        eprintln!("[zenpicker-train] DISTILLED student. Teacher (HistGB) provenance:");
        eprintln!(
            "    soft targets   = {} (sha256 {})",
            d.soft_targets_path, d.soft_targets_sha256
        );
        eprintln!(
            "    teacher        = {} cells/{} got a per-cell teacher (else nanmean)",
            d.n_cells_with_teacher, ds.n_cells
        );
        if let (Some(ta), Some(to)) = (
            d.teacher_heldout_argmin_acc,
            d.teacher_heldout_overhead_mean,
        ) {
            eprintln!(
                "    teacher heldout= argmin_acc {ta:.4} | overhead mean {to:.3} (the distillation ceiling)"
            );
        }
    }
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
            distillation,
            transforms: if fitted.is_all_identity() {
                None
            } else {
                Some(&fitted)
            },
            shaping_mode: &shaping_mode_str,
            scalar_heads: &scalar_heads,
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

/// In-flight distillation state threaded from the teacher step into the
/// student fit + manifest.
struct DistillState {
    soft_path: String,
    soft_sha256: String,
    export_path: String,
    export_sha256: String,
    n_cells_with_teacher: usize,
    teacher_argmin: Option<f64>,
    teacher_overhead: Option<f64>,
    /// Dense `n_rows × n_cells` soft targets in dataset row order.
    soft: Vec<f64>,
}

/// `--distill` orchestration: export the dataset → shell to the Python
/// HistGB teacher (one-time offline target-gen) → load the soft targets.
/// The Rust runtime never imports Python; this is a build-side step.
fn orchestrate_teacher(
    ds: &zenpicker_train::PickerDataset,
    train_rows: &[usize],
    val_rows: &[usize],
    out_path: &Path,
    teacher_params: &TeacherParams,
    opts: &DistillOpts,
) -> Result<DistillState, Box<dyn std::error::Error>> {
    let export_path = opts
        .export_dataset
        .clone()
        .unwrap_or_else(|| sibling(out_path, "teacher_export.parquet"));
    let soft_path = opts
        .soft_targets
        .clone()
        .unwrap_or_else(|| sibling(out_path, "soft_targets.parquet"));
    let stats_path = sibling(out_path, "teacher_stats.json");
    let script = opts
        .teacher_script
        .clone()
        .unwrap_or_else(|| default_teacher_script().display().to_string());

    eprintln!("[zenpicker-train] [distill] exporting teacher dataset → {export_path}");
    let export_sha = export_teacher_dataset(ds, train_rows, val_rows, Path::new(&export_path))?;
    eprintln!("[zenpicker-train] [distill] export sha256 {export_sha}");

    eprintln!(
        "[zenpicker-train] [distill] running teacher: {} {} (HistGB max_iter={} max_depth={} lr={} l2={} min_cell_rows={})",
        opts.python,
        script,
        teacher_params.max_iter,
        teacher_params.max_depth,
        teacher_params.learning_rate,
        teacher_params.l2_regularization,
        teacher_params.min_cell_rows,
    );
    let status = std::process::Command::new(&opts.python)
        .arg(&script)
        .arg("--export")
        .arg(&export_path)
        .arg("--out")
        .arg(&soft_path)
        .arg("--stats-out")
        .arg(&stats_path)
        .arg("--n-cells")
        .arg(ds.n_cells.to_string())
        .arg("--max-iter")
        .arg(teacher_params.max_iter.to_string())
        .arg("--max-depth")
        .arg(teacher_params.max_depth.to_string())
        .arg("--learning-rate")
        .arg(teacher_params.learning_rate.to_string())
        .arg("--l2-regularization")
        .arg(teacher_params.l2_regularization.to_string())
        .arg("--min-cell-rows")
        .arg(teacher_params.min_cell_rows.to_string())
        .arg("--random-state")
        .arg(teacher_params.random_state.to_string())
        .status()
        .map_err(|e| format!("failed to launch teacher script {script:?}: {e}"))?;
    if !status.success() {
        return Err(format!(
            "teacher script {script:?} exited with {status}; cannot distill without soft targets"
        )
        .into());
    }

    let soft = load_soft_targets(Path::new(&soft_path), ds.n_rows(), ds.n_cells)?;
    // Integrity gate: the teacher must have been fit on the export we wrote.
    if !soft.source_export_sha256.is_empty() && soft.source_export_sha256 != export_sha {
        return Err(format!(
            "soft-target provenance mismatch: teacher was fit on export sha256 {}, but we wrote {}",
            soft.source_export_sha256, export_sha
        )
        .into());
    }

    // Optional: parse the teacher's own held-out numbers from the stats JSON.
    let (teacher_argmin, teacher_overhead) = read_teacher_stats(&stats_path);

    Ok(DistillState {
        soft_path,
        soft_sha256: soft.sha256.clone(),
        export_path,
        export_sha256: export_sha,
        n_cells_with_teacher: soft.n_cells_with_teacher,
        teacher_argmin,
        teacher_overhead,
        soft: soft.soft,
    })
}

/// Build a sibling path `<out_stem>.<suffix>` next to the bake output.
fn sibling(out_path: &Path, suffix: &str) -> String {
    let mut s = out_path.as_os_str().to_os_string();
    s.push(".");
    s.push(suffix);
    PathBuf::from(s).display().to_string()
}

/// Read `{argmin_acc, overhead_mean}` from the teacher stats JSON, if present.
fn read_teacher_stats(path: &str) -> (Option<f64>, Option<f64>) {
    let Ok(txt) = std::fs::read_to_string(path) else {
        return (None, None);
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) else {
        return (None, None);
    };
    let argmin = v.get("argmin_acc").and_then(|x| x.as_f64());
    let overhead = v.get("overhead_mean").and_then(|x| x.as_f64());
    (argmin, overhead)
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
