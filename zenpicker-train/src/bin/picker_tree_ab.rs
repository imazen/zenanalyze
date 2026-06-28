//! `picker_tree_ab` — does a tabular **tree** picker beat the hand-trained
//! **MLP** picker? (`--features tree-ab`)
//!
//! The picker chooses, per `(image, requested-quality)`, the codec config
//! cell with the least bytes that reaches the target. GBDT is tabular SOTA
//! and often beats small MLPs on ~100-feature problems — so this A/B trains
//! all three on the SAME held-out split and compares the decision metric
//! the codec actually cares about: **held-out argmin accuracy** + the
//! **byte overhead** of the pick (mean AND the p90/p99/worst tail).
//!
//! All three predict the same target — per-cell `bytes_log` (regression),
//! masked to reachable cells — then `argmin(pred, mask=reach)`:
//!   - **MLP**: one net, `n_cells` outputs, masked-NaN MSE (the shipped trainer).
//!   - **GBDT** (`gbdt`): one gradient-boosted regressor per cell.
//!   - **RF** (`smartcore` RandomForestRegressor): same per-cell shape.
//!
//! ## Origin split (vs the in-tool grouped split)
//!
//! The shipped `--val-frac` path groups by `image_path` — which LEAKS
//! across renditions (different sizes of one origin land on both sides).
//! For a fair held-out number, pass `--split-map <parquet>` (a tiny
//! image_path→split table) + `--eval-split val|test`; rows are then
//! partitioned by the canonical even/odd-by-origin `split` column (no
//! leakage). Train rows are always `split=="train"`.
//!
//! ## Dataset export (for the Python CART comparison)
//!
//! `--dump-dir <DIR>` writes the EXACT built dataset (raw features incl.
//! zq_norm, per-cell `bytes_log`, oracle pick, per-row split) so an
//! external sklearn CART can be fit/evaluated against the IDENTICAL
//! cells/reach/oracle — making its overhead directly comparable to the
//! GBDT/MLP numbers here. Also writes per-model per-row overhead TSVs.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use arrow::array::{Array, StringArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use gbdt::config::Config as GbdtConfig;
use gbdt::decision_tree::{Data, DataVec, ValueType};
use gbdt::gradient_boost::GBDT;
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

use zenpicker_train::{
    GridPoint, MlpConfig, PickerDataset, ScalarAxisSpec, build_picker_dataset_with,
    default_zq_targets, fit_standardizer, grouped_split_picker, run_search, standardize_all,
};

/// Minimum reaching train rows to fit a tree for a cell; below this we fall
/// back to the constant mean (a near-empty cell carries no learnable signal).
const MIN_CELL_ROWS: usize = 8;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut codec: Option<String> = None;
    let mut val_frac = 0.2f64;
    let mut seed = 0u64;
    let mut split_map: Option<String> = None;
    let mut eval_split = "val".to_string();
    let mut dump_dir: Option<String> = None;
    let mut codec_tag = "codec".to_string();
    let mut skip_rf = false;
    let mut skip_mlp = false;
    let mut max_train = 0usize;
    let mut mlp_iter = 250usize;
    let mut rf_trees = 60usize;
    let mut perm_val_cap = 6000usize;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().cloned(),
            "--codec" => codec = it.next().cloned(),
            "--val-frac" => val_frac = it.next().and_then(|s| s.parse().ok()).unwrap_or(val_frac),
            "--seed" => seed = it.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--split-map" => split_map = it.next().cloned(),
            "--eval-split" => eval_split = it.next().cloned().unwrap_or(eval_split),
            "--dump-dir" => dump_dir = it.next().cloned(),
            "--codec-tag" => codec_tag = it.next().cloned().unwrap_or(codec_tag),
            "--skip-rf" => skip_rf = true,
            "--skip-mlp" => skip_mlp = true,
            "--max-train" => {
                max_train = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_train)
            }
            "--mlp-iter" => mlp_iter = it.next().and_then(|s| s.parse().ok()).unwrap_or(mlp_iter),
            "--rf-trees" => rf_trees = it.next().and_then(|s| s.parse().ok()).unwrap_or(rf_trees),
            "--perm-val-cap" => {
                perm_val_cap = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(perm_val_cap)
            }
            "-h" | "--help" => {
                eprintln!(
                    "picker_tree_ab --input <parquet> [--codec C] [--split-map <parquet> --eval-split val|test] [--val-frac F] [--seed N] [--dump-dir DIR] [--codec-tag TAG] [--skip-rf] [--skip-mlp] [--max-train N] [--mlp-iter N] [--rf-trees N] [--perm-val-cap N]"
                );
                return;
            }
            other => {
                eprintln!("unknown arg {other:?}");
                std::process::exit(1);
            }
        }
    }
    let Some(input) = input else {
        eprintln!("--input <parquet> required");
        std::process::exit(1);
    };

    // No scalar axes: the unified plan-cell schema makes the cell key the
    // `cell` value directly (see pareto_dataset::cell_key_from_knob), so
    // every distinct categorical config is its own cell.
    let axes: Vec<ScalarAxisSpec> = Vec::new();
    let zq = default_zq_targets();
    let ds = build_picker_dataset_with(Path::new(&input), codec.as_deref(), &zq, &axes)
        .expect("build picker dataset");

    // --- Split: origin (split column) if a split-map is given, else the
    // in-tool grouped-by-image_path split (leaky across renditions). ---
    let (train, val, split_kind) = if let Some(sm) = &split_map {
        let map = load_split_map(sm);
        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut missing = 0usize;
        for (i, img) in ds.image_ids.iter().enumerate() {
            match map.get(img).map(String::as_str) {
                Some("train") => train.push(i),
                Some(s) if s == eval_split => val.push(i),
                Some(_) => {} // a third split not selected for eval
                None => missing += 1,
            }
        }
        if missing > 0 {
            eprintln!("[picker_tree_ab] WARN {missing} rows had no split-map entry (dropped)");
        }
        (train, val, format!("origin split (train -> {eval_split})"))
    } else {
        let (t, v) = grouped_split_picker(&ds, val_frac);
        (
            t,
            v,
            format!("grouped-by-image_path val_frac={val_frac} (LEAKY)"),
        )
    };

    // Subsample train (all models + standardizer see the SAME rows) so the
    // hand-rolled single-threaded MLP stays tractable. 0 = use all.
    let train = if max_train > 0 && train.len() > max_train {
        let perm = shuffled_perm(train.len(), seed ^ 0x5A5A_5A5A);
        let mut t: Vec<usize> = perm.iter().take(max_train).map(|&i| train[i]).collect();
        t.sort_unstable();
        eprintln!(
            "[picker_tree_ab] train subsampled to {} rows (was {})",
            t.len(),
            perm.len()
        );
        t
    } else {
        train
    };

    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);

    eprintln!(
        "[picker_tree_ab] codec={codec_tag} {} rows | {} features (+zq_norm) | {} cells | {} | train {} / val {}",
        ds.n_rows(),
        ds.feature_names.len(),
        ds.n_cells,
        split_kind,
        train.len(),
        val.len()
    );

    let valrows: Vec<Vec<f64>> = val
        .iter()
        .map(|&r| x_std[r * ds.n_in..(r + 1) * ds.n_in].to_vec())
        .collect();

    // --- MLP (single 128,128 fit, matches the parity run). Keep the model
    // so we can extract per-row predictions for the tail analysis. ---
    let grid = vec![GridPoint {
        hidden: vec![128, 128],
        lr: 2e-3,
        seed,
    }];
    let base = MlpConfig {
        seed,
        max_iter: mlp_iter,
        n_iter_no_change: 25,
        ..Default::default()
    };
    let mlp_res = if skip_mlp {
        None
    } else {
        run_search(&ds, &x_std, &train, &val, &grid, &base, |_| {})
    };
    let mlp_pred: Vec<f64> = match &mlp_res {
        Some(r) => {
            let n_cells = ds.n_cells;
            let mut pred = vec![f64::NAN; val.len() * n_cells];
            for (vi, row) in valrows.iter().enumerate() {
                let out = r.best_model.predict(row);
                pred[vi * n_cells..(vi + 1) * n_cells].copy_from_slice(&out[0..n_cells]);
            }
            pred
        }
        None => vec![f64::NAN; val.len() * ds.n_cells],
    };
    let (mlp_acc, mlp_mean, mlp_rows) = score_rows(&mlp_pred, ds.n_cells, &ds, &val);
    let mlp_srocc = mlp_res
        .as_ref()
        .map(|r| r.best_eval.bytes_panel.srocc)
        .unwrap_or(f64::NAN);
    let mlp_niter = mlp_res.as_ref().map(|r| r.best_model.n_iter).unwrap_or(0);
    eprintln!("[picker_tree_ab] MLP done: argmin={mlp_acc:.4} (n_iter={mlp_niter}/{mlp_iter})");

    // --- GBDT (gbdt, per cell) — retained for permutation importance. ---
    let gbdt_cells = train_gbdt_cells(&ds, &x_std, &train);
    let gbdt_pred = predict_gbdt(&gbdt_cells, ds.n_cells, &valrows);
    let (gbdt_acc, gbdt_mean, gbdt_rows) = score_rows(&gbdt_pred, ds.n_cells, &ds, &val);
    eprintln!("[picker_tree_ab] GBDT done: argmin={gbdt_acc:.4}");

    // --- RF (smartcore, per cell) ---
    let (rf_acc, rf_mean, rf_rows) = if skip_rf {
        (f64::NAN, f64::NAN, Vec::new())
    } else {
        let rf_pred = train_per_cell(&ds, &x_std, &train, &val, |xrows, y, _n_in| {
            let xm = DenseMatrix::from_2d_vec(&xrows.to_vec()).expect("dense matrix");
            let params = RandomForestRegressorParameters::default()
                .with_n_trees(rf_trees)
                .with_max_depth(8)
                .with_seed(seed);
            let rf = RandomForestRegressor::fit(&xm, &y.to_vec(), params).expect("rf fit");
            Box::new(move |valrows: &[Vec<f64>]| -> Vec<f64> {
                let xv = DenseMatrix::from_2d_vec(&valrows.to_vec()).expect("dense matrix");
                rf.predict(&xv).expect("rf predict")
            })
        });
        let (a, m, r) = score_rows(&rf_pred, ds.n_cells, &ds, &val);
        eprintln!("[picker_tree_ab] RF done: argmin={a:.4}");
        (a, m, r)
    };

    // --- Report table: argmin acc + overhead mean + TAIL (p50/p90/p99/worst) ---
    let mlp_ov: Vec<f64> = mlp_rows.iter().flatten().map(|r| r.ov).collect();
    let gbdt_ov: Vec<f64> = gbdt_rows.iter().flatten().map(|r| r.ov).collect();
    let rf_ov: Vec<f64> = rf_rows.iter().flatten().map(|r| r.ov).collect();
    let (m_mean, m_p50, m_p90, m_p99, m_worst) = summarize(&mlp_ov);
    let (g_mean, g_p50, g_p90, g_p99, g_worst) = summarize(&gbdt_ov);
    let (r_mean, r_p50, r_p90, r_p99, r_worst) = summarize(&rf_ov);
    let _ = (mlp_mean, gbdt_mean, rf_mean);

    println!("\n=== picker A/B — codec={codec_tag} | {split_kind} ===");
    println!(
        "model            argmin_acc   ov_mean   ov_p50   ov_p90   ov_p99   ov_WORST   (extra)"
    );
    println!(
        "MLP (128,128)    {mlp_acc:.4}      {:.4}   {:.4}   {:.4}   {:.4}   {:.4}   (bytes-SROCC {mlp_srocc:.4})",
        m_mean, m_p50, m_p90, m_p99, m_worst
    );
    println!(
        "GBDT (gbdt)      {gbdt_acc:.4}      {:.4}   {:.4}   {:.4}   {:.4}   {:.4}",
        g_mean, g_p50, g_p90, g_p99, g_worst
    );
    if !skip_rf {
        println!(
            "RF (smartcore)   {rf_acc:.4}      {:.4}   {:.4}   {:.4}   {:.4}   {:.4}",
            r_mean, r_p50, r_p90, r_p99, r_worst
        );
    }
    let best = [("MLP", mlp_acc), ("GBDT", gbdt_acc), ("RF", rf_acc)]
        .into_iter()
        .filter(|(_, a)| a.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((name, acc)) = best {
        println!("\nwinner (argmin accuracy): {name} @ {acc:.4}");
    }

    // --- Permutation feature importance on the GBDT ---
    let feat_name = |f: usize| -> String {
        if f < ds.feature_names.len() {
            ds.feature_names[f].clone()
        } else {
            "zq_norm".to_string()
        }
    };
    // Cap val for the (ranking-robust) permutation importance to bound cost.
    let perm_val: Vec<usize> = if val.len() > perm_val_cap {
        let step = (val.len() / perm_val_cap).max(1);
        val.iter().step_by(step).copied().collect()
    } else {
        val.clone()
    };
    let perm_valrows: Vec<Vec<f64>> = perm_val
        .iter()
        .map(|&r| x_std[r * ds.n_in..(r + 1) * ds.n_in].to_vec())
        .collect();
    let perm_pred = predict_gbdt(&gbdt_cells, ds.n_cells, &perm_valrows);
    let (perm_base_acc, _, _) = score_rows(&perm_pred, ds.n_cells, &ds, &perm_val);
    let per_feat = permutation_importance(
        &gbdt_cells,
        ds.n_cells,
        &ds,
        &perm_val,
        &perm_valrows,
        perm_base_acc,
        seed,
    );
    println!("\n=== GBDT permutation importance — per feature (argmin-acc drop) ===");
    for (f, drop) in per_feat.iter().take(20) {
        println!("  {:+.4}  {}", drop, feat_name(*f));
    }
    let nonpos = per_feat.iter().filter(|(_, d)| *d <= 0.0).count();
    println!(
        "  ... ({nonpos}/{} features individually non-positive — many redundant twins; see groups)",
        per_feat.len()
    );

    let groups = corr_groups(&x_std, ds.n_in, &train, 0.9);
    let grp_imp = group_perm_importance(
        &gbdt_cells,
        ds.n_cells,
        &ds,
        &perm_val,
        &perm_valrows,
        perm_base_acc,
        &groups,
        seed,
    );
    println!(
        "\n=== GBDT permutation importance — by rho>=0.9 redundancy group ({} feats -> {} groups) ===",
        ds.n_in,
        groups.len()
    );
    for (grp, drop) in grp_imp.iter().take(15) {
        let mut members: Vec<String> = grp.iter().take(5).map(|&f| feat_name(f)).collect();
        if grp.len() > 5 {
            members.push(format!("+{} more", grp.len() - 5));
        }
        println!("  {:+.4}  [{}]", drop, members.join(", "));
    }

    // --- Dataset + per-row dump for the external CART comparison ---
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).expect("mkdir dump-dir");
        let train_set: HashMap<usize, ()> = train.iter().map(|&r| (r, ())).collect();
        dump_dataset(dir, &codec_tag, &ds, &split_map, &eval_split, &train_set);
        if !skip_mlp {
            dump_perrow(dir, &codec_tag, "mlp", &ds, &val, &mlp_rows);
        }
        dump_perrow(dir, &codec_tag, "gbdt", &ds, &val, &gbdt_rows);
        if !skip_rf {
            dump_perrow(dir, &codec_tag, "rf", &ds, &val, &rf_rows);
        }
        eprintln!("[picker_tree_ab] dumped dataset + per-row TSVs to {dir}");
    }
}

/// Read a tiny image_path/image_basename -> split table.
fn load_split_map(path: &str) -> HashMap<String, String> {
    let file = File::open(path).expect("open split-map parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("split-map reader builder");
    let schema = builder.schema().clone();
    let pos = |name: &str| schema.fields().iter().position(|f| f.name() == name);
    let img_idx = pos("image_basename")
        .or_else(|| pos("image_path"))
        .expect("split-map needs image_basename or image_path");
    let split_idx = pos("split").expect("split-map needs split column");
    let parquet_schema = builder.parquet_schema().clone();
    let mask = ProjectionMask::roots(&parquet_schema, [img_idx, split_idx]);
    let reader = builder
        .with_projection(mask)
        .build()
        .expect("split-map reader");
    let mut map = HashMap::new();
    for batch in reader {
        let batch = batch.expect("split-map batch");
        let bs = batch.schema();
        let bi = bs
            .fields()
            .iter()
            .position(|f| f.name() == "image_basename" || f.name() == "image_path")
            .expect("img in batch");
        let bsp = bs
            .fields()
            .iter()
            .position(|f| f.name() == "split")
            .expect("split in batch");
        let imgs = batch
            .column(bi)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("img col utf8");
        let splits = batch
            .column(bsp)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("split col utf8");
        for i in 0..imgs.len() {
            if !imgs.is_null(i) && !splits.is_null(i) {
                map.entry(imgs.value(i).to_string())
                    .or_insert_with(|| splits.value(i).to_string());
            }
        }
    }
    map
}

/// Per-row score: overhead, picked cell, true-best cell, reachable count.
struct RowScore {
    ov: f64,
    pick: usize,
    best: usize,
    n_reach: usize,
}

/// Per-row argmin accuracy + mean byte overhead + per-row detail from a
/// `val.len() x n_cells` prediction matrix vs the within-cell-optimal.
fn score_rows(
    pred: &[f64],
    n_cells: usize,
    ds: &PickerDataset,
    val: &[usize],
) -> (f64, f64, Vec<Option<RowScore>>) {
    let mut per = Vec::with_capacity(val.len());
    let mut hits = 0usize;
    let mut scored = 0usize;
    let mut ovsum = 0.0f64;
    for (vi, &r) in val.iter().enumerate() {
        let reach = &ds.reach[r * n_cells..(r + 1) * n_cells];
        let truth = &ds.bytes_log[r * n_cells..(r + 1) * n_cells];
        let pick = argmin_masked(&pred[vi * n_cells..(vi + 1) * n_cells], reach);
        let best = argmin_masked(truth, reach);
        let n_reach = reach.iter().filter(|&&b| b).count();
        if let (Some(pk), Some(bk)) = (pick, best) {
            scored += 1;
            if pk == bk {
                hits += 1;
            }
            let ov = (truth[pk] - truth[bk]).exp() - 1.0;
            ovsum += ov;
            per.push(Some(RowScore {
                ov,
                pick: pk,
                best: bk,
                n_reach,
            }));
        } else {
            per.push(None);
        }
    }
    let acc = if scored > 0 {
        hits as f64 / scored as f64
    } else {
        f64::NAN
    };
    let mean = if scored > 0 {
        ovsum / scored as f64
    } else {
        f64::NAN
    };
    (acc, mean, per)
}

/// (mean, p50, p90, p99, worst) of a slice of overheads.
fn summarize(ovs: &[f64]) -> (f64, f64, f64, f64, f64) {
    if ovs.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let mut v = ovs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| -> f64 {
        let idx = (((v.len() - 1) as f64) * p).round() as usize;
        v[idx.min(v.len() - 1)]
    };
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (mean, pct(0.50), pct(0.90), pct(0.99), *v.last().unwrap())
}

/// Index of the smallest `vals[c]` over cells where `reach[c]`. NaN never wins.
fn argmin_masked(vals: &[f64], reach: &[bool]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut bv = f64::INFINITY;
    for (c, (&v, &re)) in vals.iter().zip(reach).enumerate() {
        if re && v < bv {
            bv = v;
            best = Some(c);
        }
    }
    best
}

/// Train one regressor per cell on its reaching train rows -> `val.len() x
/// n_cells` row-major prediction matrix.
fn train_per_cell<F>(
    ds: &PickerDataset,
    x_std: &[f64],
    train: &[usize],
    val: &[usize],
    mut fit_cell: F,
) -> Vec<f64>
where
    F: FnMut(&[Vec<f64>], &[f64], usize) -> Box<dyn FnOnce(&[Vec<f64>]) -> Vec<f64>>,
{
    let n_cells = ds.n_cells;
    let n_in = ds.n_in;
    let valrows: Vec<Vec<f64>> = val
        .iter()
        .map(|&r| x_std[r * n_in..(r + 1) * n_in].to_vec())
        .collect();
    let mut pred = vec![f64::NAN; val.len() * n_cells];
    for c in 0..n_cells {
        let mut xrows: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for &r in train {
            let label = ds.bytes_log[r * n_cells + c];
            if label.is_finite() {
                xrows.push(x_std[r * n_in..(r + 1) * n_in].to_vec());
                y.push(label);
            }
        }
        if xrows.len() < MIN_CELL_ROWS {
            let m = if y.is_empty() {
                f64::INFINITY
            } else {
                y.iter().sum::<f64>() / y.len() as f64
            };
            for vi in 0..val.len() {
                pred[vi * n_cells + c] = m;
            }
            continue;
        }
        let predict = fit_cell(&xrows, &y, n_in);
        let cell_preds = predict(&valrows);
        for (vi, &p) in cell_preds.iter().enumerate() {
            pred[vi * n_cells + c] = p;
        }
    }
    pred
}

/// A per-cell GBDT regressor, or a constant fallback.
enum GbdtCell {
    Model(GBDT),
    Const(f64),
}

fn train_gbdt_cells(ds: &PickerDataset, x_std: &[f64], train: &[usize]) -> Vec<GbdtCell> {
    let n_cells = ds.n_cells;
    let n_in = ds.n_in;
    let mut cells = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let mut td: DataVec = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for &r in train {
            let label = ds.bytes_log[r * n_cells + c];
            if label.is_finite() {
                let feat: Vec<ValueType> = x_std[r * n_in..(r + 1) * n_in]
                    .iter()
                    .map(|&v| v as ValueType)
                    .collect();
                td.push(Data::new_training_data(
                    feat,
                    1.0 as ValueType,
                    label as ValueType,
                    None,
                ));
                ys.push(label);
            }
        }
        if td.len() < MIN_CELL_ROWS {
            let m = if ys.is_empty() {
                f64::INFINITY
            } else {
                ys.iter().sum::<f64>() / ys.len() as f64
            };
            cells.push(GbdtCell::Const(m));
            continue;
        }
        let mut cfg = GbdtConfig::new();
        cfg.set_feature_size(n_in);
        cfg.set_max_depth(4);
        cfg.set_iterations(80);
        cfg.set_shrinkage(0.1 as ValueType);
        cfg.set_loss("SquaredError");
        let mut g = GBDT::new(&cfg);
        g.fit(&mut td);
        cells.push(GbdtCell::Model(g));
    }
    cells
}

fn predict_gbdt(cells: &[GbdtCell], n_cells: usize, valrows: &[Vec<f64>]) -> Vec<f64> {
    let mut pred = vec![f64::NAN; valrows.len() * n_cells];
    for (c, cell) in cells.iter().enumerate() {
        match cell {
            GbdtCell::Model(g) => {
                let test: DataVec = valrows
                    .iter()
                    .map(|row| {
                        Data::new_test_data(row.iter().map(|&v| v as ValueType).collect(), None)
                    })
                    .collect();
                for (vi, &pv) in g.predict(&test).iter().enumerate() {
                    pred[vi * n_cells + c] = pv as f64;
                }
            }
            GbdtCell::Const(m) => {
                for vi in 0..valrows.len() {
                    pred[vi * n_cells + c] = *m;
                }
            }
        }
    }
    pred
}

fn shuffled_perm(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn permutation_importance(
    cells: &[GbdtCell],
    n_cells: usize,
    ds: &PickerDataset,
    val: &[usize],
    valrows: &[Vec<f64>],
    base_acc: f64,
    seed: u64,
) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(ds.n_in);
    for f in 0..ds.n_in {
        let perm = shuffled_perm(valrows.len(), seed.wrapping_add(f as u64 + 1));
        let orig: Vec<f64> = valrows.iter().map(|row| row[f]).collect();
        let mut sv = valrows.to_vec();
        for (i, row) in sv.iter_mut().enumerate() {
            row[f] = orig[perm[i]];
        }
        let pred = predict_gbdt(cells, n_cells, &sv);
        let (acc, _, _) = score_rows(&pred, n_cells, ds, val);
        out.push((f, base_acc - acc));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn corr_groups(x_std: &[f64], n_in: usize, train: &[usize], thr: f64) -> Vec<Vec<usize>> {
    let n = train.len() as f64;
    let mut parent: Vec<usize> = (0..n_in).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut c = x;
        while p[c] != c {
            let nx = p[c];
            p[c] = r;
            c = nx;
        }
        r
    }
    for a in 0..n_in {
        for b in (a + 1)..n_in {
            let mut dot = 0.0;
            for &r in train {
                dot += x_std[r * n_in + a] * x_std[r * n_in + b];
            }
            if (dot / n).abs() >= thr {
                let ra = find(&mut parent, a);
                let rb = find(&mut parent, b);
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut map: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for f in 0..n_in {
        let r = find(&mut parent, f);
        map.entry(r).or_default().push(f);
    }
    map.into_values().collect()
}

#[allow(clippy::too_many_arguments)]
fn group_perm_importance(
    cells: &[GbdtCell],
    n_cells: usize,
    ds: &PickerDataset,
    val: &[usize],
    valrows: &[Vec<f64>],
    base_acc: f64,
    groups: &[Vec<usize>],
    seed: u64,
) -> Vec<(Vec<usize>, f64)> {
    let mut out = Vec::with_capacity(groups.len());
    for (gi, grp) in groups.iter().enumerate() {
        let perm = shuffled_perm(valrows.len(), seed.wrapping_add(1000 + gi as u64));
        let mut sv = valrows.to_vec();
        for &f in grp {
            let orig: Vec<f64> = valrows.iter().map(|row| row[f]).collect();
            for (i, row) in sv.iter_mut().enumerate() {
                row[f] = orig[perm[i]];
            }
        }
        let pred = predict_gbdt(cells, n_cells, &sv);
        let (acc, _, _) = score_rows(&pred, n_cells, ds, val);
        out.push((grp.clone(), base_acc - acc));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// Dump the exact built dataset: raw features (incl. zq_norm as the last
/// column) + per-cell bytes_log + per-row split/oracle, so an external
/// sklearn CART can be fit/evaluated against the IDENTICAL cells/oracle.
fn dump_dataset(
    dir: &str,
    codec_tag: &str,
    ds: &PickerDataset,
    split_map: &Option<String>,
    eval_split: &str,
    train_set: &HashMap<usize, ()>,
) {
    let n_rows = ds.n_rows();
    let n_in = ds.n_in;
    let n_cells = ds.n_cells;

    // Per-row split label.
    let smap = split_map.as_ref().map(|p| load_split_map(p));
    let split_of = |img: &str| -> String {
        match &smap {
            Some(m) => m.get(img).cloned().unwrap_or_else(|| "?".into()),
            None => "?".into(),
        }
    };

    // X (raw features incl. zq_norm), f32 LE, row-major n_rows x n_in.
    let xpath = format!("{dir}/{codec_tag}_X.f32");
    let mut xw = BufWriter::new(File::create(&xpath).expect("create X"));
    let mut buf = Vec::with_capacity(n_in * 4);
    for r in 0..n_rows {
        buf.clear();
        for j in 0..n_in {
            buf.extend_from_slice(&(ds.features[r * n_in + j] as f32).to_le_bytes());
        }
        xw.write_all(&buf).expect("write X row");
    }
    xw.flush().ok();

    // bytes_log, f32 LE, row-major n_rows x n_cells (NaN = unreachable).
    let bpath = format!("{dir}/{codec_tag}_byteslog.f32");
    let mut bw = BufWriter::new(File::create(&bpath).expect("create byteslog"));
    let mut bbuf = Vec::with_capacity(n_cells * 4);
    for r in 0..n_rows {
        bbuf.clear();
        for c in 0..n_cells {
            bbuf.extend_from_slice(&(ds.bytes_log[r * n_cells + c] as f32).to_le_bytes());
        }
        bw.write_all(&bbuf).expect("write byteslog row");
    }
    bw.flush().ok();

    // rows.tsv: idx split image_id target_zq oracle_cell n_reach
    let rpath = format!("{dir}/{codec_tag}_rows.tsv");
    let mut rw = BufWriter::new(File::create(&rpath).expect("create rows.tsv"));
    writeln!(
        rw,
        "idx\tsplit\timage_id\ttarget_zq\toracle_cell\tn_reach\tin_train"
    )
    .ok();
    for r in 0..n_rows {
        let reach = &ds.reach[r * n_cells..(r + 1) * n_cells];
        let truth = &ds.bytes_log[r * n_cells..(r + 1) * n_cells];
        let oracle = argmin_masked(truth, reach).map(|c| c as i64).unwrap_or(-1);
        let n_reach = reach.iter().filter(|&&b| b).count();
        let in_train = train_set.contains_key(&r) as u8;
        writeln!(
            rw,
            "{r}\t{}\t{}\t{}\t{oracle}\t{n_reach}\t{in_train}",
            split_of(&ds.image_ids[r]),
            ds.image_ids[r],
            ds.target_zq[r]
        )
        .ok();
    }
    rw.flush().ok();

    // meta.json
    let meta = serde_json::json!({
        "codec": codec_tag,
        "eval_split": eval_split,
        "n_rows": n_rows,
        "n_in": n_in,
        "n_image_feats": n_in - 1,
        "n_cells": n_cells,
        "feature_names": ds.feature_names,
        "cell_labels": ds.cell_labels,
        "zq_targets": ds.zq_targets,
        "layout": "X.f32 row-major [n_rows x n_in] (last col = zq_norm); byteslog.f32 row-major [n_rows x n_cells], NaN=unreachable",
    });
    std::fs::write(
        format!("{dir}/{codec_tag}_meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .expect("write meta");
}

/// Per-row overhead TSV for one model on the eval rows.
fn dump_perrow(
    dir: &str,
    codec_tag: &str,
    model: &str,
    ds: &PickerDataset,
    val: &[usize],
    rows: &[Option<RowScore>],
) {
    let path = format!("{dir}/{codec_tag}_perrow_{model}.tsv");
    let mut w = BufWriter::new(File::create(&path).expect("create perrow tsv"));
    writeln!(
        w,
        "model\timage_id\ttarget_zq\tn_reach\toverhead\tpick_cell\tbest_cell\thit"
    )
    .ok();
    for (vi, &r) in val.iter().enumerate() {
        if let Some(rs) = &rows[vi] {
            writeln!(
                w,
                "{model}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}",
                ds.image_ids[r],
                ds.target_zq[r],
                rs.n_reach,
                rs.ov,
                rs.pick,
                rs.best,
                (rs.pick == rs.best) as u8
            )
            .ok();
        }
    }
    w.flush().ok();
}
