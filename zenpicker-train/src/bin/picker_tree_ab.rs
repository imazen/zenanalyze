//! `picker_tree_ab` — does a tabular **tree** picker beat the hand-trained
//! **MLP** picker? (`--features tree-ab`)
//!
//! The picker chooses, per `(image, requested-quality)`, the codec config
//! cell with the least bytes that reaches the target. GBDT is tabular SOTA
//! and often beats small MLPs on ~100-feature problems — so this A/B trains
//! all three on the SAME grouped-by-image held-out split and compares the
//! decision metric the codec actually cares about: **held-out argmin
//! accuracy** (fraction of rows whose predicted least-bytes cell equals the
//! true least-bytes cell) + the **byte overhead** of the pick.
//!
//! All three predict the same target — per-cell `bytes_log` (regression),
//! masked to reachable cells — then `argmin(pred, mask=reach)`:
//!   - **MLP**: one net, `n_cells` outputs, masked-NaN MSE (the shipped trainer).
//!   - **GBDT** (`gbdt`): one gradient-boosted regressor per cell, fit on that
//!     cell's reaching rows (the distillation-teacher shape).
//!   - **RF** (`smartcore` RandomForestRegressor): same per-cell shape.
//!
//! Tree models train per-cell over independent cells, so they parallelize
//! trivially with rayon-over-cells if needed (kept sequential here — the MLP
//! fit dominates wall time anyway).
//!
//! Usage:
//!   picker_tree_ab --input <parquet> [--codec zenjpeg] [--val-frac 0.2] [--seed 0]
//!
//! On the real zenjpeg sweep the scalar knobs (chroma_scale/lambda) are
//! excluded from the cell key (they're within-cell scalars), giving the 12
//! categorical cells the MLP parity run used.

use std::path::Path;

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
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().cloned(),
            "--codec" => codec = it.next().cloned(),
            "--val-frac" => val_frac = it.next().and_then(|s| s.parse().ok()).unwrap_or(val_frac),
            "--seed" => seed = it.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "-h" | "--help" => {
                eprintln!("picker_tree_ab --input <parquet> [--codec C] [--val-frac F] [--seed N]");
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

    // Exclude the scalar knobs from the cell key so cells are the categorical
    // tuple (matches the MLP parity run's 12 cells), not the full config grid.
    let axes = vec![
        ScalarAxisSpec::new("chroma_scale", None),
        ScalarAxisSpec::new("lambda", Some(0.0)),
    ];
    let zq = default_zq_targets();
    let ds = build_picker_dataset_with(Path::new(&input), codec.as_deref(), &zq, &axes)
        .expect("build picker dataset");
    let (train, val) = grouped_split_picker(&ds, val_frac);
    let (mean, scale) = fit_standardizer(&ds.features, ds.n_in, &train);
    let x_std = standardize_all(&ds.features, ds.n_in, &mean, &scale);

    eprintln!(
        "[picker_tree_ab] {} rows | {} features (+zq_norm) | {} cells | train {} / val {}",
        ds.n_rows(),
        ds.feature_names.len(),
        ds.n_cells,
        train.len(),
        val.len()
    );

    // --- MLP (single 128,128 fit, matches the parity run) ---
    let grid = vec![GridPoint {
        hidden: vec![128, 128],
        lr: 2e-3,
        seed,
    }];
    let base = MlpConfig {
        seed,
        ..Default::default()
    };
    let (mlp_acc, mlp_ov, mlp_srocc) =
        match run_search(&ds, &x_std, &train, &val, &grid, &base, |_| {}) {
            Some(r) => (
                r.best_eval.argmin_acc,
                r.best_eval.overhead_mean,
                r.best_eval.bytes_panel.srocc,
            ),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
    eprintln!("[picker_tree_ab] MLP done: argmin={mlp_acc:.4}");

    // --- GBDT (gbdt, per cell) — retain the cell models so we can permute
    // features for importance below. ---
    let valrows: Vec<Vec<f64>> = val
        .iter()
        .map(|&r| x_std[r * ds.n_in..(r + 1) * ds.n_in].to_vec())
        .collect();
    let gbdt_cells = train_gbdt_cells(&ds, &x_std, &train);
    let gbdt_pred = predict_gbdt(&gbdt_cells, ds.n_cells, &valrows);
    let (gbdt_acc, gbdt_ov) = score_predmatrix(&gbdt_pred, ds.n_cells, &ds, &val);
    eprintln!("[picker_tree_ab] GBDT done: argmin={gbdt_acc:.4}");

    // --- RF (smartcore, per cell) ---
    let rf_pred = train_per_cell(&ds, &x_std, &train, &val, |xrows, y, _n_in| {
        let xm = DenseMatrix::from_2d_vec(&xrows.to_vec()).expect("dense matrix");
        let params = RandomForestRegressorParameters::default()
            .with_n_trees(100)
            .with_max_depth(8)
            .with_seed(seed);
        let rf = RandomForestRegressor::fit(&xm, &y.to_vec(), params).expect("rf fit");
        Box::new(move |valrows: &[Vec<f64>]| -> Vec<f64> {
            let xv = DenseMatrix::from_2d_vec(&valrows.to_vec()).expect("dense matrix");
            rf.predict(&xv).expect("rf predict")
        })
    });
    let (rf_acc, rf_ov) = score_predmatrix(&rf_pred, ds.n_cells, &ds, &val);
    eprintln!("[picker_tree_ab] RF done: argmin={rf_acc:.4}");

    println!("\n=== picker A/B (held-out, grouped by image) ===");
    println!("model            argmin_acc   byte_overhead_mean");
    println!("MLP (128,128)    {mlp_acc:.4}       {mlp_ov:.4}   (bytes-SROCC {mlp_srocc:.4})");
    println!("GBDT (gbdt)      {gbdt_acc:.4}       {gbdt_ov:.4}");
    println!("RF (smartcore)   {rf_acc:.4}       {rf_ov:.4}");
    let best = [("MLP", mlp_acc), ("GBDT", gbdt_acc), ("RF", rf_acc)]
        .into_iter()
        .filter(|(_, a)| a.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((name, acc)) = best {
        println!("\nwinner (argmin accuracy): {name} @ {acc:.4}");
    }

    // --- Permutation feature importance on the GBDT (the A/B winner) ---
    // Shuffle a feature (or a whole ρ≥0.9 redundancy group) across the
    // held-out rows; the drop in argmin accuracy is its importance. The
    // GROUPED view is the honest one: correlated twins split single-feature
    // credit, so a feature can look useless alone while its group carries
    // real signal (the §4.2 methodology point).
    let feat_name = |f: usize| -> String {
        if f < ds.feature_names.len() {
            ds.feature_names[f].clone()
        } else {
            "zq_norm".to_string()
        }
    };
    let per_feat =
        permutation_importance(&gbdt_cells, ds.n_cells, &ds, &val, &valrows, gbdt_acc, seed);
    println!("\n=== GBDT permutation importance — per feature (argmin-acc drop) ===");
    for (f, drop) in per_feat.iter().take(15) {
        println!("  {:+.4}  {}", drop, feat_name(*f));
    }
    let nonpos = per_feat.iter().filter(|(_, d)| *d <= 0.0).count();
    println!(
        "  ... ({nonpos}/{} features individually non-positive — many are redundant twins; see groups)",
        per_feat.len()
    );

    let groups = corr_groups(&x_std, ds.n_in, &train, 0.9);
    let grp_imp = group_perm_importance(
        &gbdt_cells,
        ds.n_cells,
        &ds,
        &val,
        &valrows,
        gbdt_acc,
        &groups,
        seed,
    );
    println!(
        "\n=== GBDT permutation importance — by ρ≥0.9 redundancy group ({} feats → {} groups) ===",
        ds.n_in,
        groups.len()
    );
    for (grp, drop) in grp_imp.iter().take(12) {
        let mut members: Vec<String> = grp.iter().take(4).map(|&f| feat_name(f)).collect();
        if grp.len() > 4 {
            members.push(format!("+{} more", grp.len() - 4));
        }
        println!("  {:+.4}  [{}]", drop, members.join(", "));
    }
}

/// Train one regressor per cell on its reaching train rows and produce a
/// `val.len() × n_cells` row-major prediction matrix. `fit_cell(xrows, y,
/// n_in)` returns a closure that predicts a batch of rows. Cells with fewer
/// than [`MIN_CELL_ROWS`] reaching rows fall back to the constant cell mean
/// (`+inf` if none, so the masked argmin never picks an unlearnable cell).
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

/// Held-out argmin accuracy + mean byte overhead from a `val.len() × n_cells`
/// prediction matrix vs the true within-cell-optimal. Pick = `argmin(pred,
/// mask=reach)`; overhead = `exp(true_bytes_log[pick] − true_bytes_log[best])
/// − 1`.
fn score_predmatrix(pred: &[f64], n_cells: usize, ds: &PickerDataset, val: &[usize]) -> (f64, f64) {
    let mut hits = 0usize;
    let mut scored = 0usize;
    let mut overhead_sum = 0.0f64;
    for (vi, &r) in val.iter().enumerate() {
        let reach = &ds.reach[r * n_cells..(r + 1) * n_cells];
        let truth = &ds.bytes_log[r * n_cells..(r + 1) * n_cells];
        let pick = argmin_masked(&pred[vi * n_cells..(vi + 1) * n_cells], reach);
        let best = argmin_masked(truth, reach);
        if let (Some(pk), Some(bk)) = (pick, best) {
            scored += 1;
            if pk == bk {
                hits += 1;
            }
            overhead_sum += (truth[pk] - truth[bk]).exp() - 1.0;
        }
    }
    let acc = if scored > 0 {
        hits as f64 / scored as f64
    } else {
        f64::NAN
    };
    let ov = if scored > 0 {
        overhead_sum / scored as f64
    } else {
        f64::NAN
    };
    (acc, ov)
}

/// Index of the smallest `vals[c]` over cells where `reach[c]`. `None` if no
/// cell is reachable. NaN predictions never win (NaN comparisons are false).
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

/// A per-cell GBDT regressor, or a constant fallback for cells with too few
/// reaching train rows to learn from.
enum GbdtCell {
    Model(GBDT),
    Const(f64),
}

/// Fit one GBDT (or constant fallback) per cell on its reaching train rows.
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

/// Predict the `valrows.len() × n_cells` bytes_log matrix from retained cells.
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

/// Deterministic Fisher–Yates permutation (SplitMix64-seeded).
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

/// Per-feature permutation importance: shuffle each input column across the
/// held-out rows, measure the drop in GBDT argmin accuracy. Sorted desc.
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
        let (acc, _) = score_predmatrix(&pred, n_cells, ds, val);
        out.push((f, base_acc - acc));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// Correlation redundancy groups: union-find features whose |Pearson| ≥ `thr`
/// over the train rows. `x_std` is per-column standardized over train, so the
/// correlation is the mean product.
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

/// Grouped permutation importance: shuffle every column in a redundancy group
/// with the SAME permutation (preserving within-group correlation, breaking
/// correlation with the target), measure the argmin-acc drop. Sorted desc.
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
        let (acc, _) = score_predmatrix(&pred, n_cells, ds, val);
        out.push((grp.clone(), base_acc - acc));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}
