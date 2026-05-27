//! Bounded hyperparameter search over the picker MLP.
//!
//! zentrain's `train_hybrid.py` exposes `--hidden`, `--activation`,
//! `--seed`, plus learning-rate / batch / iter defaults; experiments
//! sweep `--hidden` (e.g. 128,128 vs 256,256) and seeds. We port a
//! BOUNDED grid over the two knobs that matter most for this corpus
//! size — hidden topology and learning rate — across a small seed set,
//! ranked by the held-out bytes-log SROCC (rank-honest, no q-leakage).
//!
//! The grid is intentionally small (default 6 candidates) so the search
//! is bounded and finishes in seconds-to-minutes on CPU — not an
//! unbounded optimizer. `cmaes` is NOT a zenanalyze workspace
//! dependency (despite the spec note), so we use the grid the spec
//! offers as the alternative.

use crate::mlp::{Mlp, MlpConfig, train_mlp};
use crate::pareto_dataset::PickerDataset;
use crate::picker_eval::{PickerEval, evaluate_picker};

/// One grid point.
#[derive(Debug, Clone)]
pub struct GridPoint {
    pub hidden: Vec<usize>,
    pub lr: f64,
    pub seed: u64,
}

/// The default bounded grid: {[64,64], [128,128]} × {1e-3, 2e-3} ×
/// {seed 0, seed 1} → but we cap at a small set to stay bounded. We
/// pick a 6-point grid: two topologies × two lrs × (seed varied only
/// on the cheapest topology) — keeping the total count low.
pub fn default_grid() -> Vec<GridPoint> {
    vec![
        GridPoint {
            hidden: vec![64, 64],
            lr: 2e-3,
            seed: 0,
        },
        GridPoint {
            hidden: vec![64, 64],
            lr: 1e-3,
            seed: 0,
        },
        GridPoint {
            hidden: vec![128, 128],
            lr: 2e-3,
            seed: 0,
        },
        GridPoint {
            hidden: vec![128, 128],
            lr: 1e-3,
            seed: 0,
        },
        GridPoint {
            hidden: vec![128, 128],
            lr: 2e-3,
            seed: 1,
        },
        GridPoint {
            hidden: vec![256, 256],
            lr: 2e-3,
            seed: 0,
        },
    ]
}

/// One row of the search trail: `(grid point, held-out bytes-log SROCC,
/// held-out argmin accuracy)`.
pub type TrailRow = (GridPoint, f64, f64);

/// Result of the search: the winning model + its eval + the per-point
/// ranking trail.
pub struct SearchResult {
    pub best_model: Mlp,
    pub best_cfg: MlpConfig,
    pub best_eval: PickerEval,
    pub trail: Vec<TrailRow>,
    pub selected_index: usize,
}

/// Run the bounded grid search. Trains one MLP per grid point on
/// `x_std[train_rows]`, evaluates on `x_std[val_rows]`, and selects the
/// candidate with the highest held-out **argmin accuracy** (the picker
/// decision-quality metric zentrain gates on via `min_argmin_acc`),
/// ties broken by the bytes-log SROCC. `base` supplies the non-swept
/// hyperparameters (batch, max_iter, …).
pub fn run_search(
    ds: &PickerDataset,
    x_std: &[f64],
    train_rows: &[usize],
    val_rows: &[usize],
    grid: &[GridPoint],
    base: &MlpConfig,
    mut log: impl FnMut(&str),
) -> Option<SearchResult> {
    if grid.is_empty() {
        return None;
    }
    let n_in = ds.n_in;
    let n_out = ds.n_cells;

    // Pack the training rows into a contiguous matrix so the MLP trainer
    // sees rows 0..n_train. (The MLP's internal val split keys off this.)
    let (x_tr, y_tr) = pack_rows(ds, x_std, train_rows);
    let n_train = train_rows.len();

    let mut best: Option<(usize, Mlp, MlpConfig, PickerEval, f64)> = None;
    let mut trail: Vec<TrailRow> = Vec::with_capacity(grid.len());

    for (gi, gp) in grid.iter().enumerate() {
        let cfg = MlpConfig {
            hidden: gp.hidden.clone(),
            lr: gp.lr,
            seed: gp.seed,
            ..base.clone()
        };
        let model = train_mlp(&x_tr, &y_tr, n_train, n_in, n_out, &cfg);
        let eval = evaluate_picker(&model, ds, x_std, val_rows);
        let (srocc, argmin) = match &eval {
            Some(e) => (e.bytes_panel.srocc, e.argmin_acc),
            None => (f64::NAN, f64::NAN),
        };
        log(&format!(
            "[search] cand {gi}: hidden={:?} lr={} seed={} -> heldout bytes-SROCC={:.4} argmin_acc={:.4} (n_iter={})",
            gp.hidden, gp.lr, gp.seed, srocc, argmin, model.n_iter
        ));
        trail.push((gp.clone(), srocc, argmin));

        if let Some(e) = eval {
            // Rank by argmin accuracy (zentrain's picker gate), ties by
            // the bytes-log SROCC.
            let key = if e.argmin_acc.is_finite() {
                e.argmin_acc
            } else {
                f64::NEG_INFINITY
            };
            let better = match &best {
                None => true,
                Some((_, _, _, be, bkey)) => {
                    key > *bkey || (key == *bkey && e.bytes_panel.srocc > be.bytes_panel.srocc)
                }
            };
            if better {
                best = Some((gi, model, cfg, e, key));
            }
        }
    }

    best.map(|(idx, model, cfg, eval, _)| SearchResult {
        best_model: model,
        best_cfg: cfg,
        best_eval: eval,
        trail,
        selected_index: idx,
    })
}

/// Pack the given rows of the standardized feature matrix + the target
/// matrix into contiguous `0..rows.len()` arrays. Returns
/// `(x_packed[rows*n_in], y_packed[rows*n_cells])`.
fn pack_rows(ds: &PickerDataset, x_std: &[f64], rows: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let n_in = ds.n_in;
    let n_cells = ds.n_cells;
    let mut x = Vec::with_capacity(rows.len() * n_in);
    let mut y = Vec::with_capacity(rows.len() * n_cells);
    for &r in rows {
        x.extend_from_slice(&x_std[r * n_in..(r + 1) * n_in]);
        y.extend_from_slice(&ds.bytes_log[r * n_cells..(r + 1) * n_cells]);
    }
    (x, y)
}
