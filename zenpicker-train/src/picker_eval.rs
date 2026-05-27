//! Held-out evaluation for the within-cell-optimal picker.
//!
//! Two complementary views, both on the grouped held-out images:
//!
//! 1. **`bytes_log` rank panel** — over every `(row, reachable-cell)`
//!    pair, the full [`zenstats`] Mohammadi-2025 panel of predicted
//!    vs actual `bytes_log`. This is the rank-honest "does the picker
//!    order config costs correctly?" measure, and it is what we report
//!    side-by-side with zentrain (zentrain's diagnostics report per-cell
//!    R² + argmin accuracy; the panel is the IQA-rigor superset).
//!
//! 2. **argmin accuracy + byte overhead** — zentrain's headline picker
//!    metric. For each held-out row, `pick = argmin(pred_bytes_log,
//!    mask = reach)`; `actual = argmin(true_bytes_log, mask = reach)`.
//!    Accuracy = fraction where `pick == actual`; overhead =
//!    `bytes(pick) / bytes(actual) - 1` (mean / p50 / p90).
//!
//! Critically, NONE of this uses the codec's chosen `q` as an input —
//! the model's only inputs are image features + the requested
//! `zq_norm`. The q-leakage that inflated the skeleton to SROCC 0.9988
//! is structurally impossible here.

use zenstats::{PanelStats, compute_panel};

use crate::mlp::Mlp;
use crate::pareto_dataset::PickerDataset;

/// Held-out picker report.
#[derive(Debug, Clone)]
pub struct PickerEval {
    /// Panel of predicted-vs-actual `bytes_log` over reachable cells.
    pub bytes_panel: PanelStats,
    /// Fraction of held-out rows whose argmin pick equals the true
    /// min-bytes reachable cell.
    pub argmin_acc: f64,
    /// Mean byte overhead of the pick vs the true best (e.g. 0.04 = 4%).
    pub overhead_mean: f64,
    pub overhead_p50: f64,
    pub overhead_p90: f64,
    /// Rows scored.
    pub n_rows: usize,
    /// (row, reachable-cell) pairs scored for the panel.
    pub n_pairs: usize,
}

/// Evaluate `model` (predicts standardized features → `n_cells`
/// `bytes_log`) on `val_rows` of `ds`. `x_std` is the standardized
/// feature matrix (row-major `n × n_in`).
pub fn evaluate_picker(
    model: &Mlp,
    ds: &PickerDataset,
    x_std: &[f64],
    val_rows: &[usize],
) -> Option<PickerEval> {
    if val_rows.is_empty() {
        return None;
    }
    let n_cells = ds.n_cells;
    let n_in = ds.n_in;

    let mut pred_pairs: Vec<f64> = Vec::new();
    let mut true_pairs: Vec<f64> = Vec::new();
    let mut overheads: Vec<f64> = Vec::new();
    let mut correct = 0usize;
    let mut scored_rows = 0usize;

    for &r in val_rows {
        let x = &x_std[r * n_in..(r + 1) * n_in];
        let pred = model.predict(x);
        debug_assert_eq!(pred.len(), n_cells);

        // Collect reachable cells for this row.
        let reach = &ds.reach[r * n_cells..(r + 1) * n_cells];
        let truth = &ds.bytes_log[r * n_cells..(r + 1) * n_cells];

        // Panel pairs over reachable cells.
        let mut any = false;
        let mut best_true_c: Option<usize> = None;
        let mut best_true_v = f64::INFINITY;
        let mut best_pred_c: Option<usize> = None;
        let mut best_pred_v = f64::INFINITY;
        for c in 0..n_cells {
            if !reach[c] {
                continue;
            }
            any = true;
            pred_pairs.push(pred[c]);
            true_pairs.push(truth[c]);
            if truth[c] < best_true_v {
                best_true_v = truth[c];
                best_true_c = Some(c);
            }
            // Masked argmin over predicted bytes (the runtime pick).
            if pred[c] < best_pred_v {
                best_pred_v = pred[c];
                best_pred_c = Some(c);
            }
        }
        if !any {
            continue;
        }
        scored_rows += 1;
        if let (Some(tc), Some(pc)) = (best_true_c, best_pred_c) {
            if pc == tc {
                correct += 1;
            }
            // Byte overhead: exp(bytes_log) ratio.
            let pick_bytes = truth[pc].exp();
            let best_bytes = best_true_v.exp();
            if best_bytes > 0.0 && pick_bytes.is_finite() {
                overheads.push(pick_bytes / best_bytes - 1.0);
            }
        }
    }

    if pred_pairs.len() < 2 {
        return None;
    }

    let bytes_panel = compute_panel(&pred_pairs, &true_pairs);
    let argmin_acc = if scored_rows > 0 {
        correct as f64 / scored_rows as f64
    } else {
        0.0
    };
    overheads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> f64 {
        if overheads.is_empty() {
            return f64::NAN;
        }
        let idx = ((overheads.len() as f64 - 1.0) * p).round() as usize;
        overheads[idx.min(overheads.len() - 1)]
    };
    let overhead_mean = if overheads.is_empty() {
        f64::NAN
    } else {
        overheads.iter().sum::<f64>() / overheads.len() as f64
    };

    Some(PickerEval {
        bytes_panel,
        argmin_acc,
        overhead_mean,
        overhead_p50: pct(0.50),
        overhead_p90: pct(0.90),
        n_rows: scored_rows,
        n_pairs: pred_pairs.len(),
    })
}
