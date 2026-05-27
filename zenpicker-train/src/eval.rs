//! Held-out evaluation via the canonical [`zenstats`] panel.
//!
//! We score the trained model on the held-out rows and run the full
//! Mohammadi 2025 panel (SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE)
//! through `zenstats::compute_panel` — the same stat path the metric
//! work uses. No hand-rolled SROCC here (per the "don't hand-roll
//! srocc/plcc" rule in zensim/CLAUDE.md).

use zenstats::{PanelStats, compute_panel};

use crate::model::RidgeModel;
use crate::parquet_input::TrainingData;

/// Held-out report: the panel plus the row count actually scored.
#[derive(Debug, Clone, Copy)]
pub struct EvalReport {
    pub panel: PanelStats,
}

/// Score `model` on `val_rows` of `data` and compute the panel
/// against the true targets. Returns `None` if there are no held-out
/// rows.
pub fn evaluate(model: &RidgeModel, data: &TrainingData, val_rows: &[usize]) -> Option<EvalReport> {
    if val_rows.is_empty() {
        return None;
    }
    let p = data.n_features;
    let mut preds: Vec<f64> = Vec::with_capacity(val_rows.len());
    let mut truth: Vec<f64> = Vec::with_capacity(val_rows.len());
    for &r in val_rows {
        let base = r * p;
        let row = &data.features[base..base + p];
        preds.push(model.predict_raw(row));
        truth.push(data.targets[r]);
    }
    let panel = compute_panel(&preds, &truth);
    Some(EvalReport { panel })
}
