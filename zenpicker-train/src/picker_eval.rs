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

use zenstats::{
    PanelStats, kendall_tau, outlier_ratio, pearson, rescale_logistic, spearman, z_rmse,
};

use crate::mlp::Mlp;
use crate::pareto_dataset::PickerDataset;

/// Number of Sensory-Threshold sample points on the PWRC SA-ST curve.
/// Must match `zenstats::pwrc_sa_st_auc`'s hard-coded `n_points = 128`
/// so [`compute_panel_lowmem`] reproduces the canonical PWRC bit-for-bit.
const PWRC_N_POINTS: usize = 128;

/// Memory-bounded re-implementation of [`zenstats::compute_panel`].
///
/// `zenstats::compute_panel` is bit-for-bit reproduced here EXCEPT for
/// the PWRC term: the canonical [`zenstats::pwrc_sa_st_auc`] →
/// `sa_st_curve` materialises a `Vec<(f64, bool)>` of **all `n·(n−1)/2`
/// pairs** (`Vec::with_capacity(n*n/2)`), which is O(n²) MEMORY. The
/// picker evaluator feeds the panel one entry per `(held-out row,
/// reachable cell)` — `n ≈ val_rows × n_cells` (≈ 59k for the dense
/// zenjpeg sweep). At that scale the all-pairs vector is
/// `59_788² / 2 × 16 B ≈ 27 GB`, which OOM-kills the trainer (observed
/// VmHWM 23 GB on a single `--hidden 128,128` distilled fit; the
/// 6-candidate grid hit the same ceiling faster).
///
/// [`pwrc_sa_st_auc_lowmem`] computes the **identical** SA-ST AUC with
/// O(`PWRC_N_POINTS`) memory and the same O(n²) time, so every reported
/// stat (SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE) is unchanged. All
/// other terms delegate to the same public `zenstats` functions
/// `compute_panel` itself calls, in the same order on the same inputs.
///
/// Shared with the ridge held-out path ([`crate::eval`]), which has the
/// same O(n²)-memory exposure whenever the held-out split is large.
pub(crate) fn compute_panel_lowmem(scores: &[f64], humans: &[f64]) -> PanelStats {
    let n = scores.len().min(humans.len());
    if n == 0 {
        return PanelStats::default();
    }
    // Identical to zenstats::compute_panel (panel.rs): SROCC/KROCC on raw,
    // PLCC/OR/PWRC/Z-RMSE on the 4-param-logistic-rescaled scores.
    let srocc = spearman(humans, scores).abs();
    let krocc = kendall_tau(humans, scores).abs();
    let rescaled = rescale_logistic(scores, humans);
    let plcc = pearson(&rescaled, humans).abs();
    let or_ratio = outlier_ratio(&rescaled, humans);
    let pwrc = pwrc_sa_st_auc_lowmem(&rescaled, humans);
    let z_rmse = z_rmse(&rescaled, humans);
    PanelStats {
        srocc,
        plcc,
        krocc,
        or_ratio,
        pwrc,
        z_rmse,
        n,
    }
}

/// O(`PWRC_N_POINTS`)-memory equivalent of `zenstats::pwrc_sa_st_auc`.
///
/// The canonical implementation builds `pairs: Vec<(gap, correct)>` over
/// every `(i, j)` pair, then sweeps `PWRC_N_POINTS` uniformly-spaced
/// Sensory Thresholds counting, per threshold, how many pairs have
/// `gap > ST` (and how many of those are concordant). The AUC of the
/// resulting (ST, accuracy) curve, normalised by `ST_max`, is the PWRC.
///
/// We compute the EXACT same curve without the all-pairs vector: a
/// difference array over the `PWRC_N_POINTS` thresholds. For each pair,
/// `kmax` = the number of leading thresholds strictly below `gap` (i.e.
/// the thresholds for which `gap > ST_k`), found with the SAME float
/// comparison the reference uses (`(k/(np−1))·ST_max < gap`), so the
/// active/correct counts per threshold — and hence the AUC — are
/// bit-identical. Time is still O(n²) (the `(i, j)` double loop); memory
/// drops from O(n²) to O(`PWRC_N_POINTS`).
fn pwrc_sa_st_auc_lowmem(scores: &[f64], humans: &[f64]) -> f64 {
    let n = scores.len().min(humans.len());
    if n < 2 {
        return 0.0;
    }
    let np = PWRC_N_POINTS;

    // Pass 1: ST_max = max |Δhuman| over pairs with a defined direction
    // (both Δhuman and Δscore finite and non-zero) — identical filter to
    // sa_st_curve. Also detect whether any such pair exists.
    let mut st_max = 0.0_f64;
    let mut any_pair = false;
    for i in 0..n {
        let (hi, si) = (humans[i], scores[i]);
        for j in (i + 1)..n {
            let dh = humans[j] - hi;
            let ds = scores[j] - si;
            if !dh.is_finite() || !ds.is_finite() || dh == 0.0 || ds == 0.0 {
                continue;
            }
            any_pair = true;
            let g = dh.abs();
            if g > st_max {
                st_max = g;
            }
        }
    }
    if !any_pair || st_max <= 0.0 {
        return 0.0;
    }

    // Pass 2: difference arrays over the np thresholds.
    // active[k] = #{pairs : gap > ST_k}; correct[k] = #{concordant pairs : gap > ST_k}.
    let mut active_diff = vec![0i64; np + 1];
    let mut correct_diff = vec![0i64; np + 1];
    for i in 0..n {
        let (hi, si) = (humans[i], scores[i]);
        for j in (i + 1)..n {
            let dh = humans[j] - hi;
            let ds = scores[j] - si;
            if !dh.is_finite() || !ds.is_finite() || dh == 0.0 || ds == 0.0 {
                continue;
            }
            let gap = dh.abs();
            let correct = dh.signum() == ds.signum();
            // kmax = count of thresholds k with ST_k < gap, where
            // ST_k = (k / (np-1)) * st_max — i.e. the set the reference's
            // `gap > st` test marks active. ST_k is monotone increasing
            // in k, so this is a prefix; binary-search the boundary with
            // the identical float comparison.
            let kmax = partition_point_st(np, st_max, gap);
            if kmax > 0 {
                active_diff[0] += 1;
                active_diff[kmax] -= 1;
                if correct {
                    correct_diff[0] += 1;
                    correct_diff[kmax] -= 1;
                }
            }
        }
    }

    // Prefix-sum to per-threshold counts, rebuild the (ST, accuracy)
    // curve, and trapezoid-integrate exactly as pwrc_sa_st_auc does.
    let mut auc = 0.0_f64;
    let mut active = 0i64;
    let mut correct = 0i64;
    let mut prev_st = 0.0_f64;
    let mut prev_sa = 0.0_f64;
    let mut last_finite_sa = 0.0_f64;
    for k in 0..np {
        active += active_diff[k];
        correct += correct_diff[k];
        let st = (k as f64 / (np - 1) as f64) * st_max;
        let sa = if active == 0 {
            // Matches sa_st_curve's "propagate last finite value" tail.
            last_finite_sa
        } else {
            let s = correct as f64 / active as f64;
            last_finite_sa = s;
            s
        };
        if k > 0 {
            let dt = st - prev_st;
            if dt > 0.0 {
                auc += 0.5 * (prev_sa + sa) * dt;
            }
        }
        prev_st = st;
        prev_sa = sa;
    }
    if st_max <= 0.0 { 0.0 } else { auc / st_max }
}

/// Number of thresholds `ST_k = (k/(np−1))·st_max` (for `k` in `0..np`)
/// that are strictly less than `gap` — i.e. the count the reference's
/// `gap > ST_k` test marks active. `ST_k` is monotone non-decreasing in
/// `k`, so the satisfying set is a prefix `0..kmax`; we binary-search the
/// boundary using the SAME float expression as the reference curve.
fn partition_point_st(np: usize, st_max: f64, gap: f64) -> usize {
    // Invariant: ST_k < gap for all k < lo; ST_k >= gap for all k >= hi.
    let mut lo = 0usize;
    let mut hi = np;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let st_mid = (mid as f64 / (np - 1) as f64) * st_max;
        if st_mid < gap {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

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
        // Hybrid-heads models append scalar blocks after the bytes_log
        // block; the picker argmin only reads `pred[0..n_cells]`.
        debug_assert!(
            pred.len() >= n_cells,
            "model output must contain at least the bytes_log block"
        );

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

    let bytes_panel = compute_panel_lowmem(&pred_pairs, &true_pairs);
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

/// Evaluate an EXTERNAL ZNPR v3 bake over the **deployed runtime path**
/// on `val_rows` of `ds`.
///
/// Unlike [`evaluate_picker`] (which scores the in-memory `Mlp` on
/// pre-standardized features), this loads the bake via zenpredict and
/// runs `predict_transformed` on the RAW (un-shaped, un-standardized)
/// feature rows — so it exercises `feature_transforms` → standardize →
/// the (possibly quantized) forward exactly as production would. Use it
/// to measure quantization / feature-bounds / output-spec variants
/// against the identical held-out split, and to confirm a quantized
/// bake preserves the argmin decision.
pub fn evaluate_picker_bake(
    bake: &[u8],
    ds: &PickerDataset,
    val_rows: &[usize],
) -> Result<Option<PickerEval>, String> {
    if val_rows.is_empty() {
        return Ok(None);
    }
    let model =
        zenpredict::Model::from_bytes(bake).map_err(|e| format!("Model::from_bytes: {e}"))?;
    let mut predictor = zenpredict::Predictor::new(&model);
    let use_tf = model.has_nontrivial_feature_transforms();
    let n_cells = ds.n_cells;
    let n_in = ds.n_in;

    let mut pred_pairs: Vec<f64> = Vec::new();
    let mut true_pairs: Vec<f64> = Vec::new();
    let mut overheads: Vec<f64> = Vec::new();
    let mut correct = 0usize;
    let mut scored_rows = 0usize;
    let mut row_f32 = vec![0.0f32; n_in];

    for &r in val_rows {
        let raw = &ds.features[r * n_in..(r + 1) * n_in];
        for (k, &v) in raw.iter().enumerate() {
            row_f32[k] = v as f32;
        }
        let pred = if use_tf {
            predictor.predict_transformed(&row_f32)
        } else {
            predictor.predict(&row_f32)
        }
        .map_err(|e| format!("predict: {e}"))?;
        debug_assert_eq!(pred.len(), n_cells);

        let reach = &ds.reach[r * n_cells..(r + 1) * n_cells];
        let truth = &ds.bytes_log[r * n_cells..(r + 1) * n_cells];
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
            pred_pairs.push(pred[c] as f64);
            true_pairs.push(truth[c]);
            if truth[c] < best_true_v {
                best_true_v = truth[c];
                best_true_c = Some(c);
            }
            if (pred[c] as f64) < best_pred_v {
                best_pred_v = pred[c] as f64;
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
            let pick_bytes = truth[pc].exp();
            let best_bytes = best_true_v.exp();
            if best_bytes > 0.0 && pick_bytes.is_finite() {
                overheads.push(pick_bytes / best_bytes - 1.0);
            }
        }
    }

    if pred_pairs.len() < 2 {
        return Ok(None);
    }
    let bytes_panel = compute_panel_lowmem(&pred_pairs, &true_pairs);
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

    Ok(Some(PickerEval {
        bytes_panel,
        argmin_acc,
        overhead_mean,
        overhead_p50: pct(0.50),
        overhead_p90: pct(0.90),
        n_rows: scored_rows,
        n_pairs: pred_pairs.len(),
    }))
}

#[cfg(test)]
mod panel_parity_tests {
    //! Prove [`compute_panel_lowmem`] reproduces `zenstats::compute_panel`
    //! bit-for-bit. This is the memory fix's correctness gate: the
    //! reported held-out panel (SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE)
    //! MUST be unchanged — only the O(n²) → O(`PWRC_N_POINTS`) memory of
    //! the PWRC term changes. Equality is asserted on the EXACT f64 bits
    //! (`to_bits`) for every stat, on the same inputs the canonical path
    //! sees, with NaN-aware comparison.

    use super::*;
    use zenstats::{compute_panel, pwrc_sa_st_auc, rescale_logistic};

    /// Bit-equal incl. NaN (all-NaN bit-patterns treated equal — the
    /// panel only produces canonical NaN, but be explicit).
    fn bits_eq(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        a.to_bits() == b.to_bits()
    }

    fn assert_panel_eq(scores: &[f64], humans: &[f64], label: &str) {
        let want = compute_panel(scores, humans);
        let got = compute_panel_lowmem(scores, humans);
        assert!(
            bits_eq(want.srocc, got.srocc),
            "{label}: srocc {} != {}",
            want.srocc,
            got.srocc
        );
        assert!(
            bits_eq(want.plcc, got.plcc),
            "{label}: plcc {} != {}",
            want.plcc,
            got.plcc
        );
        assert!(
            bits_eq(want.krocc, got.krocc),
            "{label}: krocc {} != {}",
            want.krocc,
            got.krocc
        );
        assert!(
            bits_eq(want.or_ratio, got.or_ratio),
            "{label}: or_ratio {} != {}",
            want.or_ratio,
            got.or_ratio
        );
        assert!(
            bits_eq(want.pwrc, got.pwrc),
            "{label}: PWRC {} != {} (bits {:#x} vs {:#x})",
            want.pwrc,
            got.pwrc,
            want.pwrc.to_bits(),
            got.pwrc.to_bits()
        );
        assert!(
            bits_eq(want.z_rmse, got.z_rmse),
            "{label}: z_rmse {} != {}",
            want.z_rmse,
            got.z_rmse
        );
        assert_eq!(want.n, got.n, "{label}: n");
    }

    /// Deterministic LCG so the test is self-contained + reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// The PWRC SA-ST curve term alone, bit-identical, across shapes.
    #[test]
    fn pwrc_lowmem_matches_canonical_exactly() {
        for seed in [1u64, 7, 42, 1234, 99999] {
            for n in [4usize, 17, 63, 200, 501] {
                let mut rng = Lcg(seed);
                let humans: Vec<f64> = (0..n).map(|_| rng.next_f64() * 100.0).collect();
                // mix of correlated + noisy + anti-correlated regimes
                let scores: Vec<f64> = humans
                    .iter()
                    .map(|&h| 0.6 * h + 40.0 * rng.next_f64() - 0.1 * h)
                    .collect();
                // pwrc_sa_st_auc consumes the logistic-rescaled scores,
                // exactly as compute_panel does internally.
                let rescaled = rescale_logistic(&scores, &humans);
                let want = pwrc_sa_st_auc(&rescaled, &humans);
                let got = pwrc_sa_st_auc_lowmem(&rescaled, &humans);
                assert!(
                    bits_eq(want, got),
                    "seed={seed} n={n}: PWRC {want} != {got} (bits {:#x} vs {:#x})",
                    want.to_bits(),
                    got.to_bits()
                );
            }
        }
    }

    /// Full panel bit-parity across random regimes.
    #[test]
    fn full_panel_matches_canonical() {
        for seed in [3u64, 11, 555, 8675309] {
            for n in [5usize, 23, 128, 400] {
                let mut rng = Lcg(seed);
                let humans: Vec<f64> = (0..n).map(|_| rng.next_f64() * 90.0 + 5.0).collect();
                let scores: Vec<f64> = humans.iter().map(|&h| h + 25.0 * rng.next_f64()).collect();
                assert_panel_eq(&scores, &humans, &format!("rand seed={seed} n={n}"));
            }
        }
    }

    /// Edge cases: ties, duplicate gaps, anti-correlation, near-constant.
    #[test]
    fn full_panel_edge_cases() {
        // Many tied human values -> many zero-gap pairs dropped.
        let humans = vec![1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0];
        let scores = vec![1.1, 0.9, 1.0, 2.2, 1.8, 3.5, 2.9, 3.1, 3.9, 5.2];
        assert_panel_eq(&scores, &humans, "ties");

        // Perfect anti-correlation.
        let humans2: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let scores2: Vec<f64> = (0..40).map(|i| (40 - i) as f64).collect();
        assert_panel_eq(&scores2, &humans2, "anti-correlation");

        // Duplicate, integer gaps (exercises the threshold-boundary
        // partition_point against exact ST_k == gap cases).
        let humans3: Vec<f64> = (0..30).map(|i| (i % 5) as f64 * 10.0).collect();
        let scores3: Vec<f64> = (0..30).map(|i| (i % 5) as f64 * 10.0 + 1.0).collect();
        assert_panel_eq(&scores3, &humans3, "integer-gaps");
    }

    /// The picker-eval scale that triggered the 27 GB OOM: ~hundreds of
    /// "rows" × tens of "cells" worth of pairs. We use a few thousand
    /// here (O(n²) canonical still cheap at n=3000: ~4.5M pairs, ~70 MB)
    /// to confirm parity holds at scale without re-OOMing the test.
    #[test]
    fn full_panel_matches_canonical_at_scale() {
        let n = 3000usize;
        let mut rng = Lcg(20260601);
        let humans: Vec<f64> = (0..n).map(|_| rng.next_f64() * 100.0).collect();
        let scores: Vec<f64> = humans
            .iter()
            .map(|&h| 0.7 * h + 30.0 * rng.next_f64())
            .collect();
        assert_panel_eq(&scores, &humans, "scale n=3000");
    }
}
