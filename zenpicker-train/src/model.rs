//! Minimal baseline model: ridge-regularized linear regression with a
//! per-feature standardizer.
//!
//! This is the END-TO-END-pipeline placeholder, NOT the mature picker
//! model. It is deterministic (closed-form normal equations, no
//! stochastic optimizer to tune), always produces finite weights for
//! finite inputs (the `λI` ridge term guarantees the Gram matrix is
//! positive-definite), and maps cleanly onto a single
//! identity-activation ZNPR layer. The non-linear MLP + hyperparameter
//! search is a documented follow-on.

use crate::TrainError;

/// Per-feature standardizer (z-score). Folded into the bake's
/// `scaler_mean` / `scaler_scale` so the runtime applies it for free.
#[derive(Debug, Clone)]
pub struct Standardizer {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

impl Standardizer {
    /// Fit from a row-major `n_rows × n_features` matrix over the
    /// given row indices. A zero/near-zero std is clamped to 1.0 so
    /// the column passes through unscaled (constant features become
    /// all-zero post-standardize, harmless under ridge).
    pub fn fit(features: &[f32], n_features: usize, rows: &[usize]) -> Self {
        let n = rows.len().max(1) as f64;
        let mut mean = vec![0.0f64; n_features];
        for &r in rows {
            let base = r * n_features;
            for j in 0..n_features {
                mean[j] += features[base + j] as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0f64; n_features];
        for &r in rows {
            let base = r * n_features;
            for j in 0..n_features {
                let d = features[base + j] as f64 - mean[j];
                var[j] += d * d;
            }
        }
        let mut scale = vec![1.0f32; n_features];
        for j in 0..n_features {
            let std = (var[j] / n).sqrt();
            scale[j] = if std > 1e-9 { std as f32 } else { 1.0 };
        }
        Self {
            mean: mean.iter().map(|&m| m as f32).collect(),
            scale,
        }
    }

    /// Standardize one row in place into `out` (length n_features).
    fn apply_row(&self, row: &[f32], out: &mut [f64]) {
        for j in 0..row.len() {
            out[j] = (row[j] - self.mean[j]) as f64 / self.scale[j] as f64;
        }
    }
}

/// A fitted ridge model: weights over standardized features plus an
/// intercept. `predict` operates on RAW (un-standardized) features by
/// applying the embedded standardizer first — same contract the baked
/// model honors via its scaler.
#[derive(Debug, Clone)]
pub struct RidgeModel {
    pub standardizer: Standardizer,
    /// One weight per standardized feature.
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub lambda: f64,
}

impl RidgeModel {
    pub fn n_features(&self) -> usize {
        self.weights.len()
    }

    /// Predict from a raw feature row (length n_features).
    pub fn predict_raw(&self, row: &[f32]) -> f64 {
        let mut z = vec![0.0f64; row.len()];
        self.standardizer.apply_row(row, &mut z);
        let mut acc = self.intercept;
        for (w, x) in self.weights.iter().zip(z.iter()) {
            acc += w * x;
        }
        acc
    }
}

/// Solve `A x = b` for a symmetric positive-definite `A` (n×n,
/// row-major) by Gaussian elimination with partial pivoting. Returns
/// `None` if the matrix is singular to working precision.
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            b.swap(col, piv);
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }
    // Back-substitution.
    let mut x = vec![0.0f64; n];
    for col in (0..n).rev() {
        let mut acc = b[col];
        for c in (col + 1)..n {
            acc -= a[col * n + c] * x[c];
        }
        x[col] = acc / a[col * n + col];
    }
    Some(x)
}

/// Fit a ridge regression on standardized features over `train_rows`.
///
/// Closed form: `w = (ZᵀZ + λI)⁻¹ Zᵀ(y - ȳ)`, intercept = `ȳ`
/// (targets are centered; features are already zero-mean post-
/// standardize, so the intercept is just the target mean). `lambda`
/// is the L2 penalty; a small positive value keeps the Gram matrix
/// well-conditioned even when `n_features ≥ n_rows`.
pub fn train_ridge(
    features: &[f32],
    targets: &[f64],
    n_features: usize,
    train_rows: &[usize],
    lambda: f64,
) -> Result<RidgeModel, TrainError> {
    if train_rows.is_empty() {
        return Err(TrainError::Degenerate("no training rows".into()));
    }
    let standardizer = Standardizer::fit(features, n_features, train_rows);

    // Target mean (intercept).
    let y_mean = train_rows.iter().map(|&r| targets[r]).sum::<f64>() / train_rows.len() as f64;

    // Accumulate Gram matrix ZᵀZ (+ λI) and Zᵀ(y - ȳ).
    let p = n_features;
    let mut gram = vec![0.0f64; p * p];
    let mut zty = vec![0.0f64; p];
    let mut z = vec![0.0f64; p];
    for &r in train_rows {
        let base = r * p;
        standardizer.apply_row(&features[base..base + p], &mut z);
        let yc = targets[r] - y_mean;
        for i in 0..p {
            zty[i] += z[i] * yc;
            let gi = i * p;
            for j in i..p {
                gram[gi + j] += z[i] * z[j];
            }
        }
    }
    // Symmetrize + ridge.
    for i in 0..p {
        for j in (i + 1)..p {
            let v = gram[i * p + j];
            gram[j * p + i] = v;
        }
        gram[i * p + i] += lambda;
    }

    let weights = solve_linear(gram, zty, p).ok_or_else(|| {
        TrainError::Degenerate("Gram matrix singular even with ridge term".into())
    })?;

    if weights.iter().any(|w| !w.is_finite()) || !y_mean.is_finite() {
        return Err(TrainError::Degenerate("non-finite solution".into()));
    }

    Ok(RidgeModel {
        standardizer,
        weights,
        intercept: y_mean,
        lambda,
    })
}
