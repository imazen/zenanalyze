//! Per-feature input shaping (zenpredict `feature_transforms`).
//!
//! Picker inputs are zenanalyze source features whose marginal
//! distributions are heavy-tailed (variance, edge density, laplacian
//! variance, …). A LeakyReLU MLP conditions far better on
//! near-Gaussian inputs, so we fit a per-feature monotone transform
//! that minimises residual skewness, apply it BEFORE the standardiser,
//! and emit `zentrain.feature_transforms` + `zentrain.feature_transform_params`
//! so the runtime's `predict_transformed` reproduces the exact same
//! shaping. The transform is applied through zenpredict's own
//! [`FeatureTransform::apply_with_params`], so train-time and
//! inference-time shaping are bit-identical by construction (no
//! re-implementation to drift).
//!
//! The dial input (`zq_norm`, the last column) is ALWAYS `Identity`:
//! it is already a normalised [0, 1] target and must pass through
//! untouched so the picker's quality axis stays linear.

use zenpredict::FeatureTransform;

/// How to choose each feature's transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapingMode {
    /// No shaping — every input is `Identity` (back-compat default).
    None,
    /// Per-feature: pick the candidate (Identity / SignedLog1p /
    /// SignedSqrt / SignedCbrt / WinsorP99 / YeoJohnson) that minimises
    /// |skewness| on the training distribution, keeping Identity unless
    /// a candidate beats it by a margin.
    Auto,
    /// YeoJohnson on every feature with a per-feature fitted lambda
    /// (the canonical Gaussianising power transform).
    Yeo,
}

impl ShapingMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" | "off" => Some(Self::None),
            "auto" => Some(Self::Auto),
            "yeo" | "yeo_johnson" | "yeojohnson" => Some(Self::Yeo),
            _ => None,
        }
    }
}

/// Per-input fitted transform + its parameters (parallel arrays of
/// length `n_in`).
#[derive(Clone, Debug)]
pub struct FittedTransforms {
    pub transforms: Vec<FeatureTransform>,
    pub params: Vec<Vec<f32>>,
}

impl FittedTransforms {
    pub fn identity(n_in: usize) -> Self {
        Self {
            transforms: vec![FeatureTransform::Identity; n_in],
            params: vec![Vec::new(); n_in],
        }
    }

    /// True when every transform is `Identity` (so the bake can skip
    /// emitting the metadata entirely).
    pub fn is_all_identity(&self) -> bool {
        self.transforms
            .iter()
            .all(|t| *t == FeatureTransform::Identity)
    }

    /// Newline-separated tokens for `zentrain.feature_transforms`.
    pub fn transforms_text(&self) -> String {
        self.transforms
            .iter()
            .map(|t| t.as_token())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Newline-separated, comma-joined params for
    /// `zentrain.feature_transform_params` (one line per input; empty
    /// line for no-param transforms). Uses a compact, round-trippable
    /// float format.
    pub fn params_text(&self) -> String {
        self.params
            .iter()
            .map(|p| {
                p.iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Count of non-identity inputs (for logging).
    pub fn n_shaped(&self) -> usize {
        self.transforms
            .iter()
            .filter(|t| **t != FeatureTransform::Identity)
            .count()
    }
}

/// Sample skewness (third standardised moment) of `xs`. Returns 0 for
/// near-constant inputs (std below a tiny epsilon).
fn skewness(xs: &[f32]) -> f64 {
    let n = xs.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in xs {
        let d = x as f64 - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    if m2 < 1e-20 {
        return 0.0;
    }
    m3 / m2.powf(1.5)
}

/// Linear-interpolated percentile (`q` in [0, 1]) of a sorted slice.
fn percentile_sorted(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = (pos - lo as f64) as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---- YeoJohnson lambda fit (mirrors zenpredict-bake/fit_yeo_johnson) ----

fn yj(x: f64, lambda: f64) -> f64 {
    if x >= 0.0 {
        if lambda.abs() < 1e-9 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < 1e-9 {
        -(-x + 1.0).ln()
    } else {
        -(((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
    }
}

/// Profile log-likelihood of the YeoJohnson transform at `lambda`
/// (constant terms dropped — only used to compare across lambdas).
fn yj_loglik(xs: &[f64], lambda: f64) -> f64 {
    let n = xs.len() as f64;
    let ys: Vec<f64> = xs.iter().map(|&x| yj(x, lambda)).collect();
    let mean = ys.iter().sum::<f64>() / n;
    let var = ys.iter().map(|&y| (y - mean) * (y - mean)).sum::<f64>() / n;
    if var < 1e-30 {
        return f64::NEG_INFINITY;
    }
    let sign_term: f64 = xs
        .iter()
        .map(|&x| (lambda - 1.0) * (x.signum() * (x.abs() + 1.0)).abs().ln())
        .sum();
    -0.5 * n * var.ln() + sign_term
}

/// Grid + golden-ish refine search for the lambda maximising the
/// profile log-likelihood (coarse grid then a local bisection refine).
fn fit_yj_lambda(xs: &[f64]) -> f32 {
    let mut best_l = 1.0f64;
    let mut best_ll = f64::NEG_INFINITY;
    // Coarse grid over the standard [-2, 3] YJ range.
    let steps = 50;
    for i in 0..=steps {
        let l = -2.0 + 5.0 * (i as f64 / steps as f64);
        let ll = yj_loglik(xs, l);
        if ll > best_ll {
            best_ll = ll;
            best_l = l;
        }
    }
    // Local refine around the grid winner.
    let mut lo = best_l - 0.1;
    let mut hi = best_l + 0.1;
    for _ in 0..40 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if yj_loglik(xs, m1) < yj_loglik(xs, m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    (0.5 * (lo + hi)) as f32
}

/// Transform a column's train values through a candidate, returning the
/// shaped values (for skewness scoring).
fn shaped_values(xs: &[f32], t: FeatureTransform, params: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| t.apply_with_params(x, params)).collect()
}

/// Fit per-feature transforms on `train_rows`. `features` is row-major
/// `n_rows × n_in`. Input index `n_in - 1` (zq_norm) is forced Identity.
pub fn fit_transforms(
    features: &[f64],
    n_in: usize,
    train_rows: &[usize],
    mode: ShapingMode,
) -> FittedTransforms {
    if mode == ShapingMode::None || n_in == 0 {
        return FittedTransforms::identity(n_in);
    }
    let mut out = FittedTransforms::identity(n_in);
    // Skip the last column (zq_norm dial) — always Identity.
    let n_feat = n_in.saturating_sub(1);
    for j in 0..n_feat {
        let col: Vec<f32> = train_rows
            .iter()
            .map(|&r| features[r * n_in + j] as f32)
            .collect();
        if col.len() < 8 {
            continue;
        }
        let mut sorted = col.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p1 = percentile_sorted(&sorted, 0.01);
        let p99 = percentile_sorted(&sorted, 0.99);
        let xs_f64: Vec<f64> = col.iter().map(|&x| x as f64).collect();

        match mode {
            ShapingMode::None => {}
            ShapingMode::Yeo => {
                let lambda = fit_yj_lambda(&xs_f64);
                // Only adopt if it meaningfully reduces |skew|.
                let base_skew = skewness(&col).abs();
                let shaped = shaped_values(&col, FeatureTransform::YeoJohnson, &[lambda]);
                if skewness(&shaped).abs() < base_skew * 0.98 {
                    out.transforms[j] = FeatureTransform::YeoJohnson;
                    out.params[j] = vec![lambda];
                }
            }
            ShapingMode::Auto => {
                // Per-feature menu: each input INDEPENDENTLY picks the
                // magnitude-preserving transform that best Gaussianises
                // it. (QuantileBins is excluded here — rank-binning
                // trivially zeroes skew by uniformising, discarding the
                // magnitude information the MLP needs; offer it only via
                // an explicit mode.)
                let base_skew = skewness(&col).abs();
                let min_v = sorted[0];
                let yj_lambda = fit_yj_lambda(&xs_f64);
                let mut candidates: Vec<(FeatureTransform, Vec<f32>)> = vec![
                    (FeatureTransform::SignedLog1p, vec![]),
                    (FeatureTransform::SignedSqrt, vec![]),
                    (FeatureTransform::SignedCbrt, vec![]),
                    (FeatureTransform::YeoJohnson, vec![yj_lambda]),
                ];
                if p99 > p1 {
                    candidates.push((FeatureTransform::WinsorP99, vec![p1, p99]));
                }
                if min_v > -1.0 {
                    candidates.push((FeatureTransform::Log1p, vec![]));
                    if p99 > p1 && p1 > -1.0 {
                        candidates.push((FeatureTransform::WinsorThenLog1p, vec![p1, p99]));
                    }
                }
                let mut best_t = FeatureTransform::Identity;
                let mut best_params: Vec<f32> = Vec::new();
                let mut best_skew = base_skew;
                for (t, params) in candidates {
                    let shaped = shaped_values(&col, t, &params);
                    if shaped.iter().any(|v| !v.is_finite()) {
                        continue;
                    }
                    let s = skewness(&shaped).abs();
                    // Require a 5% |skew| improvement to beat the
                    // incumbent (avoid chasing sampling noise).
                    if s < best_skew * 0.95 {
                        best_skew = s;
                        best_t = t;
                        best_params = params;
                    }
                }
                if std::env::var("ZENPICKER_SHAPE_DEBUG").is_ok() && j < 24 {
                    eprintln!(
                        "[shape-dbg] feat {j}: base|skew|={base_skew:.3} -> {} (|skew|={best_skew:.3}) min={min_v:.4} p1={p1:.4} p99={p99:.4}",
                        best_t.as_token()
                    );
                }
                out.transforms[j] = best_t;
                out.params[j] = best_params;
            }
        }
    }
    out
}

/// Apply fitted transforms IN PLACE to every row of `features`
/// (`n_rows × n_in`, row-major), through zenpredict's own apply so
/// train-time shaping is identical to `predict_transformed`.
pub fn apply_inplace(features: &mut [f64], n_in: usize, fitted: &FittedTransforms) {
    if fitted.is_all_identity() || n_in == 0 {
        return;
    }
    let n_rows = features.len() / n_in;
    for r in 0..n_rows {
        let base = r * n_in;
        for j in 0..n_in {
            let t = fitted.transforms[j];
            if t == FeatureTransform::Identity {
                continue;
            }
            let x = features[base + j] as f32;
            features[base + j] = t.apply_with_params(x, &fitted.params[j]) as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mode_is_noop() {
        let feats = vec![1.0, 2.0, 3.0, 4.0];
        let f = fit_transforms(&feats, 2, &[0, 1], ShapingMode::None);
        assert!(f.is_all_identity());
        let mut g = feats.clone();
        apply_inplace(&mut g, 2, &f);
        assert_eq!(feats, g);
    }

    #[test]
    fn dial_column_stays_identity() {
        // 3 inputs: 2 features + dial. Heavy-tailed feature cols.
        let n_in = 3;
        let rows = 64usize;
        let mut feats = vec![0.0f64; rows * n_in];
        for r in 0..rows {
            feats[r * n_in] = ((r * r) as f64).exp().min(1e6); // very skewed
            feats[r * n_in + 1] = (r as f64) * 0.5;
            feats[r * n_in + 2] = r as f64 / rows as f64; // dial
        }
        let train: Vec<usize> = (0..rows).collect();
        let f = fit_transforms(&feats, n_in, &train, ShapingMode::Auto);
        assert_eq!(
            f.transforms[n_in - 1],
            FeatureTransform::Identity,
            "dial must stay identity"
        );
    }

    #[test]
    fn auto_reduces_skew_on_heavy_tail() {
        let n_in = 2;
        let rows = 200usize;
        let mut feats = vec![0.0f64; rows * n_in];
        for r in 0..rows {
            // Exponentially distributed -> strong positive skew.
            feats[r * n_in] = ((r as f64) / 20.0).exp();
            feats[r * n_in + 1] = r as f64 / rows as f64;
        }
        let train: Vec<usize> = (0..rows).collect();
        let f = fit_transforms(&feats, n_in, &train, ShapingMode::Auto);
        assert_ne!(
            f.transforms[0],
            FeatureTransform::Identity,
            "heavy-tailed feature should get shaped"
        );
        // Token + params round-trip cleanly. Use split('\n') (NOT
        // .lines(), which collapses trailing empty lines) to match the
        // runtime parser `parse_feature_transform_params`, which counts
        // newline-separated rows including empties.
        assert_eq!(f.transforms_text().split('\n').count(), n_in);
        assert_eq!(f.params_text().split('\n').count(), n_in);
    }
}
