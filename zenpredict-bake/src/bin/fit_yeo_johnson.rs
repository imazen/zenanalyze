//! Fit a per-feature Yeo-Johnson shape parameter λ via MLE.
//!
//! Yeo-Johnson is the modern "auto-pick the right power transform"
//! default. Unlike Box-Cox it works on the full real line; unlike
//! `Pow` the shape parameter is fit from data instead of user-supplied.
//! See `zenpredict::FeatureTransform::YeoJohnson` for the runtime
//! math and the algorithmic reference (Yeo & Johnson 2000).
//!
//! ## MLE objective
//!
//! For a feature column `x[1..n]` and candidate `λ`:
//!
//! ```text
//!     LL(λ) = (λ − 1) · Σ_i sign(x_i) · ln(|x_i| + 1)
//!             − (n / 2) · ln( var(y) )
//! ```
//!
//! where `y_i = YeoJohnson(x_i, λ)` and `var(y)` is the biased
//! sample variance. The first term is the log-Jacobian, the second
//! is the Gaussian-fit log-likelihood after centering. Maximizing
//! `LL` over λ yields the value that makes the transformed
//! distribution as close to Gaussian as the YJ family allows.
//!
//! Search domain is λ ∈ [−2, 2] — the same default scipy uses for
//! `scipy.stats.yeojohnson`. Golden-section search converges in ~50
//! evaluations to ~1e-5 precision, more than enough for downstream
//! standardize-then-fit.
//!
//! ## Usage
//!
//! ```text
//! cargo build --release -p zenpredict-bake --features fit-yj --bin fit_yeo_johnson
//! ./target/release/fit_yeo_johnson <parquet> <feature_col_name>
//!     [--feature-idx N] [--max-rows N] [--print json|plain]
//! ```
//!
//! Outputs `λ = 0.234567` on stdout (plain) or
//! `{"lambda": 0.234567, "n": 196086, "ll": -1234567.89}` (json).
//!
//! ## Numerical robustness
//!
//! - Non-finite inputs are dropped.
//! - Constant columns (`var(x) < 1e-10`) print `λ = 1.0` (identity
//!   for YJ at λ=1) and emit a warning to stderr.
//! - Catastrophic underflow in `var(y)` falls back to `var(y) = ε`
//!   to avoid `-inf` log-likelihood.
//!
//! The fit prioritises being close-enough-to-scipy rather than
//! bit-equivalent. The shape parameter only needs to be good enough
//! that the post-YJ distribution is Gaussian-ish; the downstream
//! standardize step absorbs scale/location.

use std::fs::File;
use std::path::PathBuf;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{Float32Type, Float64Type};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// YJ math. Mirrors `zenpredict::FeatureTransform::YeoJohnson` exactly
/// — re-implemented here in f64 so MLE numerics survive heavy tails.
#[inline]
fn yj(x: f64, lambda: f64) -> f64 {
    const LAMBDA_EPS: f64 = 1.0e-9;
    if x >= 0.0 {
        if (lambda - 0.0).abs() < LAMBDA_EPS {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else if (lambda - 2.0).abs() < LAMBDA_EPS {
        -((-x + 1.0).ln())
    } else {
        let exp = 2.0 - lambda;
        -(((-x) + 1.0).powf(exp) - 1.0) / exp
    }
}

/// MLE log-likelihood of YJ-transformed data assuming Gaussian
/// post-transform. Higher is better.
fn yj_loglik(xs: &[f64], lambda: f64) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return f64::NEG_INFINITY;
    }

    // Log-Jacobian: Σ (λ - 1) * sign(x) * ln(|x| + 1)
    // = (λ - 1) * Σ sign(x_i) * ln(|x_i| + 1)
    let log_jac_sum: f64 = xs
        .iter()
        .map(|&x| {
            let s = if x >= 0.0 { 1.0 } else { -1.0 };
            s * (x.abs() + 1.0).ln()
        })
        .sum();
    let log_jac = (lambda - 1.0) * log_jac_sum;

    // Transform + center + var(y).
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut any_nonfinite = false;
    for &x in xs {
        let y = yj(x, lambda);
        if !y.is_finite() {
            any_nonfinite = true;
            break;
        }
        sum += y;
        sum_sq += y * y;
    }
    if any_nonfinite {
        return f64::NEG_INFINITY;
    }
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(1e-30);
    log_jac - 0.5 * n * var.ln()
}

/// Golden-section search for argmax. Returns `(lambda, ll)`.
fn golden_search(xs: &[f64], lo: f64, hi: f64, tol: f64) -> (f64, f64) {
    // Golden ratio.
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618

    let mut a = lo;
    let mut b = hi;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = yj_loglik(xs, c);
    let mut fd = yj_loglik(xs, d);

    let mut iters = 0;
    while (b - a).abs() > tol && iters < 200 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = yj_loglik(xs, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = yj_loglik(xs, d);
        }
        iters += 1;
    }
    let best = (a + b) / 2.0;
    let best_ll = yj_loglik(xs, best);
    (best, best_ll)
}

fn load_column(
    path: &std::path::Path,
    col_name: Option<&str>,
    col_idx: Option<usize>,
    max_rows: Option<usize>,
) -> Result<Vec<f64>, String> {
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| format!("parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let resolved_idx = if let Some(name) = col_name {
        schema
            .index_of(name)
            .map_err(|_| format!("column '{name}' not found in {}", path.display()))?
    } else if let Some(i) = col_idx {
        if i >= schema.fields().len() {
            return Err(format!(
                "--feature-idx {i} out of range (schema has {} columns)",
                schema.fields().len()
            ));
        }
        i
    } else {
        return Err("must pass <feature_col_name> or --feature-idx".into());
    };
    let field = schema.field(resolved_idx);
    let field_name = field.name().clone();
    let dtype = field.data_type().clone();
    let mask = ProjectionMask::leaves(builder.parquet_schema(), std::iter::once(resolved_idx));
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| format!("parquet build reader: {e}"))?;

    let mut out: Vec<f64> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("read batch: {e}"))?;
        // After projection there is exactly one column.
        let arr = batch.column(0);
        match &dtype {
            arrow::datatypes::DataType::Float32 => {
                let a = arr.as_primitive::<Float32Type>();
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        let v = a.value(i) as f64;
                        if v.is_finite() {
                            out.push(v);
                        }
                    }
                    if let Some(cap) = max_rows {
                        if out.len() >= cap {
                            break;
                        }
                    }
                }
            }
            arrow::datatypes::DataType::Float64 => {
                let a = arr.as_primitive::<Float64Type>();
                for i in 0..a.len() {
                    if a.is_valid(i) {
                        let v = a.value(i);
                        if v.is_finite() {
                            out.push(v);
                        }
                    }
                    if let Some(cap) = max_rows {
                        if out.len() >= cap {
                            break;
                        }
                    }
                }
            }
            other => {
                return Err(format!(
                    "column '{field_name}' has unsupported dtype {other:?}; expected Float32 or Float64"
                ));
            }
        }
        if let Some(cap) = max_rows {
            if out.len() >= cap {
                break;
            }
        }
    }
    Ok(out)
}

fn print_help() {
    eprintln!(
        "fit_yeo_johnson — fit Yeo-Johnson λ via MLE for one feature column.

USAGE
  fit_yeo_johnson <parquet> <feature_col_name> [OPTIONS]
  fit_yeo_johnson <parquet> --feature-idx N [OPTIONS]

OPTIONS
  --feature-idx N      Read column by zero-based index. Mutually exclusive
                       with the positional <feature_col_name>.
  --max-rows N         Sample at most N finite rows. Default: read all.
  --print plain|json   Output format. Default: plain (\"λ = ...\").
  --grid-min L         λ search lower bound. Default: -2.0.
  --grid-max H         λ search upper bound. Default:  2.0.
  --tol T              Golden-section tolerance. Default: 1e-5.
  --help               Show this message.

EXIT CODES
  0 on success, 1 on input error, 2 on numerical failure.

EXAMPLES
  # Fit λ for the f156 column of the canonical safesyn parquet.
  fit_yeo_johnson canonical-2026-05-21/train/safesyn.parquet f156

  # Same, by index, JSON output, capped at 50k rows.
  fit_yeo_johnson canonical-2026-05-21/train/safesyn.parquet \\
      --feature-idx 162 --max-rows 50000 --print json
"
    );
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return std::process::ExitCode::SUCCESS;
    }

    let mut path: Option<PathBuf> = None;
    let mut col_name: Option<String> = None;
    let mut col_idx: Option<usize> = None;
    let mut max_rows: Option<usize> = None;
    let mut print_json = false;
    let mut grid_min = -2.0_f64;
    let mut grid_max = 2.0_f64;
    let mut tol = 1e-5_f64;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--feature-idx" => {
                i += 1;
                let v: usize = args.get(i).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
                    eprintln!("error: --feature-idx requires a non-negative integer");
                    std::process::exit(1);
                });
                col_idx = Some(v);
            }
            "--max-rows" => {
                i += 1;
                max_rows = Some(args[i].parse().unwrap_or_else(|_| {
                    eprintln!("error: --max-rows requires a non-negative integer");
                    std::process::exit(1);
                }));
            }
            "--print" => {
                i += 1;
                match args[i].as_str() {
                    "plain" => print_json = false,
                    "json" => print_json = true,
                    other => {
                        eprintln!("error: --print accepts 'plain' or 'json', got '{other}'");
                        std::process::exit(1);
                    }
                }
            }
            "--grid-min" => {
                i += 1;
                grid_min = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("error: --grid-min must be an f64");
                    std::process::exit(1);
                });
            }
            "--grid-max" => {
                i += 1;
                grid_max = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("error: --grid-max must be an f64");
                    std::process::exit(1);
                });
            }
            "--tol" => {
                i += 1;
                tol = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("error: --tol must be an f64");
                    std::process::exit(1);
                });
            }
            other if other.starts_with("--") => {
                eprintln!("error: unknown flag '{other}'");
                return std::process::ExitCode::from(1);
            }
            _ => {
                if path.is_none() {
                    path = Some(PathBuf::from(a));
                } else if col_name.is_none() && col_idx.is_none() {
                    col_name = Some(a.clone());
                } else {
                    eprintln!("error: unexpected positional argument '{a}'");
                    return std::process::ExitCode::from(1);
                }
            }
        }
        i += 1;
    }

    let Some(path) = path else {
        eprintln!("error: missing parquet path. Pass --help for usage.");
        return std::process::ExitCode::from(1);
    };
    if col_name.is_none() && col_idx.is_none() {
        eprintln!("error: missing column name or --feature-idx. Pass --help for usage.");
        return std::process::ExitCode::from(1);
    }
    if !(grid_min < grid_max) {
        eprintln!("error: --grid-min must be < --grid-max");
        return std::process::ExitCode::from(1);
    }

    let xs = match load_column(&path, col_name.as_deref(), col_idx, max_rows) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error loading column: {e}");
            return std::process::ExitCode::from(1);
        }
    };
    if xs.is_empty() {
        eprintln!("error: no finite samples in column");
        return std::process::ExitCode::from(2);
    }

    // Guard against constant columns.
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    if var < 1e-10 {
        eprintln!(
            "warning: feature column has near-zero variance ({var:.3e}); \
            returning λ = 1.0 (YJ identity branch)"
        );
        if print_json {
            println!(
                r#"{{"lambda": 1.0, "n": {}, "ll": null, "warning": "constant_column"}}"#,
                xs.len()
            );
        } else {
            println!("λ = 1.0");
        }
        return std::process::ExitCode::SUCCESS;
    }

    let (lambda, ll) = golden_search(&xs, grid_min, grid_max, tol);
    if !lambda.is_finite() || !ll.is_finite() {
        eprintln!("error: golden_search returned non-finite (λ={lambda}, LL={ll})");
        return std::process::ExitCode::from(2);
    }

    if print_json {
        println!(
            r#"{{"lambda": {lambda:.6}, "n": {}, "ll": {ll:.4}}}"#,
            xs.len()
        );
    } else {
        println!("λ = {lambda:.6}");
        eprintln!("(n = {}, LL = {ll:.4})", xs.len());
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yj_at_lambda_one_is_identity() {
        for x in [-3.0, 0.0, 3.0, 10.0_f64] {
            assert!((yj(x, 1.0) - x).abs() < 1e-9);
        }
    }

    #[test]
    fn yj_zero_input_is_zero() {
        for l in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0_f64] {
            assert!(yj(0.0, l).abs() < 1e-9);
        }
    }

    #[test]
    fn golden_search_finds_optimum_on_synthetic_gaussian() {
        // Synthetic data: draw from N(0, 1) and shift positive to
        // force a non-trivial λ. λ ≈ 0..1 should win.
        use std::f64::consts::PI;
        let n = 1000;
        let xs: Vec<f64> = (0..n)
            .map(|i| {
                let u1 = ((i + 1) as f64) / (n + 1) as f64;
                let u2 = ((i + 1) as f64 * 1.61803398875) % 1.0;
                // Box-Muller, deterministic
                (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .map(|z| z.exp()) // log-normal — log-scale
            .collect();
        let (lambda, _ll) = golden_search(&xs, -2.0, 2.0, 1e-5);
        // λ near 0 indicates "use log transform" — correct for
        // log-normal data.
        assert!(
            lambda.abs() < 0.5,
            "expected λ near 0 for log-normal data, got {lambda}"
        );
    }
}
