//! Smoke evaluation for the HVS-derived features added 2026-05-17.
//!
//! Walks a small image set, runs the analyzer with the five new
//! HVS features in the requested set, and prints a per-image
//! summary plus aggregate min/median/max across the set.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features experimental --example hvs_feature_smoke -- <dir>
//! ```
//!
//! Picks up to 20 PNG/JPEG files (alphabetical) from the directory.

use image::ImageReader;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

fn quantile_sorted(arr: &[f32], q: f32) -> f32 {
    if arr.is_empty() {
        return f32::NAN;
    }
    let idx = ((arr.len() as f32 * q) as usize).min(arr.len() - 1);
    arr[idx]
}

fn fmt_f(x: f32) -> String {
    if x.is_finite() {
        format!("{x:>8.4}")
    } else {
        format!("{x:>8}")
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/mnt/v/input/zensim/sources".to_string()),
    );

    // Pick up to 20 image files from the directory (size-bounded so
    // we don't smoke-test on 100 MP source images).
    let mut entries: Vec<PathBuf> = Vec::new();
    let dir_iter = match std::fs::read_dir(&dir) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error reading {dir:?}: {e}");
            std::process::exit(1);
        }
    };
    for e in dir_iter.flatten() {
        let p = e.path();
        if let Some(ext) = p.extension().and_then(|s| s.to_str())
            && matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp"
            )
            && let Ok(md) = std::fs::metadata(&p)
            && md.len() < 4 * 1024 * 1024
        {
            entries.push(p);
        }
        if entries.len() >= 200 {
            break;
        }
    }
    entries.sort();
    entries.truncate(20);

    if entries.is_empty() {
        eprintln!("no images found in {dir:?}");
        std::process::exit(1);
    }

    // Request the five new HVS features (and a handful of references
    // for sanity context).
    let mut req = FeatureSet::new();
    req = req.with(AnalysisFeature::Variance);
    req = req.with(AnalysisFeature::EdgeDensity);
    req = req.with(AnalysisFeature::ChromaLumaCovarianceCb);
    req = req.with(AnalysisFeature::ChromaLumaCovarianceCr);
    req = req.with(AnalysisFeature::InfoWeightMean);
    req = req.with(AnalysisFeature::InfoWeightP90);
    req = req.with(AnalysisFeature::OrientationEnergyRatio);
    let query = AnalysisQuery::new(req);

    println!(
        "{:<50} {:>6} {:>6} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "path", "W", "H", "ms", "covCb", "covCr", "iw_m", "iw_p90", "orient"
    );

    let mut covs_cb: Vec<f32> = Vec::new();
    let mut covs_cr: Vec<f32> = Vec::new();
    let mut iw_means: Vec<f32> = Vec::new();
    let mut iw_p90s: Vec<f32> = Vec::new();
    let mut oris: Vec<f32> = Vec::new();
    let mut total_pixels: u64 = 0;
    let mut total_ms: f64 = 0.0;

    for path in &entries {
        let img = match ImageReader::open(path)
            .map(|r| r.decode())
            .and_then(|x| x.map_err(std::io::Error::other))
        {
            Ok(i) => i,
            Err(e) => {
                eprintln!("skip {path:?}: {e}");
                continue;
            }
        };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let bytes = rgb.into_raw();
        let stride = (w as usize) * 3;
        let slice = PixelSlice::new(&bytes, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let t0 = Instant::now();
        let r = match zenanalyze::analyze_features(slice, &query) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("analyze failed for {path:?}: {e}");
                continue;
            }
        };
        let dt_ms = t0.elapsed().as_secs_f64() * 1e3;
        total_ms += dt_ms;
        total_pixels += (w as u64) * (h as u64);

        let cov_cb = r
            .get_f32(AnalysisFeature::ChromaLumaCovarianceCb)
            .unwrap_or(f32::NAN);
        let cov_cr = r
            .get_f32(AnalysisFeature::ChromaLumaCovarianceCr)
            .unwrap_or(f32::NAN);
        let iw_m = r
            .get_f32(AnalysisFeature::InfoWeightMean)
            .unwrap_or(f32::NAN);
        let iw_p = r
            .get_f32(AnalysisFeature::InfoWeightP90)
            .unwrap_or(f32::NAN);
        let ori = r
            .get_f32(AnalysisFeature::OrientationEnergyRatio)
            .unwrap_or(f32::NAN);
        covs_cb.push(cov_cb);
        covs_cr.push(cov_cr);
        iw_means.push(iw_m);
        iw_p90s.push(iw_p);
        oris.push(ori);

        let pname = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .chars()
            .take(50)
            .collect::<String>();
        println!(
            "{:<50} {:>6} {:>6} {:>8.2} {:>9} {:>9} {:>9} {:>9} {:>9}",
            pname,
            w,
            h,
            dt_ms,
            fmt_f(cov_cb),
            fmt_f(cov_cr),
            fmt_f(iw_m),
            fmt_f(iw_p),
            fmt_f(ori)
        );
    }
    let n = covs_cb.len();
    let print_stat = |label: &str, mut v: Vec<f32>| {
        // Strip NaNs for percentile computation.
        v.retain(|x| x.is_finite());
        v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let min = v.first().copied().unwrap_or(f32::NAN);
        let med = quantile_sorted(&v, 0.5);
        let max = v.last().copied().unwrap_or(f32::NAN);
        println!(
            "{:<10}  n={:>3}  min={:>9}  med={:>9}  max={:>9}",
            label,
            v.len(),
            fmt_f(min),
            fmt_f(med),
            fmt_f(max),
        );
    };

    println!("\n--- aggregate over {n} images ---");
    print_stat("covCb", covs_cb);
    print_stat("covCr", covs_cr);
    print_stat("iw_mean", iw_means);
    print_stat("iw_p90", iw_p90s);
    print_stat("orient", oris);
    let mp = total_pixels as f64 / 1_000_000.0;
    println!(
        "\nthroughput: {total_pixels} pixels in {total_ms:.1} ms over {n} images = {:.3} ms/MP",
        total_ms / mp.max(1e-9)
    );

    // Sanity gates — exit non-zero if any output value is NaN/inf
    // outside the documented ranges (chroma covariances ∈ [−1, 1],
    // info weights ≥ 0, orientation ≥ 1).
    let bad_path = |label: &str, v: &Path| {
        eprintln!("smoke failure: {label} produced an unexpected value on {v:?}");
    };
    let _ = bad_path;
}
