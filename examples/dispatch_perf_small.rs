//! Small-image size sweep for the issue-#53 / PR-#54 dispatch plan.
//! Issue imazen/zenanalyze#50 — quantify the wall-time the Stage 1.5
//! grayscale gate saves at SMALL image sizes, where its roughly-fixed
//! saving (skipping the chroma / tier-2 passes) is a meaningful fraction
//! of total analysis time. The companion large-corpus harness
//! (`dispatch_perf.rs`) established gating is FREE at <8 MP but could
//! not measure the WIN: that corpus is large-image-heavy (median 9 MP),
//! so the fixed grayscale saving is buried under the per-pixel cost.
//!
//! This harness DOWNSCALES a representative sample of imazen-26
//! originals to a log-spaced size set and times the two entry points at
//! each size, both with `FeatureSet::SUPPORTED`:
//!
//!   full  = `analyze_features`                  (no gating)
//!   gated = `analyze_with_dispatch_plan(.., None)` (default hints —
//!           Stage 2 OFF, only the grayscale gate is enabled)
//!
//! ## Downscale-only, principled kernel
//!
//! Each source is resized to longest-side ∈ {48,64,96,128,192,256,384,
//! 512,768,1024} with Lanczos3 (matching the picker-feature extractor).
//! Sizes >= the source's longest side are SKIPPED — we never upscale
//! (synthetic upscale has no high-freq detail and would mislead the
//! per-pixel fit). Resize happens once, in memory, BEFORE timing; only
//! the `analyze_*` call is timed.
//!
//! ## Interleaving (anti-thermal/turbo-bias)
//!
//! For each (image, size) we run `REPS` paired reps; inside each rep we
//! time full then gated, alternating which goes first per rep so neither
//! side systematically sits in the warmer slot. The reported per-side
//! time is the MEDIAN over reps. A 3-rep untimed warmup settles caches /
//! branch predictors first.
//!
//! ## Grayscale subset
//!
//! The 8 imazen-26 grayscale originals (discovered by the large-corpus
//! run: `is_grayscale==1`) are included explicitly so the grayscale gate
//! fires at every small size. Their rows carry `is_grayscale=1`; the
//! per-size summary breaks out the grayscale-fired subset specifically —
//! that is the ONLY enabled gate and the only place a real win can show.
//!
//! Output: per-(image,size) TSV + a per-size stderr summary with the
//! α + β·pixels fit for both paths. The TSV is the committed artifact.
//!
//! Usage:
//!
//!     cargo run --release --example dispatch_perf_small -- \
//!         --corpus /home/lilith/work/codec-corpus/imazen-26 \
//!         --output benchmarks/dispatch_perf_small_2026-06-05.tsv \
//!         --screen-n 5 --photo-n 5 --seed 1 --reps 21
//!
//! Build WITHOUT target-cpu=native — runtime SIMD dispatch is what
//! ships. The corpus is READ-ONLY (only decoded).

use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageReader};
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenanalyze::{analyze_features, analyze_with_dispatch_plan};
use zenpixels::{PixelDescriptor, PixelSlice};

/// Log-spaced longest-side targets per CLAUDE.md size-sweep discipline.
const SIZES: &[u32] = &[48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

/// The 8 imazen-26 grayscale originals (from the large-corpus run's
/// `is_grayscale==1` rows). All are large, so every SIZE target is a
/// genuine downscale (no upscale). 6 are document / line-art grayscale,
/// 2 are photo grayscale.
const GRAYSCALE_RELPATHS: &[&str] = &[
    "office-documents/office_irs_f1040sa_p01_page1.png",
    "office-documents/office_irs_f1040sd_p01_page1.png",
    "office-documents/office_irs_f941_p01_page1.png",
    "office-documents/office_irs_fw7_p01_page1.png",
    "office-documents/office_uspto-11292620_lunar-landing-pad_p04_drawings.png",
    "office-documents/office_uspto-12190202_quantum-chip_p04_drawings.png",
    "unsplash/unsplash-people/ian-taylor-BloPN4jcI7s-unsplash.jpg",
    "unsplash/unsplash-renders/logan-voss-z_e_ZrFty00-unsplash.jpg",
];

struct Args {
    corpus: PathBuf,
    output: PathBuf,
    screen_n: usize,
    photo_n: usize,
    seed: u64,
    reps: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut corpus = None;
        let mut output = None;
        let mut screen_n = 5usize;
        let mut photo_n = 5usize;
        let mut seed = 1u64;
        let mut reps = 21usize;
        let raw: Vec<String> = env::args().collect();
        let mut it = raw.iter().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "-h" | "--help" => {
                    eprintln!(
                        "Usage: dispatch_perf_small --corpus DIR --output PATH \
                         [--screen-n N] [--photo-n N] [--seed S] [--reps R]"
                    );
                    std::process::exit(0);
                }
                "--corpus" => corpus = it.next().map(PathBuf::from),
                "--output" => output = it.next().map(PathBuf::from),
                "--screen-n" => {
                    screen_n = it
                        .next()
                        .ok_or("--screen-n")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--photo-n" => {
                    photo_n = it
                        .next()
                        .ok_or("--photo-n")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--seed" => {
                    seed = it
                        .next()
                        .ok_or("--seed")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--reps" => {
                    reps = it
                        .next()
                        .ok_or("--reps")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                other => return Err(format!("unknown arg {other}")),
            }
        }
        Ok(Args {
            corpus: corpus.ok_or("--corpus DIR required")?,
            output: output.ok_or("--output PATH required")?,
            screen_n,
            photo_n,
            seed,
            reps: reps.max(1),
        })
    }
}

fn is_decodable(p: &Path) -> bool {
    matches!(
        p.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg")
    )
}

fn collect_images(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if is_decodable(&p) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Deterministic stride sample of `n` items, seeded for reproducibility.
fn stride_sample(items: &[PathBuf], n: usize, seed: u64) -> Vec<PathBuf> {
    if items.is_empty() || n == 0 {
        return Vec::new();
    }
    if items.len() <= n {
        return items.to_vec();
    }
    let stride = items.len() as f64 / n as f64;
    let start = (seed as usize) % items.len();
    let mut picked = Vec::with_capacity(n);
    let mut seen = std::collections::BTreeSet::new();
    for k in 0..n {
        let idx = ((start as f64 + k as f64 * stride) as usize) % items.len();
        let mut j = idx;
        while seen.contains(&j) {
            j = (j + 1) % items.len();
        }
        seen.insert(j);
        picked.push(items[j].clone());
    }
    picked.sort();
    picked
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Downscale to longest-side `target` with Lanczos3. Returns `None` when
/// the source is already <= target (we never upscale).
fn downscale_to(src: &DynamicImage, target: u32) -> Option<DynamicImage> {
    let (w, h) = src.dimensions();
    if w.max(h) <= target {
        return None;
    }
    let ratio = target as f64 / w.max(h) as f64;
    let new_w = ((w as f64) * ratio).round().max(2.0) as u32;
    let new_h = ((h as f64) * ratio).round().max(2.0) as u32;
    Some(src.resize_exact(new_w, new_h, FilterType::Lanczos3))
}

#[inline(never)]
fn time_full(buf: &[u8], w: u32, h: u32, stride: usize, query: &AnalysisQuery) -> f64 {
    let s = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let t0 = Instant::now();
    let r = analyze_features(black_box(s), query).unwrap();
    let dt = t0.elapsed().as_nanos() as f64 / 1000.0;
    black_box(&r);
    dt
}

#[inline(never)]
fn time_gated(buf: &[u8], w: u32, h: u32, stride: usize, query: &AnalysisQuery) -> f64 {
    let s = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let t0 = Instant::now();
    let r = analyze_with_dispatch_plan(black_box(s), query, None).unwrap();
    let dt = t0.elapsed().as_nanos() as f64 / 1000.0;
    black_box(&r);
    dt
}

/// Ordinary-least-squares fit `y = a + b*x`. Returns (a, b). `a` is the
/// fixed overhead (ms at zero pixels), `b` is the per-pixel slope
/// (ms per megapixel when x is in MP).
fn ols_fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return (0.0, 0.0);
    }
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return (sy / n, 0.0);
    }
    let b = (n * sxy - sx * sy) / denom;
    let a = (sy - b * sx) / n;
    (a, b)
}

/// One timed datapoint at one (image, size) cell.
struct Cell {
    is_gray: bool,
    mp: f64,
    full_ms: f64,
    gated_ms: f64,
}

/// Per-size accumulator. Keeps raw cells so we can fit α/β and split the
/// grayscale subset.
#[derive(Default)]
struct SizeAcc {
    cells: Vec<Cell>,
}

impl SizeAcc {
    fn push(&mut self, c: Cell) {
        self.cells.push(c);
    }

    fn split(&self) -> (Vec<&Cell>, Vec<&Cell>) {
        let gray: Vec<&Cell> = self.cells.iter().filter(|c| c.is_gray).collect();
        let color: Vec<&Cell> = self.cells.iter().filter(|c| !c.is_gray).collect();
        (gray, color)
    }
}

fn summarize_group(cells: &[&Cell]) -> Option<(f64, f64, f64, f64, f64, f64)> {
    // Returns (med_full_ms, med_gated_ms, med_pct, med_ms_saved, med_mp, n).
    if cells.is_empty() {
        return None;
    }
    let mut full: Vec<f64> = cells.iter().map(|c| c.full_ms).collect();
    let mut gated: Vec<f64> = cells.iter().map(|c| c.gated_ms).collect();
    let mut pct: Vec<f64> = cells
        .iter()
        .filter(|c| c.full_ms > 0.0)
        .map(|c| (c.full_ms - c.gated_ms) / c.full_ms * 100.0)
        .collect();
    let mut saved: Vec<f64> = cells.iter().map(|c| c.full_ms - c.gated_ms).collect();
    let mut mp: Vec<f64> = cells.iter().map(|c| c.mp).collect();
    Some((
        median(&mut full),
        median(&mut gated),
        median(&mut pct),
        median(&mut saved),
        median(&mut mp),
        cells.len() as f64,
    ))
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    // Grayscale originals (explicit list — the gate must fire).
    let mut gray_imgs: Vec<PathBuf> = Vec::new();
    for rel in GRAYSCALE_RELPATHS {
        let p = args.corpus.join(rel);
        if p.exists() {
            gray_imgs.push(p);
        } else {
            eprintln!("warning: grayscale source missing: {}", p.display());
        }
    }

    // Screen sample (color).
    let screen_all = collect_images(&args.corpus.join("screen"));
    let screen_sample = stride_sample(&screen_all, args.screen_n, args.seed);

    // Photo sample (color) — same dirs as the large-corpus harness, minus
    // office-documents (those are the grayscale documents).
    let photo_dirs = [
        "unsplash",
        "nasa",
        "noaa",
        "national-park-service",
        "epa",
        "internet-archive-scans",
        "skitter",
    ];
    let mut photo_all: Vec<PathBuf> = Vec::new();
    for d in photo_dirs {
        photo_all.extend(collect_images(&args.corpus.join(d)));
    }
    photo_all.sort();
    // Drop the explicit grayscale photos so they aren't double-counted as
    // color (they're decoded grayscale anyway, but be explicit).
    photo_all.retain(|p| !gray_imgs.iter().any(|g| g == p));
    let photo_sample = stride_sample(&photo_all, args.photo_n, args.seed);

    eprintln!(
        "corpus={} | grayscale {} | screen {}/{} | photo {}/{} | reps={} | sizes={:?}",
        args.corpus.display(),
        gray_imgs.len(),
        screen_sample.len(),
        screen_all.len(),
        photo_sample.len(),
        photo_all.len(),
        args.reps,
        SIZES,
    );

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("open {}: {e}", args.output.display()));
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "path\tstratum\ttarget_size\twidth\theight\tmegapixels\tis_grayscale\t\
         full_ms\tgated_ms\tsavings_pct\tms_saved"
    )
    .unwrap();

    // tag each source with its stratum, decode once, downscale per size.
    let sources: Vec<(&'static str, &PathBuf)> = gray_imgs
        .iter()
        .map(|p| ("grayscale", p))
        .chain(screen_sample.iter().map(|p| ("screen", p)))
        .chain(photo_sample.iter().map(|p| ("photo", p)))
        .collect();

    // One accumulator per target size.
    let mut size_accs: Vec<SizeAcc> = (0..SIZES.len()).map(|_| SizeAcc::default()).collect();

    let mut n_cells = 0u64;
    let mut n_failed = 0u64;

    for (stratum, path) in &sources {
        let dyn_img = match ImageReader::open(path).map(|r| r.decode()) {
            Ok(Ok(img)) => img,
            _ => {
                n_failed += 1;
                eprintln!("decode-failed: {}", path.display());
                continue;
            }
        };

        for (si, &target) in SIZES.iter().enumerate() {
            let Some(small) = downscale_to(&dyn_img, target) else {
                // source already <= target: skip (no upscale).
                continue;
            };
            let (wd, ht) = small.dimensions();
            if wd < 2 || ht < 2 {
                continue;
            }
            let rgb8 = small.to_rgb8();
            let rgb = rgb8.as_raw();
            let stride = wd as usize * 3;
            if PixelSlice::new(rgb, wd, ht, stride, PixelDescriptor::RGB8_SRGB).is_err() {
                continue;
            }
            let mp = (wd as f64 * ht as f64) / 1_000_000.0;

            // Classify via the full analysis (read the gate signal).
            let s = PixelSlice::new(rgb, wd, ht, stride, PixelDescriptor::RGB8_SRGB).unwrap();
            let full_res = analyze_features(s, &query).unwrap();
            let is_gray = full_res
                .get(AnalysisFeature::IsGrayscale)
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            // Warmup: 3 untimed paired calls.
            for _ in 0..3 {
                let _ = time_full(rgb, wd, ht, stride, &query);
                let _ = time_gated(rgb, wd, ht, stride, &query);
            }

            // Interleaved A/B.
            let mut full_us = Vec::with_capacity(args.reps);
            let mut gated_us = Vec::with_capacity(args.reps);
            for r in 0..args.reps {
                if r % 2 == 0 {
                    full_us.push(time_full(rgb, wd, ht, stride, &query));
                    gated_us.push(time_gated(rgb, wd, ht, stride, &query));
                } else {
                    gated_us.push(time_gated(rgb, wd, ht, stride, &query));
                    full_us.push(time_full(rgb, wd, ht, stride, &query));
                }
            }
            let full_ms = median(&mut full_us) / 1000.0;
            let gated_ms = median(&mut gated_us) / 1000.0;
            let savings_pct = if full_ms > 0.0 {
                (full_ms - gated_ms) / full_ms * 100.0
            } else {
                0.0
            };
            let ms_saved = full_ms - gated_ms;

            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{:.5}\t{:.5}\t{:+.3}\t{:+.5}",
                path.display(),
                stratum,
                target,
                wd,
                ht,
                mp,
                is_gray as u8,
                full_ms,
                gated_ms,
                savings_pct,
                ms_saved,
            )
            .ok();

            size_accs[si].push(Cell {
                is_gray,
                mp,
                full_ms,
                gated_ms,
            });
            n_cells += 1;
        }
        w.flush().ok();
        eprintln!("done source {} ({})", path.display(), stratum);
    }
    w.flush().ok();

    // ---- per-size summary ----
    eprintln!("\n================ SMALL-IMAGE DISPATCH PERF ================");
    eprintln!(
        "cells timed: {n_cells} (decode-failed: {n_failed}); reps/cell: {}",
        args.reps
    );
    eprintln!("build: release, NO target-cpu=native; runtime SIMD dispatch");
    eprintln!(
        "gated = analyze_with_dispatch_plan(.., None): Stage 2 OFF, only grayscale gate enabled\n"
    );

    eprintln!(
        "{:>6} {:>5} {:>10} {:>10} {:>8} {:>9}  | grayscale-fired subset",
        "size", "n", "full_ms", "gated_ms", "pct", "ms_saved"
    );
    eprintln!(
        "{:>6} {:>5} {:>10} {:>10} {:>8} {:>9}  | {:>5} {:>9} {:>8} {:>9}",
        "", "", "(med)", "(med)", "delta", "(med)", "n", "full_ms", "pct", "ms_saved"
    );

    // Collect for α/β fit across sizes (all-cells, per path).
    let mut fit_mp_all: Vec<f64> = Vec::new();
    let mut fit_full_all: Vec<f64> = Vec::new();
    let mut fit_gated_all: Vec<f64> = Vec::new();
    let mut fit_mp_gray: Vec<f64> = Vec::new();
    let mut fit_full_gray: Vec<f64> = Vec::new();
    let mut fit_gated_gray: Vec<f64> = Vec::new();
    let mut fit_mp_color: Vec<f64> = Vec::new();
    let mut fit_full_color: Vec<f64> = Vec::new();
    let mut fit_gated_color: Vec<f64> = Vec::new();

    for (si, &target) in SIZES.iter().enumerate() {
        let acc = &size_accs[si];
        if acc.cells.is_empty() {
            continue;
        }
        let (gray, _color) = acc.split();
        let all_refs: Vec<&Cell> = acc.cells.iter().collect();
        let all = summarize_group(&all_refs);
        let gray_sum = summarize_group(&gray);

        // Feed the fits.
        for c in &acc.cells {
            fit_mp_all.push(c.mp);
            fit_full_all.push(c.full_ms);
            fit_gated_all.push(c.gated_ms);
            if c.is_gray {
                fit_mp_gray.push(c.mp);
                fit_full_gray.push(c.full_ms);
                fit_gated_gray.push(c.gated_ms);
            } else {
                fit_mp_color.push(c.mp);
                fit_full_color.push(c.full_ms);
                fit_gated_color.push(c.gated_ms);
            }
        }

        if let Some((f, g, p, sv, _mp, n)) = all {
            let (gn, gf, _gg, _gp, gsv) = match gray_sum {
                Some((gf, _gg, gp, gsv, _gmp, gn)) => (gn as u64, gf, _gg, gp, gsv),
                None => (0u64, 0.0, 0.0, 0.0, 0.0),
            };
            if gn > 0 {
                eprintln!(
                    "{target:>6} {:>5} {f:>10.4} {g:>10.4} {p:>+7.2}% {sv:>+9.5}  | \
                     {gn:>5} {gf:>9.4} {_gp:>+7.2}% {gsv:>+9.5}",
                    n as u64,
                );
            } else {
                eprintln!(
                    "{target:>6} {:>5} {f:>10.4} {g:>10.4} {p:>+7.2}% {sv:>+9.5}  | (no gray)",
                    n as u64,
                );
            }
        }
    }

    // ---- α/β fits (x = MP, so β is ms/MP, α is fixed-overhead ms) ----
    eprintln!("\n---- linear fit  total_ms = α + β·MP  (OLS over all cells) ----");
    let (a_full, b_full) = ols_fit(&fit_mp_all, &fit_full_all);
    let (a_gated, b_gated) = ols_fit(&fit_mp_all, &fit_gated_all);
    eprintln!(
        "ALL cells    full : α={a_full:+.5} ms  β={b_full:.4} ms/MP   (n={})",
        fit_mp_all.len()
    );
    eprintln!("ALL cells    gated: α={a_gated:+.5} ms  β={b_gated:.4} ms/MP");
    eprintln!(
        "ALL cells    Δα (full-gated fixed saving) = {:+.5} ms   Δβ = {:+.4} ms/MP",
        a_full - a_gated,
        b_full - b_gated
    );

    let (a_gf, b_gf) = ols_fit(&fit_mp_gray, &fit_full_gray);
    let (a_gg, b_gg) = ols_fit(&fit_mp_gray, &fit_gated_gray);
    eprintln!(
        "\nGRAYSCALE    full : α={a_gf:+.5} ms  β={b_gf:.4} ms/MP   (n={})",
        fit_mp_gray.len()
    );
    eprintln!("GRAYSCALE    gated: α={a_gg:+.5} ms  β={b_gg:.4} ms/MP");
    eprintln!(
        "GRAYSCALE    Δα (fixed saving) = {:+.5} ms   Δβ (per-MP saving) = {:+.4} ms/MP",
        a_gf - a_gg,
        b_gf - b_gg
    );

    let (a_cf, b_cf) = ols_fit(&fit_mp_color, &fit_full_color);
    let (a_cg, b_cg) = ols_fit(&fit_mp_color, &fit_gated_color);
    eprintln!(
        "\nCOLOR        full : α={a_cf:+.5} ms  β={b_cf:.4} ms/MP   (n={})",
        fit_mp_color.len()
    );
    eprintln!("COLOR        gated: α={a_cg:+.5} ms  β={b_cg:.4} ms/MP");
    eprintln!(
        "COLOR        Δα (fixed saving) = {:+.5} ms   Δβ = {:+.4} ms/MP  \
         (≈0 expected — gate never fires on color, this is pure dispatch overhead)",
        a_cf - a_cg,
        b_cf - b_cg
    );

    eprintln!("\n==========================================================\n");
    eprintln!("per-(image,size) TSV written to {}", args.output.display());
    ExitCode::from(0)
}
