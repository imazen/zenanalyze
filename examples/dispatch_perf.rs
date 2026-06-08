//! Perf measurement for the issue-#53 / PR-#54 dispatch plan.
//! Issue imazen/zenanalyze#50 — quantify the wall-time saved by the
//! Stage 1.5 content-class gates.
//!
//! For every decodable corpus image we time TWO entry points, both with
//! `FeatureSet::SUPPORTED`:
//!
//!   full  = `analyze_features`            (no gating — every tier runs)
//!   gated = `analyze_with_dispatch_plan`  (Stage 0/1.5/2 — grayscale
//!                                          gate drops the chroma tiers
//!                                          on strict R==G==B input)
//!
//! ## Interleaving (anti-thermal/turbo-bias)
//!
//! We do NOT run all-full-then-all-gated — that bakes in turbo/thermal
//! drift across the two halves of the run. Instead, for each image we
//! run `REPS` repetitions and inside each rep we time `full` then
//! `gated` back-to-back, alternating which one goes first per rep
//! (full-first on even reps, gated-first on odd reps) so neither side
//! systematically sits in the warmer/cooler slot. The reported per-side
//! time is the MEDIAN over reps (robust to the occasional scheduler
//! hiccup). The single-core 7950X here doesn't thermally throttle, but
//! the round-robin also cancels any residual turbo-ramp asymmetry.
//!
//! ## Per-gate breakdown
//!
//! A gate only saves time on the images it FIRES on. We classify each
//! image by the gate signals (read from the full analysis) and report
//! mean/median savings overall AND within each subset:
//!
//! - grayscale-fired: `is_grayscale = true` (chroma tiers dropped).
//! - content-unif: `uniformity > 0.95 AND edge_density < tau` — the
//!   DISABLED content-aware uniformity gate's would-fire condition,
//!   measured so the worth-it call has the wall-time it WOULD save.
//! - neither: full work, gated ≈ full (parity overhead only).
//!
//! Output: per-image TSV + a stderr summary. The TSV is the committed
//! artifact.
//!
//! Usage:
//!
//!     cargo run --release --features "experimental,hdr" \
//!         --example dispatch_perf -- \
//!         --corpus /home/lilith/work/codec-corpus/imazen-26 \
//!         --output benchmarks/dispatch_perf_2026-06-04.tsv \
//!         --screen-n 50 --photo-n 100 --seed 1 --reps 9 \
//!         --content-unif-edge-max 0.0005
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

use image::{GenericImageView, ImageReader};
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenanalyze::{analyze_features, analyze_with_dispatch_plan};
use zenpixels::{PixelDescriptor, PixelSlice};

struct Args {
    corpus: PathBuf,
    output: PathBuf,
    screen_n: usize,
    photo_n: usize,
    seed: u64,
    reps: usize,
    content_unif_edge_max: f32,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut corpus = None;
        let mut output = None;
        let mut screen_n = 50usize;
        let mut photo_n = 100usize;
        let mut seed = 1u64;
        let mut reps = 9usize;
        let mut content_unif_edge_max = 0.0005f32;
        let raw: Vec<String> = env::args().collect();
        let mut it = raw.iter().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "-h" | "--help" => {
                    eprintln!(
                        "Usage: dispatch_perf --corpus DIR --output PATH \
                         [--screen-n N] [--photo-n N] [--seed S] [--reps R] \
                         [--content-unif-edge-max TAU]"
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
                "--content-unif-edge-max" => {
                    content_unif_edge_max = it
                        .next()
                        .ok_or("--content-unif-edge-max")?
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
            content_unif_edge_max,
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
/// Identical selection logic to `validate_dispatch_gates` so the two
/// artifacts cover the same images.
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

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// One timed call to `analyze_features` (microseconds). The PixelSlice
/// rebuild + black_box keep LLVM from hoisting work out of the loop.
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Subset {
    Grayscale,
    ContentUnif,
    Neither,
}

#[derive(Default)]
struct SubsetAcc {
    full_ms_per_mp: Vec<f64>,
    gated_ms_per_mp: Vec<f64>,
    full_ms: Vec<f64>,
    gated_ms: Vec<f64>,
    savings_pct: Vec<f64>,
    mp: Vec<f64>,
    n: u64,
}

impl SubsetAcc {
    fn push(&mut self, full_ms: f64, gated_ms: f64, mp: f64) {
        self.full_ms.push(full_ms);
        self.gated_ms.push(gated_ms);
        if mp > 0.0 {
            self.full_ms_per_mp.push(full_ms / mp);
            self.gated_ms_per_mp.push(gated_ms / mp);
        }
        if full_ms > 0.0 {
            self.savings_pct
                .push((full_ms - gated_ms) / full_ms * 100.0);
        }
        self.mp.push(mp);
        self.n += 1;
    }

    fn report(&mut self, label: &str) {
        if self.n == 0 {
            eprintln!("  [{label}]  n=0 (gate never fired on this sample)");
            return;
        }
        let med_full = median(&mut self.full_ms.clone());
        let med_gated = median(&mut self.gated_ms.clone());
        let med_full_mp = median(&mut self.full_ms_per_mp.clone());
        let med_gated_mp = median(&mut self.gated_ms_per_mp.clone());
        let mean_sav = mean(&self.savings_pct);
        let med_sav = median(&mut self.savings_pct.clone());
        let med_mp = median(&mut self.mp.clone());
        eprintln!("  [{label}]  n={}", self.n);
        eprintln!(
            "    median image: {med_mp:.2} MP | full={med_full:.3} ms ({med_full_mp:.3} ms/MP) | \
             gated={med_gated:.3} ms ({med_gated_mp:.3} ms/MP)"
        );
        eprintln!(
            "    per-image savings: mean={mean_sav:+.2}%  median={med_sav:+.2}%  \
             (median wall delta {:.3} ms)",
            med_full - med_gated
        );
    }
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let screen_dir = args.corpus.join("screen");
    let screen_all = collect_images(&screen_dir);
    let screen_sample = stride_sample(&screen_all, args.screen_n, args.seed);

    let photo_dirs = [
        "unsplash",
        "nasa",
        "noaa",
        "national-park-service",
        "epa",
        "internet-archive-scans",
        "skitter",
        "office-documents",
    ];
    let mut photo_all: Vec<PathBuf> = Vec::new();
    for d in photo_dirs {
        photo_all.extend(collect_images(&args.corpus.join(d)));
    }
    photo_all.sort();
    let photo_sample = stride_sample(&photo_all, args.photo_n, args.seed);

    eprintln!(
        "corpus={} | screen {}/{} | photo/mixed {}/{} | reps={} | content-unif edge_max={}",
        args.corpus.display(),
        screen_sample.len(),
        screen_all.len(),
        photo_sample.len(),
        photo_all.len(),
        args.reps,
        args.content_unif_edge_max,
    );

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }

    let supported = FeatureSet::SUPPORTED;
    let query = AnalysisQuery::new(supported);

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("open {}: {e}", args.output.display()));
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "path\tstratum\twidth\theight\tmegapixels\tis_grayscale\tuniformity\tedge_density\t\
         subset\tfull_ms\tgated_ms\tfull_ms_per_mp\tgated_ms_per_mp\tsavings_pct"
    )
    .unwrap();

    let mut all_acc = SubsetAcc::default();
    let mut gray_acc = SubsetAcc::default();
    let mut cunif_acc = SubsetAcc::default();
    let mut neither_acc = SubsetAcc::default();

    let mut n_images = 0u64;
    let mut n_failed = 0u64;

    let all: Vec<(&'static str, &PathBuf)> = screen_sample
        .iter()
        .map(|p| ("screen", p))
        .chain(photo_sample.iter().map(|p| ("photo", p)))
        .collect();

    for (stratum, path) in &all {
        let dyn_img = match ImageReader::open(path).map(|r| r.decode()) {
            Ok(Ok(img)) => img,
            _ => {
                n_failed += 1;
                continue;
            }
        };
        let (wd, ht) = dyn_img.dimensions();
        if wd < 2 || ht < 2 {
            n_failed += 1;
            continue;
        }
        let rgb8 = dyn_img.to_rgb8();
        let rgb = rgb8.as_raw();
        let stride = wd as usize * 3;
        if PixelSlice::new(rgb, wd, ht, stride, PixelDescriptor::RGB8_SRGB).is_err() {
            n_failed += 1;
            continue;
        }
        let mp = (wd as f64 * ht as f64) / 1_000_000.0;

        // Classify the image by gate signals (read from full analysis).
        let s = PixelSlice::new(rgb, wd, ht, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let full_res = analyze_features(s, &query).unwrap();
        let is_gray = full_res
            .get(AnalysisFeature::IsGrayscale)
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let uniformity = full_res.get_f32(AnalysisFeature::Uniformity).unwrap_or(0.0);
        let edge_density = full_res
            .get_f32(AnalysisFeature::EdgeDensity)
            .unwrap_or(0.0);
        let subset = if is_gray {
            Subset::Grayscale
        } else if uniformity > 0.95 && edge_density < args.content_unif_edge_max {
            Subset::ContentUnif
        } else {
            Subset::Neither
        };

        // Warmup: 3 untimed paired calls to settle caches / branch
        // predictors before timing.
        for _ in 0..3 {
            let _ = time_full(rgb, wd, ht, stride, &query);
            let _ = time_gated(rgb, wd, ht, stride, &query);
        }

        // Interleaved A/B: alternate which side runs first per rep.
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

        let subset_label = match subset {
            Subset::Grayscale => "grayscale",
            Subset::ContentUnif => "content_unif",
            Subset::Neither => "neither",
        };
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.6}\t{:.6}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:+.3}",
            path.display(),
            stratum,
            wd,
            ht,
            mp,
            is_gray as u8,
            uniformity,
            edge_density,
            subset_label,
            full_ms,
            gated_ms,
            if mp > 0.0 { full_ms / mp } else { 0.0 },
            if mp > 0.0 { gated_ms / mp } else { 0.0 },
            savings_pct,
        )
        .ok();

        all_acc.push(full_ms, gated_ms, mp);
        match subset {
            Subset::Grayscale => gray_acc.push(full_ms, gated_ms, mp),
            Subset::ContentUnif => cunif_acc.push(full_ms, gated_ms, mp),
            Subset::Neither => neither_acc.push(full_ms, gated_ms, mp),
        }

        n_images += 1;
        if n_images.is_multiple_of(20) {
            w.flush().ok();
            eprintln!("[{n_images}] timed (failed={n_failed})");
        }
    }
    w.flush().ok();

    eprintln!("\n================ DISPATCH PERF SUMMARY ================");
    eprintln!(
        "images timed: {n_images} (decode-failed/skipped: {n_failed}); reps/image: {}",
        args.reps
    );
    eprintln!("build: release, NO target-cpu=native; runtime SIMD dispatch");
    eprintln!();
    eprintln!("OVERALL (all images — gated = grayscale-gate-enabled shipped plan):");
    all_acc.report("ALL");
    eprintln!();
    eprintln!("PER-GATE-FIRED SUBSET (a gate only saves time on images it fires on):");
    gray_acc.report("grayscale-gate FIRED (chroma tiers dropped)");
    cunif_acc.report(&format!(
        "content-unif gate WOULD-fire (uniformity>0.95 AND edge_density<{}) — DISABLED, measured for worth-it call",
        args.content_unif_edge_max
    ));
    neither_acc.report("neither gate fired (gated ≈ full — pure dispatch overhead)");
    eprintln!("======================================================\n");
    eprintln!("per-image TSV written to {}", args.output.display());
    ExitCode::from(0)
}
