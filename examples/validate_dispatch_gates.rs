//! Picker-free gate-correctness validation for the issue-#53 dispatch
//! plan (Stage 1.5 + Stage 2). PR imazen/zenanalyze#54.
//!
//! For each image we run BOTH entry points with `FeatureSet::SUPPORTED`:
//!
//!   full  = `analyze_features`            (no gating)
//!   gated = `analyze_with_dispatch_plan`  (Stage 0/1.5/2)
//!
//! A Stage 1.5 gate is SAFE iff, on the images where it fired, the
//! features it DROPPED were what `analyze_features` would have computed
//! anyway as negligible (~ default / below a meaningfulness threshold).
//! We therefore record, per dropped feature, the value the FULL
//! analysis actually produced on that image, and report the worst-case
//! magnitude across all gated images. If a gate ever dropped a feature
//! whose full value was meaningfully non-trivial, that gate is UNSAFE.
//!
//! We also cross-check the grayscale gate against a ground-truth chroma
//! measure computed directly from the RGB8 bytes (independent of the
//! analyzer's `is_grayscale` classifier): the fraction of sampled
//! pixels whose max-min channel spread exceeds 2 (8-bit levels). If the
//! gate fires on an image with real chroma, it misfired.
//!
//! Output: a per-image TSV (one row per image) plus a per-gate summary
//! printed to stderr. The TSV is the committed artifact.
//!
//! Usage:
//!
//!     cargo run --release --features "experimental,hdr" \
//!         --example validate_dispatch_gates -- \
//!         --corpus /home/lilith/work/codec-corpus/imazen-26 \
//!         --output benchmarks/dispatch_gate_validation_2026-06-04.tsv \
//!         --screen-n 50 --photo-n 100 --seed 1
//!
//! The corpus directory is treated as READ-ONLY — we only decode files.

use std::collections::BTreeMap;
use std::env;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use image::{GenericImageView, ImageReader};
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenanalyze::{analyze_features, analyze_with_dispatch_plan};
// Hints are always `None` here — Stage 0/1.5/2 derive every decision
// from image dimensions + Tier 1 output, never from a hint.
use zenpixels::{PixelDescriptor, PixelSlice};

/// Mirror of `feature::CHROMA_DROP_FEATURES` (which is `pub(crate)`),
/// reconstructed from public `AnalysisFeature` variants. The dispatch
/// plan drops these on `is_grayscale = true`. Kept in lockstep with
/// the const in `src/feature.rs` — the per-image drop detection below
/// does NOT rely on this list (it compares full-vs-gated presence
/// directly); this is only used to enumerate the TSV drop columns and
/// to attribute drops to the grayscale gate.
fn chroma_drop_features() -> FeatureSet {
    use AnalysisFeature as F;
    FeatureSet::new()
        .with(F::ChromaComplexity)
        .with(F::CbSharpness)
        .with(F::CrSharpness)
        .with(F::Colourfulness)
        .with(F::SkinToneFraction)
        .with(F::CbHorizSharpness)
        .with(F::CbVertSharpness)
        .with(F::CbPeakSharpness)
        .with(F::CrHorizSharpness)
        .with(F::CrVertSharpness)
        .with(F::CrPeakSharpness)
        .with(F::DctCompressibilityUV)
        .with(F::NoiseFloorUV)
        .with(F::NoiseFloorUvP25)
        .with(F::NoiseFloorUvP50)
        .with(F::NoiseFloorUvP75)
        .with(F::NoiseFloorUvP90)
        .with(F::QuantSurvivalUv)
        .with(F::QuantSurvivalUvP10)
        .with(F::QuantSurvivalUvP25)
        .with(F::QuantSurvivalUvP50)
        .with(F::QuantSurvivalUvP75)
}

/// Mirror of `feature::SATURATING_DROP_FEATURES` (`pub(crate)`),
/// reconstructed from public variants. Dropped on `uniformity > 0.95`.
fn saturating_drop_features() -> FeatureSet {
    use AnalysisFeature as F;
    FeatureSet::new()
        .with(F::LaplacianVarianceP50)
        .with(F::LaplacianVarianceP75)
        .with(F::LaplacianVarianceP90)
        .with(F::LaplacianVarianceP99)
        .with(F::LaplacianVariancePeak)
        .with(F::PatchFractionFast)
        .with(F::AqMapP50)
        .with(F::AqMapP75)
        .with(F::AqMapP90)
        .with(F::AqMapP95)
        .with(F::AqMapP99)
}

/// 8-bit channel-spread threshold for "this pixel carries real chroma".
/// max(R,G,B) - min(R,G,B) > 2 levels means the pixel is not neutral
/// gray within rounding. Independent of the analyzer.
const CHROMA_LEVEL_THRESHOLD: u8 = 2;

/// "Meaningfulness" threshold for a dropped feature's full value. Most
/// dropped features are sharpness / percentile / fraction signals that
/// sit near 0 on grayscale / uniform content. A worst-case full value
/// above this on a gated image is the red flag the validation hunts
/// for. Reported as raw magnitude regardless; this is just the
/// safe/unsafe classifier line in the summary.
const MEANINGFUL_MAGNITUDE: f32 = 1.0;

struct Args {
    corpus: PathBuf,
    output: PathBuf,
    screen_n: usize,
    photo_n: usize,
    seed: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut corpus = None;
        let mut output = None;
        let mut screen_n = 50usize;
        let mut photo_n = 100usize;
        let mut seed = 1u64;
        let raw: Vec<String> = env::args().collect();
        let mut it = raw.iter().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "-h" | "--help" => {
                    eprintln!(
                        "Usage: validate_dispatch_gates --corpus DIR --output PATH \
                         [--screen-n N] [--photo-n N] [--seed S]"
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
                other => return Err(format!("unknown arg {other}")),
            }
        }
        Ok(Args {
            corpus: corpus.ok_or("--corpus DIR required")?,
            output: output.ok_or("--output PATH required")?,
            screen_n,
            photo_n,
            seed,
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

/// Recursively collect decodable image paths under `dir`.
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

/// Deterministic stride sample of `n` items from `items`, seeded so
/// runs are reproducible. Documents the exact selection in the TSV.
fn stride_sample(items: &[PathBuf], n: usize, seed: u64) -> Vec<PathBuf> {
    if items.is_empty() || n == 0 {
        return Vec::new();
    }
    if items.len() <= n {
        return items.to_vec();
    }
    // Even stride from a seed-derived start. Deterministic, spreads the
    // pick across the sorted list, no RNG dependency.
    let stride = items.len() as f64 / n as f64;
    let start = (seed as usize) % items.len();
    let mut picked = Vec::with_capacity(n);
    let mut seen = std::collections::BTreeSet::new();
    for k in 0..n {
        let idx = ((start as f64 + k as f64 * stride) as usize) % items.len();
        // Linear-probe to the next unseen index so duplicates from
        // rounding collisions don't shrink the sample.
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

/// Ground-truth chroma stats from RGB8 bytes, independent of the
/// analyzer. Samples every `step`-th pixel; returns the fraction of
/// sampled pixels with channel spread > CHROMA_LEVEL_THRESHOLD and the
/// maximum channel spread seen.
fn chroma_ground_truth(rgb: &[u8], step: usize) -> (f64, u8) {
    let step = step.max(1);
    let mut chroma_pixels = 0u64;
    let mut total = 0u64;
    let mut max_spread = 0u8;
    let mut i = 0usize;
    while i + 2 < rgb.len() {
        let r = rgb[i];
        let g = rgb[i + 1];
        let b = rgb[i + 2];
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        let spread = mx - mn;
        if spread > max_spread {
            max_spread = spread;
        }
        if spread > CHROMA_LEVEL_THRESHOLD {
            chroma_pixels += 1;
        }
        total += 1;
        i += 3 * step;
    }
    let frac = if total > 0 {
        chroma_pixels as f64 / total as f64
    } else {
        0.0
    };
    (frac, max_spread)
}

#[derive(Default, Clone)]
struct GateStats {
    fired: u64,
    /// Per dropped feature: worst-case (max) absolute full-analysis
    /// value observed on a gated image.
    worst_dropped: BTreeMap<String, f32>,
    /// Count of (image, feature) drop events where the full value was
    /// >= MEANINGFUL_MAGNITUDE — these are the unsafe events.
    meaningful_drop_events: u64,
    /// Worst single offending example path + feature + value.
    worst_example: Option<(String, String, f32)>,
}

impl GateStats {
    fn record_drop(&mut self, feat: &str, full_val: f32, path: &str) {
        let mag = full_val.abs();
        let slot = self.worst_dropped.entry(feat.to_string()).or_insert(0.0);
        if mag > *slot {
            *slot = mag;
        }
        if mag >= MEANINGFUL_MAGNITUDE {
            self.meaningful_drop_events += 1;
            let replace = self
                .worst_example
                .as_ref()
                .map(|(_, _, v)| mag > *v)
                .unwrap_or(true);
            if replace {
                self.worst_example = Some((path.to_string(), feat.to_string(), mag));
            }
        }
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

    // ----- Stratified sample ----------------------------------------
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
        "corpus={} | screen: {} available, {} sampled | photo/mixed: {} available, {} sampled",
        args.corpus.display(),
        screen_all.len(),
        screen_sample.len(),
        photo_all.len(),
        photo_sample.len()
    );

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }

    let supported = FeatureSet::SUPPORTED;
    let query = AnalysisQuery::new(supported);

    // Features each gate can drop, intersected with SUPPORTED (so we
    // only count features actually present in this build).
    let chroma_drop: Vec<AnalysisFeature> = chroma_drop_features()
        .iter()
        .filter(|f| supported.contains(*f))
        .collect();
    let saturating_drop: Vec<AnalysisFeature> = saturating_drop_features()
        .iter()
        .filter(|f| supported.contains(*f))
        .collect();

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("open {}: {e}", args.output.display()));
    let mut w = BufWriter::new(file);

    // Per-image columns. We list every droppable feature's full value
    // so the row is self-describing; empty when the feature was not
    // dropped on that image.
    write!(
        w,
        "path\tstratum\twidth\theight\tis_grayscale\tuniformity\t\
         chroma_frac_gt2\tmax_channel_spread\tgray_gate_fired\tuniformity_gate_fired\t\
         n_chroma_dropped\tn_saturating_dropped\tgray_gate_misfire"
    )
    .unwrap();
    for f in chroma_drop.iter() {
        write!(w, "\tdrop_C_{}", f.name()).unwrap();
    }
    for f in saturating_drop.iter() {
        write!(w, "\tdrop_S_{}", f.name()).unwrap();
    }
    writeln!(w).unwrap();

    let mut gray_gate = GateStats::default();
    let mut unif_gate = GateStats::default();
    let mut n_images = 0u64;
    let mut n_failed = 0u64;
    // Gray gate accuracy cross-check.
    let mut gray_fire_with_real_chroma = 0u64;
    let mut gray_fire_total = 0u64;

    let all: Vec<(&'static str, &PathBuf)> = screen_sample
        .iter()
        .map(|p| ("screen", p))
        .chain(photo_sample.iter().map(|p| ("photo", p)))
        .collect();

    for (stratum, path) in &all {
        let dyn_img = match ImageReader::open(path).map(|r| r.decode()) {
            Ok(Ok(img)) => img,
            _ => {
                eprintln!("skip (decode fail): {}", path.display());
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
        let slice_full = match PixelSlice::new(rgb, wd, ht, stride, PixelDescriptor::RGB8_SRGB) {
            Ok(s) => s,
            Err(_) => {
                n_failed += 1;
                continue;
            }
        };
        let slice_gated = PixelSlice::new(rgb, wd, ht, stride, PixelDescriptor::RGB8_SRGB).unwrap();

        let full = analyze_features(slice_full, &query).expect("full analyze");
        let gated = analyze_with_dispatch_plan(slice_gated, &query, None).expect("gated analyze");

        // Determine gate firing from the FULL analysis signals (the same
        // signals the dispatch plan reads internally from Tier 1).
        let is_gray = full
            .get(AnalysisFeature::IsGrayscale)
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let uniformity = full.get_f32(AnalysisFeature::Uniformity).unwrap_or(0.0);
        let gray_gate_fired = is_gray;
        let unif_gate_fired = uniformity > 0.95;

        // Ground-truth chroma (independent of the analyzer).
        // Step keeps the sample bounded on large images.
        let pix = (wd as usize) * (ht as usize);
        let step = (pix / 200_000).max(1);
        let (chroma_frac, max_spread) = chroma_ground_truth(rgb, step);

        // Gray-gate misfire: gate fired but the image carries real
        // chroma (>0.1% of sampled pixels are non-neutral). On strict
        // R==G==B grayscale this fraction is exactly 0.
        let gray_misfire = gray_gate_fired && chroma_frac > 0.001;

        if gray_gate_fired {
            gray_gate.fired += 1;
            gray_fire_total += 1;
            if chroma_frac > 0.001 {
                gray_fire_with_real_chroma += 1;
            }
        }
        if unif_gate_fired {
            unif_gate.fired += 1;
        }

        // Per-feature drop assessment. We measure the COUNTERFACTUAL:
        // "if this gate's condition fired, what value did the FULL
        // analysis compute for each feature the gate would drop?" — and
        // record that magnitude. This is independent of whether the
        // dispatch plan currently ACTS on the gate (the uniformity gate
        // is disabled in the shipped plan), so the artifact documents
        // gate safety permanently. A gate is safe iff every would-drop
        // feature is consistently negligible on the images where its
        // condition fired. We also note whether the plan actually
        // dropped the feature (parity sanity), but the safety verdict
        // is driven by the counterfactual.
        let mut n_chroma_dropped = 0u32;
        let mut n_saturating_dropped = 0u32;
        let path_str = path.display().to_string();

        let mut chroma_cells: Vec<String> = Vec::with_capacity(chroma_drop.len());
        for f in chroma_drop.iter() {
            let full_v = full.get_f32(*f);
            let plan_dropped = full_v.is_some() && gated.get_f32(*f).is_none();
            if plan_dropped {
                n_chroma_dropped += 1;
            }
            // Counterfactual: gate condition fired ⇒ assess the full
            // value the gate would discard.
            if gray_gate_fired && let Some(fv) = full_v {
                if !fv.is_nan() {
                    gray_gate.record_drop(f.name(), fv, &path_str);
                    chroma_cells.push(format!("{fv:.6}"));
                } else {
                    chroma_cells.push(String::from("nan"));
                }
            } else {
                chroma_cells.push(String::new());
            }
        }

        let mut sat_cells: Vec<String> = Vec::with_capacity(saturating_drop.len());
        for f in saturating_drop.iter() {
            let full_v = full.get_f32(*f);
            let plan_dropped = full_v.is_some() && gated.get_f32(*f).is_none();
            if plan_dropped {
                n_saturating_dropped += 1;
            }
            if unif_gate_fired && let Some(fv) = full_v {
                if !fv.is_nan() {
                    unif_gate.record_drop(f.name(), fv, &path_str);
                    sat_cells.push(format!("{fv:.6}"));
                } else {
                    sat_cells.push(String::from("nan"));
                }
            } else {
                sat_cells.push(String::new());
            }
        }

        write!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}",
            path_str,
            stratum,
            wd,
            ht,
            is_gray as u8,
            uniformity,
            chroma_frac,
            max_spread,
            gray_gate_fired as u8,
            unif_gate_fired as u8,
            n_chroma_dropped,
            n_saturating_dropped,
            gray_misfire as u8,
        )
        .ok();
        for c in &chroma_cells {
            write!(w, "\t{c}").ok();
        }
        for c in &sat_cells {
            write!(w, "\t{c}").ok();
        }
        writeln!(w).ok();

        n_images += 1;
        if n_images.is_multiple_of(20) {
            w.flush().ok();
            eprintln!("[{n_images}] processed (failed={n_failed})");
        }
    }
    w.flush().ok();

    // ----- Per-gate summary to stderr -------------------------------
    eprintln!("\n================ GATE VALIDATION SUMMARY ================");
    eprintln!("images scored: {n_images} (decode-failed/skipped: {n_failed})");
    eprintln!();
    summarize_gate(
        "GRAYSCALE (CHROMA_DROP_FEATURES)",
        &gray_gate,
        n_images,
        true,
    );
    eprintln!(
        "  grayscale-gate accuracy: fired {gray_fire_total} times, \
         {gray_fire_with_real_chroma} with real chroma (>0.1% non-neutral px) \
         => misfires={gray_fire_with_real_chroma}"
    );
    eprintln!();
    summarize_gate(
        "UNIFORMITY (SATURATING_DROP_FEATURES)",
        &unif_gate,
        n_images,
        false,
    );
    eprintln!("========================================================\n");

    eprintln!("per-image TSV written to {}", args.output.display());
    ExitCode::from(0)
}

fn summarize_gate(label: &str, g: &GateStats, n_images: u64, shipped_enabled: bool) {
    let fire_rate = if n_images > 0 {
        g.fired as f64 / n_images as f64 * 100.0
    } else {
        0.0
    };
    eprintln!("{label}");
    eprintln!(
        "  shipped: {}",
        if shipped_enabled {
            "ENABLED in dispatch plan"
        } else {
            "DISABLED in dispatch plan (counterfactual assessment below)"
        }
    );
    eprintln!(
        "  condition fire rate: {}/{} ({fire_rate:.1}%)",
        g.fired, n_images
    );
    // Worst-case would-drop-feature magnitude across all gated images
    // (counterfactual: the full-analysis value on images where the
    // gate condition fired).
    let worst = g
        .worst_dropped
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
    match worst {
        Some((feat, mag)) => {
            eprintln!("  worst-case would-drop magnitude: {mag:.6} ({feat})");
        }
        None => eprintln!("  worst-case would-drop magnitude: (condition never fired)"),
    }
    eprintln!(
        "  meaningful would-drop events (|full value| >= {MEANINGFUL_MAGNITUDE}): {}",
        g.meaningful_drop_events
    );
    if let Some((p, f, v)) = &g.worst_example {
        eprintln!("  worst offending: {f}={v:.6} on {p}");
    }
    let safe = g.meaningful_drop_events == 0;
    let verdict = match (safe, shipped_enabled) {
        (true, true) => {
            "VALIDATED-SAFE, shipped ENABLED (no would-drop feature exceeded the threshold)"
        }
        (true, false) => "SAFE on this sample but shipped DISABLED (kept off pending more data)",
        (false, true) => "REVIEW: shipped ENABLED but a would-drop feature exceeded the threshold",
        (false, false) => {
            "UNSAFE on this sample — correctly shipped DISABLED (would-drop feature exceeded the threshold)"
        }
    };
    eprintln!("  verdict: {verdict}");
}
