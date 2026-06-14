//! Full-feature extractor for the imazen-26 corpus with zoom/region
//! crop derivatives.
//!
//! For each manifest image it generates source variants:
//!   - `full`            — the whole image
//!   - `c50_{center,tl,tr,bl,br}` — a 50%×50% window at 5 positions
//!   - `c25_{center,tl,tr,bl,br}` — a 25%×25% window at 5 positions
//!
//! Each variant is then run through a dense max-dim resize grid
//! (Lanczos3), emitting only **downscales** (`target < variant_maxdim`)
//! plus a `native` row — never an upscale (which would just duplicate
//! the native row). `zenanalyze::analyze_features_rgb8` runs with
//! `FeatureSet::SUPPORTED`, so building `--features experimental,hdr`
//! emits ALL features (incl. palette_density / xyb444 / xyb_bquarter /
//! HDR-depth).
//!
//! Output TSV schema:
//!   image_path  image_sha  split  content_class  source
//!   crop_label  size_class  width  height  feat_<name>...
//!
//! Usage:
//!   cargo run --release --features experimental,hdr \
//!     --example extract_features_imazen26_crops -- \
//!     --manifest /mnt/v/output/imazen-26-features/imazen26_manifest.tsv \
//!     --output   /mnt/v/output/imazen-26-features/imazen26_features.tsv \
//!     [--sizes 32,48,...,4096] [--crop-fractions 0.5,0.25] [--limit N]

use image::{DynamicImage, GenericImageView, ImageReader, imageops::FilterType};
use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};

struct ManifestEntry {
    sha256: String,
    split: String,
    content_class: String,
    source: String,
    path: PathBuf,
}

struct Args {
    manifest: PathBuf,
    output: PathBuf,
    sizes: Vec<u32>,
    fractions: Vec<f64>,
    limit: usize,
    threads: usize,
}

const DENSE_SIZES: &[u32] = &[
    32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];

impl Args {
    fn parse() -> Result<Self, String> {
        let mut manifest = None;
        let mut output = None;
        let mut sizes: Vec<u32> = DENSE_SIZES.to_vec();
        let mut fractions: Vec<f64> = vec![0.5, 0.25];
        let mut limit = usize::MAX;
        // Default to run-heavy's core cap (RAYON_NUM_THREADS=nproc-4) when
        // present, else available parallelism minus a little headroom.
        let mut threads = env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                thread::available_parallelism()
                    .ok()
                    .map(|n| n.get().saturating_sub(4).max(1))
            })
            .unwrap_or(8);
        let raw: Vec<String> = env::args().collect();
        let mut iter = raw.iter().skip(1);
        while let Some(a) = iter.next() {
            match a.as_str() {
                "-h" | "--help" => return Err("see file header for usage".into()),
                "--manifest" => manifest = iter.next().map(PathBuf::from),
                "--output" => output = iter.next().map(PathBuf::from),
                "--sizes" => {
                    sizes = iter
                        .next()
                        .ok_or("--sizes needs a value")?
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
                "--crop-fractions" => {
                    fractions = iter
                        .next()
                        .ok_or("--crop-fractions needs a value")?
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
                "--limit" => {
                    limit = iter
                        .next()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(usize::MAX);
                }
                "--threads" => {
                    threads = iter
                        .next()
                        .and_then(|s| s.parse().ok())
                        .filter(|&n| n >= 1)
                        .unwrap_or(threads);
                }
                other => return Err(format!("unknown arg {other}")),
            }
        }
        Ok(Args {
            manifest: manifest.ok_or("--manifest required")?,
            output: output.ok_or("--output required")?,
            sizes,
            fractions,
            limit,
            threads,
        })
    }
}

fn read_manifest(path: &Path) -> Result<Vec<ManifestEntry>, String> {
    let r = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let mut out = Vec::new();
    let mut hdr: HashMap<String, usize> = HashMap::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.map_err(|e| format!("line {i}: {e}"))?;
        let cols: Vec<&str> = line.split('\t').collect();
        if i == 0 {
            for (idx, name) in cols.iter().enumerate() {
                hdr.insert(name.to_string(), idx);
            }
            continue;
        }
        let get = |k: &str| {
            hdr.get(k)
                .and_then(|&idx| cols.get(idx).copied())
                .unwrap_or("")
                .to_string()
        };
        let path_str = get("path");
        if path_str.is_empty() {
            continue;
        }
        out.push(ManifestEntry {
            sha256: get("sha256"),
            split: get("split"),
            content_class: get("content_class"),
            source: get("source"),
            path: PathBuf::from(path_str),
        });
    }
    Ok(out)
}

/// Crop variants: `full` + `c{frac}_{pos}` for each fraction × position.
/// Window is `frac·W × frac·H` (aspect-preserving), at center + 4 corners.
fn crop_variants(src: &DynamicImage, fractions: &[f64]) -> Vec<(String, DynamicImage)> {
    let (w, h) = src.dimensions();
    let mut v = vec![("full".to_string(), src.clone())];
    for &f in fractions {
        let cw = ((w as f64 * f).round() as u32).clamp(1, w);
        let ch = ((h as f64 * f).round() as u32).clamp(1, h);
        let pct = (f * 100.0).round() as u32;
        let positions: [(&str, u32, u32); 5] = [
            ("center", (w - cw) / 2, (h - ch) / 2),
            ("tl", 0, 0),
            ("tr", w - cw, 0),
            ("bl", 0, h - ch),
            ("br", w - cw, h - ch),
        ];
        for (pos, x, y) in positions {
            v.push((format!("c{pct}_{pos}"), src.crop_imm(x, y, cw, ch)));
        }
    }
    v
}

/// Lanczos3 downscale to `target` maxdim; `target == 0` → native clone.
fn resize_to_maxdim(src: &DynamicImage, target: u32) -> DynamicImage {
    let (w, h) = src.dimensions();
    if target == 0 || w.max(h) <= target {
        return src.clone();
    }
    let ratio = target as f64 / w.max(h) as f64;
    let nw = ((w as f64) * ratio).round().max(1.0) as u32;
    let nh = ((h as f64) * ratio).round().max(1.0) as u32;
    src.resize_exact(nw, nh, FilterType::Lanczos3)
}

fn feature_value_str(a: &zenanalyze::feature::AnalysisResults, f: AnalysisFeature) -> String {
    if let Some(v) = a.get_f32(f) {
        if v.is_nan() {
            return String::new();
        }
        return format!("{v:.6}");
    }
    match a.get(f) {
        Some(FeatureValue::F32(x)) if !x.is_nan() => format!("{x:.6}"),
        Some(FeatureValue::U32(x)) => format!("{x}"),
        Some(FeatureValue::Bool(b)) => format!("{}", b as u8),
        _ => String::new(),
    }
}

/// Process one manifest entry → its TSV rows (no header) + row count.
/// Returns `None` if the image fails to decode.
fn process_entry(
    e: &ManifestEntry,
    cols: &[AnalysisFeature],
    sizes: &[u32],
    fractions: &[f64],
) -> Option<(String, usize)> {
    let img = ImageReader::open(&e.path).ok()?.decode().ok()?;
    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let mut buf = String::new();
    let mut count = 0usize;
    for (crop_label, variant) in crop_variants(&img, fractions) {
        let (vw, vh) = variant.dimensions();
        let vmax = vw.max(vh);
        // native (0) + every target strictly below the variant's maxdim —
        // never an upscale (that would just dup the native row).
        for &target in std::iter::once(&0u32).chain(sizes.iter()) {
            if target != 0 && target >= vmax {
                continue;
            }
            let resized = resize_to_maxdim(&variant, target);
            let (rw, rh) = resized.dimensions();
            let rgb8 = resized.to_rgb8();
            let row = analyze_features_rgb8(rgb8.as_raw(), rw, rh, &query);
            let size_class = if target == 0 {
                "native".to_string()
            } else {
                target.to_string()
            };
            let _ = write!(
                buf,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                e.path.display(),
                e.sha256,
                e.split,
                e.content_class,
                e.source,
                crop_label,
                size_class,
                rw,
                rh
            );
            for c in cols {
                let _ = write!(buf, "\t{}", feature_value_str(&row, *c));
            }
            buf.push('\n');
            count += 1;
        }
    }
    Some((buf, count))
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let manifest = match read_manifest(&args.manifest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let cols: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    eprintln!(
        "manifest {} images, {} features/row, sizes {:?}, fractions {:?}",
        manifest.len(),
        cols.len(),
        args.sizes,
        args.fractions
    );
    if let Some(p) = args.output.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("open {}: {e}", args.output.display()));
    let mut w = BufWriter::new(file);
    write!(
        w,
        "image_path\timage_sha\tsplit\tcontent_class\tsource\tcrop_label\tsize_class\twidth\theight"
    )
    .unwrap();
    for c in &cols {
        write!(w, "\tfeat_{}", c.name()).unwrap();
    }
    writeln!(w).unwrap();

    let entries: Vec<&ManifestEntry> = manifest.iter().take(args.limit).collect();
    let n = entries.len();
    let nthreads = args.threads.min(n.max(1));
    let chunk_size = n.div_ceil(nthreads).max(1);
    eprintln!("processing {n} images across {nthreads} threads");

    let cols_ref = &cols;
    let sizes_ref = &args.sizes;
    let fractions_ref = &args.fractions;
    let results: Vec<(String, usize, usize)> = thread::scope(|s| {
        let handles: Vec<_> = entries
            .chunks(chunk_size)
            .map(|chunk| {
                s.spawn(move || {
                    let mut buf = String::new();
                    let (mut done, mut failed) = (0usize, 0usize);
                    for e in chunk {
                        match process_entry(e, cols_ref, sizes_ref, fractions_ref) {
                            Some((rows, count)) => {
                                buf.push_str(&rows);
                                done += count;
                            }
                            None => {
                                eprintln!("skip (decode fail): {}", e.path.display());
                                failed += 1;
                            }
                        }
                    }
                    (buf, done, failed)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    let (mut total_done, mut total_failed) = (0usize, 0usize);
    for (buf, done, failed) in &results {
        w.write_all(buf.as_bytes()).ok();
        total_done += done;
        total_failed += failed;
    }
    w.flush().ok();
    eprintln!("final: rows={total_done} failed={total_failed}");
    ExitCode::from(0)
}
