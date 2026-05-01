//! Feature extractor for picker-training Pareto sweeps that key by
//! `(image_path, size_class)`.
//!
//! Reads a TSV manifest (sha256, split, content_class, source,
//! size_bytes, path), resizes each image to N target maxdim sizes via
//! Lanczos3 (matching how `jxl-encoder/examples/lossy_pareto_calibrate`
//! and `zenavif/examples/extract_features` do it), runs
//! `zenanalyze::analyze_features_rgb8` with `FeatureSet::SUPPORTED`,
//! and emits the standard zentrain features TSV schema:
//!
//!   image_path  size_class  width  height  feat_<name>...
//!
//! plus the manifest pass-through columns:
//!
//!   image_sha  split  content_class  source
//!
//! This binary lives in the zenanalyze workspace so we don't have to
//! reach into a sibling crate's target dir / lockfile.
//!
//! Usage:
//!
//!     cargo run --release --example extract_features_for_picker -- \
//!         --manifest /tmp/zenjxl_manifest.tsv \
//!         --output ~/work/zen/zenjxl/benchmarks/zenjxl_lossy_features_2026-05-01.tsv \
//!         --sizes 64,256,1024,4096 \
//!         --image-path-from-sha
//!
//! With `--image-path-from-sha`, the `image_path` column is
//! synthesized as `sha:<first-16-chars-of-sha>` so it joins against
//! the matching `zenjxl_oracle_adapter.py` output. Without that flag
//! the manifest's actual filesystem path is emitted.

use image::{DynamicImage, GenericImageView, ImageReader, imageops::FilterType};
use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};

#[derive(Clone, Debug)]
struct ManifestEntry {
    sha256: String,
    split: String,
    content_class: String,
    source: String,
    path: PathBuf,
}

#[derive(Clone, Debug)]
struct Args {
    manifest: PathBuf,
    output: PathBuf,
    sizes: Vec<u32>,
    image_path_from_sha: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut manifest = None;
        let mut output = None;
        // Default matches `jxl-encoder/examples/lossy_pareto_calibrate.rs`:
        // 0 = native (no resize). Override on the CLI with --sizes.
        let mut sizes: Vec<u32> = vec![64, 256, 1024, 0];
        let mut image_path_from_sha = false;
        let raw: Vec<String> = env::args().collect();
        let mut iter = raw.iter().skip(1);
        while let Some(a) = iter.next() {
            match a.as_str() {
                "-h" | "--help" => {
                    eprintln!(
                        "Usage: extract_features_for_picker --manifest PATH --output PATH \
                         [--sizes 64,256,...] [--image-path-from-sha]"
                    );
                    std::process::exit(0);
                }
                "--manifest" => manifest = iter.next().map(PathBuf::from),
                "--output" => output = iter.next().map(PathBuf::from),
                "--sizes" => {
                    sizes = iter
                        .next()
                        .ok_or("--sizes")?
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            if s == "native" || s == "0" {
                                0
                            } else {
                                s.parse().expect("size must be uint or 'native'")
                            }
                        })
                        .collect();
                }
                "--image-path-from-sha" => image_path_from_sha = true,
                other => return Err(format!("unknown arg {other}")),
            }
        }
        let manifest = manifest.ok_or("--manifest PATH required")?;
        let output = output.ok_or("--output PATH required")?;
        Ok(Args {
            manifest,
            output,
            sizes,
            image_path_from_sha,
        })
    }
}

fn read_manifest(path: &Path) -> Result<Vec<ManifestEntry>, String> {
    let f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    let mut header_idx: HashMap<String, usize> = HashMap::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.map_err(|e| format!("read line {i}: {e}"))?;
        let cols: Vec<&str> = line.split('\t').collect();
        if i == 0 {
            for (idx, name) in cols.iter().enumerate() {
                header_idx.insert(name.to_string(), idx);
            }
            continue;
        }
        let get = |k: &str| {
            header_idx
                .get(k)
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

/// Same Lanczos3 resize-to-maxdim policy as
/// `jxl-encoder/examples/lossy_pareto_calibrate.rs::resize_to`.
///
/// `target_maxdim == 0` means "native — no resize". When the source
/// is already <= target, return as-is (no upscale, no skip — this
/// matches the pareto sweep's behaviour of treating "image fits
/// inside target bucket" as the same as native). The size_class
/// label is determined by the *target* bucket, NOT the actual
/// dimensions, again matching the calibrate script.
fn resize_to_maxdim(src: &DynamicImage, target_maxdim: u32) -> DynamicImage {
    let (w, h) = src.dimensions();
    if target_maxdim == 0 || w.max(h) <= target_maxdim {
        return src.clone();
    }
    let ratio = target_maxdim as f64 / w.max(h) as f64;
    let new_w = ((w as f64) * ratio).round().max(1.0) as u32;
    let new_h = ((h as f64) * ratio).round().max(1.0) as u32;
    src.resize_exact(new_w, new_h, FilterType::Lanczos3)
}

fn size_class_label(target_size: u32) -> &'static str {
    // Match lossy_pareto_calibrate.rs label conventions.
    match target_size {
        64 => "tiny",
        256 => "small",
        1024 => "medium",
        4096 | 0 => "large",
        _ => "other",
    }
}

fn feature_value_str(
    analysis: &zenanalyze::feature::AnalysisResults,
    f: AnalysisFeature,
) -> String {
    if let Some(v) = analysis.get_f32(f) {
        // NaN sentinel from the analyzer means "not enough samples for
        // this percentile feature on this image" — pass it through as
        // empty so the downstream Python loader treats it consistently.
        if v.is_nan() {
            return String::new();
        }
        format!("{v:.6}")
    } else if let Some(v) = analysis.get(f) {
        match v {
            FeatureValue::F32(x) => {
                if x.is_nan() {
                    String::new()
                } else {
                    format!("{x:.6}")
                }
            }
            FeatureValue::U32(x) => format!("{x}"),
            FeatureValue::Bool(b) => format!("{}", b as u8),
            _ => String::new(),
        }
    } else {
        String::new()
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

    let manifest = match read_manifest(&args.manifest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    eprintln!(
        "manifest: {} images, sizes: {:?}, image_path_from_sha={}",
        manifest.len(),
        args.sizes,
        args.image_path_from_sha
    );

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }

    let cols: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    eprintln!("extracting {} features per (image, size)", cols.len());

    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&args.output)
        .unwrap_or_else(|e| panic!("open {}: {e}", args.output.display()));
    let mut w = BufWriter::new(file);

    write!(
        w,
        "image_path\timage_sha\tsplit\tcontent_class\tsource\tsize_class\twidth\theight"
    )
    .unwrap();
    for c in &cols {
        write!(w, "\tfeat_{}", c.name()).unwrap();
    }
    writeln!(w).unwrap();

    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let mut total_done = 0usize;
    let mut total_failed = 0usize;

    for (idx, entry) in manifest.iter().enumerate() {
        let dyn_img = match ImageReader::open(&entry.path).and_then(|r| Ok(r.decode())) {
            Ok(Ok(img)) => img,
            _ => {
                eprintln!("skip (decode fail): {}", entry.path.display());
                total_failed += 1;
                continue;
            }
        };

        for &target in &args.sizes {
            let resized = resize_to_maxdim(&dyn_img, target);
            let (rw, rh) = resized.dimensions();
            let rgb8 = resized.to_rgb8();
            let row = analyze_features_rgb8(rgb8.as_raw(), rw, rh, &query);

            let image_path = if args.image_path_from_sha {
                format!("sha:{}", &entry.sha256[..16])
            } else {
                entry.path.display().to_string()
            };
            let size_class = size_class_label(target);

            write!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                image_path,
                entry.sha256,
                entry.split,
                entry.content_class,
                entry.source,
                size_class,
                rw,
                rh
            )
            .ok();
            for c in &cols {
                write!(w, "\t{}", feature_value_str(&row, *c)).ok();
            }
            writeln!(w).ok();
            total_done += 1;
        }

        if (idx + 1) % 10 == 0 {
            w.flush().ok();
            eprintln!("[{}/{}] done={total_done} failed={total_failed}", idx + 1, manifest.len());
        }
    }

    w.flush().ok();
    eprintln!("final: done={total_done} failed={total_failed}");
    ExitCode::from(0)
}
