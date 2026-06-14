//! Render the budget-selected imazen-26 training variants to actual image
//! files via **zenresize Mitchell-sharp**, and re-extract zenanalyze features
//! on the Mitchell renditions (so the training features describe the exact
//! pixels that get encoded — not the Lanczos3 proxy used during selection).
//!
//! Input: the FPS-ordered variant manifest from `imazen26_budget_select`
//! (`rank cumulative_gp image_path scale_w scale_h megapixels content_class
//! variant_name`). Renders the prefix with `cumulative_gp <= --max-gp`.
//!
//! Each variant: load source `.sdr.png` → if target == source dims, copy;
//! else zenresize `Filter::Mitchell` + `resize_sharpen(--sharpen)` downscale →
//! write `<variant_name>.png` → `analyze_features_rgb8(FeatureSet::SUPPORTED)`.
//! Build `--features experimental,hdr` to emit all features.
//!
//! SDR only (the `.sdr.png` mirror); HDR renditions wait on the heic/UltraHDR
//! decode path landing in zencodecs.
//!
//! Usage:
//!   cargo run --release --features experimental,hdr \
//!     --example render_imazen26_variants -- \
//!     --manifest /mnt/v/output/imazen-26-features/imazen26_train_variants_2026-06-14.tsv \
//!     --out-dir  /mnt/v/output/imazen-26-features/train_renditions_2026-06-14 \
//!     --features-out /mnt/v/output/imazen-26-features/imazen26_train_features_2026-06-14.tsv \
//!     [--max-gp 1.5] [--sharpen 10.0] [--limit N] [--threads N]

use image::{ImageReader, RgbImage};
use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

struct Variant {
    image_path: PathBuf,
    scale_w: u32,
    scale_h: u32,
    content_class: String,
    variant_name: String,
}

struct Args {
    manifest: PathBuf,
    out_dir: PathBuf,
    features_out: PathBuf,
    max_gp: f64,
    sharpen: f32,
    limit: usize,
    threads: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut m = None;
        let mut out_dir = None;
        let mut features_out = None;
        let mut max_gp = 1.5;
        let mut sharpen = 10.0;
        let mut limit = usize::MAX;
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
        let mut it = raw.iter().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--manifest" => m = it.next().map(PathBuf::from),
                "--out-dir" => out_dir = it.next().map(PathBuf::from),
                "--features-out" => features_out = it.next().map(PathBuf::from),
                "--max-gp" => max_gp = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_gp),
                "--sharpen" => sharpen = it.next().and_then(|s| s.parse().ok()).unwrap_or(sharpen),
                "--limit" => limit = it.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
                "--threads" => {
                    threads = it
                        .next()
                        .and_then(|s| s.parse().ok())
                        .filter(|&n| n >= 1)
                        .unwrap_or(threads)
                }
                "-h" | "--help" => return Err("see file header for usage".into()),
                other => return Err(format!("unknown arg {other}")),
            }
        }
        Ok(Args {
            manifest: m.ok_or("--manifest required")?,
            out_dir: out_dir.ok_or("--out-dir required")?,
            features_out: features_out.ok_or("--features-out required")?,
            max_gp,
            sharpen,
            limit,
            threads,
        })
    }
}

fn read_manifest(path: &Path, max_gp: f64, limit: usize) -> Result<Vec<Variant>, String> {
    let r = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let mut hdr: HashMap<String, usize> = HashMap::new();
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.map_err(|e| format!("line {i}: {e}"))?;
        let c: Vec<&str> = line.split('\t').collect();
        if i == 0 {
            for (idx, n) in c.iter().enumerate() {
                hdr.insert(n.to_string(), idx);
            }
            continue;
        }
        let g = |k: &str| hdr.get(k).and_then(|&j| c.get(j).copied()).unwrap_or("");
        if g("cumulative_gp").parse::<f64>().unwrap_or(f64::MAX) > max_gp {
            continue;
        }
        out.push(Variant {
            image_path: PathBuf::from(g("image_path")),
            scale_w: g("scale_w").parse().unwrap_or(0),
            scale_h: g("scale_h").parse().unwrap_or(0),
            content_class: g("content_class").to_string(),
            variant_name: g("variant_name").to_string(),
        });
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

fn feat_str(a: &zenanalyze::feature::AnalysisResults, f: AnalysisFeature) -> String {
    if let Some(v) = a.get_f32(f) {
        return if v.is_nan() {
            String::new()
        } else {
            format!("{v:.6}")
        };
    }
    match a.get(f) {
        Some(FeatureValue::F32(x)) if !x.is_nan() => format!("{x:.6}"),
        Some(FeatureValue::U32(x)) => format!("{x}"),
        Some(FeatureValue::Bool(b)) => format!("{}", b as u8),
        _ => String::new(),
    }
}

/// Render one variant to `<out_dir>/<variant_name>.png` and return its
/// features row, or `None` on decode/render failure.
fn render_one(
    v: &Variant,
    out_dir: &Path,
    sharpen: f32,
    cols: &[AnalysisFeature],
    query: &AnalysisQuery,
) -> Option<String> {
    let img = ImageReader::open(&v.image_path)
        .ok()?
        .decode()
        .ok()?
        .to_rgb8();
    let (sw, sh) = (img.width(), img.height());
    let (dw, dh) = (v.scale_w, v.scale_h);
    let out_rgb: Vec<u8> = if (dw, dh) == (sw, sh) {
        img.into_raw() // native: copy, no resample/sharpen
    } else {
        let cfg = ResizeConfig::builder(sw, sh, dw, dh)
            .filter(Filter::Mitchell)
            .resize_sharpen(sharpen)
            .format(PixelDescriptor::RGB8_SRGB)
            .build();
        Resizer::new(&cfg).resize(img.as_raw())
    };
    let rendered = RgbImage::from_raw(dw, dh, out_rgb.clone())?;
    rendered
        .save(out_dir.join(format!("{}.png", v.variant_name)))
        .ok()?;
    let row = analyze_features_rgb8(&out_rgb, dw, dh, query);
    let mut s = String::new();
    let _ = write!(s, "{}\t{}\t{}\t{}", v.variant_name, v.content_class, dw, dh);
    for c in cols {
        let _ = write!(s, "\t{}", feat_str(&row, *c));
    }
    s.push('\n');
    Some(s)
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let variants = match read_manifest(&args.manifest, args.max_gp, args.limit) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    std::fs::create_dir_all(&args.out_dir).ok();
    let cols: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    let n = variants.len();
    let nthreads = args.threads.min(n.max(1));
    let chunk = n.div_ceil(nthreads).max(1);
    eprintln!(
        "rendering {n} variants @<= {} GP via Mitchell+sharpen({}) across {nthreads} threads -> {}",
        args.max_gp,
        args.sharpen,
        args.out_dir.display()
    );

    let cols_ref = &cols;
    let out_dir = args.out_dir.as_path();
    let results: Vec<(String, usize, usize)> = thread::scope(|s| {
        variants
            .chunks(chunk)
            .map(|ch| {
                s.spawn(move || {
                    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
                    let mut buf = String::new();
                    let (mut ok, mut fail) = (0usize, 0usize);
                    for v in ch {
                        match render_one(v, out_dir, args.sharpen, cols_ref, &query) {
                            Some(r) => {
                                buf.push_str(&r);
                                ok += 1;
                            }
                            None => {
                                eprintln!("skip (render fail): {}", v.variant_name);
                                fail += 1;
                            }
                        }
                    }
                    (buf, ok, fail)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    use std::io::Write as _;
    let f = File::create(&args.features_out)
        .unwrap_or_else(|e| panic!("create {}: {e}", args.features_out.display()));
    let mut w = std::io::BufWriter::new(f);
    write!(w, "variant_name\tcontent_class\twidth\theight").ok();
    for c in &cols {
        write!(w, "\tfeat_{}", c.name()).ok();
    }
    writeln!(w).ok();
    let (mut tok, mut tfail) = (0usize, 0usize);
    for (buf, ok, fail) in &results {
        w.write_all(buf.as_bytes()).ok();
        tok += ok;
        tfail += fail;
    }
    w.flush().ok();
    eprintln!(
        "final: rendered {tok}, failed {tfail} -> {}",
        args.features_out.display()
    );
    ExitCode::from(0)
}
