//! Re-extract analyzer features over an EXISTING rendition set (no re-render).
//!
//! Reads a variant-keyed TSV (`variant_name\tcontent_class\t…`) for the variant
//! list + content_class, loads each `<renditions_dir>/<variant_name>.png`, runs
//! `analyze_features_rgb8(FeatureSet::SUPPORTED)`, and writes a features TSV in
//! the SAME shape as `render_imazen26_variants` (so omni_to_pareto joins it
//! unchanged):
//!
//!   variant_name  content_class  width  height  feat_<name>…
//!
//! Built with `--features full-budgets`, the analysis runs at full sampling
//! budget (pixel_budget = usize::MAX) — the eval for whether full-budget chroma
//! sharpens subsampling discrimination on the budget-limited medium size.
//!
//!   cargo run --release --features full-budgets --example extract_existing_features -- \
//!     --vn-tsv  /mnt/v/output/clean-picker-corpus-2026-06-26/clean_features_vn.tsv \
//!     --renditions-dir /mnt/v/output/clean-picker-corpus-2026-06-26 \
//!     --out /mnt/v/output/clean-picker-corpus-2026-06-26/clean_features_vn_fullbudget.tsv \
//!     --threads 24

use image::ImageReader;
use std::env;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};

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

struct Args {
    vn_tsv: PathBuf,
    renditions_dir: PathBuf,
    out: PathBuf,
    threads: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut vn_tsv = None;
    let mut renditions_dir = None;
    let mut out = None;
    let mut threads = 16usize;
    let raw: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--vn-tsv" => {
                i += 1;
                vn_tsv = Some(PathBuf::from(&raw[i]));
            }
            "--renditions-dir" => {
                i += 1;
                renditions_dir = Some(PathBuf::from(&raw[i]));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&raw[i]));
            }
            "--threads" => {
                i += 1;
                threads = raw[i].parse().map_err(|_| "bad --threads")?;
            }
            other => return Err(format!("unknown arg {other}")),
        }
        i += 1;
    }
    Ok(Args {
        vn_tsv: vn_tsv.ok_or("missing --vn-tsv")?,
        renditions_dir: renditions_dir.ok_or("missing --renditions-dir")?,
        out: out.ok_or("missing --out")?,
        threads,
    })
}

/// (variant_name, content_class) from the reference TSV's first two columns.
fn read_variants(path: &Path) -> Result<Vec<(String, String)>, String> {
    let r = BufReader::new(File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?);
    let mut lines = r.lines();
    let header = lines
        .next()
        .ok_or("empty tsv")?
        .map_err(|e| e.to_string())?;
    let cols: Vec<&str> = header.split('\t').collect();
    let vn_i = cols.iter().position(|c| *c == "variant_name").unwrap_or(0);
    let cc_i = cols.iter().position(|c| *c == "content_class");
    let mut out = Vec::new();
    for line in lines {
        let line = line.map_err(|e| e.to_string())?;
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let vn = f.get(vn_i).copied().unwrap_or("").to_string();
        let cc = cc_i
            .and_then(|i| f.get(i).copied())
            .unwrap_or("")
            .to_string();
        if !vn.is_empty() {
            out.push((vn, cc));
        }
    }
    Ok(out)
}

fn extract_one(
    vn: &str,
    cc: &str,
    dir: &Path,
    cols: &[AnalysisFeature],
    query: &AnalysisQuery,
) -> Option<String> {
    let png = dir.join(format!("{vn}.png"));
    let img = ImageReader::open(&png).ok()?.decode().ok()?.to_rgb8();
    let (w, h) = (img.width(), img.height());
    let row = analyze_features_rgb8(img.as_raw(), w, h, query);
    let mut s = String::new();
    let _ = write!(s, "{vn}\t{cc}\t{w}\t{h}");
    for c in cols {
        let _ = write!(s, "\t{}", feat_str(&row, *c));
    }
    s.push('\n');
    Some(s)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let variants = match read_variants(&args.vn_tsv) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };
    let cols: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    let col_headers: Vec<String> = cols.iter().map(|c| format!("feat_{}", c.name())).collect();
    let n = variants.len();
    let full = cfg!(feature = "full-budgets");
    eprintln!(
        "extracting {n} variants (full_budgets={full}) over {} threads -> {}",
        args.threads,
        args.out.display()
    );
    let nthreads = args.threads.min(n.max(1));
    let chunk = n.div_ceil(nthreads).max(1);
    let cols_ref = &cols;
    let dir = args.renditions_dir.as_path();
    let results: Vec<(String, usize, usize)> = thread::scope(|s| {
        variants
            .chunks(chunk)
            .map(|ch| {
                s.spawn(move || {
                    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
                    let mut buf = String::new();
                    let (mut ok, mut fail) = (0usize, 0usize);
                    for (vn, cc) in ch {
                        match extract_one(vn, cc, dir, cols_ref, &query) {
                            Some(r) => {
                                buf.push_str(&r);
                                ok += 1;
                            }
                            None => {
                                eprintln!("skip (load/analyze fail): {vn}");
                                fail += 1;
                            }
                        }
                    }
                    (buf, ok, fail)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });
    let mut out = match File::create(&args.out) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: create {}: {e}", args.out.display());
            return ExitCode::from(1);
        }
    };
    let _ = writeln!(
        out,
        "variant_name\tcontent_class\twidth\theight\t{}",
        col_headers.join("\t")
    );
    let (mut tot_ok, mut tot_fail) = (0usize, 0usize);
    for (buf, ok, fail) in &results {
        let _ = out.write_all(buf.as_bytes());
        tot_ok += ok;
        tot_fail += fail;
    }
    eprintln!(
        "done: {tot_ok} ok, {tot_fail} failed -> {}",
        args.out.display()
    );
    if tot_ok == 0 {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
