//! Extract zenanalyze features on the **16-bit PQ/HLG HDR renditions** produced
//! by `hdr-corpus-convert` (`*.hdr.png`), via the generic bit-depth/transfer-aware
//! `analyze_features` so the **depth tier** (peak nits, hdr_present, headroom,
//! wide-gamut, effective_bit_depth) reads real HDR signal — the features that are
//! dead/constant in the SDR-only dataset.
//!
//! Per file: parse the PNG `cICP` chunk for primaries/transfer, decode the 16-bit
//! RGB (the `image` crate yields native-endian u16), wrap it in a u16 `PixelSlice`
//! with the matching `PixelDescriptor` (`RGB16` + transfer + primaries), and run
//! `analyze_features(FeatureSet::SUPPORTED)`. Build `--features experimental,hdr`.
//!
//! Usage:
//!   cargo run --release --features experimental,hdr --example extract_hdr_features -- \
//!     --hdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
//!     --out     /mnt/v/output/imazen-26-features/imazen26_hdr_features_2026-06-14.tsv

use image::ImageReader;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};
use zenpixels::{ColorPrimaries, PixelDescriptor, PixelSlice, TransferFunction};

/// Parse the PNG `cICP` chunk → (primaries_code, transfer_code), if present.
fn read_cicp(bytes: &[u8]) -> Option<(u8, u8)> {
    let mut i = 8; // skip signature
    while i + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[i..i + 4].try_into().ok()?) as usize;
        let typ = &bytes[i + 4..i + 8];
        if typ == b"cICP" && i + 8 + 4 <= bytes.len() {
            let d = &bytes[i + 8..i + 12];
            return Some((d[0], d[1]));
        }
        if typ == b"IDAT" {
            break;
        }
        i += 12 + len; // len + type(4) + crc(4) + data(len)
    }
    None
}

fn map_primaries(code: u8) -> ColorPrimaries {
    match code {
        9 => ColorPrimaries::Bt2020,
        12 => ColorPrimaries::DisplayP3,
        _ => ColorPrimaries::Bt709, // 1 (and fallback)
    }
}
fn map_transfer(code: u8) -> TransferFunction {
    match code {
        18 => TransferFunction::Hlg,
        _ => TransferFunction::Pq, // 16 (and fallback — these are HDR renditions)
    }
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

fn collect_hdr(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_hdr(&p, out);
            } else if p.to_string_lossy().ends_with(".hdr.png") {
                out.push(p);
            }
        }
    }
}

fn main() -> ExitCode {
    let mut hdr_dir = None;
    let mut out = None;
    let raw: Vec<String> = env::args().collect();
    let mut it = raw.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hdr-dir" => hdr_dir = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            _ => {}
        }
    }
    let (hdr_dir, out) = match (hdr_dir, out) {
        (Some(d), Some(o)) => (d, o),
        _ => {
            eprintln!("usage: --hdr-dir DIR --out TSV");
            return ExitCode::from(2);
        }
    };
    let mut files = Vec::new();
    collect_hdr(&hdr_dir, &mut files);
    files.sort();
    let cols: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    eprintln!(
        "{} .hdr.png files, {} features/row",
        files.len(),
        cols.len()
    );

    let f = File::create(&out).unwrap_or_else(|e| panic!("create {}: {e}", out.display()));
    let mut w = BufWriter::new(f);
    write!(
        w,
        "variant_name\tcontent_class\twidth\theight\tprimaries\ttransfer"
    )
    .ok();
    for c in &cols {
        write!(w, "\tfeat_{}", c.name()).ok();
    }
    writeln!(w).ok();

    let (mut ok, mut fail) = (0usize, 0usize);
    for (idx, p) in files.iter().enumerate() {
        let raw = match std::fs::read(p) {
            Ok(b) => b,
            Err(_) => {
                fail += 1;
                continue;
            }
        };
        let (pc, tc) = read_cicp(&raw).unwrap_or((1, 16)); // default 709-PQ
        let img = match ImageReader::open(p).map(|r| r.decode()) {
            Ok(Ok(i)) => i.into_rgb16(),
            _ => {
                eprintln!("skip (decode): {}", p.display());
                fail += 1;
                continue;
            }
        };
        let (wd, ht) = (img.width(), img.height());
        let u16s = img.into_raw(); // native-endian u16, RGB
        let bytes: Vec<u8> = u16s.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let desc = PixelDescriptor::RGB16
            .with_transfer(map_transfer(tc))
            .with_primaries(map_primaries(pc));
        let slice = match PixelSlice::new(&bytes, wd, ht, (wd * 6) as usize, desc) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip (slice {}): {e:?}", p.display());
                fail += 1;
                continue;
            }
        };
        let r = match analyze_features(slice, &query) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skip (analyze {}): {e:?}", p.display());
                fail += 1;
                continue;
            }
        };
        let stem = p.file_name().unwrap().to_string_lossy();
        let variant = stem.trim_end_matches(".hdr.png");
        let cclass = p
            .parent()
            .and_then(|d| d.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        write!(w, "{variant}\t{cclass}\t{wd}\t{ht}\t{pc}\t{tc}").ok();
        for c in &cols {
            write!(w, "\t{}", feat_str(&r, *c)).ok();
        }
        writeln!(w).ok();
        ok += 1;
        if (idx + 1) % 25 == 0 {
            w.flush().ok();
            eprintln!("[{}/{}] ok={ok} fail={fail}", idx + 1, files.len());
        }
    }
    w.flush().ok();
    eprintln!("final: ok={ok} fail={fail} -> {}", out.display());
    ExitCode::from(0)
}
