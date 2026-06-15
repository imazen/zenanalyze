//! HDR size grid — bit-depth-preserving, LINEAR-LIGHT downscales of the 16-bit
//! PQ/HLG HDR renditions (from `hdr-corpus-convert`), mirroring the SDR
//! Mitchell-sharp render grid, with zenanalyze features (depth tier live)
//! extracted per rendition. Completes the HDR training corpus across sizes,
//! not just native resolution.
//!
//! Why not `resize_u16`: it hardcodes the sRGB transfer to linearize, which is
//! WRONG for PQ/HLG and would corrupt highlights. So this linearizes via the
//! correct transfer (`linear_srgb::tf::{pq,hlg}_to_linear`), resizes in linear
//! f32 (zenresize Mitchell + `resize_sharpen`), then re-encodes to 16-bit
//! PQ/HLG. Native size is read straight from the source u16 (no round-trip) so
//! it matches the `imazen26_hdr_features` native rows exactly.
//!
//! Renditions (`--out-dir`) are written via the `image` crate (correct pixels +
//! endianness), then a `cICP` chunk is spliced in after IHDR so they stay
//! self-describing (pixel data untouched). Build `--features experimental,hdr`.
//!
//! Usage:
//!   cargo run --release --features experimental,hdr \
//!     --example extract_hdr_size_grid -- \
//!     --hdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
//!     --features-out /mnt/v/output/imazen-26-features/imazen26_hdr_grid_features_2026-06-14.tsv \
//!     [--out-dir /mnt/v/output/imazen-26-hdr-grid-2026-06-14] [--sharpen 10] [--threads N] [--limit N]

use image::{ImageBuffer, ImageReader, Rgb};
use linear_srgb::tf::{hlg_to_linear, linear_to_hlg, linear_to_pq, pq_to_linear};
use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};
use zenpixels::{ColorPrimaries, PixelDescriptor, PixelSlice, TransferFunction};
use zenresize::{Filter, ResizeConfig, Resizer};

/// Longest-edge downscale targets (log-spaced ~1.25×). Downscale-only; native
/// is always emitted. Honors the dense-size sweep discipline (~16 sizes/source).
const LONGEST_EDGES: &[u32] = &[
    128, 160, 200, 256, 320, 400, 512, 640, 800, 1024, 1280, 1600, 2048, 2560, 3072,
];

/// Parse the PNG `cICP` chunk → (primaries_code, transfer_code), if present.
fn read_cicp(bytes: &[u8]) -> Option<(u8, u8)> {
    let mut i = 8;
    while i + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[i..i + 4].try_into().ok()?) as usize;
        let typ = &bytes[i + 4..i + 8];
        if typ == b"cICP" && i + 12 <= bytes.len() {
            let d = &bytes[i + 8..i + 12];
            return Some((d[0], d[1]));
        }
        if typ == b"IDAT" {
            break;
        }
        i += 12 + len;
    }
    None
}

fn map_primaries(code: u8) -> ColorPrimaries {
    match code {
        9 => ColorPrimaries::Bt2020,
        12 => ColorPrimaries::DisplayP3,
        _ => ColorPrimaries::Bt709,
    }
}
fn map_transfer(code: u8) -> TransferFunction {
    match code {
        18 => TransferFunction::Hlg,
        _ => TransferFunction::Pq,
    }
}

/// 16-bit code → linear (normalized), via the encoded transfer.
fn to_linear(code: u16, tc: u8) -> f32 {
    let v = code as f32 / 65535.0;
    if tc == 18 {
        hlg_to_linear(v)
    } else {
        pq_to_linear(v)
    }
}
/// linear (normalized) → 16-bit code, via the encoded transfer. Clamps to the
/// valid [0,1] PQ/HLG domain (sharpen overshoot can leave it).
fn from_linear(lin: f32, tc: u8) -> u16 {
    let l = lin.clamp(0.0, 1.0);
    let v = if tc == 18 {
        linear_to_hlg(l)
    } else {
        linear_to_pq(l)
    };
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// PNG CRC-32 (ISO-HDLC), bitwise — only used for the tiny cICP chunk.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Splice a `cICP` chunk in after IHDR (offset 8 sig + 25 IHDR = 33). Pixel /
/// IDAT data is after the insertion point and is left byte-for-byte intact.
fn inject_cicp(path: &Path, pc: u8, tc: u8) -> std::io::Result<()> {
    let mut bytes = std::fs::read(path)?;
    let data = [pc, tc, 0u8, 1u8]; // primaries, transfer, matrix=0, full-range
    let mut crc_in = Vec::with_capacity(8);
    crc_in.extend_from_slice(b"cICP");
    crc_in.extend_from_slice(&data);
    let mut chunk = Vec::with_capacity(16);
    chunk.extend_from_slice(&4u32.to_be_bytes());
    chunk.extend_from_slice(b"cICP");
    chunk.extend_from_slice(&data);
    chunk.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    let ins = 33.min(bytes.len());
    bytes.splice(ins..ins, chunk);
    std::fs::write(path, bytes)
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

/// Analyze one u16 RGB buffer and append its TSV row.
#[allow(clippy::too_many_arguments)]
fn emit_row(
    rows: &mut String,
    u16s: &[u16],
    w: u32,
    h: u32,
    variant: &str,
    cclass: &str,
    pc: u8,
    tc: u8,
    desc: PixelDescriptor,
    cols: &[AnalysisFeature],
    query: &AnalysisQuery,
) -> bool {
    let bytes: Vec<u8> = u16s.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let slice = match PixelSlice::new(&bytes, w, h, (w * 6) as usize, desc) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let r = match analyze_features(slice, query) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let _ = write!(rows, "{variant}\t{cclass}\t{w}\t{h}\t{pc}\t{tc}");
    for c in cols {
        let _ = write!(rows, "\t{}", feat_str(&r, *c));
    }
    rows.push('\n');
    true
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

/// Native row + the full downscale grid for one source. Returns (rows, ok, fail).
fn process_source(
    path: &Path,
    sharpen: f32,
    out_dir: Option<&Path>,
    cols: &[AnalysisFeature],
    query: &AnalysisQuery,
) -> (String, usize, usize) {
    let mut rows = String::new();
    let (mut ok, mut fail) = (0usize, 0usize);

    let raw = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return (rows, 0, 1),
    };
    let (pc, tc) = read_cicp(&raw).unwrap_or((1, 16));
    let img = match ImageReader::open(path).map(|r| r.decode()) {
        Ok(Ok(i)) => i.into_rgb16(),
        _ => return (rows, 0, 1),
    };
    let (sw, sh) = (img.width(), img.height());
    let src_u16 = img.into_raw();
    let desc = PixelDescriptor::RGB16
        .with_transfer(map_transfer(tc))
        .with_primaries(map_primaries(pc));
    let stem = path
        .file_name()
        .map(|s| s.to_string_lossy().trim_end_matches(".hdr.png").to_string())
        .unwrap_or_default();
    let cclass = path
        .parent()
        .and_then(|d| d.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Native — from the original u16, no transfer round-trip.
    if emit_row(
        &mut rows, &src_u16, sw, sh, &stem, &cclass, pc, tc, desc, cols, query,
    ) {
        ok += 1;
    } else {
        fail += 1;
    }

    // Linearize the full-res source once, reuse across targets.
    let lin_full: Vec<f32> = src_u16.iter().map(|&c| to_linear(c, tc)).collect();
    let longest = sw.max(sh);
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for &edge in LONGEST_EDGES {
        if edge >= longest {
            continue; // downscale-only
        }
        let scale = edge as f64 / longest as f64;
        let dw = ((sw as f64 * scale).round() as u32).max(1);
        let dh = ((sh as f64 * scale).round() as u32).max(1);
        if !seen.insert((dw, dh)) {
            continue;
        }
        let cfg = ResizeConfig::builder(sw, sh, dw, dh)
            .filter(Filter::Mitchell)
            .resize_sharpen(sharpen)
            .format(PixelDescriptor::RGBF32_LINEAR)
            .build();
        let lin_out = Resizer::new(&cfg).resize_f32(&lin_full);
        let out_u16: Vec<u16> = lin_out.iter().map(|&l| from_linear(l, tc)).collect();
        let variant = format!("{stem}.scale{dw}x{dh}");
        if emit_row(
            &mut rows, &out_u16, dw, dh, &variant, &cclass, pc, tc, desc, cols, query,
        ) {
            ok += 1;
        } else {
            fail += 1;
            continue;
        }
        if let Some(od) = out_dir {
            let p = od.join(format!("{variant}.hdr.png"));
            if let Some(buf) = ImageBuffer::<Rgb<u16>, _>::from_raw(dw, dh, out_u16)
                && buf.save(&p).is_ok()
            {
                let _ = inject_cicp(&p, pc, tc);
            }
        }
    }
    (rows, ok, fail)
}

struct Args {
    hdr_dir: PathBuf,
    features_out: PathBuf,
    out_dir: Option<PathBuf>,
    sharpen: f32,
    threads: usize,
    limit: usize,
}

fn parse_args() -> Result<Args, String> {
    let (mut hdr_dir, mut features_out, mut out_dir) = (None, None, None);
    let mut sharpen = 10.0f32;
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
            "--hdr-dir" => hdr_dir = it.next().map(PathBuf::from),
            "--features-out" => features_out = it.next().map(PathBuf::from),
            "--out-dir" => out_dir = it.next().map(PathBuf::from),
            "--sharpen" => sharpen = it.next().and_then(|s| s.parse().ok()).unwrap_or(sharpen),
            "--limit" => limit = it.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
            "--threads" => {
                threads = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n >= 1)
                    .unwrap_or(threads)
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        hdr_dir: hdr_dir.ok_or("--hdr-dir required")?,
        features_out: features_out.ok_or("--features-out required")?,
        out_dir,
        sharpen,
        threads,
        limit,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let mut files = Vec::new();
    collect_hdr(&args.hdr_dir, &mut files);
    files.sort();
    if files.len() > args.limit {
        files.truncate(args.limit);
    }
    if let Some(od) = &args.out_dir {
        std::fs::create_dir_all(od).ok();
    }
    let cols: Vec<AnalysisFeature> = FeatureSet::SUPPORTED.iter().collect();
    let n = files.len();
    let nthreads = args.threads.min(n.max(1));
    let chunk = n.div_ceil(nthreads).max(1);
    eprintln!(
        "{n} HDR sources, {} features/row, Mitchell+sharpen({}) linear-light grid across {nthreads} threads",
        cols.len(),
        args.sharpen,
    );

    let cols_ref = &cols;
    let out_dir = args.out_dir.as_deref();
    let sharpen = args.sharpen;
    let results: Vec<(String, usize, usize)> = thread::scope(|s| {
        files
            .chunks(chunk)
            .map(|ch| {
                s.spawn(move || {
                    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
                    let mut buf = String::new();
                    let (mut ok, mut fail) = (0usize, 0usize);
                    for p in ch {
                        let (rows, o, f) = process_source(p, sharpen, out_dir, cols_ref, &query);
                        buf.push_str(&rows);
                        ok += o;
                        fail += f;
                    }
                    (buf, ok, fail)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });

    let f = File::create(&args.features_out)
        .unwrap_or_else(|e| panic!("create {}: {e}", args.features_out.display()));
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
    let (mut tok, mut tfail) = (0usize, 0usize);
    for (buf, ok, fail) in &results {
        w.write_all(buf.as_bytes()).ok();
        tok += ok;
        tfail += fail;
    }
    w.flush().ok();
    eprintln!(
        "final: {tok} rows ok, {tfail} fail -> {}",
        args.features_out.display()
    );
    ExitCode::from(0)
}
