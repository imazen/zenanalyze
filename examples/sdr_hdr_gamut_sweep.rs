//! SDR ↔ HDR ↔ wide-gamut feature-consistency sweep on **real content**.
//!
//! The sibling `sdr_hdr_consistency` probe proved (on synthetic content) that
//! the f32 + sRGB-OETF linear-light path makes SDR-in-PQ match SDR-normal below
//! diffuse white. This sweep extends that to real images and to the two axes a
//! codec picker actually faces, to settle the MLP-architecture question
//! (conditioned trunk vs separate vs deeper):
//!
//!   * **SDR↔HDR (transfer / dynamic range)** — a super-white *headroom ladder*:
//!     the brightest `--highlight-pct` of pixels boosted ×N above diffuse white.
//!     `hdr_h1` (no boost) is the OETF round-trip baseline; `h2/h4/h8` extend the
//!     highlights. Which features stay flat (consistent) vs ramp with headroom?
//!   * **Colour gamut (primaries)** — the same values reinterpreted in Bt2020.
//!     Which features shift (chroma) vs stay (luma)?
//!
//! Every variant is the SAME displayable content; the cross-axis shift is what
//! we measure, plus whether it is *explained* by the depth-tier regime features
//! (headroom / peak / gamut-coverage) — high explanation ⇒ one regime-conditioned
//! model suffices.
//!
//! ```text
//! cargo run --release --features experimental,hdr --example sdr_hdr_gamut_sweep -- \
//!   --sdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
//!   --features-out /tmp/sdr_hdr_gamut_sweep.tsv --limit 40 --highlight-pct 5 --max-dim 1024
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::{ImageReader, imageops::FilterType};
use linear_srgb::tf::{linear_to_pq, srgb_to_linear};
use zenanalyze::feature::{AnalysisQuery, AnalysisResults, FeatureSet};
use zenanalyze::{analyze_features, analyze_features_rgb8};
use zenpixels::{
    ColorContext, ColorPrimaries, DiffuseWhite, PixelDescriptor, PixelSlice, TransferFunction,
};

const DIFFUSE_WHITE_NITS: f32 = 203.0;
const HEADROOMS: &[f32] = &[1.0, 2.0, 4.0, 8.0];

fn luma(px: &[u8]) -> f32 {
    0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32
}

/// Top-`pct`% luma threshold — pixels at/above it are "highlights".
fn highlight_threshold(rgb: &[u8], pct: f32) -> f32 {
    let mut lumas: Vec<f32> = rgb.chunks_exact(3).map(luma).collect();
    lumas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if lumas.is_empty() {
        return 255.0;
    }
    let idx = (((100.0 - pct) / 100.0) * (lumas.len() as f32 - 1.0)).round() as usize;
    lumas[idx.min(lumas.len() - 1)]
}

/// PQ16 (sRGB primaries) variant: highlights boosted ×`headroom`. Diffuse white
/// (sRGB white) maps to `DIFFUSE_WHITE_NITS`; boosted highlights to
/// `DIFFUSE_WHITE_NITS × headroom` — super-white once headroom > 1.
fn synth_pq(rgb: &[u8], thresh: f32, headroom: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 6);
    for px in rgb.chunks_exact(3) {
        let boost = if luma(px) >= thresh { headroom } else { 1.0 };
        for &c in px {
            let lin = srgb_to_linear(c as f32 / 255.0) * boost;
            let pq = linear_to_pq(lin * DIFFUSE_WHITE_NITS / 10000.0).clamp(0.0, 1.0);
            let v = (pq * 65535.0 + 0.5) as u16;
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn feature_cells(results: &AnalysisResults) -> Vec<String> {
    FeatureSet::SUPPORTED
        .iter()
        .map(|f| {
            results
                .get_f32(f)
                .map(|v| format!("{v:.7e}"))
                .unwrap_or_else(|| "nan".to_string())
        })
        .collect()
}

fn header_cells() -> Vec<String> {
    FeatureSet::SUPPORTED
        .iter()
        .map(|f| format!("feat_{}", f.name()))
        .collect()
}

fn content_class(path: &Path) -> String {
    path.parent()
        .and_then(|d| d.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn collect_sdr(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_sdr(&p, out);
            } else if p.to_string_lossy().ends_with(".sdr.png") {
                out.push(p);
            }
        }
    }
}

struct Args {
    sdr_dir: PathBuf,
    features_out: PathBuf,
    limit: usize,
    highlight_pct: f32,
    max_dim: u32,
}

fn parse_args() -> Result<Args, String> {
    let (mut sdr_dir, mut features_out) = (None, None);
    let (mut limit, mut highlight_pct, mut max_dim) = (usize::MAX, 5.0f32, 1024u32);
    let raw: Vec<String> = std::env::args().collect();
    let mut it = raw.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sdr-dir" => sdr_dir = it.next().map(PathBuf::from),
            "--features-out" => features_out = it.next().map(PathBuf::from),
            "--limit" => limit = it.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
            "--highlight-pct" => {
                highlight_pct = it.next().and_then(|s| s.parse().ok()).unwrap_or(5.0)
            }
            "--max-dim" => max_dim = it.next().and_then(|s| s.parse().ok()).unwrap_or(1024),
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        sdr_dir: sdr_dir.ok_or("--sdr-dir required")?,
        features_out: features_out.ok_or("--features-out required")?,
        limit,
        highlight_pct,
        max_dim,
    })
}

fn emit(
    buf: &mut String,
    stem: &str,
    cc: &str,
    variant: &str,
    hr: f32,
    dim: (u32, u32),
    cells: &[String],
) {
    buf.push_str(stem);
    buf.push('\t');
    buf.push_str(cc);
    buf.push('\t');
    buf.push_str(variant);
    buf.push('\t');
    buf.push_str(&format!("{hr}\t{}\t{}", dim.0, dim.1));
    for c in cells {
        buf.push('\t');
        buf.push_str(c);
    }
    buf.push('\n');
}

fn process(
    path: &Path,
    args: &Args,
    gamma_q: &AnalysisQuery,
    linear_q: &AnalysisQuery,
) -> Option<String> {
    let img = ImageReader::open(path).ok()?.decode().ok()?;
    let img = if img.width().max(img.height()) > args.max_dim {
        let (w, h) = (img.width(), img.height());
        let s = args.max_dim as f32 / w.max(h) as f32;
        img.resize(
            ((w as f32 * s).round() as u32).max(1),
            ((h as f32 * s).round() as u32).max(1),
            FilterType::Lanczos3,
        )
    } else {
        img
    };
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let rgb = rgb.into_raw();
    let stem = path
        .file_name()
        .map(|s| s.to_string_lossy().trim_end_matches(".sdr.png").to_string())
        .unwrap_or_default();
    let cc = content_class(path);
    let mut buf = String::new();

    // SDR baseline — gamma, sRGB primaries.
    let sdr = analyze_features_rgb8(&rgb, w, h, gamma_q);
    emit(
        &mut buf,
        &stem,
        &cc,
        "sdr",
        1.0,
        (w, h),
        &feature_cells(&sdr),
    );

    // Wide-gamut — same values, Bt2020 primaries, gamma.
    let wg = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::Bt2020);
    if let Ok(slice) = PixelSlice::new(&rgb, w, h, (w * 3) as usize, wg)
        && let Ok(r) = analyze_features(slice, gamma_q)
    {
        emit(
            &mut buf,
            &stem,
            &cc,
            "widegamut",
            1.0,
            (w, h),
            &feature_cells(&r),
        );
    }

    // HDR ladder — highlights boosted, PQ, sRGB primaries, linear-light.
    let thresh = highlight_threshold(&rgb, args.highlight_pct);
    let hdr = PixelDescriptor::RGB16
        .with_transfer(TransferFunction::Pq)
        .with_primaries(ColorPrimaries::Bt709);
    for &hr in HEADROOMS {
        let pq = synth_pq(&rgb, thresh, hr);
        let ctx = Arc::new(
            ColorContext::default().with_diffuse_white(DiffuseWhite::new(DIFFUSE_WHITE_NITS)),
        );
        if let Ok(slice) = PixelSlice::new(&pq, w, h, (w * 6) as usize, hdr)
            && let Ok(r) = analyze_features(slice.with_color_context(ctx), linear_q)
        {
            emit(&mut buf, &stem, &cc, "hdr", hr, (w, h), &feature_cells(&r));
        }
    }
    Some(buf)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let mut files = Vec::new();
    collect_sdr(&args.sdr_dir, &mut files);
    files.sort();
    if files.len() > args.limit {
        files.truncate(args.limit);
    }
    eprintln!(
        "{} sources, highlight-pct={}, headrooms={:?}, max-dim={}",
        files.len(),
        args.highlight_pct,
        HEADROOMS,
        args.max_dim
    );

    let gamma_q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let linear_q = AnalysisQuery::new(FeatureSet::SUPPORTED).with_linear_light(true);

    let mut out = String::from("stem\tcontent_class\tvariant\theadroom\twidth\theight");
    for c in header_cells() {
        out.push('\t');
        out.push_str(&c);
    }
    out.push('\n');

    let (mut ok, mut fail) = (0usize, 0usize);
    for p in &files {
        match process(p, &args, &gamma_q, &linear_q) {
            Some(rows) => {
                out.push_str(&rows);
                ok += 1;
            }
            None => fail += 1,
        }
    }
    std::fs::write(&args.features_out, out)
        .map_err(|e| format!("write {}: {e}", args.features_out.display()))?;
    eprintln!(
        "done: {ok} ok, {fail} fail -> {}",
        args.features_out.display()
    );
    Ok(())
}
