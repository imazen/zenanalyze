//! Prototype A/B for the opt-in linear-light path (the "prototype opt-in mode"
//! decision). Measures, per feature, how much computing the SDR luma-domain
//! tier-1 features in **linear light** (and at **higher precision**) changes
//! them vs the shipped gamma/8-bit path — so we can decide which features
//! actually earn a production linear kernel before building f32 SIMD kernels.
//!
//! It reimplements the four luma-domain tier-1 features (variance, edge
//! density, uniformity, laplacian variance) as a scalar f32 kernel that takes a
//! precomputed luma plane — so the ONLY thing that varies between arms is how
//! the luma plane is built. Three measurements:
//!
//!  - **SDR fidelity:** my gamma kernel vs shipped `analyze_features` — confirms
//!    the reimplementation tracks the real feature (full-image vs stripe-sampled,
//!    so correlation, not equality).
//!  - **SDR linear-light effect:** my gamma vs my linear (sRGB→linear, same
//!    [0,255] scale) — the domain effect, sampling held constant.
//!  - **HDR precision effect:** linear-from-16-bit vs linear-from-narrowed-8-bit
//!    on the PQ renditions — does real >8-bit precision move the features.
//!
//! Usage:
//!   cargo run --release --features experimental --example linear_light_ab -- \
//!     --sdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
//!     --hdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
//!     --out /mnt/v/output/imazen-26-features/linear_light_ab_2026-06-14.tsv [--limit N]

use image::ImageReader;
use linear_srgb::tf::{pq_to_linear, srgb_to_linear};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature as AF, AnalysisQuery, FeatureSet};

const EDGE_THRESH_SQ: f32 = 400.0;

/// The four luma-domain tier-1 features, computed from a luma plane in [0,255].
/// Mirrors tier1.rs reductions exactly (full-image, not stripe-sampled).
fn subset(luma: &[f32], w: usize, h: usize) -> [f32; 4] {
    let n = (w * h) as f64;
    // variance
    let (mut s, mut sq) = (0.0f64, 0.0f64);
    for &l in luma {
        s += l as f64;
        sq += (l as f64) * (l as f64);
    }
    let mean = s / n;
    let variance = (sq / n - mean * mean).max(0.0) as f32;
    // edge density: |∇L|² > 400 over interior (forward diffs)
    let mut edge = 0u64;
    let mut interior = 0u64;
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let i = y * w + x;
            let gx = luma[i + 1] - luma[i];
            let gy = luma[i + w] - luma[i];
            if gx * gx + gy * gy > EDGE_THRESH_SQ {
                edge += 1;
            }
            interior += 1;
        }
    }
    let edge_density = if interior > 0 {
        (edge as f64 / interior as f64) as f32
    } else {
        0.0
    };
    // uniformity: 8×8 blocks with luma variance < 25
    let (bx, by) = (w / 8, h / 8);
    let (mut uniform, mut total) = (0u32, 0u32);
    for byi in 0..by {
        for bxi in 0..bx {
            let (mut bs, mut bsq) = (0.0f64, 0.0f64);
            for dy in 0..8 {
                for dx in 0..8 {
                    let l = luma[(byi * 8 + dy) * w + (bxi * 8 + dx)] as f64;
                    bs += l;
                    bsq += l * l;
                }
            }
            let bm = bs / 64.0;
            let bv = (bsq / 64.0 - bm * bm).max(0.0);
            if bv < 25.0 {
                uniform += 1;
            }
            total += 1;
        }
    }
    let uniformity = if total > 0 {
        uniform as f32 / total as f32
    } else {
        1.0
    };
    // laplacian variance: var(∇²L) over interior, /1e3 · √MP
    let (mut ls, mut lsq, mut lc) = (0.0f64, 0.0f64, 0u64);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let lap = luma[i - 1] + luma[i + 1] + luma[i - w] + luma[i + w] - 4.0 * luma[i];
            ls += lap as f64;
            lsq += (lap as f64) * (lap as f64);
            lc += 1;
        }
    }
    let laplacian_variance = if lc > 0 {
        let lm = ls / lc as f64;
        let lv = (lsq / lc as f64 - lm * lm).max(0.0);
        let mp = (w as f64 * h as f64 / 1_000_000.0).max(1e-3);
        ((lv / 1e3) * mp.sqrt()) as f32
    } else {
        0.0
    };
    [variance, edge_density, uniformity, laplacian_variance]
}

fn luma_from_rgb8(rgb: &[u8], linearize: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(rgb.len() / 3);
    for px in rgb.chunks_exact(3) {
        let (r, g, b) = if linearize {
            (
                srgb_to_linear(px[0] as f32 / 255.0) * 255.0,
                srgb_to_linear(px[1] as f32 / 255.0) * 255.0,
                srgb_to_linear(px[2] as f32 / 255.0) * 255.0,
            )
        } else {
            (px[0] as f32, px[1] as f32, px[2] as f32)
        };
        out.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    out
}

/// HDR luma plane in [0,255]: PQ→linear, optionally narrowing to 8-bit first.
fn luma_from_pq16(rgb: &[u16], narrow_to_8bit: bool) -> Vec<f32> {
    let lin = |code: u16| -> f32 {
        let norm = if narrow_to_8bit {
            // round(code*255/65535) then back to [0,1] — the 8-bit narrowing.
            ((code as f32 * 255.0 / 65535.0).round()) / 255.0
        } else {
            code as f32 / 65535.0
        };
        pq_to_linear(norm) * 255.0
    };
    let mut out = Vec::with_capacity(rgb.len() / 3);
    for px in rgb.chunks_exact(3) {
        out.push(0.299 * lin(px[0]) + 0.587 * lin(px[1]) + 0.114 * lin(px[2]));
    }
    out
}

fn collect(dir: &Path, suffix: &str, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, suffix, out);
            } else if p.to_string_lossy().ends_with(suffix) {
                out.push(p);
            }
        }
    }
}

/// Pearson correlation + median & max relative |a−b| over paired samples.
fn stats(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let n = a.len() as f64;
    if n < 2.0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let pearson = if va > 0.0 && vb > 0.0 {
        cov / (va.sqrt() * vb.sqrt())
    } else {
        f64::NAN
    };
    let mut rel: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-6))
        .collect();
    rel.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let med = rel[rel.len() / 2];
    let max = *rel.last().unwrap();
    (pearson, med, max)
}

fn main() -> ExitCode {
    let (mut sdr_dir, mut hdr_dir, mut out) = (None, None, None);
    let mut limit = usize::MAX;
    let raw: Vec<String> = env::args().collect();
    let mut it = raw.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sdr-dir" => sdr_dir = it.next().map(PathBuf::from),
            "--hdr-dir" => hdr_dir = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--limit" => limit = it.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
            _ => {}
        }
    }
    let names = [
        "variance",
        "edge_density",
        "uniformity",
        "laplacian_variance",
    ];

    // ---- SDR: gamma vs linear (domain effect) + gamma-vs-shipped (fidelity) ----
    let mut sg = [(); 4].map(|_| Vec::<f64>::new()); // my gamma
    let mut sl = [(); 4].map(|_| Vec::<f64>::new()); // my linear
    let mut ss = [(); 4].map(|_| Vec::<f64>::new()); // shipped gamma
    if let Some(d) = &sdr_dir {
        let mut files = Vec::new();
        collect(d, ".sdr.png", &mut files);
        files.sort();
        files.truncate(limit);
        eprintln!("SDR: {} images", files.len());
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        let shipped_feats = [
            AF::Variance,
            AF::EdgeDensity,
            AF::Uniformity,
            AF::LaplacianVariance,
        ];
        for (k, p) in files.iter().enumerate() {
            let Ok(Ok(img)) = ImageReader::open(p).map(|r| r.decode()) else {
                continue;
            };
            let img = img.to_rgb8();
            let (w, h) = (img.width() as usize, img.height() as usize);
            if w < 16 || h < 16 {
                continue;
            }
            let rgb = img.as_raw();
            let g = subset(&luma_from_rgb8(rgb, false), w, h);
            let l = subset(&luma_from_rgb8(rgb, true), w, h);
            let r = analyze_features_rgb8(rgb, w as u32, h as u32, &q);
            for i in 0..4 {
                sg[i].push(g[i] as f64);
                sl[i].push(l[i] as f64);
                if let Some(v) = r.get_f32(shipped_feats[i]) {
                    ss[i].push(v as f64);
                } else {
                    ss[i].push(f64::NAN);
                }
            }
            if (k + 1) % 250 == 0 {
                eprintln!("  SDR [{}/{}]", k + 1, files.len());
            }
        }
    }

    // ---- HDR: linear-from-16bit vs linear-from-narrowed-8bit (precision) ----
    let mut hf = [(); 4].map(|_| Vec::<f64>::new());
    let mut hn = [(); 4].map(|_| Vec::<f64>::new());
    if let Some(d) = &hdr_dir {
        let mut files = Vec::new();
        collect(d, ".hdr.png", &mut files);
        files.sort();
        files.truncate(limit);
        eprintln!("HDR: {} images", files.len());
        for (k, p) in files.iter().enumerate() {
            let Ok(Ok(img)) = ImageReader::open(p).map(|r| r.decode()) else {
                continue;
            };
            let img = img.into_rgb16();
            let (w, h) = (img.width() as usize, img.height() as usize);
            if w < 16 || h < 16 {
                continue;
            }
            let rgb = img.into_raw();
            let full = subset(&luma_from_pq16(&rgb, false), w, h);
            let nar = subset(&luma_from_pq16(&rgb, true), w, h);
            for i in 0..4 {
                hf[i].push(full[i] as f64);
                hn[i].push(nar[i] as f64);
            }
            if (k + 1) % 25 == 0 {
                eprintln!("  HDR [{}/{}]", k + 1, files.len());
            }
        }
    }

    // ---- report ----
    let mut lines = vec!["axis\tfeature\tpearson\tmedian_rel_delta\tmax_rel_delta\tn".to_string()];
    let mut emit = |axis: &str, a: &[Vec<f64>; 4], b: &[Vec<f64>; 4]| {
        for i in 0..4 {
            // drop NaN-paired samples (shipped may omit)
            let (mut xa, mut xb) = (Vec::new(), Vec::new());
            for j in 0..a[i].len() {
                if a[i][j].is_finite() && b[i][j].is_finite() {
                    xa.push(a[i][j]);
                    xb.push(b[i][j]);
                }
            }
            let (p, med, mx) = stats(&xa, &xb);
            lines.push(format!(
                "{axis}\t{}\t{p:.4}\t{med:.4}\t{mx:.4}\t{}",
                names[i],
                xa.len()
            ));
        }
    };
    emit("sdr_fidelity_gamma_vs_shipped", &sg, &ss);
    emit("sdr_linear_effect_gamma_vs_linear", &sg, &sl);
    emit("hdr_precision_full16_vs_narrow8", &hf, &hn);

    let text = lines.join("\n");
    println!("\n{text}\n");
    if let Some(o) = &out {
        std::fs::write(o, format!("{text}\n")).ok();
        eprintln!("wrote {}", o.display());
    }
    ExitCode::from(0)
}
