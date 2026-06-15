//! Characterization probe for the "SDR features consistent across channel
//! types, unless real extra precision is present, computed in linear light"
//! requirement. Feeds the SAME sRGB content five ways and compares a set of
//! SDR (tier 1/3) features:
//!
//!   A) u8  RGB8_SRGB                          — baseline
//!   B) u16 RGB16_SRGB = (A<<8)|A  (= A*257)   — LOSSLESS promotion, NO new info
//!   C) u16 RGB16_SRGB = B + sub-8-bit detail  — genuine sub-8-bit detail
//!   D) u8  RGB8 (LINEAR transfer), same bytes — gamma-vs-linear sensitivity
//!   E) f32 RGBF32 (sRGB), c/255.0             — 3rd channel type, same content
//!
//! NB: the lossless 8→16 promotion is byte-replication `(c<<8)|c` (=c*257,
//! 255→65535), NOT `c<<8` (=c*256, 255→65280). The converter narrows with
//! correct full-range rounding `round(v*255/65535)`, so a `c<<8` promotion
//! would read back ~0.39% low — an artifact, not a real inconsistency.
//!
//! Expectations to verify:
//!   - A == B  (consistency: lossless promotion adds no info ⇒ identical features)
//!   - A vs C  (precision: if real precision is *utilized*, C differs from A;
//!     if the analyzer narrows to 8-bit, the sub-8-bit detail is lost ⇒ C ≈ A)
//!   - A == D  (transfer: identical bytes tagged linear vs sRGB. If equal, the
//!     analyzer is transfer-blind — it computes on code values, NOT linear light)
//!
//! Run: cargo run --release --features experimental --example channel_consistency_probe

use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisFeature as AF, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice, TransferFunction};

const W: u32 = 256;
const H: u32 = 256;

fn make_u8() -> Vec<u8> {
    // Horizontal sRGB ramp + a faint per-row low-amplitude texture so edge /
    // laplacian / variance features have something to chew on.
    let mut v = vec![0u8; (W * H * 3) as usize];
    for y in 0..H {
        for x in 0..W {
            let base = (x * 255 / (W - 1)) as u8;
            let tex = if (x + y) % 7 == 0 { 3 } else { 0 };
            let p = ((y * W + x) * 3) as usize;
            v[p] = base.saturating_add(tex);
            v[p + 1] = base;
            v[p + 2] = base.saturating_sub(tex);
        }
    }
    v
}

fn feats(slice: PixelSlice<'_>, cols: &[AF]) -> Vec<(AF, f32)> {
    let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let r = analyze_features(slice, &q).expect("analyze");
    cols.iter()
        .filter_map(|&c| r.get_f32(c).map(|v| (c, v)))
        .collect()
}

fn main() {
    let cols = [
        AF::Variance,
        AF::EdgeDensity,
        AF::ChromaComplexity,
        AF::Uniformity,
        AF::HighFreqEnergyRatio,
        AF::LumaHistogramEntropy,
        #[cfg(feature = "experimental")]
        AF::LaplacianVariance,
        #[cfg(feature = "experimental")]
        AF::AqMapMean,
        #[cfg(feature = "experimental")]
        AF::NoiseFloorY,
    ];

    let u8buf = make_u8();
    let a = feats(
        PixelSlice::new(&u8buf, W, H, (W * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap(),
        &cols,
    );

    // B: LOSSLESS promote u8 -> u16 via byte replication (c<<8)|c == c*257.
    let b_u16: Vec<u16> = u8buf
        .iter()
        .map(|&c| ((c as u16) << 8) | c as u16)
        .collect();
    let b_bytes: Vec<u8> = b_u16.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let b = feats(
        PixelSlice::new(
            &b_bytes,
            W,
            H,
            (W * 6) as usize,
            PixelDescriptor::RGB16_SRGB,
        )
        .unwrap(),
        &cols,
    );

    // C: lossless base + genuine sub-8-bit detail. Amplitude capped at 0x3f
    // (63/257 ≈ 0.245 of an 8-bit LSB) so it NEVER crosses a narrowing rounding
    // boundary — unambiguously sub-8-bit. Real 16-bit signal, zero 8-bit trace.
    let c_u16: Vec<u16> = u8buf
        .iter()
        .enumerate()
        .map(|(i, &c)| (((c as u16) << 8) | c as u16).saturating_add((i as u16 * 37) & 0x3f))
        .collect();
    let c_bytes: Vec<u8> = c_u16.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let c = feats(
        PixelSlice::new(
            &c_bytes,
            W,
            H,
            (W * 6) as usize,
            PixelDescriptor::RGB16_SRGB,
        )
        .unwrap(),
        &cols,
    );

    // D: identical bytes to A, but tagged LINEAR (RGB8). If features == A, the
    // analyzer is transfer-blind (operates on code values, not linear light).
    let d = feats(
        PixelSlice::new(&u8buf, W, H, (W * 3) as usize, PixelDescriptor::RGB8).unwrap(),
        &cols,
    );

    // E: f32 sRGB-encoded floats c/255.0 — same SDR content as a 3rd channel
    // type. Confirms consistency holds for f32 too (narrows back to A's RGB8).
    let e_f32: Vec<f32> = u8buf.iter().map(|&c| c as f32 / 255.0).collect();
    let e_bytes: Vec<u8> = e_f32.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let e_desc = PixelDescriptor::RGBF32.with_transfer(TransferFunction::Srgb);
    let e = feats(
        PixelSlice::new(&e_bytes, W, H, (W * 12) as usize, e_desc).unwrap(),
        &cols,
    );

    println!(
        "{:<24} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "feature", "A:u8", "B:u16loss", "C:u16+det", "D:u8 lin", "E:f32 srgb"
    );
    let (mut ab, mut ac, mut ad, mut ae, mut n) = (0, 0, 0, 0, 0);
    for i in 0..a.len() {
        let (f, va) = a[i];
        let (vb, vc, vd, ve) = (b[i].1, c[i].1, d[i].1, e[i].1);
        let eq = |x: f32, y: f32| (x - y).abs() <= 1e-6 * x.abs().max(1.0);
        let (abeq, aceq, adeq, aeeq) = (eq(va, vb), eq(va, vc), eq(va, vd), eq(va, ve));
        ab += abeq as i32;
        ac += aceq as i32;
        ad += adeq as i32;
        ae += aeeq as i32;
        n += 1;
        println!(
            "{:<24} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}  {}{}{}{}",
            format!("{f:?}"),
            va,
            vb,
            vc,
            vd,
            ve,
            if abeq { "B=" } else { "B≠!" },
            if aceq { "C=" } else { "C≠prec" },
            if adeq { "D=" } else { "D≠lin" },
            if aeeq { "E=" } else { "E≠!" },
        );
    }
    println!(
        "\nA==B (u16 lossless): {ab}/{n} identical  — channel-type consistency, u16  (want {n}/{n})"
    );
    println!(
        "A==E (f32 sRGB):     {ae}/{n} identical  — channel-type consistency, f32  (want {n}/{n})"
    );
    println!(
        "A==C (u16 +detail):  {ac}/{n} identical  — {} features utilize genuine sub-8-bit precision (want >0 to satisfy 'unless precision present')",
        n - ac
    );
    println!(
        "A==D (linear tag):   {ad}/{n} identical  — {n}/{n} ⇒ transfer-BLIND: math on code values, NOT linear light"
    );
}
