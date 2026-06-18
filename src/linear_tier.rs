//! **Diffuse-white-normalized linear-light** tier-1 override, opt-in via
//! [`crate::feature::AnalysisQuery::with_linear_light`].
//!
//! The default tier 1/2/3 path is transfer-blind — it computes on gamma-encoded
//! code values, and narrows HDR (PQ/HLG) to RGB8 by clipping against the
//! 10000-nit PQ peak, which crushes SDR-in-HDR content to near-black (see
//! `benchmarks/linear_light_precision_review_2026-06-14.md` and the
//! `sdr_in_hdr_envelope_matches_sdr_normal` test). This module recomputes the
//! supported feature in **linear light, normalized to the diffuse-white anchor**
//! — *not* tonemapped. Today that is `Variance` (the feature whose ranking moved
//! most in the A/B); coverage grows as the kernels land.
//!
//! ## Pipeline (reuses zenpixels-convert for all colour math)
//!
//! 1. `zenpixels_convert::RowConverter` → `RGBF32_LINEAR`: the EOTF + primaries,
//!    giving *absolute* linear for PQ (`1.0` = 10000 cd/m²) and *relative* linear
//!    for SDR transfers (`1.0` = display white).
//! 2. **Diffuse-white anchor** (a single multiply, the only normalization done
//!    here): for PQ, scale by `10000 / diffuse_white_nits` so the anchor maps to
//!    `1.0`; SDR/HLG are already relative (scale `1.0`). The anchor is read from
//!    the slice's [`ColorContext`](zenpixels::ColorContext) `diffuse_white`,
//!    defaulting to BT.2408 (203 nits) when unsignaled.
//! 3. Luma → **f32 SIMD** variance (4-accumulator `#[magetypes]` reduction; f32
//!    so true-HDR highlights survive as `>1.0` signal, never clipped).
//!
//! This makes **SDR-in-HDR ≡ SDR by construction**: sub-diffuse-white content
//! maps to identical relative-linear values whether it arrived as sRGB or as PQ.
//! Works on any bit depth / format `RowConverter` accepts (u8/u16/f32, RGB/RGBA,
//! sRGB/PQ/HLG). Luma is kept on a ×255 scale so the variance has the [0,255]²
//! magnitude of the gamma `Variance` it replaces (highlights push past that).

use archmage::{incant, magetypes};
use zenpixels::{PixelDescriptor, PixelSlice, TransferFunction};
use zenpixels_convert::RowConverter;

/// BT.601 luma weights — match `tier1`'s baseline for the sRGB/BT.709 path.
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;
/// PQ absolute peak (SMPTE ST 2084).
const PQ_PEAK_NITS: f32 = 10000.0;
/// BT.2408 diffuse white — the default anchor when a source doesn't signal one.
const DEFAULT_DIFFUSE_WHITE_NITS: f32 = 203.0;

/// SIMD sum + sum-of-squares of an f32 luma plane. 4 independent accumulators
/// break the reduction dependency chain; f32 reduced to f64 at the end (the
/// values can exceed [0,255] for true-HDR highlights, so no integer container).
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn sum_sumsq_simd(token: Token, p: &[f32]) -> (f64, f64) {
    let mut s = [f32x8::zero(token); 4];
    let mut sq = [f32x8::zero(token); 4];
    let mut it = p.chunks_exact(32);
    for c in &mut it {
        for k in 0..4 {
            let v = f32x8::from_slice(token, &c[k * 8..k * 8 + 8]);
            s[k] += v;
            sq[k] = v.mul_add(v, sq[k]);
        }
    }
    let (mut st, mut sqt) = (0.0f64, 0.0f64);
    for k in 0..4 {
        st += s[k].reduce_add() as f64;
        sqt += sq[k].reduce_add() as f64;
    }
    for &x in it.remainder() {
        st += x as f64;
        sqt += (x as f64) * (x as f64);
    }
    (st, sqt)
}

fn sum_sumsq(p: &[f32]) -> (f64, f64) {
    incant!(sum_sumsq_simd(p))
}

/// The diffuse-white anchor scale: the factor that maps the converter's linear
/// output to **diffuse-white-relative** linear (anchor → 1.0). PQ's linear is
/// absolute (1.0 = 10000 nits) so it scales by `10000 / diffuse_white`; SDR and
/// HLG are already relative, scale 1.0.
///
/// TODO(zenpixels-bump): honor a *signaled* anchor from the slice
/// `ColorContext::diffuse_white`. zenanalyze pins zenpixels 0.2.11, whose
/// `ColorContext` is `{icc, cicp}` only — the `diffuse_white` field landed in
/// 0.2.15. Until the dep bumps, every PQ source uses the BT.2408 default (203).
fn anchor_scale(slice: &PixelSlice<'_>) -> f32 {
    if slice.descriptor().transfer != TransferFunction::Pq {
        return 1.0;
    }
    PQ_PEAK_NITS / DEFAULT_DIFFUSE_WHITE_NITS
}

/// Diffuse-white-normalized linear-light luma variance over any [`PixelSlice`],
/// row-stride sampled to roughly `pixel_budget` samples. Returns `None` only if
/// the converter can't be built for the source descriptor.
pub(crate) fn linear_variance(slice: &PixelSlice<'_>, pixel_budget: usize) -> Option<f32> {
    let desc = slice.descriptor();
    let w = slice.width() as usize;
    let h = slice.rows() as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let mut converter = RowConverter::new(desc, PixelDescriptor::RGBF32_LINEAR).ok()?;
    let scale = anchor_scale(slice) * 255.0; // ×255 → [0,255] luma magnitude

    // Scalar gather is unavoidable (the EOTF lives in the converter); build a
    // normalized-linear luma plane over sampled rows, then SIMD-reduce it.
    let row_stride = ((w * h) / pixel_budget.max(1)).max(1);
    let dst_bytes = w * 3 * 4; // RGBF32_LINEAR = 3 × f32
    let mut dst = vec![0u8; dst_bytes];
    let mut luma: Vec<f32> = Vec::with_capacity((h / row_stride + 1) * w);
    let mut y = 0usize;
    while y < h {
        converter.convert_row(slice.row(y as u32), &mut dst, w as u32);
        for px in dst.chunks_exact(12) {
            let r = f32::from_ne_bytes(px[0..4].try_into().unwrap());
            let g = f32::from_ne_bytes(px[4..8].try_into().unwrap());
            let b = f32::from_ne_bytes(px[8..12].try_into().unwrap());
            luma.push((KR * r + KG * g + KB * b) * scale);
        }
        y += row_stride;
    }
    let n = luma.len();
    if n == 0 {
        return None;
    }
    let (sum, sumsq) = sum_sumsq(&luma);
    let mean = sum / n as f64;
    Some((sumsq / n as f64 - mean * mean).max(0.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_ramp_rgb8(w: usize, h: usize, max_v: u8) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 3;
                let v = (x as u32 * max_v as u32 / (w as u32 - 1)) as u8;
                buf[p] = v;
                buf[p + 1] = v;
                buf[p + 2] = v;
            }
        }
        buf
    }

    /// Shadows are where the sRGB curve diverges most from linear, so a *dark*
    /// ramp (0..63) has dramatically lower variance in linear light than gamma.
    #[test]
    fn linear_variance_collapses_in_shadows() {
        let (w, h) = (64usize, 64usize);
        let buf = gray_ramp_rgb8(w, h, 63);
        let slice =
            PixelSlice::new(&buf, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let lin = linear_variance(&slice, 500_000).expect("rgb8") as f64;
        let vals: Vec<f64> = (0..w).map(|x| x as f64).collect();
        let mean = vals.iter().sum::<f64>() / w as f64;
        let gamma = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / w as f64;
        assert!(
            lin.is_finite() && lin > 0.0,
            "linear var positive, got {lin}"
        );
        assert!(
            lin < 0.3 * gamma,
            "linear-light should collapse shadow variance: linear {lin} vs gamma {gamma}"
        );
    }

    /// End-to-end: the opt-in flag changes the reported `Variance`.
    #[test]
    fn linear_light_flag_changes_variance_end_to_end() {
        use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
        let (w, h) = (64u32, 64u32);
        let buf = gray_ramp_rgb8(w as usize, h as usize, 63);
        let mk =
            || PixelSlice::new(&buf, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
        let fs = FeatureSet::new().with(AnalysisFeature::Variance);
        let gamma = crate::analyze_features(mk(), &AnalysisQuery::new(fs))
            .unwrap()
            .get_f32(AnalysisFeature::Variance)
            .unwrap();
        let linear = crate::analyze_features(mk(), &AnalysisQuery::new(fs).with_linear_light(true))
            .unwrap()
            .get_f32(AnalysisFeature::Variance)
            .unwrap();
        assert!(
            linear < gamma * 0.5,
            "flag must lower shadow variance: gamma {gamma} linear {linear}"
        );
    }

    /// A uniform field reads ~zero variance through the converter + SIMD reduce.
    #[test]
    fn uniform_field_is_zero_variance() {
        let buf = vec![137u8; 40 * 40 * 3];
        let slice = PixelSlice::new(&buf, 40, 40, 40 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let lin = linear_variance(&slice, 500_000).expect("rgb8");
        assert!(lin.abs() < 1e-2, "uniform field variance ~0, got {lin}");
    }

    /// The general path handles non-RGB8 (here RGBA8) — the converter does the
    /// layout, unlike the old RGB8-only stub.
    #[test]
    fn rgba8_supported() {
        let buf = vec![80u8; 32 * 32 * 4];
        let slice = PixelSlice::new(&buf, 32, 32, 32 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
        assert!(linear_variance(&slice, 500_000).is_some());
    }
}
