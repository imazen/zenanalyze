//! **Linear-light** tier-1 override, opt-in via
//! [`crate::feature::AnalysisQuery::with_linear_light`].
//!
//! The default tier 1/2/3 path is transfer-blind — it computes on gamma-encoded
//! code values (see `benchmarks/linear_light_precision_review_2026-06-14.md`).
//! This module recomputes the supported feature in **linear light**: it
//! linearizes each sample through the sRGB EOTF before the statistic. Today that
//! is just `Variance` (the feature whose ranking moved most in the A/B —
//! `benchmarks/linear_light_ab_2026-06-14.md`); coverage grows as the kernels
//! land.
//!
//! ## Representation (per `benchmarks/linear_repr_bench_2026-06-14.md`)
//!
//! Linearize via a 256-entry sRGB→**i12** LUT (12-bit linear round-trips 8-bit
//! sRGB without shadow banding; 10-bit collapses shadow codes). The variance
//! *reduction* runs in **i32x8** SIMD (magetypes `#[magetypes]` + `incant!`):
//! the squared term needs i32 lanes regardless (luma² overflows i16), real
//! `i32x8` ties hand-tuned `f32x8` on throughput, and the i32→i64 flush is
//! bit-exact where f32 accumulation of millions of squares loses mantissa. The
//! gather + luma stay scalar (an inherent table lookup). RGB8-layout sources
//! only; returns `None` otherwise so the caller keeps the gamma value.
//!
//! Luma is kept on an ×16 i12 scale (LUT peak 4080 = 255·16), so the final
//! variance rescales by ÷256 to the **[0,255]² magnitude** of the gamma
//! `Variance` it replaces — a drop-in for `RawAnalysis::variance`.

use archmage::{incant, magetypes};
use zenpixels::{PixelDescriptor, PixelSlice};

/// Fixed-point BT.601 luma weights, sum 256 → `(QR·r + QG·g + QB·b) >> 8` keeps
/// a gray pixel's luma equal to its i12 linear value.
const QR: i32 = 77;
const QG: i32 = 150;
const QB: i32 = 29;

/// 256-entry sRGB→linear LUT on an ×16 i12 scale (peak 4080 = 255·16). i12
/// precision round-trips 8-bit sRGB without the shadow banding 8-/10-bit linear
/// hits; the ×16 makes the final ÷256 rescale land on the [0,255]² scale.
fn srgb_to_linear_i12_lut() -> [i32; 256] {
    let mut lut = [0i32; 256];
    for (v, e) in lut.iter_mut().enumerate() {
        *e = (linear_srgb::tf::srgb_to_linear(v as f32 / 255.0) * 4080.0 + 0.5) as i32;
    }
    lut
}

/// SIMD sum + sum-of-squares of an i12 luma plane. 4 independent accumulators
/// break the reduction dependency chain; i32 lanes flush to i64 (via `to_array`)
/// every 120 chunks — luma² ≤ 4080² ≈ 16.6M, 120·16.6M ≈ 2.0e9 < `i32::MAX`.
#[magetypes(define(i32x8), v4, v3, neon, wasm128, scalar)]
fn sum_sumsq_simd(token: Token, p: &[i32]) -> (i64, i64) {
    const FLUSH: usize = 120;
    let mut s = [i32x8::zero(token); 4];
    let mut sq = [i32x8::zero(token); 4];
    let (mut st, mut sqt) = (0i64, 0i64);
    let mut cnt = 0usize;
    let mut it = p.chunks_exact(32);
    for c in &mut it {
        for k in 0..4 {
            let v = i32x8::from_slice(token, &c[k * 8..k * 8 + 8]);
            s[k] += v;
            sq[k] += v * v;
        }
        cnt += 1;
        if cnt == FLUSH {
            for k in 0..4 {
                for lane in s[k].to_array() {
                    st += lane as i64;
                }
                for lane in sq[k].to_array() {
                    sqt += lane as i64;
                }
                s[k] = i32x8::zero(token);
                sq[k] = i32x8::zero(token);
            }
            cnt = 0;
        }
    }
    for k in 0..4 {
        for lane in s[k].to_array() {
            st += lane as i64;
        }
        for lane in sq[k].to_array() {
            sqt += lane as i64;
        }
    }
    for &x in it.remainder() {
        let v = x as i64;
        st += v;
        sqt += v * v;
    }
    (st, sqt)
}

fn sum_sumsq(p: &[i32]) -> (i64, i64) {
    incant!(sum_sumsq_simd(p))
}

/// Linear-light luma variance over an RGB8-layout source, row-stride sampled to
/// roughly `pixel_budget` samples (matching tier1's sample count). Returns
/// `None` for non-RGB8 layouts (caller keeps the gamma `Variance`).
pub(crate) fn linear_variance_rgb8(slice: &PixelSlice<'_>, pixel_budget: usize) -> Option<f32> {
    if !slice.descriptor().layout_compatible(PixelDescriptor::RGB8) {
        return None;
    }
    let w = slice.width() as usize;
    let h = slice.rows() as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let lut = srgb_to_linear_i12_lut();

    // Scalar gather + integer luma → i12 luma buffer over sampled rows.
    let row_stride = ((w * h) / pixel_budget.max(1)).max(1);
    let mut luma: Vec<i32> = Vec::with_capacity((h / row_stride + 1) * w);
    let mut y = 0usize;
    while y < h {
        let row = &slice.row(y as u32)[..w * 3];
        for px in row.chunks_exact(3) {
            let l = QR * lut[px[0] as usize] + QG * lut[px[1] as usize] + QB * lut[px[2] as usize];
            luma.push(l >> 8); // i12 luma, ×16 scale
        }
        y += row_stride;
    }
    let n = luma.len();
    if n == 0 {
        return None;
    }

    let (sum, sumsq) = sum_sumsq(&luma);
    let mean = sum as f64 / n as f64;
    let var_i12 = (sumsq as f64 / n as f64 - mean * mean).max(0.0);
    // luma_i12 == 16 · luma_[0,255]  ⇒  variance scales ×256. Rescale back.
    Some((var_i12 / 256.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_ramp(w: usize, h: usize, max_v: u8) -> Vec<u8> {
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
    /// (On a full 0..255 ramp the effect is only ~3%; the A/B's ~21% median is
    /// on real photos. Shadows are the clean, robust demonstration.)
    #[test]
    fn linear_variance_collapses_in_shadows() {
        let (w, h) = (64usize, 64usize);
        let buf = gray_ramp(w, h, 63);
        let slice =
            PixelSlice::new(&buf, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let lin = linear_variance_rgb8(&slice, 500_000).expect("rgb8 source") as f64;

        // Gamma luma == code value for grayscale; population variance of the ramp
        // (gray_ramp with max_v == w-1 makes v == x).
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

    /// End-to-end: the opt-in flag actually changes the reported `Variance`,
    /// and the default (flag off) is untouched.
    #[test]
    fn linear_light_flag_changes_variance_end_to_end() {
        use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
        let (w, h) = (64u32, 64u32);
        let buf = gray_ramp(w as usize, h as usize, 63);
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

    /// Equal content as u8 / lossless-u16 promotion is moot here (RGB8-only),
    /// but a uniform field must read ~zero variance through the SIMD reduction.
    #[test]
    fn uniform_field_is_zero_variance() {
        let buf = vec![137u8; 40 * 40 * 3];
        let slice = PixelSlice::new(&buf, 40, 40, 40 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let lin = linear_variance_rgb8(&slice, 500_000).expect("rgb8");
        assert!(lin.abs() < 1e-3, "uniform field variance ~0, got {lin}");
    }

    /// Non-RGB8 layout → None (caller keeps gamma).
    #[test]
    fn non_rgb8_returns_none() {
        let buf = vec![0u8; 16 * 4 * 4]; // RGBA8
        let slice = PixelSlice::new(&buf, 16, 4, 16 * 4, PixelDescriptor::RGBA8).unwrap();
        assert!(linear_variance_rgb8(&slice, 500_000).is_none());
    }
}
