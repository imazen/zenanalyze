//! Prototype **linear-light** tier-1 override, opt-in via
//! [`crate::feature::AnalysisQuery::with_linear_light`].
//!
//! The default tier 1/2/3 path is transfer-blind — it computes on gamma-encoded
//! code values (see `benchmarks/linear_light_precision_review_2026-06-14.md`).
//! This module recomputes the supported features in **linear light**: it
//! linearizes each sample through the sRGB EOTF before the statistic. Today that
//! is just `Variance` (the feature whose ranking moved most in the A/B —
//! `benchmarks/linear_light_ab_2026-06-14.md`); coverage grows as the kernels
//! land.
//!
//! Scope (prototype): **RGB8-layout sources only**, **scalar**. Returns `None`
//! for other layouts so the caller keeps the gamma value. The representation
//! bench (`benchmarks/linear_repr_bench_2026-06-14.md`) settled the eventual
//! production kernel on an **i16 / i12** linear intermediate; this scalar f32
//! pass produces the same *values* (precision is irrelevant per the A/B) and is
//! the correctness-first stub the SIMD kernel will replace.
//!
//! The linear luma is kept on the **[0,255] scale** (linearize to [0,1] then
//! ×255) so the returned variance has the same magnitude as the gamma
//! `Variance` it replaces — a drop-in for `RawAnalysis::variance`.

use zenpixels::{PixelDescriptor, PixelSlice};

/// BT.601 luma weights — match `tier1`'s f32 path for the sRGB/BT.709 baseline.
const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;

/// 256-entry sRGB→linear LUT on the [0,255] scale.
fn srgb_to_linear_255_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for (v, e) in lut.iter_mut().enumerate() {
        *e = linear_srgb::tf::srgb_to_linear(v as f32 / 255.0) * 255.0;
    }
    lut
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
    let lut = srgb_to_linear_255_lut();

    // Sample every `row_stride`-th row so total samples ≈ pixel_budget.
    let row_stride = ((w * h) / pixel_budget.max(1)).max(1);
    let (mut s, mut sq, mut n) = (0.0f64, 0.0f64, 0u64);
    let mut y = 0usize;
    while y < h {
        let row = slice.row(y as u32);
        let row = &row[..w * 3];
        for px in row.chunks_exact(3) {
            let luma =
                KR * lut[px[0] as usize] + KG * lut[px[1] as usize] + KB * lut[px[2] as usize];
            let l = luma as f64;
            s += l;
            sq += l * l;
            n += 1;
        }
        y += row_stride;
    }
    if n == 0 {
        return None;
    }
    let mean = s / n as f64;
    Some((sq / n as f64 - mean * mean).max(0.0) as f32)
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

    /// Non-RGB8 layout → None (caller keeps gamma).
    #[test]
    fn non_rgb8_returns_none() {
        let buf = vec![0u8; 16 * 4 * 4]; // RGBA8
        let slice = PixelSlice::new(&buf, 16, 4, 16 * 4, PixelDescriptor::RGBA8).unwrap();
        assert!(linear_variance_rgb8(&slice, 500_000).is_none());
    }
}
