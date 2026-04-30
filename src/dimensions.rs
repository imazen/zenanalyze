//! Dimension features — pure descriptor math, no per-pixel work.
//!
//! Computes 11 features from `(width, height, descriptor)` alone:
//! `PixelCount`, `LogPixels`, `MinDim`, `MaxDim`, `BitmapBytes`,
//! `AspectMinOverMax`, `LogAspectAbs`, `BlockMisalignment{8,16,32,64}`,
//! `ChannelCount`. See issue #42 for rationale.
//!
//! All cheap enough that we always populate the `RawAnalysis` fields
//! unconditionally — the `into_results` filter then drops any features
//! the caller didn't request. The cost is a handful of integer ops
//! plus one `ln`; under 10 ns per call on a 7950X.
//!
//! ## Block-misalignment math
//!
//! `block_loss(N)` is the fraction of *padding* pixels needed to
//! round the visible image up to a complete `N × N` grid:
//!
//! ```text
//! padded(N) = ((N - w % N) % N + w) * ((N - h % N) % N + h)
//! loss(N)   = padded(N) / (w * h) - 1.0
//! ```
//!
//! Examples:
//! - 256×256 at N=8: 0.0 (already aligned)
//! - 257×257 at N=8: 264*264/(257*257) - 1 ≈ 5.5 %
//! - 1000×1 at N=8: 1000*8/(1000*1) - 1 = 7.0
//!
//! The codec applies its per-block overhead (DCT, prediction signal-
//! ing, motion vectors) to padded blocks too — this fraction is
//! exactly the "wasted" overhead vs a perfectly-aligned image.

use zenpixels::PixelDescriptor;

use crate::feature::RawAnalysis;

/// Populate the dimension fields on `raw`. Always-on — no
/// `requested` gate here; the `into_results` translator filters per
/// the caller's `FeatureSet` after the fact.
pub(crate) fn populate_dimensions(
    raw: &mut RawAnalysis,
    width: u32,
    height: u32,
    descriptor: PixelDescriptor,
) {
    let w64 = width as u64;
    let h64 = height as u64;
    let pixels_u64 = w64 * h64;

    raw.pixel_count = pixels_u64.min(u32::MAX as u64) as u32;
    raw.log_pixels = if pixels_u64 == 0 {
        0.0
    } else {
        (pixels_u64 as f64).ln() as f32
    };
    raw.min_dim = width.min(height);
    raw.max_dim = width.max(height);

    let channels = descriptor.layout().channels() as u64;
    let bytes_per_sample = descriptor.channel_type().byte_size() as u64;
    raw.channel_count = channels as u32;
    let bytes_u64 = pixels_u64
        .saturating_mul(channels)
        .saturating_mul(bytes_per_sample);
    raw.bitmap_bytes = bytes_u64 as f32;

    let (mn, mx) = if width <= height {
        (width as f32, height as f32)
    } else {
        (height as f32, width as f32)
    };
    raw.aspect_min_over_max = if mx == 0.0 { 0.0 } else { mn / mx };
    // |ln(w/h)| = ln(max/min) since max ≥ min ≥ 0; the abs falls out
    // of the swap above. Avoid a redundant ln(w/h) + abs() — costs a
    // log either way but this form skips one reciprocal.
    raw.log_aspect_abs = if mn == 0.0 || mx == 0.0 {
        0.0
    } else {
        (mx / mn).ln()
    };

    raw.block_misalignment_8 = block_misalignment(width, height, 8);
    raw.block_misalignment_16 = block_misalignment(width, height, 16);
    raw.block_misalignment_32 = block_misalignment(width, height, 32);
    raw.block_misalignment_64 = block_misalignment(width, height, 64);

    // Log / derivative variants — give the network multiple
    // numerical handles for the same underlying resolution signal.
    // Lets ablation distinguish "real size dependence" from "memo-
    // rized PixelCount" (the latter shows up as PixelCount having
    // huge ablation impact while every log variant is near zero).
    if pixels_u64 > 0 {
        let pixels_f64 = pixels_u64 as f64;
        raw.log2_pixels = (pixels_f64.log2()) as f32;
        raw.log10_pixels = (pixels_f64.log10()) as f32;
        // Round ln(pixels) to the nearest 0.5 — quantized smooth
        // bucket signal. `(ln*2).round()/2` form.
        raw.log_pixels_rounded = ((pixels_f64.ln() * 2.0).round() / 2.0) as f32;
        raw.sqrt_pixels = (pixels_f64.sqrt()) as f32;
    }
    if bytes_u64 > 0 {
        raw.log_bitmap_bytes = ((bytes_u64 as f64).ln()) as f32;
    }
    if width > 0 && height > 0 {
        raw.log_min_dim = ((width.min(height) as f64).ln()) as f32;
        raw.log_max_dim = ((width.max(height) as f64).ln()) as f32;
    }
    raw.log_padded_pixels_8 = log_padded_pixels(width, height, 8);
    raw.log_padded_pixels_16 = log_padded_pixels(width, height, 16);
    raw.log_padded_pixels_32 = log_padded_pixels(width, height, 32);
    raw.log_padded_pixels_64 = log_padded_pixels(width, height, 64);
}

/// `ln(ceil(w/N)*N × ceil(h/N)*N)` — log of the block-padded
/// encoded surface area at block size N. For aligned images equals
/// `ln(w*h)`; off-bucket sizes are slightly larger.
fn log_padded_pixels(width: u32, height: u32, block: u32) -> f32 {
    if width == 0 || height == 0 || block == 0 {
        return 0.0;
    }
    let pad_w = (block - width % block) % block;
    let pad_h = (block - height % block) % block;
    let padded_w = (width + pad_w) as u64;
    let padded_h = (height + pad_h) as u64;
    let padded = padded_w * padded_h;
    if padded == 0 {
        0.0
    } else {
        ((padded as f64).ln()) as f32
    }
}

/// Returns `padded(N) / (w * h) - 1.0` — the fraction of *padding*
/// pixels needed to round up to a complete N×N grid. `0.0` if both
/// dims are exact multiples of N.
fn block_misalignment(width: u32, height: u32, block: u32) -> f32 {
    if width == 0 || height == 0 || block == 0 {
        return 0.0;
    }
    let pad_w = (block - width % block) % block;
    let pad_h = (block - height % block) % block;
    let padded_w = (width + pad_w) as u64;
    let padded_h = (height + pad_h) as u64;
    let pixels = (width as u64) * (height as u64);
    let padded = padded_w * padded_h;
    if pixels == 0 {
        0.0
    } else {
        (padded as f64 / pixels as f64 - 1.0) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::PixelDescriptor;

    fn rgb8() -> PixelDescriptor {
        PixelDescriptor::RGB8_SRGB
    }

    #[test]
    fn aligned_square_has_zero_misalignment() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 256, 256, rgb8());
        assert_eq!(raw.pixel_count, 65_536);
        assert_eq!(raw.min_dim, 256);
        assert_eq!(raw.max_dim, 256);
        assert!((raw.aspect_min_over_max - 1.0).abs() < 1e-6);
        assert!(raw.log_aspect_abs.abs() < 1e-6);
        assert_eq!(raw.block_misalignment_8, 0.0);
        assert_eq!(raw.block_misalignment_16, 0.0);
        assert_eq!(raw.block_misalignment_32, 0.0);
        assert_eq!(raw.block_misalignment_64, 0.0);
    }

    #[test]
    fn off_by_one_257_square_loses_about_5_percent_at_8() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 257, 257, rgb8());
        // padded 264×264, loss = 264*264/(257*257) - 1 ≈ 0.0552
        assert!(
            (raw.block_misalignment_8 - 0.0552).abs() < 1e-3,
            "got {}",
            raw.block_misalignment_8
        );
    }

    #[test]
    fn extreme_strip_pegs_aspect_signals() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 1024, 1, rgb8());
        assert_eq!(raw.min_dim, 1);
        assert_eq!(raw.max_dim, 1024);
        assert!(
            (raw.aspect_min_over_max - 1.0 / 1024.0).abs() < 1e-6,
            "got {}",
            raw.aspect_min_over_max
        );
        // |ln(1024/1)| = ln(1024) ≈ 6.931
        assert!(
            (raw.log_aspect_abs - 6.931_472).abs() < 1e-3,
            "got {}",
            raw.log_aspect_abs
        );
    }

    #[test]
    fn portrait_and_landscape_have_same_log_aspect_abs() {
        let mut a = RawAnalysis::default();
        let mut b = RawAnalysis::default();
        populate_dimensions(&mut a, 800, 600, rgb8());
        populate_dimensions(&mut b, 600, 800, rgb8());
        // Symmetry check: a strip is the same extremity score whether
        // it's tall or wide.
        assert!((a.log_aspect_abs - b.log_aspect_abs).abs() < 1e-6);
        assert!((a.aspect_min_over_max - b.aspect_min_over_max).abs() < 1e-6);
    }

    #[test]
    fn bitmap_bytes_includes_channels_and_byte_depth() {
        let mut raw = RawAnalysis::default();
        // RGB8 = 3 channels × 1 byte/sample.
        populate_dimensions(&mut raw, 100, 100, PixelDescriptor::RGB8_SRGB);
        assert_eq!(raw.bitmap_bytes, (100 * 100 * 3 * 1) as f32);
        assert_eq!(raw.channel_count, 3);

        // RGBA16 = 4 channels × 2 bytes/sample.
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 100, 100, PixelDescriptor::RGBA16_SRGB);
        assert_eq!(raw.bitmap_bytes, (100 * 100 * 4 * 2) as f32);
        assert_eq!(raw.channel_count, 4);

        // RGBAF32 = 4 channels × 4 bytes/sample.
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 100, 100, PixelDescriptor::RGBAF32_LINEAR);
        assert_eq!(raw.bitmap_bytes, (100 * 100 * 4 * 4) as f32);
        assert_eq!(raw.channel_count, 4);
    }

    #[test]
    fn log_pixels_matches_expected() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 1024, 1024, rgb8());
        // ln(1024^2) = 2 ln 1024 = 13.863
        assert!((raw.log_pixels - 13.863).abs() < 1e-2);
    }

    #[test]
    fn block_misalignment_strict_strip() {
        // 1000 × 1 at block=8 → padded to 1000 × 8, loss = 8x - 1 = 7.
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 1000, 1, rgb8());
        assert!(
            (raw.block_misalignment_8 - 7.0).abs() < 1e-3,
            "got {}",
            raw.block_misalignment_8
        );
    }

    #[test]
    fn log_variants_match_expected() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 1024, 1024, rgb8());
        // 1024² = 1 048 576
        // ln = 13.8629, log2 = 20, log10 = 6.0206, sqrt = 1024
        assert!(
            (raw.log2_pixels - 20.0).abs() < 1e-3,
            "log2_pixels: got {}",
            raw.log2_pixels
        );
        assert!((raw.log10_pixels - 6.0206).abs() < 1e-3);
        assert!((raw.sqrt_pixels - 1024.0).abs() < 0.5);
        // Rounded ln(1024²) = round(13.8629 * 2)/2 = round(27.7258)/2 = 28/2 = 14.0
        assert!(
            (raw.log_pixels_rounded - 14.0).abs() < 1e-3,
            "log_pixels_rounded: got {}",
            raw.log_pixels_rounded
        );
    }

    #[test]
    fn log_min_max_dim_separates_strips() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 1024, 16, rgb8());
        // ln(16) ≈ 2.773, ln(1024) ≈ 6.931
        assert!((raw.log_min_dim - 2.773).abs() < 1e-2);
        assert!((raw.log_max_dim - 6.931).abs() < 1e-2);
    }

    #[test]
    fn log_padded_pixels_aligned_matches_log_pixels() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 256, 256, rgb8());
        // 256 is a multiple of 8/16/32/64 — all padded values equal
        // the visible log_pixels.
        assert!(
            (raw.log_padded_pixels_8 - raw.log_pixels).abs() < 1e-5,
            "8-aligned: log_padded {} vs log_pixels {}",
            raw.log_padded_pixels_8,
            raw.log_pixels
        );
        assert!((raw.log_padded_pixels_16 - raw.log_pixels).abs() < 1e-5);
        assert!((raw.log_padded_pixels_32 - raw.log_pixels).abs() < 1e-5);
        assert!((raw.log_padded_pixels_64 - raw.log_pixels).abs() < 1e-5);
    }

    #[test]
    fn log_padded_pixels_off_by_one_pads_up() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 257, 257, rgb8());
        // 257 → padded to 264 (block 8), 272 (block 16), 288 (32), 320 (64).
        let expect_8 = (264.0_f64 * 264.0).ln() as f32;
        let expect_16 = (272.0_f64 * 272.0).ln() as f32;
        let expect_32 = (288.0_f64 * 288.0).ln() as f32;
        let expect_64 = (320.0_f64 * 320.0).ln() as f32;
        assert!(
            (raw.log_padded_pixels_8 - expect_8).abs() < 1e-5,
            "8: got {} expect {}",
            raw.log_padded_pixels_8,
            expect_8
        );
        assert!((raw.log_padded_pixels_16 - expect_16).abs() < 1e-5);
        assert!((raw.log_padded_pixels_32 - expect_32).abs() < 1e-5);
        assert!((raw.log_padded_pixels_64 - expect_64).abs() < 1e-5);
        // Padded values should be strictly greater than the
        // unpadded `log_pixels`.
        assert!(raw.log_padded_pixels_8 > raw.log_pixels);
        assert!(raw.log_padded_pixels_64 > raw.log_padded_pixels_32);
    }

    #[test]
    fn log_bitmap_bytes_matches_expected() {
        let mut raw = RawAnalysis::default();
        // 100×100 RGBA16 = 100*100*4*2 = 80 000 bytes.
        populate_dimensions(&mut raw, 100, 100, PixelDescriptor::RGBA16_SRGB);
        let expect = (80_000.0_f64).ln() as f32;
        assert!((raw.log_bitmap_bytes - expect).abs() < 1e-3);
    }

    #[test]
    fn zero_dim_inputs_are_safe() {
        let mut raw = RawAnalysis::default();
        populate_dimensions(&mut raw, 0, 100, rgb8());
        // No panics, no NaN, all zeros.
        assert_eq!(raw.pixel_count, 0);
        assert_eq!(raw.bitmap_bytes, 0.0);
        assert_eq!(raw.block_misalignment_8, 0.0);
        assert_eq!(raw.aspect_min_over_max, 0.0);
        assert_eq!(raw.log_aspect_abs, 0.0);
        assert_eq!(raw.log_pixels, 0.0);
        assert_eq!(raw.log2_pixels, 0.0);
        assert_eq!(raw.log10_pixels, 0.0);
        assert_eq!(raw.sqrt_pixels, 0.0);
        assert_eq!(raw.log_padded_pixels_8, 0.0);
    }
}
