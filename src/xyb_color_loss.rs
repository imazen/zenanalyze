//! XYB-vs-YCbCr **color-conversion-loss** picker feature
//! ([`crate::feature::AnalysisFeature::Xyb444ColorLoss`]).
//!
//! A zenjpeg JPEG can encode in BT.601 YCbCr or in libjxl-style XYB. The
//! two color spaces shed *different* information even at 4:4:4 (no chroma
//! subsampling): XYB's opsin → cube-root → 8-bit-sample round trip clips a
//! sliver of warm-gamut detail (saturated red/orange/yellow) that the
//! well-conditioned BT.601 matrix preserves. `xyb444_color_loss` is the
//! **favor-YCbCr** discriminant: the fraction of an image's pixels whose
//! color XYB's 4:4:4 conversion sheds while YCbCr 4:4:4 keeps it.
//!
//! It is orthogonal to the chroma-*sharpness* family
//! ([`crate::feature::AnalysisFeature::CbHorizSharpness`] et al.): those
//! measure chroma spatial *detail* (what 2× subsampling drops); this
//! measures gamut/saturation *clipping* in the unsubsampled transform.
//! Validation (all-GPU ssim2 + butteraugli over 459 images, 2026-06-05)
//! found it carries a unique signal — residual −0.26 vs an existing-feature
//! model — that no chroma-sharpness feature provides; the subsampling-loss
//! variants the prototype also explored were redundant and were dropped.
//! See `/mnt/v/zen/color-loss-research/accuracy_eval_2026-06-02.md`.
//!
//! ## Cost
//!
//! `xyb444_color_loss` is a property of the *color*, not the layout: a
//! given `(r,g,b)` either survives both 4:4:4 round trips or it doesn't.
//! The expensive opsin+cbrt predicate is therefore evaluated **offline**,
//! once, over a quantized color grid and baked into a 4 KiB `.rodata`
//! density LUT ([`CONV_LOSS_DENSITY`]). At runtime the feature is a strided
//! sample of cheap table lookups — near Tier-1 cost, no `cbrt`, no heap,
//! and no dependency on the transfer-function crate. The exact per-pixel
//! reference and the LUT regenerator live in the test module below.

// ---- runtime path: density LUT + strided sample (no cbrt, no alloc) ------

/// Embedded coarse per-RGB-region conversion-loss DENSITY LUT: 4 bits per
/// channel (16³ = 4096 bins), one byte per bin = the fraction of that bin's
/// 8-bit colors XYB's 4:4:4 conversion sheds, scaled `0..=255`. **4 KiB of
/// `.rodata`** — no runtime build (cold start == warm cost), no allocation.
/// Generated offline by the `regenerate_density_lut` test, which runs the
/// exact `cbrt` predicate (`reference::conv_only_lost_at`) over all 16.7 M
/// colors; the `lut_matches_predicate` test gates that the embedded bytes
/// still match the predicate. Regenerate after any opsin-constant change.
static CONV_LOSS_DENSITY: &[u8; 4096] = include_bytes!("conv_loss_density.bin");

/// 4-bit-per-channel bin index into [`CONV_LOSS_DENSITY`].
#[inline]
fn density_idx(r: u8, g: u8, b: u8) -> usize {
    ((r as usize >> 4) << 8) | ((g as usize >> 4) << 4) | (b as usize >> 4)
}

/// Per-image sample budget. `xyb444_color_loss` is a color statistic that
/// tolerates strided sampling; this caps per-image cost near Tier-1.
const SAMPLE_TARGET: usize = 1 << 16; // 65 536

/// Mean conversion-loss density over a strided sample of `rgb` (a tightly
/// packed `w*h*3` sRGB8 buffer; `n = w*h` pixels). Returns the expected
/// fraction of pixels in colors XYB's 4:4:4 conversion sheds — the
/// favor-YCbCr signal, range `[0, 1]`. `0.0` for an empty image.
pub(crate) fn xyb444_color_loss_rgb8(rgb: &[u8], w: usize, h: usize) -> f32 {
    let n = w * h;
    debug_assert_eq!(rgb.len(), n * 3, "rgb buffer must be tightly packed w*h*3");
    if n == 0 {
        return 0.0;
    }
    let stride = (n / SAMPLE_TARGET).max(1);
    let (mut sum, mut cnt) = (0u64, 0u64);
    let mut i = 0;
    while i < n {
        let base = i * 3;
        sum += CONV_LOSS_DENSITY[density_idx(rgb[base], rgb[base + 1], rgb[base + 2])] as u64;
        cnt += 1;
        i += stride;
    }
    sum as f32 / (cnt.max(1) as f32 * 255.0)
}

// ---- exact reference + LUT regenerator (test-only) -----------------------

#[cfg(test)]
mod reference {
    //! The exact per-color conversion-loss predicate the LUT is baked from.
    //! The conversion math mirrors `zenjpeg::color::xyb` /
    //! `zenjpeg::color::ycbcr` exactly (same opsin matrix, scale offsets,
    //! and `linear-srgb` transfer crate the encoder uses), so the LUT is
    //! numerically faithful to the encoder it models. Constants are
    //! duplicated from `zenjpeg/src/foundation/consts.rs` intentionally
    //! (zenanalyze does not depend on zenjpeg) and must be kept in sync.

    use linear_srgb::default::{linear_to_srgb, srgb_to_linear};

    /// Opsin absorbance matrix (row-major 3×3).
    const OPSIN: [f32; 9] = [
        0.30,
        0.622,
        0.078, // row 0
        0.23,
        0.692,
        0.078, // row 1
        0.243_422_69,
        0.204_767_44,
        0.551_809_87, // row 2
    ];
    /// Opsin absorbance bias (same for all three channels).
    const OPSIN_BIAS: f32 = 0.003_793_073_3;
    /// Inverse opsin matrix (row-major 3×3), from `xyb_to_linear_rgb`.
    const INV_OPSIN: [f32; 9] = [
        11.031_567, -9.866_944, -0.164_623, -3.254_147, 4.418_770, -0.164_623, -3.658_851,
        2.712_923, 1.945_928,
    ];
    /// Scaled-XYB offsets `[x, y, b]` (`scale_xyb`).
    const SCALED_OFFSET: [f32; 3] = [0.015_386_134, 0.0, 0.277_704_59];
    /// Scaled-XYB scales `[x, y, b]` (`scale_xyb`).
    const SCALED_SCALE: [f32; 3] = [22.995_788_804, 1.183_000_077, 1.502_141_333];

    /// BT.601 RGB→YCbCr on `0..255`.
    fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168_735_9 * r - 0.331_264_1 * g + 0.5 * b + 128.0;
        let cr = 0.5 * r - 0.418_687_6 * g - 0.081_312_4 * b + 128.0;
        (y, cb, cr)
    }
    /// BT.601 YCbCr→RGB on `0..255`.
    fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
        let cb = cb - 128.0;
        let cr = cr - 128.0;
        (
            y + 1.402 * cr,
            y - 0.344_136 * cb - 0.714_136 * cr,
            y + 1.772 * cb,
        )
    }
    /// sRGB8 → scaled XYB (`zenjpeg::color::xyb`), through 0..1 linear.
    fn srgb_to_scaled_xyb(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
        let lr = srgb_to_linear(r as f32 / 255.0);
        let lg = srgb_to_linear(g as f32 / 255.0);
        let lb = srgb_to_linear(b as f32 / 255.0);
        let or = (OPSIN[0] * lr + OPSIN[1] * lg + OPSIN[2] * lb + OPSIN_BIAS).max(0.0);
        let og = (OPSIN[3] * lr + OPSIN[4] * lg + OPSIN[5] * lb + OPSIN_BIAS).max(0.0);
        let ob = (OPSIN[6] * lr + OPSIN[7] * lg + OPSIN[8] * lb + OPSIN_BIAS).max(0.0);
        let nb = -OPSIN_BIAS.cbrt();
        let cr = or.cbrt() + nb;
        let cg = og.cbrt() + nb;
        let cb = ob.cbrt() + nb;
        let x = 0.5 * (cr - cg);
        let y = 0.5 * (cr + cg);
        let sx = (x + SCALED_OFFSET[0]) * SCALED_SCALE[0];
        let sy = (y + SCALED_OFFSET[1]) * SCALED_SCALE[1];
        let sb = (cb - y + SCALED_OFFSET[2]) * SCALED_SCALE[2];
        (sx, sy, sb)
    }
    /// Scaled XYB → sRGB8 (`xyb_to_linear_rgb` + inverse opsin).
    fn scaled_xyb_to_srgb(sx: f32, sy: f32, sb: f32) -> (u8, u8, u8) {
        let y = sy / SCALED_SCALE[1] - SCALED_OFFSET[1];
        let x = sx / SCALED_SCALE[0] - SCALED_OFFSET[0];
        let b_xyb = sb / SCALED_SCALE[2] - SCALED_OFFSET[2] + y;
        let nb = -OPSIN_BIAS.cbrt();
        let cr = y + x - nb;
        let cg = y - x - nb;
        let cb = b_xyb - nb;
        let mixed_cube = |v: f32| if v < 0.0 { -((-v).powi(3)) } else { v.powi(3) };
        let or = mixed_cube(cr) - OPSIN_BIAS;
        let og = mixed_cube(cg) - OPSIN_BIAS;
        let ob = mixed_cube(cb) - OPSIN_BIAS;
        let lr = INV_OPSIN[0] * or + INV_OPSIN[1] * og + INV_OPSIN[2] * ob;
        let lg = INV_OPSIN[3] * or + INV_OPSIN[4] * og + INV_OPSIN[5] * ob;
        let lb = INV_OPSIN[6] * or + INV_OPSIN[7] * og + INV_OPSIN[8] * ob;
        let to_u8 = |l: f32| (linear_to_srgb(l.clamp(0.0, 1.0)) * 255.0).round() as u8;
        (to_u8(lr), to_u8(lg), to_u8(lb))
    }
    fn clip_u8(v: f32) -> u8 {
        v.round().clamp(0.0, 255.0) as u8
    }
    fn max_abs_diff(a: (u8, u8, u8), b: (u8, u8, u8)) -> u32 {
        let dr = (a.0 as i32 - b.0 as i32).unsigned_abs();
        let dg = (a.1 as i32 - b.1 as i32).unsigned_abs();
        let db = (a.2 as i32 - b.2 as i32).unsigned_abs();
        dr.max(dg).max(db)
    }

    /// The exact 4:4:4 conversion-loss predicate for one color: XYB's
    /// opsin+cbrt+8-bit round trip sheds it (`> 8` max-channel error) while
    /// BT.601 YCbCr's keeps it (`<= 3`). This is what the density LUT bins.
    pub(crate) fn conv_only_lost_at(r: u8, g: u8, b: u8) -> bool {
        let orig = (r, g, b);
        let (y, cb, cr) = rgb_to_ycbcr(r as f32, g as f32, b as f32);
        let yc = ycbcr_to_rgb(y.round(), cb.round(), cr.round());
        let e_yc = max_abs_diff(orig, (clip_u8(yc.0), clip_u8(yc.1), clip_u8(yc.2)));
        let (sx, sy, sb) = srgb_to_scaled_xyb(r, g, b);
        let xq = (sx * 255.0).round() / 255.0;
        let yq = (sy * 255.0).round() / 255.0;
        let bq = (sb * 255.0).round() / 255.0;
        let xy = scaled_xyb_to_srgb(xq, yq, bq);
        let e_xy = max_abs_diff(orig, xy);
        e_xy > 8 && e_yc <= 3
    }

    /// Exact density (scaled `0..=255`) for one 4-bit-per-channel bin:
    /// the fraction of the bin's 4096 member colors that are conversion-lost.
    pub(crate) fn bin_density(rbin: usize, gbin: usize, bbin: usize) -> u8 {
        let mut lost = 0u32;
        for dr in 0..16u32 {
            for dg in 0..16u32 {
                for db in 0..16u32 {
                    let r = (rbin as u32 * 16 + dr) as u8;
                    let g = (gbin as u32 * 16 + dg) as u8;
                    let b = (bbin as u32 * 16 + db) as u8;
                    lost += conv_only_lost_at(r, g, b) as u32;
                }
            }
        }
        ((lost as f32 / 4096.0) * 255.0).round() as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    /// The embedded LUT must match the exact `cbrt` predicate it claims to
    /// bin. Full enumeration of every 256th bin's 4096 colors — a strided
    /// but exhaustive sample (no approximation) — gated as a hard equality.
    #[test]
    fn lut_matches_predicate() {
        let mut checked = 0;
        for bin in (0..4096usize).step_by(7) {
            let (rb, gb, bb) = ((bin >> 8) & 15, (bin >> 4) & 15, bin & 15);
            let want = reference::bin_density(rb, gb, bb);
            assert_eq!(
                CONV_LOSS_DENSITY[bin], want,
                "bin {bin} (R{rb} G{gb} B{bb}): embedded {} != predicate {want} — \
                 conv_loss_density.bin is stale; rerun regenerate_density_lut",
                CONV_LOSS_DENSITY[bin]
            );
            checked += 1;
        }
        assert!(checked > 500, "too few bins checked ({checked})");
    }

    /// Neutral gray survives both 4:4:4 round trips → ~zero conversion loss.
    #[test]
    fn gray_has_no_conversion_loss() {
        for v in [0u8, 64, 128, 192, 255] {
            let loss = xyb444_color_loss_rgb8(&solid(v, v, v, 32, 32), 32, 32);
            assert!(loss < 0.05, "gray {v}: loss {loss} should be ~0");
        }
    }

    /// Saturated warm gamut (where XYB's transform clips) must fire above
    /// neutral content, and the result stays a valid `[0, 1]` fraction.
    #[test]
    fn saturated_warm_fires() {
        let warm = xyb444_color_loss_rgb8(&solid(255, 40, 0, 32, 32), 32, 32);
        let gray = xyb444_color_loss_rgb8(&solid(128, 128, 128, 32, 32), 32, 32);
        assert!((0.0..=1.0).contains(&warm), "out of range: {warm}");
        assert!(
            warm > gray,
            "saturated warm ({warm}) should exceed gray ({gray})"
        );
    }

    /// Strided sampling is deterministic and tolerates non-square images.
    #[test]
    fn deterministic_and_nonsquare() {
        let img = {
            // checkerboard of saturated red and gray so the sample is mixed
            let (w, h) = (37usize, 19usize);
            let mut v = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    if (x + y) & 1 == 0 {
                        v.extend_from_slice(&[230, 30, 10]);
                    } else {
                        v.extend_from_slice(&[120, 120, 120]);
                    }
                }
            }
            (v, w, h)
        };
        let a = xyb444_color_loss_rgb8(&img.0, img.1, img.2);
        let b = xyb444_color_loss_rgb8(&img.0, img.1, img.2);
        assert_eq!(a, b, "must be deterministic");
        assert!(a.is_finite() && (0.0..=1.0).contains(&a));
        assert_eq!(xyb444_color_loss_rgb8(&[], 0, 0), 0.0, "empty image → 0");
    }

    /// Regenerate `conv_loss_density.bin` from the exact predicate over all
    /// 16.7 M colors. Not a correctness gate (that's `lut_matches_predicate`);
    /// run explicitly when the opsin constants change:
    /// `cargo test --features experimental -p zenanalyze regenerate_density_lut -- --ignored --nocapture`
    #[test]
    #[ignore = "regeneration tool — full 16.7M-color sweep, run on demand"]
    fn regenerate_density_lut() {
        let mut lut = vec![0u8; 4096];
        for rb in 0..16 {
            for gb in 0..16 {
                for bb in 0..16 {
                    lut[(rb << 8) | (gb << 4) | bb] = reference::bin_density(rb, gb, bb);
                }
            }
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/conv_loss_density.bin");
        std::fs::write(path, &lut).unwrap();
        eprintln!("wrote {} bytes to {path}", lut.len());
    }
}
