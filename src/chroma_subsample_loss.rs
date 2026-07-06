//! IDCT-roundtrip chroma 4:2:0-subsampling cost feature.
//!
//! The direct 4:2:0-vs-4:4:4 discriminator, complementing the spatial
//! [`crate::xyb_color_loss::xyb_bquarter_chroma_loss_from_stream`] proxy: per 8×8
//! chroma block, run the real codec spectral path — forward DCT → quantize /
//! dequantize at a representative chroma quality (jpegli distance 2) → inverse
//! DCT — then measure the energy a 2× chroma subsample+upsample would remove from
//! THAT quantized reconstruction.
//!
//! High ⇒ detail survives 4:4:4 quantization but dies under a 4:2:0 subsample
//! (favor 4:4:4); low ⇒ quantization already removed it, so 4:2:0 is free. This
//! captures the quality-dependence a pure spatial chroma-variance proxy cannot:
//! coarse quant → smooth reconstruction → subsampling is cheap; fine quant →
//! detail retained → subsampling costs. Cb and Cr are averaged.

use crate::row_stream::RowStream;
use crate::tier3::{DCT_COEF, JPEGLI_QUANT_C_D2};

/// Per-image sampled-block budget (~Tier-3 high-freq class).
const SAMPLE_BLOCKS: usize = 256;

/// BT.601 chroma, centered around 0 (the pre-DCT convention). The +128 offset is
/// intentionally dropped — the subsampling loss is a difference (`recon - mean`)
/// in which any DC offset cancels.
#[inline]
fn cbcr(r: f32, g: f32, b: f32) -> (f32, f32) {
    let cb = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    (cb, cr)
}

/// 2D 8×8 forward DCT-II: `Y = D · X · Dᵀ` (D = orthonormal [`DCT_COEF`]).
#[inline]
fn fdct2d(x: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut t = [[0f32; 8]; 8]; // t = D · X
    for u in 0..8 {
        for y in 0..8 {
            let mut s = 0.0;
            for k in 0..8 {
                s += DCT_COEF[u][k] * x[k][y];
            }
            t[u][y] = s;
        }
    }
    let mut yc = [[0f32; 8]; 8]; // Y = t · Dᵀ  →  Y[u][v] = Σ_y t[u][y]·D[v][y]
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0.0;
            for y in 0..8 {
                s += t[u][y] * DCT_COEF[v][y];
            }
            yc[u][v] = s;
        }
    }
    yc
}

/// 2D 8×8 inverse DCT-II: `X = Dᵀ · Y · D` (D orthonormal ⇒ D⁻¹ = Dᵀ).
#[inline]
fn idct2d(yc: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut t = [[0f32; 8]; 8]; // t = Dᵀ · Y  →  t[x][v] = Σ_u D[u][x]·Y[u][v]
    for x in 0..8 {
        for v in 0..8 {
            let mut s = 0.0;
            for u in 0..8 {
                s += DCT_COEF[u][x] * yc[u][v];
            }
            t[x][v] = s;
        }
    }
    let mut xr = [[0f32; 8]; 8]; // X = t · D  →  X[x][y] = Σ_v t[x][v]·D[v][y]
    for x in 0..8 {
        for y in 0..8 {
            let mut s = 0.0;
            for v in 0..8 {
                s += t[x][v] * DCT_COEF[v][y];
            }
            xr[x][y] = s;
        }
    }
    xr
}

/// Marginal 4:2:0 cost on one chroma block: forward DCT, quantize + dequantize at
/// `q` (a row-major 8×8 quant table), inverse DCT, then the RMS of what a 2× box
/// subsample + nearest upsample removes from that reconstruction.
fn block_subsample_dct_loss(block: &[[f32; 8]; 8], q: &[f32; 64]) -> f32 {
    let mut yc = fdct2d(block);
    for u in 0..8 {
        for v in 0..8 {
            let qq = q[u * 8 + v];
            yc[u][v] = (yc[u][v] / qq).round() * qq;
        }
    }
    let recon = idct2d(&yc);
    // 2×2 box subsample, then nearest upsample; accumulate the removed energy.
    let mut ss = 0.0f32;
    for by in 0..4 {
        for bx in 0..4 {
            let (y, x) = (2 * by, 2 * bx);
            let avg =
                0.25 * (recon[y][x] + recon[y][x + 1] + recon[y + 1][x] + recon[y + 1][x + 1]);
            for dy in 0..2 {
                for dx in 0..2 {
                    let d = recon[y + dy][x + dx] - avg;
                    ss += d * d;
                }
            }
        }
    }
    (ss / 64.0).sqrt()
}

/// Mean IDCT-roundtrip 4:2:0 chroma-subsampling loss over sampled 8×8 blocks (Cb
/// and Cr averaged). The favor-4:4:4 signal. `NaN` for sub-8×8 inputs — no 8×8
/// block exists to measure, so there is nothing to report; `0.0` would be a
/// misleading sentinel (indistinguishable from a genuinely computed
/// zero-loss result) rather than "not computed" (matches the crate-wide NaN
/// convention: [`crate::feature::RawAnalysis::into_results`] drops NaN
/// fields from the results, same as any other dropped feature). Walks the
/// `RowStream` 8 rows at a time; same block-stride sampling as the other
/// Tier-3 block features.
pub(crate) fn chroma_subsample_dct_loss_from_stream(stream: &mut RowStream<'_>) -> f32 {
    let w = stream.width() as usize;
    let h = stream.height() as usize;
    let (bw, bh) = (w / 8, h / 8);
    let nblocks = bw * bh;
    if nblocks == 0 {
        return f32::NAN;
    }
    let bstride = (nblocks / SAMPLE_BLOCKS).max(1);
    let row_bytes = w * 3;
    let mut rows = vec![0u8; 8 * row_bytes];
    let (mut sum, mut cnt) = (0.0f64, 0u64);
    for by in 0..bh {
        // first sampled block in this block-row (raster block order)
        let first = (bstride - (by * bw) % bstride) % bstride;
        if first >= bw {
            continue;
        }
        let y0 = (by * 8) as u32;
        stream.fetch_range(y0..y0 + 8, &mut rows);
        let mut bx = first;
        while bx < bw {
            let px0 = bx * 8;
            let mut cb = [[0f32; 8]; 8];
            let mut cr = [[0f32; 8]; 8];
            for ry in 0..8 {
                let row = &rows[ry * row_bytes..(ry + 1) * row_bytes];
                for cx in 0..8 {
                    let o = (px0 + cx) * 3;
                    let (b_cb, b_cr) = cbcr(row[o] as f32, row[o + 1] as f32, row[o + 2] as f32);
                    cb[ry][cx] = b_cb;
                    cr[ry][cx] = b_cr;
                }
            }
            sum += block_subsample_dct_loss(&cb, &JPEGLI_QUANT_C_D2) as f64;
            sum += block_subsample_dct_loss(&cr, &JPEGLI_QUANT_C_D2) as f64;
            cnt += 2;
            bx += bstride;
        }
    }
    (sum / cnt.max(1) as f64) as f32
}

/// Contiguous-buffer reference — test-only (production path is the row stream).
#[cfg(test)]
pub(crate) fn chroma_subsample_dct_loss_rgb8(rgb: &[u8], w: usize, h: usize) -> f32 {
    if w < 8 || h < 8 {
        // Match chroma_subsample_dct_loss_from_stream's contract exactly: NaN
        // ("not computed"), not 0.0 (a misleading sentinel indistinguishable
        // from a genuinely computed zero-loss result).
        return f32::NAN;
    }
    let slice = zenpixels::PixelSlice::new(
        rgb,
        w as u32,
        h as u32,
        w * 3,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    chroma_subsample_dct_loss_from_stream(&mut RowStream::new(slice).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checker() -> [[f32; 8]; 8] {
        let mut b = [[0f32; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                b[i][j] = if (i + j) % 2 == 0 { 60.0 } else { -60.0 };
            }
        }
        b
    }

    #[test]
    fn fdct_idct_is_identity() {
        // Orthonormal DCT: inverse∘forward reconstructs the block (no quant).
        let x = checker();
        let r = idct2d(&fdct2d(&x));
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (x[i][j] - r[i][j]).abs() < 1e-3,
                    "roundtrip {} vs {}",
                    x[i][j],
                    r[i][j]
                );
            }
        }
    }

    #[test]
    fn flat_block_zero_loss() {
        // Flat chroma → DC-only spectrum → subsample removes nothing.
        let flat = [[37.0f32; 8]; 8];
        assert!(block_subsample_dct_loss(&flat, &JPEGLI_QUANT_C_D2) < 1e-3);
    }

    #[test]
    fn horizontal_nyquist_loss_is_amplitude() {
        // Near-lossless quant (all 1s): a ±40 horizontal alternation survives the
        // roundtrip; a 2×2 box average cancels it exactly, so the removed RMS == 40.
        let q = [1.0f32; 64];
        let mut b = [[0f32; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                b[i][j] = if j % 2 == 0 { 40.0 } else { -40.0 };
            }
        }
        let l = block_subsample_dct_loss(&b, &q);
        assert!(
            (l - 40.0).abs() < 0.5,
            "horizontal-Nyquist loss {l} (expected ~40)"
        );
    }

    #[test]
    fn detailed_chroma_exceeds_flat_image() {
        // A high-chroma-detail image scores strictly above a flat-gray one.
        let (w, h) = (32, 32);
        let flat: Vec<u8> = vec![128; w * h * 3];
        let mut detail = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * 3;
                // alternate strongly-saturated red/blue per pixel → heavy Cb/Cr detail
                let (r, g, b) = if (x + y) % 2 == 0 {
                    (220, 20, 20)
                } else {
                    (20, 20, 220)
                };
                detail[o] = r;
                detail[o + 1] = g;
                detail[o + 2] = b;
            }
        }
        let lf = chroma_subsample_dct_loss_rgb8(&flat, w, h);
        let ld = chroma_subsample_dct_loss_rgb8(&detail, w, h);
        assert!(lf < 1e-3, "flat image loss {lf}");
        assert!(
            ld > 1.0,
            "detailed image loss {ld} should be well above flat"
        );
    }

    #[test]
    fn too_small_is_nan_not_a_misleading_zero() {
        // A sub-8x8 input has no 8x8 block to measure at all — NaN ("not
        // computed"), not 0.0 (which would be indistinguishable from a
        // genuinely computed zero-loss result). Was previously (wrongly)
        // 0.0; this test used to pin that exact misleading-sentinel bug.
        assert!(chroma_subsample_dct_loss_rgb8(&[0u8; 4 * 4 * 3], 4, 4).is_nan());
    }
}
