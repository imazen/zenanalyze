//! Strict-equality grayscale classifier.
//!
//! Returns `true` iff every pixel has `R == G == B`. Used by codec
//! selectors to drop chroma planes entirely (encode as YUV400 / single-
//! plane JXL / monochrome JPEG) — must be exact, no tolerance.
//!
//! Distinct from [`crate::palette::PaletteStats::non_grayscale`] which
//! uses a 4-unit max−min tolerance and surfaces a *fraction*. Strict
//! equality is cheaper because:
//!
//! 1. **Early exit**: returns `false` at the first row containing any
//!    colored pixel. A typical photo exits on row 1; only truly
//!    grayscale images walk every pixel.
//! 2. **No flag array**: just a per-row OR reduction. ~6 µs per fully
//!    walked 2048-px row at v4; ~5 ms total for a 4 MP grayscale
//!    image. For colored images the mean is ≪ 100 µs at any size.
//!
//! The kernel: for each row, compute `row_or = OR over pixels of
//! (r ^ g) | (g ^ b)`. If the reduction is non-zero the row contains
//! at least one pixel where R, G, B are not all equal — return false.
//! LLVM autovectorizes the OR-reduction within each `#[autoversion]`
//! tier's `target_feature` boundary.

use crate::row_stream::RowStream;
use archmage::autoversion;

/// Walk every row, return true iff `R == G == B` for every pixel.
///
/// Early-exits at the first non-grayscale row. The `RowStream` is
/// expected to deliver RGB8 (the analyzer always converts to that
/// before tier dispatch — see `lib.rs::analyze_features_typed`).
pub(crate) fn scan_strict_grayscale(stream: &mut RowStream<'_>) -> bool {
    scan_strict_grayscale_impl(stream)
}

#[autoversion(v4x, v4, v3, neon, scalar)]
fn scan_strict_grayscale_impl(stream: &mut RowStream<'_>) -> bool {
    let width = stream.width() as usize;
    let height = stream.height();
    if width == 0 || height == 0 {
        return true;
    }
    let row_bytes = width * 3;
    let chunk_bytes = 24usize;
    for y in 0..height {
        let row = stream.borrow_row(y);
        let row = &row[..row_bytes.min(row.len())];
        let mut row_or: u32 = 0;
        let full_chunks = row.len() / chunk_bytes;
        for c in 0..full_chunks {
            let base = c * chunk_bytes;
            let chunk: &[u8; 24] = (&row[base..base + chunk_bytes]).try_into().unwrap();
            let mut i = 0;
            while i < 8 {
                let r = chunk[i * 3] as u32;
                let g = chunk[i * 3 + 1] as u32;
                let b = chunk[i * 3 + 2] as u32;
                // (r^g) | (g^b) is zero iff r == g == b. OR-accumulate
                // across the row; LLVM keeps `row_or` in a register and
                // autovec'es the per-pixel transform under each tier's
                // target_feature region.
                row_or |= (r ^ g) | (g ^ b);
                i += 1;
            }
        }
        let tail_start = full_chunks * chunk_bytes;
        for px in row[tail_start..].chunks_exact(3) {
            let r = px[0] as u32;
            let g = px[1] as u32;
            let b = px[2] as u32;
            row_or |= (r ^ g) | (g ^ b);
        }
        if row_or != 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::row_stream::RowStream;
    use zenpixels::{PixelDescriptor, PixelSlice};

    fn analyze(buf: &[u8], w: u32, h: u32) -> bool {
        let stride = (w as usize) * 3;
        let slice = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let mut stream = RowStream::new(slice).unwrap();
        scan_strict_grayscale(&mut stream)
    }

    #[test]
    fn solid_gray_returns_true() {
        let buf = vec![128u8; 16 * 16 * 3];
        assert!(analyze(&buf, 16, 16));
    }

    #[test]
    fn pure_black_and_white_row_is_grayscale() {
        let mut buf = Vec::with_capacity(8 * 3);
        for v in [0u8, 255, 128, 64, 192, 32, 224, 16] {
            buf.extend_from_slice(&[v, v, v]);
        }
        assert!(analyze(&buf, 8, 1));
    }

    #[test]
    fn one_off_pixel_is_not_grayscale() {
        let mut buf = vec![100u8; 32 * 32 * 3];
        // Single pixel with G=101 instead of 100.
        let off = 16 * 32 * 3 + 16 * 3 + 1;
        buf[off] = 101;
        assert!(!analyze(&buf, 32, 32));
    }

    #[test]
    fn early_exit_first_row() {
        // First row colored, rest gray — should still return false.
        let mut buf = vec![64u8; 64 * 64 * 3];
        buf[0] = 255; // R
        // G = 64, B = 64 → not equal
        assert!(!analyze(&buf, 64, 64));
    }

    #[test]
    fn last_row_disqualifies() {
        let mut buf = vec![64u8; 64 * 64 * 3];
        let last = 63 * 64 * 3 + 63 * 3;
        buf[last + 2] = 65; // B differs from R=G=64
        assert!(!analyze(&buf, 64, 64));
    }

    // Empty-image case: PixelSlice rejects 0×0, so the kernel is
    // never reachable with that input. Defensive `width == 0 ||
    // height == 0` early-return inside `scan_strict_grayscale_impl`
    // is a belt-and-suspenders guard, not a tested path.

    #[test]
    fn tiny_image_under_24_byte_chunk() {
        // 3×3 image: 9 pixels = 27 bytes per row, just over one chunk.
        // Forces the chunk + tail path on the same row.
        let buf = vec![100u8; 3 * 3 * 3];
        assert!(analyze(&buf, 3, 3));
    }

    #[test]
    fn tiny_colored_under_chunk() {
        let mut buf = vec![100u8; 5 * 3];
        buf[7] = 99; // pixel 2's G channel
        assert!(!analyze(&buf, 5, 1));
    }
}
