//! Always-full-scan palette tier — counts colour bins on every pixel.
//!
//! Why a separate tier: the corpus convergence study (139 images,
//! 1-50% sampling) showed `distinct_color_bins` is the single feature
//! that does NOT converge under stride sampling. At 25% sampling the
//! median error vs full scan is 27%; even at 50% it's 13%. Other
//! features (variance, edges, chroma, etc.) reach <5% by 10-25%
//! sampling, so it makes sense to budget them differently.
//!
//! Pre-0.1.0 the analyser also surfaced a Chao1-corrected estimate
//! (`DistinctColorBinsChao1`). At full sampling — which is the only
//! path the crate ships today — the Chao1 correction collapses to
//! zero and the estimate is identical to the raw count. Surfacing
//! two fields that are always equal invited bug reports, so the
//! variant was retired pre-0.1.0; its discriminant id is reserved
//! for the future budget-sampled palette variant where Chao1 *would*
//! diverge from the raw count.
//!
//! ## Storage choice: 32 KB byte-flag array, not 4 KB bitset
//!
//! Earlier revisions used a 4 KB `[u64; 512]` presence bitset with
//! `bits[idx>>6] |= 1<<(idx&63)` per pixel. That looks compact, but
//! every update is a read-modify-write on a 64-bit word — the load
//! must complete before the OR-store can issue, even when the bit was
//! already set. On a 4096×2048 RGB8 image that RMW chain hit ~6 ms.
//!
//! Switching to a `[u8; 32_768]` array indexed by the 15-bit RGB-bin
//! id, with `flags[idx] = 1` (pure store, no read), drops the same
//! scan to ~4 ms — a measured 1.47× on low-diversity (photo) input
//! and 1.17× on high-diversity (saturated 32K-bin) input. The 8×
//! larger working set still fits in L1 (32 KB matches the line size of
//! a single L1D way on most x86-64 / ARM cores). The final pop is a
//! linear sum / filter over the 32 KB byte array, which LLVM
//! autovectorizes into a tight reduction.
//!
//! A separately-prototyped magetypes `u32x8` SIMD index-compute path
//! that fed the same flag store turned out **slower** than the scalar
//! loop: the deinterleave-to-arrays + vector-load + store-back +
//! 8-iteration store roundtrip costs more than just letting LLVM
//! autovectorize the scalar `r >> 3 | g >> 3 << 5 | …`. Kept simple.

use super::row_stream::RowStream;
use archmage::autoversion;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PaletteStats {
    /// Distinct 5-bit-per-channel RGB bins observed (true full-image
    /// count under this full-pass tier). Set to 0 by `scan_palette_quick`
    /// — that path doesn't compute the exact count, only the
    /// fits-in-256 signal.
    pub distinct: u32,
    /// `ceil(log2(distinct))` clamped to `[1, 15]`, with `0`
    /// reserved for "truecolor" (> 32 768 distinct bins, which
    /// saturates the 5-bit-per-channel bin storage). Drives PNG /
    /// GIF / JXL palette-mode breakpoints.
    ///
    /// Computed exactly by `scan_palette` (full path). The
    /// `scan_palette_quick` path early-exits at 257 distinct bins
    /// and so produces values in `[1, 8]` or `0` only — JXL's
    /// 9..15 breakpoints require a full-path consumer
    /// (`DistinctColorBins`) to be co-requested.
    pub palette_log2_size: u32,
    /// Convenience: `distinct ≤ 256`.
    pub fits_in_256: bool,
    /// Total pixels walked by the scan (`width × height`). Used to
    /// compute fractions like `GrayscaleScore` from `non_grayscale`.
    /// Set by `scan_palette` only; `scan_palette_quick` may have
    /// early-exited before walking every pixel and leaves this at 0.
    pub total_pixels: u64,
    /// Pixels that failed the grayscale gate (`max(R, G, B) − min ≤ 4`).
    /// Computed on the same full walk that builds the bin histogram.
    /// **100 % coverage** — at stripe-sampled budgets a single colour
    /// pixel in a sea of gray would have ~95 % chance of being missed,
    /// breaking the binary "is this image grayscale?" classifier.
    /// Hosting it on the always-full-scan palette tier gives the
    /// right answer at no extra cost (the row walk is bandwidth-bound
    /// on the flag-array store; the 4-instruction max−min gate fits
    /// in the same target_feature loop).
    pub non_grayscale: u64,
}

/// Map a distinct-colour count to `ceil(log2(count))` clamped to
/// `[1, 15]`, with `24` as the saturation sentinel for "bin array is
/// full — image is truecolor; true 8-bit colour count is in
/// `[32 768, 2²⁴]`".
///
/// **Why 24 and not 0 for truecolor:** the trained picker is a small
/// MLP; a discontinuous `..., 14, 15, 0` jump fights the gradient
/// signal ("more colours → lower value" is wrong). Emitting 24 keeps
/// the scale monotonic — `..., 14, 15, [9-unit gap], 24` — and the
/// gap itself is a meaningful feature ("we ran out of measurement
/// resolution; this is unambiguously truecolor"). 24 is the bit-width
/// the source would need with no palette compression at all (8 bits
/// per channel × 3 channels), so it's a defensible upper bound.
///
/// Encoding rules:
/// - `count == 0` (empty image, edge case): `1` — degenerate, treated
///   as solid colour. We don't surface a separate sentinel; empty
///   images shouldn't reach this code path in the first place.
/// - `count == 1` (solid colour): `1` — 1-BPP indexed still encodes it.
/// - `count ∈ [2, 32_767]`: `ceil(log2(count))` ∈ `[1, 15]`.
///   `(count - 1).leading_zeros()` gives `32 - ceil_log2`.
/// - `count == 32_768` (bin array filled to capacity): `24`. The 5-bit
///   binning stores at most 32 768 distinct cells; reaching this is a
///   strong heuristic signal that the true 8-bit colour count is much
///   higher. False-positive rate: a synthetic image with exactly
///   32 768 distinct 5-bit-binned colours and not-truecolor source
///   would emit 24, but such images are vanishingly rare in real
///   workloads and the encoder would still try indexed mode if it
///   cares.
#[inline]
fn palette_log2_size_from_count(count: u32) -> u32 {
    if count >= 32_768 {
        // Saturated bin storage — truecolor heuristic. count > 32_768
        // is impossible from the full-scan path (the bin array can
        // hold at most 32 768 distinct cells), but the comparison is
        // safe and self-documenting.
        24
    } else if count <= 1 {
        // Solid-colour and degenerate-empty both fold to 1 (1 BPP
        // indexed still encodes a single colour). No 0 sentinel.
        1
    } else {
        // ceil(log2(count)) for count >= 2: 32 - (count-1).leading_zeros()
        32 - (count - 1).leading_zeros()
    }
}

/// Walk every row of the stream, count 5-bit-per-channel RGB bins.
///
/// Storage is a 32 KB `[u8; 32_768]` flag array indexed by the 15-bit
/// `(r>>3, g>>3, b>>3)` bin id. Each pixel does one unconditional byte
/// store (no read-modify-write); the final count is a linear pop over
/// the array, which LLVM autovectorizes. At full scan the raw count is
/// the population truth — no extrapolation needed.
pub(crate) fn scan_palette(stream: &mut RowStream<'_>, want_grayscale: bool) -> PaletteStats {
    let width = stream.width() as usize;
    let height = stream.height();
    if width == 0 || height == 0 {
        return PaletteStats::default();
    }
    // Boxed to avoid a 32 KB stack frame. One alloc per call; the
    // `flags[idx] = 1` pattern is overwhelmingly faster than the
    // earlier 4 KB bitset RMW chain (1.47× measured on a 8 MP photo,
    // 1.17× on a 32K-bin-saturated worst case).
    let mut flags: Box<[u8; 32_768]> = vec![0u8; 32_768]
        .into_boxed_slice()
        .try_into()
        .expect("32 KB heap alloc for palette flag array");

    // One autoversion dispatch for the entire scan: row loop + chunked
    // index compute + per-byte stores + final sum, all inside one
    // `#[target_feature]` boundary. Earlier per-row dispatch paid the
    // ~10 ns lookup `height` times and the wins from chunks-of-24
    // never amortized. Pulling the row loop in lets v3/v4 autovec the
    // index batch *and* the final reduction in the same code generation.
    let (distinct, non_grayscale) = scan_and_count(stream, &mut flags, want_grayscale);
    PaletteStats {
        distinct,
        palette_log2_size: palette_log2_size_from_count(distinct),
        fits_in_256: distinct <= 256,
        total_pixels: if want_grayscale {
            (width as u64) * (height as u64)
        } else {
            0
        },
        non_grayscale,
    }
}

/// Early-exit palette scan: walks pixels with a running distinct-count
/// and bails as soon as the count exceeds 256. Produces only the
/// quick-path signals (`palette_log2_size`, `fits_in_256`); the exact
/// distinct count is left at 0 because the scan stopped before the
/// final tally.
///
/// **Speed.** Same `#[autoversion]` + `chunks_exact(24)` treatment as
/// the full scan, plus the per-pixel `if flags[idx] == 0` running-count
/// check. An A/B against the full scan on a 16-colour 8 MP synthetic
/// (early-exit never fires; every pixel walked) measured the quick
/// path at **0.96× the full path's time** — the running-count branch
/// is well-predicted in steady state and autovec still applies to the
/// index-compute side; the saved 32 KB final pop-count more than pays
/// for the branch overhead. On truecolor content, early-exit lifts
/// that to a 2.4× win.
///
/// Used when the caller's [`crate::feature::FeatureSet`] requests
/// only the quick-path features ([`crate::feature::AnalysisFeature::PaletteLog2Size`]
/// or [`crate::feature::AnalysisFeature::PaletteFitsIn256`]) and not
/// the exact-count features. Note that `PaletteLog2Size` from the
/// quick path saturates at `8` (≤ 256 colours); JXL's 9..15
/// breakpoints require co-requesting `DistinctColorBins` to force
/// the full-scan path. Any overlap with the full-path features
/// (`DistinctColorBins` / `PaletteDensity`) routes through
/// `scan_palette` instead, which produces both signal classes from a
/// single pass.
pub(crate) fn scan_palette_quick(stream: &mut RowStream<'_>) -> PaletteStats {
    let width = stream.width() as usize;
    let height = stream.height();
    if width == 0 || height == 0 {
        return PaletteStats::default();
    }
    let mut flags: Box<[u8; 32_768]> = vec![0u8; 32_768]
        .into_boxed_slice()
        .try_into()
        .expect("32 KB heap alloc for palette flag array");
    let (count, exceeded) = scan_quick_inner(stream, &mut flags);
    // The quick path doesn't compute grayscale (the early-exit
    // structure can bail before walking all pixels, and the
    // grayscale gate has no analogous early-exit at this
    // granularity). Callers that request `GrayscaleScore` route
    // through the full `scan_palette` path because `GrayscaleScore`
    // is in `PALETTE_FULL_FEATURES`.
    PaletteStats {
        distinct: 0,
        palette_log2_size: if exceeded {
            0
        } else {
            palette_log2_size_from_count(count)
        },
        fits_in_256: !exceeded,
        total_pixels: 0,
        non_grayscale: 0,
    }
}

/// Drive the entire `RowStream` through the flag array and return
/// the popcount in one shot.
///
/// Chunked 8 pixels (24 bytes) at a time so LLVM sees a fixed-size
/// `[u8; 24]` view per chunk. The fixed-size proof eliminates interior
/// bounds checks and lets the autovectorizer (running under each
/// `#[autoversion]` tier's `#[target_feature]`) unroll the 8-pixel
/// batch into a tight register-resident sequence.
///
/// We can't pass `&mut RowStream` into the autoversioned function
/// because the `borrow_row` mutable-borrow lifetime is tied to the
/// stream object; the compiler can still hoist the row fetch loop
/// inside this fn since `borrow_row` is `#[inline]` and itself a tiny
/// pointer indirection. The hot work — chunked index compute + 32 KB
/// reduction — is what we wanted in the v4x context anyway.
/// Two-mode scan: with-grayscale or without. The runtime `want_gray`
/// flag dispatches to one of two `#[autoversion]`-tier inner kernels
/// so the without-grayscale path keeps its tight original codegen
/// (no per-pixel max/min/compare in the inner loop) while the
/// with-grayscale path adds the gate inside the same target_feature
/// region.
fn scan_and_count(
    stream: &mut RowStream<'_>,
    flags: &mut [u8; 32_768],
    want_gray: bool,
) -> (u32, u64) {
    if want_gray {
        scan_and_count_gray(stream, flags)
    } else {
        (scan_and_count_no_gray(stream, flags), 0)
    }
}

#[autoversion(v4x, v4, v3, neon, scalar)]
fn scan_and_count_no_gray(stream: &mut RowStream<'_>, flags: &mut [u8; 32_768]) -> u32 {
    let width = stream.width() as usize;
    let height = stream.height();
    let row_bytes = width * 3;
    let chunk_bytes = 24usize;
    for y in 0..height {
        let row = stream.borrow_row(y);
        let row = &row[..row_bytes.min(row.len())];
        let full_chunks = row.len() / chunk_bytes;
        for c in 0..full_chunks {
            let base = c * chunk_bytes;
            let chunk: &[u8; 24] = (&row[base..base + chunk_bytes]).try_into().unwrap();
            let mut i = 0;
            while i < 8 {
                let r = (chunk[i * 3] >> 3) as usize;
                let g = (chunk[i * 3 + 1] >> 3) as usize;
                let b = (chunk[i * 3 + 2] >> 3) as usize;
                let idx = (r << 10) | (g << 5) | b;
                flags[idx] = 1;
                i += 1;
            }
        }
        let tail_start = full_chunks * chunk_bytes;
        for px in row[tail_start..].chunks_exact(3) {
            let idx = (((px[0] >> 3) as usize) << 10)
                | (((px[1] >> 3) as usize) << 5)
                | ((px[2] >> 3) as usize);
            flags[idx] = 1;
        }
    }
    flags.iter().map(|&f| f as u32).sum()
}

#[autoversion(v4x, v4, v3, neon, scalar)]
fn scan_and_count_gray(stream: &mut RowStream<'_>, flags: &mut [u8; 32_768]) -> (u32, u64) {
    let width = stream.width() as usize;
    let height = stream.height();
    let row_bytes = width * 3;
    let chunk_bytes = 24usize;
    let mut non_gray: u64 = 0;
    for y in 0..height {
        let row = stream.borrow_row(y);
        let row = &row[..row_bytes.min(row.len())];
        let full_chunks = row.len() / chunk_bytes;
        for c in 0..full_chunks {
            let base = c * chunk_bytes;
            let chunk: &[u8; 24] = (&row[base..base + chunk_bytes]).try_into().unwrap();
            let mut i = 0;
            while i < 8 {
                let r = chunk[i * 3];
                let g = chunk[i * 3 + 1];
                let b = chunk[i * 3 + 2];
                let idx =
                    (((r >> 3) as usize) << 10) | (((g >> 3) as usize) << 5) | ((b >> 3) as usize);
                flags[idx] = 1;
                let mx = r.max(g).max(b);
                let mn = r.min(g).min(b);
                non_gray += (mx - mn > 4) as u64;
                i += 1;
            }
        }
        let tail_start = full_chunks * chunk_bytes;
        for px in row[tail_start..].chunks_exact(3) {
            let r = px[0];
            let g = px[1];
            let b = px[2];
            let idx =
                (((r >> 3) as usize) << 10) | (((g >> 3) as usize) << 5) | ((b >> 3) as usize);
            flags[idx] = 1;
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            non_gray += (mx - mn > 4) as u64;
        }
    }
    let distinct: u32 = flags.iter().map(|&f| f as u32).sum();
    (distinct, non_gray)
}

/// Same row+chunks(24) skeleton as `scan_and_count`, but with a
/// running distinct-count check and early-exit at 256. Returns
/// `(count, exceeded)` — `exceeded == true` means the scan bailed
/// before finishing; `count` is `≤ 257` either way.
///
/// An A/B against the unconditional `scan_and_count` on a 16-colour
/// 8 MP synthetic (early-exit never fires; every pixel walked) showed
/// this inner loop running at **0.96×** the unconditional scan's
/// time. The per-pixel branch is well-predicted in steady state and
/// the autovectorizer still applies to the index-compute side; the
/// 32 KB final pop-count saved by the quick path more than pays for
/// the branch overhead. The earlier "17 % slower" measurement was
/// entirely due to *missing* autoversion + chunks(24) — the branch
/// itself isn't the bottleneck.
#[autoversion(v4x, v4, v3, neon, scalar)]
fn scan_quick_inner(stream: &mut RowStream<'_>, flags: &mut [u8; 32_768]) -> (u32, bool) {
    let width = stream.width() as usize;
    let height = stream.height();
    let row_bytes = width * 3;
    let chunk_bytes = 24usize;
    let mut count: u32 = 0;
    for y in 0..height {
        let row = stream.borrow_row(y);
        let row = &row[..row_bytes.min(row.len())];
        let full_chunks = row.len() / chunk_bytes;
        for c in 0..full_chunks {
            let base = c * chunk_bytes;
            let chunk: &[u8; 24] = (&row[base..base + chunk_bytes]).try_into().unwrap();
            let mut i = 0;
            while i < 8 {
                let r = (chunk[i * 3] >> 3) as usize;
                let g = (chunk[i * 3 + 1] >> 3) as usize;
                let b = (chunk[i * 3 + 2] >> 3) as usize;
                let idx = (r << 10) | (g << 5) | b;
                if flags[idx] == 0 {
                    flags[idx] = 1;
                    count += 1;
                    if count > 256 {
                        return (count, true);
                    }
                }
                i += 1;
            }
        }
        // Tail: < 8 pixels.
        let tail_start = full_chunks * chunk_bytes;
        for px in row[tail_start..].chunks_exact(3) {
            let idx = (((px[0] >> 3) as usize) << 10)
                | (((px[1] >> 3) as usize) << 5)
                | ((px[2] >> 3) as usize);
            if flags[idx] == 0 {
                flags[idx] = 1;
                count += 1;
                if count > 256 {
                    return (count, true);
                }
            }
        }
    }
    (count, false)
}

#[cfg(test)]
mod log2_size_tests {
    use super::palette_log2_size_from_count;

    #[test]
    fn codomain_matches_indexed_bpp_breakpoints() {
        // Edge cases — empty + solid both fold to 1 (no 0 sentinel).
        assert_eq!(palette_log2_size_from_count(0), 1, "empty → 1 (no 0 sentinel)");
        assert_eq!(palette_log2_size_from_count(1), 1, "solid color → 1 BPP");

        // PNG-indexed bit-widths.
        assert_eq!(palette_log2_size_from_count(2), 1, "PNG-1");
        assert_eq!(palette_log2_size_from_count(3), 2, "GIF BPP=2 / PNG-2");
        assert_eq!(palette_log2_size_from_count(4), 2, "PNG-2 boundary");
        assert_eq!(palette_log2_size_from_count(5), 3, "GIF BPP=3");
        assert_eq!(palette_log2_size_from_count(8), 3, "GIF BPP=3 boundary");
        assert_eq!(palette_log2_size_from_count(9), 4, "PNG-4");
        assert_eq!(palette_log2_size_from_count(16), 4, "PNG-4 boundary");
        assert_eq!(palette_log2_size_from_count(17), 5, "GIF BPP=5");
        assert_eq!(palette_log2_size_from_count(32), 5);
        assert_eq!(palette_log2_size_from_count(33), 6, "GIF BPP=6");
        assert_eq!(palette_log2_size_from_count(64), 6);
        assert_eq!(palette_log2_size_from_count(65), 7, "GIF BPP=7");
        assert_eq!(palette_log2_size_from_count(128), 7);
        assert_eq!(palette_log2_size_from_count(129), 8);
        assert_eq!(palette_log2_size_from_count(256), 8, "PNG-8 / GIF-8 boundary");

        // JXL palette breakpoints (full-scan path only).
        assert_eq!(palette_log2_size_from_count(257), 9);
        assert_eq!(palette_log2_size_from_count(512), 9);
        assert_eq!(palette_log2_size_from_count(513), 10);
        assert_eq!(palette_log2_size_from_count(1024), 10, "JXL 1024 break");
        assert_eq!(palette_log2_size_from_count(4096), 12, "JXL 4096 break");
        assert_eq!(palette_log2_size_from_count(16_384), 14);
        assert_eq!(palette_log2_size_from_count(32_767), 15, "just under cap");

        // Truecolor saturation sentinel — count == 32_768 (bin array
        // filled) folds to 24 to keep the trained-MLP gradient signal
        // monotonic. count > 32_768 is impossible from the full-scan
        // path but we guard it for safety.
        assert_eq!(palette_log2_size_from_count(32_768), 24, "saturated → 24");
        assert_eq!(palette_log2_size_from_count(u32::MAX), 24, "guard");

        // Monotonicity sanity over the whole codomain. The 9-unit gap
        // from 15 → 24 is the only discontinuity; everywhere else the
        // mapping is non-decreasing in count.
        let mut prev = 0u32;
        for c in [
            0, 1, 2, 4, 16, 256, 1024, 4096, 16384, 32_767, 32_768,
        ] {
            let v = palette_log2_size_from_count(c);
            assert!(v >= prev, "non-monotonic: count={c} → {v} after {prev}");
            prev = v;
        }
    }
}
