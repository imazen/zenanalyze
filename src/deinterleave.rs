//! RGB-interleaved → planar f32 primitive.
//!
//! `rgb24_to_planes` deinterleaves 8 packed RGB pixels (24 bytes) into three
//! independent `[f32; 8]` arrays — one per channel.  The arrays are returned
//! by value and are intended to be immediately loaded into `f32x8` SIMD
//! vectors by the calling tier.
//!
//! ## Architecture dispatch
//!
//! Uses the archmage `#[arcane]` / `incant!` pattern — zero `unsafe` blocks.
//!
//! | Context             | Path               | Key instructions                          |
//! |---------------------|--------------------|-------------------------------------------|
//! | x86_64 + AVX2+SSSE3 | `_v3`              | `vpshufb`×6 + `vpmovzxbd`×3 + `vcvtdq2ps`×3 |
//! | aarch64 (NEON)      | `_neon`            | scalar loop compiled with NEON features; LLVM autovectorizes |
//! | wasm32 (simd128)    | `_wasm128`         | scalar loop compiled with simd128 features |
//! | everything else     | `_scalar`          | scalar byte-to-float scatter              |
//!
//! The `#[arcane]` wrappers for NEON and wasm128 compile the scalar loop
//! under the respective `#[target_feature]` context, letting LLVM
//! autovectorize without explicit structure-load intrinsics.  This matches
//! garb's approach: explicit `vld3q`/`vst3q` structure loads were found to
//! be slower than autovectorized scalar on Ampere, Apple Silicon, and
//! Snapdragon.

use archmage::prelude::*;

// ============================================================================
// x86_64 AVX2+SSSE3
// ============================================================================

/// AVX2+SSSE3 deinterleave via `vpshufb`.
///
/// Loads two overlapping 16-byte windows covering all 24 source bytes,
/// applies `vpshufb` masks to scatter each channel's bytes into lanes 0–7
/// of an XMM register, then uses `vpmovzxbd` + `vcvtdq2ps` to widen and
/// convert.
///
/// `X64V3Token::TARGET_FEATURES` includes `ssse3`, so `_mm_shuffle_epi8`
/// is available in this function's feature context.  Value-based intrinsics
/// are safe inside `#[target_feature]` functions since Rust 1.87+.
/// Memory intrinsics use the `safe_unaligned_simd` reference-based wrappers
/// brought in via `archmage::prelude::*`.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn rgb24_to_planes_v3(_token: X64V3Token, chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let z: i8 = -128; // vpshufb: bit-7 set → zero the output lane

    // Shuffle masks: collect R/G/B bytes from the lo (bytes 0–15) and hi
    // (bytes 8–23) 16-byte windows into output lanes 0–7.
    //
    // Position layout (lo window bytes 0-based):
    //   lo = R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    //   hi = B2 R3 G3 B3 R4 G4 B4 R5 G5 B5 R6 G6 B6 R7 G7 B7
    //         (hi[0..16] = chunk[8..24])
    //
    //   R from lo: positions 0,3,6,9,12,15  → R0..R5
    //   R from hi: positions 10,13          → R6,R7
    //   G from lo: positions 1,4,7,10,13    → G0..G4
    //   G from hi: positions 8,11,14        → G5..G7
    //   B from lo: positions 2,5,8,11,14    → B0..B4
    //   B from hi: positions 9,12,15        → B5..B7
    let shuf_r_lo = _mm_setr_epi8(0, 3, 6, 9, 12, 15, z, z, z, z, z, z, z, z, z, z);
    let shuf_r_hi = _mm_setr_epi8(z, z, z, z, z, z, 10, 13, z, z, z, z, z, z, z, z);
    let shuf_g_lo = _mm_setr_epi8(1, 4, 7, 10, 13, z, z, z, z, z, z, z, z, z, z, z);
    let shuf_g_hi = _mm_setr_epi8(z, z, z, z, z, 8, 11, 14, z, z, z, z, z, z, z, z);
    let shuf_b_lo = _mm_setr_epi8(2, 5, 8, 11, 14, z, z, z, z, z, z, z, z, z, z, z);
    let shuf_b_hi = _mm_setr_epi8(z, z, z, z, z, 9, 12, 15, z, z, z, z, z, z, z, z);

    // Two overlapping unaligned 16-byte loads — safe via archmage::prelude::*
    // which shadows core::arch::x86_64::_mm_loadu_si128 with a reference-based
    // wrapper (safe_unaligned_simd).
    let lo_arr: &[u8; 16] = chunk[..16].try_into().unwrap();
    let hi_arr: &[u8; 16] = chunk[8..].try_into().unwrap();
    let lo = _mm_loadu_si128(lo_arr);
    let hi = _mm_loadu_si128(hi_arr);

    // Each vpshufb+vpor places the 8 channel bytes in XMM lanes 0..7.
    let r_xmm = _mm_or_si128(_mm_shuffle_epi8(lo, shuf_r_lo), _mm_shuffle_epi8(hi, shuf_r_hi));
    let g_xmm = _mm_or_si128(_mm_shuffle_epi8(lo, shuf_g_lo), _mm_shuffle_epi8(hi, shuf_g_hi));
    let b_xmm = _mm_or_si128(_mm_shuffle_epi8(lo, shuf_b_lo), _mm_shuffle_epi8(hi, shuf_b_hi));

    // vpmovzxbd: zero-extend 8 × u8 → 8 × i32 (YMM).
    // vcvtdq2ps: 8 × i32 → 8 × f32 (YMM).
    let r_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r_xmm));
    let g_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_xmm));
    let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_xmm));

    // archmage::prelude::* _mm256_storeu_ps takes &mut [f32; 8] (not *mut f32).
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    _mm256_storeu_ps(&mut r, r_f32);
    _mm256_storeu_ps(&mut g, g_f32);
    _mm256_storeu_ps(&mut b, b_f32);

    (r, g, b)
}

// ============================================================================
// NEON — scalar body, compiled with NEON target_feature
// ============================================================================

/// NEON path.
///
/// The `#[arcane]` wrapper compiles the scalar loop under
/// `#[target_feature(enable = "neon")]`, which lets LLVM autovectorize it
/// without explicit `vld3`/`vst3` structure-load intrinsics.  This is the
/// same strategy garb uses for 3-channel operations on aarch64.
#[cfg(target_arch = "aarch64")]
#[arcane]
fn rgb24_to_planes_neon(_token: NeonToken, chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    deinterleave_scalar(chunk)
}

// ============================================================================
// wasm128 — scalar body, compiled with simd128 target_feature
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[arcane]
fn rgb24_to_planes_wasm128(
    _token: Wasm128Token,
    chunk: &[u8; 24],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    deinterleave_scalar(chunk)
}

// ============================================================================
// Scalar
// ============================================================================

fn rgb24_to_planes_scalar(_token: ScalarToken, chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    deinterleave_scalar(chunk)
}

/// Inner scalar implementation shared by every non-AVX2 tier.
#[inline(always)]
fn deinterleave_scalar(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..8 {
        r[i] = chunk[i * 3] as f32;
        g[i] = chunk[i * 3 + 1] as f32;
        b[i] = chunk[i * 3 + 2] as f32;
    }
    (r, g, b)
}

// ============================================================================
// Public dispatch
// ============================================================================

/// Deinterleave 8 packed RGB pixels (24 bytes) into three planar `[f32; 8]`
/// arrays.
///
/// Returns `(r, g, b)` where each array holds 8 channel values in the
/// original pixel order, as raw `f32` values in `[0.0, 255.0]`.  No gamma
/// or transfer-function adjustment is applied — this is a pure
/// byte-to-float widening operation.
///
/// Dispatch: x86_64+AVX2 → `vpshufb` shuffle path; aarch64/wasm32 → scalar
/// body compiled with the arch's target_feature (LLVM autovectorizes); all
/// others → plain scalar.
///
/// # Example
///
/// ```ignore
/// let chunk: &[u8; 24] = ...;
/// let (r_arr, g_arr, b_arr) = rgb24_to_planes(chunk);
/// let r = f32x8::load(token, &r_arr);
/// let g = f32x8::load(token, &g_arr);
/// let b = f32x8::load(token, &b_arr);
/// ```
#[inline(always)]
pub(crate) fn rgb24_to_planes(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    incant!(rgb24_to_planes(chunk), [v3, neon, wasm128, scalar])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(pixels: &[[u8; 3]; 8]) -> [u8; 24] {
        let mut out = [0u8; 24];
        for (i, px) in pixels.iter().enumerate() {
            out[i * 3] = px[0];
            out[i * 3 + 1] = px[1];
            out[i * 3 + 2] = px[2];
        }
        out
    }

    #[test]
    fn scalar_identity() {
        let pixels = [
            [10u8, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [100, 110, 120],
            [130, 140, 150],
            [160, 170, 180],
            [190, 200, 210],
            [220, 230, 240],
        ];
        let chunk = make_chunk(&pixels);
        let token = ScalarToken::summon().unwrap();
        let (r, g, b) = rgb24_to_planes_scalar(token, &chunk);
        for (i, px) in pixels.iter().enumerate() {
            assert_eq!(r[i], px[0] as f32, "R[{i}]");
            assert_eq!(g[i], px[1] as f32, "G[{i}]");
            assert_eq!(b[i], px[2] as f32, "B[{i}]");
        }
    }

    #[test]
    fn scalar_extremes() {
        let chunk = make_chunk(&[
            [0, 0, 0],
            [255, 255, 255],
            [0, 255, 0],
            [255, 0, 255],
            [128, 64, 32],
            [1, 2, 3],
            [253, 254, 255],
            [127, 128, 129],
        ]);
        let token = ScalarToken::summon().unwrap();
        let (ref_r, ref_g, ref_b) = rgb24_to_planes_scalar(token, &chunk);
        let (r, g, b) = rgb24_to_planes(&chunk);
        assert_eq!(r, ref_r);
        assert_eq!(g, ref_g);
        assert_eq!(b, ref_b);
    }

    #[test]
    fn dispatched_matches_scalar() {
        // Verify that the dispatched path (AVX2 on x86_64, NEON on aarch64,
        // scalar elsewhere) produces bit-identical results to the reference.
        let token = ScalarToken::summon().unwrap();
        for seed in 0u8..=255 {
            let mut pixels = [[0u8; 3]; 8];
            for (i, px) in pixels.iter_mut().enumerate() {
                px[0] = seed.wrapping_add(i as u8 * 3);
                px[1] = seed.wrapping_add(i as u8 * 3 + 1);
                px[2] = seed.wrapping_add(i as u8 * 3 + 2);
            }
            let chunk = make_chunk(&pixels);
            let (ref_r, ref_g, ref_b) = rgb24_to_planes_scalar(token, &chunk);
            let (r, g, b) = rgb24_to_planes(&chunk);
            assert_eq!(r, ref_r, "R mismatch at seed={seed}");
            assert_eq!(g, ref_g, "G mismatch at seed={seed}");
            assert_eq!(b, ref_b, "B mismatch at seed={seed}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        let Some(token_v3) = X64V3Token::summon() else {
            return; // skip on CPUs without AVX2
        };
        let token_sc = ScalarToken::summon().unwrap();
        for seed in 0u8..=255 {
            let mut pixels = [[0u8; 3]; 8];
            for (i, px) in pixels.iter_mut().enumerate() {
                px[0] = seed.wrapping_add(i as u8 * 7);
                px[1] = seed.wrapping_add(i as u8 * 7 + 1);
                px[2] = seed.wrapping_add(i as u8 * 7 + 2);
            }
            let chunk = make_chunk(&pixels);
            let (ref_r, ref_g, ref_b) = rgb24_to_planes_scalar(token_sc, &chunk);
            // rgb24_to_planes_v3 is the safe #[arcane] wrapper — no unsafe needed.
            let (r, g, b) = rgb24_to_planes_v3(token_v3, &chunk);
            assert_eq!(r, ref_r, "AVX2 R mismatch at seed={seed}");
            assert_eq!(g, ref_g, "AVX2 G mismatch at seed={seed}");
            assert_eq!(b, ref_b, "AVX2 B mismatch at seed={seed}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        let Some(token_neon) = NeonToken::summon() else {
            return;
        };
        let token_sc = ScalarToken::summon().unwrap();
        for seed in 0u8..=255 {
            let mut pixels = [[0u8; 3]; 8];
            for (i, px) in pixels.iter_mut().enumerate() {
                px[0] = seed.wrapping_add(i as u8 * 7);
                px[1] = seed.wrapping_add(i as u8 * 7 + 1);
                px[2] = seed.wrapping_add(i as u8 * 7 + 2);
            }
            let chunk = make_chunk(&pixels);
            let (ref_r, ref_g, ref_b) = rgb24_to_planes_scalar(token_sc, &chunk);
            // rgb24_to_planes_neon is the safe #[arcane] wrapper — no unsafe needed.
            let (r, g, b) = rgb24_to_planes_neon(token_neon, &chunk);
            assert_eq!(r, ref_r, "NEON R mismatch at seed={seed}");
            assert_eq!(g, ref_g, "NEON G mismatch at seed={seed}");
            assert_eq!(b, ref_b, "NEON B mismatch at seed={seed}");
        }
    }
}
