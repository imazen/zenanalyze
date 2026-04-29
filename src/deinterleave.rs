//! RGB-interleaved → planar f32 primitive.
//!
//! `rgb24_to_planes` deinterleaves 8 packed RGB pixels (24 bytes) into three
//! independent `[f32; 8]` arrays — one per channel.  The arrays are returned
//! by value and are intended to be immediately loaded into `f32x8` SIMD
//! vectors by the calling tier.
//!
//! ## Architecture dispatch
//!
//! | Context               | Path                                    | Key instructions             |
//! |-----------------------|-----------------------------------------|------------------------------|
//! | x86_64 + AVX2+SSSE3   | `rgb24_to_planes_avx2`                  | `vpshufb` × 6 + `vpmovzxbd` × 3 + `vcvtdq2ps` × 3 |
//! | aarch64 (NEON)        | `rgb24_to_planes_neon`                  | `vld3.8` + widen + `ucvtf`   |
//! | everything else       | `rgb24_to_planes_scalar`                | scalar byte-to-float scatter |
//!
//! The NEON path is taken unconditionally on `aarch64` because NEON is the
//! ABI baseline.  The AVX2 path is gated by a one-time cached runtime check
//! (`is_x86_feature_detected!`); when called from within a
//! `#[target_feature(enable = "avx2")]` monomorphization the branch is
//! always-taken and the branch predictor keeps it free.

/// Deinterleave 8 packed RGB pixels (24 bytes) into three planar `[f32; 8]`
/// arrays.
///
/// Returns `(r, g, b)` where each array holds 8 channel values in the
/// original pixel order, as raw `f32` values in `[0.0, 255.0]`.  No gamma
/// or transfer-function adjustment is applied — this is a pure
/// byte-to-float widening operation suitable for zenanalyze's sRGB-encoded
/// hotpath.
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
    // x86_64: prefer AVX2+SSSE3 shuffle path.
    // `is_x86_feature_detected!` uses a cached atomic — one cheap load per
    // call. When inlined into an AVX2 #[target_feature] monomorphization the
    // branch is perfectly predicted (always taken on any modern host).
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: the runtime check above guarantees AVX2 (which implies
        // SSSE3) is available on this CPU.
        return unsafe { rgb24_to_planes_avx2(chunk) };
    }

    // aarch64: NEON is the ABI baseline — always available.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always present on aarch64.
        return unsafe { rgb24_to_planes_neon(chunk) };
    }

    // Scalar fallback for all other targets (or x86_64 without AVX2).
    rgb24_to_planes_scalar(chunk)
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

#[inline(always)]
fn rgb24_to_planes_scalar(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
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

// ---------------------------------------------------------------------------
// x86_64 AVX2 path
// ---------------------------------------------------------------------------

/// AVX2+SSSE3 deinterleave.
///
/// Strategy: load two overlapping 16-byte SSE registers covering the full
/// 24-byte pixel run, apply `vpshufb` masks to extract each channel into the
/// low 8 bytes, OR the halves together, then use `vpmovzxbd` + `vcvtdq2ps`
/// to widen and convert in one AVX2 step.
///
/// Mask derivation (hi = chunk[8..24], lo = chunk[0..16]):
///
/// ```text
/// lo[0..16] = R0 G0 B0 | R1 G1 B1 | R2 G2 B2 | R3 G3 B3 | R4 G4 B4 | R5
/// hi[0..16] = B2 R3 G3 | B3 R4 G4 | B4 R5 G5 | B5 R6 G6 | B6 R7 G7 | B7
///
/// R from lo: positions 0,3,6,9,12,15  → R0..R5
/// R from hi: positions 10,13          → R6,R7
/// G from lo: positions 1,4,7,10,13    → G0..G4
/// G from hi: positions 8,11,14        → G5..G7
/// B from lo: positions 2,5,8,11,14    → B0..B4
/// B from hi: positions 9,12,15        → B5..B7
/// ```
///
/// Mask byte = -128 (bit 7 set) ⇒ `vpshufb` zeroes that output lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,ssse3")]
unsafe fn rgb24_to_planes_avx2(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    use core::arch::x86_64::*;

    // -128_i8 = 0x80 → vpshufb zeroes that output lane.
    let z: i8 = -128;

    // R channel: lo positions 0,3,6,9,12,15 → R0..R5; hi positions 10,13 → R6,R7
    let shuf_r_lo = _mm_setr_epi8(0, 3, 6, 9, 12, 15, z, z, z, z, z, z, z, z, z, z);
    let shuf_r_hi = _mm_setr_epi8(z, z, z, z, z, z, 10, 13, z, z, z, z, z, z, z, z);

    // G channel: lo positions 1,4,7,10,13 → G0..G4; hi positions 8,11,14 → G5..G7
    let shuf_g_lo = _mm_setr_epi8(1, 4, 7, 10, 13, z, z, z, z, z, z, z, z, z, z, z);
    let shuf_g_hi = _mm_setr_epi8(z, z, z, z, z, 8, 11, 14, z, z, z, z, z, z, z, z);

    // B channel: lo positions 2,5,8,11,14 → B0..B4; hi positions 9,12,15 → B5..B7
    let shuf_b_lo = _mm_setr_epi8(2, 5, 8, 11, 14, z, z, z, z, z, z, z, z, z, z, z);
    let shuf_b_hi = _mm_setr_epi8(z, z, z, z, z, 9, 12, 15, z, z, z, z, z, z, z, z);

    // Two overlapping 16-byte loads covering all 24 bytes, and the channel
    // shuffle + OR, zero-extend, and float convert — all inside a single
    // `unsafe {}` block.
    unsafe {
        // Two overlapping 16-byte loads covering all 24 bytes.
        let lo = _mm_loadu_si128(chunk.as_ptr() as *const __m128i); // bytes  0..16
        let hi = _mm_loadu_si128(chunk.as_ptr().add(8) as *const __m128i); // bytes  8..24

        // Each shuffle + OR places the 8 channel bytes in lanes 0..7 of the XMM register.
        let r_xmm =
            _mm_or_si128(_mm_shuffle_epi8(lo, shuf_r_lo), _mm_shuffle_epi8(hi, shuf_r_hi));
        let g_xmm =
            _mm_or_si128(_mm_shuffle_epi8(lo, shuf_g_lo), _mm_shuffle_epi8(hi, shuf_g_hi));
        let b_xmm =
            _mm_or_si128(_mm_shuffle_epi8(lo, shuf_b_lo), _mm_shuffle_epi8(hi, shuf_b_hi));

        // vpmovzxbd: zero-extend first 8 u8 lanes → 8 × i32 (YMM).
        // vcvtdq2ps: convert 8 × i32 → 8 × f32 (YMM).
        let r_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r_xmm));
        let g_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g_xmm));
        let b_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b_xmm));

        let mut r = [0.0f32; 8];
        let mut g = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        _mm256_storeu_ps(r.as_mut_ptr(), r_f32);
        _mm256_storeu_ps(g.as_mut_ptr(), g_f32);
        _mm256_storeu_ps(b.as_mut_ptr(), b_f32);

        (r, g, b)
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON path
// ---------------------------------------------------------------------------

/// NEON deinterleave using the hardware structure-load `vld3.8`.
///
/// `vld3_u8` (8-element) loads 24 bytes and deinterleaves them into three
/// `uint8x8_t` lanes in a single instruction.  Two widening steps then
/// produce `float32x4_t` pairs per channel, avoiding the scalar scatter
/// loop entirely.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgb24_to_planes_neon(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
    use core::arch::aarch64::*;

    // u8×8 → u16×8 → u32×4 (lo/hi) → f32×4 (lo/hi) per channel.
    unsafe {
        // vld3.8 {d0,d1,d2}, [ptr] — 1 instruction, 3 deinterleaved u8x8 lanes.
        let rgb = vld3_u8(chunk.as_ptr());
        // rgb.0 = [R0,R1,R2,R3,R4,R5,R6,R7]
        // rgb.1 = [G0,G1,G2,G3,G4,G5,G6,G7]
        // rgb.2 = [B0,B1,B2,B3,B4,B5,B6,B7]

        let (r_arr, g_arr, b_arr) = (rgb.0, rgb.1, rgb.2);

        let mut r = [0.0f32; 8];
        let mut g = [0.0f32; 8];
        let mut b = [0.0f32; 8];

        for (arr, out) in [(r_arr, &mut r), (g_arr, &mut g), (b_arr, &mut b)] {
            let u16v = vmovl_u8(arr);
            let lo_f32 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16v)));
            let hi_f32 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16v)));
            vst1q_f32(out.as_mut_ptr(), lo_f32);
            vst1q_f32(out.as_mut_ptr().add(4), hi_f32);
        }

        (r, g, b)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: scalar deinterleave, used to validate the
    /// platform-specific paths.
    fn reference(chunk: &[u8; 24]) -> ([f32; 8], [f32; 8], [f32; 8]) {
        rgb24_to_planes_scalar(chunk)
    }

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
        let (r, g, b) = rgb24_to_planes_scalar(&chunk);
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
        let (ref_r, ref_g, ref_b) = reference(&chunk);
        let (r, g, b) = rgb24_to_planes(&chunk);
        assert_eq!(r, ref_r);
        assert_eq!(g, ref_g);
        assert_eq!(b, ref_b);
    }

    #[test]
    fn dispatched_matches_scalar() {
        // Verify that the dispatched path (AVX2 on x86_64, NEON on aarch64,
        // scalar elsewhere) produces bit-identical results to the reference.
        for seed in 0u8..=255 {
            let mut pixels = [[0u8; 3]; 8];
            for (i, px) in pixels.iter_mut().enumerate() {
                px[0] = seed.wrapping_add(i as u8 * 3);
                px[1] = seed.wrapping_add(i as u8 * 3 + 1);
                px[2] = seed.wrapping_add(i as u8 * 3 + 2);
            }
            let chunk = make_chunk(&pixels);
            let (ref_r, ref_g, ref_b) = reference(&chunk);
            let (r, g, b) = rgb24_to_planes(&chunk);
            assert_eq!(r, ref_r, "R mismatch at seed={seed}");
            assert_eq!(g, ref_g, "G mismatch at seed={seed}");
            assert_eq!(b, ref_b, "B mismatch at seed={seed}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return; // skip on machines without AVX2
        }
        for seed in 0u8..=255 {
            let mut pixels = [[0u8; 3]; 8];
            for (i, px) in pixels.iter_mut().enumerate() {
                px[0] = seed.wrapping_add(i as u8 * 7);
                px[1] = seed.wrapping_add(i as u8 * 7 + 1);
                px[2] = seed.wrapping_add(i as u8 * 7 + 2);
            }
            let chunk = make_chunk(&pixels);
            let (ref_r, ref_g, ref_b) = reference(&chunk);
            // SAFETY: guarded by is_x86_feature_detected above.
            let (r, g, b) = unsafe { rgb24_to_planes_avx2(&chunk) };
            assert_eq!(r, ref_r, "AVX2 R mismatch at seed={seed}");
            assert_eq!(g, ref_g, "AVX2 G mismatch at seed={seed}");
            assert_eq!(b, ref_b, "AVX2 B mismatch at seed={seed}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar() {
        for seed in 0u8..=255 {
            let mut pixels = [[0u8; 3]; 8];
            for (i, px) in pixels.iter_mut().enumerate() {
                px[0] = seed.wrapping_add(i as u8 * 7);
                px[1] = seed.wrapping_add(i as u8 * 7 + 1);
                px[2] = seed.wrapping_add(i as u8 * 7 + 2);
            }
            let chunk = make_chunk(&pixels);
            let (ref_r, ref_g, ref_b) = reference(&chunk);
            // SAFETY: NEON is always present on aarch64.
            let (r, g, b) = unsafe { rgb24_to_planes_neon(&chunk) };
            assert_eq!(r, ref_r, "NEON R mismatch at seed={seed}");
            assert_eq!(g, ref_g, "NEON G mismatch at seed={seed}");
            assert_eq!(b, ref_b, "NEON B mismatch at seed={seed}");
        }
    }
}
