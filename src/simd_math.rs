//! Deterministic SIMD math helpers, and a cross-platform determinism probe.
//!
//! magetypes' `rsqrt_approx()` is a **hardware instruction** whose precision (and
//! exact result) differs per backend — x86 `rsqrtps` (~12-bit), AVX-512
//! `vrsqrt14` (~14-bit), NEON `vrsqrte` (~8-bit) — and its Newton-refined sibling
//! `rsqrt()` seeds from that hardware estimate, so it inherits the per-backend
//! difference in its low bits. Either one feeds a feature with **per-architecture
//! divergence** (see `docs/feature-cross-platform-divergence-2026-06-20.md`).
//!
//! [`rsqrt_stable!`] is a drop-in that is **bit-identical on every backend**: a
//! software bit-trick seed (integer ops on the float bits) refined by Newton-
//! Raphson in pure f32 `*`/`-` (NO `mul_add` → no FMA, which is itself
//! backend-dependent). f32 mul/sub are IEEE-correctly-rounded, so the whole thing
//! is deterministic. It keeps the speed of an approximation (no hardware `sqrt`
//! latency) while removing the cross-platform divergence.
//!
//! It is a `macro_rules!` rather than a `fn` so it expands inside a `#[magetypes]`
//! body against that body's per-tier `f32x8` (which the macro re-types to
//! `f32x16`/`f32x4`/… per backend) — a plain generic `fn` can't ride that
//! re-typing. The magic constant routes through `f32x8::splat(from_bits(..))
//! .bitcast_to_i()` so the integer vector is the *matching-width* companion of
//! the float vector without ever naming `i32xN`.

use archmage::magetypes;

/// Deterministic reciprocal square root `≈ 1/√x` (bit-identical across x86 /
/// AVX-512 / NEON / i686 / wasm). `$vec` is the in-scope SIMD float type, `$token`
/// the archmage token, `$x` a positive `$vec`. Quake seed
/// `0x5f3759df - (bits(x) >> 1)` + 2 Newton steps `y·(1.5 − 0.5·x·y²)`, explicit
/// `*`/`-` only (no FMA). ~5e-4 relative — better than the ~8–14-bit hardware
/// `rsqrt_approx`, and the *same* value on every architecture.
macro_rules! rsqrt_stable {
    ($vec:ident, $token:expr, $x:expr) => {{
        let x = $x;
        // Magic-constant int vector = matching-width companion of `x`, obtained
        // without naming i32xN: splat the float whose bits are the magic, reinterpret.
        let magic = $vec::splat($token, f32::from_bits(0x5f37_59df)).bitcast_to_i32();
        let y0 = (magic - x.bitcast_to_i32().shr_logical_const::<1>()).bitcast_to_f32();
        let half = $vec::splat($token, 0.5);
        let onehalf = $vec::splat($token, 1.5);
        // Two Newton steps, explicit mul/sub (mul_add would fuse to a
        // backend-dependent FMA and break determinism).
        let y1 = y0 * (onehalf - half * x * y0 * y0);
        y1 * (onehalf - half * x * y1 * y1)
    }};
}
pub(crate) use rsqrt_stable;

/// Scalar counterpart of [`rsqrt_stable!`], **bit-identical** to one SIMD lane
/// (same f32 ops, same order) — so a kernel's SIMD body and its scalar tail agree
/// exactly. Same Quake seed + 2 Newton steps, explicit `*`/`-` (no `mul_add`).
#[inline]
pub(crate) fn rsqrt_stable_scalar(x: f32) -> f32 {
    let y0 = f32::from_bits(0x5f37_59df - (x.to_bits() >> 1));
    let y1 = y0 * (1.5 - 0.5 * x * y0 * y0);
    y1 * (1.5 - 0.5 * x * y1 * y1)
}

/// Compute the gradient magnitude `√x` four ways per input, so a test can measure
/// each method's cross-platform spread. `x` is `grad_sq` (the edge kernel's
/// quantity); magnitude = `x · rsqrt(x)`, except `exact` = `√x`. Lengths equal and
/// a multiple of 8.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn magnitude_methods(
    token: Token,
    x: &[f32],
    approx: &mut [f32],
    mt_rsqrt: &mut [f32],
    stable: &mut [f32],
    exact: &mut [f32],
) {
    let chunks = x.len() / 8;
    for c in 0..chunks {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let xv = f32x8::load(token, arr);
        let mut buf = [0.0f32; 8];

        (xv * xv.rsqrt_approx()).store(&mut buf);
        approx[off..off + 8].copy_from_slice(&buf);

        (xv * xv.rsqrt()).store(&mut buf);
        mt_rsqrt[off..off + 8].copy_from_slice(&buf);

        (xv * rsqrt_stable!(f32x8, token, xv)).store(&mut buf);
        stable[off..off + 8].copy_from_slice(&buf);

        xv.sqrt().store(&mut buf);
        exact[off..off + 8].copy_from_slice(&buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use archmage::incant;

    fn fnv1a(vals: &[f32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in vals {
            for &b in &v.to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    fn max_rel(a: &[f32], ref_: &[f32]) -> f32 {
        a.iter()
            .zip(ref_)
            .map(|(&x, &r)| (x - r).abs() / r.abs().max(1.0))
            .fold(0.0f32, f32::max)
    }

    /// Inputs = `grad_sq` spanning [1, ~130050], integer-derived so the grid is
    /// byte-identical on every platform (no host transcendental in setup).
    fn grid() -> Vec<f32> {
        (0..256u32).map(|i| (i * 509 + 1) as f32).collect()
    }

    /// Measures the cross-platform divergence of each √ method AND asserts the
    /// deterministic ones reproduce the x86-blessed hash on every CI platform.
    ///
    /// The printed `RSQRTPROBE` line lets us read the per-platform hashes for the
    /// hardware methods (`approx`, `mt_rsqrt`) out of the CI logs — they differ
    /// per SIMD tier. `stable` and `exact` MUST match the committed hash on every
    /// platform; a mismatch fails CI (the determinism guard).
    #[test]
    fn magnitude_method_determinism_and_accuracy() {
        // Bit-identical hashes of `stable` / `exact`, blessed on x86-64; the
        // determinism guard for the portable methods (must hold on every arch).
        const STABLE_GOLDEN_HASH: u64 = 0x6b40_4d4c_5e62_e664;
        const EXACT_GOLDEN_HASH: u64 = 0x3aaf_412b_356d_ac68;

        let x = grid();
        let n = x.len();
        let (mut approx, mut mt, mut stable, mut exact) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        incant!(magnitude_methods(
            &x,
            &mut approx,
            &mut mt,
            &mut stable,
            &mut exact
        ));

        let (ha, hm, hs, he) = (fnv1a(&approx), fnv1a(&mt), fnv1a(&stable), fnv1a(&exact));
        println!(
            "RSQRTPROBE approx_hash={ha:016x} mt_rsqrt_hash={hm:016x} \
             stable_hash={hs:016x} exact_hash={he:016x}"
        );
        println!(
            "RSQRTPROBE max_rel_vs_exact: approx={:.3e} mt_rsqrt={:.3e} stable={:.3e}",
            max_rel(&approx, &exact),
            max_rel(&mt, &exact),
            max_rel(&stable, &exact),
        );

        // Accuracy floor: stable must be well under 1% (it's ~5e-4).
        assert!(max_rel(&stable, &exact) < 1.0e-2, "rsqrt_stable accuracy");

        // Determinism guards — assert once the goldens are blessed (non-zero).
        if STABLE_GOLDEN_HASH != 0 {
            assert_eq!(
                hs, STABLE_GOLDEN_HASH,
                "rsqrt_stable must be deterministic across platforms"
            );
            assert_eq!(
                he, EXACT_GOLDEN_HASH,
                "exact sqrt must be deterministic across platforms"
            );
        }
    }
}
