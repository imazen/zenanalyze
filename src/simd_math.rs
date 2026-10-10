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
//! **software** bit-trick seed (integer ops on the float bits) refined by Newton-
//! Raphson, with explicit `*`/`-` only. The determinism comes from the *software
//! seed* replacing the hardware `rsqrt_approx`, and from not using `mul_add`:
//! magetypes' `mul_add` is a hardware FMA (one rounding) on x86 v3/v4 and NEON,
//! but `a * b + c` (two roundings) on the **scalar** backend and possibly on
//! wasm (the engine's madd). Only `mul_add_portable` rounds once everywhere (in
//! software where needed). The analyzer kernels route their multiply-adds
//! through [`TierMulAdd`] so the scalar tier matches the FMA tiers bit for bit.
//! The other non-deterministic primitives are the hardware *approximations*
//! (`rsqrt_approx`, `rcp_approx`). `rsqrt_stable!` keeps approximation speed (no
//! hardware `sqrt` latency) while removing the cross-platform divergence.
//!
//! It is a `macro_rules!` rather than a `fn` so it expands inside a `#[magetypes]`
//! body against that body's per-tier `f32x8` (which the macro re-types to
//! `f32x16`/`f32x4`/… per backend) — a plain generic `fn` can't ride that
//! re-typing. The magic constant routes through `f32x8::splat(from_bits(..))
//! .bitcast_to_i()` so the integer vector is the *matching-width* companion of
//! the float vector without ever naming `i32xN`.

use archmage::magetypes;

/// Deterministic reciprocal square root `≈ 1/√x` (bit-identical across x86 /
/// AVX-512 / NEON / i686 / wasm). `$x` is a positive SIMD float vector; `$vec` /
/// `$token` are kept for call-site compatibility but are now vestigial. Quake seed
/// `0x5f3759df - (bits(x) >> 1)` + 2 Newton steps `y·(1.5 − 0.5·x·y²)`, explicit
/// `*`/`-` only (no FMA) — better than the ~8–14-bit hardware `rsqrt_approx`, and
/// the *same* value on every architecture.
///
/// This now delegates to magetypes 0.9.27's portable reciprocal-sqrt:
/// `rsqrt_approx_portable` is the same Quake seed + 1 Newton step, and the
/// trailing `rsqrt_newton_portable` is the 2nd step — so it is **bit-for-bit
/// identical** to the body this macro used to inline by hand (same seed, same
/// `y·(1.5 − 0.5·x·y²)` non-FMA Newton, same order). The committed
/// `STABLE_GOLDEN_HASH` in the tests below is the determinism guard that proves it.
macro_rules! rsqrt_stable {
    ($vec:ident, $token:expr, $x:expr) => {{
        let x = $x;
        x.rsqrt_approx_portable().rsqrt_newton_portable(x)
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

/// `self * a + b` rounded the way the x86 v3/v4 and NEON tiers round it.
///
/// magetypes' `mul_add` is a hardware FMA (one rounding) on x86 v3/v4 and NEON,
/// but on the **scalar** backend it is `a * b + c` (two roundings) — so every
/// `mul_add` in a kernel rounded differently on the scalar tier, and the
/// cancellation-prone statistics (`spectral_slope_y`, `patch_fraction`, the
/// Pearson covariances) drifted up to ~11 % from the SIMD-tier values that the
/// golden and every stored feature vector were produced on. `tier_mul_add`
/// keeps `mul_add` (identical instruction, identical bits) on the SIMD tiers and
/// uses a correctly rounded software FMA (magetypes' `fmaf_soft`, what
/// `mul_add_portable` uses there) on the scalar tier, so scalar is
/// **bit-identical** to v3/v4/NEON. The scalar tier is slower for it; it only
/// runs on CPUs without AVX2+FMA / NEON and on 32-bit x86.
///
/// The **wasm128** tier deliberately keeps `mul_add` (the engine's madd, which
/// may round twice) — unchanged values and speed; making it single-rounding too
/// would cost ~8× on these ops (`mul_add_portable` fuses in software there).
pub(crate) trait TierMulAdd: Sized {
    /// `self * a + b` with the tier's rounding (see the trait docs).
    fn tier_mul_add(self, a: Self, b: Self) -> Self;
}

/// Tiers whose `mul_add` already rounds once (hardware FMA) or whose current
/// rounding is kept (wasm128): plain `mul_add`. (32-bit x86 dispatches to the
/// scalar tier only, so it has no native impl.)
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
macro_rules! tier_mul_add_native {
    ($($token:ty),* $(,)?) => {$(
        impl TierMulAdd for magetypes::simd::generic::f32x8<$token> {
            #[inline(always)]
            fn tier_mul_add(self, a: Self, b: Self) -> Self {
                self.mul_add(a, b)
            }
        }
    )*};
}
#[cfg(target_arch = "x86_64")]
tier_mul_add_native!(archmage::X64V3Token, archmage::X64V4Token);
#[cfg(target_arch = "aarch64")]
tier_mul_add_native!(archmage::NeonToken);
#[cfg(target_arch = "wasm32")]
tier_mul_add_native!(archmage::Wasm128Token);

/// Scalar tier: magetypes' correctly rounded software FMA (`fmaf_soft`: exact
/// f64 product, TwoSum, round-to-odd, narrow) per lane. Same bits as
/// `mul_add_portable`, which wraps the same routine, but without re-probing the
/// CPU tier on every lane (`nostd_math::fmaf` checks for an FMA instruction per
/// call) — the scalar tier only runs when that probe already said no.
impl TierMulAdd for magetypes::simd::generic::f32x8<archmage::ScalarToken> {
    #[inline(always)]
    fn tier_mul_add(self, a: Self, b: Self) -> Self {
        let (x, y, z) = (self.to_array(), a.to_array(), b.to_array());
        Self::from_array_t(
            archmage::ScalarToken,
            core::array::from_fn(|i| magetypes::nostd_math::fmaf_soft(x[i], y[i], z[i])),
        )
    }
}

/// Deterministic horizontal sum of 8 SIMD lanes into f64 — widen each lane to f64
/// (exact) and sum in fixed lane order `0..8`. Unlike a hardware `reduce_add()`,
/// whose add-tree shape is arch-specific (hadd pairs / `vaddvq` / scalar), this is
/// the **same f64 add order on every backend**, so flushing a lane accumulator
/// through it makes cancellation-prone reductions (variance, the Pearson
/// chroma–luma covariances) bit-identical across SIMD tiers. It runs once per
/// flush (every `FLUSH` iters), not per element, so it adds no per-pixel cost.
///
/// (i686 runs the scalar tier, which reduces through this same function; its
/// former divergence came from the scalar tier's double-rounded `mul_add`, now
/// fixed by [`TierMulAdd`] — not from this reduction.)
#[inline]
pub(crate) fn fixed_reduce8(lanes: [f32; 8]) -> f64 {
    let mut s = 0.0f64;
    let mut i = 0;
    while i < 8 {
        s += lanes[i] as f64;
        i += 1;
    }
    s
}

/// `out[i] = rsqrt_stable(x[i])` — the deterministic bit-hack + 2-Newton path.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn rsqrt_stable_into(token: Token, x: &[f32], out: &mut [f32]) {
    let n = x.len() / 8;
    for c in 0..n {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let xv = f32x8::load_t(token, arr);
        let mut buf = [0.0f32; 8];
        rsqrt_stable!(f32x8, token, xv).store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
    }
}

/// `out[i] = rsqrt(x[i])` — magetypes' hardware estimate + 1 Newton (≥16-bit; the
/// `rsqrt_approx_12` general/ARM path, the published stand-in for it).
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn rsqrt_nt_into(token: Token, x: &[f32], out: &mut [f32]) {
    let n = x.len() / 8;
    for c in 0..n {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let mut buf = [0.0f32; 8];
        f32x8::load_t(token, arr).rsqrt().store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
    }
}

/// `out[i] = rsqrt_approx(x[i])` — the raw hardware estimate (≥8-bit; x86 ~12-bit).
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn rsqrt_hw_into(token: Token, x: &[f32], out: &mut [f32]) {
    let n = x.len() / 8;
    for c in 0..n {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let mut buf = [0.0f32; 8];
        f32x8::load_t(token, arr).rsqrt_approx().store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
    }
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
        let xv = f32x8::load_t(token, arr);
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

/// `out[i] = log2_lowp(x[i])` (magetypes low-precision SIMD log2 — bit-ops +
/// `mul_add` polynomial, no hardware approximation). Lengths equal, multiple of 8.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn log2_lowp_into(token: Token, x: &[f32], out: &mut [f32]) {
    let chunks = x.len() / 8;
    for c in 0..chunks {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let mut buf = [0.0f32; 8];
        f32x8::load_t(token, arr).log2_lowp().store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
    }
}

/// `out[i] = log2_midp(x[i])` (magetypes mid-precision SIMD log2). Companion of
/// [`log2_lowp_into`] for accuracy/perf comparison.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn log2_midp_into(token: Token, x: &[f32], out: &mut [f32]) {
    let chunks = x.len() / 8;
    for c in 0..chunks {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let mut buf = [0.0f32; 8];
        f32x8::load_t(token, arr).log2_midp().store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
    }
}

/// `out[i] = ln_midp(x[i])` (magetypes mid-precision SIMD natural log). Deterministic
/// on FMA arches (bit-ops + `mul_add` poly); ~11× faster than scalar `f32::ln`.
/// Used for the spectral-slope `log|F|` accumulation. `x > 0`; lengths a multiple of 8.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn ln_midp_into(token: Token, x: &[f32], out: &mut [f32]) {
    let chunks = x.len() / 8;
    for c in 0..chunks {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let mut buf = [0.0f32; 8];
        f32x8::load_t(token, arr).ln_midp().store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
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

    /// `TierMulAdd` on the scalar tier is a correctly rounded fused
    /// multiply-add, lane by lane — the same bits `f32::mul_add` (and the x86
    /// v3/v4 / NEON hardware FMA) gives — including the cancellation cases
    /// where an unfused `a * b + c` differs.
    #[test]
    fn scalar_tier_mul_add_is_fused() {
        use super::TierMulAdd;
        use archmage::{ScalarToken, SimdToken};
        use magetypes::simd::generic::f32x8;
        let t = ScalarToken::summon().expect("scalar token");
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let m = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
            m * 2f32.powi(((state >> 8) % 20) as i32 - 10)
        };
        let mut unfused_differs = 0;
        for _ in 0..2_000 {
            let a: [f32; 8] = core::array::from_fn(|_| next());
            let b: [f32; 8] = core::array::from_fn(|_| next());
            // c = -a*b (rounded) makes the fused result the exact rounding error.
            let c: [f32; 8] =
                core::array::from_fn(|i| if i % 2 == 0 { -(a[i] * b[i]) } else { next() });
            let got = f32x8::from_array_t(t, a)
                .tier_mul_add(f32x8::from_array_t(t, b), f32x8::from_array_t(t, c))
                .to_array();
            for i in 0..8 {
                let want = a[i].mul_add(b[i], c[i]);
                assert_eq!(
                    got[i].to_bits(),
                    want.to_bits(),
                    "lane {i}: {a:?} {b:?} {c:?}"
                );
                unfused_differs += usize::from((a[i] * b[i] + c[i]).to_bits() != want.to_bits());
            }
        }
        assert!(
            unfused_differs > 1_000,
            "test inputs must exercise fused vs unfused"
        );
    }

    /// Measures magetypes `log2_lowp` / `log2_midp` cross-platform determinism +
    /// accuracy. Both are bit-ops + a `mul_add` polynomial (no hardware approx), so
    /// they're byte-identical on every **FMA-capable** arch — CI-confirmed on x86-64,
    /// macOS-ARM, and Windows-ARM, asserted here against the x86-blessed hash. The
    /// one exception is **i686**, which runs magetypes' scalar backend, where
    /// `mul_add` is the unfused `a * b + c` (two roundings) inside magetypes' own
    /// polynomial, so the CI cross job `--skip`s this assert there; `rsqrt_stable`
    /// (mul_add-free) stays identical even on i686. The `LOGPROBE` line surfaces the
    /// hashes in CI.
    #[test]
    fn log2_lowp_midp_determinism_and_accuracy() {
        const LOWP_GOLDEN_HASH: u64 = 0x67c2_346b_644a_0119;
        const MIDP_GOLDEN_HASH: u64 = 0xc4b7_1ece_d59d_3a08;

        let x = grid();
        let n = x.len();
        let (mut lowp, mut midp) = (vec![0.0; n], vec![0.0; n]);
        incant!(log2_lowp_into(&x, &mut lowp));
        incant!(log2_midp_into(&x, &mut midp));

        // f64 reference for accuracy (host, not used for the determinism hash).
        let refv: Vec<f32> = x.iter().map(|&v| (v as f64).log2() as f32).collect();
        let (hl, hm) = (fnv1a(&lowp), fnv1a(&midp));
        println!(
            "LOGPROBE lowp_hash={hl:016x} midp_hash={hm:016x} \
             max_rel_vs_exact: lowp={:.3e} midp={:.3e}",
            max_rel(&lowp, &refv),
            max_rel(&midp, &refv),
        );

        if LOWP_GOLDEN_HASH != 0 {
            assert_eq!(
                hl, LOWP_GOLDEN_HASH,
                "log2_lowp must be deterministic across platforms"
            );
            assert_eq!(
                hm, MIDP_GOLDEN_HASH,
                "log2_midp must be deterministic across platforms"
            );
        }
    }

    /// Relative perf of `log2_lowp` vs `log2_midp` vs scalar `f32::log2` (libm).
    /// Modest interleaved workload — always runs (prints `LOGPERF` ns/elem); the
    /// absolute numbers are rough under CI noise but the lowp↔midp ratio is the
    /// point: are they "far apart" or not. Interleaved + checksummed to defeat
    /// thermal bias and dead-code elimination.
    #[test]
    fn log2_lowp_vs_midp_perf() {
        use std::time::Instant;
        let x: Vec<f32> = (0..16_384u32).map(|i| (i + 1) as f32).collect();
        let n = x.len();
        let (mut lowp, mut midp) = (vec![0.0f32; n], vec![0.0f32; n]);
        const ITERS: u32 = 200;
        let (mut t_low, mut t_mid, mut t_scal) = (0u128, 0u128, 0u128);
        let mut sink = 0.0f32;
        for _ in 0..ITERS {
            // Interleave so each method sees the same thermal/turbo state.
            let a = Instant::now();
            incant!(log2_lowp_into(&x, &mut lowp));
            t_low += a.elapsed().as_nanos();
            sink += lowp[0];

            let b = Instant::now();
            incant!(log2_midp_into(&x, &mut midp));
            t_mid += b.elapsed().as_nanos();
            sink += midp[1];

            let c = Instant::now();
            let mut s = 0.0f32;
            for &v in &x {
                s += v.log2();
            }
            t_scal += c.elapsed().as_nanos();
            sink += s;
        }
        let per = |t: u128| t as f64 / (ITERS as f64 * n as f64);
        println!(
            "LOGPERF ns/elem: lowp={:.3} midp={:.3} scalar_log2={:.3}  (midp/lowp={:.2}x)  sink={sink}",
            per(t_low),
            per(t_mid),
            per(t_scal),
            per(t_mid) / per(t_low).max(1e-9),
        );
        assert!(sink.is_finite());
    }

    /// The realistic spectral-slope-binning win: per 8×8 DCT block, accumulate
    /// `log|F|` per radial bin the OLD way (scalar `f32::ln` per coefficient) vs the
    /// NEW way (batch all 64 through SIMD `ln_midp`, then scalar bin-scatter).
    /// Prints `SPECTRALPERF` ns/block + speedup. Interleaved + checksummed.
    #[test]
    fn spectral_ln_binning_perf() {
        use std::time::Instant;
        const NBLOCKS: usize = 512;
        const FLOOR: f32 = 1.0;
        // Synthetic DCT-like coefficient blocks, integer-derived (deterministic).
        let blocks: Vec<[f32; 64]> = (0..NBLOCKS)
            .map(|b| {
                let mut blk = [0.0f32; 64];
                for (i, v) in blk.iter_mut().enumerate() {
                    // u64 math so the LCG doesn't overflow 32-bit usize on i686.
                    *v = (((b * 64 + i) as u64 * 2_654_435 + 1) % 401) as f32 - 200.0;
                }
                blk
            })
            .collect();
        // Radial bin per flattened index (idx = v*8 + u).
        let mut binmap = [0usize; 64];
        for (idx, b) in binmap.iter_mut().enumerate() {
            let (u, v) = (idx % 8, idx / 8);
            let rr = u * u + v * v;
            *b = if rr < 4 {
                0
            } else if rr < 9 {
                1
            } else if rr < 21 {
                2
            } else if rr < 36 {
                3
            } else {
                4
            };
        }

        const ITERS: u32 = 200;
        let (mut t_old, mut t_simd, mut t_prod) = (0u128, 0u128, 0u128);
        let mut sink = 0.0f32;
        for _ in 0..ITERS {
            // OLD: scalar `f32::ln` per above-floor coefficient.
            let a = Instant::now();
            for blk in &blocks {
                let mut binsum = [0.0f32; 5];
                for (i, &c) in blk.iter().enumerate().skip(1) {
                    let mag = c.abs();
                    if mag >= FLOOR {
                        binsum[binmap[i]] += mag.ln();
                    }
                }
                sink += binsum[0] + binsum[4];
            }
            t_old += a.elapsed().as_nanos();

            // SIMD: batch all 64 through ln_midp, then scatter.
            let d = Instant::now();
            for blk in &blocks {
                let mut mags = [0.0f32; 64];
                for (m, &c) in mags.iter_mut().zip(blk.iter()) {
                    *m = c.abs().max(FLOOR);
                }
                let mut lns = [0.0f32; 64];
                incant!(ln_midp_into(&mags, &mut lns));
                let mut binsum = [0.0f32; 5];
                for (i, &c) in blk.iter().enumerate().skip(1) {
                    if c.abs() >= FLOOR {
                        binsum[binmap[i]] += lns[i];
                    }
                }
                sink += binsum[0] + binsum[4];
            }
            t_simd += d.elapsed().as_nanos();

            // PRODUCT: accumulate the f64 product per bin (cheap multiplies), then
            // ONE ln per bin — Σln(mag) = ln(Πmag). 5 lns/block instead of ~30.
            let e = Instant::now();
            for blk in &blocks {
                let mut binprod = [1.0f64; 5];
                for (i, &c) in blk.iter().enumerate().skip(1) {
                    let mag = c.abs();
                    if mag >= FLOOR {
                        binprod[binmap[i]] *= mag as f64;
                    }
                }
                let mut binsum = [0.0f32; 5];
                for b in 0..5 {
                    binsum[b] = binprod[b].ln() as f32;
                }
                sink += binsum[0] + binsum[4];
            }
            t_prod += e.elapsed().as_nanos();
        }
        let per = |t: u128| t as f64 / (ITERS as f64 * NBLOCKS as f64);
        println!(
            "SPECTRALPERF ns/block: scalar_ln={:.1} simd_ln_midp={:.1} product_then_ln={:.1}  \
             (simd {:.2}x, product {:.2}x)  sink={sink}",
            per(t_old),
            per(t_simd),
            per(t_prod),
            per(t_old) / per(t_simd).max(1e-9),
            per(t_old) / per(t_prod).max(1e-9),
        );
        assert!(sink.is_finite());
    }

    /// Relative perf of the rsqrt primitives (`RSQRTPERF` ns/elem) — `rsqrt_stable`
    /// (bit-hack + 2 Newton) vs `rsqrt` (hardware + 1 Newton, the rsqrt_approx_12
    /// stand-in) vs raw `rsqrt_approx` (hardware estimate). Interleaved.
    #[test]
    fn rsqrt_methods_perf() {
        use std::time::Instant;
        let x: Vec<f32> = (0..16_384u32).map(|i| (i + 1) as f32).collect();
        let n = x.len();
        let mut out = vec![0.0f32; n];
        const ITERS: u32 = 300;
        let (mut t_st, mut t_nt, mut t_hw) = (0u128, 0u128, 0u128);
        let mut sink = 0.0f32;
        for _ in 0..ITERS {
            let a = Instant::now();
            incant!(rsqrt_stable_into(&x, &mut out));
            t_st += a.elapsed().as_nanos();
            sink += out[0];
            let b = Instant::now();
            incant!(rsqrt_nt_into(&x, &mut out));
            t_nt += b.elapsed().as_nanos();
            sink += out[1];
            let c = Instant::now();
            incant!(rsqrt_hw_into(&x, &mut out));
            t_hw += c.elapsed().as_nanos();
            sink += out[2];
        }
        let per = |t: u128| t as f64 / (ITERS as f64 * n as f64);
        println!(
            "RSQRTPERF ns/elem: rsqrt_stable={:.3} rsqrt(hw+1nt)={:.3} rsqrt_approx(hw)={:.3}  \
             (stable/nt={:.2}x)  sink={sink}",
            per(t_st),
            per(t_nt),
            per(t_hw),
            per(t_st) / per(t_nt).max(1e-9),
        );
        assert!(sink.is_finite());
    }
}
