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
//! Raphson. The determinism comes from the *software seed* replacing the hardware
//! `rsqrt_approx` — NOT from avoiding `mul_add`. `mul_add` is the IEEE
//! correctly-rounded fused-multiply-add (magetypes lowers it to hardware FMA where
//! present, else a correctly-rounded `fmaf`), so it is itself deterministic; the
//! Newton steps here happen to use explicit `*`/`-` but `mul_add` would be equally
//! portable (and the `log2_lowp`/`log2_midp` probes confirm magetypes' `mul_add`
//! polynomials are byte-identical across arches). The only non-deterministic
//! primitives are the hardware *approximations* (`rsqrt_approx`, `rcp_approx`).
//! `rsqrt_stable!` keeps approximation speed (no hardware `sqrt` latency) while
//! removing the cross-platform divergence.
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
        // Two Newton steps. (`mul_add` would be equally deterministic — it is the
        // correctly-rounded fma — but explicit mul/sub keeps the bless stable.)
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

/// Deterministic horizontal sum of 8 SIMD lanes into f64 — widen each lane to f64
/// (exact) and sum in fixed lane order `0..8`. Unlike a hardware `reduce_add()`,
/// whose add-tree shape is arch-specific (hadd pairs / `vaddvq` / scalar), this is
/// the **same f64 add order on every backend**, so flushing a lane accumulator
/// through it makes cancellation-prone reductions (variance, the Pearson
/// chroma–luma covariances) bit-identical across SIMD tiers. It runs once per
/// flush (every `FLUSH` iters), not per element, so it adds no per-pixel cost.
///
/// (i686 still diverges here: its `f64` is x87 80-bit, a precision axis orthogonal
/// to lane order, which only `-Z build-std`-with-sse2 or accepting it can address —
/// see the i686-relaxed budgets in `versioning`.)
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
        let xv = f32x8::load(token, arr);
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
        f32x8::load(token, arr).rsqrt().store(&mut buf);
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
        f32x8::load(token, arr).rsqrt_approx().store(&mut buf);
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

/// `out[i] = log2_lowp(x[i])` (magetypes low-precision SIMD log2 — bit-ops +
/// `mul_add` polynomial, no hardware approximation). Lengths equal, multiple of 8.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
pub(crate) fn log2_lowp_into(token: Token, x: &[f32], out: &mut [f32]) {
    let chunks = x.len() / 8;
    for c in 0..chunks {
        let off = c * 8;
        let arr: &[f32; 8] = x[off..off + 8].try_into().unwrap();
        let mut buf = [0.0f32; 8];
        f32x8::load(token, arr).log2_lowp().store(&mut buf);
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
        f32x8::load(token, arr).log2_midp().store(&mut buf);
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
        f32x8::load(token, arr).ln_midp().store(&mut buf);
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

    /// Measures magetypes `log2_lowp` / `log2_midp` cross-platform determinism +
    /// accuracy. Both are bit-ops + a `mul_add` polynomial (no hardware approx), so
    /// they're byte-identical on every **FMA-capable** arch — CI-confirmed on x86-64,
    /// macOS-ARM, and Windows-ARM, asserted here against the x86-blessed hash. The
    /// one exception is **i686** (no hardware FMA → magetypes' `mul_add` uses the
    /// software `fmaf` fallback, which doesn't match hardware FMA bit-for-bit), so
    /// the CI cross job `--skip`s this assert there; `rsqrt_stable` (mul_add-free)
    /// stays identical even on i686. The `LOGPROBE` line surfaces the hashes in CI.
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
