# Transcendental / expensive-math audit + optimization (2026-06-20)

A full sweep of every `sqrt`/`rsqrt`/`log`/`exp`/`pow`/`cbrt` in `zenanalyze`, to
make each one **as fast as it can be** and **cross-platform deterministic** where
it feeds a serialized feature. Driven by the rsqrt determinism finding
(`feature-cross-platform-divergence-2026-06-20.md`).

## The determinism axis (the key constraint)

A SIMD math primitive is cross-platform deterministic **iff** it uses only
integer ops + IEEE-correctly-rounded f32 `*`/`-`/`sqrt` — i.e. **no hardware
approximation instruction and no `mul_add` (FMA)**. FMA fuses to one rounding and
is hardware-availability-dependent, so `a*b+c` and `fma(a,b,c)` differ and the
backend picks differently per arch. Consequences for the options:

| approach | speed | cross-platform deterministic? |
|---|---|---|
| exact `sqrt()` (hardware, IEEE) | slow | ✅ (correctly-rounded is unique) |
| hardware `rsqrt_approx` / `rsqrt` | fast | ❌ per-arch instruction/seed |
| magetypes `*_lowp` (log2/exp2/pow) | fast | ❌ **uses `mul_add`** (FMA) |
| **software poly, explicit `*`/`-`** | fast | ✅ |
| scalar `libm` (`f32::ln`, `powf`, `cbrt`) | slow | ❌ libm differs per platform |

So "use magetypes" gives speed but **not** determinism for the log/exp family
(they're built on `mul_add`). A deterministic *and* fast log/exp needs a software
polynomial evaluated with explicit mul/sub — which is what `../work/polyfit` is
for (fit offline → bake `const` coefficients → SIMD Horner, no FMA).

## sqrt / rsqrt — DONE

| site | was | now |
|---|---|---|
| `tier1.rs` edge magnitude (SIMD) | hardware `rsqrt_approx` (per-arch ~8–14 bit) | `simd_math::rsqrt_stable!` — software Quake seed + 2 Newton, explicit mul/sub. **Deterministic** (CI-measured identical on x86/ARM), ~4.6e-6 accurate. |
| `tier1.rs` edge magnitude (scalar tail) | exact `sqrt()` | `rsqrt_stable_scalar` — matches the SIMD body bit-for-bit (no seam). |
| `tier3.rs` spectral-slope radial bin | `√(u²+v²)` per coefficient + 4 branches | integer `rr` vs squared thresholds — **no sqrt**, provably identical bins, no re-bless. |
| ~13 cold `.sqrt()` (post-reduction feature scalars) | exact `sqrt()` | left — once per image, already deterministic, not hot. |

`rsqrt_stable!` lives in `src/simd_math.rs` with a CI `rsqrt-probe` job that
measures the cross-platform spread and asserts determinism.

## log / exp / pow — QUEUED (needs a determinism-vs-speed decision)

Hot sites that currently use scalar `libm` (slow + per-platform):

| site | quantity | loop | note |
|---|---|---|---|
| `tier3.rs:1264` `mag.ln()` | spectral-slope log\|F\| | per AC coefficient (~20–40/block) | hottest log; SIMD-vectorizable |
| `tier3.rs:643` `p*p.log2()` | entropy | per histogram bin (~50–200) | LUT-able (256-entry) but quantizes → re-bless |
| `tier_depth.rs:136` `(1+nits).log2()` | HDR histogram bin | per sampled pixel | LUT-able (~256–512) |
| `tier_depth.rs:157` `signal.powf(2.2)` | Gamma-2.2 EOTF | per sampled pixel | 256-entry LUT if input is u8 |

Already optimal: `tier3.rs:1832` AQ log uses magetypes `log10_lowp` (deliberate
fast SIMD path); all cold/post-reduction `log10`/`ln` (tier1:703, dimensions,
tier_depth:571) run once per image.

`xyb_color_loss.rs` `cbrt`/`powf` are **test-only** reference math — the runtime
path is LUT-based already, so no hot cbrt/powf to optimize there.

### The decision

These log/exp sites currently drift only 0.01–0.3 % across platforms (libm
variance), within the 0.5 % global `REL_TOLERANCE`. Three ways forward, per the
table above:

1. **Leave as libm** — simplest; non-deterministic but within tolerance today.
2. **magetypes `*_lowp`** — ~3–8× faster (SIMD), but `mul_add`-based → *adds*
   cross-platform divergence (the thing we just removed from the edge kernel), and
   ~1 % accuracy change → re-bless.
3. **polyfit → deterministic SIMD Horner** — fit a minimax log2/exp2 over the
   mantissa with `polyfit::CurveFit::new_weighted(|x| x.log2(), range, degree)`,
   bake `const` coefficients, evaluate with explicit mul/sub. Fast **and**
   deterministic. More work (per-function fit + range reduction + re-bless), and
   the right call only if these features need cross-machine reuse precision tighter
   than libm gives.

Recommendation: option 3 for `tier3:1264` (the hottest, and a feature serialized
for training reuse), option 1 for the cold ones, and option 3-or-LUT for the
`tier_depth` HDR path. Pending a steer on how much cross-platform precision these
specific features need.
