# Transcendental / expensive-math audit + optimization (2026-06-20)

A full sweep of every `sqrt`/`rsqrt`/`log`/`exp`/`pow`/`cbrt` in `zenanalyze`, to
make each one **as fast as it can be** and **cross-platform deterministic** where
it feeds a serialized feature. Driven by the rsqrt determinism finding
(`feature-cross-platform-divergence-2026-06-20.md`).

## The determinism axis (the key constraint)

A SIMD math primitive is cross-platform deterministic **iff** it avoids hardware
*approximation* instructions (`rsqrt_approx`, `rcp_approx`) and uses only
integer ops + IEEE-correctly-rounded f32 operations. **`mul_add` is fine on
FMA-capable arches** — it is the IEEE correctly-rounded fused-multiply-add, so
hardware-FMA platforms (x86-64, AArch64/Apple/Windows-ARM) all yield the unique
correctly-rounded result. (My first pass wrongly flagged `mul_add` as the cause;
the magetypes `rsqrt` divergence is its hardware `rsqrt_approx` **seed**, not the
Newton `mul_add`.) **One exception, CI-measured: i686** — 32-bit x86 has no
hardware FMA, so magetypes' `mul_add` takes the software `fmaf` fallback, which
does *not* match the hardware-FMA result bit-for-bit (`log2_lowp` hashed
`8af3fc30…` on i686 vs `67c2346b…` on every FMA platform). So `mul_add`-based
primitives are deterministic across FMA arches but i686 is a reference-scope
outlier (already so for `golden_is_stable`). `rsqrt_stable` (mul_add-**free**,
explicit mul/sub) is identical even on i686 — the most portable. Consequences:

| approach | speed | accurate | cross-platform deterministic? |
|---|---|---|---|
| exact `sqrt()` (hardware, IEEE) | slow | exact | ✅ |
| hardware `rsqrt_approx` / `rsqrt` | fast | 8–14 bit | ❌ per-arch instruction/seed |
| **magetypes `*_lowp`** (bit-ops + `mul_add` poly) | fast | ~2e-6 | ✅ (no hardware approx) |
| **magetypes `*_midp`** | fast | ~1e-7 | ✅ |
| software bit-trick + Newton (`rsqrt_stable`) | fast | ~5e-6 | ✅ |
| scalar `libm` (`f32::ln`, `powf`, `cbrt`) | slow | exact-ish | ❌ libm differs per platform |

So `magetypes::*_lowp`/`*_midp` give speed **and** determinism for the log/exp
family — they're bit-ops + a `mul_add` polynomial, no hardware approximation. No
polyfit needed for log2/exp2/pow (magetypes covers them); polyfit remains the tool
for functions magetypes lacks (e.g. `cbrt`, currently test-only).

### Measured (x86-64 AVX-512, release, no `target-cpu=native`)

| `log2` method | ns/elem | accuracy vs exact | deterministic |
|---|--:|--:|---|
| `log2_lowp` | 0.129 | 1.9e-6 | ✅ (CI-asserted) |
| `log2_midp` | 0.203 | 1.1e-7 | ✅ (CI-asserted) |
| scalar `f32::log2` (libm) | 2.293 | exact | ❌ |

`midp` is only **1.58×** `lowp` but **~11×** faster than scalar libm, and near
f32-exact (1.1e-7). **Recommendation: `midp`** for the log/exp conversions —
near-exact + deterministic + 11× faster than libm; drop to `lowp` only on a
profiled hot path that needs the extra 1.58×. Probes + asserts live in
`src/simd_math.rs` (the `rsqrt-probe` CI job runs them across x86/ARM).

## sqrt / rsqrt — DONE

| site | was | now |
|---|---|---|
| `tier1.rs` edge magnitude (SIMD) | hardware `rsqrt_approx` (per-arch ~8–14 bit) | `simd_math::rsqrt_stable!` — software Quake seed + 2 Newton, explicit mul/sub. **Deterministic** (CI-measured identical on x86/ARM), ~4.6e-6 accurate. |
| `tier1.rs` edge magnitude (scalar tail) | exact `sqrt()` | `rsqrt_stable_scalar` — matches the SIMD body bit-for-bit (no seam). |
| `tier3.rs` spectral-slope radial bin | `√(u²+v²)` per coefficient + 4 branches | integer `rr` vs squared thresholds — **no sqrt**, provably identical bins, no re-bless. |
| ~13 cold `.sqrt()` (post-reduction feature scalars) | exact `sqrt()` | left — once per image, already deterministic, not hot. |

`rsqrt_stable!` lives in `src/simd_math.rs` with a CI `rsqrt-probe` job that
measures the cross-platform spread and asserts determinism.

## log / exp / pow — QUEUED (decided: magetypes `*_midp`)

Decision (user steer): convert the hot scalar-`libm` log/exp sites to magetypes
SIMD `*_midp` — near-exact (1.1e-7), deterministic, and ~11× faster than libm
(per the measurement above). Each changes the feature value (libm → midp) → a
re-bless; allowed under the 0.2.x feature-drift contract.

Hot sites that currently use scalar `libm`:

| site | quantity | loop | plan |
|---|---|---|---|
| `tier3.rs:1264` `mag.ln()` | spectral-slope log\|F\| | per AC coefficient (~20–40/block) | SIMD `ln_midp` over a block's coeffs, then scalar bin-scatter |
| `tier3.rs:643` `p*p.log2()` | entropy | per bin (~256, once/image) | low value (once/image); `log2_midp` over the bin array if convenient |
| `tier_depth.rs:136` `(1+nits).log2()` | HDR histogram bin | per sampled pixel | batch the depth scan → `log2_midp` |
| `tier_depth.rs:157` `signal.powf(2.2)` | Gamma-2.2 EOTF | per sampled pixel | `pow_midp` (or a 256-LUT if input is u8) |

Already optimal: `tier3.rs:1832` AQ log uses magetypes `log10_lowp`; all
cold/post-reduction `log10`/`ln` (tier1:703, dimensions, tier_depth:571) run once
per image — leave as libm.

`xyb_color_loss.rs` `cbrt`/`powf` are **test-only** reference math — the runtime
path is LUT-based already, so no hot cbrt/powf to optimize there. (If a hot `cbrt`
ever appears, magetypes lacks it → fit one with `../work/polyfit`.)
