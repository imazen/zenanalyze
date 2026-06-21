# Cross-platform feature divergence — diagnosis (2026-06-20)

When the content-hash feature-versioning golden (`src/versioning.rs` +
`versioning_golden.tsv`) first ran across the full CI matrix, the
`golden_is_stable` value-stability tripwire fired on **9 of ~97 features**. This
is the model working as designed — it measured the analyzer's own cross-platform
reproducibility floor. This doc records what it found and why.

## What diverges, measured

Golden blessed on x86-64 AVX-512 (Ryzen 7950X). Max relative deviation of each
drifting feature across the CI matrix (ubuntu x86-64 AVX2, i686, aarch64 +
macOS-ARM + Windows-ARM NEON), from the run on commit `41c390f`:

| feature | max rel. dev | nature |
|---|--:|---|
| `chroma_luma_covariance_cb` / `_cr` | ~11 % (0.11 absolute) | Pearson, ill-conditioned near-gray |
| `edge_slope_stdev` | 5.3 % (and structural 0.0-vs-1.30) | rsqrt-approx + cancellation |
| `patch_fraction` | 0.29 % | threshold count |
| `spectral_slope_y` | 0.14 % | f64 reduction order |
| `dct_compressibility_y` | 0.067 % | f64 reduction order |
| `aq_map_std` | 0.041 % | f64 reduction order |
| `quant_survival_y` | 0.025 % | f64 reduction order |
| `variance` | 0.012 % | f64 reduction order |

Values are **bit-identical within a SIMD tier** (every AVX2 box gives 43.669, every
NEON box gives 45.846) and differ **across** tiers — i.e. deterministic per-tier,
not random noise. Two regimes:

- **Six features at 0.01–0.29 %** — plain f64 lane-wise reduction-order divergence
  (the SIMD inner loops sum sum/sum-of-squares lane-wise; a different lane count
  reduces in a different order). Harmless; covered by the 0.5 % global
  `REL_TOLERANCE`.
- **Three cancellation-prone outliers** — `edge_slope_stdev` and the two
  `chroma_luma_covariance_*`. These need looser per-feature budgets and have a
  deeper root cause (below).

The `i686` / NEON runs also produced **structural** `0.0`-vs-nonzero flips
(`edge_slope_stdev[7] = 0.0` vs `1.30`; `chroma_luma_covariance_cb[19] = 0.0` vs
`-0.266`): a per-tier-divergent value crossing a *decision threshold* (the edge
`grad_sq > EDGE_THRESH_SQ` count, or the covariance degeneracy guard) the other way.
No float tolerance can bridge `0.0` vs `1.30`.

## Root cause of the `edge_slope_stdev` outlier

`src/tier1.rs`, the FULL edge accumulation. Two distinct numerical paths for the
**same** quantity (the per-pixel luma-gradient magnitude `√(gx² + gy²)`):

- **SIMD chunk** (≈line 1506): `safe_grad_sq.rsqrt_approx()` then
  `grad_sq * inv_sqrt` — a **~12-bit approximate** reciprocal-sqrt. The comment
  assumes "~12-bit precision (well above the edge-slope stddev's noise floor)" —
  but `rsqrt_approx` lowers to a **different instruction with different precision
  per backend**: x86 `rsqrtps` (~12-bit), AVX-512 `vrsqrt14ps` (~14-bit), NEON
  `vrsqrte`+step (~8-bit). So the magnitude itself differs per architecture, most
  on NEON (coarsest estimate → the +5.6 % outlier).
- **Scalar tail** (≈line 1560): exact `grad_sq.sqrt()`. So the rightmost 1–7
  edge pixels per row (non-multiple-of-8 widths) use a *different, exact* sqrt than
  the SIMD body — an internal SIMD-vs-tail seam even on one machine.

Both feed `edge_slope_stdev = √(E[g²] − E[g]²)`, the cancellation-prone variance
formula. On near-uniform-edge content (regular lines) the true variance ≈ 0, so the
per-backend magnitude error and the catastrophic cancellation dominate → the
`0.0`-vs-`1.30` flip. On varied content (checkerboard, value ≈ 43) the rsqrt
per-backend error still shows as the 5.3 % spread.

`chroma_luma_covariance_*` uses exact `.sqrt()` (line 758) — its divergence is pure
Pearson cancellation `(nΣXY − ΣXΣY)/√…` on the near-gray gradient case, where
Cb/Cr ≈ 0 makes the numerator nearly cancel and the degeneracy guard flip per tier.

## Is it a correctness problem?

For **codec decisions**: no. The divergence occurs at each feature's *degenerate
floor* (near-uniform edges → `edge_slope_stdev` ≈ 0; near-gray → covariance
degenerate), far from any decision threshold (`edge_slope_stdev > 35 ⇒ screen`,
etc.), so it never flips a classification. The 5.3 % on the checkerboard sits at
43–46, both well above 35 → same verdict.

For **serialized-feature reuse across machines** (the point of the versioning
work): yes, partially. `edge_slope_stdev` is only reproducible to ~6 % across
architectures, so a model trained on x86 features and inferring on NEON-extracted
features sees up to a 6 % shift in this one feature. The per-feature version
tolerance now encodes exactly that (10 % budget) — an honest statement of the
feature's cross-platform reusability.

It is, strictly, a "two code paths produce different output for the same
operation" case (SIMD `rsqrt_approx` vs scalar exact `sqrt`, and per-backend
rsqrt).

## Recommended fix (not yet applied — shipped-feature + perf change)

Replace the SIMD `rsqrt_approx()` magnitude with an **exact SIMD `sqrt()`** to
match the scalar tail:

```rust
// was: rsqrt_approx + the max(1.0) NaN-guard (0 * rsqrt(0) = NaN)
let g_mag_v = grad_sq_v.sqrt() * mask_v;   // sqrt(0)=0, mask zeros non-edges
```

IEEE-754 mandates correctly-rounded f32 `sqrt`, so this is **bit-identical across
SSE/AVX2/AVX-512/NEON and the scalar tail**, collapsing the dominant ~6 % divergence
to the residual f32 `reduce_add` lane-order floor (~0.4 %; eliminable too by
widening the horizontal sum to f64).

Tradeoffs, why it is **deferred to a decision** rather than applied here:

1. **Perf** — `rsqrt_approx` was a deliberate "~3× faster than batch sqrt" choice.
   The absolute cost is one extra-latency sqrt per 8-px chunk on *stripe-sampled*
   Tier-1 rows (not the full image), so the expected impact on total analyze time
   is small — but it removes a deliberate optimization and must be **measured**
   (zenbench `tier1_bench`), not assumed.
2. **Re-bless** — it changes `edge_slope_stdev`'s value (≈0.1–0.5 % on x86, larger
   on NEON toward the exact value), so the golden must be re-blessed and its
   version hash bumps. Allowed under the 0.2.x feature-drift contract, but it
   churns the hash + any provenance blocks generated against it.
3. **Downstream models** — picker models trained on the current `edge_slope_stdev`
   re-validate on the new definition. Decision impact is expected negligible (the
   change is sub-threshold), but it is a behavior change.

The covariance outliers are pure cancellation, not rsqrt; a stable formulation
(or excluding near-degenerate cases from the corpus) is a separate, smaller item.

## How the versioning model handles it today

- **Hash** is computed from the committed golden *text* → platform-independent;
  used everywhere.
- **`REL_TOLERANCE` = 0.5 %** covers the six well-conditioned drifters.
- **`F32_TOLERANCE_OVERRIDES`** carries the three measured outliers
  (`edge_slope_stdev` 10 %, covariances 20 %).
- **`golden_is_stable`** (the live re-extraction tripwire) is **scoped to an
  x86-64 reference** (`golden-reference` CI job); the portable Test/Cross jobs
  `--skip` it (caller-controlled in `ci.yml`, no source `#[ignore]`). It still runs
  on the developer's x86-64 locally, so behavior regressions are caught at dev time.

## Update 2026-06-20 — fixed via `rsqrt_stable`, with a deterministic seed (not exact sqrt)

The dominant outlier (`edge_slope_stdev`) is **fixed**, and the fix is better than
the originally-recommended exact `sqrt`: rather than pay exact-sqrt latency, the
edge kernel now uses `simd_math::rsqrt_stable!` — a software Quake seed
(`0x5f3759df - (bits(x)>>1)`, integer ops) + 2 Newton steps in **explicit f32
mul/sub** (no `mul_add`, so no backend-dependent FMA). f32 mul/sub are
IEEE-correctly-rounded, so the result is bit-identical on every backend while
keeping approximation speed. The scalar tail uses the matching `rsqrt_stable_scalar`
(same f32 ops, one lane) so there is no SIMD-vs-tail seam.

A CI `rsqrt-probe` job measured all four √ methods over a fixed grid across
x86-64, macOS-ARM, and Windows-ARM (FNV hash of the output vector):

| method | x86-64 | ARM (NEON, both) | deterministic |
|---|---|---|---|
| `rsqrt_approx` (hardware) | `da07b12b4fb7391c` | `4fa90f0776894ab7` | no |
| magetypes `rsqrt` (Newton from hw seed) | `16e013cdd1251458` | `931bb8fa1f55dd34` | **no** |
| `rsqrt_stable` (ours) | `6b404d4c5e62e664` | `6b404d4c5e62e664` | **yes** |
| exact `sqrt` | `3aaf412b356dac68` | `3aaf412b356dac68` | yes |

The key result: magetypes' Newton-refined `rsqrt` is **not** cross-platform
deterministic either — Newton from a per-arch hardware seed leaves per-arch low
bits. Only a software seed (or exact sqrt) is deterministic. Accuracy vs exact on
x86: `rsqrt_approx` 1.96e-4, magetypes `rsqrt` 1.4e-7, `rsqrt_stable` 4.6e-6 — all
far below `edge_slope_stdev`'s decision granularity.

**Detour, tried and reverted (2329a9a → reverted):** the SIMD body was briefly
swapped from `rsqrt_stable!` to magetypes `.rsqrt()` for speed (intended as a
stand-in for a `rsqrt_approx_12` design). It was reverted because (a) per the
table above `.rsqrt()` is non-deterministic across arches — it reintroduced the
very `edge_slope_stdev` divergence this section fixed; (b) it left the scalar
tail on `rsqrt_stable_scalar`, creating a SIMD-vs-tail seam the body/tail were
specifically built to avoid; and (c) the measured gain was ~0.02 ns/px — below
the per-tier noise floor. The inverse-sqrt is a negligible slice of the edge
kernel, so determinism wins. Do not re-swap to `.rsqrt()`/`rsqrt_approx` here
without a *cross-arch-deterministic* approximation (and then update both the body
and the tail, and make `rsqrt-probe` assert the kernel's actual method).

The `edge_slope_stdev` golden was re-blessed (only that feature's row changed). Its
`F32_TOLERANCE_OVERRIDES` budget could now be tightened toward the f64-reduction
floor, but is left at 10 % for safety until a follow-up re-measures the post-fix
cross-platform spread (the f32 `reduce_add` lane-order residual, ~0.4 %, remains).
The two `chroma_luma_covariance_*` outliers are pure Pearson cancellation (exact
sqrt already), a separate item.

## Update 2026-06-21 — one platform blesses, EVERY platform checks (+ tolerances shrunk from CI)

The x86-only scoping was replaced. `golden_is_stable` now runs on the portable
(macOS / Windows-ARM) **and** cross (i686 / aarch64) matrices too; the x86-64
`golden-reference` job (`ZENANALYZE_GOLDEN_REFERENCE=1`) blesses the values, and
every other platform re-extracts live and checks within the per-feature tolerance —
so cross-platform divergence is now *bounded*, not merely measured on the reference.
The test prints each platform's spread (max rel-dev vs the x86 golden); those CI
logs sized the tolerances below.

**Measured spread on the actual CI matrix (2026-06-21), worst platform per feature:**

| feature | worst | platform | handling |
|---|--:|---|---|
| `chroma_luma_covariance_{cb,cr}` | 26.6 % | i686 | `XPLAT_STRUCTURAL_EXEMPT` (guard flips per tier); enforced on x86 ref at 15 % |
| `spectral_slope_y` | 7.23 % | **i686 only** | `I686_X87_EXEMPT`; tight (0.5 %) on every 64-bit platform |
| `patch_fraction` | 6.25 % | **i686 only** | `I686_X87_EXEMPT`; tight on every 64-bit platform |
| `dct_compressibility_y` | 0.067 % | i686 | global 0.5 % |
| `aq_map_std` / `quant_survival_y` / `variance` / … | ≤ 0.041 % | i686 | global 0.5 % |
| `edge_slope_stdev` | **≈ 0 %** | — | **override retired** (deterministic after the `rsqrt_stable` revert) |

**Tolerances now:** global `REL_TOLERANCE` 0.5 %; one override
(`chroma_luma_covariance_*` 15 %, was 20 %); `edge_slope_stdev`'s 10 % override
**retired**.

**Two things no float tolerance can bound** (exempt, not fixed):
- **covariances** — the Pearson degeneracy guard returns `0.0`-vs-nonzero across
  SIMD tiers on i686/NEON. Exempt on non-reference platforms.
- **`spectral_slope_y` / `patch_fraction` on i686 only** — x87 80-bit excess
  precision in *precompiled `std`/libm* (`.ln()`, a DCT-energy threshold count).
  `+sse2` was tried and reverted: byte-identical spread, because `cross` doesn't
  rebuild `std` (would need `-Z build-std`). These pass on every 64-bit platform
  incl. aarch64/NEON, so they're tight there; i686 stands in for WASM, whose float
  model is SSE2-like (no x87) — the x87 divergence is unrepresentative.

**Remaining follow-up** to make *all* tolerances xplat-enforced: fixed-order f64
reduction (so the covariance guard stops flipping per tier) and either `build-std`
+sse2 for i686 or accepting the x87 stand-in gap.
