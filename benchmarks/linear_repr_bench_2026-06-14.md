# Linear-intermediate representation throughput bench (2026-06-14)

Decides the working representation for the opt-in linear-light path — **i12/i16
vs u16 vs f32** — by measured throughput, not precision argument (zenanalyze is
hot-path). Also corrects an earlier over-claim: u16 PQ/HLG **is** a valid HDR
carrier (the `.hdr.png` renditions are exactly that), so u16 is in contention,
not f32-only.

## Method

`examples/bench_linear_repr.rs`, zenbench interleaved/paired (kills thermal
bias). The choice hinges on two effects, benched separately:

1. **Math lane width.** i16/u16 pack 2× per SIMD register vs f32 — but
   variance/laplacian need the *squared* term, which widens i16→i64 and eats the
   advantage, while in-width ops (min/max for uniformity, threshold compares for
   edges) keep the full 2×. So:
   - `variance_reduce` — sum + sum-of-squares (widening reduction).
   - `minmax_reduce` — block min/max (in-width reduction).
2. **Linearize cost** — the opt-in's only added work over the gamma path:
   - `sdr_linearize` — u8→linear luma via 256-LUT (→f32, →i16) and via scalar
     compute (`srgb_to_linear`).
   - `hdr_linearize` — u16 PQ→linear luma via 64k-LUT (→f32, →i16), scalar
     compute (`pq_to_linear`), and `pq_direct` (no linearize — luma on PQ codes,
     the floor).

Sizes 1MP + 4MP. **Caveat:** plain autovectorized loops, not the hand-tuned
magetypes kernels tier1 ships — first-order numbers, enough to pick a direction
and flag whether a hand-tuned bake-off is warranted.

## Results

7950X, `--release`, 4 rounds/group (long groups → noisy, but the pattern is
identical at 1MP and 4MP). Throughput = luma elements/s. Raw zenbench JSON
(52 KB, block storage, not committed):
`/mnt/v/output/imazen-26-features/bench_linear_repr_2026-06-14.json` —
regenerate via the command at the bottom.

**Reductions (Gops/s, higher better):**

| group | f32 | i16 | u16 |
|---|--:|--:|--:|
| variance_reduce 1MP | 0.44 | 3.26 | 6.38 |
| variance_reduce 4MP | 0.43 | 3.23 | 7.32 |
| minmax_reduce 1MP | 5.27 | 45.4 | 20.3 |
| minmax_reduce 4MP | 5.12 | 37.8 | 18.9 |

**Linearize (Gops/s):**

| group | best→worst |
|---|---|
| sdr_linearize 4MP | lut_f32 **1.75** · lut_i16 1.23 · compute_f32 **0.030** |
| hdr_linearize 4MP | pq_direct **1.92** · pq_lut_f32 0.89 · pq_lut_i16 0.62 · pq_compute_f32 **0.024** |

## Interpretation

**1. Integer reductions crush scalar f32 — but read the caveat.** `variance` is
7–17× faster in i16/u16 than f32, `minmax` 4–9×. The cause is **FP
non-associativity**: the f32 sum-of-squares is a serial recurrence
(`sq = mul_add(x,x,sq)`) that LLVM may not reassociate, so it runs at ~1 element
per mul-add latency; integer adds *are* associative, so LLVM auto-vectorizes with
multiple accumulators. Tell: f32 `minmax` (5.1G, min/max vectorizes) is 12× f32
`variance` (0.43G) — same data, only the reduction differs. **So the headline 7×
overstates the gap vs a *hand-tuned* f32 kernel** — tier1's shipped `f32x8`
kernel uses explicit lane accumulators and would recover most of it. The honest
read: **integer gets good codegen *for free* (no fighting the autovectorizer) and
keeps the 2× lane-density edge; f32 only competes when hand-tuned.**

**2. Linearize must be a LUT, never per-pixel compute.** `compute_f32` (scalar
`srgb_to_linear` / `pq_to_linear`) is **30–58× slower** than the LUT — the EOTF
branch / `pow` is fatal in a hot loop. 256-LUT (SDR) is L1-resident and fast
(~1.2–1.75G). The **64k PQ LUT is cache-bound** (~0.6–0.9G, roughly half the
256-LUT) — it doesn't fit L1.

**3. HDR linear light is the genuinely expensive corner.** `pq_direct` (luma on
PQ codes, no linearize) is fastest (1.92G) — but that's *PQ/perceptual* domain,
**not linear light**, so it doesn't meet the goal. Real HDR linear light needs
the 64k LUT (cache-bound) or a **small interpolated LUT** (e.g. 1–4k entries +
lerp, fits L1) — slight precision loss, which the precision A/B says is fine.

## Verdict (representation for the opt-in linear path)

**Use i16 (i12) linear, not u16, not f32 — for reasons throughput *confirms* but
doesn't decide alone:**

- **Signedness (decisive):** laplacian `∇²L` and edge gradients are signed; i16
  carries them, u16 can't without an offset. u16's marginal `variance` edge
  (noisy 2×) doesn't buy back losing the signed ops.
- **Headroom:** i12-in-i16 keeps signed laplacian (`±4·4095 = ±16380`) inside
  16-bit lanes; full-range u16 overflows to i32 and halves lane density.
- **Throughput:** integer auto-vectorizes for free and keeps 2× density; the raw
  f32 gap is inflated by autovec, but i16 is *at worst* on par with a hand-tuned
  f32 kernel and needs less hand-tuning.
- **Precision:** i12 round-trips 8-bit sRGB (separate finding); precision barely
  moves the features (the A/B), so 12-bit linear is plenty.

**SDR linearize:** 256-entry sRGB→i12 LUT. **HDR:** small interpolated PQ→linear
LUT (avoid the 64k cache thrash); or, if a downstream A/B shows PQ-domain
analysis is acceptable, `pq_direct` is ~3× faster but is *not* linear light.

**Confirm before baking the production kernel:** a hand-tuned `magetypes`
`i16x16`-vs-`f32x8` bake-off on the real tier-1 kernel — these autovec numbers
say "integer is at least as good and free," not "f32 is 7× worse." That
bake-off + the downstream picker A/B gate the default flip.

## Addendum — hand-tuned variance bake-off (`bench_repr_handtuned`)

Tested whether a *multi-accumulator* f32 closes the gap (variance is the
contested widening case). 1MP+4MP, consistent:

| variant | Gops/s |
|---|--:|
| f32_serial | 0.45 |
| f32_multi16 (16 accumulators) | **0.63** |
| i16_i64_direct | 3.5 |
| i16_i32_flush | 3.3 |
| u16_i64_direct | 4.4 |

Sixteen independent f32 accumulators gained only **+40%** — they did NOT
vectorize. Likely the explicit `mul_add` intrinsic plus LLVM's refusal to
reassociate FP; integer adds are associative so the same plain-array pattern
auto-vectorizes to 3–4G. **Honest caveat:** this means f32 needs an *explicit*
`f32x8` SIMD kernel (tier1 ships those) to compete — my plain `f32_multi16` is a
floor, not f32's ceiling, so the literal 5.6× is not the hand-tuned-vs-hand-tuned
ratio.

**It does not change the decision, though:** i16/i12 wins regardless because
(1) **signedness is decisive** — laplacian/edge are signed, u16 can't and f32's
only path to parity is explicit SIMD; (2) i16 reaches 3.5G from *plain*
maintainable code while f32 demands careful explicit SIMD to not be reduction-
bound; (3) headroom + i12 round-trip + precision-irrelevance all hold. f32 has no
advantage that overrides signedness. **Verdict locked: i16/i12.** Build the
production kernel there; the explicit-`f32x8` ceiling is moot for the choice.

## Reproduce

```bash
cargo run --release --example bench_linear_repr
```
