# Native kernel format bench — f32 vs integer vs u8

**Question:** the content tiers deinterleave RGB8 → f32 and compute in `f32x8`
(garb only emits f32 planes; tier3 is the lone i32 fixed-point exception).
Does the f32 choice cost speed vs a u8/integer-native compute?

- **Command:** `cargo run --release --example bench_kernel_format`
- **Harness:** zenbench (interleaved, paired), `run-heavy` (nice/cgroup)
- **Commit:** `ee38c55` · **Host:** WSL2 Ryzen 9 7950X · **Date:** 2026-06-18
- **Raw:** `/mnt/v/output/imazen-26-features/bench_kernel_format_2026-06-18.json`
- **Caveat:** autovec multi-accumulator (K=16), NOT the production garb+f32x8
  hand-tuned path — this is *relative format cost*, not absolute tier1 MP/s.
  Variance group got thin sampling (4 rounds; i32@4MP CI crosses zero). The
  min/max group is rock-solid (39/22 calls × 4 rounds, huge effect size).

## Result 1 — reduction path (RGB8 → luma → variance): f32 WINS

| size | f32 | i32 fixed-point | i16 luma |
|---|--:|--:|--:|
| 1 MP | **1.09 Gops/s** | 953 M (+11–18%) | 1.03 G (+12–14%) |
| 4 MP | **1.08 Gops/s** | 949 M (−6%…+30%) | 997 M (+14–22%) |

f32 is the **fastest** by ~11–18%. The integer paths are *slower*, not faster.

**Why:** the squared-sum forces the difference. `luma²` ≤ 65 025; summed over
a megapixel it blows past `i32` (2³¹) almost immediately, so an *exact* integer
variance must accumulate in **`i64` → 4 lanes per 256-bit vector**. f32 keeps the
accumulate in **`f32x8` → 8 lanes** (24-bit mantissa absorbs the partial sums,
final reduce to f64). Double the lanes on the dominant op ⇒ f32 wins. The luma
dot-product itself (3 mul + 2 add) is a wash; FMA vs `imul`+shift doesn't move it.

## Result 2 — in-width op (luma min/max, uniformity extent): u8 DOMINATES

| size | u8 | i16 | f32 |
|---|--:|--:|--:|
| 1 MP | **74.7 Gops/s** | 36.2 G (2.0× slower) | 5.17 G (**14.5× slower**) |
| 4 MP | **65.7 Gops/s** | 34.7 G (1.9× slower) | 5.07 G (**13.9× slower**) |

For a pure in-width comparison (no widening), **u8 is ~14× faster than f32** and
2× faster than i16. Lane count, directly: `pminub`/`pmaxub` chew **16–32 lanes**
per vector; `pminsw` 8–16; `minps` only 4–8 (and carries NaN-handling baggage —
note the f32 drift flag). Promoting a min/max-style predicate to f32 would be a
~14× regression.

## Takeaway — the current split is correct, and it's principled

- **Reductions / anything that needs widening** (variance, sum-of-squares,
  edge energy, chroma stats): **f32** is the right native format. Integer isn't
  faster — overflow forces it to a 4-lane i64 accumulate. tier1/tier2 = f32. ✓
- **In-width comparisons** (min/max for uniformity extent, range, thresholds):
  keep them **u8**. f32 here is a 14× cliff.
- **tier3's i32 fixed-point** is for *binning/histogram/DCT* (exact integer
  indices, not a reduction or a min/max) — a third, separate justification.

Rule of thumb for new kernels: **does the op widen? → f32. Is it a same-width
compare/select? → u8.** The "native format" isn't one format — it's per-op, and
the existing tier split already lands on the fast side of both.
