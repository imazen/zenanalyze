# Per-TIER cost — finding & fixing the real hotspot

Per-FEATURE cost is meaningless: features share combined passes, so requesting
any one feature runs its whole tier. `per_feature_cost` confirmed it — every
feature's solo cost was a flat ~2200-2400µs (the shared RowStream+tier1 floor),
LOO marginals ~0/negative. The right granularity is per-TIER.

- **Harness:** `examples/per_tier_cost.rs` (zenbench, one gating feature per tier)
- **Host:** WSL2 Ryzen 9 7950X · **Date:** 2026-06-18 · all features on
- **Raw:** `/mnt/v/output/imazen-26-features/per_tier_cost_2026-06-18.json`

## Tier costs at 4MP (solo, RGB8 SDR input)

| tier | cost |
|---|--:|
| alpha | 2.3ms |
| tier2 (chroma sharpness) | 2.4ms |
| tier1 (variance/edge/chroma/uniformity) | 3.6ms |
| tier3 hist (entropy) | 3.6ms |
| palette | 4.5ms |
| tier3 dct (hf ratio) | 5.6ms |
| **depth (HDR/nits/gamut)** | **13.6ms** ← 54% of ALL |
| **ALL** | **25.4ms** |

The depth tier dominated — and it was on U8 SDR content, the case the removed
fast path used to short-circuit. The per-pixel sRGB EOTF (transcendental) plus a
per-pixel gamut matrix projection were the cost.

## Fix (bit-identical) → depth 1.7×, ALL 1.3×

Two exact, value-preserving optimizations in `scan_depth`:

1. **u8 EOTF LUT.** u8 sources have 256 distinct samples → precompute
   `lut[i] = eotf(tf, i/255)` once and look up per pixel instead of a
   transcendental. Bit-identical (`lut[byte] == eotf(tf, byte/255)`).
2. **Bt709 gamut skip.** Bt709 ⊆ sRGB(=Bt709) and ⊆ P3, so every pixel is in
   both gamuts — skip the per-pixel matrix projections (exact: coverage 1.0).

| 4MP | before | after | speedup |
|---|--:|--:|--:|
| depth tier | 13.6ms | 8.0ms | **1.70×** |
| ALL features | 25.4ms | 19.4ms | **1.31×** |

Verified: all 9 `tier_depth` tests pass unchanged, `consistency_matrix` still
108/108. Depth is now 41% of ALL (down from 54%).

## Remaining lever (NOT bit-identical, deferred)

`nits_to_bin` does a `log2` per pixel for the log-spaced percentile histogram —
the last per-pixel transcendental. A fast-log2 approximation or a linear-bin
restructure would shave it but shifts p99 by up to a bin (~3%), so it's a
value-changing tradeoff, left for a deliberate call rather than folded into this
bit-identical pass.
