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

## FINAL resolution — restore the SDR fast path (supersedes the LUT)

The LUT/gamut-skip above optimized the depth *walk*, but the walk should never
run for SDR in the first place. The depth tier had a U8-SDR fast path that
short-circuited to the canonical SDR profile (no walk); an earlier consistency
change had removed it (which is why depth was walking + showing up as the 13.6ms
hotspot). The right fix was to **restore the fast path and extend it to all
non-HDR transfers** (u8/u16/f32) — every SDR source short-circuits, only true
HDR (PQ/HLG) walks (`0595292`).

| 4MP depth tier | cost |
|---|--:|
| fast path removed (content walk) | 8.0ms |
| **fast path restored + extended** | **2.3ms** (in line with the other tiers) |

| 4MP ALL | cost |
|---|--:|
| fast path removed | 19.4ms |
| **fast path restored** | **13.1ms** (−32%) |

So the depth tier is no longer a hotspot — it's back to ~free for SDR. The
LUT + Bt709 gamut-skip now apply only to the HDR-only walk. The per-pixel
`log2` in `nits_to_bin` is likewise an HDR-walk-only concern now (deferred,
value-changing).
