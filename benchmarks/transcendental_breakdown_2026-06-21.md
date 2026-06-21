# Per-pixel transcendental breakdown — 5f6ba93 (2026-06-21)

Host: Ryzen 9 7950X, x86-64 release, no `target-cpu=native`. Sources: random
RGB8 (`per_tier_cost`) + PQ-u16 (`hdr_depth_scan_perf`). Sampling budgets:
pixel_budget=500k (tier1/depth/alpha), hf_max_blocks=1024 (tier3).

## SDR pipeline, all features (per_tier_cost, 1 MP, ns/px)

| component | ns/px | transcendental content |
|---|--:|---|
| baseline (RowStream traversal, shared) | ~2.3 | none |
| tier1 (variance/edge/chroma SIMD stats) | +1.4 | edge `rsqrt_stable` ~0.06 ns/px + ~8 cold sqrt once/img |
| tier2 (Cb/Cr sharpness) | ~0 | none |
| tier3_hist (histogram + entropy) | +0.3 | entropy log2 once/img (~0 ns/px) |
| tier3_dct (DCT + spectral + AQ) | +1.6 | product-then-ln 5/block × 1024 (~0.01 ns/px), AQ log10_lowp SIMD |
| palette (colour counting) | +0.4 | none |
| alpha / depth (SDR fast path) | ~0 | none |
| **ALL** | **~8.1** | **< 0.1 ns/px total (~1%)** |

→ SDR transcendentals are effectively zero — the `rsqrt_stable` + product-then-ln
passes already took them there. The ~8 ns/px is SIMD mul/add/compare (reductions,
DCT, counting), not transcendentals.

## HDR depth tier, added on top (PQ-u16, 1 MP, ns/px)

scan_total = **7.05 ns/px** (→ HDR ALL ≈ 15 ns/px). Split:

| component | ns/px | |
|---|--:|---|
| EOTF (linear-srgb SIMD slice, optimized) | ~1.65 | 3 ch × 0.55 |
| **nits_to_bin `log2` (scalar libm)** | **~2.06** | **~29% of the scan — the remaining transcendental** |
| gamut mat3 projections + gather + binning + bit-probes | ~3.3 | non-transcendental |

(measured: nits_to_bin = 4.32 ns/sampled-px over ~500k sampled.)

## Where to focus next

1. **nits_to_bin `log2` (HDR)** — 2.06 ns/px, ~29% of the HDR depth scan. A
   two-pass SIMD `log2_midp` (store nits → SIMD-bin) cuts the log2 from 2.29 ns to
   ~0.2 ns, saving ~1.6 ns/px for HDR content. The last meaningful transcendental.
2. After that, transcendentals are exhausted. Remaining levers are
   **non-transcendental**: HDR gamut mat3 projections (~3.3 ns/px, 2 muls/pixel —
   vectorizable) and the SDR stats passes (tier1 1.4, tier3_dct 1.6 ns/px — already
   SIMD mul/add, marginal).

Size scaling: tier1/tier3 use fixed sampling budgets, so their per-pixel cost
*drops* at 4 MP (tier1 0.9 ns/px, ALL 3.2 ns/px) — transcendentals shrink further.
