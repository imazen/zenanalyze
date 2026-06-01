# zenjpeg Profile-A dial — quality-targeting precision + byte efficiency (2026-06-01)

Applies IQA-grade statistics (the zenstats panel + percentile distributions) to
the **dial** question, not just the picker's argmin: *when we target a Profile-A
(zensim:a) score, how precisely do we land, and what does it cost in bytes — at
p50/p75/p90/p95 across the corpus?*

## Data + method

- Source: `picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet` — the full RD
  grid, **320 images × 36 cells × 29 q-values**, each row carrying `encoded_bytes`
  + `score_zensim` (Profile-A). zensim+bytes is complete; **ssim2/cvvdp are NOT
  joinable** (the picker corpus overlaps the cvvdp/ssim2 sidecars by 1 image — see
  gaps below).
- For a target grid T ∈ {30…90}, per image:
  - **ORACLE** = the least-byte `(cell, q)` with `zensim:a ≥ T` (the least-waste
    optimum), its achieved zensim (q-granularity precision = `achieved − T`).
  - Cross-image distributions (mean/std/p50/p75/p90/p95) of byte cost + hit error,
    plus a calibration RMSE.
- Scripts: `scripts/dial/{dial_quality,plot_dial}.py`. Plot (116 KB, >30 KB so not
  in git): `/mnt/v/zen/dial-quality-2026-06-01/dial_frontier.png`. Raw frontier:
  `oracle_frontier.pkl` (same dir).

## Findings

**1. Dial precision is strongly target-dependent — coarse at low-q, tight at high-q.**
Quality-hit RMSE (achieved − target) drops monotonically:

| target | 30 | 50 | 70 | 80 | 85 | 90 |
|---|---|---|---|---|---|---|
| hit-err mean | +4.46 | +2.29 | +1.32 | +0.84 | +0.60 | +0.39 |
| hit-err std | 5.63 | 2.66 | 1.33 | 0.81 | 0.53 | 0.35 |
| RMSE | 7.18 | 3.51 | 1.87 | 1.17 | 0.80 | 0.52 |

At low targets the q-grid is too coarse to land near T (the nearest reaching q
overshoots by ~4–7 points); at high quality the dial is tight (±0.5). This is the
single most actionable dial fact: **"give me zensim 40" is ±5; "give me zensim 88"
is ±0.5.** Densifying low-q q-steps (or sub-q interpolation in the loop) is the
lever if low-q precision matters.

**2. The byte cost of a target has an enormous, heavy-tailed cross-image spread.**
At T=70 the least-waste optimum is p50≈5.8 KB but p95≈108 KB (~18×); the spread
widens with quality (mean ≫ median throughout — a few high-detail images dominate).
Any single "bytes at quality X" number is meaningless without the percentile band;
the plot shows median/p75/p90/p95 on a log axis.

**3. Config selection is worth ~7–13% bytes** at most targets (naive 4:2:0-e1 vs the
oracle cell, on common-reach images) — the headroom the picker exists to capture.

## Top-K picker prune — FALSIFIED at the bake level (keep the 108-bake)

The CV (5-fold) confirmed pruning helps the **teacher** (every K<108 beats full-108
on overhead; top-30 teacher SROCC 0.996). But the shipped bake is a **distilled
student**, and there the result inverts: even with the full hyperparameter search
(`[64,64]`/`[128,128]`, multiple seeds/LRs), the top-30 student caps at **bytes-SROCC
~0.63 vs the 108-bake's 0.906**. The MLP student needs the redundant features as a
richer basis to reproduce the teacher's 36-cell surface from. The top-30's *lower
overhead* (3.0% vs 3.35%) is the flat-surface artifact the SROCC collapse warns about
(near-tied cells → near-random picks score low overhead by accident). **Lesson:
teacher-level feature importance does not transfer to the distilled student — judge
prunes on the bake's full panel (SROCC), not the teacher's importance or overhead
alone.** The 108-bake remains the ship.

## Gaps + next chunks

- **Cross-metric (ssim2 + cvvdp)** — the user wants the hit-error/byte spread also
  in ssim2 + cvvdp (does zensim:a=T map *consistently* to ssim2/cvvdp, or is there
  spread?). The picker corpus isn't in the existing sidecars, so this needs a
  **rescore** of the picker encodes: ssim2 is cheap (CPU, fast-ssim2); cvvdp is GPU.
  This is the key missing axis for "is our dial consistent across metrics."
- **Picker-vs-oracle ("OUR" error)** — the frontier above is the *achievable*
  optimum. The picker's own per-target byte-waste distribution needs a per-row
  decision dump from `evaluate_picker_bake` (picked cell, oracle cell, target) —
  a small Rust addition + rebuild — then the same percentile treatment.
- **XYB color-path axis** — separately justified (see `rd_explore_b_sweep_2026-04-15.md`):
  XYB wins content-dependently; a focused XYB grid sweep would add it as a picker axis.
