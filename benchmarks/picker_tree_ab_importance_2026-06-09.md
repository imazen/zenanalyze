# Picker A/B + feature-importance ablation — zenjpeg (2026-06-09)

GBDT vs MLP vs RF on the **real** zenjpeg sweep, + GBDT permutation feature
importance (the input-side ablation, Track-A Phase 0). Instrument:
`zenpicker-train/src/bin/picker_tree_ab.rs --features tree-ab`.

## Setup

- **Data:** `/mnt/v/zen/zenpicker-train-parity/zq_pareto_2026-04-29_adapted.parquet`
  (3.5M rows → **9,733 (image, target_zq) picker rows**, 92 zenanalyze features
  + `zq_norm`, **12 categorical cells** = `{color, sub, trellis_on, sa}`).
- **Split:** grouped-by-image held-out (val_frac 0.2 → train 7,800 / val 1,933).
- **Run:** `run-heavy -- picker_tree_ab --input … --codec zenjpeg --val-frac 0.2 --seed 0`
  (139 s, peak-RSS 5.25 GiB, min-avail 39 GiB — box never stressed).
- **Caveat:** this is the **12-cell / 6-knob slice** (the other ~9 zenjpeg knobs
  are pinned at defaults). Importances are for predicting the pick *within* that
  slice; the **output/knob** ablation (which pinned knobs earn a head) is a
  separate sweep, not measurable from this data.

## A/B — GBDT beats the MLP

| model | argmin_acc | byte_overhead_mean |
|---|--:|--:|
| **GBDT** (gbdt 0.1.3) | **0.5649** | **0.0198** |
| MLP (128,128) | 0.5266 | 0.0277 (bytes-SROCC 0.9258) |
| RF (smartcore 0.5.0) | 0.2613 | 0.2571 |

GBDT wins on both accuracy (+3.8 pp) and byte overhead (1.98% vs 2.77%) — the
tabular-SOTA expectation holds. (RF-regression-per-cell does poorly here; a tree
*classifier* over the cell label would be the fairer RF framing.)

## Feature importance — `zq` dominates; content = a few clusters

**Per-feature permutation** (shuffle each input across val, drop in argmin acc):

```
+0.2162  zq_norm                 ← the target quality, by far
+0.0383  feat_cr_sharpness
+0.0372  feat_quant_survival_y
+0.0238  feat_cb_sharpness
+0.0176  feat_distinct_color_bins
+0.0155  feat_aq_map_p50
+0.0140  feat_laplacian_variance_p50
 ...     (50/93 features individually non-positive — redundant twins)
```

**Grouped by ρ≥0.9 redundancy cluster** (93 feats → 50 groups; the honest view —
shuffling a whole cluster, so correlated twins can't mask each other):

```
+0.2074  [zq_norm]
+0.0512  [cr_sharpness, dct_compressibility_uv, quant_survival_uv, noise_floor_uv_p50, +3]   ← CHROMA/UV detail
+0.0455  [edge_density, noise_floor_y, quant_survival_y, noise_floor_y_p25, +14]             ← LUMA detail/edge
+0.0217  [cb_sharpness]
+0.0155  [distinct_color_bins, palette_density]
+0.0135  [uniformity, aq_map_mean, aq_map_p50, noise_floor_y_p50, +1]
+0.0129  [dct_compressibility_y]
 ...
```

## Findings

1. **`zq_norm` carries ~0.21 of the ~0.28 above-random argmin signal.** The
   picker is first and foremost a **quality-conditioned chooser**; all 92 content
   features together add only ~0.07. This is the data-grounded version of "`zq`
   is an input" — it's the *dominant* input.
2. **Content signal = ~3 clusters**, mapping cleanly to the real zenjpeg
   decisions: a **chroma/UV-detail** cluster (~0.05 → the 444-vs-420 subsampling
   call), a **luma-detail/edge** cluster (~0.045 → trellis/quant), and
   color-count (`distinct_color_bins`/`palette` ~0.015 → colorspace). Everything
   else is sub-0.013.
3. **Massive redundancy — prunable to ~10–15 features.** 93 features collapse to
   50 ρ≥0.9 groups, and only ~6–8 groups carry >0.01 signal. The percentile
   families (`noise_floor_*_p*`, `aq_map_p*`, `laplacian_variance_p*`) are mostly
   twins. A pruned set of the cluster representatives (zq_norm + cr/cb_sharpness +
   one luma-detail rep + quant_survival_y + distinct_color_bins + aq_map_mean)
   should hold accuracy at a fraction of the extraction cost.
4. **Grouping changes the ranking** — per-feature understates the chroma cluster
   (top single chroma feature 0.038 vs the chroma *group* 0.051), confirming the
   "ablate by cluster, not feature" rule.

## Next

Input-side first-cut done. The bigger lever is the **output/knob** ablation:
sweep the ~9 pinned knobs (OAT, oracle byte-Δ at matched quality) to see which
earn a picker head — that needs fresh encodes (not this parquet). The pruned
feature set above should then be re-validated against that richer output space.

Reproduce: `run-heavy -- cargo build --release -p … --features tree-ab` then the
run command above. (gbdt 0.1.3 + smartcore 0.5.0, both pure-Rust, behind the
`tree-ab` feature; forust-ml 0.6.0-rc.1 is the production GBDT follow-up.)
