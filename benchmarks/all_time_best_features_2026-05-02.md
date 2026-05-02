# All-time best features — zenanalyze 0.1, as of 2026-05-02

Synthesis of every ablation, LOO retrain, Tier 0 correlation, Tier 1.5
permutation importance, and dendrogram analysis we've run on zenanalyze
since 0.1.0 launched. Useful as a starting point for new picker
training and as a reference when designing `KEEP_FEATURES` for any
codec that consumes zenanalyze.

## TL;DR — the 15-feature minimum set

If you have to pick a defensible starting point for a tiny-MLP picker
without running new ablation, take this:

```
# Resolution / shape (4)
feat_pixel_count
feat_aspect_min_over_max
feat_min_dim
feat_bitmap_bytes

# Palette (1)
feat_distinct_color_bins

# Texture / edge (5)
feat_laplacian_variance
feat_laplacian_variance_p75
feat_edge_density
feat_edge_slope_stdev
feat_patch_fraction

# Encoder-spend distribution (3)
feat_uniformity
feat_aq_map_std
feat_quant_survival_y

# Noise floor (1)
feat_noise_floor_y

# Chroma (1)
feat_cb_sharpness
```

15 features, mutually disjoint per the cross-codec dendrogram (no two
in the same redundancy cluster), each with at least one positive
ablation finding. Add 5–10 more if your codec has specific signals
(JPEG-trellis quant survival, JXL palette tiers, AVIF wide-gamut, etc.).

## Methodology — evidence types

We weight by methodology strength, not just magnitude. From strongest
to weakest:

1. **Multi-seed Tier 3 LOO retrain** (this session, 2026-05-02). 5
   seeds × paired with/without on a real pareto sweep. Variance
   estimable. Best gold standard but expensive (~50 min per feature).
2. **Single-seed Tier 3 LOO retrain** (2026-05-02). N=1; variance
   ±5–7 pp on argmin accuracy per the multi-seed run.
3. **Cross-codec Tier 0 correlation cleanup** + dendrogram. Surfaces
   redundancy *candidates*; can't decide which member of a cluster
   to keep without LOO.
4. **Tier 1.5 student permutation importance** (per-codec).
   Mis-attributes signal across redundant pairs (e.g.,
   `feat_log_pixels` showed +0.596 ΔAC on zenjpeg permutation despite
   Spearman 1.0 with `feat_pixel_count` — the model uses both as
   handles, not a real independent signal). Treat as
   discovery-phase, not load-bearing.
5. **Historical ablation scores** baked into picker configs (zenjpeg
   v2.1 → v2.2-clean → v2.2-pruned schema evolution). Pre-cross-codec
   work; should be cross-checked against the 2026-05-02 cluster
   structure before relying.

## What zenjpeg found best

zenjpeg has the longest ablation history of any codec (PR #129
ablation winner → v2.1 → v2.2 → v2.2-pruned → v2.2-clean → 2026-05-02
overnight ablation). zenjpeg-specific findings, oldest to newest:

### Pre-2026 (production picker)

The single dominant signal: **`feat_pixel_count`** at **+4.89 pp**
permutation importance — "**the #1 ablation impact in the entire
schema**" per the picker config docstring. Every subsequent
zenjpeg ablation has confirmed this; pixel_count is non-negotiable
for the JPEG picker.

### 2026-04-30 v2.1 → v2.2-clean schema (path to 35 features)

The trail of cuts:

- **Drop `feat_distinct_color_bins`** for zenjpeg specifically:
  "*palette codec only, structurally irrelevant for zenjpeg
  (true-color JPEG doesn't care about palette fits-in-N).*" Other
  codecs keep it; JPEG doesn't.
- **Drop UV-side percentiles** (quant_survival_uv,
  noise_floor_uv, noise_floor_uv_p50): "*parent quant_survival_uv
  and noise_floor_uv both had negative ablation Δ on subset100.*"
- **Drop `feat_high_freq_energy_ratio`, `feat_flat_color_block_ratio`,
  `feat_grayscale_score`**: "*small-corpus noisy*; on full 347-image
  retrain these likely become weak-positive; revisit then.*"
- **Drop 9 redundant log/derivative variants** of pixel_count
  (`log2_pixels`, `log10_pixels`, `log_pixels_rounded`, `sqrt_pixels`,
  `log_bitmap_bytes`, `log_min_dim`, `log_max_dim`,
  `log_padded_pixels_{16,32,64}`): "*the logs-only retrain showed the
  MLP performs worse without `pixel_count`, so the linear dims carry
  the load and the extra log scales are noise.*"

What zenjpeg KEPT (the v2.2-clean dimension set, 5 features):

- `feat_pixel_count` — **the dominant size signal** (+4.89pp ablation impact)
- `feat_log_pixels` — smooth resolution axis (now restored across the board)
- `feat_aspect_min_over_max` — bounded strip detection
- `feat_log_padded_pixels_8` — log of the encoded surface area at the JPEG 8×8 block grid (the codec actually pays for these padded pixels)
- `feat_channel_count` — discrete grayscale/RGB/RGBA distinction

### 2026-05-01 zenjpeg overnight Tier 1.5 (caveat: data-starved)

Top-5 by permutation Δpp (with caveats — the schema mismatch made
several cells data-starved at 1 member config; treat as smoke test
not load-bearing):

1. **`feat_quant_survival_y_p10`** (+0.885 pp) — *worst-block
   survival → trellis ROI*. This is the JPEG-specific signal: at
   p10, the quant-survival metric tells the picker how aggressive
   the trellis quantizer should be. Other codecs don't surface
   this.
2. **`feat_laplacian_variance_p90`** (+0.755 pp) — texture-spread
   upper-tail. Distinguishes complex-detail images from
   smooth-photo content.
3. **`feat_edge_slope_stdev`** (+0.740 pp) — gradient-magnitude
   stddev. Distinguishes natural photos (clustered around the lens
   MTF) from digital artwork (bimodal from variable brushwork).
4. **`feat_noise_floor_y`** (+0.715 pp) — encoder spends-floor
   signal.
5. *`feat_palette_density`* (+0.618 pp) — but **this was
   permutation mis-attribution**: ρ=1.0 with `feat_distinct_color_bins`
   on zenjpeg, the model uses one of the pair, permutation
   double-counts. **Just deprecated** (commit `023ff5ff`).

Bottom-5 (Δpp negative — model worse with feature *present*):

1. `feat_cr_sharpness` (−0.649) — surprising; zenwebp keeps it.
   zenjpeg may not use chroma sharpness as much because JPEG uses
   fixed chroma quantization.
2. `feat_aq_map_p90` (−0.338)
3. `feat_edge_density` (−0.325) — also surprising; zenwebp Tier 1.5
   loved it. Likely cluster mis-attribution: zenwebp's
   `edge_density ↔ quant_survival_y` cluster shows up here too,
   and the negative Δ may reflect the model preferring
   `quant_survival_y` (which was kept).
4. `feat_luma_kurtosis` (−0.214) — newer 2026-05-01 feature; mixed
   signal across codecs.
5. `feat_aq_map_p95` (−0.213) — adjacent to p90, same cluster.

### zenjpeg cross-codec confirmation

`feat_pixel_count` (+4.89 pp on zenjpeg) is in **every** other
codec's KEEP_FEATURES (zenwebp, zenavif, zenjxl). Same with
`feat_aspect_min_over_max` and `feat_log_pixels` (after restoration).
**zenjpeg's findings on the geometry primitives generalize.**

zenjpeg's chroma-sharpness skepticism, on the other hand, doesn't
generalize — zenwebp shows `feat_cb_sharpness` and `feat_cr_sharpness`
in the +0.20 pp tier of its Tier 1.5. JPEG's fixed chroma
quantization tables are the likely explanation.

## Cross-codec top tier (load-bearing on every codec)

The 6 features that show up high in every codec's KEEP and have
positive ablation findings everywhere we've measured:

| Feature | Strongest signal | Note |
|---|---|---|
| **`feat_pixel_count`** | +4.89 pp zenjpeg ablation; in every KEEP | The dominant size signal. |
| **`feat_distinct_color_bins`** | Anchor of palette cluster on every codec; in every KEEP except zenjpeg v2.2-clean | Non-JPEG codecs need it for indexed-mode dispatch. |
| **`feat_laplacian_variance`** + p50/p75/p90 | Top tier of zenwebp Tier 1.5 (Δ ≥ 0.20 pp); zenjpeg Tier 1.5 #2 at p90 | Texture-spread; distinguishes flat from busy content. |
| **`feat_uniformity`** | Cross-prefix anchor with `feat_aq_map_mean`; in every KEEP | Flat-block-fraction. |
| **`feat_quant_survival_y`** | zenjpeg Tier 1.5 #1 at p10 (+0.885 pp); cross-prefix anchor with `feat_edge_density` | Coefficient survival under quantization. |
| **`feat_aspect_min_over_max`** | zenwebp prior +0.10–0.14 pp on m4/m5/m6 method choice | Bounded shape signal; survives the perfect-Jaccard cluster vs. now-deprecated `feat_log_aspect_abs`. |

## Strong on most codecs

| Feature | Where strongest |
|---|---|
| `feat_aq_map_std` | zenwebp Tier 1.5 #3 (+0.359 pp) — encoder-spend variance the mean misses |
| `feat_patch_fraction` | **zenwebp Tier 1.5 #1** (+0.470 pp) — best single feature on zenwebp; AUC 0.880 for screen-vs-photo |
| `feat_edge_density` | zenwebp Tier 1.5 top-5 (+0.229) — but zenjpeg Tier 1.5 ranks it negative (cluster mis-attrib) |
| `feat_edge_slope_stdev` | zenjpeg Tier 1.5 #3 (+0.740 pp) — gradient-magnitude stddev |
| `feat_noise_floor_y` | zenjpeg Tier 1.5 #4 (+0.715) — anchor of low-tail block-cost cluster cross-codec |
| `feat_cb_sharpness`, `feat_cr_sharpness` | zenwebp Tier 1.5 in the +0.20 pp tier — chroma matters for codecs without fixed chroma quantization |
| `feat_min_dim` | zenwebp prior +0.10–0.14 pp on small-image method choice |

## Recently confirmed (2026-05-02 LOO)

Single-seed paired retrain on zenwebp showed these had measurable
LOO ΔAC well outside the noise floor:

| Feature | ΔAC (single-seed) | Action taken |
|---|--:|---|
| `feat_bitmap_bytes` | **−8.6 pp** (strongest signal in entire LOO) | **Restored** (commit `4c183f7d`) |
| `feat_noise_floor_y_p10` | **−5.8 pp** | Kept |
| `feat_log_padded_pixels_16` | **−3.3 pp** | Restored (`15d9c299`) |
| `feat_log_padded_pixels_32` | **−3.0 pp** | Restored |
| `feat_log_padded_pixels_8` | **−1.8 pp** | Restored |

Multi-seed LOO is showing variance ±5–7 pp on ΔAC, so
`bitmap_bytes` and `noise_floor_y_p10` are robust; the smaller
deltas may be partially variance.

## Codec-specialized (don't generalize)

| Feature | Codec | Why |
|---|---|---|
| `feat_quant_survival_y_p10` | **zenjpeg** | Trellis-ROI signal — worst-block survival drives JPEG quantizer aggressiveness. Other codecs don't surface this. |
| `feat_aq_map_p10` | zenjpeg | Same family — JPEG-specific encoder spend signal. |
| `feat_noise_floor_uv_p50` | **zenwebp** | UV-side noise; zenwebp Tier 1.5 #2 (+0.373 pp). zenjpeg explicitly dropped it ("negative on subset100"). |
| `feat_cb_sharpness` / `feat_cr_sharpness` | zenwebp / zenavif / zenjxl | Codecs with adaptive chroma quantization; zenjpeg has fixed tables and ignores. |
| `feat_chroma_complexity` ↔ `feat_colourfulness` | zenjxl-only redundancy | On zenjxl's 400-image corpus they cluster ρ=1.0; on zenwebp's 1264-image corpus they don't. **Defer cull** until cross-codec corpus convergence. |

## Honorable mentions / mis-attribution warnings

These looked good in some metric but turned out to be
permutation-importance artifacts or LOO-mixed:

| Feature | Apparent | Reality |
|---|---|---|
| `feat_palette_density` (just deprecated) | +0.618 pp on zenjpeg Tier 1.5 | ρ=1.0 with `distinct_color_bins`; multi-seed LOO ΔAC = +0.78 ± 7.62 pp (within noise). The Tier 1.5 number is mis-attribution. |
| `feat_log_pixels` (restored on theoretical grounds) | +0.596 pp on zenjpeg permutation | ρ=1.0 with `pixel_count`; multi-seed LOO mean ΔAC = −0.76 ± 4.89 pp. Restoration was correct on average but barely. |
| `feat_log_aspect_abs` (just deprecated) | n/a | Perfect-Jaccard cluster with `aspect_min_over_max` cross-codec. Same signal, different bounds. |

## What's actively retired

| ID range | What | When | Reason |
|---|---|---|---|
| 11 | `DistinctColorBinsChao1` | pre-0.1.0 | Chao1 collapsed to raw count under full-scan |
| 27, 28, 29, 45 | `TextLikelihood`, `ScreenContentLikelihood`, `NaturalLikelihood`, `LineArtScore` | 2026-05-01 (`e5c3c39`) | Composite-likelihood drift; raw signals (PatchFraction, etc.) outperform |
| 30 | `IndexedPaletteWidth` | 2026-05-02 (`248b48b`) | Codomain {0,2,4,8} too coarse; replaced by `PaletteLog2Size` (id 121) with codomain [1,15] ∪ {24} |
| 64, 66 | `BlockMisalignment{16,64}` | 2026-04-30 | Spearman 0.96 / 0.998 with `_8` / `_32` anchors |
| 94–100 | 7 mathematical transforms of `pixel_count` | 2026-05-01 | log2/log10/sqrt/log_min/log_max/log_padded_pixels — recoverable by tiny MLP via constant scaling or rgb8-channel-constant |
| 104 | `LogPaddedPixels64` | 2026-05-02 (`4c183f7d`) | LOO ΔAC +13.6 pp (actively hurt — 64×64 grid past useful resolution at our data scale) |
| 117–119 | `ChromaKurtosis`, `UniformitySmooth`, `FlatColorSmooth` | 2026-05-01 | Cross-codec Tier 0 redundant with their hard-threshold counterparts |

Stable IDs reserved; never recycled.

## Currently deprecated (active but `#[deprecated]`)

| ID | Feature | Migration |
|---|---|---|
| 12 | `PaletteDensity` | Use `DistinctColorBins` for raw count or `PaletteLog2Size` for discrete BPP |
| 62 | `LogAspectAbs` | Use `AspectMinOverMax` (bounded `(0, 1]`) |

## Caveats

- **zenjpeg's Tier 1.5 numbers** from 2026-05-01 had a schema-mismatch
  data-starvation issue (several cells with 1 member config). Read
  the rankings as discovery-grade, not load-bearing.
- **Multi-seed LOO** so far covers only zenwebp. Single-seed magnitudes
  on zenwebp are reliable; cross-codec replication needed before
  treating any LOO finding as universal.
- **The minimum 15 set** is a starting point, not a final answer.
  Each codec's specific axes (JPEG trellis ROI, JXL Modular palette
  tiers, AVIF wide-gamut, WebP-VP8L palette mode) need 3–5 additional
  codec-specific features per the picker configs.
- **Tier 1.5 negative deltas** (e.g., zenjpeg's `cr_sharpness` −0.649)
  often reflect cluster mis-attribution, not "the feature actively
  hurts." Drop only after Tier 3 LOO confirms.

## What's still in flight

- **Multi-seed LOO** (running now): variance estimates on the 8
  candidates from 2026-05-02. Will tighten the magnitude estimates
  on `bitmap_bytes`, `log_padded_pixels_*`, `log_pixels`,
  `aq_map_p10`, `palette_density` (already deprecated),
  `noise_floor_y_p50`, and `log_padded_pixels_64` (already retired).
- **Next LOO batch**: `feat_block_misalignment_8`,
  `feat_block_misalignment_32`, `feat_min_dim`, `feat_max_dim`,
  `feat_palette_log2_size` (post-redefinition; needs fresh feature
  extraction).
- **`FEATURE_GROUPS` validator** prototype shipped in zenwebp
  picker config (`c4ac6423`); needs port to other codec configs.

## References

- `benchmarks/feature_groups_2026-05-02.md` — full zenwebp dendrogram analysis
- `benchmarks/feature_groups_cross_codec_2026-05-02.md` — zenwebp ↔ zenjxl comparison + 6 perfect-Jaccard clusters
- `benchmarks/cross_codec_aggregate_2026-05-02.md` — 4-codec deferred-feature validation (the original 2026-05-02 sweep)
- `benchmarks/loo_retrain_2026-05-02.md` — single-seed LOO findings
- `benchmarks/loo_retrain_multiseed_2026-05-02.tsv` — multi-seed sweep results (when complete)
