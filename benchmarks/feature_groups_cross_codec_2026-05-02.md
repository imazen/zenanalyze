# Cross-codec feature-group comparison — zenwebp vs zenjxl, 2026-05-02

Same hierarchical-clustering analysis (Spearman, average linkage, cuts
at ρ ∈ {0.99, 0.95, 0.90, 0.85}) run on both codecs:

- **zenwebp**: 1 264 image-instance rows × 100 active features
  (`zenwebp_pareto_features_2026-05-01_combined_filled.tsv`)
- **zenjxl**: 400 (image, size) rows × 87 active features (lossy and
  lossless TSVs are essentially identical at the cluster level since
  features are computed per source image, not per encode)
  (`/mnt/v/output/jxl-encoder/picker-oracle-2026-04-30/lossy_pareto_features_2026-05-01_v2_with_experimental.tsv`
  and `lossless_*.tsv`)

Both codecs surface **13 non-trivial clusters at ρ≥0.95**. The Jaccard
overlap between codec clusterings is the question — does zenjxl
confirm the design insights from zenwebp?

## Headline: 6 perfect-Jaccard agreements

These six clusters are **structurally identical** between zenwebp and
zenjxl, despite the codecs encoding different bitstream formats with
different perceptual/quality metrics:

| Cluster | Members | Jaccard |
|---|---|--:|
| **Median block cost** | `aq_map_p50 ↔ noise_floor_y_p50 ↔ quant_survival_y_p50` | **1.000** |
| **Aspect** | `aspect_min_over_max ↔ log_aspect_abs` | **1.000** |
| **Palette boolean** | `indexed_palette_width ↔ palette_fits_in_256` | **1.000** |
| **UV chroma compressibility** | `dct_compressibility_uv ↔ quant_survival_uv` | **1.000** |
| **Edge / coef survival** | `edge_density ↔ quant_survival_y` | **1.000** |
| **p1 low-tail** | `aq_map_p1 ↔ noise_floor_y_p1` | **1.000** |

This level of cross-codec agreement is structural, not corpus
artifact — these six clusters are how the analyzer's metric system
*works*, not how a particular corpus rendered them.

The single most actionable finding stays the same: **the
`{aq_map, noise_floor_y, quant_survival_y}` trio measures the same
physical signal at the median, on both codecs.** Pick one
representative per percentile, not three.

## Strong but not identical (Jaccard 0.5–0.8)

| zenwebp cluster | zenjxl cluster | J | Note |
|---|---|--:|---|
| `aq_map_p5/10` + `noise_floor_y_p5/10` (4) | + `noise_floor_y` parent (5) | 0.80 | zenjxl extends the cluster to include the unsmoothed mean |
| `aq_map/noise_floor_y/quant_survival_y` p75 (3) | first two only (2) | 0.67 | zenjxl drops `quant_survival_y_p75` — different per-cell percentile shape |
| `aq_map_mean` + `uniformity` (2) | + `uniformity_smooth` (3) | 0.67 | zenjxl confirms smooth+hard pair lives in same cluster as the mean |

Same physical signal, slightly different cluster boundaries.

## Different cluster boundaries, same semantic group

| Cluster | zenwebp | zenjxl |
|---|---|---|
| Resolution dimension | 16 features (incl. all log/sqrt/log_padded variants) | 5 features (only the kept ones) |
| `noise_floor_uv` adjacent | `_p25 ↔ _p50` | `floor ↔ _p25` |
| Flat-color | `flat_color_block_ratio ↔ screen_content_likelihood` (retired composite) | `flat_color_block_ratio ↔ flat_color_smooth` (the smooth-vs-hard pair) |

These differences come from **schema differences** between the two
features TSVs — zenwebp's TSV was generated against a wider feature
schema (114 columns) that includes log-derivatives + retired
composites that zenjxl's TSV (98 columns) doesn't have. The
underlying redundancy structure is the same; what merges with what
depends on which features the TSV happens to expose.

## zenjxl-only finding: `chroma_complexity ↔ colourfulness`

This 2-feature cluster appears on zenjxl (Jaccard 0 against zenwebp)
but NOT on zenwebp at the ρ≥0.95 threshold.

```
feat_chroma_complexity, feat_colourfulness
```

Both measure color spread:
- `colourfulness` = Hasler-Süsstrunk M3 (`σ²_rg + σ²_yb + 0.3·μ_term`)
- `chroma_complexity` = `√(σ²_Cb + σ²_Cr)`

Algebraically these are NOT identical (different color spaces, different
weighting), but on zenjxl's 400-image corpus they correlate at ρ≥0.95.
On zenwebp's 1264-image corpus they don't — likely because zenwebp's
expanded-multiaxis corpus has more grayscale + indexed-palette content
where the Hasler-Süsstrunk and Cb/Cr-variance forms diverge.

**Recommendation**: keep both for now (zenwebp's wider-corpus signal
suggests they DO carry independent signal in places). If a future
multi-codec ablation confirms ρ≥0.95 across more corpora, fold to
one (probably `colourfulness` — Hasler-Süsstrunk is the more
established perceptual metric).

## VIF tail (zenjxl)

The infinity-VIF (perfectly recoverable) features differ between codecs:

| zenwebp | zenjxl |
|---|---|
| `log_aspect_abs` (cluster #003) | `distinct_color_bins` (#24) |
| `pixel_count` + 9 log/dim variants (cluster #067) | `palette_density` (#25) |
| | `pixel_count` + `bitmap_bytes` (#27) |
| | `noise_floor_y` + `noise_floor_y_p10` (#54) |
| | `laplacian_variance_p1` (#66) |

zenjxl's `palette_density = distinct_color_bins / min(pixel_count, 32_768)`
saturates VIF — the corpus has enough resolution variance that the
ratio is perfectly recoverable. (LOO showed this carries marginal
signal anyway; the picker MLP can't compute the division but it's
weakly load-bearing.)

zenjxl's `noise_floor_y ↔ noise_floor_y_p10` infinity-VIF says **the
mean and the 10th percentile of the noise floor are linearly
recoverable from each other on this corpus** — the noise-floor
distribution is so concentrated at its low tail that p10 IS the
mean (or a constant offset from it). Strong signal that the two are
duplicates for this picker training run. zenwebp shows ρ≈0.99 too
(cluster #054 had them) — same story, weaker VIF only because the
larger zenwebp corpus has slightly more dispersion.

## Practical implications for `FEATURE_GROUPS`

The cross-codec analysis tightens the proposed group definitions
from the zenwebp-only report:

1. **High-confidence groups** (perfect-Jaccard cross-codec agreement
   — ship in the first prototype validator):
   - `median_block_cost`: `{aq_map_p50, noise_floor_y_p50, quant_survival_y_p50}`, max_picked=1
   - `aspect`: `{aspect_min_over_max, log_aspect_abs}`, max_picked=1
   - `palette_boolean`: `{indexed_palette_width, palette_fits_in_256}`, max_picked=1 *(legacy — id 30 retired and replaced by `palette_log2_size` in commit `248b48b`; the 0.1.x analyzer ships the new variant)*
   - `uv_chroma_compressibility`: `{dct_compressibility_uv, quant_survival_uv}`, max_picked=1
   - `edge_coef_survival`: `{edge_density, quant_survival_y}`, max_picked=1
   - `low_tail_block_cost_p1`: `{aq_map_p1, noise_floor_y_p1}`, max_picked=1

2. **Strong-agreement groups** (slight cross-codec boundary
   differences but same core insight):
   - `low_tail_block_cost_p5_p10`: `{aq_map_p5, aq_map_p10, noise_floor_y_p5, noise_floor_y_p10}`, max_picked=1 (or 2 if p5 vs p10 have different LOO signatures)
   - `upper_tail_block_cost`: `{aq_map_p75, noise_floor_y_p75, quant_survival_y_p75}`, max_picked=1
   - `flat_block_signal`: `{aq_map_mean, uniformity, uniformity_smooth}`, max_picked=1

3. **Defer pending more data**:
   - `chroma_complexity ↔ colourfulness` — zenjxl says yes, zenwebp says
     no. Run the analysis on zenjpeg + zenavif before deciding.

## Files

| Codec | Dendrogram SVG | Heatmap SVG | Clusters TSV |
|---|---|---|---|
| zenwebp | `feature_groups_zenwebp_dendrogram_2026-05-02.svg` | `feature_groups_zenwebp_heatmap_2026-05-02.svg` | `feature_groups_zenwebp_clusters_2026-05-02.tsv` |
| zenjxl_lossy | `feature_groups_zenjxl_lossy_dendrogram_2026-05-02.svg` | `feature_groups_zenjxl_lossy_heatmap_2026-05-02.svg` | `feature_groups_zenjxl_lossy_clusters_2026-05-02.tsv` |
| zenjxl_lossless | `feature_groups_zenjxl_lossless_dendrogram_2026-05-02.svg` | `feature_groups_zenjxl_lossless_heatmap_2026-05-02.svg` | `feature_groups_zenjxl_lossless_clusters_2026-05-02.tsv` |

All under `benchmarks/`. The .py driver
(`feature_groups_2026-05-02.py`) takes `--tsv` and `--label` so you
can re-run on any features TSV:

```bash
python3 benchmarks/feature_groups_2026-05-02.py --label <slug> --tsv <path>
```

## Next moves

1. Run the same analysis on zenjpeg and zenavif features TSVs once
   their schemas converge with the post-cull state.
2. Implement the `FEATURE_GROUPS` validator on the high-confidence
   group set (the 6 perfect-Jaccard clusters above).
3. The multi-seed LOO sweep (currently running on zenwebp) will
   provide the per-feature ranking that determines `max_picked`
   when a group has multiple LOO-positive members.
