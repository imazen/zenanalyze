# Feature group structure — zenwebp Tier 0 dendrogram, 2026-05-02

Hierarchical clustering on the zenwebp pareto features TSV
(`zenwebp_pareto_features_2026-05-01_combined_filled.tsv`,
1 264 image-instance rows × 114 features → 100 features after
dropping 14 corpus-constant columns: alpha_*, hdr_*, wide_gamut_*,
gamut_coverage_*, effective_bit_depth, etc.).

**Method:** distance = `1 − |ρ_Spearman|`, average linkage,
`scipy.cluster.hierarchy`. Cuts at ρ ∈ {0.99, 0.95, 0.90, 0.85}
yielding 79 / 69 / 59 / 51 clusters respectively.

## Visualizations

- `benchmarks/feature_groups_dendrogram_2026-05-02.svg` — full
  dendrogram with the four threshold lines drawn. Read top-down:
  the deeper a fork sits in the tree, the tighter the redundancy.
- `benchmarks/feature_groups_heatmap_2026-05-02.svg` — reordered
  correlation heatmap (rows + cols permuted by leaf order). Block
  structure along the diagonal = redundancy clusters; bright pixels
  off-diagonal = inter-cluster correlations the dendrogram merged at
  a higher level.
- `benchmarks/feature_groups_clusters_2026-05-02.tsv` — per-feature
  cluster id at each threshold + VIF + semantic-prefix overlay,
  100 rows.

## The five non-trivial clusters at ρ≥0.95

These are the actionable groups the dendrogram surfaces. The
single-codec, single-corpus result is suggestive — multi-codec
replication would tighten the bound.

### 1. The 16-feature resolution cluster (cluster #067 at ρ≥0.85)

```
feat_pixel_count, feat_log_pixels, feat_min_dim, feat_max_dim,
feat_bitmap_bytes, feat_log2_pixels, feat_log10_pixels,
feat_log_pixels_rounded, feat_sqrt_pixels, feat_log_bitmap_bytes,
feat_log_min_dim, feat_log_max_dim,
feat_log_padded_pixels_{8,16,32,64}
```

VIF saturates at ∞ for most members (perfect linear recoverability).
Despite this, the LOO retrain showed `bitmap_bytes` (+0.90 pp OH,
−8.6 pp AC), `log_padded_pixels_{8,16,32}` (+0.43 to +0.69 pp OH,
−1.8 to −3.3 pp AC), and `log_pixels` (mixed) all carry signal — the
tiny-MLP-expressivity argument. The picker MLP uses these as separate
numerical handles even though they're algebraically identical.

**Group design**: `resolution_dimension`, `max_picked = 4–5`. Rank
order from LOO: `pixel_count > bitmap_bytes > log_padded_pixels_16 >
log_padded_pixels_32 > log_padded_pixels_8 > log_pixels > min_dim >
max_dim` — pick top-K. Already-retired members (`log2/log10/sqrt/
log_rounded/log_bitmap/log_min/max_dim/log_padded_pixels_64`) are
linear scalings the MLP can recover.

### 2. The aq_map/noise_floor low-tail cluster (cluster #055, ρ≥0.95)

```
feat_aq_map_p5, feat_aq_map_p10,
feat_noise_floor_y_p5, feat_noise_floor_y_p10
```

Plus the related cluster #054: `aq_map_p1 ↔ noise_floor_y_p1`. **Two
different metric families converge in their low percentiles** — the
"smoothest blocks" concept comes out the same regardless of which
metric you measure with. This is the "naming-disagrees-with-data"
finding.

**Group design**: `low_tail_dist`, `max_picked = 1`. The pair
{aq_map_pN, noise_floor_y_pN} for N ∈ {1, 5, 10} should pick exactly
one — they're measuring the same thing.

### 3. The aq_map/noise_floor/quant_survival p50 cluster (cluster #057)

```
feat_aq_map_p50, feat_noise_floor_y_p50, feat_quant_survival_y_p50
```

Three different metric families at the *median*. These three measure
"how much the encoder spends on a typical block" from different
angles (DCT energy / pixel-domain noise / coefficient survival) and
converge at p50. Same content drives all three.

**Group design**: `median_block_cost`, `max_picked = 1`. LOO would
need to pick whichever has the cleanest gradient signal.

### 4. The aq_map/noise_floor/quant_survival p75 cluster (cluster #063)

```
feat_aq_map_p75, feat_noise_floor_y_p75, feat_quant_survival_y_p75
```

Identical pattern to p50, at the upper-tail. Confirms the structural
finding: **the {aq_map, noise_floor_y, quant_survival_y} trio
describes the same physical signal** (per-block encode cost
distribution) at every percentile we tested.

**Group design**: `upper_tail_block_cost`, `max_picked = 1`.

### 5. Adjacent-percentile clusters (singleton-prefix)

- cluster #058: `feat_noise_floor_uv_p25 ↔ feat_noise_floor_uv_p50`
- cluster #064: `feat_aq_map_p90 ↔ feat_aq_map_p95`

These are the expected "adjacent percentiles inside the same metric
correlate" — well-named clusters, low surprise.

**Group design**: per-metric percentile groups, each with
`max_picked = 2 or 3` (mean + a few well-separated percentiles, not
the whole tail).

## Cross-prefix surprises (naming-disagrees-with-data)

These are the design insights the auto-clustering surfaces that the
schema names hide. Each is a single auto-cluster that spans two
semantic prefixes:

| Cluster | Members | Insight |
|---|---|---|
| #029 | `feat_uniformity` ↔ `feat_aq_map_mean` | Pixel-domain flatness and block-energy mean track each other → flat content drives both. |
| #031 | `feat_flat_color_block_ratio` ↔ `feat_screen_content_likelihood` | The composite likelihood (already retired) WAS a linear function of `flat_color_block_ratio`. Confirms the cull. |
| #037 | `feat_edge_density` ↔ `feat_quant_survival_y` | Edges drive DCT coefficient survival — physically grounded. |
| #022 | `feat_dct_compressibility_uv` ↔ `feat_quant_survival_uv` | UV-side equivalent: chroma compressibility = chroma quant survival. |
| #054/#055/#057/#063 | aq_map ⟷ noise_floor_y at every percentile | The two metrics ARE measuring the same thing on this corpus. |

## VIF tail

Top 15 features by VIF (saturated at 10⁶ for the resolution cluster):

```
VIF=1.0e6  feat_log_aspect_abs            (cluster #003: aspect)
VIF=1.0e6  feat_pixel_count               (cluster #067: resolution)
VIF=1.0e6  feat_log_pixels                (#067)
VIF=1.0e6  feat_bitmap_bytes              (#067)
VIF=1.0e6  feat_log2_pixels               (#067)
VIF=1.0e6  feat_log10_pixels              (#067)
VIF=1.0e6  feat_log_bitmap_bytes          (#067)
VIF=1.0e6  feat_log_min_dim               (#067)
VIF=1.0e6  feat_log_max_dim               (#067)
VIF=1.0e6  feat_log_padded_pixels_8       (#067)
VIF=3.6e5  feat_log_padded_pixels_16      (#067)
VIF=8.7e4  feat_sqrt_pixels               (#067)
VIF=4.0e4  feat_log_padded_pixels_32      (#067)
VIF=2.5e4  feat_min_dim                   (#067)
VIF=2.0e4  feat_max_dim                   (#067)
```

Every high-VIF feature lands in the resolution cluster. The dendrogram
and VIF agree perfectly here.

## Concrete `FEATURE_GROUPS` proposal (zenwebp)

Seeded from this analysis; `max_picked` values reflect the LOO data
where available, otherwise conservative defaults:

```python
FEATURE_GROUPS = {
    "resolution_dimension": {
        "members": [
            "feat_pixel_count",
            "feat_log_pixels",
            "feat_bitmap_bytes",
            "feat_log_padded_pixels_8",
            "feat_log_padded_pixels_16",
            "feat_log_padded_pixels_32",
            "feat_min_dim",
            "feat_max_dim",
        ],
        "max_picked": 5,  # LOO showed ≥5 carry signal; rank by ΔAC
    },
    "aspect": {
        "members": ["feat_aspect_min_over_max", "feat_log_aspect_abs"],
        "max_picked": 1,  # cluster #003 confirms
    },
    "low_tail_block_cost": {
        "members": [
            "feat_aq_map_p1", "feat_aq_map_p5", "feat_aq_map_p10",
            "feat_noise_floor_y_p1", "feat_noise_floor_y_p5",
            "feat_noise_floor_y_p10",
        ],
        "max_picked": 1,  # all merge at p1/5/10
    },
    "median_block_cost": {
        "members": [
            "feat_aq_map_p50",
            "feat_noise_floor_y_p50",
            "feat_quant_survival_y_p50",
        ],
        "max_picked": 1,
    },
    "upper_tail_block_cost": {
        "members": [
            "feat_aq_map_p75",
            "feat_noise_floor_y_p75",
            "feat_quant_survival_y_p75",
        ],
        "max_picked": 1,
    },
    "aq_map_extreme_tail": {
        "members": ["feat_aq_map_p90", "feat_aq_map_p95", "feat_aq_map_p99"],
        "max_picked": 1,
    },
    "noise_floor_uv_dist": {
        "members": [
            "feat_noise_floor_uv",
            "feat_noise_floor_uv_p25",
            "feat_noise_floor_uv_p50",
            "feat_noise_floor_uv_p90",
        ],
        "max_picked": 1,
    },
    "uniformity_aq_mean": {
        "members": ["feat_uniformity", "feat_aq_map_mean"],
        "max_picked": 1,  # cross-prefix #029
    },
    "edge_signals": {
        "members": ["feat_edge_density", "feat_quant_survival_y"],
        "max_picked": 1,  # cross-prefix #037
    },
    "dct_compressibility_uv": {
        "members": [
            "feat_dct_compressibility_uv",
            "feat_quant_survival_uv",
        ],
        "max_picked": 1,
    },
    "palette": {
        "members": [
            "feat_distinct_color_bins",
            "feat_palette_density",
            "feat_palette_fits_in_256",
            "feat_palette_log2_size",
        ],
        "max_picked": 2,
    },
    # ... cb_sharpness / cr_sharpness families ... (per-codec)
}
```

The validator at picker-config-load time enforces
`count(KEEP_FEATURES ∩ group.members) ≤ group.max_picked` for every
group. Violations error out before training.

## Limitations

- **Single codec.** zenjpeg / zenavif / zenjxl Tier 0 outputs surface
  the same {aq_map ↔ noise_floor_y, palette_density ↔
  distinct_color_bins} clusters — but the per-codec details (which
  specific percentile combinations cluster) likely differ. Cross-codec
  agreement on group boundaries needs replication.
- **Single corpus.** The 1264-image v0.2 corpus is photographic-heavy.
  The expanded multi-axis corpus (97 MP / 463 images) hasn't been run
  through this analysis yet — its richer alpha/HDR/palette/screen-
  content axes will move some clusters.
- **Linear-only.** Spearman is monotonic-correlation. A genuinely
  nonlinear signal that two features encode at different scales would
  appear independent here. Mutual-information would catch it but is
  expensive to estimate.
- **Single-corpus VIF.** VIF computed on the same TSV — collinearity
  is also corpus-dependent.

## Next moves

1. **Multi-seed LOO sweep already running** (8 features × 5 seeds × 2,
   ~70 min wall) — feeds into the per-group ranking that determines
   `max_picked` per group.
2. **Cross-codec replication**: run the same dendrogram on the four
   features TSVs (zenjpeg/zenwebp/zenavif/zenjxl) and look at where
   the auto-clusters agree vs. diverge.
3. **Implement the validator**: add `FEATURE_GROUPS` + `validate_keep_features(keep, groups)`
   to one picker config (zenwebp) as a prototype. Document in
   `zentrain/PRINCIPLES.md`.
4. **LOO Δ-signature embedding** (deferred): once we have multi-seed
   data on multiple codecs, cluster features by Δ-signature similarity
   as a secondary group definition. Catches functional redundancy
   that pure correlation misses.
