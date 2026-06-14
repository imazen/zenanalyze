# Per-codec feature redundancy clusters — KEEP_FEATURES guidance (2026-06-13)

This file collates the ρ≥0.90 feature redundancy clusters that the
2026-05-02 `feature_groups` dendrogram runs and the 2026-06-09 zenjpeg
tree-A/B importance run surfaced, **per codec**, as a reference for
choosing a codec's `KEEP_FEATURES` list. The companion machine-readable
table is `feature_redundancy_clusters_2026-06-13.json`; the advisory
lint that reads it is `../zentrain/tools/lint_keep_features.py`.

## Principle: redundancy is PER-CODEC — this is guidance, NOT a gate

Two features that correlate at ρ≥0.90 on **one** codec's corpus may
decorrelate on another's. The cross-codec report
(`feature_groups_cross_codec_2026-05-02.md`) documents exactly this:
`chroma_complexity ↔ colourfulness` clusters on zenjxl's 400-image
corpus but **not** on zenwebp's 1264-image corpus at ρ≥0.95, because
zenwebp's wider corpus has more grayscale + indexed-palette content
where the Hasler-Süsstrunk and Cb/Cr-variance forms diverge. The
resolution cluster also differs in size between codecs purely because
the two TSVs were generated against different feature schemas.

So:

- **Do NOT gate, hide, or remove a feature** because it lands in a
  redundancy cluster. The analyzer's job is to surface signals; the
  consumer decides. A feature that is redundant on one corpus is
  load-bearing on another.
- **DO prefer ≤1 member per cluster per head** when you assemble a
  codec's `KEEP_FEATURES`, unless you have codec-specific LOO/ablation
  evidence that a second member carries independent signal (the
  resolution cluster is the documented exception — see below).
- The lint is **advisory and non-blocking** (always exits 0). It warns
  when a `KEEP_FEATURES` list picks ≥2 members of one cluster so you
  re-check whether the duplication is intentional.

### Sources

| Source file | What it provides |
|---|---|
| `feature_groups_2026-05-02.md` | zenwebp dendrogram narrative; the five ρ≥0.95 clusters + cross-prefix surprises; `FEATURE_GROUPS` proposal with `max_picked`. |
| `feature_groups_cross_codec_2026-05-02.md` | zenwebp-vs-zenjxl Jaccard comparison; the 6 perfect-agreement clusters; the per-codec divergences. |
| `feature_groups_zenwebp_clusters_2026-05-02.tsv` | zenwebp per-feature cluster-id @ ρ∈{0.99,0.95,0.90,0.85} + VIF. **The ρ≥0.90 cluster column (`cluster_at_0_90`) is transcribed below.** |
| `feature_groups_zenjxl_lossy_clusters_2026-05-02.tsv` / `…_lossless_…` | zenjxl per-feature cluster-id @ same thresholds + VIF. **The two files are byte-identical** (features are computed per source image, not per encode — confirmed by `cmp`), so one zenjxl section covers both. |
| `picker_tree_ab_importance_2026-06-09.md` | zenjpeg GBDT permutation importance, **grouped by ρ≥0.9 redundancy cluster** — used for the zenjpeg section + representative ranking. |
| `all_time_best_features_2026-05-02.md` | cross-reference only — its "15-feature minimum set" is stated to be mutually disjoint across these clusters. |

**Threshold note.** The `.md` narratives report the ρ≥0.95 cut. The
TSVs carry all four cuts; **the tables below use the ρ≥0.90 cut
(`cluster_at_0_90`)** because that is the lint's stated threshold and
it captures slightly looser-but-still-redundant pairs (e.g. the
sharpness horiz/vert pairs) that the 0.95 cut splits. Where the
narrative singled a cluster out as ρ≥0.95 or perfect-Jaccard
cross-codec, that is noted on the row. VIF values quoted are from the
TSVs; a VIF of `1.0e6` is the saturation value the driver writes for
perfectly-recoverable (∞-VIF) features.

---

## zenwebp — 17 multi-member clusters at ρ≥0.90

Source: `feature_groups_zenwebp_clusters_2026-05-02.tsv`
(`cluster_at_0_90` column; 1 264 image-instance rows × 100 active
features). Cluster ids are the TSV's own numbering. "Rep" = the member
the narrative/ablation flags as the one to keep when picking one;
blank where the source does not single one out.

| Cluster id | Members (feat_ prefix dropped) | Rep (source-flagged) | Notes / source |
|---|---|---|---|
| cl3 | `aspect_min_over_max`, `log_aspect_abs` | `aspect_min_over_max` | "aspect" cluster, perfect-Jaccard cross-codec (cross_codec.md). `log_aspect_abs` is ∞-VIF. |
| cl4 | `block_misalignment_8`, `block_misalignment_16` | — | Block-misalignment scale twins. |
| cl8 | `indexed_palette_width`, `palette_fits_in_256` | — | "palette boolean", perfect-Jaccard cross-codec (cross_codec.md). NB id 30 retired → replaced by `palette_log2_size` in the 0.1.x analyzer (cross_codec.md note). |
| cl16 | `chroma_complexity`, `colourfulness` | `colourfulness` (tentative) | Clusters on zenwebp at ρ≥0.90 but the .md flagged this as ρ≥0.95 **zenjxl-only**; recommendation is keep both until multi-codec ablation confirms (cross_codec.md). |
| cl17 | `cr_sharpness`, `cr_horiz_sharpness` | `cr_sharpness` | Cr sharpness mean vs horizontal axis. |
| cl19 | `dct_compressibility_uv`, `quant_survival_uv` | — | "UV chroma compressibility", perfect-Jaccard cross-codec (cross_codec.md): chroma compressibility = chroma quant survival. |
| cl20 | `cb_sharpness`, `cb_horiz_sharpness` | `cb_sharpness` | Cb sharpness mean vs horizontal axis. |
| cl25 | `uniformity`, `aq_map_mean`, `noise_floor_y` | `uniformity` or `aq_map_mean` | Cross-prefix #029: pixel-domain flatness ↔ block-energy mean; flat content drives both (feature_groups.md). |
| cl26 | `flat_color_block_ratio`, `screen_content_likelihood` | `flat_color_block_ratio` | Cross-prefix #031: the (retired) composite likelihood WAS a linear function of `flat_color_block_ratio` (feature_groups.md). `screen_content_likelihood` is retired in the 0.1.x analyzer. |
| cl32 | `edge_density`, `quant_survival_y` | `edge_density` | Cross-prefix #037 / "edge-coef survival", perfect-Jaccard cross-codec (cross_codec.md): edges drive DCT coefficient survival. |
| cl49 (n=7) | `aq_map_p1`, `aq_map_p5`, `aq_map_p10`, `noise_floor_y_p1`, `noise_floor_y_p5`, `noise_floor_y_p10`, `noise_floor_y_p25` | one only (`max_picked=1`) | Low-tail block cost. `{aq_map_pN, noise_floor_y_pN}` measure the same "smoothest blocks" concept; the .md proposes `max_picked=1` (feature_groups.md §2). The p1 pair is perfect-Jaccard cross-codec. |
| cl50 (n=3) | `aq_map_p50`, `noise_floor_y_p50`, `quant_survival_y_p50` | one only (`max_picked=1`) | **Median block cost** — perfect-Jaccard (1.000) cross-codec; the single most-cited finding: the `{aq_map, noise_floor_y, quant_survival_y}` trio measures the same physical signal at the median. Pick one (feature_groups.md §3, cross_codec.md). |
| cl51 | `noise_floor_uv_p25`, `noise_floor_uv_p50` | — | Adjacent UV-noise percentiles (feature_groups.md §5, cluster #058). |
| cl53 (n=3) | `noise_floor_uv_p75`, `noise_floor_uv_p90`, `quant_survival_uv_p75` | one only | Upper-tail UV noise / quant survival. |
| cl54 (n=3) | `aq_map_p75`, `noise_floor_y_p75`, `quant_survival_y_p75` | one only (`max_picked=1`) | **Upper-tail block cost** — same trio as cl50 at p75 (feature_groups.md §4). zenjxl drops `quant_survival_y_p75` from this cluster (Jaccard 0.67, cross_codec.md). |
| cl55 (n=3) | `aq_map_p90`, `aq_map_p95`, `aq_map_p99` | one or two | aq_map extreme tail; adjacent percentiles inside one metric (feature_groups.md §5, cluster #064 was the p90↔p95 pair). |
| cl57 (n=16) | `pixel_count`, `bitmap_bytes`, `log_pixels`, `log2_pixels`, `log10_pixels`, `log_pixels_rounded`, `sqrt_pixels`, `log_bitmap_bytes`, `min_dim`, `max_dim`, `log_min_dim`, `log_max_dim`, `log_padded_pixels_8`, `log_padded_pixels_16`, `log_padded_pixels_32`, `log_padded_pixels_64` | `pixel_count` (rank #1); `max_picked = 4–5` | **Resolution cluster — the documented EXCEPTION.** All members are algebraically related (VIF saturates at ∞ for most), but LOO showed ≥4–5 carry independent tiny-MLP signal. LOO rank: `pixel_count > bitmap_bytes > log_padded_pixels_16 > log_padded_pixels_32 > log_padded_pixels_8 > log_pixels > min_dim > max_dim` (feature_groups.md §1). The retired `log2/log10/sqrt/log_rounded/log_bitmap/log_min/log_max/log_padded_pixels_64` are linear scalings the MLP recovers. |

---

## zenjxl (lossy and lossless) — 16 multi-member clusters at ρ≥0.90

Source: `feature_groups_zenjxl_lossy_clusters_2026-05-02.tsv` (byte-
identical to the lossless TSV). 400 (image, size) rows × 87 active
features.

| Cluster id | Members (feat_ prefix dropped) | Rep (source-flagged) | Notes / source |
|---|---|---|---|
| cl1 | `aspect_min_over_max`, `log_aspect_abs` | `aspect_min_over_max` | "aspect", perfect-Jaccard cross-codec. |
| cl6 | `chroma_complexity`, `colourfulness` | `colourfulness` (tentative) | The **zenjxl-only** ρ≥0.95 cluster the cross_codec.md highlights; both measure color spread. Keep both pending more corpora (cross_codec.md). |
| cl14 | `dct_compressibility_y`, `aq_map_p90` | — | Luma DCT compressibility ↔ aq_map upper tail (zenjxl-specific boundary). |
| cl22 | `indexed_palette_width`, `palette_fits_in_256` | — | "palette boolean", perfect-Jaccard cross-codec. |
| cl26 (n=5) | `pixel_count`, `bitmap_bytes`, `log_pixels`, `min_dim`, `max_dim` | `pixel_count` | Resolution cluster — **only 5 members here** vs zenwebp's 16, because zenjxl's TSV doesn't expose the log/sqrt/log_padded derivatives (cross_codec.md). `pixel_count` and `bitmap_bytes` are ∞-VIF. |
| cl27 (n=6) | `aq_map_p50`, `noise_floor_y_p50`, `quant_survival_y_p50`, `uniformity`, `uniformity_smooth`, `aq_map_mean` | one only | **Median block cost (extended)** — the perfect-Jaccard median trio, here merged with the flat-block signal (`uniformity`, `uniformity_smooth`, `aq_map_mean`). zenjxl extends the cluster to absorb the smooth+hard uniformity pair (Jaccard 0.67 on the uniformity sub-cluster, cross_codec.md). |
| cl29 (n=3) | `aq_map_p75`, `noise_floor_y_p75`, `quant_survival_y_p75` | one only | **Upper-tail block cost** — full trio on zenjxl (zenwebp keeps it too; here `quant_survival_y_p75` stays in, unlike the cross_codec.md's "drops it" note which referred to the 0.95 cut). |
| cl30 (n=4) | `edge_density`, `quant_survival_y`, `laplacian_variance_p90`, `luma_kurtosis` | `edge_density` | "edge / coef survival", perfect-Jaccard core (`edge_density ↔ quant_survival_y`); zenjxl extends to absorb `laplacian_variance_p90` + `luma_kurtosis` at ρ≥0.90. |
| cl33 | `noise_floor_uv_p75`, `quant_survival_uv_p75` | — | Upper-tail UV noise ↔ UV quant survival. |
| cl34 (n=3) | `dct_compressibility_uv`, `quant_survival_uv`, `noise_floor_uv_p90` | — | "UV chroma compressibility" core (`dct_compressibility_uv ↔ quant_survival_uv`, perfect-Jaccard) extended with `noise_floor_uv_p90`. |
| cl35 | `cb_sharpness`, `cr_sharpness` | `cb_sharpness` | On zenjxl the Cb and Cr **mean** sharpness merge (they don't on zenwebp at ρ≥0.90 — zenwebp pairs each with its own horiz axis instead). |
| cl36 | `cr_horiz_sharpness`, `cr_vert_sharpness` | — | Cr horizontal ↔ vertical axis. |
| cl37 | `cb_horiz_sharpness`, `cb_vert_sharpness` | — | Cb horizontal ↔ vertical axis. |
| cl40 (n=3) | `noise_floor_uv`, `noise_floor_uv_p25`, `noise_floor_uv_p50` | — | UV noise floor mean + low/median percentiles (zenjxl boundary is `floor ↔ p25` per cross_codec.md). |
| cl41 | `aq_map_p1`, `noise_floor_y_p1` | one only | p1 low-tail pair — perfect-Jaccard (1.000) cross-codec. |
| cl42 (n=8) | `flat_color_block_ratio`, `flat_color_smooth`, `aq_map_p5`, `aq_map_p10`, `noise_floor_y_p5`, `noise_floor_y_p10`, `noise_floor_y`, `noise_floor_y_p25` | one only (mostly) | Low-tail block cost (p5/p10/p25) merged with the flat-color smooth+hard pair. On zenjxl the flat-color pair is `flat_color_block_ratio ↔ flat_color_smooth` (vs zenwebp's `↔ screen_content_likelihood`, cross_codec.md). `noise_floor_y ↔ noise_floor_y_p10` are ∞-VIF on zenjxl (mean ≈ p10 — the noise-floor distribution is concentrated at its low tail). |

---

## zenjpeg — redundancy clusters from the 2026-06-09 tree-A/B importance

Source: `picker_tree_ab_importance_2026-06-09.md`, "Grouped by ρ≥0.9
redundancy cluster" section. This run clustered the **93 zenjpeg
features → 50 ρ≥0.9 groups** on the real zenjpeg sweep
(`zq_pareto_2026-04-29_adapted.parquet`, 9 733 picker rows) and reports
**permutation importance per cluster** (shuffling the whole cluster so
correlated twins can't mask each other). The doc lists only the
high-signal clusters explicitly (with `+N` placeholders for the
remaining members it did not enumerate); those are transcribed verbatim
below. **`+N` means N further unnamed members in that cluster** — the
doc did not list them, so they are not invented here.

| Cluster (named members) | Cluster permutation importance | Maps to (doc) |
|---|---|---|
| `cr_sharpness`, `dct_compressibility_uv`, `quant_survival_uv`, `noise_floor_uv_p50`, **+3 more** | +0.0512 | **CHROMA / UV detail** → the 444-vs-420 subsampling call. (Top single chroma feature is only 0.038 — the group beats it, confirming "ablate by cluster.") |
| `edge_density`, `noise_floor_y`, `quant_survival_y`, `noise_floor_y_p25`, **+14 more** | +0.0455 | **LUMA detail / edge** → trellis / quant. |
| `distinct_color_bins`, `palette_density` | +0.0155 | color-count → colorspace decision. |
| `uniformity`, `aq_map_mean`, `aq_map_p50`, `noise_floor_y_p50`, **+1 more** | +0.0135 | flat-block / median encoder spend. |
| `cb_sharpness` (singleton in the grouped view) | +0.0217 | Cb chroma detail (stayed its own group at ρ≥0.9 on the zenjpeg corpus). |
| `dct_compressibility_y` (singleton) | +0.0129 | luma DCT compressibility. |
| `zq_norm` (singleton — the target quality) | +0.2074 | dominant input (not a content feature). |

The doc's takeaway: 93 features collapse to 50 ρ≥0.9 groups, only ~6–8
groups carry >0.01 signal, and the set is prunable to ~10–15 features
by keeping the cluster representatives (`zq_norm + cr_sharpness +
cb_sharpness + one luma-detail rep + quant_survival_y +
distinct_color_bins + aq_map_mean`).

**Caveat (from the source):** these importances are for the 12-cell /
6-knob zenjpeg slice (the other ~9 knobs pinned); cluster membership is
on that corpus. The `+N` members are not enumerated in the source, so
the machine-readable JSON for zenjpeg lists only the **named** members
of each cluster — the lint will under-report zenjpeg redundancy for the
unnamed members. Treat zenjpeg cluster coverage as partial.

---

## cross-codec — the high-confidence clusters (ship-first set)

Source: `feature_groups_cross_codec_2026-05-02.md`, "6 perfect-Jaccard
agreements" + "Practical implications". These clusters are
**structurally identical between zenwebp and zenjxl** (Jaccard 1.000) —
the safest default redundancy table to apply to **any new codec** that
has no per-codec cluster run yet. The lint falls back to this `cross_codec`
table when the requested codec has no entry.

| Cross-codec cluster | Members (feat_ prefix dropped) | Jaccard | max_picked (cross_codec.md) |
|---|---|--:|--:|
| Median block cost | `aq_map_p50`, `noise_floor_y_p50`, `quant_survival_y_p50` | 1.000 | 1 |
| Aspect | `aspect_min_over_max`, `log_aspect_abs` | 1.000 | 1 |
| Palette boolean | `indexed_palette_width`, `palette_fits_in_256` | 1.000 | 1 |
| UV chroma compressibility | `dct_compressibility_uv`, `quant_survival_uv` | 1.000 | 1 |
| Edge / coef survival | `edge_density`, `quant_survival_y` | 1.000 | 1 |
| p1 low-tail | `aq_map_p1`, `noise_floor_y_p1` | 1.000 | 1 |
| Low-tail p5/p10 (strong, not perfect) | `aq_map_p5`, `aq_map_p10`, `noise_floor_y_p5`, `noise_floor_y_p10` | 0.5–0.8 | 1 (or 2 if p5/p10 LOO-differ) |
| Upper-tail block cost (strong) | `aq_map_p75`, `noise_floor_y_p75`, `quant_survival_y_p75` | 0.67 | 1 |
| Flat-block signal (strong) | `aq_map_mean`, `uniformity`, `uniformity_smooth` | 0.67 | 1 |

`chroma_complexity ↔ colourfulness` is **deliberately NOT** in the
ship-first cross-codec table: zenjxl says it's redundant, zenwebp says
it isn't. The cross_codec.md defers it pending zenjpeg + zenavif runs,
so the cross-codec JSON entry omits it; the per-codec zenwebp/zenjxl
tables keep it (cl16 / cl6) because it clusters on those individual
corpora.

---

## How to use this when building a codec's KEEP_FEATURES

1. **Start from the cross-codec ship-first set** (the perfect-Jaccard
   clusters above) as your redundancy map if you have no per-codec
   cluster run; otherwise use the per-codec section for your codec.
2. **Prefer ≤1 member per cluster per head.** When two features land in
   the same cluster, keep the source-flagged representative (or the one
   your own LOO/ablation ranks higher) and drop the rest — *unless* you
   have codec-specific evidence the second member carries independent
   signal. The **resolution cluster (zenwebp cl57 / zenjxl cl26) is the
   standing exception**: LOO showed 4–5 members carry tiny-MLP signal,
   so `max_picked = 4–5` there (feature_groups.md §1).
3. **Run the lint:**
   ```bash
   python3 zentrain/tools/lint_keep_features.py <codec> \
       zentrain/examples/<codec>_picker_config.py
   ```
   It loads `feature_redundancy_clusters_2026-06-13.json` for `<codec>`
   (falling back to `cross_codec`), greps the `KEEP_FEATURES` list out
   of the config, and prints a `WARNING` for every cluster where ≥2
   KEEP members collide, suggesting you keep ≤1. It **always exits 0** —
   it is advisory, because redundancy is per-codec and the analyzer
   never gates a feature.
4. **Cross-check against the 15-feature minimum set** in
   `all_time_best_features_2026-05-02.md` — that set is stated to be
   mutually disjoint across these clusters, so it makes a clean
   redundancy-free seed to extend.

## Provenance

- Cluster tables: 2026-05-02 `feature_groups` runs (Spearman, average
  linkage, `cluster_at_0_90` cut) + 2026-06-09 zenjpeg `picker_tree_ab`
  GBDT permutation importance.
- Transcribed 2026-06-13 from the source files cited per-section above.
  No ρ values or cluster memberships were invented; the zenjpeg `+N`
  placeholders reflect members the source did not enumerate.
- zenjxl lossy/lossless TSVs verified byte-identical via `cmp`.
