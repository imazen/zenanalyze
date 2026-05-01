# Cross-codec ablation: expanded multi-axis corpus, 2026-05-02

Aggregates four-codec Tier 0 correlation runs and two-codec Tier 1.5
permutation-importance runs against the new
`/mnt/v/output/codec-corpus-2026-05-01-multiaxis/` corpus (97.0 MP,
463 images, 9 axis-class buckets). Goal: confirm three deferred
math-redundancy features can be retired and check whether previously
corpus-bound features (alpha, HDR, wide-gamut, aspect, grayscale,
block-misalignment) gained any signal on the expanded fixture set.

## TL;DR

- **`feat_log_pixels` and `feat_bitmap_bytes` are confirmed Tier 0
  redundant on all 4 codecs** (Spearman ≥ 0.9656 vs `feat_pixel_count`
  on every run; ≥ 0.9999 at the 0.99 threshold). They survive in the
  next breaking release alongside the existing 13-transforms purge.
- **`feat_indexed_palette_width` is confirmed redundant against
  `feat_palette_fits_in_256` on all 4 codecs (min |r| 0.97–0.99) but
  the corpus only exercises `n_unique = 4` for it** — the redundancy
  finding is real, the *generalisation* claim deserves more
  small-palette content before we cull. Marking as
  borderline-pending-corpus-densification.
- **None of the corpus-bound features gained signal**: alpha stayed
  constant (analyzer's `analyze_features_rgb8` strips alpha — known
  zenanalyze limitation, not a corpus problem); HDR / wide-gamut /
  effective-bit-depth stayed constant on the two codecs that built
  with `--features hdr` (zenjpeg, zenjxl) because the corpus's
  "hdr_widegamut" axis class is gain-map JPEGs whose base renders
  SDR. The corpus needs native PQ/HLG content before these can be
  validated.

## Corpus expansion in numbers

| Field | Value |
|------:|:------|
| Path | `/mnt/v/output/codec-corpus-2026-05-01-multiaxis/manifest.tsv` |
| Total images | 463 |
| Total pixels | 97.02 MP |
| Axis classes | 9 (illustration, screen, hdr-widegamut, indexed-palette, grayscale-nonpow2, alpha-nonpow2, ultrawide, portrait, landscape) |

Axis-class breakdown:

| Class | n |
|:------|---:|
| `content_illustration` | 130 |
| `content_screencontent` | 70 |
| `hdr_widegamut` | 53 |
| `indexed_palette_smalledge` | 50 |
| `grayscale_nonpow2_small` | 50 |
| `alpha_nonpow2_portrait` | 50 |
| `wide_aspect_ultrawide_21x9` | 20 |
| `wide_aspect_portrait_9x16` | 20 |
| `wide_aspect_landscape_16x9` | 20 |

Per-codec analyzed-row counts (different by codec because each
encoder downstream-filtered for its sweep grid):

| Codec | n_rows | n_features | Has `--features hdr`? |
|:------|------:|---:|:-:|
| zenjpeg | 768 | 99 | yes |
| zenwebp | 1348 | 89 | no |
| zenavif | 669 | 89 | no |
| zenjxl | 421 | 99 | yes |

## Tier 0 — deferred math redundancies

### Cross-codec table for the three deferred features

`anchor` is the cluster anchor (the survivor); `min |r|` is the
weakest absolute Spearman in the cluster the feature was assigned to.

| Feature | zenjpeg | zenwebp | zenavif | zenjxl | Verdict |
|:--------|:--------|:--------|:--------|:-------|:-------|
| `feat_log_pixels` | dropped → `pixel_count`, min \|r\| **1.0000** | dropped → `pixel_count`, **0.9656** | dropped → `pixel_count`, **0.9712** | dropped → `pixel_count`, **0.9769** | **CULL** (4/4) |
| `feat_bitmap_bytes` | dropped → `pixel_count`, min \|r\| **1.0000** | dropped → `pixel_count`, **0.9656** | dropped → `pixel_count`, **0.9712** | dropped → `pixel_count`, **0.9769** | **CULL** (4/4) |
| `feat_indexed_palette_width` | dropped → `palette_fits_in_256`, **0.9843** (n_unique=4) | dropped → `palette_fits_in_256`, **0.9972** (n_unique=4) | low-variance only, n_unique=4 | dropped → `palette_fits_in_256`, **0.9680** (n_unique=4) | **BORDERLINE** (4-bin only) |

At the stricter 0.99 threshold, log_pixels and bitmap_bytes still
cluster against `pixel_count` on every codec (zenjpeg 1.0000, zenavif
0.9999999..., zenjxl 0.9999999..., zenwebp 0.9989662 in the
pareto-matched run with n=1140). The redundancy is structural, not
sampling noise.

`indexed_palette_width` survives in zenavif at the 0.95 threshold
*only because* the zenavif n_unique = 4 made the column hit the
low-variance pre-filter before clustering started — same underlying
finding (4 indistinct values), different report path. The feature
*is* redundant against `palette_fits_in_256`, but every codec's
corpus only sees 4 distinct width values (0, the small-palette band,
and a couple of wider buckets). Until the corpus has more indexed
content with varied palette widths (16, 32, 64, 128, full 256), we
can't prove it adds nothing.

### Recommendation

- **`feat_log_pixels` and `feat_bitmap_bytes`**: ship the cull.
  These are pure deterministic transforms of `feat_pixel_count`
  (`log_pixels` = ln(width·height); `bitmap_bytes` = width·height·4
  as f32 — see commit `89f07ac`). The 4-codec consensus matches the
  algebraic prediction; no further data needed.
- **`feat_indexed_palette_width`**: leave for now, queue for the
  same breaking-release window with a corpus-densification gate.
  Add ≥ 100 indexed PNG/GIF samples spanning palette widths
  {2, 4, 8, 16, 32, 64, 128, 256}, re-run Tier 0, then decide.

## Tier 0 — corpus-gap features

Which previously-corpus-bound features moved from constant/low-var to
varying on the new corpus.

### `is_grayscale`

Stayed **low-variance n_unique = 2** on every codec. The
`grayscale_nonpow2_small` axis (50 images) generates rows where the
flag = 1; the rest are 0. Boolean by definition, so `n_unique = 2`
is the ceiling. Move it from "low-variance pre-filter" to "expected
binary" in the analyzer documentation; downstream tree learners can
still use it as a leaf split.

### `alpha_*` (alpha_present, alpha_used_fraction, alpha_bimodal_score)

**Still constant 0.0 on all 4 codecs.** This is *not* a corpus gap
even though we added 50 `alpha_nonpow2_portrait` images. The
`analyze_features_rgb8` entry strips the alpha channel before the
analyzer sees the buffer (zenanalyze CLAUDE.md "Alpha — stride-sampled,
**reads source bytes directly**" applies only when callers go
through the alpha-aware entry; the codec-eval driver feeds RGB8). To
surface alpha signal the codec wrappers (zenwebp, zenavif, zenjxl)
need a separate features pass that goes through the RGBA-aware
analyzer. **Logged as zenanalyze entry-point limitation, not a
corpus problem.** No drop list change.

### `hdr_*` and `wide_gamut_*` and `effective_bit_depth`

Constant on **zenjpeg** and **zenjxl** (both with `--features hdr`
enabled, 99 features). Absent from **zenwebp** and **zenavif**
(89 features — built without the `hdr` cargo feature).

The 53 `hdr_widegamut` images are gain-map JPEGs whose base layer
renders sRGB SDR; the analyzer reads the rendered base, not the gain
map, and reports SDR. Logged as a corpus-content bug — replace with
native PQ/HLG content (e.g. AVIF HDR samples, JXL HDR
distance-encoded samples) before claiming HDR features have signal.

### `aspect_min_over_max` and `log_aspect_abs`

Both **kept (not in any cluster)** on zenjpeg, zenwebp, zenavif —
each has independent signal. **zenjxl is the exception**: at
threshold 0.95 it dropped `aspect_min_over_max` against
`log_aspect_abs` (min |r| = 0.9939, cluster size 2), and the same
holds at 0.99. zenjxl needs only one of them; the other three codecs
treat them as separate inputs. This is consistent with zenjxl's
finer-grained spatial reasoning around block sizes — log-aspect and
ratio carry the same information once block-size choice is in the
mix.

Recommendation: keep both features. Codec-specific picker training
will route around the redundancy (the zenjxl picker will pick one).

### `block_misalignment_8` and `block_misalignment_32`

Kept (not in any cluster) on every codec. Earlier work suspected
these were always-zero on power-of-2 corpora; the new
`grayscale_nonpow2_small`, `alpha_nonpow2_portrait`, and the
ultrawide/portrait axes broke that assumption. **First Tier 0 run
where these features cleared the variance gate on every codec.** No
action — they earned their keep on the new corpus.

### `palette_fits_in_256` and `distinct_color_bins`

`palette_fits_in_256` is still low-variance n_unique = 2 (boolean by
definition). `distinct_color_bins` is the **anchor** of a cluster on
all 4 codecs (with `palette_density` consistently dropped against it,
min |r| 0.9614–0.9871). No change to the existing finding — both
features are kept; `palette_density` is an existing cull target
across the board.

## Tier 1.5 — permutation importance (zenjpeg + zenwebp only)

Only zenjpeg and zenwebp completed Phase C this run; zenjxl Phase B/C
was skipped to avoid concurrent-agent collision and zenavif Phase B/C
was skipped because the harness fails on non-power-of-2 sizes.

### zenjpeg top-5 / bottom-5 (Δpp on holdout, 5 repeats, seed 42)

Baseline `mean_pct = 7.30`, `argmin_acc = 0.549`, n_val = 505.

Top 5 (most damaging to permute → most informative):

| Δpp | feature |
|----:|:--------|
| +0.885 | `feat_quant_survival_y_p10` |
| +0.755 | `feat_laplacian_variance_p90` |
| +0.740 | `feat_edge_slope_stdev` |
| +0.715 | `feat_noise_floor_y` |
| +0.618 | `feat_palette_density` |

Bottom 5 (cull candidates):

| Δpp | feature |
|----:|:--------|
| -0.649 | `feat_cr_sharpness` |
| -0.338 | `feat_aq_map_p90` |
| -0.325 | `feat_edge_density` |
| -0.214 | `feat_luma_kurtosis` |
| -0.213 | `feat_aq_map_p95` |

### zenwebp top-5 / bottom-5

Baseline `mean_pct = 2.49`, `argmin_acc = 0.416`, n_val = 2274.
Note: zenwebp's training already excluded the math-redundant transforms
(log_pixels, bitmap_bytes, indexed_palette_width are absent from
`feat_cols`), so the Tier 1.5 permutation top list is uncontaminated.

Top 5:

| Δpp | feature |
|----:|:--------|
| +0.470 | `feat_patch_fraction` |
| +0.373 | `feat_noise_floor_uv_p50` |
| +0.359 | `feat_aq_map_std` |
| +0.274 | `feat_noise_floor_uv` |
| +0.229 | `feat_edge_density` |

Bottom 5:

| Δpp | feature |
|----:|:--------|
| -0.155 | `feat_noise_floor_y_p50` |
| -0.147 | `feat_uniformity` |
| -0.140 | `feat_quant_survival_uv` |
| -0.094 | `feat_luma_histogram_entropy` |
| -0.060 | `feat_quant_survival_y_p50` |

### Cross-codec consistency

No feature appears in both top-5 lists, but `noise_floor_*` (Y on
zenjpeg, UV on zenwebp) and the AQ-map-derived percentiles dominate
both rankings. The zenjpeg top-5 is sharpness/noise driven (consistent
with quality decisions); the zenwebp top-5 leans on chroma noise and
patch fraction (consistent with screen-content branching). Codec-
specific pickers are doing what they should — drawing on different
signal subsets.

### Caveat: Tier 1.5 alone misses redundancy

`feat_log_pixels` showed **Δpp = +0.596** on zenjpeg permutation
*despite* being Spearman 1.0 with `feat_pixel_count`. A learner with
both inputs splits arbitrarily between them; permutation importance
will mis-attribute signal across the redundant pair. **Tier 0
correlation is required to catch this; Tier 1.5 by itself would
falsely "validate" log_pixels.** This is exactly why the cross-codec
discipline runs Tier 0 first and only uses Tier 1.5 to find
*non-redundant* low-impact features.

## Recommended next actions (queued for next breaking release)

### Cull list — 4-codec consensus, ship next break

- `feat_log_pixels` (already in queue, now confirmed)
- `feat_bitmap_bytes` (already in queue, now confirmed)

### Borderline — queue with corpus-densification gate

- `feat_indexed_palette_width`: cull *after* corpus has ≥ 100
  indexed-palette samples spanning palette widths
  {2, 4, 8, 16, 32, 64, 128, 256}.

### Don't cull yet (architecturally limited, not corpus-bound)

- `feat_alpha_present`, `feat_alpha_used_fraction`,
  `feat_alpha_bimodal_score`: surface them via the RGBA-aware
  analyzer entry point in codec wrappers, then re-run.
- `feat_hdr_present`, `feat_peak_luminance_nits`,
  `feat_p99_luminance_nits`, `feat_hdr_headroom_stops`,
  `feat_hdr_pixel_fraction`, `feat_wide_gamut_peak`,
  `feat_wide_gamut_fraction`, `feat_effective_bit_depth`,
  `feat_gamut_coverage_srgb`, `feat_gamut_coverage_p3`: replace the
  gain-map-base hdr_widegamut axis with native PQ/HLG content, then
  re-run.

### Other findings worth committing

- The zenjpeg overnight Pareto sweep produced a schema mismatch and
  needs a proper rebake (file `zenjpeg_overnight_hybrid_2026-05-02.json`
  has only 1 KB of useful results; data starvation flagged in the
  overnight log).
- zenwebp pareto-matched (n_rows = 1140) and unmatched (n_rows = 1348)
  give the same redundancy verdicts → matching is not changing the
  Tier 0 conclusions for the deferred features.

## Caveats / known gaps

- **zenjxl Phase B/C skipped** — the in-flight zenjxl agent held the
  marker, so this run only produced Tier 0 for zenjxl. Schedule a
  Tier 1.5 run for zenjxl before any breaking release that touches
  jxl-specific features.
- **zenavif Phase B/C skipped** — the picker training harness asserts
  power-of-2 dimensions in places that fire on the
  `wide_aspect_*` rows. Fix is harness-side, not feature-side. Ticket
  the harness fix before claiming zenavif validation.
- **HDR axis class is mis-corpused** — gain-map JPEGs render SDR,
  defeating the point of the axis. Corpus author needs to swap in
  native PQ/HLG content (AVIF, JXL HDR distance-encoded).
- **Alpha analyzer entry strips alpha** — codec wrappers route
  through `analyze_features_rgb8` which discards alpha by design.
  This is documented in the zenanalyze tier architecture
  (`Alpha — stride-sampled, reads source bytes directly`); the codec
  wrappers need a separate RGBA-aware entry to surface alpha
  signals.

## Artifact paths

Inputs (block storage):

- `/home/lilith/work/zen/zenjpeg--ablate-overnight-2026-05-02/benchmarks/zenjpeg_correlation_2026-05-02.json`
- `/home/lilith/work/zen/zenjpeg--ablate-overnight-2026-05-02/benchmarks/zenjpeg_correlation_2026-05-02.report.txt`
- `/home/lilith/work/zen/zenjpeg--ablate-overnight-2026-05-02/benchmarks/zenjpeg_student_perm_2026-05-02.json`
- `/home/lilith/work/zen/zenwebp--ablate-overnight-2026-05-02/benchmarks/zenwebp_correlation_2026-05-02.json`
- `/home/lilith/work/zen/zenwebp--ablate-overnight-2026-05-02/benchmarks/zenwebp_correlation_paretomatched_2026-05-02.json`
- `/home/lilith/work/zen/zenwebp--ablate-overnight-2026-05-02/benchmarks/zenwebp_student_perm_2026-05-02.json`
- `/home/lilith/work/zen/zenavif/benchmarks/zenavif_correlation_2026-05-02.json`
- `/home/lilith/work/zen/zenavif/benchmarks/zenavif_correlation_2026-05-02_t99.json`
- `/home/lilith/work/zen/zenanalyze/benchmarks/zenjxl_correlation_2026-05-02.json`
- `/home/lilith/work/zen/zenanalyze/benchmarks/zenjxl_correlation_thresh99_2026-05-02.json`
- `/mnt/v/output/codec-corpus-2026-05-01-multiaxis/manifest.tsv`

Output (this report, committed to zenanalyze):

- `/home/lilith/work/zen/zenanalyze/benchmarks/cross_codec_aggregate_2026-05-02.md`

Run config: zenanalyze HEAD `eaaa2e2` at the time of this report;
99 features in the default `FeatureSet::SUPPORTED` (with `hdr` cargo
feature enabled = 99 + 10 HDR/depth gated; without = 89). The 3
features culled at commit `e5c3c39` (uniformity_smooth,
flat_color_smooth, chroma_kurtosis) are not in any of these inputs.
