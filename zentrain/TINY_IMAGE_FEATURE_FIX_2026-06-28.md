# Tiny-image picker feature fix + feature-vs-size trend audit (2026-06-28)

## Problem

SDR per-codec pickers (zenjpeg / zenwebp / zenavif) could not bake legitimately —
`bake_picker.py` refused on `DATA_STARVED_SIZE` (the `(tiny, target_zq)` cells had
0 train rows). Two independent causes, both measured:

1. **Tiny renditions were dropped for NaN content features.** 13 of the 50 picker
   `KEEP_FEATURES` are percentile/windowed features that are undefined ("" / NaN)
   for an image too small to satisfy their per-feature minimum-sample/block floor
   (zenanalyze #49): `aq_map_p75/p90/p95/p99`, `noise_floor_y_p50/p90`,
   `laplacian_variance_p50/p75/p90/p99/peak`, `quant_survival_y_p10`,
   `luma_kurtosis`. `train_hybrid.load_features` dropped any (image, size) row with
   a NaN feature → the entire tiny size-class had ~no training rows → the gate
   fired. Adding more tiny renditions (the dense-small re-sweep) could not fix this
   — the added renditions were dropped the same way.

2. **`large` was demanded but absent.** The corpus is web-focused and tops out at
   medium (≤ 1 MP); `SIZE_CLASSES = [tiny, small, medium, large]` made the gate
   require `(large, *)` rows the corpus never contained.

## Fix

**(a) Content-aware recovery, now INTRINSIC in the zenanalyze extractor.**
A constant fill (0.0 / feature-min) clears the gate but makes every too-small image
identical in those 13 features → a degenerate tiny picker that can't distinguish
content (gate-gaming). The correct fix lives in the extractor itself, so EVERY
caller (training + every codec's inference path + any future consumer) gets
content-aware features at any size with **zero external handling** (user directive
2026-06-28). `analyze_features` (`src/lib.rs`) now, for an input below the
percentile floors in either axis, mirror-tiles the source up to ≥ **128 px**
(alternating H/V flips — seamless; a plain repeat would inject false edges and
inflate `laplacian_variance`), re-extracts, and fills ONLY the would-be-NaN
features from the tiled pass. Valid native features are kept unchanged
(native-primary). Each too-small image gets its OWN content-derived percentile
values.

Two silent-bad-value sources had to be fixed first, so the would-be-NaN features
actually surface as NaN (→ `None` → recoverable) instead of a stale `0.0`:
- **tier3** (`src/tier3.rs`): `dct_stats`'s early returns for `width < 8` /
  `total_blocks == 0` now emit `NaN` (not `0.0`) for the percentile fields, and
  `src/lib.rs` no longer gates `populate_tier3` on `width/height >= 8` (so the
  zero-block path runs and produces those NaNs; the per-pixel luma histogram, which
  needs no 8×8 block, now also runs sub-8 — its entropy is a real finite value).
- **tier1** (`src/tier1.rs`): the laplacian-percentile / kurtosis floor now also
  checks the **geometric** interior count `(w-2)*(h-2)` (the SIMD pass over-counts
  padded stripe lanes at tiny widths, inflating the histogram total past the
  `total < 1024` floor), and the `n < 1` bail (image shorter than one stripe, e.g.
  4 rows) NaNs the laplacian percentiles before returning.

- **Min tile dim = 128**, chosen by measurement: mirror-tiling to **96 px already
  recovers ALL 97 extracted features** (64 px leaves 30 NaN). 128 is a safe margin.
- Canonical `mirror_tile` spec now lives in `mirror_tile_packed` (`src/lib.rs`),
  generalized over bytes-per-pixel so a u16 source recovers exactly like its u8
  promotion (the `u16 ≡ u8` invariant holds at tiny sizes). The interim Python
  `tile_fill_tiny_features.py` is **RETIRED** (its algorithm is preserved verbatim
  in `mirror_tile_packed`'s doc comment + git history).
- Tested in `src/tests.rs::sample_count_floor`: recovery byte-matches an explicit
  mirror-tile reference across 4×4 … 127×10 (incl. the extreme 2×32 / 64×4 aspect
  ratios); the `versioning_golden.tsv` golden was re-blessed to capture the
  recovered tiny-rendition values. `spectral_slope_y` is the only feature tiling
  does not recover — NOT a picker feature, so it is reported, never filled.

**(b) Size-grid scoped to corpus-present sizes.** `train_hybrid` now derives
`SIZE_CLASSES` from the sizes actually present in the pareto (here
`[tiny, small, medium]`); `large` is excluded so the gate enforces coverage only
for sizes the sweep produced. Override with `PICKER_SIZE_CLASSES=...`.

## Feature-vs-size trend audit (catch SILENT non-NaN bad values)

NaN is the obvious failure (dropped). A windowed/normalized feature returning a
WRONG-but-finite value at small size would silently corrupt training and trip no
NaN check. Audited via **10 k-means-representative images** (feature-space
centroids, min-dim ≥ 512) × **crop sweep** (center-crops at native scale →
area-dependence) × **resize sweep** (Lanczos downscale → scale-dependence) × all 97
extracted features, vs the 512 px value as ground truth. Curves:
`/mnt/v/output/picker-feature-size-audit-2026-06-28/{curves_raw.csv,
feature_size_deviation.csv, keep_feature_audit.csv}`.

Findings:
- **No picker KEEP feature has a sentinel/overflow value at small native size**
  (`|value| > 1e5`: NONE). The one genuine sentinel artifact, `palette_fits_in_256`
  (= 1e6 at small size), is NOT in any picker KEEP set — no action.
- The KEEP features with high crop-deviation (`*_sharpness`, `laplacian_variance`,
  `colourfulness`, `chroma_complexity`, ...) reflect **legitimate content variation**
  — a center-crop shows different content — not a size artifact. They are computed
  at native size on the actual tiny content, so native-primary uses them correctly.
- **Tiling validation:** the 13 KEEP NaN-features are all recovered by mirror-tiling
  to ≥ 128 px (none stay NaN). `spectral_slope_y` is the only feature tiling does
  not recover (not a picker feature → reported, not used).

## Train/inference consistency (REQUIRED when a bin is wired)

The fix is a feature-extraction step, so the picker RUNTIME must apply the same
tile-before-extract to too-small images. Today none of the new bins are wired into
a codec runtime (they ship as `benchmarks/` artifacts; only zenwebp wires an older
bin), so this is a forward requirement, specified here:

- `extract_raw_features_rgb8` (each codec): if `analyze_features_rgb8` returns a
  missing/NaN feature, mirror-tile the RGB to ≥ 128 px via the canonical
  `mirror_tile` (byte-identical to `tile_fill_tiny_features.py::mirror_tile` —
  integer flips, deterministic) and re-extract; native-primary fill. Do NOT
  `unwrap_or(0.0)` the percentile features.
- `engineered_features`: build the `size_class` one-hot of length
  `len(model.size_classes)` (now 3: tiny/small/medium), not a hardcoded 4; map
  images larger than the modeled range to the last class.

## Bake results (clean, no --allow-unsafe)

<!-- appended after the clean_bake run; TEST argmin/top-K + committed .bin SHAs -->
