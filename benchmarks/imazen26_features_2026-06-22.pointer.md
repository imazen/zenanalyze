# imazen-26 full-feature re-extraction (2026-06-22) — data pointer

Re-extraction of the imazen-26 SDR feature dataset on the current `main`, after
this cycle's feature-determinism work changed several feature *values*. Supersedes
[`imazen26_features_2026-06-13.pointer.md`](imazen26_features_2026-06-13.pointer.md)
as the canonical full-feature artifact. Large data lives in block storage + Tower,
NOT git.

## Artifact

| | |
|---|---|
| **Parquet (canonical)** | `/mnt/v/output/imazen-26-features/imazen26_features_2026-06-22.parquet` |
| sha256 | `8cc4040b8bdd2d6cbadb77a24499149640531021cd2bceea1d0301aa7d04cf8b` |
| size / shape | 60 MB · **246,819 rows × 119 cols** (9 meta + 110 `feat_*`) — same shape as 2026-06-13 |
| Raw TSV | `…/imazen26_features_2026-06-22.tsv` (268 MB, kept alongside) |
| Manifest | `…/imazen26_manifest.tsv` (unchanged — same 2157 images) |
| **Tower mirror** | `/mnt/tower/output/imazen-26-features/imazen26_features_2026-06-22.parquet` (sha256-verified ✓) |

## Provenance

- **build_commit:** `6cb86df9` (`perf,test(zenanalyze): fixed-order f64 reduction`)
  — the tip of the feature-determinism cycle. Extractor unchanged
  (`examples/extract_features_imazen26_crops.rs`), `--features experimental,hdr`
  → `FeatureSet::SUPPORTED` (110 features). `std::thread::scope`, 24 threads,
  **0 decode failures**, 826 s, peak RSS 4.9 GiB.
- **Corpus / crops / resize grid / schema:** identical to 2026-06-13 — see that
  pointer. Same 2157 images, same 11 crop labels, same downscale-only resize grid.

## What changed vs 2026-06-13 (validated)

`benchmarks/validate_reextract.py` (aligned by content key) confirms **exactly the
7 features this cycle touched changed; the other 103 are byte-identical**, no
accidental drift:

| feature | rows changed | max rel-dev | cause |
|---|--:|--:|---|
| `edge_slope_stdev` | 92.6 % | 100 %* | `rsqrt_approx` → deterministic `rsqrt_stable` (be389a1) |
| `variance` | 48.8 % | 0.43 % | fixed-order f64 reduction (`fixed_reduce8`, 6cb86df9) |
| `chroma_luma_covariance_cb` | 10.3 % | 100 %* | `fixed_reduce8` (guard no longer flips per tier) |
| `chroma_luma_covariance_cr` | 10.1 % | 100 %* | `fixed_reduce8` |
| `spectral_slope_y` | 1.0 % | <0.01 % | `Σ ln` → `ln(Π)` product-then-ln (378fdbe) |
| `chroma_complexity` | 0.1 % | <0.01 % | derived from the changed chroma reductions |
| `colourfulness` | 0.03 % | <0.01 % | `fixed_reduce8` on the rg/yb accumulators |

\* 100 % max rel-dev = the cancellation-prone `0.0`-vs-nonzero cases (the rel
denominator floors at 1.0), not a full-scale swing — typical magnitudes are tiny.

**HDR features unchanged on SDR** (`peak_luminance_nits` / `hdr_headroom_stops` /
`wide_gamut_peak` — the depth tier fast-paths on 8-bit SDR), and the **tier3
reduction features unchanged** (`fixed_reduce8` is tier1-only) — both as expected.

**Downstream impact:** any model trained on the 2026-06-13 features for the 7
changed columns (especially `edge_slope_stdev`) carries a stale feature version —
retrain against this artifact. Their `feature_version_hash`es moved accordingly.

## Regenerate

```bash
cargo build --release --features experimental,hdr --example extract_features_imazen26_crops
target/release/examples/extract_features_imazen26_crops \
  --manifest /mnt/v/output/imazen-26-features/imazen26_manifest.tsv \
  --output   /mnt/v/output/imazen-26-features/imazen26_features_<DATE>.tsv
python3 benchmarks/tsv_to_parquet.py --keep-tsv <…>.tsv
python3 benchmarks/validate_reextract.py <old>.parquet <new>.parquet --expect <changed feats>
```
