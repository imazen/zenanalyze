# Per-(image, size_class) zensim ceilings — canonical zenwebp (2026-08-28)

zenanalyze#51's premise, measured on the canonical 2026-06-27 zenwebp lossy
dataset (`~/tmp/canonical/zenwebp_lossy/{train,validate,test}.parquet` →
`canonical_to_pareto.py`, which now emits `effective_max_zensim` /
`effective_max_ssim2` per `(image_path, size_class)` = max score over every
`(cell, q)` row of that key). 944,370 Pareto rows, 4,497 (image, size) pairs,
30 VP8 cells, q grid as swept. No `large` renditions exist in this dataset.

| size_class | pairs | ceiling p5 | p25 | p50 | p95 | min | < 94 | < 90 | < 85 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | 414 | 77.6 | 87.4 | 90.1 | 93.7 | 54.0 | 97.8 % | 48.6 % | 15.2 % |
| small | 1,656 | 79.1 | 87.3 | 89.9 | 93.7 | 58.4 | 97.3 % | 51.3 % | 13.9 % |
| medium | 2,427 | 78.0 | 86.8 | 89.4 | 93.4 | 62.0 | 96.5 % | 60.7 % | 18.3 % |

- 97 % of pairs cannot reach zensim 94 at any of the 30 cells × swept q; the
  median ceiling is ≈ 90. (The issue's 2026-04 probe saw ceilings near 96 at
  ≥ 96 px — a different zensim profile; the shape, not the level, is the
  point: the ceiling is content-dependent at every size here, not only tiny.)
- Against the default `ZQ_TARGETS` grid (0..70 step 5, 70..100 step 2 — 27
  targets), **28,269 of 134,910 (image, size, zq) decision cells (21.0 %) are
  physically unreachable**. Those rows never trained (a target above the
  ceiling is, by construction, one no config reaches, so `build_dataset`
  dropped them as "nothing reached"); what the column changes is the
  DIAGNOSIS. Verified by re-training the #56 rd_time bake on the regenerated
  Pareto: identical 91,413 decision rows and val mean overhead 4.609 % both
  ways, but `UNCAPPED_ZQ_GRID` no longer fires (`sweep_ceilings.n_with_ceiling
  = 4497`) and `DATA_STARVED_SIZE` can tell a sweep gap from the ceiling tail.
- The 30 `DATA_STARVED_SIZE` cells that remain are every `tiny/zq*` cell:
  `train_hybrid` drops 636 / 4,497 keys with residual NaN in KEEP features
  (tiny images skip the percentile features, zenanalyze#49; the canonical
  features TSV has no tiled re-extraction to fill them), which removes all 414
  tiny renditions from this bench. That is a features-pipeline gap, not a
  ceiling problem — worth fixing before any tiny-size conclusion is drawn
  from the canonical bench.
- `effective_max_ssim2` median: tiny 52.2, small 86.2, medium 87.7 — the ssim2
  ceiling collapses on tiny images far harder than zensim's, which is why a
  picker trained against ssim2 must read `effective_max_ssim2`, never the
  zensim column (`train_hybrid.ceiling_column_for`).

Feature-based prediction of the ceiling (work item 3, `fit_zensim_ceiling.py`):
`zensim_ceiling_fit_zenwebp_2026-08-28.md`.

Reproduce: `zentrain/tools/canonical_to_pareto.py` on the three splits, then
the numpy summary in this file's commit message / `just zentrain-pytests` for
the emission test.
