# Picker v0.3 — zenjxl, 2026-05-04

## Summary

Trained a fresh hybrid-heads MLP picker for zenjxl (jxl-encoder
adapted oracle) against the existing schema-compatible Pareto sweep
+ features TSVs. Used the existing `zenjxl_picker_config` codec
module (unchanged). 16 cells (cell_id × ac_intensity ×
enhanced_clustering × gaborish × patches), 3 scalar heads.

## CRITICAL CAVEAT — metric substitution

Per `zenjxl_picker_config.py`, the `zensim` column in zenjxl's
adapted Pareto TSV is actually **ssim2 (SSIMULACRA2)** values, not
zenpipe's XYB-Butteraugli zensim. Numbers in this report under
"argmin overhead" / "achieved zensim" are ssim2-based, NOT
comparable to the zenwebp / zenavif zensim values.

## Training inputs

| Field | Value |
|-------|-------|
| Pareto TSV | `~/work/zen/zenjxl/benchmarks/zenjxl_lossy_pareto_2026-05-01.tsv` |
| Pareto rows | 610,593 (100 unique images) |
| Features TSV | `~/work/zen/zenjxl/benchmarks/zenjxl_lossy_features_2026-05-01.tsv` |
| Feature rows | 400 |
| KEEP_FEATURES | 68 |
| Cells | 16 = cell_id × ac × enhanced_clustering × gaborish × patches |
| Scalar heads | k_info_loss_mul, k_ac_quant, entropy_mul_dct8 |
| Output dim | 64 (16 cells × (1 bytes_log + 3 scalars)) |

## Architecture

MLP `138 -> 128 -> 128 -> 64`, ReLU, 42,560 params. Engineered
input: `feats[68] + size_oh[4] + poly[5] + zq*feats[68] + icc[1] =
138 floats`.

## Held-out (training-internal val split) metrics

| Metric | Teacher (boosted-trees) | Student (MLP) |
|---|---:|---:|
| Argmin mean overhead | 5.86% | 4.57% |
| Argmin accuracy | 8.9% | 11.2% |
| k_info_loss_mul RMSE | 0.1728 | 0.2380 |
| k_ac_quant RMSE | 0.0610 | 0.0689 |
| entropy_mul_dct8 RMSE | 0.0792 | 0.1169 |
| Train rows | — | 4,984 |
| Val rows | — | 1,241 |
| Train→val gap (overhead) | — | +0.81pp |

## Safety violations (carried)

- `LOW_ARGMIN`: val argmin_acc 11.2% < 30.0% threshold
- `DATA_STARVED_SIZE`: 21 (size_class, zq) cells with <50 rows
- `UNCAPPED_ZQ_GRID`: ZQ_TARGETS includes zq=94 with no
  effective_max_zensim column

Bake via `--allow-unsafe`. 16 cells × 100 unique images is heavily
data-starved at the cell level (each cell averages ~6 imgs/cell at
the unique-image level after the train/val split). Argmin acc 11.2%
is roughly 2× a uniform 16-way prior (6.25%), so the model has SOME
signal — but it's weak.

**Recommendation:** Same as zenavif — ship for external A/B archival;
do not promote to in-runtime use until the sweep expands.

## Artifacts

Local:
- `~/work/zen/zenanalyze/benchmarks/zenjxl_picker_v0.3_2026-05-04.bin`
  (53,768 bytes i8, n_inputs=138, n_outputs=64, n_layers=3,
  schema_hash=`0x5447813df14cd70d`)
- `~/work/zen/zenanalyze/benchmarks/zenjxl_picker_v0.3_2026-05-04.manifest.json`
- `~/work/zen/zenanalyze/benchmarks/zenjxl_hybrid_2026-05-01_v0.3_2026-05-04.json`

R2:
- `s3://zentrain/zenjxl/pickers/zenjxl_picker_v0.3_2026-05-04.bin`
- `s3://zentrain/zenjxl/pickers/zenjxl_picker_v0.3_2026-05-04.manifest.json`
- `s3://zentrain/zenjxl/pickers/zenjxl_hybrid_v0.3_train.json`

## Provenance

- Trainer: `~/work/zen/zenanalyze/zentrain/tools/train_hybrid.py`
- Codec config: `~/work/zen/zenanalyze/zentrain/examples/zenjxl_picker_config.py`
- Bake: `~/work/zen/zenanalyze/tools/bake_picker.py --dtype i8 --allow-unsafe`
