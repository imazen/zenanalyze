# train_hybrid student trainers — multi-seed comparison (zenanalyze#68)

Date: 2026-08-28 · codec config `canonical_picker_config` · hidden `128,128,128` · seeds ['0xcafe', '0xbeef', '0xface'] · arms ['relu', 'leakyrelu', 'leakyrelu+wd=1e-4']

## Methodology

Every arm runs the full production `train_hybrid.py` pipeline (HistGradientBoosting teacher per cell → MLP student on the teacher's soft targets, per-head normalisation on) on the same Pareto + features files, the same canonical origin even/odd split (`origin_split.py`) and the same hidden shape; only the student trainer differs. Metrics are the student's val argmin diagnostics vs the reachable per-row optimum, `mean ± stdev` over the seeds.

| arm | backend | init | L2 |
|---|---|---|---|
| `relu` | sklearn | glorot_uniform | mlpregressor_alpha = 0.0001 |
| `leakyrelu` | torch | kaiming_uniform | adam_weight_decay = 0.0 |
| `leakyrelu+wd=1e-4` | torch | kaiming_uniform | adam_weight_decay = 0.0001 |

## Headline numbers (val, mean ± stdev over seeds)

| metric | `relu` | `leakyrelu` | `leakyrelu+wd=1e-4` |
|---|---|---|---|
| mean overhead | 4.82 ± 0.38 % | 4.71 ± 0.14 % | 4.58 ± 0.17 % |
| p50 overhead | 1.05 ± 0.30 % | 1.10 ± 0.14 % | 0.93 ± 0.21 % |
| p90 overhead | 10.86 ± 1.46 % | 9.58 ± 0.13 % | 10.11 ± 0.98 % |
| p95 overhead | 28.81 ± 0.41 % | 28.47 ± 0.03 % | 28.64 ± 0.35 % |
| p99 overhead | 58.62 ± 1.49 % | 58.40 ± 1.39 % | 58.62 ± 0.66 % |
| max overhead | 296.16 ± 36.73 % | 281.99 ± 31.15 % | 270.56 ± 1.12 % |
| argmin accuracy | 11.2 ± 1.2 pp | 9.1 ± 0.6 pp | 12.3 ± 1.0 pp |

## Per-seed values (val)

| arm | seed | mean | p50 | p90 | p95 | p99 | max | argmin_acc | model |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `relu` | 0xcafe | 4.77% | 1.02% | 9.86% | 28.68% | 58.41% | 253.75% | 11.4% | `zenwebp_canonical_hybrid_relu_seed_cafe.json` |
| `relu` | 0xbeef | 5.22% | 1.36% | 12.54% | 29.27% | 60.20% | 317.37% | 9.9% | `zenwebp_canonical_hybrid_relu_seed_beef.json` |
| `relu` | 0xface | 4.46% | 0.77% | 10.19% | 28.49% | 57.24% | 317.37% | 12.3% | `zenwebp_canonical_hybrid_relu_seed_face.json` |
| `leakyrelu` | 0xcafe | 4.57% | 0.94% | 9.59% | 28.49% | 59.94% | 271.21% | 8.4% | `zenwebp_canonical_hybrid_leakyrelu_seed_cafe.json` |
| `leakyrelu` | 0xbeef | 4.84% | 1.21% | 9.70% | 28.43% | 58.03% | 257.66% | 9.6% | `zenwebp_canonical_hybrid_leakyrelu_seed_beef.json` |
| `leakyrelu` | 0xface | 4.73% | 1.15% | 9.43% | 28.48% | 57.23% | 317.10% | 9.3% | `zenwebp_canonical_hybrid_leakyrelu_seed_face.json` |
| `leakyrelu+wd=1e-4` | 0xcafe | 4.40% | 0.69% | 10.19% | 28.89% | 59.13% | 271.21% | 13.1% | `zenwebp_canonical_hybrid_leakyrelu_wd1e-4_seed_cafe.json` |
| `leakyrelu+wd=1e-4` | 0xbeef | 4.74% | 1.03% | 11.04% | 28.81% | 58.85% | 269.28% | 11.2% | `zenwebp_canonical_hybrid_leakyrelu_wd1e-4_seed_beef.json` |
| `leakyrelu+wd=1e-4` | 0xface | 4.60% | 1.07% | 9.09% | 28.24% | 57.88% | 271.21% | 12.5% | `zenwebp_canonical_hybrid_leakyrelu_wd1e-4_seed_face.json` |

## Per-row overhead TSVs

Columns `image, size_class, zq, pick, actual_best, overhead`; one per (arm, seed) under `/Users/lilith/tmp/backend-gap-zenwebp` (not committed).

