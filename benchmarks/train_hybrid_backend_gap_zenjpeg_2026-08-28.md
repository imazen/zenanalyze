# train_hybrid student trainers — multi-seed comparison (zenanalyze#68)

Date: 2026-08-28 · codec config `canonical_picker_config` · hidden `128,128,128` · seeds ['0xcafe'] · arms ['relu', 'leakyrelu', 'leakyrelu+wd=1e-4']

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
| mean overhead | 12.04 ± 0.00 % | 12.71 ± 0.00 % | 12.37 ± 0.00 % |
| p50 overhead | 6.29 ± 0.00 % | 7.03 ± 0.00 % | 6.70 ± 0.00 % |
| p90 overhead | 32.57 ± 0.00 % | 33.92 ± 0.00 % | 33.36 ± 0.00 % |
| p95 overhead | 42.65 ± 0.00 % | 44.17 ± 0.00 % | 43.50 ± 0.00 % |
| p99 overhead | 68.74 ± 0.00 % | 69.05 ± 0.00 % | 68.66 ± 0.00 % |
| max overhead | 262.35 ± 0.00 % | 284.26 ± 0.00 % | 284.26 ± 0.00 % |
| argmin accuracy | 24.6 ± 0.0 pp | 21.7 ± 0.0 pp | 25.2 ± 0.0 pp |

## Per-seed values (val)

| arm | seed | mean | p50 | p90 | p95 | p99 | max | argmin_acc | model |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `relu` | 0xcafe | 12.04% | 6.29% | 32.57% | 42.65% | 68.74% | 262.35% | 24.6% | `zenjpeg_canonical_hybrid_relu_seed_cafe.json` |
| `leakyrelu` | 0xcafe | 12.71% | 7.03% | 33.92% | 44.17% | 69.05% | 284.26% | 21.7% | `zenjpeg_canonical_hybrid_leakyrelu_seed_cafe.json` |
| `leakyrelu+wd=1e-4` | 0xcafe | 12.37% | 6.70% | 33.36% | 43.50% | 68.66% | 284.26% | 25.2% | `zenjpeg_canonical_hybrid_leakyrelu_wd1e-4_seed_cafe.json` |

## Per-row overhead TSVs

Columns `image, size_class, zq, pick, actual_best, overhead`; one per (arm, seed) under `/Users/lilith/tmp/backend-gap-zenjpeg` (not committed).

