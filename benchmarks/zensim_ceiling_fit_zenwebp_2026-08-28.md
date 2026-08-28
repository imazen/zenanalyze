# Feature → achievable-zensim-ceiling fit (zenanalyze#51)

Date 2026-08-28 · pareto `/Users/lilith/tmp/canonical/zenwebp_lossy/derived/pareto.parquet` · features `/Users/lilith/tmp/canonical/zenwebp_lossy/derived/features.tsv` (97 feat_ + 5 axes) · rows 4497 (train 2307 / val 1382 / test 808, origin even/odd split) · seed 51966

`over_m` = fraction of rows whose PREDICTED ceiling exceeds the real one by more than m points — the codec-side risk of attempting an unreachable target when it subtracts margin m.

## teacher (HistGradientBoosting)

| split | size_class | n | R² | MAE | p90 abs | max abs | over_0 | over_2 | over_5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | all | 2307 | 0.997 | 0.17 | 0.37 | 4.85 | 48.9 % | 0.0 % | 0.0 % |
| train | tiny | 212 | 0.993 | 0.26 | 0.54 | 4.85 | 47.2 % | 0.5 % | 0.0 % |
| train | small | 848 | 0.997 | 0.18 | 0.38 | 1.29 | 48.6 % | 0.0 % | 0.0 % |
| train | medium | 1247 | 0.998 | 0.15 | 0.32 | 2.16 | 49.3 % | 0.0 % | 0.0 % |
| val | all | 1382 | 0.891 | 0.91 | 2.12 | 11.50 | 43.7 % | 6.4 % | 0.9 % |
| val | tiny | 128 | 0.879 | 1.02 | 2.35 | 6.77 | 46.1 % | 9.4 % | 0.0 % |
| val | small | 512 | 0.886 | 0.90 | 2.26 | 6.78 | 42.2 % | 6.2 % | 0.6 % |
| val | medium | 742 | 0.894 | 0.90 | 1.99 | 11.50 | 44.3 % | 5.9 % | 1.3 % |
| test | all | 808 | 0.853 | 0.99 | 2.06 | 17.62 | 48.3 % | 7.8 % | 2.2 % |
| test | tiny | 74 | 0.718 | 1.29 | 2.48 | 17.62 | 58.1 % | 10.8 % | 4.1 % |
| test | small | 296 | 0.863 | 0.93 | 2.05 | 12.85 | 48.0 % | 7.8 % | 1.4 % |
| test | medium | 438 | 0.869 | 0.98 | 1.99 | 13.72 | 46.8 % | 7.3 % | 2.5 % |

## student (MLP student 64,64)

| split | size_class | n | R² | MAE | p90 abs | max abs | over_0 | over_2 | over_5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | all | 2307 | 0.953 | 0.70 | 1.42 | 18.09 | 53.1 % | 1.3 % | 0.1 % |
| train | tiny | 212 | 0.894 | 1.03 | 1.95 | 18.09 | 35.8 % | 2.4 % | 0.5 % |
| train | small | 848 | 0.948 | 0.72 | 1.45 | 15.85 | 46.3 % | 1.5 % | 0.1 % |
| train | medium | 1247 | 0.969 | 0.62 | 1.29 | 5.13 | 60.5 % | 1.0 % | 0.0 % |
| val | all | 1382 | 0.628 | 1.96 | 4.18 | 23.74 | 56.9 % | 22.2 % | 4.6 % |
| val | tiny | 128 | 0.714 | 1.75 | 3.92 | 7.24 | 49.2 % | 15.6 % | 2.3 % |
| val | small | 512 | 0.599 | 1.95 | 4.32 | 11.30 | 58.2 % | 23.0 % | 4.5 % |
| val | medium | 742 | 0.626 | 2.01 | 4.15 | 23.74 | 57.4 % | 22.8 % | 5.0 % |
| test | all | 808 | 0.298 | 2.19 | 4.36 | 75.21 | 52.8 % | 23.3 % | 4.7 % |
| test | tiny | 74 | -2.457 | 3.17 | 4.64 | 75.21 | 47.3 % | 14.9 % | 8.1 % |
| test | small | 296 | 0.347 | 2.25 | 4.91 | 27.38 | 51.0 % | 23.6 % | 5.4 % |
| test | medium | 438 | 0.709 | 1.98 | 3.87 | 20.17 | 55.0 % | 24.4 % | 3.7 % |

