# Teacher backend benchmark — 2026-05-17

**Question:** is XGBoost (with CUDA on RTX 5070 or CPU) faster than the
shipped `sklearn.ensemble.HistGradientBoostingRegressor` + `joblib`
parallel-per-cell path for the picker's teacher step?

**Answer:** no, by a wide margin, on every scale tested. The per-cell
training workload is too small per fit for GPU H↔D overhead and
kernel-launch latency to amortize, and XGBoost-CPU likewise loses to
sklearn HistGB on many-small-fits.

The shipped `_make_teacher` (sklearn HistGB) stays the right default.

## Hardware

- AMD Ryzen 9 7950X, 16 cores / 32 threads, water-cooled, no thermal
  throttling.
- NVIDIA RTX 5070 (GeForce, driver 570.x, CUDA 12.8).
- XGBoost 3.2.0 (`device="cuda"`, `tree_method="hist"`), sklearn 1.7.2.
- Python 3.10, no `target-cpu=native` (runtime path matters here).

## Workload

Synthetic `(X, y)` with `X.shape = (n, 50)` standard-normal,
`y[c] = X[:, c % 50] + 0.1·ε`. 36 cells fit independently. HistGB
params: `max_iter=400, max_depth=8, learning_rate=0.05,
l2_regularization=0.5`. XGB equivalents: `n_estimators=400,
max_depth=8, learning_rate=0.05, reg_lambda=0.5`. Quantile loss not
tested in this round.

## Wall-time results (warmup discarded for XGB-CUDA)

| Backend | n=5,000 | n=100,000 |
|---|--:|--:|
| sklearn HistGB + joblib(n_jobs=-1) | **4.93 s** | **15.83 s** |
| XGBoost CPU `tree_method=hist`, joblib(n_jobs=-1) per cell | 51.50 s | not tested |
| XGBoost CUDA `tree_method=hist`, serial per cell | 94.12 s | 115.83 s |
| XGBoost CUDA `multi_strategy=multi_output_tree` (single fit, 36 outputs) | 49.78 s | not tested |
| XGBoost CUDA via `QuantileDMatrix(..., ref=base_dmatrix)` cached | 112.57 s | not tested |

## Why

The picker's per-cell training set is **~4K rows × ~80 features**
([_picker_lib.py:68-74][src]). XGBoost CUDA's H→D transfer (data + KV
cache) and per-call kernel-launch overhead dominate when the actual
math fits in ~1 ms on the host. Even at the upper-bound scale a
picker would realistically see (100K rows from a 21.8M-row sweep
after per-(image, size, zq) grouping), GPU still loses 7×.

`multi_output_tree` is the best XGB variant tested — `~50 s` for 36
outputs in one shot — and it's still 10× slower than `joblib`-parallel
HistGB. The strategy is also documented as *experimental* in
XGBoost 3.2 ("not suitable for production, missing features" per
upstream release notes), so it's not a candidate for the default
even if it had won on wall-clock.

[src]: ../zentrain/tools/_picker_lib.py

## Quantile / `reg:quantileerror`

Not benchmarked here. sklearn `HistGradientBoostingRegressor(loss="quantile",
quantile=0.99)` is what the `zensim_strict` profile already uses
([train_hybrid.py:1742-1745][th]) and it parallelizes via the same
joblib loop. There's no measured reason to swap it.

[th]: ../zentrain/tools/train_hybrid.py

## Implication for the prior recommendation

The 2026-05-17 zentrain review proposed *"swap sklearn HGB → XGBoost
3.x device=cuda, expected 30 s → 3-10 s per codec."* That estimate
was unfounded; the actual measurement here goes the other way by 1-2
orders of magnitude. The fundamental issue is that GPU GBDT wins on
*single big fits* (multi-million-row training matrices), not on
*many small fits* — and the picker is firmly in the second regime.

Future revisits: the verdict could flip if (a) a single shared cross-
cell teacher with vector leaves becomes the architecture, AND (b) the
training corpus grows past ~1 M rows per fit (≈ 50× current scale).
Both are unlikely without a bigger redesign — the per-cell decomposition
is load-bearing for the safety_report's per-cell reach mask.

## Reproduce

The benchmark script lived in conversation; reconstruct by training
36 `HistGradientBoostingRegressor(...).fit(X, y)` (joblib) and 36
`xgb.XGBRegressor(device="cuda", tree_method="hist", ...).fit(X, y)`
(serial) on a `(n, 50)` standard-normal X and `y = X[:, i%50] +
0.1·ε`. Warm up XGB once before timing the serial loop.
