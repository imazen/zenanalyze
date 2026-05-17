# LOO warm-start benchmark — 2026-05-17

Synthetic-data smoke test for `_picker_lib`'s new
`fit_distilled_teacher` / `fit_residual_teacher` /
`train_loo_teachers_distilled_parallel` helpers.

## Setup

- Rows: 4500 (train 3600 / val 900)
- Features: 80 (30 latent + 50 noise)
- Cells: 12
- LOO features evaluated: 10 (6 latent + 4 noise)
- Retrain presets: `HISTGB_FULL` (max_iter=400, max_depth=8) and `HISTGB_FAST` (max_iter=100, max_depth=4 — what `feature_ablation.py` actually uses today).
- Distilled presets: `HISTGB_FAST` and `HISTGB_MEDIUM` (max_iter=200, max_depth=6).
- Hardware: 16-core box, `n_jobs=-1`

Full-schema baseline argmin_acc: **0.8433** (mean overhead 11.20%); full teachers fit in 8.13s.

## Per-feature results

| feature | retrain-FULL t/acc | retrain-FAST t/acc | distill-FAST t/acc | distill-MED t/acc | residual t/acc |
|---|--:|--:|--:|--:|--:|
| `feat_lat_7` | 4.01s / 0.8322 | 1.30s / 0.8222 | 0.92s / 0.8111 | 1.53s / 0.8311 | 5.47s / 0.8344 |
| `feat_lat_9` | 2.89s / 0.7733 | 0.40s / 0.7622 | 0.77s / 0.7511 | 1.52s / 0.7522 | 2.16s / 0.7800 |
| `feat_lat_17` | 2.88s / 0.7900 | 0.42s / 0.7833 | 0.77s / 0.7844 | 1.55s / 0.7778 | 2.47s / 0.8111 |
| `feat_lat_18` | 3.04s / 0.7689 | 0.41s / 0.7744 | 0.80s / 0.7656 | 1.51s / 0.7567 | 2.25s / 0.7867 |
| `feat_lat_20` | 2.83s / 0.7967 | 0.40s / 0.7933 | 0.80s / 0.7811 | 1.59s / 0.7822 | 2.22s / 0.8133 |
| `feat_lat_23` | 3.38s / 0.8089 | 0.42s / 0.7878 | 0.86s / 0.7878 | 1.51s / 0.7811 | 2.28s / 0.8011 |
| `feat_noise_16` | 3.14s / 0.8422 | 0.41s / 0.8278 | 0.81s / 0.8089 | 1.79s / 0.8311 | 2.41s / 0.8400 |
| `feat_noise_26` | 2.95s / 0.8411 | 0.42s / 0.8367 | 0.87s / 0.8133 | 1.53s / 0.8378 | 2.30s / 0.8411 |
| `feat_noise_39` | 2.80s / 0.8411 | 0.42s / 0.8378 | 0.74s / 0.8200 | 1.55s / 0.8367 | 2.18s / 0.8444 |
| `feat_noise_49` | 3.70s / 0.8467 | 0.47s / 0.8389 | 0.89s / 0.8144 | 1.39s / 0.8356 | 2.69s / 0.8411 |

## Aggregate (vs HISTGB_FULL retrain — comparable to production teachers)

| strategy | mean wall (s) | std (s) | mean acc | speedup | acc loss (pp) |
|---|--:|--:|--:|--:|--:|
| retrain (HISTGB_FULL) | 3.16 | 0.39 | 0.8141 | 1.00× | 0.00 |
| retrain (HISTGB_FAST) | 0.51 | 0.26 | 0.8064 | 6.24× | +0.77 |
| distilled (HISTGB_FAST) | 0.82 | 0.06 | 0.7938 | 3.84× | +2.03 |
| distilled (MEDIUM) | 1.55 | 0.09 | 0.8022 | 2.05× | +1.19 |
| residual (HISTGB_FAST) | 2.64 | 0.96 | 0.8193 | 1.20× | -0.52 |

## Aggregate (vs HISTGB_FAST retrain — apples-to-apples for feature_ablation.py)

`feature_ablation.py` already uses max_iter=100, max_depth=4 for its retrain LOO, so this is the realistic comparison for any wiring change.

| strategy | speedup | acc loss vs FAST retrain (pp) |
|---|--:|--:|
| distilled (HISTGB_FAST) | 0.62× | +1.27 |
| distilled (MEDIUM) | 0.33× | +0.42 |
| residual (HISTGB_FAST) | 0.19× | -1.29 |

## Verdict

Compared to HISTGB_FULL retrain (the production target):

- HOLD (distilled HISTGB_FAST): 3.84× speedup but +2.03pp acc loss > 1.0pp budget.
- DROP (distilled MEDIUM): both speed (2.05×) and acc (+1.19pp) miss targets.
- HOLD (residual HISTGB_FAST): acc OK (-0.52pp) but only 1.20× speedup < 3× target.

Compared to HISTGB_FAST retrain (what feature_ablation.py uses today):

- DROP (distilled HISTGB_FAST): both speed (0.62×) and acc (+1.27pp) miss targets.
- HOLD (distilled MEDIUM): acc OK (+0.42pp) but only 0.33× speedup < 3× target.
- HOLD (residual HISTGB_FAST): acc OK (-1.29pp) but only 0.19× speedup < 3× target.

## Conclusion — negative result; warm-start does NOT replace LOO retrain

**The warm-start / distillation hypothesis fails the wall-time gate at the
preset `feature_ablation.py` already uses.**

Two ways to read the table; both arrive at the same conclusion.

**1. Why `HISTGB_FAST` retrain is the right baseline.** `feature_ablation.py`
declared its own `HISTGB_KW` (max_iter=100, max_depth=4) — i.e., it
already uses `HISTGB_FAST` for LOO retrain. The docstring says the
quality drop is ~1-2pp vs the production fit but the *ranking*
across feature ablations is stable. So the realistic question is
"can warm-start beat `HISTGB_FAST` retrain at LOO time?" — and the
answer (row 50-52 above) is **no**. The cheapest distilled variant
runs 0.62× as fast (i.e., 60% slower) and loses 1.27pp accuracy.

**2. Why this isn't a tuning issue.** The distillation cost is
dominated by re-fitting a HistGB on `(X_loo, soft_labels)`. The soft
label benefit (smoother target → fewer iterations needed) only pays
off if you started from a *very* expensive `HISTGB_FULL` retrain and
want to substitute a `HISTGB_FAST`-ish model. But the existing tool
already accepts that tradeoff directly: row 39 shows `HISTGB_FAST`
retrain gives **6.24× speedup over `HISTGB_FULL` at only +0.77pp
accuracy loss** — a strictly better tradeoff than any warm-start
strategy measured here.

In other words: if you have HISTGB_FULL teachers and want to LOO
cheaply, *just use HISTGB_FAST retrain*. Distillation is dominated.

**3. What residual fitting buys you (and doesn't).** The residual
path delivered the highest accuracy of any LOO strategy (-0.52pp vs
HISTGB_FULL retrain — i.e., it actually beat retrain on average,
because the imputed full-teacher prediction is already a strong
prior on val). But its wall is 2.64s — only 1.20× the HISTGB_FULL
retrain — because the dominant cost is the `full_teacher.predict()`
call plus a HISTGB_FAST fit on residuals, and we can't avoid the
prior call without losing the whole reason this works.

If a future picker pipeline ends up with very expensive teachers
(e.g., 1000 iterations on millions of rows), `fit_residual_teacher`
becomes attractive again because the relative cost of the prior-
predict step drops. The helpers stay in `_picker_lib.py` for that
case.

## Decision

- **Helpers KEPT** in `_picker_lib.py` (`fit_distilled_teacher`,
  `fit_residual_teacher`, `fit_warm_continue_teacher`,
  `train_loo_teachers_distilled_parallel`,
  `_ResidualTeacher`) — they're additive and small.
- **`feature_ablation.py` wiring SHIPPED but off by default** behind
  `--use-warm-start={off,distilled,residual}`. The default stays at
  the existing `HISTGB_FAST` retrain path. Anyone who finds a regime
  where warm-start wins on real codec data can flip the flag without
  a code change; the existing benchmark is the apples-to-apples
  baseline they must beat.
- **Negative finding committed** so the next session doesn't redo
  this investigation. Rerunning the bench with bigger row counts or
  expensive teacher presets is the right way to relitigate; this
  table is the current state.

## Reproduction

```
python3 benchmarks/loo_warmstart_bench_2026-05-17.py
```

Raw per-feature data: `loo_warmstart_bench_2026-05-17.tsv`.

(Synthetic data — 4500 rows × 80 features × 12 cells. Real codec
sweeps with 50+ features and 36 cells may shift the numbers, but
the structural argument above — that `feature_ablation.py` already
runs at `HISTGB_FAST` — holds independently of scale.)
