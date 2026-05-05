# zensim → butteraugli translation MLP — v06 data, 2026-05-05

Trained a small MLP (93 → 64 → 32 → 2) that maps `(image features ⊕ zensim score)` to `(log butteraugli_max, log butteraugli_pnorm3)`.

## Data
- 129,233 cells from v06 sweep (77/100 chunks done at training time)
- 385 unique images, 77 held out (seed 7)
- 92 zenanalyze features + 1 zensim score input

## Held-out results (25,872 cells)

| metric | mae | rmse | r² | spearman | pearson |
|---|---:|---:|---:|---:|---:|
| butteraugli_max | 1.67 | 23.57 | -97.1 | **0.937** | 0.18 |
| butteraugli_p3  | 0.65 | 7.96  | -58.6 | **0.938** | 0.22 |

## Baselines

| baseline | metric | spearman |
|---|---|---:|
| zensim only (linear) | butter_max | 0.826 |
| zensim only (linear) | butter_p3  | 0.874 |
| MLP (features+zensim) | butter_max | **0.937** |
| MLP (features+zensim) | butter_p3  | **0.938** |

The MLP improves rank correlation by ~0.07 vs linear-on-zensim alone.
R² is misleadingly negative because of outlier predictions on tiny
synthetic images (32px) — those skew RMSE but don't affect the
rank-order signal that runtime butter-targeting needs.

## Worst per-image rel-error

- size-dense-renders 32px renders: 60-2400% rel-err (training distribution
  rarely had tiles that small with butter measurements)
- synthetic thin-lines patterns: 258% rel-err
- Real-world content (cid22, clic) under 30% rel-err

## Implication for runtime butter-targeting

Spearman 0.94 is enough for **rank-preserving** runtime decisions. A
runtime caller measuring zensim once + running the z2b MLP can estimate
butter for picker decisions without paying the per-cell butter compute
cost (which dominates the metric pipeline at ~300 ms/cell on CPU).

Caveats:
- Train on production-content images (skip synthetic + tiny)
- Re-train per zensim version (V0_2 weights vs V0_5/V0_6 will give different
  outputs)
- For a SHIPPING runtime, pin to the safest predictions (high Spearman
  + low MAE on the actual content domain)
