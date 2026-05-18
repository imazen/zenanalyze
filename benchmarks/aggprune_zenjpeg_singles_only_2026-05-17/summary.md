# Multi-seed confirm — zenjpeg_picker_config_aggprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 5/5 ran successfully. Wall: 215.7s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 27.5% | 35.4% | +7.89 pp | -0.33 pp |
| `0xbeef` | 33.2% | 36.0% | +2.72 pp | -1.16 pp |
| `0xface` | 24.7% | 23.3% | -1.40 pp | +3.77 pp |
| `0xdead` | 33.4% | 29.6% | -3.81 pp | +3.81 pp |
| `0xbabe` | 28.9% | 27.7% | -1.16 pp | +1.90 pp |

## Aggregate

- baseline argmin (median): **28.9%**
- recommended argmin (median): **29.6%**
- Δ argmin (median): **-1.16 pp** (stdev 4.58 pp, range -3.81..+7.89)
- Δ mean overhead (median): **+1.90 pp** (stdev 2.29 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
