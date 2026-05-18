# Multi-seed confirm — zenjpeg_picker_config_ultraprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 5/5 ran successfully. Wall: 219.1s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 35.4% | 31.0% | -4.48 pp | +0.77 pp |
| `0xbeef` | 34.4% | 35.4% | +1.03 pp | +0.02 pp |
| `0xface` | 36.6% | 33.3% | -3.24 pp | +2.09 pp |
| `0xdead` | 31.6% | 33.8% | +2.21 pp | +1.27 pp |
| `0xbabe` | 34.2% | 37.6% | +3.33 pp | -0.69 pp |

## Aggregate

- baseline argmin (median): **34.4%**
- recommended argmin (median): **33.8%**
- Δ argmin (median): **+1.03 pp** (stdev 3.44 pp, range -4.48..+3.33)
- Δ mean overhead (median): **+0.77 pp** (stdev 1.08 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
