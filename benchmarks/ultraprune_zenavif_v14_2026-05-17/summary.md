# Multi-seed confirm — zenavif_picker_config_ultraprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 5/5 ran successfully. Wall: 58.2s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 16.7% | 12.3% | -4.41 pp | +0.52 pp |
| `0xbeef` | 9.5% | 20.2% | +10.68 pp | +1.33 pp |
| `0xface` | 17.3% | 16.7% | -0.67 pp | +0.07 pp |
| `0xdead` | 12.2% | 21.6% | +9.33 pp | -1.03 pp |
| `0xbabe` | 15.7% | 27.2% | +11.47 pp | -1.99 pp |

## Aggregate

- baseline argmin (median): **15.7%**
- recommended argmin (median): **20.2%**
- Δ argmin (median): **+9.33 pp** (stdev 7.30 pp, range -4.41..+11.47)
- Δ mean overhead (median): **+0.07 pp** (stdev 1.31 pp)
- **verdict**: `ship`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
