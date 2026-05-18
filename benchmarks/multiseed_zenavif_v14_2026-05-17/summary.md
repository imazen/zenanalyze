# Multi-seed confirm — zenavif_picker_config

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 82.9s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 19.2% | 20.8% | +1.65 pp | +0.54 pp |
| `0xbeef` | 30.7% | 29.6% | -1.14 pp | -1.20 pp |
| `0xface` | 14.3% | 23.7% | +9.43 pp | -2.72 pp |

## Aggregate

- baseline argmin (median): **19.2%**
- recommended argmin (median): **23.7%**
- Δ argmin (median): **+1.65 pp** (stdev 5.48 pp, range -1.14..+9.43)
- Δ mean overhead (median): **-1.20 pp** (stdev 1.63 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
