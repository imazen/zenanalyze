# Multi-seed confirm — zenjpeg_picker_config

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 203.3s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 29.5% | 25.7% | -3.81 pp | +1.02 pp |
| `0xbeef` | 36.9% | 30.1% | -6.81 pp | +2.77 pp |
| `0xface` | 30.7% | 23.9% | -6.86 pp | +2.17 pp |

## Aggregate

- baseline argmin (median): **30.7%**
- recommended argmin (median): **25.7%**
- Δ argmin (median): **-6.81 pp** (stdev 1.75 pp, range -6.86..-3.81)
- Δ mean overhead (median): **+2.17 pp** (stdev 0.89 pp)
- **verdict**: `regress`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
