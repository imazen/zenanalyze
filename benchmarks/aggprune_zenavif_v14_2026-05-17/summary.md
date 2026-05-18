# Multi-seed confirm — zenavif_picker_config_aggprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 41.0s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 20.0% | 36.0% | +15.98 pp | +1.29 pp |
| `0xbeef` | 11.5% | 13.7% | +2.18 pp | +0.78 pp |
| `0xface` | 14.1% | 9.1% | -4.95 pp | -3.84 pp |

## Aggregate

- baseline argmin (median): **14.1%**
- recommended argmin (median): **13.7%**
- Δ argmin (median): **+2.18 pp** (stdev 10.64 pp, range -4.95..+15.98)
- Δ mean overhead (median): **+0.78 pp** (stdev 2.83 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
