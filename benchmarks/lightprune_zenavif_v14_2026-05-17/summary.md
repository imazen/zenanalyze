# Multi-seed confirm — zenavif_picker_config_lightprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 5/5 ran successfully. Wall: 64.7s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 10.7% | 7.3% | -3.31 pp | +0.64 pp |
| `0xbeef` | 15.1% | 24.3% | +9.13 pp | -3.03 pp |
| `0xface` | 19.3% | 30.6% | +11.24 pp | -1.37 pp |
| `0xdead` | 22.5% | 20.0% | -2.53 pp | +2.10 pp |
| `0xbabe` | 7.1% | 28.2% | +21.09 pp | -4.26 pp |

## Aggregate

- baseline argmin (median): **15.1%**
- recommended argmin (median): **24.3%**
- Δ argmin (median): **+9.13 pp** (stdev 10.22 pp, range -3.31..+21.09)
- Δ mean overhead (median): **-1.37 pp** (stdev 2.60 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
