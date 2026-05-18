# Multi-seed confirm — zenwebp_picker_config_ultraprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 5/5 ran successfully. Wall: 106.5s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 47.4% | 40.7% | -6.73 pp | +0.56 pp |
| `0xbeef` | 32.1% | 39.4% | +7.29 pp | -0.43 pp |
| `0xface` | 31.0% | 31.1% | +0.15 pp | -1.10 pp |
| `0xdead` | 37.9% | 36.2% | -1.75 pp | -0.07 pp |
| `0xbabe` | 45.8% | 44.6% | -1.20 pp | -0.75 pp |

## Aggregate

- baseline argmin (median): **37.9%**
- recommended argmin (median): **39.4%**
- Δ argmin (median): **-1.20 pp** (stdev 5.05 pp, range -6.73..+7.29)
- Δ mean overhead (median): **-0.43 pp** (stdev 0.64 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
