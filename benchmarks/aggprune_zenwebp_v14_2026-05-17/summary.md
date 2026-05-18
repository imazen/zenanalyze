# Multi-seed confirm — zenwebp_picker_config_aggprune

Screen metric: `z_rmse`. Epochs: 60. Seeds: 5/5 ran successfully. Wall: 90.8s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 28.7% | 33.3% | +4.56 pp | -0.63 pp |
| `0xbeef` | 24.5% | 42.3% | +17.85 pp | -2.24 pp |
| `0xface` | 22.9% | 43.4% | +20.50 pp | -1.91 pp |
| `0xdead` | 21.9% | 37.3% | +15.38 pp | -1.14 pp |
| `0xbabe` | 28.9% | 31.0% | +2.02 pp | -0.86 pp |

## Aggregate

- baseline argmin (median): **24.5%**
- recommended argmin (median): **37.3%**
- Δ argmin (median): **+15.38 pp** (stdev 8.26 pp, range +2.02..+20.50)
- Δ mean overhead (median): **-1.14 pp** (stdev 0.69 pp)
- **verdict**: `ship`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
