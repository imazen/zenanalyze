# Multi-seed confirm — zenwebp_picker_config

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 723.7s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 20.8% | 45.3% | +24.54 pp | -2.62 pp |
| `0xbeef` | 22.4% | 39.2% | +16.80 pp | -2.65 pp |
| `0xface` | 19.7% | 46.9% | +27.21 pp | -2.92 pp |

## Aggregate

- baseline argmin (median): **20.8%**
- recommended argmin (median): **45.3%**
- Δ argmin (median): **+24.54 pp** (stdev 5.41 pp, range +16.80..+27.21)
- Δ mean overhead (median): **-2.65 pp** (stdev 0.17 pp)
- **verdict**: `ship`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
