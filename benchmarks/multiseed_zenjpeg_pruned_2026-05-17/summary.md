# Multi-seed confirm — zenjpeg_picker_config_pruned

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 99.2s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 29.9% | 37.3% | +7.39 pp | -3.44 pp |
| `0xbeef` | 30.7% | 31.9% | +1.22 pp | +0.89 pp |
| `0xface` | 30.5% | 31.8% | +1.29 pp | +1.33 pp |

## Aggregate

- baseline argmin (median): **30.5%**
- recommended argmin (median): **31.9%**
- Δ argmin (median): **+1.29 pp** (stdev 3.54 pp, range +1.22..+7.39)
- Δ mean overhead (median): **+0.89 pp** (stdev 2.63 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
