# Multi-seed confirm — zenjpeg_picker_config_v3_stable

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 157.7s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 8.1% | 41.7% | +33.57 pp | -27.70 pp |
| `0xbeef` | 12.0% | 39.5% | +27.54 pp | -15.06 pp |
| `0xface` | 5.9% | 42.4% | +36.54 pp | -32.73 pp |

## Aggregate

- baseline argmin (median): **8.1%**
- recommended argmin (median): **41.7%**
- Δ argmin (median): **+33.57 pp** (stdev 4.58 pp, range +27.54..+36.54)
- Δ mean overhead (median): **-27.70 pp** (stdev 9.10 pp)
- **verdict**: `ship`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
