# Multi-seed confirm — zenwebp_picker_config_v3_stable

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 81.7s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 48.6% | 43.7% | -4.86 pp | +0.28 pp |
| `0xbeef` | 40.0% | 41.1% | +1.04 pp | -0.38 pp |
| `0xface` | 54.1% | 39.6% | -14.53 pp | +1.36 pp |

## Aggregate

- baseline argmin (median): **48.6%**
- recommended argmin (median): **41.1%**
- Δ argmin (median): **-4.86 pp** (stdev 7.86 pp, range -14.53..+1.04)
- Δ mean overhead (median): **+0.28 pp** (stdev 0.88 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
