# Multi-seed confirm — zenavif_picker_config_pruned

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 33.5s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 10.1% | 18.7% | +8.63 pp | -1.48 pp |
| `0xbeef` | 25.2% | 18.4% | -6.85 pp | +1.66 pp |
| `0xface` | 22.7% | 14.5% | -8.19 pp | -0.37 pp |

## Aggregate

- baseline argmin (median): **22.7%**
- recommended argmin (median): **18.4%**
- Δ argmin (median): **-6.85 pp** (stdev 9.35 pp, range -8.19..+8.63)
- Δ mean overhead (median): **-0.37 pp** (stdev 1.59 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
