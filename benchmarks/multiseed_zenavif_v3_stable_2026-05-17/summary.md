# Multi-seed confirm — zenavif_picker_config_v3_stable

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 65.3s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 5.1% | 8.0% | +2.85 pp | -10.66 pp |
| `0xbeef` | 28.1% | 14.4% | -13.69 pp | +0.76 pp |
| `0xface` | 6.5% | 9.4% | +2.95 pp | -7.16 pp |

## Aggregate

- baseline argmin (median): **6.5%**
- recommended argmin (median): **9.4%**
- Δ argmin (median): **+2.85 pp** (stdev 9.58 pp, range -13.69..+2.95)
- Δ mean overhead (median): **-7.16 pp** (stdev 5.85 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
