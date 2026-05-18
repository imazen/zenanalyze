# Multi-seed confirm — zenjpeg_picker_config_aggprune20

Screen metric: `z_rmse`. Epochs: 60. Seeds: 3/3 ran successfully. Wall: 96.6s.

## Per-seed

| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |
|---|--:|--:|--:|--:|
| `0xcafe` | 27.5% | 35.4% | +7.89 pp | -0.33 pp |
| `0xbeef` | 33.2% | 36.0% | +2.72 pp | -1.16 pp |
| `0xface` | 24.7% | 23.3% | -1.40 pp | +3.77 pp |

## Aggregate

- baseline argmin (median): **27.5%**
- recommended argmin (median): **35.4%**
- Δ argmin (median): **+2.72 pp** (stdev 4.65 pp, range -1.40..+7.89)
- Δ mean overhead (median): **-0.33 pp** (stdev 2.64 pp)
- **verdict**: `noise`

Verdict logic:
- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)
- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)
- `regress` otherwise (clearly negative)
