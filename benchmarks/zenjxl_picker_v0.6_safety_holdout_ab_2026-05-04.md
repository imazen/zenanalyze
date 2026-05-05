# zenjxl held-out A/B (distance-banded)

**Verdict: HOLD**

- Holdout: 248 images (frac=0.2, seed=7); 4696 (image, distance) cells.
- Default cell: effort7
- Picker cell preference: {'effort9': 1675, 'effort5': 1197, 'effort3': 1634, 'effort7': 190}

## Per-band results

| band (distance) | n | mean Δbytes % | median Δbytes % | win rate | mean Δzensim_pp |
|---|---:|---:|---:|---:|---:|
| tight (0.05..1.0) | 1722 | +0.49 | +0.28 | 37.3% | -0.02 |
| mid (1.0..3.0) | 1238 | +0.83 | +0.46 | 36.3% | +0.29 |
| loose (3.0..15) | 1736 | +2.12 | +0.59 | 34.3% | +0.98 |
| overall | 4696 | +1.18 | +0.42 | 35.9% | +0.43 |
