# zenjxl held-out A/B (distance-banded)

**Verdict: HOLD**

- Holdout: 248 images (frac=0.2, seed=7); 4695 (image, distance) cells.
- Default cell: effort7
- Picker cell preference: {'effort9': 2258, 'effort3': 958, 'effort5': 970, 'effort7': 463, 'effort1': 46}

## Per-band results

| band (distance) | n | mean Δbytes % | median Δbytes % | win rate | mean Δzensim_pp |
|---|---:|---:|---:|---:|---:|
| tight (0.05..1.0) | 1721 | +1.27 | +0.55 | 23.4% | +0.01 |
| mid (1.0..3.0) | 1238 | +1.46 | +0.66 | 27.9% | +0.30 |
| loose (3.0..15) | 1736 | +1.89 | +0.00 | 35.7% | +1.22 |
| overall | 4695 | +1.55 | +0.41 | 29.1% | +0.54 |
