# Unified v06+v07 picker — 198466 cells, 542 imgs, 108 holdout

- Cell taxonomy: 96 cells (union of v06 effort×biters×ziters + v07 extras)
- Default cell: (7, 0, 0, None, None, 'single', True, None, None, None, True)
- Architecture: 256x128 dropout=0.2, weight decay 1e-5
- Held-out 248 imgs (seed 7), 1386 cells

## A/B vs default

- val acc: 0.631
- mean Δbytes: -1.948%
- mean Δzensim: +0.2478pp
- mean Δms: -23.57%
- default picks: 57.9%
- alt picks: 42.1%
- alt picks using v07-only knobs: 0.0%
- pick effort distribution: {3: 238, 5: 321, 7: 782, 9: 9}
