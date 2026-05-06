# Unified v06+v08 picker — 416216 cells, 495 imgs, 99 holdout

- Cell taxonomy: 88 cells (union of v06 effort×biters×ziters + v07 extras)
- Default cell: (7, 0, 0, None, None, 'single', True, None, None, None, True)
- Architecture: 256x128 dropout=0.2, weight decay 1e-5
- Held-out 248 imgs (seed 7), 1386 cells

## A/B vs default

- val acc: 0.495
- mean Δbytes: -1.424%
- mean Δzensim: +0.0270pp
- mean Δms: -17.08%
- default picks: 69.8%
- alt picks: 30.2%
- alt picks using v08-only knobs: 0.0%
- pick effort distribution: {3: 168, 7: 968, 5: 250}
