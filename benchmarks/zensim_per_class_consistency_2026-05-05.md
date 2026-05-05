# zensim per-content-class consistency vs SSIM2/butteraugli — v06 sweep

- Source data: v06 sweep, 165895 (image, knob) cells
- Per-class sample counts:
  - photo: 147399 cells
  - synthetic: 9525 cells
  - screen: 8971 cells

## Pairwise rank correlation (Spearman) of zensim vs other metrics, per class

Higher zensim = better quality. Higher SSIM2 = better quality. Lower butter = better quality (sign-flipped to align).

| class | n | zensim×ssim2 | zensim×butter_max | zensim×butter_p3 |
|---|---:|---:|---:|---:|
| photo | 147399 | +0.9114 | +0.7857 | +0.8355 |
| synthetic | 9525 | +0.9466 | +0.8813 | +0.8659 |
| screen | 8971 | +0.9864 | +0.9731 | +0.9711 |

## Interpretation

- High SROCC (>0.95) = zensim agrees with that metric on this class. Zensim works there.
- Low SROCC (<0.85) = zensim diverges from that metric. Possible blind spot.
- If photo class shows >0.95 across the board but screen/synthetic show <0.85, zensim is photo-tuned and less reliable on those classes.

## Per-class score distribution

| class | n | zensim mean | zensim std | ssim2 mean | butter_max mean |
|---|---:|---:|---:|---:|---:|
| photo | 147399 | 65.77 | 49.90 | 68.95 | 2.729 |
| synthetic | 9525 | 84.72 | 26.26 | 81.48 | 1.684 |
| screen | 8971 | 83.80 | 13.63 | 77.79 | 2.742 |
