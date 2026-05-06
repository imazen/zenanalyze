# v06 picker variant comparison

- 66120 cells, 95 images, 19 held out (seed 7)
- Default cell: (effort=7, biters=0, ziters=0), noise=False
- Safety mask: bytes <99% AND target not regressed AND ms <=105%

| variant | acc | n | mean Δbytes | mean Δtarget | mean Δms | default % | top picks |
|---|---:|---:|---:|---:|---:|---:|---|
| zensim_mask_mlp | 0.735 | 551 | -0.679% | +0.0275 | -11.29% | 70.6% | e7=389, e5=87, e3=75 |
| zensim_mask_histgb | 0.724 | 551 | -0.896% | +0.0063 | -7.44% | 76.0% | e7=420, e5=88, e3=43 |
| butter_max_mask_mlp | 0.668 | 551 | -1.984% | -0.0166 | -15.43% | 59.7% | e7=329, e5=193, e3=29 |
| butter_p3_mask_mlp | 0.733 | 551 | -1.334% | -0.0060 | -12.96% | 66.1% | e7=364, e5=158, e3=29 |
| multi_mask_mlp | 0.802 | 551 | -1.078% | -0.0143 | -7.51% | 78.9% | e7=435, e5=87, e3=29 |
| zensim_nomask_mlp | 0.579 | 551 | -5.288% | -0.7117 | +156.99% | 16.3% | e5=240, e7=217, e3=94 |
