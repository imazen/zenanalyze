# v06 picker variant comparison

- 129233 cells, 385 images, 77 held out (seed 7)
- Default cell: (effort=7, biters=0, ziters=0), noise=False
- Safety mask: bytes <99% AND target not regressed AND ms <=105%

| variant | acc | n | mean Δbytes | mean Δtarget | mean Δms | default % | top picks |
|---|---:|---:|---:|---:|---:|---:|---|
| zensim_mask_mlp | 0.617 | 1078 | -2.190% | +0.1620 | -25.24% | 49.0% | e7=528, e5=281, e3=256, e9=13 |
| zensim_mask_histgb | 0.656 | 1078 | -1.662% | +0.1715 | -17.27% | 59.6% | e7=645, e5=235, e3=191, e9=7 |
| butter_max_mask_mlp | 0.642 | 1078 | -1.655% | -0.0349 | -20.13% | 60.5% | e7=652, e5=397, e3=15, e9=14 |
| butter_p3_mask_mlp | 0.734 | 1078 | -1.840% | -0.0107 | -23.49% | 56.0% | e7=604, e5=429, e3=31, e9=14 |
| multi_mask_mlp | 0.742 | 1078 | -1.104% | +0.0724 | -11.00% | 78.4% | e7=845, e5=173, e3=46, e9=14 |
| zensim_nomask_mlp | 0.434 | 1078 | -8.407% | -1.5921 | +250.67% | 4.2% | e5=453, e3=391, e7=194, e9=40 |
