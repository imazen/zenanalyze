# v06 picker variant comparison

- 44893 cells, 82 images, 16 held out (seed 7)
- Default cell: (effort=7, biters=0, ziters=0), noise=False
- Safety mask: bytes <99% AND target not regressed AND ms <=105%

| variant | acc | n | mean Δbytes | mean Δtarget | mean Δms | default % | top picks |
|---|---:|---:|---:|---:|---:|---:|---|
| zensim_mask_mlp | 0.844 | 64 | -0.593% | +0.0287 | -17.41% | 68.8% | e7=44, e5=20 |
| zensim_mask_histgb | 0.906 | 64 | -0.450% | +0.0061 | -14.39% | 75.0% | e7=48, e5=16 |
| butter_max_mask_mlp | 0.781 | 64 | -0.562% | +0.0026 | -14.89% | 70.3% | e7=45, e5=19 |
| butter_p3_mask_mlp | 0.797 | 64 | -0.810% | +0.0021 | -21.47% | 56.2% | e7=36, e5=28 |
| multi_mask_mlp | 0.844 | 64 | -0.215% | +0.0106 | -5.96% | 89.1% | e7=57, e5=7 |
| zensim_nomask_mlp | 0.719 | 64 | -7.533% | -2.4006 | +286.98% | 0.0% | e5=60, e7=4 |
