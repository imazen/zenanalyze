# v06 picker variant comparison

- 165895 cells, 495 images, 99 held out (seed 7)
- Default cell: (effort=7, biters=0, ziters=0), noise=False
- Safety mask: bytes <99% AND target not regressed AND ms <=105%

| variant | acc | n | mean Δbytes | mean Δtarget | mean Δms | default % | top picks |
|---|---:|---:|---:|---:|---:|---:|---|
| zensim_mask_mlp | 0.656 | 1386 | -1.985% | +0.3117 | -21.25% | 57.1% | e7=792, e3=299, e5=284, e9=11 |
| zensim_mask_histgb | 0.664 | 1386 | -1.879% | +0.4021 | -3.40% | 62.8% | e7=873, e5=268, e3=234, e9=11 |
| butter_max_mask_mlp | 0.639 | 1386 | -1.733% | -0.0104 | -20.75% | 61.5% | e7=853, e5=473, e3=46, e9=14 |
| butter_p3_mask_mlp | 0.734 | 1386 | -1.716% | -0.0079 | -21.49% | 57.8% | e7=801, e5=563, e9=12, e3=10 |
| multi_mask_mlp | 0.747 | 1386 | -1.063% | +0.1694 | -12.55% | 76.5% | e7=1060, e5=277, e3=40, e9=9 |
| zensim_nomask_mlp | 0.450 | 1386 | -8.662% | -1.2392 | +306.37% | 1.9% | e5=552, e3=498, e7=267, e9=69 |
