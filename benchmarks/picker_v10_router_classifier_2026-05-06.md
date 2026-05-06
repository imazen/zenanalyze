# v10 multi-codec router (HistGradientBoosting classifier)

- 624 (image, band) training cells, 153 train imgs / 38 hold imgs
- features: 92 zenanalyze feat_* + 1 band
- baseline (always zenavif): acc=0.480, bytes=13599859
- router: acc=0.616, bytes=12063319 (-11.30% vs baseline)
- oracle ceiling: bytes=11316529 (-16.79% vs baseline)

## Holdout classification report

```
              precision    recall  f1-score   support

     zenavif      0.625     0.750     0.682        60
      zenjxl      0.683     0.549     0.609        51
     zenwebp      0.333     0.286     0.308        14

    accuracy                          0.616       125
   macro avg      0.547     0.528     0.533       125
weighted avg      0.616     0.616     0.610       125

```
