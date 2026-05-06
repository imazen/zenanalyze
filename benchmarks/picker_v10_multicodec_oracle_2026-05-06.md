# v10 multi-codec picker analysis (zenjxl / zenavif / zenwebp)

- Source: v10 sweep, 40000 rows, 200 unique images
- Bands: zensim midpoints [70.0, 75.0, 80.0, 85.0, 90.0] ± 1.5

## Per-band winner distribution

Number of images where each codec produced the smallest encode meeting the band.

| band | n_images | jxl wins | avif wins | webp wins | jxl% | avif% | webp% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 135 | 37 | 70 | 28 | 27.4% | 51.9% | 20.7% |
| 75 | 113 | 45 | 49 | 19 | 39.8% | 43.4% | 16.8% |
| 80 | 108 | 31 | 66 | 11 | 28.7% | 61.1% | 10.2% |
| 85 | 112 | 37 | 65 | 10 | 33.0% | 58.0% | 8.9% |
| 90 | 156 | 100 | 52 | 4 | 64.1% | 33.3% | 2.6% |

## Median bytes per band (across images with all 3 codecs)

| band | n_imgs | jxl med | avif med | webp med | jxl/avif | jxl/webp | avif/webp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 70 | 25 | 9867 | 10810 | 12454 | 0.913 | 0.792 | 0.868 |
| 75 | 43 | 12767 | 10763 | 11698 | 1.186 | 1.091 | 0.920 |
| 80 | 30 | 12279 | 9693 | 13172 | 1.267 | 0.932 | 0.736 |
| 85 | 34 | 10968 | 10786 | 11454 | 1.017 | 0.958 | 0.942 |
| 90 | 68 | 19226 | 18107 | 27544 | 1.062 | 0.698 | 0.657 |

## Mean Δbytes vs single-codec baseline (lower = better)

If we always picked the SAME codec for every image, what would mean bytes be?
vs the multi-codec oracle (best per image)?

| band | n | jxl-only | avif-only | webp-only | oracle | oracle vs best-single |
|---:|---:|---:|---:|---:|---:|---:|
| 70 | 25 | 23620 | 23004 | 28014 | 22290 | -3.11% |
| 75 | 43 | 18700 | 15845 | 20228 | 15288 | -3.52% |
| 80 | 30 | 18535 | 15479 | 20285 | 14915 | -3.64% |
| 85 | 34 | 27806 | 25853 | 33888 | 24007 | -7.14% |
| 90 | 68 | 39028 | 38656 | 55651 | 35027 | -9.39% |

## Interpretation

- 'Oracle vs best-single' shows how much we'd save with a perfect per-image codec router.
- Negative % means routing helps (oracle smaller than always picking the same codec).
- A small gap (~0-2%) means one codec dominates this band; a large gap (5%+) means routing matters.
