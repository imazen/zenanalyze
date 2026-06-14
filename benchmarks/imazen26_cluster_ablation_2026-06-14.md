# imazen-26 representative-subset ablation (2026-06-14)

Selects a **feasible, coverage-preserving subset of `(image, crop)` sources**
from the imazen-26 feature dataset so the costly downstream — encode-sweeping
each source across q × size × config to train picker MLPs — is affordable
without losing feature-space coverage. Feature extraction is cheap; encode
sweeps are ~100× costlier, so per the ML-data discipline we reduce the
*representative-source count* (not size/q resolution) via k-means on content
features, keeping each cluster's centroid-nearest member.

- **Population:** 23,727 native `(image, crop)` units (2157 images × 11 crops),
  from `imazen26_features_2026-06-13.parquet` (`size_class == native`).
- **Space:** 84 zenanalyze **content** features, z-scored; the 13 geometry
  features (`pixel_count`, `*_dim`, `aspect*`, `block_misalignment*`,
  `log_padded*`, `bitmap_bytes`, `channel_count`) are **excluded** — size is the
  densification axis applied *to* the chosen reps, not a selection axis.
- **Method:** `benchmarks/imazen26_cluster_ablation_2026-06-14.py`
  (sklearn k-means, n_init 10, seed 0) — the Rust equivalent is
  `zenpicker-train/src/bin/cluster_features.rs`.

## Coverage vs K

`var_explained` = fraction of total feature variance captured; `dist_pNN` =
percentiles of each corpus unit's distance to its cluster centroid (how far a
typical source sits from its nearest representative, standardized space).

| K | var_explained | dist_p50 | dist_p95 | dist_max | singletons |
|--:|--:|--:|--:|--:|--:|
| 100 | 0.803 | 3.38 | 6.71 | 28.0 | 1 |
| 200 | 0.848 | 3.05 | 5.77 | 28.0 | 2 |
| 300 | 0.870 | 2.85 | 5.36 | 23.2 | 6 |
| **500** | **0.894** | **2.61** | **4.79** | **16.4** | **18** |
| 750 | 0.910 | 2.42 | 4.38 | 11.7 | 37 |
| 1000 | 0.922 | 2.28 | 4.12 | 11.0 | 54 |
| 1500 | 0.936 | 2.09 | 3.70 | 8.1 | 110 |

Knee at **K≈300–500**: variance-explained climbs steeply to 0.87 by 300, then
flattens (only 0.936 by 1500); p95 nearest-rep distance falls 6.7→4.8 by 500,
diminishing after. K=300 drops `2200-unsplash-renders` to **0** reps (a coverage
gap); **K=500 keeps every content class (≥2)**, matches the prior v06 sweep
precedent, and retains 18 outlier singletons.

## Recommendation: K = 500

500 `(image, crop)` reps spanning **414 distinct source images** with a balanced
crop spread (`full` + all c50/c25 positions, 20–61 each) — crops contribute real
content diversity, not redundancy.

**k-means rebalances by feature diversity, not raw count** — exactly the
anti-modal-bias property the discipline wants:

| class | corpus share | K=500 rep share | effect |
|---|--:|--:|---|
| `9226-ai-products` | 0.347 | 0.218 | down-weighted (homogeneous white-bg shots collapse) |
| `7000-plots` | 0.058 | 0.148 | up-weighted (feature-diverse) |
| `2400-textures` | 0.005 | 0.012 | up-weighted (diverse) |
| `6600-manuscript-illustrations` | 0.017 | 0.040 | up-weighted |

## Artifacts (block storage + Tower; not in git)

| K | rows | path (`/mnt/v/output/imazen-26-features/` + Tower mirror) | sha256-16 |
|--:|--:|---|---|
| 300 | 300 | `imazen26_representatives_K300_2026-06-14.tsv` | `fdf20d4d95889e13` |
| **500** | **500** | `imazen26_representatives_K500_2026-06-14.tsv` | `00c79fd8ffb7099c` |
| 1000 | 1000 | `imazen26_representatives_K1000_2026-06-14.tsv` | `100e66e0fbcec957` |

Schema: `image_path  crop_label  content_class  cluster_id  cluster_size`.
Tower: `/mnt/tower/output/imazen-26-features/`.

## How to use

Each rep row is a `(source PNG, crop window)`. To build the picker training set,
apply the **dense size grid + q grid + config axes** to the K reps (densify on
representatives — do NOT cluster those axes away):

```
encodes ≈ K · (≈10 downscale sizes) · (q grid, e.g. 21–30) · (configs)
```

So K is the lever for total encode cost. Start at **K=500** (≈ the v06
precedent); drop to 300 only if a class-coverage gap is acceptable, or go to
1000 for +0.03 variance-explained when budget allows.

## Regenerate / re-select at another K

```bash
python3 benchmarks/imazen26_cluster_ablation_2026-06-14.py \
  --parquet /mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet \
  --select-k <K>   # omit --select-k to re-print the coverage sweep
```
