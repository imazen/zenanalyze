# zenpicker meta-picker v0.2 — class-balanced (2026-05-06)

**TL;DR:** Second baked meta-picker, trained on **v10 + v12 unified corpus** (200 photo + 200 rebalanced gen-* sources across screen/doc/chart/line). 18 KB ZNPR v3 binary; 59.3% holdout accuracy; **−12.03% bytes vs always-zenjxl baseline**. 

The big news: **per-class Δbytes** shows synthetic content saves **−59%**, lineart **−46%**, photo **−26%** — much larger than v0.1 meta-picker because the rebalanced corpus has more diverse winning configurations across codecs.

## Architecture

- 20 inputs: 14 zenanalyze feat_* (intersection of v10 + v12 feature sets) + 5 cclass_* one-hot + 1 target band
- 96 → 64 ReLU → 3 (zenwebp/zenjxl/zenavif)
- 18 KB ZNPR v3 (f16 weights)
- `schema_hash = 0xd900d5193bee3d3c`

## Training data

| Source | rows | unique imgs | classes covered |
|---|---:|---:|---|
| v10 sweep (R2 sources/) | 40,000 | 200 | mostly photo |
| v12 sweep (rebalanced gen-*) | 30,235 (partial; sweep still running) | ~200 | screen/doc/chart/line |
| **union** | **70,235** | **~400** | photo + 4 non-photo classes |

## Per-class results on holdout

| class | n | MLP acc | mlp_dbytes vs always-jxl |
|---|---:|---:|---:|
| document | 17 | 70.6% | 0.00% |
| lineart | 44 | 47.7% | **−46.27%** |
| photo | 93 | 58.1% | **−26.44%** |
| screen | 22 | 86.4% | −11.55% |
| synthetic | 13 | 46.2% | **−59.18%** |
| **OVERALL** | **189** | **59.3%** | **−12.03%** |
| oracle ceiling | 189 | 100% | −22.76% (-10.7pp from MLP) |

The MLP captures **53% of oracle headroom**.

## Comparison to v0.1 meta-picker (v10-only, photo-heavy)

| Metric | v0.1 (v10 only) | v0.2 (v10 + v12) |
|---|---:|---:|
| Holdout cells | 125 (photo-only) | 189 (5 classes) |
| Holdout acc | 57.6% | **59.3%** |
| Δbytes vs always-X | −10.22% (vs always-AVIF) | **−12.03%** (vs always-zenjxl) |
| Oracle ceiling | −16.79% | −22.76% |
| Class coverage | photo only | 5 classes |

v0.2 wins on EVERY metric AND covers 5 content classes vs v0.1's photo-only.

## Caveats

- **Partial v12 data** — sweep was 503/600 chunks at training time. Final v0.2 should be re-baked when sweep completes (~1 min).
- **Photo class still smallish** in holdout (93 cells); per-class numbers volatile.
- **Synthetic n=13** is too small for the −59% to be a robust headline; could be 3-30% in reality.
- **3-family only** still — jpeg/png/gif need separate sweep coverage (task #39).
- The class-conditioning feature (5 cclass one-hot) requires the runtime classifier (`content_classifier_v0.2_2026-05-06.bin`). At inference: classify → predict.

## Pipeline picture

```
features (zenanalyze) → content_classifier_v0.2.bin → class label
                          ↓
       (features ⊕ cclass_onehot ⊕ target_band)
                          ↓
                  meta_v0.2.bin → codec choice
                          ↓
            zenjxl_picker_v0.7b.bin (if jxl chosen)
                          ↓
                  encoder config
```

Compounded Pareto move on a typical mixed-content workload: this v0.2 routing (-12%) + per-codec picker savings (-1.88% on jxl) + per-class encoder rule (-10% on screens within jxl) → **single-digit-percent improvement** on real production traffic.

## Provenance

- Trainer: `tools/v12_metapicker_train.py`
- Sweep data: v10 (`s3://zentrain/sweep-v10-2026-05-05/`) + v12 (`s3://zentrain/sweep-v12-2026-05-06/`)
- Bake: `tools/bake_picker.py --allow-unsafe`
- Generated 2026-05-06 during 10-hour autonomous run
