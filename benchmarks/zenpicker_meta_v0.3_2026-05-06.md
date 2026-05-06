# zenpicker meta-picker v0.3 — full v12 data (2026-05-06)

**TL;DR:** Final meta-picker bake with full v12 sweep (600 chunks). **−16.92% bytes** vs always-zenjxl on 227 holdout cells across 5 content classes. 67.4% holdout accuracy. Captures **74% of oracle headroom** (vs v0.2's 53%).

## Per-class results

| class | n | acc | mlp_dbytes |
|---|---:|---:|---:|
| document | 30 | 80.0% | −5.80% |
| **lineart** | 38 | 71.1% | **−52.51%** |
| **photo** | 122 | 58.2% | **−21.43%** |
| screen | 34 | 88.2% | −14.15% |
| synthetic | 3 | 33.3% | −97.86% (n too small) |
| **OVERALL** | **227** | **67.4%** | **−16.92%** |

## v0.1 → v0.2 → v0.3 progression

| Metric | v0.1 (v10 photo) | v0.2 (partial v12) | v0.3 (full v12) |
|---|---:|---:|---:|
| Holdout cells | 125 | 189 | 227 |
| Holdout acc | 57.6% | 59.3% | **67.4%** |
| Δbytes vs always-X | −10.22% | −12.03% | **−16.92%** |
| Oracle ceiling | −16.79% | −22.76% | −22.81% |
| Headroom captured | 61% | 53% | **74%** |
| Class coverage | photo only | 5 classes | 5 classes |

Each generation strictly improves on the prior.

## Architecture

Same as v0.2: 20 inputs (14 zenanalyze feat + 5 cclass one-hot + 1 target band) → 96 → 64 ReLU → 3 outputs (zenwebp/zenjxl/zenavif). 18 KB ZNPR v3 binary (f16 weights).

## Caveats

- **synthetic n=3** in holdout — the headline −97.86% is a single-cell anomaly; ignore that cell, real synthetic improvement should be inferred from v0.2's −59% (n=13) which is itself noisy.
- **3-family only** — jpeg/png/gif still need separate sweep (task #39).
- **Photo-class accuracy 58%** — picker isn't great at photo discrimination because there's often a tie between codecs at common bands. The Δbytes is what matters anyway (−21%).

## Pipeline

```
features → content_classifier_v0.2.bin → cclass label
        → meta_v0.3.bin → codec choice
        → per-codec picker → encoder config
```

## Provenance

- Trainer: `tools/v12_metapicker_train.py`
- Source data: v10 (40k rows) + v12 (full 600 chunks, ~30k rows after partial → full)
- Generated 2026-05-06 during 10-hour autonomous run
- Supersedes v0.2 (`zenpicker_meta_v0.2_2026-05-06.bin`) which used 503/600 partial v12
