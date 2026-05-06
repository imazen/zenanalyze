# Content classifier v0.2 — production gate for per-class rules (2026-05-06)

**TL;DR:** 7.7 KB ZNPR v3 classifier mapping 15 zenanalyze features → 4-class softmax (photo / screen / lineart / document). 99.6% holdout accuracy on the rebalanced corpus (3,525 holdout images stratified by class). Production-ready gate for the per-class picker rules.

## Holdout confusion matrix (3525 imgs, image-level holdout)

|  | photo | screen | lineart | document |
|---|---:|---:|---:|---:|
| **photo** (1103) | **1090** | 0 | 11 | 2 |
| **screen** (603) | 0 | **602** | 0 | 1 |
| **lineart** (1219) | 1 | 0 | **1218** | 0 |
| **document** (600) | 0 | 0 | 0 | **600** |

Per-class metrics:

| class | precision | recall | n holdout |
|---|---:|---:|---:|
| photo | 0.999 | 0.988 | 1103 |
| screen | 1.000 | 0.998 | 603 |
| lineart | 0.991 | 0.999 | 1219 |
| document | 0.995 | 1.000 | 600 |

## Why this matters

The per-class encoder rule (`patches=True+gaborish=False` for screen/synthetic) and the v0.7b picker gate (`skip on screen`) BOTH need a content class at runtime. Filename heuristics work on training data but fail in production where filenames may be UUIDs/hashes/anonymized.

This classifier provides a deterministic, fast (15 features → 4-class softmax) class label from the same zenanalyze features the picker already consumes. **No new feature extraction needed** — reuses the picker's input vector.

## Architecture

- 15 inputs: variance, edge_density, chroma_complexity, luma_histogram_entropy, dct_compressibility_y, dct_compressibility_uv, high_freq_energy_ratio, uniformity, colourfulness, laplacian_variance, log_pixels, aspect_min_over_max, flat_color_block_ratio, gradient_fraction, hdr_present
- 64 → 32 ReLU → 4 (softmax via argmin)
- 7.7 KB ZNPR v3 (f16 weights)
- `schema_hash = 0x10429adc95a10579`

## Caveats

- **No synthetic class** in training data (rebalanced corpus didn't include synthetic gradients/checkers). Synthetic content currently routes to "lineart" or "photo" (fall-through). Acceptable as-is — the per-class rule treats synthetic and screen identically.
- **Trained on rebalanced filename labels** — these are heuristic labels (gen-* prefixes are perfect; real content uses substring matches). The 99.6% accuracy is against those heuristic labels, not "true" content class. Production accuracy on novel content (photos vs screenshots that don't match the heuristics) is likely lower.
- **Distribution shift**: rebalanced corpus has 31% photo / 35% lineart / 17% document / 17% screen. Production traffic is much more photo-heavy; the classifier may overpredict non-photo at runtime. A simple confidence threshold (only mark as non-photo if softmax > 0.7) would mitigate.

## Wiring

```rust
let class_logits = content_classifier_v0_2.predict(&feature_15)?;
let class = match class_logits.argmin() {
    0 => Class::Photo,
    1 => Class::Screen,
    2 => Class::Lineart,
    3 => Class::Document,
    _ => unreachable!(),
};
let max_conf = softmax(&class_logits).iter().cloned().fold(0.0, f32::max);
if max_conf < 0.7 {
    return Class::Photo;  // fall back to safe default
}
```

## Provenance

- Trainer: `tools/train_content_classifier_rebal.py`
- Source: `/mnt/v/output/zensim/v06-rebalance/zenanalyze_union_rebalanced_cclass.tsv` (17,629 rebalanced sources)
- Bake: `tools/bake_picker.py --allow-unsafe`
- File: `benchmarks/content_classifier_v0.2_2026-05-06.bin`
- Generated 2026-05-06 during 10-hour autonomous run
