# zenjxl picker v0.7 lineage — content-class-aware variants (2026-05-06)

Two variants trained today using the v06 sweep + zenanalyze features, both
adding content_class one-hot as input but differing in training data:

| Variant | Train set | Holdout overall Δbytes | Notes |
|---|---|---:|---|
| **v0.6_mlp** | all classes | (N/A — see audit) | Photo-tuned, +41% on screen. baseline |
| **v0.7** | all classes + cclass input | unknown | Added cclass → photo improves but screen STILL +50% |
| **v0.7b** | non-screen + cclass input | **−0.95% bytes overall** | Hard gate: screen → default |

## v0.7b per-class on HOLDOUT (98 imgs)

| class | n | Δbytes% | Δzensim_pp | def_pick% |
|---|---:|---:|---:|---:|
| photo | 1162 | −1.120% | +0.072 | 77.5 |
| **lineart** | 56 | **−3.853%** | +0.168 | 32.1 |
| screen | 70 | 0.000% | 0.000 | 100.0 (gated) |
| synthetic | 70 | +0.016% | +0.285 | 48.6 |
| **overall (weighted)** | **1358** | **−0.948%** | varied | varied |

## Why v0.7 (cclass-as-input only) failed on screen

Adding cclass features didn't fix the screen regression — the picker still
saw +50% bytes regression on screens. The reason: even with `cclass_screen=1.0`
in the input, the MLP learns from photo-majority training data and applies
photo-shaped decision boundaries (effort=3 → save time, lose bytes). The
safety mask filters LABELS but not the FEATURE-space pull.

## Why v0.7b works

By excluding screen samples from training, the MLP never learns to pick
non-default cells "like the photo majority did" for screen-shaped features.
At inference, the runtime gate routes `class==screen` straight to default
(effort=7, biters=0, ziters=0) — never invoking the picker.

## Lineart improvement (−3.85%)

The cclass-as-input lets the picker learn "lineart prefers higher effort
with biters/ziters tweaks" patterns. v0.6 picker saw 0% improvement on
lineart because it couldn't distinguish lineart from photo in feature
space; now it can.

## Architecture

- 92 zenanalyze feat_* + 5 cclass_* one-hot + 1 log_distance = **98 inputs**
- 128 → 128 ReLU → 20 active cells (subset of 24 (effort, biters, ziters))
- 65 KB ZNPR v3 (f16 weights)
- `schema_hash = 0x5896532033934e16`

## Provenance

- Trainer: `tools/v07b_zenjxl_picker_no_screen.py` (committed)
- Data: v06 sweep (`~/sweep-data/zenjxl_v06.tsv`)
- Bake: `tools/bake_picker.py --allow-unsafe`
- File: `benchmarks/zenjxl_picker_v0.7b_2026-05-06.bin`
- Generated 2026-05-06 during 10-hour autonomous run

## Wiring into zenjxl crate

```rust
// Pseudocode for the codec runtime
fn pick_zenjxl_config(features: &[f32], target_distance: f32, content_class: ContentClass) -> JxlConfig {
    if content_class == ContentClass::Screen {
        return JxlConfig::default();  // effort=7, biters=0, ziters=0
    }
    let mut input = features.to_vec();
    input.extend_from_slice(&one_hot_cclass(content_class));
    input.push(target_distance.ln());
    let pick = picker_v0_7b.argmin_masked(&input, &mask, ...)?;
    JxlConfig::from_cell(pick.cell_idx)
}
```

## Caveats

- The 70 screen / 56 lineart / 70 synthetic holdout cells are TINY. Per-class
  numbers are noisy; the photo result (1162 cells) is the only one inside
  noise floor.
- Trained on v06 grid only (effort × biters × ziters). The bigger Pareto move
  is **adding the per-class encoder rule** (`patches=True, gaborish=False`
  for screen/synthetic — see `zenjxl/benchmarks/per_class_encoder_rule_v07_v08_2026-05-06.md`).
  Combined: v0.7b picks (effort, biters, ziters) for non-screen, and the
  encoder rule sets (patches, gaborish) per class.
- v0.6_mlp/v0.7/v0.7b all use the SAME feature set; differences are
  training data and gate.
