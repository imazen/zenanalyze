# zenpicker meta-picker v0.1 — first ever baked (2026-05-06)

**TL;DR:** First baked codec-family meta-picker (`zenpicker::MetaPicker`). 3-family routing (zenwebp / zenjxl / zenavif) trained on v10 multi-codec sweep. 24 KB ZNPR v3 binary; 57.6% holdout accuracy; **−10.22% bytes vs always-AVIF baseline**. Captures 61% of the oracle ceiling (-16.79%).

## Architecture

- Inputs: 92 zenanalyze `feat_*` features + 1 target zensim band → total 93
- Layers: 93 → 64 → 64 → 3 (ReLU activation, identity output)
- Output classes (CodecFamily indices used): zenwebp=0, zenjxl=1, zenavif=2
- Format: ZNPR v3 f16 weights (24 KB on disk)
- `schema_hash = 0xcb6e6e91690cf6d5`
- `schema_version_tag = "zenpicker.metapicker.v0.1"`

## Training

- Source: v10 sweep (`s3://zentrain/sweep-v10-2026-05-05/`)
- Filter: cells where ≥2 codecs land within ±1.5 zensim of target band {70, 75, 80, 85, 90}
- Sample size: 624 (image, band) pairs; 499 train / 125 holdout (image-level split)
- Class distribution: zenavif 302 (48%), zenjxl 250 (40%), zenwebp 72 (12%)
- Seed: 7

## Holdout results

| Strategy | Acc | Bytes (sum) | vs always-AVIF |
|---|---:|---:|---:|
| Always-AVIF (baseline) | 0.480 | 13,599,859 | 0.00% |
| **MLP router (THIS)** | **0.576** | **12,209,415** | **−10.22%** |
| Oracle (best per image) | 1.000 | 11,316,529 | −16.79% |

Captures 61% of available headroom. Mistakes are typically routing to AVIF when JXL would have been ~5% smaller; near-misses on the 90-band where JXL dominates (64% wins).

## Comparison to v10 HistGB router (same data)

The earlier non-bakeable HistGradientBoosting variant achieved 61.6% acc and -11.30% bytes (`picker_v10_router_classifier_2026-05-06.md`). The MLP variant is 4 percentage points worse on accuracy and 1 point worse on bytes — but is bakeable to ZNPR v3 and shippable via `zenpicker::MetaPicker::from_bytes`.

## Caveats — why this is v0.1 not v1.0

- **Small holdout (125 cells across 38 images)**: noisy headline number; need 10× more data to ship as production default.
- **3-family only**: zenpicker enum has 6 families (jpeg/webp/jxl/avif/png/gif). v10 sweep didn't cover jpeg/png/gif. Caller must pass `AllowedFamilies::all().deny(Jpeg).deny(Png).deny(Gif)` until 6-family bake exists.
- **WebP undersampled**: only 12% of training examples have webp as winner; the model can predict webp but its confidence intervals are wide.
- **No content-class conditioning**: trained globally; per-class behavior may differ. v10 corpus was photo-heavy.

## Wiring into zenpicker

```rust
use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker};
use zenpredict::Model;

#[repr(C, align(16))]
struct Aligned<const N: usize>([u8; N]);
const META_BIN: &[u8] = &Aligned(*include_bytes!(
    "../benchmarks/zenpicker_meta_v0.1_2026-05-06.bin"
)).0;

let model = Model::from_bytes_with_schema(META_BIN, /* schema hash */ 0xcb6e6e91690cf6d5)?;
let mut meta = MetaPicker::new(model);
let allowed = AllowedFamilies::all()
    .deny(CodecFamily::Jpeg)   // not in training data
    .deny(CodecFamily::Png)
    .deny(CodecFamily::Gif);
let chosen = meta.pick(&zenanalyze_features_with_target_band, &allowed)?;
```

The model's class-index → CodecFamily mapping is currently **anonymous** (the bake metadata names them `zenwebp_0`, `zenjxl_1`, `zenavif_2`). Need a separate manifest entry (`zenpicker.family_order`) so MetaPicker's `validate_family_order` check passes against the runtime enum order. **TODO before shipping**: add `family_order = "zenwebp,zenjxl,zenavif"` to the bake metadata.

## Provenance

- Trainer: `tools/v10_router_mlp_train.py` (committed)
- Bake: `tools/bake_picker.py --model v10_router_mlp_model.json --out benchmarks/zenpicker_meta_v0.1_2026-05-06.bin --dtype f16 --allow-unsafe`
- Generated 2026-05-06 during 10-hour autonomous run
- Cross-references:
  - HistGB router (better metrics, not bakeable): `picker_v10_router_classifier_2026-05-06.md`
  - Oracle analysis: `picker_v10_multicodec_oracle_2026-05-06.md`
  - zenpicker crate: `~/work/zen/zenanalyze/zenpicker/`
