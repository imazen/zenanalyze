# External-caller pattern for joint-pretrained per-codec bakes

## Problem

The joint-trunk pretrain methodology (`zentrain/tools/train_multi_codec.py`)
produces **one per-codec `.bin` per participating codec**. Each bake is a normal
ZNPR v3 single-output MLP — no v3.2 multi_codec_schema wire-format section
required. But each codec's bake has UNION-schema input (not the codec's natural
feature schema), so the caller (the codec's encoder, or an external picker
orchestrator) must assemble the union input vector before calling
`Predictor::predict`.

## Wire-format reminder

There is **no special runtime section** for joint-pretrained bakes. They look
identical to per-codec single-codec bakes from the runtime's perspective. The
only difference is that:

- `Model::n_inputs()` returns a larger number than the codec's natural feature
  count (= 2 × union_feat_count + 4 + 1 + 1 + n_codecs).
- The bake's `scaler_mean` / `scaler_scale` were trained against the union
  input shape.
- `feat_cols` metadata (if present) lists union feature names, not just the
  codec's natural feats.

`Predictor::predict(features: &[f32])` validates `features.len() == n_inputs()`
and then runs forward. The caller is responsible for assembling the right
vector.

## Union-input vector layout

Per the joint trainer's `_concat_codec_input`:

```text
[ union_feat_values  (U,)   — zenanalyze features per codec, scattered into
                              union slots by per-codec map. Zero in slots this
                              codec doesn't carry.
  presence_mask      (U,)   — 1.0 in slots populated by this codec, 0.0 elsewhere.
  size_onehot        (4,)   — one-hot for size_class ∈ {tiny, small, medium, large}.
  log_pixels         (1,)   — natural log of image's pixel count.
  zq_norm            (1,)   — target zensim quality, normalized as
                              (zq - 50) / 50.
  codec_onehot       (C,)   — one-hot for which codec is invoking (this codec's
                              slot = 1.0). ]
```

Total `n_inputs = 2*U + 4 + 1 + 1 + C`.

For the 3-codec (zenjpeg + zenwebp + zenavif) joint pretrain from 2026-05-17,
`U = 65, C = 3, n_inputs = 139`.

## Per-codec metadata (where to find U, C, slot map)

The joint trainer emits this info in the bake's metadata via
`tools/bake_picker.py`. Inspect with:

```bash
zenpredict inspect path/to/joint_pretrain_<codec>_picker.bin
```

Look for metadata keys:
- `joint.union_feat_count` (u32) — `U`
- `joint.n_codecs` (u32) — `C`
- `joint.codec_id` (u32) — this codec's one-hot index in `codec_onehot`
- `joint.codec_name` (utf8) — for cross-check
- `joint.union_feat_cols` (utf8, newline-separated) — names of the U union slots
- `joint.codec_feat_to_union_slot` (utf8, newline-separated) — for each of this
  codec's natural feat_cols (in the order the codec's
  zenanalyze::analyze_features returns), the union slot index 0..U-1 to
  scatter into.

## Caller pseudocode

```rust,ignore
use zenpredict::{Model, Predictor};

let bake_bytes: &[u8] = include_bytes!("joint_pretrain_zenjpeg.bin");
static MODEL: OnceLock<Model> = OnceLock::new();
let model = MODEL.get_or_init(|| Model::from_bytes(bake_bytes).unwrap());

// Cached from the bake's metadata at init time:
let union_count: usize = read_u32_meta(model, "joint.union_feat_count");
let n_codecs: usize = read_u32_meta(model, "joint.n_codecs");
let codec_id: usize = read_u32_meta(model, "joint.codec_id");
let slot_map: Vec<u32> = read_u32_lines(model, "joint.codec_feat_to_union_slot");

let n_in = model.n_inputs();
debug_assert_eq!(n_in, 2 * union_count + 4 + 1 + 1 + n_codecs);

let mut input = vec![0.0f32; n_in];

// Codec's natural features (one f32 per column)
let codec_features: &[f32] = analyze_features_rgb8(&image, ...);

// Scatter codec's feats into union slots + presence mask.
for (i, &slot) in slot_map.iter().enumerate() {
    let slot = slot as usize;
    input[slot] = codec_features[i];
    input[union_count + slot] = 1.0;
}

// Size one-hot.
let size_off = 2 * union_count;
let size_idx = match max(image.width, image.height) {
    0..=63 => 0,    // tiny
    64..=255 => 1,  // small
    256..=1023 => 2, // medium
    _ => 3,         // large
};
input[size_off + size_idx] = 1.0;

// log_pixels + zq_norm.
input[size_off + 4] = (image.width * image.height) as f32).ln();
input[size_off + 5] = (zq_target - 50.0) / 50.0;

// codec one-hot.
input[size_off + 6 + codec_id] = 1.0;

// Run picker.
let mut predictor = Predictor::new(model);
let bytes_log_per_cell: &[f32] = predictor.predict(&input).unwrap();
let argmin = argmin_masked(bytes_log_per_cell, &allowed_cells_mask);
```

## Single-codec fallback

Each codec should also ship a fallback **per-codec-only** bake (the
`bake_picker.py` output without joint pretrain) for the case where the joint
bake is unavailable, hasn't been ported to a target platform, or doesn't carry
sufficient metadata. The per-codec fallback uses the codec's natural feature
schema only — no union scatter needed.

Recommended runtime strategy:
1. Try to load joint bake; check metadata for the union slot map.
2. If joint bake load fails OR slot-map metadata is missing, fall back to the
   per-codec single-codec bake.
3. Both produce `bytes_log_per_cell` outputs that the codec's argmin / cell-
   dispatch logic consumes identically.

## When to use which bake

- **Joint pretrain bake**: when shipping the codec to environments where the
  ~50 KB joint bake is acceptable, the runtime knows the union slot map, and
  cross-codec generalization is valuable (zenjpeg + zenavif both win ≥10pp
  argmin from joint training per `benchmarks/joint_pretrain_breakthrough_2026-05-17.md`).
- **Per-codec single bake**: when shipping the codec in isolation, the union
  slot map isn't easily accessible, or zenwebp's per-codec already-shipping
  methodology is sufficient (zenwebp doesn't benefit from joint training).
