# Multi-codec joint picker — external-caller pattern

**Status: 2026-05-17.** The Rust runtime + the bake pipeline + the
Python emitter all land in 2026-05-17's commits. A production-grade
joint bake (hyperparameter-tuned, variance-clamp fix in, longer
training) is the next step; the wiring is ready.

## Why this lives outside the codec crates

The user's constraint, 2026-05-17: *"all codecs should rely on external
caller to do prediction and tuning, they cannot depend on zenanalyze
and zenpredict yet"*. The multi-codec joint picker therefore wires up
**without touching `zenjpeg`, `zenwebp`, `zenavif`, or `zenjxl`**. Each
codec continues to expose its existing encode API that takes an
`EncoderConfig`-style struct; the picker output gets translated into
that struct by an external orchestrator.

Existing single-codec pickers embedded in each codec crate (e.g.
`zenjpeg_picker_v2.1_full.bin`) stay as-is. The multi-codec path is
*additive* — when adopted, the orchestrator can choose to call the
multi-codec joint picker instead of (or alongside) each codec's
embedded single-codec picker.

## End-to-end pipeline

```
                  ┌───────────────────────────────────┐
                  │ External orchestrator             │
                  │   (e.g. imageflow, image-cdn,     │
                  │    a CLI tool, or a service)      │
                  └─────────────────┬─────────────────┘
                                    │
                                    │ 1. extract features
                                    ▼
                   ┌──────────────────────────────┐
                   │ zenanalyze::analyze_features │
                   │   → Vec<feat_value>          │  per-codec natural schema
                   └─────────────┬────────────────┘
                                 │
                                 │ 2. apply per-codec transforms
                                 │    (log, log1p, etc — listed in
                                 │     the joint bake's
                                 │     per_codec_manifests entry)
                                 ▼
              ┌─────────────────────────────────────┐
              │ zenpredict::Predictor               │
              │   .predict_multi_codec(codec_id,    │
              │       codec_features, size_class,   │
              │       log_pixels, zq_norm)          │
              │   → &[f32] for that codec's range   │
              └─────────────────┬───────────────────┘
                                │
                                │ 3. argmin over bytes_log head;
                                │    read scalar heads for chosen cell;
                                │    clamp to caller constraints
                                ▼
                ┌──────────────────────────────────┐
                │ Build per-codec EncoderConfig    │
                │   from the picker's cell index   │
                │   + scalar predictions           │
                └─────────────────┬────────────────┘
                                  │ 4. encode
                                  ▼
                ┌──────────────────────────────────┐
                │ zenjpeg / zenwebp / zenavif /    │
                │ zenjxl encode_*(EncoderConfig)   │
                └──────────────────────────────────┘
```

**Codec crates own steps 4 only.** The orchestrator owns 1–3.

## Loading the joint bake

```rust
use std::fs;
use zenpredict::{Model, Predictor};

#[repr(C, align(16))]
struct AlignedModel<const N: usize>([u8; N]);

const MODEL_BYTES: &[u8] = &AlignedModel(*include_bytes!(
    "joint_picker_v0.1.bin"
)).0;

let model = Model::from_bytes(MODEL_BYTES)?;
assert!(model.has_multi_codec_schema());

let schema = model.multi_codec_schema().expect("multi_codec_schema");
// `codec_id` is the index into schema.per_codec; resolve from your
// caller-facing codec enum once at startup.
let codec_id_for_zenjpeg = schema.per_codec
    .iter().position(|m| m.codec_name == "zenjpeg")
    .expect("zenjpeg in joint bake") as u32;
```

## Per-encode call

```rust
let mut p = Predictor::new(&model);

// `codec_features` is in the codec's NATURAL feat_cols order — the
// order zenanalyze emits them, AFTER applying that codec's per-feature
// transforms (log, log1p — see per_codec_manifests[*].codec_feat_cols
// and feature_transforms in the joint bake's manifest).
let codec_features: Vec<f32> = orchestrator
    .extract_codec_features(image, "zenjpeg");

let size_class: u32 = match image.classify_size() {
    SizeClass::Tiny   => 0,
    SizeClass::Small  => 1,
    SizeClass::Medium => 2,
    SizeClass::Large  => 3,
};
let log_pixels: f32 = (image.width * image.height).max(1) as f32 \
    .ln() as f32;
let zq_norm: f32 = target_zq as f32 / 100.0;

let out = p.predict_multi_codec(
    codec_id_for_zenjpeg,
    &codec_features,
    size_class,
    log_pixels,
    zq_norm,
)?;
// `out` is a borrowed slice into the predictor's internal buffer —
// `output_range.1 - output_range.0` long. The layout matches the
// per_codec_manifests entry's `output_layout_relative` block.
```

## Mapping picker output → EncoderConfig

The joint bake JSON's `joint_multi_codec.per_codec_manifests[i]`
block carries:

- `output_range`: half-open range into `predict_multi_codec`'s output
- `n_cells`: number of config cells (the categorical pick space)
- `categorical_axes`: per-cell axis names (e.g.
  `["color_mode", "subsampling", "trellis_on", "sa_piecewise"]`)
- `scalar_axes`: ordered scalar prediction axes (e.g.
  `["chroma_scale", "lambda"]`)
- `cells`: per-cell metadata — `id`, `label`, per-axis values,
  `member_config_ids`
- `output_layout_relative`: byte-head and scalar-head index ranges
  WITHIN the per-codec slice (zero-based — caller adds nothing; the
  slice is already trimmed to that codec)
- `config_names`: map from `config_id` (int) → original config_name
  string emitted by the pareto sweep harness

Typical mapping in the orchestrator:

```rust
// out has length output_range.1 - output_range.0 already.
let n_cells = manifest.n_cells as usize;
let bytes_log_block = &out[..n_cells];

// 1. Argmin over allowed cells — caller's constraint mask hides
//    cells that violate caller intent (e.g. forbid_xyb).
let cell = bytes_log_block
    .iter()
    .enumerate()
    .filter(|(i, _)| caller_constraint_mask[*i])
    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(i, _)| i)
    .ok_or(NoAllowedCell)?;

// 2. Look up that cell's encoder-config tuple.
let cfg = manifest.cells[cell].clone();  // categorical axes already populated

// 3. Read scalar predictions for the chosen cell, clamp to caller range.
let scalar_idx = |axis: &str| {
    let r = manifest.output_layout_relative[axis];
    out[r.0 + cell]    // per-axis block of length n_cells, indexed by cell
};
let chroma_scale = scalar_idx("chroma_scale")
    .clamp(caller.chroma_min, caller.chroma_max);
let lambda = if cfg.trellis_on {
    Some(scalar_idx("lambda").clamp(caller.lambda_min, caller.lambda_max))
} else {
    None
};

// 4. Hand to the codec — zenjpeg::encode (or whichever) takes its
//    own ConfigStruct. The orchestrator owns the translation from
//    `cfg` + scalars → that struct.
let encoder_cfg = zenjpeg::ConfigStruct::from_picker_cell(
    cfg.color_mode,
    cfg.subsampling,
    chroma_scale,
    lambda,
);
zenjpeg::encode(image, &encoder_cfg)?;
```

The codec sees `ConfigStruct` — never `zenpredict::Predictor`, never
`zenanalyze::analyze_features`. Adding a 5th codec is a pure
orchestrator-side change.

## What the codec crate must NOT depend on

- `zenanalyze` — feature extractor
- `zenpredict` / `zenpredict-bake` — picker runtime + bake format

What it can depend on: its own internal types, the standard library,
and the orchestrator's `ConfigStruct` (which is plain-old-data).

## Smoke-test recipe

```bash
# 1. Joint training + joint bake JSON
PYTHONPATH=zenanalyze/zentrain/examples:zenanalyze/zentrain/tools \
  python3 zenanalyze/zentrain/tools/train_multi_codec.py \
    --codec zenjpeg=examples/zenjpeg_picker_config.py:/path/to/zenjpeg \
    --codec zenwebp=examples/zenwebp_picker_config.py:/path/to/zenwebp \
    --codec zenavif=examples/zenavif_picker_config.py:/path/to/zenavif \
    --out-dir /tmp/joint_smoke \
    --hidden 96,48 --epochs 200 \
    --joint-bake /tmp/joint_smoke/joint.bake.json

# 2. Bake JSON → .bin
python3 zenanalyze/tools/bake_picker.py \
    --model /tmp/joint_smoke/joint.bake.json \
    --out /tmp/joint_smoke/joint.bin \
    --dtype f32 --no-manifest \
    --bake-bin target/release/zenpredict-bake

# 3. Verify the runtime accepts the bake and runs predict_multi_codec.
cargo run --release -p zenpredict-bake --example inspect_multi_codec \
    -- /tmp/joint_smoke/joint.bin
```

Expected output of step 3:

```
Loaded model: schema_hash=0x... n_inputs=139 n_outputs=70
multi_codec_schema: union_feat_count=65 n_codecs=3
  codec[0] zenjpeg    slots=51  output_range=[0,36)  head_n_cells=12 head_n_heads=3
  codec[1] zenwebp    slots=33  output_range=[36,60) head_n_cells=6  head_n_heads=4
  codec[2] zenavif    slots=52  output_range=[60,70) head_n_cells=10 head_n_heads=1
predict_multi_codec(0, zeros) → output_len=36 first3=[..., ..., ...]
```

(Exact `n_inputs` / `output_dim` / slot counts depend on the codecs'
`feat_cols` lists.)

## Known follow-ups

- **Production joint bake.** The 2026-05-17 smoke training was 20
  epochs. Real shipping needs hyperparameter sweep + longer training +
  the variance-clamp floor fix flagged in
  `benchmarks/multi_codec_bench_2026-05-17.md`.
- **`zenpredict inspect` doesn't yet surface the
  `multi_codec_schema` section.** The
  `inspect_multi_codec` example exists for now; merging that into
  the unified `zenpredict inspect` reporter is a small follow-up.
- **`feature_transforms` per-codec parity.** The joint bake's
  top-level `feature_transforms` is identity; each codec's per-feature
  log/log1p needs to be applied by the orchestrator BEFORE scattering.
  The transforms live in the joint bake's
  `joint_multi_codec.per_codec_manifests[i].codec_feat_cols` and the
  source codec_config Python files. Surfacing them more accessibly
  (e.g. embedding the codec's `FEATURE_TRANSFORMS` map in the
  per-codec manifest) is queued.
- **`SizeClass` / `log_pixels` / `zq_norm` are caller-supplied.** The
  picker doesn't compute these from the image; the orchestrator must.

## Cross-references

- `zenpredict-bake/examples/inspect_multi_codec.rs` — runnable
  reference for steps 1-3 of the per-encode call sequence
- `zenpredict-bake/tests/multi_codec.rs` — 10 binary-format tests
- `zenpredict-bake/tests/json_bake.rs::json_round_trip_with_multi_codec_schema`
  — JSON in → bake → predict round-trip
- `zentrain/tools/train_multi_codec.py::write_joint_bake_json` —
  bake JSON emitter
- `tools/bake_picker.py::encode_multi_codec_schema` — JSON forwarder
- `benchmarks/multi_codec_bench_2026-05-17.md` — joint-vs-baseline measurements
- `benchmarks/distill_2026-05-17/README.md` — why distillation can't
  preserve the joint training win
