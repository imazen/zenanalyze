# zenpicker ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpicker?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpicker?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpicker) ![docs.rs](https://img.shields.io/docsrs/zenpicker?style=flat-square) ![License](https://img.shields.io/crates/l/zenpicker?style=flat-square)

Codec-agnostic picker runtime. Loads a packed MLP from bytes, runs SIMD inference, returns argmin under caller-supplied constraints. Used by zenjpeg / zenwebp / zenavif / zenjxl to translate `(image_features, target_quality, constraints)` into a concrete encoder configuration.

`#![forbid(unsafe_code)]`. `no_std + alloc`. AGPL-3.0-only / Commercial dual license.

---

## Why a separate crate

Each codec ships its own picker model (different config space, different feature schema), but they all want the same runtime:

1. Load a baked binary blob, validate header + schema_hash.
2. Run a small MLP forward pass.
3. Argmin over the outputs, restricted to a caller-supplied mask.
4. Optionally apply additive cost adjustments (caller's ICC profile size, format-specific overhead).

zenpicker owns this. Each codec crate owns its `ConfigSpec` enumeration, constraint API, baked `.bin` model, and the schema declaration for which features it consumes — all things zenpicker can't know.

---

## End-to-end workflow

```
                ┌─────────────────────┐
                │  Pareto sweep       │  ← cargo run zq_pareto_calibrate
                │  120 cfg × 30 zq    │     N images × M cells = pareto.tsv
                │  + zenanalyze feats │     features.tsv
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │  Train teacher      │  ← scripts/zq_bytes_distill.py
                │  HistGB per-config  │     n_configs × HistGradientBoosting
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │  Distill student    │  ← same script, MLP soft targets
                │  small shared MLP   │     ~60 KB f32 / ~30 KB f16
                └──────────┬──────────┘
                           ↓ JSON
                ┌─────────────────────┐
                │  Bake to v1 binary  │  ← tools/bake_picker.py
                │  + manifest.json    │     verifies round-trip via Rust loader
                └──────────┬──────────┘
                           ↓ .bin
                ┌─────────────────────┐
                │  Codec crate        │  ← include_bytes!(..) + ZqConstraints
                │  loads at runtime   │     argmin on every encode decision
                └─────────────────────┘
```

### Phase 1 — Pareto sweep (codec-side)

The codec crate runs a sweep harness that encodes a corpus through every config in the search space, measures bytes + perceptual quality (via zensim or similar), and emits two TSVs:

- `benchmarks/zq_pareto_<DATE>.tsv` — per `(image, size, config, q)` cell: `bytes`, `zensim`, `encode_ms`.
- `benchmarks/zq_pareto_features_<DATE>.tsv` — per `(image, size)`: zenanalyze feature vector.

For zenjpeg the harness is `cargo run --release -p zenjpeg --features 'target-zq trellis' --example zq_pareto_calibrate`. Each codec writes its own.

Per the project-wide sweep discipline (size × quality × mode × content axes; see the `Sweep / Calibration / Source-informing Benchmark Discipline` section of `~/work/claudehints/CLAUDE.md`), the corpus and grid must cover tiny + small + medium + large image sizes and dense quality coverage in the perceptibility band.

### Phase 2 — Train + distill

`scripts/zq_bytes_distill.py` (in the codec repo) does both halves:

1. **Teacher** — per-config HistGradientBoostingRegressor (max_iter=400, max_depth=8). Predicts log-bytes from the simple feature vector. Accuracy ceiling, but bakes to MB-class artifacts.
2. **Student** — single shared MLP (`n_inputs → 64 → 64 → n_configs`) with engineered cross-terms (`zq × feat[i]`, `log_pixels`, polynomials, `icc_bytes`). Trained on the teacher's soft targets, not raw labels — closes most of the gap to the teacher.

Output: `benchmarks/zq_bytes_distill_<DATE>.json` carrying `n_inputs`, `n_outputs`, `feat_cols`, `scaler_mean`, `scaler_scale`, `layers[]` (each with `W` and `b`), and a `config_names` mapping for the manifest.

### Phase 3 — Bake to binary

```bash
python3 tools/bake_picker.py \
    --model benchmarks/zq_bytes_distill_2026-04-29.json \
    --out   models/zenjpeg_picker_v1.bin \
    --dtype f16
```

Emits the v1 binary blob (~30 KB f16 / ~60 KB f32) plus `models/zenjpeg_picker_v1.manifest.json` describing the per-output config metadata. Both ship in the codec crate via `include_bytes!`.

### Phase 4 — Round-trip verify

```bash
python3 tools/bake_roundtrip_check.py \
    --model benchmarks/zq_bytes_distill_2026-04-29.json \
    --dtype f16
```

Runs the Python forward pass against the Rust loader output on a deterministic input. Tolerances tuned per dtype; this is the only way to guarantee the binary format and the Rust loader agree with the bake script.

### Phase 5 — Codec consumes the model

```rust,ignore
// 1. Embed the baked model. Wrap in an aligned struct so the loader
//    can zero-copy borrow weight slices from the bytes.
#[repr(C, align(8))]
struct AlignedModel<const N: usize>([u8; N]);
const MODEL_BYTES: &[u8] = &AlignedModel(*include_bytes!("zenjpeg_picker_v1.bin")).0;

// 2. Load once at startup; hard-fail on schema mismatch.
let model = zenpicker::Model::from_bytes(MODEL_BYTES)?;
assert_eq!(model.schema_hash(), MY_SCHEMA_HASH);

// 3. On each encode: build features + mask, pick.
let mut picker = zenpicker::Picker::new(model);
let features = my_codec::extract_features(&analysis, target_zq);
let mask = my_codec::allowed_configs(&caller_constraints);
let pick = picker.argmin_masked(&features, &mask, None)?
    .expect("at least one config must be allowed");
let cfg = my_codec::CONFIGS[pick];
```

---

## Inputs: scalar features alongside the image features

The picker is not just `bytes(image_features)`. It also takes:

- **target perceptual quality** — `zq` ∈ [0, 100] in zenjpeg; `target_distance` for jxl/avif. The single most important input — the picker is asking "what config minimizes bytes given the user wants this much quality".
- **`log_pixels`** — the image area as `log(width × height)`. Captures the size axis: small images have huge fixed-byte overhead (headers, ICC) per pixel; large images amortize it. Picker decisions differ accordingly.
- **`icc_bytes`** — caller-supplied additive byte cost from their ICC profile (or other metadata). Doesn't affect argmin in a pure log-bytes regressor (additive constants are argmin-invariant), but combined with per-output offsets (e.g., XYB intrinsic ICC vs YCbCr-no-ICC) it can shift the pick.
- **size onehot** — categorical bucket {tiny, small, medium, large} so the model can learn intercept differences without relying purely on the continuous `log_pixels`.

The codec-side feature extraction packs all of these into the `&[f32]` input vector. Order matters — it must match the bake's schema exactly. The schema_hash gates this at load time.

### Engineered cross-terms

The distill student takes engineered features beyond the raw image features:

- `log_pixels`, `log_pixels²`
- `zq_norm`, `zq_norm²`, `zq_norm × log_pixels`
- `zq_norm × feat[i]` for each image feature

These let the small MLP learn the non-linear interactions the per-config HistGB teacher captured naturally. The bake tool's `derive_extra_axes()` documents the exact layout so the schema_hash is stable.

---

## Outputs: bytes + scalar parameter heads (hybrid heads, v0.2)

### Today: pure categorical

In v0.1, every output is "predicted log-bytes for config index `i`". Codec runtime does `argmin_masked()` over allowed indices and picks one. The `config_names` manifest maps each output index to a tuple of discrete encoder settings (color_mode, sub, scan, sa_piecewise, …) — those are baked into the codec's compile-time `CONFIGS` table.

This works when every encoder axis is genuinely discrete. For zenjpeg's 120-cell grid that's almost true.

### v0.2: hybrid heads for continuous control axes

For jxl / avif / webp, **effort** is a single scalar axis (libjxl effort 1–9, libwebp method 0–6, libavif speed 0–10). For zenjpeg, **chroma_quality** and **trellis_lambda** are also continuous. Discretizing them blows up cell count combinatorially without paying back in accuracy.

The right shape: two output heads.

1. **Categorical bytes head** over discrete-only axes (color_mode, sub, scan, sa_piecewise) — small, ~8–16 cells.
2. **Continuous parameter heads** — per categorical bucket, predict optimal scalar values (`chroma_scale`, `lambda`, `effort`) as f32 outputs.

At inference: argmin the categorical head, read off its scalar predictions, **clamp to caller-supplied scalar constraints** (`chroma_scale ∈ [0.8, 1.2]`, `effort ≤ 5`). The MLP itself does the interpolation across training-grid points — that's what gradient descent gives you for free.

Format extension: same v1 binary, just additional output channels. `manifest.json` declares which output indices are categorical-bytes vs scalar-predictions and which scalar axis each predicts. Codec consumers route categorical outputs through `argmin_masked()` and scalar outputs through clamping.

```rust,ignore
// v0.2 sketch:
let pick = picker.predict(&features)?;
let cat_bytes = &pick[..8];                     // 8 categorical cells
let chroma_pred = &pick[8..16];                 // per-cell chroma_scale prediction
let lambda_pred = &pick[16..24];                // per-cell trellis_lambda prediction
let effort_pred = &pick[24..32];                // per-cell effort_level prediction

let cat_idx = mask.argmin_masked(cat_bytes)?;
let cfg = CategoricalSpec::from(cat_idx);
let chroma = chroma_pred[cat_idx].clamp(constraints.chroma_min, constraints.chroma_max);
let lambda = lambda_pred[cat_idx].clamp(constraints.lambda_min, constraints.lambda_max);
let effort = effort_pred[cat_idx]
    .round()
    .clamp(constraints.effort_min as f32, constraints.effort_max as f32) as u8;
```

---

## Constraints

### Categorical constraints — `AllowedMask`

`AllowedMask<'a>` is a `&[bool]` over `n_outputs`. Codec crate translates user intent into the bitmask:

```rust,ignore
pub struct ZqConstraints {
    color_mode: Option<ColorMode>,        // require_xyb / require_ycbcr
    forbid_trellis: bool,
    max_chroma_subsampling: Option<Subsampling>,
    forbid_progressive_optimize: bool,
}

impl ZqConstraints {
    fn matches(&self, spec: &ConfigSpec) -> bool {
        if let Some(cm) = self.color_mode && spec.color_mode != cm { return false; }
        if self.forbid_trellis && spec.hybrid_lambda.is_some() { return false; }
        if let Some(max_sub) = self.max_chroma_subsampling
            && spec.subsampling.h_factor() < max_sub.h_factor() { return false; }
        true
    }

    pub fn allowed_mask(&self) -> [bool; ZQ_N_CONFIGS] {
        core::array::from_fn(|i| self.matches(&CONFIGS[i]))
    }
}
```

Same shape for **jxl** (effort 1–9), **webp** (effort 0–9, method 0–6), **avif** (speed 0–10): effort/speed is a field in the `ConfigSpec`. `Constraints::max_effort(u8)` becomes the bitmask predicate `spec.effort <= max`.

### Scalar constraints — clamp at output time (v0.2)

For continuous axes, the constraint is a `[min, max]` clamp on the picker's scalar prediction:

```rust,ignore
pub struct WebpConstraints {
    pub max_effort: Option<u8>,           // 0..=6
    pub forbid_lossless: bool,
    pub min_chroma_quality: Option<f32>,
}

// At inference (sketch, v0.2):
let cat_idx = picker.argmin_masked(...)?;
let effort_raw = scalar_outputs[cat_idx][effort_axis];
let effort = constraints
    .max_effort
    .map(|m| effort_raw.clamp(0.0, m as f32))
    .unwrap_or(effort_raw)
    .round() as u8;
```

The picker predicts the *Pareto-optimal* scalar value for the given image + target quality. Caller's clamp is the *constraint*. They're separate concerns: the picker says "the optimal effort for this image is 4.7"; the caller says "I'll accept anything ≤ 5". Result: `min(4.7, 5) = 4.7 → round → 5` (or whatever the codec rounds to).

### Cost adjustments

`CostAdjust` lets callers add (a) a global additive byte cost (their planned ICC / EXIF / XMP overhead) and (b) per-output additive bytes (format-specific overhead the model couldn't learn).

```rust,ignore
let adjust = zenpicker::CostAdjust {
    additive_bytes: caller_icc.len() as f32 + caller_exif.len() as f32,
    per_output_offset: Some(&MY_FORMAT_OVERHEAD_TABLE),
};
let pick = picker.argmin_masked(&features, &mask, Some(adjust))?;
```

Argmin in log-bytes space ignores additive constants on its own — they only matter when combined with per-output offsets that differ between configs (e.g., XYB intrinsic 720-byte ICC vs YCbCr-no-ICC).

---

## Binary format (v1)

Little-endian throughout. Header is 32 bytes; subsequent sections are 4-byte aligned.

```text
Header (32 bytes):
  [0..4]   magic: b"ZNPK"
  [4..6]   version: u16 = 1
  [6..8]   header_size: u16 = 32
  [8..12]  n_inputs: u32
  [12..16] n_outputs: u32
  [16..20] n_layers: u32
  [20..28] schema_hash: u64
  [28..32] flags: u32 (reserved)

Scaler section:
  scaler_mean[n_inputs]:  f32
  scaler_scale[n_inputs]: f32

Per-layer (× n_layers):
  in_dim: u32
  out_dim: u32
  activation: u8        (0=Identity, 1=ReLU)
  weight_dtype: u8      (0=F32, 1=F16)
  reserved: [u8; 2]
  weights: row-major (in_dim major), in_dim*out_dim values
  biases: f32, out_dim values
```

Weights are stored input-major: `W[i * out_dim + o]` is the contribution from input `i` to output `o`. This layout streams `out_dim` outputs in chunks of 8 across each input row — what `magetypes::f32x8` wants when SIMD lands.

f16 storage is built in (no feature gate, no `half` dep) — the conversion is ~15 lines of integer bit math in `inference::f16_bits_to_f32`. Halves the model size at ~no accuracy cost.

---

## Schema hashing

The model header carries a `schema_hash: u64`. Codec consumers should hash their compile-time feature schema and compare to this on load. Mismatch means the model was baked against a different feature order or set; load with hard error rather than silently producing nonsense predictions.

```rust,ignore
const SCHEMA: &[&str] = &[
    "feat_variance", "feat_edge_density", /* … */ "log_pixels",
    "zq_norm", "zq_norm_sq", "zq_norm_x_log_pixels",
    "zq_x_feat_variance", /* cross terms */
    "icc_bytes",
];
const SCHEMA_HASH: u64 = compute_hash(SCHEMA, "zenpicker.v1.shared-mlp.distill+icc");
```

The bake tool computes the same hash from `feat_cols + extra_axes + SCHEMA_VERSION_TAG`. Bumping `SCHEMA_VERSION_TAG` forces every codec to re-bake.

---

## Performance

Inference cost is dominated by the layer-0 matmul. For a 48-input × 64-output layer 0 with 64×64 hidden and 64×120 output, total ≈ 15K weights → ~15µs scalar Rust on a Ryzen 9 7950X. f32x8 SIMD via magetypes (v0.2) cuts that to ~3µs.

This is negligible compared to image analysis (~ms) and encoding (~tens of ms). The picker's contribution to per-encode latency is in the noise.

Memory: 30 KB (f16) or 60 KB (f32) embedded; one prediction call allocates nothing past the `Picker::new()` scratch buffers.

---

## What zenpicker is **not**

- Not a training framework. Use sklearn / PyTorch / whatever; emit the binary format from `tools/bake_picker.py`.
- Not codec-aware. The crate has zero `cfg(feature = "jpeg")`, no codec-specific shortcuts.
- Not an inference engine for arbitrary architectures. Strictly: scaler → series of (linear, activation) layers → output. Adding new activations or layer types is additive in the format.
- Not the analyzer. zenpicker doesn't know how to extract features from pixels; that's [zenanalyze](https://crates.io/crates/zenanalyze)'s job.

---

## Roadmap

- **v0.1** (now): scalar inference, f32 + f16 storage built in, parse + validate v1 format. Categorical-only outputs.
- **v0.2**: `#[magetypes]`-dispatched matmul (AVX-512 / AVX2 / NEON / WASM SIMD128 / scalar). 8-wide f16 → f32 via F16C / FCVT through magetypes. **Hybrid heads** for continuous control axes (effort, chroma_quality, lambda).
- **v0.3**: i8 quantized weights option (per-row scale) for the case where ~50 KB still isn't small enough.

The format header has reserved fields for future expansion. New `weight_dtype` values, new activations, and new layer types are additive.

---

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See [LICENSE-AGPL3](../LICENSE-AGPL3) and [LICENSE-COMMERCIAL](../LICENSE-COMMERCIAL).
