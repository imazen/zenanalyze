# zenpicker ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpicker?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpicker?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpicker) ![docs.rs](https://img.shields.io/docsrs/zenpicker?style=flat-square) ![License](https://img.shields.io/crates/l/zenpicker?style=flat-square)

Codec-agnostic picker runtime. Loads a packed MLP from bytes, runs SIMD inference, returns argmin under caller-supplied constraints. Used by zenjpeg / zenwebp / zenavif / zenjxl to translate `(image_features, target_quality, constraints)` into a concrete encoder configuration.

`#![forbid(unsafe_code)]`. `no_std + alloc`. AGPL-3.0-only / Commercial dual license.

## Documentation map

This README is the canonical entry point — every concept (workflow, output layout, constraints, safety profiles, binary format, performance, roadmap) is documented here. Two subsidiary docs exist for content that's too long to inline:

| Doc | Audience |
|---|---|
| **[README.md](README.md)** (this file) | Anyone reading the picker code or designing a new model. Single source of truth for API, format, training shape, safety profiles, codec-side patterns, and roadmap |
| **[FOR_NEW_CODECS.md](FOR_NEW_CODECS.md)** | Tutorial. Walk a new codec from "I have a config grid" to a shipped bake in ~30 min. Skip if you've integrated a codec before |
| **[SAFETY_PLANE.md](SAFETY_PLANE.md)** | Codec-implementation deep dive. Two-shot rescue protocol, RescuePolicy field layout, calibration questions still open |

Reference code + tools:

| File | What it is |
|---|---|
| **[examples/hybrid_heads_codec_sketch.rs](examples/hybrid_heads_codec_sketch.rs)** | Runnable codec-side reference: `CELLS` table, constraint mask, `argmin_masked_in_range` + scalar clamp |
| **[examples/load_baked_model.rs](examples/load_baked_model.rs)** | Smoke test for the loader on any bake artifact |
| **[examples/zenjpeg_picker_config.py](examples/zenjpeg_picker_config.py)** | Reference codec config the training pipeline imports — paths, KEEP_FEATURES, ZQ_TARGETS, `parse_config_name`. New codecs copy and edit |
| **[tools/](tools/README.md)** | Codec-agnostic training pipeline (`train_hybrid.py`, `train_distill.py`, `feature_ablation.py`, …). Each script imports `_picker_lib.py` and a codec config module — see [tools/README.md](tools/README.md) |
| **[../tools/bake_picker.py](../tools/bake_picker.py)** | sklearn JSON → v1 binary + manifest. Codec-agnostic |
| **[../tools/bake_roundtrip_check.py](../tools/bake_roundtrip_check.py)** | Verify Rust loader matches numpy reference forward pass |

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
                │  Train teacher      │  ← zenpicker/tools/train_hybrid.py
                │  HistGB per-cell    │     N cells × 3 heads, parallel via joblib
                │  (parallel)         │     ~30 s on 16 cores
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

The training pipeline lives at [`zenpicker/tools/`](tools/README.md) — codec-agnostic. Each codec writes a small **codec config module** declaring its TSV paths, feature subset, target_zq grid, and config-name parser; the training scripts import it via `--codec-config <module-name>`.

```bash
PYTHONPATH=<zenanalyze>/zenpicker/examples:<zenanalyze>/zenpicker/tools \
    python3 <zenanalyze>/zenpicker/tools/train_hybrid.py \
        --codec-config zenjpeg_picker_config
```

Two phases inside the script:

1. **Teacher** — per-cell HistGradientBoostingRegressor (max_iter=400, max_depth=8). Predicts log-bytes + per-cell scalar values from the simple feature vector. Trained in parallel via joblib (~30 s for 36 models on a 16-core box vs ~25 min serial).
2. **Student** — single shared MLP (`n_inputs → 128 → 128 → 3*N_cells`) with engineered cross-terms (`zq × feat[i]`, `log_pixels`, polynomials, `icc_bytes`). Trained on the teacher's soft targets, not raw labels — closes most of the gap to the teacher.

Output JSON carries `n_inputs`, `n_outputs`, `feat_cols`, `scaler_mean`, `scaler_scale`, `layers[]` (each with `W` and `b`), and a `hybrid_heads_manifest` declaring the categorical-vs-scalar layout. See [tools/README.md](tools/README.md) for the full file map and the codec-config contract.

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
// 1. Embed the baked hybrid-heads model. Wrap in an aligned struct so the
//    loader can zero-copy borrow weight slices from the bytes.
#[repr(C, align(8))]
struct AlignedModel<const N: usize>([u8; N]);
const MODEL_BYTES: &[u8] =
    &AlignedModel(*include_bytes!("zenjpeg_picker_v2.0_hybrid.bin")).0;

// 2. Load once at startup; hard-fail on schema mismatch.
let model = zenpicker::Model::from_bytes(MODEL_BYTES)?;
assert_eq!(model.schema_hash(), MY_SCHEMA_HASH);

// 3. On each encode: build features + mask, pick categorical cell,
//    read scalar parameter predictions, clamp to caller constraints.
let mut picker = zenpicker::Picker::new(model);
let features = my_codec::extract_features(&analysis, target_zq);
let mask = my_codec::allowed_cells(&caller_constraints);

let cell_idx = picker
    .argmin_masked_in_range(&features, (0, N_CELLS), &mask, None)?
    .expect("at least one cell allowed");
let out = picker.predict(&features)?;     // re-read for scalar heads
let chroma = out[N_CELLS + cell_idx].clamp(c_min, c_max);
let lambda = out[2 * N_CELLS + cell_idx].clamp(l_min, l_max);
let cfg = build_encoder_config(CELLS[cell_idx], chroma, lambda);
```

See [`examples/hybrid_heads_codec_sketch.rs`](examples/hybrid_heads_codec_sketch.rs) for a complete runnable codec-side reference (constraint translation, mask building, scalar clamp).

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

## Outputs: hybrid heads (recommended) vs pure categorical (legacy)

### Recommended: hybrid heads

The picker has two kinds of outputs in one flat vector:

1. **Categorical bytes head** over discrete-only axes (color_mode, sub, scan, sa_piecewise, trellis_on/off) — typically 8–16 cells.
2. **Continuous parameter heads** — per categorical cell, predict optimal scalar values (chroma_quality / trellis_lambda / effort / speed) as f32 outputs.

At inference: argmin the categorical head over the allowed mask, read off its scalar predictions, **clamp to caller-supplied scalar constraints** (`chroma_scale ∈ [0.8, 1.2]`, `effort ≤ 5`). The MLP itself does the interpolation across training-grid points — that's what gradient descent gives you for free.

**Why hybrid is the default:** the zenjpeg v2.0 hybrid model (12 cells × 3 heads = 36 outputs) achieves **2.76% mean overhead** vs the v1.1 pure-categorical 120-cell model's **8.20%** — 3× tighter on the held-out corpus, 5× higher argmin accuracy (52% vs 10.5%). The categorical space is 10× smaller so individual cell picks are confident; the within-cell scalar choice (chroma_scale, lambda) becomes a smooth f32 regression instead of buried under a categorical noise floor.

For codecs with intrinsically scalar control axes — **jxl** (effort 1–9), **webp** (effort 0–9, method 0–6), **avif** (speed 0–10) — hybrid heads are the only sensible shape. Discretizing those axes into the categorical grid blows up cell count combinatorially without paying back in accuracy.

`manifest.json` declares which output indices are categorical-bytes vs scalar-predictions, the cell list, and which scalar axis each prediction maps to. Codec consumers slice `predict()` for the scalar reads and use `Picker::argmin_masked_in_range()` for the categorical pick.

```rust,ignore
// Recommended pattern (zenjpeg v2.0-shaped, 12 cells × 3 heads):
const N_CELLS: usize = 12;

let out = picker.predict(&features)?;
let bytes_log = &out[..N_CELLS];
let chroma_pred = &out[N_CELLS..2 * N_CELLS];
let lambda_pred = &out[2 * N_CELLS..3 * N_CELLS];

let cell_idx = picker
    .argmin_masked_in_range(&features, (0, N_CELLS), &mask, None)?
    .expect("at least one cell allowed");

let cfg = CELLS[cell_idx];                        // (color, sub, trellis_on, sa)
let chroma_scale = chroma_pred[cell_idx]
    .clamp(constraints.chroma_min, constraints.chroma_max);
let lambda = if cfg.trellis_on {
    Some(lambda_pred[cell_idx].clamp(constraints.lambda_min, constraints.lambda_max))
} else {
    None
};
```

A complete codec-side reference lives in [`examples/hybrid_heads_codec_sketch.rs`](examples/hybrid_heads_codec_sketch.rs).

---

## Safety profiles: `size_optimal` vs `zensim_strict`

The default model minimizes **mean bytes** subject to "achieves target_zq on average." That's the right shape for most consumer traffic, but for SLA-bound deployments — image proxies that promise minimum perceptual quality on every request — you want a model whose pick **reliably** hits the target, paying ~10-15 % extra bytes as the cost of that reliability.

zenpicker's binary format and the bake tool support shipping **two model variants per codec, side by side**, so the codec picks at session start based on caller intent:

| Profile | Loss | Reach-rate gate | Effective behavior | When to ship it |
|---|---|---|---|---|
| **`size_optimal`** (default) | mean log-bytes | none | Picks smallest *expected* bytes that probably hits target. p99 zensim shortfall unbounded — picker may miss target on adversarial / out-of-distribution images | Default consumer traffic |
| **`zensim_strict`** (opt-in) | quantile loss at p99 of bytes given target_zq | mask cells with empirical reach rate < 0.99 at target_zq | Picks smallest *worst-case-safe* bytes; cells that historically miss target are excluded before argmin | Quality-SLA traffic, contractually-bound proxies, archival pipelines |

### Why two models is the right number

A "constraint-flavored" third variant doesn't pay back — caller constraints (forbid_xyb, max_subsampling, …) are a *runtime* axis handled by the [`AllowedMask`](#categorical-constraints--allowedmask), not a *training* axis. Held-out ablation showed dropping XYB or trellis as a runtime constraint costs +0.16 pp on the picker, well within the noise floor.

The `size_optimal` ↔ `zensim_strict` split is *real* because the two models train against different objectives. You can't get strict-zensim behavior by masking — the underlying loss function is different.

### What changes between the two bakes

Same architecture, same input schema, same output layout. Differences are entirely on the training side:

1. **Loss function.** `size_optimal` uses mean log-bytes regression (current). `zensim_strict` uses [pinball loss](https://en.wikipedia.org/wiki/Quantile_regression) at q=0.99 of bytes given target_zq, so the predicted byte cost *is* the worst-case-safe estimate.
2. **Reach-rate gate** baked into the manifest. At training time, for each `(cell, target_zq)` pair, compute the empirical reach rate across the training corpus — fraction of images where that cell achieved ≥ target_zq. ~12 cells × 30 zq targets × 1 byte = 360-byte table per model, fits in the manifest's open extension area. Codec-side helper AND's the gate into the constraint mask before argmin runs.
3. **Calibration.** Held-out **p99 zensim shortfall** (worst-1% miss vs target) is computed at bake time and recorded in the manifest. Codecs surface it to callers so they know what reliability to advertise.

### Codec-side API

```rust,ignore
pub enum PickerProfile { SizeOptimal, ZensimStrict }

const PICKER_SIZE_OPTIMAL: &[u8]  = include_bytes!("zenjpeg_picker_size_optimal_v2.0.bin");
const PICKER_ZENSIM_STRICT: &[u8] = include_bytes!("zenjpeg_picker_zensim_strict_v2.0.bin");

pub fn load_picker(profile: PickerProfile) -> Result<Picker<'static>, PickerError> {
    let bytes = match profile {
        PickerProfile::SizeOptimal  => PICKER_SIZE_OPTIMAL,
        PickerProfile::ZensimStrict => PICKER_ZENSIM_STRICT,
    };
    let model = Model::from_bytes(bytes)?;
    debug_assert_eq!(model.schema_hash(), MY_SCHEMA_HASH);
    Ok(Picker::new(model))
}

// At pick time, the strict profile's manifest carries a reach-rate
// table the codec AND's into the constraint mask:
let mut mask = constraints.allowed_mask();
if let Some(reach_table) = manifest.reach_rate_for(target_zq) {
    for (i, allowed) in mask.iter_mut().enumerate() {
        *allowed &= reach_table[i] >= 0.99;
    }
}
let cell_idx = picker.argmin_masked_in_range(&features, (0, N_CELLS), &AllowedMask::new(&mask), None)?;
```

The size and runtime cost of carrying both bakes is small (~100 KB embedded total per codec, no inference-path changes — it's just two `Picker` instances). The codec exposes `PickerProfile` on its public encode API; imageflow / proxy operators flip per-request based on SLA requirements.

### Caller-facing semantics

- **`size_optimal`** advertises: "P50 of achieved zensim ≥ target_zq across web traffic; some images may miss by up to ~5 pp."
- **`zensim_strict`** advertises: "P99 of achieved zensim ≥ target_zq − 1; ~10-15 % extra bytes vs `size_optimal` average; cells with bad reach gates are masked out."

Combined with the [safety plane's two-shot rescue](SAFETY_PLANE.md), `zensim_strict` becomes the high-reliability tier; `size_optimal` + rescue covers most cases at lower mean cost.

---

### Legacy: pure categorical (v0.1, v1.0/v1.1 zenjpeg bakes)

In the categorical-only shape, every output is "predicted log-bytes for config index `i`". Codec runtime does `argmin_masked()` over allowed indices and picks one. The `config_names` manifest maps each output index to a tuple of discrete encoder settings — baked into the codec's compile-time `CONFIGS` table.

This works only when every encoder axis is genuinely discrete *and* the cell count stays small. zenjpeg's 120-cell pure-categorical bakes (v1.0, v1.1) ship as a fallback for deployments that haven't migrated to v2.0 yet, but new codec integrations should target hybrid heads from day one.

```rust,ignore
// Legacy pattern — v1.x bakes only:
let mask = my_codec::allowed_configs(&caller_constraints);  // [bool; 120]
let pick = picker.argmin_masked(&features, &mask, None)?
    .expect("at least one config allowed");
let cfg = my_codec::CONFIGS[pick];   // 120-entry table
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

### Scalar constraints — clamp at output time

For continuous axes, the constraint is a `[min, max]` clamp on the picker's scalar prediction:

```rust,ignore
pub struct WebpConstraints {
    pub max_effort: Option<u8>,           // 0..=6
    pub forbid_lossless: bool,
    pub min_chroma_quality: Option<f32>,
}

// At inference:
let cell_idx = picker
    .argmin_masked_in_range(features, (0, N_CELLS), &mask, None)?
    .expect("at least one cell allowed");
let out = picker.predict(features)?;
let effort_raw = out[N_CELLS + cell_idx];
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

| Status | Item |
|---|---|
| ✅ v0.1 | Scalar inference, f32 + f16 storage built in, parse + validate v1 format |
| ✅ v0.1 | **Hybrid heads** (categorical bytes + scalar parameter outputs) via `Picker::argmin_masked_in_range()` — recommended shape for new bakes |
| ✅ v0.1 | Codec-agnostic bake tool (`tools/bake_picker.py`) — codec declares `extra_axes` + `schema_version_tag` in JSON, no script edits needed |
| 🔜 v0.1 | **`zensim_strict` profile**: pinball-loss training + reach-rate manifest gate. Pairs with `size_optimal` (current) so codecs ship two bakes side by side |
| 🔜 v0.1 | **Two-shot rescue** loop in zenpicker (codec-agnostic; codec injects verify + rescue strategy). See [SAFETY_PLANE.md](SAFETY_PLANE.md) |
| ⏳ v0.2 | `#[magetypes]`-dispatched matmul (AVX-512 / AVX2 / NEON / WASM SIMD128 / scalar). 8-wide f16 → f32 via F16C / FCVT |
| ⏳ v0.3 | i8-quantized weights option (per-row scale) for the case where ~50 KB still isn't small enough |
| ⏳ v0.3 | Generational re-encode picker (round-trip JPEG-source case): see [imazen/zenanalyze#13](https://github.com/imazen/zenanalyze/issues/13) |

The format header has reserved fields for future expansion. New `weight_dtype` values, new activations, and new layer types are all additive — bakes against future versions still load on older runtimes (with the unsupported field flagged) until the format major-version bumps.

### Shipped zenjpeg models

Same architecture, same input schema, different training objectives. Codec links all three at compile time and picks per request:

| Bake | Size | Mean overhead | Argmin acc | Profile | Recommended |
|---|---:|---:|---:|---|---|
| `zenjpeg_picker_v1.0_19feat.bin` | 31 KB | 7.20 % | 13.8 % | size_optimal | legacy / broadest 19-feature schema |
| `zenjpeg_picker_v1.1_8feat.bin` | 28 KB | 8.20 % | 10.5 % | size_optimal | legacy / reduced 8-feature schema |
| **`zenjpeg_picker_v2.0_hybrid.bin`** | **50 KB** | **2.76 %** | **52.0 %** | size_optimal | **default** — hybrid heads, 8-feature schema |
| `zenjpeg_picker_v2.0_zensim_strict.bin` (planned) | ~50 KB | (TBD: ~12 % expected) | — | zensim_strict | SLA-bound traffic — see [Safety profiles](#safety-profiles-size_optimal-vs-zensim_strict) |

Codec consumers default to v2.0 `size_optimal`; v1.x bakes remain in tree as a fallback for deployments that pinned before the hybrid rollout. The `zensim_strict` bake ships once trained.

### What's not blocking lock-in

Per [Safety plane](SAFETY_PLANE.md), the codec-side rescue loop is the next implementation gap. It re-uses zenjpeg's existing `ZqTarget` iteration scaffolding, so the codec change is small. Open questions captured in SAFETY_PLANE.md: rescue threshold calibration (held-out p99 zensim shortfall), pre-filter ROC, always-verify cost budget at 1MP / 4K.

For the per-codec burden to stay low, the framework lives in zenpicker (codec-agnostic two-shot loop, generic `verify` + `rescue_pick` callbacks); the codec ships a thin glue layer per [FOR_NEW_CODECS.md](FOR_NEW_CODECS.md).

---

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See [LICENSE-AGPL3](../LICENSE-AGPL3) and [LICENSE-COMMERCIAL](../LICENSE-COMMERCIAL).
