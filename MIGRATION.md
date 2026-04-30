# Migration: zenpicker (Rust) → zenpredict + zenpicker (Python)

`zenpicker` was scaffolded as a Rust crate inside this repo for the codec
config picker runtime. It was **never published** to crates.io — the only
in-tree consumer was `zenwebp` behind its `picker` feature flag.

`zenpredict` is the new Rust runtime. It supersedes the unpublished
`zenpicker` Rust crate, generalizes the API surface so `zensim`'s V0.4
perceptual scorer can share it, and adopts a forwards-compatible binary
format (ZNPR v2) with a typed-TLV metadata blob.

`zenpicker` keeps the name — for the **Python training pipeline** that lives
at [`zenpicker/tools/`](zenpicker/tools/) (pareto sweep, teacher fit, distill,
ablation, holdout probes, safety reports, `.bin` bake). Versioned
independently of the Rust runtime.

The contract between the two is the **ZNPR v2 binary format**.

---

## What moved where

| Item | Was | Is now |
|---|---|---|
| Rust runtime crate name | `zenpicker` (path-dep, unpublished) | [`zenpredict`](zenpredict/) (`0.1.x`, dual-licensed) |
| Binary magic | `ZNPK` | `ZNPR` |
| Binary format version | v1 (32-byte header, positional layout) | v2 (128-byte `#[repr(C)]` header, offset-table sections, TLV metadata) |
| `Picker::argmin_masked(features, mask, adjust)` | `CostAdjust { additive_bytes, per_output_offset }` | `Predictor::argmin_masked(features, mask, transform, offsets)` with `ScoreTransform::{Identity, Exp}` and `ArgminOffsets { uniform, per_output }` |
| `reach_gate_mask(rates, threshold, out)` | picker-domain | `argmin::threshold_mask(rates, threshold, out)` (generic) |
| `RescuePolicy.rescue_threshold_pp` | field name | `RescuePolicy.rescue_threshold` (units unchanged — points on the codec's quality scale) |
| `PickerError` | not `#[non_exhaustive]` | `PredictError` is `#[non_exhaustive]` from day 1 |
| Loader convenience | manual `assert_eq!(model.schema_hash(), MY_HASH)` | `Model::from_bytes_with_schema(bytes, expected_hash)` |
| Feature bounds + OOD | feature-bounds in manifest JSON, `first_out_of_distribution` lived in the codec | top-level `feature_bounds` Section in the bin; `FeatureBound` + `first_out_of_distribution` are first-class in zenpredict |
| Reach-rate table | top-level Section was planned | Now lives in metadata under `zenpicker.reach_rates` + `zenpicker.reach_zq_targets` (codec copies out at startup) |
| Bake metadata | sibling JSON manifest | Embedded in the `.bin` via the typed-TLV metadata blob (utf8 / numeric / bytes); `manifest.json` keeps only training-side provenance not consumed at runtime |

`zensim` V0.4's previously-vendored copy of the v1 MLP runtime moves to the
same `zenpredict` dependency; the on-disk format change requires a one-shot
re-bake of the V0.4 weights through the v2 baker.

---

## Cargo migration

### `Cargo.toml`

```diff
 [dependencies]
-zenpicker = { path = "../zenanalyze/zenpicker", optional = true }
+zenpredict = { path = "../zenanalyze/zenpredict", optional = true }

 [features]
-picker = ["dep:zenpicker"]
+picker = ["dep:zenpredict"]
```

zenpredict has two cargo features (both default-on for codec consumers):

| Feature | What it gates |
|---|---|
| `std` | `std::error::Error` impls and `f32::exp` for `ScoreTransform::Exp` |
| `bake` | Rust-side ZNPR v2 byte-stream composer for round-trip tests / placeholder weights |

`zensim` and other consumers that only need the inference core can opt out of
`bake`:

```toml
zenpredict = { version = "0.1", default-features = false, features = ["std"] }
```

The picker / rescue / bounds / metadata API surface is **not** behind feature
flags — it's all generic decision math, useful to codecs and metrics crates
alike.

---

## Source migration

### Loading a model

```diff
-let model = zenpicker::Model::from_bytes(MODEL_BYTES)?;
-assert_eq!(model.schema_hash(), MY_SCHEMA_HASH);
-let mut picker = zenpicker::Picker::new(model);
+let model = zenpredict::Model::from_bytes_with_schema(MODEL_BYTES, MY_SCHEMA_HASH)?;
+let mut predictor = zenpredict::Predictor::new(model);
```

**Alignment note:** zenpredict requires the input bytes to be at least 4-byte
aligned for f32 weight slices (8-byte-aligned recommended). Wrap your
`include_bytes!` blob in an `#[repr(C, align(16))]` struct — see the example
in [`zenpredict/README.md`](zenpredict/README.md). The previous v1 loader
silently allocated to copy on misalignment; the v2 loader fails loudly with
`PredictError::SectionMisaligned`.

### Argmin with offsets

```diff
-let pick = picker.argmin_masked(
-    &features,
-    &mask,
-    Some(zenpicker::CostAdjust {
-        additive_bytes: caller_icc_size as f32,
-        per_output_offset: Some(&FORMAT_OVERHEAD),
-    }),
-)?;
+use zenpredict::{ArgminOffsets, ScoreTransform};
+let pick = predictor.argmin_masked(
+    &features,
+    &mask,
+    ScoreTransform::Exp,                       // log-bytes regressor → linear bytes
+    Some(&ArgminOffsets {
+        uniform: caller_icc_size as f32,
+        per_output: Some(&FORMAT_OVERHEAD),
+    }),
+)?;
```

Two changes:

1. The implicit `exp` clamp that v1 did inside `argmin_masked` when
   `CostAdjust` was `Some` is now an explicit `ScoreTransform::Exp` argument.
   Pass `ScoreTransform::Identity` for raw-score argmin (most non-codec uses);
   pass `ScoreTransform::Exp` for codec log-bytes regressors.
2. `CostAdjust { additive_bytes, per_output_offset }` →
   `ArgminOffsets { uniform, per_output }`. Same shape, generic naming —
   nothing in this type pretends the values are bytes.

### Hybrid-heads (categorical bytes + scalar prediction heads)

```diff
-let cell_idx = picker.argmin_masked_in_range(&features, (0, N_CELLS), &mask, None)?
-    .expect("at least one cell allowed");
+let cell_idx = predictor.argmin_masked_in_range(
+    &features,
+    (0, N_CELLS),
+    &mask,
+    ScoreTransform::Exp,
+    None,
+)?.expect("at least one cell allowed");
-let out = picker.predict(&features)?;
+let out = predictor.predict(&features)?;
 let chroma = out[N_CELLS + cell_idx].clamp(c_min, c_max);
 let lambda = out[2 * N_CELLS + cell_idx].clamp(l_min, l_max);
```

### Top-K + confidence

```diff
-let top2 = picker.argmin_masked_top_k::<2>(&features, &mask, None)?;
-let (best, gap) = picker.pick_with_confidence(&features, &mask, None)?
+let top2 = predictor.argmin_masked_top_k::<2>(
+    &features, &mask, ScoreTransform::Exp, None,
+)?;
+let (best, gap) = predictor.pick_with_confidence(
+    &features, &mask, ScoreTransform::Exp, None,
+)?
     .expect("mask non-empty");
```

### Reach-rate gate

```diff
-let mut gate = [false; N_CELLS];
-zenpicker::reach_gate_mask(&rates, threshold, &mut gate);
+let mut gate = [false; N_CELLS];
+zenpredict::threshold_mask(&rates, threshold, &mut gate);
 for (i, allowed) in mask.iter_mut().enumerate() {
     *allowed &= gate[i];
 }
```

Codec consumers read `rates` from
`metadata.get_numeric("zenpicker.reach_rates")` (and pair them with
`metadata.get_numeric("zenpicker.reach_zq_targets")`) at startup, then keep
the parsed `&[f32]` for per-encode use. The previous v1 layout shipped these
in a sibling `manifest.json`; v2 embeds them.

### Rescue policy

```diff
-use zenpicker::rescue::{RescuePolicy, RescueStrategy, RescueDecision, should_rescue};
-let policy = RescuePolicy::default();
+use zenpredict::{RescuePolicy, RescueStrategy, RescueDecision, should_rescue};
+let mut policy = RescuePolicy::default();
+// `RescuePolicy` is `#[non_exhaustive]`; construct via `Default` then
+// mutate fields, or just take the default.
+policy.rescue_threshold = 3.0;
```

The field rename `rescue_threshold_pp` → `rescue_threshold` drops the
documentation suffix from the field name (units unchanged — points on the
codec's quality scale). The non-exhaustive struct attribute means external
callers must construct via `Default::default()` and assign fields, rather
than struct-init.

### Feature bounds + OOD

```diff
-// v1: codec compiled in its own FEATURE_BOUNDS table:
-const FEATURE_BOUNDS: &[FeatureBounds] = &[ ... ];
-if let Some(idx) = first_out_of_distribution(&features, FEATURE_BOUNDS) { ... }
+// v2: bounds ship in the .bin; loader exposes them as a borrowed slice:
+let bounds = model.feature_bounds();   // &[zenpredict::FeatureBound]
+if let Some(idx) = zenpredict::first_out_of_distribution(&features, bounds) { ... }
```

If `feature_bounds` is absent in the bin (`Section.len == 0`), the loader
returns an empty slice; OOD detection becomes a no-op for that bake.

### Metadata access

```rust,ignore
use zenpredict::keys;

let md = model.metadata();
let profile: u8 = md.get_pod(keys::PICKER_PROFILE).unwrap_or(0);
let bake_name = md.get_utf8(keys::BAKE_NAME).unwrap_or("(unnamed)");
let metrics: [f32; 3] = md.get_pod(keys::CALIBRATION_METRICS).unwrap_or([0.0; 3]);
let cell_config = md.get_bytes("zenjpeg.cell_config")?;  // codec-private
```

Standard zenpredict-defined keys live under `zenpicker.*` (because the
training side that emits them is the Python `zenpicker` pipeline — namespace
matches the producer, not the consumer). Codec-private keys live under
`<codec>.*` (`zenjpeg.cell_config`, `zenwebp.method_grid`, …). The loader
hands these back as opaque `&[u8]`; the codec deserializes its own format.

---

## Bake-side migration (Python)

The Python tooling at [`zenpicker/tools/`](zenpicker/tools/) is being updated
to emit ZNPR v2 instead of v1. Tracked separately from this Rust-runtime
move; the trainer continues to emit v1 until the v2 baker lands. Until then,
old v1 bakes need to be rebaked to load on zenpredict — `zenpredict::Model::from_bytes`
fails fast with `PredictError::UnsupportedVersion` on a v1 file.

The training side (loss functions, teacher/distill split, ablation,
calibration, safety reports, manifest emission) is unaffected by the Rust
runtime move — the only change is the bake step's output format.

---

## Timeline / coordination

| Status | Item |
|---|---|
| ✅ Done | `zenpredict` v0.1.0 scaffolded, 75 tests passing, ZNPR v2 format documented |
| ⏳ In flight | `tools/bake_picker.py` → emit ZNPR v2 instead of v1 |
| ⏳ In flight | `zenwebp` swap from `zenpicker` to `zenpredict` (single Cargo consumer) |
| ⏳ Pending | `zensim` V0.4 swap to `zenpredict` (currently has a vendored v1 mlp/) |
| ⏳ Pending | Delete `zenanalyze/zenpicker/src/` Rust shell once `zenwebp` no longer references it |
| 🚫 No-op | crates.io yank — zenpicker was never published, no external consumers |

Each consumer migration is a separate PR; they don't have to land together
because the v2 loader runs alongside any v1 bakes the codecs haven't moved
yet. (Strictly speaking: a v1 bake won't load through the v2 loader, but
that's a fail-loud at startup, not a silent regression — every codec
ships its bake bundled with its build, so a single PR per codec covers the
swap and the bake regen together.)

---

## Why the rename, why now

Three forces converged:

1. **`zenpicker` (Rust crate) was misnamed for what it actually does.** It
   was 80% generic MLP runtime + decision math, 20% codec-flavored offset
   semantics. Calling it a "picker" only made sense from one consumer's
   perspective; `zensim` couldn't reuse it without renaming all the picker-
   flavored types in error messages and docs.
2. **The v1 binary format had a forwards-compat hole** — per-layer block
   had no `layer_header_size`, no metadata section embedded in the bin, no
   typed value system for arbitrary key/value annotations. v2 fixes all
   three with a `#[repr(C)]` Header + offset table + TLV metadata blob.
3. **The Python training pipeline deserves the `zenpicker` name** —
   `pareto_sweep` + `train_hybrid` + `bake_picker` + `feature_ablation` +
   `safety_report` are unmistakably "the picker workflow," and the team
   says "I'm running zenpicker" when they mean those scripts. Letting Rust
   own the name was confusing.

`zenpicker` was never published; nobody outside this repo can be affected.
The swap is a one-PR-per-consumer affair and the format change ships
alongside.
