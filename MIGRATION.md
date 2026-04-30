# Migration: zenpicker (Rust shell, unpublished) → zenpredict + zenpicker (meta-picker) + zentrain

This repo hosted a `zenpicker` Rust crate that was scaffolded as the
codec config picker runtime. It was **never published** to crates.io —
the only in-tree consumer was `zenwebp` behind its `picker` feature
flag.

The runtime, the codec-family meta-picker, and the Python training
pipeline have now been split into three siblings:

| Identity | Where | Versioning |
|---|---|---|
| **Generic Rust runtime** — ZNPR v2 parser, forward pass, masked argmin, metadata blob | [`zenpredict/`](zenpredict/) | `0.1.x` |
| **Codec-family meta-picker** (Rust) — given features + target + family mask, picks `{jpeg, webp, jxl, avif, png, gif}` | [`zenpicker/`](zenpicker/) | `0.1.x` |
| **Training pipeline** (Python) — pareto sweep, teacher fit, distill, ablation, safety reports, bake to ZNPR v2 | [`zentrain/`](zentrain/) | tooling |

The binary contract between them is **ZNPR v2**.

---

## What moved where

| Item | Was | Is now |
|---|---|---|
| Rust runtime crate name | `zenpicker` (path-dep, unpublished, ZNPK v1 format) | [`zenpredict`](zenpredict/) (`0.1.x`, dual-licensed, ZNPR v2) |
| Binary magic | `ZNPK` | `ZNPR` |
| Binary format version | v1 (32-byte header, positional layout) | v2 (128-byte `#[repr(C)]` header, offset-table sections, TLV metadata) |
| `Picker::argmin_masked(features, mask, adjust)` | `CostAdjust { additive_bytes, per_output_offset }` | `Predictor::argmin_masked(features, mask, transform, offsets)` with `ScoreTransform::{Identity, Exp}` and `ArgminOffsets { uniform, per_output }` |
| `reach_gate_mask(rates, threshold, out)` | picker-domain | `argmin::threshold_mask(rates, threshold, out)` (generic) |
| `RescuePolicy.rescue_threshold_pp` | field name | `RescuePolicy.rescue_threshold` (units unchanged — points on the codec's quality scale) |
| `PickerError` | not `#[non_exhaustive]` | `PredictError` is `#[non_exhaustive]` from day 1 |
| Loader convenience | manual `assert_eq!(model.schema_hash(), MY_HASH)` | `Model::from_bytes_with_schema(bytes, expected_hash)` |
| Feature bounds + OOD | feature-bounds in manifest JSON, `first_out_of_distribution` lived in the codec | top-level `feature_bounds` Section in the bin; `FeatureBound` + `first_out_of_distribution` are first-class in `zenpredict` |
| Reach-rate table | top-level Section was planned | Now lives in metadata under `zentrain.reach_rates` + `zentrain.reach_zq_targets` (codec copies out at startup) |
| Bake metadata | sibling JSON manifest | Embedded in the `.bin` via the typed-TLV metadata blob (utf8 / numeric / bytes); legacy sibling `manifest.json` still emitted by `bake_picker.py` for codecs that haven't migrated |
| Codec-family selection | implicit / hardcoded in each codec wrapper | First-class in [`zenpicker`](zenpicker/) with `CodecFamily`, `AllowedFamilies`, `MetaPicker` |
| Python training pipeline path | `zenpicker/tools/`, `zenpicker/examples/` | [`zentrain/tools/`](zentrain/tools/), [`zentrain/examples/`](zentrain/examples/) |
| Trainer-emitted metadata-key namespace | `zenpicker.*` (in the early v2 drafts) | `zentrain.*` (matches the producer) |
| Runtime metadata-key namespace | (none — runtime read from manifest JSON) | `zenpicker.*` reserved for the meta-picker (`zenpicker.family_order`); `zentrain.*` for trainer-emitted; `<codec>.*` for codec-private |

`zensim` V0.4's previously-vendored copy of the v1 MLP runtime moves
to the same `zenpredict` dependency; the on-disk format change
requires a one-shot re-bake of the V0.4 weights through the v2 baker.

---

## Cargo migration

### `Cargo.toml`

```diff
 [dependencies]
-zenpicker = { path = "../zenanalyze/zenpicker", optional = true }
+zenpredict = { path = "../zenanalyze/zenpredict", optional = true }
+# Optional: pull in the codec-family meta-picker if your codec
+# wrapper does cross-codec selection.
+zenpicker  = { path = "../zenanalyze/zenpicker",  optional = true }

 [features]
-picker = ["dep:zenpicker"]
+picker = ["dep:zenpredict"]
```

`zenpredict` has two cargo features (both default-on for codec consumers):

| Feature | What it gates |
|---|---|
| `std` | `std::error::Error` impls and `f32::exp` for `ScoreTransform::Exp` |
| `bake` | Rust-side ZNPR v2 byte-stream composer + JSON-driven baker |

`zensim` and other consumers that only need the inference core can
opt out of `bake`:

```toml
zenpredict = { version = "0.1", default-features = false, features = ["std"] }
```

The picker / rescue / bounds / metadata API surface is **not** behind
feature flags — it's all generic decision math, useful to codecs and
metrics crates alike.

`zenpicker` (the new meta-picker) is `no_std + alloc` capable with
one optional `std` feature mirroring zenpredict's.

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

**Alignment note:** `zenpredict` requires the input bytes to be at
least 4-byte aligned for f32 weight slices (8-byte-aligned recommended).
Wrap your `include_bytes!` blob in an `#[repr(C, align(16))]` struct —
see the example in [`zenpredict/README.md`](zenpredict/README.md). The
previous v1 loader silently allocated to copy on misalignment; the v2
loader fails loudly with `PredictError::SectionMisaligned`.

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
   `CostAdjust` was `Some` is now an explicit `ScoreTransform::Exp`
   argument. Pass `ScoreTransform::Identity` for raw-score argmin
   (most non-codec uses); pass `ScoreTransform::Exp` for codec
   log-bytes regressors.
2. `CostAdjust { additive_bytes, per_output_offset }` →
   `ArgminOffsets { uniform, per_output }`. Same shape, generic
   naming — nothing in this type pretends the values are bytes.

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
`metadata.get_numeric("zentrain.reach_rates")` (paired with
`metadata.get_numeric("zentrain.reach_zq_targets")`) at startup, then
keep the parsed `&[f32]` for per-encode use. The previous v1 layout
shipped these in a sibling `manifest.json`; v2 embeds them.

### Rescue policy

```diff
-use zenpicker::rescue::{RescuePolicy, RescueStrategy, RescueDecision, should_rescue};
-let policy = RescuePolicy::default();
+use zenpredict::{RescuePolicy, RescueStrategy, RescueDecision, should_rescue};
+let mut policy = RescuePolicy::default();
+// `RescuePolicy` is `#[non_exhaustive]`; construct via `Default`
+// then mutate fields, or just take the default.
+policy.rescue_threshold = 3.0;
```

The field rename `rescue_threshold_pp` → `rescue_threshold` drops the
documentation suffix from the field name (units unchanged — points on
the codec's quality scale). `#[non_exhaustive]` means external callers
must construct via `Default::default()` and assign fields, rather than
struct-init.

### Feature bounds + OOD

```diff
-// v1: codec compiled in its own FEATURE_BOUNDS table:
-const FEATURE_BOUNDS: &[FeatureBounds] = &[ ... ];
-if let Some(idx) = first_out_of_distribution(&features, FEATURE_BOUNDS) { ... }
+// v2: bounds ship in the .bin; loader exposes them as a borrowed slice:
+let bounds = model.feature_bounds();   // &[zenpredict::FeatureBound]
+if let Some(idx) = zenpredict::first_out_of_distribution(&features, bounds) { ... }
```

If `feature_bounds` is absent in the bin (`Section.len == 0`), the
loader returns an empty slice; OOD detection becomes a no-op for that
bake.

### Metadata access

```rust,ignore
use zenpredict::keys;

let md = model.metadata();
let profile: u8 = md.get_pod(keys::PROFILE).unwrap_or(0);
let bake_name = md.get_utf8(keys::BAKE_NAME).unwrap_or("(unnamed)");
let metrics: [f32; 3] = md.get_pod(keys::CALIBRATION_METRICS).unwrap_or([0.0; 3]);
let cell_config = md.get_bytes("zenjpeg.cell_config")?;  // codec-private
```

Standard zenpredict-defined keys live under `zentrain.*` (because the
producer is the Python trainer). The Rust meta-picker reserves
`zenpicker.*` for its own keys (e.g. `zenpicker.family_order`).
Codec-private keys live under `<codec>.*`. The loader hands these
back as opaque `&[u8]`; the codec deserializes its own format.

### Codec-family meta-picker (new)

```rust,ignore
use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker};
use zenpredict::Model;

let model = Model::from_bytes_with_schema(META_BIN, META_SCHEMA_HASH)?;
let mut meta = MetaPicker::new(model);
meta.validate_family_order()?;            // hard-fail on bake/runtime mismatch

let allowed = AllowedFamilies::all().deny(CodecFamily::Gif).deny(CodecFamily::Png);
let chosen = meta.pick(&features, &allowed)?;
match chosen {
    Some(CodecFamily::Webp) => /* dispatch to per-codec webp picker */,
    Some(CodecFamily::Jxl)  => /* … */,
    None                    => /* nothing allowed; caller fallback */,
    _ => /* … */
}
```

The meta-picker is optional. Codecs that always know which family
they're about to encode (e.g. a pure-zenjpeg pipeline) skip
`zenpicker` entirely and depend on `zenpredict` directly.

---

## Bake-side migration (Python)

The Python tooling moved from `zenpicker/tools/` → `zentrain/tools/`.
`tools/bake_picker.py` (top-level) now emits a portable
`BakeRequestJson` and shells out to the `zenpredict-bake` binary. The
byte-packing (struct.pack, magic constants, i8 quantization, alignment
padding) all moved to Rust.

The trainer-side metadata namespace moved from `zenpicker.*` (in the
v2 design drafts) to `zentrain.*` so the producer's name is
unambiguous. Bakes need to be re-emitted to pick up the new keys; old
v1 bakes already need re-baking for the format change, so this is a
no-extra-cost rename.

---

## Timeline / coordination

| Status | Item |
|---|---|
| ✅ Done | `zenpredict` v0.1.0 — generic Rust runtime, ZNPR v2 format, 80 tests |
| ✅ Done | `zenpicker` v0.1.0 — codec-family meta-picker (renamed from the placeholder `zenpickerchoose`) |
| ✅ Done | `zentrain/` — Python training pipeline (renamed from `zenpicker/tools/`) |
| ✅ Done | `tools/bake_picker.py` rewritten to emit ZNPR v2 via Rust baker |
| ✅ Done | `tools/bake_roundtrip_check.py` updated for v2 + all three activations |
| ✅ Done | `tools/test_bake_roundtrip.py` regression covers 3 activations × 3 dtypes |
| ⏳ In flight | `zenwebp` swap from old `zenpicker` (Rust shell) to `zenpredict` (currently broken on `spike/zenpicker-knobs`) |
| ⏳ Pending | `zensim` V0.4 swap from vendored mlp/ to `zenpredict` |
| 🚫 No-op | crates.io yank — neither `zenpicker` (Rust shell) nor any other variant was ever published |

---

## Why the renames

1. **`zenpicker` (Rust crate)** ended up as 80 % generic MLP runtime + 20 %
   codec-flavored offset semantics. It became `zenpredict` so other crates
   (`zensim` V0.4 perceptual scoring) can share it without renaming the
   picker-flavored types in their docs.
2. **`zenpicker` (the name)** moved to the codec-family meta-picker —
   the actual call site is `zenpicker::MetaPicker::pick`, which reads
   like what users would naturally type.
3. **`zentrain`** is unambiguous about what the Python pipeline is. No
   one will confuse it with a Rust crate.
4. **Trainer metadata under `zentrain.*`** matches the producer of
   those keys — keys produced by the trainer should be namespaced
   after the trainer; keys consumed by the meta-picker should be
   namespaced after the meta-picker (`zenpicker.family_order`).

The unpublished Rust shell at the previous `zenpicker/Cargo.toml` is
deleted; nobody outside this repo can be affected. The swap is a
one-PR-per-consumer affair (just `zenwebp` and `zensim`).
