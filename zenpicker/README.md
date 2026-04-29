# zenpicker ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpicker?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpicker?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpicker) ![docs.rs](https://img.shields.io/docsrs/zenpicker?style=flat-square) ![License](https://img.shields.io/crates/l/zenpicker?style=flat-square)

Codec-agnostic picker runtime. Loads a packed MLP from bytes, runs inference, returns argmin under caller-supplied constraints. Used by zenjpeg / zenwebp / zenavif / zenjxl to translate `(image_features, target_quality, constraints)` into a concrete encoder configuration.

`#![forbid(unsafe_code)]`. `no_std + alloc`. AGPL-3.0-or-later / Commercial dual license.

## Why a separate crate

Each codec ships its own picker model (different config space, different feature schema), but they all want the same thing at runtime:

1. Load a baked binary blob with format validation.
2. Run a small MLP forward pass.
3. Argmin over the outputs, restricted to a caller-supplied mask.
4. Optionally apply additive cost adjustments (caller's ICC profile size, format-specific overhead).

This is the codec-agnostic part. zenpicker owns it. Each codec crate owns its `ConfigSpec` enumeration, constraint API surface, baked `.bin` model, and the schema declaration for which features it consumes — all things zenpicker can't know.

## Usage

```rust,ignore
// 1. Embed the baked model. Wrap in an aligned struct so the loader
//    can zero-copy borrow f32 weights from the bytes.
#[repr(C, align(8))]
struct AlignedModel<const N: usize>([u8; N]);
const MODEL_BYTES: &[u8] = &AlignedModel(*include_bytes!("zenjpeg_picker_v1.bin")).0;

// 2. Load.
let model = zenpicker::Model::from_bytes(MODEL_BYTES)?;
assert_eq!(model.schema_hash(), MY_SCHEMA_HASH);  // codec verifies its own schema

// 3. Pick.
let mut picker = zenpicker::Picker::new(model);
let features: [f32; 48] = my_codec::extract_features(&analysis, target_zq);
let mask = my_codec::allowed_configs(&caller_constraints);
let pick = picker.argmin_masked(&features, &mask, None)?;
```

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

Weights are stored input-major: `W[i * out_dim + o]` is the contribution from input `i` to output `o`. This layout lets the matmul stream `out_dim` outputs in chunks of 8 across each input row, which is what `magetypes::f32x8` wants when SIMD lands.

## Schema hashing

The model header carries a `schema_hash: u64`. Codec consumers should hash their compile-time feature schema and compare to this on load. Mismatch means the model was baked against a different feature order or set; load with hard error rather than silently producing nonsense predictions.

```rust,ignore
const SCHEMA: &[&str] = &[
    "feat_variance", "feat_edge_density", /* ... */ "log_pixels",
    "zq_norm", "zq_norm_sq", "zq_norm_x_log_pixels",
    "zq_x_feat_variance", /* ... cross terms ... */
    "icc_bytes",
];
const SCHEMA_HASH: u64 = compute_hash(SCHEMA); // const-fn FxHash or similar
```

## Cost adjustments

`CostAdjust` lets callers add (a) a global additive byte cost (their planned ICC / EXIF / XMP overhead) and (b) per-output additive bytes (format-specific overhead the model couldn't learn). Argmin in log-bytes space ignores additive constants on its own — they only matter when combined with per-output offsets that differ between configs.

## What zenpicker is **not**

- Not a training framework. Use sklearn / PyTorch / whatever; emit the binary format from a bake tool.
- Not codec-aware. The crate has zero `cfg(feature = "jpeg")`, no codec-specific shortcuts.
- Not an inference engine for arbitrary architectures. Strictly: scaler → series of (linear, activation) layers → output. Adding new activations or layer types is additive in the format.

## Roadmap

- **v0.1** (now): scalar inference, f32 + f16 storage built in (no feature gate, no `half` dep — the conversion is ~15 lines of bit math), parse + validate v1 format.
- **v0.2**: `#[magetypes]`-dispatched matmul (AVX-512 / AVX2 / NEON / WASM SIMD128 / scalar). 8-wide f16 → f32 via F16C / FCVT through magetypes.
- **v0.3**: i8 quantized weights option (per-row scale) for the case where ~50 KB still isn't small enough. Hybrid heads (categorical + continuous outputs) for codecs with scalar control axes (effort, chroma_quality, lambda).

The format header has reserved fields for future expansion. New `weight_dtype` values, new activations, and new layer types are additive.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial. See [LICENSE-AGPL3](../LICENSE-AGPL3) and [LICENSE-COMMERCIAL](../LICENSE-COMMERCIAL).
