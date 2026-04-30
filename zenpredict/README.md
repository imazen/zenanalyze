# zenpredict ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpredict?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpredict?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpredict) ![docs.rs](https://img.shields.io/docsrs/zenpredict?style=flat-square) ![License](https://img.shields.io/crates/l/zenpredict?style=flat-square)

Zero-copy MLP runtime. Parse a packed binary model (ZNPR v2), run scaler + layer-by-layer forward pass, surface typed metadata, run masked argmin for codec-config selection. Core of zenjpeg / zenwebp / zenavif / zenjxl picker selection and zensim V0.4 perceptual distance.

`#![forbid(unsafe_code)]`. `no_std + alloc` capable. MIT / Apache-2.0 dual license — the runtime is intentionally permissive so it can be embedded in any consumer (including MIT/Apache `zensim`). The training pipeline (`zentrain`) and codec dispatch logic (`zenpicker`, `zenanalyze`) remain `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`.

## Crate boundary

zenpredict is the **Rust runtime**. Three pieces compose with it:

- [`zenanalyze`](../) — feature extractor (one pass over a `zenpixels::PixelSlice`, returns the numeric features the model consumes).
- [`zenpicker`](../zenpicker/) — codec-family meta-picker that wraps `zenpredict::Predictor`; picks `{jpeg, webp, jxl, avif, png, gif}` ahead of the per-codec config picker.
- [`zentrain`](../zentrain/) — Python training pipeline: pareto sweep, teacher fit, distill, ablation, holdout probes, safety reports, `.bin` bake (via `tools/bake_picker.py` → `zenpredict-bake`).

All three version independently from `zenpredict`; the binary format (ZNPR v2) is the contract between them.

## Two consumer shapes

**Codec picker** — `argmin` over a constrained set of encoder configurations:

```rust,ignore
use zenpredict::{AllowedMask, ArgminOffsets, Model, Predictor, ScoreTransform};

#[repr(C, align(16))]
struct Aligned<const N: usize>([u8; N]);
const MODEL: &[u8] = &Aligned(*include_bytes!("zenjpeg_picker_v3.bin")).0;

let model = Model::from_bytes_with_schema(MODEL, MY_SCHEMA_HASH)?;
let mut predictor = Predictor::new(model);

let features = my_codec::extract_features(&analysis, target_zq);
let mask = AllowedMask::new(&my_codec::allowed_cells(&caller_constraints));

let pick = predictor.argmin_masked(
    &features,
    &mask,
    ScoreTransform::Exp,
    Some(&ArgminOffsets {
        uniform: caller_icc_size as f32,
        per_output: Some(&FORMAT_OVERHEAD),
    }),
)?;
```

**Perceptual scorer** — single forward pass, read first output:

```rust,ignore
use zenpredict::{Model, Predictor};

let model = Model::from_bytes(include_bytes!("zensim_v04.bin"))?;
let mut predictor = Predictor::new(model);
let distance = predictor.predict(&features)?[0];
```

## Format (ZNPR v2)

Fixed-shape `#[repr(C)]` header (128 bytes) + offset-table `LayerEntry[n_layers]` (48 bytes each) + aligned data sections + a typed-TLV metadata blob. Every weight slice is a zero-copy borrow into the input bytes — wrap `include_bytes!` in `#[repr(C, align(16))]` (see example above) to satisfy alignment. Full byte layout in [`src/model.rs`](src/model.rs).

Three weight dtypes:

- **F32** — full precision.
- **F16** — half the size at ~no accuracy cost. Conversion is built in (no `half` dep) — ~15 lines of integer bit math, see [`f16_bits_to_f32`](src/inference.rs).
- **I8** — `1/4` size with one f32 scale per output neuron. Per-output (column-wise) scaling — each output has its own dynamic range so one big-magnitude column doesn't waste i8 resolution on the small-magnitude ones.

Three activations: `Identity`, `ReLU`, `LeakyReLU(α=0.01)`.

## Metadata

The TLV metadata blob carries everything that's not raw weights: `zentrain.profile` (size_optimal vs zensim_strict), `zentrain.feature_columns`, `zentrain.calibration_metrics`, `zentrain.reach_rates` for the strict reach-rate gate, codec-private `<codec>.cell_config` payloads. Typed accessors (`get_utf8`, `get_numeric`, `get_bytes`) fail loudly on type mismatch instead of silently misreading.

Three value types: `bytes`, `utf8`, `numeric`. Numeric width is implied by `value_len`; per-key loader knows the exact shape.

## Decision math

The `argmin` family is generic — it's "argmin over a slice with a boolean filter," not codec-specific. `ScoreTransform::Exp` lets log-domain regressors mix with linear-domain offsets when the codec wants raw-byte argmin. `ArgminOffsets` carries uniform + per-output additive offsets in the post-transform score space.

`threshold_mask(rates, threshold, &mut out)` is the generalized form of the picker's reach-rate gate — codec ANDs the result against its constraint mask before argmin.

`rescue` module ships the two-shot encode-verify-rescue policy types (`RescuePolicy`, `RescueStrategy`, `RescueDecision`, `should_rescue`). Codec wires `encode` + `verify` + `encode_bumped`; zenpredict provides the threshold predicate and shared vocabulary.

## Features

| Feature | Default | What it gates |
|---|---|---|
| `std` | yes | `std::error::Error` impls; `f32::exp` for `ScoreTransform::Exp` |
| `bake` | yes | Rust-side ZNPR v2 composer (placeholder weights, round-trip tests) |

`no_std + alloc` builds drop both — `ScoreTransform::Exp` falls back to identity (loses linear-space mixing). Everything else works.

## License

MIT OR Apache-2.0, at your option. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

zenpredict is the runtime — it loads + scores baked models — and is intentionally licensed permissively so it can be embedded in any consumer (including the MIT/Apache-2.0 [`zensim`](https://github.com/imazen/zensim) perceptual scorer). The training pipeline ([`zentrain`](../zentrain/)) and codec dispatch logic ([`zenpicker`](../zenpicker/), [`zenanalyze`](../)) remain `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` — the IP lives on the bake-time and decision-tree side, not the runtime forward pass.
