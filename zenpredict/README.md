# zenpredict ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpredict?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpredict?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpredict) ![docs.rs](https://img.shields.io/docsrs/zenpredict?style=flat-square) ![License](https://img.shields.io/crates/l/zenpredict?style=flat-square)

Zero-copy MLP runtime. Parse a packed binary model (ZNPR v3), run scaler + layer-by-layer forward pass, surface typed metadata, run masked argmin for codec-config selection. Core of zenjpeg / zenwebp / zenavif / zenjxl picker selection and zensim perceptual distance.

`#![forbid(unsafe_code)]`. `no_std + alloc` capable. MIT / Apache-2.0 dual license — the runtime is intentionally permissive so it can be embedded in any MIT/Apache consumer.

## Crate split (0.2)

The bake-side composer lives in the **sibling [`zenpredict-bake`](../zenpredict-bake/) crate**. Runtime consumers (`include_bytes!` + parse + predict) depend only on `zenpredict`; trainers and tooling depend on `zenpredict-bake`.

This split exists so codec-runtime binaries don't pay for `serde_json` and the JSON-baker glue they never call. Before 0.2 the JSON baker was ~30–40 % of zenpredict's monomorphization budget for zero in-process consumers.

## Crate boundary

- [`zenanalyze`](../) — feature extractor (one pass over a `zenpixels::PixelSlice`, returns the numeric features the model consumes).
- [`zenpicker`](../zenpicker/) — codec-family meta-picker that wraps `zenpredict::Predictor`; picks `{jpeg, webp, jxl, avif, png, gif}` ahead of the per-codec config picker.
- [`zenpredict-bake`](../zenpredict-bake/) — Rust composer + JSON baker + `zenpredict-bake` / `zenpredict-inspect` CLIs.
- [`zentrain`](../zentrain/) — Python training pipeline: pareto sweep, teacher fit, distill, ablation, holdout probes, safety reports, `.bin` bake (via `tools/bake_picker.py` shelling out to `zenpredict-bake`).

All version independently. The binary format (ZNPR v3) is the contract between them.

**Hard fork at 0.2.0** — v2 bins do not load. Migrate existing bakes via [`zentrain/tools/migrate_znpr_v2_to_v3.py`](../zentrain/tools/migrate_znpr_v2_to_v3.py); the rewrite is byte-perfect for the layer payloads (only the header version field changes).

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

let model = Model::from_bytes(include_bytes!("zensim_v018.bin"))?;
let mut predictor = Predictor::new(model);
let distance = predictor.predict(&features)?[0];
```

## Format (ZNPR v3)

Fixed-shape `#[repr(C)]` header (128 bytes) + offset-table `LayerEntry[n_layers]` (48 bytes each) + aligned data sections + a typed-TLV metadata blob + optional `output_specs` / `discrete_sets` / `sparse_overrides` sections. Every weight slice is a zero-copy borrow into the input bytes — wrap `include_bytes!` in `#[repr(C, align(16))]` (see example above) to satisfy alignment.

Wire layout: [`src/model.rs`](src/model.rs) (byte-by-byte). Shared offset constants: [`src/wire.rs`](src/wire.rs). Detailed format notes: [`docs/ZNPR_V3.md`](docs/ZNPR_V3.md).

Three weight dtypes:

- **F32** — full precision.
- **F16** — half the size at ~no accuracy cost. Conversion is built in (no `half` dep) — compact integer bit math, see [`f16_bits_to_f32`](src/inference.rs).
- **I8** — `1/4` size with one f32 scale per output neuron. Per-output (column-wise) scaling — each output has its own dynamic range so one big-magnitude column doesn't waste i8 resolution on the small-magnitude ones.

Three activations: `Identity`, `ReLU`, `LeakyReLU(α=0.01)`.

## Metadata

The TLV metadata blob carries everything that's not raw weights: `zentrain.profile`, `zentrain.feature_columns`, `zentrain.calibration_metrics`, codec-private `<codec>.cell_config` payloads, and (under the `advanced` cargo feature) the safety / rescue / output-bounds keys. Typed accessors (`get_utf8`, `get_numeric`, `get_bytes`) fail loudly on type mismatch instead of silently misreading.

Three value types: `bytes`, `utf8`, `numeric`. Numeric width is implied by `value_len`; per-key loader knows the exact shape.

## Decision math

The `argmin` family is generic — it's "argmin over a slice with a boolean filter," not codec-specific. `ScoreTransform::Exp` lets log-domain regressors mix with linear-domain offsets when the codec wants raw-byte argmin. `ArgminOffsets` carries uniform + per-output additive offsets in the post-transform score space.

Default-on: `argmin_masked`, `argmin_masked_in_range`.

Behind the `advanced` feature (default-off): `argmin_masked_top_k*`, `pick_with_confidence*`, `argmin_masked_with_scorer*`, `threshold_mask`, the two-shot `rescue` policy types, `safety::*` accessors, the typed `output_spec` API (`predict_with_specs`, `OutputValue`, `apply_spec`), and `bounds::*_out_of_distribution`. Wire-format slots for `output_specs` / `discrete_sets` / `sparse_overrides` parse unconditionally; the feature gates only the typed Rust API.

## Features

| Feature | Default | What it gates |
|---|---|---|
| `std` | yes | `std::error::Error` impls; `f32::exp` for `ScoreTransform::Exp` |
| `advanced` | no | safety / rescue / output_specs typed API / top-K argmin + scorer hybrids — see "Decision math" above |
| `compressed-weights` | no | `WeightDtype::I8Lz4` (wire byte 3) + the vendored LZ4 block decoder. Pair with `zenpredict-bake`'s matching `lz4` feature to emit compressed bakes. See "Compressed weights" below. |

`no_std + alloc` builds drop `std::error::Error` impls and `f32::exp` — `ScoreTransform::Exp` falls back to identity (loses linear-space mixing). Everything else works.

## Compressed weights

The `compressed-weights` feature adds `WeightDtype::I8Lz4` — i8 weights with an LZ4-block-compressed byte stream. Wire layout: `[u32 decompressed_len_bytes][lz4_block_payload]`, followed by the same per-output f32 scales section the plain I8 path uses.

**Decoder** is pure-safe-Rust, single-alloc, ~280 LOC, no_std-compatible. Lives at `src/lz4_block.rs`. The per-Predictor scratch is allocated once at `Predictor::new` (sized to `max(decompressed_len)` across all `I8Lz4` layers, zero if no compressed layers); each `predict()` re-decompresses fresh into it. No global decoded-weight cache.

**Measured trade-off** on V0_17 → V0_18 shape (228 → 384 → 1, 87.5K i8 weights):

| variant | bake bytes | first-parse µs | per-predict µs |
|---|---:|---:|---:|
| I8 raw | 93,064 | 0.1 | 140.3 |
| I8Lz4 + τ=0.005 zerobias | **37,976 (-59 %)** | 0.7 | 179.8 (+40) |

Pair lz4 with `zenpredict-bake::apply_zero_bias_in_place` to get the shrink; LZ4 alone on raw i8 weights *expands* (uniform-LSB byte streams have no runs to compress). For binary-size-sensitive consumers the 59 % shrink is real; for predict-throughput-sensitive ones the 40 µs/predict overhead is the cost.

**Resource limits**: every parse is bounded by `MAX_BAKE_BYTES = 64 MiB`, `MAX_DIM = 65,536`, `MAX_LAYERS = 256`. Limits are enforced before any allocation against the value being bounded; a 1 GB-claiming header fails in O(1). Constants exposed at `zenpredict::limits`. Fuzz targets at `fuzz/` cover parser + lz4 decoder + the full predict pipeline.

## License

MIT OR Apache-2.0, at your option. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

zenpredict is intentionally licensed permissively so it can be embedded in any MIT/Apache consumer (including [`zensim`](https://github.com/imazen/zensim)). The training pipeline ([`zentrain`](../zentrain/)) and codec dispatch logic ([`zenpicker`](../zenpicker/), [`zenanalyze`](../)) remain `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` — the IP lives on the bake-time and decision-tree side, not the runtime forward pass.
