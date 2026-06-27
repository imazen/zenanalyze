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

// The bake compiled in. `Model::from_bytes*` copies it into an owned,
// internally-aligned buffer (see "Alignment" below), so a plain
// `include_bytes!` works as-is — the `#[repr(C, align(16))]` wrapper is
// optional and only matters for the runtime-file-load path.
const MODEL_BYTES: &[u8] = include_bytes!("zenjpeg_picker_v3.bin");

// The schema hash is a plain `u64` chosen at bake time and stored in the
// header. Compile-in the value the bake was produced with (read it via
// `zenpredict-inspect`, or `model.schema_hash()` on a known-good build) so a
// stale/mismatched bake fails loudly at load instead of silently mispicking.
const MY_SCHEMA_HASH: u64 = 0x0123_4567_89ab_cdef;

// `Model` owns its copy of the bake. `Predictor` BORROWS the `Model`, so the
// `Model` must outlive the `Predictor` (here both are locals; for a process-
// wide singleton put the `Model` in a `static OnceLock<Model>`).
let model: Model = Model::from_bytes_with_schema(MODEL_BYTES, MY_SCHEMA_HASH)?;
let mut predictor = Predictor::new(&model); // note: &model — `new` takes `&Model`

// RAW features — pass them un-normalized. The bake's embedded scaler
// standardizes internally (`x' = (x - mean) / scale`) before the first layer.
// Length MUST equal `model.n_inputs()`, in the `zentrain.feature_columns`
// order the bake was trained with (a wrong length is a `FeatureLenMismatch`).
let features: Vec<f32> = my_codec::extract_features(&analysis, target_zq);

// `AllowedMask::new` takes `&[bool]`: one flag per output/config cell, `true`
// = "this config may be picked". It must be at least `model.n_outputs()` long
// (a short mask panics). `&[true; N]` admits every cell.
let allowed: Vec<bool> = my_codec::allowed_cells(&caller_constraints);
let mask = AllowedMask::new(&allowed);

// `Predictor::argmin_masked` runs the forward pass for `features`, then
// argmins the outputs under the mask. Returns `Result<Option<usize>, _>`:
//   - `Err(..)`   — bad feature length, offsets-length mismatch, etc.
//   - `Ok(None)`  — no cell allowed by the mask (or every allowed cell scored
//                   NaN) — fall back to a default config.
//   - `Ok(Some(i))` — `i` indexes the FULL output space (same indexing as
//                   `allowed` and the model's config table); map it back with
//                   your own config table, e.g. `my_codec::CONFIGS[i]`.
let pick: Option<usize> = predictor.argmin_masked(
    &features,
    &mask,
    ScoreTransform::Exp, // model emits log-bytes → exponentiate so argmin is
                         // in raw-byte space and mixes with byte-space offsets
    Some(&ArgminOffsets {
        // `uniform: f32` is added to every cell's post-transform score (same
        // for all cells, so it can't change the pick on its own)...
        uniform: caller_icc_size as f32,
        // ...but `per_output: Option<&[f32]>` (per-cell additive overhead,
        // here per-config container/ICC bytes) can. When `Some`, its length
        // must equal the argmin's working slice — full `n_outputs` here.
        per_output: Some(&FORMAT_OVERHEAD),
    }),
)?;

let config = match pick {
    Some(i) => my_codec::CONFIGS[i], // the chosen encoder configuration
    None => my_codec::DEFAULT_CONFIG, // mask admitted nothing — use a default
};
```

**Perceptual scorer** — single forward pass, read first output:

```rust,ignore
use zenpredict::{Model, Predictor};

// `predict` takes RAW features `&[f32]` of length `model.n_inputs()` (the
// embedded scaler standardizes them) and returns `Result<&[f32], _>` borrowing
// the predictor's internal output buffer (one f32 per model output). A scorer
// bake has a single output, so index `[0]`. `include_bytes!` is fine —
// `from_bytes` copies into an internally-aligned buffer.
let model = Model::from_bytes(include_bytes!("zensim_v018.bin"))?;
let mut predictor = Predictor::new(&model); // &model — `new` borrows the Model
let distance: f32 = predictor.predict(&features)?[0];
```

> **Picking from `predict` directly.** `Predictor::argmin_masked` is just
> `predict` + `argmin::argmin_masked` rolled together. If you want the raw
> scores too, call `predictor.predict(&features)?` to get `&[f32]`, then
> `zenpredict::argmin::argmin_masked(scores, &mask, transform, offsets)` — that
> free function returns the same `Option<usize>` (full-output-space index;
> `None` when no cell is allowed). Same `ScoreTransform` / `ArgminOffsets`
> semantics either way.

## Format (ZNPR v3)

Fixed-shape `#[repr(C)]` header (128 bytes) + offset-table `LayerEntry[n_layers]` (48 bytes each) + aligned data sections + a typed-TLV metadata blob + optional `output_specs` / `discrete_sets` / `sparse_overrides` sections.

Wire layout: [`src/model.rs`](src/model.rs) (byte-by-byte). Shared offset constants: [`src/wire.rs`](src/wire.rs). Detailed format notes: [`docs/ZNPR_V3.md`](docs/ZNPR_V3.md).

### Alignment & lifetime

`Model::from_bytes` / `from_bytes_with_schema` **copy the input into an owned, heap-allocated `Box<[u8]>`** at construction (decompressing first for a compressed bake). Two consequences:

- **The input slice has no alignment requirement** — `include_bytes!(...)` can be passed straight to `from_bytes`. Misaligned input is *not* UB and does *not* panic; it's copied like any other. The `#[repr(C, align(16))]` wrapper used in some examples is therefore **optional** for the compile-in case.
- **The input bytes do not need to outlive the `Model`.** The `Model` owns its copy; weight/scaler/metadata slices are zero-copy borrows into **`self.bytes`** (the owned buffer), not into the original input. You may drop the input immediately after `from_bytes` returns.

The lifetime that *does* matter is the other one: **`Predictor<'a>` borrows `&'a Model`, so the `Model` must outlive the `Predictor`.** For a process-wide singleton, park the `Model` in a `static OnceLock<Model>` (it's `Send + Sync`) and give each thread its own `Predictor` over a `&'static Model` — no `Mutex` needed.

One subtlety for the **runtime-file-load** path (`std::fs::read` → `Vec<u8>`): the owned buffer is cast in place to typed slices, and a `Vec<u8>` allocation isn't *guaranteed* to be 4/8-byte aligned. If the allocator ever hands back a misaligned buffer, the typed-slice cast fails cleanly with `PredictError::SectionMisaligned` (again, an error — never UB). In practice global allocators return ≥16-aligned blocks so this is rare, but if you hit it, re-align into a `u64`-backed buffer before parsing — see [`examples/load_baked_model.rs`](examples/load_baked_model.rs) for the pattern.

Three weight dtypes:

- **F32** — full precision.
- **F16** — half the size at ~no accuracy cost. Conversion is built in (no `half` dep) — compact integer bit math, see [`f16_bits_to_f32`](src/inference.rs).
- **I8** — `1/4` size with one f32 scale per output neuron. Per-output (column-wise) scaling — each output has its own dynamic range so one big-magnitude column doesn't waste i8 resolution on the small-magnitude ones.

Three activations: `Identity`, `ReLU`, `LeakyReLU(α=0.01)`.

## Metadata

The TLV metadata blob carries everything that's not raw weights: `zentrain.profile`, `zentrain.feature_columns`, `zentrain.calibration_metrics`, codec-private `<codec>.cell_config` payloads, and (under the `advanced` cargo feature) the safety / rescue / output-bounds keys. Typed accessors (`get_utf8`, `get_numeric`, `get_bytes`) fail loudly on type mismatch instead of silently misreading.

Three value types: `bytes`, `utf8`, `numeric`. Numeric width is implied by `value_len`; per-key loader knows the exact shape.

## Decision math

The `argmin` family is generic — it's "argmin over a slice with a boolean filter," not codec-specific.

- **`AllowedMask::new(&[bool])`** — one flag per output cell; `true` = pickable. Must be ≥ the scored slice length (a short mask panics in debug *and* release).
- **`argmin_masked(...) -> Option<usize>`** (free fn) / **`Predictor::argmin_masked(...) -> Result<Option<usize>, _>`** (method, which runs `predict` first). `Some(i)` is an index into the **full output/config space** (the same indexing as the mask and your config table); `None` means no allowed cell (or all allowed cells scored NaN — NaN is silently skipped). Ties break to the **lowest index**. Use `*_in_range` to argmin over a sub-slice of the outputs (hybrid-heads bakes that pack `[bytes.., scalar1.., …]`); for those the returned index is into the sub-range.
- **`ScoreTransform`** (applied per score *before* offsets and argmin): `Identity` (default — outputs are already in argmin-target space, e.g. perceptual distances) or `Exp` (the model emits **log-domain** values, typically log-bytes; `exp` brings them into raw-byte space so a byte-space `ArgminOffsets` table mixes correctly). `Exp` clamps its input to `[-30, 30]` and uses a true `exp` under both `std` and `no_std` (via `libm`), so the linear-space argmin is identical across build configs.
- **`ArgminOffsets<'a>`** — additive cost adjustments in the post-transform score space, fields:
  - `uniform: f32` — added to every cell's score. Same for all cells, so on its own it never changes the pick (e.g. caller-side ICC/EXIF overhead).
  - `per_output: Option<&'a [f32]>` — per-cell additive (e.g. per-config container/ICC bytes). When `Some`, its length **must equal the argmin's working slice** (full `n_outputs` for `argmin_masked`, the sub-range length for `*_in_range`), else `PredictError::OffsetsLenMismatch`. This is the term that can actually shift the pick. Pass `None` (or `ArgminOffsets::uniform(x)` for uniform-only) when you have no per-cell table.

Default-on (stable, always compiled): `argmin_masked`, `argmin_masked_in_range`.

Behind the opt-in **`topk`** feature (default-off, stable — `--features topk`, also reachable under `advanced`): **`argmin_masked_top_k::<K>`** / **`argmin_masked_top_k_in_range::<K>`** (top-`K` lowest-scoring indices, ascending, for the "predict-top-K then encode-verify" picker path — `K = 3` typically) as both free fns and `Predictor` methods, plus **`tier_mask`** + the compute-tier reader (see [Compute-tier masking](#compute-tier-masking)). Kept off the default surface so a `predict` / `argmin_masked`-only consumer pays no extra public API or top-K monomorphization — the cost is opt-in.

Behind the `advanced` feature (default-off): the closure-scorer hybrids `argmin_masked_with_scorer*` / `argmin_masked_top_k_with_scorer*`, `pick_with_confidence*`, `threshold_mask`, the two-shot `rescue` policy types, `safety::*` accessors, the typed `output_spec` API (`predict_with_specs`, `OutputValue`, `apply_spec`), and the output-space OOD check (`OutputBound`, `output_first_out_of_distribution`). `advanced` also re-exposes the whole `topk` surface for back-compat. The feature-space OOD check (`FeatureBound`, `first_out_of_distribution`) is default-on. Wire-format slots for `output_specs` / `discrete_sets` / `sparse_overrides` parse unconditionally; the feature gates only the typed Rust API.

### Compute-tier masking

Behind the opt-in `topk` feature (also reachable under `advanced`). A bake MAY carry an optional `zentrain.cell_compute_tier` metadata key — `[u8; n_outputs]`, one small compute-tier rank per output cell (lower = cheaper to encode; e.g. JXL effort `e1`..`e9`, or any codec's opaque cost rank). `Model::cell_compute_tiers()` (or `Predictor::cell_compute_tiers()`) returns it as a zero-copy `&[u8]`, **empty** when the bake omits it — so older bins still load and the caller just skips tier masking. The generic `argmin::tier_mask(tiers, max_tier, out)` fills a `&mut [bool]` admitting only cells whose tier is `<= max_tier`; AND it into your constraint mask before `argmin_masked` / `argmin_masked_top_k` to express "fast configs only" under a time budget, with no per-codec tier table.

## Features

| Feature | Default | What it gates |
|---|---|---|
| `std` | yes | `std::error::Error` trait impls |
| `topk` | no | **stable, minimal** picker surface: top-`K` argmin query (`argmin_masked_top_k*`, free fns + `Predictor` methods) + compute-tier masking (`tier_mask`, `Model`/`Predictor::cell_compute_tiers`, `keys::CELL_COMPUTE_TIER`) — see "Decision math" above |
| `advanced` | no | safety / rescue / output_specs typed API / closure-scorer argmin hybrids / `pick_with_confidence` / `threshold_mask` — see "Decision math" above. Also re-exposes the `topk` surface for back-compat. |

The default surface and the `topk` surface follow normal 0.x semver (and the default surface stays byte-identical to a `predict`-only consumer — enabling `topk` is the only way to add the top-K API + its compile cost). The `advanced` surface is **not yet stabilized** — items behind it may change shape or be removed in a 0.2.x patch, so pin a version if you depend on them. (`advanced` gates extra API; it's a different axis from `zenanalyze`'s `experimental`, which gates unstable feature numerics.)

`no_std + alloc` builds drop only the `std::error::Error` impls. All numeric work — including `f32::exp` for `ScoreTransform::Exp` — runs identically via the unconditional `libm` dependency.

## Compressed bakes

A bake's payload (everything after the 128-byte header) may be LZ4-block-compressed as a single envelope. There is **no feature flag** — the loader always links the decoder (`lz4_flex`, `safe-decode` only, ~4 KB) and handles compression transparently:

- Header `flags` bit 0 marks a compressed payload; the algorithm nibble selects LZ4 (`lz4_flex::block`). Header byte offset 96 carries `decompressed_payload_len: u32`.
- At load the parser allocates `128 + decompressed_payload_len` bytes, copies the header verbatim, and decompresses the payload into place. From there it parses exactly as an uncompressed bake — section offsets are written as if uncompressed, so the compression is purely an envelope.

The composer (`zenpredict-bake`) decides whether to emit a compressed envelope; the runtime accepts either form. The earlier per-layer `WeightDtype::I8Lz4` scheme was removed in favour of this whole-payload approach — see [`WIRE_FORMAT_V3_1.md`](WIRE_FORMAT_V3_1.md). Weight dtypes are now exactly `F32` / `F16` / `I8`.

**Resource limits**: every parse is bounded by `MAX_BAKE_BYTES = 64 MiB`, `MAX_DIM = 65,536`, `MAX_LAYERS = 256`, and the decompressed payload is additionally bounded to `MAX_BAKE_BYTES` minus the header. Limits are enforced before any allocation against the value being bounded; a 1 GB-claiming header fails in O(1). Constants exposed at `zenpredict::limits`. Fuzz targets at `fuzz/` cover parser + decompression + the full predict pipeline.

## License

MIT OR Apache-2.0, at your option. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).

zenpredict is intentionally licensed permissively so it can be embedded in any MIT/Apache consumer (including [`zensim`](https://github.com/imazen/zensim)). The training pipeline ([`zentrain`](../zentrain/)) and codec dispatch logic ([`zenpicker`](../zenpicker/), [`zenanalyze`](../)) remain `AGPL-3.0-only OR LicenseRef-Imazen-Commercial` — the IP lives on the bake-time and decision-tree side, not the runtime forward pass.
