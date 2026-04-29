# zenpicker — status snapshot (2026-04-29)

Working branch: `feat/zenpicker-v0.1` (PR #14, draft). CI: 19/19 green.
**Hybrid-heads v2.0 is the default** for new codec integrations.

## Done

**Runtime crate (`zenpicker`, v0.1.0)**

- Binary format v1: 32-byte header (`ZNPK` magic, version, dims, schema_hash, flags), scaler section, per-layer (in_dim, out_dim, activation, weight_dtype, weights, biases). Little-endian, 4-byte aligned. Zero-copy borrow when input bytes are aligned.
- Inference: scaler → series of (linear, ReLU/Identity) layers → output. f32 + f16 storage **both built in** (no feature flag). f16→f32 via 15-line `inference::f16_bits_to_f32` covering all IEEE-754 classes — same answer F16C hardware gives. Drops `half` dependency entirely.
- Argmin masking: `AllowedMask<'a>(&[bool])` plus `CostAdjust` for caller ICC/EXIF tax + per-output offsets.
- **Hybrid heads** via `Picker::argmin_masked_in_range((start, end), …)` — codec slices the output vector for scalar predictions and feeds the bytes-only sub-range to the categorical pick. No format change; manifest declares the layout.
- 18 tests passing. Clippy clean with `-D warnings` on `--all-targets --all-features`. `#![forbid(unsafe_code)]`. `no_std + alloc`.

**Models shipped (zenpicker/models/)**

| Bake | Size | Mean overhead | Argmin acc | Status |
|---|---:|---:|---:|---|
| `zenjpeg_picker_v1.0_19feat.bin` | 31 KB | 7.20% | 13.8% | legacy / broadest features |
| `zenjpeg_picker_v1.1_8feat.bin` | 28 KB | 8.20% | 10.5% | legacy / reduced schema |
| **`zenjpeg_picker_v2.0_hybrid.bin`** | **50 KB** | **2.76%** | **52.0%** | **default — recommended** |

The hybrid v2.0 collapses the categorical space from 120 cells to 12 (color × sub × trellis_on × sa_piecewise) and adds two scalar heads per cell (chroma_scale, lambda). 3× lower mean overhead and 5× higher argmin accuracy than the v1.1 pure-categorical bake at the same 8-feature schema.

**Bake tooling (`tools/`)**

- `bake_picker.py` — codec-agnostic. Reads sklearn JSON; honors model-side `extra_axes` and `schema_version_tag` overrides so other codecs can bake without editing the script.
- `bake_roundtrip_check.py` — bakes both dtypes, runs Rust `load_baked_model` example, compares to numpy reference. Round-trip max rel diff `4e-6` on the v2.0 hybrid model.
- `examples/load_baked_model.rs` — runs inference on a deterministic input, prints outputs.
- `examples/hybrid_heads_codec_sketch.rs` — full codec-side reference: `CONFIGS` table, constraint translation, scalar clamp.

**Safety plane** ([SAFETY_PLANE.md](SAFETY_PLANE.md)) — two-shot rescue design for catastrophic-zensim failures. Codec-side, layered on the existing `ZqTarget` iteration loop.

**Companion issues**

- [imazen/zenanalyze#13](https://github.com/imazen/zenanalyze/issues/13): round-trip generational re-encode picker — source-provenance features, sealed web-representative corpus, two-model dispatch architecture.

## Open architectural questions

1. **ConfigSpec table sync.** Bake emits `manifest.json` with config names as strings; codec crate's compile-time `CONFIGS` table needs a stable verification mechanism (manifest hash + schema hash both checked).

## Pending (next session)

- Wire `Decoder::source_provenance()` accessor in zenjpeg for the round-trip picker (separate PR; see #13).
- Sealed corpus manifest + tower storage layout for the round-trip training data.
- Round-trip sweep harness (extend `zq_pareto_calibrate` to take pre-encoded sources).
- Implement the safety-plane `RescuePolicy` in zenjpeg's `ZqTarget` iteration loop (see SAFETY_PLANE.md). Calibrate `rescue_threshold_pp` from held-out shortfall p99.
- Train zenwebp / zenavif / zenjxl picker models using the codec-agnostic bake path (set `extra_axes` and `schema_version_tag` in their training-script JSON).

## What's safe to revisit / change

- Output count, output semantics, manifest format: **all additive**. v1 binary format itself is stable; manifest fields can grow without rebaking existing models.
- Picker model architecture (hidden width, cross-term recipe). No format change required to swap models — codec compares schema_hash on load and re-bakes when needed.
