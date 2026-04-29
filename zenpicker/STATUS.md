# zenpicker — status snapshot (2026-04-29)

Working branch: `feat/zenpicker-v0.1` (PR #14, draft).

## Done

**Runtime crate (`zenpicker`, v0.1.0)**

- Binary format v1: 32-byte header (`ZNPK` magic, version, dims, schema_hash, flags), scaler section, per-layer (in_dim, out_dim, activation, weight_dtype, weights, biases). Little-endian, 4-byte aligned. Zero-copy borrow when input bytes are aligned.
- Inference: scaler → series of (linear, ReLU/Identity) layers → output. f32 + f16 storage **both built in** (no feature flag). f16→f32 via 15-line `inference::f16_bits_to_f32` covering all IEEE-754 classes — same answer F16C hardware gives. Drops `half` dependency entirely.
- Argmin masking: `AllowedMask<'a>(&[bool])` plus `CostAdjust` for caller ICC/EXIF tax + per-output offsets.
- 17 tests passing. Clippy clean with `-D warnings`. `#![forbid(unsafe_code)]`. `no_std + alloc`.
- `archmage` + `magetypes` on the dep list (free in workspace) — reserved for v0.2 SIMD matmul.

**Bake tooling (`tools/`)**

- `bake_picker.py` — sklearn JSON → v1 binary + manifest.json.
- `bake_roundtrip_check.py` — bakes both dtypes, runs Rust `load_baked_model` example, compares to numpy reference. Verified on `zq_bytes_distill_2026-04-29.json`:
  - f32: 59 KB blob, max rel diff vs numpy `3.2e-7`
  - **f16: 31 KB blob, max rel diff vs numpy `3.0e-7`** — saves 50 % size, no observable accuracy cost
- `examples/load_baked_model.rs` — runs inference on a deterministic input, prints outputs.

**Companion issue**

- [imazen/zenanalyze#13](https://github.com/imazen/zenanalyze/issues/13): round-trip generational re-encode picker — source-provenance features, sealed web-representative corpus, 14M-cell sweep plan, two-model dispatch architecture.

## Open architectural questions

1. **Hybrid heads** for continuous control axes (chroma_quality, lambda, effort). Current v1.0 model is purely categorical (120 cells, each baking specific scalar values). Wrong shape for jxl/avif/webp where effort is a single integer axis the user controls. Right shape is two heads: categorical bytes-prediction over discrete-only axes (color_mode, sub, scan, sa_piecewise) + continuous parameter heads predicting optimal scalar values. Format already supports this — just additional output channels with manifest semantics. **Recommend deciding before any v1.0 zenjpeg bake.**

2. **ConfigSpec table sync.** Bake emits `manifest.json` with config names as strings; codec crate's compile-time `CONFIGS` table needs a stable verification mechanism (manifest hash + schema hash both checked).

## In progress (this session)

- Feature ablation: train HistGB with each feature removed, rank by accuracy delta. Goal: drop the zenanalyze features that don't pay for themselves in picker accuracy → reduces zenanalyze hot-path cost. Script: `zenjpeg/scripts/zq_feature_ablation.py`. Output: `zenjpeg/benchmarks/zq_feature_ablation_2026-04-29.log`.

## Pending (next session)

- Settle hybrid-heads design on issue #13. Sketch the manifest schema for "this output index is categorical bytes / scalar prediction / scalar gradient".
- Bake the existing 120-cell distillation as v1.0 stepping stone (already trained, format supports it). Codec-side API in zenjpeg routes through `EncoderConfig::for_perceptual_target`.
- Wire `Decoder::source_provenance()` accessor in zenjpeg (separate PR).
- Sealed corpus manifest + tower storage layout.
- Round-trip sweep harness (extend `zq_pareto_calibrate`).

## What's safe to revisit / change

- Output count, output semantics, manifest format: **all open**. v1 binary format itself is locked once we ship a real model; everything above the wire format is still mutable.
- The 19-feature zenanalyze schema. Pending ablation result. If we drop features, the picker schema_hash changes — that's a re-bake, not a format change.
