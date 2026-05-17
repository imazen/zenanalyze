# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that ship together in the next minor (0.x patch
     bumps stay non-breaking). Persist across patch releases. Only
     clear when the breaking release ships. -->

- `Header.*` fields became `pub(crate)` in this 0.2.0 cut. zenavif
  migrated to `model.n_outputs()` ahead of release. Other internal
  consumers (zenwebp, zenjpeg, zenpicker) build clean.
- `LayerEntry.*` and `LayerView.*` field tightening may follow once
  `WeightStorage` variant pattern-matching has a stable accessor
  pair (likely `LayerView::weights() -> &WeightStorage<'_>`).

### Added

## [0.2.1] - 2026-05-17

### Added

- **5 new stacked `FeatureTransform` variants** identified as universal
  high-win stacks across zenjpeg / zenwebp / zenavif by the
  2026-05-17 stacks sweep:
  - `WinsorThenLog` (params `[p1, p99]`) — `ln(clamp(x, p1, p99))`.
    The dominant stack winner (12+10+13 wins across codecs).
  - `WinsorThenLog1p` (params `[p1, p99]`) —
    `ln(1 + clamp(x, p1, p99))`. Secondary winner.
  - `WinsorThenSignedCbrt` (params `[p1, p99]`) —
    `signed_cbrt(clamp(x, p1, p99))`. Scattered wins.
  - `SignedCbrtThenWinsor` (params `[q1, q99]`) —
    `clamp(signed_cbrt(x), q1, q99)`. `q1`, `q99` are bounds in
    cbrt-domain, distinct semantics from `WinsorThenSignedCbrt`.
  - `ClipThenLog1pThenWinsor` (params `[ε, q1, q99]`) —
    `clamp(ln(1 + max(0, x − ε)), q1, q99)`. Second most common
    stack winner (7+8+8).
  Math mirrors `zentrain/tools/feature_transform_sweep.py`'s
  `_make_stack` helper byte-identically so bake round-trips match.
  All five are additive on the `#[non_exhaustive]` enum (no
  semver break).

- **Multi-codec shared-trunk picker runtime (ZNPR v3.2).** New
  optional `multi_codec_schema` header section (offset 116..124)
  carrying the per-codec scatter map for joint-trained pickers.
  When present, `Predictor::predict_multi_codec(codec_id,
  codec_features, size_class, log_pixels, zq_norm)` composes the
  trunk's input vector (union features + presence mask + size
  onehot + scalars + codec onehot) from a single codec's natural
  feature vector and returns that codec's output range. Backwards
  compatible — single-codec bakes leave the section empty and load
  unchanged. New public types: `MultiCodecSchema`, `PerCodecMap`,
  `HeadMeta`. New `Model::multi_codec_schema()` /
  `has_multi_codec_schema()` accessors. New
  `PredictError::MultiCodecNotSupported` /
  `UnknownCodecId` / `CodecFeatureLenMismatch` variants on the
  `#[non_exhaustive]` enum. Parse-time validation cross-checks the
  schema against the trunk's `n_inputs` (must equal `2*U + 6 + C`).

## [0.2.0] - 2026-05-13

### Added: `compressed-weights` feature (LZ4 + zerobias)

New cargo feature `compressed-weights` (default-off) ships
`WeightDtype::I8Lz4` (wire byte 3) for layers whose i8 weight bytes
are LZ4-block-compressed. Bake-side support is in
[`zenpredict-bake`](../zenpredict-bake/) behind its parallel `lz4`
feature.

**Decoder**: pure-safe-Rust `src/lz4_block.rs` (~280 LOC, single
allocation, `#![forbid(unsafe_code)]`-compatible). Hand-rolled
instead of vendoring `lz4_flex` to keep the binary cost under 1 KB
and stay free of `unsafe` blocks; round-trip-tested against
`lz4_flex`'s encoder. 9 unit tests cover overlap-RLE semantics,
truncated input, invalid offsets, output-overflow, and a fuzz-style
random-input probe.

**Wire layout**: layer's weights section is `[u32 decompressed_len_bytes][lz4_block_payload]`,
followed by the existing per-output f32 scales section unchanged.
`decompressed_len_bytes` must equal `in_dim * out_dim`; the scales
section is byte-identical to plain `I8`. Switching `I8 ↔ I8Lz4` is
a wire-format change only — quantized weights, matmul kernel, and
score outputs are bit-equivalent.

**Inference**: single-alloc scratch buffer per Predictor sized at
`Predictor::new` to `max(decompressed_len)` across all `I8Lz4`
layers; zero alloc if the bake has no compressed layers. Per
`predict()`, the scratch is re-decompressed fresh — "very fast
uncached expansion" per the 2026-05-13 design directive. No global
decoded-weight cache.

**Measured on V0_17 → V0_18 shape (228 → 384 → 1, 87.5K i8 weights)**:

| variant | bake bytes | shrink | first-parse µs (median) | per-predict µs (median) |
|---|---:|---:|---:|---:|
| I8 raw | 93,064 | — | 0.1 | 140.3 |
| I8 zerobias (τ=0.005, per-layer) | 93,064 | 0 | 0.1 | 140.9 |
| I8Lz4 raw | 93,280 | +0.2 % (expansion) | 0.8 | 142.8 |
| **I8Lz4 zerobias (τ=0.005)** | **37,976** | **-59.2 %** | **0.7** | **179.8** |

Reading: lz4 alone on raw weights _expands_ (i8 LSBs are near-uniform-
random). Pair lz4 with τ-zerobias and the bake shrinks 59 % with
only ~40 µs of added per-predict cost (the per-`predict()`
decompression). First-parse is ~0.6 µs slower with lz4 (scratch
allocation in `Predictor::new`), still under 1 µs.

**Trade-off summary**: compress for binary-size-sensitive embeds;
keep uncompressed for predict-throughput-sensitive consumers. The
40 µs/predict overhead matters for batch sweeps that call
`predict()` thousands of times per second; for one-shot zensim
scoring it's invisible.

### Added: Resource limits

`src/limits.rs` exposes four constants enforced at parse time
**before** any allocation against the value being checked:

- `MAX_BAKE_BYTES = 64 MiB` — bytes-slice ceiling. Every shipped
  bake is < 1 MiB; the limit bounds fuzz / adversarial input.
- `MAX_DIM = 65,536` — per-dim ceiling on `n_inputs`, `n_outputs`,
  `in_dim`, `out_dim`. Largest shipped dim is 384.
- `MAX_LAYERS = 256` — every shipped bake has ≤ 4 layers.
- `MAX_LZ4_DECOMPRESSED_BYTES` (with `compressed-weights`): caps
  the per-layer decompressed scratch.

A 1 GB-claiming header now fails in O(1).

### Added: Fuzz targets

`fuzz/` directory with three `libfuzzer-sys` targets covering
`Model::from_bytes`, `lz4_block::decompress_into`, and the full
`Predictor::predict` pipeline against arbitrary bytes. 5K-iteration
smoke runs pass clean on every target. Corpora live under
`/mnt/v/fuzzes/zenanalyze/` per CLAUDE.md fuzz-corpus policy
(not committed to git).


This is a **hard fork**: v2-format bakes do not load. Migrate via
[`zentrain/tools/migrate_znpr_v2_to_v3.py`](../zentrain/tools/migrate_znpr_v2_to_v3.py)
— the rewrite is header-only (layer payloads byte-identical between v2 and v3).

### Changed (breaking)

- Format: parser now accepts only ZNPR **v3** bytes; v1 and v2 fail with
  `PredictError::UnsupportedVersion`. v3 differs from v2 only in the
  `version` byte plus three optional new sections (`output_specs`,
  `discrete_sets`, `sparse_overrides`).
- **Bake-side composer extracted to a separate crate
  [`zenpredict-bake`](../zenpredict-bake/)**. The `bake` cargo feature
  is gone; consumers building bakes import `zenpredict-bake` directly.
  The runtime's monomorph budget drops ~30-40 % (no more serde_json +
  JSON-visitor glue compiled into every codec binary).
- `bake_v2` (when it lived in this crate) renamed to `bake` and now
  lives at `zenpredict_bake::bake`.
- `__experimental_versions` feature does not exist here — that was a
  zensim-side feature.
- Default features changed from `["std", "bake"]` to `["std"]`.
- `OutputTransform::from_byte` promoted from `pub(crate)` to `pub` so
  the external bake crate's validator can reject unknown variants.
- `Section.offset` / `Section.len` tightened to `pub(crate)`. New
  `Section::new(offset, len)`, `Section::offset() -> u32`,
  `Section::len_bytes() -> u32` accessors.

### Added

- New cargo feature `advanced` (default-off) bundles four speculative
  subsystems behind a single flag so consumers paying for a lean
  runtime don't link them: `safety::*` (SafetyCompact, CellHint,
  FallbackEntry, SafetyProfile, fallback_for; Model accessors for the
  matching wire-format sections), `rescue::*` (RescuePolicy,
  RescueStrategy, RescueDecision, should_rescue), the typed
  `output_spec` API (`predict_with_specs`,
  `predict_with_specs_transformed`, `OutputValue`, `apply_spec`), and
  the top-K / scorer hybrid argmin family (`argmin_masked_top_k*`,
  `pick_with_confidence*`, `argmin_masked_with_scorer*`,
  `threshold_mask`, plus `bounds::*_out_of_distribution`,
  `OutputBound`). Wire-format slots still parse unconditionally — the
  feature gates only the typed Rust API.
- New `zenpredict::wire` module exposes the shared byte-offset
  constants (`HEADER_SIZE`, `LAYER_ENTRY_SIZE`, `SECTION_OFF_*`) the
  parser and `zenpredict-bake` composer both consume. Ends a
  parser/composer drift risk that existed in 0.1.
- `FeatureTransform::from_token` promoted from `pub(crate)` to `pub`
  (was a dead doc-link target before).

### Documentation

- README rewritten end-to-end for v3-only + crate split. Adds the
  migration tool pointer and the `advanced`-feature surface map.
- Comprehensive review doc at `docs/v0_2_review_2026-05-13.md`
  driving this release.
- "ZNPR v2" strings stripped from every source comment + the README;
  one mention remains in `Model` docs as a footnote explaining the
  migration path.
- 4 rustdoc warnings fixed (broken intra-doc links to
  `EncodeMetrics`, `AllowedMask::all_allowed`, `BakeRequest::default`,
  `as_token → from_token`).

### Tests

- 7 integration test files moved from `zenpredict/tests/` to
  `zenpredict-bake/tests/` (they need the composer to mint fixtures).
- `zenpredict-bake` pulls `zenpredict` with `--features advanced` as a
  dev-dep so the integration tests exercise both feature surfaces.
- New `benches/predict.rs` (in `zenpredict-bake`) covers the two
  production shapes: V0_18-zensim 228→384→1 I8, and zenwebp picker
  51→64→24 F16.

## [0.1.0] - 2025

Initial release. ZNPR v2 format. Bake composer + JSON baker bundled
in-crate behind the `bake` feature. AGPL-3.0-only or
LicenseRef-Imazen-Commercial dual-license; relicensed to MIT OR
Apache-2.0 for crates.io publication.
