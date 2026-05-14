# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that ship together in the next minor (0.x patch
     bumps stay non-breaking). Persist across patch releases. Only
     clear when the breaking release ships. -->

- `Header.*` and `LayerEntry.*` fields will become `pub(crate)` with
  accessor methods. Currently `pub` for compat with `zenavif::auto_tune`
  reading `model.header().n_outputs` directly. Migration: switch to
  `model.n_outputs()` accessor; the cross-repo update lands alongside
  the bump.
- `LayerView.*` fields may follow the same tightening once `WeightStorage`
  variant pattern-matching has a stable accessor pair (likely
  `LayerView::weights() -> &WeightStorage<'_>`).

### ROADMAP: compressed-weights feature (planned for ~zenpredict 0.3)

A `compressed-weights` cargo feature will add `WeightDtype::I8Lz4`
(byte value 3 in the wire format) for layers whose i8 weight bytes
are LZ4-block-compressed. This is the queued size-shrink path the
RLE/zero-bias and compression-eval reviews concluded on
(`zensim/benchmarks/zenpredict_rle_zerobias_eval_2026-05-13.md`,
`zensim/benchmarks/zenpredict_compression_eval_2026-05-13.md`).

**Prerequisites (must land first, in order):**

1. **Zero-bias support on the bake side** (zenpredict-bake change,
   no zenpredict-runtime change). Add a `zero_bias_threshold: f32`
   field to `BakeLayer`. When > 0, the composer thresholds
   `|W[i, o]| < tau * max(|W[:, o]|)` to exactly 0 before computing
   the per-output i8 scale. At τ=0.005 the V0_18 weight tensor goes
   from 1.4 % zeros to 87.5 % zeros at SROCC cost ≤ 0.0001 on CID22.
   This makes the weight bytes compressible — without it, raw V_X
   weight bytes are near-uniform-random and zstd/lz4/deflate all
   plateau at ratio 0.93.
2. **A baked-with-zero-bias V_X model** (zensim-side bake call adds
   the threshold). V0_18-zerobiased keeps the same MLP, just with
   weights pre-thresholded. Methodology doc accompanies the bake
   per the per-bake requirement in `zensim/CLAUDE.md`.
3. **LZ4 decoder vendor or dep** in zenpredict. The compression-eval
   report measured `lz4_flex` block decoder at 1.8 µs / 1 alloc /
   4 KB compiled binary — fastest of every candidate tested,
   meets all four bars (ratio ≤ 0.40 after zero-bias, ≤ 100 µs, ≤ 2
   allocs, ≤ 30 KB binary). Vendor scope: `src/block/decompress_safe.rs`
   (485 LOC), `src/sink.rs` (335 LOC), `src/block/mod.rs` (180 LOC)
   — ~1 KLOC total, MIT licensed, zero deps when frame format
   skipped. Skip `src/block/decompress.rs` (uses `unsafe` pointer
   copies; `#![forbid(unsafe_code)]` requires the `_safe` variant)
   and `src/frame/` (requires `twox_hash` for checksums).

**Format / API design:**

- Wire format: `WeightDtype::I8Lz4 = 3` (next free dtype byte).
  Layer bytes are: `[u32 decompressed_len_bytes][lz4_block_payload]`,
  followed by the existing per-output f32 scales section unchanged.
- Inference: single-alloc decompress into a per-layer scratch buffer
  reused across `predict()` calls (allocated at `Predictor::new`
  time, not per call). Then existing `saxpy_matmul_i8` consumes the
  decompressed i8 slice — same hot path as today, zero extra allocs
  per predict.
- "Very fast uncached expansion and evaluation" per 2026-05-13 user
  directive: decompress is per-layer, fused with the matmul (cache-
  blocking the decompressed weight stream against the f32 input
  accumulator to keep decompressed weights in L1/L2). No global
  decompressed-weight cache — re-decompress every predict call,
  trusting lz4's ~88 GiB/s decode throughput to stay under the
  matmul's own bandwidth budget.

**API:**

- New cargo feature `compressed-weights` (default-off) gates
  `saxpy_matmul_i8_lz4` and the `WeightDtype::I8Lz4` variant of the
  match arm. Without the feature, encountering byte 3 fails with
  `PredictError::UnknownWeightDtype { byte: 3 }`, so consumers
  building lean runtimes that never load compressed bakes don't
  pay the lz4 decoder cost.
- `zenpredict-bake` always supports the `I8Lz4` write path (even
  without `compressed-weights` on the runtime side); bakes built
  with i8_lz4 layers fail to load only at zenpredict runtimes
  that didn't enable the feature.

**Not doing yet**: shipping any of this in 0.2.0. The 4.5-9 % shrink
from compression alone on raw weights doesn't justify the binary
cost. The 73 % shrink lives in the **zero-bias rebake**, which is
prerequisite (1) above. Until V_X is rebaked with τ > 0, lz4 stays
on the roadmap.

## [0.2.0] - 2026-05-13

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
