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
