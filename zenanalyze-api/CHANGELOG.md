# Changelog

## [Unreleased]

### Added
- `Select::Names(&[&str])` — version-**agnostic** selection by bare feature name, next to the
  version-pinned `Select::Features`. Threshold heuristics, content classifiers, diagnostics
  and bulk export can now name features without naming a `zenanalyze` version — the pressure
  that previously pushed such consumers onto a direct `zenanalyze` dependency. Models must
  keep using `Features`: the code-drift miss is the safety property `Names` gives up.
- `FeatureProvider` — the extraction **intermediary**, object-safe (`&dyn FeatureProvider`):
  `analyzer_version()`, `catalog() -> OwnedCatalog`,
  `extract_rgb8(rgb, w, h, request) -> Result<OwnedOffer, ProviderError>`. A codec with no
  `Offer` to reuse can now run its own pass without naming a `zenanalyze` type; the host picks
  the version and injects the impl (`zenanalyze` ships one behind its `api` feature).
- `ProviderError` (`BadInput` / `Unavailable` / `OutOfMemory`, `#[non_exhaustive]`,
  `core::error::Error`).
- `OwnedCatalog` — the owned twin of `Catalog`, for a provider whose qualified names are only
  known at runtime: `new`, `available()` (lends borrowed `NamedFeature`s), `len`, `is_empty`,
  and the same `offers` / `has_name` / `unmet` / `union` queries.

### Verified

- `cargo semver-checks check-release --baseline-version 0.1.0`: **196 checks, 196
  pass, 0 fail — "no semver update required."** Every addition above is
  compatible with published `0.1.0`, which is the property the freeze plan rests
  on: the contract can grow the surface consumers need without splitting anyone,
  because private fields and `#[non_exhaustive]` make additions non-breaking.
  Run this before every release of this crate — a break here splits the
  ecosystem, and it is the one crate where that is unrecoverable.
- `cargo package -p zenanalyze-api`: 9 files, 25.2 KiB compressed.

### Documentation
- README gained a **Compatibility rules** section — the four rules that make a multi-version
  build work: one dependency source (a crates.io *version*, never a git rev; override with a
  single workspace-root `[patch.crates-io]`), sole contract (a codec's library code names only
  this crate; direct `zenanalyze` belongs to the host and to dev tooling), no zenanalyze types
  across the boundary, and identity-versioned feature layout.
- The README coverage tripwire now parses `pub trait` bodies, so a new trait method is
  required to be documented like any other public item.

## [0.1.0] - 2026-06-23

First release — the version-unifying feature contract for the zenanalyze picker tree.
`no_std + alloc`, no dependencies, `forbid(unsafe_code)`. The README is the crate doc (every
example is a compiled doctest) plus a coverage tripwire that fails the build if a public item
is undocumented.

### Added
- `NamedFeature` — a feature's cross-version identity carried as a single qualified
  `name@hex8` string (the per-feature **code** version folded in via `fold_hash`). `const`
  `parse` (validating, builds compile-time-checked presets) / `from_qualified` (unchecked) /
  `is_valid_name` / `fold_hash` / `qualified_for`, plus `name` / `version_hash` /
  `qualified_name` accessors, `Display`, and `TryFrom<&str>`.
- `Value` (`F32` / `U32` / `U64` / `Bool`, `#[non_exhaustive]`) mirroring zenanalyze's native
  feature outputs, with the canonical `to_f32` projection built in. `FeatureResult` carries it:
  `value()` is the native `Value`, `float()` the canonical `f32` (the model-input currency).
- `Offer` and its owned twin `OwnedOffer` (of `OwnedFeatureResult` cells) — the self-describing
  result, negotiated **purely by qualified name** (`satisfies`, `reuse_for`, `get`), with the
  `schema_hash` file-blend gate. `OwnedOffer::new` (from deserialized parts — the parquet/TSV
  path) and `OwnedOffer::parse` (the dependency-free `zenanalyze-features/1` text form).
- `Provenance` (`#[non_exhaustive]`, builder) — informational `analyzer_version` + optional
  `config_hash` / `descriptor_hash`, feeding `schema_hash`; not a reuse gate.
- `Select` / `Request` (the consumer's ask), `Catalog` (`union` + availability queries),
  `FormatError`.

[Unreleased]: https://github.com/imazen/zenanalyze/compare/zenanalyze-api-v0.1.0...HEAD
[0.1.0]: https://github.com/imazen/zenanalyze/releases/tag/zenanalyze-api-v0.1.0
