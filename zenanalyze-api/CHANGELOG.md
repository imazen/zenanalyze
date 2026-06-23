# Changelog

## [Unreleased]

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

[Unreleased]: https://github.com/imazen/zenanalyze/commits/main
