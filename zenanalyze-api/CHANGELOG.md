# Changelog

## [0.1.1] - unreleased

> `Cargo.toml` is at `0.1.1`; crates.io still serves `0.1.0`.

**0.1.1 = 0.1.0 + one enum variant.** That is the whole release.

### Added
- `Select::Names(&[&str])` — version-**agnostic** selection by bare feature name, next to the
  version-pinned `Select::Features`. Threshold heuristics, content classifiers, diagnostics
  and bulk export can now name features without naming a `zenanalyze` version — the pressure
  that previously pushed such consumers onto a direct `zenanalyze` dependency. Models must
  keep using `Features`: the code-drift miss is the safety property `Names` gives up.

  This is the selector the contract's actual flow needs. The model is *push* — the host runs
  one pass and **gives** the codec the data, the codec answers yes/no, and on "no" it runs its
  own scan. Answering "is this enough?" has to work *across analyzer versions*, which is
  precisely matching by bare name at whatever version is on offer. `Features` pins the version
  hash and `All` is everything; `Names` is the only selector that expresses the question.

### Removed before publication

Four items were added during 0.1.1's development and **cut before it shipped**. They were
never on crates.io, so nothing outside this repo could depend on them; this section exists so
the record is not silently rewritten.

- `FeatureProvider` (trait: `analyzer_version` / `catalog` / `extract_rgb8`)
- `ProviderError` (`BadInput` / `Unavailable` / `OutOfMemory`)
- `OwnedCatalog` (struct + `new` / `available` / `len` / `is_empty` / `offers` / `has_name` /
  `unmet` / `union`)

17 permanent public items — 3 types, 3 trait methods, 3 enum variants, 8 inherent methods,
plus `Display` and `core::error::Error` impls — removed for one reason: **the contract is
data, not behaviour.** In order of force —

1. **The trait was the wrong direction of control.** Push hands data across a boundary; a
   `&dyn` lets a codec reach *back* into a live analyzer to pull values. Every verb the
   intended flow needs — `satisfies` (yes/no), `reuse_for` (the values), `get` (which wants
   were missing, and whether a present one drifted) — was already in **0.1.0**. The trait sat
   on none of those steps. `push_model_answers_yes_no_and_names_the_gaps` in `src/tests.rs`
   is that flow end to end, in 0.1.0 verbs plus `Select::Names`, and it is the proof there was
   no gap to fill.
2. **Data serializes; a trait does not.** An `Offer` crosses a process, a file, a cache and a
   *version* boundary — `to_block` / `parse` are right here. A `&dyn` crosses none of them.
   For a contract whose entire purpose is letting analyzer versions coexist, serializable data
   is strictly more powerful than dynamic dispatch.
3. **A trait is a promise about _how_.** This surface freezes at `1.0` and never breaks after.
   A decade is too long to freeze someone else's method set.
4. **It dragged a pixel-buffer shape into a contract that otherwise has nothing to do with
   pixels — and got it wrong.** `extract_rgb8` took tightly-packed RGB8 with no row stride,
   against this workspace's pixel-buffer rule, and it had already cost a consumer a
   full-image copy per call to work around (`zensr`'s `center_crop_rgb8` allocated a
   512×512×3 buffer for exactly this reason). With no pixel buffer in the contract, that whole
   class of mistake is unrepresentable.

`ProviderError` and `OwnedCatalog` existed only to serve the trait — the former as its error
type, the latter as `catalog()`'s return. `Catalog<'a>` covers the borrowed case and has been
frozen in 0.1.0 all along.

The producer-side replacement lives in `zenanalyze`, not in the contract:
`zenanalyze::offer_for_request(rgb, w, h, &request)` runs one pass answering a `Request` and
returns an `OwnedOffer`. The `Select` → `FeatureSet` resolution stays in one place, and a
codec calls it from a function *body* — where a direct `zenanalyze` dependency has been
explicitly permitted since 2026-08-28.

### Verified

- `cargo semver-checks check-release --baseline-version 0.1.0` — see the release notes below;
  re-run before every release of this crate. A break here splits the ecosystem, and it is the
  one crate where that is unrecoverable. Treat a green run as a lower bound, not a clearance:
  it has no lint for an inherent method's return type and cannot model behavioural breaks.
  Compiling the real consumers is the actual proof.

### Documentation

- README gained **"Why there is no provider trait"** — the four reasons above, kept in the
  docs so the trait is not re-proposed in a year as an obvious gap. It is a decision, not an
  omission.
- **Policy correction (owner, 2026-08-28, verbatim: "a direct dep is okay though, a
  reanalysis might be needed anyway if the upstream provided features are
  insufficient").** The README's compatibility rules had been written as a prohibition —
  a codec's library code may name this crate and nothing else. That was too strict.
  This crate is the **interchange boundary**, not a ban on depending on `zenanalyze`: a
  codec whose offer doesn't cover what it needs should re-analyse. What stays hard is the
  *boundary* — a public signature naming `zenanalyze::feature::*` pins every caller to your
  analyzer version — plus registry-version deps (never a git rev) and no absolute-path pins.
- README gained a **Compatibility rules** section — the four rules that make a multi-version
  build work: one dependency source (a crates.io *version*, never a git rev; override with a
  single workspace-root `[patch.crates-io]`), interchange types at the boundary, transport-only
  and dependency-free, and identity-versioned feature layout.
- The README coverage tripwire now parses `pub trait` bodies, so a trait method would be
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
