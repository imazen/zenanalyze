# Publishing the zenanalyze workspace

The [pre-publication API audit](API_NECESSITY_2026-10-28.md) identifies
output-transform correctness blockers and traces the controller APIs' intended workflows.
Resolve those findings before treating the current surface as the release candidate.

See [ecosystem use and approaches to reconsider](ECOSYSTEM_USAGE_2026-10-25.md)
before treating every unpublished research interface as a release prerequisite.

## Registry state and dependency order

Verified 2026-10-25 against source commit `8ca647c5` and the
[crates.io API](https://crates.io/api/v1/crates/zenanalyze). A version in a
manifest or changelog does not establish publication. Recheck each crate at
`https://crates.io/api/v1/crates/<name>` before preparing a release.

| Crate | Source version | Published version | Immediate prerequisite |
|---|---|---|---|
| zenanalyze-api | 0.1.1 | 0.1.0 | None; zero dependencies |
| zenanalyze | 0.2.0 | 0.1.0 | zenanalyze-api 0.1.1 |
| zenpredict | 0.2.0 | 0.1.0 | No unpublished normal dependency |
| zenpredict-bake | 0.1.0 | None (registry 404) | zenpredict 0.2.0 |
| zenpicker | 0.1.0 | None (registry 404) | zenpredict 0.2.0 |

`zenpredict-viz` and `apidoc` set `publish = false`. `zenpicker-train` is
workspace-excluded local tooling with versionless path dependencies, including
zenstats; its manifest does not currently set `publish = false`. These tools
are not required to publish zenanalyze or to satisfy zenjpeg's runtime graph.

The two independent dependency chains are:

- zenanalyze-api 0.1.1, then zenanalyze 0.2.0.
- zenpredict 0.2.0, then zenpredict-bake and zenpicker if those consumers need
  registry releases. Neither bake nor picker is a normal dependency of zenjpeg.

## What the unpublished versions contain

- zenanalyze-api adds `Select::Names` and the corresponding selection/union
  behavior. The source delta against published 0.1.0 has no provider trait.
  `Select` was already non-exhaustive, so the new variant is additive.
- zenanalyze removes the four composite likelihood variants,
  `IndexedPaletteWidth` and the `composites` feature; deprecates
  `PaletteDensity`; adds the versioned feature-offer producer, expanded raw
  feature families, separate HDR gating and default-on experimental features.
  Current main also carries gray-exact covariance and scalar/FMA consistency
  repairs, with feature-definition version 2.
- zenpredict changes the model format from v2 to v3, separates the bake
  composer into zenpredict-bake, changes model/field APIs, and adds runtime
  selection, provenance, transforms, bounds, compressed payload support, FMA
  dispatch, load-time f16 expansion and SmoothHinge activation. Version 0.1.0
  cannot substitute for this runtime.
- zenpicker provides codec-family routing. Its shipped models do not match
  current analyzer feature identities; `default_route` reports the heuristic
  fallback. See [router status](../zenpicker/README.md#router-status-2026-10-10-disabled-pending-retrain).

## zenjpeg integration

At zenjpeg `702fbf9b`, zenanalyze 0.2.0 is an unconditional normal
requirement: `encode/adaptive.rs` uses `XybBquarterChromaLoss`, absent from
registry zenanalyze 0.1.0. The `target-zq` feature additionally needs
zenpredict 0.2.0. Optional dependencies still need valid registry requirements
when packaging.

The consumer lockfile resolves analyzer/API to `e6c72f99`, before current
main's covariance/scalar fixes. Its embedded picker feature order includes
both covariance columns. Publishing current main therefore needs consumer
feature/model qualification; it is not merely replacing an identical git
source with a registry source. No consumer compatibility run against the new
analyzer was performed in this audit.

## Verification on 2026-10-25

- `cargo package --locked -p zenanalyze-api` and the equivalent zenpredict
  command pass, including Cargo's package verification build.
- `cargo package --locked --no-verify -p zenanalyze` refuses the missing
  registry zenanalyze-api 0.1.1; zenpredict-bake and zenpicker refuse missing
  registry zenpredict 0.2.0. These are measured resolution blockers.
- API semver comparison against 0.1.0 with `--release-type patch` passes.
  Forced-patch comparisons expose the deliberate analyzer/predictor breaks
  (three and ten failing lint categories respectively), consistent with their
  planned 0.2.0 versions. This is not a failed compatibility claim for 0.2.0.
- `cargo test --locked -p zenanalyze-api -p zenpredict --all-targets` and
  `--doc` pass. `cargo test --locked -p zenanalyze --lib --features api,hdr`
  and the corresponding doctests pass; existing ignores remain unchanged.
- Source-commit [CI](https://github.com/imazen/zenanalyze/actions/runs/39293039724)
  passed, including Windows ARM, macOS ARM and i686.

This audit does not authorize publication. Final-candidate all-target/doc
tests, README approval, release metadata and required CI remain release gates.
