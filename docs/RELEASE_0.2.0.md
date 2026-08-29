# Releasing zenanalyze 0.2.0

Prepared 2026-08-28. **Nothing here publishes anything** — every command below
was run in dry-run / check form and its result recorded. The publishing steps at
the end are for the repo owner.

State at preparation time: `Cargo.toml` says `0.2.0`, **crates.io serves
`0.1.0`** (2026-04-28). Everything since is unreleased. `zenanalyze-api` is in
the same shape: `0.1.1` in tree, `0.1.0` published.

## 0. The blocker this release clears

`jxl-encoder`'s `learned-admission` feature is **default-on** and declares
`zenanalyze = { version = "0.2.0" }`, resolved locally by a
`[patch.crates-io]` to the sibling checkout. That patch cannot follow the crate
to crates.io, so **jxl-encoder 0.4.0 cannot be published until zenanalyze 0.2.0
is.** Same shape in `zenavif` (`version = "0.2.0", path = "../zenanalyze"`).

## 1. Publish order (strict)

| # | Crate | Version | Why this position |
|---|---|---|---|
| 1 | `zenanalyze-api` | 0.1.1 | `zenanalyze 0.2.0`'s `api` feature imports `FeatureProvider` / `OwnedCatalog` / `ProviderError`, which do not exist in published 0.1.0. **Measured**: build the packaged crate with `--features api` against the registry and it fails `E0432: unresolved imports`. `Cargo.toml`'s requirement is now `0.1.1`, so until this publishes, `cargo publish --dry-run -p zenanalyze` cannot resolve — that failure is correct, not a regression. |
| 2 | `zenanalyze` | 0.2.0 | The crate this document is about. |

Not part of this release, and not blocking it: `zenpicker`, `zenpredict`,
`zenpredict-bake`, `zenpredict-viz` are workspace members but **not**
dependencies of `zenanalyze` — `zenanalyze`'s only in-repo dependency is
`zenanalyze-api`. `zenpicker-train` and `apidoc` are workspace-`exclude`d and
unpublishable by design (`zenpicker-train` pulls a cross-repo path dep on
`zenstats` in the sibling `zenmetrics` checkout).

### External dependency audit (checked 2026-08-28 against crates.io)

| dep | required | published latest | note |
|---|---|---|---|
| `zenpixels` | ^0.2.14 | 0.2.16 | resolves 0.2.16; fine |
| `zenpixels-convert` | ^0.2.14 | 0.2.16 | fine |
| `archmage` | ^0.9.27 | 0.9.28 | fine |
| `magetypes` | ^0.9.27 | 0.9.28 | fine |
| `garb` | ^0.2.8 | 0.2.8 | exact latest |
| `linear-srgb` | ^0.6.12 | 0.6.12 (0.7.0 exists) | deliberate: staying on the 0.6 line |
| `zenresize` (dev) | ^0.3.1 | 0.3.1 | dev-only, not in the published graph |
| `zenbench` (dev) | ^0.1.8 | 0.1.9 | dev-only |

Every non-dev dependency is a registry version, published, and satisfiable — no
path-only dependency has to publish first except `zenanalyze-api`.

## 2. What is breaking, and that it is deliberate

The complete list, with the lint that found each and the rationale, is in
`CHANGELOG.md` under `[0.2.0] - unreleased` → "Breaking changes since published
0.1.0". Summary: 5 removed `AnalysisFeature` variants (ids 27/28/29/30/45), the
removed `composites` cargo feature, and `PaletteDensity` (12) becoming
`#[deprecated]`. All five ids are in `RESERVED_RETIRED_IDS` and are never
recycled.

How it was measured — and why the obvious invocation is useless:

```bash
# Reports "no semver update required" and checks NOTHING: 0.1 -> 0.2 is a 0.x
# minor bump, which semver-checks treats as allowed-to-break, so it skips all
# 253 checks.
cargo semver-checks check-release -p zenanalyze --baseline-version 0.1.0

# The one that measures. --release-type patch forces every check to run.
cargo semver-checks check-release -p zenanalyze --baseline-version 0.1.0 \
  --release-type patch          # 223 checks: 220 pass, 3 fail
cargo semver-checks check-release -p zenanalyze --baseline-version 0.1.0 \
  --release-type patch --all-features   # identical 3 failures
```

Two breaks are invisible to `semver-checks` because they move items between
cargo features rather than removing them, and are recorded in the changelog:
`tier_depth` moved from `experimental` to the new `hdr` feature, and
`experimental` became default-on.

## 3. Verified before this document was written

| Check | Command | Result |
|---|---|---|
| Break list | `cargo semver-checks … --release-type patch` | 223 checks, 220 pass, **3 fail**, all deliberate and listed |
| Public-API snapshots | `ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml` | was **stale** (173-line snapshot vs a 211-line surface); regenerated and committed, now passes |
| Packaging | `cargo publish --dry-run -p zenanalyze` | **rc=0** — 66 files, 1.4 MiB (430.5 KiB compressed), verify-build clean *(run before the `zenanalyze-api` requirement bump; see §1 for why it now blocks on step 1)* |
| `api` feature off the registry | build `target/package/zenanalyze-0.2.0` with `--features api` | **failed** `E0432` → fixed by requiring `zenanalyze-api 0.1.1` |
| Feature counts | `cargo run --example list_features` × 5 combos | 97 base / **101** default / 117 default+hdr / 113 hdr-only — README corrected |
| README claims | manual, against source + `benchmarks/` | 6 wrong numbers + 8 crates.io-dead relative links fixed |

## 4. CI lanes

`.github/workflows/ci.yml` covers every lane the release policy requires:

| lane | job | status |
|---|---|---|
| `windows-11-arm` | `test` matrix `os` | present |
| macOS Intel | `test` matrix `macos-15-intel` | present |
| macOS aarch64 | `test` matrix `macos-latest` | present |
| Linux x86-64 | `test` matrix `ubuntu-latest` | present |
| `i686-unknown-linux-gnu` | `cross` matrix | present (via `cross`) |
| `aarch64-unknown-linux-gnu` | `cross` matrix | present |

The `test` matrix runs all four feature combinations (`""`, `experimental`,
`hdr`, `experimental,hdr`) on all four OSes, plus the `api` feature, the
golden-value tripwire (`golden-reference`, x86-64 blesses / every platform
checks) and the `rsqrt-probe` cross-platform determinism job.

**Gap worth knowing about before publishing:** no CI job runs
`cargo publish --dry-run` or `cargo package`, so packaging problems (the `api`
E0432 above is exactly one) cannot be caught by CI. Adding a `package` job is a
one-liner and is recommended, but is a change to CI policy and is left to the
owner.

## 5. Consumers that unblock (verified read-only, 2026-08-28)

Checked **statically** — every `zenanalyze::` / `zenanalyze_api::` path each
consumer names was looked up in this build's regenerated public-API snapshot and
in `zenanalyze-api/src/lib.rs`. Their working copies all held another agent's
uncommitted work, so nothing was built or modified in them.

| repo | names | verdict |
|---|---|---|
| `jxl-encoder` | library (`src/vardct/learned_admission.rs`, `src/s4_eps.rs`, default-on `learned-admission`): `try_analyze_features_rgb8`, `AnalysisQuery::new`, `FeatureSet::just`/`with`, `FeatureValue::{U32,F32}`, and 11 `AnalysisFeature` variants (`DistinctColorBins`, `EdgeSlopeStdev`, `LaplacianVariance`, `LaplacianVarianceP99`, `FlatColorBlockRatio`, `GradientFraction`, `HighFreqEnergyRatio`, `AqMapStd`, `GrayscaleScore`, `LumaHistogramEntropy`, `QuantSurvivalY`) | **all present on the default (no-`experimental`, no-`hdr`) surface** — satisfied by 0.2.0 |
| `zenavif` | `analyze_features_rgb8`, `analyzer_version`, `versioning::feature_version_hash_by_name`, `FeatureSet::{new,SUPPORTED}`, `AnalysisQuery::new`, `AnalysisResults`, 8 `AnalysisFeature` variants; contract side `Request::new`, `Select::Features`, `NamedFeature::{from_qualified,qualified_for,fold_hash}`, `Offer::new`, `OwnedOffer::new`, `OwnedFeatureResult`, `Provenance::new` | **all present** on zenanalyze main + zenanalyze-api main |
| `zenpipe` / `zencodecs` | `zenanalyze_api::Offer` only, contract by registry version with one root `[patch.crates-io]` | **satisfied**; this is the exemplar shape (`docs/sole-contract.md`) |

Note `jxl-encoder` is the one consumer still naming `zenanalyze::` from library
code — that is [imazen/jxl-encoder#98](https://github.com/imazen/jxl-encoder/issues/98)
and is orthogonal to this release: it compiles today and will keep compiling.

## 6. Steps for the owner

Do them in this order; **stop at the first failure.**

1. Re-read `README.crates.md` (the crates.io page) and approve it. It is
   generated — edit `README.md` and re-run
   `sh ../zenutils/scripts/gen-readme-crates.sh .`, never edit it directly.
2. `cargo test --all-targets` and `cargo test --doc` locally, green.
3. `cargo semver-checks check-release -p zenanalyze-api --baseline-version 0.1.0`
   — must be **0 fail** (a break in the contract splits the ecosystem; last run:
   196 checks, 196 pass).
4. Push, wait for CI green on **every** lane in §4 — including
   `windows-11-arm`, `macos-15-intel`, and both `cross` targets.
5. Tag and release `zenanalyze-api` 0.1.1, then `cargo publish -p zenanalyze-api`.
6. `cargo publish --dry-run -p zenanalyze` — now resolvable; must be rc=0.
7. `git tag v0.2.0 && git push origin v0.2.0`, then
   `gh release create v0.2.0 --title "v0.2.0" --generate-notes`.
8. `cargo publish -p zenanalyze`.
9. Unblock the consumers: drop the `[patch.crates-io]` in `jxl-encoder` and the
   `path =` in `zenavif`, then release `jxl-encoder 0.4.0`.
