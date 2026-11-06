# zenpredict 0.2.0 readiness review — 2026-11-05

Publication is not authorized. The controller changes require signature review
through a PR. The runtime, ZNPR v3 artifact format, transport contract and baker
remain available. The no_std correctness repair is on main at `7064f1df`.

The declaration census contains **491** public declaration rows. This includes
members and re-exported paths, not 491 independent controller capabilities.

## Caller qualification and placement

The generated [declaration census](../benchmarks/predict_readiness_2026-11-05.pointer.md)
inventories every public declaration from the supported, feature and hidden
zenpredict snapshots. Its source matches are candidates, not symbol-resolved
call counts: similarly named types in other crates are not proof of use.
The following workflow review determines placement.

| Surface | Qualified workflow and source receipt | Placement / migration |
|---|---|---|
| `Model`, `Predictor`, loading errors, layers, weights, dimensions, feature transforms | zensim's crate-private MLP adapter; zenavif `src/backend_tuner.rs` loads schema-checked models and selects raw/transformed prediction; zenpicker `src/lib.rs` predicts and selects | Preserve the runtime, including inspection accessors required by the baker. No caller change. |
| `wire`, `Section`, `Header`, `LayerEntry`, activation/dtype constants, limits | `zenpredict-bake/src/composer.rs` imports wire primitives and writes the sections the runtime reads; `src/cli.rs` inspects all weight dtypes | Preserve the shared format and baker. Absence of a direct codec call does not make a wire item unused. |
| Output specs, sparse overrides, feature bounds and metadata | The baker serializes these; the runtime parses and exposes them. Bake integration suites exercise typed finishing and feature/output transforms | Keep artifact support. Existing `advanced` finishing and safety accessors remain opt-in. |
| Free argmin, range/top-K selection and quality/cost masks | zenpicker calls masked argmin; zencodecs ranks a constrained family set. Top-K/masks implement the existing shortlist workflow with shared NaN/tie/mask semantics | Preserve the common selection math and the inherited public contract; an encoder still owns verification. |
| Closure scorer | zenpipe `7aaa33ce`, `zencodecs/src/picker.rs:189` adds degraded-effort cost only to permitted family candidates | Keep the existing `advanced` opt-in. No caller change; do not conflate this with a trial controller. |
| `EncodeMode`, `PickerStrategy` | zencodecs names `EncodeMode` at its routing boundary; zenpicker uses it for viability. `EncodeMode::strategy` returns `PickerStrategy` | Keep both default-visible; moving the return type alone would break that contract. |
| Knob-veto parser/types and metadata reader | `zenpredict-bake/src/cli.rs:230` calls `Model::knob_vetoes`, inspects each typed rule and prints it | Preserve parsing/inspection and the existing apply primitive. The trainer and deployed runtime must agree on the metadata meaning. |
| Size-zone typed metadata reader and parser | No independent qualified caller found; the pre-argmin policy composes it with masks | Move typed accessors/types behind `experimental` with that policy. Generic metadata parsing preserves the artifact bytes in every configuration. A stored ceiling describes the model's policy, not a universal codec limit. |
| `EncodeBudget` | No qualified deployed caller in the named consumer sources; existing tests define the mandatory first encode, zero allowance and empty-candidate floor | Move behind `experimental`; preserve every existing budget expectation. Other users opt in rather than losing the workflow. |
| `QualityTarget`, `Trial`, `next_trial`, `best_trial` | No qualified deployed caller found. Intended workflow: codec-owned shortlist → encode → measure → direct next probe → select measured winner | Move behind `experimental`. NaN measurements cannot guide search or win; infinities keep numeric ordering; untried candidates carry no optimality guarantee. |
| `PreArgminDecision`, `resolve_pre_argmin` | No qualified deployed caller found. Intended workflow combines model reachability, size fallback and knob vetoes before constrained selection | Move behind `experimental`; add a separate hard mask and test both forbidden fallback and unreachable-but-permitted fallback. |
| Rescue/confidence/compact safety extras | Existing `advanced` runtime API, qualified by baker round trips and the documented verify/rescue workflow rather than a deployed consumer in this census | Preserve their existing opt-in. This review does not stabilize their policy or delete published capabilities. |

`advanced` also enables `experimental`, preserving existing advanced consumers'
opt-in. The default MLP and baker do not need experimental policy. The audit
example is the only identified direct call requiring the additional hard-mask
argument; it is updated in this change. Future trial-controller callers must
enable `experimental` explicitly (or retain their existing `advanced` opt-in).

The named source census covers zensim, zenpicker, zenpicker-train, zencodecs,
zenwebp, zenavif and zenjpeg. Zentrain uses the baker through its CLI and JSON
artifact interface (`tools/train_hybrid.py`, `tools/bake_picker.py`), rather
than invoking Rust controller functions. Its metadata producers remain intact.
Source evidence is the clean cloned codec mains and this workspace; it is not
a claim that every feature-gated codec consumer has been rebuilt yet.

## Controller edge contracts

The old five-argument `resolve_pre_argmin` accepted a reach mask but returned a
zone fallback without inspecting it. Treating that mask as both reachability
and a hard constraint was unsound: an unreachable target needs permission to
bypass reachability, while format/subsampling/latency exclusions must survive.

The proposed signature adds `hard_allowed: &AllowedMask<'_>` before the mutable
reach mask. It intersects that mask with hard constraints. A declared zone
fallback is returned only if its cell is hard-allowed; an absent mask entry
denies it. Otherwise the function applies vetoes and returns `Argmin`. An empty
mask authorizes no encode. A later retry may relax soft vetoes but must keep
hard constraints. Existing tests retain their expected fallback and veto order.

`best_trial` excludes NaN achieved quality under both quality and byte
preferences. No non-NaN measurement means `None`; a NaN quality target also
means `None`. Infinities retain their ordinary numeric ordering, matching
current main. Reaching trials still minimize bytes, and fitting trials still
maximize achieved quality. The existing best-effort ordering and finite tie
behavior remain unchanged. `next_trial` ignores NaN predictions/measurements
and history entries whose candidate index cannot supply a prediction. A failed
trial remains tried. Quality-based navigation is a heuristic; it does not prove
byte monotonicity. The owner selected this NaN-only policy when reconciling the
PR with main on 2026-11-05; the PR's earlier all-non-finite expectations were
updated with that explicit approval.

`EncodeBudget::passes(0)` and an empty candidate list still resolve to one.
That is the established trial allowance with a mandatory encode, not a hard
admission ceiling. The caller must reject an empty shortlist before encoding.
`Auto` and `MultiShot` still share budget resolution; no changed policy is
hidden behind a documentation correction.

## Producer signature review

`analyze_features(PixelSlice, &AnalysisQuery)` is the canonical strided, typed,
fallible producer. `extract_offer` instead takes packed RGB8 and a caller's
descriptor hash and calls the panicking RGB8 wrapper. Its only found calls
outside its own definition are local offer tests. WebP's classifier has a real
packed `offer_for_request` caller (`src/encoder/analysis/classifier.rs:601`).

Recommended signatures for review, before implementation:

```rust,ignore
pub fn extract_offer(
    input: zenpixels::PixelSlice<'_>,
    query: &AnalysisQuery,
) -> Result<zenanalyze_api::OwnedOffer, AnalyzeError>;

pub fn offer_for_request_pixels(
    input: zenpixels::PixelSlice<'_>,
    request: &zenanalyze_api::Request<'_>,
    query: &AnalysisQuery,
) -> Result<zenanalyze_api::OwnedOffer, AnalyzeError>;
```

The typed front door derives descriptor provenance before consuming the slice
and preserves query configuration while resolving the requested feature set.
One private request resolver and one results-to-offer conversion serve both
front doors. Keep the existing packed request helper for WebP; it validates
geometry and forwards a canonical RGB8 descriptor. Do not silently accept the
caller-supplied descriptor hash or add extraction behavior to the frozen
transport crate. The proposed producer signatures are reviewed here, not
implemented or declared stable by this PR. Implementing them requires their
signature review and producer parity/stride tests.

Name negotiation matches qualified training identities, or bare names for
heuristics. It does not check config/descriptor provenance. A consumer must
compare expected provenance before reuse, or establish the compatibility
precondition for a grouped extraction pass. Analyzer semver is informational.
Source comments naming nonexistent `Request::config_hash` and `Offer::matches`
were corrected; the transport implementation is unchanged.

Selection convenience wrappers retain `predict`'s raw-model-input semantics.
Raw source features compose as `predict_transformed` → free selection. No new
wrapper matrix or change to the existing raw prediction contract is introduced.

## zensim's public boundary

At zensim `fb64941bcfa9`, complete `cargo +nightly public-api -p zensim` listings
were generated separately with defaults, `legacy-models`, `training`, and all
features. No listing contains `zenpredict`, including error types, associated
signatures and `From` implementations. Each includes three `From` impls.

| Configuration | Listing lines | zenpredict references |
|---|---:|---:|
| default | 1,441 | 0 |
| legacy-models | 1,447 | 0 |
| training | 1,615 | 0 |
| all features | 2,018 | 0 |

`zensim/src/lib.rs` keeps `mlp` crate-private, and its adapter's zenpredict
re-exports are `pub(crate)`. No public-boundary fix is needed. The full listings
and SHA-256 receipts are in the evidence pointer linked above.

## Current-main reconciliation

This review preserves main `17d9cf8e` and the prior PR head `c55554da` as merge
parents, preserving the published review history. The LZ4 0.14.0 and
Arrow/Parquet 60.1.0 dependency updates are retained. The current-main
provenance documentation is retained without introducing automatic transport
checks. The directed-search implementation and all main regression tests are
kept verbatim; the PR's additional NaN cases follow the owner's NaN-only
choice. The producer signatures and frozen transport remain unchanged.

The refreshed declaration census still contains 491 rows. A current source
scan of zenpipe, zensim, zensimx, zensr, zenavif, zenwebp, zenjpeg and zenpicker
found no new qualified caller of the newly gated controller APIs. zenpipe's
`EncodeBudget` and zenavif/zenpipe `QualityTarget` hits are their own types.
This is a source review, not a claim that every consumer configuration has
been rebuilt. Evidence belongs to the current refresh directory linked from
the [receipt](../benchmarks/predict_readiness_2026-11-05.pointer.md).

Main advanced again to `9c8d6c55` during the refresh. The PR also merges that
commit, preserving head `33c49dcd`. It widens the archmage/magetypes upper bounds
to `<0.10.0`; registry resolution still uses 0.9.30 for both, and `cargo update`
changes zero packages. This admits a future stable 0.9.31 without claiming it
has been published or tested here. The standalone fuzz lock resolves its former
beta dependency to stable 0.9.30 under the new range. The source implementation, root dependency lock
and public-surface proposal are unchanged by this final base refresh.

## Current qualification

Runtime readiness passes without defaults, with no_std plus advanced, and with
all features. The workspace's all-feature library/integration tests pass,
including the recorded metapicker artifact and current shipped zensim bake.
Workspace doctests, strict all-target Clippy, fuzz compilation and unchanged API
snapshot checks pass. Predictor package verification also passes.

Both native and forced-scalar reference goldens pass on x86. The strict
linear-light reference mode is x86-only, as documented in `src/versioning.rs`.
Forcing that mode on ARM reproduces the same 39 failing feature/value rows on
main `17d9cf8e` and this PR; ARM's intended portable golden passes. No golden
value, tolerance or ignore was changed.

The publication recipe now passes the exact predictor manifest to semver-checks,
so Cargo's staged package copy cannot make its package selection ambiguous.
The current 0.1.0 → 0.2.0 comparisons pass for default/all features by recognizing
the planned pre-1.0 breaking bump: each runs zero compatibility checks and skips
254. They do not establish patch compatibility. The forced-patch audit runs
223 checks: 212 pass, 11 fail and 31 skip (`rc=100`), exposing the planned
v2/v3/baker-split breaks. The full x86 cohort and corrected publication recipe
complete together with `rc=0`; logs and source identities are in the current
refresh receipt. This evidence does not authorize publication or PR merge.

## Earlier gate evidence

`just predict-readiness` exercises no defaults, no_std plus advanced, and all
features, then strict Clippy with default-free, experimental and all-feature
configurations. It includes nonzero no_std transforms and controller boundary
cases; established tests and ignored doctests are preserved.

`just predict-publish-check` runs runtime/API all-target tests and doctests,
baker compatibility tests/doctests, package verification and semver comparisons
against registry 0.1.0 with default and all features. Its output and CI results
must be checked before treating this as a release candidate. No publish, tag
or release command is part of either recipe.

Historical validation at the earlier review head (before the current-main refresh): 144 no-default runtime tests, 159 no_std-plus-advanced
tests, 162 all-feature tests, strict Clippy in all three configurations, the
runtime/API and baker compatibility suites/doctests, package verification,
snapshot regeneration, and analyzer api/hdr library/doctests (235 library
tests pass; four existing ignores remain). The wrapper completed with `rc=0`,
92 seconds, peak RSS 0.54 GiB. No performance result is inferred from this run.

The declared 0.1.0 → 0.2.0 comparisons pass for default and all features, but
semver-checks treats that pre-1.0 minor bump as breaking and skips all 260
compatibility checks. To inspect the actual break set, `just predict-semver-audit`
forces patch compatibility: **229 checks, 218 pass, 11 failing lint categories,
31 skipped** (`rc=100`, expected incompatibility evidence). The failures include
the pre-existing v2→v3/baker split, changed Model lifetime and tightened wire
fields; this is not a claim that 0.2.0 is patch-compatible with 0.1.0.

The already-landed correctness commit's CI is green:
[CI run 41990408982](https://github.com/imazen/zenanalyze/actions/runs/41990408982).
The controller PR must receive its own CI and owner review before landing.

The review was refreshed on shared-pixel main `bb88b421` while retaining the
existing review head as a merge parent. `just predict-readiness
sibling-cohort-check` passes on that merged source: 144 no-default, 159
no_std-plus-advanced and 162 all-feature runtime tests; the full workspace passes
674 runtime tests and 21 doctests, with existing ignores unchanged. Strict
runtime/workspace lint and fuzz compilation pass. Wrapper: rc0, 38s, peak RSS
0.86 GiB, minimum available 7,885 MiB, peak load 3.13. This is qualification on
the new dependency cohort; the earlier package/semver evidence remains identified
separately above. The analyzer's unchanged feature-name/value goldens pass.

The strict-reference follow-up supersedes the initial dependency-cohort refresh:
main `38dc80f7` repairs linear-light stored-color compatibility and includes both
CI reference modes in local qualification. The review merges that main while
retaining review head `99eeb787`. Its full cohort passes674 runtime tests,
21 doctests and both native/scalar reference goldens, with existing ignores
unchanged; strict lint and fuzz compilation pass. Wrapper: rc0,41s, peak RSS
0.86GiB, minimum available7,688MiB, peak load2.99. The predictor proposal itself
is unchanged; the package and semver evidence above remains separately identified.
