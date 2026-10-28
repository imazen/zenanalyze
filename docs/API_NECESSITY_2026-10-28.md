# Pre-publication API necessity audit

Audited source `208ee7ec` on 2026-10-28. **Keep the extractor, transport contract,
MLP runtime and baker. Narrow the controller surface before publishing it.**
There are also reproducible output-transform defects to resolve before release.
This audit changes no library API, model, numeric behavior or test expectation.

## Release baseline

The crates.io API still reports **0.1.0** for each of
[zenpredict](https://crates.io/api/v1/crates/zenpredict),
[zenanalyze](https://crates.io/api/v1/crates/zenanalyze), and
[zenanalyze-api](https://crates.io/api/v1/crates/zenanalyze-api), checked today.
Downloaded published source confirms that zenanalyze-api's entire new public
surface is `Select::Names`. Its accompanying changes implement selection and
catalog union; no provider trait remains.

**Top-K, scorer, confidence, rescue and compact safety types already existed
in published zenpredict 0.1.0.** Their movement between default and `advanced`
features is not creation of a new API. Encode budgets/modes, directed search,
knob vetoes and unachievable zones are new relative to that release. The v3
parser, separate baker and feature-transform pipeline are also new. Distinguish
these groups when deciding what can be trimmed before 0.2.0.

Consumer evidence includes the immutable-ref
[October 25 census](ECOSYSTEM_USAGE_2026-10-25.md) and its
[evidence pointer](../benchmarks/dependency_usage_2026-10-25.pointer.md).
Dependency edges, comments and experimental branches do not establish a live
production caller. That census is a dated observation, not today's deployment
inventory. A negative source search is insufficient grounds to delete wire
support or a published item.

A supplementary symbol pass today scanned captured branch/remote tips in seven
repositories: jxl-encoder, zenavif, zenmetrics, zenpipe, zensimx, zentone and zenwebp.
An existing lock excluded zenjpeg; zensr and coefficient were absent from that
checkout inventory and remain covered only by the earlier census. Refs were not
refreshed, and working-file edits were excluded. This pass found no calls to
`PickerStrategy`, `next_trial`, `best_trial`, `resolve_pre_argmin`,
`apply_knob_vetoes`, `unachievable_zones` or `pick_with_confidence`. It found the
zencodecs closure scorer and offer-request calls in WebP. Its `EncodeBudget`
matches include **zencodecs' own type**, not imports of zenpredict's type;
jxl-encoder also has its own unrelated `EncodeMode`. Namesake hits are not
evidence of runtime API adoption. No zenanalyze `feature_vector` or
`resolve_feature_ids` calls were identified in that supplementary sample.

## Findings to resolve before publication

### 1. Output transforms silently compute different functions without std

[`OutputTransform::apply`](../zenpredict/src/output_spec.rs) substitutes identity
for `Exp` without `std`; `sigmoid` returns 0.5 for every input. These are callable
on the default API even without `advanced`. The `SigmoidScaled` result consequently
collapses to the midpoint. Existing sigmoid/exp tests are std-gated.

The probe prints:

```text
std=false
Exp(1)=1
Sigmoid(1)=0.5
std=true
Exp(1)=2.7182817
Sigmoid(1)=0.7310586
```

This contradicts the README's claim that no_std computes the same functions.
`libm` is already an unconditional dependency and `ScoreTransform::Exp` already
uses it. **Fix these branches and add nonzero no_std cases.** No additional API
or dependency is needed. This is a correctness blocker, not a reason to weaken
the no_std contract.

### 2. Round is numerically wrong in both builds

The same module implements rounding as `(x + signed_half) as i64 as f32`.
Measured in both std and no_std builds:

```text
Round(16777215)=16777216, reference=16777215
Round(inf)=9223372000000000000, reference=inf
```

The integer cast also saturates large finite inputs. **Use correct rounding
semantics and cover boundaries/special values.** The public variant's comment
incorrectly calls `f32::round` banker's rounding. Correct the comment with the fix.

### 3. Selection wrappers and baked input transforms have two meanings of input

[`Predictor::argmin_masked`](../zenpredict/src/predictor.rs) and the top-K/range
wrappers call `predict`, which intentionally bypasses baked feature transforms.
`predict_transformed` applies them. For one valid log1p bake with raw input `[3]`,
the wrapper chooses cell **1**, while transformed prediction followed by the free
argmin chooses cell **0**. This is a demonstrated API composition hazard; it does
not prove that every current consumer misuses the wrappers.

The existing raw `predict` behavior has explicit regression coverage. **Preserve
that behavior unless an API change is approved.** Prefer one documented public
composition: raw features → `predict_transformed` → free selection function.
Before retaining more convenience wrappers, decide explicitly whether they take
caller-transformed or raw model inputs. Internally share feature preparation and
output post-processing between the transformed/typed entry points; they currently
repeat the transform/expander routing. Do not add another complete wrapper matrix
to paper over this distinction.

### 4. The new offer producer freezes an unnecessarily narrow pixel shape

[`extract_offer` and `offer_for_request`](../src/offer.rs) take packed RGB8,
width and height, with no stride. `extract_offer` calls the panicking RGB8
wrapper and accepts a caller-supplied descriptor hash. The caller can therefore
stamp framing independently of the pixels actually analyzed. The request helper
checks dimensions but always constructs the default analysis query.

The canonical `analyze_features(PixelSlice, query)` already supports strided and
typed input. **Retain one request resolver and one result-to-offer conversion,
then make the producer front door fallible and descriptor-aware.** A PixelSlice
entry can derive framing from the actual source and preserve query configuration.
Keep packed convenience entry points only for demonstrated callers. This needs
signature review before implementation; it belongs in zenanalyze, not in the
frozen transport crate. Changing a signature is not authorized by this audit.

### 5. Qualified names do not validate analysis configuration

Two offers carrying the same qualified feature names but config hashes 0 and 1
both satisfy the same `Select::Features` request. Their schema hashes differ.
This is intentional current transport behavior, confirmed by the probe; it is
not a newly introduced break. A constant configuration within one pass does not
prove compatibility with a model trained under a different configuration.

**Keep the frozen contract intact.** The orchestrator/consumer must compare
existing provenance and the model's expected config before reuse, or establish
and document that precondition when grouping passes. Feature identities must
come from training, not be regenerated from the current extractor. Neither
analyzer semver nor a global definition counter substitutes for these checks.

Several current comments still describe nonexistent `Request::config_hash`,
`Offer::matches` or an automatic three-part reuse gate: see
`AnalysisQuery::config_hash`, `analyzer_version`, metadata provenance keys and
the [historical contract design](feature-contract-pr-2026-06-19.md). Reconcile
these with the implemented name-only contract before release.

### 6. Safety/search abstractions still contain policy and boundary gaps

The probe makes `resolve_pre_argmin` return fallback cell **1** while
`allowed[1]` is **false**. Its zone branch returns before inspecting the mask.
The documented encode-directly composition can therefore bypass caller
constraints. If retained, fallback selection must honor a separate hard
constraint mask; predicted reachability and hard constraints are different.

`best_trial` chooses a trial whose achieved quality is NaN over a finite-quality
trial when none reaches the target. `f32::total_cmp` treats that NaN as larger.
Define invalid-measurement handling before making this a generic public search
contract. `next_trial` also assumes predicted quality provides a useful search
ordering; no general relationship to encoded bytes follows from that alone.

`EncodeBudget::passes(0)` resolves to **1**, as does an empty candidate set.
This floor is explicitly tested and documented, so it is **not** an accidental
regression to fix silently. It is evidence that this type means a trial allowance
with a mandatory encode, rather than a universally binding resource ceiling.
`Auto` and `MultiShot` currently execute identical resolution logic.

**Proposal:** defer the unused controller policy rather than commit consumers
to these semantics now. If any part stays public, specify these cases and test
composition against actual codec constraints first. Preserve existing expectations
until a changed contract is approved.

## Recommended surface

| Surface | Disposition and reason |
|---|---|
| zenanalyze-api `Select::Names` | Keep. WebP's heuristic classifier has an actual caller; this is the missing version-agnostic selector. |
| Existing transport types and methods | Keep frozen. Do not add a provider/extraction trait, pixel arguments, an owned catalog or a request-builder framework. |
| `analyze_features`, typed queries/results, requested feature sets | Keep. Canonical implementation and sparse requests have real codec/export roles. |
| Offer production/request resolution | Keep the capability; repair input shape and configuration semantics before stabilizing convenience signatures. |
| Numeric-id `feature_vector*`, lookup/all/packed wrappers | Candidate to trim unpublished convenience wrappers after a final consumer check. They are adapters over the typed extractor, not another implementation, but their bool/NaN contract discards failure detail. No adoption was identified in the supplementary sample; do not expand this parallel facade without a caller needing it. |
| `Model`, `Predictor`, transforms, dtype support, v3 compression, bounds | Keep. Runtime/artifact use supports these; none of the audit probes justifies dropping wire capabilities. |
| Free argmin/top-K and runtime masks | Keep shared selection math. They avoid replicated NaN, tie and mask semantics. Top-K is inherited API, not a new speculative release dependency. |
| Closure scorer | Keep the demonstrated capability: zencodecs calls it. Do not sweep it away with unused confidence/rescue wrappers. |
| `EncodeMode` | An actual zencodecs boundary names it, and zenpicker consumes it. Consider relocating application policy only with that consumer migration. |
| `EncodeBudget`, `PickerStrategy`, directed search | Propose keeping out of the permanent default surface pending concrete callers and policy qualification. Do not equate their tests with a demonstrated codec requirement. |
| Knob-veto / unachievable-zone / pre-argmin composition | Propose experimental or tool-side ownership until deployed composition is demonstrated. Preserve metadata readability and the baker's canonical encoding. |
| Rescue/confidence/typed output extras | Inherited or feature-gated capabilities; assess individual consumers before pruning. `advanced` reduces default exposure but does not make a public API private. |
| `wire`, `Section`, limits, model inspection views | Keep the cross-crate needs of the separate baker/inspection tools. Move code internally before inventing a new public wire crate or duplicate parser. |
| New graph runtime or generic controller framework | Defer. No inspected v3 artifact requires a DAG. See the existing graph compatibility finding in the ecosystem audit. |

## Internal complexity worth reducing

1. **Share transformed prediction preparation and typed output finishing.** Keep
   raw versus transformed semantics explicit; remove repeated dispatch bodies.
2. **Avoid owned-offer lookup allocations.** `OwnedOffer::satisfies` currently
   builds a temporary `Vec<FeatureResult>` before scanning; `reuse_for` builds
   that vector and then the requested output vector. Borrowed and owned storage
   can use private shared iteration without adding a public trait or cache.
   No speed or memory improvement was measured here.
3. **Keep SIMD and row-format abstraction.** The u8/f32 `ChunkInput` dispatch
   shares tier arithmetic and preserves HDR input. It is functioning production
   code, not the unfinished foundation described by some older comments. Removing
   it would change behavior; private generic kernels are justified complexity.
4. **Keep tier gating and fixed sampling budgets.** They implement narrow queries
   and bounded work. Do not replace them with public tuning knobs while slimming
   the API. Existing numerical golden and dispatch checks remain requirements.
5. **Separate diagnostics from compatibility.** Qualified training identities,
   config preconditions, schema layout and informational analyzer versions have
   different jobs. Stale prose currently makes them look like competing gates.

## Verification and limits

Reproduce the observations with `just api-necessity-probe` and
`just api-necessity-transforms`. These examples print observed behavior; they
are investigation tools, not regression tests blessing the defects.

The runtime/baker/API sources were copied unchanged into an isolated three-member
workspace for compilation, with the root package and unrelated members omitted.
The source lockfile was copied; the initial standalone build was allowed to
reconcile workspace resolution, and subsequent runs used `--locked`. No sibling
checkout was edited. Existing zenpredict library tests pass with default features
and without defaults; zenanalyze-api library, README coverage and doctests pass.
Both new examples pass strict Clippy. Passing existing tests does not cover the
defects demonstrated above.

Supplementary source evidence and full command logs are preserved through
[the audit evidence pointer](../benchmarks/api_necessity_2026-10-28.pointer.md).

Resource-wrapper evidence: the combined API probe completed with `rc=0`,
`peak-RSS 0.30GiB`; no_std/std output probes completed with `rc=0`,
`peak-RSS 0.24GiB`/`0.18GiB`. These are wrapper observations, not performance
claims. No new RD, model qualification, analyzer golden run, full release matrix
or all-target release gate was performed by this audit.

**Order of work:** repair output-transform correctness; decide the retained
controller surface; review the strided/fallible producer signatures; reconcile
reuse documentation and caller preconditions; then regenerate API snapshots,
run semver checks and the release gates in [PUBLISHING.md](PUBLISHING.md).
Publication remains subject to README approval and the established release gates.
