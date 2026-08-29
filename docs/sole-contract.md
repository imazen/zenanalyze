# `zenanalyze-api` is the sole contract and intermediary

**Owner directive, 2026-08-28:** *"Zenanalyze-api should be the sole contract and
intermediary so different zenanalyze versions can compile together."*

This is the rule of record for every crate in the zen tree that touches image features.
The canonical, always-current statement of the mechanics lives in
[`zenanalyze-api/README.md`](../zenanalyze-api/README.md) (which *is* that crate's rustdoc,
with compiled doctests); this page is the repo-facing version — the rule, why it exists, and
how to tell which side of the line a given crate is on.

## The rule

> A codec crate's **library code** depends on `zenanalyze-api` and nothing else from the
> zenanalyze family. It receives values as a `zenanalyze_api::Offer`, or extracts them
> through a `&dyn zenanalyze_api::FeatureProvider` the host injects. It never names
> `zenanalyze::…`.

A direct `zenanalyze` dependency is legitimate in exactly two roles:

| Role | Example | Why it's fine |
|---|---|---|
| **Host / orchestrator** | the application or pipeline that runs one analysis pass and hands out the `Offer`; anything implementing `FeatureProvider` | It is the layer whose job is to *choose* the version. There is one of it. |
| **Dev tooling** | `dev/` binaries, `examples/`, `benches/`, training and sweep extractors | Not linked into the product graph, so its pin can't collide with another crate's. |

Everything else — the encoder library, the picker, the classifier — speaks the contract.

## Why: two codecs, two versions, one binary

A product links many `zenanalyze` versions at once. Each codec pins the version its model was
trained against, because a model's coefficients were fit against specific feature *definitions*
and re-defining a feature silently changes what the numbers mean. Cargo happily links
`zenanalyze 0.2` and `zenanalyze 0.3` side by side — but their types are then distinct, so
nothing typed in terms of `zenanalyze::feature::AnalysisResults` can cross between the two
codecs. `zenanalyze-api` is deliberately the one crate everything agrees on: a single version,
zero dependencies, transport types only.

### The failure this prevents, concretely

`zenwebp`'s `analyzer` feature named `zenanalyze::feature::AnalysisFeature` variants directly
while pinning published `zenanalyze 0.1.0`. When `IndexedPaletteWidth` was replaced by
`PaletteLog2Size` upstream, `cargo build --features analyzer` stopped compiling — the crate
simply could not be built with its classifier enabled. Its manifest carried a comment saying
so and telling local developers to patch in a sibling checkout by hand.

Nothing about that classifier needed a `zenanalyze` version: it reads ten feature values and
thresholds them. Under the contract it asks for them by bare name
(`Select::Names`) and takes an `Offer` or a `&dyn FeatureProvider` — which is exactly what
the rename would then have been free to do.

## The four compatibility rules

Restated from `zenanalyze-api/README.md#compatibility-rules`, which has the detail:

1. **One dependency source.** `zenanalyze-api = "0.1.0"` — a crates.io *version*. Never
   `{ git = …, rev = … }` in a consumer manifest: a registry dep and a git dep are different
   Cargo sources, and two git deps at *different revs* are different sources too. Either way
   you get two `Offer` types that don't interconvert, and the error ("expected `Offer`, found
   `Offer`") surfaces far from the cause. Unreleased changes go in **one**
   `[patch.crates-io]` at the workspace root, which rewrites the registry entry everywhere
   and keeps unification.
2. **Sole contract** — the rule above.
3. **No zenanalyze types cross the boundary.** The contract carries transport only, and takes
   no dependencies: anything it pulled in would become another axis that can force a split.
4. **Identity-versioned layout.** A column is identified by its qualified `name@hex8`, and a
   value vector is built in request order by `reuse_for` — so there is no global layout to
   keep in sync. `Select::Features` pins the code version (what a model must use);
   `Select::Names` matches any version (for threshold heuristics, diagnostics, export);
   `schema_hash` gates blending *stored* offers.

## Which `Select` do I want?

| Consumer | Variant | Because |
|---|---|---|
| A compiled model / picker | `Select::Features` | Coefficients were fit against one code version per column. A drift **must** miss, or the prediction is silently corrupted. |
| A threshold heuristic, content classifier, diagnostic, or bulk export | `Select::Names` | A drift doesn't invalidate it, and pinning would force it to hard-code hashes it can't know across builds — the pressure that pushes a crate back onto a direct `zenanalyze` dep. |
| An extractor that wants everything this build has | `Select::All` | Resolved provider-side against its `Catalog`. |

## Producer side, in this repo

Both live in [`src/offer.rs`](../src/offer.rs), behind the `api` cargo feature:

- `extract_offer(rgb, w, h, &query, descriptor_hash) -> OwnedOffer` — one pass, bundled as a
  self-describing offer with each feature's code version folded into its qualified name.
- `Analyzer` — this build as a `zenanalyze_api::FeatureProvider`. This is the intermediary a
  host injects so a codec can run its own pass without naming a `zenanalyze` type.

`versioning::feature_qualified_names()` is the build's vocabulary (and what `Analyzer::catalog`
publishes); `benchmarks/feature_qualified_names.tsv` is the committed copy a golden tripwire
keeps in sync, and is what off-Rust tooling should read rather than re-deriving the hash.

## Cross-repo audit, 2026-08-28

State of every crate in `~/work` and `~/work/zen` that named a zenanalyze-family
dependency, at the time the directive landed. "Library" means production code linked
into the product graph; dev tooling and host wiring are called out separately.

| Repo | Before | After |
|---|---|---|
| **zenpipe** / `zencodecs` | Already correct — `zenanalyze-api = "0.1.0"` in `zencodecs`, one root `[patch.crates-io]` to the git repo. Its manifest comment records the E0308 that taught it. | Unchanged. **The exemplar** — copy this shape. |
| **zensquoosh** | Correct by shape: the contract appears only inside the root `[patch.crates-io]`, so one source. Rev-pinned by that table's deliberate reproducibility policy. | Unchanged. Re-syncs when zenpipe re-pins, per its own note. |
| **zenwebp** | Library named `zenanalyze::feature::AnalysisFeature` against a registry `0.1.0` pin. `--features analyzer` **had not compiled** since `IndexedPaletteWidth` → `PaletteLog2Size`; CI never enabled it. | Classifier reads by bare name through the contract (`Select::Names` — thresholds, not coefficients). `analyzer` = contract only, builds anywhere and is CI-covered; `analyzer-bundled` adds `zenanalyze::Analyzer`. |
| **zenavif** | `zenanalyze-api` git-rev `47b4d0f5` — old enough that the crate compiled against a **superseded contract API** and could not have built beside a correctly-pinned consumer. | Registry version + one root patch. Five reuse sites version-pinned per feature (`Select::Features`) — all feed fitted coefficients or thresholds. |
| **zenjpeg** | Two *different* git revs of the same repo (`zenanalyze` 13d40c3, `zenanalyze-api` 47b4d0f5); also on the superseded contract API. | Registry versions + one root patch. `pick_config_from_offer` gates on the model's stamps, then per feature on its code version. |
| **zensr** | Library `chooser` called `analyze_features_rgb8` directly against git-rev `a7d8224`. | Contract-only `chooser`; `chooser-bundled` supplies the provider. Known gap recorded in-source: the fit didn't record its training-time feature versions, so reuse is by bare name until the next re-fit stamps them. |
| **jxl-encoder** | Library `s4_eps.rs` (feature `learned-admission`, **default-on**) names `zenanalyze::feature::*`. | **Not migrated** — another agent held uncommitted work in its working copy. Highest-value remaining item: a default-on feature puts a concrete zenanalyze in every jxl-encoder graph. Tracked: [imazen/jxl-encoder#98](https://github.com/imazen/jxl-encoder/issues/98). |
| **zenjxl** | `extract_features_multiaxis` dev tool pins zenanalyze by an absolute path (`/home/lilith/...`) that doesn't resolve on every machine. | **Not migrated** — another agent had paused mid-task with uncommitted work. Dev tooling, so the direct dep is fine; only the absolute path needs fixing. Tracked: [imazen/zenjxl#19](https://github.com/imazen/zenjxl/issues/19). |
| **zenmetrics** (`zenfleet-vastai`), **zensim** (`zensim-picker-prep`) | Direct `zenanalyze` path deps in worker/extractor binaries. | Unchanged, and correct: these are the producer and dev-tooling roles the rule allows. |

Two findings worth keeping:

1. **"Pin every codec to the same rev" is the trap, not the fix.** Three repos carried a
   rev pin under a comment explaining that identical revs keep the contract type unified.
   Cargo unifies by *source*, so each rev is its own source, and the pins had already
   drifted apart (`47b4d0f5`, `7b84d53c`, floating). A rev pin also cannot rewrite
   `zenanalyze`'s **internal** `{ version, path }` dep on the contract; a root patch can.
2. **A rev pin silently freezes the API, not just the version.** zenavif and zenjpeg were
   both compiling against a `Request::new(names, analyzer_version, defs_version,
   config_hash)` that no longer exists. Nothing failed, because nothing ever built them
   next to a current consumer — which is the whole failure mode, arriving quietly.

## Auditing a consumer

```bash
# Library code that names zenanalyze directly — each hit is a violation unless the file
# is a dev tool, example, bench, or the host's own provider wiring.
grep -rn --include='*.rs' 'zenanalyze::' src/

# Manifest form: a git rev pin on the contract splits unification.
grep -rn --include=Cargo.toml 'zenanalyze-api' .
```
