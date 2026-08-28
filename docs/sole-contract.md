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

## Auditing a consumer

```bash
# Library code that names zenanalyze directly — each hit is a violation unless the file
# is a dev tool, example, bench, or the host's own provider wiring.
grep -rn --include='*.rs' 'zenanalyze::' src/

# Manifest form: a git rev pin on the contract splits unification.
grep -rn --include=Cargo.toml 'zenanalyze-api' .
```
