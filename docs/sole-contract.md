# `zenanalyze-api` is the preferred contract and interchange boundary

> **Filename note:** this page is still `sole-contract.md` so existing links keep
> working. The policy it describes is no longer "sole" — see the correction below.

**Owner directive, 2026-08-28:** *"Zenanalyze-api should be the sole contract and
intermediary so different zenanalyze versions can compile together."*

**Owner correction, same day, verbatim:** *"a direct dep is okay though, a reanalysis
might be needed anyway if the upstream provided features are insufficient."*

The first version of this page read the directive as a prohibition — library code may name
`zenanalyze-api` and nothing else. That was too strict, and the correction says why: a codec
handed an offer that doesn't cover what it needs has to re-analyse, and forcing it through a
provider it cannot rely on buys nothing. What actually matters is the **interchange boundary**,
not the dependency list.

The canonical statement of the mechanics lives in
[`zenanalyze-api/README.md`](../zenanalyze-api/README.md) (which *is* that crate's rustdoc,
with compiled doctests); this page is the repo-facing version.

## The policy

**Preferred — the interchange boundary.** Negotiation types (`Request` / `Offer` / `Catalog` /
`Select`) and the `FeatureProvider` injection point flow through `zenanalyze-api`, so a host
and a codec built against *different* `zenanalyze` versions can still talk. Reach for the
shared `Offer` first: it is free, the host already paid for the pass.

**Permitted — a direct `zenanalyze` dependency in a codec.** Specifically for **re-analysis
when the host-provided features are insufficient**: a feature the offer doesn't carry, the
wrong tier, a stale or version-drifted offer, or no offer at all. A codec is entitled to run
its own pass with its own `zenanalyze`, and that is a normal thing to do rather than a
failure to migrate.

The design that follows from both together — and what the migrated codecs now do — is:

1. try the shared `Offer`;
2. else a `&dyn FeatureProvider` if the host injected one;
3. else its own `zenanalyze` pass;
4. and whatever the source, hand results across crate boundaries as `zenanalyze-api` types.

Step 3 is the one this correction restores. Steps 1, 2 and 4 are what keep a mixed-version
graph linkable.

## The rules that are hard

These are not preferences. Each one caused a real failure in this tree.

1. **Depend by crates.io registry version — never a git-rev pin.**
   `zenanalyze-api = "0.1.0"`, `zenanalyze = "0.2.0"`. Cargo unifies by *source*, so a rev pin
   is its own source: two consumers on different revs get two `Offer` types that don't
   interconvert, and the error ("expected `Offer`, found `Offer`") surfaces far from the cause.
   Worse, and this is what actually bit, **a rev pin silently freezes the API too** — zenavif
   and zenjpeg were both compiling against a `Request::new` shape that no longer exists, and
   nothing failed because nothing ever built them next to a current consumer. Unreleased
   changes go in **one** `[patch.crates-io]` at the workspace root, which rewrites every edge
   at once, including `zenanalyze`'s own internal `{ version, path }` dep on the contract that
   a rev pin cannot reach.
2. **No absolute-path pins.** `path = "/home/lilith/work/zen/zenanalyze"` resolves on exactly
   one machine. Relative sibling paths are tolerable where a repo already relies on a sibling
   checkout; absolute ones are never right.
3. **Interchange types at crate boundaries come from `zenanalyze-api`, not `zenanalyze`.**
   A public signature naming `zenanalyze::feature::AnalysisResults` pins every caller to your
   `zenanalyze` version. Take an `Offer`, return `Offer`/`OwnedOffer`, accept
   `&dyn FeatureProvider` — then a host on a different version can still call you, however you
   sourced the numbers internally.

Rule 3 is the one that makes rule "permitted" safe: a codec can depend on `zenanalyze`
directly *because* its boundary doesn't leak it.

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

Two separate faults, worth untangling now that the policy distinguishes them. The **fatal**
one was rule 1: a stale registry pin the crate could not build against. The **avoidable** one
was rule 3: the classifier's signature was written in terms of upstream enum variants, so a
rename reached all the way into it. It reads ten values and thresholds them — asking by bare
name through `Select::Names` costs nothing and leaves the rename free to happen.

Note what was *not* wrong: depending on `zenanalyze` at all. Had the pin been a current
registry version, the direct dependency would have been fine.

## Compatibility details

`zenanalyze-api/README.md#compatibility-rules` has the full statement; beyond the three hard
rules above, one property is worth restating because it is easy to assume away:

**The feature-vector layout is versioned by identity, not by position.** A column is identified
by its qualified `name@hex8`, and a value vector is built in *request order* by `reuse_for`, so
there is no global layout to keep in sync. `Select::Features` pins the code version (what a
model must use); `Select::Names` matches any version (threshold heuristics, diagnostics,
export); `schema_hash` gates blending *stored* offers.

## Which `Select` do I want?

| Consumer | Variant | Because |
|---|---|---|
| A compiled model / picker | `Select::Features` | Coefficients were fit against one code version per column. A drift **must** miss, or the prediction is silently corrupted. |
| A threshold heuristic, content classifier, diagnostic, or bulk export | `Select::Names` | A drift doesn't invalidate it, and pinning would force it to hard-code hashes it can't know across builds. |
| An extractor that wants everything this build has | `Select::All` | Resolved provider-side against its `Catalog`. |

## Producer side, in this repo

Both live in [`src/offer.rs`](../src/offer.rs), behind the `api` cargo feature:

- `extract_offer(rgb, w, h, &query, descriptor_hash) -> OwnedOffer` — one pass, bundled as a
  self-describing offer with each feature's code version folded into its qualified name.
- `Analyzer` — this build as a `zenanalyze_api::FeatureProvider`. This is what a host injects
  so a codec *can* run a pass without naming a `zenanalyze` type. It is an option offered to
  codecs, not a hoop they must jump through: a codec with its own `zenanalyze` is free to call
  `extract_offer` (or `analyze_features_rgb8`) directly.

`versioning::feature_qualified_names()` is the build's vocabulary (and what `Analyzer::catalog`
publishes); `benchmarks/feature_qualified_names.tsv` is the committed copy a golden tripwire
keeps in sync, and is what off-Rust tooling should read rather than re-deriving the hash.

## Cross-repo audit, 2026-08-28

State of every crate in `~/work` and `~/work/zen` that named a zenanalyze-family
dependency. Re-scored against the corrected policy: a direct `zenanalyze` dep is **not** a
finding here — only a rev pin, an absolute-path pin, or a leaked type at a crate boundary is.

| Repo | Before | After |
|---|---|---|
| **zenpipe** / `zencodecs` | Already correct — `zenanalyze-api = "0.1.0"` in `zencodecs`, one root `[patch.crates-io]` to the git repo. Its manifest comment records the E0308 that taught it. | Unchanged. **The exemplar** — copy this shape. |
| **zensquoosh** | Correct by shape: the contract appears only inside the root `[patch.crates-io]`, so one source. Rev-pinned by that table's deliberate reproducibility policy. | Unchanged. Re-syncs when zenpipe re-pins, per its own note. |
| **zenwebp** | Stale registry `0.1.0` pin (rule 1), and the classifier's body written in terms of upstream enum variants (rule 3). `--features analyzer` **had not compiled** since `IndexedPaletteWidth` → `PaletteLog2Size`; CI never enabled it. | Classifier reads by bare name through the contract (`Select::Names` — thresholds, not coefficients). `analyzer` needs no `zenanalyze`; `analyzer-bundled` adds `zenanalyze::Analyzer` as a default provider. Both CI-covered. |
| **zenavif** | `zenanalyze-api` git-rev `47b4d0f5` — old enough that the crate compiled against a **superseded contract API** and could not have built beside a correctly-pinned consumer. | Registry version + one root patch. Five reuse sites version-pinned per feature (`Select::Features`) — all feed fitted coefficients or thresholds. |
| **zenjpeg** | Two *different* git revs of the same repo (`zenanalyze` 13d40c3, `zenanalyze-api` 47b4d0f5); also on the superseded contract API. | Registry versions + one root patch. `pick_config_from_offer` gates on the model's stamps, then per feature on its code version. |
| **zensr** | Git-rev pin `a7d8224` (rule 1); the `chooser` also had no way to accept a shared offer. | Registry version + root patch. `chooser` can take an `Offer` or a provider; `chooser-bundled` supplies `zenanalyze::Analyzer`. Known gap recorded in-source: the fit didn't record its training-time feature versions, so reuse is by bare name until the next re-fit stamps them. |
| **jxl-encoder** | `s4_eps.rs` (feature `learned-admission`, default-on) runs its own `zenanalyze` pass. | **Compliant as-is under the corrected policy** — that is exactly the permitted re-analysis case, the dep is already a registry version, and the extraction is `pub(crate)` so nothing leaks. Re-scored and closed: [imazen/jxl-encoder#98](https://github.com/imazen/jxl-encoder/issues/98). |
| **zenjxl** | `extract_features_multiaxis` pins zenanalyze by an **absolute** path (`/home/lilith/...`), so it builds on one machine only. | **Still a rule-2 violation** and still open. The direct dep is fine; only the absolute path is wrong. Not fixed here — another agent had paused mid-task with uncommitted work. Tracked: [imazen/zenjxl#19](https://github.com/imazen/zenjxl/issues/19). |
| **zenmetrics** (`zenfleet-vastai`), **zensim** (`zensim-picker-prep`) | Direct `zenanalyze` path deps in worker/extractor binaries. | Unchanged and fine — extractors that own their analyzer version, with no crate boundary to leak across. |

Findings worth keeping:

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

Check the three hard rules. A direct `zenanalyze` dep is **not** on this list.

```bash
# Rule 1 — a git-rev pin on either crate. Any hit is a finding.
grep -rn --include=Cargo.toml -E 'zenanalyze(-api)?.*(git|rev)' .

# Rule 2 — an absolute path pin.
grep -rn --include=Cargo.toml -E 'zenanalyze.*path *= *"/' .

# Rule 3 — a zenanalyze type in a PUBLIC signature (pub fn / pub struct field / pub trait).
# `pub(crate)` and function bodies are fine; only crate-boundary leaks pin your callers.
grep -rn --include='*.rs' -E '^\s*pub (fn|struct|enum|trait|type).*zenanalyze::' src/
```

If all three are clean, the crate is compliant however it gets its feature values.
