# The feature contract — frozen for a decade

**Status:** 2026-06-19. `zenanalyze-api` is **1.0** and frozen. The full Rust
producer surface (contract, `zenanalyze` config digest, `zenpredict` model
accessors, `zenpredict-bake` stamps) and the Python forwarder have landed. The
remaining work is *consumer-side* (codecs build `Request`s, the orchestrator
groups + runs the union pass) and *upstream* (the trainer recording extraction
provenance) — in their own repos.

## The requirement

A product (e.g. imageflow) links **a dozen codecs, each pinning a different
`zenanalyze`/`zenpredict` version** — their models were trained against different
feature *definitions* and must be fed by the *exact* version they trained on. So
`zenanalyze@0.1…2.0` all coexist in one binary (verified safe: pure fns + `const`
data — no `#[no_mangle]`, `links`, `static mut`, `extern "C"`). This makes
definition drift impossible by construction — but it means **no `zenanalyze` type
can cross a crate boundary** (`zenanalyze0_2::Foo` ≠ `zenanalyze1_0::Foo`). The
contract must be expressed in types that *unify*.

## Three layers, one rule

| layer | crate | versioning | holds |
|---|---|---|---|
| **contract** | `zenanalyze-api` | **1.0, frozen, unifies** | `Request`, `Offer`, negotiation, `union_names` — pure transport |
| **implementation** | `zenanalyze@X` | multi-version | feature defs + extraction + `config_hash()` + version-local accessors |
| **consumers** | `zenpicker` / codec pickers | version-agnostic | build `Request`s, negotiate `Offer`s, own-pass via their `zenanalyze@X` |

**The rule:** *names a feature → impl; pure transport → contract.* `zenanalyze-api`
has zero deps and zero feature knowledge, which is why it can freeze and unify —
feature math churns every release; `name → value + a reuse key + gather-by-name`
does not.

## Why `1.0` and `#[non_exhaustive]` are the two un-fixable-later decisions

Everything else in this design is additive-safe — *if and only if* these two are
right, and they cannot be changed after the crate ships:

- **`1.0`, never `2.0`.** Cargo unifies only semver-compatible versions: all `1.*`
  unify; a `2.0` splits the linked ecosystem into two incompatible contract
  types — the exact failure this crate exists to prevent. A `0.x` line is worse
  (`0.1`/`0.2` don't even unify, so the *first* additive release would split it).
  So: ship at `1.0`, evolve additively, never bump major.
- **`#[non_exhaustive]` + `new()` constructors.** This is what lets `Request`/`Offer`
  grow fields within `1.x` without breaking. With it, every conceivable
  extension below is a *non-breaking addition the day a consumer needs it*; the
  frozen core stays minimal now and open forever.

## The reuse key — `(analyzer major.minor, defs_version, config_hash)`

A feature is named the same across versions, but its *value* can change three
independent ways without the name changing. Reuse is gated on all three:

1. **`analyzer_version`** (`major.minor`) — different math in a different release.
   Patch ignored.
2. **`defs_version`** — a within-`major.minor` numeric-definition bump (a
   `zenanalyze` compile-time const).
3. **`config_hash`** — the value-affecting **runtime analysis config**.

### Why `config_hash` exists (a proven hole, closed)

Two adversarial read-only investigations (consumer-needs survey + config-vs-identity
trace) converged on this: `zenanalyze`'s `AnalysisQuery` has a caller-settable
flag, `with_linear_light(true)`, that makes the **same feature name** (`variance`)
produce a **different value**, while the name *and* `feature_defs_version` (const
`1`) stay identical. Test: `src/linear_tier.rs::linear_light_flag_changes_variance_end_to_end`.
Without a config component, a codec could reuse linear-light `variance` against a
gamma-trained model and feed it silently wrong inputs — a correctness bug, the
kind the user's pixels are sacred against.

`config_hash` is an **opaque `u64` digest** (`AnalysisQuery::config_hash()`;
`0` = canonical gamma default). It hashes only the **caller-settable value-mode
flags** — not the feature *set* ("which features," handled by offer names) and
not the crate-internal sampling budgets (invariants folded into `defs_version`).
A flag mixes in only on *deviation from its default*, so the default stays `0` as
flags land, and new config axes fold into the hash **without ever touching the
frozen crate**.

### Why source-dependent axes are deliberately NOT in the key

The investigations also surfaced primaries / transfer-function / diffuse-white /
pixel-layout / Native-vs-Convert as value-affecting. They are **not** reuse
hazards and are correctly excluded: they're pinned by *which image* you analyze,
and an `Offer` is only ever reused for the **same image** (the orchestrator
analyzes image X once, every codec encoding X reuses it). Cargo-feature axes
(`hdr`/`experimental`) are likewise non-hazards: Cargo unifies features per
crate-version within a binary, so all consumers of one analyzer version share one
feature set, and anything absent falls through `gather → None` to an own-pass.

## The flow

```text
1. each codec builds a Request from its baked model
       Request::new(model.feature_columns(), analyzer_version, defs_version, config_hash)  …× a dozen
2. the caller GROUPS requests by (defs_version, config_hash), unions the names in
   each group, picks the best zenanalyze it has, and runs ONE pass per group
                                                       ──▶  Offer{ name→value, "1.0", defs, config }
3. each codec negotiates:  offer.reuse_for(my_request)
       Some(vec) → reuse (no second extraction)
       None      → own zenanalyze@X pass → &[f32]
   → predict → params
```

Grouping by config is the caller's job; mixing configs into one pass is *safe*
(mismatches fall through `reuse_for` to `None` + own-pass), only a missed-reuse
perf cost.

## Data structures

**`zenanalyze-api`** (frozen `1.0`; `no_std + alloc`, zero deps,
`#[non_exhaustive]` structs + `new()`):
```rust
pub struct Request<'a> { names: &'a [&'a str], analyzer_version: &'a str,
                         defs_version: u32, config_hash: u64 }   // + Request::new(..)
pub struct Offer<'a>   { names: &'a [&'a str], values: &'a [f32],
                         analyzer_version: &'a str, defs_version: u32, config_hash: u64 } // + Offer::new(..)
impl Offer { fn matches(major_minor, defs, config)->bool;  fn get(name)->Option<f32>;
             fn gather(names)->Option<Vec<f32>>;  fn reuse_for(&Request)->Option<Vec<f32>> }
pub fn union_names(reqs)->Vec<&str>;   // call per (defs, config) group
```
`matches` gates on `major.minor + defs_version + config_hash`; `gather`/`reuse_for`
return `None` on any missing name (never a silent zero). Lifetime asymmetry: offer
names are `'static` (from `zenanalyze::feature_name`), request names are borrowed
(from model bytes) — `&'a [&'a str]` accepts both by variance, no `Feature`/registry
type needed (no consumer introspects).

**The model** (`zenpredict`, ZNPR metadata) self-describes the whole key:
```text
feature_columns      → Model::feature_columns()      (names + order)
analyzer_version     → Model::analyzer_version()      -> Option<&str>   ("0.2.7")
feature_defs_version → Model::feature_defs_version()  -> Option<u32>    (LE-decoded)
feature_config_hash  → Model::feature_config_hash()   -> Option<u64>    (LE-decoded, None⇒0)
schema_hash          → Model::schema_hash()           -> u64            (names/order, existing)
```

## What's done vs. pending

- ✅ `zenanalyze-api` **1.0** — frozen shape: `config_hash` reuse key,
  `#[non_exhaustive]` + constructors, soundness docs, tests (config gating +
  `major_minor` pre-release/degenerate edges).
- ✅ `zenanalyze` — **actually produces an `Offer`**: the opt-in `api` feature +
  `OwnedOffer::extract(rgb, w, h, &query)` → `as_offer()`, plus
  `AnalysisQuery::config_hash()` and the version-local primitives. An end-to-end
  test (extract → offer → `reuse_for`) validates the frozen borrowed-`Offer` shape
  under real use, and confirms the producer's owned holder belongs in the impl
  crate — *not* the frozen contract.
- ✅ `zenpredict` — `keys::{ANALYZER_VERSION, FEATURE_DEFS_VERSION, FEATURE_CONFIG_HASH}`
  + the four `Model` accessors (numerics LE-explicit for i686/any-endian).
- ✅ `zenpredict-bake` — first-class `analyzer_version` / `feature_defs_version` /
  `feature_config_hash` fields; **Rust owns the byte encoding** (UTF-8 / LE-u32 /
  LE-u64); explicit metadata entry wins, dup-guarded.
- ✅ `bake_picker.py` — forwards all three values when the trainer recorded them.
- ✅ `zenpicker` — **consumes** the contract: opt-in `api` feature +
  `MetaPicker::feature_request()` builds a `Request` from the model (names
  collected once at construction; stamps from the accessors) for a caller to
  negotiate a shared `Offer` before `pick`. Test validates reuse + drift-rejection.
- ✅ `zentrain` — **outside-in provenance**: `tools/_provenance.py` +
  `train_hybrid.py` / `train_multi_codec.py` emit the three stamps into the model
  JSON from a codec config's optional `ANALYSIS_PROVENANCE` (bake_picker forwards).
  Source-of-truth is *declared*, not auto-guessed (the extractor's exact zenanalyze
  version isn't verifiable from here, and a wrong stamp is a soundness risk);
  undeclared → unstamped → safe own-pass. Template in the reference config.
- ⏳ **per-codec configs** — *declare* `ANALYSIS_PROVENANCE` with the verified
  extractor version so their pickers actually reuse (the mechanism is wired and
  safe-by-default until they do).
- ⏳ **codec migrations** (their repos) — build a `Request`, negotiate the `Offer`,
  drop the `pub use zenanalyze::feature::*`.
- ⏳ **caller/orchestrator** (the product) — group by key, union per group,
  best-version-pick, single pass → shared `Offer`.

## Deliberately NOT added now — and the proof it's safe to add later

These came up in the survey; each is omitted on purpose (YAGNI for a frozen
surface) and is **provably additive-safe** under `1.0 + #[non_exhaustive]`, so
*not* adding them now is correct minimalism, not a deferred risk:

- **Owned/cacheable `Offer` *in the frozen crate*** (0 current consumers; 1
  hypothetical caching server). The producer's owned holder already exists as
  `zenanalyze::OwnedOffer` (impl-crate, lends a borrowed `Offer`) — which
  *validated* that the frozen contract needs no owned variant. A version-agnostic
  cacheable `Offer::to_owned()` in `zenanalyze-api` is still additive-later if a
  consumer ever needs to store an offer across versions.
- **`gather_into(&mut [f32])`** (no-alloc consumers). New method — additive.
- **Non-`f32` values.** Every current feature is exactly representable in `f32`'s
  24-bit mantissa (counts ≤ ~32k, bit-depths ∈ {8..32}, bools → 0/1). If that
  ever stops holding, add a parallel typed channel — additive under
  `#[non_exhaustive]`.
- **Per-feature metadata** (units, ranges). Bounds already live in the ZNPR
  model; if the contract ever needs them, a new method/struct is additive.

## Why earlier drafts were wrong (so they don't recur)

- A `FeatureSchema` *adapter* in zenanalyze — a version-specific type can't be a
  cross-version contract; dropped for the api crate.
- Putting the `Offer` type *in zenanalyze* — multi-version, wouldn't unify.
- A `Feature`/registry type — no consumer introspects (all name-based).
- **Shipping at `0.1.0` with exhaustive structs** — would split the ecosystem on
  the first additive release; fixed to `1.0` + `#[non_exhaustive]`.
- **A 3-tuple reuse key without `config_hash`** — unsound the moment a
  value-affecting `AnalysisQuery` flag (linear-light) is used; fixed.
- **"Python bakes the stamps"** — the Rust `zenpredict-bake` is the byte-writer;
  Python only supplies values.
