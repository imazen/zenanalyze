# The feature contract: request → union → one pass → offer → reuse-or-own

**Status:** 2026-06-19. `zenanalyze-api` (the contract crate) and the full
**Rust** producer surface — `zenanalyze`, `zenpredict` (model accessors), and
`zenpredict-bake` (writes the stamps) — have landed additively. The only
producer remainder is zentrain *passing* the two values into the bake request;
codec migrations are follow-ups in their repos.

## The requirement

A product (e.g. imageflow) links **a dozen codecs, each pinning a different
`zenanalyze`/`zenpredict` version** — their models were trained against
different feature *definitions*, so they must be fed features by the *exact*
version they trained on. So `zenanalyze@0.1…2.0` all coexist in one binary
(verified safe: zenanalyze is pure fns + `const` data — no `#[no_mangle]`,
`links`, `static mut`, or `extern "C"`, so Cargo links them side by side).

This makes definition drift impossible by construction — but it means **no
`zenanalyze` type can cross a crate boundary** (`zenanalyze0_2::Foo` ≠
`zenanalyze1_0::Foo`). The contract must be expressible in types that *unify*.

## Three layers, one rule

| layer | crate | versioning | holds |
|---|---|---|---|
| **contract** | `zenanalyze-api` | frozen, **unifies** (one version everywhere) | `Request`, `Offer`, negotiation, `union_names` — pure transport |
| **implementation** | `zenanalyze@X` | multi-version | feature defs + extraction + version-local accessors |
| **consumers** | `zenpicker` / codec pickers | version-agnostic | build `Request`s, negotiate `Offer`s, own-pass via their `zenanalyze@X` |

**The rule that keeps it stable:** *does it name a feature → impl; is it pure
transport → contract.* `zenanalyze-api` has zero deps and zero feature
knowledge, which is exactly why it can freeze and unify — feature math churns
every release; `name → value + a version stamp + gather-by-name` does not.

## The flow

```text
1. each codec builds a Request from its baked model
       Request{ names: model.feature_columns(), analyzer_version, defs_version }   …× a dozen
2. the caller unions the requested names, picks the best zenanalyze it has,
3. and runs ONE zenanalyze@Y pass over the union        ──▶  Offer{ name→value, "1.0", defs=1 }
4. each codec negotiates:  offer.reuse_for(my_request)
       Some(vec) → reuse (no second extraction)
       None      → own zenanalyze@X pass → &[f32]
   → predict → params
```

## Data structures

**`zenanalyze-api`** (frozen; `no_std + alloc`, zero deps):
```rust
pub struct Request<'a> { pub names: &'a [&'a str],       // model names (borrowed from model bytes)
                         pub analyzer_version: &'a str, pub defs_version: u32 }
pub struct Offer<'a>   { pub names: &'a [&'a str],       // analyzer names ('static via variance)
                         pub values: &'a [f32],
                         pub analyzer_version: &'a str, pub defs_version: u32 }
impl Offer { fn matches(major_minor,defs)->bool;  fn get(name)->Option<f32>;
             fn gather(names)->Option<Vec<f32>>;  fn reuse_for(&Request)->Option<Vec<f32>> }
pub fn union_names(reqs)->Vec<&str>;
```
`matches` gates on `major.minor + defs_version`; `gather`/`reuse_for` return
`None` on any missing name (never a silent zero). Note the lifetime asymmetry:
**offer names are `'static`** (from `zenanalyze::feature_name`), **request names
are borrowed** (parsed from model bytes) — `&'a [&'a str]` accepts both by
variance, no `Feature`/registry type needed (no consumer introspects).

**The model** (`zenpredict`, ZNPR metadata) self-describes:
```text
feature_columns      → Model::feature_columns() -> impl Iterator<&str>   (names + order)
analyzer_version     → Model::analyzer_version() -> Option<&str>          ("0.2.7")
feature_defs_version → Model::feature_defs_version() -> Option<u32>
schema_hash          → Model::schema_hash() -> u64                        (names/order, existing)
```

**`zenanalyze@X`** provides the version-local primitives that populate an offer
and run an own-pass: `feature_name`, `feature_id_by_name`, `resolve_feature_ids`,
`feature_vector`, `analyzer_version`, `feature_defs_version`.

## Version guarantees, layered
1. **Cross-major** (0.2 vs 1.0): the **Cargo pin** — a 0.2 model's codec links
   only 0.2, so it can't be fed 1.0 features.
2. **Within-major numeric drift** (0.2.3 vs 0.2.7): `feature_defs_version`, baked
   vs runtime (`Offer::matches`).
3. **Name/order drift**: the existing `schema_hash`.

## What's done vs. pending
- ✅ `zenanalyze-api` crate (Request/Offer/negotiation/union, frozen, tested).
- ✅ `zenanalyze`: the version-local primitives (`from_name`, `resolve_feature_ids`,
  `feature_vector`, `analyzer_version`, `feature_defs_version`).
- ✅ `zenpredict`: `keys::ANALYZER_VERSION` + `keys::FEATURE_DEFS_VERSION` +
  `Model::feature_columns()/analyzer_version()/feature_defs_version()` (the u32
  decoded LE-explicit so it round-trips on i686/any-endian).
- ✅ `zenpredict-bake`: first-class `analyzer_version` + `feature_defs_version`
  fields on `BakeRequestJson` that write the two reuse-key metadata entries —
  **the Rust baker owns the byte encoding** (UTF-8 / LE-`u32`), so Python never
  hand-rolls LE hex. An explicit `metadata` entry for the same key still wins
  (dup-guarded). The baker (not Python) is and stays the byte-writer.
- ⏳ **zentrain** (`bake_picker.py`): *pass the two values through* to the bake
  request — `analyzer_version` + `feature_defs_version` captured at extraction
  time. Python supplies the values; the Rust baker writes the bytes (this is the
  existing inversion, not a Python baker).
- ⏳ **codec migrations** (their repos): build a `Request`, negotiate `Offer` (or
  own-pass), drop the `pub use zenanalyze::feature::*`.
- ⏳ **caller** (orchestrator/product): the union + best-version-pick + single
  pass that produces the shared `Offer`.

## Why earlier drafts were wrong (so they don't recur)
- A `FeatureSchema` *adapter* in zenanalyze — a version-specific type can't be a
  cross-version contract; dropped for free functions + the api crate.
- Putting the `Offer` type *in zenanalyze* — multi-version, wouldn't unify;
  moved to the frozen `zenanalyze-api`.
- A `Feature`/registry type — no consumer introspects (all name-based), so it
  violates minimal and risks the freeze; omitted.
