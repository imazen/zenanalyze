# zenanalyze-api ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenanalyze-api?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenanalyze-api?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenanalyze-api) ![docs.rs](https://img.shields.io/docsrs/zenanalyze-api?style=flat-square) ![License](https://img.shields.io/crates/l/zenanalyze-api?style=flat-square)

The version-unifying **feature contract** for the zenanalyze picker tree — iterating at
`0.1.x`, freezing at `1.0`.

A product links **many `zenanalyze` versions at once** — a dozen codecs each pin
the version their model was trained against, and `zenanalyze0_2::*` ≠
`zenanalyze1_0::*` are incompatible types. So no `zenanalyze` type can be what
crosses between layers. **This crate is that thing**: depend on it at a single
version and it *unifies* across the whole build.

The flow it serves: an orchestrator collects each codec's `Request`, **unionizes** them,
runs **one** analysis pass, and hands every codec the resulting `Offer`; each codec checks
`offer.satisfies(&its_request)` and reuses, or runs its own pass — the latter through a
`FeatureProvider` the host injects, so *even the fallback* stays version-free.
`no_std + alloc`, **no dependencies**, `forbid(unsafe_code)`.

**What this crate is for:** it is the **interchange boundary**, so a host and a codec built
against different `zenanalyze` versions can still talk. It is not a prohibition on depending
on `zenanalyze` — a codec whose offer doesn't cover what it needs should re-analyse. See
[Compatibility rules](#compatibility-rules); the dependency-source rule is the one that
silently bites.

## Quick start

```rust
use zenanalyze_api::{NamedFeature, FeatureResult, Offer, Provenance, Request, Select, Catalog};

// Each codec hands the orchestrator a Request — its model's columns, as qualified names.
let jpeg_wants = [
    NamedFeature::parse("variance@11111111").unwrap(),
    NamedFeature::parse("edge_density@abcdef01").unwrap(),
];
let jpeg_req = Request::new(Select::Features(&jpeg_wants));

// The orchestrator unionizes the requests against what this build can produce, then runs
// ONE analysis pass for the union.
let available = [
    NamedFeature::parse("variance@11111111").unwrap(),
    NamedFeature::parse("edge_density@abcdef01").unwrap(),
    NamedFeature::parse("uniformity@00000009").unwrap(),
];
let catalog = Catalog::new(&available);
assert_eq!(catalog.union(&[jpeg_req]), vec!["variance", "edge_density"]);

// The pass becomes a self-describing Offer, stamped with one Provenance.
let results = [
    FeatureResult::new(NamedFeature::parse("variance@11111111").unwrap(), 0.5),
    FeatureResult::new(NamedFeature::parse("edge_density@abcdef01").unwrap(), 12.0),
    FeatureResult::new(NamedFeature::parse("uniformity@00000009").unwrap(), 0.9),
];
let offer = Offer::new(&results, Provenance::new("0.2.7"));

// Each codec checks the shared Offer against its own Request, and re-runs if it can't:
assert!(offer.satisfies(&jpeg_req));
assert_eq!(offer.reuse_for(&jpeg_req), Some(vec![0.5, 12.0]));

// A codec whose `variance` is a newer code-version can't reuse it → runs its own pass:
let drift = [NamedFeature::parse("variance@ffffffff").unwrap()];
assert!(!offer.satisfies(&Request::new(Select::Features(&drift))));
```

## The identity is the qualified name — and it's the **code** version

A `NamedFeature` **is** its qualified name `variance@b4a1c2d3` — `name` `@` 8 lowercase hex
of the 32-bit `version_hash`. One `&str`, so equality/hashing/reuse are a single string
compare; `name()` and `version_hash()` are zero-alloc on-demand splits.

The version hash is the feature's **code** version (`NamedFeature::fold_hash` of
`zenanalyze::feature_version_hash`) — build-stable, so the qualified name is too. It does
*not* fold in the runtime analysis config or input framing: those are constant within one
analysis pass, so reuse against that pass's `Offer` is safe by code-version alone, and they
travel on the per-offer `Provenance` instead.

## Negotiation — `satisfies`, `reuse_for`, and the miss rationale

- `offer.satisfies(&req) -> bool` — does the offer cover every wanted column? The
  "do I need to re-run?" decision, without building a value vector.
- `offer.reuse_for(&req) -> Option<Vec<f32>>` — `Some(values)` (canonical f32, in request
  order) when satisfied; `None` ⇒ run own pass.

Both compare **purely by qualified name** — no global config/descriptor gate, because within
a pass those are constant.

When a full re-run isn't possible (a deserialized offer with no original) you can classify
each miss directly from `get()` — by *bare name*, then compare versions:

```rust
# use zenanalyze_api::{NamedFeature, FeatureResult, Offer, Provenance};
# let feats = [FeatureResult::new(NamedFeature::parse("variance@11111111").unwrap(), 0.5)];
# let offer = Offer::new(&feats, Provenance::new("0.2.7"));
let want = NamedFeature::parse("variance@22222222").unwrap();
match offer.get(want.name()) {
    None => { /* missing — must extract */ }
    Some(f) if f.feature().version_hash() == want.version_hash() => { /* reuse f.float() */ }
    Some(_) => { /* present, but a code-version drift */ }
}
```

### Version-pinned (`Features`) vs version-agnostic (`Names`)

`Select::Features` matches the **whole qualified name**, so a code drift in any column is a
miss. That is what a compiled model needs: its coefficients were fit against one code
version of each column, and silently eating a re-defined feature would corrupt the
prediction with no error.

`Select::Names` matches the **bare name at whatever version is on offer**. It exists for the
consumers a drift doesn't invalidate — threshold heuristics and content classifiers,
diagnostics, bulk column export — and, crucially, it lets those consumers name features
*without naming a `zenanalyze` version*. Pinning would force them to hard-code hashes they
can't know across builds, which is exactly the pressure that pushes a crate back onto a
direct `zenanalyze` dependency.

```rust
use zenanalyze_api::{NamedFeature, FeatureResult, Offer, Provenance, Request, Select};

let feats = [FeatureResult::new(NamedFeature::parse("variance@11111111").unwrap(), 0.5)];
let offer = Offer::new(&feats, Provenance::new("0.2.7"));

// Pinned to a version the offer doesn't carry ⇒ miss (a model must re-run).
let pinned = [NamedFeature::parse("variance@ffffffff").unwrap()];
assert!(!offer.satisfies(&Request::new(Select::Features(&pinned))));

// By bare name ⇒ reuses across the drift (a threshold heuristic is fine with that).
let names = ["variance"];
assert_eq!(offer.reuse_for(&Request::new(Select::Names(&names))), Some(vec![0.5]));
```

**Never feed a compiled model from `Names`** — that miss is the safety property you'd be
giving up.

## The intermediary — `FeatureProvider`

A codec with no `Offer` to reuse still has to get values from somewhere. `FeatureProvider`
is one way: extraction expressed as a contract trait, so the **host** picks the `zenanalyze`
version, implements the trait over it, and injects `&dyn FeatureProvider` (`zenanalyze` ships
its own impl behind its `api` feature). A codec written against `&dyn FeatureProvider` can be
driven by a host on any `zenanalyze` version.

It is an *option*, not an obligation. A codec is equally free to depend on `zenanalyze`
directly and run its own pass — which is often the right answer, since a shared offer may
simply not carry what it needs. What matters is that the values cross the **crate boundary**
as the types below, so your callers aren't pinned to your analyzer version.

```rust
use zenanalyze_api::{FeatureProvider, Request, Select};

/// A codec's picker: no `zenanalyze` type anywhere in the signature or the body.
fn pick(offer: Option<&zenanalyze_api::Offer<'_>>,
        provider: Option<&dyn FeatureProvider>,
        rgb: &[u8], w: u32, h: u32) -> Option<Vec<f32>> {
    const WANTED: [&str; 2] = ["variance", "edge_density"];
    let req = Request::new(Select::Names(&WANTED));
    if let Some(values) = offer.and_then(|o| o.reuse_for(&req)) {
        return Some(values);          // the shared pass covered us
    }
    let owned = provider?.extract_rgb8(rgb, w, h, &req).ok()?;  // our own pass, version-free
    owned.reuse_for(&req)
}
```

`catalog()` reports what a provider build can produce as an `OwnedCatalog` — the owned twin
of `Catalog`, for a provider whose qualified names are only known at runtime. Extraction
failures are a `ProviderError`: `BadInput` (buffer length / dimensions), `Unavailable` (this
build can't produce a wanted identity — `OwnedCatalog::unmet` says which), `OutOfMemory`.

An implementor must honor the `Request` exactly. Returning an offer that silently drops a
want is a contract violation: `satisfies` is a *reuse* decision on the consumer side, not a
correctness backstop.

## Native values, canonical f32

A value keeps its native type: `Value` mirrors zenanalyze's outputs — `F32` / `U32` / `U64`
/ `Bool`. `FeatureResult::new` takes any of them (`0.5f32` / `4096u32` / `true`, via
`Into<Value>`). `value() -> Value` keeps the exact type and precision — so a `u32` count past
2²⁴ survives serialization that a bare f32 would round; `float() -> f32` is the canonical
model-input projection (`Value::to_f32`, identical to zenanalyze's `FeatureValue::to_f32`).

```rust
use zenanalyze_api::{NamedFeature, FeatureResult, Value};

let pixels = FeatureResult::new(NamedFeature::parse("pixel_count@00112233").unwrap(), 16_777_217u32);
assert_eq!(pixels.value(), Value::U32(16_777_217)); // exact
assert_eq!(pixels.float(), 16_777_216.0);            // canonical f32 rounds 2^24+1
```

## Compile-time-validated presets

`NamedFeature::parse` is `const`, so a consumer's preset columns are checked **at build
time** — a typo fails the build rather than silently producing a degenerate identity:

```rust
use zenanalyze_api::NamedFeature;

const MODEL_COLUMNS: [NamedFeature<'static>; 2] = [
    NamedFeature::parse("variance@b4a1c2d3").unwrap(),
    NamedFeature::parse("edge_density@abcdef01").unwrap(),
];
assert_eq!(MODEL_COLUMNS[0].name(), "variance");
```

(`TryFrom<&str>` gives the same validation at runtime; a trait method can't be `const`,
which is why `parse` is the one that builds a `const`. `NamedFeature::from_qualified` is the
`const`, unchecked wrap for a string a producer already built with `qualified_for`.)

## Serializing — and the owned twins

Bulk data is columnar **parquet/TSV**, and the contract (being `no_std` + dependency-free)
doesn't write it — it hands a writer the *pieces*: `features()` (qualified name → column,
value → cell) + `provenance()` + `schema_hash` → file metadata. A reader rebuilds the owned
twin from those with `OwnedOffer::new(cells, provenance)` — a `Vec` of `OwnedFeatureResult`
carrying the **same negotiation surface**, so a deserialized offer reuses directly.
`features()` lends the owned cells zero-cost; `OwnedFeatureResult::as_ref()` bridges back to a
borrowed `FeatureResult`.

`to_block` / `OwnedOffer::parse` are the **dependency-free text** form of the same offer — for
debug dumps, tests, and compact self-describing stamps, *not* the bulk path.

Across stored offers (different images, configs) the qualified name alone isn't enough to
know two columns are comparable, so `schema_hash` is the blend gate: a `u64` over the *set*
of qualified names plus the recorded `config_hash` + `descriptor_hash`. Equal ⇒ safe to stack.

```rust
use zenanalyze_api::{NamedFeature, FeatureResult, Offer, OwnedOffer, OwnedFeatureResult,
                     Provenance, Request, Select};

let nf = NamedFeature::parse("variance@b4a1c2d3").unwrap();
let results = [FeatureResult::new(nf, 0.5)];
let offer = Offer::new(&results, Provenance::new("0.2.0").with_descriptor(9));

// Bulk path: a parquet reader rebuilds the owned twin from deserialized cells + metadata.
let owned = OwnedOffer::new(
    vec![OwnedFeatureResult::new("variance@b4a1c2d3", 0.5)],
    Provenance::new("0.2.0").with_descriptor(9),
);
assert!(owned.satisfies(&Request::new(Select::All))); // negotiates directly, no original needed
assert_eq!(owned.schema_hash(), offer.schema_hash());

// Text path: the dependency-free debug/stamp form.
let reparsed = OwnedOffer::parse(&offer.to_block()).unwrap();
assert_eq!(reparsed.get("variance").unwrap().float(), 0.5);
assert_eq!(offer.schema_hash(), 15671083119752945687); // the value in the block below
```

The text block (parquet maps these to columns + metadata):

```text
zenanalyze-features/1
analyzer_version=0.2.0
config_hash=0
descriptor_hash=9
schema_hash=15671083119752945687
[features]
variance@b4a1c2d3=0.5
```

## API reference

Every struct keeps its fields **private** behind accessors. `#[non_exhaustive]` appears on
the **enums** and on `Provenance` (the open-ended metadata bag). There are **no free
functions** — every entry point hangs off the type it concerns.

### `struct NamedFeature<'a>`

A feature's cross-version identity carried as a **single borrowed `&str`** (`Copy`). Accessors:
`qualified_name() -> &str` (`const`, the identity), `name() -> &str`, `version_hash() -> u32`
(zero-alloc splits). Constructors: `NamedFeature::parse(s) -> Option<Self>` (`const`,
validating) and `impl TryFrom<&str>`, or `NamedFeature::from_qualified(s)` (`const`,
unchecked). Associated: `NamedFeature::is_valid_name(name) -> bool` (`const`),
`NamedFeature::fold_hash(full: u64) -> u32` (`const`), `NamedFeature::qualified_for(name,
version_hash) -> String`. `impl Display`.

### `enum Value` `#[non_exhaustive]`

A feature's native value, mirroring zenanalyze's outputs: `F32(f32)`, `U32(u32)`, `U64(u64)`,
`Bool(bool)`. `to_f32() -> f32` (`const`) is the canonical model-input projection; `From`
impls let `0.5f32` / `4096u32` / `1u64` / `true` `.into()` it.

### `struct FeatureResult<'a>`

An identity plus its [`Value`]: `feature() -> NamedFeature`, `value() -> Value` (the native
value), `float() -> f32` (canonical projection), `name() -> &str`. Build with
`FeatureResult::new(feature, impl Into<Value>)`.

### `struct OwnedFeatureResult`

The owned twin of `FeatureResult` (owns its qualified name + `Value`). Same readers
(`qualified_name`/`name`/`version_hash`/`value`/`float`), plus `OwnedFeatureResult::new(qualified_name,
impl Into<Value>)`, `From<FeatureResult>`, and `as_ref() -> FeatureResult` (the owned→borrowed
bridge).

### `struct Provenance<'a>` `#[non_exhaustive]`

**Informational** record (not a reuse gate), the open-ended metadata bag. Mandatory `analyzer_version`; optional via a
builder: `Provenance::new(version)`, `with_config(h)`, `with_descriptor(h)`, read by
`analyzer_version()`/`config_hash()`/`descriptor_hash()`. Config/descriptor feed `schema_hash`.

### `struct Offer<'a>`

`Offer::new(features, provenance)` (`const`), `features() -> &[FeatureResult]`, `provenance()
-> Provenance`, `get(name) -> Option<&FeatureResult>` (by bare name — also classifies a reuse
miss), `satisfies(request) -> bool`, `reuse_for(request) -> Option<Vec<f32>>`, `schema_hash()
-> u64`, `to_block() -> String`.

### `enum Select<'a>` `#[non_exhaustive]` / `struct Request<'a>`

`Select` is `All`, `Features(&[NamedFeature])` (version-pinned — for models), or
`Names(&[&str])` (version-agnostic by bare name — for threshold heuristics, diagnostics,
export). `Request::new(select)`, `select() -> Select`.

### `struct Catalog<'a>`

What a build can produce: `Catalog::new(available)`, `available() -> &[NamedFeature]`,
`offers(want) -> bool`, `has_name(name) -> bool`, `unmet(wants) -> Vec<&str>`, and
`union(requests) -> Vec<&str>` (the "unionize" step, resolving `Select::All`).

### `struct OwnedCatalog`

The owned twin of `Catalog`, for a provider whose vocabulary is built at runtime.
`OwnedCatalog::new(qualified_names)`, `available() -> impl Iterator<Item = NamedFeature>`
(the owned→borrowed bridge), `len()`, `is_empty()`, plus the same `offers` / `has_name` /
`unmet` / `union` queries.

### `trait FeatureProvider`

The extraction intermediary — object-safe, used as `&dyn FeatureProvider`.
`analyzer_version() -> &str`, `catalog() -> OwnedCatalog`, and
`extract_rgb8(rgb, width, height, request) -> Result<OwnedOffer, ProviderError>` over a
tightly-packed 8-bit sRGB buffer (`rgb.len() == width * height * 3`).

### `enum ProviderError` `#[non_exhaustive]`

`BadInput` / `Unavailable` / `OutOfMemory`; implements `core::error::Error`.

### `struct OwnedOffer`

The owned twin of `Offer`. Build from parts with `OwnedOffer::new(features, provenance)` (the
parquet/TSV path) or from text with `OwnedOffer::parse(text) -> Result<_, FormatError>`, then
the SAME surface over owned cells: `provenance()`, `features() -> &[OwnedFeatureResult]`,
`get(name) -> Option<&OwnedFeatureResult>`, `satisfies`, `reuse_for`, `schema_hash`.

### `enum FormatError` `#[non_exhaustive]`

`UnknownFormat` / `MissingHeader` / `BadLine`; implements `core::error::Error`.

## Compatibility rules

These are what make "many `zenanalyze` versions in one build" actually work. All four are
load-bearing; breaking any one of them re-splits the ecosystem.

### 1. One dependency source — a version from crates.io, never a git rev

Cargo unifies two dependencies only when they resolve to the **same source**. A registry
dependency and a git dependency on the same crate are two different packages, and two git
dependencies pinned to *different revs* are also two different packages. Either way you get
two `zenanalyze_api::Offer` types that don't interconvert, and the error surfaces far from
the cause ("expected `Offer`, found `Offer`").

So every consumer declares:

```toml
zenanalyze-api = "0.1.0"     # a crates.io version — the ONLY correct form
```

Never `{ git = "…", rev = "…" }` in a consumer manifest. When you need an unreleased change,
override it in **one** place — a `[patch.crates-io]` at the workspace root — which rewrites
the registry entry everywhere and *keeps* unification:

```toml
[patch.crates-io]
zenanalyze-api = { git = "https://github.com/imazen/zenanalyze" }
```

Drop the patch once the version is published.

### 2. Interchange types at crate boundaries come from this crate

A public signature naming `zenanalyze::feature::AnalysisResults` pins every caller to your
`zenanalyze` version. Take an `Offer`, return `Offer` / `OwnedOffer`, accept a
`&dyn FeatureProvider` — then a host on a different version can still call you, however you
sourced the numbers internally. Function bodies and `pub(crate)` items are unconstrained.

**A direct `zenanalyze` dependency is fine**, including in a codec's library code, and is the
right answer when a host-provided offer is insufficient — a missing feature, the wrong tier, a
version-drifted or absent offer. Don't route around your own analyzer for its own sake; route
your *boundary* through these types.

Prefer the shared `Offer` when it covers you, because the host already paid for that pass.

### 3. This crate itself stays transport-only, and dependency-free

It carries names, values, and a reuse key. It must never re-export or mirror a `zenanalyze`
type, and it must never grow a dependency: anything it pulls in becomes another axis that can
force a version split. Absolute-path dependency pins are likewise never acceptable anywhere in
the family — they resolve on one machine.

### 4. The feature-vector layout is versioned by identity, not by position

There is no global "feature vector layout" to keep in sync. A column is identified by its
qualified `name@hex8`, and a value vector is built in **request order** by `reuse_for`, so a
consumer's layout is defined by its own `Request`. Two consequences:

- A feature whose *code* changes gets a new qualified name, so a model pinned with
  `Select::Features` misses rather than silently reading drifted values.
- Across *stored* offers (different images/configs), `schema_hash` is the blend gate —
  equal hash ⇒ identical columns under identical conditions ⇒ safe to stack.

## Version policy — iterating at `0.1.x`, freezing at `1.0` (then never `2.0`)

The surface is still settling, so the crate is at **`0.1.x`** — treat it as a single-version
in-house contract for now (Cargo doesn't unify `0.1` with `0.2`). Once it stabilizes it
**freezes at `1.0` and never goes to `2.0`**: from `1.0` on Cargo unifies every `1.x`, so a
dozen `zenanalyze` versions can share one contract type — which a `2.0` (or staying on `0.x`)
would split. The shape is already built for that freeze — private fields make field additions
non-breaking, and the enums (`Value`, `Select`, `FormatError`) plus `Provenance` are
`#[non_exhaustive]` — so reaching `1.0` is a version bump, not a redesign.

## License

MIT OR Apache-2.0.
