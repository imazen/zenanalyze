# PR: a co-versioning, type-free feature contract for the picker crate tree

**Status:** drafted 2026-06-19. zenanalyze-side code landed additively
(patch-bump; `cargo semver-checks`: no breaking change). Codec + `zenpredict`
migrations are follow-ups in their own repos.

## The hard requirement that drives the design

**A single build must link multiple zenanalyze majors at once — 0.1, 0.2, 0.3,
1.0, 2.0 — simultaneously.** Different codec crates pin different versions
because their baked models were trained against different feature *definitions*,
and a product (e.g. imageflow) pulls in several codecs. zenanalyze is pure
functions + `const` data — no `#[no_mangle]`, no `links`, no `static mut` /
`thread_local`, no `extern "C"` — so Cargo links every major/0.x-minor as a
separate crate, side by side, with no symbol or state clash. Verified.

This makes feature-definition **drift impossible by construction**: each model
is fed features by the *exact* version it was trained on, because that's the only
version that codec links. It replaces the fragile "pin a patch and re-validate"
contract with a build-time guarantee.

## The cardinal rule it forces

**No zenanalyze type may cross a crate boundary.** Every zenanalyze type is
version-specific: in a five-version build, `zenanalyze0_2::AnalysisFeature` and
`zenanalyze1_0::AnalysisFeature` are *different, incompatible* types — five
copies of everything. So the data-sharing contract is **core types only**
(`u16`, `u32`, `&str`, `Vec<u16>`, `&[f32]`) plus the shared `PixelSlice` pixel
currency. This is why there is no `FeatureSchema` adapter (an earlier draft had
one — a version-specific struct can't be a cross-version contract, and it's a
footgun precisely because it *looks* like one). The bind logic is free functions
that return core types.

## The data structures

### 1. What the MODEL carries — the source of truth (baked, in ZNPR metadata)

The model is self-describing. zenanalyze does **not** hold the schema — the model
does, and `zenpredict` (the version-agnostic runtime) owns the format.

```text
model.feature_columns      : Vec<String>   // feature NAMES, in model-input order
model.analyzer_version     : String        // "0.2" — the zenanalyze major.minor it was extracted with
model.feature_defs_version : u32           // within-major numeric-definition version at extract time
model.schema_hash          : u64           // BLAKE2b over feature_columns (names + order)
```

`feature_columns` + `schema_hash` already exist in ZNPR (`zentrain.feature_columns`,
header bytes 24..32). `analyzer_version` + `feature_defs_version` are the
additions (one new metadata TLV each). Names are the durable key — **not ids**,
which are version-local.

### 2. What CROSSES crate boundaries — core types only

```text
codec ── PixelSlice + &[u16] ids ──▶ zenanalyze@X      (extract)
codec ◀──────── &[f32] values ─────── zenanalyze@X      (in model order)
codec ──────── &[f32] values ──────▶ zenpredict         (predict)
codec ◀──────── &[f32] outputs ────── zenpredict         (codec params)
```

Never a `Vec<AnalysisFeature>`, an `AnalysisResults`, or any `@X` struct. The
shared predictor sees only `&[f32]`, so it works with features from *any*
zenanalyze version.

### 3. What the CODEC holds at runtime — core types it owns

```text
ids  : Vec<u16>   // resolved id order (from resolve_feature_ids, version-local)
defs : u32        // the model's baked feature_defs_version, for the load check
```

The codec may wrap these in *its own* struct; it holds no zenanalyze type, so its
public API — and its downstreams — stay free of zenanalyze across the version
boundary.

## The flow (example)

```rust
// ── at model load, once ────────────────────────────────────────────────────
// the model is self-describing: names + the version it was baked with.
let names: &[String] = model.feature_columns();              // ["variance", "edge_density", …]

// resolve against THIS crate's pinned zenanalyze -> a core Vec<u16> we own:
let ids: Vec<u16> = zenanalyze::resolve_feature_ids(names)
    .ok_or(MyErr::FeatureDrift)?;                            // a retired name => fall back, never a silent 0

// within-major drift backstop (cross-major is already guaranteed by the Cargo pin):
if zenanalyze::feature_defs_version() != model.feature_defs_version() {
    log::warn!("zenanalyze feature math moved within the major; re-validate this model");
}

// ── per image ──────────────────────────────────────────────────────────────
let mut feats = vec![0.0f32; ids.len()];
zenanalyze::feature_vector(slice, &ids, &mut feats);         // PixelSlice in, &[f32] out
let params = predictor.predict(&feats)?;                     // core &[f32] both directions
```

## How names + versions are shared — the lifecycle

```text
TRAIN  (zentrain, Python)
  extract features named feat_<name>  ──▶  train  ──▶  bake:
     feature_columns      = [variance, edge_density, …]     (names + order)
     analyzer_version     = "0.2"                            (the zenanalyze used to extract)
     feature_defs_version = zenanalyze::feature_defs_version()   (e.g. 1)
     schema_hash          = blake2b(feature_columns)

BUILD  (each codec's Cargo.toml pins the model's major.minor)
     zenjpeg → zenanalyze = "=0.2.7"
     zenwebp → zenanalyze = "=1.0.3"
     ⇒ Cargo links BOTH as separate crates — no conflict (pure-fn, no global state)

RUN
     each codec:  names ─resolve_feature_ids(its pinned @X)→ Vec<u16> ─feature_vector→ &[f32]
     shared predictor:   &[f32] ─predict→ &[f32]            (names no zenanalyze type)
```

## The version guarantees, layered
1. **Cross-major** (0.2 vs 1.0): the **Cargo pin**. A 0.2-model's codec links only
   0.2; it *cannot* be fed 1.0 features because 1.0 isn't in that codec's graph.
2. **Within-major numeric drift** (0.2.3 vs 0.2.7): `feature_defs_version` — baked
   vs runtime, mismatch warns.
3. **Name/order drift**: the existing `schema_hash` (zenpredict checks at load).

## Coexistence (the Cargo picture)
```toml
# imageflow/Cargo.toml — pulls in codecs that pin different analyzer majors
zenjpeg = "…"   # → zenanalyze =0.2.x
zenwebp = "…"   # → zenanalyze =1.0.x
```
```text
$ cargo tree -i zenanalyze
zenanalyze v0.2.7  ← zenjpeg
zenanalyze v1.0.3  ← zenwebp     # both present, both correct
```

## The zenanalyze surface (final — all free functions, all core types)
- `feature_count() / feature_ids(&mut [u16]) / feature_name(u16)`
- `feature_id_by_name(&str) -> Option<u16>` · `AnalysisFeature::from_name`
- `resolve_feature_ids(&[S]) -> Option<Vec<u16>>`  ← the bind step, returns a core `Vec<u16>`
- `feature_vector(slice, &[u16], &mut [f32]) -> bool`
- `feature_defs_version() -> u32`

No `FeatureSchema`, no exported struct on the contract path. The rich typed
surface (`AnalysisFeature`/`AnalysisResults`/`FeatureSet`) stays for in-process
callers that pin one version — it is simply never used *across* a boundary.

## Follow-ups (other repos)
- **zenpredict-bake** — bake `analyzer_version` + `feature_defs_version` into the
  model metadata.
- **zenpredict** — `Model::feature_columns()` / `.feature_defs_version()` /
  `.analyzer_version()` accessors (read-only, core types).
- **codecs** — bind via `resolve_feature_ids` + drop the `pub use
  zenanalyze::feature::*`; pin the model's analyzer major.minor.

## Why this dissolves the old worries
- **Drift** — gone, by co-versioning.
- **"Freeze at 1.0"** — no longer urgent: ship `0.3` with the pruned dense set
  *while* old codecs stay on `0.2`, coexisting. The set evolves across majors;
  each consumer migrates on its own schedule (see the dense-percentiles dedup
  doc). Promoting the dedup winners = cut a new major, force no one.
- **Type re-export** — structurally impossible to rely on: the types can't cross,
  only `&[f32]` does.
