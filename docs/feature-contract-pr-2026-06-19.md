# PR: a robust, type-free feature contract for the picker crate tree

**Status:** drafted 2026-06-19. Code landed additively (patch-bump, `cargo
semver-checks`: no breaking change). Codec migrations are follow-ups in their
own repos.

## Problem

zenanalyze produces feature vectors that feed compiled-in trained models across
a growing set of codec crates. Reading the three production integrations
(`zenwebp`, `zenjpeg`, `zenavif`) shows the contract between "the model's
feature columns" and "the analyzer's feature ids, in order" is bound **three
different ways, none blessed**:

| codec | bind | failure mode |
|---|---|---|
| zenwebp | hardcoded `ANALYSIS_FEATURES` enum array ↔ `FEAT_COLS` names, **position-aligned**, compile-time length check + schema-hash at load | brittle: add/reorder a feature ⇒ hand-resync two arrays; compile-time-bound |
| zenjpeg | embedded `feature_order.txt` of names, resolved at runtime via a hand-built `name→variant` map over `SUPPORTED.iter()` | re-implements the lookup; OK otherwise |
| zenavif | reads feature-column names **from the model metadata**, hand-built name→variant map | re-implements the lookup |

Three concrete flaws fall out:

1. **No `from_name`.** The reverse of `feature_name` doesn't exist, so every
   name-resolving codec hand-rolls a `HashMap` over `SUPPORTED.iter()`.
2. **Type re-export.** Codecs that bind by enum (zenwebp) or expose analysis
   results (`zenjpeg` does `pub use zenanalyze::feature::{AnalysisFeature,
   AnalysisQuery, FeatureSet}`) leak zenanalyze's typed surface into their own
   public API and their downstreams.
3. **Silent numeric-definition drift.** The schema hash a model carries (in
   `zenpredict`, over feature *names*) protects names + order, but a feature can
   keep its id/name while its numeric definition drifts — allowed within a 0.x
   minor — and every model silently eats the drift. No tripwire.

The previous "expose tier/cost/value-kind/schema-hash accessors" idea was wrong:
the runtime extracts a *fixed* set (cost/tier is a *training*-time ablation
concern), the vector is all f32 (value-kind is moot), and a (id,name) hash
duplicates the one `zenpredict` already bakes. None of those serve a real
consumer.

## Design

One **flat data plane** (core types only — no `AnalysisFeature` named by the
consumer) plus one **adapter**.

### Flat surface (added)
- `feature_id_by_name(name: &str) -> Option<u16>` and
  `AnalysisFeature::from_name(&str) -> Option<Self>` — the missing reverse.
  Accepts the `feat_` training-column prefix; `None` for unknown / retired /
  cfg-disabled (caller falls back rather than feed a silent zero).
- `feature_defs_version() -> u32` — monotonic version of the feature *numeric
  definitions*, independent of crate version and the id/name set.

### Adapter (added) — `FeatureSchema`
The blessed bridge. Resolve once from the model's column-name list, extract per
image:

```rust
let schema = zenanalyze::FeatureSchema::resolve(model.feature_columns())
    .ok_or(MyErr::FeatureDrift)?;          // a retired feature => fall back
if schema.defs_version() != model.baked_defs_version() {
    warn!("analyzer feature math moved since training; re-validate");
}
// per image:
let mut feats = vec![0.0f32; schema.len()];
schema.extract(slice, &mut feats);          // PixelSlice in, &[f32] out
let params = predictor.predict(&feats)?;    // core &[f32] -> codec params
```

`FeatureSchema` holds the resolved ids (in the model's order) + the defs
version. In/out are core types plus the shared `PixelSlice` pixel currency.

### Passing data without re-exporting types
This is what closes flaw #2. A codec uses `FeatureSchema` **internally**; its own
public API exposes only `&[f32]` (or its own newtype). It never re-exports
`AnalysisFeature` / `AnalysisQuery` / `AnalysisResults`. The data plane is the
"separate surface", `FeatureSchema` is the "adapter" — either is enough to keep
zenanalyze's typed surface out of a consumer's API. The rich typed surface stays
for callers who *want* it (it isn't removed — additive PR).

### Drift tripwire
`feature_defs_version()` is bumped in lockstep with the threshold-contract
CHANGELOG entries (any change to a feature's computed value). A model bakes the
version it trained against; the codec compares at load. Coarse but honest, and
it matches the existing "pin a patch, re-validate on bump" contract — it freezes
at 1.0. (A per-feature-set fingerprint is a future refinement; the global
counter is the minimal correct tripwire.)

## Codec migration (follow-ups, each in its own repo)
- **zenavif** is already close — switch its hand-built name→variant map to
  `FeatureSchema::resolve(model.feature_columns())`.
- **zenjpeg** — replace `resolve_features()` + the `pub use` with `FeatureSchema`;
  drop the re-export so its public API is core-typed.
- **zenwebp** — retire the position-aligned `ANALYSIS_FEATURES`/`FEAT_COLS`
  arrays for `FeatureSchema::resolve(FEAT_COLS)`; bake `feature_defs_version`
  next to the existing schema-hash check.

All three then bind the *same* way, forward-compatibly (additions ignored,
retirements caught by `resolve` returning `None`), with a definition-drift
tripwire and no type re-export.

## Explicitly out of scope (and why)
- **Range / units / monotonicity metadata.** Worth adding — but *after* 1.0, as
  a documented contract over a frozen set. Adding it now bakes a slot into the
  `features_table!` grammar for ~102 rows over a set that's still churning.
- **Redundancy-cluster membership.** Codec-specific + training-derived + the
  data files are partly stale. Belongs in `zenpicker`/`zentrain`; zenanalyze
  "doesn't bake content-class assumptions".
- **A definition-aware schema *hash*.** A hash can't cheaply capture algorithm
  drift; the monotonic `defs_version` counter is the pragmatic tripwire.

The deeper fix for drift is the contract's own endgame: converge the feature set
and freeze at 1.0 (see the dense-percentiles dedup doc — that's the gate).
