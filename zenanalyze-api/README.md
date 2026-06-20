# zenanalyze-api ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenanalyze-api?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenanalyze-api?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenanalyze-api) ![docs.rs](https://img.shields.io/docsrs/zenanalyze-api?style=flat-square) ![License](https://img.shields.io/crates/l/zenanalyze-api?style=flat-square)

The frozen, version-unifying **feature contract** for the zenanalyze picker tree.

A product links **many `zenanalyze` versions at once** — a dozen codecs each pin
the `zenanalyze`/`zenpredict` version their model was trained against, and
`zenanalyze0_2::*` ≠ `zenanalyze1_0::*` are incompatible types. So no `zenanalyze`
type can be what crosses between layers. **This crate is that thing**: depend on
it at a single version and it *unifies* across the whole build.

It carries **only transport** — feature *names*, *values*, and a *reuse key* —
never feature definitions, ids, or extraction. That is exactly why it can stay
frozen: the feature math churns every `zenanalyze` release; `name → value + a
reuse key + gather-by-name` does not.

## The flow

```text
1. each codec declares a Request (its model's feature names + its reuse key)
2. the caller groups Requests by reuse key, unions the names in each group,
   picks the best zenanalyze it has, runs ONE pass per group  ────────▶  an Offer
3. each codec:  offer.reuse_for(my_request)?
                   Some(vec) => reuse (no second extraction)
                   None      => run its own zenanalyze@X pass
```

## Types

- **`Request`** — what a consumer wants: feature column names + the reuse key
  `(analyzer_version, defs_version, config_hash)` it needs them at. Build via
  `Request::new`.
- **`Offer`** — a self-describing result: name→value pairs + the reuse key of the
  pass that produced them, with `matches` / `get` / `gather` / `reuse_for`. Build
  via `Offer::new`.
- **`union_names`** — the distinct names across a set of requests (one group's
  single-pass work-list).

### The reuse key — why three parts

A feature is named the same across versions, but its *value* can change three
ways without the name changing, so reuse is gated on all three:

- **`analyzer_version`** (`major.minor`) — different math in a different release.
- **`defs_version`** — a within-`major.minor` numeric-definition bump.
- **`config_hash`** — the value-affecting **analysis config**
  (`AnalysisQuery::config_hash()`; `0` = default). The same build computes a
  different `variance` under linear-light vs gamma; this catches it. Opaque, so
  new config axes never touch this crate.

`no_std + alloc`, **no dependencies**, `forbid(unsafe_code)`. **1.0, additive-only,
never 2.0** — `Request`/`Offer` are `#[non_exhaustive]` so they grow without
breaking. Stability is the whole point: a breaking change splits the ecosystem.

## License

MIT OR Apache-2.0.
