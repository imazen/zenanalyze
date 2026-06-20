# zenanalyze-api ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenanalyze-api?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenanalyze-api?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenanalyze-api) ![docs.rs](https://img.shields.io/docsrs/zenanalyze-api?style=flat-square) ![License](https://img.shields.io/crates/l/zenanalyze-api?style=flat-square)

The frozen, version-unifying **feature contract** for the zenanalyze picker tree.

A product links **many `zenanalyze` versions at once** — a dozen codecs each pin
the `zenanalyze`/`zenpredict` version their model was trained against, and
`zenanalyze0_2::*` ≠ `zenanalyze1_0::*` are incompatible types. So no `zenanalyze`
type can be what crosses between layers. **This crate is that thing**: depend on
it at a single version and it *unifies* across the whole build.

It carries **only transport** — feature *names*, *values*, and *version stamps* —
never feature definitions, ids, or extraction. That is exactly why it can stay
frozen: the feature math churns every `zenanalyze` release; `name → value + a
version stamp + gather-by-name` does not.

## The flow

```text
1. each codec declares a Request (its model's feature names + the version it needs)
2. the caller unions the Requests, picks the best zenanalyze it has,
   and runs ONE pass over the union  ─────────────────────────▶  an Offer
3. each codec:  offer.reuse_for(my_request)?
                   Some(vec) => reuse (no second extraction)
                   None      => run its own zenanalyze@X pass
```

## Types

- **`Request`** — what a consumer wants: feature column names + the
  `(analyzer_version, defs_version)` it needs them at.
- **`Offer`** — a self-describing result: name→value pairs + the version stamp of
  the pass that produced them, with `matches` / `get` / `gather` / `reuse_for`.
- **`union_names`** — the distinct names across a set of requests (the single
  pass's work-list).

`no_std + alloc`, **no dependencies**, `forbid(unsafe_code)`. Stability is the
whole point — this crate must not break, or it splits the ecosystem.

## License

MIT OR Apache-2.0.
