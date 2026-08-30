# `zenanalyze-api` 0.1.1 freeze review — 2026-08-30

Fine-tooth-comb review of everything added between published `0.1.0` and unpublished
`0.1.1`, against the standard *"it will be set in stone for a decade, less is more"*:
default verdict **cut**, burden of proof on keeping.

**Outcome: 0.1.1 = 0.1.0 + `Select::Names`.** One enum variant. Roughly twelve permanent
public items were removed before publication.

Verified mechanically — the complete public-surface delta against the published crate:

```console
$ diff -u ~/.cache/cargo-read/zenanalyze-api-0.1.0/src/lib.rs src/lib.rs \
    | grep '^+' | grep -E 'pub (fn|struct|enum|trait|const|mod|use)|^\+\s+[A-Z]'
    Names(&'a [&'a str]),
```

## Baseline

Established from the published crate (`cargo read zenanalyze-api` → `0.1.0`), not from the
CHANGELOG's claims. The delta was exactly what commit `47c3c69` added: `FeatureProvider`,
`ProviderError`, `OwnedCatalog`, `Select::Names`, plus a private `union_impl` refactor.
Nothing else moved — no derive changes, no signature changes to 0.1.0 items.

## Verdicts

| Item | Real consumers, measured | Verdict |
|---|---|---|
| `Select::Names(&[&str])` | zenwebp `classifier.rs:459`; zenanalyze `offer.rs` (producer). zenavif / zenjpeg reference it in comments as the thing they deliberately do *not* use. | **KEEP** — the keystone |
| `FeatureProvider` (3 methods) | zenwebp `classifier.rs:559`, zensr `chooser.rs:162` — one-line detours, both crates already had `Offer`-taking paths | **CUT** |
| `ProviderError` (3 variants) | none — existed solely as the trait's error type | **CUT** |
| `OwnedCatalog` (8 methods) | **zero.** Only `zenanalyze/src/offer.rs:130` *constructs* one. No consumer called any of the eight methods; the only call sites of `.catalog()` / `.has_name()` / `.unmet()` / `.len()` were tests inside the two crates that define and implement it. | **CUT** |

### `Select::Names` — keep, and it is the whole reason 0.1.1 exists

The intended model is **push**: the host runs one pass and *gives* the codec the data, the
codec answers yes/no, and on "no" it runs its own scan. Answering "is this enough?" has to
work *across analyzer versions* — which is exactly matching by bare name at whatever version
is on offer. `Select::Features` (0.1.0) pins the version hash; `Select::All` is everything.
`Names` is the only selector that expresses the question, which is why three codecs reached
for it. Cutting it would force a threshold heuristic to hard-code version hashes it cannot
know across builds — the precise pressure that pushes a crate back onto a direct `zenanalyze`
dependency.

Additive on a `#[non_exhaustive]` enum, so a 0.1.0-compiled consumer already has a wildcard
arm and is unaffected.

### `FeatureProvider` — cut. Not merely unused: the wrong direction of control.

The load-bearing finding. **Every verb the intended flow needs already shipped in 0.1.0:**

| step in the model | API | shipped in |
|---|---|---|
| codec is *given* the data | `Offer<'a>` / `OwnedOffer` | **0.1.0** |
| yes/no — is it enough? | `offer.satisfies(&Request) -> bool` | **0.1.0** |
| the values, when yes | `offer.reuse_for(&Request) -> Option<Vec<f32>>` | **0.1.0** |
| *which* wants missed, and whether a present one drifted | `offer.get(name) -> Option<&FeatureResult>` | **0.1.0** |
| if no — own scan | direct `zenanalyze` dep, permitted since 2026-08-28 | n/a |

The trait sat on **none** of those steps. The model is push; a `&dyn` is pull — it lets a
codec reach *back* into a live analyzer. Nothing in the flow ever calls it.

Three further reasons, recorded in `zenanalyze-api/README.md` ("Why there is no provider
trait") so the trait is not re-proposed in a year as an obvious gap:

1. **Data serializes; a trait does not.** An `Offer` crosses a process, a file, a cache and a
   *version* boundary — `to_block` / `parse` are right there. A `&dyn` crosses none of them.
   For a contract whose entire purpose is letting analyzer versions coexist, serializable data
   is strictly more powerful than dynamic dispatch.
2. **A trait is a promise about _how_.** This surface freezes at `1.0` and never breaks after.
   A decade is too long to freeze someone else's method set.
3. **It dragged a pixel buffer into a contract that has nothing to do with pixels — and got
   it wrong.** See below.

### The stride defect, and why deleting beat fixing

`extract_rgb8` hard-coded tightly-packed 8-bit sRGB RGB with no row padding. Measured:

- **zenanalyze already supports stride natively** — `analyze_features(PixelSlice, …)`
  (`src/lib.rs:405`) and `packed8_slice(bytes, w, h, row_stride, channels)` (`src/lib.rs:1302`),
  with the `row_stride == 0 ⇒ tightly packed` convention already established in this workspace.
- **A consumer was already paying for the gap.** `zensr`'s `center_crop_rgb8`
  (`chooser.rs:93-103`) allocates a full 512×512×3 buffer and copies row by row — solely
  because the contract could not express a strided sub-region.
- The workspace's Pixel Buffer APIs rule requires native stride support anywhere a function
  takes "the whole image".

So the contract was strictly narrower than both the producer above it and the consumer below
it, and that narrowness was about to be frozen for a decade. Adding `row_stride: usize` would
have fixed the symptom. **Removing the pixel buffer from the contract entirely makes the whole
class of mistake unrepresentable**, which is the better resolution — a contract that carries no
pixels cannot describe them wrongly.

### `ProviderError` / `OwnedCatalog` — cut, cascading

`ProviderError` existed only as the trait's error type. `OwnedCatalog` existed only as
`catalog()`'s return; `Catalog<'a>` covers the borrowed case and has been frozen in 0.1.0 all
along.

`OwnedCatalog` deserves a note because it was the clearest case on its own merits: **eight
permanent public methods, zero production callers**, six of them (`len`, `is_empty`, `offers`,
`has_name`, `unmet`, `union`) straight duplicates of `Catalog`'s. Even had the trait survived,
it should have shrunk to two.

## The one thing I checked before agreeing to cut

Whether `satisfies` / `reuse_for` fully express the yes/no — specifically whether a codec can
learn *which* wants were missing from an `Offer` alone, without a catalog type.

**They can; there is no gap, and nothing needed adding.** `Offer::get(name)` (0.1.0) is by
bare name and classifies each want directly: `None` ⇒ missing, `Some` whose `version_hash`
disagrees ⇒ a code drift. `Catalog::unmet` answers a different question (what a *build* can
produce, not what an *offer* carries) and is already frozen in 0.1.0 for the borrowed case.

This is now a permanent test rather than an assertion —
`push_model_answers_yes_no_and_names_the_gaps` in `zenanalyze-api/src/tests.rs` walks the
entire flow in 0.1.0 verbs plus `Select::Names`.

## Cost of the removals: none

The brief anticipated a breaking-change cost in two published codecs and asked for the version
bump each needs. **There is no cost.** Checked against crates.io:

| crate | published max | provider surface present there? |
|---|---|---|
| `zenanalyze-api` | 0.1.0 | no — 0.1.1 never published |
| `zenwebp` | **0.4.4** | **no `analyzer` feature at all** (verified in the published `Cargo.toml`) |
| `zensr-zenjpeg` | **not published** | n/a |

`classify_image_type_with_provider`, `classify_with_provider` and both `bundled_provider()`s
are entirely unreleased. **No version bump is owed by anyone**, and no downstream user can be
holding a reference to any removed item.

## Migration landed

- **zenanalyze** — `Analyzer` + its trait impl replaced by
  `offer_for_request(rgb, w, h, &request) -> Result<OwnedOffer, AnalyzeError>`. The
  `Select` → `FeatureSet` resolution stays in **one** place instead of being duplicated into
  each codec (this repo's no-duplicate-implementations rule).
- **zenwebp** — `classify_image_type_from_owned_offer` added; `_with_provider` and
  `bundled_provider` removed; `classify_image_type_rgb8` / `_diag` keep their signatures and
  scan via `offer_for_request`.
- **zensr** — `classify_rgb8_scanning` (returns the error) replaces `classify_with_provider`;
  `classify_rgb8` keeps its signature and its `Photo`-on-failure precision bias.

## Feature flags: `analyzer-bundled` and `chooser-bundled` removed

Both existed only to make the concrete `zenanalyze` dep optional — a property nobody needs now
that a direct dep is policy-permitted. Both actively harmed, and the evidence is not
theoretical:

- **zenwebp: the flag split the lint gate.** CI's `Clippy` job ran only
  `cargo clippy --features analyzer --all-targets -- -D warnings`. The five `dev/` targets were
  gated behind `analyzer-bundled`, so **they were never linted at all** — six findings had
  accumulated unseen (`manual_clamp` ×3, `manual_is_multiple_of` ×2, `&PathBuf`→`&Path`). All
  fixed. This is the "a flag creates a configuration nobody builds, and bugs live there" shape
  reproducing exactly.
- **zensr: the flag made a CI step vacuous.** `Test (chooser)` ran a configuration in which
  every behavioural chooser test was `#[cfg]`'d out — a green step asserting only that the
  crate compiled. The real tests lived behind `chooser-bundled`.

Now one flag each, one configuration, one gate. The rule that actually matters is unchanged
and independently checkable by reading signatures: **no `zenanalyze` type appears in any public
signature** of either codec, so a host on a different `zenanalyze` version can still drive
`classify_image_type_from_offer` / `classify_from_offer` with its own pass.

## Gates

| gate | result |
|---|---|
| `zenanalyze-api` tests | 18 lib + 7 doctests + README-coverage tripwire — green |
| `zenanalyze` tests (`--features api`) | 211 passed, 0 failed |
| `zenwebp` (`--features analyzer`) | builds; 322 lib tests pass |
| `zensr-zenjpeg` (`--features chooser`) | builds; 31 tests pass |
| `zenjpeg`, `zenavif` (`--features auto-tune`) | build clean — neither ever used the trait |
| clippy `-D warnings` | clean in every file touched, all repos |
| `cargo fmt` | clean |
| `cargo semver-checks` vs published 0.1.0 | 196 checks, 196 pass — *"no semver update required"* |

**On semver-checks:** a green run here is a lower bound, not a clearance — it has no lint for
an inherent method's return type changing and cannot model behavioural breaks. The actual proof
is that all four real consumers compile and pass against the trimmed tree.

## The surface is now recorded, and one gate is missing

Two things found by going looking, not by any gate firing:

- **`docs/public-api/zenanalyze.features.txt` was stale.** It still listed
  `pub struct Analyzer`, `Analyzer::new`, and the roster line
  `Analyzer: … zenanalyze_api::FeatureProvider` after the trait was cut. Regenerated.
- **`zenanalyze-api` had no snapshot at all.** The apidoc runner carries an explicit crate
  list and the contract crate was not on it — so the one published crate in this repo that
  must *never* break was the only one with no committed surface record. Added, listed first.
  `docs/public-api/zenanalyze-api.txt` is now the diffable record of exactly what freezes at
  `1.0`: 11 types, 73 public lines, `Select::Names` present and no `FeatureProvider` /
  `ProviderError` / `OwnedCatalog`.

**The gap I did not close:** these snapshots are **not gated in CI**. `apidoc/` is
workspace-excluded precisely so no CI job compiles it or runs rustdoc, and the only check is
the manual `just api-doc-check`. That is why the stale entry above survived. A gate would have
caught it, but adding one trades against a deliberate choice to keep rustdoc out of CI — the
owner's call, not mine to make unilaterally.

Worth knowing: **`zencodec` has the identical gap and it has already bitten.** Its
`docs/public-api/zencodec-testkit.txt` is stale by one item (`check_gain_map_roundtrip`,
`zencodec-testkit/src/lib.rs:1459`, added in `857fab1` after the last regeneration), and its
CI has no `api-doc` job either. `zenpixels` *does* gate it — which is exactly why zenpixels'
snapshot was caught and regenerated while these two drifted.

## Loose ends, for the owner

1. **`zensr`'s graph resolves two `zenanalyze` sources.** `zenjpeg`'s rev pin
   (`rev=13d40c3be60e`, via `zenjpeg rev=e277e9c9`) sits alongside the patched `main`, so
   `zenanalyze` 0.2.0 is in the build twice. **`zenanalyze-api` resolves to exactly one**, so
   the `Offer` type still unifies — the contract did its job and contained the damage. Belongs
   to zenjpeg's pin, not to this change; left alone.
2. **`zenavif/Cargo.toml:208`** still names `FeatureProvider` in a comment. zenavif was outside
   the editable scope; it compiles fine (the mention is prose only).
3. Prior-session WIP was found and **preserved, not discarded**, in three repos: a fuzz dep-pin
   (zenanalyze), three nested lockfiles + CHANGELOG (zenwebp), and a `chain444` sweep output
   (zensr, still at the repo root — relocating it to `benchmarks/` with a dated name is queued).
   Each is now its own described commit.
4. `zenavif` and `zenjpeg` have modified `Cargo.lock`s from the `cargo update` needed to compile
   them as proof. Pure regeneration (only the `zenanalyze` git rev moved); nothing else in those
   read-only repos was touched.
