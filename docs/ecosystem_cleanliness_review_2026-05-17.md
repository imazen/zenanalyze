# Ecosystem cleanliness + testability review (2026-05-17)

Concrete, prioritized list of cleanups across the
zenpredict / zenpredict-bake / zentrain / per-codec-picker /
zensim pipeline. Each entry has a **cost** estimate and a **payoff**
note; sort by payoff/cost ratio when picking the next batch.

## P0 — bugs / risks (do soon)

### 1. `zenpredict-bake` is a workspace member but CI ignores it

`Cargo.toml`'s `[workspace] members = ["zenpicker", "zenpredict",
"zenpredict-bake"]`, but `.github/workflows/ci.yml` has **zero**
references to zenpredict-bake. Its 70+ integration tests
(`tests/json_bake.rs`, `tests/multi_codec.rs`, `tests/lifecycle.rs`,
`tests/compression_and_perm.rs`, …) run **only** on local
`cargo test -p zenpredict-bake`. The five JSON-knob tests I just
landed for `zerobias_tau` / `compressed` / `optimize` are not
CI-gated.

**Fix**: clone the `zenpredict` job in `ci.yml`, swap the crate name,
add the matrix axes (`ubuntu-latest`, `windows-11-arm`, `macos-15-intel`,
`macos-latest`, plus i686-via-cross). ~30 lines of YAML.
**Cost**: 10 min. **Payoff**: catches regressions instead of finding
them post-publish.

### 2. Feature-transform token vocabulary is duplicated 3 ways

- `zenpredict/src/feature_transform.rs` — authoritative enum (14 variants).
- `zenpredict-bake/src/composer.rs` — bake-side validator (string ↔ enum).
- `zentrain/tools/train_hybrid.py:777` — `_VALID_FEATURE_TRANSFORMS` set (14 strings).
- `zenanalyze/tools/bake_picker.py` — `_VALID_RUNTIME_TRANSFORMS` set (14 strings, copied).
- `zentrain/tools/feature_transform_sweep.py` — `TRANSFORMS` dict (function table, same 14).

Whenever a new variant lands, **four files** must change in lockstep.
The current 14-stack set landed mostly-correct because one person
shipped it end-to-end in one PR, but the next stack proposal (e.g.
"asinh") will be a coin-flip on whether anyone remembers all four.

**Fix**: emit a generated `feature_transform_tokens.json` from
`zenpredict-bake`'s build script (or a `tokens` subcommand on
`zenpredict-bake`), have the Python sets load it at module
init. Or simpler: a `_picker_lib.py` constant that all Python
consumers import. The Rust side becomes the canonical source via a
build-emitted artifact.
**Cost**: 1 hour. **Payoff**: removes a recurring drift class.

### 3. `BakeRequestJson` is not `#[non_exhaustive]`

`zenpredict-bake/src/json.rs:134`. Today I added three optional
fields (`zerobias_tau`, `compressed`, `optimize`). Anyone holding
struct-literal `BakeRequestJson { schema_hash, scaler_mean, … }`
in Rust now fails to compile — including (eventually) any in-tree
test that constructs one directly. Each future addition is a
silent Rust API break.

**Fix**: add `#[non_exhaustive]` to `BakeRequestJson` and a
`BakeRequestJson::new(...)` constructor with the required four
fields. Document in CHANGELOG as a one-time break (consistent
with the multi-codec / stacks breaks already in `[Unreleased]`).
**Cost**: 15 min. **Payoff**: future additions are non-breaking.

### 4. Multi-seed lock is not enforced

`zentrain/tools/feature_transform_sweep.py` writes
`recommended_transforms.py` from a single seed. `promote_recommended_to_config.py`
generates a v2 codec config from that output without checking
whether the recommendation has been multi-seed-confirmed. Today's
session showed two of three single-seed picks were actually
regress / noise — that means **anyone running the canonical pipeline
can productize a regression in one command**.

**Fix**: `promote_recommended_to_config.py` should require a
`--multiseed-aggregate <path>` argument pointing at a
`multi_seed_confirm.py` `aggregate.json`, and refuse to write the
v2 config unless the verdict is `ship`. Provide a `--force` escape
hatch for explicit user override, with a banner in the generated
config docstring.
**Cost**: 30 min. **Payoff**: structural prevention of last
session's exact failure mode.

## P1 — testability gaps (high payoff)

### 5. No Python tests in CI

The Python pipeline (bake_picker, train_hybrid, feature_transform_sweep,
multi_seed_confirm, seed_stable_screen, build_pruned_config, …) is
~10 kLOC and has zero automated tests. Local-only tests exist
(`tools/test_bake_roundtrip.py`) but aren't run by CI.

**Fix**: add a `python-tests` job to `ci.yml` that runs
`python3 -m pytest zentrain/tests/ tools/tests/`. Start with
just `test_bake_roundtrip.py` to prove the wiring; new bake-knob
tests can land alongside the JSON-side ones I added.
**Cost**: 1 hour (CI + 2-3 starter tests). **Payoff**:
catches Python-side regressions, which are currently caught only
by someone running the script and noticing it broke.

### 6. End-to-end test crossing Python → Rust does not exist

`test_bake_roundtrip.py` round-trips a *synthetic* model through
the bake CLI back through Rust — but it doesn't:
- Train an actual tiny model (PyTorch or sklearn)
- Apply real feature transforms via `_apply_feature_transform`
- Bake with `--zerobias-tau --compress --optimize`
- Load via Rust runtime and compare predict outputs

So a per-feature transform bug in Python `_apply_feature_transform`
that disagrees with Rust `FeatureTransform::apply` would survive
both crates' unit tests.

**Fix**: a single `tests/e2e_python_rust_parity.py` that runs
a tiny pipeline end-to-end and asserts `predict(features)` agrees
within 1e-5 between the Python forward pass and Rust runtime.
Iterate over every transform token + a representative param set.
**Cost**: 3 hours. **Payoff**: catches Python/Rust drift in
transforms — currently the highest-risk silent-divergence class.

### 7. `bake_picker.py` is 1,151 lines monolithic

Mix of: argparse, model JSON loading, schema-hash derivation,
metadata encoding (cell_rescue_hints / zq_fallback_table /
output_bounds / feature_transforms / feature_transform_params /
multi_codec_schema), feature-bound encoding, output-spec encoding,
sparse-override encoding, binary location, subprocess invocation,
legacy manifest emission, the new compression flags.

Hard to unit-test individual encoders without re-implementing the
whole entrypoint. The `encode_*` functions are good split candidates.

**Fix**: split into
- `bake_picker/__main__.py` (argparse + orchestration)
- `bake_picker/json_build.py` (`build_bake_request_json` + encoders)
- `bake_picker/metadata.py` (encoders for cell_rescue_hints, etc.)
- `bake_picker/invocation.py` (find_bake_bin + invoke_bake)
- `bake_picker/manifest.py` (legacy manifest)

Then unit-test each encoder against a fixture model JSON. `bake()`
becomes ~30 lines.
**Cost**: 4 hours (large but pure refactor + tests). **Payoff**:
the encoders become testable in isolation; trainer changes that
add a new metadata key get a small obvious test, not a giant
end-to-end one.

### 8. Per-codec configs use module-globals convention, not schema

`zentrain/examples/zen{jpeg,webp,avif,jxl}_picker_config.py` are
just modules with top-level constants: `PARETO = Path(...)`,
`KEEP_FEATURES = [...]`, `CATEGORICAL_AXES = [...]`,
`FEATURE_TRANSFORMS = {...}`, `FEATURE_GROUPS = {...}` (some configs
only), `OUTPUT_SPECS = {...}` (some), `parse_config_name(name) -> dict`.

Problems:
- Trainer does `getattr(module, "KEEP_FEATURES", default_list)` —
  silent fallbacks when names drift.
- No schema validation. A typo in a key (`KEEP_FEATURE` vs `KEEP_FEATURES`)
  silently uses the default.
- Hard to test config validity without spinning up the trainer.
- 24 codec config files exist in `zentrain/examples/`, most of
  which are historical (v04, v04full, v06, v07, v01_schema, _hvs,
  _v2, _v3_stable, _pruned, _expanded, _overnight, …). The trainer
  has no way to mark some as deprecated.

**Fix**: introduce `CodecConfig` dataclass with explicit fields
+ a `validate(self)` method that catches typos, dimension
mismatches, and missing transforms. Each `zen<codec>_picker_config.py`
becomes a module that exports `CONFIG: CodecConfig = CodecConfig(...)`.
Migration: keep the legacy globals as a deprecated path for one
release.
**Cost**: 1 day (refactor + migrate all live configs).
**Payoff**: silent-fallback class of bugs goes away; configs become
unit-testable; archive easier (deprecated configs can be marked).

## P2 — duplication / dead code (medium payoff)

### 9. 11 historical `tools/v0*` + `tools/v1*` per-version picker trainers

`tools/v0_2_zenjpeg_picker_train.py`, `v06_champ_per_class.py`,
`v06_zenjxl_picker_mlp_train.py`, `v07b_zenjxl_picker_no_screen.py`,
`v07_zenjxl_picker_with_cclass.py`, `v10_router_mlp_train.py`,
`v12_metapicker_train.py`, `v14_metapicker_train.py`,
`v15_compare_pickers.py`, `v15_metapicker_train.py`,
`v15_zenjpeg_picker_train.py` — none are imported by the current
trainer (`zentrain/tools/train_hybrid.py` is the canonical entry).

These accreted from the per-version codec-picker exploration
(March-May 2026). They each declare their own argparse / load
their own paretos / call `bake_picker`. Reading them is necessary
to understand the picker history but they're not load-bearing.

**Fix**: move to `tools/legacy/` with a README explaining the
v0_2 → v14 → train_hybrid evolution, OR just delete and rely on
git history (`git log -- tools/v0_2_zenjpeg_picker_train.py`).
**Cost**: 30 min. **Payoff**: ~11 files / ~4 kLOC out of the
top-level tools/ namespace; faster `ls` for new contributors.

### 10. 24 files in `zentrain/examples/` codec configs

After the cleanups in P1.8, audit which configs are still live:

| Current files | Status |
|---|---|
| `zenjpeg_picker_config.py` | LIVE — canonical zenjpeg trainer config |
| `zenwebp_picker_config.py` | LIVE — canonical zenwebp |
| `zenavif_picker_config.py` | LIVE — canonical zenavif |
| `zenjxl_picker_config.py` | LIVE — canonical zenjxl |
| `*_v2.py`, `*_v3_stable.py` | shipping candidates this session |
| `*_pruned.py` | experimental from this session |
| `*_v04.py`, `*_v04full.py`, `*_v06.py` | HISTORICAL |
| `*_overnight.py`, `*_expanded.py`, `*_v01_schema.py`, `*_hvs.py` | EXPERIMENTAL/HISTORICAL |

**Fix**: move historical to `zentrain/examples/legacy/` with date
banners. Leave a README. Mark `*_v2 / _v3_stable / _pruned` as
"sweep result, not production" in their docstrings (already partly
done via the retract banners in this session).
**Cost**: 20 min. **Payoff**: discoverability — finding which
config is canonical for a codec is currently a forensic exercise.

### 11. Several Python wrappers redo "find baker binary"

In zenanalyze: `tools/bake_picker.py:find_bake_bin()` — well-formed.
In zensim (different repo, can't touch): `bake_to_znpr.py:_find_bake_bin`,
`bake_znpr_v3.py:find_baker` — both reimplement the same fallback
list with slightly different orderings.

**Fix**: publish a tiny `zenpredict-bake-python` helper module in
`zenpredict-bake/python/zenpredict_bake.py` that exports
`find_baker_binary() -> Path` and `bake_from_dict(req: dict) -> bytes`.
All three Python callers (and any future ones) import that instead.
**Cost**: 1 hour. **Payoff**: removes the per-wrapper version-drift
risk; one canonical search path. Especially valuable as zensim's
copies WILL drift from zenanalyze's over time. Also makes the
imminent zensim-side update trivial — they just import the helper.

### 12. Two CLI binaries with overlapping scope

`zenpredict-bake` (legacy, bake only) and `zenpredict bake | inspect
| repack` (unified, three subcommands). Both ship as bin targets
in `zenpredict-bake/Cargo.toml`. The unified `zenpredict` binary
delegates to the same `run_*_cli` functions the legacy binaries
call, but the legacy ones are documented in README and the unified
one is documented in CHANGELOG. New contributors don't know which
to use.

**Fix**: pick one and deprecate the other (with a print-to-stderr
deprecation note that exits non-zero after a release). Mention in
CHANGELOG. The unified is the obvious target.
**Cost**: 30 min. **Payoff**: a future user docs pass doesn't have
to explain both.

## P3 — nice-to-have (low cost)

### 13. Doc-link inversion in `_VALID_FEATURE_TRANSFORMS`

`train_hybrid.py:777` ships the list with comments referring to
"zenpredict 0.2.1+" for stacks but the list at `tools/bake_picker.py`
doesn't. Same set, different provenance comments.

**Fix**: after P0.2 (shared token source), this dissolves.

### 14. `zerobias_rebake.py` (~360 lines) is now redundant

It existed because the JSON pipeline couldn't do zerobias. Now it
can. Most use cases are subsumed; only "I have a baked .bin and
want to apply zerobias retroactively without re-training" remains.

**Fix**: add a soft-deprecation header (✅ done today) directing
users to the JSON pipeline; keep the script for the rebake-only
case. Schedule removal once `zenpredict repack` covers the
retroactive case (it already does — `zenpredict repack --zerobias
0.005 --dtype i8 --compress`).
**Cost**: 0 (already done). **Payoff**: clarifies which tool to
reach for in new work.

### 15. `safety_report.violations` is text-only

`train_hybrid.py:1850` ships violations as f-string error
messages (`PER_ZQ_TAIL: zq=92 p99 overhead 99.6% > threshold 80.0%`).
Programmatic gating must regex-parse those strings (which is what
`bake_picker.py:953` does to count violations).

**Fix**: emit violations as `{code, args, threshold, observed,
message}` structured dicts. Human message stays in `message`;
gates can switch on `code`.
**Cost**: 1 hour. **Payoff**: future safety gates (e.g.
"refuse to ship if argmin < threshold UNLESS the only violation
is DATA_STARVED_SIZE") become trivially configurable.

### 16. Versioning policy for zenpredict-bake unstated

`zenanalyze/CLAUDE.md` documents 0.2.x policy for `zenanalyze`.
`zenpredict/CHANGELOG.md` is currently shipping breaking changes.
`zenpredict-bake/CHANGELOG.md` is accumulating breaking changes
under `[Unreleased]` with no per-crate policy. Each crate now
publishes independently (CHANGELOG says so).

**Fix**: a one-paragraph "API stability" section in
`zenpredict-bake/CLAUDE.md` matching the zenanalyze pattern.
**Cost**: 10 min. **Payoff**: future contributors know what's
breakable.

---

## Summary order (by payoff/cost)

1. **#1** CI for zenpredict-bake — 10 min, prevents publish regressions
2. **#3** `BakeRequestJson` `#[non_exhaustive]` — 15 min, prevents future API breaks
3. **#4** Multi-seed lock enforced in promotion — 30 min, prevents this session's bug class
4. **#9** Move legacy `tools/v0*`+`v1*` to `tools/legacy/` — 30 min, discoverability
5. **#10** Move legacy `zentrain/examples/` configs to `legacy/` — 20 min, discoverability
6. **#12** Pick a canonical bake CLI — 30 min, one less footgun
7. **#11** Shared `zenpredict_bake.py` helper module — 1 hour, removes cross-repo drift
8. **#2** Generated shared token list — 1 hour, removes drift class
9. **#5** Python tests in CI — 1 hour, catches Python-side regressions
10. **#15** Structured safety violations — 1 hour, configurability win
11. **#6** End-to-end Python ↔ Rust parity test — 3 hours, highest-risk class
12. **#7** Split `bake_picker.py` into a package — 4 hours, testability win
13. **#8** `CodecConfig` dataclass — 1 day, removes silent-fallback bugs

The top six are all under an hour each and clear most of the
structural risk. Days 1-2 cumulative wall time would deliver every
P0+P1 item.
