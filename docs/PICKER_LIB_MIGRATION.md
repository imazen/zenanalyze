# `_picker_lib` Migration — what's done, what's left

Tracking doc for the Tier-1 #6 dedup chunk
(`benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md`, line 46): "zenanalyze
ablation/probe forks — 4 tools carry byte-identical `load_pareto` /
`load_features` predating `_picker_lib.py`; 3 reimplement numpy
forward-pass". The first half (loaders) shipped in this chunk; the
second half (forward-pass) needs `_picker_lib` to grow a runtime helper
before it can migrate, and is tracked here.

## Status

| Concern | Status |
|---|---|
| `load_pareto` / `load_features` in 4 zentrain ablation tools | **SHIPPED** (this chunk) |
| `_picker_lib.load_features_raw(strict=False)` lenient mode | **SHIPPED** (this chunk) |
| numpy forward-pass duplication in 3 probe tools | **DEFERRED** — needs new helper |
| `build_dataset` (dense `config_id_remap`) | **STAYS LOCAL** (intentional variation) |

## What shipped

Four `zentrain/tools/` scripts replaced their local
`load_pareto` / `load_features` impls with delegations to
`_picker_lib.load_pareto_raw` / `_picker_lib.load_features_raw`:

- `zentrain/tools/feature_ablation.py`
- `zentrain/tools/feature_group_ablation.py`
- `zentrain/tools/validate_schema.py`
- `zentrain/tools/capacity_sweep.py`

Each tool's local `load_pareto` is now a 5-line wrapper that calls
`_picker.load_pareto_raw(path)` and merges the returned `config_names`
into the module-global `CONFIG_NAMES` (preserving the side-effect that
downstream `build_dataset` relies on for either dense remap or
`max(CONFIG_NAMES)+1` sizing).

`capacity_sweep`'s pre-dedup `load_features` was lenient — silently
dropped `KEEP_FEATURES` columns absent from the TSV. The pre-existing
`_picker_lib.load_features_raw` raised `SystemExit` on missing columns
(strict). This chunk added a `strict: bool = True` keyword argument:
True (default) preserves the existing 3-strict-tool behavior plus
`load_or_build_dataset`'s internal caller; `capacity_sweep` passes
`strict=False`. New unit test
`zentrain/tools/test_picker_lib_strict.py` covers all 5 modes (no
filter / subset / strict raise / lenient drop / lenient all-missing).

`build_dataset` stayed local in every tool because it carries
tool-specific logic the canonical lib doesn't have:

- `feature_ablation.py`: dense `config_id_remap = {cid: dense for
  dense, cid in enumerate(sorted(CONFIG_NAMES))}` to compact sparse
  hashed config_id spaces (zenavif phase-1a).
- `feature_group_ablation.py`: per-feature-group masking semantics
  (drops cols not in the active group).
- `validate_schema.py`: schema-validation post-processing.
- `capacity_sweep.py`: cross-term recipe dispatcher (calls
  `make_engineered_v{1..5}` to vary the X feature space).

A future chunk could extract the dense-remap path as an opt-in
`_picker_lib.build_dataset_remapped(...)` if any of the 3
`max(CONFIG_NAMES)+1` tools want to harden against sparse config_ids,
but that's not required for the current corpora.

## Per-cell evidence

Each migrated tool ships an adjacent `<tool>_migration_evidence.txt`
with summary stats from a BEFORE/AFTER run on the live
`/home/lilith/work/zen/zenjpeg/benchmarks/zq_pareto_2026-04-29.tsv` +
`zq_pareto_features_2026-04-29.tsv` (1388 cells × 3,497,409 samples ×
120 configs × 19 feat cols). `diff before.json after.json` was empty
across all 4 tools (BYTE-IDENTICAL).

## Deferred — numpy forward-pass duplication

The Tier-1 #6 synthesis line "3 reimplement numpy forward-pass" maps
to three (and likely a fourth) inference-side scripts that hand-roll
the standard `Z = X @ W + b; activate; ...` loop instead of routing
through a shared runtime helper:

| File | Function | Notes |
|---|---|---|
| `zentrain/tools/zerobias_rebake.py` | `forward(features, mean, scale, layers)` | Standardize-then-MLP forward. |
| `zentrain/tools/inspect_picker.py` | `forward(model, features)` / `forward_batch(model, X)` | Two near-identical impls in the same file. |
| `tools/v15_compare_pickers.py` | `forward(model: dict, X)` | LeakyReLU + linear head. |
| `tools/holdout_ab_lookup.py` | `forward(model_json, x)` | Same shape. |
| `zentrain/tools/student_permutation.py` | `forward_one` / `forward_batch` (inline closures) | Different signature, less obvious dup. |

`_picker_lib` does not currently expose a numpy forward-pass — its
scope is training-dataset construction (load + cache + pivot + train
teachers + evaluate argmin). The runtime concern is owned by
`zenpredict::Predictor` (Rust) and conceptually mirrored by the
in-tree `bake_roundtrip_check.py` reference path. The right place
for a Python forward-pass helper is **probably**:

- A new `zentrain/tools/_predict_lib.py` (sibling to `_picker_lib`)
  that exposes `forward(model_json, X)` and `forward_batch(...)`
  matching the canonical ZNPR v3 bake spec, OR
- Promote the relevant functions out of `bake_roundtrip_check.py`
  (which already encodes the canonical Rust-vs-numpy parity check)
  and have the 4 probe scripts above import that.

Either approach is one separate chunk. Constraint: the canonical
forward-pass MUST match `bake_roundtrip_check.py`'s reference (it is
the regression gate for `zenpredict-bake`); any extracted helper
should depend on that reference rather than become a parallel
implementation.

## Out of scope

Other zenanalyze scripts that define `load_features` but with
non-byte-identical semantics (NOT migrated by this chunk):

- `zentrain/tools/correlation_cleanup.py` — `load_features(path: Path) ->
  (matrix, feat_cols, n_rows)` returns a 2-D numpy array (NOT the dict
  keyed by `(image, size_class)` that `_picker_lib` returns) plus
  NaN-row filtering for tier3 percentile-on-tiny-image edge cases.
  Different return type → not a candidate for the standard delegation.
- `zentrain/tools/student_permutation.py` — `load_features(path: Path)
  -> (dict, list[str])` with NaN-row drop and `KEEP_FEATURES` filter.
  Returns the same dict shape as `_picker_lib`, but additionally drops
  rows whose features contain NaN (tiny images that skip percentile
  features, per zenanalyze#49). Could be migrated if `_picker_lib`
  grows an optional `drop_nan_rows: bool = False` parameter, but that
  change pulls in policy decisions about how NaN rows interact with
  the cache key and downstream `build_dataset_simple` (which currently
  has no NaN guard). Filed as a separate follow-on chunk.
- `tools/picker_v06_*.py`, `tools/v0_2_zenjpeg_picker_train.py`,
  `tools/v06_champ_per_class.py`, `tools/v10_router_mlp_train.py` —
  these live under `tools/` (not `zentrain/tools/`) and predate the
  move that promoted `_picker_lib` to its current location. They share
  the same `load_features` patterns but live in a different sub-tree;
  a future consolidation chunk should grep them once `_picker_lib`
  covers whatever shapes they need (most use the simple no-filter or
  `KEEP_FEATURES`-filter strict shape that's already supported).
- `tools/v14_metapicker_train.py`, `tools/v15_metapicker_train.py`,
  `tools/v15_compare_pickers.py` — **MIGRATED** (DEDUP-C, 2026-05-26)
  to a SIBLING shared module `zentrain/tools/_metapicker_lib.py`
  rather than to `_picker_lib.py`. See DEDUP-C addendum below for
  rationale (meta-picker scaffolding is structurally distinct from
  the per-codec picker training scaffolding).

## DEDUP-C addendum — `_metapicker_lib` (Tier-1 #5)

Closes Tier-1 #5 from the verified synthesis (`~/work/zen/zensim/
benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md` line 45): "zenanalyze
metapicker copy-forward forks — v10→v12→v14→v15 standalone copies, no
shared base, 60-80% skeleton each; `classify_stem` byte-identical
v14↔v15".

### What shipped (2026-05-26, commit pending)

| Concern | File | Status |
|---|---|---|
| Shared metapicker scaffolding | `zentrain/tools/_metapicker_lib.py` (NEW, 451 LOC) | **SHIPPED** |
| Unit tests for the shared lib | `zentrain/tools/test_metapicker_lib.py` (NEW, 199 LOC, 8 tests PASS) | **SHIPPED** |
| v14 4-codec trainer | `tools/v14_metapicker_train.py` (430 → 280 LOC, -34.9%) | **MIGRATED** |
| v15 5-codec trainer | `tools/v15_metapicker_train.py` (839 → 582 LOC, -30.6%) | **MIGRATED** |
| v15 comparator | `tools/v15_compare_pickers.py` (357 → 267 LOC, -25.2%) | **MIGRATED** |
| v12 3-codec trainer | `tools/v12_metapicker_train.py` (253 → 268 LOC) | **DEPRECATED** (banner, source kept) |

Aggregate: 1879 → 1397 LOC across the four scripts that existed
pre-extraction (-482 LOC, -25.7%) plus +650 LOC in the new shared
lib + its tests, for a net +168 LOC in tree but ~480 LOC of duplicated
scaffolding eliminated and a regression harness gained.

### What lives in `_metapicker_lib.py`

The 11-symbol API:

- `BANDS_DEFAULT` / `BAND_TOL_DEFAULT` / `SEED_DEFAULT` /
  `HOLDOUT_FRAC_DEFAULT` / `CCLASSES` — constants used by all
  versions.
- `classify_stem(stem)` / `cclass_one_hot(cls)` — content-class
  derivation. `classify_stem` is byte-identical to the v12/v14/v15
  inline impls (proven by the parity tests).
- `load_features(features_tsv, named_feats)` — features TSV loader.
- `load_sweep_tsvs(data_dir)` — sweep TSV loader (includes v15's
  symlink-dir support).
- `build_band_winners(df, bands, band_tol, classes)` — per-(image,
  band) min-bytes-per-codec table.
- `image_disjoint_split(samples, seed, holdout_frac)` — deterministic
  train/holdout split.
- `cell_bytes_for(s, codec)` — bytes-with-worst-case-fallback.
- `bytes_delta_vs_baseline(hold, pred, baseline)` — bytes-Δ
  reporting.
- `format_per_class_report` / `format_per_class_winner_distribution`
  / `format_per_codec_accuracy` — per-class report lines.
- `write_metapicker_json(...)` — ZNPR-bakeable JSON writer.
- `forward_metapicker(model, X)` — numpy forward-pass for a baked
  metapicker JSON (used by `v15_compare_pickers.py`). Subsumes
  the `tools/v15_compare_pickers.py` entry from the deferred
  "numpy forward-pass duplication" table above. The remaining
  4 forward() impls (zerobias_rebake, inspect_picker, holdout_ab_
  lookup, student_permutation) are still candidates for a future
  `_predict_lib.py` chunk.

### Why a sibling lib, not an extension of `_picker_lib.py`

The two concerns are structurally distinct:

- `_picker_lib.py` (867 LOC, per-codec PICKER training) is a HistGB
  teacher → distilled-MLP pipeline that operates on per-codec Pareto
  TSVs. Its dataset shape is (image, knob_tuple) → bytes/quality
  regression.
- `_metapicker_lib.py` (451 LOC, cross-codec META-PICKER training)
  is a classifier-over-codecs that operates on cross-codec sweep
  TSVs. Its dataset shape is (image, target_zensim_band) → codec
  argmin.

Wiring meta-picker concerns into `_picker_lib` would have meant
either (a) growing the older module with code that has no per-codec
picker use, or (b) coupling the picker and meta-picker test surfaces
in a way that makes future per-codec picker work depend on changes
to meta-picker code. Keeping them siblings preserves both modules'
right-sized API surfaces.

### v12 specifically

v12 (3-codec) was NOT migrated structurally. It was DEPRECATED with
a startup warning banner pointing callers at v14 (4-codec
superseding) — but the source is preserved for audit. Reasoning:

- v12 depends on `/tmp/v12-sweep-data` which is typically absent.
- v12's two-features-TSV column-intersection logic is sufficiently
  different from v14/v15's single-TSV path that the wrapper would
  be larger than the saved LOC.
- v12 has never been re-run after v14 superseded it (per the
  v15_compare_pickers.py headline table — v12's output isn't even
  listed; v14 is the 4-codec champion).

The deprecation banner is the minimum-viable signal to future agents
that v12 isn't on the live training path. The source survives for
auditability + as a template if anyone needs to revive a 3-codec
trainer.

### Migration evidence

Each migrated script ships a `<script>_migration_evidence.txt`
adjacent to it (mirroring the DEDUP-B convention). The evidence
files document the 9-point structural equivalence (per-helper
inline-vs-lib comparison) AND why no live before/after diff was
captured (sweep data isn't currently staged in /tmp). Live A/B
regen instructions are included in each evidence file.

The 8 unit tests in `zentrain/tools/test_metapicker_lib.py` are
the live regression gate: they include byte-identical reference
impls of v12's, v14's, and v15's pre-migration `classify_stem` and
`forward()` and assert equality on 26 probe stems + 30 random MLP
inputs.

### Follow-on candidates beyond DEDUP-B2/B3

After completing DEDUP-C, the highest-EV remaining items are:

1. **`_predict_lib.py`** (Tier-1 #6 second half — see deferred
   table above). The 4 remaining inference-side scripts
   (zerobias_rebake, inspect_picker, holdout_ab_lookup,
   student_permutation) each reimplement standardize-then-MLP.
   `_metapicker_lib.forward_metapicker` shows the shape; could
   either promote that to a top-level `_predict_lib.py` or have
   `_predict_lib` import from `_metapicker_lib`. The cleanest design
   is to extract the forward-pass to `_predict_lib`, have
   `_metapicker_lib` re-export it (so existing v15_compare_pickers
   import stays valid), and migrate the other 4 scripts to
   `_predict_lib` directly.

2. **Tier-1 #4** (zensim recipe-driver forks). 21 scripts across 3
   fork families with measured 50-80% overlap. Different repo
   (zensim), but the methodology proven by DEDUP-B/C maps directly.

3. **Tier-2 #9** (CodecFamily enum order — `coefficient/src/
   constraints.rs:30` diverges from `coefficient/src/oracle_picker
   .rs:99`, single-mislabel risk). Smallest fix, highest correctness
   value of the remaining items.

Tier-1 #1 (zensim score_row inline rerolls) and #2 (spearman/pearson
recomputed 7×) are in the zensim repo and would each be a single
chunk of similar scope to DEDUP-B/C.
