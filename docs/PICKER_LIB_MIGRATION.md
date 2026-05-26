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
  `tools/v06_champ_per_class.py`, `tools/v10_router_mlp_train.py`,
  `tools/v14_metapicker_train.py`, `tools/v15_*.py` — these live
  under `tools/` (not `zentrain/tools/`) and predate the move that
  promoted `_picker_lib` to its current location. They share the same
  `load_features` patterns but live in a different sub-tree; a future
  consolidation chunk should grep them once `_picker_lib` covers
  whatever shapes they need (most use the simple no-filter or
  `KEEP_FEATURES`-filter strict shape that's already supported).
