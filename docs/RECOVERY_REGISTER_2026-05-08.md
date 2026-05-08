# zenanalyze recovery register — 2026-05-08

Compiled from a read-only sweep of zenanalyze (zenpredict + zentrain + zenpicker), the `zenanalyze--zenpredict-pre-v3` and `zenanalyze--time-budgeted` worktrees, the 2026-04-30 commit storm, and the v15-5codec-metapicker / feat/dense-percentiles / feat/dispatch-plan / bench/picker-v06-rebake-attempt branches. Cross-ref: `~/work/zen/RECOVERY_PLAN_2026-05-08.md`.

## Verdict table

| crate | branch | commit | date | item | what it adds | verdict | files |
|---|---|---|---|---|---|---|---|
| zenpredict | main | `6b552a5` | 2026-05-06 | **ZNPR v3 format** | `output_specs`, `discrete_sets`, `sparse_overrides`, `feature_transforms`; fluent `BakeRequest` builder; `#[non_exhaustive]` on Header / BakeRequest / BakeError | **kept — unpublished, on main** | `zenpredict/src/{model.rs,bake.rs,output_spec.rs}` |
| zenanalyze | main | `6b552a5` | 2026-05-06 | retired feature IDs | TextLikelihood (27), ScreenContentLikelihood (28), NaturalLikelihood (29), LineArtScore (45), IndexedPaletteWidth (30→PaletteLog2Size) — IDs reserved permanently | kept (already on main) | `src/feature.rs` `RESERVED_RETIRED_IDS` |
| zenanalyze | main | `6b552a5` | 2026-05-06 | HDR features feature-gated | 10 HDR/wide-gamut features (32–39, 46, 47) behind `hdr` cargo feature; `tier_depth` module `cfg(feature="hdr")` | kept | `src/tier_depth.rs`, `feature.rs` |
| zenanalyze | main | `6b552a5` | 2026-05-06 | cross-codec redundancy purge | removed ChromaKurtosis (117), UniformitySmooth (118), FlatColorSmooth (119); collapsed block-misalignment 16/64 → 8/32; removed pixel-count transform 94–100, 104 | kept | `src/feature.rs`, `tier3.rs` |
| zenanalyze | main | `6b552a5` | 2026-05-06 | `dimensions.rs` + `grayscale.rs` | pure-descriptor block_loss math (8/16/32 grids) ~10 ns; strict R==G==B classifier with multi-arch SIMD via archmage | kept | `src/dimensions.rs`, `grayscale.rs` |
| zenanalyze | main | `6b552a5` | 2026-05-06 | dense percentile features (experimental) | IDs 122–211, 90 dense-grid percentile variants for LaplacianVariance / AqMap / NoiseFloor / QuantSurvival; behind `experimental` cargo feature | kept (gated) | `src/percentile_features.rs`, `tier1.rs`, `tier3.rs` |
| zenanalyze | feat/dense-percentiles | `fc1f26a` | 2026-05-07 | QEMU cross-build skips | `Cross.toml`; skip timing/inference tests on `CROSS_RUNTIME` targets; 512×512 instead of 2048×2048 for perf tests | unverified — in-flight | `Cross.toml`, `tests/perf.rs` |
| zenanalyze | feat/dispatch-plan | `dae665d` | 2026-05-07 | dispatch tree (stages 0+1.5) | `analyze_with_dispatch_plan`; `DispatchHints` struct (`#[non_exhaustive]`, empty seats for future stages) | open (PR #54) | `src/dispatch.rs`, `lib.rs` |
| zenanalyze | feat/time-budgeted-objective | `dae665d` | 2026-05-07 | encode-time-budgeted picker objective | tradeoff: encode time in objective alongside RD | open (worktree exists) | `tools/_picker_lib.py` |
| zenanalyze | deps/garb-0.2.8 | `96005e9` | 2026-05-08 (today) | tokenless deinterleave API | bumped garb 0.2.8 — no archmage `token()` at call site | shipped (dep bump) | `Cargo.{toml,lock}` |
| zentrain | main | `6b552a5` | 2026-05-06 | hybrid-heads picker trainer | categorical cells (bytes head) + K scalar regression heads per cell; PyTorch LeakyReLU student with early stopping; feature_transforms metadata key; per-class holdout slicing | kept (basis of v15r/v15rc pickers) | `zentrain/tools/train_hybrid.py` |
| zentrain | v15-5codec-metapicker | `36c717c` | 2026-05-07 | regression+classification hybrid trainer | 14 zenanalyze + 5 cclass + 1 target_zensim → 64×64; subsampling/progressive softmax; aq/auto_optimize sigmoid; scalar regression on chroma_scale/q with clamps | open (branch work, not yet baked) | `tools/v0_2_zenjpeg_picker_train.py` |
| zentrain | v15-5codec-metapicker | `0cba306` | 2026-05-07 | zenjpeg picker v0.1 from full v15r sweep | 1.79M+514K cells; hybrid softmax/sigmoid/regression on chroma_distance_scale + q; honest headline −3.43% vs −8.30% strawman | open (branch) | `benchmarks/zenjpeg_picker_v0.2_2026-05-07.json` |
| zenpicker | main | `6b552a5` | 2026-05-06 | meta-picker v0.4 → v0.5 5-codec | v0.4 jpeg/webp/jxl/avif; v0.5 +png; trained on v12+v13 sweep data | kept (baked artifacts) | `benchmarks/zenpicker_meta_v0.{4,5}.bin` |
| zenpicker | bench/picker-v06-rebake-attempt | `d5f7cba` | 2026-05-06 | per-content-class audit (zenjxl v0.6) | per-class regression: photo −0.45%, lineart 0%, **screen +41.4% (catastrophic)** — MLP overfits to 94% photo majority | verified — recommends class-gated inference | `benchmarks/picker_v06_per_class_audit_2026-05-06.md` |
| zenanalyze | docs/sa-piecewise-v5-finding-2026-05-04 | (same) | 2026-05-04 | SA-piecewise finding | analysis doc for piecewise SA-quant | research note | `docs/...` |
| zenanalyze | feat/student-permutation-ablation | (same) | 2026-05-01 | permutation importance for distilled student | feature-importance ranking for student MLP | research note | `tools/student_permutation.py` |
| zenanalyze | research/i8-quant-impact | `cdb8da4` | 2026-04-29 | i8 quant accuracy study | quant impact analysis | research note | (locked worktree) |

## ZNPR v3 — staged feature inventory (already on zenanalyze main)

These need **no further design** — Phase 4 is finalize doc + minimize API + publish.

- **`output_specs`** (header bytes 72–96): per-output pipeline (activation, clamp, snap-to-discrete, sentinel). 32 B POD, zero-copy.
- **`discrete_sets`**: f32 pool referenced by OutputSpecs. Codec-side bounds checked.
- **`sparse_overrides`**: hand-tuned `(idx, value)` patches applied post-spec; 8 B POD. Per-output sentinel fallback.
- **`feature_transforms`** (metadata key `zentrain.feature_transforms`): per-feature `identity|log|log1p` applied **before** forward pass — closes train/serve skew.
- **`output_value` enum**: `Override(f32)` post-processed value OR `Default` ("use codec's built-in default").

All sections optional → v2 bakes parse identically.

## zenpredict public API minimization (Phase 4 yagni-trim plan)

Live call sites: zensim, zenavif (caller-supplied bake), zenwebp (caller-supplied bake), zenpicker, the `zenpredict-bake` CLI, the `zenpredict-bake-roundtrip-check` CLI.

| Item | Used by | Decision |
|---|---|---|
| `Model::from_bytes`, `Predictor::new`, `Predictor::predict`, `PredictError`, `error::*` | all | **keep public** |
| `argmin::{threshold_mask, argmin_masked, argmin_in_range}`, `AllowedMask`, `ArgminOffsets`, `ScoreTransform` | zenpicker only | keep public — picker depends on it |
| `output_spec::*`, `OutputSpec`, `OutputTransform`, `OutputValue`, `apply_spec` | zenpicker, zensim (potentially) | keep public |
| `rescue::*`, `RescuePolicy`, `RescueStrategy`, `RescueDecision` | zenpicker | keep public |
| `bake::{BakeRequest, BakeError, build_bake}` | zenpredict-bake CLI, training-time tools | **gate behind `bake` cargo feature** (default-on for dev, off for lean runtime) |
| `inference::{LayerKind, forward_f32, forward_f16, forward_i8}`, `f16_bits_to_f32`, `scale_i8_row` | none externally | **demote to `pub(crate)`** |

Recommendation: when zensim depends on zenpredict, use `default-features = false` to drop the bake module. zenavif/zenwebp same.

## Gaps in zentrain that the v06-rebalance Rust trainer had

| feature | v06-rebalance Rust | zentrain Python (main) | gap |
|---|---|---|---|
| FiLM heads (per-image conditioning) | yes | no | **MISSING** — `train_hybrid.py` is plain MLP |
| MoE (mixture of experts per cell) | yes (sparse routing) | no | **MISSING** |
| cclass conditioning (one-hot content class) | yes | yes (on `v15-5codec-metapicker` branch only — not main) | partial — port to main |
| magnitude-matching loss | yes | no (MSE only) | **MISSING** |
| sampler bias (low-band oversample) | yes | no (uniform) | **MISSING** |
| dct_hf zenanalyze-feature appender | yes (3 features) | no (training-time tools don't pull from zenanalyze TSV) | **MISSING** |
| `--also` dataset mixing (multiple corpora w/ per-class weights) | yes | no (single PARETO path) | **MISSING** |
| `--val-policy=min` (mean-over-groups val selection) | yes | no | **MISSING** |

These are exactly what the **Phase 3 zentrain port** (`zenanalyze/zentrain/tools/zensim_metric_train.py`) needs to implement.

## Cherry-picks for main (anti-bloat: each one has documented value)

1. **feat/dispatch-plan (PR #54)** — `analyze_with_dispatch_plan` skeleton + `DispatchHints` struct. Currently empty seats; future-proofs the dispatch architecture without bloat. Merge.
2. **feat/dense-percentiles QEMU skips** — pragmatic fix for cross-build CI; isolated to `Cross.toml` + perf test guards. Merge.
3. **bench/picker-v06-rebake-attempt audit doc** — `benchmarks/picker_v06_per_class_audit_2026-05-06.md` documents zenjxl v0.6's catastrophic screen regression. Move to main as evidence of the "class-gated fallback" requirement that informs the zentrain port. Doc-only.
4. (Phase 3 deliverable, not cherry-pick) **port the v06-rebalance Rust trainer features into `zentrain/tools/zensim_metric_train.py`** — see "Gaps" table above for the explicit feature list.

## Drop / archive (no measured value or superseded)

- v15-5codec-metapicker **branch work that hasn't beat v0.5 baseline** — keep on branch, don't merge until validated.
- Any prior session's branches with last commit < 2026-04-25 (zenjxl-adapter-2026-05-01 etc) and no associated unmerged measurement — archive.

## Notable design / process docs to preserve

- `zenanalyze/zentrain/PRINCIPLES.md` — cross-codec contract, re-bake triggers, four-dimension sweep discipline.
- `zenanalyze/zentrain/ABLATION.md` — Tier 0–4 ablation pipeline + deprecation gates.
- `zenanalyze/zentrain/INSPECTION.md`, `INVERSION.md`, `SAFETY_PLANE.md`, `FOR_NEW_CODECS.md`.
- `zenanalyze/benchmarks/picker_v06_per_class_audit_2026-05-06.md` (cherry-pick to main per above).
- `zenanalyze/benchmarks/audits-2026-05-02/zentrain_zenanalyze.md`.
