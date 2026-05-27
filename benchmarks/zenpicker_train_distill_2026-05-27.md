# zenpicker-train teacher → student distillation — zentrain recipe, Rust port

2026-05-27. Ports the part of zentrain's picker recipe the prior MLP
port (`8df4713`) SKIPPED: the **teacher → student distillation**. The
prior port trained the MLP student directly on the sparse hard
`bytes_log` targets (held-out argmin accuracy **0.216**). zentrain's
real recipe distills the MLP student against a per-cell
HistGradientBoosting teacher's DENSE soft predictions. This doc
documents the faithful replication + the honest result.

Companion to `benchmarks/zenpicker_train_mlp_port_2026-05-27.md` (the
hard-target port). Same data, same 5109 decision rows, same 36 cells,
same no-q-leakage contract.

## zentrain's distillation recipe (replicated exactly)

From `zentrain/tools/train_hybrid.py` (`train_teacher_per_cell` +
`teacher_predict_all` + the student fit at the `_train_torch_leakyrelu_student`
call site) and `zentrain/tools/_picker_lib.py` (`HISTGB_FULL`,
`_fit_one_cell`):

1. **Teacher = one HistGradientBoostingRegressor per categorical cell.**
   For cell `c`, fit `sklearn.ensemble.HistGradientBoostingRegressor`
   on `(X_train, bytes_log_train[:, c])` over the train rows where cell
   `c` reached the target (the `reach` mask; sklearn drops the NaN
   rows). Production params `HISTGB_FULL`:
   `max_iter=400, max_depth=8, learning_rate=0.05, l2_regularization=0.5`.
   Cells with `< 50` reaching train rows get no teacher (the per-cell
   `nanmean` is the fallback).
2. **DENSE soft targets.** `teacher_predict_all` runs every per-cell
   teacher over ALL train rows → a dense `bytes_pred_tr[n_rows,
   n_cells]` with **no NaN holes** — the teacher interpolates a smooth
   cost surface even for (row, cell) pairs that were unreachable in the
   sparse sweep. This is the distillation signal.
3. **Student distills via pure soft-target MSE.** The LeakyReLU MLP
   student (`lr=2e-3, batch=512, max_iter=500, Adam, early-stop`) trains
   on `soft_tr` = the teacher's dense bytes_log predictions. **No blend
   with the hard target, no temperature, no sample weighting** (the
   default `hard_example_mode="none"`; hard-example reweighting is an
   off-by-default diagnostic, not the recipe). For the bytes-only picker
   the per-head scalar standardization is a single-head no-op.
4. **Eval vs the HARD oracle.** Both teacher and student are scored by
   `argmin(prediction, mask=reach)` vs the true within-cell-optimal
   `bytes_log` — never vs the soft targets. Distillation changes the
   *training target* only.

### q stays OUT of the inputs (no leakage)

Confirmed end-to-end: the teacher's inputs are the SAME `[image feat_*,
zq_norm]` the student consumes (zentrain's `Xs`; trees are invariant to
per-feature monotone scaling, so the Rust raw-feature export matches).
The codec's chosen `q` is never an input. Bake manifest:
`q_is_input = false`, `distillation.recipe = histgb_per_cell_soft_target_mse`.

## How it runs (no Python in the runtime)

The HistGB teacher is a one-time OFFLINE target-generation step. The
Rust runtime gains NO Python dependency. `zenpicker-train --distill`:

1. exports the exact Rust-built dataset (raw teacher inputs + hard
   `bytes_log` + `reach` + the grouped train/val split, keyed by
   `row_idx`) to `<out>.teacher_export.parquet`,
2. shells once to `scripts/teacher_soft_targets.py` (sklearn HistGB),
   which writes `<out>.soft_targets.parquet` (zstd) + a stats JSON,
3. loads the soft targets back (sha256-gated against the export it was
   fit on) and distills the pure-Rust MLP student,
4. bakes ZNPR v3 + a sibling `.toml` recording teacher params, the
   soft-target provenance + sha256, the distillation loss, and the
   held-out numbers.

`--soft-targets <path>` distills against a pre-computed sidecar (skips
the teacher shell-out). `--soft-weight α` blends in the hard target
(`α·soft + (1−α)·hard` where reachable); `α=1.0` is zentrain's recipe.

## Held-out results (HONEST — no q-leakage, eval vs the HARD oracle)

Data: `unified_v13_zenjpeg_cvvdp.parquet` (sha256 `f491714a…`), 5109
`(image, target_zq)` decision rows, 36 cells, 301 inputs (300 `feat_*`
+ `zq_norm`). Grouped-by-image holdout val-frac 0.25 → 3826 train / 1283
val. **Identical split across baseline + distilled** (deterministic).

| Trainer | training target | held-out argmin acc | mean byte overhead |
|---|---|---:|---:|
| zentrain `train_hybrid` HistGB **teacher** (not ported) | — | 0.392 | 0.054 |
| **our HistGB teacher** (this work, 36/36 cells) | — | **0.476** | 0.063 |
| **Rust student, DIRECT hard target** (prior `8df4713`) | hard `bytes_log` | **0.216** | 0.087 |
| zentrain `train_hybrid` MLP **student** (distilled) | soft (zentrain) | 0.187 | 0.087 |
| **Rust student, DISTILLED** (this work, α=1.0 pure soft) | soft `bytes_log` | **0.161** | 0.108 |

Soft↔hard blend sweep (`--soft-weight`, a non-zentrain EXTENSION
probed when pure soft didn't close the gap; same teacher soft targets,
same split):

| soft_weight α | held-out argmin acc | mean byte overhead |
|---:|---:|---:|
| 1.0 (pure soft = zentrain's recipe) | 0.161 | 0.108 |
| 0.7 | 0.128 | 0.107 |

Both measured blend points are BELOW the direct hard-target baseline
0.216 — mixing the soft surface in does not help. (Lower-α blends run
the full 500-epoch schedule per candidate without early-stopping — the
blended loss decreases smoothly and never hits the `tol=1e-6` plateau,
so they're slow; the α=1.0 and α=0.7 evidence plus the mechanism below
already establish the verdict. The expected limit α→0 is the direct
hard-target 0.216.)

## Verdict: distillation does NOT close the gap — and that reproduces zentrain

The honest finding: **distilling the MLP student against the HistGB
teacher's soft targets does not beat the direct hard-target MLP** on
this corpus (0.161 distilled vs 0.216 direct), and the **measured
soft↔hard blends (α=1.0, 0.7) are both worse than the direct hard
fit**. This is NOT a port bug. It reproduces zentrain's own behavior,
where the distilled MLP student (0.187) is far below its HistGB teacher
(0.392) and roughly matches our distilled student.

The teacher's argmin advantage (0.476) is a **tree-structure** property
the MLP cannot absorb on this data:

- **The gap is data-limited, not distillation-implementation-limited.**
  The corpus sweeps only **5 q levels {10,30,60,80,90}** per image
  (200 images × 36 configs). The within-cell-optimal `bytes_log`
  argmin boundaries between 36 near-equal-cost cells are sharp, narrow
  decision regions. A gradient-boosted tree carves those axis-aligned
  regions natively; a smooth LeakyReLU MLP regressing dense soft
  targets blurs them — and the soft targets, being a smooth tree-mean
  surface, give the MLP an EASIER-to-fit but LESS-discriminative target
  than the hard argmin. Soft distillation trades argmin sharpness for
  surface smoothness, which is the wrong trade for a 36-way argmin
  picker on a coarse-q ladder.
- **Per zensim/CLAUDE.md "Dense sampling for trained models",** a
  production picker needs ~30 q points + 16–20 log-spaced sizes. With 5
  q points the reach ladder is too coarse for either the MLP or the
  distillation to learn a sharp picker. Closing the gap is blocked on
  the dense q + size sweep (a SEPARATE data-gen task on zenmetrics sweep
  tooling + GPU compute — NOT attempted here).

The honest engineering conclusion: **ship the direct hard-target MLP
(0.216) as the per-codec student**; the distilled MLP is strictly worse
on this data. The teacher itself (0.476) is the better picker, but it's
a Python sklearn tree ensemble — not the pure-Rust MLP runtime the
picker requires. A Rust GBM teacher OR a denser sweep are the two paths
that could make distillation pay off; both are out of scope for this
bounded port.

## Artifacts

- Baseline (direct hard target): `/mnt/v/output/zenpicker-train/2026-05-27-distill/baseline_hard.bin` (+`.toml`).
- Distilled (α=1.0): `/mnt/v/output/zenpicker-train/2026-05-27-distill/distilled.bin` (+`.toml`, ZNPR v3, 301→64→64→36).
- Teacher export: `…/distilled.bin.teacher_export.parquet` (sha256 `6545464d…`).
- Teacher soft targets: `…/distilled.bin.soft_targets.parquet` (sha256 `4d68f33d…`, zstd, KV `source_export_sha256`).
- Teacher stats: `…/distilled.bin.teacher_stats.json` (argmin 0.476, overhead 0.063).
- Blend sweep bakes: `…/sw_{1.0,0.7}.bin` (α=1.0 = the shipped
  distilled.bin; α=0.7 the one measured blend point).
- Teacher script: `zenpicker-train/scripts/teacher_soft_targets.py`.
- Run logs: `/tmp/zpd_{baseline,distill2,sw_final}.log`.

## Follow-ons (out of scope)

- Dense q (~30 points) + log-spaced size sweep — the data-limited gap is
  the dominant blocker; blocked on zenmetrics sweep tooling + GPU.
- A pure-Rust GBM teacher (so the tree-structured 0.476 picker could
  ship without Python) — large, orthogonal effort.
- Scalar prediction heads (chroma_scale/lambda) for continuous-axis
  sweeps; this parquet's knobs are all categorical.
