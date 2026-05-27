# zenpicker-train

Per-codec quality picker trainer for the zen codec stack. Ports
zentrain's established within-cell-optimal picker formulation
(`zentrain/tools/train_hybrid.py`) to Rust, trains a LeakyReLU MLP,
and emits a **ZNPR v3** bake loadable by `zenpredict::Model` /
`zenpicker::MetaPicker`.

## What the picker predicts (and why `q` is NOT an input)

A picker's job is to choose codec encode parameters that hit a
**requested** quality at minimum size. So the inputs are IMAGE features
plus the *target* quality the user asks for — never the codec's chosen
`q`. `q` is the decision the picker makes, and it is monotone with the
achieved score, so feeding it in trivially inflates any "predict the
score" task. (A prior skeleton put `q` in the features and reported
SROCC 0.9988 — that was q-leakage, not signal.)

Following zentrain's `build_dataset`, this trainer:

1. Factors each codec config into a **categorical cell** — the discrete
   knob combination from `knob_tuple_json` (for zenjpeg:
   `subsampling | progressive | sharp_yuv | effort`).
2. For each `(image, target_zq)` over the `ZQ_TARGETS` grid (step 5 in
   0..70, step 2 in 70..100 — the **requested** quality, an INPUT),
   computes the within-cell optimal:
   `bytes_log[cell] = ln(min encoded_bytes over configs in the cell
   whose score_zensim ≥ target_zq)`, with a `reach[cell]` mask.
3. Trains an MLP `(image feat_*, zq_norm) → bytes_log[0..n_cells]`.
4. At inference the codec picks `cell = argmin(predicted bytes_log,
   mask = reach)` — the smallest config that reaches the requested
   quality.

The supervised target is per-cell `bytes_log`, not the achieved score.

## Model + search

- **MLP** (default mode): LeakyReLU(0.01) hidden layers, identity
  output, Adam, MSE, internal early stopping — matching zentrain's
  student topology (`--hidden`, default `128,128`). NaN targets
  (unreachable cells) are masked out of the loss.
- **Bounded grid search** over `{hidden topology} × {learning rate} ×
  {seed}` (6 candidates), ranked by held-out **argmin accuracy** (the
  picker decision-quality metric zentrain gates on via
  `min_argmin_acc`). `cmaes` is not a zenanalyze workspace dependency,
  so the grid is the bounded search the spec offers as the alternative.
- **`--mode ridge`**: a legacy single-layer linear baseline
  (`feat_* [+q] → --target-column`). NOTE: `ridge --include-q` IS
  q-leakage by construction — it exists only for the legacy baseline
  comparison; the MLP mode never sees `q`.

## Held-out evaluation (honest)

A **grouped-by-image** split (≥20% of *images*, no image in both
sides). The report covers:

- the full **`zenstats`** Mohammadi-2025 panel (SROCC / PLCC / KROCC /
  OR / PWRC / Z-RMSE) of predicted-vs-actual `bytes_log` over reachable
  cells, and
- zentrain's headline picker metrics: **argmin accuracy** + byte
  overhead (mean / p50 / p90).

Once `q` is out of the features the numbers are MUCH lower than the
leaky 0.9988 — that is correct and the point. A picker over 36
fine-grained cells with sparse-q data is a hard rank problem.

## Usage

```sh
# Real MLP picker (default mode):
zenpicker-train \
    --input  unified_v13_zenjpeg_cvvdp.parquet \
    --codec  zenjpeg \
    --out    zenjpeg_picker_mlp.bin \
    --val-frac 0.25

# Single explicit topology (disables the search):
zenpicker-train --input ... --codec zenjpeg --out ... --hidden 128,128 --seed 0

# Legacy ridge baseline:
zenpicker-train --mode ridge --input ... --codec zenjpeg \
    --target-column score_zensim --out ridge.bin

# zentrain teacher → student distillation (per-cell HistGB teacher's
# DENSE soft bytes_log → pure-Rust MLP student via soft-target MSE):
zenpicker-train --input ... --codec zenjpeg --out distilled.bin \
    --val-frac 0.25 --distill
#   --soft-weight 0.5   # optional soft↔hard blend (1.0 = pure soft = zentrain)
#   --export-dataset E  # just export the teacher dataset parquet and exit
#   --soft-targets S    # distill against a pre-computed soft-target sidecar
```

A TOML recipe can supply defaults (`--manifest recipe.toml`); CLI
flags override.

## Rust-vs-zentrain comparison

`scripts/unified_to_zentrain.py` converts a unified parquet into
zentrain's Pareto + features TSVs (same images, same cells, same zq
grid, same `feat_0..feat_N` feature set), so
`zentrain/tools/train_hybrid.py --codec-config codec_config` can be run
on the SAME data — isolating the trainer implementation from feature /
cell / grid differences. The held-out argmin accuracy + byte overhead
of the two should be in the same ballpark (the port is faithful).

## Teacher → student distillation (`--distill`)

Faithfully replicates zentrain's full recipe: a per-cell
`HistGradientBoostingRegressor` teacher (`max_iter=400, max_depth=8,
lr=0.05, l2=0.5`; `< 50` reaching-row cells fall back to the per-cell
nanmean) produces DENSE per-`(row, cell)` soft `bytes_log`, and the
pure-Rust MLP student distills against them via **pure soft-target
MSE** (no hard blend / temperature / sample weighting — zentrain's
default). Held-out eval is `argmin(prediction, mask=reach)` vs the HARD
oracle. The teacher is a one-time OFFLINE step
(`scripts/teacher_soft_targets.py`, sklearn) — the Rust runtime gains
**no Python dependency**; `--distill` shells to it once, then distills
in Rust. Inputs are still image features + `zq_norm` only (no
q-leakage; manifest `q_is_input = false`).

**Honest result on this corpus:** distillation does **not** close the
gap. HistGB teacher held-out argmin **0.476**; distilled MLP student
**0.161** — worse than the direct hard-target MLP **0.216**; the
measured soft↔hard blends (α=1.0, 0.7) are below baseline too. This
reproduces zentrain's own behavior
(distilled student 0.187 ≪ teacher 0.392): the teacher's argmin
advantage is a tree-structure property the smooth MLP can't absorb on
a 5-q-level corpus. The gap is **data-limited**, not
implementation-limited. Ship the direct hard-target MLP; closing the
gap needs the dense q + size sweep (a separate data-gen task). Full
numbers + verdict: `benchmarks/zenpicker_train_distill_2026-05-27.md`.

## Data-coverage caveat

The available `unified_v13_zenjpeg_cvvdp.parquet` sweeps only **5 q
levels {10,30,60,80,90}** per image, so the "reaches target_zq" ladder
is COARSE — sparse on quality. Per zensim/CLAUDE.md "Dense sampling for
trained models", a production picker needs ~30 q points + 16–20
log-spaced sizes. This crate validates the **formulation** and the
Rust-vs-zentrain port; it is **not** a production picker.

## Bake shape

`n_in → h0 → … → n_cells` LeakyReLU layers (identity on the output),
F32 ZNPR v3, with the input standardizer folded into the bake's
`scaler_mean` / `scaler_scale`. The bake is loadable by
`zenpredict::Model::from_bytes` and by `zenpicker::MetaPicker::new`
(which wraps the same `Predictor`).

> This is a per-codec picker bake, not a 6-family categorical
> meta-picker — it does not emit `zenpicker.family_order`, so
> `MetaPicker::validate_family_order` will (correctly) reject it as a
> family picker. `MetaPicker` is used here only as the runtime wrapper
> to confirm load + forward pass.

## Follow-ons (out of scope for this chunk)

- **Dense q (~30 points) + log-spaced size sweep** before any
  production bake (zensim/CLAUDE.md training-data discipline). The
  available parquet is sparse on q — a dense sweep is a separate
  data-gen task.
- **Scalar prediction heads** (chroma_scale / lambda) for sweeps that
  carry continuous Pareto axes — this parquet's knobs are all
  categorical, so the bytes head IS the whole picker here.
- **CubeCL GPU acceleration** of the inner MLP training loop.
- **Cross-codec `MetaPicker` auto-regeneration** when a per-codec bake
  updates.
- **Per-band panel gate** + ship/no-ship verdict integration.
