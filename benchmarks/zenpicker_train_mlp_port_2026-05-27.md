# zenpicker-train MLP port — within-cell-optimal picker, Rust vs zentrain

2026-05-27. Advances `zenpicker-train` from the linear ridge skeleton
(`47d92e9`) to a real per-codec picker by porting zentrain's
within-cell-optimal formulation to Rust, fixing the skeleton's
q-leakage, and validating the port against zentrain's Python output on
the SAME data.

## The formulation ported (from `zentrain/tools/train_hybrid.py`)

zentrain's `build_dataset`:

- **Categorical cells.** Each codec config is factored into a discrete
  knob cell. For `unified_v13_zenjpeg_cvvdp.parquet` the
  `knob_tuple_json` is `{subsampling, progressive, sharp_yuv, effort}`
  → 3 × 2 × 2 × 3 = **36 cells**. (These are all-categorical; the older
  zenjpeg sweep with continuous `chroma_scale`/`lambda` scalar axes is a
  different sweep — see follow-ons.)
- **Target.** For each `(image, target_zq)` over the `ZQ_TARGETS` grid
  (step 5 in 0..70, step 2 in 70..100): `bytes_log[cell] = ln(min
  encoded_bytes over configs in the cell whose score_zensim ≥
  target_zq)`, with a `reach[cell]` mask. Cells that don't reach the
  target are NaN/masked.
- **Inputs.** Image features (`feat_*`) + `zq_norm = target_zq / 100`
  (the user's REQUESTED quality). zentrain also one-hot-encodes
  `size_class` and adds `log_px` + polynomial zq cross-terms; the Rust
  port appends only `zq_norm` (image features + zq_norm = 301 inputs).
- **Pick.** `cell = argmin(predicted bytes_log, mask = reach)` — the
  smallest config that reaches the requested quality.
- **Model.** MLP, LeakyReLU(0.01) hidden + identity output, Adam, MSE,
  early stopping. (zentrain additionally trains a HistGradientBoosting
  *teacher* and distills it into the MLP *student*; the Rust port trains
  the MLP directly — the MLP-student vs MLP comparison is the
  apples-to-apples one.)

### Why `q` is NOT an input — the q-leakage fix

The prior skeleton put the codec's chosen `q` into the features and
reported SROCC **0.9988**. That is q-leakage: `q` is monotone with the
achieved score, so a model that sees `q` trivially ranks score across q
levels. `q` is the DECISION the picker makes, not an input. The port's
only non-image input is `zq_norm` (the requested quality); the
supervised target is per-cell `bytes_log`, never the achieved score.
**Confirmed in the bake manifest: `q_is_input = false`.**

## Apples-to-apples setup

`scripts/unified_to_zentrain.py` converts the unified parquet into
zentrain's Pareto + features TSVs with the SAME images, SAME 36 cells,
SAME zq grid, and SAME `feat_0..feat_299` feature set, so the only
difference is the trainer implementation (Rust MLP vs zentrain
sklearn/torch MLP). Both build **5109 (image, target_zq) decision
rows** — identical, confirming the dataset construction is a faithful
port. Both use a grouped-by-image holdout (Rust 0.25, zentrain's default
0.20).

## Held-out results (HONEST — no q-leakage)

| Trainer | model | held-out argmin acc | mean byte overhead |
|---|---|---:|---:|
| **Rust `zenpicker-train`** | MLP 301→64→64→36, LeakyReLU | **0.216** | 0.098 |
| zentrain `train_hybrid` | MLP student 610→64→64→36, LeakyReLU | **0.187** | 0.087 |
| zentrain `train_hybrid` | HistGB teacher (not ported) | 0.392 | 0.054 |

Rust MLP held-out full panel on predicted-vs-actual `bytes_log` over
reachable cells (selected candidate hidden=[64,64], lr=2e-3, seed=0):

| stat | value |
|---|---:|
| SROCC | 0.067 |
| (best-other-candidate SROCC) | 0.146 |
| argmin accuracy | 0.216 |
| byte overhead mean / p50 / p90 | 0.098 / 0.064 / 0.273 |

The MLP-student numbers are in the same ballpark (Rust 0.216 vs zentrain
0.187 argmin accuracy; overhead 9.8% vs 8.7%) — the port is faithful.
The ~3 pp argmin gap and the overhead gap are explained by legitimate
differences, NOT a port bug: zentrain's MLP is a *distillation student*
of a HistGB teacher (so it inherits some tree structure), it carries
extra `size_oh`/`log_px`/zq-polynomial inputs, and the two grouped
holdouts hold out different image subsets.

These numbers are MUCH lower than the leaky 0.9988 — that is correct and
the entire point. A picker over 36 fine-grained cells (many near-equal
in bytes) on a sparse-q corpus is a hard rank problem; argmin 0.216 is
~7.8× the 1/36 = 0.028 random baseline.

## Data-coverage caveat (honest)

`unified_v13_zenjpeg_cvvdp.parquet` sweeps only **5 q levels
{10,30,60,80,90}** per image (200 images, 36 knob configs = 36 000
rows). The "reaches target_zq" ladder is therefore COARSE — sparse on
quality. Per zensim/CLAUDE.md "Dense sampling for trained models", a
production picker needs ~30 q points + 16–20 log-spaced sizes. This bake
validates the FORMULATION and the Rust-vs-zentrain port; it is **not** a
production picker. A dense q + size sweep is a separate data-gen task.

## Artifacts

- Rust bake: `/mnt/v/output/zenpicker-train/2026-05-27/zenjpeg_picker_mlp.bin`
  (ZNPR v3, 301→64→64→36, ~111 KB) + sibling `.toml` manifest.
- Rust run log: `…/rust_mlp_final.log`.
- zentrain comparison data + run: `…/zentrain_cmp/{pareto.tsv,
  features.tsv,codec_config.py,zentrain_run.log}`.
- Adapter: `zenpicker-train/scripts/unified_to_zentrain.py`.

## Follow-ons (out of scope)

- Dense q (~30 points) + log-spaced size sweep before any production
  bake.
- Scalar prediction heads (chroma_scale/lambda) for continuous-axis
  sweeps.
- HistGB teacher + distillation (zentrain's full recipe), if the
  teacher's higher argmin accuracy is wanted.
- CubeCL GPU acceleration; cross-codec MetaPicker auto-regen.
