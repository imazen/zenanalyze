# zenjpeg picker retrain — Profile A + ZNPR v3, exercising zenpredict v3 capabilities

**Date:** 2026-06-01 · **Codec:** zenjpeg · **Target:** zensim Profile A
**Trainer:** `zenanalyze/zenpicker-train` (commit adds input-shaping + eval-bake)

User directive: "all codecs need to be retrained to target profile a and v3
file format and use the best of the training systems" → "Pilot zenjpeg
end-to-end first" → "implement and try everything new in zenpredict" +
"output shaping/spline stuff also exists".

## 0. The formulation (unchanged, correct)

Input = 108 zenanalyze **source** features (per-image, reproducible at encode
time) ⊕ `zq_norm` (target A score / 100). Output = `bytes_log[0..36]` per
categorical cell (subsampling{420,422,444} × progressive × sharp_yuv ×
effort{0,1,2}). Pick = `argmin(pred_bytes_log, mask=reach)`. q is NOT an
input — it's the decision. Target A scores come from rescoring the sweep's
372 pair-features under `ZensimProfile::A` (the `rescore_parquet` tool).

## 1. CRITICAL DATA BUG (found + fixed)

`picker_dense_full_zenjpeg_A_sourcefeat.parquet` had **all 108 `feat_*`
columns = 0.0** for every image — a broken join (empty→0.0 imputation)
between the sweep rows and the source-feature TSV. **The picker was learning
from `zq_norm` alone; image content never entered the model.** All "argmin
0.2168 / overhead 4.97%" numbers below the FIXED row are the zq_norm-only
baseline.

- Source TSV (`zenjpeg_source_features_full.tsv`) had REAL features; only the
  parquet join broke.
- Fix: `rebuild_sourcefeat_parquet.py` re-joins by `<source_stem>@sz<width>`
  (parquet `image_basename` ↔ TSV `source`+`width`), **fails loudly** on any
  unmatched row (no re-impute), writes `*_FIXED.parquet` + `feature_order.txt`.
- Verify: feat_0 (=`feat_variance`) distinct-across-images = 320, range
  18.67–6961 (was all-zero).
- **Detection trick:** a picker that ignores its features still scores
  argmin ≈ 2–3× random off the dial — looks "fine" until you dedup features
  per image and check they're non-constant.
- **Impact of the fix (headline):** held-out bytes-SROCC jumped from **0.34**
  (zq_norm-only) to **0.93** (real features, FIXED cand 0 [64,64]). The
  picker went from "cannot predict the cost surface" to "predicts it
  strongly." Everything below this point that uses the FIXED parquet inherits
  this — the prior bake was a degenerate dial-only model.

## 2. Distillation = best training system (measured)

Teacher = Python sklearn `HistGradientBoostingRegressor` per cell
(`scripts/teacher_soft_targets.py`). **No Rust GBM exists anywhere in
`~/work`** (exhaustive search) — the `TeacherParams` struct is config only;
distill.rs explicitly shells to Python. Student = LeakyReLU MLP distilled on
the teacher's dense soft `bytes_log`.

| picker | argmin_acc | overhead_mean | overhead_p90 |
|---|---|---|---|
| single-fit (no distill, zq-only data) | 0.0588 | 5.95% | 15.1% |
| HistGB teacher (zq-only data) | 0.3007 | 5.11% | — |
| distilled [128,128] (zq-only data) | 0.2168 | 4.97% | 12.5% |
| HistGB teacher (**FIXED real features**) | 0.2486 | **4.39%** | — |
| **distilled student (FIXED, no shaping)** [64,64] | 0.1402 | **3.39%** | 9.9% |

Distillation beats single-fit by ~1pp overhead. Real features cut **student
overhead to 3.39%** (p50 0.72%) — vs 4.97% (zq-only distilled) and 5.95%
(single-fit). Held-out **bytes-SROCC 0.906** (vs 0.34 zq-only). argmin_acc
drops with real features because the picker makes image-specific near-optimal
picks rather than memorising the modal cell — **overhead is the product
metric, not argmin**. This is the shipped FIXED bake.

## 3. Input shaping — per-feature `feature_transforms` (implemented)

`--input-shaping none|auto|yeo` (`src/input_shaping.rs`). `auto` picks, per
feature independently, the transform minimising |skewness| over
{SignedLog1p/Sqrt/Cbrt, WinsorP99, Log1p, WinsorThenLog1p, YeoJohnson}; the
dial column is forced Identity. Applied through
`zenpredict::FeatureTransform::apply_with_params` so train-time shaping is
bit-identical to the runtime's `predict_transformed` (no re-impl drift).
Emits `zentrain.feature_transforms` (+ params) as newline-separated UTF-8
metadata.

**Result (FIXED features): FALSIFIED — do not ship shaping.** `auto` shaped
**84/108** features (mechanism confirmed working) but **hurt bytes-SROCC at
every matched config** (e.g. [64,64] lr.002: 0.93 → 0.84; [128,128] lr.002:
0.92 → 0.80). The selected auto bake reached a lower *overhead* (2.89% vs
3.39%) but on a different selected topology (confounded) and with a noticeably
noisier cost surface (SROCC 0.84) — not a robust win. This reproduces the
V_20 learning: a LeakyReLU MLP already absorbs per-feature non-linearity, and
WinsorP99 clipping discards tail information the picker uses. Shaping is
**implemented + available** (`--input-shaping auto`) but **off by default**;
the shipped FIXED bake is `none`.

## 4. Quantization (`zenpredict repack`) — tried + measured

On the distilled bake (146 KB f32):

| variant | size | vs f32 | argmin | overhead_mean | srocc |
|---|---|---|---|---|---|
| f32 | 146 KB | — | 0.2168 | 4.97% | 0.3424 |
| f16 + zerobias 0.005 + compress | 73 KB | 2.0× | 0.0521 | 4.87% | 0.3424 |
| i8 + zerobias + compress + optimize | 39 KB | 3.76× | 0.0000 | 4.76% | 0.3423 |

**Insight:** quantization is essentially free on the product metric —
overhead stays 4.76–4.97%, srocc identical. But argmin_acc is hyper-sensitive:
the per-cell cost surface is flat, so i8's ~0.04 round-trip Δ flips argmin to
a near-equal neighbour (argmin→0) while bytes barely move. **Judge picker
quant on overhead, not argmin_acc.** f16 is the safe default; i8 if size
dominates and you optimise overhead.

**On the shipped FIXED [64,64] bake (real features), quant is even cleaner —
both f16 AND i8 are overhead-neutral** (the real-feature surface is less
degenerate than the zq-only one, so i8 keeps a sane argmin too):

| FIXED none bake | size | vs f32 | overhead_mean | bytes_srocc | argmin |
|---|---|---|---|---|---|
| f32 | 59,571 | — | 3.39% | 0.9060 | 0.140 |
| **f16** (shipped quant) | 29,870 | 2.0× | 3.35% | 0.9061 | 0.151 |
| i8 + zerobias + compress + optimize | 17,001 | 3.5× | 3.43% | 0.9057 | 0.107 |

Ship `picker_zenjpeg_A_FIXED_none_v3_f16.bin` (29.9 KB; held-out quality
identical to f32). Runtime-path eval (`--eval-bake`) confirms the toml
numbers exactly.

## 5. Capability applicability matrix (the rest of "everything new")

| zenpredict v3 capability | applies to argmin picker? | why |
|---|---|---|
| `feature_transforms` (input shaping) | **YES** — implemented | better MLP conditioning |
| quantization (f16/i8/zerobias/compress/optimize) | **YES** — measured | size; overhead-neutral |
| `feature_bounds` | informational only | **inert in `forward`** (inference.rs: input→standardize→layers, no bound clamp); consumed only by the optional rescue/safety path |
| `output_specs` (bounds/Sigmoid/Exp/Round) | N/A for the decision | needs the `advanced` feature + `predict_with_specs`; the argmin path uses plain `predict` + `argmin_masked_in_range(…, ScoreTransform::Exp)` which already exps |
| `discrete_sets` / `sparse_overrides` | N/A | no scalar/categorical output heads in this bytes-only picker |
| `output_calibration_spline` (PCHIP) | N/A for argmin | a monotone shared output transform does not change `argmin`; it's a zensim **metric-dial** concept (`zentrain.output_calibration_spline`), not a picker one |

So "everything new" for an argmin picker = input shaping + quantization
(+ distillation as the training recipe). The output-shaping / spline /
discrete / sparse families are real zenpredict capabilities but structurally
do not affect an argmin-over-cells decision — exercised on the metric side,
not here.

## 6. Infrastructure added (committed to zenanalyze)

- `input_shaping.rs` — per-feature transform fit + apply + metadata.
- `evaluate_picker_bake` + `--eval-bake <path>` — score any external bake
  (quantized / shaped) on the held-out split via the deployed runtime path.
- `bake_mlp_picker_to_znpr_v3` takes transforms; manifest records shaping.

## 7. Reproducibility (scripts in git, data on R2)

**Code (git, zenanalyze repo):**
- Trainer + input-shaping + eval-bake: commit `f7bb5915`
  (`feat(zenpicker-train): input shaping + eval-bake mode`).
- Data-fix recipe + this doc + pointer:
  `zenpicker-train/recipes/` (committed alongside, see pointer.md).

**Data (R2 — bucket `zentrain`, endpoint
`https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com`, creds
`~/.config/cloudflare/r2-credentials`). Canonical prefix:**
`s3://zentrain/picker-zenjpeg-2026-06-01/`

| R2 path | sha256 (16) | bytes | role |
|---|---|---|---|
| `train/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet` | `9c735461213e5a27` | 2,878,673 | **canonical training input** (real features) |
| `train/picker_dense_full_zenjpeg_A_sourcefeat_BROKEN_allzero.parquet` | `17057014cd00b144` | 2,723,266 | the buggy all-zero-feature input (audit/provenance) |
| `source/zenjpeg_source_features_full.tsv` | `bcacb8d8f040dc93` | 374,796 | 108 source features per (image,size), real values |
| `source/feature_order.txt` | `dc6882fe6168447c` | 3,321 | feat_i → zenanalyze feature name (runtime extraction order) |
| `scripts/rebuild_sourcefeat_parquet.py` | `90e896ceb1d58891` | 4,352 | the join-fix that produced the FIXED parquet |
| `METHODOLOGY_zenpredict_caps_2026-06-01.md` | `043d35796d8c4d1a` | 6,991 | this doc |

Bakes (uploaded after the FIXED search; see `bakes/` prefix +
pointer.md for final sha256s).

**Repro chain (from R2 + git):**
```sh
# 0. creds
set -a; . ~/.config/cloudflare/r2-credentials; set +a
export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
EP=https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com
# 1. fetch the canonical training input
aws s3 cp s3://zentrain/picker-zenjpeg-2026-06-01/train/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet ./FIXED.parquet --endpoint-url $EP
# 2. (optional) rebuild it from source: fetch TSV + BROKEN parquet, run the recipe
#    python3 zenpicker-train/recipes/rebuild_sourcefeat_parquet.py   # paths inside
# 3. build trainer (git commit f7bb5915)
cargo build --release -p zenpicker-train --bin zenpicker-train
# 4. train (distill = teacher->student; optional per-feature input shaping)
./target/release/zenpicker-train --input ./FIXED.parquet --codec zenjpeg \
    --out picker.bin --distill [--input-shaping auto]
# 5. score any bake on the held-out split (runtime path)
./target/release/zenpicker-train --input ./FIXED.parquet --codec zenjpeg --eval-bake picker.bin
# 6. quantize (size; overhead-neutral)
zenpredict repack picker.bin picker_f16.bin --dtype f16 --zerobias 0.005 --compress
```
The teacher step shells to `zenpicker-train/scripts/teacher_soft_targets.py`
(Python sklearn 1.7.2 HistGradientBoosting) — the Rust runtime has no Python
dep. Same parquet + same commit + `--seed 0` reproduces the bake.

## 8. Next

- Final FIXED bake (best distilled config + shaping if it helps) → quant →
  wire into zenjpeg encode path (108-feat extraction in `feature_order.txt`
  order → `predict_transformed` → 36-cell argmin → `EncoderConfig`).
- Replicate recipe to zenwebp / zenavif / zenjxl.
