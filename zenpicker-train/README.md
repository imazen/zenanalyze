# zenpicker-train

Per-codec quality picker trainer for the zen codec stack. Loads a
unified sweep parquet, trains a model that maps image features (plus
the encode-quality knob) to an achieved-quality metric, and emits a
**ZNPR v3** bake loadable by `zenpredict::Model` / `zenpicker`.

This crate is the **bounded first chunk** of `zenpicker-train` as
described in §4 of
`zenmetrics/docs/ZEN_CLOUD_AND_CONSOLIDATION_SPEC_2026-05-26.md`. It is
a working END-TO-END skeleton, **not** the full picker trainer. See
[Follow-ons](#follow-ons) for what is intentionally out of scope.

## What it does (this chunk)

1. Loads a unified sweep parquet — schema
   `(image_path, codec, q, knob_tuple_json, score_*, feat_0..feat_N)`.
2. Filters to one codec (`--codec`), builds `(feat_* [+ q] →
   --target-column)` training rows, dropping rows with non-finite
   features or target.
3. Trains a **deterministic ridge-regularized linear baseline**
   (closed-form normal equations; a per-feature standardizer folded
   into the bake scaler). This is an honest baseline, not SOTA.
4. Emits a ZNPR v3 bake through the **`zenpredict-bake` JSON
   pipeline** (`bake_from_json_str`) — never a hand-rolled wire format
   — plus a sibling `<out>.toml` reproduce-this manifest recording the
   input path + sha256, row counts, feature names, model hyperparams,
   bake sha256, and held-out panel numbers.
5. Evaluates on a **grouped held-out split** (≥20% of *images*, no
   image leaks across the split) via the canonical **`zenstats`**
   Mohammadi 2025 panel (SROCC / PLCC / KROCC / OR / PWRC / Z-RMSE).

## Usage

```sh
zenpicker-train \
    --input  unified_v13_zenjpeg_cvvdp.parquet \
    --codec  zenjpeg \
    --target-column score_zensim \
    --include-q \
    --out    zenjpeg_picker_skeleton.bin
```

A TOML recipe can supply defaults (`--manifest recipe.toml`); CLI
flags override. Recipe keys mirror the flags: `input`, `codec`,
`target_column`, `out`, `lambda`, `val_frac`, `include_q`.

## Bake shape

One identity-activation F32 ZNPR layer mapping `n_features → 1`. The
standardizer lives in the bake's `scaler_mean` / `scaler_scale`; the
linear weights are the layer weights; the intercept is the bias. The
bake is loadable by `zenpredict::Model::from_bytes` and by
`zenpicker::MetaPicker::new` (which wraps the same `Predictor`).

> Note: this is a **per-codec quality regression** bake (1 output),
> not a 6-family categorical meta-picker. It deliberately does NOT
> emit `zenpicker.family_order`, so `MetaPicker::validate_family_order`
> will (correctly) reject it as a family picker — `MetaPicker` is used
> here only as the runtime wrapper to confirm load + forward pass.

## Follow-ons

Documented as TODO so the next session knows the skeleton is a
skeleton:

- **Full hyperparameter search.** Port the scikit-learn-parity
  search (cmaes / grid) the Python `zentrain` pipeline does. This
  chunk ships a deterministic ridge baseline only — no non-linear
  MLP, no search.
- **CubeCL GPU acceleration** of the inner training loop (spec §4).
- **Cross-codec `MetaPicker` auto-regeneration** — regenerate the
  cross-codec family meta-bake whenever a per-codec bake updates
  (spec §4 data-flow).
- **Dense size/quality sampling discipline** per zensim/CLAUDE.md
  (≥16–20 log-spaced sizes, q-dense in the perceptibility band,
  k-means-stratified representative images). The skeleton trains on
  whatever the input parquet contains.
- **Per-band panel gate** and ship/no-ship verdict integration
  (zenstats per-band).
- **TOML manifest as the full reproduce-this recipe** (this chunk
  writes a basic manifest and reads a basic recipe; the §3 trainer's
  full manifest schema — every input file with sha256/row-count/mirror
  URL + post-training steps — is the target).
