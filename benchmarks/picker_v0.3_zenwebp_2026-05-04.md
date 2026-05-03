# Picker v0.3 — zenwebp, 2026-05-04

## Summary

Trained a fresh hybrid-heads MLP picker for zenwebp against the existing
v0.1 schema-compatible Pareto sweep + features TSVs (the new 2026-05-04
zen-metrics sweep had a TSV-schema + grid-shape mismatch with the
v0.1 trainer/runtime — see `picker_v0.3_blockers_2026-05-04.md`).
Used `zenwebp_picker_config_v01_schema` (newly added — see
`zentrain/examples/zenwebp_picker_config_v01_schema.py`) which mirrors
`zenwebp/src/encoder/picker/spec.rs::FEAT_COLS` exactly (36 features,
none dropped) and the v0.1 cell taxonomy (6 cells = method×segments,
3 scalar heads = sns_strength, filter_strength, filter_sharpness).

## Training inputs

| Field | Value |
|-------|-------|
| Pareto TSV | `~/work/zen/zenwebp/benchmarks/zenwebp_pareto_2026-04-30_combined.tsv` |
| Pareto rows | 2,730,240 (514 unique images × 1,265 cells) |
| Features TSV | `~/work/zen/zenwebp/benchmarks/zenwebp_pareto_features_2026-04-30_combined.tsv` |
| Feature rows | 1,264 (one per image × size_class) |
| KEEP_FEATURES | 36 (mirror of `spec.rs::FEAT_COLS`) |
| ZQ_TARGETS | `range(30, 70, 5) + range(70, 96, 2)` (21 points) |
| Cells | 6 = method × segments  |
| Scalar heads | sns_strength, filter_strength, filter_sharpness |
| Output dim | 24 (6 cells × (1 bytes_log + 3 scalars)) |

## Architecture

MLP `82 -> 128 -> 128 -> 128 -> 24`, leaky ReLU, 46,744 params.
Engineered input vector follows the v0.1 layout `feats[36] +
size_oh[4] + poly[5] + zq*feats[36] + icc[1] = 82 floats`.

## Held-out training-internal split metrics

The trainer holds out 20% of rows (image-level split) per the global
seed. These are unseen during student fitting.

| Metric | Teacher (boosted-trees) | Student (MLP) |
|---|---:|---:|
| Argmin mean overhead | 2.99% | **1.39%** |
| Argmin accuracy | 36.8% | **58.7%** |
| sns_strength RMSE  | 33.2484 | 33.3823 |
| filter_strength RMSE | 22.7691 | 22.8010 |
| filter_sharpness RMSE | 2.0302 | 2.0391 |
| Train rows | — | 10,642 |
| Val rows | — | 2,718 |
| Train→val gap (overhead) | — | +0.13pp |

Student outperforms teacher on argmin (knowledge-distillation working
as intended). Scalar RMSEs match teacher closely — student preserves
the regression signal.

## Comparison to existing v0.1 / v0.2 baked models

| Bake | argmin_acc (val) | mean overhead | Notes |
|------|---:|---:|------|
| v0.1 (`src/encoder/picker/zenwebp_picker_v0.1.bin`) | — | — | -1.20% bytes vs bucket-table on cid22-val Apr 30 |
| v0.2 (`zenwebp_hybrid_2026-05-01_v0.2.json`)        | 16.6% | 3.55% | 12 cells × 4 scalars, weaker schema |
| **v0.3 (this bake)** | **58.7%** | **1.39%** | 6 cells × 3 scalars, full v0.1 schema match |

v0.3 is a clear improvement over v0.2 on val argmin acc (+42pp) and
mean overhead (-2.16pp). Direct comparison vs v0.1 (-1.20% bytes vs
bucket-table) requires the held-out cid22-val A/B encode — see
"Limitations" below.

## Artifacts

Local:
- `~/work/zen/zenwebp/benchmarks/zenwebp_picker_v0.3_2026-05-04.bin`
  (99,248 bytes f16, n_inputs=82, n_outputs=24, n_layers=4,
  schema_hash=`0x139d73665fb030c7`)
- `~/work/zen/zenwebp/benchmarks/zenwebp_picker_v0.3_2026-05-04.manifest.json`
- `~/work/zen/zenwebp/benchmarks/zenwebp_hybrid_v0.3_v01schema.json` (training JSON)
- `~/work/zen/zenwebp/benchmarks/zenwebp_hybrid_v0.3_v01schema.log`

R2:
- `s3://zentrain/zenwebp/pickers/zenwebp_picker_v0.3_2026-05-04.bin`
- `s3://zentrain/zenwebp/pickers/zenwebp_picker_v0.3_2026-05-04.manifest.json`
- `s3://zentrain/zenwebp/pickers/zenwebp_hybrid_v0.3_v01schema_train.json`
- `s3://zentrain/zenwebp/pickers/zenwebp_hybrid_v0.3_v01schema_train.log`

## Schema-hash drift vs in-runtime spec.rs

- `spec.rs::SCHEMA_HASH = 0xb2aca28a2d7a34ec`
- v0.3 .bin schema_hash = `0x139d73665fb030c7`

The hashes differ even though FEAT_COLS matches. Cause: the trainer's
current bake_picker.derive_extra_axes built-in layout emits names
`size_tiny..size_large + log_pixels + log_pixels_sq + zq_norm +
zq_norm_sq + zq_norm_x_log_pixels + [zq_x_<feat> for feat in FEAT_COLS] +
icc_bytes` (46 axes for n_feat=36) which is the legacy zenjpeg-shape
layout. The v0.1 .bin (shipped 2026-04-30) was baked with a different
trainer revision that used a smaller curated extra_axes list (24
axes; see `zenwebp_picker_v0.manifest.json`).

**Implication:** This v0.3 .bin is not drop-in-loadable into the
current `pick_tuning` runtime — the runtime will reject it with
`PickError::SchemaMismatch`. Two paths to drop-in:

1. (preferred) Update zenwebp's `spec.rs::SCHEMA_HASH` constant to
   `0x139d73665fb030c7` and replace `zenwebp_picker_v0.1.bin` with
   the v0.3 .bin. Out of scope for this session per the brief
   ("DO NOT touch any codec source repo's runtime.rs or in-codec
   picker integration").
2. Patch the trainer / bake_picker to emit the legacy curated
   extra_axes list (`zq_x_feat_screen_content` etc., subset that
   matched the original v0.1 features). Larger trainer change with
   cross-codec impact.

The .bin is fully usable as an external picker artifact (call
`zenpredict::Model::from_bytes` directly with the extracted feature
vector to get cell predictions; map cell → encoder knobs via the
v0.1 cell taxonomy).

## Limitations

- **No cid22-val held-out re-encode A/B.** The brief asked for
  encode-and-compare-bytes against `~/work/zentrain-corpus/mlp-validate/cid22-val/`
  with picker-predicted knobs vs bucket-defaults. That requires
  per-codec encoding harness binaries (`<codec>/dev/picker_ab_eval.rs`
  takes a `--label picker|bucket` flag and rebuilds the codec with
  the picker .bin embedded; the v0.3 .bin is not embeddable due to
  the schema_hash drift documented above). The trainer's internal
  20% val split is used as a proxy held-out.
- **Sweep is the original v0.1 grid (144 configs, 514 imgs).** The
  brief proposed re-sweeping with a richer grid + full mlp-tune-fast
  corpus; not done in this session due to time budget. The current
  v0.3 bake is constrained to the existing sweep's coverage.
- **Safety violations carried (DATA_STARVED_SIZE @ tiny/zq86+;
  UNCAPPED_ZQ_GRID).** Same as v0.1; see imazen/zenanalyze#51.
  Bake used `--allow-unsafe`.

## Provenance

- Trainer: `~/work/zen/zenanalyze/zentrain/tools/train_hybrid.py`
- Codec config: `~/work/zen/zenanalyze/zentrain/examples/zenwebp_picker_config_v01_schema.py`
- Inject specs: `~/work/zen/zenwebp/dev/inject_v3_specs.py`
- Bake: `~/work/zen/zenanalyze/tools/bake_picker.py --dtype f16 --allow-unsafe`
- zenpredict-bake: `~/work/zen/zenanalyze/target/release/zenpredict-bake`
