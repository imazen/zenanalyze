# Picker v0.3 — zenavif, 2026-05-04

## Summary

Trained a fresh hybrid-heads MLP picker for zenavif (rav1e backend)
against the existing v0.1 schema-compatible Pareto sweep + features
TSVs. Used the existing `zenavif_picker_config` codec module
(unchanged). 10 cells (speed = 1..10), no scalar heads in this
phase-1a config.

## Training inputs

| Field | Value |
|-------|-------|
| Pareto TSV | `~/work/zen/zenavif/benchmarks/rav1e_phase1a_2026-04-30.tsv` |
| Pareto rows | 89,600 (200 unique images) |
| Features TSV | `~/work/zen/zenavif/benchmarks/rav1e_phase1a_features_2026-05-01.tsv` |
| Feature rows | 448 |
| KEEP_FEATURES | 56 (zenavif phase-1a v0.2 schema, post-LOO cull) |
| Cells | 10 = speed ∈ {1..10}  |
| Scalar heads | (none) |
| Output dim | 20 (10 cells × (1 bytes_log + 1 time_log)) |

## Architecture

MLP `114 -> 128 -> 128 -> 20`, ReLU, 33,812 params. Engineered
input vector: `feats[56] + size_oh[4] + poly[5] + zq*feats[56] +
icc[1] = 114 floats`.

## Held-out (training-internal val split) metrics

| Metric | Teacher (boosted-trees) | Student (MLP) |
|---|---:|---:|
| Argmin mean overhead | 3.53% | 5.60% |
| Argmin accuracy | 28.9% | 23.3% |
| Train rows | — | 4,116 |
| Val rows | — | 1,089 |
| Train→val gap (overhead) | — | +2.08pp |

## Safety violations (carried)

- `OVERFIT`: train→val mean gap +2.08pp > 2.00pp threshold
- `LOW_ARGMIN`: val argmin_acc 23.3% < 30.0% threshold
- `DATA_STARVED_SIZE`: 63 (size_class, zq) cells with <50 rows
  (mostly tiny/zq30..55 = 0)
- `UNCAPPED_ZQ_GRID`: ZQ_TARGETS includes zq=94, no
  effective_max_zensim column to discriminate

Bake forced via `--allow-unsafe`. The student is weaker than the
teacher on argmin (23.3% vs 28.9% — distillation degrading rather
than improving). Likely cause: 10-way categorical with only 200
unique images is data-starved; teacher overfits and student can't
recover the signal cleanly.

**Recommendation:** This v0.3 bake is shipped for archival /
external-A/B inspection but **should not** be used as a drop-in
runtime replacement until either (a) the sweep is expanded with
more images per content class, or (b) the cell taxonomy is reduced
to fewer cells.

## Artifacts

Local:
- `~/work/zen/zenavif/benchmarks/zenavif_picker_v0.3_2026-05-04.bin`
  (41,656 bytes i8, n_inputs=114, n_outputs=20, n_layers=3,
  schema_hash=`0x0c2a0033461f537a`)
- `~/work/zen/zenavif/benchmarks/zenavif_picker_v0.3_2026-05-04.manifest.json`
- `~/work/zen/zenavif/benchmarks/rav1e_phase1a_hybrid_2026-05-01_v0.3_2026-05-04.json`

R2:
- `s3://zentrain/zenavif/pickers/zenavif_picker_v0.3_2026-05-04.bin`
- `s3://zentrain/zenavif/pickers/zenavif_picker_v0.3_2026-05-04.manifest.json`
- `s3://zentrain/zenavif/pickers/zenavif_hybrid_v0.3_train.json`

## Limitations

Same as zenwebp: no cid22-val held-out re-encode A/B in this session
(requires per-codec encoding harness). Sweep is phase-1a (200 imgs,
qm=1, vaq=0, strength=1.0, tune=1 fixed) — does not cover the full
zenavif knob surface.

## Provenance

- Trainer: `~/work/zen/zenanalyze/zentrain/tools/train_hybrid.py`
- Codec config: `~/work/zen/zenanalyze/zentrain/examples/zenavif_picker_config.py`
- Bake: `~/work/zen/zenanalyze/tools/bake_picker.py --dtype i8 --allow-unsafe`
