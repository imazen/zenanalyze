# Picker v0.3 — cross-codec summary, 2026-05-04

Trained, baked, and uploaded v0.3 hybrid-heads pickers for zenwebp,
zenavif, and zenjxl. Per-codec details in
`picker_v0.3_<codec>_2026-05-04.md`.

## Decision: train against existing v0.1-schema sweeps

The 2026-05-04 zen-metrics sweep TSVs (documented in
`picker_v0.3_blockers_2026-05-04.md`) had three structural blockers:
TSV schema mismatch with the trainer, missing features for ~57% of
the new sweep corpus, and a degenerate 2-cell binary-knob grid that
doesn't match any v0.1 runtime cell taxonomy.

After investigation, the schema-compatible v0.1 sweeps + features
already in the repo for all 3 codecs were used — they have the
correct `(image_path, size_class, width, height, config_id,
config_name, bytes, zensim)` shape that `train_hybrid.py::load_pareto`
expects, the 6/10/16-cell taxonomies match each codec's existing
runtime, and they cover real production knobs rather than the
narrow {method:[4,6]} / {speed:[6,8]} / {effort:[3,7]} binary grids
of the new sweep.

A re-sweep with richer grids matching the runtime cell taxonomies
(zenwebp `method∈{4,5,6} × segments∈{1,4} × scalar variation`,
zenavif `speed∈{1..10} × qm × vaq × tune`, zenjxl `cell_id×ac×ec×g×p`)
remains the right next step but is a multi-hour compute arc that
exceeded this session's budget.

## Cross-codec metrics table

| Codec | n_imgs | cells | scalars | Student argmin_acc | Student mean overhead | Train→val gap |
|-------|---:|---:|---:|---:|---:|---:|
| zenwebp | 514 | 6 | 3 | **58.7%** | **1.39%** | +0.13pp |
| zenavif | 200 | 10 | 0 | 23.3% | 5.60% | +2.08pp |
| zenjxl  | 100 | 16 | 3 | 11.2% | 4.57% | +0.81pp |

zenwebp v0.3 is the strongest of the three — argmin acc up from
v0.2's 16.6%, scalar RMSE matches teacher exactly. zenavif and
zenjxl are data-starved at the cell level (200 / 100 imgs across
10 / 16 cells respectively); their .bins are shipped as archival
artifacts, not promoted to in-runtime status.

## Schema-hash drift vs in-runtime spec.rs

zenwebp's v0.3 .bin schema_hash (`0x139d73665fb030c7`) does not
match `spec.rs::SCHEMA_HASH` (`0xb2aca28a2d7a34ec`). Cause:
`bake_picker.derive_extra_axes` emits the canonical 5+N+1
engineered-axis layout the *trainer* uses, but the existing v0.1
.bin (shipped 2026-04-30) was baked from a trainer revision that
emitted a smaller curated extra_axes list (24 axes, see
`zenwebp/src/encoder/picker/zenwebp_picker_v0.manifest.json`).

zenavif and zenjxl don't have an existing in-runtime .bin to drop
into (their picker plumbing is in zenanalyze itself), so the
schema_hash drift is moot for those.

**Implication for zenwebp:** This v0.3 .bin is shipped as an
external artifact but is not drop-in into the current
`pick_tuning` runtime. Future arc to land it in-runtime: either
update `spec.rs::SCHEMA_HASH` to match the new bake, or add a
codec-supplied `EXTRA_AXES = [...]` override to the picker config
matching the v0.1 manifest's 24-axis layout.

## Held-out re-encode A/B — DEFERRED

The brief asked for a held-out cid22-val re-encode A/B comparing
picker-predicted knobs vs codec-default bucket-table knobs. Not
done in this session. Doing so requires:

1. For each codec, a Rust binary that:
   - Loads the v0.3 .bin via zenpredict
   - Runs zenanalyze feature extraction against held-out images
   - Calls picker.predict → cell argmin + scalar heads
   - Maps cell + scalars → encoder knobs
   - Encodes with picker knobs AND with bucket-table defaults
   - Scores both with zensim
2. zenwebp specifically would need the schema_hash bump (or runtime
   relaxation to load mismatched .bins) before its v0.3 .bin could
   feed `pick_tuning`.

The trainer's internal 20% val split is used as a proxy held-out
in the per-codec reports above.

## R2 layout

```
s3://zentrain/zenwebp/pickers/
  zenwebp_picker_v0.3_2026-05-04.bin
  zenwebp_picker_v0.3_2026-05-04.manifest.json
  zenwebp_hybrid_v0.3_v01schema_train.json
  zenwebp_hybrid_v0.3_v01schema_train.log

s3://zentrain/zenavif/pickers/
  zenavif_picker_v0.3_2026-05-04.bin
  zenavif_picker_v0.3_2026-05-04.manifest.json
  zenavif_hybrid_v0.3_train.json

s3://zentrain/zenjxl/pickers/
  zenjxl_picker_v0.3_2026-05-04.bin
  zenjxl_picker_v0.3_2026-05-04.manifest.json
  zenjxl_hybrid_v0.3_train.json
```

## What's still pending

1. Re-sweep with richer grids on the full mlp-tune-fast corpus (587
   imgs across 7 sub-corpora) using the `zen-metrics sweep`
   subcommand at `~/work/turbo-metrics/target/release/zen-metrics`.
   Expected wall: ~9 hr at 8 parallel jobs across 3 codecs.
2. Held-out re-encode A/B vs cid22-val for all 3 codecs (per-codec
   Rust harness binaries).
3. Schema-hash reconciliation for zenwebp drop-in (either runtime
   bump or trainer extra_axes override).
4. Address `effective_max_zensim` upstream
   (`imazen/zenanalyze#51`) so the safety report can distinguish
   physical-unreachable cells from sweep gaps and the bake pipeline
   doesn't need `--allow-unsafe`.

— Lilith River, 2026-05-04
