# picker dense q+size FULL sweep — data pointer (2026-05-27)

Full-scale run of the dense q+size zenjpeg picker data pipeline that the
dry-run (`picker_dense_sweep_dryrun_2026-05-27.md`) scoped and verified.
**LOCAL, $0, RTX 5070.** Target = ssim2-gpu (monotone; the shipped zensim
metric is non-monotone on photo content, the v39 defect). zensim-gpu
carried through as `metric_zensim_gpu` for v39 characterization. cvvdp
SKIPPED (slow, not needed for the picker, backfillable from the persisted
content-addressed encodes).

## Grid

- **sources:** K=20 k-means clusters (centroid-nearest) over a 600-image
  stratified embedding of the `_1024sq` source pool (denser than the
  dry-run's 300@K=8). Content classes: chart/screen/doc/line/mixed/photo.
- **sizes:** 16 log-spaced maxdims (32,40,48,64,80,96,128,160,192,256,
  320,384,512,640,768,1024), Lanczos, downscale-only. All 20 sources are
  1024px so all 16 sizes apply → 320 (image,size) units.
- **q:** 29-level dense grid (step-5 in 5..69, step-2 in 70..100).
- **cells:** 36 = subsampling{444,422,420} × progressive{f,t} ×
  sharp_yuv{f,t} × effort{0,1,2}. Matches unified_v13's cell space — so
  argmin accuracy is directly comparable to the 0.216 sparse-q baseline.
- **Encodes:** 320 × 36 × 29 = 334,080 (no upscale prune since uniform
  1024px sources). sz32 smoke: 20,880/20,880 cells, 0 fail.

## Data root (block storage — NOT in git)

`/mnt/v/zen/picker-dense-full-2026-05-27/`

- `parquet/picker_dense_full_zenjpeg.parquet` — picker training parquet
  (image_basename, codec, q, knob_tuple_json, score_zensim [=ssim2 reach
  ladder], encoded_bytes, encoded_sha256, size_class, metric_* {ssim2_gpu,
  zensim_gpu}, feat_0..feat_371).
- `artifacts/<sha256>.jpg` — content-addressed encoded bytes (free dedup).
- `features/feat_sz<N>.parquet` — 372-feat zensim sidecars per size.
- `pareto_sz<N>.tsv` — per-size sweep Pareto (carries both metric scores).
- `score_pairs_ssim2.tsv` — reach-ladder ssim2 target, derived from the
  pareto inline `score_ssim2_gpu` (verified byte-identical to batch path).
- `bake/zenjpeg_picker_dense_full.bin` (+`.toml`) — retrained ZNPR v3 bake.
- `chosen_sources.txt`, `cluster_assignment.json`,
  `cluster_sample_features.tsv`, `embedding_manifest.tsv` — source select.
- `build_embedding_manifest.py`, `resize_sources.py` — run-specific setup.
- `logs/` — all sweep/build/retrain logs.

## Scripts (in git)

- `zenmetrics/scripts/sweep/picker_dense_sweep_full.sh` (155a2259) —
  resumable encode+score+persist+build+retrain driver.
- `zenanalyze/zenpicker-train/scripts/build_picker_parquet.py` (0d96dec3)
  — parallel content-address + hardlink (full-run perf).
- `zenanalyze/zenpicker-train/scripts/cluster_sources.py` — K-means select.

## Backup

- Tower: `/mnt/tower/output/zensim/picker-dense-full-2026-05-27/` (at run end).
- R2: deferrable follow-on (note; not blocking).

## cvvdp backfill (deferred)

To add cvvdp scores later, run `zen-metrics batch --metric cvvdp
--gpu-runtime cuda` over a pairs TSV built from `artifacts/<sha>.jpg` +
their source refs, then join into the parquet on `encoded_sha256`. No
re-encode needed — the bytes are persisted.
