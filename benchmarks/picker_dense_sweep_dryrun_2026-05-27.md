# Dense q+size sweep dry-run — zenjpeg picker data pipeline

2026-05-27. Builds and verifies, at a **tiny local no-spend scale**, the
dense q + size sweep that produces the training data the §4 zenjpeg
picker needs. Two prior §4 experiments
(`benchmarks/zenpicker_train_mlp_port_2026-05-27.md`,
`benchmarks/zenpicker_train_distill_2026-05-27.md`) proved the picker is
**DATA-limited, not model-limited**: the `unified_v13_zenjpeg_cvvdp`
parquet sweeps only **5 q levels {10,30,60,80,90}**, so the within-cell
reach ladder is too coarse and held-out argmin accuracy is stuck at
0.216 regardless of model (distillation made it *worse*). This dry-run
de-risks the dense-sweep data-gen by closing the whole loop end-to-end
on 8 clustered sources, then specifies the full-scale grid for greenlight.

**Honest scope:** this proves the PIPELINE runs and that dense-q data is
PRODUCIBLE. The held-out argmin number here (n=12 train images, 4 cells)
is statistically near-meaningless — it is **not** a verdict that the
dense sweep "works" or beats 0.216. The accuracy gain is what the
full (user-greenlit) run demonstrates.

## Pipeline (each stage verified)

```
sources (clustered)
  └─ resize (Lanczos, downscale-only)         → size variants
       └─ zen-metrics sweep (encode + score + persist)
            ├─ encode: zenjpeg × dense-q × knob-cells
            ├─ score:  zensim-gpu + ssim2-gpu (ALL variants, local CUDA)
            ├─ persist: encoded bytes + 372-feat zensim parquet + Pareto TSV
            └─ pairs TSV
       └─ zen-metrics batch ssim2-gpu (correct-monotone reach-ladder score)
  └─ build_picker_parquet.py (join + content-address sha256 + all metrics)
       └─ zenpicker-train (MLP, no q-leakage, grouped-by-image holdout)
            └─ ZNPR v3 bake (loads via zenpredict::Model / MetaPicker)
```

### 1. Source selection — k-means clustering (not random)

Per zensim/CLAUDE.md "Dense sampling for trained models" stratification
rule. `extract_features_for_picker` (zenanalyze) extracted 33 zenanalyze
content features over a 300-image distinct-content sample of the
`/mnt/v/input/zensim/sources` pool (17,083 PNGs, 512sq variants).
`zenpicker-train/scripts/cluster_sources.py` ran k-means (K=8, 17
non-constant standardized features, seed=0) and picked the
centroid-nearest member of each cluster:

| cluster | size | representative | content class |
|--:|--:|---|---|
| 6 | 96 | `gen-doc__00275…` | document |
| 5 | 66 | `gen-screen__00017…` | screenshot |
| 3 | 48 | `2156881…` | photo |
| 0 | 30 | `gen-chart__00405…` | chart |
| 2 | 19 | `gen-mixed__00035…` | mixed |
| 7 | 19 | `1435d2e01fb3e90c…` | photo |
| 1 | 11 | `gen-chart__00345…` | chart |
| 4 | 11 | `gen-line__00255…` | line-art |

Spread covers photo / screen / line-art / chart / doc / mixed — the
content classes CLAUDE.md requires. (The synthetic-pool `gen-<class>__`
prefixes are self-documenting; cluster sizes are recorded in
`cluster_assignment.json`.)

### 2. Encode dense-q

`scripts/sweep/picker_dense_sweep_dryrun.sh` (zenmetrics) drives
`zen-metrics sweep --codec zenjpeg`:

- **q grid (dense):** step 5 in 5..69 + step 2 in 70..100 = **29 levels**
  (matches the picker's `ZQ_TARGETS` density).
- **knob grid (4 cells):** `subsampling ∈ {444,420} × progressive ∈
  {false,true} × sharp_yuv {false} × effort {1}`. The
  `knob_tuple_json` the sweep emits is canonical sorted JSON, parsed
  identically by the picker's `cell_key_from_knob`.
- **sizes:** native 512sq + a **256px Lanczos downscale** (downscale
  only, no upscale per CLAUDE.md). 2 size classes for the dry-run.
- 8 img × 29 q × 4 cells × 2 sizes = **1,856 encodes, 0 failures**.

### 3. Score (local GPU)

Local **NVIDIA RTX 5070 / CUDA**, confirmed working for `zensim-gpu`,
`ssim2-gpu`, and `cvvdp` (identity pair → 100 / 99.99 / 10.0). The sweep
scores every cell with `zensim-gpu` + `ssim2-gpu` (both variants
persisted). Throughput ≈ 16 pairs/sec at 512px (incl cubecl init).

> **score_zensim source — IMPORTANT.** The shipped zensim metric has a
> documented correctness defect on photo content (MEMORY.md
> `project_v39_correctness_defect`): on `1435d2e…` it scores q90 (−100)
> *worse* than q30 (−36) — inverted and non-monotone — while ssim2
> correctly ranks q90 (89.8) > q30 (67.6). A non-monotone reach ladder
> corrupts the picker target, so the dry-run uses the **batch-path
> ssim2-gpu** score (correct-monotone, q90>q30 for all 8 images) as the
> `score_zensim` reach-ladder target. zensim is *trained to predict
> ssim2*, so this is a sound stand-in; the raw zensim-gpu score is still
> carried through as `metric_zensim_gpu`. The full run should likewise
> use a correct-monotone quality target (ssim2 or a fixed zensim), not
> the raw defective zensim metric.

### 4. Persist (content-addressed)

Per CLAUDE.md "always persist encoded variants":
`build_picker_parquet.py` (zenanalyze) sha256-content-addresses every
encoded `.jpg` into `artifacts/<sha256>.jpg` (896 unique — free dedup
across identical low-q / 444≈420 cells), and each parquet row carries
`encoded_sha256`. **ALL** metric variants are carried as `metric_*`
columns, not just the reach-ladder scalar. **Verification gate passed:**
artifacts confirmed on disk before assembly (1,856 encoded → 896
deduped, 27 MB; 372-feat sidecars 928 rows/size; Pareto TSVs with
`encoded_bytes`/`encoded_filename`/`score_zensim_gpu`/`score_ssim2_gpu`).

Picker parquet: **1,856 rows × 382 cols** (8 base + 2 metric variants +
372 feat), schema matching `build_picker_dataset`:
`image_basename, codec, q, knob_tuple_json, score_zensim, encoded_bytes,
encoded_sha256, size_class, metric_*, feat_0..feat_371`.
score_zensim range −0.7..96.3 (full dynamic range — the dense q gives a
genuine reach ladder vs unified_v13's 5 coarse levels).
sha256 `c29703e88813c268…`.

### 5. Feature extraction (372col)

The sweep's `--feature-output` emits the **zensim 372-feature** sidecar
(`with-iw` regime) per cell — the same `feat_*` schema the picker
consumes. (The dry-run uses these directly; the zenanalyze
`extract_features_for_picker` content-feature extractor was used only
for the source clustering in step 1.)

### 6. Retrain the picker (no q-leakage)

`zenpicker-train --codec zenjpeg --val-frac 0.25` on the dense-q parquet:

- 428 (image, target_zq) decision rows; **373 inputs** (372 image feat +
  `zq_norm`); **q is NOT an input** (manifest `q_is_input = false`).
- Grouped-by-image holdout: 319 train / 109 val (4 of 16 image@size held
  out). 4 categorical cells.
- Selected hidden=[128,128], lr=2e-3, seed=0.

**Held-out picker panel (n=12 train images — NOT a verdict):**

| stat | value |
|---|--:|
| argmin accuracy | **0.6147** |
| byte overhead | mean 0.039 / p50 0.000 / p90 0.157 |
| bytes_log SROCC | 0.6525 |
| bytes_log PLCC | 0.6725 |
| bytes_log KROCC | 0.4931 |
| bytes_log PWRC | 0.9137 |
| bytes_log Z-RMSE | 0.7401 |
| bytes_log OR | 0.0235 |

> **tiny-n caveat.** argmin 0.6147 is over a 4-cell space (random = 0.25)
> with only 12 training images and 4 held-out. It is NOT comparable to
> the 0.216 sparse-q baseline (which had 36 cells, random 0.028, ~160
> train images) — different cell counts, different n. The dry-run proves
> the loop closes and the dense-q ladder is producible; it does **not**
> prove a dense sweep beats 0.216. That requires the full run below.

Bake: ZNPR v3 (`5a4e5052 03` header), 373→128→128→4 LeakyReLU+identity,
270 KB, loads via `zenpredict inspect` / `zenpredict::Model` /
`zenpicker::MetaPicker`. zenpicker-train tests pass.

> The bake manifest's `data_coverage_caveat` text is hardcoded in the
> binary to reference unified_v13's 5-q sweep — it is stale for this
> dense-q corpus (the numeric fields are correct). A follow-on should
> source the caveat from the input parquet's provenance.

## FULL-SCALE GRID SPEC (for user greenlight)

The dry-run scales to the production data-gen by widening 4 axes. Per
CLAUDE.md "Dense sampling for trained models":

| axis | dry-run | **full-scale** |
|---|---|---|
| sources | 8 (K=8 clusters) | **K=20 clustered** (centroid-nearest, denser feature embedding) |
| sizes | 2 (512, 256) | **16–20 log-spaced** maxdims (32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024) — downscale-only from native; small sources cover only the low end |
| q | 29 (dense) | **same 29-level dense grid** (already production-density) |
| cells | 4 | **36** (`subsampling{444,422,420} × progressive{f,t} × sharp_yuv{f,t} × effort{0,1,2}` = 3×2×2×3) — match unified_v13's cell space |

**Cell count estimate (per-image, per-size):** 36 cells × 29 q = 1,044
encodes. Across 20 sources × 18 sizes = 360 (image,size) units (fewer in
practice — large sources only, no upscale): ~360 × 1,044 ≈ **376k
encodes** (upper bound; realistically ~200k after the no-upscale prune).

**Compute (LOCAL GPU):**

- Encode: ~17 ms/cell at 512px (faster at small sizes) → ~200k encodes ≈
  **~1 GPU-host-hour of encode** (CPU-bound, parallel).
- Score: ~16 pairs/sec/metric at 512px steady-state. 200k cells × 2
  metrics (zensim+ssim2) + 1 cvvdp ≈ 600k GPU scores ÷ ~25/sec (small
  sizes faster) ≈ **~7 GPU-hours**. Add cvvdp (slower, ~5/sec) → **~12
  GPU-hours total** on the local RTX 5070.

**Storage (block, /mnt/v + R2):** ~31 KB avg encoded × 200k ÷ dedup
(~0.5) ≈ **~3 GB content-addressed artifacts**; 372-feat parquet ~3 KB
× 200k ≈ **0.6 GB**; picker parquet ~0.2 GB. Total **~4 GB** → R2
$0.06/mo. Diffmaps (if a perceptual metric emits them) add ~the same.

**LOCAL vs FLEET tradeoff:** ~12 local-GPU-hours is an overnight run on
the RTX 5070 at **$0 spend** — recommended. A vast.ai fleet would cut
wall-clock to ~1 hour but costs ~$2–5 + image-build overhead and trips
the CLAUDE.md fleet-spend rules. **Recommend the local overnight run**
unless wall-clock is critical.

**Greenlight checklist before the full run:**
1. Re-cluster K=20 on a denser feature embedding over the full pool.
2. Resize each source to the 16–18 log-spaced sizes (Lanczos,
   no-upscale).
3. Run `picker_dense_sweep_dryrun.sh` with `--knob-grid` widened to 36
   cells + the size-variant dirs + a correct-monotone score target.
4. Persist + content-address (verification gate after first chunk).
5. Mirror canonical parquet + artifacts to R2 + Tower before any cleanup.
6. Retrain + report the full Mohammadi panel vs the 0.216 sparse-q
   baseline on a real held-out image split (≥20% of images).

## Artifacts (all on /mnt/v, none in git)

- Data root: `/mnt/v/zen/picker-dense-dryrun-2026-05-27/`
  - `parquet/picker_dense_dryrun_zenjpeg.parquet` (1,856×382, sha256 `c29703e88813c268…`)
  - `artifacts/<sha256>.jpg` (896 content-addressed encodes)
  - `features/feat_{sz512,sz256}.parquet` (372-feat sidecars)
  - `pareto_{sz512,sz256}.tsv`, `score_pairs_ssim2.tsv`
  - `bake/zenjpeg_picker_dense_dryrun.bin` (+`.toml`, ZNPR v3, 270 KB)
  - `chosen_sources.txt`, `cluster_assignment.json`, `cluster_sample_features.tsv`
  - pointer file in repo: `benchmarks/picker_dense_sweep_dryrun_2026-05-27.pointer.md`
- Scripts:
  - `zenmetrics/scripts/sweep/picker_dense_sweep_dryrun.sh` (encode+score+persist driver)
  - `zenanalyze/zenpicker-train/scripts/cluster_sources.py` (k-means source selection)
  - `zenanalyze/zenpicker-train/scripts/build_picker_parquet.py` (join + content-address)
- Logs: `/tmp/picker_dryrun_sweep.log`, `/tmp/picker_retrain.log`,
  `/mnt/v/zen/picker-dense-dryrun-2026-05-27/logs/`.

## Tool gaps hit (none blocking)

- `unified_v13_zenjpeg_cvvdp.parquet` is not on local disk under that
  name (the prior picker work used a worktree-local / R2 copy). The dry-
  run regenerates everything from the source pool, so this didn't block.
- The shipped zensim metric's photo-content non-monotonicity (above) —
  worked around with ssim2 as the reach-ladder target; flagged for the
  full run.
