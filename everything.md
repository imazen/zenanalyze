# everything.md — zen training/picker/metric ecosystem

**Compiled 2026-05-09. Forensic reconstruction across `zenmetrics`, `zenanalyze`
(zenpicker + zenpredict + zentrain), and `zensim`. Cross-references the
`RECOVERY_PLAN_2026-05-08.md` / `RECOVERY_HANDOFF_2026-05-08.md` recovery cycle.**

This is the central tracking doc for what's shipping, what's in flight, what's
parked, and what needs to happen next. Always cross-check `git log` and the
per-repo `docs/RECOVERY_REGISTER_2026-05-08.md` files before acting on anything
here — recovery cycles are still landing.

---

## 0. Quick-reference map

```
~/work/zen/
├── zenanalyze/                    Image-feature extractor + ZNPR runtime + meta-picker + Python trainer
│   ├── src/                       102 features (102 active IDs 0–121, gaps for retired; 90 dense-percentile IDs 122–211 on feat/dense-percentiles ONLY)
│   ├── zenpredict/                ZNPR v3 binary parser + Predictor + bake CLI. crates.io 0.1.0 is v2-only; v3 unpublished on main
│   ├── zenpicker/                 CodecFamily meta-picker (wraps zenpredict::Predictor for codec-family routing)
│   ├── zentrain/                  Python training pipeline (train_hybrid.py, bake_picker.py, ablation/inspection tools)
│   ├── tools/                     Python picker trainers (per-codec + meta-picker variants)
│   ├── benchmarks/                Picker bakes (.bin/.manifest.json), training results, audit MDs
│   └── docs/                      RECOVERY_REGISTER_2026-05-08.md, HANDOFF-2026-05-04.md, r2-zentrain-layout.md
├── zenmetrics/                    GPU metric crates + zen-metrics CLI + sweep infra (vast.ai + Docker)
│   ├── crates/butteraugli-gpu/    CubeCL multi-vendor butteraugli (max + pnorm3)
│   ├── crates/dssim-gpu/          CubeCL DSSIM
│   ├── crates/ssim2-gpu/          CubeCL SSIMULACRA2
│   ├── crates/zensim-gpu/         CubeCL zensim 228-feature extractor
│   ├── crates/zen-metrics-cli/    Unified score / batch / compare / sweep CLI (binary `zen-metrics`)
│   ├── crates/zenmetrics-corpus/  Test image corpus
│   ├── scripts/sweep/             vast.ai launcher, onstart_v3.sh worker, janitor, atomic chunk claim
│   ├── Dockerfile.sweep.v13       Latest sweep image (build context = ~/work/zen for sibling path-deps)
│   └── docs/                      CUBECL_GOTCHAS, SSIM2_GPU_HANDOFF, RECOVERY_REGISTER_2026-05-08
└── zensim/                        Perceptual quality metric crate (XYB pyramid, 228/300 features)
    ├── zensim/                    Core library — V0_2 always-on, V0_4 gated behind __experimental_versions
    ├── zensim-bench/              dataset_metric_baseline + score-mapping calibration + KonJND anchor
    ├── zensim-regress/            Visual regression testing (independent semver)
    ├── zensim-validate/           Linear-weight trainer + dataset loaders (CID22/KADID/TID/KonJND)
    ├── zensim-wasm-tests/         WASM cross-arch parity
    ├── scripts/v_next/            Python trainer (PARKED — to be removed once zentrain port lands)
    └── docs/                      v_next_status, v0_5_multicodec_postmortem, score_quality_v04, CID22_PAPER_NOTES
```

The contract between training and runtime is **ZNPR v3** + the **`schema_hash`
u64** + the **`feat_cols` list**. The contract between zenanalyze and
downstream is the **stable `AnalysisFeature` u16 discriminants** + the **0.1.x
threshold contract** (numeric drift permitted in patches; signatures frozen).

`zenanalyze` ships under 0.1.x **forever** per `zenanalyze/CLAUDE.md` —
"There will never be a 0.2.x." Every change must fit within the additive
contract.

---

## 1. Boundaries of responsibility

### `zenanalyze` (the analyzer crate, top-level)
Owns: `features_table!` macro and the stable `u16` discriminants;
tier-pass implementations (Tier 1/2/3, Palette, Alpha, depth);
`RowStream` Native/Convert dispatch over any `zenpixels::PixelSlice`;
public `analyze_features` / `analyze_features_rgb8` /
`try_analyze_features_rgb8` entry points.

Does **not** own: picker training (zentrain), runtime (zenpredict), or
meta-picker selection (zenpicker). Feature *semantics* are the contract;
how a downstream consumer turns features into encoder configs is its
problem.

API freeze: 0.1.x only. Numeric drift on features permitted; signatures
frozen.

### `zenpredict` (the runtime crate, sibling under zenanalyze/)
Owns: ZNPR v3 binary format (parser, header, LayerEntry, Section,
output_specs / discrete_sets / sparse_overrides / feature_transforms);
forward-pass arithmetic across f32/f16/i8 weight storages;
scratch-owning Predictor wrapper; masked argmin and top-K math; score
transforms; threshold gating; OOD bounds; two-shot rescue decision logic;
typed-TLV metadata reader; Rust-side composer/JSON baker behind the
`bake` cargo feature.

Does **not** own: training (no codec specifics, no loss functions, no
calibration). Codec-agnostic by construction — codecs and metric crates
compose on top.

API surface lock plan (Phase 4 yagni-trim, gated on user approval):
- Keep public: `Model::from_bytes`, `Predictor::{new, predict}`, `argmin::*`,
  `output_spec::*`, `rescue::*`, `bounds::*`, `metadata::*`,
  `feature_transform::*`, `error::*`.
- Gate behind `bake` cargo feature: `bake::{BakeRequest, BakeError, build_bake}`.
- Demote to `pub(crate)`: `inference::{LayerKind, forward_*}`,
  `f16_bits_to_f32`, `scale_i8_row`.

### `zenpicker` (the meta-picker crate, sibling under zenanalyze/)
Owns: `CodecFamily {Jpeg=0, Webp=1, Jxl=2, Avif=3, Png=4, Gif=5}` enum
and its stable order contract (`zenpicker.family_order` metadata key,
validated at load time); `AllowedFamilies` mask type; `MetaPicker`
struct that wraps `zenpredict::Predictor` sized to `CodecFamily::COUNT`.

Does **not** own: per-codec configuration selection — that's each
codec's per-codec picker, also a `zenpredict::Predictor`, also baked
through zentrain.

### `zentrain` (the training pipeline, sub-directory under zenanalyze/)
Python tooling, **not a Rust crate**. Owns: per-codec config modules
(one per codec under `examples/`, declaring paths, `KEEP_FEATURES`,
`ZQ_TARGETS`, axis decomposition, config-name parsers); shared trainer
(`tools/train_hybrid.py`, 2743 lines, codec-agnostic hybrid-heads MLP);
ablation/permutation/holdout/size-invariance probes;
`tools/bake_picker.py` (shells out to `zenpredict-bake` CLI for byte
packing).

Phase-3 port queued: 8 missing trainer features that the v06-rebalance
Rust trainer had — FiLM heads, MoE, magnitude-matching loss, sampler
bias (low-band oversample), dct_hf appender, `--also` dataset mixing,
`--val-policy=min`, cclass-as-input on main. Scaffolded at
`tools/zensim_metric_train.py` (5×-committed 2026-05-08 17:04–17:11);
not yet integrated with `--codec-config` ecosystem.

### `zensim` (the metric crate, separate repo)
Owns: the perceptual metric (XYB pyramid, modified SSIM, edge artifact
+ detail loss + HF + peak features), the score mapping, the diffmap
(per-pixel error map for encoder loops), profile dispatch, and the
V0_4+ MLP *runtime* (a thin re-export of zenpredict).

Does **not** own: training (Python pipeline lives in
zenanalyze/zentrain), bake-time tooling (`zenpredict-bake` CLI),
feature extraction for non-zensim features (zenanalyze contributes the
33 named `feat_*` and content-class one-hot tail).

V0_4+ pattern: zenanalyze produces unified parquets + zenpredict ZNPR
bake → zensim `include_bytes!` loads it under `__experimental_versions`.

### `zenmetrics` (the GPU metric + sweep crate, separate repo)
Owns: the path from `(reference_image, distorted_image) →
metric_values` plus the orchestrator that drives a codec-grid sweep on
vast.ai. Specifically: (a) the unified `zen-metrics` CLI
(score/batch/compare/sweep), (b) four CubeCL-based multi-vendor GPU
metric crates (butteraugli-gpu, ssim2-gpu, dssim-gpu, zensim-gpu), (c)
the docker image and onstart script that workers run, (d) the
chunk-claim + janitor + diagnostics fleet management, (e) the per-cell
zensim 300-feature parquet sidecar that lands as training data.

Does **not** own: model training, picker decisions, encoder defaults,
rate-distortion curves. Does not own the metric implementations
themselves on the *CPU* side: those are external crates (`butteraugli`,
`ssimulacra2`, `dssim-core`, `zensim`); the GPU metrics target parity
with those CPU crates.

---

## 2. Cutting edge — what's the best of each thing today

### Per-codec pickers (intra-codec)

| Codec | Best known | Date | Path | Holdout result | Wired? |
|---|---|---|---|---|---|
| **zenjpeg** | `zenjpeg_picker_v0.1_2026-05-07.json` | 2026-05-07 | `benchmarks/zenjpeg_picker_v0.1_2026-05-07.{json,txt}` | -8.30% bytes pred / -13.08% oracle (63% capture) on 711 cells | **NO** — sklearn JSON only, **not yet baked to ZNPR v3 .bin**. v0.3.bin (2026-05-04) baked-and-shipped but has q90 -13pp zensim cliff. zenjpeg crate has no `with_picker()` integration (issue #128). |
| **zenwebp** | `zenwebp_picker_v0.3_2026-05-04.bin` | 2026-05-04 | `~/work/zen/zenwebp/benchmarks/zenwebp_picker_v0.3_2026-05-04.bin` (also on R2 at `s3://zentrain/zenwebp/pickers/`) | -3.44% bytes vs bucket on cid22-val 41-img holdout, +0.068pp zensim parity, argmin_acc 58.7% | **PARTIAL** — schema_hash drift (`0x139d73665fb030c7` vs runtime `0xb2aca28a2d7a34ec`); only `zenwebp_picker_v0.1.bin` is in production today. Drop-in requires schema_hash bump in `zenwebp/src/encoder/picker/spec.rs`. Baker bug: `feature_transforms` length 36 instead of 82. |
| **zenjxl** | `zenjxl_picker_v0.7b_2026-05-06.bin` | 2026-05-06 | `benchmarks/zenjxl_picker_v0.7b_2026-05-06.bin` (98 inputs, 20 cells, ZNPR v3 f16, 65 KB, schema_hash `0x5896532033934e16`) | overall -0.95% bytes; photo -1.12%, lineart -3.85%, screen 0.000% (gated to default), synthetic +0.02% | **NO** — zenjxl crate has no `with_picker()` API. With per-class encoder rule (`patches=True, gaborish=False` for screen/synthetic) → -3.32% overall. The shipped v0.6_mlp **catastrophically regresses on screen content (+41.4% bytes)** per `picker_v06_per_class_audit_2026-05-06.md` — DO NOT SHIP v0.6_mlp standalone. |
| **zenavif** | `zenavif_picker_v0.5_2026-05-04.bin` (R2-only) | 2026-05-04 | `s3://zentrain/zenavif/pickers/zenavif_picker_v0.5_2026-05-04.bin` (no copy in `benchmarks/` on main) | distance-banded SHIP -6.70% bytes / +17.54 pp zensim; argmin_acc 72.6%, mean overhead 5.49% | **NO** — `auto_tune()` is TODO in zenavif crate. PR `imazen/zenavif#11`. v0.3.bin in `benchmarks/` is archival only (val argmin acc 23.3%, fails OVERFIT/LOW_ARGMIN/DATA_STARVED safety gates). |
| **zenpng** | (none) | n/a | — | n/a | PNG enters via meta-picker v0.5 only. |

**Note**: There's a small reconciliation between agent reports — the
best_tuners agent flagged `zenavif_picker_v0.3` as best-on-disk (since
v0.5 is R2-only) while the zenanalyze agent quoted v0.5 as the SHIP
result. Both are correct: v0.5 is the validated ship-quality bake but
lives only on R2; v0.3 is the only zenavif `.bin` in main `benchmarks/`
and explicitly fails its own safety gates. **For practical purposes:
there is no shippable zenavif picker today** — v0.5 needs to be pulled
from R2 and committed to `benchmarks/`, AND zenavif crate needs
`with_picker()` API.

### Cross-codec routers (inter-codec)

| Generation | Bake | Date | Codecs | Holdout | Δ bytes vs always-X | Honest oracle | Acc |
|---|---|---|---|---|---|---|---|
| **v0.4 (4-codec)** ← current best **when PNG optional** | `benchmarks/zenpicker_meta_v0.4_4codec_2026-05-06.bin` (20 inputs, 4 outputs, schema_hash `0x67da414af13b267d`) | 2026-05-06 | jpeg/webp/jxl/avif | 142 cells / 40 imgs | **-6.72% vs always-jxl** | -28.39% oracle | 58.5% |
| **v0.5 (5-codec)** ← current best **when PNG required** | `benchmarks/zenpicker_meta_v0.5_5codec_2026-05-06.bin` (20 inputs, 5 outputs, schema_hash `0x30b45862aa501ff0`) | 2026-05-06 | jpeg/webp/jxl/avif/png | 142 cells | **-2.33% vs always-jxl** (regression vs v0.4 because adding zenpng on n=7 holdout hurt boundaries) | -28.39% oracle | 52.1% |
| v0.3 (3-codec) | `benchmarks/zenpicker_meta_v0.3_2026-05-06.bin` (20 inputs, 3 outputs, schema_hash `0xd900d5193bee3d3c`) | 2026-05-06 | webp/jxl/avif | 227 cells | -16.92% (filtered subset) → **honest -7 to -10% on full holdout** per `picker_schema_mismatch_2026-05-06.md` | -22.81% oracle | 67.4% |
| v0.2 (3-codec, partial v12) | `benchmarks/zenpicker_meta_v0.2_2026-05-06.bin` | 2026-05-06 | webp/jxl/avif | 189 cells / 5 classes | -12.03% vs always-jxl | -22.76% | 59.3% |
| v0.1 (3-codec, photo-only) | `benchmarks/zenpicker_meta_v0.1_2026-05-06.bin` (93 inputs, 3 outputs, schema_hash `0xcb6e6e91690cf6d5`, 24 KB) | 2026-05-06 | webp/jxl/avif | 125 cells / 38 imgs | -10.22% vs always-AVIF | -16.79% | 57.6% |

**Honest oracle ceiling**: -13.26% not -16.92% per
`7d4356d finding: honest oracle ceiling is -13.26%, not -16.92%`. The
-16.92% headline on v0.3 was on a multi-codec-filtered subset that
biases toward routing-friendly cells.

**v0.5 5-codec is currently regressed vs v0.4 4-codec.** Per-codec
accuracy on zenavif winners drops 0.625 (v0.3) → 0.406 (v0.4) →
**0.188 (v0.5)**. Adding zenpng (only 7 holdout cases, acc=0.143) made
the model worse, not better. Ship v0.4 if PNG is optional.

**Suspicious**: v0.4 has not yet had per-class audit. There may be a
hidden screen regression analogous to v0.6 zenjxl's +41% — since v0.4
training corpus is ~94% photo by image count.

### Content classifier

| Bake | Date | Path | Result |
|---|---|---|---|
| **v0.2 (4-class)** | 2026-05-06 | `benchmarks/content_classifier_v0.2_2026-05-06.bin` (15 inputs, 4 outputs, schema_hash `0x10429adc95a10579`, 7.7 KB ZNPR v3 f16) | **99.6% holdout accuracy** on 3,525 imgs from 17,629-source rebalanced corpus (filename heuristics labels) |

Confusion matrix shows ≥0.988 precision/recall per class. Per-class:
photo (1090/1103), screen (602/603), lineart (1218/1219), document (600/600).

**Open issue**: missing **synthetic** class. Meta-picker v0.{2,3} expects
5-cclass input (photo/screen/lineart/document/**synthetic**). Schema
mismatch — synthetic content always maps to one of the 4 known classes
at runtime, absorbing a bias the meta-picker learned during training.

**Recommendation**: retrain `content_classifier_v0.3` with 5 outputs.
Cost ≤30 min. Synthetic labels exist in the rebalanced corpus's
`gen-synthetic__*` paths.

### zensim metric profile

| Profile | Path | Status | CID22 SROCC | KADID | TID |
|---|---|---|---|---|---|
| **PreviewV0_2** | `zensim/profile.rs:443-674` (228-element f64 array) | **Default ship; `ZensimProfile::latest()`** | 0.8676 (with 475-pair leak; true number lower, not re-measured) | 0.8192 | 0.8427 |
| **PreviewV0_4** | `zensim/weights/v0_4_2026-04-30.bin` (60932 B, ZNPR v2) | **Ship under `__experimental_versions`** | 0.8893 | 0.8432 | 0.8401 |
| **V0_5 SSIM2-proxy** (cherry-pick candidate) | `/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin` | Recovery register's recommended swap-in | **0.8934** | **0.8505** | **0.8492** |
| V0_5 multi-codec (FAILED) | `s3://zentrain/v_next-training/2026-05-07/bakes/v0_5_2026-05-07_multicodec.bin` | **Archived, not shipping** | 0.8609 | 0.3697 (catastrophic) | 0.6298 |
| V0_6 + FiLM rebalanced (val_mean leader, unverified) | `/mnt/v/output/zensim/v06-rebalance/runs/v06_film_rebal_20260506T081152.bin` | PR #31 OPEN; **CID22 missing from val set** | unverified | 0.8488 | 0.8386 |
| V0_6 dct_hf | `/mnt/v/output/zensim/synthetic-v2/runs/v06_dct_hf_20260501T164958.bin` | parked on v06-rebalance branch | 0.8935 | 0.8496 | 0.8416 |
| V0_6 mixed-supervision (parked) | scripts/v_next/train_v_next_mlp.py | trainer **marked for removal** per recovery register | KADID 0.81 / TID 0.78 / CID22 0.84 (recovery from V0_5 collapse but doesn't match V0_4) | | |
| V0_6 + MoE | `~/work/zen/zensim--v06-moe/docs/moe_architecture.md` | PR #32 OPEN; **architecture only — no training run** | n/a | n/a | n/a |
| V0_7 e1-fill | `/mnt/v/output/zensim-v07-e1-ablation-2026-05/runs/v07_e1_*pct.bin` | **Abandoned** — every fraction regresses | every variant -0.006 to -0.034 sum-Δ vs V0_6 baseline | | |
| V0_7 canonical+xcodec (latest run, 2026-05-07 19:22) | `/mnt/v/zen/zensim-training/2026-05-07/runs/20260507T192218_v07_canonical_plus_xcodec/` | does **not** beat V0_5 incumbent | 0.7670 (well below 0.8934) | 0.8507 | 0.7464 |

**Decision**: per `RECOVERY_REGISTER_2026-05-08.md` line 24, **swap
V0_4 bake to V0_5 SSIM2-proxy MLP**. Same byte format. +0.004 CID22 /
+0.007 KADID / +0.009 TID. Trivial mechanical change (one-file flip +
docstring update at `zensim/src/profile.rs:160-181`). NOT YET DONE.

### ZNPR file format

| Version | Status |
|---|---|
| v1 (ZNPK) | retired (was vendored in zensim before PR #24) |
| v2 | `zenpredict 0.1.0` on crates.io reads only this; consumers (zensim, zenavif, zenwebp, zenpicker) link 0.1.0 |
| **v3** | **Current; on `zenanalyze/main` since commit `6b552a5` (2026-05-06); UNPUBLISHED**. `bake_v2()` (name kept for source-compat) actually emits v3. v2 bins fail with `PredictError::UnsupportedVersion`. v3 adds `output_specs` (per-output activation/clamp/snap-to-discrete/sentinel pipeline), `discrete_sets` (f32 pool referenced by output_specs), `sparse_overrides` (post-spec patches), `feature_transforms` (per-input identity/log/log1p applied before scaler). Header is 128 bytes `#[repr(C)]`, magic `b"ZNPR"`. `#[non_exhaustive]` on Header / BakeRequest / BakeError / PredictError. |

**Phase 4 publish gate** (per recovery handoff): re-bake all consumers
to v3 + zenavif/zenwebp ship caller-supplied bake API + yagni-trim
zenpredict public API + user explicitly approves publish window. Until
then, zenpredict 0.1.0 stays on crates.io as v2-only.

---

## 3. Vast.ai sweep pipeline (zenmetrics)

### v15 sweep — production recipe, COMPLETE
- 30 vast.ai boxes, 983 chunks (1 image per chunk), all `zenjpeg`
- 19-q grid × 11 knob axes (subsampling/progressive_mode/effort/chroma_distance_scale/aq_enabled/auto_optimize/sharp_yuv/optimize_huffman/deringing/quant_source)
- Metrics: `[zensim, ssim2-gpu, butteraugli-gpu]` (CPU dssim missing — features-backfill is the queued fix)
- Launcher: `scripts/sweep/v15/launch_gpu.sh`
- Chunks: `scripts/sweep/v15/chunks_gpu.jsonl`
- Tracking: `/tmp/v15-prep/v15_instances.txt`
- TSV outputs landed at `s3://zentrain/sweep-v15-2026-05-06/zenjpeg/`

### Docker image
- **Current**: `Dockerfile.sweep.v13` (commit `aba984c`, 2026-05-08, on local `master` only)
- Image: `ghcr.io/imazen/zen-metrics-sweep:0.6.3`
- Build context: **`~/work/zen` (parent of zenmetrics + zenjpeg + zenanalyze)** — required for sibling path-deps
  ```
  docker build -f zenmetrics/Dockerfile.sweep.v13 \
               -t ghcr.io/imazen/zen-metrics-sweep:0.6.3 \
               ~/work/zen
  ```
- Baked binary: `/usr/local/bin/zen-metrics`
- ENTRYPOINT: `/usr/local/bin/zen-metrics-worker` (= `onstart_v3.sh`)
- Env defaults: `WORKDIR=/workspace/sweep`, `SWEEP_GPU_RUNTIME=cpu`
  (overridable per-worker; v15 launcher passes `-e SWEEP_GPU_RUNTIME=cuda`)

### Worker (onstart_v3.sh)
1. Imports R2_*, SWEEP_*, WORKER_*, STATS_* from `/proc/1/environ`
2. Statically installs s5cmd 2.2.2 + jq 1.7.1 + mc to `/usr/local/bin`
3. **Computes parallelism cgroup-aware** — reads `/sys/fs/cgroup/{cpu,memory}.max`,
   takes `min(cgroup_cores, ram_gb*2/3) - 2`. This is the 2026-05-04 fix
   that gave 3-5× throughput on multi-core boxes (`vast.ai`'s `nproc`
   reports host cores not container limit).
4. Prefers image-baked binary; falls back to `SWEEP_BIN_OVERRIDE` (s3:// or URL).
5. Spawns `stats_loop` background heartbeat every 60s.
6. Syncs sources from R2.
7. **Atomic-ish chunk claim**: skip if `s3://zentrain/<run>/<codec>/<chunk_id>.tsv`
   present; read-back-verify own claim token after 1.5s settle; drops
   duplicate-work to <1% (vs ~22% with prior plain-cp claim).
8. Stages images, invokes `zen-metrics sweep` with feature-output parquet.
9. Mid-chunk partial flush every 60s to `s3://coefficient/partials/...`.
10. On success: ship TSV + parquet to R2; on fail: log to errors prefix.

### Janitor (`scripts/sweep/sweep_janitor.py`, commit `d1560b8`)
Reaps a worker when `wall_min ≥ 8 AND (cells_min < 100 OR cpu_recent < 5%)`.
Destroys whole fleet when TSVs reach target.

### Diagnostics (`scripts/sweep/sweep_diag.py`, commit `20bb75d`)
Prints fleet-aggregate work_min vs waste_min %, per-worker recent vs
lifetime cells/min.

### v16 cross-codec sweep — BLOCKED
- 25-box vast.ai launch (v16w / v16a / v16j) at $1.25/hr produced TSV
  rows with **all metric columns blank**. Workers reported `[done]` rows
  with chunk-key fields populated but no measurements.
- Same `zen-metrics-0.6.8-linux-x86_64-gpu` binary works locally on the
  same source-image directory.
- Diagnosis: environment-side; "most likely the
  `ghcr.io/imazen/zen-metrics-sweep:0.6.3` docker image is missing
  something the binary dlopens (libwebp / libaom / libjxl runtime), or
  something in the worker's onstart pipeline got truncated."
- Cost: $0.64 of $31.74 vast.ai credit. **All workers destroyed; no
  retry attempted.**
- Recovery plan documented in `zensim/docs/v_next_status_2026-05-07.md`:
  1. Spin up a SINGLE vast.ai box, drop into `docker exec`, run binary
     manually, look at stderr.
  2. If runtime lib gap: rebuild docker image with missing libs.
  3. Run a 1-chunk smoke before scaling.

### Parquet sidecar schema (zen-metrics-cli + sweep)
305 columns:
```
image_path: utf8 not null
codec: utf8 not null
q: uint32 not null
knob_tuple_json: utf8 not null
zensim_score: float32 not null
feat_0..feat_299: float32 not null  (300 zensim extended features)
```
zstd level 3, FLUSH_EVERY=256 rows/batch (~311 KiB/batch). Per-chunk
file naming `features-<chunk_id>.parquet`. Joins back to TSV by
`(image_path, codec, q, knob_tuple_json)`.

`NUM_FEATURES = 300` per `feature_writer.rs:44` ("4 scales × 3 channels
× 25 features/channel"); the *trained* extractor produces 228 (4 × 3 ×
19); the extra 72 are zero-weight masked features included for
forward-compat.

---

## 4. Data layout (training authority)

### Canonical V_X result store (v_next pipeline)
`/mnt/v/zen/zensim-training/2026-05-07/unified/` — **2,374,666 rows × 50 cols**
across 7 parquets:

| File | Size | Rows |
|---|---|---|
| `unified_v15r_zenjpeg.parquet` | 496 MB | **1,785,696** |
| `unified_v15rc_zenjpeg.parquet` | 695 MB | **513,570** |
| `unified_v13_zenjpeg.parquet` | 36 MB | 36,000 |
| `unified_v12_zenjxl.parquet` | 20 MB | 32,000 |
| `unified_v12_zenavif.parquet` | 16 MB | 4,000 |
| `unified_v14_zenpng.parquet` | 14 MB | 2,400 |
| `unified_v12_zenwebp.parquet` | 14 MB | 1,000 |

Schema (uniform): `image_path, codec, q, knob_tuple_json, encoded_bytes,
encode_ms, decode_ms, score_zensim, score_ssim2, feat_0..feat_299 (300
zensim feat_*), metric_runtime, sweep_id, image_basename, content_class,
size_class, width, height, corpus_feat_*` (32 named zenanalyze corpus
features).

Generated 2026-05-07 by
`zensim/scripts/v_next/build_unified_parquet.py`, git commit
`9d66e6fc2dc02964ce86c531d53626bf66890fc3`.

Mirror: `s3://zentrain/v_next-training/2026-05-07/unified/`.

### Picker bakes (in source repo, NOT on /mnt/v)
All in `/home/lilith/work/zen/zenanalyze/benchmarks/`:

| Bake | Size |
|---|---|
| `content_classifier_v0.2_2026-05-06.bin` | 7.7 KB |
| `zenjxl_picker_v0.3_2026-05-04.bin` | 53.8 KB |
| `zenjxl_picker_v0.6_mlp_2026-05-06.bin` (HAZARDOUS — +41% screen) | 66.5 KB |
| `zenjxl_picker_v0.6_safety_2026-05-04.bin` | 40.0 KB |
| `zenjxl_picker_v0.7b_2026-05-06.bin` (best) | 67.8 KB |
| `zenpicker_meta_v0.1_2026-05-06.bin` | 24.5 KB |
| `zenpicker_meta_v0.2_2026-05-06.bin` | 18.4 KB |
| `zenpicker_meta_v0.3_2026-05-06.bin` | 18.4 KB |
| `zenpicker_meta_v0.4_4codec_2026-05-06.bin` (best 4-codec) | 7.6 KB |
| `zenpicker_meta_v0.5_5codec_2026-05-06.bin` (best 5-codec, regressed vs v0.4) | 7.9 KB |

zenjpeg / zenwebp / zenavif pickers live in their own codec-repo
`benchmarks/` (or only on R2 — zenavif v0.5).

### Anchor datasets

| Dataset | Path | Used as |
|---|---|---|
| **CID22** (gold standard) | `/mnt/v/dataset/cid22/CID22_validation_set.csv` | Validation only. 49 references held-out, 4,292 fully-disjoint pairs. SSIM2 maps 1:1 to MCOS per CID22 paper Table 5. |
| **KADID-10k** | `/mnt/v/dataset/kadid10k/` | 7,125 train + 3,000 val. |
| **TID2013** | `/mnt/v/dataset/tid2013/` | 2,160 train + 840 val. |
| **KonJND-1k** | `/mnt/v/dataset/konjnd-1k/` | Calibration anchor only — visually-lossless fixed points. NEGATIVE result as training signal. |
| **Synthetic-v2** | `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv` | 218,089 pairs, 0 CID22 validation leak. **The canonical training base.** |

### Tower NAS state
- Mounted: `tower:/mnt/user/coefficient` → `/mnt/tower/` (NFS v4.2)
- **Severely lagging**: only 5 features bins from `synthetic-v2/` are on
  tower; zero of 80+ run dirs, 1.6 GB cv-results, 25 GB total zensim
  outputs are mirrored.
- CLAUDE.md states tower is source of truth — **observed reality is
  inverted**: `/mnt/v` holds the active state, `/mnt/tower` holds an
  early-2026 snapshot.
- `r2-zentrain-layout.md` recommends R2 + /mnt/v as the canonical pair.

### Disk pressure
| Subtree | Size |
|---|---|
| `/mnt/v/output/zensim/` | 25 GB |
| `/mnt/v/output/corpus-builder/` | ~18 GB |
| `/mnt/v/zen/zensim-training/2026-05-07/` | 6.6 GB |
| `/mnt/v/output/codec-corpus-2026-05-01-multiaxis/` | 140 MB |
| `/mnt/v/output/coefficient/` | ~1 GB |
| `/mnt/v/output/zenpicker/` | 57 MB |
| `/mnt/v/output/zensim-regress/` | 15 MB |
| `/mnt/v/output/zenanalyze/` | 28 KB (only 2 reports) |

Recommended archive (push-to-tower-then-evict, ≈5 GB):
- 3 × 837 MB `training_ext_*.broken_ext.features.bin` (interrupted
  pipeline run from 2026-03-04, mirrored on tower)
- `cv-results/` 1.6 GB (2026-03-05)
- 191 MB + 49 MB + 43 MB " - Copy" duplicates from 2026-02-26

---

## 5. Suspicious post-2026-05-05 work (reconciled forensic view)

User's warning: "work after last Tuesday is suspect, due to context loss
work started over from an ancient branch ignoring ten iterations of
experiments."

**Reconciled finding**: most post-2026-05-05 work is legitimate progress
(recovery cycle, picker iterations, recovery register documentation).
What the user was describing is most likely:

1. **Two "background-agent clobbers" on `feat/dense-percentiles`** —
   sub-agent truncated `src/feature.rs` twice via base64 corruption.
   Fixes: `498e86a fix(tier3): decode base64-corrupted file; restore
   correct Rust source`, `8cd1688 fix: restore correct feature.rs
   (background agent pushed truncated version)`, `107aae9 fix: restore
   correct feature.rs again (second background-agent clobber)`. **The
   dense-percentile branch (IDs 122–211) is research-only — quarantine
   until corpus story is settled. Not trained on by any picker bake.**

2. **5× retried `zensim_metric_train.py` scaffold commits** between
   2026-05-08 17:04–17:11 (`fe6b977`, `1f544c1`, `467846a`, `4c12591`,
   `4d820ca`) — Phase 3 zentrain port scaffolding that the recovery
   session redid multiple times. Currently 280 lines, not yet integrated
   with `--codec-config` ecosystem.

3. **Parking commits** (`017fc07`, `7d2c823`, `12e5322`, `875cdbe`,
   `35f18b3`, `20775d9`, `2198170`) auto-parked WIP that the session
   never described. `f3057d1 Revert "(parking — user WIP on this WC;
   user to describe when ready)"` reverted one chain.

4. **Parallel hash-only commits** in zenmetrics (`8e099ed`, `cbabe56`,
   `b0176b1`, `835a0e9`) — empty-subject jj snapshots needing explicit
   `jj describe`.

5. **Hazardous bake on main**: `zenjxl_picker_v0.6_mlp_2026-05-06.bin`
   stays in `benchmarks/` despite +41.4% screen regression.

**The "ten iterations" maps cleanly to**: v0.1 → v15 zenjpeg picker
series (only v0.1/v0.2 land on main); meta-picker v0.1 → v0.5 (5
versions); zenjxl v0.6_safety / v0.6_mlp / v0.7 / v0.7b chain; dense
percentile sweep IDs 122–211; per-class signal probe B / B′ (NO-SHIP);
SA piecewise v5 (NO-SHIP); i8-quant impact research; time-budgeted
objective experiments; student permutation ablation; cclass-as-input
experiments → v0.7b screen-gate.

The user's WIP at HEAD was **just rustfmt drift** (cargo fmt --check
exit=0 confirmed), now committed and pushed in `chore(zenpredict):
rustfmt drift cleanup` (zenanalyze) and `chore(zensim-validate):
rustfmt drift cleanup` (zensim).

### Crashed-predecessor session
- Session ID: `7c3af8e8-2d71-4c9a-985a-e10c4317be63` ("claude-recovery-2026-05-08")
- Started: 2026-05-08T04:14:40 in `~/work/zen/zenanalyze--zenpredict-pre-v3`
- Ran: ~42 hours (May 8 04:14 → May 9 22:10)
- Substantive work that landed on main: recovery register, ZNPR v3 spec,
  zentrain scaffold, v3 API hardening (`#[non_exhaustive]` + builder),
  zenpicker safety audit artifacts, garb 0.2.8 dep bump (PR #75 merged).
- Only WIP that didn't land before crash: 4 lines of cargo-fmt
  formatting drift across two repos (now resolved this session).

---

## 6. Pipeline data flow (numbered, end-to-end)

1. **Codec sweep harness** (per-codec, codec-side, e.g.
   `zenjpeg/dev/zq_pareto_calibrate.rs`,
   `jxl-encoder/examples/lossy_pareto_calibrate.rs`,
   `zenavif/benchmarks/dev/rav1e_phase1a_pareto.rs`,
   `zenwebp/dev/zenwebp_pareto.rs`) encodes corpus × every config,
   measures bytes + zensim + encode_ms. Owner: codec crate.

2. **Output**: per-codec sweep TSV `image_path | size_class | width |
   height | config_id | config_name | q | bytes | zensim | encode_ms`.
   Format target: Parquet (zstd-3) once `>50 MB` per
   `~/work/claudehints/topics/parquet-vs-tsv.md`. Owner: codec crate.

3. **R2 upload**: `s3://zentrain/sweep-<id>-<DATE>/<codec>/<chunk>.tsv`
   plus `_manifest.json`. R2 endpoint
   `https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`.
   Public dev URL `pub-c8010c5b1ac84b968fa3d3b5cd3c2dae.r2.dev`. Owner:
   codec sweep harness via `zen-metrics-cli` upload step.

4. **Mirror download**: locally
   `s3cmd cp s3://zentrain/<run>/...` → `/mnt/v/zen/zensim-training/<run>/`.
   Owner: training session.

5. **Feature extraction**: `zenanalyze::analyze_features_rgb8` over a
   per-codec manifest. Centralized:
   `zenanalyze/examples/extract_features_for_picker.rs`. Output:
   `feat_<name>...` columns plus `image_sha`, `split`, `content_class`,
   `source` pass-through. Owner: zenanalyze.

6. **Schema adapter**: `zentrain/tools/zenmetrics_sweep_adapter.py`
   translates `zen-metrics 0.3.0+` sweep TSV to zentrain pareto schema.
   Owner: zentrain.

7. **(Optional) Backfill 300-feature parquet**: `zen-metrics
   features-backfill --input-tsv chunk.tsv --output-parquet chunk.parquet`.
   **NOT YET LANDED** (`feat/features-backfill` branch only — 752 LOC +
   286 LOC tests at commit `bd86239`). Owner: zenmetrics.

8. **Refresh orchestration**: `zentrain/tools/refresh_features.py`
   (Tier 1 of INVERSION.md, landed 2026-05-02). Owner: zentrain.

9. **Trainer**: `zentrain/tools/train_hybrid.py` (2743 lines). Two-phase
   teacher-student: HistGradientBoostingRegressor per cell + shared MLP
   `n_inputs → 128 → 128 → 3*N_cells` LeakyReLU. Output: `OUT_JSON` +
   `OUT_LOG`. Owner: zentrain.

10. **Safety gate**: `bake_picker.py` reads `safety_report.passed` and
    refuses to bake when false unless `--allow-unsafe`. Owner: zentrain.

11. **Bake**: `tools/bake_picker.py --model X.json --out X.bin --dtype
    {f32,f16,i8}` shells out to `zenpredict-bake` CLI. Output: ZNPR v3
    `.bin` (~30 KB f16 / ~60 KB f32 / ~15 KB i8) + `.manifest.json`
    sidecar. Owner: zentrain Python; zenpredict Rust binary handles
    byte-packing.

12. **Round-trip verify**: `tools/bake_roundtrip_check.py`. Owner: zentrain.

13. **Ablation/inspection**: `feature_ablation.py`,
    `student_permutation.py`, `correlation_cleanup.py`,
    `feature_group_ablation.py`, `size_invariance_probe.py`,
    `inspect_picker.py`, `diagnose_picker.py`, `adversarial_probe.py`.
    Owner: zentrain.

14. **Held-out A/B**: `tools/holdout_ab_lookup.py` (zq-banded for
    zenwebp/zenavif), `tools/holdout_ab_lookup_jxl.py` (distance-banded
    for zenjxl, NEW 2026-05-04). Owner: zentrain.

15. **Codec consumption**: codec embeds `.bin` via `include_bytes!()`
    + aligned wrapper, calls
    `zenpredict::Model::from_bytes_with_schema`, instantiates
    `Predictor::new(model)`. Per encode: extract features →
    `argmin_masked` → resolve config. Owner: each codec crate.

16. **Meta-picker (cross-codec)**: `zenpicker::MetaPicker` wraps a
    `zenpredict::Predictor` whose `n_outputs == CodecFamily::COUNT`.
    Caller passes `(features, AllowedFamilies)` → returns
    `Option<CodecFamily>`. Owner: zenpicker.

---

## 7. What needs revisiting (priority backlog)

### IMMEDIATE — production blockers
1. **Land the `feat/features-backfill` zenmetrics branch** (commit
   `bd86239`, 752 LOC + 286 LOC tests at `crates/zen-metrics-cli/src/backfill.rs`).
   The recovery register lists this as "kept" but it never merged; the
   files don't exist on master. Recovery action says "mirror v15 TSVs
   to R2 + backfill CPU dssim before training pickers/zensim". Without
   the dssim column, the multi-target trainer in zentrain has to drop a
   metric or run an additional sweep.

2. **Reconcile zenmetrics local `master` ↔ `origin/master`**. Local
   master is 14 commits ahead with all the v15 sweep infrastructure
   (rayon parallelism, janitor, atomic claim, zenjpeg+zenpng codecs, v15
   launcher). If local working copy is wiped, that work is gone unless
   pushed. Recovery handoff explicitly asks for this. Options: land as
   PRs, or rebase main onto master and force-push as a topic branch.

3. **Re-train `content_classifier_v0.3` with 5 outputs (add synthetic
   class)**. Cost ≤30 min. Removes the schema mismatch between
   `content_classifier_v0.2` (4-class) and `zenpicker_meta_v0.{2,3}` (5
   cclass inputs). Unblocks per-class A/B audits on production traffic.
   Synthetic labels exist in rebalanced corpus's `gen-synthetic__*`
   paths. **best_tuners agent's "if you can only train ONE thing"
   recommendation.**

4. **Bake-swap zensim V0_4 to V0_5 SSIM2-proxy MLP** per recovery
   register: `zensim/weights/v0_4_2026-04-30.bin` ←
   `/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin`.
   Same byte format. +0.004 CID22 / +0.007 KADID / +0.009 TID. Update
   docstring at `zensim/src/profile.rs:160-181` and the V0_4 bake
   function at line 178. **Trivial mechanical change.**

### HIGH — correctness blockers
5. **Remove or quarantine `zenjxl_picker_v0.6_mlp_2026-05-06.bin`**
   from `benchmarks/`. It causes +41.4% bytes regression on screen
   content. v0.7b (with screen gate) supersedes it. Risk: accidental
   inclusion via `include_bytes!` if someone ships zenjxl with picker
   integration before v0.7b is the canonical default.

6. **Per-class audit on `zenpicker_meta_v0.4_4codec`**. Headline -6.72%
   has not yet been broken down per content class. v0.6 zenjxl had
   +41% hidden screen regression at -1.879% photo-weighted aggregate;
   v0.4 4-codec is at risk of the same since training corpus is ~94%
   photo. **Mandatory before promoting v0.4 to production.**

7. **Bake `zenjpeg_picker_v0.1_2026-05-07` to ZNPR v3** (currently
   sklearn JSON only). Then re-run cid22-val A/B to see if it beats
   v0.3 across all bands including q≥85. After that, densify
   `chroma_distance_scale` grid (0.5/0.8/1.0/1.4) per the v0.1 report's
   recommendation.

8. **Land `zenwebp` schema_hash bump**: `spec.rs::SCHEMA_HASH` from
   `0xb2aca28a2d7a34ec` to `0x139d73665fb030c7`, and replace
   `zenwebp_picker_v0.1.bin` with `v0.3.bin`. Then per-class audit on
   the 4-class rebalanced corpus to confirm no hidden screen
   regression. Also fix the baker bug in `feature_transforms` length
   (currently 36 instead of 82).

9. **CID22 bench v06-rebalance FiLM bake** — recovery register's
   biggest gap. PR #31 is blocked on this. Run
   `dataset_metric_baseline --cid22 ... --v04-bake
   .../v06_film_rebal_20260506T081152.bin` plus the per-class
   `c0..c4_*.bin` dispatch.

### MEDIUM — recovery sequence Phase 3
10. **Port v06-rebalance Rust trainer features into
    `zenanalyze/zentrain/tools/zensim_metric_train.py`**:
    - `train_loop` end-to-end wiring + bake to ZNPR v3 (~½ day)
    - zenanalyze `dct_hf` feature appender
      (`attach_zenanalyze_features`) (~2 hrs)
    - magnitude-matching loss (`magnitude_match_term`) (~1 hr)
    - sampler bias (`--low-band-oversample`) (~2 hrs)
    - FiLM heads + 5-per-class bake + manifest (~½ day)
    - MoE (unverified — train+eval before deciding to keep)
    - multi-target loss (ssim2 + butteraugli_p3) (~2 hrs)
    - cclass-as-input on main (currently only on `v15-5codec-metapicker` branch)
    - `--also` dataset mixing (multiple corpora w/ per-class weights)
    - `--val-policy=min` (mean-over-groups val selection)

11. **Wire pickers into codecs** (Phase 0 of `rapid_iteration_plan`,
    ~10 hr total):
    - zenjpeg: issue #128 — runtime uses fixed single-config LUT, **34-85% byte savings unrealized**
    - zenjxl: no `with_picker()` API in wrapper
    - zenavif: `auto_tune()` is TODO
    Currently **only `zenwebp_picker_v0.1.bin` is wired into a codec in
    production today.** Every other artifact in `benchmarks/` is
    candidate / archival / pending.

12. **Retry v16 cross-codec sweep** with diagnosed docker fix:
    1. Single vast.ai box; `docker exec`; run binary manually; capture stderr.
    2. If runtime lib gap: rebuild docker image with missing libs.
    3. 1-chunk smoke before scaling.
    4. Mirror back to `s3://zentrain/sweep-v16{w,a,j}-2026-05-07/`.

13. **Refit V0_4 `score_mapping`** — currently `(18.0, 0.7)` (V0_2-classic).
    `+7..+11 zensim-vs-ssim2 bias` and the calibration findings
    recommend either `(5.0, 1.21, offset=-10.87)` or piecewise-21 (RMSE
    27.67, exactly equal to noise floor; monotonic; 42 f64 = 336 bytes).
    Requires additive `ScoreMapping` enum (public API change).
    `benchmarks/v04_calibrate_mapping_2026-05-01.md:243-272`.

### MEDIUM — recovery sequence Phase 4 (gated on Phase 3 completion)
14. **ZNPR v3 publish (zenpredict 0.2.0)**:
    - Yagni-trim public API per `zenpredict/docs/ZNPR_V3.md` (gate
      `bake::*` behind `bake` feature; demote
      `inference::{LayerKind,forward_*}`/`f16_bits_to_f32`/`scale_i8_row`
      to `pub(crate)`)
    - Migrate consumers: zensim re-bake `weights/v0_4_2026-04-30.bin`
      to v3 + bump zenpredict dep to 0.2.0; zenavif/zenwebp merge
      `feat/expert-internal-params` + caller-supplied bake API
    - Bump zenpredict to 0.2.0 in Cargo.toml + CHANGELOG.md
    - End-to-end smoke build (no publish)
    - Tag + request publish window from user

### LOW — future research
15. **Score-quality regression on V0_4** (8.26% non-monotonic q-step
    rate vs ssim2's 5.08%). TV regularizer queued as `--tv-weight` flag
    in the parked Python trainer; needs to land in zentrain. Acceptance:
    V0_5/V0_6 should track or beat ssim2's 5.08%. (`zensim/docs/score_quality_v04_2026-05-07.md`.)

16. **Add scales 5+6 to zensim** — CID22 paper uses 6, zensim uses 4.
    Multi-scale invariance handoff TODO §4.3.
    `zensim/docs/CID22_PAPER_NOTES_2026-05-07.md:118-123`.

17. **Re-train v0.7c zenjxl picker on rebalanced corpus** (3000+
    gen-screen sources at `~/work/zen/zensim--v06-rebalance`). This
    should let the picker pick non-default cells for screens (likely
    `patches=True`) instead of gating to default. Combine with the
    per-class encoder rule at the codec layer to validate the -3.32%
    end-to-end claim from `end_to_end_pareto_simulation_2026-05-06.md`.

18. **Resume zenavif picker v0.4 sweep** from R2 chunks if still on
    disk; or re-launch on vast.ai with the spec from
    `picker_v0.4_data_starvation_spec_zenavif_zenjxl.md`. Cell taxonomy
    is now (speed ∈ {3,5,7,9} × tune ∈ {0,1}) = 8 cells, much less
    data-starved.

19. **Multi-seed LOO on candidate cross-codec consensus list**
    (`feat_log_pixels`, `feat_bitmap_bytes`, `feat_indexed_palette_width`)
    BEFORE any breaking release that touches these. Single-seed signals
    are large but variance is 4-8pp on zenwebp.

20. **Phase D1 of multi-MLP safety doc**: ship `Pick` enum +
    `Predictor::pick()` + `ParamClamp` strategy (~3 days). Then wire
    each codec to use `pick()` (zenwebp first, zenjpeg next per
    `rapid_iteration_plan_2026-05-02.md` Phase 0).

---

## 8. Open experimental branches

### zenanalyze
- `feat/dense-percentiles` (PR #74 OPEN) — IDs 122–211 dense percentile
  sweep; QEMU cross-build skips on top. **Quarantined** — has 2
  background-agent clobber fixes; not trained on by any picker bake.
- `feat/dispatch-plan` (PR #54 OPEN) — `analyze_with_dispatch_plan`
  skeleton + `DispatchHints` struct. Recovery register: "Currently empty
  seats; future-proofs the dispatch architecture without bloat. Merge."
- `feat/zenpicker-i8-agreement` (PR #76 OPEN) — only currently OPEN PR
  on the meta-picker. Adds `load_meta_picker_v0_1` lib.rs additions and
  i8 vs f16 agreement example. Worth merging.
- `feat/time-budgeted-objective` — encode-time-budgeted picker
  objective (PR #64 merged earlier; now garb 0.2.8 dep bump tip).
- `v15-5codec-metapicker` — 5-codec metapicker trainer + zenjpeg v0.1
  picker trainer + chroma calibration. Not merged. Trainer source on
  branch; .bin/.manifest artifacts ARE on main.
- `salvage/zenwebp-picker-prior-agent-2026-04-30` — prior-agent WIP
  salvaged. The "salvage" name flags prior-agent damage that this
  session preserved.
- `bench/per-class-bprime-2026-05-04` (PR #72 merged) — v10 multi-codec
  router + 4 partial-data picker reports.
- `bench/picker-v06-rebake-attempt` (PR #73 merged) — picker pipeline
  schema mismatch findings; the +41% screen audit doc.

### zensim
- `v04-mlp` — partially merged via #29; remaining bake artifact swap
  pending.
- `v06-rebalanced-corpus` (PR #31 OPEN) — current val_mean leader
  (FiLM); CID22 bench pending.
- `v06-moe` (PR #32 OPEN) — architecture only, no training run.
- `v06-film` — superseded by v06-rebalance.
- `v06-content-class` — precursor to v06-rebalance.
- `v07-e1-ablation` — abandoned (every fill fraction regresses).

### zenmetrics
- `feat/butteraugli-multi-column` — kept (required for zentrain
  multi-target loss).
- `feat/sweep-v12-balanced` — partial; informs v16.
- `feat/migrate-sweep-scripts` — kept (reorg under `scripts/sweep/`).
- `feat/features-backfill` — **NEVER MERGED**, see #1 in revisit list.
- `fix/sweep-binary-path` — kept (rayon-binary path fix).
- `fix/patch-jxl-encoder-dos-fix` — kept (security).
- `security-fixes-h1-h4` — kept.

---

## 9. Vast.ai / Docker / GPU notes

**Hardware**: water-cooled AMD Ryzen 9 7950X, 128 GB RAM, NVIDIA GPU
with CUDA 13.2.1 SDK. nvcc not on PATH by default; cubecl-cuda dlopens
CUDA at runtime so `--features sweep,gpu,gpu-cuda` builds without nvcc.

**vast.ai pinning** (`scripts/sweep/v15/launch_gpu.sh`):
- `N_BOXES=30 MAX_DPH=0.20 MIN_CORES=8 MIN_RAM_GB=12 MIN_DISK_GB=25`
- `cuda_max_good>=12 num_gpus=1` filter
- No `verified=true` (per CLAUDE.md "verified=true excludes most cheap offers")
- GHCR auth via `gh auth token`
- Per-instance `SWEEP_BIN_OVERRIDE=s3://coefficient/binaries/zen-metrics-0.6.7-linux-x86_64-gpu`
  (workspace `Cargo.toml` says version 0.6.0 — the 0.6.7 binary was
  built and uploaded out-of-tree)
- Per-instance `SWEEP_GPU_RUNTIME=cuda`, `SWEEP_RUN_ID=sweep-v15-2026-05-06`

**Build flavours** (per `zenmetrics/CLAUDE.md`):
- Default dev: `cargo build --release -p zen-metrics-cli` (CPU + sweep codecs)
- Forced GPU-only worker:
  `cargo build --release -p zen-metrics-cli --no-default-features --features sweep,png,gpu,gpu-cuda`
  → drops cpu-metrics so workers can't silently fall back to slow CPU scoring.
- WGPU variant (broader GPU compatibility, no CUDA SDK required):
  `--no-default-features --features sweep,png,gpu,gpu-wgpu`

**GPU metric parity targets**:
- `butteraugli-gpu` ↔ `butteraugli` v0.9.2 (max + pnorm_3 fused reduction)
- `ssim2-gpu` ↔ `ssimulacra2` v0.5.1 (Charalampidis recursive Gaussian)
- `dssim-gpu` ↔ `dssim-core` v3.4 (5 pyramid scales, two-pass 3×3 Gaussian, custom-Lab)
- `zensim-gpu` ↔ `zensim` v0.2.8 with `WEIGHTS_PREVIEW_V0_2` (228 features = 4 × 3 × 19)

**CubeCL gotchas** (per `docs/CUBECL_GOTCHAS.md`, 30-entry catalogue):
- G1.1 `f32::exp` not registered → use `powf(2.0, x*LOG2_E)`
- G3.x **Metal silently no-ops `Atomic<f32>::fetch_add`** (the
  `fast-reduction` feature is broken on Metal; verified working on CUDA
  / Windows DX12 / HIP)

---

## 10. ZNPR v3 wire format quick reference

Magic = `b"ZNPR"`, version `u16 = 3`. Little-endian throughout.

128-byte `#[repr(C)]` Header:
```
0..4    magic = b"ZNPR"
4..6    version: u16 = 3
6..8    flags: u16 (reserved)
8..12   n_inputs: u32
12..16  n_outputs: u32
16..20  n_layers: u32
20..24  _pad0
24..32  schema_hash: u64
32..40  scaler_mean: Section
40..48  scaler_scale: Section
48..56  layer_table: Section
56..64  feature_bounds: Section (len=0 when absent)
64..72  metadata: Section (len=0 when absent)
72..80  output_specs: Section (NEW IN v3; n_outputs * 32)
80..88  discrete_sets: Section (NEW IN v3; pool of f32)
88..96  sparse_overrides: Section (NEW IN v3; n_overrides * 8)
96..128 reserved: [u32; 8]
```

`Section = (offset: u32, len: u32)`, `len = 0` means absent.

48-byte LayerEntry:
```
0..4    in_dim: u32
4..8    out_dim: u32
8..9    activation: u8 (0=Identity, 1=Relu, 2=LeakyRelu)
9..10   weight_dtype: u8 (0=F32, 1=F16, 2=I8)
10..12  flags: u16
12..20  weights: Section
20..28  scales: Section (len=0 unless I8)
28..36  biases: Section
36..48  reserved
```

Output spec pipeline (`zenpredict/src/output_spec.rs:43-90`):
1. `transform`: Identity / Sigmoid / SigmoidScaled / Exp / Round
2. `bounds` clamp (inclusive `[low, high]`)
3. snap to nearest value in discrete set, if non-empty
4. sentinel match → `OutputValue::Default` if equal
5. sparse override: replace with override value if idx matches

Feature transforms (per-input, in metadata key
`zentrain.feature_transforms`): `identity | log | log1p` applied
**before** scaler. UTF-8 newline-separated tokens parallel to
`feature_columns`. Hard-fail on length/token mismatch.

Quantization sizes:
- F32: 1× size
- F16: 0.5× size (built-in `f16_bits_to_f32`, no `half` dep)
- I8: 0.25× size, per-output column scale
  (`scales[o] = max_i |W| / 127.0`, `i8_w[i,o] = round(W / scales[o]).clamp(-128, 127)`)

`bake_picker.py --dtype i8` is the default per zentrain CHANGELOG;
holdout argmin-acc delta f32→i8 is < 0.5 pp.

Bake API (`zenpredict::bake::v2`, name kept for source-compat — emits v3):
```rust
use zenpredict::bake::{BakeLayer, BakeRequest, bake_v2};
let layers = [BakeLayer { in_dim, out_dim, activation, dtype, weights, biases }];
let bytes = BakeRequest::builder(schema_hash, flags, &scaler_mean, &scaler_scale, &layers)
    .with_metadata(/* TLV pairs */)
    .with_output_specs(&specs)
    .with_discrete_sets(&pools)
    .with_feature_transforms(&transforms)
    .bake()?;
```

CLI:
- Inspector: `cargo run --release -p zenpredict --bin zenpredict-inspect <model.bin>`
- Baker: `cargo run --release -p zenpredict --bin zenpredict-bake -- <input.json>`

---

## 11. Recovery cycle status (as of 2026-05-08 → 2026-05-09)

| Phase | Status | What landed | What's pending |
|---|---|---|---|
| Phase 0 — Inventory snapshot | ✅ DONE | `~/work/zen/RECOVERY_PLAN_2026-05-08.md` | — |
| Phase 1 — Read & distill | ✅ DONE | Per-repo `RECOVERY_REGISTER_2026-05-08.md` files | — |
| Phase 2 — Per-repo merge plans | 🟡 PARTIAL | Recovery register commits landed where working tree was clean. Uncommitted user WIP in zenanalyze, zenmetrics, zenavif, zenwebp, coefficient was preserved untouched. | Cherry-picks #2 (PR #54 dispatch-plan), #3 (per-class audit doc), #4 (Phase 3 zentrain port). |
| Phase 3 — Re-train champion via zentrain | ❌ NOT STARTED | `zenanalyze/zentrain/tools/zensim_metric_train.py` SCAFFOLDED (5×-committed at 2026-05-08 17:04–17:11). | Implement 8 trainer features (FiLM/MoE/cclass/dct_hf/magnitude-matching/sampler-bias/--also/--val-policy=min). Train champion. Bake to ZNPR v3. Validate against held-out CID22 + KonJND-1k. **Acceptance gate**: CID22 SROCC ≥ 0.8893 (V0_4 baseline). |
| Phase 4 — zenpredict v3 + minimization | ❌ BLOCKED on Phase 3 | ZNPR v3 already on `zenanalyze/main` (commit `6b552a5`, 2026-05-06). Spec at `zenpredict/docs/ZNPR_V3.md`. v3 API hardening (`#[non_exhaustive]` + builder) at `0935914`. | Yagni-trim public API. Re-bake all consumers. Bump zenpredict to 0.2.0. End-to-end smoke. Tag + request publish window. |

**Cleanup to land before next session**: `.workongoing` markers in
zenanalyze/zenmetrics/zenavif/zenwebp/coefficient should be refreshed
or removed based on activity. Parking commits in zenanalyze and
zenmetrics need user attention to claim.

---

## 12. Forensic-trace caveats

1. The **zenanalyze recon agent** says best zenavif picker is **v0.5
   (2026-05-04, distance-banded SHIP -6.70%)** on R2; the
   **best_tuners agent** says best on-disk is **v0.3 (archival, fails
   safety gates)**. Both are correct — v0.5 is the validated ship-quality
   bake but lives only on R2; **for practical purposes there is no
   shippable zenavif picker today**.

2. The **zenmetrics agent** says local `master` already has the v15
   sweep infra; the recovery handoff said "main has rayon parallelism
   etc that master doesn't." This was ambiguous — the handoff was
   referring to **`origin/master`**, which is missing the 14 unpushed
   commits. Local `master` = ahead of `origin/master` by exactly those
   14 commits.

3. The **zensim agent** confirmed the user WIP at HEAD `04e2d82` is **a
   single rustfmt expansion**, not "ancient branch context loss". The
   "ten iterations" refers to v06-* branch work that hasn't been
   integrated into the v_next training pipeline.

4. The **disk recon agent** confirmed the canonical V_X result store is
   at `/mnt/v/zen/zensim-training/2026-05-07/unified/` with 2.37M-row
   parquets across 7 codec/sweep combinations. It also confirmed picker
   bakes are NOT on /mnt/v but in the source repo
   `/home/lilith/work/zen/zenanalyze/benchmarks/`.

---

## 13. One-line recommendations (independent)

- **If you can only train ONE thing next**: `content_classifier_v0.3` with 5 outputs (~30 min). Removes schema mismatch, unblocks per-class audits.
- **If you can only fix ONE zenmetrics thing next**: revive `feat/features-backfill` branch and run on v15 TSVs. Adds dssim column without re-encoding.
- **If you can only land ONE zensim change next**: bake-swap V0_4 to V0_5 SSIM2-proxy MLP per recovery register. One-file flip + docstring update.
- **If you can only land ONE zenanalyze change next**: PR #54 (`feat/dispatch-plan`) per recovery register.
- **If you can only retire ONE risky artifact**: remove or quarantine `benchmarks/zenjxl_picker_v0.6_mlp_2026-05-06.bin` (+41% screen regression).

---

## 14. Source agents

This document was synthesized 2026-05-09 from five parallel forensic
research agents:
- `/tmp/recon_zenmetrics.md` (4,372 words, 14 sections)
- `/tmp/recon_zenanalyze.md` (7,872 words, 1,044 lines, 12 sections)
- `/tmp/recon_zensim.md` (~7,600 words, 593 lines, 12 sections)
- `/tmp/best_tuners.md` (5,239 words, 441 lines, 10 sections)
- `/tmp/recon_disk.md` (~3,000 words, 10 sections)

Plus the Claude session's own deep reading of:
- `~/work/zen/RECOVERY_PLAN_2026-05-08.md`
- `~/work/zen/RECOVERY_HANDOFF_2026-05-08.md`
- `zenanalyze/{CLAUDE.md, CONTEXT-HANDOFF.md, MIGRATION.md, docs/RECOVERY_REGISTER_2026-05-08.md}`
- `zenanalyze/zentrain/{PRINCIPLES.md, INVERSION.md, SAFETY_PLANE.md}`
- `zenanalyze/zenpredict/docs/ZNPR_V3.md`
- `zenanalyze/benchmarks/{all_time_best_features_2026-05-02.md, picker_v06_per_class_audit_2026-05-06.md}`
- `zensim/{CLAUDE.md, docs/RECOVERY_REGISTER_2026-05-08.md, docs/v_next_status_2026-05-07.md, docs/v0_5_multicodec_postmortem_2026-05-07.md}`
- `zenmetrics/{CLAUDE.md, docs/RECOVERY_REGISTER_2026-05-08.md}`

Update this doc when the cutting edge moves; treat it as living
inventory. The "what needs revisiting" list in §7 is the active
backlog — items move out of it as PRs land or get superseded.
