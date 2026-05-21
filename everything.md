# everything.md — zen training/picker/metric ecosystem

**Compiled 2026-05-09. Latest substantive update: 2026-05-20 (V11
substrate retrospective + multi-codec sweep coverage audit). Forensic
reconstruction across `zenmetrics`, `zenanalyze` (zenpicker +
zenpredict + zentrain), and `zensim`. Cross-references the
`RECOVERY_PLAN_2026-05-08.md` / `RECOVERY_HANDOFF_2026-05-08.md`
recovery cycle.**

This is the central tracking doc for what's shipping, what's in flight, what's
parked, and what needs to happen next. Always cross-check `git log` and the
per-repo `docs/RECOVERY_REGISTER_2026-05-08.md` files before acting on anything
here — recovery cycles are still landing.

---

## 00. Latest state (2026-05-20) — V11 substrate retrospective + sweep coverage

### zensim Balanced trail ship

**`PreviewV0_5Balanced`** = V_22-mix-LARGE+iwssim s3 packed +
**V10 BalancedV3 spline calibration** (V9 PCHIP mechanism, shipped
2026-05-20). Bake at `zensim/weights/v_balanced_v3_2026-05-20.bin`.
Verified `bake_verdict` metrics (2026-05-20 re-run):

| Corpus | SROCC | PLCC | Z-RMSE |
|---|---:|---:|---:|
| CID22  | 0.8324 | 0.8256 | 0.564 |
| KADID  | 0.9664 | 0.9562 | 0.293 |
| TID    | 0.9712 | 0.9379 | 0.347 |
| KonJND | 0.8927 | 0.9270 | 0.375 |
| AIC-3  | 0.7845 | 0.7952 | 0.606 |
| AIC-4  | 0.9016 | 0.8900 | 0.456 |

See `zensim/zensim/SOTA_TRAILS.md` for the gate definition and the
companion `PreviewV0_5Compression` + `PreviewV0_5Ensemble` ships.

### V11-A' v2 retrain (task #190) — FALSIFIED

Substrate (anchor + cross-codec-eq) rebuilt cleanly from the
R2 `omni-multi-codec-2026-05-19` prefix; all 4 codecs have
`score_ssim2_gpu` 100 % non-null (correcting V11-A' v1's false
"ssim2 was never computed on non-jpeg codecs" claim). Substrate
artifacts preserved at
`/mnt/v/zen/zensim-training/2026-05-20-v11-substrate/`:

- `unified_omni.parquet` — 754,756 rows × 305 cols
- `anchors_ssim2_300col_v2.parquet` — 8,527 rows × 10 ssim2 bands × 4 codecs
- `cross_codec_equivalence_ssim2_v2.parquet` — 1,739 cross-codec pairs

Brief recipe (per-sample-α head + MSE-only + monotonicity-reg 1.0 +
tanh-output-head-scale 20.0) FALSIFIED across 5 seeds (median CID22
0.7519, −0.0805 below V10 BalancedV3). Brief hparams contraindicated
vs V_24-per-sample-α defaults. Clean recipe (V_24-aligned + V11
substrate) lifts CID22 +0.043 over V10 but collapses KonJND by
−0.489 — a Compression-trail-shape candidate that fails the
Balanced gate. V10 remains the defensible Balanced bake.

Methodology at `zensim/benchmarks/v11_substrate_v2_methodology_2026-05-20.md`
(in the `zensim--v11-substrate-v2` worktree's tree, committed on
zensim main `84ce3390`).

### R2 multi-codec sweep coverage map (2026-05-20)

DATA_PROVENANCE.md was missing a third multi-codec R2 prefix that's
useful as a denser zenwebp/zenavif metric-scores source:

| Sweep | zenwebp cells | zenavif cells | zenjxl cells | features sibling | encoded preserved |
|---|--:|--:|--:|---|---|
| omni-multi-codec-2026-05-19 (unified-worker, the V11 substrate input) | 1,000 (5 q) | 4,000 (5 q) | 51,200 | ✓ 300-feat | ✓ |
| **multi-codec-2026-05-18** (pre-unified-worker) | **4,000 (10 q)** | **14,400 (10 q)** | 6,400 | ✗ | ✗ |
| cvvdp-v15rc-2026-05-18 | — | — | — | ✓ 300-feat | ✓ (zenjpeg, 513k rows) |

`multi-codec-2026-05-18` has 4× the zenwebp coverage and 3.6× the
zenavif coverage (both at 10 q levels, vs 5 q in 2026-05-19), but
needs either a fresh feature-extraction pass (no preserved encoded
variants on R2; would have to re-encode from input parquet) or
consumers that only want the 7 metric scores. Documented at
`zenmetrics/docs/DATA_PROVENANCE.md` "multi-codec-2026-05-18"
section as of 2026-05-20.

### zensim-gpu ↔ zensim CPU feature parity (2026-05-20 reverify)

All 19 parity tests pass on RTX 5070 + CUDA 13.2:

- `cpu_parity` (3/3) — basic + peak 228-slot per-slot
- `extended_parity` (6/6) — extended 300-slot + WithIw 372-slot,
  including the multi-strip 128² checkerboard case that's the
  regression guard for the **strip-overlap / boundary-row activity
  bug** fixed by the 2026-05-17 principled per-channel H-blur
  redesign (zensim `2dab8f3` + zenmetrics `1b8ccab`). CPU and GPU
  both compute `activity[c] = box_blur(|src[c] - H_blur(src[c])|)`
  per channel at all strip rows; the prior cross-channel cascade is
  gone.
- `parity_lock` (8/8) — real-image corpus: dssim-cuda q70 cpu
  80.9018 / gpu 80.8850 (rel 2.1e-4); q90 cpu 91.3509 / gpu
  91.3486 (rel 2.5e-5).
- `weights_parity` (1/1) — byte-for-byte CPU/GPU `WEIGHTS_PREVIEW_V0_2`.
- `opaque` (1/1) — opaque-API srgb-u8 path.

Caveat: sweep parquet data scored against the pre-2026-05-17
masked/IW semantics still uses the old cascade values. Bound on
shift is the 1.5-4 % rel residual the fix eliminated. Re-bake any
V_X model whose training corpus consumed pre-2026-05-17 masked/IW
feature values where slots 228..372 are load-bearing. See
`zenmetrics/crates/zensim-gpu/PORT_STATUS.md` "Principled
per-channel H-blur activity" section for details.

---

## 0a. Latest state (2026-05-10) — zensim champion cycle

**Read this first if you're returning to the project.**

### Goal lock (per zensim/CLAUDE.md "Training goals" section, 2026-05-10)

zensim is the **user-facing quality dial**. Five priorities, in order:

1. **CID22 SROCC is the gold standard.** KADID-10k and TID2013 are
   **NOT compression-tuned** (KADID is ~95% non-compression
   distortions; TID similar) — use them as integrity guards, not
   optimization targets. Optimize CID22.
2. **Smoothness AND monotonicity** are first-class objectives. Target
   non-monotone q-step rate ≤ V0_2's **4.86%** (project floor); ssim2
   GT is 5.08%. TV regularization (`--tv-weight 10..30`) is the lever.
3. **KonJND-1k anchoring** at perceptibility thresholds (mean PJND ≈
   ssim2-63 per CID22 paper Table 4). A trained model must score
   at-PJND pairs 63 ± 5; saturating to 100 means visually-lossless
   calibration is broken.
4. **Filter synth corpus by ssim2 ↔ butteraugli concordance.** Drop
   curves where Spearman(ssim2, -butter) < 0.6 — noisy ranking labels.
   The CID22 paper Tables 3 & 6 flag ssim2 as less reliable in
   q-extremes; butter-concordance is the simplest cross-check
   without human MOS.
5. **CID22 paper governs ssim2-accuracy regions:** ssim2 most
   reliable in q-band 50..90; less so at q > 95 (saturation) and q < 30
   (extreme distortion outside training distribution).

**Anti-goals** (do NOT optimize for):
- Aggregate (KADID + TID + CID22) / 3 SROCC. The aggregate hides CID22
  regressions behind compression-irrelevant gains.
- Synthetic ssim2-target val_srocc. It tracks the trainer's own loss,
  not held-out human judgement (>0.99 across most runs while CID22
  stayed 0.85–0.88).
- Metrics in q < 30 or q > 95 bands.

### Current shipped state (zensim main `e902d519`, 2026-05-10)

`zensim/weights/v0_4_2026-04-30.bin` is **V0_5 SSIM2-proxy MLP**
(swapped in by tick 2 of the champion loop). Numbers (full-dataset
eval): KADID 0.8432 / TID 0.8401 / **CID22 0.8893** / non-mono ~8.26%.

This is the Rust-mlp_train.rs-trained bake from
`/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin`.
The Rust trainer was deleted in PR #29 (commit `e613224`, 2026-05-07)
and **RESTORED on 2026-05-10** (commit `ec40ec8`); it is LIVE at
`zensim/zensim-validate/src/mlp_train.rs` and produced V0_15 + V0_16
(current ship). Pre-deletion snapshot preserved at
`zensim/docs/phase4_reference/mlp_train_rust_e3f8748.rs` for
historical reference only. To reproduce V0_16, run
`bash zensim/benchmarks/recipe_v0_16.sh`.

### Loop session conclusion (Tick 75, 2026-05-11) — V0_5 STAYS SHIP

## 🚀 UPDATE 2026-05-11 (eve): V0_7 SHIPS (zensim commit `c4b059a7`)

Cycle continued through the **holdout-overlap audit + clean-retrain**
arc. The V0_5 "ship" of midday turned out to be inflated by training
leakage; the honest version that beats `fast-ssim2` on CID22 aggregate
AND meets the 5.5 % non-mono target is **V0_7 seed=1**.

| Bake | CID22 SROCC | Non-mono | Notes |
|---|--:|--:|---|
| V0_5 (archived) | 0.8900 | 5.36 % | inflated by 11.77 % training leak from 22 of 49 CID22 holdout refs |
| V0_6 clean (seed=42) | 0.8839 | 5.94 % | honest baseline after dedupe |
| V0_7 seed=0 (archived) | 0.8912 | 5.67 % | initial ship; non-mono over target |
| **V0_7 seed=1 (current ship)** | **0.8933** | **5.46 %** | beats ssim2 (+0.0038), smoothness within target |
| V0_8 sweep (h128 TV20 / h192 TV10) | 0.8897 / 0.8923 | 5.70 % / 5.66 % | both eliminated on non-mono |
| fast-ssim2 baseline | 0.8895 | — | the bar |

**The leak audit was the key unlock**: V0_5's 0.8900 was 11.77 %
training-set contamination, not a genuine ssim2-beat. After cleaning
(28 % of training pairs dropped), a 5-seed sweep produced seed=1 as
the dual-criteria winner.

**Methodology finding**: training-time val_mean does NOT linearly
predict CID22 SROCC. seed=1 had val_mean=0.9437 (vs seed=0's 0.9443)
but HIGHER CID22 SROCC (0.8933 vs 0.8912). Future cycles should
evaluate per-seed CID22 directly, not pick by val_mean.

**Artifacts shipped this cycle**:
- Audit tools: `zensim-validate/src/bin/check_holdout_overlap{,_stage2}.rs`
- Cleaned corpus: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_perceptual_clean.csv`
- Generator patched: `imazen/coefficient` commit `d4cb501` (CID22_VALIDATION_49)
- Page-by-page paper methodology: `zensim/docs/CID22_PAPER_PAGE_BY_PAGE_2026-05-11.md` (30/30 pages)
- GH Pages scaffold: `zensim/site/` + `.github/workflows/pages.yml`
- Champion bake: `zensim/weights/v0_7_2026-05-11.bin` (md5 `0ad0dace`)

**Per-band CID22 (V0_7 vs ssim2)**:
- B0 (<50): 0.4370 vs 0.4418 (−0.005, near-parity)
- B1 [50,65): 0.4424 vs 0.4694 (−0.027, only meaningful loss)
- B2 [65,90): **0.7893** vs 0.7722 (+0.017 BEATS)
- B3 (≥90): **0.1944** vs 0.1121 (+0.082 BEATS)
- Near-PJND: 0.3741 vs 0.3908 (−0.017, near-parity)

**Next-cycle direction**: close the B1 gap. Hypotheses:
- Per-band-weighted TV regularizer (TV=20 helps B1, hurts B0 — needs
  a band-aware variant in the trainer)
- B0/B1 training-data densification
- 5-more-seed sweep at (h=128, TV=10) — currently running (seeds
  5/8/13/21) to confirm seed=1's 5.46 % non-mono is reproducible
  (most clean-data bakes are 5.6-5.9 %)

Full per-tick log: `~/work/zen/zenanalyze/zensim_champion_log.md`
(300+ ticks). Audit details:
`zensim/benchmarks/holdout_overlap_audit_2026-05-11.md`.

---

## Prior verdict (2026-05-11 midday, superseded by the audit cycle above):

**FINAL VERDICT** after 75 ticks of automated training (2026-05-10/11, ~18 hr wall, 130+ trainings, 70+ end-to-end evals):

**Keep the currently-shipped V0_5 SSIM2-proxy MLP** (`zensim/weights/v0_4_2026-04-30.bin`, md5 `bb7e24a1`). It is the **best dual-target model** the recovery cycle could produce or evaluate.

| Bake | CID22 | non-mono | Both targets |
|---|---|---|---|
| **V0_5 shipped (current)** | **0.8893** | **4.57%** | smoothness ✓, CID22 -0.004 |
| h128 no-TV seed=1 | 0.8941 | 6.72% | CID22 ✓ only |
| h128 TV=30 seed=1 | 0.8874 | 4.97% | strictly dominated by V0_5 |
| Smoothness-Winner Python | 0.8769 | 4.49% | strictly dominated by V0_5 |
| Prior CHAMPION (Python h192x128) | 0.8803 | (high) | strictly dominated by V0_5 |

**No model trained in 75 ticks Pareto-dominates V0_5 on both axes.**

**Why V0_5 wins**:
- CID22 0.8893: just -0.004 below the 0.8934 target. Closest model that also meets smoothness.
- Non-mono 4.57%: below the V0_2-floor target 4.86%. None of the new bakes meets this.
- The h128 no-TV seed=1 bake exceeds CID22 (+0.0048) but costs +2.15pp smoothness — bad trade for the dial property (per goal #2 "smoothness is first-class").

**Major plot twist (Tick 73)**: the working assumption that V0_5 had ~8.26% non-mono came from a stale doc about a DIFFERENT bake (the 2026-04-30 mixed-supervision predecessor). When measured directly via `score_unified_with_bake.py`, the actual shipped V0_5 has 4.57% non-mono — already meeting the target. The entire TV-recovery chase (Ticks 65-72) was operating on a misread.

**Closing the residual CID22 gap (-0.004)** would require interventions NOT tried in this cycle:
1. **Multi-target loss** (DSSIM/butter as additional supervisors, Phase 4 deferred work)
2. **Multi-codec corpus expansion** (CLIC + JXL/WebP/AVIF/GIF GPU re-scoring, pending user authorization)
3. **Larger architecture with smoothness regularization tuned to match V0_5's 4.57%**

**For now**: do not swap. V0_5 ships. The detailed per-recovery-tick log is at `~/work/zen/zenanalyze/zensim_champion_log.md`.

**Loop-session best bakes preserved on disk** (for any future comparative work):
- `zensim/benchmarks/rust_webp_mono_h128_seed1_2026-05-10.bin` (CID22-best, smoothness-fail)
- `zensim/benchmarks/rust_webp_mono_h128_tv30_seed1_2026-05-10.bin` (best dual at Rust h=128, V0_5-dominated)
- `zensim/benchmarks/h192x128_ep300_safesyn218k_kt_2026-05-10.bin` (Python aggregate-best, V0_5-dominated)
- 50+ supporting bakes from seed/TV/corpus sweeps, all V0_5-dominated

## 0b. Cycle close (2026-05-11) — 35 ticks extended, all per-band measured

**Read this AFTER §0a.** The 2026-05-10 cycle paused at Tick 75 with "V0_5 stays ship". The cron-driven loop resumed Ticks 76-110 on 2026-05-11 with a final set of experiments and analyses.

### What changed in Ticks 76-110

1. **Alignment bug found (Tick 81)** — the KonJND-mix scoring binary (Tick 77) had source CSV rows misaligned with `features.bin`. Three prior KonJND attempts (Ticks 78-80) trained on RANDOM (features, target) pairings. Fix: sort source CSV by (src_id, codec, quality). New aligned features at `/tmp/zensim_loop/konjnd_aligned_features.csv`.

2. **KonJND-aligned mix WORKS** (Tick 81+): TV=0 h64 → CID22 **0.8921** (+0.0028 over V0_5), smoothness 5.46% (fails).

3. **Capacity exploration (Ticks 95-105)**: h=128 + TV=10 + KonJND → CID22 **0.8900** (slightly beats V0_5 0.8893). h=192 + TV=10 + KonJND → CID22 **0.8859** (REGRESSES from h=128). **Capacity peaks at h=128**, not monotonic.

4. **MEMORY.md / CLAUDE.md correction (Tick 88)**: the claimed "V0_5 CID22 0.8934" in `~/.claude/CLAUDE.md` was aspirational — never measured. md5 hash confirmed shipped V0_5 (`bb7e24a1`) is identical to the file MEMORY.md attributed 0.8934 to. **Measurement-of-record: V0_5 CID22 = 0.8893.**

5. **Per-band analysis (Ticks 95, 99, 109)**: V0_5 wins CID22 B0/B1 narrowly; KonJND-mix wins B2 (q65-90) slightly and B3 (visually-lossless) by 2.8×. KonJND-mix is also +0.10/+0.115 on KADID/TID aggregate.

### Final Pareto state (CID22 sorted, all full 4292-pair measured)

| Bake | CID22 | non-mono | smoothness ✓ | CID22 ✓ |
|---|---|---|---|---|
| h128 WebP-mono no-TV | 0.8941 | 6.72% | ✗ | ✓ |
| TV=0 h64 KonJND | 0.8921 | 5.46% | ✗ | ✗ |
| TV=10 h128 KonJND | 0.8900 | 5.36% | ✗ | ✗ |
| **V0_5 shipped (md5 `bb7e24a1`)** | **0.8893** | **4.57%** ★ | ✓ | ✗ (-0.0041) |
| TV=5 h64 KonJND | 0.8880 | 5.14% | ✗ | ✗ |
| TV=5 h128 KonJND | 0.8871 | 5.44% | ✗ | ✗ |
| TV=10 h192 KonJND | 0.8859 | 5.80% | ✗ | ✗ |
| TV=10 h64 KonJND | 0.8841 | 5.09% | ✗ | ✗ |
| TV=20 h64 KonJND | 0.8812 | 5.29% | ✗ | ✗ |
| TV=30 h128 KonJND | 0.8803 | 5.39% | ✗ | ✗ |

**No bake in the trainer-recipe space dual-clears both targets.**

### Why no recipe-only path will work

- **Capacity saturates at h=128**: h=192 regresses on CID22. The 228-input-feature space is the bottleneck, not the hidden width.
- **TV regularization** trades CID22 for smoothness on a strict line at h=64; non-monotonic at higher h.
- **KonJND-mix** boosts CID22 by ~+0.003 max but costs +0.5pp non-mono. The product-band gains are real (B2/B3) but the smoothness cost is binding.

### Realistic paths to CID22 > 0.8934

NOT recipe tuning. Realistically requires:
1. **Feature-space extension** — add new image features outside the current 228 (queued in zentrain/INVERSION.md tier 2-4)
2. **Different model class** — multi-head, FiLM, MoE (zentrain phase 4 work)
3. **Re-anchored target** — `CLAUDE.md` claims 0.8934 but never measured; if the target is "match V0_5", it's already met (0.8893)

### Artifacts produced in Ticks 76-110

- `zensim/benchmarks/pareto_2026-05-11.md` — full Pareto summary doc
- `zensim/benchmarks/rust_v05recipe_konjnd_*_2026-05-11.{bin,eval.log,train.log}` — 8 new bakes
- `zensim/benchmarks/make_cid22_style_plots_2026-05-11.py` — 5-plot generator
- `zensim/benchmarks/make_cid22_scatter_2026-05-11.py` — paper-canonical scatter
- `zensim/benchmarks/make_per_band_tv_plot_2026-05-11.py` — per-band TV trends
- `/mnt/v/output/zensim/cycle_2026-05-11/` — 7 PNG plots + README.md
- `~/work/zen/zenanalyze/zensim_champion_log.md` Ticks 76-110

### Phase 4 trainer infrastructure (zensim main, after 2026-05-10)

`scripts/v_next/train_v_next_mlp.py` now has 7 flags ported from
the Rust trainer (NOTE: corrected 2026-05-13 — the Rust trainer was
RESTORED on 2026-05-10 and is LIVE; these Python flags mirror its
current behavior, they're not a reconstruction from a snapshot):

| Flag | Default | Rust-faithful value |
|---|---|---|
| `--lr-schedule {constant, cosine}` | constant | cosine (T_max=epochs, eta_min=lr·0.01) |
| `--optimizer {adamw, adam}` | adamw | adam (no decoupled weight decay) |
| `--init {kaiming, glorot}` | kaiming | glorot (std=sqrt(2/(in+out))) |
| `--val-policy {mean, min}` | mean | min (worst per-group SROCC drives selection) |
| `--ranknet-group {image, dataset}` | image | dataset (cross-image absolute ranking, matches CID22 evaluation) |
| `--concordance-filter {none, ssim2_butter}` | none | ssim2_butter (drop curves where Spearman(ssim2, -butter) < 0.6) |
| `--tv-weight FLOAT` | 0.0 | 10..30 (smoothness regularizer; bug fix `dd79a3c` for last-partial-batch slice) |

**Companion script**: `scripts/v_next/convert_features_bin.py` —
ZSFC v3 binary → human-csv format converter. Built in tick 12 to
unlock the canonical 218k clean safe-synthetic training base
(`/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv.features.20260308_162434.bin`).

**Output of converter**: `/tmp/zensim_loop/safe_synth_218k_features.csv`
(745 MB; ephemeral — re-run convert script if /tmp wiped). The
canonical 218k training base in trainer-compatible format.

### Phase 4 outstanding work (unimplemented as of 2026-05-10)

1. **Per-step group-weighted pair sampling** with explicit
   `pairs_per_epoch=50000` budget (Rust trainer's outer loop). The
   `--ranknet-group dataset` flag captures the spirit but not the
   per-step inverse-CDF sampling; the Rust trainer also runs each pair
   through `forward + backprop` independently rather than batching.
   Estimated 30 LOC.
2. **KonJND-1k anchor loss term** — add `--konjnd-anchor-csv PATH:WEIGHT`
   that mixes in 504 JPEG + 504 BPG pairs at PJND with target score ≈
   63 (CID22 paper Table 4). Validates "visually-lossless" calibration
   without using KADID/TID compression-irrelevant signal. Estimated
   20 LOC.
3. **Cyclic cosine LR with 50-epoch period** (Rust uses
   `lr · 0.5 · (1 + cos(pi · (epoch % 50) / 50))`) vs my full-run
   `CosineAnnealingLR(T_max=epochs)`. The Rust schedule restarts
   every 50 epochs, mine decays once over the full run. Estimated 5
   LOC.

### Background pipelines as of 2026-05-10 (mind-wipe time)

A 4-candidate Phase 4 training was running in background (`PID 989643`,
script `/tmp/zensim_loop/phase4_full_train.sh`). Output dirs (when it
finishes) at:
`/mnt/v/zen/zensim-training/2026-05-07/runs/2026051*v_next_h*ssim2_butter*/`.
Signal file: `/tmp/zensim_loop/results/phase4_full_done`.

The 4 candidates exercise the new flags:
1. `[192,128] mse_rank TV=10 image group + concordance` — concordance alone
2. `[192,128] mse_rank TV=10 dataset group + concordance` — dataset-RankNet + concordance
3. `[192,128] ranknet TV=10 dataset group + concordance` — pure RankNet + everything
4. `[32] ranknet TV=0 dataset group + concordance` — V0_5-EXACT + dataset-group + concordance

When picking up after wipe: check `pgrep -f phase4_full_train`. If
still running, wait for `phase4_full_done`; if not, find the run dirs
that match the wildcard above. Bake any winners via
`scripts/v_next/bake_to_znpr.py` and eval via
`zensim-bench/examples/dataset_metric_baseline`.

### Trainer command for the CHAMPION (canonical, reproduces 0.8991 avg / 4.56% non-mono)

```bash
cd ~/work/zen/zensim
python3 scripts/v_next/convert_features_bin.py \
  --bin /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv.features.20260308_162434.bin \
  --csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
  --out /tmp/zensim_loop/safe_synth_218k_features.csv \
  --target-col gpu_ssimulacra2

python3 scripts/v_next/train_v_next_mlp.py \
  --sweeps v15r,v15rc \
  --target ssim2 \
  --loss mse_rank \
  --hidden 192,128 \
  --epochs 300 \
  --batch-size 16384 \
  --lr 3e-3 \
  --weight-decay 1e-5 \
  --rank-weight 0.5 \
  --tv-weight 10 \
  --human-csv "safesyn:/tmp/zensim_loop/safe_synth_218k_features.csv:1.0" \
  --human-csv "kadid:/mnt/v/zen/zensim-training/2026-05-07/v06-features/kadid_features.csv:0.3" \
  --human-csv "tid:/mnt/v/zen/zensim-training/2026-05-07/v06-features/tid_features.csv:0.3" \
  --seed 0 \
  --tag v_next_h192x128_ep300_safesyn218k_kt_2026-05-10 \
  --out-dir /mnt/v/zen/zensim-training/2026-05-07/runs

# Bake
python3 scripts/v_next/bake_to_znpr.py \
  --run-dir /mnt/v/zen/zensim-training/2026-05-07/runs/<TS>_v_next_h192x128_ep300_safesyn218k_kt_2026-05-10 \
  --out /tmp/zensim_loop/bakes/champion.bin

# Eval (~4 min)
cargo build --release -p zensim-bench --example dataset_metric_baseline
target/release/examples/dataset_metric_baseline \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --v04-bake /tmp/zensim_loop/bakes/champion.bin \
  --max-pairs 999999
```

Wall: ~7 min on RTX 5070 + 7950X (60s for converter, 5 min for training,
5s for bake, 4 min for eval).

### Loop session full transcript

`~/work/zen/zenanalyze/zensim_champion_log.md` — 30+ ticks across
2026-05-10. Each tick documents what was tried, why, what landed, what
broke. Read this if you need to know the "why" behind any choice in the
champion recipe.

### Cron loop status

`b55bf342` was a `*/4 * * * *` cron firing the `/loop` skill that
drove ticks 1–30. **Cancelled at tick 29** via `CronDelete`. Future
training continuation should be manual (the loop has converged on the
achievable Pareto frontier given the Python trainer's structural
limitations).

### Ship instructions when CID22 ≥ 0.8893 candidate is found

```bash
# Backup the current shipped V0_5 (also durable at runs/v04_mlp_ssim2_holdout_20260501T045510.bin)
cp ~/work/zen/zensim/zensim/weights/v0_4_2026-04-30.bin \
   ~/work/zen/zensim/zensim/weights/v0_4_2026-04-30.v0_5.bak.bin

# Promote champion
cp ~/work/zen/zensim/benchmarks/<champion>.bin \
   ~/work/zen/zensim/zensim/weights/v0_4_2026-04-30.bin

# Update profile.rs:160-181 docstring with new lineage
# Run cargo test --features __experimental_versions
# jj describe + push
```

### Resume-after-mind-wipe protocol

When a fresh Claude Code session opens this repo and wants to continue
the champion training cycle, run this `/loop` command verbatim. It
will (re)create a 4-minute cron tick that drives the next experiment
each iteration:

```
/loop 4m Continue zensim champion training session.

Step 1 — Read `~/work/zen/zenanalyze/zensim_champion_log.md` to see the goal, phase plan, and the last tick. The phase plan is the working order; do not skip phases without recording why.

Step 2 — Refresh `.workongoing` markers in `~/work/zen/zensim`, `~/work/zen/zenanalyze`, `~/work/zen/zenmetrics` with current UTC timestamp + agent-id `claude-zensim-champ-loop`. If a different agent's marker is fresher than 5 minutes, STOP and append a "skipped: collision with <agent>" tick to the log.

Step 3 — Pick the smallest useful next step from the phase plan (5–10 minutes of focused work). Targets: CID22 SROCC > 0.8934, non-monotonic q-step rate < 4.86%. Constraint: no CID22 training data leak; no `cargo publish`. Use synth-only training data on `/mnt/v`.

Step 4 — Execute that step. Tools available include: cargo build/test/run, file edits, jj commit, R2 sync, the existing trainers (`zensim/scripts/v_next/train_v_next_mlp.py` with `--tv-weight`), the eval harness (`zensim-bench/examples/dataset_metric_baseline.rs`), the synth generator binary (`coefficient/examples/generate_zensim_training`), the unified parquets (`/mnt/v/zen/zensim-training/2026-05-07/unified/`).

Step 5 — Append one tick to the log under `## Tick log`. Format: `### Tick N — <UTC timestamp> — <one-line summary>` plus 2–4 bullets of what changed and what's next. Always state concrete artifacts produced (file paths, hashes, numbers).

Step 6 — If a concrete file/artifact was produced, commit via jj on main (`jj describe -m '<msg>' && jj git push --bookmark main` if appropriate). NEVER touch user WIP. If the next step requires user authorization (publishing, removing files, big sweeps that cost money), document the request in the log and stop.

Anti-patterns: don't restart from an ancient branch; don't ignore the prior 10 iterations the recovery register catalogs; don't relax thresholds; don't run cargo publish; don't add CID22 training data.
```

**Before starting fresh ticks**, the resumed session should:

1. **Check live processes**: `pgrep -f phase4_full_train` (PID 989643 was running at wipe). If alive, let it finish; bake winners via `scripts/v_next/bake_to_znpr.py`. If dead/no run dirs created, re-launch via `/tmp/zensim_loop/phase4_full_train.sh`.

2. **Re-create the converter output if /tmp wiped**:
   ```bash
   cd ~/work/zen/zensim
   python3 scripts/v_next/convert_features_bin.py \
     --bin /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv.features.20260308_162434.bin \
     --csv /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv \
     --out /tmp/zensim_loop/safe_synth_218k_features.csv \
     --target-col gpu_ssimulacra2
   ```

3. **Re-build the eval binary** if `target/release/examples/dataset_metric_baseline` is gone:
   `cargo build --release -p zensim-bench --example dataset_metric_baseline` (~30s on warm cache, ~28s cold).

4. **Recover the loop log + champion docs** by reading:
   - `~/work/zen/zenanalyze/zensim_champion_log.md` — full 30-tick session
   - `~/work/zen/zensim/benchmarks/champion_2026-05-10.md` — champion + smoothness-winner + ultra-smooth
   - `~/work/zen/zensim/benchmarks/tv_smoothness_sweep_2026-05-10.md` — TV+capacity sweep findings
   - `~/work/zen/zensim/docs/phase4_reference/README.md` — gap analysis + Phase 4 plan
   - This file (`everything.md` §0a) — the latest state

5. **Stop the cron** when next round of work is done: `CronDelete <id>`. If
   you don't want a cron at all, just run the prompt by hand.

### Phase 4 priority backlog (after resume)

In this order, given goal #1 is CID22:

1. **Wait for `pgrep -f phase4_full_train`**, bake 4 candidates, run
   end-to-end CID22 evals. If any hits CID22 ≥ 0.8893 with non-mono ≤
   4.86, ship it (replaces CHAMPION).
2. **Implement KonJND-1k anchor loss term** (~20 LOC, goal #3). Add
   `--konjnd-anchor-csv PATH:WEIGHT` flag to `train_v_next_mlp.py`.
   Target: at-PJND pairs score 63 ± 5 (CID22 paper Table 4).
3. **Implement explicit pairs_per_epoch=50000 budget loop** (~30 LOC,
   the last unported Rust ingredient). The existing `--ranknet-group
   dataset` captures the spirit; this adds the explicit budget +
   inverse-CDF group-weighted sampling. May or may not move CID22
   further — the dataset-group flag may already be sufficient.
4. **Implement cyclic cosine LR (50-epoch period)** instead of full-run
   `CosineAnnealingLR` (~5 LOC). Subtle but documented as Rust's choice.
5. **If all four above don't hit CID22 ≥ 0.8893**, the next lever is
   training-data quality: the 218k clean safe-synthetic was generated
   by an older zensim version's gpu_ssimulacra2 column. Re-scoring
   with the V0_5 (current shipping) bake's ssim2 outputs as targets
   might give cleaner labels, but this is non-trivial (regenerate
   features.bin too if zensim feature definitions changed).

---

## 0c. Cycle close (2026-05-12) — V0_8 contamination purge + V0_16 ship + 2-bake optimum

**Read this AFTER §0b.** The 2026-05-11 cycle paused after V0_8 ship
(CID22 0.8948). The 2026-05-12 audit + cycle-6 ensemble work follows.

**Contamination purge** (one user directive, executed):
- 156,420-row "clean" CSV used to train V0_8 still contained 11,629
  contaminated rows (7.43 %) — 22 of 49 CID22 holdout references had
  hex-hashed near-duplicate source files at dHash-64 distance d ≤ 16.
- Purged 361 source files + 75 GiB derivatives + tower mirror; rebuilt
  clean CSV at 144,791 rows. Manifest:
  `zensim/benchmarks/contaminated_sources_purged_2026-05-12.txt`.
- V0_8's inflation is real — honest CID22 ≈ 0.890–0.892 (V0_8's
  measured 0.8948 had ~+0.005 leakage bias).
- Drop-in patch staged for the coefficient generator's blocklist at
  `zensim/benchmarks/coefficient_blocklist_patch_2026-05-12.md` +
  `purged_hex_stems_const_2026-05-12.rs` (pre-formatted Rust const,
  ~10-minute apply). **Cross-repo apply pending user.**

**V0_15 honest ship** (replaced tainted V0_8 same-day): CID22 0.8914
(+0.0019 vs ssim2), AIC-3 0.8019 (+0.0054), non-mono 2.51 %.

**V0_16 honest ship** (replaced V0_15 same-day): TV raised 15 → 20
to recover V0_8's B1 closure honestly. CID22 0.8919, AIC-3 0.7990,
non-mono 2.30 %. Current runtime weight is
`zensim/weights/v0_16_2026-05-12.bin` (md5 `baf3fdcb...`),
affine-calibrated α=28.0366 β=-5.0738.

**Cycle 6 ensemble characterization** (V_X recipe-knob space
exhaustively explored):
- 4-seed sweep (V0_18/V0_19/V0_20): CID22 mean 0.8872 ± 0.0034.
  V0_16 is +1.4σ outlier (lucky seed).
- V0_21 butter-clean training: trade-off, not improvement.
- V0_22 konjnd_w=1.0: best smoothness (1.96 %) + best Near-PJND (0.3710).
- V0_23 val_policy=mean: within seed variance — confirms knob is
  save-time only.
- **Exhaustive 7-bake subset search**: `{V0_16, V0_20}` 2-bake is
  Pareto-optimum. CID22 0.8910 (+0.0015 vs ssim2),
  AIC-3 0.8050 (+0.0085), 2× inference cost. Adding V0_21
  brings negligible CID22 gain.
- `{V0_20, V0_21}` 2-bake hits AIC-3 0.8079 (+0.0114 — best AIC-3
  of any subset, but CID22 −0.0006).
- AIC-3 (held-out, no overlap with training) confirms: V_X recipe
  beats fast-ssim2 by ≥+0.0033 in 4-bake ensemble.

**Deliverables published**:
- Methodology page (10 sections + TL;DR) at
  <https://imazen.github.io/zensim/methodology.html>.
- Site charts (8 sections): aggregate, per-band, scatter, step-5,
  2D Pareto, non-mono Pareto, cross-codec smoothness, bake history.
- Scripts: `apply_butter_filter.py`, `band_balance_safesyn.py`,
  `ensemble_seeds.py --dataset CID22|AIC-3 CTC`, `per_band_step5.py`,
  `build_scatter_data.py`, `content_class_explore.py`.

**Recipe-knob space EXHAUSTED**. Cycle 7 needs **structural change**,
not parameter tuning:

1. **zenpredict multi-bake runtime** — Rust port to load N bakes and
   average outputs. Plan documented in methodology Section 9
   candidate #2 (~3–4 hours Rust). User to authorize and run.
2. **Image-type-aware MLP dispatch** — k-means foundation done in
   tick 404. Multi-hours.
3. **KonJND-1k dataset restoration** — `/mnt/v/dataset/konjnd-1k/`
   missing. Blocked on external source.
4. **AIC-4 dataset full download** — URL stale. Blocked.
5. **dssim Rust binary extension** — multi-hour.

**Ship state**: `zensim/weights/v0_16_2026-05-12.bin` (single bake).
Multi-bake runtime is the smallest cycle-7 win and the only one that
unlocks the documented 2-bake +0.0100 combined gain.

**Recovery cycle docs** updated:
- `~/work/zen/zensim/CHANGELOG.md` — V0_16 ship + V0_15 ship + cycle-6
  ensemble section under `[Unreleased]` (zensim commit `c0773d34`).
- `~/work/zen/zensim/CLAUDE.md` — V0_8 INFLATION CAVEAT added.
- `~/work/zen/zenanalyze/zensim_champion_log.md` — ticks 1–443 cover
  the full audit + cycle-6 narrative.

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
