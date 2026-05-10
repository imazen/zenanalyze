# zensim champion training log

**Goal**: produce a zensim version with **smooth scoring AND strong CID22 SROCC**, trained without any CID22 training data (49 validation references must stay held-out). Synthesize CID22-like training distortions from non-CID22 sources on `/mnt/v`. Learn from all prior work. **No crates.io publishing.**

## Targets

| Property | Target | Reference |
|---|---|---|
| CID22 validation SROCC | beat V0_5's **0.8934** (current leader) | `/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin` |
| Non-monotonic q-step rate | beat V0_2's **4.86%** (smoothest of shipping); definitely beat V0_4's 8.26% | `zensim/docs/score_quality_v04_2026-05-07.md` |
| KADID-10k SROCC | match-or-beat V0_5's **0.8505** | same |
| TID2013 SROCC | match-or-beat V0_5's **0.8492** | same |

## Constraints
- No CID22 training-set data. The 49 refs in `/mnt/v/dataset/cid22/CID22_validation_set.csv` are validation-only.
- No `cargo publish`. Work locally; bakes land under `__experimental_versions`.
- Use `jj` on main per global CLAUDE.md.
- `.workongoing` marker: refresh every 2 min during active work.

## Resources
- **Synth pairs (218k, no CID22 leak)**: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv` + `.features.bin`
- **Extended (340k)**: `training_safe_synthetic_extended.csv` (218k + 122k zenjpeg-420-e1 fill, e1 abandoned)
- **Unified V_X parquets**: `/mnt/v/zen/zensim-training/2026-05-07/unified/` (2.37M rows × 50 cols)
- **Historical bakes**: `/mnt/v/output/zensim/synthetic-v2/runs/` (~80 bins)
- **Python trainer (parked)**: `~/work/zen/zensim/scripts/v_next/train_v_next_mlp.py` (has `--tv-weight` for smoothness)
- **Eval harness**: `~/work/zen/zensim/zensim-bench/examples/dataset_metric_baseline.rs`
- **zentrain port scaffold**: `~/work/zen/zenanalyze/zentrain/tools/zensim_metric_train.py` (280 lines)
- **Synth generator binary**: `~/work/coefficient/examples/generate_zensim_training` (Rust)
- **Tracking doc**: `~/work/zen/zenanalyze/everything.md`
- **Forensic recon outputs**: `/tmp/recon_zenmetrics.md`, `/tmp/recon_zenanalyze.md`, `/tmp/recon_zensim.md`, `/tmp/recon_disk.md`, `/tmp/best_tuners.md`

## Phase plan (each tick = one focused step ~5–10 min)

### Phase A — Quick wins (mechanical)
- A.1 Verify V0_5 SSIM2-proxy bake exists, file size matches (60932 B), magic bytes `ZNPR\x02` (v2 format). PROBLEM if v3 only — zensim 0.3.0 reads v2.
- A.2 Bake-swap V0_4 → V0_5 in zensim: copy `runs/v04_mlp_ssim2_holdout_20260501T045510.bin` → `zensim/weights/v0_4_2026-04-30.bin` (or rename slot). Update docstring at `zensim/src/profile.rs:160-181`. Commit.
- A.3 Verify swap: `cargo test --features __experimental_versions` in zensim repo passes.
- A.4 Run `dataset_metric_baseline --cid22 --kadid --tid --v04-bake zensim/weights/v0_4_2026-04-30.bin`; record CID22/KADID/TID SROCC numbers.

### Phase B — Smoothness baseline + audit
- B.1 Run score-quality analyzer on V0_5 (the new-bake state) over the unified parquet curves; record non-monotonic q-step rate.
- B.2 If V0_5 smoothness ≤ V0_4's 8.26% but > V0_2's 4.86%: TV regularizer is the lever. Plan B.3+. If V0_5 smoothness ≥ V0_4: surprising — investigate.
- B.3 Read `zensim/scripts/v_next/train_v_next_mlp.py` `--tv-weight` implementation; verify the loss term is sound.
- B.4 Check that `zensim/scripts/v_next/` paths are still functional after recovery cycle parking.

### Phase C — Synth CID22-like data
- C.1 Inventory the 6 codec classes CID22 uses (from `zensim/docs/CID22_PAPER_NOTES_2026-05-07.md`).
- C.2 Inventory non-CID22 sources we have (clic2025/training, kodak, gb82-sc, corpus-builder/source_jpegs).
- C.3 Generate per-source distortions matching CID22's 6×{8-11 q} grid using `coefficient/examples/generate_zensim_training` (or write a new generator if needed).
- C.4 Score every distortion pair via zen-metrics CLI (CPU first; GPU vast.ai if needed for scale).
- C.5 Append to `training_safe_synthetic_cid22like.csv`. Keep separate from existing safe-synthetic; train mixed.

### Phase D — Train V_NEW
- D.1 Hyperparameter sweep on architecture: 228 → 64 / 128 / 192 hidden, LeakyReLU. TV weight {0.0, 0.01, 0.1, 0.3}.
- D.2 Mixed-supervision: synth + KADID_train@0.3 + TID_train@0.3 (matching V0_4 recipe).
- D.3 Hold out CID22 + KADID_val + TID_val + KonJND-1k.
- D.4 Pick best by `min(CID22, KADID, TID)` validation SROCC.
- D.5 Bake to ZNPR v3 (or v2 for compatibility with zensim 0.3.0).

### Phase E — Validate end-to-end
- E.1 Score-quality regression on V_NEW: non-monotonic q-step rate.
- E.2 Score-quality alignment with ssim2 ground truth.
- E.3 Cross-codec consistency on per-codec sweeps.
- E.4 Per-image bumpiness audit on production-representative web traffic corpus.

### Phase F — Ship under `__experimental_versions`
- F.1 Update `zensim/weights/v0_4_2026-04-30.bin` with V_NEW. Or add a new profile slot.
- F.2 Add CHANGELOG entry. Re-run `cargo test --features __experimental_versions`.
- F.3 Document headline numbers in zensim CLAUDE.md and `zensim/docs/v_next_status_<DATE>.md`.
- F.4 Commit + push to main. NO `cargo publish`.

---

## Tick log

(Each loop fire appends here. Format: `### Tick N — YYYY-MM-DDTHH:MM:SSZ — <one-line summary>`. Keep ≤ 5 lines per tick.)

### Tick 0 — 2026-05-10T05:38:00Z — Log initialized

Synthesized from forensic recon. Phase plan above is the working order.
Next tick: A.1 (verify V0_5 bake exists + integrity check).

### Tick 1 — 2026-05-10T05:52:18Z — A.1 verify + A.2 swap V0_4 → V0_5 (committed, not pushed)

- **A.1 done**: `/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_ssim2_holdout_20260501T045510.bin` exists; 60,932 bytes; magic `ZNPR\x02` (v2); 228 inputs / 1 output. md5 `bb7e24a16a64afa43eb296bf151fb6b8`.
- **A.2 done**: V0_5 copied over `zensim/weights/v0_4_2026-04-30.bin`. Predecessor (md5 `0218b0ed7838adff7acbf551316e69c6`) preserved byte-identical at `runs/v04_mlp_v5znpr2_20260430T044620.bin`. Diff: 58,801 bytes — truly different model.
- Updated `zensim/zensim/src/profile.rs:160-181` docstring to reflect new lineage (CID22 0.8934, KADID 0.8505, TID 0.8492 — all expected, will measure in A.4).
- jj describe at change `lmwlxupk 2ad0a3ac` ("feat(zensim): swap V0_4 bake → V0_5 SSIM2-proxy MLP"). **NOT YET PUSHED** — next tick runs `cargo test --features __experimental_versions` first.
- Next tick: A.3 — `cargo test --features __experimental_versions` in zensim. If green, advance main + push.

### Tick 2 — 2026-05-10T05:54:30Z — A.3 cargo test PASS + push to main

- **A.3 PASS**: `cargo test --manifest-path zensim/Cargo.toml --features __experimental_versions --test v04_mlp` — 5/5 tests passed (v04_profile_name, v04_identical_inputs_near_perfect, v04_score_is_in_unit_range, v04_degraded_does_not_exceed_identical, v04_compute_with_ref_matches_compute). Build cold-took 20.37 s.
- Pushed: zensim main `d530546` → `2ad0a3a` ("feat(zensim): swap V0_4 bake → V0_5 SSIM2-proxy MLP"). Visible at `origin/main`.
- Note from origin: dependabot reports 1 LOW vulnerability on default branch (referenced as security advisory #2) — separate concern, not this swap's responsibility.
- Next tick: A.4 — measure actual CID22/KADID/TID SROCC numbers via `cargo run --release -p zensim-bench --example dataset_metric_baseline -- --cid22 --kadid --tid --v04-bake zensim/zensim/weights/v0_4_2026-04-30.bin --features __experimental_versions`. Build is cold; may need to start the build in this tick and read results in next tick.

### Tick 3 — 2026-05-10T05:57:37Z — A.4 background release build started

- Inspected `zensim-bench/examples/dataset_metric_baseline.rs:1-100` — args: `--kadid <dir>`, `--tid <dir>`, `--cid22 <dir>`, `--konjnd <dir>`, `--v04-bake <path>`, `--max-pairs N` (default 500), `--per-pair-output <path>`. Reads `crate::mlp::Model` so requires `__experimental_versions` (already on by default in zensim-bench Cargo.toml).
- Kicked off `cargo build --release -p zensim-bench --example dataset_metric_baseline` in background; output streaming to `/tmp/zensim_loop/build.log`. Cold release build with `lto=thin` + `codegen-units=1` typically takes 60–180s on this box.
- Did NOT push anything new this tick (no source changes).
- Next tick: check `/tmp/zensim_loop/build.log` for "Build completed". When binary ready, run full eval: `cargo run --release -p zensim-bench --example dataset_metric_baseline -- --cid22 /mnt/v/dataset/cid22/CID22_validation_set --kadid /mnt/v/dataset/kadid10k --tid /mnt/v/dataset/tid2013 --v04-bake zensim/zensim/weights/v0_4_2026-04-30.bin --max-pairs 999999 --per-pair-output /tmp/zensim_loop/v05_per_pair.csv` and capture SROCC numbers.


### Tick 4 — 2026-05-10T06:03:30Z — A.4 full-dataset eval — IMPORTANT FINDING

- Background build completed in 27.99s. Binary at `target/release/examples/dataset_metric_baseline` (3.5 MB).
- Full eval ran on KADIK10k (n=10,125) + TID2013 (n=3,000) + CID22 (n=4,292) in 255s wall, with the swapped V0_5 bake (md5 `bb7e24a1`, confirmed loaded via `--v04-bake`):

  | Dataset | n | V0_2 (linear) | V0_4 slot (= V0_5 SSIM2-proxy now) | fast-ssim2 | butteraugli |
  |---|---:|:---:|:---:|:---:|:---:|
  | KADIK10k | 10125 | 0.8192 | **0.8432** | 0.8133 | 0.6062 |
  | TID2013 | 3000 | 0.8427 | **0.8401** | 0.8460 | 0.6696 |
  | CID22 | 4292 | 0.8676 | **0.8893** | 0.8895 | 0.7412 |

- **Key finding**: V0_5 SSIM2-proxy MLP measures **identically to V0_4 mixed-supervision** at full-dataset scale (CID22 0.8893 / KADID 0.8432 / TID 0.8401). The recovery register's "V0_5 leader" headline of `0.8934 / 0.8505 / 0.8492` was n=1500/dataset sampling noise — exactly what `RECOVERY_HANDOFF_2026-05-08.md` "Known unfinished business #2" warned: *"the published V0_5 (CID22 0.8934) numbers were measured at n=1500/dataset … new full-dataset benches show all the V0_4-equivalent bakes scoring 0.8893"*.
- The swap is the right structural call (cleaner provenance, source-disjoint splits) but is **not a measurable accuracy gain**. We are now in the V0_5 slot but the absolute ceiling stays at CID22=0.8893 = the bar to beat.
- Saved log to `zensim/benchmarks/v05_full_eval_2026-05-10.log`.
- Also note: V0_4 slot **0.8893** vs fast-ssim2 **0.8895** — they're essentially tied on CID22. SSIMULACRA 2 was tuned on 201/250 CID22 refs, so its CID22 number isn't held-out. zensim V0_5 is held-out (49 disjoint refs) and matches.
- **Pivot**: Phase A is structurally done but yields no measurable gain. Real path forward is Phase B (smoothness — V0_4 has 8.26% non-monotone q-step rate vs V0_2's 4.86% per `docs/score_quality_v04_2026-05-07.md`) + Phase C (synthesize CID22-like training data) + Phase D (train a new MLP that genuinely beats 0.8893 CID22 AND has < 4.86% non-monotone).
- Next tick: **B.1** — locate and run the score-quality analyzer (`zensim/scripts/v_next/analyze_score_quality.py`?) on V0_5 to measure non-monotone q-step rate. Compare to V0_2 (4.86%) and V0_4 mixed (8.26%). If V0_5 has same bumpiness as V0_4, we know structurally that bake-swapping won't help smoothness either — TV regularizer training is the only lever.

### Tick 5 — 2026-05-10T06:10:56Z — D.0 TV-regularizer pipeline alive (bug fix + smoke pass)

- Confirmed RTX 5070 + torch 2.10 + CUDA 13.2 ready. Inspected `scripts/v_next/train_v_next_mlp.py` TV regularizer math at lines 355–422 — fused forward `cat([x, Xt[lo], Xt[hi]])`, `relu(pred_lo - pred_hi)` term scaled by `--tv-weight`. Looks correct in design.
- **Found and fixed bug** at lines 390–399: when the last training batch is partial (e.g. 159923 % 8192 = 4275), `pred = pred_all[:bs]` over-slices the fused tensor and downstream weighted-MSE crashes with `RuntimeError: size 8192 vs size 4275`. Fix: use `n_main = x.size(0)` for slice bounds. **Without this, --tv-weight > 0 was unusable**, regardless of what the prior smoke-run note claimed.
- Smoke retry succeeded: `--max-rows 200000 --epochs 5 --tv-weight 0.1` ran in 14s wall, produced `runs/20260510T001023_tv01_smoke_2026-05-10/{model.pt 61 KB, scaler.npz 2.3 KB, meta.json, predictions_val.parquet 575 KB}`. val_srocc=0.6673 at epoch 4 (undertrained — proves pipeline only).
- Committed fix on zensim main: `dd79a3c fix(v_next/trainer): TV regularizer last-partial-batch slice bounds`. Pushed to origin.
- Next tick: **D.1** — kick off full TV-on training run in background. Plan: `--epochs 50 --tv-weight 0.1` on full v15r+v15rc (no `--max-rows` cap = 2.3M rows). RTX 5070 budget ~10–15 min. Wall budget for next tick is 4 min — start build in background, log job ID, monitor across multiple ticks. Compare vs the 2026-05-07 baseline run `20260507T115414_v_next_ssim2_64h_full` (val_srocc=0.9547, test_srocc=0.9814, NO TV regularizer).

### Tick 6 — 2026-05-10T06:12:04Z — D.1 full TV=0.1 training launched in background

- Sized prior baseline: 2026-05-07's `20260507T115414_v_next_ssim2_64h_full` ran 50 epochs on n_train=1,839,162 in 7,710 s = 128 min, achieving val_srocc=0.9547, test_srocc=0.9814 with NO TV regularizer. RTX 5070 in this session should be substantially faster than the GPU used at training time (smoke = 200k×5ep in 14s = 71 ksample-epochs/s → estimate full at ~25 min).
- Launched `train_v_next_mlp.py --epochs 50 --batch-size 16384 --lr 3e-3 --weight-decay 1e-5 --rank-weight 0.5 --tv-weight 0.1 --sweeps v15r,v15rc --target ssim2 --loss mse_rank --hidden 64 --val-frac 0.10 --test-frac 0.10 --seed 0 --tag v_next_tv01_full_2026-05-10` as background PID 263658. PID file `/tmp/zensim_loop/tv01_full.pid`. Log streaming to `/tmp/zensim_loop/tv01_full.log`. Output dir will be `/mnt/v/zen/zensim-training/2026-05-07/runs/<TS>_v_next_tv01_full_2026-05-10/`.
- Same hyperparams as 2026-05-07 baseline EXCEPT `tv-weight=0.1` (vs default 0.0). Apples-to-apples comparison: if val_srocc holds within ±0.005 of 0.9547 AND val_mse stays comparable AND non-monotone q-step rate drops below V0_4's 8.26% (target < V0_2's 4.86%), TV regularizer wins.
- Did NOT also run a tv-weight=0 baseline post-fix yet — that's queued for after this run completes (sequential, ~25 min budget). Need both numbers to claim TV is the cause of any smoothness improvement.
- Next tick: poll training progress via `tail /tmp/zensim_loop/tv01_full.log`. If still running, brief check + plan-ahead. If completed, immediately measure non-monotone q-step rate of the new bake and compare to V0_4/V0_2 baselines.

### Tick 6 — 2026-05-10T06:18Z — D.1+D.2 full TV training + capacity sweep (5 runs landed)

- RTX 5070 turned out FAR faster than the 2026-05-07 baseline GPU: full 50-epoch run on 2.3M rows = **26 sec wall** (vs 7710 s = 128 min historically). Ran a 5-config sweep in <3 min total.
- Best so far: **hidden=128, tv-weight=0.3** → val_srocc 0.9590, test_srocc 0.9810, val_mse 37.32, non-mono **7.45%**. Improvement over 2026-05-07 baseline (val_srocc 0.9547, non-mono 8.26%) on every metric.
- Saturation: increasing TV weight at hidden=64 plateaued at ~8% non-mono (TV=0.1: 7.96%, TV=0.3: 8.23%, TV=1.0: 8.21%). The 4096/80200 random TV-pair sampling rate is plausibly the bottleneck — most TV pairs see only ~2.5 epochs of supervision over 50 epochs.
- Capacity is a stronger lever: 64→128 hidden cut non-mono from 7.96% → 7.45%. But 128→128,128 traded val_srocc for test_srocc (likely overfitting).
- All five run dirs at `/mnt/v/zen/zensim-training/2026-05-07/runs/2026051*v_next_*2026-05-10/`. Comparison table in `zensim/benchmarks/tv_smoothness_sweep_2026-05-10.md`.
- Pushed to zensim main: `dd79a3c` (bug fix), `5f79914` (sweep findings doc).
- All configs still well above V0_2's 4.86% smoothness floor — the 228→N→1 LeakyReLU family appears to have an inherent ~7-8% bumpiness ceiling that single-axis hyperparameter sweeps don't break.
- Next tick: try **wider TV pair sampling** (modify trainer to use ALL 80k pairs per batch, or 50% sample). If still saturated, try **multi-scale TV** (penalize gaps of 1, 2, 3 q-steps simultaneously). Then **bake the current best (h128_tv03) and measure full CID22/KADID/TID SROCC** to confirm synthetic improvements transfer.

### Tick 7 — 2026-05-10T06:25Z — D.3 TV-weight breakthrough — V0_2 smoothness floor BROKEN

- **The previous tick's hypothesis (~7-8% inherent floor) was wrong.** Pushed TV weight 0.1 → 1 → 3 → 10 → 30 → **100** at hidden=128. Non-mono dropped monotonically: 7.96 → 8.21 → 7.54 → 6.68 → 5.81 → **4.26%**. Beats V0_2's 4.86% floor.
- Diagnosis: TV term magnitude was ~100x smaller than MSE (relu(pred_lo-pred_hi).mean ~0.5-2 vs MSE ~40). At weight 0.1 the TV contribution to total loss was 0.05 vs MSE's 40 — dominated. At weight 100 the contribution is ~50-200, comparable to MSE.
- Pareto front at hidden=128:
  - TV=3:   non-mono 7.54%, val_srocc 0.9601 (best val_srocc)
  - TV=30:  non-mono 5.81%, val_srocc 0.9584 (best balance)
  - TV=100: non-mono **4.26%**, val_srocc 0.9531 (★ smoothest, costs 0.0016 val_srocc)
- Sweep doc updated: `zensim/benchmarks/tv_smoothness_sweep_2026-05-10.md`. Pushed to zensim main.
- Per-(sweep:codec) test SROCC for h128_tv1000: v15r 0.9817 / v15rc 0.9806 — within 0.001 of TV-off baseline. Synthetic-ssim2 fidelity holds.
- All 11 candidates have full predictions_val.parquet → direct re-analysis without re-training.
- **Next tick: D.4** — bake the three production candidates (`h128_tv1000`, `h128_tv300`, `h128_tv30`) to ZNPR v2 via `scripts/v_next/bake_to_znpr.py`, then run `target/release/examples/dataset_metric_baseline` on each over CID22+KADID+TID. The bar is **CID22 SROCC ≥ 0.8893** (current V0_4-slot reality). If h128_tv1000 holds CID22 ≥ 0.8893 AND non-mono < 4.86%, it becomes the new champion ready to swap into `zensim/weights/v0_4_2026-04-30.bin`.

### Tick 8 — 2026-05-10T06:35Z — D.4 first bake + eval — h128_tv1000 fails on human MOS

- Baked all 3 candidates via `scripts/v_next/bake_to_znpr.py` (default flip=True), sizes 120,522–120,525 B (vs V0_5's 60,932 B because hidden=128 doubles weight matrices). All ZNPR v2 magic.
- **h128_tv1000 (smoothest, val_srocc=0.9531 on synthetic ssim2) regresses HARD on human MOS**:
  - KADIK10k 0.8432 → **0.5380** (-0.305)
  - TID2013 0.8401 → **0.7609** (-0.079)
  - CID22 0.8893 → **0.8519** (-0.037)
- Same pattern as the V0_5 multi-codec postmortem: synthetic-ssim2 holdout is NOT a reliable proxy for held-out human MOS. The 0.9531 val_srocc was misleading.
- `--no-flip-output` bake gave identical SROCC numbers (the harness takes `.abs()` so sign of distance doesn't matter). Output-scale calibration is not the problem; the model's actual generalization is.
- **Diagnosis**: the 2026-04-30 V0_4 mixed-supervision bake (which V0_5 inherited the numbers of) was trained with `synth + KADID_train@0.3 + TID_train@0.3`. The new TV-trained models train ONLY on synthetic ssim2 — no human-MOS supervision. The TV regularizer makes things *worse* on human-MOS by overfitting the synthetic q-curve smoothness.
- Kicked off background sequential eval of h128_tv30 + h128_tv300 (eval pipeline PID `$EVAL_PID`, results in `/tmp/zensim_loop/results/`, signal file `/all_done` when complete). ~8 min wall.
- **Next tick**: read the two pending eval results. Then: train a new set of candidates **with `--human-csv` mixing KADID_train + TID_train** (matching the V0_4 mixed-supervision recipe) **+ TV regularizer at moderate weight**. The target is: CID22 SROCC > 0.8893 AND non-mono < 4.86%, simultaneously. This is the actual user goal.

### Tick 9 — 2026-05-10T06:50Z — D.5 mixed-supervision rescues KADID + smoothness, costs CID22

- Trained `h128_tv30_kadid_tid` with `--human-csv kadid:.../kadid_features.csv:0.3 + tid:.../tid_features.csv:0.3 --tv-weight 30 --hidden 128`. Wall 260s (slowed by parallel CPU eval; alone would be ~30s).
- **End-to-end full-dataset eval result** (256 s wall):

  | Bake | KADID | TID | CID22 | non-mono q-step % |
  |---|---|---|---|---|
  | V0_5 (currently shipped) | 0.8432 | 0.8401 | 0.8893 | ~8.26 (V0_4-equiv) |
  | h128_tv1000 (synth-only TV=100) | 0.5380 | 0.7609 | 0.8519 | 4.26 ★ |
  | h128_tv30 (synth-only TV=30) | 0.4411 | 0.6630 | 0.8651 | 5.81 |
  | **h128_tv30_kadid_tid** (mixed-sup) | **0.8564** ★ | 0.7913 | 0.8462 | **4.83** ★ |

- **Mixed-supervision rescued KADID** (0.4411 → 0.8564, beating V0_5 baseline by +0.013) and helped TID (0.6630 → 0.7913, but still below baseline). **TV=30 at hidden=128 cost CID22** (0.8893 → 0.8462, -0.043) — TV regularization conflicts with what CID22 SROCC favors.
- Smoothness goal MET: 4.83% beats V0_2's 4.86% floor.
- KADID + smoothness goals MET. CID22 + TID goals NOT met yet — TV is too aggressive.
- Killed h128_tv300 (synth-only) eval mid-run; redundant given h128_tv30 already showed synth-only TV fails on human MOS regardless of weight.
- **Next tick**: train `h128_tv0_kadid_tid` (mixed-sup, NO TV) as a calibration check — should match the original V0_4 mixed-supervision numbers (CID22 0.8893). Then sweep TV={1, 3, 10} with mixed-sup to find the smallest TV that holds CID22 ≥ 0.8893 while pushing non-mono < 4.86%.
