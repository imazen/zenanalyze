# Recovery Phase 3 — trainer port + falsification note

**Date:** 2026-05-21 (task #201)

Companion to `~/work/zen/RECOVERY_PLAN_2026-05-08.md` +
`RECOVERY_HANDOFF_2026-05-08.md`. This doc records what the 2026-05-21
agent landed against the Phase-3 plan, what the modern (2026-05-20)
state forced the agent to alter, and the honest verdict on the
plan's ship gate.

## What landed

`zentrain/tools/zensim_metric_train.py` was scaffolded with TODOs by
the 2026-05-08 recovery cycle. The 2026-05-21 agent completed ports
1–4 (the foundation set per `RECOVERY_HANDOFF_2026-05-08.md` §"Phase
3 — what to port next").

| # | Port | Status |
|---|---|---|
| 1 | `train_loop` end-to-end wiring + ZNPR v3 bake | DONE (smoke-passed) |
| 2 | `attach_zenanalyze_features` zenanalyze sidecar appender | DONE (smoke-passed) |
| 3 | `magnitude_match_term` magnitude-matching loss | DONE (smoke-passed) |
| 4 | `--low-band-oversample` sampler bias | DONE (smoke-passed) |
| 5 | FiLM heads + 5-per-class bake + manifest | NOT IMPLEMENTED (architectural — see deferral notes) |
| 6 | MoE | NOT IMPLEMENTED (architectural — see deferral notes) |
| 7 | Multi-target loss (ssim2 + butteraugli_p3) | NOT IMPLEMENTED (can be approximated via `--target-col` switching) |

### Port 1 smoke (3 epochs, h=64, canonical-2026-05-21)

```
train safesyn:.../canonical-2026-05-21/train/safesyn.parquet:1.0
train kadid:.../canonical-2026-05-21/train/kadid.parquet:0.3
val cid22:.../canonical-2026-05-21/val/cid22.parquet
val kadid:.../canonical-2026-05-21/val/kadid.parquet
target-col human_score, hidden=64, epochs=3, seed=1

Best: epoch 1 sel_metric=+0.8424; per-ds val SROCC = {cid22: 0.8424, kadid: 0.8501}
Bake: 99,207 bytes, ZNPR v3 (header byte 4 = 0x03), 372→64→1 MLP.
bake_verdict CID22 aggregate: SROCC=0.8424 PLCC=0.8404 KROCC=0.6466
                              PWRC=0.9034 Z-RMSE=0.542
```

Port 1 is a working end-to-end trainer. The Mohammadi panel above
demonstrates the bake loads through `bake_verdict` + scores
non-degenerate values on canonical-2026-05-21 features.

### Port 2 smoke

Joined a synthetic 50-ref sidecar with 3 zenanalyze-style features;
trainer correctly:
- Read TSV via `attach_zenanalyze_features`.
- Joined by `ref_basename` ← `stem`.
- Appended cols as `feat_372..feat_374`.
- Reported NaN-fraction (193,014/196,086 unjoined for the synthetic
  case where the sidecar only covered 50 refs).
- Trainer's NaN-row filter dropped unjoined rows correctly.

### Port 3 smoke

`magnitude_match_term` shape: `λ · mean((|α·target| − |pred|)²)`.
2-epoch smoke with `λ=0.1, α=0.3` on safesyn ran cleanly; magnitude
contributes ~25-30% of total loss at α=0.3 for human_score on the
[0,1] scale. Verified the loss term doesn't NaN.

### Port 4 smoke

`build_low_band_sample_weights` builds per-row weights for the
`torch.multinomial` sampler in `ranknet_loss`. Smoke with
`oversample_ratio=4.0, low_band_cutoff=0.6` on safesyn produced
non-degenerate training; RankNet pair sampler now over-represents
the B0..B5 band.

## Ports 5–7 — deferred with measured reason

### Why port 5 (FiLM) is deferred

The FiLM head architecture conditions per-layer affine on content
class, requiring a per-row `content_class` column AND a per-class
bake-manifest. canonical-2026-05-21 does NOT carry `content_class`
in the train parquets. Implementing FiLM right would require:

- Extracting content_class for the canonical safesyn rows (zenanalyze
  inference at corpus build time — ~1 hr of work).
- Adding a `FilmHead` MLP variant with a Cclass-conditional layer.
- Multi-bake output (5 .bin per class + manifest.tsv).
- Per-row runtime dispatch in zensim's `apply_mlp_scoring`.

Per the task spec wall budget (~6-8 hr total for all ports + train +
eval + ship), port 5 alone is ~half-day of work. The zensim/CLAUDE.md
"Architecture is open" allowance covers adding this, but the wall
budget does not — and the FiLM Rust implementation in `mlp_train::
FilmHead` (per recovery-handoff §"Phase 3 #5") is the better
production landing point anyway.

### Why port 6 (MoE) is deferred

The recovery handoff itself flags MoE as "unverified — train+eval
before deciding to keep". MoE adds K experts + a routing network,
significantly more state than FiLM. The architecturally-cleaner home
is the Rust trainer, which already has the substrate per Recovery
Handoff §"What's on each repo's main / branches now".

### Why port 7 (multi-target) is approximable

The current trainer's `--target-col` argument can already train
against ssim2 / cvvdp / iwssim / mix_* / pjnd_target. Switching
targets across epochs gives a poor-man's multi-target. A proper
multi-target loss (weighted sum across columns) is ~1 hour of work;
it was not landed this pass to focus on the foundation.

## Structural conflict with the recovery plan's ship gate

The RECOVERY_PLAN Phase 3 gate is **CID22 SROCC ≥ 0.8893 (V0_4
baseline)**. The 2026-05-08 plan assumes:

1. The V0_7 corpus `training_safe_synthetic_perceptual_clean.csv`
   exists and is canonical.
2. The 228-feature zensim feature schema is canonical.
3. The recovery-source worktrees (v04-mlp, v06-film, v06-moe,
   v06-rebalance, v07-e1-ablation) are on disk for line-by-line
   port reference.

**Reality check (2026-05-20 / 2026-05-21):**

1. The V0_7 CSV at
   `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_perceptual_clean.csv`
   still exists (37 MB, last modified 2026-05-12). HOWEVER it was
   produced by the **2026-05-12 d≤16 perceptual-overlap purge** which
   was REVERTED on 2026-05-14 (zensim/CLAUDE.md §"dHash threshold").
   Bakes trained against this CSV would carry the same overaggressive
   contamination flags that the revert documented as false positives.
2. The canonical training corpus is now
   `/mnt/v/zen/zensim-training/canonical-2026-05-21/` with **372
   features** plus 18+ target columns (ssim2, cvvdp, iwssim, mix_*,
   pjnd_target). The 228+3=231 V0_7 dct_hf schema is structurally
   incompatible with the canonical corpus.
3. The v04-mlp / v07-e1-ablation worktrees are **gone from disk**
   (verified: `/home/lilith/work/zen/zensim--v04-mlp/`,
   `/home/lilith/work/zen/zensim--v07-e1-ablation/` do not exist).
   They remain as branches in the main repo but the working-tree
   line-by-line port reference the recovery plan assumed is no
   longer accessible without a `jj new <branch>` checkout.
4. Current ship for the Balanced trail is V_22-mix-LARGE+iwssim s3
   (CID22 0.8324 per the modern `bake_verdict`'s 4-param-logistic
   rescale + canonical-2026-05-18 features). The Compression trail
   ships V_24-per-sample-α s4 at CID22 0.8641. Both are **less than**
   the 0.8893 V0_7 baseline number from the recovery plan — but that
   gap is partly the eval harness change (the V0_7 0.8893 was
   measured by an older `dataset_metric_baseline` decoder against
   pre-canonical features). They are NOT directly comparable, and
   `bake_verdict`'s number is the only honest current ceiling.

**Implication for the ship gate**: re-running the V0_7 recipe today
against canonical-2026-05-21 would produce a 372-feature bake
trained against the 196k-row canonical safesyn (not the 218k V0_7
CSV). Even with all 7 ports landed, the resulting bake would be:

- Trained on **different data** than V0_7 (canonical-2026-05-21
  vs `training_safe_synthetic_perceptual_clean.csv`).
- A **different architecture** input width (372 vs 228 features).
- Subject to **different evaluation** (`bake_verdict` 4-param-logistic
  Z-RMSE vs older `dataset_metric_baseline`).

That bake's CID22 SROCC ≥ 0.8893 gate would be **meaningless in
isolation** — it would either trivially exceed the gate (because
the canonical-2026-05-21 corpus is richer than V0_7) or trivially
fail it (because the new harness is more demanding). The gate as
written in the recovery plan does not survive 12 days of corpus +
harness evolution.

## What WAS measurable

The smoke-trained 3-epoch bake at `/tmp/smoke_recovery_phase3.bin`
scores CID22 SROCC=0.8424 on `bake_verdict` — **same harness as
current ship**. That's:

- −0.022 vs Compression trail ship (V_24-per-sample-α s4 @ 0.8641)
- +0.010 vs Balanced trail ship (V_22-mix-LARGE+iwssim s3 @ 0.8324)

A 3-epoch, h=64, single-MLP, no-magnitude-match, no-low-band-oversample
control bake **already ties the Balanced ship on CID22**, with no
intervention from ports 2/3/4. Whether ports 2/3/4 + longer training
(50-100 epochs) actually exceed both ship gates is testable — but
the V0_7 recipe (228+3=231 features, KADID/TID at 0.3 weight,
specific magnitude-match alpha) is not directly applicable to the
372-feature corpus.

## Recommended next steps (NOT executed this session)

If the user wants to continue the recovery-phase-3 line:

1. **Sweep h ∈ {64, 128, 192} × lr ∈ {1e-3, 3e-3} × seed ∈ {1..5}**
   on canonical-2026-05-21 with the new trainer. Targets:
   `--target-col mix_cv40_iw60` (the active Compression trail
   target) and `--target-col human_score` (synth ssim2 anchor).
   5-seed CI per (h, lr) cell, eval via `bake_verdict` full panel.

2. **Test ports 3 + 4 effect size**: train a h=128 baseline (port 1
   only) + h=128 with `--magnitude-match-lambda 0.1
   --magnitude-match-alpha 0.3` + h=128 with
   `--low-band-oversample 4.0 --low-band-cutoff 0.6`. Compare
   CID22 B3/B4/B5 lift across the three.

3. **Bake-compare against current ship**: the modern `bake_verdict`
   harness produces apples-to-apples panel-vs-panel comparisons.
   No retraining of the current ship needed; just compare new bakes
   against the current ships' published bakes.

4. **Per the user's CLAUDE.md feedback "feedback_no_prs_no_branches"**:
   if any new bake passes either trail's gate per `SOTA_TRAILS.md`
   §A.9 decisive rule, commit direct to zensim main and update the
   SOTA_TRAILS.md ship matrix in the same commit.

If the recovery-phase-3 ports are sufficient as a research /
ablation foundation (which is the scaffold's stated role per its
own module docstring), no further action is required from this
task.

## Files touched

- `zentrain/tools/zensim_metric_train.py` — implemented ports 1–4
  (was scaffolded only).
- `zentrain/RECOVERY_PHASE3_2026-05-21.md` — this doc.

## Commit shas

- zenanalyze: see commit attached to this file via `jj log`.
- zensim: no edits this session (zensim ship rotation is gated on
  ports actually beating current ship, which requires a training
  sweep beyond this session's wall budget).
