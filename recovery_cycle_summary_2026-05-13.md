# Recovery cycle summary — 2026-05-13

This document is the zenanalyze-side index for the zensim recovery
cycle that ran 2026-05-07 → 2026-05-13. The detailed
per-cycle outcomes docs live in `~/work/zen/zensim/benchmarks/`;
this file is a high-level cross-reference for future agents
resuming this work.

## Goal

Match or exceed `fast-ssim2`'s CID22 SROCC (0.8895) per the
zensim CLAUDE.md "Training goals" #1 (locked 2026-05-11):
*match-or-exceed fast-ssim2 across all quality bands*.

**V0_16 SHIP** already meets this at aggregate (CID22 0.8919).
The recovery cycle searched for a candidate that could improve
on V0_16, particularly on:
- Per-band coverage (especially B0 and B3 where ssim2 is weaker)
- JPEG-AI codec ranking (cycle-7's original motivation)
- Cross-corpus generalization (AIC-3 + AIC-4)

## Final state (2026-05-13 03:28 UTC)

**V0_16 SHIP retained as the runtime weight** — not unseated by
any recovery candidate. The 0.020-CID22 gap between V0_16 and
the best recovery candidate (V0_38) appears to be in unrecoverable
per-run state (split seed, batch sampling order).

**Live comparison site** (https://imazen.github.io/zensim/) now
hosts 4 zensim variants for user toggling:

| Bake | Recipe | CID22 | AIC-3 | AIC-4 | Best for… |
|---|---|--:|--:|--:|---|
| V0_16 SHIP | unknown extras | **0.8919** | 0.7965 | 0.9127 | balanced (gold standard) |
| V0_26 | + KonJND w=1.0 | 0.8639 | 0.8027 | 0.9097 | JPEG-AI codec |
| V0_31 | + KonJND w=0.5 | 0.8628 | 0.8031 | **0.9176** | AIC-4 (cross-codec) |
| V0_38 | + KonJND+KADID+TID | **0.8817** | 0.7969 | 0.9027 | B0/B3 low-q + lossless |

## Cycle outcomes (5 docs in zensim/benchmarks/)

| Cycle | Lever | Verdict | Doc filename |
|---|---|---|---|
| 7 | dssim co-training / cosine LR / smaller LR | FALSIFIED | `cycle_7_dssim_outcomes_2026-05-12.md` |
| 8 | KonJND-weight Pareto sweep | PARTIAL (V0_31 wins AIC-4) | `cycle_8_konjnd_pareto_outcomes_2026-05-13.md` |
| 9 | Low-q row-weight MSE boost | FALSIFIED | `cycle_9_lowq_boost_outcomes_2026-05-13.md` |
| 9b | Low-q RankNet pair-loss boost | FALSIFIED | `cycle_9b_pair_boost_outcomes_2026-05-13.md` |
| 10 | KADID+TID mixed supervision + sub-leverage | **VERIFIED** (V0_38) | `cycle_10_kadid_tid_outcomes_2026-05-13.md` |

## Tick log range

All 30 ticks of the recovery cycle live in
`zensim_champion_log.md` in this repo:

- Cycle-7 work: ticks 482–497 (approximately; V0_X numbering
  came from earlier sessions)
- Cycle-8 work: ticks 502–507
- Cycle-9 work: ticks 508–511
- Cycle-9b work: ticks 513–515
- Cycle-10 work: ticks 516–527

## Trainer infrastructure delivered (3 new flags)

All added to `zensim/scripts/v_next/train_v_next_mlp.py`, dormant
by default (no behavior change unless invoked):

| Flag | Commit (zensim) | Cycle | Purpose |
|---|---|---|---|
| `--low-q-boost <float>` | `4b998258` | 9 | Multiply train_weight for B0/B1 rows by factor |
| `--low-q-pair-boost <float>` | `a700b10f` | 9b | Weight RankNet pair-loss by max(boost_i, boost_j) for B0/B1 endpoints |
| `--tv-pairs-file <path>` | `c4cacfba` | 10h | Load pre-built adjacent-q TV pairs from TSV (filters out-of-range indices) |

## Key lessons recorded for future cycles

### 1. Single-seed comparisons mislead when Δ < 1σ

Cycle-9 (V0_34) and cycle-9b (V0_pairboost seed=1) both showed
"big wins" at single-seed comparisons that vanished or reversed
when multi-seed sweeps were run. Always run ≥3 seeds + Welch's
t-test before declaring "X is a real improvement".

### 2. Seed variance of V_X MLP recipe

On the V_X 228-feat MLP trained for 300 epochs at h=128 + V0_31
base recipe:
- CID22 SROCC: σ ≈ 0.004 (V0_31 family) → σ ≈ 0.007 (V0_kadid_tid family)
- AIC-4 SROCC: σ ≈ 0.003 across all variants
- AIC-3 SROCC: similar to CID22, σ ≈ 0.005

Use these as priors when comparing future bakes.

### 3. Data axis dwarfs recipe axis

Cycle-7/8/9/9b recipes all sit within ~0.01 CID22 of each other.
Adding KADID+TID DATA in cycle-10 lifted +0.013 CID22 (p=0.033).
The data axis is where the real signal lives; recipe knobs
(boost, LR schedule, init, val_policy) are second-order at best.

### 4. V0_16's exact recipe contains unrecoverable per-run state

After 10 cycles of attempts, V0_16's CID22 0.8919 remains +3.0σ
above the V0_kadid_tid 8-seed distribution mean (0.8712, σ=0.0068).
The remaining 0.020 gap is in:
- V0_16's specific train/val split seed
- V0_16's specific batch sampling order
- Possibly an undocumented preprocessing/calibration step

Cannot reproduce V0_16 exactly in autonomous mode without
serialized training state. **V0_16 SHIP stays as ground truth;
recovery candidates are alternatives, not replacements.**

## Cycle-11 strategic options (requires user direction)

The autonomous recovery cycle has exhausted cheap recipe levers
on the existing data. Future cycles need a strategic pivot:

1. **Data acquisition** — JPEG-AI public test corpus, more
   recent AIC challenges, additional human-MOS datasets. Cycle-10a
   showed +0.013 CID22 from adding KADID+TID; more data could
   compound.
2. **Architecture changes** — 300-feat input (vs current 228) is
   the cheapest. Requires zensim profile.rs + bake-format changes
   (multi-tick). Could also try deeper MLP (192→128 or 256→128
   hidden).
3. **Different training objective** — current is `mse_rank`
   (MSE + RankNet). Could try learn-to-rank-only, or contrastive
   pair losses (NT-Xent, etc).
4. **Ensemble at runtime** — currently V0_16 is single bake. An
   ensemble of {V0_16, V0_26, V0_31, V0_38} could win per-band
   coverage without a single new training run.

## How to resume this work

If a future agent picks up the recovery cycle:

1. Read this summary.
2. Read the 5 cycle outcomes docs in `zensim/benchmarks/`.
3. Read `zensim_champion_log.md` in this repo for tick-level history.
4. Note that `safe_synth_clean_features_with_dssim_qc.csv` (at
   `/tmp/zensim_loop/`) IS the 144k V0_16-clean training data
   (just renamed).
5. The cycle-10a recipe (V0_kadid_tid) is the strongest
   reproducible Pareto improvement; default to it as the
   experimental baseline.

Standing baseline (V0_kadid_tid, n=8 seeds): CID22 0.8712 ± 0.0068,
AIC-4 0.9046 ± 0.0027.

## Artifacts inventory

All bakes preserved in `/tmp/zensim_loop/bakes/`:
- 30+ ZNPR binaries from cycle-7 through cycle-10i
- Compatible with `dataset_metric_baseline --v04-bake` runtime

Run dirs in `/mnt/v/zen/zensim-training/2026-05-07/runs/`:
- ~30+ directories with `model.pt`, `meta.json`, `predictions_val.parquet`

Per-pair eval CSVs in `/tmp/zensim_loop/v0_*_per_pair.csv`:
- ~30 files with (dataset, reference, distorted, codec, version,
  human_score, v02_distance, v04_distance, fast_ssim2_score, butter_3norm)

Site live commits (zensim main):
- `ab7d6fd8` — V0_26 merged
- `115b1020` — V0_31 merged + dropdown
- `4edc426c` — V0_38 merged + dropdown
- `27c86ee1` — cycle-8 outcomes doc
- `0adaacdc` — cycle-9 outcomes doc
- `a855419f` — cycle-9b outcomes doc
- `b311aa6e` — cycle-10 outcomes doc
- `f791a778` — cycle-10 closure section
- `4b998258` — `--low-q-boost` flag
- `a700b10f` — `--low-q-pair-boost` flag
- `c4cacfba` — `--tv-pairs-file` flag
