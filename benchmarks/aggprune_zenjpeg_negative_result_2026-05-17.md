# zenjpeg aggressive pruning — measured negative result (2026-05-17)

## TL;DR

Pruning zenjpeg's `KEEP_FEATURES` 51 → 34 (LOO-driven, drop top-17
most-harmful) shifts the v14+z_rmse sweep verdict from `regress`
to `noise` but does **not** reach `ship`. The 5-seed median is
**−1.16 pp** with stdev 4.58 pp. The original 3-seed result of
+2.72 pp was sampling luck — 2 of 3 seeds happened to land positive.

## Methodology

Same pipeline as zenwebp's successful v3_stable:
- LOO multi-seed → identify features whose removal helps argmin
- Drop top-K with `Δargmin >= 0.5 pp` and `Δoverhead <= 0.5 pp`
- Run `feature_transform_sweep.py --confirm` with `z_rmse` screen
- 5-seed `multi_seed_confirm.py`

Drop threshold K = 17 (matches the LOO's "clear signal" cutoff).
Resulting `KEEP_FEATURES`: 34 (close to zenwebp's 33).

## Results

### 5-seed verdict (the honest measurement)

| seed | baseline argmin | recommended argmin | Δargmin |
|---|--:|--:|--:|
| `0xCAFE` | 27.5% | 35.4% | **+7.89 pp** |
| `0xBEEF` | 33.2% | 36.0% | +2.72 pp |
| `0xFACE` | 24.7% | 23.3% | −1.40 pp |
| `0xDEAD` | 33.4% | 29.6% | **−3.81 pp** |
| `0xBABE` | 28.9% | 27.7% | −1.16 pp |

- Median: **−1.16 pp**
- Stdev: 4.58 pp
- Range: −3.81..+7.89
- Verdict: **noise**

### 3-seed pre-result

The first 3 seeds (`0xCAFE, 0xBEEF, 0xFACE`) gave median +2.72 pp
and looked promising. The 4th and 5th seeds (`0xDEAD, 0xBABE`)
both came in negative, dragging the median to −1.16 pp.

This is why `multi_seed_confirm.py` exists and why the
`promote_recommended_to_config.py` gate now requires aggregate.json
with `verdict == "ship"` (not just a positive 3-seed median).

## Comparison across pruning depth

| Variant | n_features | Δargmin median | stdev | verdict |
|---|--:|--:|--:|---|
| zenjpeg unpruned v14+z_rmse | 51 | −6.81 pp | 1.75 | regress (3-seed) |
| zenjpeg pruned (drop-10) | 41 | −0.20 pp | ? | noise (3-seed) |
| zenjpeg aggprune (drop-17) | 34 | **−1.16 pp** | 4.58 | **noise (5-seed)** |

Pruning monotonically improves median (less regress), but does
NOT cross zero into `ship`. The compound-overfit hypothesis from
the prior session's diagnosis (`benchmarks/screen_seed_stability_findings_2026-05-17.md`)
is real but incomplete — even at 34 features the screen-based
methodology produces noise on zenjpeg, not ship.

## Compare to zenwebp

zenwebp at 33 features ships with median **+24.54 pp** (stdev 5.41,
range +16.8..+27.2). Every seed strongly positive. That's a real
signal, not luck.

zenjpeg at 34 features: median −1.16 pp, signs split (+/+/−/−/−).
No real signal.

## Hypothesized causes (not tested in this session)

1. **Different feature distribution shape.** zenjpeg's features
   (post-percentiles, post-DCT analysis) may have heavier tails
   or more outliers than zenwebp's, making the WinsorP99 + stack
   variants behave differently.

2. **Wider config grid.** zenjpeg has 120 configs vs zenwebp's 72.
   The picker has more candidates to discriminate. Pruning features
   alone can't compensate for an 80% larger output space.

3. **Trellis/sa-aware cell structure** — zenjpeg's categorical axes
   (`color`, `sub`, `trellis_on`, `sa`) include a trellis_off
   cluster where lambda is a sentinel. The screen may pick
   transforms that work for trellis-on cells and confuse the
   trellis-off cluster.

## Decision

**Keep zenjpeg on the original (no-overrides) `zenjpeg_picker_config`.**
Pruning gets us closer to a working methodology but doesn't deliver
a multi-seed-locked ship. Future work (queued, not in this session):

- Try singles-only screen on pruned zenjpeg (drop the 5 runtime
  stacks). The stacks may compound overfit even at 34 features.
- Per-cell pruning (`trellis_off` cells use a different feature
  subset than `trellis_on`).
- Joint-trunk pretraining (the value the v3.2 design was reaching
  for, recoverable as a training-time technique per ecosystem review #5).
- Wider sweep grid for low-q regime where most production traffic lives.

## Files

- `benchmarks/aggprune_zenjpeg_v14_5seed_2026-05-17/aggregate.json` — 5-seed measurement
- `benchmarks/aggprune_zenjpeg_v14_2026-05-17/aggregate.json` — 3-seed pre-result
- `benchmarks/aggprune20_zenjpeg_v14_2026-05-17/aggregate.json` — drop-20 identical to drop-17 (only 17 features met `min-delta >= 0.2`)
- `zentrain/examples/zenjpeg_picker_config_aggprune.py` — the 34-feature config

The methodology works for zenwebp because zenwebp's data + config grid
+ feature distribution happen to align. The methodology does NOT
generalize to zenjpeg/zenavif by pruning alone.

## Follow-up experiments (also negative)

### Ultraprune at 28 features (drop-25, more aggressive)

| Seed | baseline argmin | recommended argmin | Δargmin |
|---|--:|--:|--:|
| `0xCAFE` | 35.4% | 31.0% | −4.48 pp |
| `0xBEEF` | 34.4% | 35.4% | +1.03 pp |
| `0xFACE` | 36.6% | 33.3% | −3.24 pp |
| `0xDEAD` | 31.6% | 33.8% | +2.21 pp |
| `0xBABE` | 34.2% | 37.6% | +3.33 pp |

- Median: **+1.03 pp** (stdev 3.44, range −4.48..+3.33)
- Verdict: **noise** (median < stdev)

Compared to drop-17 (34 features), drop-25 (28 features) shifts
median upward (−1.16 → +1.03) and shrinks stdev (4.58 → 3.44).
But baseline ALSO drifted up (28.9% → 34.4%), so the absolute
recommended ends up similar at 33.8%. The recommended is essentially
unchanged across pruning depths; only the baseline argmin moves.

### Singles-only equivalence (drop-17)

Running the drop-17 5-seed sweep with `--enable-stacks` OFF
(singles vocab only, 9 transforms) produces **byte-identical
results** to with-stacks: same baseline, same recommended, same
delta on every seed. The screen picks singles for zenjpeg on every
seed; stacks were never on the table.

The compound-overfit hypothesis from the prior session pointed at
the 5 stack variants as a potential culprit. Verified false: the
stacks aren't being chosen.

## Final cross-depth summary

| Variant | n_features | Δargmin median | stdev | verdict |
|---|--:|--:|--:|---|
| unpruned v14+z_rmse | 51 | −6.81 pp | 1.75 | regress (3-seed) |
| drop-10 | 41 | −0.20 pp | ? | noise (3-seed) |
| drop-17 (aggprune) | 34 | −1.16 pp | 4.58 | noise (5-seed) |
| drop-17, singles-only | 34 | −1.16 pp | 4.58 | noise (5-seed) |
| drop-25 (ultraprune) | 28 | **+1.03 pp** | **3.44** | **noise (5-seed)** |

Pruning monotonically improves median, stacks vs singles is a wash,
but no methodology configuration ships. The screen-based transform
sweep has an effective ceiling on zenjpeg around 0-3 pp of marginal
argmin lift, which is consistently inside seed-variance.

## What would unblock zenjpeg?

Things this session ruled out:
- More aggressive pruning beyond 28 features (didn't reach drop-30+
  threshold; might continue trend but compute-cost gets steep)
- Stack-variant removal (no effect; stacks never picked)

Things still untested:
- **Different screen metric.** Pearson (default) or Spearman might
  pick different transforms; z_rmse was chosen because it beat
  Pearson on zenwebp.
- **Different student capacity.** 192³ may be too small; the
  picker has 120 configs × 12 cells, larger heads might absorb
  more transform information.
- **More training data.** zenjpeg's 484-row val set with 12 cells
  means ~40 val rows/cell. Doubling val data might tighten variance.
- **Per-cell pruning.** Different cell types (trellis_on/off,
  444/420 sub) may use different feature subsets effectively.
- **Joint-trunk pretraining** (the value the v3.2 design was
  reaching for, recoverable as a training-time technique per
  ecosystem review #5).

Document this as the conclusion of the 2026-05-17 picker
methodology investigation. Future work should focus on data
expansion or architectural changes, not further screen variants.
