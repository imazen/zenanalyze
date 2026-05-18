# Screen-step seed instability — diagnosis (2026-05-17)

## The problem

`feature_transform_sweep.py --confirm` produces a single-seed
recommendation that often does not survive multi-seed confirmation:

| Codec | 1-seed --confirm | 3-seed median | Verdict reversal? |
|---|--:|--:|---|
| zenjpeg | +3.74 pp | **-6.81 pp** | YES — regress |
| zenwebp | +24.54 pp | +24.54 pp | no — ship |
| zenavif | +2.57 pp | +1.65 pp | YES — noise |

Two of three single-seed point estimates were wrong; the third was
genuinely shippable. Production decisions made off the 1-seed numbers
would have shipped two regressions.

## Why it varies

The screen step ranks (feature × transform × params) tuples by a
per-cell aggregate score (Pearson, Spearman, or z-RMSE) on a single
train/val split. With a different seed:

- Different rows land in train vs val → different per-cell distributions
- Different cells get different fold-sizes → different Pearson stability
- The top-K candidates per feature reorder

For features whose top-2 candidates have nearly identical screen
scores, this reordering is dominant. Measured across 3 seeds on the
v14+z_rmse sweep:

| Codec | Features recommended in ANY seed | Same transform in ALL 3 | Different per seed | Missing in ≥1 |
|---|--:|--:|--:|--:|
| zenjpeg | 42 | 20 | 7 | 15 |
| zenavif | 42 | 13 | **23** | 6 |
| zenwebp | 32 | 20 | 11 | 1 |

zenavif's 23/42 feature-instability is the worst, consistent with
its 3-seed `noise` verdict. zenwebp's 20/32 stable is the best,
consistent with the `ship` verdict.

## Why zenwebp is the exception

- Smallest config space: 72 candidate configs (vs zenjpeg 120, zenavif 200)
- Fewest cells: 6 (vs zenjpeg 12, zenavif 10)
- Best data-coverage ratio: 583 val rows / 72 configs ≈ 8.1 rows/config
  (vs zenjpeg 484/120 = 4.0, zenavif 218/200 = 1.1)

When each cell has more data, the per-cell aggregate is more stable
across seeds, and the screen's top-K reordering is less likely.

## Mitigation (this session)

`zentrain/tools/seed_stable_screen.py` — majority-vote across already-
run seed sweep directories. A feature is kept iff a strict majority of
seeds (≥51%) agreed on the same transform; otherwise dropped (= identity).

Applied to the 3-seed v14+z_rmse runs:

| Codec | Recommended in any seed | Stable (majority) | Dropped: no majority | Dropped: low coverage |
|---|--:|--:|--:|--:|
| zenjpeg | 42 | 31 | 0 | 11 |
| zenavif | 42 | 31 | 8 | 3 |
| zenwebp | 32 | 31 | 1 | 0 |

The seed-stable recommendations are saved under
`benchmarks/seed_stable_<codec>_v14_2026-05-17/` and adopted by
`zentrain/examples/<codec>_picker_config_v3_stable.py`.

## 3-seed confirm results on v3_stable (measured)

Each v3_stable config sets its `FEATURE_TRANSFORMS` to the majority-vote
output from above. Multi-seed `--confirm` runs them through the SAME
A/B harness; the comparison is `v3_stable transforms` vs `whatever the
screen finds for this seed`.

| Codec | v3_stable baseline_argmin median | re-screen argmin median | delta verdict | KEEP_FEATURES count |
|---|--:|--:|---|--:|
| zenjpeg | **8.1%** | 41.7% | `ship` (re-screen wins +33.6 pp) | 51 |
| zenavif | **6.5%** | 9.4% | `noise` (re-screen +2.8 pp) | 52 |
| zenwebp | **48.6%** | 41.1% | `noise` (re-screen -4.9 pp) | 33 |

**The seed-stable picks themselves are diagnostic:**

| Codec | Original config baseline (no v3) | v3_stable baseline (= stable picks applied) | Δ |
|---|--:|--:|--:|
| zenjpeg | 30.7% | 8.1% | **-22.6 pp** ⚠️ |
| zenavif | 19.2% | 6.5% | **-12.7 pp** ⚠️ |
| zenwebp | 20.8% | 48.6% | **+27.8 pp** ✅ |

The seed-stable picks **catastrophically regress** for zenjpeg /
zenavif but **substantially improve** zenwebp. The methodology is
**not codec-agnostic**.

## Root cause: feature-count compounding overfit

Inspecting `KEEP_FEATURES` counts reveals the pattern:

- **zenwebp KEEP_FEATURES = 33** (pre-pruned)
- zenjpeg KEEP_FEATURES = 51
- zenavif KEEP_FEATURES = 52

With 50+ features, the screen ranks 50 (feature, transform, params)
tuples in parallel. Even with majority-vote stability, the cumulative
effect of small per-feature spurious lifts compounds into a model that
overfits the screen's particular distribution. The same screen picks
that look stable across 3 seeds STILL encode the seed-class's split
biases when applied as a whole.

With a pre-pruned 33-feature set (zenwebp), each individual screen
pick is informative on average, and the cumulative effect generalizes.

## Updated methodological recommendations

1. **The screen-based transform sweep is conditioned on PRIOR feature
   pruning.** Do not run it on full 50+ feature sets without first
   pruning to ~30-35 features by single-feature lift on val argmin.

2. **`zenwebp_picker_config_v3_stable` (seed-stable) marginally beats
   `zenwebp_picker_config_v2` (single-seed --confirm).** Both ship;
   v3_stable is the safer choice.

3. **For zenjpeg / zenavif, the path forward is FEATURE PRUNING
   followed by transform sweep**, NOT direct transform sweep on the
   full feature set. Pruning ablation tooling exists in zentrain but
   has not been run codec-wide for jpeg/avif yet.

4. **DO NOT bake `zenjpeg_picker_config_v3_stable` or
   `zenavif_picker_config_v3_stable`.** Both files exist as
   reproducibility artifacts only.

5. The 3-seed `noise` verdict for both v3_stable runs (zenavif,
   zenwebp) means re-screening is also seed-sensitive; for zenwebp
   the v3_stable is sufficiently close to re-screen that either
   ships, but for zenavif neither approach beats the original
   no-transforms config.

## Methodological recommendations going forward

1. **Never bake on a 1-seed sweep result.** `multi_seed_confirm.py`
   must gate every transform recommendation before it touches a
   production config. The wall-time cost (~3 min for 3 seeds at
   60 epochs) is trivial vs the cost of shipping a regression.

2. **Prefer seed-stable transforms over single-seed greedy winners.**
   `seed_stable_screen.py` filters by majority vote across N seeds;
   features that don't agree across seeds are dropped. The stable
   set is smaller but more reliable.

3. **For codecs with thin data (data-rows/config < 5), expect the
   transform sweep to find noise.** zenavif's 1.1 rows/config ratio
   means most "improvements" the sweep proposes are within seed
   variance. Production improvements there require expanding the
   sweep corpus, NOT tuning transforms.

4. **Watch out for stacks specifically.** The 5 runtime-supported
   stacks (`clip_then_log1p_then_winsor`, `winsor_then_*`, etc.) had
   the highest instability rate across seeds (per the unstable-
   features dumps). Their parameterization is wider so the screen
   has more candidates to choose between for marginal lift.

5. **Diagnostic to add to the trainer**: per-cell data-density
   warning. If any cell has < 50 val rows or < 100 train rows, log
   the transform-sweep results as low-confidence regardless of
   single-seed lift.

## Files

- `zentrain/tools/seed_stable_screen.py` — majority-vote tool (new)
- `benchmarks/multiseed_<codec>_v14_2026-05-17/` — 3-seed runs of v14+z_rmse
- `benchmarks/seed_stable_<codec>_v14_2026-05-17/` — majority-vote output
- `zentrain/examples/<codec>_picker_config_v3_stable.py` — codec configs adopting stable recs
- `benchmarks/multiseed_<codec>_v3_stable_2026-05-17/` — 3-seed confirm of stable picks (pending)
