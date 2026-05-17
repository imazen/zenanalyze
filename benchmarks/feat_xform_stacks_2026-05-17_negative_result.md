# Stacked feature transforms — measured regression (2026-05-17)

**Question:** does adding two-step transform stacks (e.g. `winsor_p99 →
log1p`, `signed_log1p → winsor_p99`, `clip_then_log1p → winsor_p99`)
to the per-feature transform screen find better combos than the
9-variant single-transform vocabulary?

**Answer:** no. The Pearson screen happily picks stacks as per-feature
winners — 54-63 % of features chose a stack — but **the end-to-end
student MLP regresses on 2 of 3 codecs**, with one catastrophic
−12 pp drop on zenavif. The stack-augmented Pearson screen is
overfitting in transform space.

## A/B vs the singles-only sweep

Same harness, same teacher, same 60-epoch student, same val split,
same seed. Only the candidate set differs (`ALL_TRANSFORMS` = 9
singles + 9 stacks for this run; the prior run was singles-only).

| Codec | Singles-only Δ argmin | With stacks Δ argmin | Stack-induced delta |
|---|--:|--:|--:|
| zenjpeg | +3.14 pp | +3.07 pp | −0.07 pp (noise) |
| zenwebp | **+18.03 pp** | +9.38 pp | **−8.65 pp** |
| zenavif | +1.84 pp | **−12.49 pp** | **−14.33 pp** |

zenavif went from a small win to a **catastrophic loss**. zenwebp's
+18 pp standout (which depended on `quantile_bins` for
`feat_high_freq_energy_ratio`) shrank by half.

## Why stacks fool the Pearson screen

With stacks, each feature now has ~50 candidate `(transform, params)`
combinations (9 singles × avg 3 param configs + 9 stacks × avg 5
nested param configs). The Pearson aggregate picks whichever
maximizes `|Pearson(transformed_feat, bytes_log)|` averaged over
reachable cells — a greedy-per-feature score that doesn't model how
the transformed feature will interact with other features in the
MLP.

Specifically:

- **`winsor_then_log`** (12 zenjpeg + 10 zenwebp + 13 zenavif wins
  — the most common stack winner) clips the tails to `[p1, p99]`
  AND log-compresses the survivors. The resulting feature has
  flat-ended saturation on both sides; the MLP loses gradient
  signal in those regions. Pearson likes the bulk-distribution
  shape; the MLP doesn't.

- **`clip_then_log1p_then_winsor`** (7 + 8 + 8 wins) is a 3-step
  stack: subtract noise floor → log1p → clip log-domain outliers.
  Three points where signal can be lost; high Pearson because the
  middle of the distribution is cleaner.

- **The Pearson aggregate doesn't penalize information loss** — a
  feature that's a near-constant `c` after transform can still
  have non-trivial cell-wise Pearson if `c` happens to correlate
  positively with the mean of each cell's bytes_log.

The screen's own caveats section warned about this:

> Greedy per-feature. The screen scores transforms one feature at
> a time. Cross-feature interactions are NOT captured in the
> Pearson aggregate. The end-to-end confirmation step (`--confirm`)
> catches interaction-driven regressions.

With singles, the search space is small enough that the screen's
Pearson winners coincide with the end-to-end winners on all 3 codecs.
With stacks, the search space is large enough that Pearson
overfits — and confirmation catches it.

## Per-feature winner distribution (stack-augmented run)

For lift ≥ 0.005, count of features choosing each transform across
the 3 codecs:

| Transform | Wins (jpeg/webp/avif) | Single or stack |
|---|---|---|
| `winsor_then_log` | 12 / 10 / 13 | **stack** |
| `clip_then_log1p_then_winsor` | 7 / 8 / 8 | **stack (3-step)** |
| `winsor_p99` | 5 / 1 / 6 | single |
| `clip_then_log1p` | 3 / 5 / 4 | single |
| `signed_cbrt` | 2 / 4 / 1 | single |
| `quantile_bins` | 4 / 1 / 2 | single |
| `winsor_then_signed_cbrt` | 0 / 1 / 1 | stack |
| `log` | 2 / 1 / 1 | single |
| (other stacks) | 0 / 1 / 1 | stacks |
| `signed_sqrt` | 0 / 0 / 2 | single |
| `signed_log1p` | 0 / 0 / 1 | single |
| `log_then_winsor` | 0 / 0 / 1 | stack |

Stacks are **54-63 %** of winners across the 3 codecs. The Pearson
screen confidently selects them; the end-to-end run says no.

## Decision

**Don't ship stacks.** The singles-only sweep
(`benchmarks/feat_xform_<codec>_2026-05-17/`) remains the production
recommender. The runtime infrastructure for parameterized singles is
already in `zenpredict::FeatureTransform`; that's what the codec
configs should adopt.

The 9 stack functions stay in `feature_transform_sweep.py` for
future experiments — gated behind a `--enable-stacks` flag would be
cleaner, but the current code includes them in `ALL_TRANSFORMS`
unconditionally. Adding the flag is queued.

## What this implies about screen design

Two takeaways from the negative result:

1. **The Pearson screen's reliability degrades with candidate-set
   size.** With 9 candidates the screen and end-to-end win agree;
   with 50 they don't. Future expansions of the transform vocabulary
   need correspondingly tighter screen objectives or denser
   end-to-end confirmation.

2. **A rank-based screen (Spearman) might be more robust.** Stacks
   that produce flat-ended saturation have the same Spearman as
   their non-stack inner transforms (rank order is preserved), so
   Spearman wouldn't reward the saturation that fools Pearson.
   Single monotonic transforms also have identical Spearman as
   identity, which neuters single-transform discrimination — but
   the parameterized variants (winsor, clip-then-log1p,
   quantile_bins) are non-monotonic in the right places to register
   on Spearman. Worth measuring; queued.

## Caveats

- 60-epoch HISTGB_FAST teacher A/B (same as the singles run). A
  production-quality HISTGB_FULL teacher MIGHT shift the picture
  enough that some stacks pay off — but the magnitudes here
  (−12.49 pp on zenavif) make that an unlikely rescue.
- Single seed (`0xCAFE`). The catastrophic zenavif drop reproduces
  under that one seed; multi-seed confirmation would establish the
  effect size more tightly. Queued.

## Cross-references

- `zentrain/tools/feature_transform_sweep.py` — harness (now
  includes `ALL_TRANSFORMS = TRANSFORMS + STACKS`)
- `benchmarks/feat_xform_zenjpeg_stacks_2026-05-17/` — raw outputs
- `benchmarks/feat_xform_zenwebp_stacks_2026-05-17/`
- `benchmarks/feat_xform_zenavif_stacks_2026-05-17/`
- `benchmarks/feat_xform_<codec>_2026-05-17/` — singles-only
  comparison (the production recommender)
- `benchmarks/feat_xform_2026-05-17_summary.md` — singles-only
  cross-codec summary
