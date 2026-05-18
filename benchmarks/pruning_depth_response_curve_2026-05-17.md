# Pruning-depth response curve (2026-05-17)

Cross-codec measurement of how aggressively to drop LOO-harmful
features. Answer to: "why not ultraprune if no downside?"

## Measurement

5-seed multi_seed_confirm on v14+z_rmse sweep at three pruning depths
per codec. Drop targets selected by `build_pruned_config.py` from the
2026-05-03 LOO multiseed TSV, sorted by `mean_delta_argmin_pp`
descending (features whose removal helps argmin most).

| Codec | n_features | Δargmin median | stdev | recommended | verdict |
|---|--:|--:|--:|--:|---|
| **zenwebp unpruned** | 33 | **+24.54 pp** | 5.41 | 45.3% | **ship** (3-seed) |
| zenwebp aggprune (drop-5) | 28 | +15.38 pp | 8.26 | 37.3% | **ship** (5-seed) |
| zenwebp ultraprune (drop-13) | 20 | **−1.20 pp** | 5.05 | 39.4% | **noise** (5-seed) |
| zenjpeg unpruned | 51 | −6.81 pp | 1.75 | — | regress (3-seed) |
| zenjpeg aggprune (drop-17) | 34 | −1.16 pp | 4.58 | 29.6% | noise (5-seed) |
| zenjpeg ultraprune (drop-25) | 28 | +1.03 pp | 3.44 | 33.8% | noise (5-seed) |

## The downside of ultraprune is real

The naive intuition: if pruning helps, more pruning helps more. The
measured curve says NO.

**zenwebp clearly has a sweet spot.** At 33 features the screen
recommendation hits 45.3% argmin and the lift over baseline is the
strongest measured (+24.54 pp). Dropping 5 features (28 → ship) costs
8 pp absolute recommended argmin. Dropping 13 features (20 → noise)
costs both the verdict and another 6 pp absolute. The 33-feature set
is **already pre-pruned** to the right operating point — additional
pruning removes load-bearing features.

**zenjpeg never ships.** At any pruning depth (51 / 34 / 28), the
median delta stays within seed variance. Pruning shifts the median
monotonically upward (−6.81 → −1.16 → +1.03) but never crosses
`median > stdev`. The picker has a structural noise floor that
neither pruning nor transforms can break.

## Visualization

```
                  recommended argmin %
  zenwebp:   ====  45.3% (33 feat, +24.54 pp ship)
             ===   37.3% (28 feat, +15.38 pp ship)
             ===   39.4% (20 feat, −1.20 pp noise)

  zenjpeg:   ==    25.7% (51 feat, regress)
             ==    29.6% (34 feat, noise)
             ===   33.8% (28 feat, noise)
```

For zenwebp, the highest absolute recommended argmin is at the most
features. For zenjpeg, it's at the fewest. Two different codecs need
two different operating points.

## Why the downside exists

When you drop a feature, you lose the picker's ability to discriminate
on whatever signal that feature carried. If the feature was actively
harmful (LOO says argmin improves on removal), dropping it helps.
If the feature was load-bearing — even just modestly — dropping it
hurts.

For codecs whose data + config grid let the screen find robust
signal (zenwebp), most features are load-bearing and pruning hurts
quickly. For codecs whose data + config grid is too sparse for the
screen to find robust signal (zenjpeg, zenavif), most features are
noise and pruning helps modestly but never enough.

**LOO single-feature ablation doesn't capture pairwise / group
interactions.** A feature may show up as `+5pp` "helps when removed"
in isolation but be **load-bearing in combination with another
feature** that's also in the keep list. The LOO ranking ranks
features by their solo removal effect, not their net contribution
to the trained picker.

## Practical recommendation

Treat pruning depth as a **hyperparameter to sweep**, not a knob to
ratchet up indefinitely:

1. Start at the codec's existing `KEEP_FEATURES` count
2. Run multi-seed confirm at current depth
3. Try drop-N at N ∈ {5%, 10%, 25%, 50%} of features
4. Pick the depth that maximizes `median - stdev` (the "ship margin")
5. **If no depth crosses `median > stdev`, accept noise and don't
   bake the recommended transforms** — keep the codec on its
   original config

For zenwebp: stay at 33 features (current shipping config) — pruning
hurts. For zenjpeg: stay at 51 features (original config) — no depth
ships. For zenavif: same (data scarcity dominates).

The methodology converges on a per-codec sweet spot, not on a
universal "more prune = better" rule.

## Files

- `benchmarks/aggprune_zenwebp_v14_2026-05-17/aggregate.json`
- `benchmarks/ultraprune_zenwebp_v14_2026-05-17/aggregate.json`
- `benchmarks/aggprune_zenjpeg_v14_5seed_2026-05-17/aggregate.json`
- `benchmarks/ultraprune_zenjpeg_v14_2026-05-17/aggregate.json`
- `benchmarks/aggprune_zenjpeg_singles_only_2026-05-17/aggregate.json`
- `benchmarks/aggprune_zenjpeg_negative_result_2026-05-17.md` (full
  zenjpeg writeup)
