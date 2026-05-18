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

## Update: zenavif depth curve (added later)

Same methodology applied to zenavif at three pruning depths:

| Codec | n_features | Δargmin median | stdev | recommended | verdict |
|---|--:|--:|--:|--:|---|
| zenavif unpruned | 52 | +1.65 pp | 5.48 | 20.8% | noise (3-seed) |
| zenavif lightprune (drop-10) | 43 | +9.13 pp | 10.22 | 24.3% | noise (5-seed) |
| zenavif aggprune (drop-20) | 35 | +2.18 pp | 10.64 | 13.7% | noise (3-seed) |
| zenavif ultraprune (drop-30) | **27** | **+9.33 pp** | **7.30** | 20.2% | **SHIP (5-seed)** |

**Surprise win**: zenavif at 27 features ships (median > stdev,
+9.33 > 7.30). The methodology DOES generalize for zenavif at
sufficient pruning depth.

But the absolute recommended argmin is bounded by data scarcity:
zenavif's 218 val rows / 200 configs = 1.1 rows/config means even
the best methodology only gets the picker to ~20-25% argmin. By
comparison, zenwebp at its sweet spot hits 45.3%.

### What zenavif's variance is doing

Across the 4 zenavif measurements, the lightprune (43 feat) recommended
hit 24.3% — better than ultraprune's 20.2%. But with much wider seed
variance (stdev 10.22 vs 7.30), the verdict couldn't lock down. This
is the data-scarcity tax: zenavif can in principle reach 24% argmin
with the right transforms, but the 218-row val set doesn't have
enough samples per config to confirm it across seeds.

Per-seed sample: lightprune saw seed_0xbabe baseline 7.1% → recommended
28.2% (+21pp). One image-set fold catastrophically picks for that
seed; another fold (0xcafe, -3.31pp) doesn't see the same lift. Until
the val corpus grows, lightprune's 24% is a coin flip.

**Conclusion for zenavif**: ship the ultraprune (27-feature) config.
Lower absolute recommended (20.2% vs 24.3%) but verdict-locked across
5 seeds. The lightprune result is "tantalizing but not statistically
confirmed" — would need more val data to lock in.

## Full cross-codec table

| Codec | n_features | Δargmin median | stdev | recommended | ship-margin | verdict |
|---|--:|--:|--:|--:|--:|---|
| zenwebp | 33 (orig) | +24.54 | 5.41 | **45.3%** | +19.13 | **ship** |
| zenwebp | 28 | +15.38 | 8.26 | 37.3% | +7.12 | **ship** |
| zenwebp | 20 | −1.20 | 5.05 | 39.4% | −6.25 | noise |
| zenjpeg | 51 (orig) | −6.81 | 1.75 | — | −8.56 | regress |
| zenjpeg | 34 | −1.16 | 4.58 | 29.6% | −5.74 | noise |
| zenjpeg | 28 | +1.03 | 3.44 | 33.8% | −2.41 | noise |
| zenavif | 52 (orig) | +1.65 | 5.48 | 20.8% | −3.83 | noise |
| zenavif | 43 | +9.13 | 10.22 | 24.3% | −1.09 | noise |
| zenavif | 35 | +2.18 | 10.64 | 13.7% | −8.46 | noise |
| zenavif | **27** | **+9.33** | **7.30** | 20.2% | **+2.03** | **ship** |

Three codec sweet spots, three different feature counts:
- zenwebp: 28-33 features (current 33 is optimal)
- zenavif: 27 features (DROP from current 52!)
- zenjpeg: no shipping depth (data + config space limit)

**Promote zenavif to drop-30 (27 features) configuration.** This is
the new SOTA for zenavif's picker methodology.

