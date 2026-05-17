# HVS-features picker eval — measured negative result (2026-05-17)

**Question:** do the 5 HVS-derived features added to zenanalyze 2026-05-17 (chroma-luma covariance Cb/Cr, info-weight mean/p90, orientation energy ratio — ids 132-136) improve picker argmin accuracy when included in the trainer set?

**Answer: no.** All 3 codecs regress when HVS features are added to KEEP_FEATURES. The features have strong individual screen lift but adding them to a ~450-params-per-head student MLP causes overfitting that exceeds their marginal information gain.

## A/B (60-epoch HISTGB_FAST teacher + 192³ student + z_rmse screen + stacks-enabled, single seed `0xCAFE`)

| Codec | Without HVS (v14+z_rmse) | With HVS | Δ from adding HVS |
|---|--:|--:|--:|
| zenjpeg | **+3.74 pp** / −2.91 pp | −2.40 pp / +3.17 pp | **−6.14 pp argmin** |
| zenwebp | **+24.54 pp** / −2.62 pp | +22.27 pp / −3.30 pp | **−2.27 pp argmin** (mean overhead improves +0.68 pp) |
| zenavif | **+2.57 pp** / +0.60 pp | +0.18 pp / +1.68 pp | **−2.39 pp argmin** |

## Screen vs end-to-end mismatch

The HVS features rank HIGH on the Pearson screen — `feat_orientation_energy_ratio` is the single biggest screen lift in zenwebp (+0.113), top-3 in zenavif (+0.062). Individual feature × bytes_log signal is real. But the picker's MLP doesn't translate it to argmin gain:

**zenjpeg top screen winners (with HVS):**
```
rank 10: feat_chroma_luma_covariance_cr  winsor_then_signed_cbrt  lift +0.037
rank 12: feat_orientation_energy_ratio   clip_then_log1p_then_winsor  lift +0.033
rank 25: feat_chroma_luma_covariance_cb  winsor_then_log  lift +0.014
rank 42: feat_info_weight_p90            clip_then_log1p_then_winsor  lift +0.005
```

**zenwebp screen — feat_orientation_energy_ratio is rank #1** with `quantile_bins` at +0.113 lift.

But adding these to the trainer's feature set regresses end-to-end accuracy. Same pattern as the stacks negative-result experiment (2026-05-17): screen Pearson lift doesn't survive the trained MLP's gradient interactions.

## Why this happens

1. **Capacity over-run.** The student MLP head has ~450 parameters. Adding 5 new inputs increases input dim from ~110 → ~115; per-cell head capacity stays fixed; overfitting on the new signal axes.

2. **Information redundancy with existing features.** Spot-check:
   - `feat_orientation_energy_ratio` correlates with screen-content discrimination — already captured by `feat_patch_fraction_fast` and `feat_edge_slope_stdev`.
   - `feat_info_weight_*` (Wang-Li IW weights) overlap with `aq_map_p*` percentiles — different transfer function but same underlying spatial-variance signal.
   - `feat_chroma_luma_covariance_*` is the most genuinely novel of the five (no existing feature captures Y-chroma cross-correlation), but its lift is moderate (+0.037 zenjpeg, +0.054 zenwebp, +0.055 zenavif).

3. **Screen-vs-end-to-end gap widens with vocabulary size.** The screen scores each (feature × transform × params) tuple in isolation. With 14 transforms × 5 new features = 70 new candidates, the screen finds many local optima; end-to-end training surfaces conflicts between them.

## Decision

**Ship WITHOUT HVS features in production picker bakes.**

The HVS features stay in `zenanalyze` (commits `cd55dc3` proposal + `40a44ba` implementation + `de31260` benches) for:
- Future research with different MLP architectures (TabM ensembles, larger heads)
- Non-picker consumers (e.g., codec dispatch heuristics, image classification)
- Diagnostic / debugging signals

Production codec bakes (next step) use the v14+z_rmse winners from `benchmarks/feat_xform_<codec>_v14_2026-05-17/` (singles+stacks vocabulary, no HVS).

## Cross-references

- `benchmarks/hvs_feature_proposals_2026-05-17.md` — research agent's proposal
- `benchmarks/feat_xform_<codec>_hvs_zrmse_2026-05-17/` — raw per-codec HVS sweep results
- `benchmarks/feat_xform_<codec>_v14_2026-05-17/` — prior best (no HVS, v14+z_rmse)
- `benchmarks/feat_xform_stacks_2026-05-17_negative_result.md` — analogous "screen says yes, end-to-end says no" pattern from stack-only experiment

## Caveats

- 60-epoch HISTGB_FAST teacher + single seed; the multi-seed confirmation (`multiseed_zenwebp_v14_2026-05-17/`) is in flight to lock in the no-HVS baseline. HVS regression magnitudes here are point estimates.
- Different MLP capacity (larger student) might let HVS features pay back. Not tested.
- A pure HVS-only feature set (drop most of the existing 102 features, keep only HVS + dimensions + size onehot) might also work differently. Not tested.

For now: ship the v14+z_rmse winners; HVS features stay available but unused by the picker.
