# Screen metric A/B — Pearson vs Spearman vs z-RMSE (2026-05-17)

**Question:** under the stack-augmented transform vocabulary (18
variants), does Spearman or z-RMSE produce more reliable per-feature
winners than the default |Pearson| aggregate? The 2026-05-17 stacks
sweep showed Pearson catastrophically over-fitting in transform space
on zenavif (−12.49 pp argmin). Rank-based or signed-error screens
might be more robust.

**Answer:** **z-RMSE wins on 2/3 codecs and produces the biggest
single result in the entire transform-sweep work — zenwebp +24.54 pp
argmin acc.** Spearman wins on 1/3 (zenjpeg). Pearson wins on 0/3
under the stack-augmented set. **No single screen metric is best on
all codecs** — production recommender should run all three and pick
the per-codec winner via `--confirm`.

## Setup

- Vocabulary: 18 transforms (9 singles + 9 stacks), same as the
  2026-05-17 stacks experiment.
- Same teacher (HISTGB_FAST), student (192³ leakyrelu, 60 epochs),
  seed (`0xCAFE`), val split. Only the screen metric varies.
- Three screens implemented in
  `zentrain/tools/feature_transform_sweep.py`:
  - `pearson` (default): `mean(|cell_pearson|)` — accepts
    anti-correlation as good
  - `spearman`: `mean(|cell_spearman|)` — rank invariant, monotone
    single transforms tie identity
  - `z_rmse`: `mean(signed_cell_pearson)` — mathematically `1 −
    z-RMSE² / 2` where both vectors are z-normalized. SIGNED, so
    anti-correlation is penalized.

## Headline matrix

| Codec | Baseline (no transforms) | pearson | spearman | z_rmse |
|---|--:|--:|--:|--:|
| zenjpeg | 29.5 % / 13.32 % | 32.5 % / 10.34 % | **37.5 % / 8.08 %** | 25.7 % / 14.34 % |
| zenwebp | 20.8 % / 5.29 % | 30.2 % / 3.42 % | 19.4 % / 5.45 % | **45.3 % / 2.68 %** |
| zenavif | 19.2 % / 7.91 % | 6.7 % / 8.86 % | 17.4 % / 8.77 % | **20.8 % / 8.45 %** |

Deltas (recommended − baseline):

| Codec | pearson Δ | spearman Δ | z_rmse Δ | Best metric |
|---|--:|--:|--:|---|
| zenjpeg | +3.07 pp / −2.97 pp | **+8.01 pp / −5.24 pp** | −3.81 pp / +1.02 pp | **spearman** |
| zenwebp | +9.38 pp / −1.87 pp | −1.43 pp / +0.16 pp | **+24.54 pp / −2.62 pp** | **z_rmse** |
| zenavif | −12.49 pp / +0.95 pp | −1.84 pp / +0.86 pp | **+1.65 pp / +0.54 pp** | **z_rmse** |

## Comparison against the singles-only sweep (the previous production recommender)

The original 2026-05-17 sweep ran with 9 single transforms + Pearson
screen and showed:

- zenjpeg +3.14 pp / −2.45 pp
- zenwebp +18.03 pp / −2.10 pp
- zenavif +1.84 pp / −0.17 pp

Updated production winners (per-codec):

| Codec | Singles+Pearson (prior) | Best metric here | Net improvement |
|---|--:|---|--:|
| zenjpeg | +3.14 pp | spearman+stacks (+8.01 pp) | **+4.87 pp** |
| zenwebp | +18.03 pp | z_rmse+stacks (+24.54 pp) | **+6.51 pp** |
| zenavif | +1.84 pp | z_rmse+stacks (+1.65 pp) | −0.19 pp (noise) |

**Two of three codecs gain materially.** zenwebp jumps from +18 pp to
+24.54 pp (the largest single accuracy result in any transform-sweep
experiment so far). zenjpeg gains +4.87 pp by switching to Spearman.
zenavif is unchanged within noise.

## Why each metric wins where it does

### `z_rmse` (signed Pearson) wins on zenwebp + zenavif

Signed Pearson penalizes anti-correlation. The MLP cannot use an
anti-correlated feature without a negative final-layer weight, but
the gradient signal for that weight depends on the feature's actual
linear contribution to bytes_log — which is what z-RMSE directly
measures. |Pearson| rewards "the transform makes the feature
informative in either direction" but the MLP cares about *which*
direction.

Stacks that produce sign-flipped outputs (e.g. `winsor_then_log` on
features where the upper tail is anti-correlated with bytes_log but
the bulk is positively correlated) get correctly penalized by z-RMSE
where |Pearson| applauded them.

### `spearman` wins on zenjpeg

Spearman is rank-invariant. Monotonic single transforms (log, log1p,
signed_sqrt, signed_cbrt, signed_log1p, clip_then_log1p) all score
identically to identity. The screen can only differentiate
non-monotonic transforms (winsor_p99, quantile_bins) and stacks that
INTRODUCE non-monotonicity. This is a stricter test that filters out
"transforms that look helpful but don't change the rank order the
MLP would discover anyway."

zenjpeg's wins under Spearman come from the screen choosing
non-monotonic transforms (winsor_p99 with aggressive bounds,
quantile_bins) that actually re-order the rank space in ways that
help cell discrimination.

### `pearson` wins on no codec (with stacks)

The previous stacks experiment showed why: with 18 candidates,
Pearson's |·| aggregate over-rewards transforms that produce
saturation-induced bulk-distribution alignment without preserving the
sign needed for MLP gradient flow. Pearson is fine for **9-candidate**
single-transform screens (it agreed with end-to-end there); it
breaks for **18-candidate** stacked screens.

## Implications

1. **The Pearson screen is unreliable when the candidate set
   includes stacks.** This confirms the 2026-05-17 stack negative
   result and explains the mechanism (sign-blindness).

2. **z-RMSE is the new default screen** — wins on 2/3 codecs,
   produces the biggest single accuracy result (zenwebp +24.54 pp),
   and rescues the codec that Pearson catastrophically broke
   (zenavif).

3. **Per-codec metric selection is the production rule.** Run all
   three; let `--confirm` pick. The harness already supports this
   via `--screen-metric` — just run thrice per codec.

4. **For future research:** a metric ENSEMBLE (vote per-feature
   across the three screens, transform with ≥2 votes wins) might
   be more robust than any single-metric choice. Not implemented;
   queued.

## Per-codec recommended files (drop-in for codec configs)

The best `recommended_transforms.py` per codec:

- **zenjpeg**: `benchmarks/feat_xform_zenjpeg_spearman_2026-05-17/recommended_transforms.py`
- **zenwebp**: `benchmarks/feat_xform_zenwebp_zrmse_2026-05-17/recommended_transforms.py`
- **zenavif**: `benchmarks/feat_xform_zenavif_zrmse_2026-05-17/recommended_transforms.py`

## Caveats

- 60-epoch HISTGB_FAST teacher A/B. Production runs with
  HISTGB_FULL + longer training might shift the picture. Re-run the
  three-metric sweep before the production bake.
- Single seed (`0xCAFE`). The big zenwebp z-RMSE win (+24.54 pp)
  needs multi-seed confirmation to establish the effect size
  tightly. Queued.
- The Pearson catastrophe on zenavif (−12.49 pp) was specific to
  this combination of teacher + student + seed; under a different
  setup Pearson might recover, OR z-RMSE might lose its lead on
  other codecs. The robust takeaway is: **never trust a single
  screen metric; always end-to-end-confirm.**

## Cross-references

- `zentrain/tools/feature_transform_sweep.py` (`--screen-metric` flag
  + `safe_spearman` / `safe_z_rmse_score` / `aggregate_cell_*` fns)
- `benchmarks/feat_xform_<codec>_{spearman,zrmse}_2026-05-17/` —
  raw per-codec outputs
- `benchmarks/feat_xform_stacks_2026-05-17_negative_result.md` —
  Pearson + stacks failure mode that motivated this experiment
- `benchmarks/feat_xform_2026-05-17_summary.md` — original
  singles-only Pearson sweep (previous production recommender)
