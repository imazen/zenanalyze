# zenjpeg picker — feature ablation / LOO (2026-06-01)

Permutation-importance + top-K study on the zenjpeg source-feature quality
picker, to find which of its 108 zenanalyze inputs are load-bearing vs noise.

## Setup

- **Data**: `/mnt/v/zen/picker-dense-full-2026-05-27/parquet/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet`
  → 8,205 `(image, zq)` rows × 109 inputs (108 source features + `zq_norm`),
  36 categorical cells. Split by **image**: 320 train / 64 val (6,572 / 1,633 rows).
- **Teacher**: per-cell `HistGradientBoostingRegressor` (the same estimator the
  shipped distillation teacher uses), predicting `bytes_log` per cell.
- **Decision metric** (held-out): masked **argmin over predicted `bytes_log`** vs
  the oracle (argmin over *true* `bytes_log`). Reported: argmin-acc (exact-cell),
  **mean byte-overhead** of the picked cell vs oracle (the real quality signal —
  many cells are near-tied so exact-argmin is low by construction), p95 overhead,
  and pooled bytes-SROCC.
- Harness: `scripts/ablation/{build_dataset,ablation}.py`. Raw outputs:
  `recipes/picker_ablation_{importance,topk,deaddrop}_*`. The 772 KB row cache
  (`picker_ds.npz`) stays at `/mnt/v/zen/picker-ablation-2026-06-01/cache/`.

## Finding 1 — the hdr/alpha block is provably inert

Dropping features that carry no signal changes the picker by **exactly zero**:

| set | K | argmin | overhead | bytes-SROCC |
|---|---|---|---|---|
| full | 108 | 0.2468 | 0.0436 | 0.9885 |
| drop 8 `std=0` (hdr+alpha) | 100 | 0.2468 | 0.0436 | 0.9885 |
| drop 17 dead-for-corpus | 91 | 0.2468 | 0.0436 | 0.9885 |

Confirms the earlier variance check: the 5 hdr + 3 alpha features (`peak/p99_luminance_nits`,
`hdr_headroom_stops`, `hdr_pixel_fraction`, `hdr_present`, `alpha_present/used_fraction/bimodal_score`)
are constant on this SDR/opaque corpus and contribute nothing. **A picker retrained
without them needs no `zenanalyze/hdr` feature** and resolves a smaller set.

## Finding 2 — the picker OVERFITS; pruning to ~20–40 features improves it

Retraining the teacher on the top-K features by importance (held-out):

| label | K | argmin | Δargmin | overhead | Δoverhead | bytes-SROCC |
|---|---|---|---|---|---|---|
| full108 | 108 | 0.2468 | — | 0.0436 | — | 0.9885 |
| top100 | 100 | 0.3007 | +0.054 | 0.0366 | **−0.0070** | 0.9877 |
| top80 | 80 | 0.2817 | +0.035 | 0.0392 | −0.0044 | 0.9885 |
| top60 | 60 | 0.2970 | +0.050 | 0.0381 | −0.0054 | 0.9893 |
| top40 | 40 | 0.2970 | +0.050 | 0.0381 | −0.0054 | **0.9893** |
| **top20** | **20** | **0.3056** | **+0.059** | **0.0355** | **−0.0081** | 0.9880 |

Every prune **reduces** mean byte-overhead and **raises** argmin accuracy vs the
full 108. top-20 is best on both (overhead 3.55% vs 4.36%, argmin +5.9pp); top-40
also nudges bytes-SROCC up (+0.0008). The full feature set is feeding the model
noise inputs it overfits — the bottom of the importance ranking has **negative**
permutation importance (removing those features *helps*).

The curve isn't perfectly monotonic (top100 < top80 on overhead) — these are
single-split point estimates, so treat the *direction* (prune helps, ~20–40 is
the sweet spot) as solid and confirm the exact K with cross-validation before
shipping a retrained bake.

## Finding 3 — what's load-bearing

Top features (all directly bytes-relevant): `pixel_count`, `laplacian_variance`,
`distinct_color_bins`, `aq_map_p75/p50/p99`, `noise_floor_uv_p75`,
`cr_peak_sharpness`, `laplacian_variance_p99`, `quant_survival_y`, `edge_density`,
`gradient_fraction_smooth`. Full ranking in `picker_ablation_importance_2026-06-01.tsv`.

## Recommendation

Retrain the picker on the **top ~20–40 features** (CV-selected K). Expected wins,
all measured here on held-out data:
- **lower byte-overhead** (~3.5–3.8% vs 4.36%) and higher argmin accuracy,
- **drops the `zenanalyze/hdr` dependency** (hdr features are all in the dead tail),
- **fewer features to resolve** → strictly more forward-compatible,
- a smaller bake.

Not yet done — the shipped bake is still the 108-feature one. This is the next
chunk: rebuild the dense dataset with the top-K columns, re-distill + bake via the
existing pipeline, re-verify on the held-out split, and swap the bake (the picker
resolves features by name, so the new bake just declares its smaller `feature_order`).
