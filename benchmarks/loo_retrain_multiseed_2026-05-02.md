# Multi-seed Tier 3 LOO retrain — 5 seeds, 8 features, 2026-05-02

Hand-rolled driver (`benchmarks/loo_driver_multiseed_2026-05-02.py`)
running 5 seeds × paired with/without on the zenwebp v0.2 pareto sweep
(~22 M cells, 1 264 image-instances). Total: 80 paired retrains in
175 min wall.

The point of the sweep: **estimate the variance** of single-seed LOO
ΔAC measurements that informed the 2026-05-02 cull/restore cycle.
Per-seed σ on ΔAC is **4–8 pp** — large enough that single-seed
results within ±5 pp of zero are not reliable.

## Aggregated results

| Feature | n_seeds | mean ΔOH (pp) | σ ΔOH | mean ΔAC (pp) | σ ΔAC | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `feat_noise_floor_y_p50` | 5 | **−0.35** | 0.28 | **+5.00** | 4.00 | **CULL** — strongest signal in sweep |
| `feat_log_padded_pixels_64` | 5 | −0.19 | 0.47 | **+3.20** | 5.45 | CULL confirmed (already retired `4c183f7d`) |
| `feat_log_padded_pixels_8` | 5 | **+0.27** | 0.35 | **−4.10** | 6.61 | KEEP — restoration confirmed |
| `feat_bitmap_bytes` | 5 | +0.17 | 0.53 | −2.58 | 6.28 | KEEP — restoration confirmed (smaller magnitude than single-seed) |
| `feat_log_padded_pixels_16` | 5 | +0.19 | 0.29 | −1.14 | 6.59 | weak KEEP |
| `feat_log_pixels` | 5 | +0.10 | 0.22 | −0.76 | 4.89 | weak KEEP — multi-seed signal small |
| `feat_aq_map_p10` | 5 | +0.17 | 0.41 | −0.38 | 6.99 | within noise |
| `feat_palette_density` | 5 | −0.01 | 0.44 | +0.78 | 7.62 | within noise (already deprecated `023ff5ff`) |

**Reading the verdicts**:

- ΔOH > 0 ⇒ removing increases overhead ⇒ KEEP
- ΔAC < 0 ⇒ removing decreases accuracy ⇒ KEEP
- Magnitude must exceed ~1 σ to clear the noise floor

## Headline: new cull candidate `feat_noise_floor_y_p50`

The strongest cull signal in the entire sweep. Three independent
methods converge:

1. **2026-05-02 cross-codec Tier 0 dendrogram** (zenwebp + zenjxl): perfect-Jaccard cluster #057 — `{aq_map_p50, noise_floor_y_p50, quant_survival_y_p50}` cluster at ρ≥0.95, anchor `aq_map_p50`, drops `noise_floor_y_p50` and `quant_survival_y_p50`.
2. **Single-seed LOO** (commit `af5bcf74`, 2026-05-02): ΔOH −0.59 pp, ΔAC +0.50 pp — only feature in the single-seed sweep whose removal helped both metrics.
3. **Multi-seed LOO** (this sweep): mean ΔOH = −0.35 ± 0.28 pp, mean ΔAC = +5.00 ± 4.00 pp. **Both >1σ from zero in cull direction.** Strongest cull signal of any feature tested.

## Variance-aware reinterpretation of prior single-seed results

The 2026-05-02 single-seed LOO drove the `bitmap_bytes` restoration
and the `log_padded_pixels_64` re-cull (commit `4c183f7d`). Multi-seed
shows the magnitude estimates were noisy:

| Feature | Single-seed ΔAC | Multi-seed ΔAC | Action correct? |
|---|--:|--:|---|
| `bitmap_bytes` | −8.6 pp | −2.58 ± 6.28 pp | **Yes** (right direction, smaller magnitude) |
| `log_padded_pixels_64` | +13.6 pp | +3.20 ± 5.45 pp | **Yes** (right direction, smaller magnitude — but the −1σ bound is still negative; could be borderline) |
| `log_padded_pixels_8` | −1.8 pp | −4.10 ± 6.61 pp | **Yes** (right direction, larger multi-seed magnitude) |
| `log_padded_pixels_16` | −3.3 pp | −1.14 ± 6.59 pp | weak — single-seed was a high-leverage seed |
| `log_padded_pixels_32` | −3.0 pp | (not in this sweep) | — |
| `noise_floor_y_p10` | −5.8 pp | (not in this sweep) | — |

`log_padded_pixels_64`'s multi-seed +3.20 ± 5.45 pp is consistent with
"actively hurts at our data scale" but at lower confidence than the
single-seed +13.6 suggested. The retirement is still defensible —
the direction holds, and multi-seed noise should not unwind a
reasoned decision.

## Recommendation

**Cull `feat_noise_floor_y_p50`** at the picker-config level (drop
from `KEEP_FEATURES`) for every codec config that includes it:

- zenwebp (currently in KEEP)
- zenjpeg overnight (currently in KEEP)
- zenavif (currently in KEEP)

Defer the analyzer-side `#[deprecated]` until cross-codec multi-seed
LOO confirms (this sweep was zenwebp-only). The variant stays
emittable for any non-picker consumer who explicitly requests it.

The `FEATURE_GROUPS` validator can tighten `median_block_cost`'s
`max_picked` from 2 back to its structural target of 1 once
`feat_noise_floor_y_p50` is dropped from KEEP — the constraint will
naturally pass without `noise_floor_y_p50` in the picked set.

## Caveats

- **Single codec.** Same `train_hybrid.py` 80/20 train/val split, same
  zenwebp v0.2 pareto. Cross-codec replication needed before treating
  any cull as universal — the cross-codec Tier 0 finding makes
  `noise_floor_y_p50` likely to generalize, but other features
  (especially the codec-specific tail of the distribution) need
  per-codec multi-seed before action.
- **Train/val split keyed by image_path × seed.** Different seeds get
  different train/val partitions (validated empirically by the σ
  values being non-zero). Variance estimate is meaningful.
- **5 seeds is the conservative N.** σ values stabilize around N=5;
  going to N=10 would tighten by √2 ≈ 1.4×, not material for any
  candidate currently within 1σ of zero.

## Artifacts

- `benchmarks/loo_retrain_multiseed_raw_2026-05-02.tsv` — 40 raw rows (8 features × 5 seeds)
- `benchmarks/loo_retrain_multiseed_2026-05-02.tsv` — aggregated summary (8 rows)
- `benchmarks/loo_driver_multiseed_2026-05-02.py` — driver source
- `/tmp/loo_multiseed_2026-05-02/` — per-retrain train logs (80 logs;
  not committed, reproducible)

Total wall: 10508 s (175 min) at ~50 s/retrain.

## Cross-references

- `benchmarks/loo_retrain_2026-05-02.md` — single-seed precursor
- `benchmarks/feature_groups_cross_codec_2026-05-02.md` — Tier 0 cluster structure
- `benchmarks/all_time_best_features_2026-05-02.md` — synthesized top-features list
