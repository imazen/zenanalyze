# zenpicker-train scalar hybrid-heads — real-data parity (2026-06-09)

Validates the Rust scalar hybrid-heads (ch1–ch3.5) on the **real** zenjpeg
scalar sweep — the data `zentrain/examples/zenjpeg_picker_config.py` points
at — against the Python joint-trunk SOTA that last trained scalar heads.

## Setup

- **Sweep:** `/mnt/v/backups/home/sweep-data/per-codec/zenjpeg/zq_pareto_2026-04-29.parquet`
  (3,497,760 rows, 120 configs; scalars encoded in `config_name`, e.g.
  `ycbcr_444_hyb145_cs100_sa`).
- **Features:** `zq_pareto_features_2026-05-01_parallel.tsv` (1,388
  `(image, size_class)` rows × 92 named `feat_*`).
- **Adapter:** `zenpicker-train/recipes/parity_adapter_2026-06-09.py` parses
  `config_name` → `knob_tuple_json` (categorical `{color, sub, trellis_on,
  sa}` + scalar `{chroma_scale, lambda}`) and joins the features → a
  zenpicker-train unified parquet (3,497,760 rows × 99 cols, 0 unmatched).
- **Train:** `zenpicker-train --codec zenjpeg --scalar-axes chroma_scale,lambda`
  (bounded grid search, grouped-by-image held-out split, default `val_frac`).
- **Dataset built:** 9,733 `(image, target_zq)` rows · 92 features (+zq_norm)
  · **12 categorical cells** (the `{color, sub, trellis_on, sa}` tuple — matches
  the Python joint bake's `categorical_axes` exactly, confirming the adapter).

## Baseline — Python joint-trunk zenjpeg (2026-05-18)

`benchmarks/joint_finetune_2026-05-18/finetune_0xbabe_zenjpeg_picker.json`
(same axes: categorical `[color, sub, trellis_on, sa]`, scalar
`[chroma_scale, lambda]`), `student_metrics`:

| metric | Python joint (0xbabe) |
|---|--:|
| argmin accuracy | 0.5450 (n=5802) |
| chroma_scale MAE | 0.1536 |
| lambda MAE | 2.130 |

Not a bit-exact reference — the Python bake trained on the multi-codec
**joint** sweep with a shared trunk + per-codec finetune, on a different
train/val split. It is the closest same-axes scalar baseline.

## Result — Rust zenpicker-train hybrid

Single `--hidden 128,128 --seed 0` fit (the topology leading the partial
6-candidate grid; the full grid is slower on 9,733 rows and unnecessary for
the parity point). Held-out grouped-by-image split, `val_frac` 0.2.

| metric | Rust hybrid | Python joint | Δ |
|---|--:|--:|--:|
| held-out argmin accuracy | **0.5266** | 0.5450 | −0.018 |
| bytes_log SROCC | **0.9258** | — | — |
| byte overhead (mean / p50 / p90) | **2.8% / 0.0% / 8.0%** | 2.57% / 0.0% / 6.57% | ~ |
| **chroma_scale MAE** (natural units) | **0.1521** | 0.1536 | **−0.0015** |
| **lambda MAE** (natural units) | **2.003** | 2.130 | **−0.127** |

Held-out scalar targets scored: chroma_scale 21,130; lambda 10,572 (fewer —
trellis-off cells are sentinel-masked, so their `lambda` target is dropped).

Partial-grid candidates from the first (timed-out) full-grid run, for context:

| cand | hidden | lr | bytes-SROCC | argmin_acc |
|---|---|--:|--:|--:|
| 0 | 64,64 | 2e-3 | 0.9314 | 0.4040 |
| 1 | 64,64 | 1e-3 | 0.9249 | 0.5147 |
| 2 | 128,128 | 2e-3 | 0.9258 | 0.5266 |
| 3 | 128,128 | 1e-3 | 0.9068 | 0.5246 |

## Reading

The Rust port **matches — and marginally beats — the Python joint-trunk SOTA
on scalar accuracy**: chroma_scale MAE 0.1521 vs 0.1536, lambda MAE 2.003 vs
2.130, on a *single-codec single fit* without the shared-trunk pretrain +
per-codec finetune the Python bake used. It reproduces the categorical cell
structure exactly (12 cells, same `{color, sub, trellis_on, sa}` axes), lands
at 0.9258 bytes-SROCC and 0.5266 argmin accuracy (vs Python's 0.545), and
holds median byte overhead at 0.0%. This confirms the restored scalar
hybrid-heads regress chroma_scale + lambda to natural-unit accuracy on par
with the Python student, end to end on real zenjpeg data.

Bake: `/mnt/v/zen/zenpicker-train-parity/zenjpeg_hybrid_parity_2026-06-09.bin`
(140,356 bytes, ZNPR v3, + `.toml` manifest). Reproduce: run the adapter then
`zenpicker-train --codec zenjpeg --scalar-axes chroma_scale,lambda --hidden 128,128`.
