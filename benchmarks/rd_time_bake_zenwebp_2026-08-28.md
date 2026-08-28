# `--objective rd_time` — first real bake (zenwebp canonical, 2026-08-28)

zenanalyze#56 acceptance: "one real bake produced on the standard corpus and
validated against held-out images: predicted ms within 25 % of measured ms on
the training-time CPU at p50, within 50 % at p99."

## Setup

- Data: canonical 2026-06-27 zenwebp lossy dataset (`s3://zentrain/canonical/2026-06-27/zenwebp_lossy/`,
  local `~/tmp/canonical/zenwebp_lossy/`), converted with
  `zentrain/tools/canonical_to_pareto.py` (944,370 Pareto rows, 4,497 images,
  30 VP8 cells, `encode_ms` per row). `encode_ms` was measured by the sweep
  hosts that produced the canonical dataset — that is the "training-time CPU"
  for this bake, not the machine that ran the trainer.
- Trainer: `zentrain/tools/train_hybrid.py --codec-config canonical_picker_config
  --objective rd_time --hidden 128,128,128 --seed 51966 --activation leakyrelu
  --out-suffix _rd_time_seed_cafe` (torch student, weight_decay 1e-4,
  `--time-loss-weight` at its rd_time default 0.5), origin even/odd split via
  `origin_split.py`: train 46,656 / val 28,281 / test 16,476 decision rows.
  Wall clock 4 m 20 s on an Apple M4 Pro (`nice -n 19`, 4 BLAS threads).
- Bake: `tools/bake_picker.py --dtype i8 --allow-unsafe` → 85,580 bytes,
  16 metadata entries, `zentrain.profile` = 2 (rd_time),
  `zentrain.hybrid_heads_layout` = 30 cells × head kinds `[bytes, time]`,
  `zentrain.median_cell_ms_per_mp` = 136.23, `zentrain.encode_ms_p99` =
  27 zq targets × 30 cells (min 76.7 ms, no unreached cell).
- Artifacts (not committed — > 30 KB): `~/tmp/canonical/zenwebp_lossy/models/zenwebp_canonical_hybrid_rd_time_seed_cafe.{json,bin,manifest.json,log}`;
  trainer stderr in `~/tmp/zenanalyze-rdtime-train.log`.

## Time head (student, validation rows, 841,943 reached (row, cell) entries)

| metric | value | #56 bar |
|---|---:|---|
| per-cell R² of `log(encode_ms)`, median | **0.993** (min 0.986; teacher median 0.995) | TIME_HEAD_R2 floor 0.6 |
| \|ms_pred − ms\| / ms, p50 | **6.6 %** | ≤ 25 % |
| p90 | 19.4 % | — |
| p99 | **38.9 %** | ≤ 50 % |
| max | 233 % | — |
| median encode_ms per megapixel | 136.2 ms/MP | baked for platform calibration |

Both acceptance percentiles hold with margin. The residual p99 tail is the
`m6_mpass` family on small images, where the multi-pass encoder's time is
bimodal on content the features do not separate.

## Bytes head (unchanged objective, for reference)

| | mean overhead | argmin_acc |
|---|---:|---:|
| teacher (HistGB per cell) | 5.45 % | 11.6 % |
| student, val | 4.61 % | 7.9 % |
| student, test (K=1) | 4.50 % | 7.1 % |

Compare the 2026-08-28 size_optimal arms on the same data
(`train_hybrid_backend_gap_zenwebp_2026-08-28.md`): leakyrelu+wd=1e-4 mean
overhead 4.58 ± 0.17 %, argmin_acc 12.3 ± 1.0 % over three seeds. The rd_time
bytes head lands on the same RD (4.61 %); argmin_acc is lower on this one seed
— it is a near-tie statistic on 30 cells that swings several pp between seeds,
and the RD-quality gates (`max_mean_overhead_pct`, per-zq/size p99) charge
nothing for RD-equivalent ties. Safety violations are the canonical-bench set
every size_optimal arm also raises on this dataset (PER_ZQ_TAIL, PER_SIZE_TAIL,
DATA_STARVED_SIZE, UNCAPPED_ZQ_GRID, WORST_ROW) plus LOW_ARGMIN; none is
time-head related (TIME_HEAD_R2 and BUDGET_INFEASIBLE are silent).

## What this bake is not

The canonical picker config is the apples-to-apples trainer bench (one
categorical `mode` axis, no scalar heads, no codec runtime schema); it is not a
shipping zenwebp picker. The point here is the time head's accuracy and the
bake-side contract (`hybrid_heads_layout` kinds, `median_cell_ms_per_mp`,
`encode_ms_p99`), which transfer unchanged to any codec config.

## Addendum — ceiling-aware Pareto (same day)

Re-trained with the regenerated Pareto carrying `effective_max_zensim`
(`--out-suffix _rd_time_ceil_seed_cafe`): identical decision rows (91,413),
identical bytes head (val mean overhead 4.609 %), identical time head
(R² median 0.993, |Δms|/ms p50 6.6 % / p99 38.9 %); `UNCAPPED_ZQ_GRID` no
longer fires. See `zensim_ceiling_zenwebp_canonical_2026-08-28.md` for why
the rows do not change and for the tiny-image NaN drop (636 / 4,497 keys)
that keeps every `tiny/zq*` cell DATA_STARVED on this bench.
