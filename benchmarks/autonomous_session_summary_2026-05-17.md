# Autonomous 4-hour session summary (2026-05-17)

User directive: keep going autonomously for 4 hours on the
transform-sweep / HVS-features picker pipeline.

## Shipped commits (chronological)

1. **`fdbd246`** — `examples/augment_features_with_hvs.rs` + per-codec
   eval configs (`zen{jpeg,webp,avif}_picker_config_hvs.py`).
   Augmenter ran on all 3 codec features TSVs (1388 + 1264 + 448
   rows) producing `<codec>_features_2026-05-17_hvs.tsv` with the 5
   new HVS feature columns.

2. **`98a5904`** — `multi_seed_confirm.py` harness + zenwebp HVS
   config import fix.

3. **`bd7e083`** — **HVS features regress picker argmin on all 3
   codecs** (measured negative result). Doc:
   `benchmarks/hvs_features_picker_eval_2026-05-17.md`.
   - zenjpeg: −6.14 pp vs no-HVS baseline
   - zenwebp: −2.27 pp
   - zenavif: −2.39 pp
   - Despite `feat_orientation_energy_ratio` ranking #1 in zenwebp's
     screen (+0.113 lift), end-to-end accuracy regresses.
   - Verdict: ship without HVS features.

4. **`2ff7916`** — v2 production codec configs
   (`zen{jpeg,webp,avif}_picker_config_v2.py`) +
   `promote_recommended_to_config.py` helper. v2 configs adopt the
   sweep winners: zenjpeg 36 transforms, zenwebp 31 transforms,
   zenavif 41 transforms.

5. **`d711416`** — **rescued `feat_spectral_slope_y` (id 137)** from
   the interrupted agent worktree (agent hit usage limit). 1/f
   exponent per block, Field 1987 math, 3 unit tests pass. zenanalyze
   test count: 149 → 152.

6. **`2ea0768`** — **multi-seed confirmation of zenwebp v14+z_rmse**:
   3 seeds × 60 epochs each, median Δ argmin **+24.54 pp** (stdev
   5.41 pp, range +16.80..+27.21); mean overhead median −2.65 pp ±
   0.17. Verdict: **ship**.

## Production training results (HISTGB_FULL + leakyrelu student)

Student argmin / student mean-overhead. Production train uses
HISTGB_FULL teacher and full 60-epoch leakyrelu student (vs
HISTGB_FAST + 60-epoch in the sweep --confirm step).

| Codec | sweep --confirm | full production train | Δ student argmin (vs sweep) | teacher | student vs teacher |
|---|--:|--:|--:|--:|--:|
| zenwebp | 45.3% / 2.68% | **58.7% / 1.36%** | +13.4 pp | — | — |
| zenjpeg | 47.1% / 2.91% | **55.5% / 2.21%** | +8.4 pp | 54.7% / 1.99% | **student > teacher** ✅ |
| zenavif | 21.8% / 8.51% | 28.4% / 3.73% | +6.6 pp | 31.7% / 3.55% | **student < teacher** ⚠️ |

Bake JSONs at `zen<codec>/benchmarks/zen<codec>_hybrid_v2_2026-05-17.json`.

**Safety violations per codec:**

- **zenwebp v2:** 2 pre-existing data-coverage issues:
  - `DATA_STARVED_SIZE` — 46 size×zq cells with < 50 train rows
  - `UNCAPPED_ZQ_GRID` — zq=94 in ZQ_TARGETS but no `effective_max_zensim` column
- **zenjpeg v2:** 4 violations:
  - 2× `PER_ZQ_TAIL` at zq=92 (p99 99.6%) and zq=94 (p99 114.7%) — tail-quality fails 80% threshold (NEW concern, not just data coverage)
  - `DATA_STARVED_SIZE` — 36 cells with < 50 train rows
  - `UNCAPPED_ZQ_GRID` — zq=100 without `effective_max_zensim`
- **zenavif v2:** 3 violations:
  - `LOW_ARGMIN` — val argmin 28.4% < 30% threshold (NEW concern, not data coverage)
  - `DATA_STARVED_SIZE` — 63 cells with < 50 train rows
  - `UNCAPPED_ZQ_GRID` — zq=94 without `effective_max_zensim`

Per imazen/zenanalyze#51, `DATA_STARVED_SIZE` + `UNCAPPED_ZQ_GRID`
are codec-sweep-harness gaps shared with the current shipped bakes.
`zenjpeg` PER_ZQ_TAIL and `zenavif` LOW_ARGMIN are genuine quality
concerns — should not paper over with `--allow-unsafe` without
investigation.

## Cumulative SOTA per codec (final)

| Codec | Sweep best metric | Recommended config | Δ argmin (single seed) | Confidence |
|---|---|---|--:|---|
| zenjpeg | singles + z_rmse | `zenjpeg_picker_config_v2.py` | +3.74 pp | 1-seed |
| zenwebp | v14 + z_rmse | `zenwebp_picker_config_v2.py` | **+24.54 pp** | **3-seed locked** |
| zenavif | v14 + z_rmse | `zenavif_picker_config_v2.py` | +2.57 pp | 1-seed |

## What's NOT in production-ready state

- **zenjpeg and zenavif multi-seed confirmation** not run (only zenwebp). 1-seed point estimates only.
- **Production .bin bakes** not produced — `bake_picker.py` needs `--allow-unsafe` to clear safety violations. Bake JSONs exist (`zen<codec>/benchmarks/zen<codec>_hybrid_v2_2026-05-17.json`).
- **zenavif `LOW_ARGMIN`** — student 28.4% is the worst of the three codecs and below the 30% safety threshold; teacher is also weak at 31.7%. Needs investigation before shipping — possibly hard-example mining, capacity bump, or different effort-encoding.
- **zenjpeg `PER_ZQ_TAIL`** — p99 overhead at zq=92/94 is 99.6%/114.7%. Picker is making catastrophic choices in the rare high-zq cells; investigate before bake.
- **HVS features in picker:** measured negative, do not ship. HVS features stay in zenanalyze 0.2.1+ (ids 132-136, 137 with spectral_slope) for non-picker consumers.

## Per-codec next steps (queued)

1. Run `multi_seed_confirm.py` on zenjpeg + zenavif v2 configs to
   3-seed-lock those too. ~10 min wall each.
2. Address `DATA_STARVED_SIZE` / `UNCAPPED_ZQ_GRID` violations by
   either (a) lowering `ZQ_TARGETS` max or (b) having the codec
   sweep harness emit `effective_max_zensim`. Then `bake_picker.py`
   without `--allow-unsafe`.
3. zenwebp shipped: copy `zenwebp_hybrid_v2_2026-05-17.bin` into
   `zenwebp` crate via `include_bytes!`. Codec runtime applies the
   FEATURE_TRANSFORMS automatically — see
   `zenpredict::Predictor::predict_transformed`.

## Test count + coverage

- zenanalyze: 152 lib tests pass (149 before HVS + 3 new for spectral slope; HVS agent landed 8 tests earlier)
- zenpredict + zenpredict-bake: 253 tests pass (5 new stack variants + bake validation + integration tests landed earlier)
- All commits: `cargo test`, `cargo fmt --check`, `cargo clippy --features experimental --all-targets` — all green.

## Files touched (~30+ commits this autonomous session)

- `zenanalyze/src/feature.rs`, `tier3.rs`, `tests.rs` — HVS features + spectral slope
- `zenanalyze/examples/augment_features_with_hvs.rs` — corpus-feature augmenter
- `zenanalyze/examples/hvs_feature_smoke.rs`, `hvs_marginal_cost.rs` — agent's eval examples
- `zenpredict/src/feature_transform.rs` — 14 transform variants (9 singles + 5 stacks)
- `zenpredict-bake/src/composer.rs` + `json.rs` — bake-side validation + JSON wire
- `tools/bake_picker.py` — Python forwarder for all 14 transforms + params
- `zentrain/tools/feature_transform_sweep.py` — Pearson/Spearman/z-RMSE screens, stacks-enabled flag
- `zentrain/tools/multi_seed_confirm.py` — N-seed confirmation harness
- `zentrain/tools/promote_recommended_to_config.py` — v2 config generator
- `zentrain/examples/*_v2.py` — production codec configs
- `zentrain/examples/*_hvs.py` — HVS-eval configs (kept for reproducibility)
- `benchmarks/*` — ~25 result directories with per-codec sweeps + raw TSV + recommended_transforms.py
