# Context handoff — zenanalyze, 2026-05-02

Session worked through (in order): a 3-feature cull, multi-axis corpus
expansion, 4-codec ablation, IndexedPaletteWidth → PaletteLog2Size
redefinition, LogPixels/BitmapBytes cull-then-restore cycle, multi-seed
LOO sweep, 7-component audit, rapid-iteration plan, orchestration
architecture (R2 + vast.ai substrate pivot), multi-MLP safety plan
with optional slope-tuned bake, and zentrain lifecycle orchestration
design. Total ~30 commits to `main`.

**Current main HEAD**: `91bfbcf5` (zentrain lifecycle orchestration doc)

## Read these first (in order)

1. `benchmarks/orchestration_architecture_2026-05-02.md` — substrate (R2 + vast.ai + storage-as-queue, replacing GCP entirely per user decision)
2. `benchmarks/zentrain_orchestration_lifecycle_2026-05-02.md` — `zentrain orchestrate` CLI; multi-session safe via content-hash determinism; pluggable workers via R2 capability advertising
3. `benchmarks/multi_mlp_safety_default_path_2026-05-02.md` — single MLP per (codec, metric) for Default + MetricTimeMasked + RdSlope via runtime masking; 5-class safety chain; ParamClamp + KnownGoodFallback; **optional slope-tuned bake** per codec where RD/time tradeoff matters (zenavif, zenjxl, zenwebp-high-effort)
4. `benchmarks/rapid_iteration_plan_2026-05-02.md` — original single-machine plan; supersedes its Phase 2 with the orchestration-layer docs above; Phase 0 (wire 3 idle pickers) still valid
5. `benchmarks/all_time_best_features_2026-05-02.md` — synthesized top-15 features + per-codec history
6. `benchmarks/feature_groups_cross_codec_2026-05-02.md` — zenwebp ↔ zenjxl dendrogram comparison; 6 perfect-Jaccard clusters
7. `benchmarks/loo_retrain_multiseed_2026-05-02.md` — 5-seed LOO; variance estimates ±5–7pp on ΔAC
8. `benchmarks/cross_codec_aggregate_2026-05-02.md` — 4-codec deferred-feature validation
9. `benchmarks/audits-2026-05-02/*.md` — 10 component audits (zenjpeg, zenwebp, zenavif, zenjxl, zenpicker_zenpredict, zensim, zentrain_zenanalyze, coefficient, squintly, coefficient_zenjpeg_opt)
10. `zentrain/INVERSION.md` — Tier 1 orchestrator (refresh_features.py landed); Tiers 2–4 mostly subsumed by lifecycle doc
11. `zentrain/PRINCIPLES.md` — picker training invariants

## Substrate decision (user-driven)

**GCP entirely abandoned.** Migrating to:
- **Cloudflare R2** (object storage, S3-compatible, free egress within Cloudflare)
- **vast.ai** (rented spot GPU/CPU compute, ~5–7× cheaper than GCP)
- **Storage-as-queue** (R2-only coordinator, no managed services)

coefficient repo at `~/work/coefficient/` already integrates all 4
zen-codecs via the `Codec` trait + `CodecRegistry`. Existing
`oracle-d2-store` substrate has 75 k encodings + 108 k metric records.
zentrain's per-codec piecemeal pipelines are largely obsolete
because coefficient is the lab.

squintly at `~/work/squintly/` is the psychovisual ground-truth
producer (Bradley-Terry from human pair-comparisons across viewing
conditions). Exports zenanalyze-compatible `pareto.tsv` (subjective
θ) and `thresholds.tsv` (q_threshold per condition; v0.2 stub).

## Feature history (zenanalyze 0.1.0 since session start)

**Active feature count progression**: 102 → 99 (3-feature cull) → 99 (IndexedPaletteWidth retired, PaletteLog2Size added) → 102 (LogPixels + LogPaddedPixels{8,16,32,64} restored) → 102 (BitmapBytes restored, LogPaddedPixels64 retired) → **102 currently**.

### Currently retired (in `RESERVED_RETIRED_IDS`)

| ID | Was | Retired | Reason |
|---:|---|---|---|
| 11 | `DistinctColorBinsChao1` | pre-0.1.0 | Chao1 collapsed to raw count under full-scan |
| 27 | `TextLikelihood` | 2026-05-01 (`e5c3c39`) | composites flag deleted |
| 28 | `ScreenContentLikelihood` | same | `PatchFraction` AUC=0.88 outperforms |
| 29 | `NaturalLikelihood` | same | composites |
| 30 | `IndexedPaletteWidth` | 2026-05-02 (`248b48bb`) | replaced by `PaletteLog2Size` (id 121) — codomain `{0, 2, 4, 8}` lacked 1-BPP case + JXL high-cap |
| 45 | `LineArtScore` | 2026-05-01 (`e5c3c39`) | composites |
| 60 | `BitmapBytes` (briefly) | culled 2026-05-02 → **restored same day** (`4c183f7d`) — strongest LOO signal in experiment, ΔAC −8.6 pp |
| 64, 66 | `BlockMisalignment{16,64}` | 2026-04-30 | Spearman 0.96/0.998 with `_8`/`_32` anchors |
| 94–100 | 7 mathematical transforms of `pixel_count` (log2, log10, sqrt, log_pixels_rounded, log_bitmap_bytes, log_min_dim, log_max_dim) | 2026-05-01 (`e5c3c39`) | constant-factor scalings of LogPixels/PixelCount; tiny MLP can recover via linear scaling |
| 104 | `LogPaddedPixels64` | 2026-05-02 (`4c183f7d`) | LOO ΔAC +13.6 pp (actively hurt — 64×64 grid past useful resolution) |
| 117 | `ChromaKurtosis` | 2026-05-01 (`e5c3c39`) | 3-of-4 codecs Tier-0 redundant |
| 118 | `UniformitySmooth` | same | Spearman 0.94 with `Uniformity` on 3/4 codecs |
| 119 | `FlatColorSmooth` | same | Spearman 0.93 with `FlatColorBlockRatio` on jxl |

### Currently deprecated (active but `#[deprecated]`)

| ID | Feature | Migration |
|---:|---|---|
| 12 | `PaletteDensity` | use `DistinctColorBins` (raw count) or `PaletteLog2Size` (discrete BPP). 3-codec Tier-0 ρ=1.0 with `distinct_color_bins`; multi-seed LOO no signal beyond count |
| 62 | `LogAspectAbs` | use `AspectMinOverMax` (bounded `(0,1]`). Cross-codec perfect-Jaccard cluster #001 with `aspect_min_over_max` |

### Active features that survived all culls (the "all-time best" 15)

```
# Resolution / shape (4)
feat_pixel_count       (id 56) — #1 zenjpeg ablation +4.89pp
feat_aspect_min_over_max (id 61)
feat_min_dim           (id 58)
feat_bitmap_bytes      (id 60) — restored after wrong cull, LOO ΔAC -8.6pp

# Palette (1)
feat_distinct_color_bins (id 10)

# Texture / edge (5)
feat_laplacian_variance  (top zenwebp Δ≥0.20pp)
feat_laplacian_variance_p75
feat_edge_density       (zenwebp Tier 1.5 top-5)
feat_edge_slope_stdev   (zenjpeg Tier 1.5 #3 +0.740pp)
feat_patch_fraction     (zenwebp Tier 1.5 #1 +0.470pp; AUC 0.88 screen-vs-photo)

# Encoder spend (3)
feat_uniformity
feat_aq_map_std
feat_quant_survival_y   (zenjpeg Tier 1.5 #1 +0.885pp)

# Noise floor (1)
feat_noise_floor_y      (zenavif Tier 1.5 #1 +0.514pp; zenjpeg #4)

# Chroma (1)
feat_cb_sharpness       (zenwebp +0.20pp tier; zenavif #5)
```

## Cross-codec dendrogram findings

Run on zenwebp (1264 rows) + zenjxl (400 rows). **6 perfect-Jaccard
(1.000) clusters** — same redundancy structure on both codecs:

1. `aq_map_p50 ↔ noise_floor_y_p50 ↔ quant_survival_y_p50` (median block cost)
2. `aspect_min_over_max ↔ log_aspect_abs` (#62 deprecated)
3. `indexed_palette_width ↔ palette_fits_in_256` (legacy; #30 retired)
4. `dct_compressibility_uv ↔ quant_survival_uv` (UV chroma compress)
5. `edge_density ↔ quant_survival_y`
6. `aq_map_p1 ↔ noise_floor_y_p1`

**3 strong-but-different (Jaccard 0.5–0.8)**:
- low_tail p5/p10 cluster (4 features)
- upper_tail_block_cost (3 features at p75)
- flat_block_signal (`aq_map_mean ↔ uniformity`)

**zenjxl-only**: `chroma_complexity ↔ colourfulness` (zenwebp's larger
corpus splits these — defer cull until cross-codec confirmation).

## FEATURE_GROUPS validator (shipped, `c4ac6423`)

Lives in `zentrain/examples/zenwebp_picker_config.py`. Enforces
mutual-exclusion at codec-config-load time via
`validate_keep_features(keep, groups)` in `train_hybrid.py`.

11 groups currently shipping (max_picked relaxed where current
zenwebp KEEP exceeds structural target):

```python
FEATURE_GROUPS = {
    # 6 perfect-Jaccard cross-codec clusters (max_picked=1 structural target;
    # relaxed where current KEEP has more, comment notes the tightening goal)
    "aspect": ["feat_aspect_min_over_max", "feat_log_aspect_abs"], max=1
    "palette_boolean": ["feat_palette_log2_size", "feat_palette_fits_in_256",
                        "feat_indexed_palette_width"], max=1
    "median_block_cost": ["feat_aq_map_p50", "feat_noise_floor_y_p50",
                          "feat_quant_survival_y_p50"], max=2 (target=1)
    "low_tail_p1": ["feat_aq_map_p1", "feat_noise_floor_y_p1"], max=1
    "uv_chroma_compressibility": ["feat_dct_compressibility_uv",
                                  "feat_quant_survival_uv"], max=2 (target=1)
    "edge_coef_survival": ["feat_edge_density", "feat_quant_survival_y"], max=2 (target=1)
    # 3 strong-agreement
    "low_tail_p5_p10": [aq_map_p5, aq_map_p10, noise_floor_y_p5,
                        noise_floor_y_p10], max=1
    "upper_tail_block_cost": [aq_map_p75, noise_floor_y_p75,
                              quant_survival_y_p75], max=3 (target=1)
    "flat_block_signal": ["feat_aq_map_mean", "feat_uniformity"], max=2 (target=1)
    # Soft-constraint
    "resolution_dimension": [pixel_count, log_pixels, bitmap_bytes,
                             log_padded_pixels_{8,16,32}, min_dim, max_dim], max=5
    "palette": [distinct_color_bins, palette_density, palette_log2_size,
                palette_fits_in_256], max=2
}
```

## LOO multi-seed results (5 seeds, zenwebp, 80 retrains, 175 min wall)

```
Feature                      mean ΔOH (pp)  σ ΔOH   mean ΔAC (pp)  σ ΔAC   Verdict
feat_noise_floor_y_p50       -0.35          0.28    +5.00          4.00    CULL (strongest signal in sweep — both metrics agree, both >1σ)
feat_log_padded_pixels_64    -0.19          0.47    +3.20          5.45    CULL confirmed (already retired)
feat_log_padded_pixels_8     +0.27          0.35    -4.10          6.61    KEEP (restore confirmed)
feat_bitmap_bytes            +0.17          0.53    -2.58          6.28    KEEP (restore confirmed; smaller magnitude than single-seed)
feat_log_padded_pixels_16    +0.19          0.29    -1.14          6.59    weak KEEP
feat_log_pixels              +0.10          0.22    -0.76          4.89    weak KEEP — restore mostly theoretical
feat_aq_map_p10              +0.17          0.41    -0.38          6.99    within noise
feat_palette_density         -0.01          0.44    +0.78          7.62    within noise (deprecated 023ff5ff)
```

**Critical lesson**: σ ΔAC is 4–8 pp. Single-seed LOO results within
±5 pp of zero are not reliable. Tier 0 correlation is the load-bearing
test, not Tier 1.5 permutation (which mis-attributes signal across
redundant pairs — `feat_log_pixels` showed +0.596 Δpp on zenjpeg
permutation despite Spearman 1.0 with `feat_pixel_count`).

## Open cull candidates pending action (multi-seed-confirmed)

1. **`feat_noise_floor_y_p50`** — 3-method consensus (cross-codec Tier 0 cluster anchor with `aq_map_p50` + single-seed LOO + multi-seed LOO mean ΔAC +5.00 ± 4.00 pp). **Drop from KEEP_FEATURES on every codec config that has it** (zenwebp, zenjpeg overnight, zenavif). Defer analyzer-side `#[deprecated]` until cross-codec multi-seed LOO confirms.

## Pending integration items

- **`feat/time-budgeted-objective` commit `1e020d1`** (worktree at `~/work/zen/zenanalyze--time-budgeted/`): adds `metric_log` head + `--time-budget-multiplier` + `BUDGET_INFEASIBLE` gate + R²/budget safety gates. Conflicts substantially with main's recent train_hybrid.py work (Parquet column-wise consumption, FEATURE_GROUPS validator, METRIC_COLUMN). Cherry-pick attempt produced empty diff (changes already on main via different SHAs?). Worth re-checking carefully.
- **`zenanalyze--zenpicker-rename` jj workspace** (commit `3eb0a225`): another session's WIP — 459 LoC including `zenpicker/examples/load_meta_picker_v0_1.rs` + `i8_vs_f16_agreement.rs` + `zenpicker/src/lib.rs` (+242). DO NOT TOUCH without checking with that session.

## Worktree state (cleaned this session)

```
~/work/zen/zenanalyze                                     [main, 91bfbcf5]
~/work/zen/zenanalyze--time-budgeted                      [feat/time-budgeted-objective, deferred]
~/work/zen/zenanalyze--zenpicker-rename                   [active session WIP]
~/work/zen/zenanalyze/.claude/worktrees/i8-quant-study    [research/i8-quant-impact, historical]
```

Cleaned during session: 18 dirs → 4. Deleted redundant local branches
already on main via squash: `feat/is-grayscale`,
`experiment/fingerprint-clean`, `feat/per-feature-bench`,
`fix/cargo-fmt-zenpicker`, `feat/size-variance-discipline`,
`worktree-zenpicker-pathology-seatbelts`. Closed PR #66
(content landed directly via `3a487de2`).

## Production gap (audit headline finding)

**Three of four trained pickers don't load at encode time**:

| Codec | Picker artifact | Loads at encode? |
|---|---|---|
| zenwebp | `picker.bin` v0 | ✅ |
| zenjpeg | regression weights baked as Rust consts | ❌ (issue #128) |
| zenjxl | `zenjxl_hybrid_2026-05-01.bin` in benchmarks/ | ❌ (no `with_picker()` API) |
| zenavif | `rav1e_picker_v0_1.json` | ❌ (`auto_tune()` is TODO) |

**This is the highest-ROI work item.** Phase 0 of
`rapid_iteration_plan` (~10 hr total): wire each picker to actually
load. Unlocks production payoff of all the offline ablation work.

## Architecture plan summary (4 docs interlock)

| Doc | Layer | Effort estimate |
|---|---|--:|
| Substrate (R2 + vast.ai) | Cloud primitives | ~1.5 weeks Phase M |
| Lifecycle orchestration (zentrain CLI) | Workflow | ~2 weeks (O1–O7) |
| Multi-MLP safety default path | Training shape | ~12 days (D1–D5) |
| Rapid iteration Phase 0 | Codec-picker wiring | ~10 hr (3 codecs) |

These compose: rapid Phase 0 unblocks production, lifecycle gives
multi-session, multi-MLP defines what gets baked, substrate defines
where it runs.

## Tonight's commits to main (in order)

```
e5c3c39  cull 3 new shape/smoothness redundancies
faf708a..b6b9903  pre-session work (already on main when started)
2bbdaf1  zenwebp expanded multiaxis picker config
eaaa2e2  zenjxl Tier 0 outputs + zenjpeg overnight picker config
155191a  cross-codec aggregate report (4-codec deferred-feature validation)
248b48b8 IndexedPaletteWidth (id 30) → PaletteLog2Size (id 121)
34fdb2e  zenwebp picker config v0.2 paths refresh
99076b75 PaletteLog2Size dependency fix + truecolor=24 sentinel + bin-bias docs
778ff2e  zenwebp picker config — drop multi_pass_stats head
e9cd04d  cull LogPixels (57) + BitmapBytes (60) — REVERTED next commit
15d9c299 restore LogPixels + LogPaddedPixels{8,16,32,64} (post tiny-MLP review)
af5bcf74 single-seed LOO results (zenwebp; first-pass)
4c183f7d restore BitmapBytes + retire LogPaddedPixels64 (LOO-confirmed)
6cdf681c feature-group structural analysis (zenwebp dendrogram + VIF)
ce899b6f cross-codec feature-group analysis (zenjxl confirms zenwebp clusters)
c4ac6423 deprecate LogAspectAbs + add FEATURE_GROUPS validator
023ff5ff deprecate PaletteDensity (id 12)
eac5e47f all-time-best-features synthesis doc
9b122043 Parquet support for pareto + features TSVs
62e3b2e3 multi-seed LOO retrain results (5 seeds)
59f502e1 multi-codec training inversion roadmap + Phase 1 Parquet conversion
fd85ad77 rapid-iteration plan + 7-component audit synthesis
a5523bf4 orchestration architecture (initial GCP version)
5a9adbe4 substrate pivot — R2 + vast.ai (no GCP)
3a487de2 land PR #66 content directly (zenjxl picker integration toolchain)
dfc59124 multi-MLP safety + default-path plan (initial)
4926cd0f revise multi-MLP plan — single MLP, multi-objective via runtime masking
d5f26ce2 add optional slope-tuned MLP per (codec, metric)
91bfbcf5 zentrain lifecycle orchestration — drive everything, multi-session safe, pluggable compute
```

## Known bugs / data-quality issues flagged

1. **HDR class corpus bug** — 53 gain-map JPEGs in `/mnt/v/output/codec-corpus-2026-05-01-multiaxis/` whose base layer is SDR. The expanded corpus' HDR axis doesn't actually exercise HDR features. Replace with native PQ/HLG content.
2. **zenjpeg overnight pareto data starvation** — schema mismatch caused several cells to have 1 member config; numeric retrain results from that run are smoke-test only.
3. **`feat_palette_density` permutation mis-attribution** — Tier 1.5 ranked it #5 on zenjpeg (+0.618 Δpp) but Tier 0 + multi-seed LOO showed it's pure noise. Permutation importance double-counts across redundant pairs.
4. **`feat_log_pixels` similar mis-attribution** — Tier 1.5 +0.596 Δpp on zenjpeg despite Spearman 1.0 with `feat_pixel_count`. Multi-seed mean ΔAC −0.76 ± 4.89 pp; restoration mostly theoretical.
5. **zenavif harness panics on non-pow2 sizes** — blocks Phase 3 LHS sweep against expanded multi-axis corpus. Documented in `audits-2026-05-02/zenavif.md`.

## Hot links

- Multi-axis corpus: `/mnt/v/output/codec-corpus-2026-05-01-multiaxis/manifest.tsv` (463 images, 97 MP)
- coefficient oracle-d2: `/home/lilith/oracle-d2-store/oracle-d2/{pareto_rows.csv, source_features.json}` (75 k encodings, 108 k metrics)
- jxl picker oracle: `/mnt/v/output/jxl-encoder/picker-oracle-2026-04-30/{lossy,lossless}_pareto_*.{tsv,parquet}`
- coefficient code: `~/work/coefficient/`, `~/work/coefficient-zenjpeg-opt/`
- squintly: `~/work/squintly/`
- LOO driver: `benchmarks/loo_driver_multiseed_2026-05-02.py`
- TSV → Parquet converter: `benchmarks/tsv_to_parquet.py`
- Feature dendrogram script: `benchmarks/feature_groups_2026-05-02.py`

## Multi-seed sweep raw outputs

`/tmp/loo_multiseed_2026-05-02/` — 80 train logs (not committed, regenerable via driver). Result TSVs at `benchmarks/loo_retrain_multiseed{,_raw}_2026-05-02.tsv`.

## Substrate migration status (Phase 1 of orchestration)

- ✅ Parquet conversion: zenjpeg `zq_pareto_2026-04-29.tsv` (540 MB → 50 MB), zenjxl `lossy_pareto_2026-04-30.tsv` (100 MB → 10 MB), zenjxl `lossless_pareto_2026-04-30.tsv` (20 MB → 2 MB), zenwebp `_combined.tsv` (3.4 GB → 0.21 GB)
- ✅ `train_hybrid.py` Parquet support (`_read_table_columns` helper)
- ❌ R2 backend in coefficient (Phase M.1, ~4 hr)
- ❌ vast.ai launcher in coefficient (Phase M.2, ~2 days)
- ❌ Storage-as-queue protocol (Phase M.3, ~1 day)

## Next-session priorities

1. **Highest ROI**: rapid_iteration_plan Phase 0 — wire 3 idle codec pickers (zenjpeg/zenjxl/zenavif). ~10 hr total. Unblocks production payoff of all offline ablation work.
2. **Cull `feat_noise_floor_y_p50`** at picker-config level on every codec config that has it. ~1 hr. Multi-seed-confirmed.
3. **Phase D1 of multi-MLP safety doc**: `Pick` + `PickerObjective` enums + `Predictor::pick()` orchestrator in zenpredict. ~3 days. Highest leverage on the runtime API.
4. **Phase O1 of lifecycle orchestration**: `zentrain orchestrate --dry-run` hash resolver. ~2 days. Foundational for everything else.

Pause-resume between any of these is fine. Each is independently
useful.

## Conventions

- jj on top of colocated git; main checkout works on `main` directly
- Marker file `.workongoing` claimed before any work
- Per-codec / per-feature commits commit immediately after compile/test pass
- All file format >50 MB ships as Parquet (zstd-3); see `~/work/claudehints/topics/parquet-vs-tsv.md`
