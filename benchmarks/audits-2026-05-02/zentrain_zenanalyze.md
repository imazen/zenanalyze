# zentrain + zenanalyze pipeline audit, 2026-05-02

## Executive Summary

The zentrain/zenanalyze stack is the shared training + feature-extraction layer consumed by 4 codec pickers (zenjpeg, zenwebp, zenavif, zenjxl) + zensim + zenpicker. The pipeline has grown piecemeal with significant duplication across codecs. **Tier 1 (refresh_features.py orchestrator) has landed and consolidates per-codec agent spawning.** Tiers 2–4 (centralized Rust extractor, pareto schema unification, config minimization) are queued.

---

## Inventory: Python Training Tools

| Tool | Lines | Purpose | Input | Output | Invocation |
|------|-------|---------|-------|--------|-----------|
| **train_hybrid.py** | 2,373 | Codec-agnostic hybrid-heads MLP trainer. Loads pareto TSV (Parquet), features TSV, pivots by categorical cell, trains per-cell students + bytes_log head, serializes ZNPR v2 JSON. | `--codec-config <module>` (PARETO, FEATURES, KEEP_FEATURES, ZQ_TARGETS, parse_config_name) | `OUT_JSON` (ZNPR v2 model file) + `OUT_LOG` (safety report, per-cell metrics) | `python3 train_hybrid.py --codec-config zenjpeg_picker_config --activation leakyrelu --seed 0x...` |
| **feature_ablation.py** | 22,391 | Leave-one-out + forward-greedy ablation. Retrains HistGB per feature subset, measures argmin overhead delta. Outputs Pareto curve of (n_features vs overhead). | Same codec-config, same PARETO/FEATURES | `*.json` (per-feature overhead ranks) + `*.log` (summary) | `--codec-config` + `--method greedy\|backward\|forward\|permutation` |
| **student_permutation.py** | 568 | Fast permutation importance (seconds, not hours). Loads trained student JSON from train_hybrid, runs forward passes on val set, permutes each feature, measures overhead delta. | `--model-json <trained model>` + `--codec-config` | `*.json` (feature → mean/stddev/delta) | `python3 student_permutation.py --model-json <model> --codec-config ...` |
| **correlation_cleanup.py** | 15,998 | Tier-0 dendrogram: identifies redundant feature pairs across cross-codec corpus. Reads all 4 codecs' features TSVs, computes Spearman correlation, clusters, suggests drop candidates. | 4 codec-config modules (can override via CLI) | `*.json` (correlation matrix + dendrogram) | `--codec-config zenjpeg_picker_config --threshold 0.92` |
| **feature_group_ablation.py** | 10,983 | Retrains with feature groups dropped (e.g., all luma features, all DCT features). Answers "can we drop an entire family?" | Same codec-config + `FEATURE_GROUPS` dict (zenjpeg_picker_config.py example) | `*.json` (group → overhead) | `--codec-config` + `--method ...` |
| **refresh_features.py** | ~300 | **[Tier 1, landed 2026-05-02]** Orchestrator: for each registered codec, invokes the codec's extractor binary, converts TSV→Parquet, updates picker config paths. Removes the "spawn 4 agents in parallel" pattern. | Hardcoded CodecRecipe list (zenjpeg/zenwebp/zenavif/zenjxl) | Updated Parquet files + picker config update suggestions | `python3 refresh_features.py [--codec zenwebp\|all] [--dry-run]` |
| **tsv_to_parquet.py** | 3.2 KB | File format converter. Reads TSV, writes zstd-compressed Parquet (16× disk savings, 35× faster load). | `--input <TSV>` | `<input>.parquet` | Invoked by refresh_features or manual conversion |
| **size_invariance_probe.py** | 444 | Validation: verifies that per-(image, target_zq) picker cell picks remain stable across size classes. | Trained model JSON + feature matrix | `*.md` (per-image stability report) | `--model-json` + codec-config |
| **validate_schema.py** | 301 | Schema-hash audit: compares `feat_*` columns in Parquet/TSV against deployed schema_hash. Alerts on drift. | `--schema-hash` + `--features-parquet` | Console warnings on mismatch | `python3 validate_schema.py --schema-hash <hash> ...` |
| **diagnose_picker.py** | 8,693 | Introspection: given a trained JSON + feature matrix, outputs per-cell argmin rates, reach gates, failure modes. | Model JSON + features TSV | `*.md` (diagnostics table) | `--model-json <model> --features ...` |
| **inspect_picker.py** | 26,464 | Rich inspection: reads model, features, pareto; outputs per-cell statistics, cross-metric correlation, per-image prediction errors. | Model JSON + pareto/features TSVs | `*.md` + optional `*.json` (detailed tables) | `--model-json` + `--pareto` + `--features` |
| **leakyrelu_experiment.py** | 15,119 | Proof-of-concept LeakyReLU student via PyTorch. Matches sklearn MLPRegressor interface. | None (test script) | None | Standalone experiment |
| **leakyrelu_seeds_runner.py** | 9,606 | Harness for LeakyReLU + sklearn ReLU comparison across seeds. | Config + feature matrices | Per-seed training logs | `python3 leakyrelu_seeds_runner.py` |
| **capacity_sweep.py** | 15,760 | Hyperparameter sweep: trains MLPs with different hidden layer sizes, measures accuracy vs model size. | Codec-config + `--hidden-sizes` list | JSON (size vs overhead Pareto) | `--codec-config` + `--hidden-sizes "64,64" "128,128" ...` |
| **holdout_size_sweep_probe.py** | 14,609 | Validates that held-out per-size metrics are within bounds. Probes for size-dependent calibration drift. | Model JSON + per-size holdout data | `*.md` (pass/fail per size class) | `--model-json` + `--holdout-data ...` |
| **adversarial_probe.py** | 7,048 | Finds pathological images: runs model on corpus, identifies images where picker made worst predictions. | Model JSON + full corpus features | JSON (ranking of worst images) | `--model-json` + `--features ...` |

**Total**: ~144 KB of Python training logic (excluding dependencies). Heavily modular; core transformation pipeline (load, pivot, train, evaluate) lives in `_picker_lib.py` (shared across tools).

---

## Inventory: Rust Public API (zenanalyze)

### Entry Points

| API | Signature | Purpose | Cost |
|-----|-----------|---------|------|
| **analyze_features** | `fn(PixelSlice, &AnalysisQuery) -> Result<AnalysisResults>` | Primary. Generic over color space / depth / alpha. Dispatches on const-bool tier gates (palette, T2, T3, depth, alpha, DCT). | ~0.5–1.5 ms/Mpx depending on tier + FeatureSet composition. Default `pixel_budget = 1 MP` on any image; lazy sampling below 1 MP. |
| **analyze_features_rgb8** | `fn(&[u8], width, height) -> Result<AnalysisResults>` | Convenience for packed RGB8 buffers (zero-copy, no RowConverter). | Equivalent to analyze_features on T1 path. |

### Feature Schema (features_table! macro)

The **features_table! macro** (lines 64–270 in feature.rs) is the single source of truth. Generates:
- `AnalysisFeature` enum (u16 discriminants, immutable once shipped)
- `RawAnalysis` dense struct (field layout matches enum order)
- `RawAnalysis::into_results()` translator
- `FeatureSet::SUPPORTED` preset (bitset of all active features)

**Versioning discipline:**
- Discriminants are **never recycled**. Retired variants keep their u16 slot forever.
- New variants get the next sequential number.
- `FeatureSet` storage is opaque; public API is const-fn set math.
- Deserialization via `AnalysisFeature::from_u16(n)` returns `Option` for graceful forward/backward compat.

**Feature count**: 29 active features shipped; schema_hash (SHA256 of feat_cols list) is immutable once deployed. Picker loads require schema-hash match at runtime.

### AnalysisResults (opaque container)

```rust
pub struct AnalysisResults { /* ... opaque ... */ }

impl AnalysisResults {
    pub fn get(&self, feature: AnalysisFeature) -> Option<FeatureValue>  // None if feature inactive or OOD
    pub fn iter(&self) -> impl Iterator<Item = (AnalysisFeature, FeatureValue)>
    pub fn geometry(&self) -> ImageGeometry  // (width, height, pixels, aspect_ratio)
    pub fn source_descriptor(&self) -> &PixelDescriptor  // original input format
}
```

**FeatureValue**: tagged enum (`F32(f32) | U32(u32) | Bool(bool)`), with lossless coercion via `to_f32()`. `NaN` signals out-of-distribution (per-feature sample-count floor, e.g., percentiles below 100 DCT blocks → NaN).

---

## Inventory: Picker Configs (codec-specific scaffolding)

| Codec | Lines | Paths | Axes | Features | Duplicated Logic |
|-------|-------|-------|------|----------|-----------------|
| **zenjpeg** | 276 | PARETO, FEATURES, OUT_JSON, OUT_LOG | categorical: (color_mode, sub, scan, sa_piecewise, trellis_on) → ~12 cells; scalar: (chroma_scale, lambda) | 73-feature set (post-ablation) | Config-name regex, axis naming, ZQ_TARGETS list, HISTGB params |
| **zenwebp** | 313 | Same | categorical: (method 0-6, segments 1-4) → ~14–20 cells; scalar: (effort, sns_strength, filter_strength, filter_sharpness) | 38-feature set (empirically culled) | Config-name regex, ZQ_TARGETS, scalar bounds |
| **zenavif** | 190 | Same | categorical: (speed 1-10) + phase-1a fixed axes (qm, vaq, tune) → 10 cells; scalar: none yet | ~99 features (post-cull) | Config-name regex, axis dict |
| **zenjxl** | 215 | Same | **lossy**: categorical (modular_mode, color_space) + scalar (effort, distance); **lossless**: categorical (lz77_method, palette) + scalar (tree_max_buckets, tree_num_properties, tree_sample_fraction) | Union of lossy + lossless feature sets | Separate parsers for lossy/lossless, two independent pickers |

**Structural redundancy**: All configs define ~150 lines of identical scaffolding:
- Import + path declarations (20 lines)
- Codec config loading boilerplate (10 lines)
- ZQ_TARGETS defaults (5 lines)
- Axis naming + bounds (20 lines)
- Config-name regex parser (40–60 lines)
- Codec-specific feature list (40–80 lines)

**The inversion roadmap (INVERSION.md) targets 4 tiers**:
1. **Tier 1 (landed)**: refresh_features.py orchestrator
2. **Tier 2 (queued)**: Centralized `zenanalyze::extract_features_from_manifest` Rust binary
3. **Tier 3 (queued)**: Pareto TSV schema unification (all codecs emit same columns)
4. **Tier 4 (queued)**: Config minimization (boilerplate moves to zentrain defaults; codecs declare only codec-specific knobs)

**Target post-Tier-4**: ~30-line configs.

---

## Inventory: Benchmarks (Source of Truth vs Archive vs Ephemeral)

| File | Size | Date | Status | Consumed By | Notes |
|------|------|------|--------|------------|-------|
| **zenjpeg_ablation_2026-05-01.json** | 37 KB | 2026-05-01 | Active | Feature ablation ranking (zenjpeg) | Full Tier-3 ablation; includes dendrogram |
| **zenjxl_hybrid_2026-05-01.json** | 1.8 MB | 2026-05-01 | Active | Deployed zenjxl picker (lossless) | ZNPR v2 model file; full training metadata |
| **zenjxl_student_perm_2026-05-01.json** | 21 KB | 2026-05-01 | Active | Post-training feature importance | Permutation importance on 5K val rows |
| **loo_retrain_multiseed_2026-05-02.tsv** | 530 B | 2026-05-02 | Active | Cross-seed LOO stability validation | 8 candidate features × 5 seeds = 40 retrains; 175 min wall-clock |
| **feature_groups_2026-05-02.py** | 12 KB | 2026-05-02 | Active | Feature grouping schema (reference) | Defines 8 semantic groups (luma, chroma, DCT, percentiles, etc.) |
| **zenavif_correlation_2026-05-01.json** | 3.3 KB | 2026-05-01 | Active | Tier-0 dendrogram (zenavif only) | Identifies redundant feature pairs |
| **cross_codec_aggregate_2026-05-02.md** | 15 KB | 2026-05-02 | Active | Multi-codec summary (markdown) | Aggregated statistics across 4 codecs |
| **all_time_best_features_2026-05-02.md** | 15 KB | 2026-05-02 | Archive? | Historical best-features ranking | Pre-dates multi-codec unification; may be stale |
| **loo_driver_multiseed_2026-05-02.py** | 10 KB | 2026-05-02 | Ephemeral | Ad-hoc LOO harness | Generated at runtime; not committed to version control |
| **zenjxl_correlation_2026-05-02.json** | 5.9 KB | 2026-05-02 | Active | Tier-0 for zenjxl (post-threshold) | Threshold-99 variant; alternative to the 2026-05-01 version |

**Pattern**: Each codec has a per-date subdirectory (benchmarks/zenjpeg_*, benchmarks/zenwebp_*, etc.). Active files are <2 weeks old; older versions are archived but not deleted. **No central "source of truth" dashboard** — must reconcile 4 separate per-codec benchmark trees.

---

## Data Flow: Canonical Training Cycle

```
┌──────────────────────────────────────────────────────────────┐
│ 0. PARETO SWEEP (codec-side, ~hours)                          │
│    ├─ Encode images × configs × quality grid                   │
│    ├─ Measure bytes, encode_ms, zensim/ssim2/butteraugli       │
│    └─ Output: TSV (image, size_class, config_name, q, bytes…)  │
└────────────────┬─────────────────────────────────────────────┘
                 │ PARETO = Path("…pareto.tsv")
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. FEATURE EXTRACTION (zenanalyze extractor, ~minutes)        │
│    ├─ Run analyzer on each image (zenanalyze::analyze_features)│
│    ├─ Emit feat_* columns (29 features × n_images)            │
│    └─ Output: TSV (image_path, size_class, feat_*, …)         │
└────────────────┬─────────────────────────────────────────────┘
                 │ FEATURES = Path("…features.tsv")
                 │ Optionally convert TSV→Parquet (tsv_to_parquet.py)
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. LOAD + PIVOT (train_hybrid.py, ~30 sec)                   │
│    ├─ Load PARETO (Parquet, 36× faster than TSV)             │
│    ├─ Load FEATURES (column indices into cache)              │
│    ├─ Filter ZQ_TARGETS (drop cells where zq > ceiling)      │
│    ├─ Pivot by CATEGORICAL_AXES → per-cell (X, Y) matrices  │
│    └─ Hold out ~20% test/val data per image                  │
└────────────────┬─────────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 3a. TEACHER TRAINING (parallel, ~30 sec)                      │
│    ├─ Per-cell HistGB fit (sklearn, multi-threaded via joblib)│
│    ├─ Predicts log-bytes per config within cell              │
│    ├─ Used to compute scalar-head targets (within-cell opt)  │
│    └─ N_cells × N_scalar_axes independent teachers           │
└────────────────┬─────────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────────┐
│ 3b. STUDENT TRAINING (leakyrelu MLP, ~20 sec)               │
│    ├─ Inputs: feat_* columns (standardized)                 │
│    ├─ Outputs: [bytes_log[N_cells], scalar_0[N_cells], …]  │
│    ├─ Loss: MSE (mean on train, L2 on weights)             │
│    ├─ Activation: LeakyReLU (sklearn ReLU is 5–15× slower) │
│    └─ Early stop on val loss (patience=30, tol=1e-6)        │
└────────────────┬─────────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. EVALUATION + SAFETY REPORT (~10 sec)                       │
│    ├─ Evaluate argmin accuracy on held-out test             │
│    ├─ Per-cell R² on val + test splits                      │
│    ├─ Reach rates per (size, zq) band (p99 above threshold?)│
│    ├─ Feature bounds (p01/p99 for OOD gate)                │
│    ├─ Output bounds (p01/p99 for hallucination gate)       │
│    ├─ Check 6 validation gates (safety report)             │
│    └─ Emit safety_compact + reach_rates + feature_bounds   │
└────────────────┬─────────────────────────────────────────────┘
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. BAKE ZNPR V2 MODEL + MANIFEST                              │
│    ├─ Serialize model weights → f16 (half model size)       │
│    ├─ Embed metadata (≤ 5 KB)                               │
│    ├─ Output: <codec>_picker.bin (~30–150 KB)              │
│    └─ Side-car: manifest.json (safety report, ablations…)   │
└──────────────────────────────────────────────────────────────┘

Wall-clock total (end-to-end, single-machine): ~5–10 min per bake
(Dominated by step 0 — codec-side pareto sweep, outside zentrain.)
```

---

## Performance Bottlenecks (with numbers)

### 1. **Parquet Adoption Incomplete** (biggest impact, already 35× gain from TSV)

- **Current**: zenjpeg/zenwebp still use TSV for Pareto (~3.4 GB each).
- **Why slow**: load_pareto() loops per-row, builds Python dicts, no columnar SIMD. zenwebp 21.8 M-row pareto takes **68 seconds** to CSV-parse → Python dicts.
- **After Parquet (live benchmark in PRINCIPLES.md)**: **1.9 seconds** (36× faster).
- **Blocker**: `train_hybrid.py` loads Parquet natively but downstream per-row dict iteration still materializes. Refactor to consume Arrow columns directly → **another 1.3× gain** (end-to-end, 68 s → 54 s).
- **Action**: Migrate all 4 codecs to Parquet (tsv_to_parquet.py is ready); unblock Arrow column consumption in train_hybrid.

### 2. **Multi-Seed LOO Takes ~175 minutes** (N=8 candidates × 5 seeds × 2 sides = 40 retrains)

- **Current**: loo_driver_multiseed_2026-05-02.py runs 40 serial retrains (zenwebp config, ~4 min each). 
- **Bottleneck**: Pure wall-clock; no parallelization at the outer loop. CPU cores sit idle between jobs.
- **Why it matters**: Feature importance validation across seeds is essential for deciding whether to drop candidates.
- **Action**: Parallelize LOO at the seed level (GNU parallel or job queue). Potential 4–5× speedup (still serial within each config, but codecs can run in parallel).

### 3. **Per-Row Dict Overhead in train_hybrid.py** (vectorization roadblock)

- **Current**: After loading pareto + features TSVs, train_hybrid pivots row-by-row into per-cell matrices.
  ```python
  for r in pareto_rows:
      parsed = parse_config_name(r['config_name'])  # regex per row
      cell_key = tuple(parsed[ax] for ax in CATEGORICAL_AXES)  # hashable
      Y_cell[cell_key].append(float(r['bytes']))  # accumulate
  ```
- **Cost**: ~O(n_rows × parse_time + dict overhead). On zenwebp 21.8 M rows, parse_time dominates even with compiled regex.
- **Fix**: Vectorize in Rust. Centralize extract_features_from_manifest + pareto_parse_and_pivot in Rust; return NumPy arrays directly.
- **Gain**: 1–2 orders of magnitude (comparable to Parquet load speedup).

### 4. **No Caching of Intermediate Results**

- **Feature extraction**: Each refresh_features invocation re-extracts all images from codecs (can take 5–10 min per codec).
- **Teacher training**: HistGB per-cell fits are deterministic; no cache key is maintained. Identical (pareto, features, cell) combos retrain from scratch.
- **Action**: Add hash-based cache (pareto_hash + features_hash + axes_schema → cached teacher JSON) in _picker_lib.py.
- **Potential**: 5–10× speedup on iteration (skip redundant teacher training during ablations).

### 5. **Single-Machine Serialization** (16-core CPU, no GPU, no distributed trainer)

- **Current**: All 4 codecs train sequentially on one box. refresh_features spawns 4 codec extractors back-to-back.
- **Parallelization available**: Extract features in parallel (4 codec extractors at once). Train pickers in parallel (no inter-codec coupling).
- **Blocker**: None technical; just orchestration. refresh_features.py has the loop.
- **Potential**: 3–4× speedup if codecs' extract time is bottleneck (depends on disk I/O, encode machinery).

---

## Reproducibility (Commit/Data/Seed Pinning)

### What's versioned:

| Artifact | Versioning | Pinning |
|----------|-----------|--------|
| **Source code (Rust + Python)** | git commit hash | Yes (CI tests HEAD, releases tag commit) |
| **Pareto sweep data** | date stamp in filename (YYYY-MM-DD) | Implicit (TSV path in config, no hash) |
| **Feature extraction schema** | schema_hash (SHA256(feat_cols list)) | **Weak** — runtime loads and errors on mismatch, but no pre-commit tracking |
| **Trained model (.bin)** | date stamp + commit in metadata | **Weak** — metadata not verified against current code |
| **Seed** | `--seed` flag to train_hybrid.py (default 0xCAFE) | Yes, per-run |

### Pain points:

1. **Pareto sweep is not content-addressed**. If a codec re-runs the pareto and outputs the same TSV path, picker configs still point to it, but the underlying data may have drifted (if encoder version bumped or encode pass parameters changed).
   - *Fix*: Pareto TSV path should include content hash (e.g., `pareto_<sha256>.parquet`). Config pins the hash.

2. **Feature schema drift is not pre-commit audited**. If zenanalyze adds a feature and an old picker config tries to load features from the new extraction, `validate_schema.py` catches it at runtime, but by then the config is already deployed.
   - *Fix*: CI should run `validate_schema.py` against all deployed picker configs whenever zenanalyze schema changes.

3. **Metadata versioning is implicit**. Bake metadata (safety_compact, reach_rates, output_bounds) is ZNPR v2, but the trainer only emits it if `--allow-unsafe false` (default — refuse to bake if safety_report.passed=false). No centralized log of "which commit baked which version."
   - *Fix*: Add a bake provenance table (git commit, sweep commit, trainer version, feature schema hash, seed, timestamp).

### Current best practice (per PRINCIPLES.md, §"Re-bake triggers"):

Codecs re-bake when:
- Feature mean drift > 0.10σ on any input feature
- `schema_hash` mismatch (compile error at load)
- Default `pixel_budget` change
- ZNPR format version change
- New corpus class added
- Encoder version bump > 1% byte cost

**No automated triggers.** Re-bakes are manual (ops run refresh_features.py + train_hybrid.py + hand-verify safety report).

---

## Test Coverage

### Rust (src/tests.rs + zenpredict/tests/)

- **Feature extraction tests**: ✓ (per-tier SIMD dispatch, various color spaces, alpha paths)
- **AnalysisResults serialization**: ✓ (schema_hash, feature_set round-trip)
- **Zenpredict model loading**: ✓ (ZNPR v2 deserialization, OOD bounds, argmin with mask)
- **Cross-platform**: ✓ (CI covers ubuntu/windows-arm/macos-intel/macos-arm, 32-bit and 64-bit via cross)
- **End-to-end trainer tests**: ✗ (no integration test that trains a model, bakes, and verifies runtime load)

### Python (zentrain/tools/)

- **train_hybrid.py**: No unit tests. Tested via manual ablation runs (feature_ablation.py serves as integration test).
- **feature_ablation.py**: ✓ Self-integration tested (trains N models, measures overhead).
- **Picker config parsing**: ✓ (each codec's parse_config_name is lightly tested in ablation logs).
- **Parquet I/O**: ✓ (tsv_to_parquet.py is simple and tested via refresh_features).
- **No regression tests on eval metrics**: Each codec bake measures accuracy independently; no cross-codec baseline to catch regressions.

### CI (GitHub Actions)

- **Rust**: Format + Clippy + tests (lib + doc) on 4 platforms + 2 cross targets.
- **Python**: None. (No Python linting or type checking in CI; local-only pre-commit hooks expected.)
- **Trainer**: Not in CI. (Pareto sweeps + trainer runs are off-CI, manual, per-codec.)

**Gap**: No integration test that:
1. Generates a toy pareto + features
2. Trains a model via train_hybrid.py
3. Bakes to ZNPR v2
4. Loads and runs the model at runtime
5. Verifies predictions are sane

---

## What Would Make This Faster/Better (Top 5 Missing Tooling)

### 1. **Centralized Feature Extractor (Tier 2, Rust)**
   - **What**: Single `zenanalyze/examples/extract_features_from_manifest.rs` binary replacing 4 per-codec extractors.
   - **Impact**: 
     - Removes per-codec extraction code duplication (codecs no longer ship their own extractors).
     - refresh_features.py becomes a thin wrapper.
     - Feature schema unification is possible (all codecs extract the same columns).
   - **Effort**: ~2 hours (Rust binary, Parquet output, cargo deps already present).
   - **Blocker**: None; independent of Tier 3/4.

### 2. **Arrow Column Consumption in train_hybrid.py (Vectorization)**
   - **What**: Load Parquet into Arrow table; consume as columnar arrays, not per-row dicts.
   - **Impact**: 
     - End-to-end load time: 68 s → 54 s (1.3× on zenwebp 21.8 M rows).
     - Unblocks downstream NumPy vectorization (parse_config_name can be SIMD'd if needed).
   - **Effort**: ~1 hour (refactor load_pareto + build_dataset to use pyarrow Tables).
   - **Blocker**: Tier 1 (Parquet migration) must land first.

### 3. **Pareto Schema Unification (Tier 3, Codec-side)**
   - **What**: All 4 codecs' pareto sweeps emit canonical schema:
     ```
     image_path  size_class  width  height  config_id  config_name
     q  bytes  metric_name  metric_value  encode_ms  effective_max_zensim
     ```
   - **Impact**: 
     - One train_hybrid.py consumes any codec without per-codec schema branches.
     - Cross-codec analysis tools (correlation cleanup, LOO) work on any subset.
     - New codecs plug in with no harness rewrites.
   - **Effort**: ~1 day per codec (4 codec harnesses to refit).
   - **Blocker**: Codec harness maintainers; independent of Tier 2.

### 4. **Orchestration Dashboard (Reproducibility)**
   - **What**: Centralized table (CSV/JSON/web) tracking all picker bakes:
     ```
     codec  | model_name      | commit_hash | sweep_commit | feature_schema_hash | seed | trained_at | safety_passed | deployed
     zenjpeg| picker_v2.2     | abc123      | def456       | sha256(...)        | 0    | 2026-05-02 | ✓             | ✓
     ```
   - **Impact**: 
     - Audit trail: "which code/data/seed produced this picker?"
     - Pre-commit hook can verify deployed pickers are reproducible.
     - Cross-codec regression detection (compare current bake vs previous).
   - **Effort**: ~4 hours (schema + bake_picker.py integration + CI step to log).
   - **Blocker**: None; can be added independently.

### 5. **Multi-Codec Ablation + LOO Driver (Tier 2+, Python)**
   - **What**: Unified ablation harness that runs across all 4 codecs in parallel, outputs cross-codec feature importance rankings.
   - **Impact**: 
     - Multi-codec consensus on "which features to drop" (avoids codec-specific overfitting).
     - Parallel LOO (codecs run independently; seeds run in parallel within each).
     - Single HTML dashboard (feature importance heatmap, per-codec + cross-codec).
   - **Effort**: ~3 hours (generalize loo_driver_multiseed_2026-05-02.py, add parallel job spawning).
   - **Blocker**: Tier 3 (pareto schema unification) simplifies this; can be done after.

---

## Top 5 Pain Points (Ranked by Impact on Time Loss)

| # | Pain Point | Symptom | Root Cause | Inversion Target |
|---|-----------|---------|-----------|------------------|
| **1** | **Multi-seed LOO = 175 min wall-clock** | Feature importance validation is slow; ablation iterations are blocked | 40 retrains run serially; no CPU parallelization at outer loop | Parallel LOO driver + pareto schema unification |
| **2** | **Per-codec config duplication** | ~150 lines of boilerplate × 4 codecs; new codecs inherit messy scaffolding | No centralized config defaults; codec-specific axes/paths scattered | Tier 4: minimize to ~30 lines (zentrain absorbs scaffolding) |
| **3** | **No caching of intermediate results** | Identical (pareto, features, cell) combos train from scratch on ablation iteration #2+ | _picker_lib.py doesn't hash pareto + features + axes; teachers not saved | Content-addressed cache (pareto_hash → teacher JSON) |
| **4** | **Cross-codec analysis requires hand-rolled scripts** | Each ablation/LOO run is codec-specific; comparing zenjpeg vs zenwebp feature importance is manual reconciliation | No unified data layout; 4 separate pareto schemas + benchmark trees | Tier 3: pareto schema unification |
| **5** | **Reproducibility is implicit** | Re-running an old bake is hard (seed/commit/data combo is not tracked centrally) | Bake provenance metadata not versioned; pareto sweeps not content-addressed | Orchestration dashboard + pareto content-hashing |

---

## Validation Gates (CI Blocks Release)

Current gates in `PRINCIPLES.md` §"Validation gates":

1. ✓ Trainer safety report: `passed = true` (no data-starved, uncapped-zq, ood-leakage errors)
2. ✓ Held-out p99 zensim shortfall < 1 pp (size_optimal) / 0.5 pp (zensim_strict)
3. ✓ Round-trip check: bake_roundtrip_check.py matches NumPy reference within tolerance
4. ✓ Size invariance probe: ≥ 90% stable picks per cell across sizes
5. ✓ Schema hash committed (issue tracking)
6. ✓ feature_bounds populated in bin
7. ✓ (On consumer bump) Pick-stability gate: ≥ 90% agreement vs previous bake on holdout web-traffic corpus

**In CI (GitHub Actions)**: Only Rust tests (#1–2 coverage; #6 partially tested). Python trainer + bake gates are manual.

---

## Known Landmines (from PRINCIPLES.md §"Known landmines")

1. **Re-bake every consumer when `pixel_budget` default moves** — feature `patch_fraction_fast` drifts ~0.1σ across budget range; picker's internal scaling is anchored to specific budget.
2. **MLP extrapolates outside trained envelope** — every codec MUST run OOD bounds check before argmin or get confident-looking garbage.
3. **Adding corpus class requires re-bake** — (photo + screen + line-art) → (photo + screen + line-art + mobile) shifts feature mean/variance, shifts StandardScaler.
4. **Composite features are hand-tuned** — TextLikelihood, ScreenContentLikelihood, etc. must stay behind `composites` cargo feature until multi-quarter stability validation.
5. **Bake metadata is not optional** — if metadata exceeds 8 KB budget or is missing, codec can't safely use the bake at runtime.
6. **Can't fix bad picks post-hoc** — by the time bytes are on wire, decision is amortized. Two-shot rescue is last line of defense.

---

## Next Steps (Tiers 2–4, Phased Rollout)

### Tier 2: Centralized Rust Extractor (Next session, ~2 hours)
- Write `zenanalyze/examples/extract_features_from_manifest.rs`
- Add Parquet + Arrow output, parallel image decoding
- Update refresh_features.py to invoke single binary instead of per-codec extractors
- Test on all 4 codecs

### Tier 3: Pareto Schema Unification (As time allows, ~1 day per codec)
- Standardize on canonical pareto schema (image_path, size_class, config_id, q, bytes, metric, encode_ms, effective_max_zensim)
- Refactor 4 codec harnesses
- Remove codec-specific branches from train_hybrid.py

### Tier 4: Config Minimization (After Tier 2, ~2 hours per codec)
- Move scaffolding to zentrain defaults (ZQ_TARGETS, HISTGB params, OUTPUT_RANGES)
- Reduce codec config to ~30 lines (KEEP_FEATURES, CATEGORICAL_AXES, SCALAR_AXES, parse_config_name regex)

### Bonus: Orchestration Dashboard + Reproducibility
- Central bake provenance table
- Content-addressed pareto TSVs (include hash in path)
- Pre-commit CI hook to validate reproducibility

