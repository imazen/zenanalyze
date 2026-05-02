# zenwebp training/picker audit, 2026-05-02

## Inventory

### Sweep/Oracle Harnesses
- `dev/zenwebp_pareto.rs` (600+ lines): Main Pareto sweep harness. Emits (image, size_class, config, q) → (bytes, zensim, encode_ms, total_ms) + features TSV. Parallelized with rayon. ~8× prior config volume (576 configs = 12 cells × scalar grid). Default corpus: CID22-512, gb82, gb82-sc, CLIC2025. Q-grid: 30 values (0..70 step 5, then 70..101 step 2). Size variants: {64, 256, 1024, native}. Categorical axes: method ∈ {4,5,6} × segments ∈ {1,4} × cost_model ∈ {false,true}; scalar axes: sns, filter_strength, filter_sharpness, partition_limit, multi_pass_stats[m4 only]. Feature-only mode available (`--features-only`).
- `dev/zenwebp_picker_sweep.rs` (150+ lines): Distill spike harness. Encodes each image at every cell for each target zensim. Outputs pareto TSV + features TSV for `train_hybrid.py`.
- `dev/picker_ab_eval.rs`: A/B evaluator for picker vs bucket-table baseline.
- `dev/zensim_ceiling_probe.rs`: Probes effective_max_zensim ceiling support. Referenced in pareto row schema.

### Feature Extraction
- `dev/zenwebp_pareto.rs::analyze_features_rgb8()`: Uses zenanalyze to extract full feature set (FeatureSet::SUPPORTED) per image. Output TSV matches `spec.rs::FEAT_COLS` order post-2026-04-30 ablation (32 features).
- Parity test: no explicit feature-extraction-only binary outside `--features-only` mode in pareto harness.

### Picker Training Entrypoints
- `benchmarks/picker_v0.1_holdout_ab_2026-04-30.md`: Documents full training pipeline:
  1. `train_hybrid.py` (2373 lines in zenanalyze repo) with `--codec-config zenwebp_picker_config` (Python config from zenanalyze/zentrain/examples/).
  2. `bake_picker.py` (zenanalyze/tools/) invoked with `--dtype f16 --allow-unsafe`. Outputs ZNPR v2 binary.
  3. Training hyperparams in report: `--hidden 128,128,128 --activation leakyrelu`.
- No Cargo.toml target for training (external Python pipeline in zenanalyze workspace).

### Picker Bake / Runtime
- `src/encoder/picker/mod.rs` (1.3K): Exposes v0.1 picker API under `picker` cargo feature. Feature-gates runtime.rs.
- `src/encoder/picker/spec.rs` (8.2K): Cell taxonomy (6 cells from method×segments Cartesian product), feature schema (32 FEAT_COLS post-ablation), schema_hash=0xb2aca28a2d7a34ec, N_OUTPUTS=24 (6 cells × 4 heads: bytes_log + 3 scalar heads).
- `src/encoder/picker/runtime.rs` (13K): Loads `zenwebp_picker_v0.1.bin` (138K, ZNPR v2 format) at first call. Wraps via AlignedModel<16> for u64 alignment requirement. Defines ANALYSIS_FEATURES const (32 features) matching FEAT_COLS. Main API: `pick_tuning(raw_feats, w, h, target_zensim, &constraints)` → `TuningPick`. Argmin over bytes_log block with allowed_mask; reads scalar heads. Falls back to bucket-table on error (silent, no misc-calibration).
- Baked model: v0.1 trained on 2026-04-30 corpus (1264 images, 4 size variants each, 30 q-values per config = ~151k training examples per cell). v0 placeholder manifest at `zenwebp_picker_v0.manifest.json` (unused).

### Tests
- No dedicated picker inference tests in src/ beyond inline doc examples.
- `dev/picker_ab_eval.rs`: Golden A/B regression gate (41 held-out CID22 images, 4 targets). Results: -1.20% bytes vs bucket-table at matched zensim. Picker beats 51.8% of (image, target) pairs, loses 46.3%, ties 1.8%.
- CI (`.github/workflows/ci.yml`): Runs `cargo test` (no default features, no_std, +imgref, +mode_debug). Does NOT run pareto sweep or picker training. Does NOT check picker inference or feature-extraction parity.

### Docs / Scripts / Runbooks
- `README.md`: Features table; no picker docs.
- `benchmarks/picker_v0.1_holdout_ab_2026-04-30.md`: Full build invocation (train → bake → A/B). Only picker documentation.
- `scripts/zenwebp_picker_distill.py` (150+ lines): Python distill harness. Loads pareto TSV + features TSV. Fits HistGradientBoostingRegressor per cell, trains MLP on soft targets. Output: JSON consumable by `bake_picker.py`. Feature schema must match `spec.rs::FEAT_COLS` (hardcoded FEAT_COLS mismatch in script: 14 features vs 32 in runtime — ***bug***).
- No CI runbook. Training is manual, requires external zenanalyze workspace.
- No schema drift detection or versioning beyond compile-time const hash.

### CI Integration
- CI does NOT run picker training. No release gate.
- No integration tests for picker inference + feature extraction parity.
- Manual A/B eval via `picker_ab_eval` (requires codec-corpus).

---

## Data Flow

1. **Pareto Sweep** (`zenwebp_pareto.rs`):
   - Load PNGs from corpora (CID22-512 training, gb82, gb82-sc, CLIC2025).
   - Resize to {64, 256, 1024, native} on long edge (Lanczos via zenresize).
   - For each image, size variant: extract 32 zenanalyze features → features TSV (row: image_path, width, height, feat_*).
   - For each image, size, config (576 total), q-value (30 total): encode → decode → measure zensim + wall time → pareto row (bytes, zensim, encode_ms, total_ms, effective_max_zensim ceiling).
   - Output: `zenwebp_pareto_YYYY-MM-DD.tsv` (image, size_class, config_name, q, bytes, zensim, encode_ms, total_ms, effective_max_zensim).
   - Parallelization: rayon, default 16 cores.
   - Wall time estimate: ~14-16h on 16 cores (1264 images × 4 sizes × 576 configs × 30 q = ~89M encodes, but batch filtering).

2. **Feature Distillation** (`scripts/zenwebp_picker_distill.py`):
   - Read pareto TSV + features TSV.
   - Per (image, target_zensim) key: collect (cell_idx, bytes) rows.
   - Build X_engineered (features + size_onehot + log_pixels_poly + cross-terms + icc_bytes = 82 inputs).
   - Fit HistGB regressor per cell (predicting log_bytes from features).
   - Train single MLP on soft targets (HistGB outputs).
   - Output: JSON (weights, biases, metadata).

3. **Bake to ZNPR v2** (`zenanalyze/tools/bake_picker.py`):
   - Load JSON from distill step.
   - Hash schema (FEAT_COLS, extra_axes, SCHEMA_VERSION_TAG).
   - Serialize to ZNPR v2 binary with dtype=f16, alignment=16.
   - Emit schema_hash.

4. **Runtime Inference** (`src/encoder/picker/runtime.rs`):
   - At encoder Preset::Auto site: call `pick_tuning(rgb, w, h, target_zensim, constraints)`.
   - Extract 32 features from rgb (zenanalyze::analyze_features_rgb8).
   - Engineer input vector (match distill spec: 82 inputs).
   - Load ZNPR v2 model (lazy, cached).
   - Verify schema_hash.
   - Forward pass → output vector [bytes_log[6], sns[6], fs[6], sh[6]].
   - Argmin(bytes_log) with allowed_mask → cell index.
   - Read scalar heads at cell index → (sns, fs, sh, method, segments).
   - Return TuningPick or fallback to bucket-table.

---

## Performance Bottlenecks

### Hot Loops (Wall Time)
1. **Pareto encode loop** (~89M encodes total):
   - Per-image: zenresize Lanczos (Θ(pixels)) → 4 resized variants per corpus image.
   - Per config: 576 × 30 = 17,280 encode tasks per image per size.
   - Encode latency: ~10-100ms per image per config (method 4 slower, m6 faster; higher Q = longer).
   - Feature extraction: ~1-5ms per image (zenanalyze full feature set).
   - **Estimate**: 1264 images × 4 sizes × 576 configs × 30 q ≈ 88M encodes @ 20ms avg = 1.6M seconds = 444 CPU-hours. On 16 cores: ~28h wall (theoretical), practical: 14-16h with rayon + batching/filtering.
   - **Savings**: `--features-only` mode skips encodes (1 second / 1000 images).

2. **Training loop**:
   - HistGradientBoostingRegressor × 6 cells (sklearn, single-threaded per cell in distill script).
   - MLP training: ~30s with PyTorch student (observed in report), 10-20× slower with sklearn-relu (default = 300-600s for full training).
   - **Estimate**: 30-60s total with leakyrelu; critical path if hyperparameters change.

3. **Bake step**:
   - ZNPR v2 serialization (weights quantization f32→f16): ~1s.
   - Fast; not a bottleneck.

4. **Inference**:
   - `pick_tuning` per image: feature extraction (zenanalyze, ~1-5ms) + forward pass (ZNPR v2, ~0.1ms) + argmin (6 cells, ~0.01ms).
   - **Total**: ~1-5ms per image, dominated by feature extraction. Called once per image at Preset::Auto site.

### Reproducibility Gaps
1. **No seed pinning** in pareto harness (rayon + crossbeam default = platform-dependent scheduling).
2. **Feature extraction**: zenanalyze features are deterministic (no RNG), but feature column ordering in `spec.rs::FEAT_COLS` is hardcoded. **Mismatch in distill script**: FEAT_COLS has 14 names (old schema) vs 32 in runtime (2026-04-30 ablation). ***Critical bug***: if a new sweep is run, distill.py will fail or use wrong feature indices.
3. **Dataset hash**: no corpus versioning (path-based loading; no content hash).
4. **Schema drift**: `SCHEMA_VERSION_TAG = "zenwebp.picker.v0.1"` is a string tag, not a structured version. Hash verification exists (0xb2aca28a2d7a34ec) but hash is hardcoded in runtime; re-baking requires manual hash update.
5. **No lockfile** for Python deps in distill/bake steps (train_hybrid.py, bake_picker.py live in zenanalyze repo; version not pinned).

### Per-Codec Duplication
1. **Pareto harness**: zenwebp has unique `zenwebp_pareto.rs` (600 lines). zenjpeg has `ssim2_pareto_sweep.rs` (examples/) + `zq_pareto_calibrate.rs` (examples/). Both custom-written; no shared framework.
2. **Feature extraction**: zenwebp uses zenanalyze (32 features post-ablation). zenjpeg would use different features (ICC encoding, JPEG-specific metrics). No shared feature schema.
3. **Training pipeline**: Both delegate to `train_hybrid.py` (zenanalyze/zentrain/tools/) but require codec-specific configs:
   - `zenwebp_picker_config.py` (zenanalyze/zentrain/examples/).
   - Would be `zenjpeg_picker_config.py` (if/when zenjpeg picker trained).
   - No shared trainer or abstraction.
4. **Bake step**: Both use `bake_picker.py` (same file); only difference is input JSON from distill step.
5. **Duplication size**: Each codec needs:
   - Custom sweep harness (~600 lines Rust).
   - Custom distill script (~150 lines Python).
   - Custom codec config for train_hybrid.py (~50 lines Python).
   - Total per codec: ~800 lines (60% duplication; feature extraction + training loop are shared via train_hybrid.py).

### Schema Drift
1. **FEAT_COLS mismatch**: `distill.py` hardcodes 14 features; `spec.rs` lists 32. This would silently produce wrong training data if a new sweep is collected with the current codebase.
2. **No version tag in data files**: TSVs don't include schema_hash or SCHEMA_VERSION_TAG. A stale pareto TSV + old distill.py would produce a model with wrong feature mapping.
3. **Cell reordering**: `spec.rs::CELLS` comment: "order is load-bearing". No test enforces this invariant. If a future refactor reorders CELLS, baked models from old cells would be silently misaligned.

### Untested Code Paths
1. **Picker inference**: No unit tests. A/B eval (picker_ab_eval) is manual; not in CI.
2. **Feature extraction parity**: No test verifying zenanalyze feature extraction matches the 32-feature schema in FEAT_COLS.
3. **Schema hash verification**: No test that runtime's SCHEMA_HASH matches a freshly baked model's hash.
4. **Fallback paths**: Picker error → bucket-table fallback is not tested. Silent failure mode.
5. **Edge cases**: Tiny images (64×64), extreme Q values (0, 100), misaligned buffers, ICC data not tested.

### Missing Tools
1. **No schema versioning tool**: Can't safely roll a new feature set without breaking old models.
2. **No feature-parity test**: Can't verify distill.py will read features in the right order.
3. **No seed reproducibility**: rayon scheduling is non-deterministic; no `--seed` flag to zenwebp_pareto.rs.
4. **No corpus hashing**: Can't verify the training corpus hasn't drifted.
5. **No picker model introspection tool**: Can't inspect baked ZNPR v2 binary (weights, layer sizes, activation functions).
6. **No config diff tool**: Can't compare two pareto TSVs to see which configs regressed.

---

## Reproducibility

### Seeds
- **Pareto sweep**: rayon default thread pool (non-deterministic scheduling). No `--seed` flag.
- **Distill script**: sklearn/PyTorch defaults. `SEED = 0xC0DE` hardcoded in script but not passed to sklearn/torch.
- **Train_hybrid.py**: No explicit seed initialization observed in report.
- **Result**: Same corpus → different models on different runs (minor, but non-zero drift).

### Schemas
- **Feature schema**: `spec.rs::FEAT_COLS` (32 post-ablation, hardcoded, order load-bearing).
- **Cell taxonomy**: `spec.rs::CELLS` (6 cells, Cartesian method×segments, order load-bearing).
- **Extra axes** (distill.py): size_onehot (4) + log_pixels + log_pixels^2 + target_z_norm + target_z_norm^2 + target_z_norm×log_pixels + target_z_norm×feat[i] + icc_bytes (implicit in engineering).
- **Schema hash**: Computed by `bake_picker.py` as hash(FEAT_COLS, extra_axes, SCHEMA_VERSION_TAG). Runtime verifies against hardcoded const.
- **Risk**: If distill.py's feature engineering changes, hash will mismatch but no signal to re-bake (hash is a compile-time const).

### Dataset Hashes
- **No content hash**: Corpus loaded by path. CID22-512, gb82, gb82-sc hardcoded in pareto.rs.
- **No version tag**: Pareto TSVs don't include corpus name/hash in header. Swapping corpora mid-run would silently produce stale data.
- **Training data**: ~1264 images × 4 size variants × 30 q-values = 151k examples per cell. Deterministic after shuffle(seed), but seed is not locked.

---

## Duplication with Other Codecs

### zenjpeg
- `zenjpeg/examples/zq_pareto_calibrate.rs` (~400 lines): Different design (Q-specific rather than Pareto sweep). Emits per-Q training targets.
- `zenjpeg/examples/ssim2_pareto_sweep.rs` (~200 lines): SSIM-based variant.
- `zenjpeg/scripts/zq_pareto_fit.py`: Fits quantization curve; different objective (Q prediction, not picker cells).
- **Duplication**: 40% (feature extraction, training pipeline delegation to train_hybrid.py are shared).

### zenavif / zenjxl
- No observed pareto/picker infrastructure yet (early in picker adoption).

### Shared Pieces (Could Be Unified)
1. **train_hybrid.py** (2373 lines, zenanalyze/zentrain/tools/): Codec-agnostic, used by all.
2. **bake_picker.py** (zenanalyze/tools/): Codec-agnostic, used by all.
3. **zenanalyze feature extraction**: Shared library call; each codec extracts full feature set, then selects subset in spec.rs.
4. **ZNPR v2 format**: Shared inference runtime (zenpredict crate).

### Duplication Costs
- **Sweep harness**: ~600 lines per codec (zenwebp ~2400 lines total; 25% of diff from zenjpeg is method/axis differences).
- **Distill script**: ~150 lines per codec (schema-aware; non-trivial to generalize).
- **Codec config** (train_hybrid.py): ~50 lines per codec (Python, lightweight).
- **Total**: ~800 lines of codec-specific boilerplate per codec. Could reduce to ~300 with shared template + macro.

---

## Test Coverage

### What's Tested
- **Encoder roundtrip** (src/lib.rs + tests/): Decode golden WebP files, re-encode, decode, compare pixels.
- **SIMD tier parity**: All SIMD tiers (SSE2, SSE4.1, AVX2, NEON, WASM128) against scalar baseline.
- **Format compliance**: VP8, VP8L, extended format, metadata, animation (via golden test suite).

### What's NOT Tested
- **Picker inference**: No unit tests for `pick_tuning()`. A/B eval is manual (picker_ab_eval, not in CI).
- **Feature extraction parity**: No test verifying zenanalyze features match schema.
- **Schema hash verification**: No test confirming baked model hash matches expected value.
- **Pareto sweep correctness**: No validation that sweep produces expected cell coverage.
- **Fallback behavior**: Picker error → bucket-table fallback not tested.
- **Edge cases**: Tiny images (64px), extreme Q (0, 100), ICC data, animation (picker only handles lossy single-frame).
- **Regression gates**: No CI gate preventing stale picker models from shipping.
- **Integration tests**: No end-to-end test (sweep → distill → bake → encode with picker).

### CI Coverage
- Default test suite: encoder/decoder roundtrip, SIMD tiers, format compliance.
- NO picker tests (gated on `picker` feature; not in default CI).
- NO ablation/regression gates (no schema hash check in CI).

---

## What Would Make This Faster/Better

### Eliminate Hot Loops
1. **Parallel distill script** (`zenwebp_picker_distill.py`): HistGB per-cell fitting is single-threaded. Replace sklearn with joblib parallelization or GPU-accelerated gradient boosting (XGBoost, LightGBM). **Gain**: 3-5× speedup (30s → 6-10s).
2. **Feature extraction caching**: Pareto harness re-extracts features for every size variant. Cache zenanalyze features per unique image, interpolate/weight for size variants. **Gain**: 20-30% sweep time reduction.
3. **Quantization-aware training**: Train on quantized weights (baked as f16) instead of f32, then quantize. Saves bake step. **Gain**: Simplify pipeline, faster iteration.

### Improve Reproducibility
4. **Deterministic rayon seed**: Add `--seed` flag to zenwebp_pareto.rs, pin rayon thread pool order. Seed train_hybrid.py and sklearn/torch. **Gain**: Reproducible models, easier debugging.
5. **Corpus content hashing**: Compute SHA256 of corpus images, emit in pareto TSV header. Refuse to train on mismatched corpus. **Gain**: Detect stale/drifted data, prevent silent model corruption.
6. **Schema versioning tool**: Auto-detect schema drift (FEAT_COLS length mismatch), refuse to train with stale distill.py. **Gain**: Catch bugs early, prevent silent feature reordering.

### Test & Gate
7. **Picker inference unit tests**: Mock 32-feature vector, call pick_tuning(), verify TuningPick structure. Test argmin correctness, fallback on error. **Gain**: Catch inference bugs, catch schema hash mismatches.
8. **Feature-extraction parity test**: Compare zenanalyze::analyze_features_rgb8() output against distill.py's feature vector construction. **Gain**: Catch feature reordering bugs.
9. **CI regression gate**: Run picker_ab_eval on held-out set in CI. Fail if Δbytes > threshold. **Gain**: Prevent regressions from shipping; early warning on model drift.

### Reduce Duplication
10. **Shared sweep template**: Extract common pareto-harness pattern (load images, resize, extract features, encode, score) into a generic library. Implement codec-specific axes + metric function. **Gain**: 50% reduction in sweep code per codec; easier to add axes (e.g., sharp_yuv for future expansion).
11. **Schema versioning framework**: Auto-generate FEAT_COLS, CELLS, schema_hash from a config file. Use in both Rust (compile-time const) + Python (distill.py). **Gain**: Single source of truth; eliminate distill.py FEAT_COLS mismatch bug.

---

## Top 3 Pain Points

1. **Feature schema drift** (distill.py mismatch bug): distill.py hardcodes 14 features; runtime expects 32. New sweep + old distill = silent wrong training data. No validation or schema versioning. **Fix**: Add schema hash check in distill.py, auto-generate feature list from spec.rs.

2. **No picker tests in CI**: Picker inference gated on `picker` feature; A/B eval is manual (not in CI). Silent fallback on error (no misc-calibration). **Fix**: Unit tests for pick_tuning(), CI gate on regression threshold.

3. **Pareto sweep wall time** (14-16h): Encode 89M images at 576 configs × 30 q-values. Dominant cost. No parallelism beyond rayon. Feature extraction not cached. **Fix**: Cache zenanalyze features per unique image; parallel distill script (GPU acceleration); quantization-aware training.

---

## Top 3 Missing Tooling Suggestions

1. **Schema versioning tool**: Auto-detect FEAT_COLS / CELLS / schema_hash drift. Generate Rust consts + Python distill.py from single config. Validate corpus before training.

2. **Deterministic reproduction kit**: `--seed` flag for zenwebp_pareto.rs + train_hybrid.py + distill.py + bake_picker.py. Corpus content hash in TSV headers. Enables "build picker reproducibly from corpus hash."

3. **CI regression gate**: Picker inference unit tests (mock features, call pick_tuning()) + A/B eval on held-out set in CI. Fail if Δbytes > +0.5%. Prevents silent regressions.

