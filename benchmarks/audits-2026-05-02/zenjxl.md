# zenjxl + jxl-encoder training/picker audit, 2026-05-02

## Inventory

### Source locations
- **zenjxl wrapper**: `/home/lilith/work/zen/zenjxl/` — thin zencodec adapter over jxl-encoder
- **jxl-encoder codec**: `/home/lilith/work/zen/jxl-encoder/jxl-encoder/` — native Rust JXL encoder with picker hooks
- **zenanalyze training infra**: `/home/lilith/work/zen/zenanalyze--zenjxl-adapter-2026-05-01/` — trainer, picker bake tools, feature extractor

### Oracle artifacts (2026-04-30 run)
- **lossy_pareto_2026-04-30.tsv**: 95 MB, 610k rows; lossy configs sampled at 8 cells × 6 k_info_loss_mul × 5 k_ac_quant × 6 entropy_mul_dct8 × 9 distances × ~N images/sizes
- **lossless_pareto_2026-04-30.tsv**: 22 MB, 165k rows
- **Feature TSVs**: `lossy/lossless_pareto_features_2026-04-30.tsv` (~199 KB each) with ~60 image features per (image, size_class) cell
- Schema: `[image_sha, split, content_class, size_class, width, height, distance, cell_id, {categorical_knobs}, {scalar_knobs}, sample_idx, bytes, encode_ms, butteraugli, ssim2]`

### Training pipeline
1. **Oracle harness**: `jxl-encoder/examples/{lossy,lossless}_pareto_calibrate.rs` (759 lines, 150+ lines for encode loop)
   - Cells: 8 (ac_intensity ∈ {compact, full} × enhanced_clustering {0,1} × gaborish {0,1} × patches {0,1})
   - Scalar bands: 6 × 5 × 6 = 180 combinations per cell/distance/size
   - Distances: 9 fixed (0.25..5.0 for lossy; none for lossless)
   - Per-image features: extracted via `zenanalyze::analyze_features_rgb8` with `FeatureSet::SUPPORTED`

2. **Adapter layer**: `zenjxl_oracle_adapter.py` (260 lines)
   - Rewrites lossy: `c{cell_id}_ac{ac_intensity}_ec{enhanced_clustering}_g{gaborish}_p{patches}_kil{k_info_loss_mul}_kaq{k_ac_quant}_ed8{entropy_mul_dct8}`
   - Rewrites lossless: `c{cell_id}_lz{lz77_method}_sq{squeeze}_p{patches}_rct{nb_rcts_to_try}_wp{wp_num_param_sets}_tmb{tree_max_buckets}_tnp{tree_num_properties}_tsf{tree_sample_fraction}`
   - Synthesizes image_path as `sha:<sha[:16]>` for join key matching
   - **CRITICAL**: Metric substitution — uses ssim2 (SSIMULACRA2) in zensim column, not XYB-Butteraugli; stored in `.meta` sidecar

3. **Feature extractor**: `extract_features_for_picker.rs` (317 lines)
   - Reads manifest (sha256, split, content_class, source, path)
   - Lanczos3 resize to target_maxdim (64, 256, 1024, native)
   - Emits standardtrain features TSV: `image_path, image_sha, split, content_class, source, size_class, width, height, feat_*`
   - Uses same size_class labels as pareto oracle (tiny/small/medium/large)

4. **Picker training**: `train_hybrid.py` (~800 lines)
   - Config-name parser invoked via `load_codec_config(zenjxl_picker_config)` — dynamically imports
   - Categorical axes: `[cell_id, ac_intensity, enhanced_clustering, gaborish, patches]`
   - Scalar axes: `[k_info_loss_mul, k_ac_quant, entropy_mul_dct8]`
   - ZQ_TARGETS: `range(30, 70, 5) + range(70, 96, 2)` = 21 quality levels (ssim2 space)
   - Outputs: JSON model + `.log` diagnostic file

5. **Bake → inference**: `bake_picker.py` → `zenpredict-bake` (Rust CLI) → `.bin` (ZNPR v2 format)
   - Activations: `relu, leakyrelu, identity`
   - Dtypes: `f32, f16, i8` (i8 per-output quantized)
   - Manifest sidecar: `.bin.manifest.json` (legacy compatibility)

### Runtime integration in jxl-encoder
- `jxl-encoder/src/api.rs`: `LossyConfig` + `LosslessConfig` both expose `.with_effort_profile_override(EffortProfile)` hook
- The picker is NOT integrated yet — oracle → adapter → features → training happens upstream; codec would need to call picker at encode time (missing)
- Current flow: manual EffortProfile tuning only

### Codec config module
- **zenjxl_picker_config.py** (217 lines):
  - Paths: `~/work/zen/zenjxl/benchmarks/zenjxl_{lossy,lossless}_pareto_*.tsv` + `zenjxl_{lossy,lossless}_features_*.tsv`
  - Output: `~/work/zen/zenanalyze/benchmarks/zenjxl_hybrid_*.{json,log}`
  - KEEP_FEATURES: 60 features (Tier 1/3 + new kurtosis/smoothness features from zenanalyze 0.1.0)
  - Scalar display ranges documented for human-readable logs

---

## Data flow (lossless vs lossy)

### Lossy path (9h oracle run → 610k rows)
```
lossy_pareto_calibrate.rs
  └─ 8 cells × 180 scalar configs × 9 distances × N images/sizes
     ├─ encode w/ single-shot quantization (butteraugli_iters=0)
     ├─ emit: [bytes, encode_ms, butteraugli, ssim2]
     └─ zenanalyze::analyze_features_rgb8 (per image, per size_class)

lossy_pareto_2026-04-30.tsv (610k rows, 20 cols)
  ↓
zenjxl_oracle_adapter.py
  ├─ synthesize config_name from knob tuple
  ├─ config_id = stable_hash(config_name) & 0x7FFF_FFFF
  ├─ image_path = sha:<sha[:16]>
  ├─ metric: ssim2 → zensim column (with .meta warning)
  └─ OUTPUT: zenjxl_lossy_pareto_2026-05-01.tsv (same row count, rewritten schema)

lossy_pareto_features_2026-04-30.tsv (per-image-per-size features)
  ↓ (no re-encoding)
  └─ zenjxl_lossy_features_2026-05-01.tsv (copied/reindexed by sha:[16])

train_hybrid.py --codec-config zenjxl_picker_config
  ├─ load Pareto TSV keyed by (image_path, size_class)
  ├─ per (image, size, zq_target): compute within-cell optimal bytes
  ├─ build dataset: X = [features + size_oh + log_px + zq_norm + cross_terms], Y = [bytes_log per cell + scalars]
  ├─ train MLPRegressor or HistGradientBoostingRegressor (safety checks, holdout eval)
  └─ OUTPUT: zenjxl_hybrid_2026-05-01.json (sklearn-serializable dict)

bake_picker.py --model zenjxl_hybrid_2026-05-01.json
  ├─ spawn zenpredict-bake (Rust)
  └─ OUTPUT: zenjxl_hybrid_2026-05-01.bin (ZNPR v2 binary format)
```

### Lossless path (165k rows, parallel structure)
- Cells: 8 (discrete combinations of lz77_method, squeeze, patches, nb_rcts_to_try, wp_num_param_sets, tree_max_buckets, tree_num_properties, tree_sample_fraction)
- No quality axis (lossless is binary: either it works or it doesn't)
- Same adapter, trainer, bake pipeline
- **Duplication**: identical column-rewriting, feature-loading, cell-index logic; only CATEGORICAL_AXES / SCALAR_AXES differ

---

## Performance bottlenecks

### Oracle bottleneck: **9-hour wall time**
- Per-image encode cost dominates: lossy_pareto_calibrate does **1,440+ encodes per image** (8 cells × 6 k_info × 5 k_ac × 6 entropy × 9 distances)
- Feature extraction is **separate, sequential** — does NOT parallelize with oracle encoding
- Row count: 610k lossy + 165k lossless = 775k total rows, emitted single-threaded from oracle harness
- **No wall-time metadata**: partial TSVs (`_v1a_partial`, `_v1b_partial`) suggest multi-stage runs, but no `.meta` file documents stages/timings

### Feature extraction bottleneck: **sequential image I/O**
- `extract_features_for_picker.rs` processes manifest sequentially, 10-image progress snapshots
- Resize + RGB→linear + 60-feature analysis per (image, size_class) is CPU-bound but not parallelized
- No batching across images; each image loaded to RAM in full

### Training bottleneck: **metric substitution + schema mismatch**
- ssim2 vs zensim cross-codec: can't aggregate JXL results with zenavif/zenwebp/zenjpeg without per-codec normalization
- ZQ_TARGETS (21 levels in ssim2 space) vs other codecs' ZQ_TARGETS (different metric) → picker generalization risk
- No "ceiling" column in lossy oracle (unlike zenjpeg): trainer's `effective_max_zensim` safety check doesn't gate unreachable targets

### Bake → inference gap: **no integration**
- Picker `.bin` file is trained but **not loaded or called at encode time**
- jxl-encoder's LossyConfig/LosslessConfig have hooks for EffortProfile override, but no code path that loads zenjxl_hybrid_*.bin and uses it
- **Missing feature**: zenjxl wrapper needs to instantiate zenpredict::Model, call .predict(features), and wire scalars back to config

---

## Reproducibility

### Oracle reproducibility: GOOD
- Scalar sampling is seeded by `fastrand::Rng::with_seed(hash(image_sha, size_class, cell_id, distance, sample_idx))`
- Distances are fixed array `[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]`
- Re-running same manifest → same configs, same encodes (assuming jxl-encoder determinism)
- **Risk**: partial runs (`_v1a`, `_v1b`) suggest restarts; no resume checkpointing, risk of duplicate/missing rows if interrupted mid-run

### Feature extraction reproducibility: GOOD
- Lanczos3 resize, zenanalyze v0.1.0 features are deterministic
- size_class labels hardcoded (64→tiny, 256→small, 1024→medium, 0/4096→large)
- **Risk**: manifest order-dependent; if manifest changes, row order shifts but values stable

### Training reproducibility: GOOD (with caveats)
- numpy random seed set explicitly in `_train_torch_leakyrelu_student`
- Holdout split is fixed by seed
- **Risk**: sklearn version drift (MLPRegressor fit is deterministic but coefficients may shift between sklearn versions)
- **Risk**: no validation split schema — `validation_fraction=0.1` is internal; external test set is separate image-holdout

### Bake reproducibility: GOOD
- zenpredict-bake is deterministic: same JSON → same binary
- **Risk**: ZNPR v2 format may change; no versioning in picker binary for model schema migrations

---

## Duplication with other codecs

### Within zenjxl (lossy vs lossless)
1. **Oracle harness**: separate `.rs` files (lossy/lossless_pareto_calibrate.rs), ~same structure but hardcoded knob grids differ
2. **Adapter**: single `zenjxl_oracle_adapter.py` handles both via `knobs` parameter + `config_synth` function pointer → **good factorization**
3. **Feature extraction**: single `extract_features_for_picker.rs` handles both (features are image-intrinsic, not codec-specific) → **good**
4. **Picker config**: single `zenjxl_picker_config.py` with paths for both lossy + lossless → **could split into codec-config module per zenanalyze design**
5. **Training**: same `train_hybrid.py`, but CATEGORICAL_AXES/SCALAR_AXES must be declared in codec config → **dynamic, not duplicated**

### Across codecs (zenjxl vs zenavif/zenwebp/zenjpeg)
- **Adapter pattern**: each codec has its own `*_oracle_adapter.py` (or equivalent); zenavif/zenwebp generate pareto TSVs natively
- **Feature extraction**: shared `zenanalyze::analyze_features_rgb8`; but codec-specific manifest format (path keying differs)
- **Trainer**: `train_hybrid.py` is codec-agnostic; each codec provides `{codec}_picker_config.py`
- **Schema drift**: each codec has hardcoded CATEGORICAL_AXES, SCALAR_AXES, ZQ_TARGETS — **no shared schema** → risk of silent misconfigurations

---

## Test coverage

### Unit tests: MINIMAL
- `/zenanalyze--zenjxl-adapter-2026-05-01/zenpredict/tests/`:
  - `lifecycle.rs`: model load/predict lifecycle
  - `json_bake.rs`: JSON bake round-trip
  - `/tools/test_bake_roundtrip.py`: sklearn → JSON → binary → Rust predict parity (tests relu/leakyrelu/identity × f32/f16/i8)
- **Missing**: oracle reproducibility test (re-run Oracle on subset, check row parity)
- **Missing**: feature extraction parity test (extract from same image → check feature stability)

### Integration tests: NONE FOUND
- No test of end-to-end: Oracle → adapter → features → training → bake
- No test of picker inference integration in jxl-encoder (because integration is incomplete)
- No cross-codec schema consistency test

### CI coverage
- `.github/workflows/ci.yml`:
  - Builds + tests zenpredict (Rust lib + integration tests for f32/f16/i8)
  - Builds + tests zenpicker, zenanalyze libs
  - Cross-compilation: i686-linux, aarch64-linux
  - **Missing**: training workflow (no CI runs `train_hybrid.py`)
  - **Missing**: oracle reproducibility check (CI doesn't re-run oracle)

---

## What would make this faster/better

### Immediate wins (1–2 weeks)
1. **Parallelize oracle**: encode loop uses `rayon::prelude::*` but manifest iteration is sequential. Batch images across cells/distances in work queue → 3–4× speedup
2. **Cache feature extractions**: memoize zenanalyze results per (image, size) across oracle runs. Re-encode lossy/lossless sweeps; reuse features
3. **Resume checkpointing**: partial TSV files should have `.offset` marker indicating last-completed image; oracle restart skips completed work

### Medium-term (2–4 weeks)
4. **Metric audit**: measure ssim2 vs zensim on JXL sweep corpus; if numerically close (<2% mismatch), substitute inline; if not, re-measure sweep with zensim
5. **Integrate picker into zenjxl**: add `with_picker(model_path: &str)` option to JxlEncoderConfig; call zenpredict at encode time to override EffortProfile
6. **Ceiling column for lossless**: oracle should measure `effective_max_bytes` per (image, size) analog to lossy's `effective_max_zensim`; blocks trainer's unreachable-target filtering

### Strategic (4–8 weeks)
7. **Unified oracle schema**: single `.rs` harness with shared cell/knob iteration logic, codec-agnostic TSV output; reduce lossy/lossless duplication
8. **Shared codec-config registry**: TOML or JSON manifest listing all codecs' categorical/scalar axes + ZQ_TARGETS; `train_hybrid.py` validates against registry before training
9. **Test suite**: regression tests for oracle (6-image smoke subset), feature extraction (determinism), and end-to-end training (known-good model comparison)

---

## Key risks

1. **Metric substitution (ssim2 vs zensim)**: JXL picker trains on SSIMULACRA2, not Butteraugli. Cross-codec aggregation or model transfer will mis-rank targets. **Mitigation**: measure cost to re-run oracle with zensim.

2. **Incomplete picker integration**: trained `.bin` file sits unused in benchmarks/. jxl-encoder encode() path doesn't load or call it. **Mitigation**: add zenpredict runtime to zenjxl, wire picker predictions to LossyConfig/LosslessConfig fields.

3. **Schema drift**: CATEGORICAL_AXES/SCALAR_AXES/ZQ_TARGETS live in codec config modules; no validation that they match oracle harness knobs. Silent mismatch → model trained on wrong columns. **Mitigation**: codec-config registry or compile-time assertion.

4. **Feature stability**: `zenanalyze::FeatureSet::SUPPORTED` changed 2026-04-30 (5 new features added). Old feature column names missing from new sweeps → NaN masking. **Mitigation**: feature versioning in oracle output; trainer warns on schema mismatch.

5. **No resumption**: oracle crashes mid-sweep, user re-runs from scratch, losing hours. Partial TSVs suggest staging but no documented protocol. **Mitigation**: add offset tracking + resume logic.

