# zenavif training/picker audit, 2026-05-02

## Inventory

**Sweep & Oracle Harnesses:**
- `examples/predictor_sweep.rs` (29.4 KB): Multi-image sweep driver for rav1e knob prediction. Axes: 11 speeds × 21 q-points (5..100:5) × 4 sizes (64, 256, 1024, 4096 maxdim) × configurable knobs. Outputs TSV schema with `(image_path, size_class, width, height, config_id, config_name, sha256, content_class, source, size_bucket, speed, q, qm, vaq, vaq_strength, tune_still, bytes, zensim, encode_ms, decode_ms)`. Resumable via `--append`. Phases: 1a baseline, 2 OAT (per-knob perturbations), 3 LHS joint (64 sampled tuples).
- `examples/encode_sweep.rs`: Alternative AVIF encode harness (16.1 KB); narrower schema for encode-time profiling.
- `benchmarks/`: 89,601 rows in `rav1e_phase1a_2026-04-30.tsv`, 2,575 rows Phase 2 OAT, 89,601 rows Phase 3 LHS.

**Feature Extraction:**
- `examples/extract_features.rs` (13.2 KB): zenanalyze-based; resizes to maxdims matching predictor_sweep. Outputs ~40 features post-ablation (from ~100 raw).
- `examples/extract_features_natural.rs` (10.0 KB): Native-size variant; preserves source dims via `size_bucket`.
- 449 rows in `rav1e_phase1a_features_2026-05-01.tsv`.

**Picker Training & Configuration:**
- `training/rav1e_picker_config.py` (8.1 KB): v0.2 config for `train_hybrid.py`.
  - CATEGORICAL_AXES: speed (1..10), qm (0/1)
  - SCALAR_AXES: vaq_strength, seg_boost, rdo_tx_off, seg_complex_on, bottomup_on, lrf_on, partition_range_idx
  - Feature set: 29 load-bearing post-ablation
  - ZQ targets: q ∈ [30,95] step 5 dense + 70..95 step 2 perceptibility band
  - Output: `benchmarks/rav1e_picker_v0_1.json` (3.5 MB hybrid-heads MLP)
- Supporting: `clean_tsv.py`, `build_encode_ms_lut.py`, `build_quality_lut.py`, `analyze_phase2_oat.py`.

**Picker Bake & Runtime:**
- `benchmarks/rav1e_picker_v0_1.json` (3.5 MB ZNPR v2)
- Runtime API: `zenavif::EncoderConfig::auto_tune(target_zensim)` planned behind `auto-tune` cargo feature — **NOT YET INTEGRATED** (TODO in CLAUDE.md)
- Features consumed: 29 + size/aspect + target_zensim_norm

**Tests:**
- `tests/encode_roundtrip.rs`: Minimal 16×16 RGB8/RGBA8 only. No picker coverage.
- CI: multi-platform (ubuntu, macos, windows; arm64 select) — no picker/sweeper tests.
- **Gap**: No regression for picker predictions, feature parity, golden TSV rows.

## Data Flow

1. Manifest TSV `(sha256, split, content_class, source, size_bytes, path)`
2. Lanczos3 resize to maxdim (64, 256, 1024, 4096); skip upscale
3. `analyze_features_rgb8` → ~100 raw features → cull 29
4. Sweep encoding: (image, size, speed, q, knobs) → encode (ravif) → decode (zenavif) → zensim → TSV row
5. Join features + pareto on `(image_path, size_class)`
6. `train_hybrid.py` → hybrid-heads MLP
7. `bake_picker.py` → ZNPR v2 f16 binary
8. Runtime: (features, target_zensim) → predictor → (speed, q, qm, knob_overrides)

## Performance Bottlenecks

1. **Phase 3 LHS = ~50 hours**: 800k encodes serial. `--enc-threads 1` default; 16-core box badly underutilized. Projected 8–10× speedup with shared rayon pool + 2 threads/encode.
2. **Phase 1a = ~6 hours**: 50 images × 4 sizes × 11 speeds × 21 q = 46,200 encodes. Median ~450ms/encode.
3. **Phase 2 OAT = ~3 hours**: 30 images × 4 sizes × 9 perturbations × 21 q = 22,680 encodes.
4. Tiny size class (64×64) underrepresented; CID22 mostly 512px source forces aggressive downscaling.

## Reproducibility

1. **No seed pinning**: `stratified_subset()` deterministic via sha256 sort, but `train_hybrid.py` seed not exposed.
2. **Feature ablation non-idempotent**: Re-running `feature_ablation.py` may produce different culls if zenanalyze or scorer changes.
3. **config_id 28-bit packing brittle**: bit layout `speed(4) q(7) qm(1) vaq(1) strength*4(4) tune(1) segb*4(4) rdo(1) segc(1) bu(1) lrf(1) pridx+1(2)`. No version tag in TSV header.
4. **Manifest format unversioned**: Adding columns silently breaks downstream parsing.
5. **v0.1 5.57pp train→val gap** flagged corpus-starved; v0.2 timeline unclear.

## Duplication with Other Codecs

1. **Three sweepers, one pattern**: zenavif `predictor_sweep.rs`, zenwebp `zenwebp_picker_sweep.rs`, zenjpeg `sweep_*.rs`. Same Phase 1a → OAT → LHS path. No shared library.
2. **Feature extractor trio**: zenavif/zenwebp/zenjxl each ship near-identical extract_features.rs.
3. **Per-codec training configs**: hardcoded paths, no templating.
4. `bake_picker.py` is the one shared piece (in zenanalyze/tools/).

## Test Coverage

1. Encode roundtrip: only 16×16, fixed q/speed
2. No picker prediction tests
3. No feature-parity tests across sweep versions
4. No golden TSV row validation
5. Manifest parsing untested (silent skip on malformed rows)
6. **Non-pow2 size assertion missing**: cross-codec report flags zenavif Phase B/C skipped due to harness panic on non-pow2 sizes (e.g., 351×468 wide_aspect images).

## What Would Make This Faster/Better

1. Parallelize encodes across image+size dims via shared rayon pool — 8-10× Phase 3 speedup.
2. Lock feature set by commit hash; CI gate against silent ablation drift.
3. Extract shared `zenpicker-harness` crate parameterizing sweep skeleton across codecs.
4. TSV schema versioning + validation on append.
5. Fix non-pow2 harness panic blocking expanded-corpus runs.
6. New `zenpicker-validation` crate: sample target_zensim + features, predict, encode, measure actual vs target. CI per-codec post-bake.
7. Resume checkpointing for sweep harnesses (`.offset` markers).
8. Encode-time-cached zensim reference (per-(image, size_bucket)).
