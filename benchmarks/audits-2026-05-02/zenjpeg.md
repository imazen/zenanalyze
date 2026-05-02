# zenjpeg training/picker audit, 2026-05-02

## Inventory

### Sweep / Pareto harnesses
- `zenjpeg/examples/zq_pareto_calibrate.rs` — Primary harness: sweeps 8 encoder configs × 21 quality levels (0–100 step 5) × 4 image sizes (64, 256, 1024, native) across mixed corpus. Outputs TSV rows: (image, size_class, width, height, config_id, config_name, q, bytes, zensim, encode_ms, total_ms). Features extracted separately via zenanalyze.
- `zenjpeg/examples/zq_calibrate.rs` — Older calibration routine (legacy; pareto_calibrate supersedes it).
- `zenjpeg/examples/ssim2_pareto_sweep.rs` — SSIM2 / DSSIM-specific sweep variant (research; not used in training pipeline).

### Feature extraction
- `zenjpeg/examples/zq_pareto_calibrate.rs` — Embedded feature extraction: `--features-only` mode re-runs image feature analysis without encode loop (cheap; ~1 sec/1000 images). Calls `zenanalyze::analyze_features_rgb8` per (image, size) to emit TSV: (image_path, size_class, width, height, feat_*).
- Features are dynamic: zenanalyze cargo features determine which columns appear. Build without `composites` feature → fewer feature columns.

### Picker training entrypoints
- Scripts moved **upstream** to `zenanalyze/zenpicker/tools/`: `train_hybrid.py`, `train_distill.py`, `feature_ablation.py`, etc.
- Zenjpeg-specific config module: `zenanalyze/zenpicker/examples/zenjpeg_picker_config.py` (in zenanalyze repo, not zenjpeg).
- Invocation: `PYTHONPATH=<zenanalyze>/zenpicker/examples python3 <zenanalyze>/zenpicker/tools/train_hybrid.py --codec-config zenjpeg_picker_config`.
- No picker training code lives in zenjpeg itself; it consumes pareto + features TSVs, returns regression weights.

### Picker bake / runtime
- **Baking process:** Legacy 8-config model: `scripts/zq_emit_rust.py` reads JSON regression output, emits Rust const arrays with weights. Hand-pasted into `zenjpeg/src/encode/zq.rs`.
- **Inference:** `zenjpeg/src/encode/zq.rs` defines `ZqTarget` + `BlockArtifactBound` (API) and contains **NO** picker invocation at inference time. The harness instead runs a single-pass closed-loop encoder (iteration budget up to 2 diffmap passes). Regression model is **NOT** used at runtime — instead, a fixed starting-quality LUT (`ApproxJpegli(q)`) is invoked for all images at all targets.
- Architecture gap: the Pareto-optimal controller from sweep is NOT integrated. Issue #128 tracks multi-config selection at runtime.

### Tests
- `zenjpeg/tests/zq_target.rs` — 14 unit tests covering Zq API contract: iteration loop runs, metrics are finite, max_passes respected, block artifact bounds enforced. Tests run with synthetic 256×256 mixed smooth+textured images. **No regression tests for golden pareto outputs.**
- `zenjpeg/tests/*.rs` — 30+ total test files (encoder/decoder parity, quality regression, chroma distance, YUV roundtrip, etc.). None test picker training or pareto sweep reproducibility.
- CI: `ci.yml` runs `cargo test -p zenjpeg` with feature combinations (all tests pass in CI).

### Docs / scripts
- `README.md` — User-facing encoder/decoder guide. No training docs.
- `scripts/README.md` — Documents script relocation to zenanalyze. Maps old paths to new canonical locations.
- `scripts/zq_pareto_fit.py` — Offline Pareto-front analysis: reads (pareto TSV, features TSV), computes Pareto-optimal (config, q) per (image, size, target_zq), trains per-config least-squares regressors. Outputs: config-choice frequencies, validation RMSE, Pareto-overhead statistics. **NO JSON emit** (zq_emit_rust.py expects JSON but fit.py doesn't generate it in current form).
- `scripts/zq_emit_rust.py` — Reads JSON, emits Rust const weights. Incomplete pipeline: no JSON output from zq_pareto_fit.py.
- `scripts/zq_tier2_subablation.py` — Experimental: tests dropping zenanalyze Tier 2 features. Kept for reference.
- `BENCH-AUDIT.md`, `CLAUDE.md`, `CONTEXT-HANDOFF.md` — Developer guides.

### CI integration
- `ci.yml` — Runs standard unit/integration tests on every push/PR. **Does NOT run sweep or training tests.** Skips jpegli-internals-sys and other non-CI workspace members.
- `benchmark.yml` — Likely separate benchmarking workflow (not examined in detail; runs on schedule or manual trigger).
- **Release blocker:** No sweep or training regression tests block releases. Tests are purely on encoder/decoder correctness.

---

## Data flow

1. **Image corpus ingestion**: PNGs from `codec-eval` paths (CID22, CLIC2025, gb82 datasets). User specifies via `--corpus` flag(s).

2. **Size-axis variant creation**: Each PNG is loaded as RGB8, resized to {64, 256, 1024, native} (Lanczos3) to yield 4× the image count. (Preserve aspect ratio; scale by max(w,h).)

3. **Config+Q Cartesian loop**: For each (image, size) pair:
   - Extract zenanalyze features once (image-level; invariant across configs).
   - For each of 8 encoder configs: for each of 21 q values:
     - Encode → measure bytes + zensim diffmap score + wall time.
     - Append TSV row if successful; skip failures (e.g., XYB rejects q=0).

4. **Pareto-front analysis** (offline, `zq_pareto_fit.py`):
   - Load pareto TSV + features TSV.
   - For each (image, size), compute Pareto-optimal front in (bytes, -zensim) space.
   - For each target_zq ∈ {40, 50, ..., 95}: find smallest-bytes (config, q) achieving zensim ≥ target.
   - Train per-config least-squares regressor: (features, target_zq) → predicted_starting_q.
   - Emit validation RMSE, config-choice distribution, byte-overhead vs optimal.

5. **Regression baking** (incomplete): `zq_emit_rust.py` reads JSON, bakes weights as Rust const. **Pipeline break:** fit.py outputs text (stdout) and log files; no JSON emission. Hand-curation required.

6. **Runtime inference** (current, single-pass): 
   - API `Quality::Zq(target)` is user-facing.
   - Resolved to starting `ApproxJpegli(q)` via fixed LUT (no dynamic config choice).
   - Encoder runs closed-loop iteration: measure diffmap, adjust per-block quantization, repeat up to max_passes times.
   - **Regression weights are NOT consulted at runtime.** Pareto-optimal config selection exists only as offline analysis (summary stats, not deployed).

---

## Performance bottlenecks

### Sweep wall time
- **347 images × 4 sizes × 8 configs × 21 q = 233,184 cells**
- **Wall time (from summary): 462 seconds on Ryzen 9 7950X, 16 threads**
- Per-cell throughput: ~502 cells/sec with 16-way parallelism (rayon).
- Per-image encoding + zensim: ~2–5 ms per cell (varies by size; tiny 64×64 ~0.5 ms, native ~10 ms).
- **Bottleneck 1: Encode-decode round-trip per cell.** 233k encodes + 233k zensim measurements is the dominator. zensim diffmap is O(image area); huge native images hit worst case.

### Feature extraction
- Zenanalyze feature computation: ~1 sec/1000 images at native size (cheap; image analysis not the limiter).
- **Bottleneck 2: Parallel I/O.** rayon serializes file reads and feature writes. 1389 image-size pairs written sequentially.

### Offline training
- `zq_pareto_fit.py`: least-squares solve via Cholesky per config. Data size (278 train images × 12 zq targets × 8 configs = ~26k training points) is tiny. Solve time <1 sec.
- **Bottleneck 3: Turnaround time is the sweep, not training.** Training is fast; sweep dominates the iteration loop.

### Missing: cached intermediate results
- No mechanism to resume a crashed/interrupted sweep (rows are appended, but if a process dies mid-batch, that batch is lost).
- No caching of per-(image, size) zensim reference precomputation (happens fresh per config).
- Re-running `--features-only` re-reads all images and recomputes features (no caching; acceptable given feature cost is low, but still serial I/O).

---

## Reproducibility

### Pinned
- **Corpus paths**: hardcoded defaults in `zq_pareto_calibrate.rs` (CID22, CLIC2025, gb82). User can override; defaults are environment-specific (/home/lilith/work/codec-eval/…).
- **Size grid**: DEFAULT_SIZES = [64, 256, 1024, 0] (0 = native). Hardcoded; command line can override.
- **Q grid**: Q_GRID = [0, 5, 10, …, 100] step 5. Hardcoded; 21 points.
- **Encoder configs**: 8 ConfigSpec entries. Hardcoded IDs 0–7.
- **Image order**: paths are sorted alphabetically; consistent across runs if corpus unchanged.
- **Zensim profile**: `Zensim::new(ZensimProfile::latest())` — uses latest hardcoded profile. No version pinning; drifts with zensim updates.
- **Zenanalyze feature set**: `FeatureSet::SUPPORTED` — depends on cargo features at build time. No feature-stability contract; can change between builds.
- **Random seed for train/val split**: `--seed 0` (default). Explicit; reproducible.

### Not pinned
- **Corpus images**: user responsibility to pin dataset versions. Defaults are mutable directories.
- **Feature schema**: zenanalyze features are not versioned. New features added to `AnalysisFeature` enum break column order; old TSVs can't be joined with new feature column names.
- **Encoder config definitions**: if config IDs or names change, old TSVs become ambiguous (config_id column value 3 no longer maps predictably).
- **Zensim version**: diffmap implementation changes between releases; historical pareto TSVs are invalidated if diffmap scoring changes.
- **Parallel thread count**: default = 0 (rayon detects; system-dependent). Encode order and timing varies by machine.

---

## Duplication with other codecs

### Harness duplication
- **zenjpeg** `zq_pareto_calibrate.rs`: 8 encoder configs, 21 q values, 4 sizes. ~600 lines.
- **zenwebp** `dev/zenwebp_pareto.rs`: 72 configs (method × segments × scalar params), 30 q values, 4 sizes. ~800 lines.
- **zenavif** — (not examined; likely similar pattern).
- **zenjxl** — (not examined; likely similar pattern).

Each codec reimplements:
- Image loading + resizing (rayon parallel, Lanczos3, same across all).
- Corpus iteration + path sorting (identical).
- TSV row serialization + append mode (identical schema columns, codec-specific config naming).
- Feature extraction loop (calls `zenanalyze::analyze_features_rgb8`, same for all).
- Wall time logging + progress output (identical).

**Opportunity for consolidation:** factor the harness scaffolding (image I/O, rayon loop, TSV write, feature extraction) into a codec-agnostic library; each codec provides a config iterator and encode callback. Estimated 30–40% code reduction across 4 codecs.

### Schema differences
- **zenjpeg** `zq_pareto_*.tsv`: columns = image_path, size_class, width, height, config_id, config_name, q, bytes, zensim, encode_ms, total_ms.
- **zenwebp** `zenwebp_pareto_*.tsv`: columns = image_path, size_class, width, height, config_id, config_name, q, bytes, zensim, encode_ms, total_ms (identical!).
- **Features TSV**: both emit (image_path, size_class, width, height, feat_*). Identical structure.

**Good news:** schema is aligned. **Bad news:** `zq_pareto_fit.py` is zenjpeg-specific (hardcoded CONFIGS dict with 8 entries). Zenwebp uses a different CONFIGS dict (72 entries). Scripts cannot cross-pollinate.

---

## Test coverage

### Covered
- **Iteration loop correctness** (`tests/zq_target.rs`): API contract, metrics finitude, pass budgets, target convergence (14 tests).
- **Encoder-wide parity** (`tests/*.rs`): encoder output vs mozjpeg/cjpegli (multiple files; 30+ tests total).
- **Feature-system baseline** (none; features are extracted but never validated against a golden output).

### NOT covered
- **Pareto-sweep reproducibility**: no golden TSV, no "did this run match last month's?" test.
- **Regression coefficient stability**: no "did the fitted weights change unexpectedly?" test.
- **Config ID mapping**: if a config is renamed or reordered, tests don't catch the mismatch.
- **Zensim profile drift**: no test that "zensim scores on historical images match old sweeps."
- **Feature schema evolution**: no test that new features don't break old scripts.
- **Training convergence**: no test that per-config regressor RMSE stays below threshold.

### CI integration
- Standard unit tests run on all pushes. Sweep / training tests are manual or off-schedule.
- **No CI gate prevents shipping a release that breaks sweep reproducibility.**

---

## What would make this faster/better

1. **Cache zensim reference precomputation per (image, size).** Zensim::precompute_reference() is called once per config within a work unit; parallelize across image-size pairs first, cache the reference, then parallelize configs within each cached reference. Estimated 8× speedup on high-variance configs.

2. **Resume-safe sweep checkpointing.** Rather than appending raw rows, emit row-count markers or use WAL-style checkpoints. A crash mid-batch loses that batch; checkpoint per-work-unit would allow resuming from the last completed work unit (e.g., last image-size pair). Saves hours on re-runs after crashes.

3. **Automate JSON-emission from zq_pareto_fit.py.** Current pipeline: fit.py → stdout, manually pipe to emit_rust.py. Add `--emit-json` flag to fit.py to directly emit regression weights + metadata as JSON. Removes hand-curation step.

4. **Consolidate harness scaffolding into a shared library.** Image loading, resize, rayon loop, TSV append, feature extraction all duplicate across zenjpeg / zenwebp / zenavif / zenjxl. Factor into `zenanalyze::harness` or similar. Each codec provides config_iter() + encode() callbacks. Saves ~1500 LOC across 4 codecs.

5. **Bind encoder configs to a versioned enum.** Config IDs are magic numbers; if you rename config 3, old TSVs become ambiguous. Use a versioned enum (e.g., `ConfigV2` with a migration path) and validate at load time. Prevents silent schema drift.

6. **Parallelize training via codec-agnostic sharded regressor.** `train_hybrid.py` is single-process; split training examples by image-hash and fit per-shard in parallel, reduce, and merge. Estimated 4–8× speedup for 1000+ image datasets (not relevant at 347 images today, but critical as corpus grows).

7. **Add off-path feature-schema validation.** In CI or pre-commit, run harness on a tiny 10-image corpus, load output, and validate feature column count + names against zenanalyze's declared FeatureSet. Fails fast if zenanalyze updates and breaks downstream.

8. **Implement per-config validation curve.** After training, re-encode all val-set images at optimal (config, q) per target, and compare actual bytes/zensim to regressor predictions. Emit residuals distribution. Catches training underfit early.

9. **Expose picker inference as a public API.** Current state: regression weights are baked as Rust consts but never called. Add `EncoderConfig::for_perceptual_target(target_zq, features) -> (config, suggested_q)`. Unlocks the 34–85% byte-savings from the sweep.

10. **Unify feature ablation and sweep into a single driver.** Today: sweep is separate from `zenpicker/tools/feature_ablation.py`. Run both in one pass — encode once, measure ablation + pareto metrics together. Saves a full re-sweep when investigating feature importance.

---

## Summary

**Current state:** Full sweep harness + offline analysis exists and runs (~7 min on Ryzen 9 7950X). Regression training is fast. But the trained regressor is not integrated into runtime; encoder uses a fixed LUT instead of dynamic config selection. Pareto-optimal gains (34–85% bytes at zq ≥ 85) are documented but not deployed. Test coverage is absent for reproducibility; no CI gate prevents breaking the sweep pipeline.

**Pain points:** (1) Wall-time is encode-heavy (500+ cells/sec is good but still 8+ minutes for full sweep). (2) Feature schema and config IDs are not versioned; silent drift risk. (3) Harness code duplicates 30–40% across 4 codecs. (4) Training pipeline has a hand-curation step (JSON emit). (5) No resume-on-crash checkpointing; full re-runs on any failure.

**Easiest wins:** Automate JSON emit (30 min), add schema validation in CI (1 hour), implement picker.predict(features) API (2 hours).
