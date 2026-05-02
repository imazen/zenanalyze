# zensim training/scoring audit, 2026-05-02

## Inventory

**Training Entrypoints:**
- `zensim-validate` (Rust CLI, 4,392 lines): canonical training harness. Invoked as `cargo run --release -p zensim-validate -- --dataset <path> --format tid2013 --train`.
- `tools/zensim-optimize.py`: Nelder-Mead/coordinate-descent optimization of encoder loop parameters (6 continuous params) via `cjxl-rs` + `djxl` + `ssimulacra2`.
- Training algorithms: coordinate descent (default), CMA-ES, RankNet SGD, proximal L1-regularized FISTA.
- Training objectives: SROCC (Spearman), KROCC (Kendall), or blended.

**Scorer Inference:**
- Rust crate `zensim` (0.2.7): `Zensim::compute(reference, distorted) -> f64`, `result.score()` mapped as `100 - 18 × d^0.7`.
- 228 features: 19/channel (X, Y, B) × 4 scales (1×, 2×, 4×, 8×). Multi-scale SSIM + edge artifacts + detail loss + MSE in XYB + high-frequency features.
- Precomputation API: `precompute_reference()` saves ~25% per comparison at 4K.
- Inference cost: ~22 ms at 1080p (multi-threaded, AVX2), ~89 ms single-threaded.

**Calibration & Validation:**
- Ground truth: TID2013 (3k pairs), KADID-10k (10.1k pairs), CID22 (4.3k pairs) — SROCC 0.8676 / 0.8192 / 0.8427 respectively (none used in training).
- Training corpus: 218k concordance-filtered synthetic pairs (6 codecs, q5-q100, filtered for human agreement).
- Two profiles: `PreviewV0_1` (344k pairs, 5-fold CV, SROCC 0.9936) and `PreviewV0_2` (218k filtered, Nelder-Mead, SROCC 0.9960). Latest = V0_2.
- Weights embedded in `/zensim/src/profile.rs` (static `WEIGHTS_PREVIEW_V0_1`, `WEIGHTS_PREVIEW_V0_2`).
- ZSFC binary feature cache (v3 uses f32 for 2× smaller files); invalidates on blur_passes/blur_radius/num_scales change.

**Versioning & Reproducibility:**
- Score mapping: `score() = 100 - 18 × raw_distance^0.7` (A=18, B=0.7 for V0_2).
- Cross-profile approximations: `approx_ssim2()`, `approx_dssim()`, `approx_butteraugli()`.
- CHANGELOG tracks breaking changes. Recent: cbrt_midp color swap (0.2.7), PrecomputedReference optimization (-65-70%), diffmap upsample fusion.
- **No per-version scorer comparison or drift detection tooling** found.

**Relationship to butteraugli / ssim2:**
- Same psychovisual foundations.
- `zensim-bench/benches/bench_compare.rs`: zensim ≈4× faster than C++ libjxl SSIMULACRA2, 18× faster at 4K.
- zensim-validate can train on GPU SSIMULACRA2 or Butteraugli ground truth (`--target-metric gpu-ssim2 | gpu-butteraugli`).
- Per-dataset weights via `--dataset-weights tid2013:1.0,kadid:1.5`.

**Duplication with codec-picker:**
- No direct code sharing observed. zensim is self-contained; codec-picker (jxl-encoder-rs) may call zensim as a quality scorer but doesn't share training pipelines.

**Test Coverage:**
- Golden-output regression via `zensim-regress` (0.4.0): byte-level checksums + perceptual diffs on mismatch.
- Unit tests: cross-platform, cross-tier (basic/peaks/extended features), ICC profile coverage, classification (error categorization).
- CI: fmt, clippy, multi-OS regression, feature permutation matrix.
- `--cross-validate 5` for k-fold CV. `--leave-one-out` for dataset hold-out.

**Docs/Scripts/Runbooks:**
- README covers speed (Table: 14ms@720p, 22ms@1080p, 91ms@4K), SROCC correlations, quick-start Rust API.
- **No training runbook.** Validation instructions at bottom of README; reproduction via `cargo run --release -p zensim-validate -- --dataset ... --format <type>`.
- `scripts/curate-repro-corpus.py` stages ICC-related images for S3.

## Top 3 Pain Points

1. **Training complexity & no reproducible walkthrough**. `zensim-validate` is 4,392 lines with 12+ algorithm choices (coord/cmaes/pairwise/proximal, sparse tuning, feature tiers). No tutorial for "train a new version." Switching between `--target-metric gpu-ssim2` vs `gpu-butteraugli` is implicit. Feature caching is opaque (ZSFC binary, version/params validation buried in code).

2. **Multi-scale convolution cost not profiled**. README claims 22ms@1080p but no per-scale breakdown. XYB conversion + 4-scale pyramid + 228 features (SIMD via archmage) likely dominates; no visibility into which scale/feature tier is hottest. Single-threaded bottleneck unknown.

3. **Zero tooling for corpus/model drift**. No per-image score visualization, cross-scorer comparison (zensim vs ssim2 per image), or "which train corpus change moved scores?" tracker. Validation is SROCC/KROCC aggregate only. Cannot see if 0.99 → 0.98 SROCC came from outlier overfitting or systematic miscalibration.

## Top 3 Missing-Tooling Suggestions

1. **Training runbook + profile manager**. Create `tools/train-profile.sh`: (a) downloads TID2013/KADID/CID22 from canonical sources, (b) runs zensim-validate with sensible defaults, (c) commits new weights to `profile.rs`, (d) generates summary report (SROCC before/after, train-time, feature tier used).

2. **Per-image score dashboard**. Add export `--dump-scores scores.json` with per-pair (reference, distorted, human_score, zensim_score, error_category). Viz script: scatter zensim vs human, highlight outliers, cross-reference corpus.

3. **Metric-comparison corpus tool**. `tools/compare-metrics.py`: given a dataset, compute zensim + butteraugli + ssim2 per-image. SROCC for each, correlation matrix. Detect (a) which metric agrees most with humans on which distortion type, (b) score-remapping accuracy (do approx_butteraugli estimates hold?).
