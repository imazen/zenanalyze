# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major
     (or minor for 0.x) release. Add items here as you discover them.
     Do NOT ship these piecemeal — batch them. -->

(none queued)

### Added

- `WeightDtype::I8` weight-storage variant (per-output-neuron 8-bit
  quantization with f32 scale-per-column). 26 % of f32 size, 52 %
  of f16 size — and on the v2.1 corpus a slight *win* on mean
  overhead (−0.11 pp) and argmin accuracy (+1.0 pp). Bake via
  `bake_picker.py --dtype i8`. Binary format extension is purely
  additive: parser rejects unknown dtype bytes with
  `UnknownWeightDtype`, existing F32/F16 bakes parse unchanged.

### Fixed

- `tools/bake_picker.py` docstring no longer claims the runtime
  *multiplies* by `scaler_scale` — that was the pre-0.1 convention
  and was fixed to divide before the first publish.

## [0.1.0] - 2026-04-30

First public release. Codec-agnostic picker runtime, baked-MLP
inference, safety machinery, and a canonical training pipeline.

### Added — runtime crate

- Core inference: `Picker::new` / `predict` / `argmin_masked` /
  `argmin_masked_in_range` (afb17e9, #14). Hybrid-heads layout —
  N categorical cells × K scalar prediction heads — with the
  bytes-log sub-range exposed for argmin (afb17e9).
- v1 binary format: `Model::from_bytes`, magic `ZNPK`, scalar
  fp32 / fp16 weight storage, manifest sidecar (afb17e9).
- `argmin_masked_top_k::<K>` + `argmin_masked_top_k_in_range::<K>`
  for cached second-best picks (612392b, #24).
- `rescue` module: `should_rescue`, `RescuePolicy`,
  `RescueStrategy`, `RescueDecision` — codec-agnostic decision
  logic for the two-shot pass framework (aefea47, #28).
- `reach_gate_mask` helper — runtime threshold parameter for the
  `zensim_strict` reach gate (8750ae6, #29).
- `Picker::pick_with_confidence` + `pick_with_confidence_in_range`
  return `(best, log_bytes_gap_to_top_2)` (b3f6253, #35).
- `FeatureBounds` struct + `first_out_of_distribution(features,
  bounds)` runtime OOD gate; codec ANDs the result into its
  constraint mask before argmin and falls through to
  `RescueStrategy::KnownGoodFallback` on OOD inputs (b3f6253, #35).

### Added — wire format / manifest

- `safety_report` block in the model JSON / manifest, shipped
  in every bake produced by current `train_hybrid.py`. Carries
  `passed`, `violations`, `thresholds`, and a `diagnostics` sub-block
  with: `argmin.{train,val}` metrics (mean / p50 / p90 / p99 / max),
  `by_zq` / `by_size` / `by_zq_size` overhead breakdowns,
  `worst_case` top-1 % rows, `per_cell` calibration deltas, `mlp`
  weight-health scan (dead neurons, layer max/median ratio, NaN/Inf
  checks), `feature_bounds` (per-feature `min/p01/p25/p50/p75/p99/max/
  mean/std`), `train_rows_by_size_zq` (e5fa489 #34, b3f6253 #35,
  635cdd0 #39).
- `feature_bounds_p01_p99` lifted to the manifest top level by
  `bake_picker.py` so codecs compile a `FEATURE_BOUNDS:
  &[FeatureBounds]` const (b3f6253, #35).
- `reach_safety` block (per-target_zq, per-cell reach rate +
  threshold-applied safe mask) for `zensim_strict` bakes
  (122bdfe, #27).
- `safety_profile` and `training_objective` manifest fields
  (122bdfe, #27).
- `categorical_axes` / `scalar_axes` / `scalar_sentinels` in the
  `hybrid_heads_manifest` (79277c6, #36).

### Added — training pipeline (`zenpicker/tools/`)

- `train_hybrid.py` — codec-agnostic hybrid-heads training
  (f76318e #23, 79277c6 #36 made axes configurable). Codec config
  module exports `PARETO`, `FEATURES`, `OUT_*`, `KEEP_FEATURES`,
  `ZQ_TARGETS`, `parse_config_name`, `CATEGORICAL_AXES`,
  `SCALAR_AXES`, optional `SCALAR_SENTINELS` /
  `SCALAR_DISPLAY_RANGES` / `SAFETY_THRESHOLDS`.
- `--objective {size_optimal, zensim_strict}` plus
  `--bytes-quantile`, `--reach-threshold` (122bdfe, #27).
- `--hidden W,W,W` student-MLP capacity sweep
  (b3f6253-related work).
- `--strict` / `--allow-unsafe` flags. Auto-strict when
  `CI` env is set (e5fa489, #34).
- `_picker_lib.py` — parallel teacher training (joblib),
  `HISTGB_FAST` / `HISTGB_FULL` presets, dataset cache, sklearn
  `HistGradientBoostingRegressor` is the default backend (we
  benchmarked lightgbm 4.6 against it and it's 2-16× *slower* on
  this many-small-fits workload).
- `feature_ablation.py --method permutation` (default) ranks
  features ~50× faster than retrain-LOO. `--strict` exits 1 when
  any feature shows a negative permutation Δ — overfit-on-noise
  signal (b3f6253, #35).
- `adversarial_probe.py` corner-case spot-check (zeros / huge /
  NaN / Inf / single-feature spike). Exits 1 in `--strict`
  (b3f6253, #35).
- `size_invariance_probe.py` resizes fixture images across all
  four `size_class` bins and asserts the picker's argmin stays
  stable. Counterpart to `train_hybrid.py`'s `PER_SIZE_TAIL` /
  `DATA_STARVED_SIZE` violations (635cdd0, #39).
- `diagnose_picker.py` — human-readable health report for any
  baked artifact, falls back to a static MLP weight scan on
  legacy bakes (b3f6253, #35).
- Canonical 6-stage end-to-end command sequence in
  `tools/README.md` (1454937 #38, 635cdd0 #39).

### Added — bake-time gate

- `bake_picker.py` reads `safety_report.passed` and refuses to
  bake when false unless `--allow-unsafe` is passed (e5fa489,
  #34). Defense in depth: `safety_report` is forwarded into
  `.manifest.json` so codec runtime can refuse to load if the
  bake-time `--allow-unsafe` was over-aggressive.

### Added — documentation

- Codec-side wiring tutorial: `FOR_NEW_CODECS.md` (#21, plus
  Step 1.5 added in #39 mandating size-class sweep coverage).
- `SAFETY_PLANE.md` — two-shot rescue protocol + complete API
  surface (b3f6253 #35, 635cdd0 #39 added "Size invariance is a
  safety property").
- `tools/README.md` — full pipeline + safety-gate contract +
  per-flag reference table (1454937 #38).
- Scaler-convention comment block at all four sites
  (`inference.rs`, `train_hybrid.py`, `train_distill.py`,
  `train_distill_reduced.py`) — clarifies that the bake stores
  sklearn's `scale_` (= std) and inference *multiplies* by it,
  so the picker scales by std rather than normalizing
  (1454937, #38). The MLP first layer absorbs the discrepancy
  at training time; do not "fix" the direction in isolation.

### Shipped models

- `models/zenjpeg_picker_v2.1_full.bin` — 195 KB f16, 35-feature
  schema, 192³ MLP, hybrid heads (12 cells × 3 scalar heads),
  full `safety_report` populated. Default for zenjpeg encoders.

### Known limitations

- `safety_report.passed = false` on the v2.1 zensim_strict
  variant (PER_ZQ_TAIL: zq=94 p99=85.4% > 80% threshold). The
  strict variant is intentionally not shipped pending feature
  pruning — see [`FOR_NEW_CODECS.md` Step 8](FOR_NEW_CODECS.md).
- `pixel_budget` and `hf_max_blocks` are `#[doc(hidden)]` in
  `zenanalyze`; v2.1 picker generalizes to 2 MP cleanly per the
  budget-research probe but we have no public knob today.
- i8 weight-storage variant landed in `[Unreleased]` (per-output
  scales, additive parser change). Re-baking shipped models with
  `--dtype i8` gives ~26 % the f32 size with no quality loss;
  re-bake step deferred to a follow-up so the codec adopters can
  decide whether to flip the on-disk format on their own
  release cadence.
- Generational re-encode picker (round-trip JPEG-source case) is
  on the v0.3 roadmap.
