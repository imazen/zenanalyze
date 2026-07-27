# Changelog

## [Unreleased]

### Fixed

- The crate is now CI-gated (`.github/workflows/ci.yml` job `zenpredict-bake`):
  fmt + clippy `--all-features --all-targets -D warnings` + test (default +
  `fit-yj`). Previously CI never built or tested it, so the items below had
  rotted unnoticed.
- `fit_yeo_johnson` golden-section test corrected. It asserted the Box-Cox fact
  "log-normal ⇒ λ≈0", but Yeo-Johnson λ=0 is `log(x+1)`, not `log(x)`, so the
  correct YJ MLE for log-normal data is ≈−0.843 (the profile log-likelihood is
  sharply peaked there: `ll(−0.843)≈−36.3` vs `ll(0)≈−144.7`). The fitter was
  always correct; the test now asserts ≈−0.843 and is renamed
  `golden_search_finds_yj_optimum_on_lognormal`.
- `predict` bench updated for `rand 0.9` (`gen_range` → `random_range`); a batch
  of accumulated clippy debt cleared so `-D warnings` passes.

### Added

- **`append_metadata_utf8` — metadata splice with score/byte identity
  guarantees** (`src/append.rs`, re-exported at crate root). Appends (or
  replaces in place, if the key exists) one UTF-8 metadata entry on an
  already-serialized ZNPR v3 bake via a section-level splice: only the metadata
  blob is re-serialized; every other section (weights / I8 scales / biases /
  scaler / bounds / output_specs / discrete_sets / sparse_overrides /
  feature_order / output_order) is copied byte-verbatim — no re-quantization,
  no HU re-reorder — with subsequent offsets shifted under the composer's own
  alignment rules. For composer-produced inputs the output is byte-identical
  to re-baking with the extended metadata list (locked by a 12-case
  dtype × compression × permutation test matrix). Compressed bakes stay
  compressed (LZ4 decompress → splice → recompress, `decompressed_payload_len`
  updated); v1/v2 bakes return a clean `AppendError::UnsupportedVersion`; the
  output is self-checked through `Model::from_bytes` before returning. Built
  for zensim's trainer to stamp `zentrain.repro` on finished bakes where
  `repack`'s dequantize→requantize round-trip (score-neutral, not byte-exact)
  is not acceptable. `lz4_flex` dep gains the `safe-decode` feature.
- **`zenpredict inspect` surfaces the embedded knob-veto safety rules** — when a
  bake carries a `zenpicker.knob_vetoes` blob, the inspect JSON now includes a
  `knob_vetoes` array (`feat_idx`, `op`, `threshold`, `cells`) parsed through the
  real runtime (`Model::knob_vetoes`), so the deploy-side safety bounds are
  human-inspectable. Absent on bakes without vetoes.
- **`zenpredict repack` writes the reuse-key stamps** — new
  `--analyzer-version <ver>` / `--feature-defs-version <u32>` / `--config-hash <u64>`
  flags inject (override semantics) the three `zenanalyze-api` metadata keys into
  an existing `.bin`, preserving all other content. This is the **codec re-bake
  path**: stamp a pre-contract picker model so it can reuse a shared `Offer`,
  without re-training.
- **`BakeRequestJson.analyzer_version` + `feature_defs_version` +
  `feature_config_hash`** — optional first-class stamps for the `zenanalyze-api`
  three-part offer/reuse key. When set, the baker writes them to
  `keys::ANALYZER_VERSION` (UTF-8), `keys::FEATURE_DEFS_VERSION` (4-byte LE
  `u32`), and `keys::FEATURE_CONFIG_HASH` (8-byte LE `u64`) metadata, read back
  via `Model::analyzer_version()` / `feature_defs_version()` /
  `feature_config_hash()`. `config_hash` is the value-affecting analysis-config
  digest (`AnalysisQuery::config_hash()`, `0` = gamma default) that keeps a
  linear-light model from reusing gamma features. Preferred over hand-rolled
  `metadata` entries — a `u32`/`u64` can't ride the `f32` repr, so manual
  encoding would mean emitting LE hex by hand; an explicit `metadata` entry for
  the same key still takes precedence (no duplicate key written). All default to
  `None`. Bakers (zentrain) pass the values through from extraction.

- **`BakeRequestJson` is now `#[non_exhaustive]`** so future
  `#[serde(default)]` field additions are non-breaking on the Rust
  side. Direct struct-literal construction outside the crate is no
  longer supported; callers must go through `serde_json::from_str` /
  `serde_json::from_slice` / `bake_from_json_str`. In-tree consumers
  (the CLI binaries, integration tests, the unified `zenpredict bake`
  subcommand) only construct via deserialization and are unaffected.

- **Three optional bake-time knobs on `BakeRequestJson` + `bake_from_json`**:
  `zerobias_tau` (f32, default 0.0), `compressed` (bool, default false),
  `optimize` (bool, default false). All three honor the same semantics
  as the corresponding Rust `BakeRequest` fields / `bake_optimized`
  entry point. When `zerobias_tau > 0.0`, weights are zeroed in-place
  via `apply_zero_bias_per_layer_in_place` BEFORE the layer's declared
  `dtype` quantization runs; `compressed: true` sets `BakeRequest.compressed`;
  `optimize: true` routes through `bake_optimized` (permutation +
  hillclimb search). All three default to off, so existing JSON callers
  see no behavior change. Lets Python wrappers (`zenanalyze/tools/bake_picker.py`,
  `zensim/scripts/v_next/bake_to_znpr.py`, `zensim/scripts/v_next/v0_20b/bake_znpr_v3.py`)
  produce compressed bakes by adding three keys to their JSON dict —
  no shell-out to `zenpredict repack` required. Calibrated `zerobias_tau`
  = 0.005 per `zensim/benchmarks/zenpredict_rle_zerobias_eval_2026-05-13.md`
  (87.5 % i8 zero density, -0.0001 SROCC on V0_18). README updated with
  the new keys + a calibration pointer.

- **Bake-side validation for the 5 new stacked `FeatureTransform`
  variants** (added in `zenpredict` 0.2.1). The composer now parses
  the optional `zentrain.feature_transforms` /
  `zentrain.feature_transform_params` metadata pair and rejects
  before write:
  - Unknown transform tokens (`UnknownFeatureTransformToken`).
  - Per-variant param arity mismatches
    (`FeatureTransformParamArityMismatch`): 1 for `ClipThenLog1p`,
    2 for `WinsorP99` / `WinsorThenLog` / `WinsorThenLog1p` /
    `WinsorThenSignedCbrt` / `SignedCbrtThenWinsor`, 3 for
    `ClipThenLog1pThenWinsor`.
  - Per-variant domain violations (`FeatureTransformParamInvalid`):
    `WinsorThenLog` with `p1 ≤ 0`, `WinsorThenLog1p` with `p1 ≤ -1`,
    inverted bounds, negative `ε` for `ClipThenLog1pThenWinsor`, etc.
  Catches misspelled tokens and out-of-domain bounds that would
  produce `NaN` / `-Inf` at runtime. The runtime parser still
  validates as a second line of defense for hand-constructed bytes.
  Two existing tests (`unknown_token_rejected_at_load`,
  `length_mismatch_rejected_at_load`) renamed to `*_at_bake` to
  reflect the new gate's earlier reject point.

- New unified `zenpredict` CLI binary with three subcommands:
  - `zenpredict bake <input.json> <output.bin>` — delegates to the
    same code path as the legacy `zenpredict-bake` binary.
  - `zenpredict inspect <bake.bin> [--weights]` — delegates to the
    same code path as the legacy `zenpredict-inspect` binary.
  - `zenpredict repack <in.bin> <out.bin> [--dtype f32|f16|i8]
    [--zerobias <tau>] [--compress] [--optimize]` — front-end for
    the logic previously only exposed as the `rebake_v3_1` example.
    Useful for re-quantizing an existing F32 bake to I8 + LZ4 +
    zero-bias for size-sensitive deployments (e.g. 200 KB → 14 KB
    on the V_22-IW v2 bake at CID22 SROCC delta < 0.001).
- `zenpredict_bake::cli` module exposing the three subcommand
  bodies as `pub fn run_{bake,inspect,repack}_cli(argv: &[String]) ->
  ExitCode`, shared between the unified `zenpredict` binary and the
  legacy single-purpose binaries.

The legacy `zenpredict-bake` and `zenpredict-inspect` binaries
remain present and produce byte-for-byte identical stdout/stderr
output, exit codes, and arg semantics.

### Reverted (pre-publish, 2026-05-17)

- **Multi-codec shared-trunk picker bake (ZNPR v3.2).** Briefly
  added earlier in the unreleased window (`BakeRequest.multi_codec_schema`,
  `MultiCodecSchemaInput`, `PerCodecMapInput`, `BakeRequestBuilder::multi_codec_schema`,
  `MultiCodec*` `BakeError` variants, `emit_multi_codec_section`).
  Reverted before publish because the joint trainer's distillation
  step regressed on zenjpeg by −7pp argmin, and no shipped codec
  ever consumed the runtime path. Per-codec bins (which all live
  consumers use) are unaffected. The transfer-learning value the
  joint-trunk design was reaching for is recoverable as a training-
  time trick (pretrain shared trunk on combined data → fine-tune
  per-codec head → bake each as a normal v3 bin); see the
  ecosystem review at `docs/ecosystem_cleanliness_review_2026-05-17.md`.

## [0.1.0] - 2026-05-13

Initial release. Extracted from `zenpredict 0.1.x`'s `bake` feature.

### Why this crate exists

`zenpredict` is the **runtime** (parse + predict, no allocations on
the hot path). The bake side pulled in `serde` + `serde_json` + a
hand-rolled JSON visitor that, together, were ~30-40 % of the runtime
crate's monomorphization budget — paid for by every codec consumer
including ones that never called the baker.

`zenpredict-bake` carries those parts as a separate crate. Codec
runtimes depend only on `zenpredict`; trainers + tooling depend on
`zenpredict-bake`. Codec build times measurably drop.

### Surface

- `bake(req: &BakeRequest) -> Result<Vec<u8>, BakeError>` — the
  ZNPR v3 byte-stream composer (formerly `zenpredict::bake::bake_v2`).
- `BakeRequest { schema_hash, flags, scaler_mean, scaler_scale,
  layers, feature_bounds, metadata, output_specs, discrete_sets,
  sparse_overrides }` with a fluent `BakeRequest::builder(...)` for
  the common "required fields, empty optional sections" pattern.
- `bake_from_json(req) -> Result<Vec<u8>, BakeJsonError>` and
  `bake_from_json_str(s)` — the JSON wire schema the zentrain Python
  pipeline emits.
- CLI binaries `zenpredict-bake` (JSON → `.bin`) and `zenpredict-inspect`
  (`.bin` → JSON dump). The Python trainer at
  `zentrain/tools/bake_picker.py` shells out to `zenpredict-bake`.

### Format

ZNPR v3 — the only format `zenpredict 0.2.0` parses. Migration tool
for older bakes lives at `zentrain/tools/migrate_znpr_v2_to_v3.py`.

### Tests + benches

- 102 integration tests (`tests/bake_roundtrip.rs`, `lifecycle.rs`,
  `output_specs.rs`, `safety_summary.rs`, `feature_transform.rs`,
  `scorer.rs`, `json_bake.rs`) cover the bake → load → predict
  lifecycle in both default-and-advanced zenpredict feature
  configurations.
- `benches/predict.rs` runs `Predictor::predict` against two
  production shapes: V0_18-zensim 228→384→1 I8 and zenwebp picker
  51→64→24 F16.
