# Changelog

## [Unreleased]

### Added

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

- **Multi-codec shared-trunk picker bake (ZNPR v3.2).** New
  `BakeRequest.multi_codec_schema: Option<MultiCodecSchemaInput>`
  field; when `Some(_)`, the composer emits the v3.2
  `multi_codec_schema` section that
  `zenpredict::Predictor::predict_multi_codec` reads at runtime.
  New public types: `MultiCodecSchemaInput`, `PerCodecMapInput`.
  New `BakeRequestBuilder::multi_codec_schema()` setter. New
  `BakeError::MultiCodecSlotOutOfRange` /
  `MultiCodecOutputRangeInvalid` / `MultiCodecEmpty` /
  `MultiCodecInputDimMismatch` variants. The composer
  cross-validates against `layers[0].in_dim` so a mis-shaped bake
  fails at bake time, not load time. The Python emitter
  (`zentrain/tools/bake_picker.py`) is not yet wired — follow-up
  to flip joint-trained checkpoints from the per-codec distill
  path to the joint-picker bake.

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
