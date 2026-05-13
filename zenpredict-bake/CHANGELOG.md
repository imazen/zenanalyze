# Changelog

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
