# zenpicker + zenpredict + ZNPR v2 audit, 2026-05-02

## Inventory

**zenpredict** (`~/zenpredict/`, 5094 LOC Rust):
- `src/lib.rs` (166) — public API surface; exports `Predictor`, `Model`, `AllowedMask`, `ArgminOffsets`, `ScoreTransform`, `FeatureBound`, `RescuePolicy`
- `src/model.rs` (712) — ZNPR v2 parser; fixed `#[repr(C)]` 128-byte header + offset-table sections + TLV metadata blob
- `src/predictor.rs` (325) — scratch-owning forward-pass wrapper; allocation happens once in `new()`, zero-copy per `predict()`
- `src/inference.rs` (352) — layer-by-layer matmul + activation; mixed-dtype (f32/f16/i8)
- `src/argmin.rs` (526) — masked argmin, top-K, score transforms, reach-rate gate, confidence gap
- `src/rescue.rs` (191) — encode-verify-rescue policy types (`RescueDecision`, `should_rescue`)
- `src/bounds.rs` (118) — feature bounds, OOD detection
- `src/metadata.rs` (364) — typed-TLV blob parser (bytes/utf8/numeric)
- `src/bake/v2.rs` + `src/bake/json.rs` — ZNPR v2 byte composer + JSON-driven baker
- `src/bin/zenpredict_bake.rs` — CLI: JSON → .bin with exit codes, summary stats
- `tests/lifecycle.rs`, `tests/json_bake.rs` — round-trip bake → load → infer tests

**zenpicker** (`~/zenpicker/`, 372 LOC Rust):
- `src/lib.rs` — `CodecFamily` enum (jpeg/webp/jxl/avif/png/gif), `AllowedFamilies` mask, `MetaPicker` wrapper
- Wraps a `zenpredict::Predictor`; output dimension = 6 families
- Family-order contract via `zenpicker.family_order` metadata key (UTF-8 CSV)
- Load-time validation: `MetaPicker::validate_family_order()` reads metadata and hard-fails on mismatch

**Format**: ZNPR v2 — 128-byte header (magic `ZNPR`, version 2, schema_hash u64) + layer-table (48 bytes × n_layers) + aligned data sections (scaler, weights, scales, biases, feature_bounds, metadata TLV).

---

## Data flow (training output → ZNPR v2 → runtime inference)

```
zentrain training sweep TSV (image, config, q, bytes, zensim)
           ↓
  zentrain Python pipeline (pareto, teacher fit, distill)
           ↓
  sklearn MLPRegressor JSON dump
           ↓
  bake_picker.py + zenpredict-bake CLI
           ↓
  ZNPR v2 .bin (16-aligned, zero-copy)
           ↓
  Codec loads: Model::from_bytes_with_schema(..., MY_SCHEMA_HASH)
  Predictor::new(model) — allocates scratch [need, need, n_out]
           ↓
  Per encode: extract_features(...) → zenanalyze &[f32]
  argmin_masked(features, allowed_mask, ScoreTransform, offsets)
           ↓
  Chosen config (cell index or scalar pair)
```

**Metadata carriage**:
- `zentrain.*` keys: profile, calibration_metrics, reach_rates, reach_zq_targets
- `<codec>.*` keys: codec-private cell_config, format_overhead
- `zenpicker.family_order`: family enum order (load-time assertion)

---

## Performance / inference cost

**Allocation profile**:
- `Predictor::new(model)`: one-time allocations:
  - `scratch_a`, `scratch_b`: `max(n_inputs, max_layer_out_dim) × sizeof(f32)` each (e.g., 4 KB for 64-input/1024-hidden model)
  - `output`: `n_outputs × sizeof(f32)` (e.g., 48 bytes for 12 outputs)
- Total for a typical picker: ~8 KB per instance. **Recommended: reuse a single `Predictor` across many encode requests.**

**Per-call cost**:
- `predict(features)`: layer-by-layer matmul + bias + activation, **zero-allocation hot path**
  - Forward pass: typical picker = 2–3 layers, fast enough for per-frame encoder decision ✓
- `argmin_masked(...)`: O(n_outputs) scan + masked comparison, no allocations
- Scratch buffers reused across calls without zeroing — **deterministic** (every layer writes bias before accumulation)

**Weight storage**:
- F32: 1× size (full precision, ~default)
- F16: 0.5× size (built-in conversion, no `half` dep) — typical use
- I8: 0.25× size (per-output column scaling) — dense models

**Scratch sizing**: Model exposes `scratch_len()` for pre-allocation estimates.

---

## Reproducibility (bake determinism, format stability)

**Bake determinism**: `bake_v2(BakeRequest)` is **deterministic** — same inputs → same byte-for-byte output (tested in `tests/json_bake.rs`). Schema hash (Blake2b-64 of feature columns + version tag) is stable; changing feature order or names invalidates old bakes (expected, design intent).

**Format stability**:
- ZNPR v2: fixed 128-byte `#[repr(C)]` header + offset-table LayerEntry[n_layers] (48 bytes each) — **fully specified in rust source** (`src/model.rs` documents byte offsets and layout)
- v1 (32-byte positional layout) **not supported** by this crate; older bakes must be rebaked through v2 baker
- **No version negotiation**: loader enforces `version == 2` at parse time; unknown version → parse error

**Format extensibility** (stability under evolution):
- Reserved fields: Header has `[u32; 14]` reserved slots; LayerEntry has `[u32; 3]` reserved + `flags: u16` for future activation/dtype kinds
- Metadata TLV: opaque key-value store — new keys don't break old loaders (unknown keys silently ignored if not required)
- Feature bounds: optional Section (len=0 when absent) — safe to add OOD guards without invalidating old bakes
- **Adding a new `CodecFamily`**: schema_hash mismatch → hard fail (design choice: family order is a bake-time contract, not negotiated at runtime)

**Cross-codec consistency**:
- All pickers (zenjpeg, zenwebp, zenavif, zenjxl) share the same ZNPR v2 format and `zenpredict::Predictor` runtime
- Feature input schema (feat_cols) differs per codec; bake's `schema_hash` captures it
- Per-codec `.bin` files are independent (one per codec family)
- Meta-picker adds a 6th dimension output (family enum) but uses identical machinery

---

## Test coverage

**Lifecycle tests** (`tests/lifecycle.rs`):
- Bake codec picker (5 inputs, 12 outputs, 2 layers, F16 + F32 mixed) → load → predict → argmin over masked outputs
- Covers feature bounds, metadata, rescue policy, reach-rate gate, confidence gap
- **Round-trip**: bake → load-with-schema → infer determinism **verified**

**JSON bake tests** (`tests/json_bake.rs`):
- Minimal JSON (2-input identity) → round-trip load-predict
- Full JSON (3-input, 4-layer, mixed dtypes, metadata, feature bounds) → load-predict-metadata-readback
- **Format stability**: JSON schema emitted by `bake_picker.py` loads correctly through v2 baker

**Internal sanity tests** (`src/tests.rs`, 1861 LOC):
- Argmin logic (identity, masking, per-output offsets, top-K)
- Threshold mask (reach-rate gate with NaN/Inf handling)
- Feature bounds OOD detection
- Header + Layer parsing edge cases (misaligned offsets, truncated files, oversized sections)
- Mixed-dtype inference (f32 → f16 → i8 layer chains)
- Metadata TLV parsing (unknown keys, type mismatch, truncation)

**NOT covered**:
- Integration between Rust-side `bake_v2` and Python-side `bake_picker.py` except via JSON test fixtures
- Determinism of `bake_picker.py` itself (scikit-learn + numpy source, not in scope)
- **Feature extraction** (zenanalyze) — separate crate
- Codec-specific feature validation (e.g., "does zenjpeg ever produce an OOD feature value?") — codec responsibility

---

## Pain points

### 1. **No bake determinism proof for `zentrain` pipeline end-to-end**

Python training (`zentrain/tools/train_hybrid.py`) produces sklearn JSON, which `bake_picker.py` converts to JSON and spawns Rust CLI. Rust side is deterministic; Python side (seed, dtype in sklearn's weight storage, numpy random) may introduce non-determinism. **Risk**: same training run re-run → same bytes? Untested. Codecs ship bakes but can't audit them for bit-for-bit reproduction without running the full Python pipeline themselves.

**Symptom**: Benchmarks consume TSVs and logs (reproducible from source); baked `.bin` files have no origin attestation.

### 2. **Format brittleness on `CodecFamily` enum evolution**

Adding a new family requires:
- Update `CodecFamily` enum in Rust (`zenpicker/src/lib.rs`)
- Re-bake the **entire meta-picker** model (schema_hash changes; old bakes refuse to load)
- Redeploy all codecs in lockstep with the new family option

If a codec ships with a stale meta-picker binary built against the old enum, `MetaPicker::validate_family_order()` hard-fails at startup. No graceful degradation; migration requires a coordinated push.

**Symptom**: Cross-codec releases tightly coupled via the meta-picker's schema_hash.

### 3. **No runtime audit trail for picker decisions**

`Predictor::argmin_masked(...)` returns `Option<usize>` (the chosen index); it doesn't expose:
- The per-output score vector (what did the model predict for all families?)
- Confidence gap (why was this family chosen over the runner-up?)
- Which mask bits were actually enabled (constraint filtering is silent)

Codecs can call `predict()` explicitly to read raw outputs, but that's an extra allocation-free call + manual argmin logic duplication.

**Symptom**: Training ablation studies (LOO, feature importance) can introspect full outputs; production codecs can't. Monitoring can't distinguish "this family won by a narrow margin" from "clear winner" without re-implementing argmin.

---

## Missing tooling

### 1. **Bake diff tool** (`zenpredict-diff`)

Compare two `.bin` files and report:
- Header differences (n_inputs, n_outputs, n_layers, schema_hash, flags)
- Layer-by-layer weight histogram / stats (does F16 vs F32 matter?)
- Metadata TLV differences (did calibration metrics change?)
- Byte-for-byte diff of weights (rule out retraining noise)

**Use case**: Codecs need to validate that "same training script" produces "same bytes" before shipping a production bake. Current workaround: manual hex inspection or Python deserialization (defeats zero-copy promise).

### 2. **Feature bounds visualizer** (`zenpredict-inspect --bounds`)

Plot feature bounds against a labeled training corpus:
- Per-feature: histogram of training data vs OOD envelope
- Sanity check: are OOD thresholds sensible given observed ranges?
- Cross-feature: are bounds over-tight (training data never used the full range)?

**Use case**: `FOR_NEW_CODECS.md` recommends feature bounds as part of the safety plane. No tool to validate bounds are correctly set.

### 3. **Schema hash audit trail** (metadata field + CLI)

Codecs embed `MY_SCHEMA_HASH` at compile time. If a trainer later adds/removes/reorders a feature, the bake's embedded hash changes but the codec binary doesn't. The mismatch is caught at runtime (good), but there's no pre-release way to audit "will this codec's compile-time hash match the production bake?"

**Missing**:
- A `SCHEMA_VERSION_TAG` constant exported by `zenanalyze` (or zentrain) that codecs can `assert_eq!` at compile time
- A `zenpredict-inspect <.bin>` subcommand to dump the schema_hash + human-readable feature list
- Optional: a `zenanalyze-audit <Cargo.toml>` that extracts the codec's compile-time hash and compares against a bake

---

## Duplication with other codecs

All four shipping codecs (zenjpeg, zenwebp, zenavif, zenjxl) have **identical** ZNPR v2 consumer patterns:
```rust
#[repr(C, align(16))]
struct Aligned<const N: usize>([u8; N]);
const PICKER_BIN: &[u8] = &Aligned(*include_bytes!("picker_v2.N.bin")).0;

let model = Model::from_bytes_with_schema(PICKER_BIN, MY_SCHEMA_HASH)?;
let mut predictor = Predictor::new(model);
…
let mask = AllowedMask::new(&allowed_cells(&caller_constraints));
let pick = predictor.argmin_masked(&features, &mask, ScoreTransform, offsets)?;
```

**Duplication**:
- Each codec repeats the alignment boilerplate + schema_hash assertion
- Each codec implements its own feature extraction (should be zenanalyze, but codecs have old custom passes)
- Each codec copies the metadata TLV reading code for its own `cell_config` blob

**Missing abstraction**: A shared `codec::PickerRuntime<'a>` that owns the loaded model, feature extraction, and argmin dispatch would eliminate 90% of duplication.

---

## What would make this faster/better

### Top 3 quick wins:

1. **Bake diff tool (`zenpredict-diff a.bin b.bin`)** — 1–2 days
   - Compare headers, layer-by-layer weight statistics, metadata TLV
   - Codecs can validate training script changes in CI before shipping
   - No additional runtime overhead; read-only tool

2. **Extend `zenpredict-inspect` to expose raw prediction outputs + confidence gap** — 1 day
   - Add `--raw-scores` flag to dump model outputs for a feature vector
   - Helps training pipeline audit decisions; helps codecs monitor picker behavior
   - Zero API surface change; opt-in diagnostic command

3. **Cross-codec feature audit script** — 2–3 days (Python)
   - Load all four codecs' `.bin` files; extract their feature schemas (from metadata)
   - Report intersection (common features), codec-specific features, schema_hash collisions
   - Detects silent mismatches in feature extraction between codecs early
   - Runs in CI; blocks releases if schemas don't align as expected

### Structural improvements (longer-term):

- **`CodecFamily` versioning**: Support reading old family enums (e.g., deserialize a v1 family-order, map to v2) instead of hard-fail on mismatch. Requires schema versioning separate from family order.
- **Feature schema embedding**: Store canonical `feat_cols` list in the `.bin` metadata, not just the schema_hash. Codecs can validate at load time without re-implementing feat extraction.
- **Predictor diagnostics mode**: Optional `Predictor::predict_with_tracing()` that returns `(scores, confidence_gap, oob_mask)` without extra allocations (scratch buffers reusable).

