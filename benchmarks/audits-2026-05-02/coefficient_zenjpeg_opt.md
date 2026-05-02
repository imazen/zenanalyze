# coefficient-zenjpeg-opt integration audit, 2026-05-02

## What it is

**coefficient** is a distributed codec benchmarking framework with complete provenance tracking. It encodes source images across multiple codecs (JPEG, AVIF, WebP, JXL, etc.) at various quality levels, computes perceptual metrics (SSIMULACRA2, Butteraugli, DSSIM), and stores results with full version/config provenance.

The framework itself is **codec-agnostic**—it plugs different codecs in via a trait interface (`Codec`), not via a hardcoded integration with zenjpeg. However, zenjpeg (the jpegli JPEG port) is a default included codec.

## The integration pattern (concrete)

**Type**: Pure Rust library trait. No gRPC, no Lambda handlers, no separate binary.

### Codec side: `src/codec/jpegli_codec.rs`
- **Struct**: `JpegliCodec` wraps zenjpeg's encoder
- **Interface**: Implements the `Codec` trait (5 methods):
  - `name()` → codec identifier (e.g., "jpegli-s420")
  - `provenance()` → codec name, crate version, git commit, config JSON
  - `encode(rgb: &[u8], width, height, quality: u8, effort: u8) → Result<EncodeResult>`
    - Returns: `EncodeResult { data: Vec<u8>, blob_hash, size_bytes, encode_time_ms }`
  - `extension()` → ".jpg"
  - `effort_range()` → (min, max) effort values supported
- **Config**: `JpegliConfig` struct with:
  - `subsampling: Subsampling` (S420, S422, S444, etc.)
  - `use_xyb: bool` (XYB color space)
  - `use_optimized_tables: bool` (SA-optimized piecewise quantization tables)

### Task side: `src/model/task.rs`
- **Encoding task**: `EncodingTask { source_hash, codec_name, codec_config (JSON), quality, effort }`
- **Metric task**: `MetricTask { encoding_id, metric_name, implementation, impl_config }`

### Worker side: `src/worker/mod.rs`
- **Execution**: `Worker::execute_encoding(&task)` and `Worker::execute_metric(&task)`
- **Codec lookup**: `CodecRegistry::get(codec_name)` → codec instance
- **Determinism**: Same `(source_hash, codec_config, quality, effort)` always produces same output
- **Storage**: Results stored in `ResultStore` (LocalStore, GcsStore, MemoryStore, etc.)

## Job spec → encode → output flow

### Job spec (manifest)
`JobManifest` (JSON, stored in GCS or local):
```
{
  job_id: "cid22-train-q50-q100",
  name: "CID22 training sweep",
  total_chunks: 8,
  chunk_index: 3,
  encoding_tasks: [
    {source_hash: "abc123...", codec_name: "jpegli-s420", 
     codec_config: {subsampling: "s420", use_optimized_tables: true},
     quality: 75, effort: 0},
    ...
  ],
  metric_tasks: [
    {encoding_id: "xyz789...", metric_name: "ssimulacra2", 
     implementation: "fast-ssim2", impl_config: {}},
    ...
  ],
  config: {
    storage: {gcs_bucket: "coefficient-results", gcs_prefix: "benchmarks/cid22"},
    skip_completed: true,
    continue_on_error: true
  }
}
```

### Encode/metric execution
1. **cloud_worker binary** (or local benchmark):
   - Loads manifest from GCS (or local)
   - Creates `CodecRegistry`, `MetricRegistry`
   - For each `EncodingTask`:
     - Fetch source PNG from store by hash
     - Get codec from registry by name (e.g., "jpegli-s420")
     - Call `codec.encode(rgb, width, height, quality, effort)`
     - Store `EncodingRecord` (metadata JSON + encoded blob)
   - For each `MetricTask`:
     - Fetch encoding by ID
     - Fetch source RGB
     - Get metric from registry
     - Call `metric.compute(reference_rgb, encoded_jpeg)`
     - Store `MetricRecord` (score, timing, config)

### Output format
**Per-image tuple** (immutable records):
```
EncodingRecord {
  id: hash(source_hash, codec_name, codec_config, quality, effort),
  source_hash, quality, effort,
  codec_name: "jpegli", codec_crate: "zenjpeg", codec_crate_version: "0.1.0",
  codec_commit: "abc123...", codec_config: {...},
  blob_hash, size_bytes, encode_time_ms, worker_id, created_at
}

MetricRecord {
  id: hash(encoding_id, metric_name, implementation, impl_config),
  encoding_id, metric_name, implementation, impl_config,
  value: 85.4,  // e.g., SSIM2 score
  metric_time_ms, metric_crate_version, worker_id, created_at
}
```

**Pareto frontier / analysis**:
- Post-hoc: `analysis::ParetoFrontier::compute(metric_records)` finds non-dominated points
- Per codec + config_id (e.g., "jpegli-opt-s420"):
  - Points: `(bytes, ssim2_score)` or `(bytes, butteraugli_distance)`
  - CSV export: `image_path, codec_name, config_id, quality, effort, bytes, metric_value, encode_ms`

## Encoder version pinning

**Mechanism**: Cargo.toml path dependency (not Crates.io).

```toml
[dependencies]
zenjpeg = { path = "../zenjpeg-formula-opt/zenjpeg", 
            features = ["test-utils", "experimental-hybrid-trellis", "optimized-tables"] }
```

**Pinning approach**:
1. Sibling directory required at build time (../zenjpeg-formula-opt/)
2. Docker image includes specific Rust binary (versioned git commit baked in)
3. Provenance captured in EncodingRecord:
   - `codec_crate: "zenjpeg"`
   - `codec_crate_version: "0.1.0"` (from zenjpeg's Cargo.toml)
   - `codec_commit: "abc123..."` (from git describe)
4. **No auto-update**: Upgrading requires manual Cargo.toml edit + rebuild

**Feature flags** control variant behavior:
- `optimized-tables`: Uses SA-optimized piecewise quantization tables (v4 hybrid, +6.5 holdout pareto)
- `experimental-hybrid-trellis`: Trellis-enabled quantization
- `test-utils`: Exposes internal APIs for tests/benchmarks

## Schema (inputs and outputs at the coefficient ↔ codec boundary)

### Input to `Codec::encode()`
- `rgb: &[u8]` — raw RGB pixels, 3 bytes per pixel, row-major
- `width: u32, height: u32` — image dimensions
- `quality: u8` — [0, 100], interpretation depends on codec:
  - JPEG codecs: libjpeg-style (75 = good, 95 = excellent)
  - AVIF: [0, 63] mapped to [0, 100]
  - WebP: [0, 100] direct
- `effort: u8` — [0, codec.effort_range().1]:
  - jpegli: [0, 4] (CPU time)
  - mozjpeg: [0, 6] (progressive, optimize_huffman flags)
  - effort=0: codec default behavior

### Output: `EncodeResult`
- `data: Vec<u8>` — compressed bytes (JPEG, AVIF, WebP, etc.)
- `blob_hash: String` — SHA256(data) for deduplication
- `size_bytes: usize` — len(data) in bytes
- `encode_time_ms: u64` — wall-clock encoding duration (milliseconds)

### Codec provenance (for deduplication)
```json
{
  "subsampling": "s420",
  "use_xyb": false,
  "use_optimized_tables": true
}
```
This becomes `EncodingTask.codec_config` and is hashed to create deterministic task IDs. Same config + same quality = same task ID = automatic deduplication if already encoded.

## What's awkward / hardcoded

1. **Codec config is untyped JSON**
   - EncodingTask stores `codec_config: serde_json::Value`
   - No schema validation at parse time
   - Typos in JSON silently become "new codec" (wrong config → wrong task ID)
   - Workaround: validation in tests, but no runtime schema enforcement

2. **Quality semantics are per-codec**
   - PlanConfig has separate quality lists: `jpeg_qualities`, `avif_qualities`, `webp_qualities`
   - No way to say "target 1.0 BPP" — must specify qualities for each codec
   - zentrain will want to sweep over codec × subsampling × quality = complex cartesian product

3. **Hardcoded zenjpeg variant selection**
   - JpegliConfig has flags like `use_optimized_tables`, but these map to feature flags at compile time
   - To switch optimized tables on/off, rebuild the binary (or runtime flags not propagated)
   - No "registry" of JpegliConfig variants — you construct them manually in code

4. **Storage path layout is hardcoded**
   - Sharded by hash prefix: `encodings/{prefix1}/{prefix2}/{full_hash}/record.json`
   - No way to customize without subclassing `ResultStore`
   - zentrain might want flat layout or per-codec buckets for scalability

5. **Effort parameter conflates meaning**
   - mozjpeg effort [0, 6]: flags (progressive, optimize_huffman, trellis, etc.)
   - jpegli effort [0, 4]: cpu_effort (affects quantization tuning)
   - WebP effort [0, 6]: method (affects compression strategy)
   - No semantic link — each codec interprets effort differently, no shared enum

6. **Metric name mismatch risk**
   - MetricTask specifies both `metric_name` ("ssimulacra2") and `implementation` ("fast-ssim2")
   - Two tasks with same metric_name but different implementations = different MetricRecord IDs
   - Could accidentally compute same metric twice with different implementations
   - Workaround: version mapping rules, but upfront confusion

## What zentrain would do differently

1. **Codec trait expansion**
   - Add `config_variants()` → list of valid JpegliConfig instances
   - Add `quality_presets()` → {"low": [30, 40, 50], "high": [80, 90, 95]}
   - Statically enumerate codec × subsampling × variant combinations

2. **Typed task specs**
   - Instead of `codec_config: serde_json::Value`, use `codec_config: JpegliConfig` in Rust
   - Serde still for transport, but deserialization fails loudly on schema mismatch

3. **Quality curve model**
   - Add `TargetBpp` and `TargetMetric` as first-class quality specifications
   - Planner can say "sweep to hit [0.5, 1.0, 1.5, 2.0] BPP" → codec-specific quality search
   - Avoid per-codec quality list maintenance

4. **Effort semantics**
   - Define effort as union type: `enum Effort { Fast, Balanced, Thorough }` at the boundary
   - Each codec maps union to its own [0, N] range
   - Common API, per-codec interpretation

5. **Codec variant registry**
   - Instead of manually constructing JpegliConfig, use:
     ```rust
     CodecVariant::from_id("jpegli-opt-s420")?
     ```
   - Centralized configuration (not code), easier to toggle features

## Reusable patterns (what we'd lift for zenwebp/zenavif/zenjxl integrations)

1. **Codec trait** (`src/codec/mod.rs: Codec trait`)
   - ✅ Reusable as-is: `encode()`, `provenance()`, `extension()` are generic
   - New codecs just implement these 5 methods

2. **EncodingTask / MetricTask** (`src/model/task.rs`)
   - ✅ Generic: no JPEG-specific logic
   - Works for any codec as long as config fits in JSON

3. **ResultStore abstraction** (`src/store/mod.rs`)
   - ✅ Generic: not codec-specific, works for any encoding/metric record
   - LocalStore, GcsStore, MemoryStore all suitable for other codecs

4. **Worker execution** (`src/worker/mod.rs`)
   - ✅ Generic: walks task list, looks up codec/metric from registry, calls execute_encoding/execute_metric
   - No codec-specific branching

5. **Manifest/JobBuilder** (`src/planner/manifest.rs`)
   - ✅ Generic: doesn't care about codec internals, just task lists and chunking

6. **Provenance tracking** (EncodingRecord fields)
   - ✅ Reusable: crate name, version, commit, config JSON all generic
   - Works for zenwebp, zenavif, zenjxl, any future codec

7. **Codec registry** (`src/codec/mod.rs: CodecRegistry`)
   - ✅ Template for codec registry pattern; structure is:
     ```rust
     pub fn with_defaults() -> Self {
       let mut r = Self::new();
       r.register(Box::new(MozjpegCodec::new(Default::default())));
       r.register(Box::new(JpegliCodec::new(Default::default())));
       ...
     }
     ```
   - Add zenwebp/zenavif/zenjxl with 3 lines per codec (new!() + register)

## Test coverage

### Unit tests
- ~144 total test cases (from CLAUDE.md)
- Each codec tested: encoding round-trip, provenance capture, error handling
- Schema validation: EncodingRecord/MetricRecord serialization

### Integration tests (`tests/integration.rs`)
- **test_full_pipeline_single_image**: encode 128×128 test image with all codecs, compute all metrics
  - Verifies size reduction, metric ranges (SSIM2 ∈ (0, 100], butteraugli ≥ 0)
  - Tests all codec+metric pairs (cartesian product)
- **test_parallel_execution**: 4 images × multiple codecs/metrics with rayon parallelism
  - Checks task deduplication (skip_completed)
  - Exercises worker thread pool

### Smoke tests (CLI examples)
- `just bench` runs a quick 3-quality benchmark on CID22 (209 images)
- Produces viewer-loadable output (can inspect visually)

### Not tested
- ❌ Firestore cloud job tracking (requires emulator, optional feature)
- ❌ GCS storage (requires credentials)
- ❌ GCP Batch job submission (manual via justfile)
- ❌ Docker builds (visual inspection: `just docker-codec-versions`)

## Deploy / CI

### Build / Push Cadence
- **Local**: `just docker-cpu`, `just docker-gpu` on-demand
- **CI**: Not visible in this repo (likely in GitHub Actions externally)
- **Registry**: Push to GCR: `just docker-push-gpu` (hardcoded to gcr.io/abuddy-483902)

### Gates / Checks
- ✅ Cargo tests pass (`cargo test`) — unit + integration
- ✅ Docker image builds without errors
- ❌ No linting (clippy not enforced)
- ❌ No type validation (JSON configs untyped)
- ❌ No provenance signature verification (trust crate version strings)

### Deployment
- **Cloud**: GCP Batch job submission via batch_cli binary
  - Fetches manifest from GCS, workers download and execute
  - Writes results back to GCS (or local)
- **Firestore tracking**: Optional; job status synced after each chunk completes
- **Stall detection**: Prototype Cloud Functions (not yet deployed) detect stuck jobs after 15min or 6h runtime

### Known deployment friction
- Must push Docker image to GCR manually before job submission
- Corpus must be pre-uploaded to GCS
- Service account requires 3 IAM roles (storage.objectAdmin, datastore.user, batch.jobsEditor)
- No dry-run without full manifest download (batch_cli has `--dry-run` but still does most planning work)

---

## Summary for zentrain integration

**Pattern to copy**:
1. Implement `Codec` trait for each codec (zenwebp, zenavif, zenjxl)
2. Add variants to `CodecRegistry::with_defaults()`
3. Define config types (ZenwebpConfig, etc.) similar to JpegliConfig
4. Tasks and metrics flow unchanged: coefficient's Worker will execute them

**Pain points to solve in zentrain**:
- Untyped JSON configs → add strongly-typed config variants + centralized registry
- Per-codec quality lists → support quality curves (BPP targets) instead
- Hardcoded sharding layout → allow pluggable storage backends (already present, just not used)
- Manual manifest creation → expose higher-level job DSL (sweep spec → manifest)

**Reusable from coefficient**:
- Entire storage/worker/planner pipeline (no changes needed)
- Codec trait, registry pattern, provenance tracking
- Integration with Firestore, GCS, Docker deployment machinery
