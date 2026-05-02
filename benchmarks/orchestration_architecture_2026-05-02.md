# Orchestration architecture: zentrain ↔ coefficient ↔ squintly ↔ codecs

Synthesis of three additional read-only audits — coefficient (cloud
codec benchmarking), squintly (psychovisual ground-truth collection),
and coefficient-zenjpeg-opt (concrete integration shim) — combined
with the seven prior single-machine audits.

## TL;DR

The picker training pipeline is fighting a problem that's already
solved at a different layer. **coefficient is the lab — built around a
task-based worker model, immutable JSON records, and a pluggable
storage backend; all 4 zen-codecs are integrated; existing oracle-d2
substrate has 75 k encodings + 108 k metric records.** **squintly
already collects subjective human labels with viewing-condition
tagging and exports zenanalyze-compatible TSVs.** The single-machine
"centralize the lab inside zentrain" plan from earlier this session is
largely **obsolete or absorbed**.

**Cloud substrate**: GCP is being abandoned. The new substrate is
**Cloudflare R2 (object storage, S3-compatible) + vast.ai (rented
spot compute) + storage-as-queue (no managed coordinator)**. Cost
estimate ~5–7× cheaper than GCP. coefficient's task-based design
maps cleanly to vast.ai (better fit than GCP Batch was) — the
migration is two new modules (~600 LoC) plus deleting the
GCP-specific paths.

zentrain's job shrinks substantially: it consumes coefficient outputs,
consumes squintly outputs, and trains pickers. The remaining work
splits into **Phase M (substrate migration: R2 + vast.ai +
storage-as-queue, ~1.5 weeks) running in parallel to Phase A (5
bridges between layers, ~1 week)**.

## The actual layer map

| Layer | What | Where it lives | Status |
|---|---|---|---|
| Source corpus | PNG images, hashes, manifests | coefficient `SourceRecord` | shipped |
| Encoding (cross-codec, scaled) | All codecs × configs × qualities | coefficient `EncodingRecord` + vast.ai | shipped, all 4 codecs integrated |
| Objective metrics | SSIM2, Butteraugli, zensim | coefficient `MetricRecord` | shipped |
| Subjective metrics | Bradley-Terry from human labels | squintly `pareto.tsv` | shipped (v0.1) |
| Viewing-condition thresholds | `q_threshold(condition_bucket)` | squintly `thresholds.tsv` | **v0.2 stub — not yet implemented** |
| Image features | zenanalyze `feat_*` | coefficient `source_features.json` | shipped |
| Picker training (MLP, ablate, bake) | zentrain `train_hybrid.py` | zenanalyze/zentrain/ | shipped, single-machine |
| Picker runtime | ZNPR v2 inference | zenpredict + zenpicker | shipped |
| Codec consumption | encoder reads picker.bin → chooses config | each codec | **3 of 4 codecs not wired** |

The infrastructure largely exists. What's missing is the **glue
between layers**.

## Existing data substrate

- **coefficient oracle-d2**: 75 000 encodings, 108 000 metric records,
  90 active sources, 8-9 quality levels each. 54 k Pareto rows × 17
  cols at `/home/lilith/oracle-d2-store/oracle-d2/pareto_rows.csv`.
  188 KB `source_features.json`.
- **squintly v0.1**: SQLite-backed trial collection with viewing
  conditions; pareto.tsv export functional; thresholds.tsv stubbed.

This is real production-shape data. zentrain's training cycle should
be drawing from it, not from per-codec hand-rolled sweeps.

## Cloud substrate: R2 + vast.ai (no GCP)

**We're abandoning GCP entirely.** The new substrate is **Cloudflare
R2 for object storage** + **vast.ai for compute**. coefficient's
architecture is mostly cloud-agnostic at the abstraction layer
(Codec trait, immutable JSON records, SplitStore pattern) — only two
modules need to change, plus a coordinator we have to build.

### What survives unchanged

- **Codec trait + CodecRegistry** — pure Rust, cloud-agnostic
- **Task-based worker model** — workers pull tasks independently, no
  central queue. **Maps cleanly to vast.ai's "rent a box and run"
  model** — better fit than GCP Batch's submit-to-managed-service
  pattern was.
- **`SourceRecord` / `EncodingRecord` / `MetricRecord`** — JSON; the
  storage backend is pluggable.
- **SplitStore pattern** — metadata local (fast lookups), blobs in
  object storage. Already supports custom backends.
- **`task_id` deduplication** — same `(source_hash, codec_config, q,
  effort)` always hashes to same task; if record exists, skip.
  Doubles as "free" idempotent queue when storage is the queue.
- **Docker image bundling the workspace at a fixed commit** — works
  identically on vast.ai instances; just `docker run` instead of
  GCP Batch's task spec.

### What needs to migrate

| Was (GCP) | Now (R2 + vast.ai) | Effort |
|---|---|---:|
| `src/store/gcs.rs` (GCS client) | `src/store/r2.rs` (S3-compatible API; same trait, custom endpoint URL) | 4 hr |
| `src/cloud/batch.rs` (GCP Batch submission) | `src/cloud/vastai.rs` (vast.ai launcher: `vast create instance`, `ssh`, kick off worker) | 2 days |
| Firestore (job tracking + status) | One of: Cloudflare D1 (SQLite-on-edge, 5 GB free), small Postgres on a Hetzner/DO VPS, or **storage-as-queue** (workers atomically claim a task by writing a lease file to R2; coordinator-less) | 1 day per option |
| Cloud Functions (stall detection) | Cron Triggers on Cloudflare Workers, or a GitHub Actions cron that polls R2 for stale lease files | 4 hr |
| GCP service account auth | Cloudflare API tokens + R2 access keys (S3-compatible) | 30 min |
| GCR (container registry) | Docker Hub or GHCR; pull on vast.ai instance startup | 1 hr |

**Net new code in coefficient: ~600 LoC.** The migration is
feasible because R2 IS S3-compatible (drop in `aws-sdk-s3` or
`rust-s3` crate with `endpoint = "https://<account>.r2.cloudflarestorage.com"`)
and vast.ai's marketplace API has Python and CLI clients (`vastai-cli`).

### The coordinator question (real new design work)

GCP Batch + Firestore handled job state and stall detection
implicitly. On the new stack, three coordinator options:

**Option A: storage-as-queue (no coordinator)**
- Workers list R2 `tasks/pending/` prefix.
- Atomically claim a task by `MoveObject(tasks/pending/X.json → tasks/in-flight/{worker_id}/X.json)`
  with `If-Match` header (R2 supports conditional writes).
- Heartbeat by re-uploading the in-flight file every 60 s with
  updated timestamp.
- Coordinator-less; **zero infrastructure beyond R2**.
- Stall detection: cron job moves `in-flight/*.json` older than 5 min
  back to `pending/`.
- Trade-off: R2 LIST throughput limits (~1000 ops/s); fine for
  ≤10k workers but not 100k.

**Option B: Cloudflare D1 (SQLite-on-edge)**
- Single SQL table for tasks; workers poll via HTTPS.
- Free tier: 5 GB, 25 M reads/day. Generous for our scale.
- Cloudflare Workers expose the API.
- Stall detection: D1 query for stale rows; cron-triggered Worker.

**Option C: Postgres on a small VPS (Hetzner/DO ~$5/month)**
- Standard SQL. SKIP LOCKED-style atomic claim.
- Stall detection: cron job.
- Trade-off: one-server bottleneck; need to keep the VPS healthy.

**Recommendation**: **Option A (storage-as-queue) for v1**. Fewest
moving parts, zero new services, R2 is already the storage substrate.
Migrate to Option B (D1) only if R2 LIST throughput becomes a
bottleneck (unlikely under 1000 concurrent workers).

### Cost picture (vs GCP)

This is all back-of-envelope; concrete numbers need a real
benchmark run.

| Resource | GCP (was) | R2 + vast.ai (now) | Est. ratio |
|---|---|---|---:|
| Object storage (1 TB) | ~$20/mo (GCS Standard) + egress | ~$15/mo (R2) **+ free egress** | ~5× cheaper for read-heavy workloads |
| Compute (1000 CPU-hr) | ~$50 (Batch n1-standard preemptible) | ~$10 (vast.ai 16-core spot) | ~5× cheaper |
| GPU compute (RTX 4090, 100 hr) | ~$300 (T4 / A100 GCP) | ~$50 (4090 on vast.ai spot) | ~6× cheaper |
| Coordinator | Firestore (~free at our scale) | R2 (in storage cost) or D1 (free tier) | ~equal |

**Headline**: ~5× cost reduction on compute + 5× on storage + free
egress within Cloudflare = roughly **5–7× cheaper end-to-end** than
GCP, with the upside that vast.ai's spot pricing on consumer GPUs
makes one-shot massive sweeps newly affordable.

The risk: vast.ai is a marketplace with variable instance
availability and reliability. Not the same SLA as managed services.
For multi-day sweeps with checkpointing, fine. For latency-sensitive
production inference, no.

## What we don't need to build (was queued in rapid_iteration_plan)

The single-machine plan's Phase 1+2 ("centralize the lab") is mostly
obsolete:

| Item | Status | Why |
|---|---|---|
| Centralized Rust extractor in zenanalyze | ❌ obsolete | coefficient emits `source_features.json` |
| Canonical pareto-row schema per codec | ❌ obsolete | coefficient defines this |
| Shared `zenpicker-harness` crate | ❌ obsolete | coefficient's `Codec` trait IS that |
| Per-codec sweep parallelism (50-hr Phase 3 LHS) | ❌ obsolete | vast.ai with 1000+ workers |
| Resume-on-crash sweep harness | ❌ obsolete | coefficient task model has automatic dedup via `task_id` |
| Encoder version pinning per sweep | ❌ obsolete | coefficient Docker images bundle workspace at fixed commit |

What stays useful:

| Item | Status | Why |
|---|---|---|
| Wire 3 idle pickers (zenjpeg/zenjxl/zenavif) | ✅ still needed | Same regardless of substrate |
| `--seed` pinning everywhere | ✅ still needed | Reproducibility independent of substrate |
| Corpus content hashing | ✅ helpful, partial | coefficient has `source_hash`; thread it through |
| Parallel multi-seed LOO | ✅ still needed | Training-side, independent of encoding |
| Teacher cache (content-addressed) | ✅ still needed | Training-side |
| Vectorized Parquet consumption | ✅ still needed | Training-side performance |
| Phase 3 quality CI (mostly) | ✅ still needed | Training-side guards |

## The five bridges

These are the specific seams where coefficient + squintly + zentrain
need to meet. Each is small (1–2 days), well-scoped, and unblocks a
concrete capability.

### Bridge 1: coefficient `pareto_rows.csv` → zentrain `PARETO` TSV

**Problem**: coefficient emits `pareto_rows.csv` shaped
`(source_hash, codec_name, codec_config_json, q, bpp, ssim2, butter,
zensim)`. zentrain `train_hybrid.py` expects shape
`(image_path, size_class, width, height, config_id, config_name, q,
bytes, zensim, encode_ms)`.

**Solution**: add `coefficient analysis --export zentrain` exporter
that emits zentrain-shaped Parquet directly. Includes:
- Synthesizes `image_path` from `source_hash` (or path from manifest)
- Maps `(source_hash, max_dim_quartile)` → `size_class`
- Synthesizes `config_name` from `codec_name + sorted_config_kvs`
  (zentrain's regex parser then extracts categorical+scalar axes)
- `bpp · pixel_count = bytes` (closes the units gap)

**Effort**: ~1 day in coefficient's analysis module. ~200 LoC.

### Bridge 2: squintly outputs → zentrain `METRIC_COLUMN` + threshold-aware `ZQ_TARGETS`

**Problem**: squintly produces subjective quality (BT-derived θ) and
viewing-condition thresholds. zentrain currently uses objective metrics
(zensim/ssim2/butteraugli) and a fixed ZQ grid. The picker objective
"minimize bytes subject to zensim ≥ T" should evolve to
"minimize bytes subject to q ≥ q_threshold(viewing_conditions)".

**Solution**:
- `train_hybrid.py` already supports `METRIC_COLUMN` config — point it
  at squintly's `theta` column when subjective training is requested.
- Add `THRESHOLD_TABLE = Path("squintly_thresholds.tsv")` knob to
  picker config.
- `ZQ_TARGETS` becomes `ZQ_TARGETS_BY_CONDITION` — per
  `condition_bucket`, a list of target θ values.
- Argmin objective consumes `condition_bucket` as a categorical input
  + per-bucket threshold.

**Effort**: 1 day in `train_hybrid.py`; 1 day to ship squintly's
`thresholds.tsv` v0.2 export. **Blocked on squintly v0.2.**

### Bridge 3: provenance chain coefficient → zentrain → codec

**Problem**: coefficient's `EncodingRecord` carries full provenance
(crate name + version + commit + config_json) per encoding. zentrain's
trained picker.bin doesn't surface this chain. We can't tell from a
.bin which corpus it was trained against, which codec commits, which
training seed.

**Solution**: ZNPR v2 metadata TLV grows new keys:
```
picker.bin metadata:
  source_corpus_hash:   sha256 of all PNGs in coefficient corpus
  encoding_run_id:      coefficient run_id
  encoding_commits:     {"zenjpeg": "abc123", "mozjpeg-rs": "def456", ...}
  features_extractor_commit: zenanalyze commit
  squintly_export_id:   if subjective metric used (else absent)
  zentrain_seed:        random seed
  train_config_hash:    sha256 of the picker config Python file
```

Each `.bin` is fully traceable to its inputs. **CI gate**: bake
determinism = same provenance → same bytes.

**Effort**: 1 day in zenpredict (TLV schema), 4 hr in zentrain (emit
the chain on bake), 4 hr in CI (determinism check).

### Bridge 4: mode switch (local vs vast.ai)

**Problem**: zentrain runs `train_hybrid.py` on the calling machine.
With coefficient as the encode source, the encoding can be local OR
on vast.ai. zentrain doesn't know how to wait on a coefficient
batch job.

**Solution**: `zentrain orchestrate-sweep --coefficient-mode {local,batch}`.
- `local`: runs coefficient's local Worker, waits for completion, then
  trains.
- `batch`: submits to coefficient vast.ai, polls D1/Postgres, fetches
  outputs from Cloudflare R2 when done, then trains.

Decision threshold: ~1k images → local; ~100k images → batch. Not a
sharp threshold; user override should be possible.

**Effort**: 2 days. Coefficient already has the batch primitives; zentrain
just orchestrates.

### Bridge 5: picker.bin deployment

**Problem**: zentrain trains picker.bin. Where does it go? Each codec
consumer needs to load a specific bin. Today zenwebp embeds it; the
other 3 codecs don't load anything (Phase 0.1–0.3).

**Solution** — hybrid:
- **Embedded baseline**: codec crate ships a `default_picker.bin` in
  `src/picker.bin` (current zenwebp pattern). Atomic with codec
  release.
- **Sideload override**: codec accepts `with_picker(path)` at runtime.
  Decoupled from codec releases.
- **coefficient-served catalog**: pickers live in coefficient's
  `analysis/pickers/` directory; codecs at runtime CAN fetch the
  latest from a versioned URL.

Implementation: each codec gets a `Picker` accessor on its config
struct. Default = embedded; runtime override = sideload; production
update mechanism = picker-update daemon (out of scope for this commit).

**Effort**: 4 hr × 4 codecs.

## Phased plan (revision of rapid_iteration_plan)

### Phase M — substrate migration (~1.5 weeks, runs in parallel to Phase A)

Phase A bridges can ship using LOCAL coefficient data (oracle-d2 is
on disk already); Phase M is the cloud migration that unlocks
million-image sweeps. They don't block each other.

| # | Item | Effort | Impact |
|---|---|---:|---|
| M.1 | `src/store/r2.rs` (S3-compatible R2 client; same trait as gcs.rs) | 4 hr | Storage migration |
| M.2 | `src/cloud/vastai.rs` (vast.ai instance launcher via `vastai-cli` API) | 2 days | Compute migration |
| M.3 | Storage-as-queue protocol (atomic claim via R2 conditional writes; lease files in `tasks/in-flight/`) | 1 day | Coordinator-less queueing |
| M.4 | Stall-detector cron (Cloudflare Workers Cron Trigger or GitHub Actions; moves stale leases back to `pending/`) | 4 hr | Resilience |
| M.5 | Docker image build + push pipeline (Docker Hub / GHCR; tagged by workspace commit) | 1 day | Worker bootstrapping |
| M.6 | End-to-end smoke test: 100-image sweep on 4 vast.ai instances writing to R2 | 1 day | Validation |
| M.7 | Rip out `src/store/gcs.rs`, `src/cloud/batch.rs`, Firestore code paths | 4 hr | Cleanup |

### Phase A — bridges (~1 week)

| # | Item | Effort | Impact |
|---|---|---:|---|
| A.1 | Wire 3 idle codec pickers (zenjpeg, zenjxl, zenavif) | 10 hr | Unblocks production output. **Highest ROI.** |
| A.2 | coefficient `analysis --export zentrain` (Bridge 1) | 1 day | Replaces 4 per-codec sweep harnesses operationally |
| A.3 | train_hybrid.py reads coefficient TSV directly | 4 hr | Closes the data loop |
| A.4 | Squintly pareto.tsv as `METRIC_COLUMN` (Bridge 2 part 1) | 1 day | Subjective-quality picker training |
| A.5 | Provenance chain TLV in picker.bin (Bridge 3) | 1 day | Reproducibility floor |

**~1 week. Recovers most of currently-wasted time AND unblocks the 3 idle pickers.**

### Phase B — scale-out (~1.5 weeks)

| # | Item | Effort | Impact |
|---|---|---:|---|
| B.1 | zentrain `--coefficient-mode batch` (Bridge 4) | 2 days | Million-file sweeps |
| B.2 | zentrain awaits coefficient vast.ai job | 1 day | Async sweep |
| B.3 | Squintly `thresholds.tsv` v0.2 implementation | 1 day | Condition-aware picker (squintly side) |
| B.4 | Threshold-aware `ZQ_TARGETS_BY_CONDITION` (Bridge 2 part 2) | 2 days | Condition-aware picker (zentrain side) |
| B.5 | Cross-codec dashboard reads coefficient outputs | 2 days | One pane of glass |

### Phase C — production reliability (~1 week)

| # | Item | Effort | Impact |
|---|---|---:|---|
| C.1 | Picker.bin sideload API (Bridge 5) per codec | 4 hr × 4 | Decouples picker updates from codec releases |
| C.2 | CI gate: coefficient → zentrain integration test | 1 day | Catches schema drift |
| C.3 | Bake determinism CI gate | 4 hr | Reproducibility |
| C.4 | A/B eval CI gate (held-out coefficient subset) | 1 day | Catches regressions |
| C.5 | Cross-codec feature-schema audit | 4 hr | Schema drift |

### Still useful from rapid_iteration_plan (training-side)

These remain regardless of substrate:

| # | Item | Effort | Impact |
|---|---|---:|---|
| R.1 | `--seed` pinning everywhere | 1 hr × 4 | Reproducibility floor |
| R.2 | Parallel multi-seed LOO | 4 hr | 175 min → 30 min |
| R.3 | Content-addressed teacher cache | 6 hr | Ablation retrains skip teachers |
| R.4 | Vectorized Parquet consumption | 4 hr | Pareto load 54s → 5s |
| R.5 | `zenpredict-diff` + `zenpredict-inspect --raw-scores` | 1 day | Decision auditing |

## What zentrain becomes

After the bridges land, zentrain's responsibilities shrink to:

1. **Consumer of coefficient outputs**: read `pareto_rows.csv` (or the
   zentrain-format export) + `source_features.json`.
2. **Consumer of squintly outputs**: optional `pareto.tsv` (subjective)
   + `thresholds.tsv` (condition-aware).
3. **Picker trainer**: `train_hybrid.py` + ablation tools + multi-seed
   LOO + dendrogram + LOO. **Fast iteration here, decoupled from
   encoding.**
4. **Bake**: ZNPR v2 with full provenance chain.
5. **Orchestrator**: `zentrain orchestrate-sweep` calls coefficient
   (local or batch) when fresh sweep data needed.

zentrain stops owning: per-codec sweep harnesses, per-codec
extractors, per-codec scaling. Those are coefficient's job and
coefficient does them at million-image scale already.

## Open questions

1. **Coordinator strategy** — storage-as-queue (R2-only, recommended
   for v1), Cloudflare D1, or Postgres-on-VPS? See "The coordinator
   question" above. Recommend storage-as-queue; revisit at >1k
   concurrent workers.
2. **vast.ai instance launching** — programmatic (`vastai-cli` API)
   or manual at first? Reproducibility: how do we pin instance
   image / Docker tag / workspace commit per sweep run?
3. **R2 region + redundancy** — single region (cheaper, lower latency
   from one source) vs auto-replicated (Cloudflare's default)?
   Affects cost and durability story.
4. **Spot reliability tradeoffs** — vast.ai spot pricing is the cost
   win, but instances can be reclaimed. Is the task model robust to
   "worker dies mid-encode" (yes — task_id dedup + atomic writes), and
   what's the right heartbeat / lease duration?
5. **Storage for picker.bin distribution** — R2 bucket
   (`pickers.coefficient.imazen.io/<codec>/<commit>.bin`) vs codec
   crate embedded vs sidecar daemon? Recommend R2 with embedded
   fallback; codecs fetch on first encode of a session.
6. **Squintly v0.2 timeline** — when does `thresholds.tsv` ship?
   Blocks Phase B.4 (condition-aware picker).
7. **Cost envelope** — typical 1 M × 4 codecs × 30 q × N configs
   cycle: ~5–7× cheaper than GCP per the substrate-migration
   estimate, but need a real benchmark run with one codec to confirm.
8. **Where does zensim training live in this model?** Standalone
   (`zensim-validate` Rust CLI) or absorbed by coefficient + zentrain?
   zensim audit said structurally separate today; with the new
   substrate, training zensim itself on vast.ai GPUs is newly
   affordable (the 4 K-image SROCC validation runs). Worth
   considering.
9. **CodecFamily enum evolution** — adding a 5th/6th codec family
   requires lockstep meta-picker rebake (zenpicker audit).
   Mitigation strategy.

## Cross-references

- `benchmarks/audits-2026-05-02/coefficient.md`
- `benchmarks/audits-2026-05-02/squintly.md`
- `benchmarks/audits-2026-05-02/coefficient_zenjpeg_opt.md`
- `benchmarks/rapid_iteration_plan_2026-05-02.md` (the
  single-machine plan; this doc supersedes large portions of its
  Phase 2)
- `zentrain/INVERSION.md` (originally framed inversion, now
  largely subsumed by this orchestration model)
- `benchmarks/audits-2026-05-02/{zenjpeg,zenwebp,zenavif,zenjxl,
  zenpicker_zenpredict,zensim,zentrain_zenanalyze}.md` (the seven
  prior audits)
