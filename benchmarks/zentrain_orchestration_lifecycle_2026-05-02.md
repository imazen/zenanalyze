# zentrain lifecycle orchestration: drive everything, multi-session safe, pluggable compute

The user's three asks resolve to one architecture:

1. **Invert control so zentrain drives everything** — codec authors stop calling sweep / train / bake scripts directly; they invoke one `zentrain orchestrate` command.
2. **Multiple sessions work independently and idempotently per codec** — N concurrent sessions on the same codec converge to the same `.bin` without duplicating expensive work.
3. **Mix in additional computers seamlessly** — a vast.ai instance, a Hetzner box, my workstation can join or leave the work pool with no manual coordination.

All three are properties of the same primitive set: **content-addressed task storage + storage-as-queue + capability-aware workers**.

## TL;DR

- Every artifact (corpus → manifest → features → pareto → picker.bin) gets a content hash deterministic from inputs. Same inputs → same hash → same file → cache hit.
- `zentrain orchestrate --codec X --objective Y --metric Z` walks back the hash chain, finds what's missing, registers tasks at content-addressed paths in R2, optionally waits.
- Workers (any machine) advertise capabilities, atomically claim tasks via R2 conditional writes, execute, write outputs at content-hash paths.
- Multi-session correctness: deterministic tasks + content-hash paths + atomic claim → races are harmless (worst case: a few duplicate encodes; coefficient's existing `task_id` dedup handles that already).
- Adding a computer: `cargo run --bin zentrain-worker --capabilities ...`. Removing: kill the process. Stall detector reclaims abandoned tasks.

## The control flow

### Orchestration entry point

```
zentrain orchestrate --codec zenwebp --objective default --metric zensim
                     [--corpus-spec <toml>]
                     [--mode local|distributed|auto]
                     [--wait|--detached]
                     [--invalidate-features|--invalidate-pareto]
```

What it does:

1. Parse spec → resolve codec config + objective + metric.
2. Walk back hashes top-down:
   ```
   picker_hash    = sha256(features_hash, pareto_hash, train_config_hash, seed, objective_id)
   pareto_hash    = sha256(manifest_hash, codec_commit, sweep_config_hash)
   features_hash  = sha256(manifest_hash, zenanalyze_schema_hash, extractor_commit)
   manifest_hash  = sha256(corpus_hash, sizes_set, content_classes_set)
   corpus_hash    = sha256(sorted source-image SHAs)
   ```
3. Check storage for each hash. Top-down: if `pickers/<codec>/<picker_hash>.bin` exists, done — return its URL.
4. If not: walk back to find the existing inputs. Produce only what's missing.
5. For each missing intermediate: write a JSON task record to `r2://tasks/pending/<task_hash>.json`.
6. Mode dispatch:
   - `local`: spawn worker pool on this machine.
   - `distributed`: just register tasks; rely on existing pool.
   - `auto`: local for ≤ 1k images; distributed for ≥ 100k images; threshold tunable.
7. If `--wait`: poll `pickers/<codec>/<picker_hash>.bin` until it exists (or any registered task fails). Otherwise return the future-path.
8. Output the path: `r2://pickers/<codec>/<picker_hash>.bin`.

### Worker entry point

```
zentrain worker [--worker-id auto]
                [--capabilities cpu=16,gpu=0,ram_gb=128,codecs=zenwebp,zenavif]
                [--pull r2://tasks/pending/]
                [--idle-strategy halt|sleep_60s]
                [--task-types encoding,features,training,bake]
```

Loop:

1. List `r2://tasks/pending/` entries matching this worker's capabilities.
2. Pick one. Atomic claim via R2 conditional write:
   ```
   PUT tasks/in-flight/<worker_id>/<task_hash>.json
       Source: tasks/pending/<task_hash>.json
       If-Match: <etag>
   ```
   If the conditional fails, another worker beat us; pick a different task.
3. Execute the task (encode / extract / train / bake) using the existing per-codec tooling under the hood.
4. Write the output to immutable storage at the content-hash path:
   ```
   r2://encodings/<codec>/<encoding_hash>.{jpg|webp|avif|jxl}
   r2://features/<codec>/<features_hash>.parquet
   r2://pareto/<codec>/<pareto_hash>.parquet
   r2://pickers/<codec>/<picker_hash>.bin
   ```
5. Move `tasks/in-flight/.../X.json` → `tasks/done/<task_hash>.json`.
6. Heartbeat: re-upload `tasks/in-flight/.../X.json` with updated timestamp every 60 s during long tasks.

Stall detector (Cloudflare Workers Cron, 5-min cadence):

- Scan `tasks/in-flight/` for tasks whose heartbeat is > 5 min stale.
- Move stale ones back to `tasks/pending/` (workers will re-claim).
- Append the abandoned-task event to `tasks/audit/abandoned.jsonl` for observability.

## Multi-session correctness

Three sessions running concurrently:

```
A: zentrain orchestrate --codec zenwebp --metric zensim
B: zentrain orchestrate --codec zenavif --metric zensim   # different codec
C: zentrain orchestrate --codec zenwebp --metric zensim   # same as A
```

What happens:

- **A** computes its `picker_hash`, sees nothing in R2, registers ~80 encoding + 1 features + 1 training + 1 bake task at deterministic content-hash paths.
- **B** does the same for zenavif. Disjoint task set; no conflict with A.
- **C** computes the SAME `picker_hash` A did (deterministic — same inputs). Sees A's tasks already in `pending/`. Registers nothing (idempotent). If `--wait`, subscribes to the picker output path; otherwise returns immediately with the future-path.

Workers pull from `pending/`, claim atomically. Each task executes once total. A's wait, B's wait, and C's wait all return when their respective bakes land at the picker path.

### Race resolution by task type

| Task type | Strategy | Reasoning |
|---|---|---|
| Encoding | Optimistic + lease | Same `(image_hash, codec_config, q)` → same bytes. If two workers race and both claim, second observes lease and skips. If lease is somehow lost: both encode; identical bytes; coefficient `task_id` dedup makes the second write a no-op. |
| Feature extraction | Optimistic + lease | Same logic — features are deterministic from `(image_hash, schema_hash)`. |
| Training | Pessimistic lease | Training is expensive (~30 s–10 min depending on corpus); duplication waste matters. Lease + 60 s heartbeat + 5 min stall reclaim. |
| Bake | Optimistic | Same inputs → same `.bin` (deterministic bake required: seed pinned, sklearn `random_state`, etc.). Bake is cheap (~5 s); harmless if duplicated. |

The hash determinism contract: every task type is a deterministic function of its inputs. Trainer pins seeds; sklearn `random_state` pinned; numpy `default_rng(seed)`. CI gate enforces this.

## Worker capability advertising

Each worker on startup writes:

```
r2://workers/<worker_id>/heartbeat.json

{
  "worker_id": "vastai-4090-abc123",
  "started":   "2026-05-02T15:00:00Z",
  "last_seen": "2026-05-02T15:30:00Z",
  "capabilities": {
    "cpu_cores": 16,
    "ram_gb":    64,
    "gpu":       "RTX 4090",
    "gpu_vram_gb": 24,
    "supports_codecs":      ["zenwebp", "zenavif", "zenjxl"],
    "supports_metrics_cpu": ["zensim", "ssim2"],
    "supports_metrics_gpu": ["ssim2-gpu", "butter-gpu"],
    "task_types":           ["encoding", "features", "training", "bake"]
  },
  "current_task": "tasks/in-flight/abc123/encoding-xyz789.json"
}
```

Tasks specify requirements:

```
{
  "task_id": "encoding-xyz789",
  "task_type": "encoding",
  "requires": {
    "codec": "zenavif",
    "metric": "ssim2-gpu"   // implies gpu: true
  },
  "inputs": {
    "image_blob": "r2://corpus/<sha>.png",
    "codec_config_json": {...},
    "quality": 75
  },
  "output": "r2://encodings/zenavif/<encoding_hash>.avif"
}
```

Workers filter `pending/` to tasks matching their capabilities. Adding a vast.ai instance is one command. Removing is killing the process. Stall detector handles uncleanly-departed workers (the job goes back to `pending/` after 5 min).

No central scheduler to update. The R2 bucket is the entire coordination layer.

## What this replaces

| Today | Tomorrow |
|---|---|
| Codec author runs `cargo run --release --example zq_pareto_calibrate` per codec | `zentrain orchestrate --codec zenjpeg` |
| Codec author runs `python3 train_hybrid.py --codec-config zenjpeg_picker_config` | (same orchestrate command) |
| Codec author runs bake script | (same) |
| Codec author copies `.bin` into codec crate | `r2://pickers.imazen/zenjpeg/<hash>.bin` — codec at runtime fetches the versioned URL (or has it embedded as default) |
| Per-session "what work should I do?" manual coordination | None — sessions read content-hash storage, work converges automatically |
| Adding a Hetzner box: write deployment scripts, register with scheduler | `cargo run --bin zentrain-worker -- --pool default` and walk away |
| Forgetting to pin a seed → reproducibility loss | CI gate; deterministic-bake test in CI |

## Phased delivery (~2 weeks for v1)

| Phase | What | Effort | Deliverable |
|---|---|---:|---|
| **O1** | Hash resolver — `zentrain orchestrate --dry-run` walks back the DAG, prints what would run | ~2 days | Spec → task graph; nothing executes |
| **O2** | Task graph builder + R2 storage backend (`tasks/{pending,in-flight,done}/`, content-hashed paths) | ~3 days | Tasks land in R2; deterministic paths |
| **O3** | Worker for encoding tasks (wraps existing codec extractors and coefficient's encoder shims) | ~2 days | Distributed encoding |
| **O4** | Worker for feature extraction tasks (wraps `zenanalyze::extract_features_from_manifest` once Tier 2 lands; falls back to per-codec extractor today) | ~1 day | Distributed features |
| **O5** | Worker for training + bake tasks (wraps `train_hybrid.py`) | ~2 days | Distributed training |
| **O6** | Atomic claim via R2 conditional writes; Cloudflare Workers Cron stall detector | ~2 days | Coordination + resilience |
| **O7** | `zentrain status` CLI: list pending / in-flight / done; worker capabilities; recent bakes per codec | ~1 day | Operator dashboard |

**After O1+O2**: orchestration registers tasks but doesn't execute. You can stand up workers on any machine and they pick up the work. Sessions invoke `orchestrate` from anywhere; tasks land in the same queue.

**After O3+O4+O5**: full distributed pipeline working end-to-end.

**After O6**: races handled cleanly; abandoned tasks reclaimed.

**After O7**: operators see the system's state in one view.

Each phase is independently useful. Pause-resume between phases is fine.

## Implementation notes

### Worker is a Rust binary, not Python

The trainer runs as a subprocess (workers shell out to `python3 train_hybrid.py ...`); the encode/feature workers shell out to existing Rust binaries. The worker itself is Rust to share the R2 client + atomic-claim logic with coefficient.

### Worker pool can be heterogeneous

A 7950X workstation can advertise `cpu=16, gpu=0, codecs=all`. A vast.ai 4090 can advertise `cpu=8, gpu=true, gpu_vram_gb=24, metrics_gpu=true`. They both pull from the same `pending/` prefix; tasks specify requirements; matching is automatic.

A small DigitalOcean droplet runs the cron stall detector (~$5/month) — no compute capability, just bookkeeping.

### Hash determinism is a CI gate

Every task type produces deterministic output for its inputs. CI runs:

- Encode task with fixed inputs (image, config, q) twice; assert byte-identical output.
- Feature extraction with fixed inputs twice; assert byte-identical output.
- Training with fixed inputs + seed twice; assert bit-identical .bin (this is the bake-determinism gate from the multi-MLP safety doc).

Failing this gate blocks release. Without determinism, content-hash storage is a lie and idempotency breaks.

### Backward compat

Existing per-codec sweep harnesses still work standalone. The orchestration layer is additive — it wraps them. A codec author who prefers running `zq_pareto_calibrate` directly can keep doing so; the result populates the same R2 paths as `zentrain orchestrate` would have, so cache hits work either way.

## Cross-references

- **`orchestration_architecture_2026-05-02.md`**: substrate (R2 + vast.ai + storage-as-queue). This doc operates one layer up — `zentrain orchestrate` is the entry point that uses the substrate primitives.
- **`rapid_iteration_plan_2026-05-02.md`**: Phase 0 codec-picker wiring (still needed; this orchestration layer doesn't change runtime consumption of `.bin` files in codecs).
- **`multi_mlp_safety_default_path_2026-05-02.md`**: training shape (Default + RdSlope + MetricTimeMasked + safety chain). This doc orchestrates the bakes; that doc defines what gets baked.
- **`zentrain/INVERSION.md`**: Tier 1 was `refresh_features.py` (orchestration of feature refresh). This doc is the full lifecycle inversion.

## Open questions

1. **Worker bootstrap on vast.ai**: how do we get the workspace + dependencies onto a freshly-rented vast.ai instance? Docker image + `vastai-cli` launch (per substrate doc) — but the Docker image needs to be public (or the vast.ai instance needs to authenticate to a private registry). GHCR with a public-read service account is probably easiest.

2. **Cost of CPU-multiplier probe** — the time-calibration story (issue #56) needs each worker to run a fixed-image encode probe at startup. Adds ~30 s of bootstrap time. Worth it for slope correctness.

3. **Storage tier** — R2 free tier is 10 GB; paid is $0.015/GB-month. At 1 M source images × 1 KB metadata + 100 KB blob each = ~100 GB ≈ $1.50/month. Reasonable.

4. **Long-tail recovery** — if a worker dies during training (most expensive task), the 5-min stall reclaim is fine. But if the cron Cloudflare Worker itself fails, no reclamation happens. Mitigation: a second cron on a different platform (GitHub Actions cron polling R2) as a backstop.

5. **Schema evolution** — the task JSON schema will evolve. Workers need to gracefully handle "unknown task type" entries (skip; don't claim). Add a `schema_version` field to every task record.

6. **Audit / observability** — every task transition (`pending → in-flight → done | abandoned`) appends to `tasks/audit/<date>.jsonl`. Human-readable. Easy to reconstruct what happened.
