# squintly project audit, 2026-05-02

## What it is

**squintly** is a browser-based psychovisual data collection tool that
gathers human perception judgments about image quality. Single Rust web
server (axum) with embedded vanilla TypeScript frontend (Vite,
@aspect/image-compare), designed primarily for mobile phones.

Users view image pairs and rate quality on a 4-point scale
(imperceptible → notice → dislike → hate) or choose which variant looks
better. Records detailed **viewing conditions** (device pixel ratio,
screen size, ambient light, calibration) alongside each judgment,
storing everything locally in SQLite.

**Purpose**: replace fixed-condition IQA datasets (KADID-10k, TID2013,
CID22) with labeled data that captures **how viewing conditions affect
perceptual quality**. zensim plateaus at SROCC 0.82 on fixed-condition
datasets; squintly's condition-aware data lets it condition on DPR,
viewing distance, intrinsic-to-device pixel ratio, and ambient light to
unlock better generalization.

## Architecture

```
[Phone / desktop browser]
  │
  ▼
Calibration (DPR, screen size, ambient light, color space)
  │
  ▼
Trial loop:
  ├── /api/trial/next → image pair (codec.q40 vs codec.q60)
  ├── User rates (1-4) or picks (left/right)
  └── /api/trial/response → recorded
  │
  ▼
SQLite v0.1 (Postgres v0.2 for multi-instance)
  │
  ▼
Bradley-Terry fit per (source, condition_bucket)
  │
  ▼
Exports (zenanalyze-compatible):
  ├── /api/export/pareto.tsv      — BT-derived θ subjective quality 0-100
  ├── /api/export/thresholds.tsv  — q_threshold(condition_bucket) (v0.2 stub)
  └── /api/export/responses.tsv   — raw trial data
```

## Role in picker-training ecosystem

**Producer of ground-truth labels**:

1. **Source of human judgments**: pairwise preferences + threshold
   estimates across viewing conditions.
2. **Training labels for zensim/zentrain**: `pareto.tsv` replaces
   zensim's generic column during zenanalyze training. Per-image θ
   value reflects subjective quality on a Bradley-Terry scale.
3. **Threshold oracle for the picker**: `thresholds.tsv` feeds picker
   optimization. Reframes "minimize bytes subject to zensim ≥ T" into
   "minimize bytes subject to q ≥ q_threshold(viewing_conditions)."
4. **NOT a consumer of pickers**: squintly doesn't call trained
   pickers or observe their decisions. It's orthogonal to codec
   optimization; strictly label-generation infrastructure.

## Data plane

**Storage**: SQLite v0.1; schema in `migrations/0001_init.sql`. Tables:
`observers`, `sessions`, `trials`, `responses`, `staircases`. Postgres
target for v0.2 (multi-instance).

**coefficient integration**: two implementations:
- `HttpCoefficient` — talks to coefficient viewer at
  `SQUINTLY_COEFFICIENT_HTTP`
- `FsCoefficient` — reads SplitStore `meta/` + `blobs/` directly
- Manifest cached on startup; refreshes every 5 min or on demand

**Trial sampling** (`src/sampling.rs`):
- Weighted toward threshold trials early (`p_single ≈ 0.65`)
- Source-inverse-weighted by coverage
- Quality-biased toward q5–q40 per source-informing-sweep rule
- Trivial-pair filtering (same codec q-gap ≥ grid/2; cross-codec bytes
  ratio > 4×)

**Bradley-Terry fit** (`src/bt.rs`):
- Per (source, condition_bucket): fit θ and ν via L-BFGS with
  monotonicity constraints
- Anchored at `θ_reference = 0`; scaled to 0-100 range

## Compute plane

- Single Rust binary (axum) embedding TS frontend via `rust-embed`
- Static HTML/JS, no Node runtime
- Railway-ready (Dockerfile included)
- Single-instance bottleneck: SQLite + synchronous export queries

## API / interface surface

**Three TSV export endpoints** (zenanalyze-compatible):
- `/api/export/pareto.tsv` — BT-derived θ subjective quality
- `/api/export/thresholds.tsv` — q_threshold per
  (codec, viewing_condition_bucket) (NOT YET IMPLEMENTED, v0.2)
- `/api/export/responses.tsv` — raw trial data for custom fitting

**Trial API**:
- `/api/trial/next` — sampling.rs picks pair
- `/api/trial/response` — record judgment
- `/api/manifest/refresh` — invalidate manifest cache

## Existing integrations

- **coefficient**: HttpCoefficient + FsCoefficient clients. Squintly
  reads coefficient's manifest to know what (source, codec, q) tuples
  exist.
- **zensim / butteraugli / ssim2**: not directly called. Squintly
  produces ground truth that zensim trains AGAINST.
- **codecs**: not directly. Squintly references codec.qN names from
  coefficient's manifest.

## Pain points

1. **Manifest staleness** — no auto-refresh when coefficient changes;
   requires manual `POST /api/manifest/refresh` or 5-min cache TTL.
2. **Condition undersampling** — no adaptive trial selection yet (v0.2);
   random pair sampling can starve low-N condition buckets for days.
3. **Single-instance bottleneck** — SQLite + synchronous export queries
   don't scale to millions of trials without Postgres + async batch
   export.
4. **No threshold model deployed** — thresholds computed per-session via
   staircase (Levitt 1971) but not aggregated across sessions offline.
   `thresholds.tsv` export is a stub.
5. **Codec-name sync** — if coefficient adds new codecs or quality
   scales, squintly's `sampling.rs::codec_browser_family()` must be
   manually updated.

## Top 3 questions for zentrain ↔ squintly orchestration

1. **Schema coupling & update cadence**
   - Squintly exports `config_name` as `{codec}.q{quality}` (e.g.,
     `mozjpeg.q40`). Does zentrain expect this format, or does it
     parameterize codec/quality naming? If zentrain gains new codecs or
     quality scales, does squintly's codec-name registry auto-track?
   - **Pain**: manifest changes faster than squintly's codec list
     risks training on stale config names.

2. **Condition-bucket alignment**
   - Squintly bucketing:
     `dpr{1|2|3}_dist{20|30|50|70|150|250}_{dark|dim|bright|outdoors}_{srgb|p3|rec2020}`
   - Does zentrain expect this granularity, or aggregate/override?
   - If zentrain wants finer buckets (dist_15/25/35 instead of
     20/30/50), can sampling strategy adapt?
   - **Pain**: low-N buckets → wide CIs on threshold estimates.

3. **Scale from v0.1 (local SQLite) to cloud (million-file jobs)**
   - Currently single-instance SQLite, one coefficient store. Bottleneck
     is coefficient image-serving speed.
   - For cloud-scale picker training (1M+ files), need: (a)
     multi-instance squintly with Postgres, (b) coefficient cached/CDN'd,
     (c) async/batch trial export into a job queue?
   - **Pain**: no parallelization story yet. Synchronous full-table
     scans on export.

## Test coverage

- Smoke test (`tests/smoke.rs`) — fake coefficient, full session →
  trial → response → export loop
- E2E suite in `web/e2e/` with Playwright (10 specs covering API, auth,
  calibration, trial loop, codec filtering)
- No load tests; no multi-observer concurrency tests

## What would make zentrain ↔ squintly orchestration cleaner

1. **`thresholds.tsv` export** — implement the v0.2 deliverable so
   zentrain has condition-aware quality targets.
2. **Typed condition_bucket schema** shared across squintly + zentrain.
3. **Auto-manifest-refresh** triggered by coefficient SourceRecord
   add-events (Firestore listener?).
4. **Postgres-backed multi-instance squintly** for scale-out trial
   collection on phones at population scale.

## Files of interest

- `src/main.rs` — entrypoint, CLI, axum router
- `src/export.rs` — TSV streaming (pareto, thresholds, responses)
- `src/sampling.rs` — trial pair selection
- `src/coefficient.rs` — coefficient client (Http + Fs impls)
- `src/bt.rs` — Bradley-Terry fitting
- `SPEC.md` — full design (read first)
- `docs/methodology.md` — methodology rationale
- `migrations/0001_init.sql` — SQLite schema
