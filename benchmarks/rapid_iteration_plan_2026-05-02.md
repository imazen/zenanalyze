# Rapid-Iteration Picker Training Plan, 2026-05-02

Synthesis of 7 parallel read-only audits covering every component
that touches picker training across the imazen codec stack: zenjpeg,
zenwebp, zenavif, zenjxl, zenpicker, zenpredict, zensim, and
zentrain+zenanalyze. Audit reports preserved at
[`benchmarks/audits-2026-05-02/`](audits-2026-05-02/).

## TL;DR

The picker training cycle takes too long because we ship four
near-identical pipelines (one per codec) instead of one pipeline
that pulls four codecs. **Three of four trained pickers don't even
load at encode time** — we're optimizing models that affect 25 % of
production output. This plan reorganizes around three principles:

1. **zentrain owns the cycle end-to-end**; codecs become consumers
   of trained `.bin` files plus a tiny strongly-typed inference
   shim.
2. **Every artifact is content-addressed and reproducible by
   construction** — corpus → manifest → features → pareto →
   picker.bin, each with a deterministic hash.
3. **Every step is parallelizable across codecs** — a 4-codec
   ablation is one zentrain command, not four agents.

Quick wins this week (parallelize ablation, cache features, pin
reproducibility, wire idle pickers) cut the loop from days to hours.
Medium-term restructuring (centralize extractor + sweep harness +
canonical schema) cuts another 5–10 ×.

## What we found

### 1. Three of four trained pickers don't run in production

| Codec | Trained model | Loaded at encode? | Production gap |
|---|---|---|---|
| zenwebp | `picker.bin` v0 | ✅ YES (gated on `picker` cargo feature) | Active. |
| zenjpeg | regression weights baked as Rust consts | ❌ NO — issue #128 | Runtime uses fixed single-config LUT. **34–85 % byte savings unrealized.** |
| zenjxl | `zenjxl_hybrid_2026-05-01.bin` in `benchmarks/` | ❌ NO — wrapper has no `with_picker()` API | Training entirely decoupled from runtime. |
| zenavif | `rav1e_picker_v0_1.json` in `benchmarks/` | ❌ NO — `EncoderConfig::auto_tune()` is TODO | Same. |

**This is the single largest waste of time in the project.** Every
ablation, multi-seed retrain, dendrogram, and cross-codec aggregate
analysis we've run has been polishing models that have no production
output. Fixing this is the highest-leverage action available.

### 2. Pipeline duplication

Each codec ships:

- **Sweep harness** (~600 lines Rust): `zq_pareto_calibrate.rs`,
  `zenwebp_pareto.rs`, `predictor_sweep.rs`, `lossy_pareto_calibrate.rs`
- **Feature extractor** (~300 lines Rust): `*_features_replay.rs`,
  `extract_features*.rs` (replicated 4 ×)
- **Picker config** (~150–300 lines Python): `*_picker_config.py`,
  ~150 lines of identical scaffolding per codec
- **ZNPR v2 consumer boilerplate** (~200 lines Rust): zenpicker's
  audit flagged "90 % copy-paste" across 4 codec consumers

Total per-codec footprint: ~1500 lines of near-identical
scaffolding. Across 4 codecs: ~6000 lines of duplication. Cross-codec
analysis tooling has to read 4 different pareto schemas and reconcile
column drift.

### 3. Sweep wall times

| Codec | Sweep | Wall time | Resume? | Cores used |
|---|---|---:|---|---|
| zenavif | Phase 3 LHS | **50 hours** | No | 1 / 16 |
| zenwebp | Pareto v0.2 | **14–16 hours** | No | rayon, but per-image serial |
| zenjxl | Oracle | **9 hours** | No | rayon, sequential per-image |
| zenjpeg | Pareto | 7+ minutes | No | rayon |

Common patterns making these slow:

- **No resume on crash.** zenjxl's mid-run failures lose 6–8 hours.
- **`zensim` reference recomputed per encode.** Could cache per
  `(image, size_bucket)` → ~30 % per-codec encode-wall savings.
- **Features re-extracted every retrain** even when source images
  unchanged. Could cache per `(image, size, schema_hash)`.
- **Teachers retrained on identical inputs** during ablations. Could
  cache per `(features_hash, pareto_hash)`.
- **Single-threaded per-image** on a 16-core box. Phase 3 LHS
  projects to ~5–6 hr at 8× parallelism.

### 4. Reproducibility gaps

- **Seeds not pinned everywhere** — sweep harnesses, ablation runs,
  some training configs. Re-running gives slightly different
  results, especially on multi-seed-sensitive metrics.
- **No corpus content-hash** → silent data drift between sweeps.
  Filename conventions (`_2026-04-30_combined.tsv`) carry the
  freshness signal; no machine-checkable provenance.
- **No commit-hash** → trained-`.bin` attestation. We can't tell
  you "this picker came from corpus X at commit Y with seed Z" from
  the file.
- **Schema versioning ad-hoc.** Feature names not versioned in TSV.
  zenavif's `config_id` packed into 28 bits with implicit field
  layout (renaming a knob silently breaks old datasets).
  `CodecFamily` enum brittle to new variants (any 8th family
  requires lockstep rebake of all 4 codecs).

### 5. CI tells you nothing about picker quality

Every audit flagged this:

- No regression tests on picker predictions
- No bake determinism tests (Rust-side `bake_v2()` is determined;
  full Python pipeline isn't proven)
- No A/B eval harness (zenwebp has `dev/picker_ab_eval.rs` but it's
  manual only)
- No feature-extraction parity tests
- No schema-hash audit

A change that breaks a picker's argmin output ships. A schema bump
that makes old `.bin` files unloadable ships. A trained model with
subtly worse accuracy ships. The 6 CI gates on `train_hybrid.py`
(safety report, p99 shortfall, round-trip, size invariance, schema
hash, feature_bounds) are run **manually per-codec** — none of them
gate a release.

### 6. Visibility gaps

Tooling we don't have:

- **`zenpredict-diff a.bin b.bin`** — compare two bakes header /
  weights / metadata / byte-for-byte
- **`zenpredict-inspect --raw-scores`** — dump per-output score
  vector + confidence gap for a feature vector. Production codecs
  see only the chosen index today
- **Cross-codec feature-schema audit script** — load all four
  pickers' `.bin` files, extract feature schemas from metadata,
  report intersections / gaps / collisions
- **Per-image score dashboard for zensim** — drift detection between
  scorer versions. Today we see SROCC/KROCC aggregates only
- **Bake determinism CLI** — is the .bin reproducible from the same
  inputs?
- **Cross-codec metric heatmap** — argmin accuracy / mean overhead /
  p99 shortfall per codec, on one page

## The vision: zentrain becomes the lab

Three principles drive the redesign.

### Principle 1: Content addressing

Every artifact gets a deterministic content hash:

```
corpus_hash    = sha256(every source image, sorted by sha)
manifest_hash  = sha256(corpus_hash, sizes, content_classes)
features_hash  = sha256(manifest_hash, zenanalyze_schema_hash, extractor_commit)
pareto_hash    = sha256(manifest_hash, codec_version, sweep_config, sweep_commit)
picker_hash    = sha256(features_hash, pareto_hash, train_config, seed)
```

Filenames become `*.{hash}.parquet`. `.bin` headers carry their
input hashes as TLV metadata. **"Reproducibility" stops being a
discipline and becomes a property of the file system.**

### Principle 2: Caching by content hash

Identical inputs → identical hash → cache hit. The 80-retrain
multi-seed LOO that took 175 min spent most of its time reloading
the same 3.4 GB pareto file 80 times, then re-training the same
teachers. A content-addressed cache (`/tmp/zentrain-cache/{hash}/teacher.json`,
`/tmp/.../student.bin`) makes redundant work skip entirely.

Projected: multi-seed LOO 175 min → 30 min on the same hardware.

### Principle 3: Parallelism across codecs

The 4 codec sweeps don't depend on each other; they should run as
one orchestrated pool. A 4-codec ablation today is "spawn 4 agents
in parallel" — tomorrow it's `zentrain ablate --codecs all`.

## Phased plan

Effort is engineer-days at one engineer's pace. ROI is wall-time
savings on the typical "explore a feature → ablate → cull → bake"
cycle (today: 1–3 days end-to-end).

### Phase 0 — quick wins this week

| # | Item | Effort | Impact |
|--:|---|---:|---|
| 0.1 | Wire zenjpeg's regression weights to `Encoder::predict()` (issue #128) | 2 hr | **34–85 % byte savings online** |
| 0.2 | Wire zenjxl's `picker.bin` to `JxlEncoderConfig::with_picker()` | 4 hr | Largest single-codec gap closed |
| 0.3 | Wire zenavif's `EncoderConfig::auto_tune()` | 4 hr | Third codec online |
| 0.4 | Add `--seed` to every sweep harness | 1 hr × 4 | Reproducibility floor |
| 0.5 | Pin `train_hybrid.py --seed` by default | 30 min | Same |
| 0.6 | Compute corpus content hashes, embed in TSV/Parquet headers | 2 hr | Drift-detection floor |
| 0.7 | Resume checkpointing for sweep harnesses (`.offset` markers) | 4 hr × 4 | No more lost-hours-on-crash |
| 0.8 | `zensim` reference cache per (image, size_bucket) | 4 hr | ~30 % encode-wall savings |
| 0.9 | Parallel multi-seed LOO via `ProcessPoolExecutor` | 4 hr | 175 min → ~30 min on the 7950X |

**Total: ~3 person-days. Recovers 50 %+ of currently-wasted time.**

### Phase 1 — content-addressed pipeline (1 week)

| # | Item | Effort | Impact |
|--:|---|---:|---|
| 1.1 | Content-addressed teacher cache (sklearn HistGB pickled per cell + features_hash) | 6 hr | Ablation retrains skip teachers entirely |
| 1.2 | Parquet column-wise consumption in `train_hybrid.py` (vectorize, no per-row dicts) | 4 hr | Pareto load 54s → ~5s — the actual 36× we promised |
| 1.3 | Schema-hash everything (TSV + Parquet headers; `feat_*` stable id audit) | 1 day | Schema drift floor |
| 1.4 | Bake input-hash chain (corpus → manifest → features → pareto → picker carry hashes through) | 1 day | Reproducibility-by-construction |

**Total: ~3 person-days. Cuts iteration time another 5–10×.**

### Phase 2 — centralize the lab (1.5–2 weeks)

This is the [`zentrain/INVERSION.md`](../zentrain/INVERSION.md)
roadmap, adapted with what audits surfaced.

| # | Item | Effort | Impact |
|--:|---|---:|---|
| 2.1 | Centralized Rust extractor (`zenanalyze::extract_features_from_manifest`) replaces 4 per-codec extractors | 2 days | Tier 2 of INVERSION.md |
| 2.2 | Canonical pareto-row schema; every codec emits same columns | 1 day × 4 | Cross-codec analysis = one tool, not four |
| 2.3 | Picker config minimization (~30 lines vs current 150–300) | 4 hr × 4 | Tier 4 of INVERSION.md |
| 2.4 | Shared `zenpicker-harness` crate parameterizing sweep skeleton across codecs | 3 days | Replaces 600-line per-codec sweep harnesses |
| 2.5 | Shared `PickerRuntime` ZNPR v2 consumer (eliminates the 90 % copy-paste flagged by zenpicker audit) | 1 day | All 4 codecs use the same inference shim |

**Total: ~10 person-days. Permanent reduction of cycle complexity.**

### Phase 3 — quality + observability (1.5 weeks)

| # | Item | Effort | Impact |
|--:|---|---:|---|
| 3.1 | CI gate: bake determinism (same input → same bytes) | 4 hr | Reproducibility guard |
| 3.2 | CI gate: A/B eval on held-out corpus, fail Δbytes > +0.5 % | 1 day | Catches regression bugs |
| 3.3 | CI gate: cross-codec feature-schema audit | 4 hr | Catches schema drift |
| 3.4 | `zenpredict-diff` CLI: bake-vs-bake comparison | 6 hr | Decision auditing |
| 3.5 | `zenpredict-inspect --raw-scores` CLI: per-output scores + confidence gap | 6 hr | Production decision visibility |
| 3.6 | Cross-codec dashboard: argmin acc, mean overhead, p99 shortfall, byte savings, schema_hash, commit | 2 days | One page tells you which picker is healthy |
| 3.7 | Per-image zensim score dashboard for scorer drift | 1 day | Scorer drift visibility |

**Total: ~8 person-days. Quality floor we can build on.**

## Tier 1.5 (queued unless time appears)

Multi-codec ablation primitives that only make sense after Phase 2:

- **Unified multi-codec ablation driver** — one command runs Tier 0
  / 1.5 / 3 across all codecs in parallel; emits cross-codec
  feature-importance heatmap.
- **Incremental retrain** — drop one feature → reuse 95 % of previous
  computation. Multi-seed LOO drops from 30 min to ~3 min per
  feature.
- **Continuous feature monitoring** — when zenanalyze ships a new
  feature, automatically run cross-codec Tier 0 + multi-seed LOO,
  post results to the dashboard. New features prove themselves
  before anyone hand-curates `KEEP_FEATURES`.

## Tonight's deliverables

This commit lands:

1. **All 7 audit reports** preserved at
   [`benchmarks/audits-2026-05-02/`](audits-2026-05-02/) — 6
   markdown files (zenavif + zensim reconstructed from agent
   summaries; the others written directly by the agents).
2. **This synthesis plan** at this path.
3. **Cross-references** updated:
   - `zentrain/INVERSION.md` reframed as a subset of this plan.
   - `CLAUDE.md` (project) gets a pointer.

Tomorrow's session can pick any Phase 0 item and ship it in a day.

## Cross-references

- `zentrain/INVERSION.md` — tier-by-tier inversion roadmap (subset
  of Phase 2 of this plan)
- `zentrain/PRINCIPLES.md` — picker training data discipline
- `~/work/claudehints/topics/parquet-vs-tsv.md` — file format
- `benchmarks/feature_groups_cross_codec_2026-05-02.md` — the
  redundancy structure that makes incremental retrain feasible
- `benchmarks/loo_retrain_multiseed_2026-05-02.md` — variance
  estimates that drove the 175-min sweep this plan proposes to
  shrink to 30 min
- `benchmarks/all_time_best_features_2026-05-02.md` — features that
  every picker needs (the "minimum 15" is what a Phase-2 unified
  schema would default to)

## Open questions

- **Do we ship a single per-codec `.bin` or one meta-`.bin`?** The
  zenpicker audit warns that `CodecFamily` enum evolution requires
  lockstep release of all 4 codecs. A single meta-bin makes the
  blast radius worse; per-codec keeps blast radius local but
  duplicates inference shims. Decision affects Phase 2.5.
- **GPU acceleration for zensim?** zensim 22 ms/1080p is the
  bottleneck for large sweeps. A GPU port (CUDA / Metal / wgpu) is
  out of scope here but worth a separate audit when sweep wall is
  the bottleneck.
- **Do we want PyTorch or pure NumPy / scikit?** `train_hybrid.py`
  uses LeakyReLU MLP via PyTorch but most of the pipeline is
  NumPy / sklearn. PyTorch dependency is heavy; could move to JAX
  or pure-numpy MLP for a smaller dependency surface.
