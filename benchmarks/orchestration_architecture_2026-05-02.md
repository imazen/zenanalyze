# Orchestration architecture: zentrain ↔ coefficient ↔ squintly ↔ codecs

Synthesis of three additional read-only audits — coefficient (cloud
codec benchmarking), squintly (psychovisual ground-truth collection),
and coefficient-zenjpeg-opt (concrete integration shim) — combined
with the seven prior single-machine audits.

## TL;DR

The picker training pipeline is fighting a problem that's already
solved at a different layer. **coefficient already encodes millions of
images cross-codec on GCP Batch with full provenance; all 4
zen-codecs are integrated.** **squintly already collects subjective
human labels with viewing-condition tagging and exports
zenanalyze-compatible TSVs.** The single-machine "centralize the lab
inside zentrain" plan from earlier this session is largely **obsolete
or absorbed** — the lab is coefficient. zentrain's job shrinks
substantially: it consumes coefficient outputs, consumes squintly
outputs, and trains pickers. The remaining work is **five specific
bridges**, not full re-engineering.

This reframes the rapid-iteration plan: many "Phase 2 centralize the
lab" deliverables disappear; new "Phase A bridges" replace them.

## The actual layer map

| Layer | What | Where it lives | Status |
|---|---|---|---|
| Source corpus | PNG images, hashes, manifests | coefficient `SourceRecord` | shipped |
| Encoding (cross-codec, scaled) | All codecs × configs × qualities | coefficient `EncodingRecord` + GCP Batch | shipped, all 4 codecs integrated |
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

## What we don't need to build (was queued in rapid_iteration_plan)

The single-machine plan's Phase 1+2 ("centralize the lab") is mostly
obsolete:

| Item | Status | Why |
|---|---|---|
| Centralized Rust extractor in zenanalyze | ❌ obsolete | coefficient emits `source_features.json` |
| Canonical pareto-row schema per codec | ❌ obsolete | coefficient defines this |
| Shared `zenpicker-harness` crate | ❌ obsolete | coefficient's `Codec` trait IS that |
| Per-codec sweep parallelism (50-hr Phase 3 LHS) | ❌ obsolete | GCP Batch with 1000+ workers |
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

### Bridge 4: mode switch (local vs GCP Batch)

**Problem**: zentrain runs `train_hybrid.py` on the calling machine.
With coefficient as the encode source, the encoding can be local OR
on GCP Batch. zentrain doesn't know how to wait on a coefficient
batch job.

**Solution**: `zentrain orchestrate-sweep --coefficient-mode {local,batch}`.
- `local`: runs coefficient's local Worker, waits for completion, then
  trains.
- `batch`: submits to coefficient GCP Batch, polls Firestore, fetches
  outputs from GCS when done, then trains.

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
| B.2 | zentrain awaits coefficient GCP Batch job | 1 day | Async sweep |
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

1. **Storage for picker.bin distribution** — coefficient `analysis/`
   vs codec crate vs sidecar daemon? Audit recommends hybrid (embedded
   default + runtime override + optional updates).
2. **Squintly v0.2 timeline** — when does `thresholds.tsv` ship? Block
   on this for Phase B.4.
3. **GCP Batch costs** — typical training cycle (1 M images × 4 codecs
   × 30 quality levels × N configs) cost envelope? Not surveyed in
   coefficient audit; need cost-per-encode estimate before committing
   to batch-by-default.
4. **Where does zensim training live in this model?** Still standalone
   (`zensim-validate` Rust CLI) or absorbed by coefficient + zentrain?
   zensim audit suggested it's structurally separate today.
5. **CodecFamily enum evolution** — adding a 5th/6th codec family
   requires lockstep meta-picker rebake (zenpicker audit). Does the
   bridge-1 exporter need to grow as new codecs land?

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
