# Multi-codec training — inversion of responsibility roadmap

The picker training pipeline has accumulated piecemeal duplication
across 4 codecs (zenjpeg / zenwebp / zenavif / zenjxl). Each codec
ships its own feature extractor, its own pareto sweep harness, its
own pareto-row schema, and its own picker config that duplicates
~150 lines of train-time scaffolding (paths, axis schema, regex
parsers). Cross-codec analysis has to read 4 different file layouts
and reconcile column drift.

Goal: **zentrain pulls more weight; codecs ship the minimum.**
Multi-codec training should be a single command, not a 4-agent
parallel-spawn pattern. New codecs should plug in with ~30 lines of
config, not ~150.

## Where we are (2026-05-02)

Per-codec ownership today:

| Layer | zenjpeg | zenwebp | zenavif | zenjxl |
|---|---|---|---|---|
| Pareto sweep harness | `dev/zq_pareto_calibrate.rs` | `dev/zenwebp_pareto.rs` | `benchmarks/dev/rav1e_phase1a_pareto.rs` | `lossy_pareto_calibrate.rs` (jxl-encoder) |
| Feature extractor | `dev/*_features_replay.rs` | (in worktree) | `examples/extract_features*.rs` | `examples/extract_features_for_picker.rs` |
| Picker config | `zenjpeg_picker_*.py` (~250 lines) | `zenwebp_picker_*.py` (~250 lines) | `zenavif_picker_config.py` (~200 lines) | `zenjxl_picker_config.py` (~250 lines) + adapter |
| Pareto schema | `image_path size_class config_id config_name q bytes zensim` | similar | similar + `effective_max_zensim` | `image_sha split content_class size_class w h cell_id <knobs> sample_idx bytes encode_ms ssim2 butteraugli` |

What zentrain owns today:

- `train_hybrid.py` — generic-shaped trainer (consumes via codec-config).
- `correlation_cleanup.py` — Tier 0 dendrogram cleanup.
- `student_permutation.py` — Tier 1.5 permutation importance.
- `feature_ablation.py` — Tier 3 LOO ablation.
- `feature_groups_*.py` (in zenanalyze/benchmarks/) — cross-codec dendrogram analysis.
- `loo_driver_*.py`, `loo_driver_multiseed_*.py` — paired-retrain LOO drivers.
- `tsv_to_parquet.py` — file-format conversion.

## Target inversion

### Tier 1: orchestration layer (zentrain) — landing in this commit

A single `zentrain/tools/refresh_features.py` orchestrator that:

- Iterates over a list of registered codecs.
- For each: invokes the codec's extractor binary, converts output to Parquet, updates picker config to point at the refreshed file.
- Single command: `python3 refresh_features.py [--codec=zenwebp|zenjpeg|...|all]`.

This pulls coordination weight away from "spawn 4 agents in parallel"
into one zentrain entrypoint. Per-codec extractors are still per-codec,
but invoked uniformly.

### Tier 2: centralized feature extractor (zenanalyze) — queued, next session

One generic Rust binary `zenanalyze/examples/extract_features_from_manifest.rs`:

- Inputs: `--manifest <TSV/Parquet>` (cols: `image_path size_class width height [content_class]`), `--output <Parquet>`, `--threads N`.
- Decodes each image (zenpng + format auto-detect for PNG / JPEG / etc.), resizes via zenresize to the recorded `(width, height)`, runs `zenanalyze::analyze_features_rgb8` with `FeatureSet::SUPPORTED`, emits Parquet with all `feat_*` columns.
- Replaces the 4 per-codec extractors. ~250 lines of Rust; deps already present (zenanalyze + image + zenpng + zenresize; need to add parquet/arrow as dev-deps).

After this lands:
- Each codec's `*_features_replay.rs` is deletable.
- Picker configs declare `MANIFEST = Path(".../manifest.tsv")` and `EXTRACTOR = "zenanalyze::extract_features_from_manifest"`.
- The orchestrator just runs the central extractor against each codec's manifest.

### Tier 3: pareto-sweep schema unification (codec side) — queued, deeper change

Standardize on a canonical pareto-row schema:

```
image_path  size_class  width  height  config_id  config_name
q  bytes  metric  metric_column_name  encode_ms  effective_max_zensim
```

Each codec's pareto harness emits this schema; `train_hybrid.py`
auto-detects `metric_column_name`. Today every codec's harness has
its own schema with bespoke columns; this is the biggest source of
cross-codec friction.

After this lands:
- One `train_hybrid.py` consumes any codec's pareto without per-codec
  schema branches.
- Cross-codec analysis tooling (correlation cleanup, dendrogram, LOO)
  works on any subset of codecs without per-codec adapters.

### Tier 4: picker-config minimization — queued, last

Codec config files reduce to ~30 lines:

```python
# zenwebp_picker_config.py  (target ~30 lines)
PARETO  = Path("benchmarks/zenwebp_pareto_2026-05-01_combined.parquet")
MANIFEST = Path("benchmarks/zenwebp_image_manifest.tsv")
KEEP_FEATURES = [...]  # the only codec-specific knob worth keeping
CATEGORICAL_AXES = ["method", "segments"]
SCALAR_AXES = ["sns_strength", "filter_strength", "filter_sharpness"]
FEATURE_GROUPS = {...}  # validator constraints
parse_config_name = "m{method}_seg{seg}_..."  # regex template, not full Python regex
```

Most of the per-codec bespoke Python disappears. The fixed scaffolding
(paths to OUT_JSON / OUT_LOG, ZQ_TARGETS, SCALAR_DISPLAY_RANGES) lives
in zentrain defaults; codecs override only when they actually differ.

## What's queued + when

| Tier | Effort | Dependency | Best time |
|---|---|---|---|
| 1: refresh_features.py orchestrator | ~30 min | none | **now** |
| 2: centralized Rust extractor | ~2 hours (Rust binary + tests + Parquet output + cargo deps) | none | next session |
| 3: pareto schema unification | ~1 day per codec (4 codec harnesses to refit) | independent of Tier 2 | as time allows |
| 4: picker config minimization | ~2 hours per codec | Tier 2 (extractor signature stable) | after Tier 2 |

## Cross-references

- `~/work/claudehints/topics/parquet-vs-tsv.md` — file format convention.
- `~/work/zen/zenanalyze/zentrain/PRINCIPLES.md` — picker training invariants.
- `~/work/zen/zenanalyze/CLAUDE.md` — project rules.
