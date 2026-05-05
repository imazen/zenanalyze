# zenanalyze — Claude project guide

## API stability — non-negotiable

**There will never be a 0.2.x.** The crate ships under 0.1.x forever
(or until a 1.0). Every change must fit within the additive contract:

- New `pub fn` / `pub struct` / `pub mod`: OK.
- New variant on a `#[non_exhaustive]` enum (`AnalysisFeature`,
  `FeatureValue`, `AnalyzeError`, etc.): OK.
- New field on a struct that's already private or `#[non_exhaustive]`:
  OK if the existing public constructors stay valid.
- Renaming, removing, or changing the signature of any existing public
  item: **NOT OK.** Even small changes ("rename a parameter for
  clarity", "tighten a return type") are forbidden once the item has
  shipped. Add a parallel item if you need different shape — see
  `try_analyze_features_rgb8` next to `analyze_features_rgb8`.
- Numeric / behavioural drift on existing features in 0.1.x patches is
  expected (the crate-level threshold contract spells it out) — but the
  *signatures* don't move.

If a future need can't be solved additively under 0.1.x, pause and
flag it to the user. Do not propose "ship it in the next major" —
there is no next major.

## Allocation contract

Today every internal allocation is infallible (`vec!` / `Box::new`).
The plan to flip to fallible (`Vec::try_reserve`, etc.) does not
require any signature change — `try_analyze_features_rgb8` and
`AnalyzeError::OutOfMemory { bytes_requested }` are already in place
to surface the OOM. When fallible internals land, no caller has to
recompile.

## Threshold contract

Numeric thresholds and normalisation scales drift during 0.1.x as
features get refined. Downstream consumers that compile-in fitted
models pin to a specific patch and re-validate when they bump.
Documented at the crate-level docstring in `src/lib.rs` and in the
README.

## Tier architecture quick reference

Five passes, gated by the requested `FeatureSet`:

- **Tier 1** — stripe-sampled RGB8 (variance, edges, chroma, uniformity, grayscale).
- **Tier 2** — full-image 3-row sliding window over RGB8 (per-axis Cb/Cr sharpness).
- **Tier 3** — sampled 8×8 DCT blocks on RGB8 (DCT energy, entropy, AQ map, noise floor, line-art, gradient fraction, patch fraction).
- **Palette** — full-image RGB8 (distinct color bins, indexed-palette signals).
- **Alpha** — stride-sampled, **reads source bytes directly** (no RowStream).
- **`tier_depth`** — stride-sampled, **reads source bytes directly** (HDR / wide-gamut / bit-depth signals; HDR signal would not survive RowConverter narrowing).

The Native-vs-Convert decision in `RowStream::new` only applies to
Tier 1/2/3 + Palette. Alpha and `tier_depth` always read the source.

## Benchmark + ablation file format — Parquet, not TSV (>50 MB)

Pareto sweeps, ablation outputs, multi-seed LOO retrain inputs —
anything tabular in `benchmarks/` that's bigger than ~50 MB SHIPS AS
PARQUET. Compare on real data (zenwebp pareto, 21.8 M rows, 3.4 GB
TSV):

| Stage | csv.DictReader | Parquet (zstd-3) | Speedup |
|---|--:|--:|--:|
| Pure file read+parse | 68 s | 1.9 s | **36×** |
| End-to-end `load_pareto` | 68 s | 54 s | 1.3× |
| Disk size | 3.4 GB | 0.21 GB | **16×** |

The end-to-end gap reflects Python per-row dict construction in
`load_pareto`'s downstream code. The 36× headline applies once
the consumer is refactored to use Arrow columns directly (queued).
**Disk savings and cold-cache wins are unconditional today.**

`zentrain/tools/train_hybrid.py`'s `_read_table_columns()` helper
auto-detects format by `.parquet` / `.pq` suffix; existing TSV
configs keep working unchanged. Convert with
`benchmarks/tsv_to_parquet.py`. Picker configs flip
`PARETO = Path(".../foo.tsv")` → `Path(".../foo.parquet")`.

Full guidance: `~/work/claudehints/topics/parquet-vs-tsv.md`.

## Multi-codec training is moving toward zentrain orchestration

Per-codec piecemeal extraction (4 codec-specific binaries, 4 picker
configs duplicating ~150 lines of scaffolding) is being replaced by
zentrain-owned orchestration. Tier 1 (single-command refresh of all
codecs' features files) shipped 2026-05-02 as
`zentrain/tools/refresh_features.py`. Tiers 2–4 (centralized Rust
extractor in zenanalyze, pareto-schema unification, picker-config
minimization) are queued in `zentrain/INVERSION.md`. Read that
roadmap before adding a new codec; new codecs should plug into the
inversion target, not the legacy piecemeal pattern.

## Don't

- Don't propose 0.2.x.
- Don't change a published function signature, even to "improve" it.
  Add a parallel `try_*` / `with_*` / `_into` variant if needed.
- Don't add new `expect()` / `unwrap()` to public entries that took
  untrusted input. The fallible parallels exist for a reason.
- Don't bake content-class assumptions into the analyzer. The job is
  to surface signals; the consumer (codec orchestrator) decides what
  to do with them.
- Don't write multi-GB TSVs to `benchmarks/` — Parquet (zstd) is
  16× smaller AND 36× faster to load. Use `tsv_to_parquet.py`.

## Picker training discipline (added 2026-05-04)

Picker work has shipped two key infrastructure additions. Read these
before training a new picker tier.

### `train_hybrid.py --safety-default-cell` flag

Per-row mask in `build_dataset` that hides any alternative cell whose
min-bytes config either takes more than `--safety-speed-tol` (default
1.05) times the default cell's encode time, OR fails to deliver
`(1 - --safety-bytes-min-gain)` (default 0.99) bytes savings. Forces
the picker to default unless an alternative is meaningfully smaller
AND not slower.

Result on zenjxl v0.6: student val argmin_acc jumped from 51% (v0.5
no mask) to 79%; train→val gap dropped from +6.0pp to +1.89pp.

The mask is OFF by default. Pass `--safety-default-cell <CELL_LABEL>`
matching `cell_label_from_key`'s output (e.g. `effort7`).

### Distance-aware A/B harness

`tools/holdout_ab_lookup_jxl.py` now queries the picker at the zq
the **default cell actually achieves at each distance**, not a dummy
zq=75. The previous v0.5 harness was structurally broken — a picker
trained for zq=75 was being graded against bands ranging zq~99 to
zq~40. Fix shipped 2026-05-04.

### Classifier picker prototypes

The MLP-regress-bytes-then-argmin chain is fragile under safety
masking. A small softmax-classifier MLP over `(image features ⊕
log(distance))` produces cleaner picks. Prototypes at:

- `tools/picker_v06_mlp_prototype.py` — PyTorch MLP, SHIP verdict on
  v05c data (-1.51% bytes / +0.15pp zensim / -5.93% encode time)
- `tools/picker_v06_classifier_prototype.py` — HistGradientBoosting
  variant for ablation
- `tools/oracle_v05c_zenjxl.py` — upper-bound oracle (only 9.1% of
  v05c cells have a strict speed-safe win available)

Classifier prototypes don't yet bake through `bake_picker.py` — that
expects bytes-log regression. Three productionization paths
documented at `docs/jxl-picker-v06-summary-2026-05-04.md` (option B
recommended: add `--head-mode classifier` to trainer + baker + 1
runtime branch).

### Investigation docs

- `docs/jxl-picker-investigation-2026-05-04.md` — v0.5 HOLD root cause
- `docs/jxl-picker-v06-summary-2026-05-04.md` — v0.6 path forward
  + productionization options A/B/C

### Sweep + bug status (2026-05-04+)

- v05c sweep on R2 `s3://zentrain/sweep-v05c-2026-05-04/` (no butteraugli)
- v06 sweep on R2 `s3://zentrain/sweep-v06-2026-05-04/` (in flight; CPU
  metrics, expanded JXL knobs via zen-metrics 0.6.0; chunks land at
  `zenjxl/<chunk_id>.tsv` with butteraugli columns)
- 500 representative images clustered from v05c via k-means on
  zenanalyze features (script lives at `/tmp/cluster_v06_subset.py`
  during the session — re-cluster as needed)
- Decoder bug filed: `imazen/zenjxl-decoder#15` — effort=9 +
  distance ≤ 0.5 + screen content produces files jxl-oxide accepts
  but zenjxl-decoder rejects (decoder bug, not encoder)
