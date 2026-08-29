# zenanalyze — Claude project guide

## API stability — 0.2.x policy (revised 2026-05-17)

**zenanalyze is now on 0.2.x — breaking changes to library APIs are
permitted.** The earlier "0.1.x forever" rule was retired when the
crate's pre-1.0 surface needed to evolve.

**USER DIRECTIVE 2026-07-19: `zenanalyze-api` is the unchanging contract.**
The zenanalyze-api crate (0.1.x, published on crates.io) is FROZEN — never
break its public surface; multi-version unification depends on every
consumer speaking the same zenanalyze-api (see Cargo.toml notes: publish
zenanalyze-api before zenanalyze when releasing with `api` enabled).
zenanalyze itself (0.2.x+) may change freely behind it. Additive growth of
zenanalyze-api (new `pub` items, new variants on its `#[non_exhaustive]`
enums) is allowed and is how the contract becomes sufficient; *breaking* it
is not.

**USER DIRECTIVE 2026-08-28: `zenanalyze-api` is the SOLE contract AND
intermediary — "so different zenanalyze versions can compile together."**

> A codec crate's **library code** depends on `zenanalyze-api` and nothing
> else from the zenanalyze family. It receives values as a
> `zenanalyze_api::Offer`, or extracts them through a
> `&dyn zenanalyze_api::FeatureProvider` the host injects. It never names
> `zenanalyze::…`.

Direct `zenanalyze` is legitimate in two roles only: the **host /
orchestrator** that chooses the version and runs the pass, and **dev
tooling** (`dev/` binaries, `examples/`, `benches/`, sweep and training
extractors) that isn't linked into the product graph.

Three things to get right, in the order they bite:

1. **Depend on the contract by crates.io version, never by git rev.** A
   registry dep and a git dep are different Cargo sources, and two git deps
   at *different revs* are different sources too — either way you get two
   `Offer` types that don't interconvert ("expected `Offer`, found
   `Offer`"). Unreleased contract changes go in **one** workspace-root
   `[patch.crates-io]`, which rewrites the registry entry everywhere and
   keeps unification.
2. **`Select::Features` for models, `Select::Names` for heuristics.** A
   compiled model's coefficients were fit against one code version per
   column, so a drift MUST miss — that's `Features`. Threshold heuristics,
   classifiers, diagnostics and bulk export use `Names` (bare name, any
   version), which is what lets them name features without naming a
   zenanalyze version.
3. **If a consumer can't migrate, extend the contract — additively — rather
   than reporting it blocked.** That is what `Select::Names`,
   `FeatureProvider`, `OwnedCatalog` and `ProviderError` exist for.

Full rules + audit recipe: `docs/sole-contract.md`. Mechanics and compiled
examples: `zenanalyze-api/README.md`.

### Rust library surface (semver-governed)

Standard 0.x semver applies:
- Breaking changes (renames, signature changes, removed items) bump
  the **minor** (0.2 → 0.3).
- Additive changes (new pub items, new enum variants on
  `#[non_exhaustive]` types) bump the **patch** (0.2.0 → 0.2.1).
- Behavioural / numeric drift is allowed within a minor.

Before any breaking change: run `cargo semver-checks` and document
the break in CHANGELOG.md's `[Unreleased]` section under
`### QUEUED BREAKING CHANGES`. Batch breaks into the next minor bump.

### Binaries — NOT part of the semver surface

The `zenpredict-bake`, `zenpredict-inspect`, `zenpredict` (and
future) command-line binaries are **tools, not API**. Their CLI
flags, subcommand names, output format, and even binary names may
be renamed, restructured, or removed at will. Downstream scripts
that pin to a specific CLI invocation are responsible for their own
pinning (commit-sha or version-tag); the crate makes no stability
promise on binary surfaces.

When restructuring binaries: update docstrings + any benchmarks /
examples / scripts that invoke them; note the change in
CHANGELOG.md but don't gate it on a major bump.

### Historical migration: 0.1 → 0.2

Pre-0.2, the crate enforced "additive only" — see `try_*` parallel
constructors like `try_analyze_features_rgb8` next to
`analyze_features_rgb8`. Those parallel pairs may now be collapsed
in the next breaking pass; flag for the user before removing
either side.

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

Six passes, gated by the requested `FeatureSet`:

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

## NO DUPLICATE IMPLEMENTATIONS — one owner per task (2026-07-15, user directive)

**Every task below has exactly ONE canonical implementation. Re-implementing
any of them — in Python, in a second Rust site, in a new `vNN_*_train.py`,
anywhere — is PROHIBITED.** If the owner can't do what you need, **extend the
owner**. Companion to the identical rule in `~/work/zen/zensim/CLAUDE.md`.

| Task | THE owner | Never |
|---|---|---|
| IQA stats (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE) | **`zenstats`** (`zenmetrics/crates/zenstats`) — Rust: dep on it; Python: `scripts/lib/zen_stats.py` | `scipy.stats.spearmanr`, a hand-rolled `srocc`, any private stat math |
| Parquet feature/pareto loading (Python) | `zentrain/tools/_picker_lib.py` — `load_pareto_raw` / `load_features_raw` | `pd.read_parquet` / `pq.read_table` in a new tool |
| Parquet loading (Rust) | `zenpicker-train/src/{parquet_input,pareto_dataset}.rs` | a second reader |
| Picker training | `zenpicker-train` (Rust) / `zentrain/tools/train_hybrid.py` | a new `vNN_*_picker_train.py` |
| zensim-metric MLP training | **zensim's** `zensim-validate/src/bin/zensim_mlp_train.rs` | `zentrain/tools/zensim_metric_train.py` (a Python fork of another repo's owner) |
| Bake bytes / ZNPR | `zenpredict-bake` (`zenpredict bake`/`inspect`/`repack`) | any other emitter |
| Train/val/test split | `zenmetrics/scripts/picker/origin_split.py` (`split_of()`) | a seeded shuffle — per-rendition shuffling leaks scale |

**Python is not banned — DUPLICATION is.** Python is correct where it IS the
owner (`_picker_lib`, corpus building, plots). The test is not "what language"
but "does this task already have an owner".

### Why this rule exists here specifically

This repo shows the failure mode most clearly. **Extraction is not migration:**
`_picker_lib.load_features_raw` exists and is the declared owner, yet
`load_features_raw` adoption went from 7-of-25 call sites to ~15-of-35 — the
lib landed, the forks kept coming, so the ratio barely moved while the absolute
count grew. Meanwhile `tools/v15_zenjpeg_picker_train.py`,
`v15_metapicker_train.py`, `v0_2_zenjpeg_picker_train.py`,
`v14_metapicker_train.py`, `v12_metapicker_train.py`,
`picker_v06_mlp_prototype.py`, `v10_router_mlp_train.py`,
`v06_zenjxl_picker_mlp_train.py` are one copy-the-last-one chain, all frozen
2026-05-26. Each was "just a variant" at the time.

Lines 101-111 below *describe* this duplication ("4 codec-specific binaries, 4
picker configs duplicating ~150 lines of scaffolding") and queue the fix in
`zentrain/INVERSION.md`. **Queueing is the bug.** A duplicate found is a
duplicate removed — same commit if it's dead, next commit if something still
calls it.

### First deletions (both queued since 2026-05-26, still un-migrated)

The 2026-05-26 IQA consolidation deferred four cross-repo callers. Two are
here, still on bare `scipy` seven weeks later:

- `zentrain/tools/zensim_metric_train.py:467` — audit item #10. Zero
  `zen_stats` imports. It is also a **1,078-line Python fork of zensim's Rust
  `zensim_mlp_train`**, i.e. duplicated across a repo boundary.
- `zentrain/tools/correlation_cleanup.py:161` — audit item #11.

### The one exception: a gated mirror

A second implementation is legitimate **only** with a measured engineering
reason AND a test holding it bit-exact against the owner. This repo has the
model example: `zenpicker-train/src/picker_eval.rs:97`'s
`pwrc_sa_st_auc_lowmem` (O(n²)→O(1) memory) is gated by
`pwrc_lowmem_matches_canonical_exactly`. Without that test it is not a mirror,
it is a fork with a good story.

The enforcement pattern worth copying is zenmetrics' `origin_split`:
`train_hybrid` **hard-errors** if `origin_split` isn't importable rather than
falling back to a leaky split. Fail loud; never silently substitute.

Audit of record: `~/work/zen/zensim/benchmarks/duplication_audit_2026-07-15.md`.
Prior art: `zensim/benchmarks/iqa_stats_consolidation_2026-05-26.md`.

## Don't

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

**Before scoping a picker, read the feature/knob ABLATION design:**
`../zenmetrics/docs/ML_FRAMEWORK_AND_PICKER_ABLATION_2026-06-09.md` (in the
zenmetrics repo). Key points: feature relevance is a conditional
**features × knobs × zq-band × mode** matrix — there is no global feature
ranking; ablate **inputs by redundancy cluster** (the `benchmarks/feature_groups_*`
ρ≥0.95 dendrogram) with permutation/LOGO importance, not gain; ablate **outputs**
by RD-spread + content-dependence and **pin knobs that don't earn a head**; use a
GBDT (forust) as the feature-selection instrument (per-knob importance + RD spread);
do **output ablation first** (bigger lever). Stratified-sampling corpus selector:
`zenpicker-train/src/bin/cluster_features.rs` (linfa k-means, commit `96ccf86`).
That doc also holds the candle/burn/linfa 3-layer verdict and the GBDT-teacher /
GD-MLP-student framing (GBDT model 975 KB / 109 KB gz vs ~27 KB ZNPR MLP →
distill teacher→student rather than ship trees).

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

### HISTORICAL (2026-05) — Classifier picker prototypes

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

### HISTORICAL (2026-05) — Sweep + bug status (2026-05-04+)

- v05c sweep on R2 `s3://zentrain/sweep-v05c-2026-05-04/` (no butteraugli)
- v06 sweep on R2 `s3://zentrain/sweep-v06-2026-05-04/` (in flight; CPU
  metrics, expanded JXL knobs via zenmetrics 0.6.0; chunks land at
  `zenjxl/<chunk_id>.tsv` with butteraugli columns)
- 500 representative images clustered from v05c via k-means on
  zenanalyze features (clustering script was under /tmp — wiped, /tmp
  is banned; re-cluster as needed)
- Decoder bug filed: `imazen/zenjxl-decoder#15` — effort=9 +
  distance ≤ 0.5 + screen content produces files jxl-oxide accepts
  but zenjxl-decoder rejects (decoder bug, not encoder)

## Known Bugs

_None currently open._

Resolved 2026-06-20 (`edge_slope_stdev` cross-platform divergence):
- The SIMD edge kernel computed the gradient magnitude with the hardware
  `rsqrt_approx()` (different-precision instruction per backend) while the scalar
  tail used exact `sqrt()`, so `edge_slope_stdev` diverged ~6 % across arches.
  **Fixed** by `simd_math::rsqrt_stable!` (software bit-trick seed + Newton in pure
  f32 mul/sub — bit-identical on every backend) in both the SIMD body and the
  scalar tail. CI probe (`rsqrt-probe` job) confirms `rsqrt_stable` is byte-identical
  on x86/ARM while both magetypes `rsqrt_approx` AND the Newton-refined `rsqrt`
  diverge. Diagnosis: `docs/feature-cross-platform-divergence-2026-06-20.md`.

## Investigation Notes

- **Feature versioning is platform-aware by construction (2026-06-20).** The
  `versioning` golden tripwire surfaced that 9 SIMD-reduced statistical features
  have per-SIMD-tier value divergence (6 at <0.3 % from f64 reduction order; 3
  cancellation-prone outliers up to ~11 %). The version *hash* is text-derived
  (platform-independent); `golden_is_stable` (live re-extraction) is therefore a
  **reference-platform (x86-64) tripwire** — asserted by the `golden-reference` CI
  job, `--skip`ped in the portable matrix. `REL_TOLERANCE`=0.5 % + 3 per-feature
  overrides are sized from the measured CI spread, not guessed.

Resolved 2026-06-19 (P0 CI-integrity pass):
- The `fit_yeo_johnson` golden-section test "failure" was a **wrong test
  expectation**, not a fitter bug. λ≈−0.843 IS the correct Yeo-Johnson MLE for
  log-normal data: YJ λ=0 is `log(x+1)`, NOT `log(x)`, so the Box-Cox fact
  "log-normal ⇒ λ≈0" does not carry over. Verified over a fine grid — the YJ
  profile log-likelihood is sharply peaked at −0.843 (`ll(−0.843)≈−36.3` vs
  `ll(0)≈−144.7`). The fitter was right; the test now asserts ≈−0.843 (renamed
  `golden_search_finds_yj_optimum_on_lognormal`).
- `zenpredict-bake` is now in CI (`.github/workflows/ci.yml` job
  `zenpredict-bake`): fmt + clippy `--all-features --all-targets -D warnings` +
  test (default + `fit-yj`, which builds the MLE bin and runs its test). The
  bench `rand 0.9` deprecations (`gen_range`→`random_range`) are fixed and the
  accumulated clippy debt was cleared so `-D warnings` passes.

## KADIS-700k dataset (zensim 2026-06-30; GPU-metrics 2026-07-01)

700,000 distorted-image cells — 140k KADIS pristine references × 1 `dist_type_1` × 5 severity
levels, each with a 372-D zensim feature vector. **The `source_features/` sidecar — a
per-reference snapshot of each *undistorted* image — is produced by THIS crate**
(`analyze_features_rgb8` with `FeatureSet::SUPPORTED`), one row per source, joinable to the cell
rows of EITHER canonical by `source_filename`. Two canonical variants (same 700k cells, same
`source_id` split key):

- **★ GPU-metrics canonical (2026-07-01) — current, richest.**
  `s3://zentrain/kadis-700k-gpu/canonical/kadis700k_canonical_gpu_2026-07-01.parquet`
  (700k×387, ~936 MB zstd, 0 nulls; sha256 `c9a6fd56…`). **7 perceptual scores** —
  `score_{zensim,ssim2,butteraugli_max,butteraugli_pnorm3,iwssim,dssim}_gpu` + `score_cvvdp_cpu_imazen_v0_1_0`
  — plus `distorted_url` (a persisted distorted PNG per cell → rescore-from-links), on top of the
  372-D `feat_*` + shared keys. Sidecars `s3://zentrain/kadis-700k-gpu/{omni,zensim_features,pairs}/`
  + `distorted/<chunk>/*.png`. (No `source_features/` sidecar of its own — join the CPU variant's
  `kadis-700k/source_features/` by `source_filename`.)
- **zensim-only canonical (2026-06-30) — earlier variant.**
  `s3://zentrain/kadis-700k/canonical/kadis700k_canonical_2026-06-30.parquet` (700k×380, ~906 MB
  zstd, 0 nulls; sha256 `b57e4b3f…`). `score_zensim` + `feat_0..feat_371`. Sidecars
  `s3://zentrain/kadis-700k/{omni,zensim_features,source_features}/` (350 each).
- **Shared keys (both):** `source_id` (stable split key 0..139999 — split on this, never on row),
  `source_filename`, `dist_type`, `dist_name`, `severity_level`, `dist_param` (signed for 7/18/25).
- **Mirrors:** `/mnt/v/datasets/kadis700k/canonical/`, `/mnt/tower/output/kadis700k/canonical/`.
- **Full README + schema:** `s3://zentrain/kadis-700k-gpu/README.md` + `s3://zentrain/kadis-700k/README.md`
  (and `~/work/kadis-distort/docs/DATASET.md`).
- **Credit:** reference images + distortion design © VQA Group, Universität Konstanz (Lin, Hosu,
  Saupe) — KADID-10k / KADIS-700k, https://database.mmsp-kn.de/kadid-10k-database.html ("freely
  available to the research community"). Cite KADID-10k (QoMEX 2019) + DeepFL-IQA (arXiv:2001.08113).
