# zenpicker training pipeline

Codec-agnostic training scripts. Each codec writes a small **codec
config module** (see [`examples/zenjpeg_picker_config.py`](../examples/zenjpeg_picker_config.py))
declaring its TSV paths, feature schema, zq target grid, and
config-name parser. Then runs these scripts against that config.

## Files

| File | Purpose |
|---|---|
| `_picker_lib.py` | Shared library: disk-cached dataset construction, parallel teacher training (joblib), HISTGB_FAST/FULL presets, subsample helper, StepTimer |
| `train_hybrid.py` | **Recommended.** Hybrid-heads training: N categorical cells + K scalar prediction heads per cell. Codec supplies the cell taxonomy via `parse_config_name`. **Always pass `--activation leakyrelu`** for new bakes — the default `relu` routes through sklearn's single-threaded `MLPRegressor.fit` and is 10–20× slower at the same shape (sklearn keeps Adam matmuls below the BLAS-threading threshold; pytorch matches our actual wall-clock budget). Other flags: `--objective {size_optimal, zensim_strict}`, `--bytes-quantile`, `--reach-threshold`, `--hidden 192,192,192`, `--out-suffix` |
| `train_distill.py` | Categorical-only distillation (legacy v1.x picker shape). Per-config HistGB teacher → small shared MLP student |
| `train_distill_reduced.py` | Same as `train_distill.py` but with a reduced feature subset declared in the codec config |
| `correlation_cleanup.py` | **Tier 0 — pre-flight.** Pairwise Spearman correlation across feature columns; clusters at \|ρ\| ≥ 0.99 produce a drop-list anchored on canonical signals (`feat_pixel_count` etc.). Separates **constants** (zero variance — corpus gap, NOT redundancy) from **low-variance** (<5 unique values — Spearman degenerate) from **true redundancy** (mathematical collinearity). Run before any model-training-based ablation. Seconds, no model fit. See [`../ABLATION.md`](../ABLATION.md) |
| `feature_ablation.py` | **Tier 1/3 — per-feature importance.** `--method permutation` (default) — train once, shuffle each column, ~50× faster than LOO. `--method loo` retrains the model with each feature dropped (gold standard, expensive). `--strict` exits 1 on features whose Δ is below `--max-negative-delta-pp` (overfit-on-noise signal). Don't trust either method's output without first running `correlation_cleanup.py` — constants will report Δ = 0 indistinguishably from real low-signal features |
| `feature_group_ablation.py` | **Tier 2 — group ablation.** Drop entire tier-aligned groups (all `*_p50/p75/p90/p99` percentiles, all chroma sharpness, all noise floor). Catches synergy: if dropping one feature in a percentile family is zero but dropping all four costs 0.5pp, the family is collectively load-bearing |
| `validate_schema.py` | Re-train production HistGB with a chosen feature subset, report held-out metrics |
| `capacity_sweep.py` | Architecture × cross-term-recipe sweep over the student MLP |
| `adversarial_probe.py` | Corner-case spot-check for any model JSON: zeros / huge / NaN / Inf / single-feature spike. Asserts no NaN/Inf in output, top-1/top-2 gap ≥ 0, predicted bytes plausible. Exits 1 in `--strict` |
| `size_invariance_probe.py` | Post-bake size-generalization gate. Resizes fixture images to each of `tiny / small / medium / large` and asserts the picker's argmin cell stays stable across them per `(image, target_zq)`. Counterpart to `train_hybrid.py`'s `PER_SIZE_TAIL` / `DATA_STARVED_SIZE` violations. Exits 1 in `--strict` when stability < threshold (default 90 %). See [SAFETY_PLANE.md](../SAFETY_PLANE.md#size-invariance-is-a-safety-property) |
| `diagnose_picker.py` | Human-readable health report for any model JSON or .manifest.json. Works on legacy bakes that pre-date `safety_report` (falls back to a static MLP weight scan) |

The [`tools/bake_picker.py`](../../tools/bake_picker.py) script (parent
directory, not this one) consumes the JSON output of any of the
training scripts and emits the v2 binary blob + manifest.

## Codec config contract

A codec config module exports:

```python
from pathlib import Path

PARETO        = Path(...)          # Pareto sweep TSV
FEATURES      = Path(...)          # Analyzer features TSV
OUT_JSON      = Path(...)          # Trained model JSON output
OUT_LOG       = Path(...)          # Training summary text output
KEEP_FEATURES = [...]              # list[str] of `feat_*` columns
ZQ_TARGETS    = [...]              # list[int] of target_zq values

def parse_config_name(name: str) -> dict:
    """Parse a config_name into categorical + scalar axes."""
    ...
```

`parse_config_name` returns a dict where keys partition into
**categorical axes** (hashable values; combined to form cells) and
**scalar axes** (floats; per-cell prediction targets, with a
sentinel value for cells where the axis is N/A).

For the zenjpeg reference, see [`examples/zenjpeg_picker_config.py`](../examples/zenjpeg_picker_config.py).

## Canonical end-to-end command sequence

Every codec bake should run this six-stage pipeline, in order. Stages 4–6 are the safety gates — they may fail and that is the point. Don't skip any of them. Run from the codec's repo root (where the Pareto sweep TSVs live in `benchmarks/`); paths shown assume `<zenanalyze>` is the path to your zenanalyze checkout, and the codec config module is `<your_codec>_picker_config` importable on `PYTHONPATH`.

```bash
PP="<zenanalyze>/zenpicker/examples:<zenanalyze>/zenpicker/tools"

# 1. Sweep. Codec's harness emits per-(image, size, config, q) Pareto rows
#    + per-(image, size) features TSV. Hours; do once, then only re-extract
#    features when zenanalyze adds variants (--features-only flag, ~1 s).
cargo run --release -p <your_codec> --features '<...>' \
    --example <your_codec>_pareto_calibrate
# (when re-extracting features only after a zenanalyze upgrade:)
cargo run --release -p <your_codec> --features '<...>' \
    --example <your_codec>_pareto_calibrate -- --features-only \
    --features-output benchmarks/<your_codec>_features_v2_1.tsv

# 2. Train. Codec config supplies CATEGORICAL_AXES, SCALAR_AXES, parser,
#    optional SAFETY_THRESHOLDS, KEEP_FEATURES. Default --hidden 128,128
#    is fine for ≤8-feature schemas; bump to 192,192,192 when n_inputs
#    grows past ~50. For SLA traffic, also bake a zensim_strict variant.
PYTHONPATH=$PP CI=1 python3 <zenanalyze>/zentrain/tools/train_hybrid.py \
    --codec-config <your_codec>_picker_config \
    --objective size_optimal --hidden 192,192,192
# zensim_strict variant (pinball-q99 bytes head + reach gate):
PYTHONPATH=$PP CI=1 python3 <zenanalyze>/zentrain/tools/train_hybrid.py \
    --codec-config <your_codec>_picker_config \
    --objective zensim_strict --hidden 192,192,192

# 3. Permutation-importance ablation. Drop features with Δ < 0 (model
#    overfits noise) before locking the schema. ~2 min wall-clock.
PYTHONPATH=$PP CI=1 python3 <zenanalyze>/zentrain/tools/feature_ablation.py \
    --codec-config <your_codec>_picker_config

# 4. Bake. bake_picker.py refuses to bake when the model JSON's
#    safety_report.passed=false — see strict-gate failures from step 2
#    and pass --allow-unsafe only after writing down a reviewer note.
python3 <zenanalyze>/tools/bake_picker.py \
    --model benchmarks/<bake-base>.json \
    --out   <zenanalyze>/zenpicker/models/<your_codec>_picker_v<X>.bin \
    --dtype f16

# 5. Round-trip check. Asserts Rust loader + numpy reference forward
#    pass agree to within tolerance; catches binary-format regressions.
python3 <zenanalyze>/tools/bake_roundtrip_check.py \
    --model benchmarks/<bake-base>.json --dtype f16

# 6. Post-bake spot checks.
#    a) Adversarial probe: zeros / huge / NaN / Inf / single-feature spike.
python3 <zenanalyze>/zentrain/tools/adversarial_probe.py --strict \
    --model benchmarks/<bake-base>.json
#    b) Size-invariance probe: resize fixture images to all four
#       size_classes and assert the picker's argmin stays stable.
#       The post-bake counterpart to train_hybrid.py's PER_SIZE_TAIL
#       + DATA_STARVED_SIZE violations. See SAFETY_PLANE.md →
#       "Size invariance is a safety property".
python3 <zenanalyze>/zentrain/tools/size_invariance_probe.py --strict \
    --model        benchmarks/<bake-base>.json \
    --features-tsv benchmarks/<features>.tsv
#    c) Human-readable health report — review by eye.
python3 <zenanalyze>/zentrain/tools/diagnose_picker.py \
    --model benchmarks/<bake-base>.json
```

End-to-end retrain on a 16-core box, with the dataset cache warm:
~3 minutes wall-clock for steps 2-6 combined (was ~25 minutes before
the parallel-teachers refactor).

**CI rule:** every codec's CI must run stages 2-3 with `CI=1`
(auto-strict) and stages 4 + 6a as separate jobs. A regression in any
of them fails the workflow before a bad bake ever reaches `models/`.

### Quick reference — which flag does what

| Flag | Stage | When |
|---|---|---|
| `--codec-config <module>` | 2, 3 | Always. Module declares paths + schema + parser |
| `--objective {size_optimal, zensim_strict}` | 2 | Default `size_optimal`. Bake `zensim_strict` alongside for SLA traffic |
| `--bytes-quantile 0.99` | 2 | Quantile for the zensim_strict bytes head |
| `--reach-threshold 0.99` | 2 | Per-cell reach-rate floor for the zensim_strict gate |
| `--hidden W1,W2,...` | 2 | MLP hidden widths. Default `128,128`; bump to `192,192,192` past ~50 inputs |
| `--out-suffix <suffix>` | 2 | Capacity sweep — keeps multiple bakes side-by-side |
| `--strict` / `CI=1` env | 2, 3, 6a | Exit 1 on safety-gate violation. Always on in CI |
| `--allow-unsafe` | 2, 3, 4 | Override strict gate. Only with a reviewer note |
| `--method {permutation, loo}` | 3 | Default `permutation` (~50× faster than LOO) |
| `--n-repeats N` | 3 | Permutation repeats per feature (default 3, bump to 5 on small val sets) |
| `--max-negative-delta-pp X` | 3 | Strict-mode threshold for negative-Δ features (default −0.05pp) |
| `--dtype {f16, f32}` | 4, 5 | Bake weight storage. f16 halves size at no measurable accuracy cost |

## Iteration vs production

`_picker_lib.py` exports two HistGB hyperparameter presets:

| Preset | `max_iter` | `max_depth` | When |
|---|---:|---:|---|
| `HISTGB_FAST` | 100 | 4 | Architecture sweeps, ablation runs, anything you'll re-run |
| `HISTGB_FULL` | 400 | 8 | Production bake — only the final pre-bake training run |

The picker ablation work confirmed feature-importance ranking is
stable between FAST and FULL; absolute mean overhead drops by ~1pp
when switching FAST→FULL. Use FAST for everything except the final
bake.

## Tuning the bake (post-training-script)

Two knobs matter most when distillation underperforms the teacher:

### Student MLP capacity (`--hidden W1,W2,...`)

Default `128,128` matches the v2.0 baseline. When the input layer
grows past ~50 cross-termed inputs (e.g. v2.1's 35-feature schema
feeds 80 inputs into the MLP), the default is undersized. Sweep with
`--out-suffix _hWxWxW` to compare side-by-side without clobbering
bakes. Empirically 3 hidden layers narrow (~192) beat 2 wide (~256):

| hidden        | params  | mean overhead | argmin acc |
|---|---:|---:|---:|
| 128x128 (default) | 24.6 K | 2.91% | 50.8% |
| 128x128x128   | 48 K  | 2.59% | 53.9% |
| 192x192x192   | 97 K  | **2.33%** | **56.3%** ← v2.1 winner |
| 256x256x256   | 162 K | 2.42% | 53.9% |
| 384x384x384   | 363 K | 3.42% | 44.1% (overfit) |

(zenjpeg v2.1, 80 inputs → 36 outputs, mean log-bytes regression.)

Numbers will differ across codecs but the shape generalizes:
**depth helps more than width**; overfitting kicks in beyond
~150 K params on this data scale. Sweep first, then bake at the
chosen size.

### Permutation feature ranking

`feature_ablation.py --method permutation` ranks features by
shuffling each column on the validation set and measuring
mean-overhead delta. Train once, permute 53 times → **~2 minutes
wall-clock** for a 53-feature × 120-config matrix vs ~1-2 hours for
retrain-LOO. Ranking tracks LOO closely (well-known result for tree
ensembles).

Workflow when starting from the broadest possible feature set:

1. Run `train_hybrid.py` once with all features → baseline.
2. Run `feature_ablation.py --method permutation`.
3. Drop features with `Δ ≤ 0.00pp` (zero-impact) — usually HDR /
   wide-gamut / palette signals on a non-HDR / non-palette corpus.
4. Optionally drop features with `Δ < 0.05pp` (very low impact).
5. Retrain with the pruned `KEEP_FEATURES`, compare held-out
   metrics, and bake.

Use `--n-repeats 5` if held-out validation set is small; default 3
is enough at our 7000-row scale.

### Safety gates (mandatory before shipping any bake)

`train_hybrid.py` runs a battery of diagnostics on every fit and emits a `safety_report` block into the model JSON. The report includes:

| Check | What it catches |
|---|---|
| `OVERFIT` | train→val mean-overhead gap exceeds threshold (default 2.0pp) |
| `LOW_ARGMIN` | val argmin accuracy below floor (default 30%) |
| `HIGH_OVERHEAD` | val mean overhead exceeds ceiling (default 10%) |
| `PER_ZQ_TAIL` | any single target_zq band's p99 overhead exceeds threshold (default 80%) — catches catastrophic misses concentrated in one quality band |
| `PER_SIZE_TAIL` | any single `size_class`'s p99 overhead exceeds `max_per_size_p99_overhead_pct` (default 80%). Size invariance is a safety property — see [SAFETY_PLANE.md](../SAFETY_PLANE.md#size-invariance-is-a-safety-property) |
| `DATA_STARVED_SIZE` | any `(size_class, target_zq)` cell has fewer than `min_train_rows_per_size_zq` training rows (default 50). Catches sweep harnesses that silently skip a size class |
| `WORST_ROW` | any single (image, size, zq) row exceeds threshold (default 200%) |
| `DATA_STARVED_CELL` | any cell has fewer member configs than threshold (default 3) |
| `NAN_WEIGHTS` / `INF_WEIGHTS` / `NAN_PREDICTIONS` | NaN or Inf in MLP weights or val predictions |
| `DEAD_NEURONS` | hidden neurons with ~0 output variance on val (default 30%) |
| `WEIGHT_BLOWUP` | max-to-median weight ratio per layer (default 1000×) |
| `NO_SAFE_CELL_AT_TOP_ZQ` | (zensim_strict only) reach gate empty at the highest zq band |

Override defaults in your codec config:

```python
SAFETY_THRESHOLDS = dict(
    max_per_zq_p99_overhead_pct=50.0,   # tighter than the default 80
    min_argmin_acc=0.40,                 # require better-than-default
    # ... any subset of DEFAULT_SAFETY_THRESHOLDS keys
)
```

**The strict gate** (`--strict`, also auto-enabled when the `CI` environment variable is set) makes `train_hybrid.py` exit 1 on any violation. JSON + log are still written so reviewers can inspect; only the exit code signals failure. **CI should always run with `--strict`** so a regressed bake fails the workflow before it gets near `bake_picker.py`.

`bake_picker.py` reads `safety_report.passed` and **refuses to bake** an unsafe model JSON unless `--allow-unsafe` is passed (only when the violation is intentional and reviewed). Defense in depth: the `safety_report` is also forwarded into the `.manifest.json`, so codec runtime can refuse to load too.

### OOD detection (runtime gate)

`train_hybrid.py` also computes per-feature distribution stats — `{min, p01, p25, p50, p75, p99, max, mean, std}` over the **training** image set — and ships them in `safety_report.diagnostics.feature_bounds`. `bake_picker.py` lifts the `(p01, p99)` pair into a top-level `manifest.feature_bounds_p01_p99` array aligned 1:1 with `feat_cols`.

Codec runtime emits a compile-time `FEATURE_BOUNDS: &[zenpredict::FeatureBound]` table from the manifest, then calls `zenpredict::first_out_of_distribution(&features, FEATURE_BOUNDS)` before `argmin_masked`. On `Some(idx)`, fall through to `RescueStrategy::KnownGoodFallback` instead of trusting an MLP extrapolation. NaN / Inf inputs always trigger the gate.

### Pick confidence (runtime metric)

`Picker::pick_with_confidence(features, mask, adjust)` returns `(best_idx, log_bytes_gap_to_2nd)`. A small `gap` (< ~0.1 in log space, i.e. < ~10% bytes) means the picker barely chose top-1 over top-2 — a strong signal the rescue path should verify even before encoding. Codec exposes the gap on `EncodeMetrics`; imageflow / proxy operators scrape per-request to detect drift.

### Permutation-importance gate (schema cleanliness)

`feature_ablation.py --strict` (auto in CI) exits 1 on any feature with `Δ < --max-negative-delta-pp` (default −0.05pp). A feature whose mean overhead *drops* when the column is shuffled is noise the model overfit on — 100% drop signal. CI should run the ablation on the same `--codec-config` that `train_hybrid.py` did and treat negative-Δ findings as a `KEEP_FEATURES` regression.

### Adversarial probe + diagnose (post-bake spot-check)

`adversarial_probe.py --model <bake>.json --strict` runs ~10 corner inputs (zeros, huge, NaN, Inf, single-feature spike) and asserts no NaN/Inf in output, top-1/top-2 gap ≥ 0, predicted bytes plausible. Cheap CI gate after `bake_picker.py`.

`size_invariance_probe.py --model <bake>.json --features-tsv <features>.tsv --strict` runs the picker against ~10 fixture images at all four `size_class` resizes and asserts the argmin cell stays stable across them per `(image, target_zq)`. The post-bake counterpart to `train_hybrid.py`'s `PER_SIZE_TAIL` / `DATA_STARVED_SIZE` violations — see [SAFETY_PLANE.md → "Size invariance is a safety property"](../SAFETY_PLANE.md#size-invariance-is-a-safety-property) for the rationale.

`diagnose_picker.py --model <bake>.json` (or `--manifest <bake>.manifest.json`) prints a human-readable health report. Useful for reviewing a bake by eye. Falls back gracefully on legacy bakes (pre-`safety_report`) by computing a static MLP weight scan from the JSON layers.

### GBM backend choice

`_picker_lib._make_teacher(params)` returns the GBM. Today: always
sklearn `HistGradientBoostingRegressor`. lightgbm 4.6 was
benchmarked and is **2-16× slower** than sklearn HistGB on our
many-small-fits workload (per-cell × per-head decomposition keeps
each fit small; lightgbm's per-fit overhead dominates). The wrapper
is kept so dispatch can flip if a future single-shot fit makes
lightgbm a win, but don't pre-emptively `pip install lightgbm` —
sklearn HistGB is the right default.

## Adding a new codec

1. Copy `examples/zenjpeg_picker_config.py` to `examples/<your_codec>_picker_config.py`.
2. Edit paths, KEEP_FEATURES, ZQ_TARGETS, and `parse_config_name` for your codec's config name pattern.
3. Add `examples/` to `PYTHONPATH` and run the training scripts as above.

The runtime crate (zenpicker) doesn't change. Your codec's encoder
crate ships the produced `.bin` via `include_bytes!` and a
compile-time `CELLS` table matching the manifest. See
[FOR_NEW_CODECS.md](../FOR_NEW_CODECS.md) for the codec-side wiring
walkthrough.
