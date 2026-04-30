# Adding zenpicker to a new codec

This walkthrough takes you from "I have a codec with N configs" to "the picker chooses one given image features and a quality target." Estimated time: 30 minutes for the first model, ~10 minutes for re-bakes after that.

The example throughout uses a hypothetical "zenwidget" codec with two scalar control axes (effort, quality) and one categorical axis (compression_mode). Substitute your own.

---

## What you need before starting

1. A codec crate that can encode + measure perceptual quality on a corpus.
2. zenanalyze installed (or your own pixel-feature extractor — anything that emits `&[f32]` works).
3. Python 3.10+ with `numpy`, `scikit-learn`, `joblib` for the training side.
4. Rust ≥ 1.93 to build zenpicker.

---

## Step 1 — produce a Pareto sweep TSV

Your codec runs every config × every quality value over a corpus and emits one row per `(image, size, config, quality)` cell:

```
image_path<TAB>size_class<TAB>width<TAB>height<TAB>config_id<TAB>config_name<TAB>q<TAB>bytes<TAB>zensim<TAB>encode_ms<TAB>total_ms
```

| Column | Type | Notes |
|---|---|---|
| `image_path` | string | Stable identifier (path or sha256) |
| `size_class` | enum | `tiny` / `small` / `medium` / `large` per the project-wide sweep discipline |
| `width`, `height` | int | pixel dims |
| `config_id` | int | 0..N over your full grid |
| `config_name` | string | Human-readable label e.g. `mode_a_effort5_q80` |
| `q` | int | 0..100 quality value swept |
| `bytes` | int | Encoded byte count |
| `zensim` | float | Achieved perceptual quality (zensim or compatible) |

For zenjpeg this is `zenjpeg/examples/zq_pareto_calibrate.rs`. Yours can be a one-shot binary — it just needs to enumerate configs and emit the rows.

**Sweep discipline reminder** (`~/work/claudehints/CLAUDE.md`): cover tiny + small + medium + large image sizes; cover q5–q60 with the same density as q60–q100. If your sweep is denser at high quality, fix it before training a model on the data — the picker will be miscalibrated everywhere the sweep is sparse.

---

## Step 2 — produce a features TSV

One row per `(image, size_class)`, columns prefixed `feat_`:

```
image_path<TAB>size_class<TAB>width<TAB>height<TAB>feat_variance<TAB>feat_edge_density<TAB>...
```

If you're using zenanalyze, declare a `FeatureSet` and walk `FeatureSet::SUPPORTED.iter()` to drive the column header. Reference: `zenjpeg/zenjpeg/examples/zq_pareto_calibrate.rs` does exactly this — its `feature_columns()` is one line (`zenanalyze::feature::FeatureSet::SUPPORTED.iter().collect()`), and any new analyzer feature lands in the TSV without code changes.

**Add a `--features-only` flag to your harness.** The Pareto encode loop is the expensive step (~hours on a typical corpus). When zenanalyze ships new features, you want to extend the existing TSV without re-running the encode pass — `--features-only` skips encoding and emits just the per-image feature rows. Re-extraction takes ~1 second on 1000-image corpora; without it iteration is gated on the encode loop. zenjpeg's harness has this; copy the pattern.

Cost rule: every feature costs hot-path zenanalyze time on every encode decision. Run a permutation-importance ablation (Step 8 below) to drop features that don't pay for themselves before locking in your schema.

---

## Step 3 — define your categorical-cell taxonomy

Decide which encoder axes are **categorical** vs **scalar**:

- **Categorical** (binary or small-finite): color_mode, subsampling, scan order, whether-trellis-is-on. These get one cell per combination.
- **Scalar**: effort level (1–9), trellis lambda, chroma_quality, anything continuous. These get one prediction head per cell.

For zenwidget: `compression_mode` (3 values) is categorical. `effort` (1–9) and `quality` (0–100) are scalar. So 3 cells × 2 scalar heads.

Your `.bin` model will have `n_outputs = 3 cells × (1 bytes + 2 scalars) = 9` outputs.

Why split? See the [zenjpeg result](README.md#shipped-zenjpeg-models-side-by-side-codec-picks): collapsing the 120-cell pure-categorical grid down to 12 cells × 3 heads dropped mean overhead from 8.20% to 2.76% on held-out images. Smaller categorical space → confident picks; the scalar choice within a cell becomes a smooth f32 regression.

---

## Step 4 — train the picker

Don't write your own fitter. **Reuse `zenpicker/tools/train_hybrid.py`** — it's codec-agnostic. You write a small **codec config module** declaring paths + schema + parser; the script handles teacher / distill / persist.

Create `examples/<your_codec>_picker_config.py` (copy `zenjpeg_picker_config.py` and edit):

```python
from pathlib import Path
import re

PARETO   = Path("benchmarks/zenwidget_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zenwidget_features_2026-04-29.tsv")
OUT_JSON = Path("benchmarks/zenwidget_hybrid_2026-04-29.json")
OUT_LOG  = Path("benchmarks/zenwidget_hybrid_2026-04-29.log")

KEEP_FEATURES = [
    "feat_variance", "feat_edge_density", ...,  # your picker schema
]
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

def parse_config_name(name: str) -> dict:
    """Decompose <your codec's config name> into categorical + scalar axes.

    Categorical axes form cells (one cell per unique combination).
    Scalar axes are per-cell prediction targets (continuous).
    """
    # ... regex-parse name, return e.g. {"mode": "fast", "trellis": False,
    # "effort": 5.0, "quality": 80.0}
    return {...}
```

Then run from your codec's repo:

```bash
PYTHONPATH=<zenanalyze>/zenpicker/examples:<zenanalyze>/zenpicker/tools \
    python3 <zenanalyze>/zenpicker/tools/train_hybrid.py \
        --codec-config zenwidget_picker_config
```

End-to-end on a 16-core box: ~3 minutes wall-clock. Useful flags:

| Flag | When |
|---|---|
| `--objective {size_optimal, zensim_strict}` | size_optimal (default) trains mean log-bytes; zensim_strict trains pinball-q99 + per-zq reach gate. Ship both side-by-side |
| `--bytes-quantile 0.99` | quantile for zensim_strict bytes head |
| `--reach-threshold 0.99` | per-cell reach-rate floor for zensim_strict gate |
| `--hidden 192,192,192` | student MLP hidden widths. Default `128,128`; bump when n_inputs grows past ~50 (see Step 8) |
| `--out-suffix _h192x192x192` | filename suffix for capacity sweeps so bakes co-exist |

---

## Step 5 — bake to binary

```bash
python3 tools/bake_picker.py \
    --model benchmarks/zenwidget_hybrid_2026-04-29.json \
    --out  models/zenwidget_picker_v1.0.bin \
    --dtype f16
```

The JSON should declare:

```json
{
  "n_inputs": 26,
  "n_outputs": 9,
  "feat_cols": ["feat_variance", "feat_edge_density", ...],
  "extra_axes": [
    "size_tiny", "size_small", "size_medium", "size_large",
    "log_pixels", "log_pixels_sq", "zq_norm", "zq_norm_sq",
    "zq_x_feat_variance", ...,
    "icc_bytes"
  ],
  "schema_version_tag": "zenwidget.v1.hybrid",
  "config_names": {"0": "mode_a_eff5_q80", ...},
  "hybrid_heads_manifest": {
    "n_cells": 3,
    "cells": [...],
    "output_layout": {
      "bytes_log":   [0, 3],
      "effort":      [3, 6],
      "quality":     [6, 9]
    }
  },
  "scaler_mean": [...],
  "scaler_scale": [...],
  "layers": [{"W": [...], "b": [...]}]
}
```

`extra_axes` and `schema_version_tag` are optional — `bake_picker.py` will fall back to a generic layout if you omit them, but for stable schema_hashes across re-bakes you should declare them explicitly.

Verify the bake:

```bash
python3 tools/bake_roundtrip_check.py \
    --model benchmarks/zenwidget_hybrid_2026-04-29.json \
    --dtype f16
```

The Rust loader runs the same forward pass as the numpy reference; `max rel diff` should be < `5e-3` for f16 (< `1e-4` for f32).

---

## Step 6 — load + use in the codec crate

```rust
use zenpicker::{Model, Picker, AllowedMask};

#[repr(C, align(8))]
struct AlignedModel<const N: usize>([u8; N]);
const MODEL_BYTES: &[u8] =
    &AlignedModel(*include_bytes!("../models/zenwidget_picker_v1.0.bin")).0;

const MY_SCHEMA_HASH: u64 = 0xDEADBEEF_CAFEBABE; // from bake's stderr
const N_CELLS: usize = 3;

#[derive(Clone, Copy, Debug)]
struct CellSpec {
    label: &'static str,
    compression_mode: CompressionMode,
}
const CELLS: &[CellSpec; N_CELLS] = &[
    CellSpec { label: "mode_a", compression_mode: CompressionMode::A },
    CellSpec { label: "mode_b", compression_mode: CompressionMode::B },
    CellSpec { label: "mode_c", compression_mode: CompressionMode::C },
];

pub struct ZenwidgetConstraints {
    pub forbid_mode_b: bool,
    pub max_effort: Option<u8>,
    pub min_quality: Option<f32>,
}

impl ZenwidgetConstraints {
    fn allowed_mask(&self) -> [bool; N_CELLS] {
        std::array::from_fn(|i| {
            let cell = &CELLS[i];
            !(self.forbid_mode_b && cell.compression_mode == CompressionMode::B)
        })
    }
}

pub fn pick_config(
    picker: &mut Picker<'_>,
    features: &[f32],
    constraints: &ZenwidgetConstraints,
) -> Result<EncoderConfig, PickError> {
    let mask_arr = constraints.allowed_mask();
    let mask = AllowedMask::new(&mask_arr);

    // Categorical pick: argmin over the bytes-log sub-range.
    let cell_idx = picker
        .argmin_masked_in_range(features, (0, N_CELLS), &mask, None)?
        .ok_or(PickError::NoAllowedCell)?;

    // Scalar reads: indexes into the rest of the output vector.
    let out = picker.predict(features)?;
    let effort_pred = out[N_CELLS + cell_idx];
    let quality_pred = out[2 * N_CELLS + cell_idx];

    // Clamp scalars to caller constraints.
    let effort = constraints
        .max_effort
        .map_or(effort_pred, |m| effort_pred.min(m as f32))
        .round()
        .clamp(1.0, 9.0) as u8;
    let quality = constraints
        .min_quality
        .map_or(quality_pred, |m| quality_pred.max(m))
        .clamp(0.0, 100.0);

    Ok(EncoderConfig {
        compression_mode: CELLS[cell_idx].compression_mode,
        effort,
        quality,
    })
}

// At codec init, do this once:
pub fn load_picker() -> Result<Picker<'static>, PickError> {
    let model = Model::from_bytes(MODEL_BYTES)?;
    if model.schema_hash() != MY_SCHEMA_HASH {
        return Err(PickError::SchemaMismatch);
    }
    Ok(Picker::new(model))
}
```

Compile-time const: codec defines `CELLS` and `MY_SCHEMA_HASH` matching the manifest, so any drift between the bake and the codec is caught at load time.

A complete runnable reference for the zenjpeg shape lives at [`examples/hybrid_heads_codec_sketch.rs`](examples/hybrid_heads_codec_sketch.rs).

---

## Step 7 — wire it into encode

Wherever your codec resolves "user wants quality target X" into a concrete config, route through `pick_config`:

```rust
pub fn encode(
    request: &EncodeRequest,
    picker: &mut Picker<'_>,
) -> Result<Vec<u8>, EncodeError> {
    let analysis = zenanalyze::analyze_features(request.pixels(), &PICKER_QUERY)?;
    let features = build_feature_vector(&analysis, request.target_quality());
    let cfg = pick_config(picker, &features, &request.constraints)?;
    encode_with_config(&cfg, request.pixels())
}
```

The features vector packs zenanalyze outputs in the order declared in the bake manifest. Keep that order in one place — a `const FEATURES_ORDER: &[AnalysisFeature]` shared between `build_feature_vector` and the `MY_SCHEMA_HASH` const.

---

## Step 8 — tune capacity + prune the schema

After your first end-to-end bake, two diagnostics tell you what to do next.

### Diagnostic 1: teacher metrics vs student metrics

The training script prints both. If teacher mean overhead < student mean overhead by more than ~0.3pp, **the student MLP is undersized** — distillation can't track the teacher's complexity at the current architecture.

Sweep `--hidden` to find the sweet spot. Empirical results from zenjpeg's v2.1 retrain (35 features, 80 cross-termed inputs):

| hidden        | params | student mean overhead | argmin acc |
|---|---:|---:|---:|
| 128x128 (default) | 24.6 K | 2.91% | 50.8% |
| 128x128x128   | 48 K  | 2.59% | 53.9% |
| 192x192x192   | 97 K  | **2.33%** | **56.3%** ← winner |
| 256x256x256   | 162 K | 2.42% | 53.9% |
| 384x384x384   | 363 K | 3.42% | 44.1% (overfit) |

**Takeaways that generalize across codecs:**

- depth helps more than width on this kind of input;
- 3 narrow hidden layers (~192) usually beat 2 wide (~256);
- overfitting kicks in past ~150 K parameters at our data scale;
- always sweep, don't guess.

Use `--out-suffix _hWxWxW` to keep the candidate bakes side-by-side without clobbering each other. Bake the winner once via `bake_picker.py` (Step 5).

### Diagnostic 2: feature-importance ranking

Run permutation ablation against your full feature set:

```bash
PYTHONPATH=<zenanalyze>/zenpicker/examples:<zenanalyze>/zenpicker/tools \
    python3 <zenanalyze>/zenpicker/tools/feature_ablation.py \
        --codec-config <your_codec>_picker_config \
        --method permutation
```

`--method permutation` (default) trains the model once, then for each feature shuffles that column on the validation set and measures the mean-overhead delta. **~50× faster than retrain-LOO** with near-identical ranking — 2 minutes wall-clock instead of 1-2 hours on the v2.1 53-feature × 120-config matrix.

Output: `benchmarks/<codec>_hybrid_<date>_ablation.json` (machine-readable) and `_ablation.log` (human). Sorted by `Δ vs baseline`, largest hurt first.

**Pruning rules:**

- Drop features with `Δ ≤ 0.00pp` (zero impact) — e.g. on zenjpeg v2.1, all HDR / wide-gamut signals scored 0.00pp because the corpus is SDR. Same will be true on most codecs unless you explicitly include HDR images.
- Drop features with `Δ < 0.05pp` if you're aggressively trimming hot-path cost.
- Keep the top 10-20 by `Δ` and retrain — typical result: same or better held-out metrics with a smaller, faster analyzer hot path.

The ranking is stable between `HISTGB_FAST` and `HISTGB_FULL` so use FAST for the ablation pass; absolute numbers will shift but the ordering won't.

### Don't pre-emptively `pip install lightgbm`

We benchmarked lightgbm 4.6 against sklearn's `HistGradientBoostingRegressor` on this picker workload (~4000 rows × 80 inputs, max_iter=400, depth=8): sklearn HistGB is **2-16× faster** because per-fit overhead dominates lightgbm on the picker's many-small-fits pattern (12 cells × 3 heads = 36 fits per training run, plus 53 permute passes in ablation). lightgbm wins on huge single-shot training matrices — exactly the wrong shape here. Stick with sklearn HistGB.

---

## What can go wrong

- **Schema mismatch panics on load.** Your codec's `MY_SCHEMA_HASH` const drifted from the baked model. Re-read `manifest.json`'s `schema_hash` field, update the const.
- **Round-trip check fails.** The Rust loader and Python forward pass disagree by more than `5e-3` (f16) or `1e-4` (f32). Usually means a layer dim or activation choice diverged between the trainer and the bake. Re-emit the JSON and re-run.
- **`argmin_masked_in_range` returns `None`.** All cells masked out — caller's constraints are unsatisfiable. Codec should surface this as `Err(NoAllowedCell)` and let the caller relax constraints.
- **Predictions are nonsensical (huge negative log-bytes, NaN scalars).** The feature vector you fed in is out of training distribution. zenpicker doesn't know that — it'll return whatever the MLP says. The [safety plane design](SAFETY_PLANE.md) covers the verify-and-rescue path for this case in zenjpeg; analogous patterns apply to other codecs.

---

## Step 9 — safety profiles + rescue (optional)

Once your codec ships a working `size_optimal` bake, you can add a second `zensim_strict` bake (worst-case-safe sizing) and wire the two-shot rescue path. Both use additive APIs zenpicker already exposes — no schema change.

| Step | Where |
|---|---|
| Train the strict variant | `tools/train_hybrid.py --objective zensim_strict --codec-config <your_codec>_picker_config` |
| Ship both bakes side-by-side via `include_bytes!` | `models/<your_codec>_picker_v2.0_{hybrid,zensim_strict}.bin` |
| Honor the runtime reach gate | `zenpicker::reach_gate_mask(reach_rates_for_target_zq, threshold, &mut gate)` then AND with your constraint mask |
| Run the two-shot rescue loop | `Picker::argmin_masked_top_k::<2>` for cached second-best + `zenpicker::rescue::should_rescue` for the threshold predicate |

The full design + the codec orchestration sketch live in [SAFETY_PLANE.md](SAFETY_PLANE.md) and the README's [Safety profiles](README.md#safety-profiles-size_optimal-vs-zensim_strict) section. Defer all of this to v2 — your codec ships fine on `size_optimal` alone.

---

## When to re-bake

- Your codec's config grid changed (added/removed a knob).
- Your feature schema changed (different `feat_cols` set).
- You re-ran the Pareto sweep on a larger or different corpus.
- The held-out mean overhead drifted by more than ~2 pp from the previous bake.

Re-bakes are cheap once the harness is set up: ~5 minutes for sweep load + dataset cache (re-uses the `/tmp/zq_picker_cache/` artifacts), ~30 seconds for parallel teacher training, ~3 minutes for student fit. Total ≈ 8 minutes for an iteration; commit the new `.bin` and `manifest.json` to your codec crate's `models/` directory.

---

## License

zenpicker, the bake tools, and the shared training library are AGPL-3.0-only OR LicenseRef-Imazen-Commercial. Models you bake are your own — the picker runtime imposes no restriction on the weights.
