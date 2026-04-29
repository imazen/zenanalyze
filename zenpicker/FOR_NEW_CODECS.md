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

If you're using zenanalyze, declare a `FeatureSet` const, run `analyze_features_rgb8` per image, dump the values. The 8-feature picker schema (validated for zenjpeg) is `FeatureSet::ZENJPEG_PICKER_V1_1` if you want a starting point — or build your own from the supported `AnalysisFeature` variants.

Cost rule: every feature costs hot-path zenanalyze time on every encode decision. Run an ablation (see `zq_feature_ablation.py`, `zq_feature_group_ablation.py` in zenjpeg) to drop features that don't pay for themselves before locking in your schema.

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

Adapt `zenjpeg/scripts/zq_hybrid_heads.py` to your codec. The shared library `zenjpeg/scripts/_zq_picker_lib.py` carries the slow steps so your fitter is short:

```python
from _zq_picker_lib import (
    HISTGB_FAST, HISTGB_FULL,
    load_or_build_dataset, split_by_image,
    train_teachers_parallel, teacher_predict_all,
    evaluate_argmin,
)

# 1. Load + cache (first call ~30s; subsequent ~100ms via /tmp cache).
Xs, Y, meta, feat_cols, config_names = load_or_build_dataset(
    pareto_path=Path("benchmarks/zenwidget_pareto_2026-04-29.tsv"),
    features_path=Path("benchmarks/zenwidget_features_2026-04-29.tsv"),
    keep_features=KEEP_FEATURES,  # your 8-or-so picker schema
)

# 2. Image-level holdout.
tr, va = split_by_image(meta, holdout_frac=0.2)
Xs_tr, Xs_va = Xs[tr], Xs[va]

# 3. Build per-cell teacher targets from the full Pareto.
#    For each (image, size, target_zq) row, find within-cell optimal
#    (bytes, scalar1, scalar2). See zq_hybrid_heads.py for the
#    grouping logic — adapt to your config naming convention.
bytes_log_tr, chroma_tr, lambda_tr, reach_tr = build_per_cell_optima(...)

# 4. Train teachers in parallel — 5-10x speedup vs serial.
t_bytes = train_teachers_parallel(Xs_tr, bytes_log_tr, params=HISTGB_FULL)
t_chroma = train_teachers_parallel(Xs_tr, chroma_tr, params=HISTGB_FULL)
t_lambda = train_teachers_parallel(Xs_tr, lambda_tr, params=HISTGB_FULL)

# 5. Distill teachers' soft targets into a small shared MLP. See the
#    same script for the engineered cross-term layout (zq×feat,
#    log_pixels², …) and the joint-output MLP fit.
student = fit_student_mlp(...)

# 6. Save the JSON the bake tool consumes.
save_model_json("benchmarks/zenwidget_hybrid_2026-04-29.json", student)
```

For iteration use `HISTGB_FAST` (max_iter=100, max_depth=4) — drops teacher training from 5–10 min to 30–60 s. Switch to `HISTGB_FULL` for the production bake.

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

## What can go wrong

- **Schema mismatch panics on load.** Your codec's `MY_SCHEMA_HASH` const drifted from the baked model. Re-read `manifest.json`'s `schema_hash` field, update the const.
- **Round-trip check fails.** The Rust loader and Python forward pass disagree by more than `5e-3` (f16) or `1e-4` (f32). Usually means a layer dim or activation choice diverged between the trainer and the bake. Re-emit the JSON and re-run.
- **`argmin_masked_in_range` returns `None`.** All cells masked out — caller's constraints are unsatisfiable. Codec should surface this as `Err(NoAllowedCell)` and let the caller relax constraints.
- **Predictions are nonsensical (huge negative log-bytes, NaN scalars).** The feature vector you fed in is out of training distribution. zenpicker doesn't know that — it'll return whatever the MLP says. The [safety plane design](SAFETY_PLANE.md) covers the verify-and-rescue path for this case in zenjpeg; analogous patterns apply to other codecs.

---

## Optional — safety profiles + rescue

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
