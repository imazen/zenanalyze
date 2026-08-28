# zenpredict-viz

Interactive web tool for exploring ZNPR v3 bake internals —
[zenanalyze#79](https://github.com/imazen/zenanalyze/issues/79).

Single-page browser app, no backend. Loads a `.bin` bake (file input,
the quick-load list, or a `#bake=<relative-url>` permalink), runs the
forward pass in WASM against the unchanged `zenpredict` crate, and
renders:

1. **Summary** — header, layer dims, dtypes, bias/weight stats, metadata
   keys, presence/absence of every optional stage.
2. **Scaler** — per-feature `(mean, scale)` heatmap. Hover for the
   feature name.
3. **L0 importance** — `scaler_scale[i] · Σ_h |W₀[i, h]|` heatmap +
   per-block stats + top/bottom-20 lists + feature search.
4. **Weights** — per-layer weight heatmap (dequantized for i8), bias /
   scale strips, value histogram.
5. **Calibration** — the zensim post-MLP stages that are present:
   `tanh_output_head`, `output_calibration_spline`,
   `per_codec_calibration`, `per_sample_alpha_head`.
6. **Forward** — paste / CSV-upload / sample-pack / synthetic input; the
   waterfall shows bake-declared `feature_transforms` → standardize →
   per-layer pre/post activation → raw output → `output_specs` → the
   calibration stages, in pipeline order.
7. **Attribution** — first-layer attribution of one input vector.
8. **Compare** — 2–4 bakes with matching schemas: scaler shift, L0
   importance reshuffle, per-layer weight RMS / L2 diff, calibration
   curve overlays.

Every canvas heatmap in the active panel exports to PNG (sidebar
button); the active panel and loaded bake are kept in the URL hash.

## Feature naming

Names come from, in order: the bake's own `zentrain.feature_columns`
(every picker bake carries them — index-aligned by construction), the
zensim 228 / 300 / 372 semantic layout keyed on `n_inputs` (with the
richer `feature_catalog.json` when present), then `f<idx>`. A 372-input
picker bake over zenanalyze features is therefore labelled with its own
column names, not the zensim layout.

## Parity with the runtime

`forward_with_taps` mirrors `zenpredict::inference` operation for
operation — same standardize, the same `f32::mul_add` SAXPY in the same
input-major order with the same zero-input skip, the same i8 epilogue,
the same activations — so the raw output is **bit-identical** to
`Predictor::predict` / `predict_transformed` (stricter than the 1-ULP
bar in #79). `tests/forward_parity.rs` asserts `to_bits()` equality on
bakes composed in-test (every dtype × activation, with and without
`feature_transforms` / `output_specs`); `shipped_bakes_parity` does the
same on real bakes and fails loud unless `ZENPREDICT_VIZ_BAKES` names
them (`just viz-test-shipped`) — the skip is the caller's decision.
Measured 2026-08-28: all 7 shipped zensim weights in
`zensim/zensim/weights/*.bin` and the canonical zenwebp rd_time picker
bit-identical.

## Build / test

```sh
just viz-test                       # fmt + clippy + parity/ONNX tests + wasm32 build (= CI)
just viz-test-shipped               # parity on ../zensim/zensim/weights/*.bin
just viz-serve                      # ./build.sh release + http.server on :3142
```

`./build.sh release` runs `wasm-pack build --target web` into `web/pkg/`,
copies every `*.bin` from the sibling zensim weights dir (override with
`ZENPREDICT_VIZ_BAKES=<dir>`) into `web/bakes/` and writes
`web/bakes/index.json` (the quick-load list is built from it, so it never
points at bakes that no longer exist), then regenerates
`web/feature_catalog.json`. Requires `wasm-pack`
(`curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`).

Publishing: `.github/workflows/pages.yml` deploys `web/` to GitHub Pages
on **manual dispatch only** (repo setting Pages → Source = GitHub
Actions is a one-time admin step).

## Native tools (feature-gated binaries)

- `znpr2onnx` (`--features onnx-export`) — standardize + MLP as ONNX
  (opset 13) for Netron / onnxruntime; prints the stages it does not
  export (calibration, `feature_transforms`, `output_specs`).
  `tests/onnx_parity.rs` evaluates the exported graph against
  `Predictor::predict` for every dtype × activation. See
  `docs/onnx_export.md`.
- `build_feature_catalog` (`--features feature-catalog`) — regenerates
  `web/feature_catalog.json` from `zensim/src/metric.rs` and
  `zenanalyze/src/feature.rs`.
- `build_sample_pack` (`--features sample-pack`) — regenerates the
  committed `web/sample_pack.json` from the canonical training parquets.

## Architecture

- **`src/lib.rs`** — wasm-bindgen wrapper around `zenpredict::Model`:
  `parse_bake`, `forward_with_taps`, `layer_weights`,
  `feature_transform_tokens` (each with a `_native` twin for tests).
- **`web/main.js`** — UI entry, panel routing, permalink, file / URL /
  CSV load, WASM init. **`web/feature_layout.js`** — feature naming.
  **`web/calibration_decoders.js`** — JS decoders for the zensim
  calibration metadata blobs. **`web/panels/*.js`** — one file per panel.

## Not done

- SHAP / integrated-gradients attribution (the MVP is one first-layer
  pass); a per-layer weight-diff heatmap in Compare (RMS + L2 only);
  the `per_sample_alpha_head` forward view (architecture diagram only);
  `training_stats` in the feature catalog.
- The #79 "4+ bakes without UI lockup on slider drag" criterion is not
  measured — Compare has no slider; it renders 2–4 bakes once.
