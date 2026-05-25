# zenpredict-viz

Interactive web tool for exploring ZNPR v3 bake internals — implements
[zenanalyze#79](https://github.com/imazen/zenanalyze/issues/79).

Single-page browser app, no backend. Loads a `.bin` bake (drag-drop or
shipped-bake quick-button), runs the standardize → MLP forward pass in
WASM against the unchanged `zenpredict` crate, and renders:

1. **Summary** — header, layer dims, dtypes, bias/weight stats, metadata
   key list, presence/absence of optional stages.
2. **Scaler** — per-feature `(mean, scale)` as a 12-row heatmap (4 scales
   × 3 channels). Hover for semantic feature name.
3. **L0 importance** — `scaler_scale[i] · Σ_h |W₀[i, h]|` heatmap +
   per-block stats + top/bottom-20 lists. Same source-of-truth math as
   `zensim/zensim-validate/examples/dump_l0_importance.rs`.
4. **Live forward pass** — paste a feature vector or click "synthetic",
   see standardize → L0 pre/post → L1 pre/post → ... → raw output as a
   waterfall with per-stage range + L2 deltas.

Post-MLP calibration stages (`zentrain.tanh_output_head`,
`zentrain.output_calibration_spline`, `zentrain.per_codec_calibration`,
`zentrain.per_sample_alpha_head`) are Stage 2 work — surface in the
metadata badges today but not applied to the waterfall yet.

## Build

```sh
./build.sh release    # → web/pkg/ + web/bakes/ populated
python3 -m http.server -d web 3142
# open http://localhost:3142/
```

Requires `wasm-pack` (install: `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`).

## Architecture

- **`src/lib.rs`** — wasm-bindgen wrapper around `zenpredict::Model`,
  exports `parse_bake`, `forward_with_taps`, `layer_weights`. Reuses
  zenpredict's forward math for parity with the production hot path.
- **`tests/forward_parity.rs`** — golden test: per-stage forward
  output matches `Predictor::predict` within f32 SAXPY drift tolerance
  (1e-4 rel + 1e-5 abs) across all 4 shipped zensim bakes.
- **`web/main.js`** — UI entry, panel routing, file/URL load, WASM init.
- **`web/parser.js`** — (Stage 2) standalone JS parser, currently unused;
  WASM does all parsing.
- **`web/feature_layout.js`** — zensim 228 / 300 / 372-feature schema
  mapping `f<idx>` → `s<scale>.<channel>.<feature_name>`.
- **`web/panels/*.js`** — one file per panel (summary, scaler,
  importance, forward).

## Roadmap

P1 next, per issue #79:
- Per-layer weight heatmap (panel 5).
- Output calibration view (panel 6) — PCHIP spline + tanh-pin + per-codec
  curves. Needs JS-side TLV decoders for the calibration metadata blobs.
- Bake comparison mode (panel 7) — load N bakes, diff scaler/importance/
  weights/calibration side by side.
