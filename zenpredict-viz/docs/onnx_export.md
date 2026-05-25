# znpr2onnx — ZNPR v3 → ONNX exporter

Native binary that converts a ZNPR v3 bake into an ONNX file readable
by [Netron](https://netron.app/), [ONNX Runtime](https://onnxruntime.ai/),
and any other ONNX-aware tooling.

## Build

```sh
cargo build --release --bin znpr2onnx --features onnx-export -p zenpredict-viz
```

## Usage

```sh
./target/release/znpr2onnx <input.bin> <output.onnx> [--name NAME]
```

## Scope

The ONNX represents the **standardize + MLP rank head** only:

- `Sub(features, scaler_mean)` → `Mul(_, 1/scaler_scale)` → standardized
- For each layer: `Gemm(prev, W, B)` → activation (`Relu` / `LeakyRelu`
  with alpha=0.01 / `Identity`).
- Final layer output is the raw MLP scalar (or hidden vector for the
  tuner-head bakes).

**NOT exported:**

- `zentrain.feature_transforms` — only identity transforms are
  representable; non-trivial transforms (e.g., log1p with per-feature
  parameters) would need ONNX `Function`s and a custom decoding step.
- `zentrain.tanh_output_head` — `100·σ(x/scale)` is trivial in ONNX but
  the user-side intent of this stage is to swap calibration, not bake
  in a fixed scalar. Round-trip fidelity for the post-MLP stages is
  better served by re-implementing them next to whatever runtime hosts
  the ONNX.
- `zentrain.output_calibration_spline` — PCHIP knots; would require
  per-segment piecewise `Where`+`Mul`+`Add` nodes that nobody else
  recognizes as a spline.
- `zentrain.per_codec_calibration` — runtime affine selection by
  codec hint; the codec id isn't an input feature, so ONNX has no
  clean signal route.
- `zentrain.per_sample_alpha_head` — multi-head architecture (rank
  head + pool head + sigmoid gate); the exporter currently emits the
  MLP backbone but does not assemble the alpha head's branching.

When a bake carries any of those stages, the binary prints a notice
to stderr listing which stages were dropped.

## Verification

The exporter is verified against `zenpredict::Predictor::predict` by
checking that ONNX Runtime and zenpredict produce the same scalar on a
sine-input feature vector, within f32 SAXPY drift tolerance
(`abs(a-b) ≤ max(|a|,|b|) · 1e-4 + 1e-5`). For `v0_18` on sine input:

| Runtime | Output |
|---|---|
| `zenpredict::Predictor` | `2984.1653` |
| ONNX Runtime (Python) | `2984.15771484375` |
| Δ | `~0.0076` (relative `~3e-6`) |

The drift is consistent with different SAXPY accumulator orders
between zenpredict's `saxpy_matmul_*` and ONNX Runtime's `Gemm`.

## Netron rendering

Upload the `.onnx` to <https://netron.app/> (no install) or run
desktop Netron. The expected graph for `v_tuner_v11`:

```
features [1, 372]
   │
   ├── Sub ── scaler_mean
   │     └── Mul ── 1/scaler_scale ── standardized
   │                  │
   │                  └── Gemm(W_0, B_0) ── LeakyRelu ── post_0
   │                                                       │
   │                                                       └── Gemm(W_1, B_1) ── Identity ── output [1, 128]
```

Tuner-head bakes have `output [1, 128]` (hidden width); the alpha head
mix happens at runtime and isn't expressed in the ONNX.

## Opset

IR 8, opset 13 — broadly compatible with Netron and ONNX Runtime ≥ 1.13.
Ops used: `Sub`, `Mul`, `Gemm`, `Identity`, `Relu`, `LeakyRelu`.
