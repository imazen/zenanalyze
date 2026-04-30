#!/usr/bin/env python3
"""
Round-trip check for `bake_picker.py`.

1. Bakes the input model JSON into a `.bin` via `bake_picker.py`.
2. Runs the Rust `load_baked_model` example to produce the
   inference output for a deterministic input vector.
3. Computes the same forward pass in Python (numpy) using the input
   JSON.
4. Compares row-by-row; fails if the maximum absolute difference
   exceeds a tight threshold.

This is the only way to guarantee the binary format and the Rust
loader are in agreement with the bake script — without it, format
bugs land silently and bake-vs-load mismatches surface as bad codec
picks at runtime.

Usage:
    python3 tools/bake_roundtrip_check.py \
        --model benchmarks/zq_bytes_distill_2026-04-29.json
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BAKE = REPO_ROOT / "tools" / "bake_picker.py"
EXAMPLE = "load_baked_model"
# f32 inference reorders sums differently between numpy (cache-blocked
# matmul) and the Rust input-major SAXPY, so two correct
# implementations can disagree at the unit-in-last-place level. Use
# relative tolerance plus an absolute floor for values near zero.
RTOL_F32 = 1e-5
ATOL_F32 = 1e-4
RTOL_F16 = 5e-3  # f16 quantization budget — ~3 mantissa decimal digits
ATOL_F16 = 1e-3
# i8 per-output-neuron quantization gives ~1 % relative RMS per layer;
# composed across the v2.1 4-layer MLP that's still well under the
# bytes_log magnitudes the picker emits (~75–80). Be generous with
# atol because the absolute-magnitude floor matters more here than
# the relative ratio.
RTOL_I8 = 2e-2
ATOL_I8 = 1.0


LEAKY_RELU_ALPHA = 0.01  # Matches zenpredict::model::LEAKY_RELU_ALPHA


def python_forward(model: dict, features: np.ndarray, dtype: str) -> np.ndarray:
    """Reference forward pass — must match Rust output exactly.

    Mirrors `zenpredict::inference::forward`:
      x' = (x - mean) / scale
      for each layer: x = activation(x @ W + b)
      last layer always uses Identity.
    """
    mean = np.array(model["scaler_mean"], dtype=np.float32)
    scale = np.array(model["scaler_scale"], dtype=np.float32)
    x = (features - mean) / scale

    activation = (
        model.get("activation", "relu").replace("-", "_").replace(" ", "_").lower()
    )
    layers = model["layers"]
    last_idx = len(layers) - 1
    for i, layer in enumerate(layers):
        W = np.array(layer["W"], dtype=np.float32)
        b = np.array(layer["b"], dtype=np.float32)
        if dtype == "f16":
            W = W.astype(np.float16).astype(np.float32)
        elif dtype == "i8":
            # Per-output-neuron quantize-then-dequant, matching the
            # Rust bake side.
            abs_max = np.abs(W).max(axis=0)
            scales = np.where(abs_max > 0.0, abs_max / 127.0, 1.0).astype(np.float32)
            q = np.round(W / scales).clip(-128, 127).astype(np.int8)
            W = q.astype(np.float32) * scales[np.newaxis, :]
        x = x @ W + b
        if i == last_idx:
            continue
        if activation == "relu":
            x = np.maximum(x, 0.0)
        elif activation in ("leakyrelu", "leaky_relu"):
            x = np.where(x >= 0.0, x, LEAKY_RELU_ALPHA * x)
        elif activation == "identity":
            pass
        else:
            raise SystemExit(f"unsupported activation in reference forward: {activation!r}")
    return x


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16", "i8"])
    args = ap.parse_args(argv)

    if not BAKE.exists():
        sys.exit(f"missing {BAKE}")
    if not args.model.exists():
        sys.exit(f"missing {args.model}")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "model.bin"
        # Step 1: bake.
        subprocess.run(
            [
                sys.executable,
                str(BAKE),
                "--model",
                str(args.model),
                "--out",
                str(out),
                "--dtype",
                args.dtype,
                # Round-trip is a math check, not a publish gate.
                # If the model JSON's safety_report.passed is false,
                # bake_picker would refuse without this. The runtime
                # is still correct to test even when the bake itself
                # is flagged.
                "--allow-unsafe",
                # Round-trip math check; no consumer reads the
                # legacy sibling manifest in this flow.
                "--no-manifest",
            ],
            check=True,
        )
        if not out.exists():
            sys.exit("bake produced no output")

        # Step 2: build + run the Rust example. f16 is always-on in
        # the runtime — no cargo feature toggle.
        rust_proc = subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "-q",
                "--manifest-path",
                str(REPO_ROOT / "Cargo.toml"),
                "-p",
                "zenpredict",
                "--example",
                EXAMPLE,
                "--",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if rust_proc.returncode != 0:
            sys.stderr.write(rust_proc.stderr)
            sys.exit(f"rust example exit {rust_proc.returncode}")
        sys.stderr.write(rust_proc.stderr)

        rust_out = np.array(
            [float(x) for x in rust_proc.stdout.split() if x.strip()],
            dtype=np.float32,
        )

    # Step 3: Python reference.
    model = json.loads(args.model.read_text())
    n_in = int(model["n_inputs"])
    features = np.array(
        [math.sin(i * 0.1) for i in range(n_in)],
        dtype=np.float32,
    )
    py_out = python_forward(model, features, args.dtype)

    # Step 4: compare.
    if rust_out.shape != py_out.shape:
        sys.exit(f"shape mismatch: rust {rust_out.shape} != python {py_out.shape}")

    abs_diff = np.abs(rust_out - py_out)
    if args.dtype == "i8":
        rtol, atol = RTOL_I8, ATOL_I8
    elif args.dtype == "f16":
        rtol, atol = RTOL_F16, ATOL_F16
    else:
        rtol, atol = RTOL_F32, ATOL_F32
    # Tolerance per element: rtol * max(|rust|, |python|) + atol.
    scale = np.maximum(np.abs(rust_out), np.abs(py_out))
    tol = rtol * scale + atol
    rel_err = abs_diff / np.maximum(scale, 1e-30)
    max_diff = float(abs_diff.max())
    max_rel = float(rel_err.max())

    print(f"dtype={args.dtype}  shape={rust_out.shape}")
    print(f"max abs diff: {max_diff:.6g}  max rel diff: {max_rel:.6g}")
    print(f"thresholds: rtol={rtol:g} atol={atol:g}")
    bad_idx = np.where(abs_diff > tol)[0]
    if bad_idx.size > 0:
        for i in bad_idx[:8]:
            print(
                f"  out[{i}]: rust={rust_out[i]:+.6g}  python={py_out[i]:+.6g}  "
                f"diff={abs_diff[i]:+.6g} rel={rel_err[i]:.3g}"
            )
        sys.exit(f"ROUND-TRIP FAILED ({bad_idx.size}/{abs_diff.size} outputs out of tolerance)")
    print("round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
