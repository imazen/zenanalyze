#!/usr/bin/env python3
"""
Regression test for the bake_picker → zenpredict-bake → loader →
forward-pass round-trip. Builds a small synthetic sklearn-style
model JSON in-memory, bakes it via `bake_picker.py`, runs the Rust
`load_baked_model` example, and compares against the numpy reference.

Run:
    python3 tools/test_bake_roundtrip.py

Exits 0 on success, non-zero with a diagnostic on failure.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BAKE = REPO_ROOT / "tools" / "bake_picker.py"
ROUNDTRIP = REPO_ROOT / "tools" / "bake_roundtrip_check.py"


def synth_model(activation: str, n_in: int = 5, n_hidden: int = 8, n_out: int = 4) -> dict:
    """A tiny but non-degenerate model: random-ish weights with a
    seeded RNG so the round-trip is reproducible.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    return {
        "n_inputs": n_in,
        "n_outputs": n_out,
        "feat_cols": [f"feat_{i}" for i in range(n_in)],
        "scaler_mean":  [0.0] * n_in,
        "scaler_scale": [1.0] * n_in,
        "activation": activation,
        "schema_version_tag": "zenpredict.v2.test",
        "layers": [
            {
                "W": rng.standard_normal((n_in, n_hidden)).astype("float32").tolist(),
                "b": rng.standard_normal(n_hidden).astype("float32").tolist(),
            },
            {
                "W": rng.standard_normal((n_hidden, n_out)).astype("float32").tolist(),
                "b": rng.standard_normal(n_out).astype("float32").tolist(),
            },
        ],
    }


def run_one(activation: str, dtype: str) -> None:
    print(f"--- activation={activation} dtype={dtype}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model_json = td / "model.json"
        model_json.write_text(json.dumps(synth_model(activation)))
        rc = subprocess.call(
            [
                sys.executable,
                str(ROUNDTRIP),
                "--model",
                str(model_json),
                "--dtype",
                dtype,
            ]
        )
        if rc != 0:
            sys.exit(f"round-trip failed for activation={activation} dtype={dtype}")


def main() -> int:
    if not BAKE.exists() or not ROUNDTRIP.exists():
        sys.exit(f"missing scripts under {REPO_ROOT}/tools/")
    # Build the bake binary and the example up front so the
    # round-trip script doesn't pay cargo cold-build per call.
    print("building zenpredict-bake + load_baked_model example…")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-q",
            "--manifest-path",
            str(REPO_ROOT / "Cargo.toml"),
            "-p",
            "zenpredict",
            "--bin",
            "zenpredict-bake",
            "--example",
            "load_baked_model",
        ],
        check=True,
    )
    for activation in ("relu", "leakyrelu", "identity"):
        for dtype in ("f32", "f16", "i8"):
            run_one(activation, dtype)
    print("\nALL ROUND-TRIPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
