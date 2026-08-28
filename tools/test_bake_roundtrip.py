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


def synth_rd_time_model(activation: str, n_cells: int = 2) -> dict:
    """An `--objective rd_time` (zenanalyze#56) shaped model: output
    layout `[bytes_log × n_cells, time_log × n_cells]` plus the
    time-head diagnostics `bake_picker.py` turns into
    `zentrain.hybrid_heads_layout` (head kinds [0, 2]),
    `zentrain.median_cell_ms_per_mp` and `zentrain.encode_ms_p99`.
    The forward pass is what the round-trip checks; the metadata is
    what must not break the Rust baker (a None in `encode_ms_p99`
    becomes the -1.0 sentinel, never NaN)."""
    m = synth_model(activation, n_out=2 * n_cells)
    m.update({
        "safety_profile": "rd_time",
        "hybrid_heads_manifest": {
            "n_cells": n_cells,
            "categorical_axes": ["mode"],
            "scalar_axes": [],
            "output_layout": {"bytes_log": [0, n_cells], "time_log": [n_cells, 2 * n_cells]},
        },
        "training_objective": {
            "name": "rd_time", "has_time_head": True, "time_column": "encode_ms",
            "time_loss_weight": 0.5, "median_cell_ms_per_mp": 136.2,
        },
        "safety_report": {
            "passed": True, "violations": [],
            "diagnostics": {
                "median_cell_ms_per_mp": 136.2,
                "encode_ms_p99": {"50": [12.0, 30.5], "80": [15.0, None]},
            },
        },
    })
    return m


def run_one(activation: str, dtype: str, shape: str = "plain") -> None:
    print(f"--- activation={activation} dtype={dtype} shape={shape}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model_json = td / "model.json"
        model = synth_model(activation) if shape == "plain" else synth_rd_time_model(activation)
        model_json.write_text(json.dumps(model))
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
            sys.exit(f"round-trip failed for activation={activation} dtype={dtype} shape={shape}")


def cargo_build_cmds(repo_root: Path = REPO_ROOT) -> list[list[str]]:
    """The cargo invocations that stage the round-trip's Rust halves.

    Two builds, one per package: the `zenpredict-bake` binary lives in the
    `zenpredict-bake` package while the `load_baked_model` example lives in
    `zenpredict`. A single `-p zenpredict --bin zenpredict-bake` was the
    pre-#85 form and fails with "no bin target named `zenpredict-bake` in
    `zenpredict` package" — `test_cargo_invocations.py` pins each
    (package, target) pair to the workspace manifests.
    """
    manifest = str(repo_root / "Cargo.toml")
    common = ["cargo", "build", "--release", "-q", "--manifest-path", manifest]
    return [
        common + ["-p", "zenpredict-bake", "--bin", "zenpredict-bake"],
        common + ["-p", "zenpredict", "--example", "load_baked_model"],
    ]


def main() -> int:
    if not BAKE.exists() or not ROUNDTRIP.exists():
        sys.exit(f"missing scripts under {REPO_ROOT}/tools/")
    # Build the bake binary and the example up front so the
    # round-trip script doesn't pay cargo cold-build per call.
    print("building zenpredict-bake + load_baked_model example…")
    for cmd in cargo_build_cmds():
        subprocess.run(cmd, check=True)
    for activation in ("relu", "leakyrelu", "identity"):
        for dtype in ("f32", "f16", "i8"):
            for shape in ("plain", "rd_time"):
                run_one(activation, dtype, shape)
    print("\nALL ROUND-TRIPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
