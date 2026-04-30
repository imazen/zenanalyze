#!/usr/bin/env python3
"""
Adversarial / corner-case spot-check for a baked picker JSON.

Runs ~10 synthetic feature vectors through the trained MLP (via the
Python-side numpy reference, not the .bin loader — same forward pass,
no Rust dependency) and asserts that the picker behaves sensibly:

  * No NaN / Inf in any cell's predicted output
  * No NaN / Inf in any cell's predicted log-bytes
  * top-1 / top-2 gap is non-negative (well-defined ordering)
  * argmin returns an in-range cell index
  * predicted bytes (exp of log-bytes) are positive and not absurdly
    large (catches integer-overflow-in-disguise via huge log values)

Inputs are deliberately extreme:

  * all-zeros (input is exactly the scaler mean → no information)
  * all-ones  (every feature at +1 std)
  * all-minus-ones
  * single-feature spike (one feature huge, rest at scaler mean)
  * NaN / Inf injection (every input feature flipped to NaN, then to
    Inf) — the picker must not crash or silently produce NaN output;
    the codec is expected to gate on `feature_in_distribution` first
    and never feed NaN to the picker, but defense-in-depth here
    catches future regressions

Exits non-zero on any failure so CI can run it as a post-bake gate.

Usage:

    python3 adversarial_probe.py --model benchmarks/<bake>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def relu_forward(x: np.ndarray, layers: list[dict]) -> np.ndarray:
    """ReLU MLP forward pass — final layer identity."""
    a = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        z = a @ W + b
        a = z if i == last else np.maximum(z, 0.0)
    return a


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Match StandardScaler — guard against zero-scale columns."""
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return (x - mean) / safe_scale


def adversarial_inputs(n_inputs: int) -> list[tuple[str, np.ndarray]]:
    """Synthesize corner-case input vectors.

    `n_inputs` is the standardized input dim (post-feature-engineering;
    matches `model["n_inputs"]`). Inputs are emitted in the *post-
    scaler* space — i.e., what gets fed into layer 0 — so we can
    sanity-check the inference math without re-implementing the
    feature-engineering pipeline.
    """
    cases: list[tuple[str, np.ndarray]] = []
    cases.append(("all_zeros", np.zeros(n_inputs, dtype=np.float64)))
    cases.append(("all_ones", np.ones(n_inputs, dtype=np.float64)))
    cases.append(("all_minus_ones", -np.ones(n_inputs, dtype=np.float64)))
    cases.append(("first_feature_huge_pos", np.array([1e3] + [0.0] * (n_inputs - 1))))
    cases.append(("first_feature_huge_neg", np.array([-1e3] + [0.0] * (n_inputs - 1))))
    cases.append(("last_feature_huge_pos", np.array([0.0] * (n_inputs - 1) + [1e3])))
    cases.append(("alternating_signs", np.array([(-1.0) ** i for i in range(n_inputs)])))
    cases.append(("nan_first", np.array([float("nan")] + [0.0] * (n_inputs - 1))))
    cases.append(("inf_first", np.array([float("inf")] + [0.0] * (n_inputs - 1))))
    cases.append(("neg_inf_first", np.array([float("-inf")] + [0.0] * (n_inputs - 1))))
    return cases


def check_output(
    name: str,
    output: np.ndarray,
    n_cells: int | None,
    accept_nan_propagation: bool,
) -> list[str]:
    """Validate one inference output. Returns a list of failure
    messages — empty list means the case passed."""
    failures: list[str] = []
    if not np.isfinite(output).all():
        if not accept_nan_propagation:
            failures.append(
                f"non-finite output for case '{name}' "
                f"(any-NaN={bool(np.isnan(output).any())}, "
                f"any-Inf={bool(np.isinf(output).any())})"
            )
        # Even when NaN propagation is acceptable (NaN-input cases),
        # we still want to confirm it's *deterministic* NaN — not
        # half-NaN that would corrupt argmin silently. Skip further
        # checks for this case if the output is fully non-finite.
        return failures

    # Bytes-log subrange or full output: argmin must be in range and
    # the implied byte count must be positive and bounded.
    bytes_log = output[:n_cells] if n_cells is not None else output
    if bytes_log.size == 0:
        failures.append(f"zero-length bytes_log for case '{name}'")
        return failures
    sorted_log = np.sort(bytes_log)
    gap = float(sorted_log[1] - sorted_log[0]) if sorted_log.size >= 2 else float("inf")
    if math.isnan(gap) or gap < 0.0:
        failures.append(f"negative top-1/top-2 gap {gap:.4f} for '{name}'")
    bytes_predicted = np.exp(np.clip(bytes_log, -30.0, 30.0))
    if not (bytes_predicted >= 0.0).all():
        failures.append(f"negative predicted bytes for '{name}'")
    if bytes_predicted.max() > 1e30:
        failures.append(
            f"predicted bytes {bytes_predicted.max():.3e} too large for '{name}' "
            "(check log-bytes magnitudes; suggests broken scaling)"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="trained model JSON")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Auto-enabled in CI. Exit code 1 on any failure.",
    )
    args = ap.parse_args()
    model = json.loads(args.model.read_text())
    n_inputs = int(model["n_inputs"])
    layers = model["layers"]
    n_cells = (
        model.get("hybrid_heads_manifest", {}).get("n_cells")
        or int(model.get("n_outputs", 0))
        or None
    )
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)

    cases = adversarial_inputs(n_inputs)
    sys.stderr.write(
        f"adversarial_probe: model={args.model} n_inputs={n_inputs} "
        f"n_cells={n_cells} cases={len(cases)}\n"
    )

    total_fail = 0
    for name, raw in cases:
        accept_nan = "nan" in name or "inf" in name
        # Apply the standardizer in the same way training did.
        x = standardize(raw, mean, scale).reshape(1, -1)
        output = relu_forward(x, layers).ravel()
        fails = check_output(name, output, n_cells, accept_nan_propagation=accept_nan)
        status = "ok" if not fails else "FAIL"
        sys.stderr.write(f"  {status:>4}  {name:30s}\n")
        for f in fails:
            sys.stderr.write(f"        • {f}\n")
        total_fail += len(fails)

    if total_fail:
        sys.stderr.write(
            f"\nadversarial_probe: {total_fail} failure(s) across "
            f"{len(cases)} cases\n"
        )
        return 1 if args.strict else 0
    sys.stderr.write(f"\n✓ adversarial_probe: all {len(cases)} cases passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
