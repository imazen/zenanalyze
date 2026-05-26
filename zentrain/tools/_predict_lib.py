"""
Shared numpy forward-pass for baked picker / metapicker MLPs.

Sibling to `_picker_lib.py` (per-codec training pipeline) and
`_metapicker_lib.py` (cross-codec classifier scaffolding). Promoted
2026-05-26 as DEDUP-B2 (Tier-1 #6 second half): the metapicker module
already had a tight forward-pass for the v15 schema, but four other
inference-side tools each hand-rolled their own standardize-then-MLP
loop with slightly different field-name conventions. This module is
the canonical home for the numeric op.

What lives here:

- `forward(model_json, X) -> np.ndarray` — full forward pass over an
  MLP described by a baked JSON. Returns the **raw output logits**
  (shape `(n_samples, n_outputs)`). Callers that want argmax /
  argmin / softmax apply the appropriate reduction themselves.
- `forward_from_layers(X, mean, scale, layers, *, default_activation=...) -> np.ndarray`
  — same forward, but accepts the standardization params + a layer
  list directly. Used by `zerobias_rebake.py` which carries the
  layers in its parsed-bake representation rather than a JSON dict,
  and which needs to mirror i8 quantization on layers whose stored
  `dtype == 2`.

What does NOT live here (intentional):

- Argmax / argmin reductions — caller's choice (the metapicker takes
  argmax of logits; the per-codec picker takes argmin of `exp(pred)`
  in bytes-space).
- Feature engineering (one-hot size, log_px, polynomial terms,
  cross-terms) — that's the per-script `engineer()` step. Callers
  pre-engineer their feature vector before calling `forward`.
- Softmax — none of the current consumers compute softmax (they
  all argmax raw logits). Adding it is a one-liner if a future
  consumer needs probabilities.

Schema flexibility:

The five pre-extraction consumers used three different layer-dict
conventions, which `forward()` auto-detects:

1. `W` / `b` (sklearn coefs_ orientation; used by
   `_metapicker_lib`, `holdout_ab_lookup`, `student_permutation`,
   `v15_compare_pickers`).
2. `weights` / `biases` (flat list, with `in_dim` / `out_dim`
   metadata for reshape; used by `inspect_picker`).
3. `weights_f32` / `biases` (with optional `dtype == 2` flag that
   triggers i8 re-quantization mirroring the on-disk bake; used by
   `zerobias_rebake`).

Activation flexibility:

- Model-level string (`model["activation"]`) — used by `W/b` schemas.
  Values: `relu` (default), `leakyrelu` / `leaky_relu`, `identity`.
- Per-layer string (`layer["activation"]`) — used by `weights` /
  `weights_f32` schemas. Same value set, plus numeric codes
  `1` (ReLU) and `2` (LeakyReLU) used by `zerobias_rebake`.

The final layer's activation is always **identity** (regardless of
declared activation) — this matches every pre-extraction consumer
and the canonical ZNPR runtime.

Regression gate:

`test_predict_lib.py` carries verbatim copies of each pre-extraction
forward() impl and asserts `forward()` produces bit-identical (or
1-ULP for f32 chain) output to each on a hand-rolled tiny model
plus 30 random MLPs. If you change anything in this module, run
that test first — `python3 zentrain/tools/test_predict_lib.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


LEAKY_RELU_ALPHA: float = 0.01


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _leaky_relu(x: np.ndarray, slope: float = LEAKY_RELU_ALPHA) -> np.ndarray:
    return np.where(x > 0, x, slope * x)


def _resolve_activation(
    layer: dict, fallback: Any
) -> str:
    """Return the activation tag for `layer`, falling back to `fallback`.

    `fallback` is the model-level activation (string or None). Per-layer
    overrides win when present. Numeric codes (1, 2) are translated to
    their string equivalents.
    """
    a = layer.get("activation", fallback)
    if a is None:
        return "relu"
    if isinstance(a, int):
        # zerobias_rebake convention: 1 = ReLU, 2 = LeakyReLU,
        # anything else (0 / -1 / etc.) = identity. Mirror that exactly.
        if a == 1:
            return "relu"
        if a == 2:
            return "leakyrelu"
        return "identity"
    s = str(a).lower()
    if s in ("leakyrelu", "leaky_relu"):
        return "leakyrelu"
    if s == "identity":
        return "identity"
    return "relu"


def _apply_activation(x: np.ndarray, tag: str) -> np.ndarray:
    if tag == "leakyrelu":
        return _leaky_relu(x)
    if tag == "identity":
        return x
    return _relu(x)


def _extract_weights(
    layer: dict, dtype: np.dtype = np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Pull `(W, b)` out of `layer`, supporting all three schemas.

    Returns arrays of `dtype` (f32 default; f64 supported for parity
    with `holdout_ab_lookup`'s pre-extraction f64 chain). Mirrors
    `zerobias_rebake`'s i8 re-quantization when `layer["dtype"] == 2`
    (that path stays f32 — the re-quant is an integer rounding step
    that does not benefit from f64).
    """
    if "W" in layer and "b" in layer:
        W = np.asarray(layer["W"], dtype=dtype)
        b = np.asarray(layer["b"], dtype=dtype)
        return W, b
    if "weights_f32" in layer:
        W = np.asarray(layer["weights_f32"], dtype=np.float32)
        b = np.asarray(layer["biases"], dtype=np.float32)
        if layer.get("dtype") == 2:
            # i8 re-quantization mirroring zerobias_rebake.forward.
            max_col = np.max(np.abs(W), axis=0)
            scales = np.where(max_col == 0, 1.0, max_col / 127.0)
            q = np.clip(np.round(W / scales[None, :]), -128, 127)
            W = q * scales[None, :]
        return W, b
    if "weights" in layer:
        W = np.asarray(layer["weights"], dtype=dtype)
        if "in_dim" in layer and "out_dim" in layer:
            W = W.reshape(int(layer["in_dim"]), int(layer["out_dim"]))
        b = np.asarray(layer["biases"], dtype=dtype)
        return W, b
    raise KeyError(
        "_predict_lib._extract_weights: layer dict must carry "
        "`W`/`b`, `weights`/`biases`, or `weights_f32`/`biases`; "
        f"got keys {sorted(layer.keys())}"
    )


def forward_from_layers(
    X: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    layers: list[dict],
    *,
    default_activation: Any = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Forward pass given pre-extracted standardization + layer list.

    Args:
        X: `(n_samples, n_inputs)` or `(n_inputs,)` input matrix /
           vector. Cast to `dtype` internally.
        mean / scale: standardization params, broadcast to X.
        layers: list of layer dicts (see `_extract_weights` for accepted
            schemas).
        default_activation: model-level activation fallback (string or
            None). Per-layer `activation` field overrides this.
        dtype: working dtype for the standardize-then-MLP chain
            (defaults to f32 matching the canonical bake / runtime).
            `holdout_ab_lookup` pre-DEDUP used f64; pass
            `dtype=np.float64` there to preserve bit-parity.

    Returns:
        Raw output logits, shape `(n_samples, n_outputs)` for matrix
        input, `(n_outputs,)` for vector input. The final layer's
        activation is always identity.
    """
    mean = np.asarray(mean, dtype=dtype)
    scale = np.asarray(scale, dtype=dtype)
    vec_input = X.ndim == 1
    Z = np.asarray(X, dtype=dtype)
    if vec_input:
        Z = (Z - mean) / scale
    else:
        Z = (Z - mean[None, :]) / scale[None, :]
    last_idx = len(layers) - 1
    for i, layer in enumerate(layers):
        W, b = _extract_weights(layer, dtype=dtype)
        Z = Z @ W + b
        if i == last_idx:
            continue
        tag = _resolve_activation(layer, default_activation)
        Z = _apply_activation(Z, tag)
    return Z


def forward(
    model_json: dict, X: np.ndarray, *, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Numpy forward pass for a baked picker / metapicker MLP JSON.

    Args:
        model_json: dict with keys `scaler_mean`, `scaler_scale`,
            `layers` (and optionally `activation` for the per-layer
            fallback). See module docstring for accepted layer schemas.
        X: `(n_samples, n_inputs)` or `(n_inputs,)` input. Cast to
            `dtype` (f32 default).
        dtype: working dtype (see `forward_from_layers`).

    Returns:
        Raw output logits (shape `(n_samples, n_outputs)` for matrix
        input, `(n_outputs,)` for vector input). Apply argmax / argmin
        / softmax in the caller as needed.
    """
    mean = model_json["scaler_mean"]
    scale = model_json["scaler_scale"]
    layers = model_json["layers"]
    default_act = model_json.get("activation")
    return forward_from_layers(
        X, mean, scale, layers, default_activation=default_act, dtype=dtype
    )


__all__ = [
    "LEAKY_RELU_ALPHA",
    "forward",
    "forward_from_layers",
]
