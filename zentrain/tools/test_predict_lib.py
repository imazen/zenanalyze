"""Smoke / parity tests for `_predict_lib`.

Validates `_predict_lib.forward` against verbatim copies of each
pre-extraction forward() impl on tiny hand-rolled models plus 30
random MLPs per schema. Bit-identical (f32 chain) is the floor.

Run with: `python3 -m pytest zentrain/tools/test_predict_lib.py`
or `python3 zentrain/tools/test_predict_lib.py` (calls every
test_* function directly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _predict_lib import (  # noqa: E402
    LEAKY_RELU_ALPHA,
    forward,
    forward_from_layers,
)


# ---------------------------------------------------------------------------
# Reference impls — verbatim copies of each pre-extraction forward().
# These are the regression gates: if _predict_lib.forward ever diverges
# from any of these, the corresponding test fails.

def _forward_inspect_picker_vector(model: dict, features: np.ndarray) -> np.ndarray:
    """Verbatim copy of zentrain/tools/inspect_picker.py:125-146."""
    mean = np.asarray(model["scaler_mean"], dtype=np.float32)
    scale = np.asarray(model["scaler_scale"], dtype=np.float32)
    x = (features.astype(np.float32) - mean) / scale
    last_idx = len(model["layers"]) - 1
    for i, layer in enumerate(model["layers"]):
        in_dim = layer["in_dim"]
        out_dim = layer["out_dim"]
        w = np.asarray(layer["weights"], dtype=np.float32).reshape(in_dim, out_dim)
        b = np.asarray(layer["biases"], dtype=np.float32)
        x = x @ w + b
        if i == last_idx:
            continue
        act = layer["activation"]
        if act == "relu":
            x = np.maximum(x, 0.0)
        elif act == "leakyrelu":
            x = np.where(x >= 0.0, x, LEAKY_RELU_ALPHA * x)
    return x


def _forward_inspect_picker_batch(model: dict, X: np.ndarray) -> np.ndarray:
    """Verbatim copy of zentrain/tools/inspect_picker.py:149-169."""
    mean = np.asarray(model["scaler_mean"], dtype=np.float32)
    scale = np.asarray(model["scaler_scale"], dtype=np.float32)
    x = (X.astype(np.float32) - mean[None, :]) / scale[None, :]
    last_idx = len(model["layers"]) - 1
    for i, layer in enumerate(model["layers"]):
        in_dim = layer["in_dim"]
        out_dim = layer["out_dim"]
        w = np.asarray(layer["weights"], dtype=np.float32).reshape(in_dim, out_dim)
        b = np.asarray(layer["biases"], dtype=np.float32)
        x = x @ w + b
        if i == last_idx:
            continue
        act = layer["activation"]
        if act == "relu":
            x = np.maximum(x, 0.0)
        elif act == "leakyrelu":
            x = np.where(x >= 0.0, x, LEAKY_RELU_ALPHA * x)
    return x


def _forward_holdout_ab_lookup(model_json, x):
    """Verbatim copy of tools/holdout_ab_lookup.py:54-74."""
    h = (np.asarray(x, dtype=np.float64) - np.asarray(model_json["scaler_mean"])) / np.asarray(model_json["scaler_scale"])
    layers = model_json["layers"]
    activation = model_json.get("activation", "relu")
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"])
        b = np.asarray(layer["b"])
        h = h @ W + b
        if i < len(layers) - 1:
            if activation == "relu":
                h = np.maximum(h, 0.0)
            elif activation == "identity":
                pass
            else:
                raise ValueError(f"unsupported activation {activation}")
    return h


def _forward_zerobias_rebake(features, mean, scale, layers):
    """Verbatim copy of zentrain/tools/zerobias_rebake.py:306-321."""
    x = (features - mean[None, :]) / scale[None, :]
    for i, layer in enumerate(layers):
        w = layer["weights_f32"]
        if layer["dtype"] == 2:
            max_col = np.max(np.abs(w), axis=0)
            scales = np.where(max_col == 0, 1.0, max_col / 127.0)
            q = np.clip(np.round(w / scales[None, :]), -128, 127)
            w = q * scales[None, :]
        x = x @ w + layer["biases"][None, :]
        if layer["activation"] == 2 and i < len(layers) - 1:
            x = np.where(x > 0, x, x * 0.01)
        elif layer["activation"] == 1 and i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def _forward_student_permutation(model_json: dict, X: np.ndarray) -> np.ndarray:
    """Distilled from zentrain/tools/student_permutation.py:265-283.

    student_permutation wraps forward in a closure that does
    feature-engineering before the standardize-then-MLP chain. The
    standardize+MLP body itself (lines 272-282) is what we mirror;
    callers do the engineering BEFORE calling forward().
    """
    activation = model_json["activation"]
    if activation not in ("relu", "leakyrelu"):
        raise ValueError(f"unsupported activation {activation}")
    leaky_slope = 0.01
    scaler_mean = np.asarray(model_json["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(model_json["scaler_scale"], dtype=np.float32)
    Ws = [np.asarray(l["W"], dtype=np.float32) for l in model_json["layers"]]
    bs = [np.asarray(l["b"], dtype=np.float32) for l in model_json["layers"]]
    out_rows = []
    for row in X:
        x = (np.asarray(row, dtype=np.float32) - scaler_mean) / scaler_scale
        for i, (W, b) in enumerate(zip(Ws, bs)):
            x = x @ W + b
            if i < len(Ws) - 1:
                if activation == "leakyrelu":
                    x = np.where(x > 0, x, leaky_slope * x)
                else:
                    x = np.maximum(x, 0)
        out_rows.append(x)
    return np.stack(out_rows)


# ---------------------------------------------------------------------------
# Tests.

def _rand_wb_model(rng: np.random.Generator, n_in: int, n_hidden: int, n_out: int,
                   activation: str) -> dict:
    """Random W/b-shape model (used by holdout_ab_lookup, metapicker,
    v15_compare, student_permutation post-engineering)."""
    return {
        "scaler_mean": rng.standard_normal(n_in).tolist(),
        "scaler_scale": (rng.standard_normal(n_in) ** 2 + 0.5).tolist(),
        "activation": activation,
        "layers": [
            {"W": rng.standard_normal((n_in, n_hidden)).tolist(),
             "b": rng.standard_normal(n_hidden).tolist()},
            {"W": rng.standard_normal((n_hidden, n_out)).tolist(),
             "b": rng.standard_normal(n_out).tolist()},
        ],
    }


def _rand_weights_model(rng: np.random.Generator, n_in: int, n_hidden: int,
                        n_out: int, activation: str) -> dict:
    """Random `weights/biases` schema (inspect_picker)."""
    layers = []
    for in_d, out_d in [(n_in, n_hidden), (n_hidden, n_out)]:
        layers.append({
            "in_dim": in_d,
            "out_dim": out_d,
            "weights": rng.standard_normal((in_d, out_d)).flatten().tolist(),
            "biases": rng.standard_normal(out_d).tolist(),
            "activation": activation,
        })
    return {
        "scaler_mean": rng.standard_normal(n_in).tolist(),
        "scaler_scale": (rng.standard_normal(n_in) ** 2 + 0.5).tolist(),
        "n_inputs": n_in,
        "n_outputs": n_out,
        "layers": layers,
    }


def _rand_weights_f32_model(rng: np.random.Generator, n_in: int, n_hidden: int,
                            n_out: int, activation_code: int, dtype_code: int) -> dict:
    """Random `weights_f32/biases` schema (zerobias_rebake)."""
    layers = []
    for in_d, out_d in [(n_in, n_hidden), (n_hidden, n_out)]:
        layers.append({
            "weights_f32": rng.standard_normal((in_d, out_d)).astype(np.float32),
            "biases": rng.standard_normal(out_d).astype(np.float32),
            "activation": activation_code,
            "dtype": dtype_code,
        })
    return {
        "scaler_mean": rng.standard_normal(n_in).astype(np.float32),
        "scaler_scale": (rng.standard_normal(n_in) ** 2 + 0.5).astype(np.float32),
        "layers": layers,
    }


# ----- inspect_picker -----

def test_inspect_picker_vector_relu():
    rng = np.random.default_rng(0xA1)
    for _ in range(30):
        m = _rand_weights_model(rng, n_in=12, n_hidden=20, n_out=5, activation="relu")
        x = rng.standard_normal(12).astype(np.float32)
        ours = forward(m, x)
        theirs = _forward_inspect_picker_vector(m, x)
        assert np.array_equal(ours, theirs), \
            f"vec relu mismatch: max|diff|={np.max(np.abs(ours - theirs))}"


def test_inspect_picker_vector_leakyrelu():
    rng = np.random.default_rng(0xA2)
    for _ in range(30):
        m = _rand_weights_model(rng, 8, 16, 4, "leakyrelu")
        x = rng.standard_normal(8).astype(np.float32)
        ours = forward(m, x)
        theirs = _forward_inspect_picker_vector(m, x)
        assert np.array_equal(ours, theirs)


def test_inspect_picker_batch_relu():
    rng = np.random.default_rng(0xA3)
    for _ in range(30):
        m = _rand_weights_model(rng, 10, 24, 6, "relu")
        X = rng.standard_normal((25, 10)).astype(np.float32)
        ours = forward(m, X)
        theirs = _forward_inspect_picker_batch(m, X)
        assert np.array_equal(ours, theirs)


def test_inspect_picker_batch_leakyrelu():
    rng = np.random.default_rng(0xA4)
    for _ in range(30):
        m = _rand_weights_model(rng, 9, 17, 3, "leakyrelu")
        X = rng.standard_normal((40, 9)).astype(np.float32)
        ours = forward(m, X)
        theirs = _forward_inspect_picker_batch(m, X)
        assert np.array_equal(ours, theirs)


# ----- holdout_ab_lookup -----

def test_holdout_ab_lookup_relu_f64_bit_identical():
    """Holdout uses f64 internally; pass dtype=np.float64 and assert
    bit-identical output."""
    rng = np.random.default_rng(0xB1)
    for _ in range(30):
        m = _rand_wb_model(rng, 16, 32, 6, "relu")
        x = rng.standard_normal(16)
        ours = forward(m, x, dtype=np.float64)
        theirs = _forward_holdout_ab_lookup(m, x)
        assert np.array_equal(ours, theirs), \
            f"holdout relu f64 max|diff|={np.max(np.abs(ours - theirs))}"


def test_holdout_ab_lookup_identity_f64_bit_identical():
    rng = np.random.default_rng(0xB2)
    for _ in range(10):
        m = _rand_wb_model(rng, 8, 12, 3, "identity")
        x = rng.standard_normal(8)
        ours = forward(m, x, dtype=np.float64)
        theirs = _forward_holdout_ab_lookup(m, x)
        assert np.array_equal(ours, theirs)


# ----- zerobias_rebake -----

def test_zerobias_rebake_relu_f32():
    """activation=1 (ReLU), dtype=1 (f32, no requant)."""
    rng = np.random.default_rng(0xC1)
    for _ in range(30):
        m = _rand_weights_f32_model(rng, 14, 22, 5, activation_code=1, dtype_code=1)
        X = rng.standard_normal((20, 14)).astype(np.float32)
        # Reference takes (features, mean, scale, layers) directly.
        theirs = _forward_zerobias_rebake(
            X, m["scaler_mean"], m["scaler_scale"], m["layers"]
        )
        ours = forward_from_layers(
            X, m["scaler_mean"], m["scaler_scale"], m["layers"]
        )
        assert np.array_equal(ours, theirs), \
            f"zerobias relu/f32 max|diff|={np.max(np.abs(ours - theirs))}"


def test_zerobias_rebake_leakyrelu_i8():
    """activation=2 (LeakyReLU), dtype=2 (triggers i8 re-quant)."""
    rng = np.random.default_rng(0xC2)
    for _ in range(30):
        m = _rand_weights_f32_model(rng, 11, 19, 4, activation_code=2, dtype_code=2)
        X = rng.standard_normal((15, 11)).astype(np.float32)
        theirs = _forward_zerobias_rebake(
            X, m["scaler_mean"], m["scaler_scale"], m["layers"]
        )
        ours = forward_from_layers(
            X, m["scaler_mean"], m["scaler_scale"], m["layers"]
        )
        assert np.array_equal(ours, theirs), \
            f"zerobias leaky/i8 max|diff|={np.max(np.abs(ours - theirs))}"


def test_zerobias_rebake_identity():
    """activation=0 (identity / no-op), dtype=1."""
    rng = np.random.default_rng(0xC3)
    m = _rand_weights_f32_model(rng, 6, 10, 3, activation_code=0, dtype_code=1)
    X = rng.standard_normal((5, 6)).astype(np.float32)
    theirs = _forward_zerobias_rebake(
        X, m["scaler_mean"], m["scaler_scale"], m["layers"]
    )
    ours = forward_from_layers(
        X, m["scaler_mean"], m["scaler_scale"], m["layers"]
    )
    assert np.array_equal(ours, theirs)


# ----- student_permutation -----

def test_student_permutation_relu():
    rng = np.random.default_rng(0xD1)
    for _ in range(30):
        m = _rand_wb_model(rng, 12, 24, 5, "relu")
        X = rng.standard_normal((10, 12)).astype(np.float32)
        ours = forward(m, X)
        theirs = _forward_student_permutation(m, X)
        assert np.array_equal(ours, theirs), \
            f"student relu max|diff|={np.max(np.abs(ours - theirs))}"


def test_student_permutation_leakyrelu():
    rng = np.random.default_rng(0xD2)
    for _ in range(30):
        m = _rand_wb_model(rng, 10, 16, 4, "leakyrelu")
        X = rng.standard_normal((20, 10)).astype(np.float32)
        ours = forward(m, X)
        theirs = _forward_student_permutation(m, X)
        assert np.array_equal(ours, theirs)


# ----- metapicker_lib.forward_metapicker still works (re-export) -----

def test_metapicker_forward_argmax_unchanged():
    """After _predict_lib lands, _metapicker_lib.forward_metapicker
    should still return argmax indices."""
    from _metapicker_lib import forward_metapicker
    rng = np.random.default_rng(0xE1)
    m = _rand_wb_model(rng, 20, 64, 5, "leakyrelu")
    X = rng.standard_normal((30, 20)).astype(np.float32)
    pred = forward_metapicker(m, X)
    # argmax → indices, shape (n_samples,), dtype intp.
    assert pred.shape == (30,)
    assert pred.dtype.kind == "i"
    # Each index is in range.
    assert int(pred.min()) >= 0 and int(pred.max()) < 5


def test_metapicker_matches_predict_lib_argmax():
    """forward_metapicker == argmax(forward) on the same model."""
    from _metapicker_lib import forward_metapicker
    rng = np.random.default_rng(0xE2)
    m = _rand_wb_model(rng, 16, 32, 6, "leakyrelu")
    X = rng.standard_normal((15, 16)).astype(np.float32)
    via_metapicker = forward_metapicker(m, X)
    via_predict_lib = np.argmax(forward(m, X), axis=1)
    assert np.array_equal(via_metapicker, via_predict_lib)


if __name__ == "__main__":
    test_inspect_picker_vector_relu()
    test_inspect_picker_vector_leakyrelu()
    test_inspect_picker_batch_relu()
    test_inspect_picker_batch_leakyrelu()
    test_holdout_ab_lookup_relu_f64_bit_identical()
    test_holdout_ab_lookup_identity_f64_bit_identical()
    test_zerobias_rebake_relu_f32()
    test_zerobias_rebake_leakyrelu_i8()
    test_zerobias_rebake_identity()
    test_student_permutation_relu()
    test_student_permutation_leakyrelu()
    test_metapicker_forward_argmax_unchanged()
    test_metapicker_matches_predict_lib_argmax()
    print("OK — all 13 predict_lib tests pass")
