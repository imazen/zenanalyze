"""Smoke tests for `_metapicker_lib`.

Validates:
- `classify_stem` matches each v12/v14/v15 trainer's local impl
  byte-for-byte on a fixed set of probe stems.
- `cclass_one_hot` returns identical vectors.
- `forward_metapicker` matches the v15_compare_pickers reference
  forward-pass on a hand-rolled tiny model.

Run with: `python3 -m pytest zentrain/tools/test_metapicker_lib.py`
or `python3 zentrain/tools/test_metapicker_lib.py` (calls every
test_* function directly).
"""
from __future__ import annotations

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _metapicker_lib import (
    BANDS_DEFAULT, BAND_TOL_DEFAULT, SEED_DEFAULT, CCLASSES,
    classify_stem, cclass_one_hot,
    cell_bytes_for, bytes_delta_vs_baseline,
    forward_metapicker,
)


# ---------------------------------------------------------------------------
# classify_stem parity vs v14/v15 inline impl. The pre-extraction
# function bodies are reproduced verbatim below for cross-check.

def _classify_stem_v14_inline(stem: str) -> str:
    """Verbatim copy of v14_metapicker_train.py:89-107."""
    s = stem.lower()
    if s.startswith("gen-screen__"):
        return "screen"
    if s.startswith("gen-doc__"):
        return "document"
    if s.startswith("gen-chart__") or s.startswith("gen-line__"):
        return "lineart"
    if s.startswith("gen-mixed__"):
        return "photo"
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]):
        return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stockquote"]):
        return "lineart"
    if any(p in s for p in ["scan", "document", "invoice"]):
        return "document"
    if any(p in s for p in ["synthetic", "checker", "noise_", "thin_lines", "gradient_v_", "gradient_h_"]):
        return "synthetic"
    return "photo"


def _classify_stem_v12_inline(stem):
    """Verbatim copy of v12_metapicker_train.py:31-41."""
    s = stem.lower()
    if s.startswith("gen-screen__"): return "screen"
    if s.startswith("gen-doc__"): return "document"
    if s.startswith("gen-chart__") or s.startswith("gen-line__"): return "lineart"
    if s.startswith("gen-mixed__"): return "photo"
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_", "browser", "ide", "editor"]): return "screen"
    if any(p in s for p in ["chart", "graph", "diagram", "logo", "infographic", "stockquote"]): return "lineart"
    if any(p in s for p in ["scan", "document", "invoice"]): return "document"
    if any(p in s for p in ["synthetic", "checker", "noise_", "thin_lines", "gradient_v_", "gradient_h_"]): return "synthetic"
    return "photo"


PROBE_STEMS = [
    # Synthetic prefixes (gen-*) — most common path.
    "gen-screen__terminal_001",
    "gen-doc__page_05",
    "gen-chart__bar_03",
    "gen-line__diagram",
    "gen-mixed__city_park",
    "Gen-Mixed__case_sensitive_check",  # uppercase: should fall through gen-* and hit photo
    # Keyword-fallback paths.
    "actual_terminal_screenshot_001",
    "macbook_screenshot",
    "browser_window_001",
    "stockquote_widget",
    "chart_quarterly_revenue",
    "infographic_company",
    "scan_2024_invoice_001",
    "synthetic_test",
    "checker_8x8",
    "noise_uniform_01",
    "thin_lines_01",
    "gradient_v_01",
    "gradient_h_02",
    # Default photo case.
    "photo_natural_landscape",
    "IMG_2034",
    "cid22_1418519",
    # Edge-cases.
    "",
    "a",
    "_underscore_first",
    "1234",
]


def test_classify_stem_parity_v14():
    for stem in PROBE_STEMS:
        ours = classify_stem(stem)
        theirs = _classify_stem_v14_inline(stem)
        assert ours == theirs, f"classify_stem mismatch v14 on {stem!r}: ours={ours!r} v14={theirs!r}"


def test_classify_stem_parity_v12():
    for stem in PROBE_STEMS:
        ours = classify_stem(stem)
        theirs = _classify_stem_v12_inline(stem)
        assert ours == theirs, f"classify_stem mismatch v12 on {stem!r}: ours={ours!r} v12={theirs!r}"


def test_cclass_one_hot():
    assert cclass_one_hot("photo") == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert cclass_one_hot("screen") == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert cclass_one_hot("lineart") == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert cclass_one_hot("document") == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert cclass_one_hot("synthetic") == [0.0, 0.0, 0.0, 0.0, 1.0]
    # Unknown → all zeros.
    assert cclass_one_hot("unknown_class") == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_cell_bytes_for():
    s = {"codec_bytes": {"zenjxl": 100, "zenwebp": 120, "zenavif": 90}}
    assert cell_bytes_for(s, "zenjxl") == 100
    assert cell_bytes_for(s, "zenwebp") == 120
    assert cell_bytes_for(s, "zenavif") == 90
    # OOB codec → max(codec_bytes).
    assert cell_bytes_for(s, "zenpng") == 120


def test_bytes_delta_vs_baseline():
    hold = [
        {"codec_bytes": {"zenjxl": 100, "zenwebp": 120, "zenavif": 90}},
        {"codec_bytes": {"zenjxl": 200, "zenwebp": 180, "zenavif": 220}},
    ]
    # Pred: pick the winner each time = avif then webp.
    pred = ["zenavif", "zenwebp"]
    base_b, mlp_b, oracle_b, pct = bytes_delta_vs_baseline(hold, pred, "zenjxl")
    assert base_b == 300  # 100 + 200
    assert mlp_b == 270   # 90 + 180
    assert oracle_b == 270  # 90 + 180
    assert abs(pct(mlp_b) - (-10.0)) < 1e-6
    assert abs(pct(oracle_b) - (-10.0)) < 1e-6


def test_forward_metapicker_relu():
    """Tiny 2-input, 3-output relu model: forward + argmax."""
    # x = [[1, 2]]
    # standardize with mean=[0,0], scale=[1,1] → [[1, 2]]
    # layer0 W=[[1,0,0],[0,1,0]] b=[0,0,1] → [[1, 2, 1]]
    # layer1 (final, identity) W=I, b=0 → [[1, 2, 1]]
    # argmax = 1
    model = {
        "scaler_mean": [0.0, 0.0],
        "scaler_scale": [1.0, 1.0],
        "activation": "relu",
        "layers": [
            {"W": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "b": [0.0, 0.0, 1.0]},
            {"W": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "b": [0.0, 0.0, 0.0]},
        ],
    }
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    pred = forward_metapicker(model, X)
    assert pred.tolist() == [1]


def test_forward_metapicker_leakyrelu_matches_v15_compare():
    """Replicate v15_compare_pickers.forward and confirm bit-parity."""
    # Reference impl from v15_compare_pickers.py:154-176.
    def _leaky_relu(x, slope=0.01):
        return np.where(x > 0, x, slope * x)

    def _relu(x):
        return np.maximum(x, 0)

    def forward_v15(model, X):
        mean = np.array(model["scaler_mean"], dtype=np.float32)
        scale = np.array(model["scaler_scale"], dtype=np.float32)
        Z = ((X - mean) / scale).astype(np.float32)
        act = model.get("activation", "relu").lower()
        af = _leaky_relu if act in ("leakyrelu", "leaky_relu") else _relu
        layers = model["layers"]
        for i, layer in enumerate(layers):
            W = np.asarray(layer["W"], dtype=np.float32)
            b = np.asarray(layer["b"], dtype=np.float32)
            Z = Z @ W + b
            if i < len(layers) - 1:
                Z = af(Z)
        return np.argmax(Z, axis=1)

    rng = np.random.default_rng(0xC0FFEE)
    model = {
        "scaler_mean": rng.standard_normal(20).tolist(),
        "scaler_scale": (rng.standard_normal(20) ** 2 + 0.5).tolist(),
        "activation": "leakyrelu",
        "layers": [
            {"W": rng.standard_normal((20, 64)).tolist(), "b": rng.standard_normal(64).tolist()},
            {"W": rng.standard_normal((64, 5)).tolist(), "b": rng.standard_normal(5).tolist()},
        ],
    }
    X = rng.standard_normal((30, 20)).astype(np.float32)
    pred_ours = forward_metapicker(model, X)
    pred_theirs = forward_v15(model, X)
    assert (pred_ours == pred_theirs).all(), \
        f"forward mismatch: ours={pred_ours} theirs={pred_theirs}"

    # Also test ReLU path.
    model["activation"] = "relu"
    pred_ours = forward_metapicker(model, X)
    pred_theirs = forward_v15(model, X)
    assert (pred_ours == pred_theirs).all()


def test_defaults():
    assert BANDS_DEFAULT == [70.0, 75.0, 80.0, 85.0, 90.0]
    assert BAND_TOL_DEFAULT == 1.5
    assert SEED_DEFAULT == 7
    assert CCLASSES == ["photo", "screen", "lineart", "document", "synthetic"]


if __name__ == "__main__":
    test_classify_stem_parity_v14()
    test_classify_stem_parity_v12()
    test_cclass_one_hot()
    test_cell_bytes_for()
    test_bytes_delta_vs_baseline()
    test_forward_metapicker_relu()
    test_forward_metapicker_leakyrelu_matches_v15_compare()
    test_defaults()
    print("OK — all 8 metapicker_lib tests pass")
