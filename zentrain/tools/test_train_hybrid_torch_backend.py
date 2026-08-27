"""`train_hybrid._train_torch_leakyrelu_student` — `weight_decay` must reach
`torch.optim.Adam` (zenanalyze#68 hypothesis 2: the torch student had no L2
while sklearn's `MLPRegressor` always applies `alpha=1e-4`).

Needs torch. Deliberately NOT in the CI `zentrain-pytests` file list (that job
installs only numpy/sklearn); run it with `just zentrain-pytests-torch`, which
provides torch through uv. Importing torch at module top means a missing torch
fails loudly instead of skipping.

Run with: `just zentrain-pytests-torch`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  — fail loud if absent; see module docstring

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = ["train_hybrid.py"]  # keep argv-at-import paths inert

import train_hybrid as th  # noqa: E402


def _fit(weight_decay: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((96, 3)).astype(np.float32)
    Y = np.stack([X[:, 0] + 0.5 * X[:, 1], X[:, 2] ** 2], axis=1).astype(np.float32)
    return th._train_torch_leakyrelu_student(
        X_tr=X,
        Y_tr=Y,
        hidden_layer_sizes=(8,),
        lr=2e-3,
        batch_size=32,
        max_iter=4,
        seed=seed,
        n_iter_no_change=30,
        tol=1e-6,
        weight_decay=weight_decay,
    )


def test_student_records_backend_and_weight_decay():
    s = _fit(1e-4)
    assert s.backend_ == "torch"
    assert s.weight_decay_ == 1e-4
    assert _fit(0.0).weight_decay_ == 0.0


def test_weight_decay_changes_the_fit():
    # Same seed, same data: the only difference is the Adam weight_decay. If
    # the flag were not wired into the optimizer the two fits would be
    # bit-identical (guarded by the determinism check below).
    a = _fit(0.0)
    b = _fit(0.0)
    for wa, wb in zip(a.coefs_, b.coefs_):
        assert np.array_equal(wa, wb), "torch student is not deterministic per seed"
    c = _fit(0.5)
    assert any(not np.allclose(wa, wc) for wa, wc in zip(a.coefs_, c.coefs_)), (
        "weight_decay had no effect on the trained weights — not reaching Adam"
    )


def test_default_weight_decay_is_zero():
    import inspect

    sig = inspect.signature(th._train_torch_leakyrelu_student)
    assert sig.parameters["weight_decay"].default == 0.0
