"""Regression test for `train_hybrid.derive_knob_vetoes`'s NaN-poisoning bug.

Rows whose `(reach & all_mask)` is entirely empty (no candidate cell at all)
have `oracle = +inf`, so `(achieved - oracle) / oracle` is NaN. Before the
fix, that single NaN row propagated through `.mean()` / `.max()` in the
greedy veto selector's `metrics()` closure, poisoning the WHOLE tail score;
`NaN > threshold` is always `False`, so `gain > 1.0` never fired and the
deriver silently returned zero vetoes -- even when an obviously-catastrophic,
easily-vetoable mis-set was present in the same batch.

Run with: `python3 -m pytest zentrain/tools/test_train_hybrid_knob_vetoes.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = ["train_hybrid.py"]  # train_hybrid parses argv at import in some paths; keep it inert

import train_hybrid as th  # noqa: E402


def _synthetic_dataset(n_cat: int, n_noncat: int, include_invalid_row: bool):
    """Two cells (chroma=420/444). `n_cat` rows have f0=0.8 and a picker that
    WRONGLY prefers the catastrophic cell1 (predicted cheap, actually 5x costly)
    -- a clean, easily-vetoable mis-set. `n_noncat` rows have f0=0.1 and both
    cells cheap (no catastrophe). Optionally appends one row with NO reachable
    cell at all (the oracle=inf case)."""
    n_rows = n_cat + n_noncat + (1 if include_invalid_row else 0)
    cells = [{"chroma": "420"}, {"chroma": "444"}]
    categorical_axes = ["chroma"]
    feat_cols = ["f0"]
    all_mask = np.array([True, True])
    meta_tr = [(f"img{i}", "small", 50) for i in range(n_rows)]
    feats: dict = {}
    bl_tr = np.zeros((n_rows, 2))
    pred_tr = np.zeros((n_rows, 2))
    rch_tr = np.ones((n_rows, 2), dtype=bool)

    for i in range(n_rows):
        if i < n_cat:
            feats[(f"img{i}", "small")] = [0.8]
            bl_tr[i, 0] = np.log(100.0)
            bl_tr[i, 1] = np.log(500.0)  # cell1 actually 5x costlier here
            pred_tr[i, 0] = np.log(100.0)
            pred_tr[i, 1] = np.log(90.0)  # but predicted cheaper -> picker mis-sets
        elif i < n_cat + n_noncat:
            feats[(f"img{i}", "small")] = [0.1]
            bl_tr[i, 0] = np.log(100.0)
            bl_tr[i, 1] = np.log(105.0)
            pred_tr[i, 0] = np.log(100.0)
            pred_tr[i, 1] = np.log(110.0)
        else:
            # The invalid row: no reachable cell at all -> oracle = +inf for it.
            feats[(f"img{i}", "small")] = [0.5]
            rch_tr[i, :] = [False, False]

    return cells, categorical_axes, pred_tr, bl_tr, rch_tr, meta_tr, feats, feat_cols, all_mask


def test_derive_knob_vetoes_ignores_no_reachable_cell_rows():
    """The all-empty-reach row must NOT poison the tail score: the deriver
    should still find the obviously-catastrophic chroma=444 veto."""
    args = _synthetic_dataset(n_cat=350, n_noncat=350, include_invalid_row=True)
    chosen = th.derive_knob_vetoes(*args)
    assert len(chosen) == 1, f"expected exactly one veto, got {chosen!r}"
    assert chosen[0]["axis"] == "chroma"
    assert chosen[0]["value"] == "444"
    assert chosen[0]["op"] == ">"


def test_derive_knob_vetoes_same_result_with_and_without_invalid_row():
    """The invalid row carries no information (no reachable cell) -- its mere
    presence must not change which veto gets derived."""
    with_invalid = th.derive_knob_vetoes(
        *_synthetic_dataset(n_cat=350, n_noncat=350, include_invalid_row=True)
    )
    without_invalid = th.derive_knob_vetoes(
        *_synthetic_dataset(n_cat=350, n_noncat=350, include_invalid_row=False)
    )
    assert with_invalid == without_invalid


if __name__ == "__main__":
    test_derive_knob_vetoes_ignores_no_reachable_cell_rows()
    test_derive_knob_vetoes_same_result_with_and_without_invalid_row()
    print("OK")
