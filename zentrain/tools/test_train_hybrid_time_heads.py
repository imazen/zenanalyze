"""The time-aware picker objectives — `--objective rd_time` (zenanalyze#56)
and `--objective time_budgeted` (zenanalyze#43) — and the bake-side keys
they emit.

Covers, without a real corpus:

- objective preconditions (`validate_objective_requirements`) and the
  `--time-loss-weight` resolution / scale-for-fit / absorb-after-fit pair;
- the #56 diagnostics (`encode_ms_p99`, `median_cell_ms_per_mp`,
  `time_head_relative_error`) and the TIME_HEAD_R2 / BUDGET_INFEASIBLE /
  METRIC_HEAD_R2 gates (`time_gate_violations`);
- an end-to-end label-extraction smoke test through the real `build_dataset`
  on a synthetic Pareto where the right answer REQUIRES honouring the time
  budget (the byte-optimal config is over budget);
- `tools/bake_picker.py`: `zentrain.hybrid_heads_layout` now declares the
  time / metric blocks (it used to drop them), the `zentrain.profile` byte
  for the new objectives, and the `zentrain.median_cell_ms_per_mp` /
  `zentrain.encode_ms_p99(_zq_targets)` keys.

Run with: `python3 -m pytest zentrain/tools/test_train_hybrid_time_heads.py`
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.argv = ["train_hybrid.py"]  # keep argv-at-import paths inert

import bake_picker as bp  # noqa: E402
import train_hybrid as th  # noqa: E402


# ---------------------------------------------------------------- objectives


def test_objective_choices_include_the_time_aware_pair():
    assert "rd_time" in th.OBJECTIVES and "time_budgeted" in th.OBJECTIVES
    assert set(th.TIME_AWARE_OBJECTIVES) == {"rd_time", "time_budgeted"}


def test_rd_time_requires_the_time_column():
    with pytest.raises(ValueError, match="encode_ms"):
        th.validate_objective_requirements(
            "rd_time", has_time_column=False, time_budget_multiplier=0.0
        )
    # Present → fine, with or without a budget.
    th.validate_objective_requirements("rd_time", has_time_column=True, time_budget_multiplier=0.0)
    th.validate_objective_requirements("rd_time", has_time_column=True, time_budget_multiplier=1.5)


def test_time_budgeted_requires_time_column_and_multiplier():
    with pytest.raises(ValueError, match="time-budget-multiplier"):
        th.validate_objective_requirements(
            "time_budgeted", has_time_column=True, time_budget_multiplier=0.0
        )
    with pytest.raises(ValueError, match="column"):
        th.validate_objective_requirements(
            "time_budgeted", has_time_column=False, time_budget_multiplier=1.5
        )
    th.validate_objective_requirements(
        "time_budgeted", has_time_column=True, time_budget_multiplier=1.5
    )


def test_legacy_objectives_have_no_time_preconditions():
    for obj in ("size_optimal", "zensim_strict"):
        th.validate_objective_requirements(obj, has_time_column=False, time_budget_multiplier=0.0)
    with pytest.raises(ValueError, match="unknown objective"):
        th.validate_objective_requirements("nope", has_time_column=True, time_budget_multiplier=1.0)


def test_time_loss_weight_defaults_only_for_time_aware_objectives():
    assert th.resolve_time_loss_weight("size_optimal", None) is None
    assert th.resolve_time_loss_weight("zensim_strict", None) is None
    assert th.resolve_time_loss_weight("rd_time", None) == th.DEFAULT_RD_TIME_LOSS_WEIGHT
    assert th.resolve_time_loss_weight("time_budgeted", None) == th.DEFAULT_RD_TIME_LOSS_WEIGHT
    # Explicit values win, 0.0 included.
    assert th.resolve_time_loss_weight("rd_time", 0.0) == 0.0
    assert th.resolve_time_loss_weight("size_optimal", 2.0) == 2.0
    with pytest.raises(ValueError):
        th.resolve_time_loss_weight("rd_time", -1.0)
    with pytest.raises(ValueError):
        th.resolve_time_loss_weight("rd_time", float("nan"))


# --------------------------------------------------- loss weight: fit + absorb


class _Student:
    def __init__(self, W_last: np.ndarray, b_last: np.ndarray):
        hidden = np.eye(W_last.shape[0])
        self.coefs_ = [hidden, W_last.copy()]
        self.intercepts_ = [np.zeros(W_last.shape[0]), b_last.copy()]


def _soft(n_rows=6, n_cells=2, n_scalar_blocks=1, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rows, n_cells * (2 + n_scalar_blocks)))


def test_scale_for_fit_is_sqrt_weight_and_absorb_inverts_it():
    soft = _soft()
    n_cells = 2
    soft_fit, absorb = th.scale_time_block_for_fit(soft, n_cells, 2 * n_cells, 4.0)
    # MSE on the scaled block == 4 × MSE on the natural block ⇔ block × sqrt(4).
    np.testing.assert_allclose(soft_fit[:, n_cells:2 * n_cells], 2.0 * soft[:, n_cells:2 * n_cells])
    np.testing.assert_array_equal(soft_fit[:, :n_cells], soft[:, :n_cells])
    np.testing.assert_array_equal(soft_fit[:, 2 * n_cells:], soft[:, 2 * n_cells:])
    assert soft_fit.shape == soft.shape
    assert absorb == ("scale", n_cells, 2 * n_cells, 0.5)
    # The input is not mutated.
    assert not np.shares_memory(soft_fit, soft)

    # A network that (pretend) learned the scaled targets: absorbing halves
    # exactly the time block's final-layer weights + biases.
    W = np.arange(3 * 6, dtype=np.float64).reshape(3, 6) + 1.0
    b = np.arange(6, dtype=np.float64) + 1.0
    st = _Student(W, b)
    th.absorb_time_block(st, absorb)
    np.testing.assert_allclose(st.coefs_[-1][:, 2:4], 0.5 * W[:, 2:4])
    np.testing.assert_allclose(st.intercepts_[-1][2:4], 0.5 * b[2:4])
    np.testing.assert_array_equal(st.coefs_[-1][:, :2], W[:, :2])
    np.testing.assert_array_equal(st.coefs_[-1][:, 4:], W[:, 4:])
    np.testing.assert_array_equal(st.intercepts_[-1][[0, 1, 4, 5]], b[[0, 1, 4, 5]])


def test_weight_zero_drops_the_block_and_reinserts_teacher_means():
    soft = _soft()
    n_cells = 2
    soft_fit, absorb = th.scale_time_block_for_fit(soft, n_cells, 2 * n_cells, 0.0)
    assert soft_fit.shape == (soft.shape[0], soft.shape[1] - n_cells)
    np.testing.assert_array_equal(soft_fit[:, :n_cells], soft[:, :n_cells])
    np.testing.assert_array_equal(soft_fit[:, n_cells:], soft[:, 2 * n_cells:])
    assert absorb == ("drop", n_cells, 2 * n_cells, None)

    # Student trained on the 4-wide fit matrix; absorb must widen it back
    # to the 6-wide output layout: zero weights + bias = teacher mean.
    W = np.ones((3, 4))
    b = np.array([10.0, 11.0, 12.0, 13.0])
    st = _Student(W, b)
    th.absorb_time_block(st, absorb, time_means=np.array([math.log(7.0), math.nan]))
    assert st.coefs_[-1].shape == (3, 6)
    assert st.intercepts_[-1].shape == (6,)
    np.testing.assert_array_equal(st.coefs_[-1][:, 2:4], 0.0)
    np.testing.assert_allclose(st.intercepts_[-1][2:4], [math.log(7.0), 0.0])
    np.testing.assert_array_equal(st.intercepts_[-1][[0, 1, 4, 5]], b)
    np.testing.assert_array_equal(st.coefs_[-1][:, [0, 1, 4, 5]], W)


def test_weight_none_or_one_is_a_no_op():
    soft = _soft()
    for w in (None, 1.0):
        soft_fit, absorb = th.scale_time_block_for_fit(soft, 2, 4, w)
        assert soft_fit is soft and absorb is None
    st = _Student(np.ones((3, 6)), np.zeros(6))
    th.absorb_time_block(st, None)
    np.testing.assert_array_equal(st.coefs_[-1], np.ones((3, 6)))


# --------------------------------------------------------- #56 diagnostics


def test_encode_ms_p99_is_per_zq_per_cell_with_none_for_never_reached():
    meta = [("a", "small", 50), ("b", "small", 50), ("c", "small", 80)]
    time_log = np.log(np.array([
        [10.0, np.nan],
        [30.0, np.nan],
        [5.0, 2.0],
    ]))
    out = th.compute_encode_ms_p99(time_log, meta, n_cells=2)
    assert list(out.keys()) == ["50", "80"]
    assert out["50"][1] is None
    assert out["80"] == pytest.approx([5.0, 2.0])
    # p99 over {10, 30} sits just under 30.
    assert 29.0 < out["50"][0] <= 30.0
    assert th.compute_encode_ms_p99(None, meta, 2) == {}


def test_median_ms_per_mp_uses_pixels_of_the_row_key():
    meta = [("a", "small", 50), ("b", "medium", 50)]
    time_log = np.log(np.array([[2.0, np.nan], [8.0, 16.0]]))
    px = {("a", "small"): 1_000_000, ("b", "medium"): 4_000_000}
    # entries: 2/1 = 2, 8/4 = 2, 16/4 = 4 → median 2.
    assert th.compute_median_ms_per_mp(time_log, meta, px) == pytest.approx(2.0)
    assert th.compute_median_ms_per_mp(time_log, meta, {}) is None
    assert th.compute_median_ms_per_mp(None, meta, px) is None


def test_relative_error_is_in_ms_space_over_reached_cells():
    actual = np.log(np.array([[100.0, 10.0], [100.0, np.nan]]))
    pred = np.log(np.array([[110.0, 5.0], [100.0, 999.0]]))  # 10 %, 50 %, 0 %; last not reached
    reach = np.array([[True, True], [True, False]])
    r = th.time_head_relative_error(pred, actual, reach)
    assert r["n"] == 3
    assert r["max"] == pytest.approx(0.5)
    assert r["p50"] == pytest.approx(0.1)
    assert th.time_head_relative_error(pred, actual, np.zeros_like(reach)) is None


def test_time_gates_fire_and_are_wired_into_safety_check_thresholds():
    thr = {"min_time_head_r2": 0.6, "min_metric_head_r2": 0.6, "max_budget_infeasible_fraction": 0.05}
    ok = {"time_head_r2": {"median": 0.9}, "metric_head_r2": {"median": 0.9}, "budget_infeasible_fraction": 0.01}
    assert th.time_gate_violations(ok, thr) == []
    bad = {"time_head_r2": {"median": 0.5}, "metric_head_r2": {"median": 0.2}, "budget_infeasible_fraction": 0.2}
    v = th.time_gate_violations(bad, thr)
    assert [x.split(":")[0] for x in v] == ["TIME_HEAD_R2", "METRIC_HEAD_R2", "BUDGET_INFEASIBLE"]
    # Heads absent → their gates are silent; the budget gate is fraction-only.
    assert th.time_gate_violations({"budget_infeasible_fraction": 0.0}, thr) == []
    # The shipped defaults carry the #43 floors (0.6 R², 5 % infeasible).
    assert th.DEFAULT_SAFETY_THRESHOLDS["min_time_head_r2"] == 0.6
    assert th.DEFAULT_SAFETY_THRESHOLDS["max_budget_infeasible_fraction"] == 0.05


# --------------------------------------- end-to-end: labels honour the budget


def _budget_fixture():
    """Two images, one size, two cells. Image A: cell1 has a byte-optimal
    but SLOW config (100 B, 1000 ms) and a slightly larger fast one
    (120 B, 10 ms); cell0 is 150 B / 5 ms. Image B: everything is far over
    any sane budget. Every config reaches zq 50."""
    A = ("/img/a.png", "small", 100, 100)
    B = ("/img/b.png", "small", 100, 100)
    pareto = {
        A: {
            "config_id": np.array([0, 1, 2], dtype=np.int64),
            "bytes": np.array([150, 100, 120], dtype=np.int64),
            "zensim": np.array([60.0, 60.0, 60.0]),
            "time_ms": np.array([5.0, 1000.0, 10.0]),
        },
        B: {
            "config_id": np.array([0, 1, 2], dtype=np.int64),
            "bytes": np.array([150, 100, 120], dtype=np.int64),
            "zensim": np.array([60.0, 60.0, 60.0]),
            "time_ms": np.array([5000.0, 6000.0, 7000.0]),
        },
    }
    feats = {
        ("/img/a.png", "small"): np.array([0.5], dtype=np.float32),
        ("/img/b.png", "small"): np.array([0.7], dtype=np.float32),
    }
    cells = [{"label": "c0"}, {"label": "c1"}]
    config_to_cell = {0: 0, 1: 1, 2: 1}
    parsed_all = {0: {}, 1: {}, 2: {}}
    return pareto, feats, cells, config_to_cell, parsed_all


def _configure_trainer(monkeypatch):
    monkeypatch.setattr(th, "ZQ_TARGETS", [50], raising=False)
    monkeypatch.setattr(th, "SCALAR_AXES", [])
    monkeypatch.setattr(th, "SCALAR_SENTINELS", {})
    monkeypatch.setattr(th, "METRIC_DIRECTION", "higher_better")
    monkeypatch.setattr(th, "REACH_UNDERSHOOT", 0.0)


def test_build_dataset_labels_pick_the_cheapest_in_budget_config(monkeypatch):
    _configure_trainer(monkeypatch)
    pareto, feats, cells, c2c, parsed = _budget_fixture()
    baselines = th.compute_time_baselines(pareto)
    # median over {5, 1000, 10, 5000, 6000, 7000} = 3000 ms; × 0.01 → 30 ms.
    assert baselines == {"small": pytest.approx(3000.0)}

    Xs, Xe, bytes_log, scalars, reach, meta, time_log, metric_log, infeasible = th.build_dataset(
        pareto, feats, ["f0"], cells, c2c, parsed,
        time_budget_multiplier=0.01, time_baselines=baselines,
    )
    # Image B has no in-budget config at all → no decision row, flagged.
    assert meta == [("/img/a.png", "small", 50)]
    assert infeasible == {("/img/b.png", "small"): True}
    # Cell 1's label is the 120 B / 10 ms config, NOT the byte-optimal
    # 100 B / 1000 ms one that a budget-blind trainer would take.
    np.testing.assert_allclose(bytes_log[0], np.log([150.0, 120.0]))
    np.testing.assert_allclose(time_log[0], np.log([5.0, 10.0]))
    assert reach[0].tolist() == [True, True]


def test_build_dataset_without_budget_takes_the_byte_optimal_config(monkeypatch):
    _configure_trainer(monkeypatch)
    pareto, feats, cells, c2c, parsed = _budget_fixture()
    *_, bytes_log, _scalars, reach, meta, time_log, _m, infeasible = th.build_dataset(
        pareto, feats, ["f0"], cells, c2c, parsed,
    )
    assert len(meta) == 2 and infeasible == {}
    np.testing.assert_allclose(bytes_log[0], np.log([150.0, 100.0]))
    np.testing.assert_allclose(time_log[0], np.log([5.0, 1000.0]))
    assert reach.all()


def test_budget_infeasible_fraction_reaches_the_gate(monkeypatch):
    _configure_trainer(monkeypatch)
    pareto, feats, cells, c2c, parsed = _budget_fixture()
    baselines = th.compute_time_baselines(pareto)
    *_, infeasible = th.build_dataset(
        pareto, feats, ["f0"], cells, c2c, parsed,
        time_budget_multiplier=0.01, time_baselines=baselines,
    )
    frac = len(infeasible) / len({(k[0], k[1]) for k in pareto})
    assert frac == pytest.approx(0.5)
    v = th.time_gate_violations({"budget_infeasible_fraction": frac}, th.DEFAULT_SAFETY_THRESHOLDS)
    assert len(v) == 1 and v[0].startswith("BUDGET_INFEASIBLE: 50.0%")


# ------------------------------------------------------------ bake_picker

# The minimum a training JSON needs for encode_metadata's unconditional
# sections (feature columns / transforms padding to n_inputs).
_BAKE_MIN = {"n_inputs": 1, "n_outputs": 2, "feat_cols": ["f0"]}


def test_profile_bytes_cover_every_objective():
    assert set(bp.PROFILE_BYTES) == set(th.OBJECTIVES)
    assert bp.PROFILE_BYTES["size_optimal"] == 0 and bp.PROFILE_BYTES["zensim_strict"] == 1
    assert bp.PROFILE_BYTES["rd_time"] == 2 and bp.PROFILE_BYTES["time_budgeted"] == 3


def test_head_kinds_follow_output_layout_including_time_and_metric_blocks():
    n = 3
    hh = {
        "n_cells": n,
        "categorical_axes": ["mode"],
        "scalar_axes": ["chroma_scale"],
        "output_layout": {
            "bytes_log": [0, n], "time_log": [n, 2 * n], "metric_log": [2 * n, 3 * n],
            "chroma_scale": [3 * n, 4 * n],
        },
    }
    assert bp.hybrid_head_kinds(hh) == bytes([0, 2, 3, 1])
    # Layout is ordered by block start, not dict insertion order.
    hh["output_layout"] = {"chroma_scale": [2 * n, 3 * n], "bytes_log": [0, n], "time_log": [n, 2 * n]}
    assert bp.hybrid_head_kinds(hh) == bytes([0, 2, 1])
    # Pre-#56 JSON without output_layout: bytes + one scalar per axis.
    assert bp.hybrid_head_kinds({"categorical_axes": ["a"], "scalar_axes": ["x", "y"]}) == bytes([0, 1, 1])
    assert bp.hybrid_head_kinds({"categorical_axes": [], "scalar_axes": ["x"]}) == b""


def test_packed_layout_declares_the_time_block(tmp_path):
    n = 4
    model = {
        **_BAKE_MIN,
        "hybrid_heads_manifest": {
            "n_cells": n, "categorical_axes": ["mode"], "scalar_axes": [],
            "output_layout": {"bytes_log": [0, n], "time_log": [n, 2 * n]},
        },
    }
    entries = bp.encode_metadata(model, tmp_path / "x.bin")
    blob = bytes.fromhex(next(e["hex"] for e in entries if e["key"] == "zentrain.hybrid_heads_layout"))
    n_cells, n_heads = struct.unpack("<II", blob[:8])
    assert (n_cells, n_heads) == (n, 2)
    assert blob[8:] == bytes([bp.HEAD_KIND_BYTES, bp.HEAD_KIND_TIME])


def test_time_head_metadata_keys_round_trip_from_diagnostics(tmp_path):
    model = {
        **_BAKE_MIN,
        "safety_profile": "rd_time",
        "safety_report": {"diagnostics": {
            "median_cell_ms_per_mp": 12.5,
            "encode_ms_p99": {"80": [1.0, None], "50": [3.0, 4.0]},
        }},
    }
    entries = bp.encode_metadata(model, tmp_path / "x.bin")
    by_key = {e["key"]: e for e in entries}
    assert by_key["zentrain.profile"]["hex"] == "02"
    assert by_key["zentrain.median_cell_ms_per_mp"]["f32"] == [12.5]
    assert bytes.fromhex(by_key["zentrain.encode_ms_p99_zq_targets"]["hex"]) == bytes([50, 80])
    flat = by_key["zentrain.encode_ms_p99"]["f32"]
    assert flat == [3.0, 4.0, 1.0, bp.ENCODE_MS_P99_UNREACHED] and bp.ENCODE_MS_P99_UNREACHED < 0


def test_time_head_metadata_absent_without_a_time_head(tmp_path):
    entries = bp.encode_metadata({**_BAKE_MIN, "safety_profile": "size_optimal"}, tmp_path / "x.bin")
    keys = {e["key"] for e in entries}
    assert not any(k.startswith("zentrain.encode_ms_p99") for k in keys)
    assert "zentrain.median_cell_ms_per_mp" not in keys
