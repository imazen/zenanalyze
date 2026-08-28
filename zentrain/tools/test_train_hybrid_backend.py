"""`train_hybrid.student_backend_record` — the per-bake student-trainer marker
(zenanalyze#68). `--activation leakyrelu` (torch, Kaiming init, no L2 unless
`--weight-decay`) and `--activation relu` (sklearn `MLPRegressor`, Glorot init,
always `alpha=1e-4`) are two trainers, and #68 measured a 12.4pp argmin_acc gap
between them; a bake must say which one produced it.

The torch-side wiring (`weight_decay` actually reaching `torch.optim.Adam`) is
covered by `test_train_hybrid_torch_backend.py`, which needs torch and is run
via `just zentrain-pytests-torch` rather than the light CI job.

Run with: `python3 -m pytest zentrain/tools/test_train_hybrid_backend.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = ["train_hybrid.py"]  # keep argv-at-import paths inert

import train_hybrid as th  # noqa: E402


def test_torch_record_carries_the_weight_decay_it_trained_under():
    rec = th.student_backend_record("leakyrelu", 1e-4)
    assert rec["backend"] == "torch"
    assert rec["activation"] == "leakyrelu"
    assert rec["init"] == "kaiming_uniform"
    assert rec["l2"] == {"kind": "adam_weight_decay", "value": 1e-4}


def test_torch_record_explicit_zero_is_unregularized():
    rec = th.student_backend_record("leakyrelu", 0.0)
    assert rec["l2"]["value"] == 0.0


def test_resolve_weight_decay_defaults_torch_to_1e_4_and_honours_explicit_values():
    # Flag not passed → the measured #68 default for the torch student only.
    assert th.DEFAULT_TORCH_WEIGHT_DECAY == 1e-4
    assert th.resolve_weight_decay("leakyrelu", None) == 1e-4
    assert th.resolve_weight_decay("relu", None) == 0.0
    # Explicit values win, including 0.0 (reproduce a pre-2026-08-28 bake).
    assert th.resolve_weight_decay("leakyrelu", 0.0) == 0.0
    assert th.resolve_weight_decay("leakyrelu", 5e-4) == 5e-4
    assert th.resolve_weight_decay("relu", 5e-4) == 5e-4
    # The record then carries exactly what was resolved.
    assert th.student_backend_record("leakyrelu", th.resolve_weight_decay("leakyrelu", None))["l2"] == {
        "kind": "adam_weight_decay", "value": 1e-4,
    }


def test_sklearn_record_reports_mlpregressor_alpha_not_the_ignored_flag():
    # --weight-decay is leakyrelu-only; the sklearn path always trains under
    # MLPRegressor's alpha, so that is what the record must say.
    rec = th.student_backend_record("relu", 0.5)
    assert rec["backend"] == "sklearn"
    assert rec["activation"] == "relu"
    assert rec["init"] == "glorot_uniform"
    assert rec["l2"] == {"kind": "mlpregressor_alpha", "value": th.SKLEARN_MLP_DEFAULT_ALPHA}
    assert th.SKLEARN_MLP_DEFAULT_ALPHA == 1e-4


def test_records_are_json_plain():
    import json

    for act in ("leakyrelu", "relu"):
        json.dumps(th.student_backend_record(act, 1e-4))
