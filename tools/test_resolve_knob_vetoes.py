#!/usr/bin/env python3
"""Regression test for bake_picker's resolve_knob_vetoes fail-loud contract.

`resolve_knob_vetoes` decomposes each `(axis, value)` knob-veto rule into the
concrete picker cell ids that carry that value. A rule whose `(axis, value)`
doesn't match ANY cell in `hybrid_heads_manifest.cells` (a typo, a stale
axis/value pair left over from a schema change, etc.) used to resolve to an
empty `cells` list silently -- a veto that vetoes nothing is a silently-lost
safety bound, not a harmless no-op, and the function's own docstring already
promised a "fails loud" contract for the analogous wire-overflow case. Now it
raises `SystemExit` immediately, matching that contract.

Self-contained (no Rust build). Run:

    python3 tools/test_resolve_knob_vetoes.py

Exits 0 on success, non-zero with a diagnostic on failure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_picker  # noqa: E402


def _model(vetoes):
    return {
        "feat_cols": ["f0"],
        "hybrid_heads_manifest": {
            "cells": [
                {"id": 0, "chroma": "420"},
                {"id": 1, "chroma": "444"},
            ],
            "knob_vetoes": vetoes,
        },
    }


def main() -> int:
    # A veto matching a real (axis, value) resolves to the right cell ids.
    ok = bake_picker.resolve_knob_vetoes(
        _model([{"axis": "chroma", "value": "444", "feat": "f0", "op": ">", "threshold": 0.5}])
    )
    assert len(ok) == 1
    assert ok[0]["cells"] == [1], f"expected cell 1 (chroma=444), got {ok[0]['cells']!r}"

    # A veto whose (axis, value) matches NOTHING must fail loud, not silently
    # resolve to an empty (no-op) cells list.
    try:
        bake_picker.resolve_knob_vetoes(
            _model(
                [{"axis": "chroma", "value": "999", "feat": "f0", "op": ">", "threshold": 0.5}]
            )
        )
    except SystemExit as e:
        assert "zero cell_ids" in str(e), f"unexpected SystemExit message: {e}"
    else:
        raise AssertionError(
            "resolve_knob_vetoes must raise SystemExit for a zero-match (axis, value)"
        )

    # No knob_vetoes at all -> empty list, no error (backward-compatible no-op).
    assert bake_picker.resolve_knob_vetoes(_model([])) == []
    assert bake_picker.resolve_knob_vetoes({"hybrid_heads_manifest": {}}) == []

    print("PASS: resolve_knob_vetoes matches its own fail-loud contract on a zero-match veto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
