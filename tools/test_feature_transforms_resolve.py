#!/usr/bin/env python3
"""Regression test for bake_picker's feature_transforms → n_inputs resolution.

The trainer emits `feature_transforms` parallel to `feat_cols` (the analyzed
features, which occupy the first n_feat input slots); the trailing engineered
`extra_axes` carry no transform and are therefore identity. `encode_metadata`
resolves the short list to the full `n_inputs` length so the runtime's strict
`len == n_inputs` parse (the safety check that prevents silently-wrong,
position-shifted transforms) accepts the bake. See issue #70-A1.

Self-contained (no Rust build). Run:

    python3 tools/test_feature_transforms_resolve.py

Exits 0 on success, non-zero with a diagnostic on failure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_picker  # noqa: E402


def transforms_entry(entries):
    e = next((e for e in entries if e["key"] == "zentrain.feature_transforms"), None)
    return e["text"].split("\n") if e is not None else None


def params_entry(entries):
    e = next((e for e in entries if e["key"] == "zentrain.feature_transform_params"), None)
    return e["text"].split("\n") if e is not None else None


def main() -> int:
    out = Path("/tmp/_ft_resolve.bin")

    # Short list (parallel to feat_cols) → resolved to n_inputs, identity tail.
    model = {
        "n_inputs": 10,
        "feat_cols": ["a", "b", "c"],
        "feature_transforms": ["log", "identity", "log1p"],
        "feature_transform_params": [[], [], []],
        "schema_version_tag": "test",
    }
    lines = transforms_entry(bake_picker.encode_metadata(model, out))
    assert lines is not None, "no feature_transforms entry emitted"
    assert len(lines) == 10, f"expected 10 lines (n_inputs), got {len(lines)}: {lines}"
    assert lines[:3] == ["log", "identity", "log1p"], f"feat_col transforms wrong: {lines[:3]}"
    assert lines[3:] == ["identity"] * 7, f"trailing extra_axes not identity: {lines[3:]}"

    # Params with a real param resolve in parallel (empty rows for extra_axes).
    model_p = {
        "n_inputs": 5,
        "feat_cols": ["a", "b"],
        "feature_transforms": ["winsor_p99", "identity"],
        "feature_transform_params": [[0.01, 0.99], []],
        "schema_version_tag": "test",
    }
    e_p = bake_picker.encode_metadata(model_p, out)
    t_p = transforms_entry(e_p)
    p_p = params_entry(e_p)
    assert t_p is not None and len(t_p) == 5, f"transforms not n_inputs: {t_p}"
    assert p_p is not None and len(p_p) == 5, f"params not n_inputs: {p_p}"
    assert p_p[0] == "0.01,0.99" and p_p[1:] == ["", "", "", ""], f"param rows wrong: {p_p}"

    # Already-n_inputs transforms are left untouched.
    model2 = {
        "n_inputs": 4,
        "feat_cols": ["a"],
        "feature_transforms": ["log", "identity", "identity", "identity"],
        "feature_transform_params": [[]] * 4,
        "schema_version_tag": "test",
    }
    assert len(transforms_entry(bake_picker.encode_metadata(model2, out))) == 4

    # All-identity (no real transform) emits no entry — runtime treats as identity.
    model3 = dict(model2, feature_transforms=["identity"])
    assert transforms_entry(bake_picker.encode_metadata(model3, out)) is None

    print("PASS: feature_transforms (+params) resolve to n_inputs; no-op/regression cases hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
