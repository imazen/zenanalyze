"""Tests for _provenance.py — the zenanalyze-api reuse-key stamps.

Run: python3 -m pytest zentrain/tools/test_provenance.py
"""

from __future__ import annotations

import re

from _provenance import stamps_from_provenance, workspace_provenance


def test_undeclared_emits_no_stamps():
    # The safe default: no provenance -> no stamps -> unstamped model -> own-pass.
    assert stamps_from_provenance(None) == {}
    assert stamps_from_provenance({}) == {}


def test_full_provenance_round_trips():
    s = stamps_from_provenance(
        {"analyzer_version": "0.2.0", "feature_defs_version": 1, "feature_config_hash": 0}
    )
    assert s == {
        "analyzer_version": "0.2.0",
        "feature_defs_version": 1,
        "feature_config_hash": 0,
    }


def test_config_hash_defaults_to_zero_when_declared_without_it():
    # Declaring provenance opts into stamping; gamma (0) is the universal default.
    s = stamps_from_provenance({"analyzer_version": "0.2.0", "feature_defs_version": 1})
    assert s["feature_config_hash"] == 0


def test_linear_light_config_hash_preserved():
    s = stamps_from_provenance(
        {"analyzer_version": "0.2.0", "feature_defs_version": 1, "feature_config_hash": 12345}
    )
    assert s["feature_config_hash"] == 12345


def test_workspace_provenance_reads_the_crate():
    p = workspace_provenance()
    # analyzer_version is this workspace's zenanalyze crate version.
    assert re.match(r"^\d+\.\d+\.\d+", p["analyzer_version"]), p["analyzer_version"]
    # feature_defs_version is the src/lib.rs const (a small int, currently 1).
    assert isinstance(p["feature_defs_version"], int)
    assert p["feature_defs_version"] >= 1
    # gamma default.
    assert p["feature_config_hash"] == 0


if __name__ == "__main__":
    test_undeclared_emits_no_stamps()
    test_full_provenance_round_trips()
    test_config_hash_defaults_to_zero_when_declared_without_it()
    test_linear_light_config_hash_preserved()
    test_workspace_provenance_reads_the_crate()
    print("all _provenance tests passed")
