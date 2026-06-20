"""Tests for _provenance.py — the zenanalyze-api reuse-key stamps.

Run: python3 -m pytest zentrain/tools/test_provenance.py
"""

from __future__ import annotations

import re

import pytest
from _provenance import (
    SERIALIZATION_MAGIC,
    assert_consistent_provenance,
    embed_provenance_in_table,
    feature_is_reusable,
    parse_provenance_block,
    provenance_from_parquet,
    read_provenance_sidecar,
    stamps_from_provenance,
    workspace_provenance,
    write_provenance_block,
)

# A representative block matching what `extract_features_for_picker --features api`
# emits (header + a few real features; the Rust side is the source of truth).
SAMPLE_BLOCK = (
    f"{SERIALIZATION_MAGIC}\n"
    "analyzer_version=0.2.0\n"
    "config_hash=0\n"
    "descriptor_hash=12714097049004338389\n"
    "[features]\n"
    "variance=18198331707036057709\n"
    "edge_density=17469975073558373165\n"
    "uniformity=12293623482497344344\n"
)


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


# ---------------------------------------------------------------------------
# Serialization-provenance block (the fine-grained per-feature layer).
# ---------------------------------------------------------------------------


def test_parse_block_extracts_headers_and_features():
    p = parse_provenance_block(SAMPLE_BLOCK)
    assert p["analyzer_version"] == "0.2.0"
    assert p["config_hash"] == 0
    assert p["descriptor_hash"] == 12714097049004338389
    assert p["features"]["variance"] == 18198331707036057709
    assert len(p["features"]) == 3


def test_write_then_parse_round_trips():
    feats = {"variance": 111, "edge_density": 222}
    text = write_provenance_block("0.2.0", 0, 9, feats)
    p = parse_provenance_block(text)
    assert p["analyzer_version"] == "0.2.0"
    assert p["config_hash"] == 0
    assert p["descriptor_hash"] == 9
    assert p["features"] == feats


def test_parse_rejects_bad_format_and_missing_headers():
    with pytest.raises(ValueError):
        parse_provenance_block("not a provenance block")
    with pytest.raises(ValueError):
        parse_provenance_block(f"{SERIALIZATION_MAGIC}\nconfig_hash=0\n")


def test_parse_ignores_unknown_headers_forward_compatibly():
    text = (
        f"{SERIALIZATION_MAGIC}\nanalyzer_version=0.2.0\nconfig_hash=0\n"
        "descriptor_hash=9\nfuture_field=whatever\n[features]\nvariance=1\n"
    )
    p = parse_provenance_block(text)
    assert p["features"]["variance"] == 1


def test_feature_is_reusable_needs_all_three_legs():
    p = parse_provenance_block(SAMPLE_BLOCK)
    v = p["features"]["variance"]
    assert feature_is_reusable(p, "variance", v, 0, 12714097049004338389)
    assert not feature_is_reusable(p, "variance", v + 1, 0, 12714097049004338389)  # code
    assert not feature_is_reusable(p, "variance", v, 5, 12714097049004338389)  # config
    assert not feature_is_reusable(p, "variance", v, 0, 7)  # descriptor framing
    assert not feature_is_reusable(p, "absent", v, 0, 12714097049004338389)


def test_assert_consistent_accepts_matching_and_absent():
    # identical blocks + a None (legacy unstamped) -> the agreed block.
    assert assert_consistent_provenance([SAMPLE_BLOCK, SAMPLE_BLOCK, None]) == SAMPLE_BLOCK
    assert assert_consistent_provenance([None, None]) is None


def test_assert_consistent_rejects_descriptor_conflict():
    other = SAMPLE_BLOCK.replace("descriptor_hash=12714097049004338389", "descriptor_hash=999")
    with pytest.raises(ValueError, match="descriptor_hash"):
        assert_consistent_provenance([SAMPLE_BLOCK, other])


def test_assert_consistent_rejects_feature_hash_conflict():
    other = SAMPLE_BLOCK.replace("variance=18198331707036057709", "variance=42")
    with pytest.raises(ValueError, match="version hash"):
        assert_consistent_provenance([SAMPLE_BLOCK, other])


def test_sidecar_read(tmp_path):
    table = tmp_path / "feats.parquet"
    table.write_bytes(b"")  # contents irrelevant; only the sibling sidecar matters
    assert read_provenance_sidecar(table) is None
    (tmp_path / "feats.provenance").write_text(SAMPLE_BLOCK)
    assert read_provenance_sidecar(table) == SAMPLE_BLOCK


def test_parquet_metadata_round_trip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"feat_variance": [1.0, 2.0], "feat_edge_density": [0.1, 0.2]})
    stamped = embed_provenance_in_table(table, SAMPLE_BLOCK)
    # Existing data columns are untouched.
    assert stamped.column_names == table.column_names
    out = tmp_path / "stamped.parquet"
    pq.write_table(stamped, out)
    assert provenance_from_parquet(out) == SAMPLE_BLOCK

    # An unstamped table reads back as None (legacy-safe).
    pq.write_table(table, tmp_path / "plain.parquet")
    assert provenance_from_parquet(tmp_path / "plain.parquet") is None

    # embed is a no-op for a falsy block.
    assert embed_provenance_in_table(table, None) is table


if __name__ == "__main__":
    test_undeclared_emits_no_stamps()
    test_full_provenance_round_trips()
    test_config_hash_defaults_to_zero_when_declared_without_it()
    test_linear_light_config_hash_preserved()
    test_workspace_provenance_reads_the_crate()
    test_parse_block_extracts_headers_and_features()
    test_write_then_parse_round_trips()
    test_parse_ignores_unknown_headers_forward_compatibly()
    test_feature_is_reusable_needs_all_three_legs()
    test_assert_consistent_accepts_matching_and_absent()
    print("all _provenance tests passed")
