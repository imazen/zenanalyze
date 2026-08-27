"""`tools/feature_inventory.py` (zenanalyze#41) — the per-bake feature
consumption aggregator: source readers, engineered-axis filtering,
positional-name detection, and the ≥2-bake intersection / never-consumed
report.

Run with: `python3 -m pytest zentrain/tools/test_feature_inventory.py`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import feature_inventory as fi  # noqa: E402


def _manifest(tmp_path: Path, name: str, feat_cols: list[str], **extra) -> Path:
    p = tmp_path / f"{name}.manifest.json"
    d = {"feat_cols": feat_cols, "schema_hash": 0x1234, "schema_version_tag": "zentrain.v1.test"}
    d.update(extra)
    p.write_text(json.dumps(d))
    return p


def test_normalise_drops_engineered_axes_and_prefixes_bare_names():
    assert fi.normalise("feat_variance") == "feat_variance"
    assert fi.normalise("variance") == "feat_variance"
    for eng in ("zq_x_feat_variance", "size_tiny", "log_pixels", "log_pixels_sq", "zq_norm",
                "zq_norm_sq", "zq_norm_x_log_pixels", "icc_bytes", "aux_03"):
        assert fi.normalise(eng) is None, eng
    # `feat_log_pixels` IS an analyzer feature (the dimension tier); only the
    # bare engineered `log_pixels` axis is dropped.
    assert fi.normalise("feat_log_pixels") == "feat_log_pixels"
    assert fi.normalise("   ") is None


def test_feature_order_reader_accepts_idx_tab_name_and_bare_lines(tmp_path):
    p = tmp_path / "feature_order.txt"
    p.write_text("feat_0\tfeat_variance\nfeat_1\tfeat_edge_density\n# comment\n\nfeat_uniformity\n")
    names, meta = fi.read_feature_order(p)
    assert names == ["feat_variance", "feat_edge_density", "feat_uniformity"]
    src = fi.load_source("jpeg", p, None)
    assert src.features == ["feat_variance", "feat_edge_density", "feat_uniformity"]
    assert not src.positional


def test_positional_only_source_is_flagged_not_counted(tmp_path):
    dump = tmp_path / "positional.json"
    dump.write_text(json.dumps({
        "metadata": [{"key": "zenpicker_train.image_feature_names", "value_text": "feat_0,feat_1,feat_2"}],
        "schema_hash": "0x0",
        "n_inputs": 3,
    }))
    src = fi.load_source("pos", dump, None)
    assert src.positional and src.features == ["feat_0", "feat_1", "feat_2"]
    named = fi.load_source("m", _manifest(tmp_path, "m", ["feat_variance"]), None)
    agg = fi.aggregate([src, named], None)
    assert agg["n_named_sources"] == 1
    assert list(agg["consumption"]) == ["feat_variance"]


def test_inspect_dump_prefers_zentrain_feature_columns(tmp_path):
    dump = tmp_path / "bake.json"
    dump.write_text(json.dumps({
        "metadata": [
            {"key": "zenpicker_train.image_feature_names", "value_text": "feat_0,feat_1"},
            {"key": "zentrain.feature_columns", "value_text": "feat_variance\nfeat_uniformity\n"},
            {"key": "zentrain.analyzer_version", "value_text": "0.2.7"},
        ],
        "schema_hash": "0xabc",
        "n_inputs": 9,
    }))
    src = fi.load_source("b", dump, None)
    assert src.features == ["feat_variance", "feat_uniformity"]
    assert src.meta["zentrain.analyzer_version"] == "0.2.7"
    assert src.meta["schema_hash"] == "0xabc"


def test_aggregate_intersection_exclusives_and_universe(tmp_path):
    a = fi.load_source("a", _manifest(tmp_path, "a", ["feat_variance", "feat_uniformity", "feat_only_a", "zq_x_feat_variance"]), None)
    b = fi.load_source("b", _manifest(tmp_path, "b", ["feat_uniformity", "feat_variance", "feat_only_b"]), None)
    c = fi.load_source("c", _manifest(tmp_path, "c", ["feat_variance"]), None)
    universe = ["feat_variance", "feat_uniformity", "feat_only_a", "feat_only_b", "feat_unused", "feat_unused2"]
    agg = fi.aggregate([a, b, c], universe)
    # sorted by consumer count desc, then name
    assert list(agg["consumption"]) == ["feat_variance", "feat_uniformity", "feat_only_a", "feat_only_b"]
    assert agg["consumption"]["feat_variance"] == ["a", "b", "c"]
    assert agg["shared"] == ["feat_variance", "feat_uniformity"]
    assert agg["exclusive"] == {"a": ["feat_only_a"], "b": ["feat_only_b"], "c": []}
    assert agg["never_consumed"] == ["feat_unused", "feat_unused2"]
    assert agg["unknown_to_universe"] == []
    assert agg["universe_size"] == 6


def test_unknown_to_universe_is_reported(tmp_path):
    a = fi.load_source("a", _manifest(tmp_path, "a", ["feat_variance", "feat_retired"]), None)
    agg = fi.aggregate([a], ["feat_variance"])
    assert agg["unknown_to_universe"] == ["feat_retired"]
    assert agg["never_consumed"] == []


def test_markdown_render_and_cli_roundtrip(tmp_path):
    a = _manifest(tmp_path, "a", ["feat_variance", "feat_uniformity"])
    b = _manifest(tmp_path, "b", ["feat_variance"])
    uni = tmp_path / "universe.txt"
    uni.write_text("feat_variance\nfeat_uniformity\nfeat_unused\n")
    out = tmp_path / "report.md"
    rc = fi.main(["--universe", str(uni), "--label", f"first={a}", str(b), "--out", str(out), "--generated", "2026-01-01"])
    assert rc == 0
    text = out.read_text()
    assert "| first |" in text and "| b |" in text
    assert "## Shared by ≥ 2 bakes (1)" in text
    assert "- `feat_variance` — first, b" in text
    assert "## Never consumed (1 of 3 in the universe)" in text
    assert "- `feat_unused`" in text
    assert "generated: 2026-01-01" in text
    # JSON mode yields the same aggregate
    j = tmp_path / "r.json"
    rc = fi.main(["--label", f"first={a}", str(b), "--json", "--out", str(j)])
    assert rc == 0
    assert json.loads(j.read_text())["shared"] == ["feat_variance"]


def test_real_in_repo_manifest_loads():
    p = REPO_ROOT / "zentrain" / "testdata" / "zenjpeg_picker_v2.1_full.manifest.json"
    src = fi.load_source("zenjpeg-v2.1", p, None)
    assert len(src.features) == 35
    assert "feat_variance" in src.features
    assert not src.positional
    assert src.meta["schema_version_tag"]
