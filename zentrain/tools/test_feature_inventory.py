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


# ---------------------------------------------------------------------------
# Cost grid join (#41 cost-vs-use, #50 Sub-A) + family presets
# ---------------------------------------------------------------------------

_COST_HEADER = "class\tside\tpixels\tcrop\ttiles\tfeature_id\tfeature\tbaseline_ns\tsolo_ns\tloo_ns\n"


def _cost_grid(tmp_path: Path) -> Path:
    """Two classes × three sides × two crops × three features. Photo LOO for
    `feat_pricey` grows linearly with pixels (1000 ns + 0.5 ns/px); the crop
    medians are deliberately asymmetric so a mean would differ from the
    median."""
    rows = ["# zenanalyze per-feature cost grid — synthetic test", "# git=abc host=t arch=x date=2026-01-01"]
    for cls, mult in (("photo", 1.0), ("screen", 2.0)):
        for side in (64, 256, 1024):
            px = side * side
            base = 500_000 + 2 * px
            for crop, jitter in ((0, 0.0), (1, 30_000.0)):
                rows.append(f"{cls}\t{side}\t{px}\t{crop}\t1\t7\tfeat_pricey\t{base:.0f}\t{200_000 + px:.0f}\t{(1000 + 0.5 * px) * mult + jitter:.0f}")
                rows.append(f"{cls}\t{side}\t{px}\t{crop}\t1\t8\tfeat_shared\t{base:.0f}\t{150_000:.0f}\t{-50 + jitter * 0.1:.0f}")
                rows.append(f"{cls}\t{side}\t{px}\t{crop}\t1\t9\tfeat_single\t{base:.0f}\t{160_000:.0f}\t{3000 + jitter * 0.01:.0f}")
    p = tmp_path / "cost.tsv"
    p.write_text("\n".join(rows[:2]) + "\n" + _COST_HEADER + "\n".join(rows[2:]) + "\n")
    return p


def _universe_variants(tmp_path: Path) -> Path:
    u = tmp_path / "universe.txt"
    u.write_text("7\tfeat_pricey\tPricey\n8\tfeat_shared\tShared\n9\tfeat_single\tSingle\n10\tfeat_unmeasured\tUnmeasured\n")
    return u


def test_load_cost_takes_crop_medians_and_cell_layout(tmp_path):
    cost = fi.load_cost(_cost_grid(tmp_path))
    assert cost["classes"] == ["photo", "screen"]
    assert cost["sides"] == [64, 256, 1024]
    cell = cost["cells"][("photo", 64)]
    assert cell["pixels"] == 4096
    assert cell["baseline_ns"] == 500_000 + 2 * 4096
    # median of the two crops (jitter 0 / 30 000) — not the min, not the max
    loo0 = 1000 + 0.5 * 4096
    assert cell["features"]["feat_pricey"]["loo_ns"] == (loo0 + loo0 + 30_000) / 2
    assert cell["features"]["feat_pricey"]["n"] == 2


def test_cost_summary_reference_cell_fit_and_class_split(tmp_path):
    cost = fi.load_cost(_cost_grid(tmp_path))
    s = fi.cost_summary(cost, None, None)
    # defaults: first class, largest side
    assert (s["ref_class"], s["ref_side"]) == ("photo", 1024)
    pf = s["per_feature"]["feat_pricey"]
    assert abs(pf["loo_us"] - ((1000 + 0.5 * 1024 * 1024) + 15_000) / 1e3) < 1e-6
    # α + β·px fit over the three sides recovers the planted slope (median over
    # crops adds a constant 15 000 ns to every side, so β is exact, α shifted)
    assert abs(pf["loo_beta_ns_per_px"] - 0.5) < 1e-9
    assert abs(pf["loo_alpha_us"] - (1000 + 15_000) / 1e3) < 1e-6
    # screen doubles the photo LOO at the reference side
    assert abs(pf["loo_us_by_class"]["screen"] - (2 * (1000 + 0.5 * 1024 * 1024) + 15_000) / 1e3) < 1e-6
    assert abs(s["baseline_beta_ns_per_px"] - 2.0) < 1e-9
    # explicit reference cell
    s2 = fi.cost_summary(cost, "screen", 256)
    assert (s2["ref_class"], s2["ref_side"]) == ("screen", 256)
    import pytest

    with pytest.raises(SystemExit):
        fi.cost_summary(cost, "lineart", 256)


def test_aggregate_cost_rankings_and_presets(tmp_path):
    a = fi.load_source("zenjpeg-v9", _manifest(tmp_path, "a", ["feat_shared", "feat_single", "feat_unmeasured"]), None)
    b = fi.load_source("zenwebp-v1", _manifest(tmp_path, "b", ["feat_shared", "feat_foreign"]), None)
    uni = fi.read_universe_meta(_universe_variants(tmp_path))
    assert uni.names == ["feat_pricey", "feat_shared", "feat_single", "feat_unmeasured"]
    assert uni.variants["feat_pricey"] == "Pricey" and uni.ids["feat_single"] == 9
    cost = fi.load_cost(_cost_grid(tmp_path))
    agg = fi.aggregate([a, b], uni, cost)
    assert agg["expensive_unused"] == ["feat_pricey"]
    assert agg["expensive_single"] == ["feat_single"]
    assert agg["hot_path"] == ["feat_shared"]
    # consumption order: consumer count desc, then name
    assert agg["consumed_not_measured"] == ["feat_foreign", "feat_unmeasured"]
    # family presets: union per family, universe order, foreign columns dropped,
    # rendered as compilable Rust from the variant identifiers
    p = agg["presets"]
    assert list(p) == ["jpeg", "webp"]
    assert p["jpeg"]["features"] == ["feat_shared", "feat_single", "feat_unmeasured"]
    assert p["webp"]["features"] == ["feat_shared"]
    assert p["webp"]["not_in_universe"] == ["feat_foreign"]
    assert "pub const JPEG_FAMILY: Self = Self::new()" in p["jpeg"]["rust"]
    assert p["jpeg"]["rust"].rstrip().endswith(".with(AnalysisFeature::Unmeasured);")
    assert ".with(AnalysisFeature::Shared)\n    .with(AnalysisFeature::Single)" in p["jpeg"]["rust"]
    # no variants in the universe → names only, no Rust
    plain = fi.aggregate([a, b], fi.read_universe(_universe_variants(tmp_path)), cost)
    assert plain["presets"]["jpeg"]["rust"] is None


def test_family_of_labels():
    assert fi.family_of("zenjpeg-a-v3-shipped") == "jpeg"
    assert fi.family_of("zenavif-rav1e-v0.1.1-shipped") == "avif"
    assert fi.family_of("jxl-lossy") == "jxl"
    assert fi.family_of("meta-v0.5-5codec") == "meta"
    assert fi.family_of("coefficient-x") == "coefficient"


def test_universe_three_column_lines_keep_the_feature_name(tmp_path):
    # `id<TAB>feat_name<TAB>Variant` must not be read as the Variant column
    assert fi.read_universe(_universe_variants(tmp_path)) == ["feat_pricey", "feat_shared", "feat_single", "feat_unmeasured"]
    two = tmp_path / "two.txt"
    two.write_text("3\tfeat_x\nfeat_y\n")
    assert fi.read_universe(two) == ["feat_x", "feat_y"]


def test_markdown_with_cost_renders_sections(tmp_path):
    a = _manifest(tmp_path, "zenjpeg-a", ["feat_shared", "feat_single"])
    b = _manifest(tmp_path, "zenwebp-b", ["feat_shared"])
    out = tmp_path / "report.md"
    rc = fi.main([
        "--universe", str(_universe_variants(tmp_path)), "--cost", str(_cost_grid(tmp_path)),
        "--cost-class", "screen", "--cost-side", "256",
        str(a), str(b), "--out", str(out), "--generated", "2026-01-01",
    ])
    assert rc == 0
    text = out.read_text()
    assert "| feature | n | solo µs | LOO µs | zenjpeg-a | zenwebp-b |" in text
    assert "Reference cell: **screen 256²**" in text
    assert "### Supported but consumed by no listed bake — ranked by LOO (1)" in text
    assert "| `feat_pricey` | 0 |" in text
    assert "### Consumed by exactly one bake — ranked by LOO (1)" in text
    assert "### Shared by ≥ 2 bakes — ranked by LOO (1)" in text
    assert "## Family presets (proposed `FeatureSet` constants)" in text
    assert "pub const WEBP_FAMILY: Self = Self::new()\n    .with(AnalysisFeature::Shared);" in text
    # JSON mode carries the cost aggregate too
    j = tmp_path / "r.json"
    assert fi.main(["--universe", str(_universe_variants(tmp_path)), "--cost", str(_cost_grid(tmp_path)), str(a), str(b), "--json", "--out", str(j)]) == 0
    d = json.loads(j.read_text())
    assert d["expensive_unused"] == ["feat_pricey"] and d["cost"]["ref_side"] == 1024


def test_real_cost_grid_in_repo_parses():
    grids = sorted((REPO_ROOT / "benchmarks").glob("per_feature_cost_grid_*.tsv"))
    assert grids, "benchmarks/per_feature_cost_grid_<date>.tsv is committed"
    cost = fi.load_cost(grids[-1])
    assert set(cost["classes"]) >= {"photo", "screen"}
    assert set(cost["sides"]) >= {64, 256, 1024, 2048, 4096}
    s = fi.cost_summary(cost, "photo", 2048)
    assert "feat_variance" in s["per_feature"]
    assert s["baseline_us"] > 0
