"""The canonical-dataset tools: `canonical_to_pareto.py` (canonical parquet →
train_hybrid PARETO/FEATURES), `fit_family_degradation.py` (effort-cap
degradation + cost tables, zenanalyze#85) and the arm parser of
`leakyrelu_seeds_runner.py` (#68). Synthetic 2-image canonical parquets built
in tmp_path; pyarrow + numpy only.

Run with: `python3 -m pytest zentrain/tools/test_canonical_tools.py`
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import canonical_to_pareto as c2p  # noqa: E402
import fit_family_degradation as ffd  # noqa: E402
import leakyrelu_seeds_runner as runner  # noqa: E402


def _canonical(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    p = tmp_path / name
    pq.write_table(pa.table(cols), p)
    return p


def _row(img, w, h, cell, q, nbytes, ms, zensim, fv):
    return {
        "image_path": img, "width": float(w), "height": float(h), "cell": cell, "q": float(q),
        "encoded_bytes": nbytes, "encode_ms": ms, "decode_ms": 1.0, "score_zensim": zensim,
        "score_ssim2": zensim - 10.0, "feat_variance": fv, "feat_edge_density": fv * 2,
        "feat_0": 99.0, "feat_371": 98.0,
    }


def test_canonical_to_pareto_rederives_size_class_and_dedupes_features(tmp_path):
    rows = [
        _row("/data/o_1000.png.scale64x48.png", 64, 48, "vp8-m4_def", 50, 1000, 2.0, 70.0, 1.5),
        _row("/data/o_1000.png.scale64x48.png", 64, 48, "vp8-m6_def", 50, 900, 4.0, 72.0, 1.5),
        _row("/data/o_1001.png.scale500x300.png", 500, 300, "vp8-m4_def", 50, 5000, 20.0, 60.0, 7.0),
    ]
    src = _canonical(tmp_path, "train.parquet", rows)
    pareto = tmp_path / "d" / "pareto.parquet"
    feats = tmp_path / "d" / "features.tsv"
    assert c2p.main(["--canonical", str(src), "--pareto-out", str(pareto), "--features-out", str(feats)]) == 0
    t = pq.read_table(pareto)
    assert t.num_rows == 3
    assert t["size_class"].to_pylist() == ["tiny", "tiny", "medium"]  # max side 64 → tiny; 500 → medium
    assert t["config_name"].to_pylist() == ["vp8-m4_def", "vp8-m6_def", "vp8-m4_def"]
    ids = t["config_id"].to_pylist()
    assert ids[0] == ids[2] != ids[1] and all(i >= 0 for i in ids)
    assert t["bytes"].to_pylist() == [1000, 900, 5000]
    assert t["zensim"].to_pylist() == [70.0, 72.0, 60.0]
    with open(feats) as f:
        rd = list(csv.DictReader(f, delimiter="\t"))
    assert [r["image_path"] for r in rd] == ["/data/o_1000.png.scale64x48.png", "/data/o_1001.png.scale500x300.png"]
    assert rd[0]["size_class"] == "tiny" and rd[1]["width"] == "500"
    # named analyzer features kept, positional zensim feat_N dropped
    assert "feat_variance" in rd[0] and "feat_edge_density" in rd[0]
    assert "feat_0" not in rd[0] and "feat_371" not in rd[0]
    assert float(rd[1]["feat_variance"]) == 7.0


def test_canonical_to_pareto_emits_per_image_size_ceilings(tmp_path):
    # Image A at one size: two cells reach 70 / 72 → ceiling 72 on BOTH rows;
    # image B alone → its own row's value. ssim2 = zensim - 10 in _row.
    rows = [
        _row("/data/o_1000.png.scale64x48.png", 64, 48, "vp8-m4_def", 50, 1000, 2.0, 70.0, 1.5),
        _row("/data/o_1000.png.scale64x48.png", 64, 48, "vp8-m6_def", 100, 900, 4.0, 72.0, 1.5),
        _row("/data/o_1001.png.scale500x300.png", 500, 300, "vp8-m4_def", 50, 5000, 20.0, 60.0, 7.0),
    ]
    src = _canonical(tmp_path, "train.parquet", rows)
    pareto = tmp_path / "d" / "pareto.parquet"
    assert c2p.main(["--canonical", str(src), "--pareto-out", str(pareto), "--features-out", str(tmp_path / "d" / "f.tsv")]) == 0
    t = pq.read_table(pareto)
    assert t["effective_max_zensim"].to_pylist() == [72.0, 72.0, 60.0]
    assert t["effective_max_ssim2"].to_pylist() == [62.0, 62.0, 50.0]
    # NaN-only keys stay NaN instead of -inf.
    out = c2p.per_key_max(["a", "a", "b"], ["tiny", "tiny", "tiny"], np.array([np.nan, np.nan, 3.0]))
    assert math.isnan(out[0]) and math.isnan(out[1]) and out[2] == 3.0


def test_canonical_to_pareto_refuses_features_that_vary_within_an_image(tmp_path):
    rows = [
        _row("/data/o_1000.png.scale64x48.png", 64, 48, "vp8-m4_def", 50, 1000, 2.0, 70.0, 1.5),
        _row("/data/o_1000.png.scale64x48.png", 64, 48, "vp8-m6_def", 50, 900, 4.0, 72.0, 1.6),
    ]
    src = _canonical(tmp_path, "train.parquet", rows)
    with pytest.raises(SystemExit, match="vary within an image"):
        c2p.main(["--canonical", str(src), "--pareto-out", str(tmp_path / "p.parquet"), "--features-out", str(tmp_path / "f.tsv")])


def _degradation_rows():
    """Two medium images; efforts m2/m4/m6. At target 70 image A reaches with
    bytes 1200 (m2), 1100 (m4), 1000 (m6); image B reaches only at m6 (1000).
    A tiny image is present too (separate size_class bucket)."""
    rows = []
    for img, w, h in (("/data/o_1.png.scale400x400.png", 400, 400), ("/data/o_2.png.scale400x400.png", 400, 400)):
        for m, nbytes, ms in ((2, 1200, 10.0), (4, 1100, 20.0), (6, 1000, 40.0)):
            reaches = (img.endswith("o_1.png.scale400x400.png")) or m == 6
            rows.append(_row(img, w, h, f"vp8-m{m}_def", 50, nbytes, ms, 75.0 if reaches else 60.0, 1.0))
    rows.append(_row("/data/o_3.png.scale32x32.png", 32, 32, "vp8-m6_def", 50, 300, 1.0, 80.0, 1.0))
    rows.append(_row("/data/o_3.png.scale32x32.png", 32, 32, "vp8-m2_def", 50, 330, 0.5, 80.0, 1.0))
    rows.append(_row("/data/o_3.png.scale32x32.png", 32, 32, "odd-cell", 50, 1, 0.5, 99.0, 1.0))
    return rows


def test_fit_family_degradation_webp_semantics(tmp_path):
    src = _canonical(tmp_path, "validate.parquet", _degradation_rows())
    out = tmp_path / "deg.tsv"
    md = tmp_path / "deg.md"
    assert ffd.main(["--codec", "zenwebp", "--canonical", str(src), "--targets", "70", "--out", str(out), "--md", str(md)]) == 0
    rows = [r for r in out.read_text().splitlines() if not r.startswith("#")]
    hdr = rows[0].split("\t")
    deg = [dict(zip(hdr, r.split("\t"))) for r in rows[1:] if r.startswith("degradation")]
    med = {r["cap_effort"]: r for r in deg if r["size_class"] == "medium"}
    # cap m2: image A reaches at 1200 (δ = ln 1.2), image B does not reach → reach 1/2, mean over the one δ
    assert med["2"]["n"] == "2" and float(med["2"]["reach_frac"]) == 0.5
    assert abs(float(med["2"]["delta_mean_ln"]) - math.log(1.2)) < 1e-4
    assert abs(float(med["2"]["delta_mean_pct"]) - 20.0) < 0.01
    assert abs(float(med["4"]["delta_mean_ln"]) - math.log(1.1)) < 1e-4
    assert float(med["6"]["delta_mean_ln"]) == 0.0 and float(med["6"]["reach_frac"]) == 1.0
    tiny = {r["cap_effort"]: r for r in deg if r["size_class"] == "tiny"}
    assert abs(float(tiny["2"]["delta_mean_ln"]) - math.log(1.1)) < 1e-4
    # cost section: ms/MP per (effort, size_class); m6 medium = 40 ms / 0.16 MP = 250
    cost_hdr_i = next(i for i, r in enumerate(rows) if r.startswith("section\tcodec\tsize_class\teffort"))
    chdr = rows[cost_hdr_i].split("\t")
    cost = [dict(zip(chdr, r.split("\t"))) for r in rows[cost_hdr_i + 1:] if r.startswith("cost")]
    m6_med = next(r for r in cost if r["effort"] == "6" and r["size_class"] == "medium")
    assert abs(float(m6_med["ms_per_mp_p50"]) - 250.0) < 1e-6
    text = md.read_text()
    assert "cap ≤ m2" in text and "cap ≤ m4" in text and "reference **m6**" in text
    assert "unmatched_cells=1" in out.read_text()


def test_fit_family_degradation_inverted_knob_for_rav1e_speed(tmp_path):
    # s2 = slowest/best (reference), s8 = fastest; capping at s8 means speeds ≥ 8 only.
    rows = []
    img = "/data/o_1.png.scale400x400.png"
    for s, nbytes in ((2, 1000), (4, 1050), (8, 1300)):
        rows.append(_row(img, 400, 400, f"s{s}-420", 50, nbytes, 10.0, 75.0, 1.0))
    src = _canonical(tmp_path, "validate.parquet", rows)
    out = tmp_path / "deg.tsv"
    md = tmp_path / "deg.md"
    assert ffd.main(["--codec", "zenavif", "--canonical", str(src), "--targets", "70", "--out", str(out), "--md", str(md)]) == 0
    lines = [r for r in out.read_text().splitlines() if r.startswith("degradation")]
    by_cap = {r.split("\t")[4]: r.split("\t") for r in lines}
    assert set(by_cap) == {"2", "4", "8"}
    assert abs(float(by_cap["8"][7]) - math.log(1.3)) < 1e-4   # only s8 allowed → 1300
    assert abs(float(by_cap["4"][7]) - math.log(1.05)) < 1e-4  # s4 and s8 allowed → 1050
    assert float(by_cap["2"][7]) == 0.0
    assert "reference **s2**" in md.read_text() and "cap ≥ s8" in md.read_text()


def test_runner_arm_parsing_and_out_json_suffix(monkeypatch, tmp_path):
    a = runner.parse_arm("leakyrelu+wd=1e-4")
    assert a == {"spec": "leakyrelu+wd=1e-4", "activation": "leakyrelu", "weight_decay": 1e-4, "slug": "leakyrelu_wd1e-4"}
    assert runner.parse_arm("relu")["weight_decay"] == 0.0 and runner.parse_arm("relu")["slug"] == "relu"
    with pytest.raises(SystemExit):
        runner.parse_arm("relu+lr=3")
    # OUT_JSON + train_hybrid's `--out-suffix` convention: suffix appended to the stem
    cfg = tmp_path / "fake_cfg.py"
    cfg.write_text("from pathlib import Path\nOUT_JSON = Path('/x/y/model.json')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert runner.out_json_for("fake_cfg", "_relu_seed_cafe") == Path("/x/y/model_relu_seed_cafe.json")
    assert runner.parse_seed("0xCAFE") == 0xCAFE and runner.parse_seed("7") == 7


def test_canonical_picker_config_parses_every_codec_cell_vocabulary(monkeypatch, tmp_path):
    # The generic config imports without the derived files present (KEEP_FEATURES = []).
    monkeypatch.setenv("ZENCANONICAL_CODEC", "zenjpeg")
    monkeypatch.setenv("ZENCANONICAL_DIR", str(tmp_path))
    sys.path.insert(0, str(HERE.parent / "examples"))
    import importlib

    cfg = importlib.import_module("canonical_picker_config")
    cfg = importlib.reload(cfg)
    assert cfg.CODEC == "zenjpeg" and cfg.OUT_JSON.name == "zenjpeg_canonical_hybrid.json"
    assert cfg.KEEP_FEATURES == [] and cfg.CATEGORICAL_AXES == ["mode"] and cfg.SCALAR_AXES == []
    for cell in ("vp8-m4_plim50-syuv", "jp3[0.5,0.5]_tr14.75+dc_small_444", "gls_t0_small_xybBq",
                 "vd-e7_zen_kinfo1.3", "s2-noqm-420-bd10"):
        assert cfg.parse_config_name(cell) == {"mode": cell}
    for bad in ("", "has space", "-leading", "tab\tcell"):
        with pytest.raises(ValueError):
            cfg.parse_config_name(bad)


def test_real_degradation_tables_in_repo_parse():
    repo = HERE.parent.parent
    tsvs = sorted((repo / "benchmarks").glob("family_degradation_*_2026-08-28.tsv"))
    assert {p.name.split("_")[2] for p in tsvs} >= {"zenwebp", "zenjxl", "zenavif"}
    for p in tsvs:
        lines = p.read_text().splitlines()
        assert lines[0].startswith("# family degradation + cost table")
        assert any(l.startswith("degradation\t") for l in lines) and any(l.startswith("cost\t") for l in lines)
