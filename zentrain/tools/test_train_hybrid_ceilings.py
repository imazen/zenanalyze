"""Per-(image, size_class) metric ceilings in the trainer (zenanalyze#51):
`ceiling_column_for` picks `effective_max_<METRIC_COLUMN>` (never a zensim
ceiling for a non-zensim picker), `load_pareto` reads it from Parquet, and
`build_dataset` skips targets above the ceiling instead of turning them into
data-starved rows.

Run with: `python3 -m pytest zentrain/tools/test_train_hybrid_ceilings.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = ["train_hybrid.py"]  # keep argv-at-import paths inert

import train_hybrid as th  # noqa: E402


def test_ceiling_column_is_metric_specific():
    cols = ["image_path", "zensim", "ssim2", "effective_max_zensim"]
    assert th.ceiling_column_for("zensim", cols) == "effective_max_zensim"
    # An ssim2 picker must NOT borrow the zensim ceiling (different units).
    assert th.ceiling_column_for("ssim2", cols) is None
    assert th.ceiling_column_for("ssim2", cols + ["effective_max_ssim2"]) == "effective_max_ssim2"
    assert th.ceiling_column_for("zensim", ["zensim"]) is None


def _pareto_parquet(path: Path) -> None:
    pq.write_table(pa.table({
        "image_path": ["/a", "/a", "/b"],
        "size_class": ["small", "small", "small"],
        "width": [100, 100, 100], "height": [100, 100, 100],
        "config_id": [0, 1, 0], "config_name": ["c0", "c1", "c0"],
        "bytes": [100, 90, 200],
        "zensim": [60.0, 65.0, 80.0], "ssim2": [50.0, 55.0, 70.0],
        "encode_ms": [1.0, 2.0, 3.0],
        "effective_max_zensim": [65.0, 65.0, 80.0],
        "effective_max_ssim2": [55.0, 55.0, 70.0],
    }), path)


def test_load_pareto_reads_the_ceiling_for_the_trained_metric(tmp_path, monkeypatch):
    p = tmp_path / "pareto.parquet"
    _pareto_parquet(p)
    monkeypatch.setattr(th, "METRIC_COLUMN", "ssim2")
    rows, ceilings, has_ceiling, has_time = th.load_pareto(p)
    assert has_ceiling and has_time
    assert ceilings == {("/a", "small"): 55.0, ("/b", "small"): 70.0}
    monkeypatch.setattr(th, "METRIC_COLUMN", "zensim")
    _rows, ceilings, has_ceiling, _t = th.load_pareto(p)
    assert has_ceiling and ceilings[("/a", "small")] == 65.0
    # Drop the ssim2 ceiling column: an ssim2 picker is uncapped, not mis-capped.
    t = pq.read_table(p).drop_columns(["effective_max_ssim2"])
    pq.write_table(t, p)
    monkeypatch.setattr(th, "METRIC_COLUMN", "ssim2")
    _rows, ceilings, has_ceiling, _t = th.load_pareto(p)
    assert not has_ceiling and ceilings == {}


def test_build_dataset_skips_targets_above_the_ceiling(monkeypatch):
    monkeypatch.setattr(th, "ZQ_TARGETS", [50, 70, 90], raising=False)
    monkeypatch.setattr(th, "SCALAR_AXES", [])
    monkeypatch.setattr(th, "SCALAR_SENTINELS", {})
    monkeypatch.setattr(th, "METRIC_DIRECTION", "higher_better")
    monkeypatch.setattr(th, "REACH_UNDERSHOOT", 0.0)
    key = ("/a", "small", 100, 100)
    pareto = {key: {
        "config_id": np.array([0, 1], dtype=np.int64),
        "bytes": np.array([100, 90], dtype=np.int64),
        "zensim": np.array([60.0, 75.0]),
    }}
    feats = {("/a", "small"): np.array([0.5], dtype=np.float32)}
    cells = [{"label": "c0"}, {"label": "c1"}]
    c2c = {0: 0, 1: 1}
    parsed = {0: {}, 1: {}}
    # No ceiling: zq 90 becomes a row with nothing reachable → dropped anyway,
    # but only because nothing reached; with a ceiling of 75 the 90 target is
    # skipped up front (same rows here, different reason — the counter).
    *_, meta_nc, _tl, _ml, _inf = th.build_dataset(pareto, feats, ["f0"], cells, c2c, parsed)
    assert [m[2] for m in meta_nc] == [50, 70]
    *_, meta_c, _tl, _ml, _inf = th.build_dataset(
        pareto, feats, ["f0"], cells, c2c, parsed, ceilings={("/a", "small"): 75.0}
    )
    assert [m[2] for m in meta_c] == [50, 70]
    # A ceiling BELOW an otherwise-reachable target removes that row: the
    # sweep says 75 is the max for this image, so a 70 target is kept and a
    # 90 one is not — and a ceiling of 65 also drops the 70 row.
    *_, meta_low, _tl, _ml, _inf = th.build_dataset(
        pareto, feats, ["f0"], cells, c2c, parsed, ceilings={("/a", "small"): 65.0}
    )
    assert [m[2] for m in meta_low] == [50]


# ---------------------------------------------------------------- fit tool


def test_fit_zensim_ceiling_learns_a_feature_driven_ceiling_and_emits_a_bakeable_student(tmp_path):
    import fit_zensim_ceiling as fzc

    rng = np.random.default_rng(1)
    n = 240
    paths, sizes, ws, hs, feats, ceilings = [], [], [], [], [], []
    for i in range(n):
        f = float(rng.uniform(0.0, 1.0))
        size = ["tiny", "small", "medium"][i % 3]
        side = {"tiny": 40, "small": 200, "medium": 800}[size]
        # ceiling is a clean function of the feature + size (+ tiny noise)
        c = 70.0 + 25.0 * f - (6.0 if size == "tiny" else 0.0) + float(rng.normal(0, 0.2))
        paths.append(f"/data/o_{1000 + i}.png.scale{side}x{side}.png")
        sizes.append(size); ws.append(side); hs.append(side); feats.append(f); ceilings.append(c)
    # two configs per image so the pareto has rows; zensim below the ceiling
    tbl = {
        "image_path": paths * 2, "size_class": sizes * 2, "width": ws * 2, "height": hs * 2,
        "config_id": [0] * n + [1] * n, "config_name": ["c0"] * n + ["c1"] * n,
        "bytes": [1000] * n + [900] * n,
        "zensim": [c - 5.0 for c in ceilings] + list(ceilings),
        "encode_ms": [1.0] * (2 * n),
        "effective_max_zensim": ceilings * 2,
    }
    pareto = tmp_path / "pareto.parquet"
    pq.write_table(pa.table(tbl), pareto)
    ftsv = tmp_path / "features.tsv"
    with open(ftsv, "w") as fh:
        fh.write("image_path\tsize_class\twidth\theight\tfeat_alpha\tfeat_noise\n")
        for p, s, w, h, f in zip(paths, sizes, ws, hs, feats):
            fh.write(f"{p}\t{s}\t{w}\t{h}\t{f}\t{rng.uniform()}\n")

    def split_of(name: str):
        d = int(name.split("o_")[1].split(".")[0]) % 10
        return "train" if d % 2 == 0 else ("val" if d in (1, 3, 5) else "test")

    out = tmp_path / "fit.tsv"
    md = tmp_path / "fit.md"
    sj = tmp_path / "student.json"
    rc = fzc.main([
        "--pareto", str(pareto), "--features", str(ftsv), "--out-tsv", str(out),
        "--md", str(md), "--student-json", str(sj), "--hidden", "32,32",
    ], split_of=split_of)
    assert rc == 0
    rows = [r.split("\t") for r in out.read_text().splitlines() if not r.startswith("#")]
    hdr = rows[0]
    recs = [dict(zip(hdr, r)) for r in rows[1:]]
    teacher_val = next(r for r in recs if r["section"] == "teacher" and r["split"] == "val" and r["size_class"] == "all")
    assert float(teacher_val["r2"]) > 0.9 and float(teacher_val["mae"]) < 1.5
    assert {r["size_class"] for r in recs if r["split"] == "val"} >= {"all", "tiny", "small", "medium"}
    assert 0.0 <= float(teacher_val["over_0"]) <= 1.0 and float(teacher_val["over_5"]) <= float(teacher_val["over_0"])
    # Student JSON is the bake_picker input shape.
    import json
    m = json.loads(sj.read_text())
    assert m["n_outputs"] == 1 and m["n_inputs"] == 2 + 5
    assert m["feat_cols"] == ["feat_alpha", "feat_noise"] and m["extra_axes"] == fzc.EXTRA_AXES
    assert len(m["scaler_mean"]) == m["n_inputs"] and len(m["layers"]) == 3
    assert m["layers"][-1]["b"] and len(m["layers"][-1]["W"][0]) == 1
    assert m["training_objective"] == {"name": "ceiling", "metric_name": "zensim"}
    assert len(m["nan_fill"]) == m["n_inputs"]
    assert "student" in md.read_text() and "over_2" in md.read_text()


def test_fit_zensim_ceiling_refuses_a_pareto_without_the_ceiling_column(tmp_path):
    import fit_zensim_ceiling as fzc

    pareto = tmp_path / "pareto.parquet"
    pq.write_table(pa.table({
        "image_path": ["/a"], "size_class": ["small"], "width": [10], "height": [10],
        "config_id": [0], "config_name": ["c"], "bytes": [10], "zensim": [50.0],
    }), pareto)
    ftsv = tmp_path / "f.tsv"
    ftsv.write_text("image_path\tsize_class\twidth\theight\tfeat_a\n/a\tsmall\t10\t10\t0.5\n")
    with pytest.raises(SystemExit, match="effective_max_zensim"):
        fzc.main(["--pareto", str(pareto), "--features", str(ftsv), "--out-tsv", str(tmp_path / "o.tsv")],
                 split_of=lambda _n: "train")
