#!/usr/bin/env python3
"""
Canonical per-codec picker dataset (s3://zentrain/canonical/<date>/<codec>/)
→ the (PARETO, FEATURES) pair `train_hybrid.py` consumes.

Why this exists
---------------
The canonical datasets (`zenmetrics/scripts/picker/build_canonical.py`) are one
row per (image × codec × q × knob-tuple) with the zenanalyze features joined in
by NAME (`feat_variance`, …) next to zensim's positional `feat_0..feat_371`.
`train_hybrid.py` wants two files instead: a Pareto table keyed by
`(image_path, size_class, width, height)` with `config_id` / `config_name` /
`bytes` / `<metric>` / `encode_ms` columns, and a features table with one row
per `(image_path, size_class)`. The 2026-07-03 zenjxl-modular bake did this
conversion in a scratch script that was never committed (see
`zentrain/examples/zenjxl_modular_picker_config.py`); this is that step, in
tree, so a canonical dataset can be trained on without re-deriving it.

What it does
------------
- `config_name` = the `cell` column (the knob tuple's human label, e.g.
  `vp8-m4_plim50-syuv`); `config_id` = `stable_config_id(config_name)` from
  `zenmetrics_sweep_adapter` (sha256-derived, stable across runs).
- `size_class` is RE-DERIVED from `width × height` with the zenwebp buckets
  (`zenmetrics_sweep_adapter.size_class`: tiny ≤ 64, small ≤ 256, medium ≤ 1024,
  large) — the canonical column is `large` for every row of the 2026-06-27
  datasets regardless of rendition size, which would collapse the picker's
  size axis.
- The features table keeps only the NAMED zenanalyze columns (`feat_<name>`);
  the positional zensim `feat_<N>` columns are dropped (they are the metric's
  inputs, not picker features). One row per image; the tool verifies the
  features are identical across every row of an image before de-duplicating.
- Pass both `train.parquet` and `validate.parquet`: `train_hybrid` derives its
  own origin even/odd split (`origin_split.py`) from the image names, so a
  single split file would leave it with zero validation rows.

Parquet reading goes through pyarrow directly: the canonical layout is not a
Pareto/features file, so `_picker_lib.load_pareto_raw` / `load_features_raw`
(the owners for THOSE shapes) do not apply — this tool is what produces their
input.

Usage
-----
    python3 canonical_to_pareto.py \\
        --canonical ~/tmp/canonical/zenwebp_lossy/train.parquet \\
                    ~/tmp/canonical/zenwebp_lossy/validate.parquet \\
        --pareto-out   ~/tmp/canonical/zenwebp_lossy/derived/pareto.parquet \\
        --features-out ~/tmp/canonical/zenwebp_lossy/derived/features.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenmetrics_sweep_adapter import size_class, stable_config_id  # noqa: E402

_POSITIONAL = re.compile(r"^feat_\d+$")
META_COLS = ["image_path", "width", "height", "cell", "q", "encoded_bytes", "encode_ms", "decode_ms", "score_zensim", "score_ssim2"]


def named_feature_columns(names: list[str]) -> list[str]:
    return [c for c in names if c.startswith("feat_") and not _POSITIONAL.match(c)]


def load_canonical(paths: list[Path], config_col: str) -> pa.Table:
    tables = []
    for p in paths:
        schema = pq.read_schema(p)
        feats = named_feature_columns(schema.names)
        want = [c if c != "cell" else config_col for c in META_COLS]
        missing = [c for c in want if c not in schema.names]
        if missing:
            raise SystemExit(f"{p}: missing columns {missing}")
        t = pq.read_table(p, columns=want + feats)
        if config_col != "cell":
            t = t.rename_columns([("cell" if c == config_col else c) for c in t.column_names])
        tables.append(t)
    if len({tuple(t.column_names) for t in tables}) != 1:
        raise SystemExit("canonical inputs disagree on columns; convert them separately")
    return pa.concat_tables(tables)


def build_pareto(t: pa.Table) -> pa.Table:
    width = t["width"].to_numpy().astype(np.int64)
    height = t["height"].to_numpy().astype(np.int64)
    sizes = [size_class(int(w), int(h)) for w, h in zip(width, height)]
    cells = t["cell"].to_pylist()
    ids = {c: stable_config_id(c) for c in set(cells)}
    return pa.table({
        "image_path": t["image_path"],
        "size_class": pa.array(sizes),
        "width": pa.array(width),
        "height": pa.array(height),
        "config_id": pa.array([ids[c] for c in cells], type=pa.int64()),
        "config_name": pa.array(cells),
        "bytes": pc.cast(t["encoded_bytes"], pa.int64()),
        "zensim": pc.cast(t["score_zensim"], pa.float64()),
        "ssim2": pc.cast(t["score_ssim2"], pa.float64()),
        "encode_ms": pc.cast(t["encode_ms"], pa.float64()),
        "decode_ms": pc.cast(t["decode_ms"], pa.float64()),
        "q": pc.cast(t["q"], pa.float64()),
    })


def build_features(t: pa.Table, pareto: pa.Table) -> tuple[list[str], list[list]]:
    """One row per image_path. Verifies every feature column is constant within
    an image (min == max, NaN-aware) before taking the first row."""
    feats = named_feature_columns(t.column_names)
    paths = np.asarray(t["image_path"].to_pylist(), dtype=object)
    uniq, first = np.unique(paths, return_index=True)
    order = np.argsort(first)
    uniq, first = uniq[order], first[order]
    grouped = t.select(["image_path"] + feats).group_by("image_path").aggregate(
        [(c, "min") for c in feats] + [(c, "max") for c in feats]
    )
    bad = []
    for c in feats:
        lo = grouped[f"{c}_min"].to_numpy(zero_copy_only=False).astype(np.float64)
        hi = grouped[f"{c}_max"].to_numpy(zero_copy_only=False).astype(np.float64)
        both_nan = np.isnan(lo) & np.isnan(hi)
        if not np.all(both_nan | (lo == hi)):
            bad.append(c)
    if bad:
        raise SystemExit(f"features vary within an image (not a per-image feature?): {bad[:10]}")
    size_by_path = dict(zip(pareto["image_path"].to_pylist(), pareto["size_class"].to_pylist()))
    w_by_path = dict(zip(pareto["image_path"].to_pylist(), pareto["width"].to_pylist()))
    h_by_path = dict(zip(pareto["image_path"].to_pylist(), pareto["height"].to_pylist()))
    cols = {c: t[c].to_numpy(zero_copy_only=False) for c in feats}
    header = ["image_path", "size_class", "width", "height"] + feats
    rows = []
    for p, i in zip(uniq, first):
        rows.append([p, size_by_path[p], w_by_path[p], h_by_path[p]] + [cols[c][i] for c in feats])
    return header, rows


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(_cell(v) for v in r) + "\n")


def _cell(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else repr(float(v))
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", nargs="+", type=Path, required=True, help="canonical split parquet(s) — pass train AND validate")
    ap.add_argument("--pareto-out", type=Path, required=True)
    ap.add_argument("--features-out", type=Path, required=True)
    ap.add_argument("--config-col", default="cell", help="column that names the knob tuple (default: cell)")
    args = ap.parse_args(argv)

    t = load_canonical(args.canonical, args.config_col)
    pareto = build_pareto(t)
    header, rows = build_features(t, pareto)
    args.pareto_out.parent.mkdir(parents=True, exist_ok=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pareto, args.pareto_out, compression="zstd")
    write_tsv(args.features_out, header, rows)

    sizes = pc.value_counts(pareto["size_class"]).to_pylist()
    n_cfg = len(set(pareto["config_name"].to_pylist()))
    sys.stderr.write(
        f"pareto: {pareto.num_rows} rows, {len(rows)} images, {n_cfg} configs, "
        f"size_class={ {d['values']: d['counts'] for d in sizes} } → {args.pareto_out}\n"
        f"features: {len(rows)} rows × {len(header) - 4} named feat_ columns → {args.features_out}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
