#!/usr/bin/env python3
"""Convert a unified sweep parquet into the zentrain Pareto + features
TSVs so `zentrain/tools/train_hybrid.py` can be run on the SAME data
the Rust `zenpicker-train` MLP port consumes — for an apples-to-apples
Rust-vs-zentrain held-out comparison.

The unified parquet (e.g. unified_v13_zenjpeg_cvvdp.parquet) has one row
per (image, knob-config, q):
  image_basename | codec | q | knob_tuple_json | encoded_bytes |
  score_zensim | feat_0..feat_299 | ...

zentrain's `load_pareto` needs:
  image_path | size_class | width | height | config_id | config_name |
  bytes | zensim
and `load_features` needs (keyed on image_path + size_class):
  image_path | size_class | feat_*

The categorical CELL is the discrete knob tuple
(subsampling|progressive|sharp_yuv|effort), identical to the Rust port.
Each distinct cell gets a stable `config_id` + `config_name`. Because
this parquet's knobs are all categorical (no continuous Pareto scalar
like chroma_scale/lambda), every config maps 1:1 to a cell, so the
hybrid trainer's bytes head IS the whole picker — same as the Rust port.

Outputs (TSV):
  <out_dir>/pareto.tsv     — the Pareto sweep
  <out_dir>/features.tsv   — the per-image features
  <out_dir>/codec_config.py — a zentrain codec-config module matching
                              the synthesized config_name grammar +
                              feat_0..feat_N feature set.

Run:
  python3 unified_to_zentrain.py --input <parquet> --out-dir <dir> [--codec zenjpeg]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def size_class_of(w: int, h: int) -> str:
    """zentrain's SIZE_CLASSES bucket by max(width, height)."""
    m = max(w, h)
    if m <= 64:
        return "tiny"
    if m <= 256:
        return "small"
    if m <= 1024:
        return "medium"
    return "large"


def cell_label(knob_json: str) -> str:
    """Mirror the Rust `cell_key_from_knob` canonical ordering, but emit
    a config_name the synthesized regex parser can read back."""
    import json

    try:
        obj = json.loads(knob_json)
    except Exception:
        return "rawcfg"
    sub = obj.get("subsampling", "420")
    prog = 1 if obj.get("progressive", False) else 0
    sharp = 1 if obj.get("sharp_yuv", False) else 0
    eff = int(obj.get("effort", 0))
    # config_name grammar: sub<S>_e<E>_p<0|1>_s<0|1>
    return f"sub{sub}_e{eff}_p{prog}_s{sharp}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--codec", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.input)
    cols = {n: table.column(n) for n in table.column_names}
    n = len(table)

    feat_names = sorted(
        [c for c in table.column_names if c.startswith("feat_")],
        key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 1 << 30,
    )

    image = cols["image_basename"].to_pylist() if "image_basename" in cols else cols["image_path"].to_pylist()
    codec = cols["codec"].to_pylist() if "codec" in cols else [""] * n
    q = cols["q"].to_numpy(zero_copy_only=False)
    knob = cols["knob_tuple_json"].to_pylist()
    enc_bytes = cols["encoded_bytes"].to_numpy(zero_copy_only=False)
    score = cols["score_zensim"].to_numpy(zero_copy_only=False)
    width = cols["width"].to_numpy(zero_copy_only=False) if "width" in cols else None
    height = cols["height"].to_numpy(zero_copy_only=False) if "height" in cols else None
    feat_arrs = {fn: cols[fn].to_numpy(zero_copy_only=False) for fn in feat_names}

    want = args.codec.lower() if args.codec else None

    # Assign stable config ids per cell label.
    cell_ids: dict[str, int] = {}
    rows_pareto = []
    feat_seen: dict[str, list] = {}

    for i in range(n):
        if want is not None and want not in str(codec[i]).lower():
            continue
        s = float(score[i])
        b = float(enc_bytes[i])
        if not np.isfinite(s) or not np.isfinite(b) or b <= 0:
            continue
        feats_ok = True
        for fn in feat_names:
            if not np.isfinite(feat_arrs[fn][i]):
                feats_ok = False
                break
        if not feats_ok:
            continue
        cl = cell_label(knob[i])
        cid = cell_ids.setdefault(cl, len(cell_ids))
        img = str(image[i])
        w = int(width[i]) if width is not None and np.isfinite(width[i]) else 256
        h = int(height[i]) if height is not None and np.isfinite(height[i]) else 256
        sc = size_class_of(w, h)
        rows_pareto.append((img, sc, w, h, cid, cl, int(round(b)), s))
        if img not in feat_seen:
            feat_seen[img] = (sc, [float(feat_arrs[fn][i]) for fn in feat_names])

    if not rows_pareto:
        raise SystemExit("no rows after filter")

    # Pareto TSV.
    pareto_path = args.out_dir / "pareto.tsv"
    with pareto_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            ["image_path", "size_class", "width", "height", "config_id", "config_name", "bytes", "zensim"]
        )
        for r in rows_pareto:
            w.writerow(r)

    # Features TSV (one row per image).
    feat_path = args.out_dir / "features.tsv"
    with feat_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["image_path", "size_class", *feat_names])
        for img, (sc, vals) in feat_seen.items():
            w.writerow([img, sc, *vals])

    # Codec config module — uses feat_0..feat_N (same set as the Rust
    # port) and the sub<S>_e<E>_p<0|1>_s<0|1> grammar. Categorical-only
    # cells; no scalar axes (SCALAR_AXES empty), matching the Rust port.
    cfg_path = args.out_dir / "codec_config.py"
    feat_list_repr = ",\n    ".join(repr(fn) for fn in feat_names)
    cfg_path.write_text(
        f'''"""Auto-generated zentrain codec config for the Rust-vs-zentrain
picker comparison. Uses feat_0..feat_N (the SAME feature set the Rust
zenpicker-train MLP consumes) and a categorical-only cell grammar
(sub<S>_e<E>_p<0|1>_s<0|1>) so the only difference vs the Rust port is
the trainer implementation, not the feature set / cells / zq grid.
"""
from __future__ import annotations
import re
from pathlib import Path

PARETO = Path("pareto.tsv")
FEATURES = Path("features.tsv")
OUT_JSON = Path("zentrain_picker.json")
OUT_LOG = Path("zentrain_picker.log")

KEEP_FEATURES = [
    {feat_list_repr}
]

# Same target grid as the Rust port: step 5 from 0..70, step 2 70..100.
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["sub", "effort", "prog", "sharp"]
SCALAR_AXES = []
SCALAR_SENTINELS = {{}}
SCALAR_DISPLAY_RANGES = {{}}
OUTPUT_SPECS = {{}}
SPARSE_OVERRIDES = []
FEATURE_TRANSFORMS = {{}}
FEATURE_TRANSFORM_PARAMS = {{}}

_CONFIG_RE = re.compile(r"^sub(?P<sub>\\w+)_e(?P<eff>\\d+)_p(?P<prog>[01])_s(?P<sharp>[01])$")


def parse_config_name(name: str) -> dict:
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {{name}}")
    return {{
        "sub": m.group("sub"),
        "effort": int(m.group("eff")),
        "prog": int(m.group("prog")),
        "sharp": int(m.group("sharp")),
    }}
'''
    )

    print(f"wrote {len(rows_pareto)} pareto rows, {len(feat_seen)} images, {len(cell_ids)} cells")
    print(f"  pareto:   {pareto_path}")
    print(f"  features: {feat_path}")
    print(f"  config:   {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
