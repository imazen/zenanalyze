"""Categorical-cell picker config over a CANONICAL 2026-06-27 per-codec dataset
(`s3://zentrain/canonical/2026-06-27/<codec>_lossy/`), in the shape
`zentrain/tools/canonical_to_pareto.py` derives from it. One module for every
codec — the codec is an environment variable, so the #68 trainer comparison
runs on identical machinery per codec:

    ZENCANONICAL_CODEC=zenwebp  ZENCANONICAL_DIR=~/tmp/canonical/zenwebp_lossy
    ZENCANONICAL_CODEC=zenjpeg  ZENCANONICAL_DIR=~/tmp/canonical/zenjpeg_lossy

Cell format: the canonical `cell` label (zenwebp: 30 VP8 modes
`vp8-m{2,4,6}_{def,parity,plim50,smooth,mpass}[-syuv]`; zenjpeg: 54
strategy × trellis × subsampling cells `{gls,jp3,moz,pw4}_{t0,tr14.5,…}_small_{420,422,444,xybBq}`;
zenjxl: 35 `vd-e{1..9}_…`; zenavif: 48 `s{2..8}[-noqm][-420][-bd10][-rgb]`), one
row per (image, q, cell). The knob tuple carries only `cell` + a fingerprint, so
there are NO scalar heads: a single categorical `mode` axis, one `bytes_log`
output per cell (+ the `time_log` block — the Pareto carries `encode_ms`). This
is NOT any codec's shipped runtime schema; it is the apples-to-apples bench for
`benchmarks/train_hybrid_backend_gap_*_2026-08-28.md`.

`ZENCANONICAL_DIR` (default `~/tmp/canonical/<codec>_lossy`) must hold
`derived/pareto.parquet` + `derived/features.tsv` from:

    python3 zentrain/tools/canonical_to_pareto.py \\
        --canonical $DIR/train.parquet $DIR/validate.parquet [$DIR/test.parquet] \\
        --pareto-out $DIR/derived/pareto.parquet --features-out $DIR/derived/features.tsv

(pass train AND validate at least — train_hybrid re-derives the origin even/odd
split itself; models land in `$DIR/models/`).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

CODEC = os.environ.get("ZENCANONICAL_CODEC", "zenwebp")
_DIR = Path(os.environ.get("ZENCANONICAL_DIR", f"~/tmp/canonical/{CODEC}_lossy")).expanduser()

PARETO = _DIR / "derived" / "pareto.parquet"
FEATURES = _DIR / "derived" / "features.tsv"
OUT_JSON = _DIR / "models" / f"{CODEC}_canonical_hybrid.json"
OUT_LOG = _DIR / "models" / f"{CODEC}_canonical_hybrid.log"

METRIC_COLUMN = "zensim"
METRIC_DIRECTION = "higher_better"


def _feature_columns() -> list[str]:
    with open(FEATURES) as f:
        header = f.readline().rstrip("\n").split("\t")
    meta = {"image_path", "size_class", "width", "height"}
    return [c for c in header if c not in meta]


KEEP_FEATURES = _feature_columns() if FEATURES.exists() else []
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["mode"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}
OUTPUT_SPECS = {"bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"}}
SPARSE_OVERRIDES: list = []

# Cell labels are `[A-Za-z0-9]` runs joined by `_` / `-` / `.` / `+` / `,` with
# optional bracketed parameter lists (zenjpeg's `jp3[0.5,0.5]_tr14.75+dc_small_444`);
# whitespace or anything else is a parse error, not a silent cell.
_CELL = re.compile(r"^[a-z0-9][a-z0-9._+,\-\[\]]*$", re.IGNORECASE)


def parse_config_name(name: str) -> dict:
    if not _CELL.match(name):
        raise ValueError(f"unparseable {CODEC} canonical cell name: {name}")
    return {"mode": name}
