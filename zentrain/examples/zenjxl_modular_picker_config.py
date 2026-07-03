"""zenjxl-modular (lossless) picker config — 2026-07-03, clean-picker-corpus-2026-06-26.

Cell format: 120 configs = 10 effort levels (e1..e10) x 3 base predictor modes
(def / wp5 / rct1) x pred6 on/off x palette on/off. The palette axis was added
2026-07-03 after discovering the prior 60-config sweep never tried disabling
palette detection -- measured 5.5x size regression on low-color-count content
(fine grids, line art, screenshots) where the always-on default is actively
harmful (zenjxl commits 7b4e10d8 + 535aca1a, jxl-encoder#69 items 2/3).
JXL modular mode is lossless, so every cell produces IDENTICAL pixels
(verified: score_zensim and score_ssim2 are both CONSTANT 100.0 across all
539,640 rows) -- "RD-optimal" reduces to min-bytes, same shape as the zenpng
picker.

Source: s3://zentrain/canonical/2026-07-03/zenjxl_lossless/{train,validate,test}.parquet
(supersedes the 2026-06-27/269,820-row canonical -- the pre-existing 60
configs' rows are byte-identical, verified 0 mismatches across all 269,820;
union'd locally; train_hybrid re-derives the origin-parity split itself).
Build script: build_pareto_features_2026-07-03.py in the 2026-07-02/03 scratch
session (regenerates the PARETO/FEATURES shape from the new canonical).
"""
from pathlib import Path

PARETO = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_pareto_2026-07-03.parquet")
FEATURES = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_features_2026-07-03.parquet")
OUT_JSON = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_modular_picker_2026-07-03.json")
OUT_LOG = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_modular_picker_2026-07-03.log")

METRIC_COLUMN = "score_ssim2"
METRIC_DIRECTION = "higher_better"


_ZENANALYZE_NAMES_TSV = Path(__file__).parent / "_data" / "zenanalyze_feature_names.tsv"


def _canon(c: str) -> str:
    if c.startswith("feat_"):
        c = c[len("feat_"):]
    if "@" in c:
        c = c[: c.rfind("@")]
    return c.lower()


def _feature_columns():
    """Genuine zenanalyze named features present in FEATURES, EXCLUDING the
    foreign zensim-internal feat_0..feat_371 basis that shares the same
    `feat_` prefix in the joined training parquet (372/469 columns are that
    unrelated system -- see `dump_feature_names` in zenanalyze/examples/,
    generated via `cargo run --release --example dump_feature_names`).
    """
    import pyarrow.parquet as pq

    with open(_ZENANALYZE_NAMES_TSV) as f:
        canonical = {line.rstrip("\n").split("\t")[1].lower() for line in f}
    schema = pq.read_schema(FEATURES)
    feat = [c for c in schema.names if c.startswith("feat_")]
    kept = [c for c in feat if _canon(c) in canonical]
    dropped = len(feat) - len(kept)
    if dropped:
        import sys
        sys.stderr.write(
            f"_feature_columns: kept {len(kept)}/{len(feat)} genuine zenanalyze "
            f"features, dropped {dropped} foreign (non-zenanalyze) feat_* columns\n"
        )
    return kept


KEEP_FEATURES = _feature_columns()
# Lossless: achieved quality is constant (100) regardless of target, but a real
# range (matching zenpng_picker_cpu2026.py) is still required so the picker
# answers correctly for the cross-codec meta-router's zq queries, and so the
# zq_norm*feature cross-term in train_hybrid's input vector isn't a degenerate
# exact duplicate of the raw features (zq_norm=1.0 for every row otherwise).
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["effort", "base", "pred6", "palette_off"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}
OUTPUT_SPECS = {"bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"}}
SPARSE_OVERRIDES: list = []

import re

_CONFIG_RE = re.compile(r"^mod-e(?P<effort>\d+)_(?P<base>def|wp5|rct1)(?P<pred6>-pred6)?(?P<pal0>-pal0)?$")


def parse_config_name(name: str) -> dict:
    """Decompose a zenjxl-modular cell name ('mod-e9_wp5-pred6-pal0') into axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenjxl-modular config name: {name}")
    return {
        "effort": int(m.group("effort")),
        "base": m.group("base"),
        "pred6": 1 if m.group("pred6") else 0,
        "palette_off": 1 if m.group("pal0") else 0,
    }
