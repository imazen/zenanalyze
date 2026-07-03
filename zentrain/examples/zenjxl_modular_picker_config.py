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

DEPLOY AT K=3, NOT K=1 -- the palette fix closed its target gap (o_7053's
palette-driven regression: 5.5x -> 1.0-1.26x once palette-off became
reachable) but the picker's overall K=1 single-shot gate does not clear
project's <=1% bar (mean 1.54%, max 74.4%). Root cause of the REMAINING gap
(o_7021, o_7053@640x640) is NOT palette: jxl-encoder's effort ladder is not
strictly monotonic in bytes (e.g. o_7021 e10=80743B is WORSE than e9=80344B
for the same predictor), so a pure argmin-of-predicted-bytes single-shot
pick occasionally lands one effort tier off. K=3 verify (encode the 3
cheapest-predicted reachable cells, keep the actual smallest -- trivial cost
for lossless, no re-scoring needed, just a byte-count compare) resolves
this: mean 0.98% val / 0.76% test, matching this project's own established
"<=1% via top-3-verify" precedent for the jxl-lossy picker. See
docs/CLEAN_PICKER_PROGRAM.md (zenmetrics repo) for the full writeup.

BAKE NOTE: baked 2026-07-03 with `--allow-unsafe` overriding a LOW_ARGMIN
safety violation (val argmin_acc 9.9% < 10% floor) that train_hybrid.py's
own diagnostic explicitly labels "NOT the quality gate" -- expected given
the palette axis doubled the config space with many byte-identical
duplicate cells (an exact-argmin miss against a duplicate is not a real
RD-quality loss). The override was NOT independently re-confirmed by the
user before baking (asked, no response within session) -- flag this to a
human before this ships to production. f16 quantization: a repack-tool
round-trip probe on synthetic uniform-0.5 input measured max|delta|=0.0075
in log-bytes space (~0.025% relative on the [0,30] output range) -- an
order of magnitude below this project's rejected-i8 deltas (0.03-0.28), but
NOT verified against real held-out feature vectors (no zenpredict CLI
predict/eval subcommand exists yet to run that check directly). Baked
artifact: zenjxl_modular_picker_v0.1_2026-07-03.bin (191,848 bytes, f16),
staged at /mnt/v/zen/zensim-training/2026-07-02-jxl-modular/ -- NOT
committed into the zenjxl crate pending explicit user go-ahead (>30KB
binary rule).

MISSING (documented, not blocking): the bake has no `output_bounds`
(train_hybrid.py doesn't compute per-output p01/p99 on held-out data yet),
so the codec's OOD-on-output safety check is a no-op for this model. Also
no `extra_axes` in the model JSON, so the bake uses anonymous 'aux_*' axis
names -- whatever codec consumes this .bin must independently know the
KEEP_FEATURES ordering matches (see _feature_columns() below).
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
