"""zenwebp picker config — 2026-06-24 CPU corpus (ssim2 target, name@hex8 features).

Cell format: 3 modes — `vp8-m4_def`, `vp8-m6_def` (lossy VP8 method 4/6), `vp8l-m4` (lossless VP8L).
Not a clean axis product (no `vp8l-m6`), so a single 3-way categorical `mode`. Trained against ssim2
from the 2026-06-24 Hetzner CPU rd_core corpus (~15 cells/image — sparse; densify later if overhead high).
"""
from pathlib import Path

PARETO = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/pareto/pareto_zenwebp_2026-06-24.parquet")
FEATURES = Path("/mnt/v/output/imazen-26-features/sdr_features_2026-06-24.tsv")
OUT_JSON = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenwebp_picker_cpu2026.json")
OUT_LOG = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenwebp_picker_cpu2026.log")

METRIC_COLUMN = "ssim2"
METRIC_DIRECTION = "higher_better"


def _feature_columns():
    with open(FEATURES) as f:
        header = f.readline().rstrip("\n").split("\t")
    meta = {"image_path", "image_sha", "split", "content_class", "source",
            "size_class", "width", "height"}
    return [c for c in header if c not in meta]


KEEP_FEATURES = _feature_columns()
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["mode"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}
OUTPUT_SPECS = {"bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"}}
SPARSE_OVERRIDES: list = []

_MODES = {"vp8-m4_def", "vp8-m6_def", "vp8l-m4"}


def parse_config_name(name: str) -> dict:
    if name not in _MODES:
        raise ValueError(f"unparseable config name: {name}")
    return {"mode": name}
