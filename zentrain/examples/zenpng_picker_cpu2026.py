"""zenpng picker config — 2026-06-24 CPU corpus (ssim2 target, name@hex8 features).

Cell format: 3 effort presets — `png-fast`, `png-balanced`, `png-intense`. PNG is lossless, so all
presets produce IDENTICAL pixels; "rd-optimal" reduces to min-bytes (the quality term is constant).
The picker therefore learns which effort preset yields the smallest file per image — a near-trivial
target, but trained for completeness. Single 3-way categorical `effort`.
"""
from pathlib import Path

PARETO = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/pareto/pareto_zenpng_2026-06-24.parquet")
FEATURES = Path("/mnt/v/output/imazen-26-features/sdr_features_2026-06-24.tsv")
OUT_JSON = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenpng_picker_cpu2026.json")
OUT_LOG = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenpng_picker_cpu2026.log")

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

CATEGORICAL_AXES = ["effort"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}
OUTPUT_SPECS = {"bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"}}
SPARSE_OVERRIDES: list = []

_EFFORTS = {"png-fast", "png-balanced", "png-intense"}


def parse_config_name(name: str) -> dict:
    if name not in _EFFORTS:
        raise ValueError(f"unparseable config name: {name}")
    return {"effort": name}
