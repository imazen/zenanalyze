"""zenavif picker config — 2026-06-24 CPU corpus (ssim2 target, name@hex8 features).

Cell format: `s<speed>[-noqm][-420][-bd10]` (24 cells = 3 speed x 2 qm x 2 chroma x 2 bitdepth).
Defaults encoded by ABSENCE: qm on (no `-noqm`), 4:4:4 chroma (no `-420`), 8-bit (no `-bd10`).
Pure categorical (no scalar axes). Trained against ssim2 from the 2026-06-24 Hetzner CPU rd_core
corpus (244k cells, 168/image), with the re-extracted name@hex8 features.

NOTE (RD_ABLATION_2026-06-24): on the rendition corpus + ssim2, `s2-noqm` (qm OFF) is the TOP front
cell — qm is NOT dead here, unlike the gif-static/zensim v12 finding. The picker learns this directly.
"""
import re
from pathlib import Path

PARETO = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/pareto/pareto_zenavif_2026-06-24.parquet")
FEATURES = Path("/mnt/v/output/imazen-26-features/sdr_features_2026-06-24.tsv")
OUT_JSON = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenavif_picker_cpu2026.json")
OUT_LOG = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenavif_picker_cpu2026.log")

METRIC_COLUMN = "ssim2"
METRIC_DIRECTION = "higher_better"


def _feature_columns():
    """All feature columns (name@hex8) present in the features TSV — id/meta cols excluded."""
    with open(FEATURES) as f:
        header = f.readline().rstrip("\n").split("\t")
    meta = {"image_path", "image_sha", "split", "content_class", "source",
            "size_class", "width", "height"}
    return [c for c in header if c not in meta]


KEEP_FEATURES = _feature_columns()

# Dense at the high-q end per the sweep discipline; the trainer interpolates the rd-optimal cell per target.
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["speed", "qm", "chroma", "bd"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}

OUTPUT_SPECS = {
    "bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"},
}
SPARSE_OVERRIDES: list = []

_CONFIG_RE = re.compile(r"^s(?P<speed>2|4|6)(?P<noqm>-noqm)?(?P<chroma>-420)?(?P<bd>-bd10)?$")


def parse_config_name(name: str) -> dict:
    """Parse `s<speed>[-noqm][-420][-bd10]` into its 4 categorical axes (defaults by absence)."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {name}")
    return {
        "speed": "s" + m.group("speed"),
        "qm": "noqm" if m.group("noqm") else "qm",
        "chroma": "420" if m.group("chroma") else "444",
        "bd": "bd10" if m.group("bd") else "bd8",
    }
