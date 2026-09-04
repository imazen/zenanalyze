"""zenavif AVIF autotune picker — CORE cell set (the ship candidate).

Same view, same features, same split as `zenavif_autotune_2026_09_04`; the only
difference is the CELL SET. A cell is admitted iff it was measured on BOTH
corpora — the 1024^2 budget crops AND the native-resolution images — so every
cell the picker can choose has cross-size evidence behind it.

WHY: 71 of the full set's 143 cells are the Stage-A A2 pairwise arms, which
exist only at 1024^2. The DOE's own transfer gate rates most knobs PARTIAL
("direction holds, magnitude flagged") and finds two whose 1024^2 effect is
known-wrong at native (`tl1.0` flips sign; `tl1.1` shrinks 8.6x). A picker asked
about a 12 MP image would have to extrapolate those cells from crop-only
evidence. This variant declines to.

The cell list is DERIVED FROM THE VIEW, not hand-written, so it cannot drift
from the data.
"""
from zenavif_autotune_2026_09_04 import (  # noqa: F401
    ANALYSIS_PROVENANCE, CATEGORICAL_AXES, FEATURE_TRANSFORMS, FEATURES,
    KEEP_FEATURES, METRIC_COLUMN, METRIC_DIRECTION, OUTPUT_SPECS,
    SCALAR_AXES, SCALAR_DISPLAY_RANGES, SCALAR_SENTINELS, SPARSE_OVERRIDES,
    SPLIT_RULE, TIME_COLUMN, VIEW, ZQ_TARGETS, parse_config_name,
)

OUT_JSON = VIEW / "models" / "zenavif_autotune_v1_core.json"
OUT_LOG = VIEW / "models" / "zenavif_autotune_v1_core.log"


# The core Pareto is emitted by the view builder itself (the rule is derived
# from the data there and recorded in `_MANIFEST.json` under `core_view`), so
# this module only points at it — the cell list is never hand-written here.
PARETO = VIEW / "pareto_avif_autotune_core.parquet"
