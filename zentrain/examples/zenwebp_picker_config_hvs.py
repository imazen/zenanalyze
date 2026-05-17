"""zenwebp picker config — HVS-features eval (2026-05-17).

Wraps the production config; overrides FEATURES path to point at the
HVS-augmented features TSV and extends KEEP_FEATURES with the 5 new
HVS features (ids 132-136 in zenanalyze, gated behind
`--features experimental`).

Run the sweep against this config to measure end-to-end lift of the
HVS features vs the production baseline.
"""

from __future__ import annotations

from pathlib import Path

# Inherit everything from the production config.
from zenwebp_picker_config import (  # noqa: F401
    PARETO,
    ZQ_TARGETS, CATEGORICAL_AXES, SCALAR_AXES, SCALAR_SENTINELS,
    SCALAR_DISPLAY_RANGES, FEATURE_GROUPS, FEATURE_TRANSFORMS,
    TIME_COLUMN,
    OUTPUT_SPECS, SPARSE_OVERRIDES,
    parse_config_name,
)
from zenwebp_picker_config import KEEP_FEATURES as BASE_KEEP_FEATURES

# Point at the HVS-augmented features TSV.
FEATURES = Path("benchmarks/zenwebp_pareto_features_2026-05-17_hvs.tsv")

# Override OUT paths so HVS results don't clobber baseline artifacts.
OUT_JSON = Path("benchmarks/zenwebp_hybrid_hvs_2026-05-17.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_hvs_2026-05-17.log")

# Extend KEEP_FEATURES with the 5 new HVS features.
KEEP_FEATURES = list(BASE_KEEP_FEATURES) + [
    "feat_chroma_luma_covariance_cb",
    "feat_chroma_luma_covariance_cr",
    "feat_info_weight_mean",
    "feat_info_weight_p90",
    "feat_orientation_energy_ratio",
]
