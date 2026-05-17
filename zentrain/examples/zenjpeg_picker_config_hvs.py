"""zenjpeg picker config — HVS-features eval (2026-05-17).

Wraps the production zenjpeg config; overrides FEATURES path to point
at the HVS-augmented features TSV and extends KEEP_FEATURES with the
5 new HVS features.
"""

from __future__ import annotations

from pathlib import Path

from zenjpeg_picker_config import (  # noqa: F401
    PARETO,
    ZQ_TARGETS, LAMBDA_NOTRELLIS_SENTINEL,
    CATEGORICAL_AXES, SCALAR_AXES, SCALAR_SENTINELS, SCALAR_DISPLAY_RANGES,
    FEATURE_TRANSFORMS,
    OUTPUT_SPECS, SPARSE_OVERRIDES,
    parse_config_name,
)
from zenjpeg_picker_config import KEEP_FEATURES as BASE_KEEP_FEATURES

FEATURES = Path("benchmarks/zq_pareto_features_2026-05-17_hvs.tsv")
OUT_JSON = Path("benchmarks/zenjpeg_hybrid_hvs_2026-05-17.json")
OUT_LOG = Path("benchmarks/zenjpeg_hybrid_hvs_2026-05-17.log")

KEEP_FEATURES = list(BASE_KEEP_FEATURES) + [
    "feat_chroma_luma_covariance_cb",
    "feat_chroma_luma_covariance_cr",
    "feat_info_weight_mean",
    "feat_info_weight_p90",
    "feat_orientation_energy_ratio",
]
