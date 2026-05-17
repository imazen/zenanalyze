"""zenavif picker config — HVS-features eval (2026-05-17)."""

from __future__ import annotations

from pathlib import Path

from zenavif_picker_config import (  # noqa: F401
    PARETO,
    ZQ_TARGETS, CATEGORICAL_AXES,
    FEATURE_TRANSFORMS,
    OUTPUT_SPECS, SPARSE_OVERRIDES,
    parse_config_name,
)
from zenavif_picker_config import KEEP_FEATURES as BASE_KEEP_FEATURES

FEATURES = Path("benchmarks/zenavif_features_2026-05-17_hvs.tsv")
OUT_JSON = Path("benchmarks/zenavif_hybrid_hvs_2026-05-17.json")
OUT_LOG = Path("benchmarks/zenavif_hybrid_hvs_2026-05-17.log")

KEEP_FEATURES = list(BASE_KEEP_FEATURES) + [
    "feat_chroma_luma_covariance_cb",
    "feat_chroma_luma_covariance_cr",
    "feat_info_weight_mean",
    "feat_info_weight_p90",
    "feat_orientation_energy_ratio",
]
