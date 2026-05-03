"""
zenwebp v0.3 picker config — pinned to the v0.1 runtime FEAT_COLS schema.

Mirrors `zenwebp/src/encoder/picker/spec.rs::FEAT_COLS` exactly (36
features) so the resulting .bin's schema_hash matches the runtime's
compile-time SCHEMA_HASH (0xb2aca28a2d7a34ec) and drops in to the
existing `pick_tuning` runtime without any source changes to zenwebp.

The companion `zenwebp_picker_config.py` is the *training-experiment*
config — KEEP_FEATURES has 33 entries (3 dropped via the 2026-05-03
multi-seed LOO cull) and produces a .bin whose schema_hash differs
from spec.rs and would therefore be rejected at runtime load. Use
this v0.1-schema config to bake the drop-in v0.3 .bin; use the
experiment config to ship the next-major picker once the codec
runtime's spec.rs is updated.

Trains against the same 2026-04-30 v0.1 sweep (`zenwebp_pareto_2026-04-30_combined.tsv`)
and the same v0.1 features TSV (`zenwebp_pareto_features_2026-04-30_combined.tsv`).
Cell taxonomy + scalar heads are unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------- Paths ----------

PARETO = Path("benchmarks/zenwebp_pareto_2026-04-30_combined.tsv")
FEATURES = Path("benchmarks/zenwebp_pareto_features_2026-04-30_combined.tsv")

OUT_JSON = Path("benchmarks/zenwebp_hybrid_v0.3_v01schema.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_v0.3_v01schema.log")


# ---------- Schema ----------

# Exact mirror of zenwebp/src/encoder/picker/spec.rs::FEAT_COLS
# (36 entries). Order is load-bearing; reordering invalidates the
# baked schema_hash and the runtime will reject the .bin.
KEEP_FEATURES = [
    # Top tier (Δ ≥ +0.20pp from the 2026-04-30 ablation)
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance",
    "feat_quant_survival_y",
    "feat_cb_sharpness",
    "feat_pixel_count",
    "feat_uniformity",
    "feat_distinct_color_bins",
    "feat_cr_sharpness",
    "feat_edge_density",
    "feat_noise_floor_y_p50",
    "feat_luma_histogram_entropy",
    # Mid tier (Δ +0.10..+0.20pp)
    "feat_natural_likelihood",
    "feat_quant_survival_y_p50",
    "feat_noise_floor_uv_p50",
    "feat_aq_map_mean",
    "feat_cr_horiz_sharpness",
    "feat_min_dim",
    "feat_edge_slope_stdev",
    "feat_laplacian_variance_p90",
    "feat_patch_fraction",
    "feat_max_dim",
    "feat_aspect_min_over_max",
    "feat_aq_map_p75",
    # Low tier (Δ +0.05..+0.10pp)
    "feat_cb_horiz_sharpness",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_uv",
    "feat_chroma_complexity",
    "feat_quant_survival_y_p75",
    "feat_aq_map_std",
    "feat_gradient_fraction",
    "feat_noise_floor_y_p75",
    "feat_screen_content_likelihood",
    "feat_high_freq_energy_ratio",
    "feat_colourfulness",
    "feat_quant_survival_uv",
]


# Skip the feature-group mutex validator — spec.rs's pin is the
# authority here, not cross-codec dendrogram constraints.
FEATURE_GROUPS = {}


ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema (matches spec.rs CELLS + RANGE_*) ----------
CATEGORICAL_AXES = ["method", "segments"]
SCALAR_AXES = [
    "sns_strength",
    "filter_strength",
    "filter_sharpness",
]
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES = {
    "sns_strength": (0, 100),
    "filter_strength": (0, 100),
    "filter_sharpness": (0, 7),
}

TIME_COLUMN = ""


# ---------- ZNPR v3 output post-processing (same as v0.1 config) ----------
OUTPUT_SPECS = {
    "bytes_log": {
        "bounds": [0.0, 30.0],
        "transform": "identity",
    },
    "sns_strength": {
        "bounds": [0.0, 100.0],
        "transform": "identity",
    },
    "filter_strength": {
        "bounds": [0.0, 100.0],
        "transform": "identity",
    },
    "filter_sharpness": {
        "bounds": [0.0, 7.0],
        "transform": "round",
        "discrete_set": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    },
}

SPARSE_OVERRIDES: list = []


# ---------- Bake-time identifiers ----------
# Must match spec.rs::SCHEMA_VERSION_TAG.
SCHEMA_VERSION_TAG = "zenwebp.picker.v0.1"
BAKE_NAME = "zenwebp_picker_v0.3"


# ---------- Per-feature pre-standardize transform (same as v0.1 config) ----------
FEATURE_TRANSFORMS = {
    "feat_pixel_count": "log",
    "feat_min_dim": "log",
    "feat_max_dim": "log",
    "feat_laplacian_variance": "log1p",
    "feat_laplacian_variance_p50": "log1p",
    "feat_laplacian_variance_p75": "log1p",
    "feat_laplacian_variance_p90": "log1p",
    "feat_edge_slope_stdev": "log1p",
}


# ---------- Config-name parser (unchanged) ----------
_CONFIG_RE = re.compile(
    r"^m(?P<method>\d+)_seg(?P<seg>\d+)"
    r"_sns(?P<sns>\d+)_fs(?P<fs>\d+)_sh(?P<sh>\d+)$"
)


def parse_config_name(name: str) -> dict:
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenwebp config name: {name}")
    return {
        "method": int(m.group("method")),
        "segments": int(m.group("seg")),
        "sns_strength": float(m.group("sns")),
        "filter_strength": float(m.group("fs")),
        "filter_sharpness": float(m.group("sh")),
    }
