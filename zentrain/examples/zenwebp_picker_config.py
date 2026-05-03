"""
Codec config for zenwebp's hybrid-heads picker — v0.1 sweep, ZNPR v3
output specs.

Trains against the 2026-04-30 v0.1 Pareto sweep (TSVs in zenwebp/benchmarks).
Schema matches the production runtime in
zenwebp/src/encoder/picker/spec.rs (36-feature pruned set, 6 cells from
method × segments, 3 scalar heads). Schema_hash 0xb2aca28a2d7a34ec.

Used by `zentrain/tools/train_hybrid.py`. After training, an
intermediate post-process step (see zenwebp/dev/inject_v3_specs.py)
appends `output_specs` and `sparse_overrides` to the model JSON before
handing it to `zenanalyze/tools/bake_picker.py`.

Run training from the zenwebp checkout:

    cd ~/work/zen/zenwebp
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/train_hybrid.py \\
            --codec-config zenwebp_picker_config

Cell taxonomy (CATEGORICAL_AXES — form cells):
  - method ∈ {4, 5, 6}    — m0-3 omitted (below production quality floor)
  - segments ∈ {1, 4}     — 2/3 omitted (rarely Pareto-winners)
  → 6 cells

Scalar prediction heads (SCALAR_AXES):
  - sns_strength       (0..100)
  - filter_strength    (0..100)
  - filter_sharpness   (0..7, integer)

So 6 cells × (1 bytes_log + 3 scalars) = 24 output dimensions.

The v0.1 Pareto sweep emits config_name strings of the form:
  `m{method}_seg{segments}_sns{sns}_fs{filter_strength}_sh{filter_sharpness}`
e.g. `m4_seg1_sns0_fs0_sh0`, `m6_seg4_sns100_fs60_sh6`.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------- Paths ----------

# v0.1 sweep (2026-04-30): the known-good baseline used to bake the
# currently-shipped zenwebp_picker_v0.1.bin. 144-config grid (6 cells ×
# 24 scalar combos) over the original 248-image CID22 corpus. ~8M
# Pareto rows; 102 zenanalyze features per (image, size_class).
PARETO = Path("benchmarks/zenwebp_pareto_2026-04-30_combined.tsv")
FEATURES = Path("benchmarks/zenwebp_pareto_features_2026-04-30_combined.tsv")

OUT_JSON = Path("benchmarks/zenwebp_hybrid_v3.json")
OUT_LOG = Path("benchmarks/zenwebp_hybrid_v3.log")


# ---------- Schema ----------

# 36-feature pruned schema. Order MUST match
# `src/encoder/picker/spec.rs::FEAT_COLS` exactly — schema_hash
# (0xb2aca28a2d7a34ec) is computed from this list + the standard
# extra_axes layout, and the runtime checks the hash at load time.
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
    # DROPPED 2026-05-03 (LOO consensus, mean ΔOH +0.226pp, σ 0.263 — m≥0.5σ
    # rule; #52). Removing it improves overhead measurably across 5 seeds.
    # "feat_cr_sharpness",
    "feat_edge_density",
    "feat_noise_floor_y_p50",
    "feat_luma_histogram_entropy",
    # Mid tier (Δ +0.10..+0.20pp)
    # feat_natural_likelihood retired in zenanalyze post-2026-05-02
    # (id 29 reserved retired). The single-source EdgeSlopeStdev now
    # carries comparable signal (AUC 0.799 vs 0.814 for the retired
    # composite); keeping it in FEAT_COLS would crash the runtime
    # `AnalysisFeature::NaturalLikelihood` lookup.
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
    # feat_screen_content_likelihood retired in zenanalyze post-2026-05-02
    # (id 28 reserved retired). UvProb / GrayscaleScore / luma_kurtosis
    # carry the screen-vs-photo signal at AUC 0.83+; keeping it in
    # FEAT_COLS would crash the runtime
    # `AnalysisFeature::ScreenContentLikelihood` lookup.
    "feat_high_freq_energy_ratio",
    "feat_colourfulness",
    "feat_quant_survival_uv",
]


# ---------- Feature-group mutual-exclusion validator ----------
#
# Subset of the full cross-codec feature-group taxonomy, reduced to the
# features actually present in KEEP_FEATURES so `validate_keep_features`
# doesn't trip on absent members.
FEATURE_GROUPS = {
    "aspect": {
        "members": ["feat_aspect_min_over_max"],
        "max_picked": 1,
    },
    "median_block_cost": {
        "members": [
            "feat_noise_floor_y_p50",
            "feat_quant_survival_y_p50",
        ],
        "max_picked": 2,
    },
    "uv_chroma_compressibility": {
        "members": ["feat_quant_survival_uv"],
        "max_picked": 1,
    },
    "edge_coef_survival": {
        "members": ["feat_edge_density", "feat_quant_survival_y"],
        "max_picked": 2,
    },
    "chroma_sharpness": {
        # feat_cr_sharpness dropped 2026-05-03 (LOO cull); only cb side present.
        "members": ["feat_cb_sharpness"],
        "max_picked": 1,
    },
    "upper_tail_block_cost": {
        "members": [
            "feat_aq_map_p75",
            "feat_noise_floor_y_p75",
            "feat_quant_survival_y_p75",
        ],
        "max_picked": 3,
    },
    "flat_block_signal": {
        "members": ["feat_aq_map_mean", "feat_uniformity"],
        "max_picked": 2,
    },
    "resolution_dimension": {
        "members": [
            "feat_pixel_count",
            "feat_min_dim",
            "feat_max_dim",
        ],
        "max_picked": 3,
    },
}


# Zq target grid: production-relevant range. Empirical sweep at the
# 51966 seed showed capping at 86 collapsed student argmin_acc by
# 17 pp (44.2% → 27.4%, mean overhead 2.4% → 3.4%). The high-zq tail
# rows act as regularization for the student even where teacher
# labels are noisy from sweep gaps. Keep them in for now; the
# DATA_STARVED_SIZE / UNCAPPED_ZQ_GRID safety_report warnings stay
# active and the bake step will require --allow-unsafe until the
# v0.2 sweep emits effective_max_zensim (imazen/zenanalyze#51).
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema (v0.1) ----------
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

# Suppress the optional time_log head — the v0.1 runtime
# (src/encoder/picker/spec.rs) lays out the 24 outputs as
# `bytes_log + sns + filter_strength + filter_sharpness` with no
# time slot in between. Picking a column that doesn't exist in the
# Pareto TSV makes train_hybrid skip the time head; without this the
# trainer would interpose a 6-wide `time_log` block at indices 6..12
# and shift every scalar head down by one block.
TIME_COLUMN = ""


# ---------- ZNPR v3 output post-processing ----------
#
# Per-output OutputSpec dicts, by SCALAR_AXES name. Used by
# `dev/inject_v3_specs.py` after training to emit a length-24
# `output_specs` array (1 bytes_log + 3 scalar heads × 6 cells, in
# `output_layout` order).
#
# - bytes_log: log-bytes prediction. Bounds [0, 30] cover up to e^30 ≈
#   10 GB which is well above any plausible WebP encode for the codec's
#   max-dimension envelope. Identity transform.
# - sns_strength / filter_strength: 0..100 integer-valued knob; bounds
#   match the codec API. Identity (we let the runtime cast to u8).
# - filter_sharpness: 0..7 discrete. Round transform snaps the
#   prediction to the nearest integer in [0,7].
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

# Per-output sparse hand-tune overrides (keyed by output index in the
# laid-out 24-vector). Empty stub for v0.1 — wire in production-tuned
# overrides once a held-out test surfaces specific cells that need
# pinning.
SPARSE_OVERRIDES: list = []


# ---------- Bake-time identifiers ----------
#
# Must match `src/encoder/picker/spec.rs::SCHEMA_VERSION_TAG`. The
# baker derives `schema_hash = blake2b(feat_cols || extra_axes ||
# schema_version_tag)`; the runtime checks the loaded hash against
# its compile-time `SCHEMA_HASH` const at load time. Drift in either
# direction surfaces as `PickError::SchemaMismatch`.
SCHEMA_VERSION_TAG = "zenwebp.picker.v0.1"
BAKE_NAME = "zenwebp_picker_v0.2"


# ---------- Per-feature pre-standardize transform ----------
#
# Trainer applies these BEFORE the StandardScaler fit; runtime must
# apply the same transform pre-scaler. Default for absent keys is
# "identity".
#
# - log: log-distributed positive features (pixel_count). standardize
#   over log-space gives the MLP a more linear feature axis.
# - log1p: heavy-tailed positive features that may be ≈0 on flat
#   patches (laplacian variance + percentiles). log1p handles the
#   x→0 tail without -inf; standardize captures the post-transform
#   spread.
# - identity (default): bounded [0,1] features (uniformity,
#   edge_density, patch_fraction), already-bounded ratios, sharpness
#   measurements, and discrete counts.
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


# ---------- Config-name parser ----------

# v0.1 schema:
#   m{method}_seg{seg}_sns{sns}_fs{fs}_sh{sh}
# Examples:
#   m4_seg1_sns0_fs0_sh0
#       → method=4, segments=1, sns=0, fs=0, sh=0
#   m6_seg4_sns100_fs60_sh6
#       → method=6, segments=4, sns=100, fs=60, sh=6
_CONFIG_RE = re.compile(
    r"^m(?P<method>\d+)_seg(?P<seg>\d+)"
    r"_sns(?P<sns>\d+)_fs(?P<fs>\d+)_sh(?P<sh>\d+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a zenwebp v0.1 config name into categorical + scalar
    axes."""
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
