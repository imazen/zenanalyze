"""
Codec config for the zenjxl lossy hybrid-heads picker — adapted from
the existing pareto sweep at
``/mnt/v/output/jxl-encoder/picker-oracle-2026-04-30/`` via
``zentrain/tools/zenjxl_oracle_adapter.py`` (no re-encoding).

Used by `zentrain/tools/feature_ablation.py`,
`zentrain/tools/correlation_cleanup.py`, and
`zentrain/tools/train_hybrid.py`.

CRITICAL — metric substitution caveat
-------------------------------------
The `zensim` column in the adapted Pareto TSV is actually
**ssim2 (SSIMULACRA2)** values, not zenpipe's XYB-Butteraugli
`zensim`. The two metrics are correlated but numerically distinct
and not directly comparable. zenanalyze trainers consume the column
under the name `zensim` because that's the schema all the other
codec configs share, so swapping it in is the cheapest path to
running Tier 0/1.5 on the JXL data — but downstream cross-codec
aggregation **MUST NOT** mix this run's overhead numbers / picker
predictions with results from sweeps that used real `zensim`. See
``zenjxl_oracle_adapter.py`` for the substitution mechanics and the
sidecar ``.meta`` file for the in-tree provenance.

Run from the zenjxl checkout (or anywhere — the paths are absolute
in the codec config module):

    cd ~/work/zen/zenjxl
    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/correlation_cleanup.py \\
            --codec-config zenjxl_picker_config

    PYTHONPATH=~/work/zen/zenanalyze/zentrain/examples:~/work/zen/zenanalyze/zentrain/tools \\
        python3 ~/work/zen/zenanalyze/zentrain/tools/feature_ablation.py \\
            --codec-config zenjxl_picker_config --method permutation

Cell taxonomy (CATEGORICAL_AXES — form cells):
  - cell_id ∈ {0..N}  (encoder cell layout, see jxl-encoder)
  - ac_intensity ∈ {compact, full}
  - enhanced_clustering ∈ {0, 1}
  - gaborish ∈ {0, 1}
  - patches ∈ {0, 1}

Scalar prediction heads (SCALAR_AXES):
  - k_info_loss_mul (continuous, ~0.5..2.0)
  - k_ac_quant      (continuous, ~0.5..1.5)
  - entropy_mul_dct8 (continuous, ~0.5..1.5)

Distance is the quality axis (analogous to q in
zenavif/zenwebp/zenjpeg sweeps); the picker iterates ZQ_TARGETS
across achieved zensim values, NOT distance values, so distance is
NOT in the config_name. The same encoder-knob combination is
sampled across the full distance grid in the source sweep.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ---------- Paths ----------

# Adapted oracle (synthesized, no re-encoding) — see
# zentrain/tools/zenjxl_oracle_adapter.py.
_ZENJXL = Path(os.path.expanduser("~/work/zen/zenjxl"))
PARETO = _ZENJXL / "benchmarks" / "zenjxl_lossy_pareto_2026-05-01.tsv"
FEATURES = _ZENJXL / "benchmarks" / "zenjxl_lossy_features_2026-05-01.tsv"

# Tier 0/1.5 outputs land under zenanalyze/benchmarks/ for the
# parent aggregator's eyes. Use the zenjxl_ prefix so the
# zenavif/zenwebp/zenjpeg results don't get clobbered.
_ZENANALYZE = Path(os.path.expanduser("~/work/zen/zenanalyze"))
OUT_JSON = _ZENANALYZE / "benchmarks" / "zenjxl_hybrid_2026-05-01.json"
OUT_LOG = _ZENANALYZE / "benchmarks" / "zenjxl_hybrid_2026-05-01.log"


# ---------- Schema ----------

# Same broad starting set as zenavif. The four-tier pipeline drops
# what doesn't earn its keep on the JXL oracle. Don't pre-cull —
# zenjxl might rank features differently.
KEEP_FEATURES = [
    # Tier 1 (sparse stripe)
    "feat_variance",
    "feat_edge_density",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_uniformity",
    "feat_flat_color_block_ratio",
    "feat_colourfulness",
    "feat_laplacian_variance",
    "feat_variance_spread",
    "feat_distinct_color_bins",
    "feat_palette_density",
    "feat_cb_horiz_sharpness",
    "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness",
    "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness",
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
    # Tier 3 (sampled DCT blocks)
    "feat_dct_compressibility_y",
    "feat_dct_compressibility_uv",
    "feat_patch_fraction",
    "feat_patch_fraction_fast",
    "feat_quant_survival_y",
    "feat_quant_survival_uv",
    "feat_aq_map_mean",
    "feat_aq_map_std",
    "feat_aq_map_p50",
    "feat_aq_map_p75",
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    "feat_aq_map_p99",
    "feat_noise_floor_y",
    "feat_noise_floor_uv",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p75",
    "feat_noise_floor_y_p90",
    "feat_noise_floor_uv_p25",
    "feat_noise_floor_uv_p50",
    "feat_noise_floor_uv_p75",
    "feat_noise_floor_uv_p90",
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance_p90",
    "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak",
    "feat_quant_survival_y_p10",
    "feat_quant_survival_y_p25",
    "feat_quant_survival_y_p50",
    "feat_quant_survival_y_p75",
    "feat_quant_survival_uv_p10",
    "feat_gradient_fraction",
    "feat_grayscale_score",
    "feat_edge_slope_stdev",
    # New experimental shape / smoothness features (post-cull,
    # zenanalyze 0.1.0 — 2 features kept 2026-05-01 after Tier-0
    # ablation removed chroma_kurtosis / uniformity_smooth /
    # flat_color_smooth as redundant).
    "feat_luma_kurtosis",
    "feat_gradient_fraction_smooth",
    # Dimension / shape
    "feat_pixel_count",
    "feat_min_dim",
    "feat_max_dim",
    "feat_aspect_min_over_max",
    "feat_log_aspect_abs",
    "feat_channel_count",
    # Alpha (corpus-conditional — JXL corpus likely has none, but
    # leave in so the ablation observes the constant column rather
    # than silently missing the signal).
    "feat_alpha_present",
    "feat_alpha_used_fraction",
    "feat_alpha_bimodal_score",
]

# Zq target grid: same shape as zenavif. ssim2 in this column
# rather than zensim (see metric-substitution caveat above), so
# "30..95" here is ssim2 30..95 and trains the picker to interpolate
# over that distance-mapped axis.
ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))


# ---------- Axis schema ----------

CATEGORICAL_AXES = ["cell_id", "ac_intensity", "enhanced_clustering", "gaborish", "patches"]
SCALAR_AXES: list[str] = ["k_info_loss_mul", "k_ac_quant", "entropy_mul_dct8"]
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {
    "k_info_loss_mul": (0.5, 2.0),
    "k_ac_quant": (0.5, 1.5),
    "entropy_mul_dct8": (0.5, 1.5),
}


# ---------- Config-name parser ----------

# Format: c{cell_id}_ac{ac_intensity}_ec{enhanced_clustering}_g{gaborish}
#         _p{patches}_kil{k_info_loss_mul}_kaq{k_ac_quant}_ed8{entropy_mul_dct8}
# The float scalars are emitted with 4 decimal places by the adapter.
_CONFIG_RE = re.compile(
    r"^c(?P<cell_id>\d+)"
    r"_ac(?P<ac_intensity>compact|full)"
    r"_ec(?P<enhanced_clustering>\d+)"
    r"_g(?P<gaborish>\d+)"
    r"_p(?P<patches>\d+)"
    r"_kil(?P<k_info_loss_mul>[\d.]+)"
    r"_kaq(?P<k_ac_quant>[\d.]+)"
    r"_ed8(?P<entropy_mul_dct8>[\d.]+)$"
)


def parse_config_name(name: str) -> dict:
    """Decompose a zenjxl lossy config name into categorical + scalar axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenjxl config name: {name}")
    return {
        # categorical
        "cell_id": int(m.group("cell_id")),
        "ac_intensity": m.group("ac_intensity"),
        "enhanced_clustering": int(m.group("enhanced_clustering")),
        "gaborish": int(m.group("gaborish")),
        "patches": int(m.group("patches")),
        # scalar
        "k_info_loss_mul": float(m.group("k_info_loss_mul")),
        "k_ac_quant": float(m.group("k_ac_quant")),
        "entropy_mul_dct8": float(m.group("entropy_mul_dct8")),
    }
