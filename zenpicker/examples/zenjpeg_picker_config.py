"""
Codec config for zenjpeg's hybrid-heads picker.

Used by `zenpicker/tools/train_hybrid.py` (and the other tools/
training scripts). The codec defines:

  - paths to the Pareto sweep + features TSVs and the desired
    output JSON / log
  - the `feat_*` column subset (KEEP_FEATURES) the picker uses
  - the target_zq grid (ZQ_TARGETS) the picker is trained against
  - `parse_config_name(name)` — the codec's regex / parser that
    decomposes a config_name string into categorical + scalar axes

Run training with:

    cd <zenjpeg checkout>
    PYTHONPATH=<zenanalyze>/zenpicker/examples \\
      python3 <zenanalyze>/zenpicker/tools/train_hybrid.py \\
        --codec-config zenjpeg_picker_config

A new codec (zenwebp / zenavif / zenjxl) writes its own copy of
this file: change paths, change feature subset, change parser
pattern, and import the same `train_hybrid.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------- Paths ----------

# zenjpeg's pareto sweep harness produces these. The pareto sweep
# itself (encode + zensim per (image, config, q)) is the expensive
# step and stays at 2026-04-29; the v2.1 retrain reuses it but pulls
# the wider analyzer feature set from the features-only
# `..._v2_1.tsv` re-extraction.
PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
# v2.2 features: same pareto, but re-extracted with the post-#42
# zenanalyze (12 dimension features + 26 percentile features added on
# top of v2.1's 35 raw features). Total 73-feature TSV.
# `_subset100` is the fast-iteration subset (100 of 347 images). For
# the final v2.2 bake, point at `_v2_2.tsv` (full corpus).
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-30_v2_2_subset100.tsv")

# Where to write the trained model + summary:
OUT_JSON = Path("benchmarks/zq_bytes_hybrid_v2_2.json")
OUT_LOG = Path("benchmarks/zq_bytes_hybrid_v2_2.log")


# ---------- Schema ----------

# v2.1 expanded schema. Original 8-feature reduced set (PR #129
# ablation winner) plus all stable + experimental analyzer features
# that are broadly applicable to the photographic / screenshot /
# line-art mix in the training corpus. After v2.1 trains, run the
# ablation (`tools/feature_ablation.py` + `tools/feature_group_ablation.py`)
# to prune.
#
# Skipped intentionally:
#   - HDR-specific features (peak/p99 luminance, hdr_*, wide_gamut_*,
#     gamut_coverage_*) — corpus is SDR-only; signals collapse to 0
#   - palette_fits_in_256 / indexed_palette_width — bool/int gating
#     features that are nearly redundant with distinct_color_bins
#   - patch_fraction (slow exact form) — superseded by patch_fraction_fast
#   - composite likelihoods (text/screen/natural_likelihood) — derived
#     from raw signals already in the set; redundant input to picker
#   - skin_tone_fraction — too narrow for codec-config picking
#   - alpha_present (bool) — usually constant per partition
#   - effective_bit_depth — corpus-wide constant 8
KEEP_FEATURES = [
    # Tier 1 (stable)
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_flat_color_block_ratio",
    "feat_colourfulness",
    "feat_laplacian_variance",
    "feat_variance_spread",
    "feat_grayscale_score",
    # Tier 2 (per-axis chroma sharpness)
    "feat_cb_horiz_sharpness",
    "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness",
    "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness",
    # Tier 3 (sampled DCT)
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
    "feat_dct_compressibility_y",
    "feat_dct_compressibility_uv",
    "feat_patch_fraction_fast",
    "feat_quant_survival_y",
    "feat_quant_survival_uv",
    "feat_aq_map_mean",
    "feat_aq_map_std",
    "feat_noise_floor_y",
    "feat_noise_floor_uv",
    "feat_edge_slope_stdev",
    "feat_gradient_fraction",
    "feat_line_art_score",
    # Palette
    "feat_distinct_color_bins",
    "feat_palette_density",
    # Alpha (kept for completeness; corpus has some RGBA)
    "feat_alpha_used_fraction",
    "feat_alpha_bimodal_score",
    # ---------- v2.2 ----------
    # Dimension features (zenanalyze#42). The picker can now drop the
    # hand-built `size_tiny/small/medium/large` one-hot from extra_axes
    # and let the model learn from continuous resolution signals.
    "feat_pixel_count",
    "feat_log_pixels",
    "feat_min_dim",
    "feat_max_dim",
    "feat_bitmap_bytes",
    "feat_aspect_min_over_max",
    "feat_log_aspect_abs",
    "feat_block_misalignment_8",
    "feat_block_misalignment_16",
    "feat_block_misalignment_32",
    "feat_block_misalignment_64",
    "feat_channel_count",
    # Log / derivative variants — disambiguate raw-PixelCount over-
    # fit risk vs real size signal. With multiple log scales available,
    # ablation impact on PixelCount itself drops if the signal is
    # genuinely about resolution; if PixelCount remains dominant, the
    # network was memorizing.
    "feat_log2_pixels",
    "feat_log10_pixels",
    "feat_log_pixels_rounded",
    "feat_sqrt_pixels",
    "feat_log_bitmap_bytes",
    "feat_log_min_dim",
    "feat_log_max_dim",
    "feat_log_padded_pixels_8",
    "feat_log_padded_pixels_16",
    "feat_log_padded_pixels_32",
    "feat_log_padded_pixels_64",
    # Percentile expansions of features whose mean alone collapsed
    # the distribution. The ablation (next pass) will tell us which
    # of these earn their bytes vs the existing parents.
    "feat_aq_map_p50",
    "feat_aq_map_p75",
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    "feat_aq_map_p99",
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
    "feat_quant_survival_uv_p25",
    "feat_quant_survival_uv_p50",
    "feat_quant_survival_uv_p75",
]

# Zq target grid: step 5 from 0..70 + step 2 from 70..100 (the
# perceptibility threshold band where 1-2 zensim points actually
# matter).
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))


# ---------- Config-name parser ----------

# Pattern examples from zenjpeg's `zq_pareto_calibrate.rs` harness:
#   ycbcr_444_noT_cs60        → ycbcr 4:4:4, no trellis, no SA, chroma_scale=0.60
#   ycbcr_444_noT_cs60_sa     → ycbcr 4:4:4, no trellis, SA on,  chroma_scale=0.60
#   ycbcr_444_hyb80_cs60      → ycbcr 4:4:4, trellis lambda=8.0, no SA, cs=0.60
#   ycbcr_444_hyb145_cs100_sa → ycbcr 4:4:4, lambda=14.5, SA on, cs=1.00
#   xyb_420_hyb250_cs150      → xyb BQuarter, trellis lambda=25.0, no SA (xyb-only), cs=1.50
#
# `hyb<N>` encodes lambda × 10 (so hyb80=8.0, hyb145=14.5, hyb250=25.0).
# `cs<N>` encodes chroma_scale × 100.

_CONFIG_RE = re.compile(
    r"^(?P<color>ycbcr|xyb)_(?P<sub>444|420)_"
    r"(?:noT|hyb(?P<lam>\d+))_cs(?P<cs>\d+)(?P<sa>_sa)?$"
)

# Sentinel value for "trellis off" cells. The picker's lambda head
# still emits a value at these cell indices; the codec ignores it
# at inference when the categorical cell has trellis_on=False. 0.0
# is clearly out-of-band relative to the real lambda range
# {8.0, 14.5, 25.0}.
LAMBDA_NOTRELLIS_SENTINEL = 0.0


# ---------- Axis schema (consumed by train_hybrid.py) ----------

# Explicit declaration of which `parse_config_name` keys are
# categorical (form cells) vs scalar (per-cell prediction heads).
# This was the implicit zenjpeg shape baked into the trainer prior to
# the codec-agnostic refactor; declaring it explicitly here lets new
# codec configs use a different shape without hardcoding.
CATEGORICAL_AXES = ["color", "sub", "trellis_on", "sa"]
SCALAR_AXES = ["chroma_scale", "lambda"]
# `lambda` is a sentinel-bearing axis: rows with lambda <= 0.0 are
# trellis-off cells where the lambda value is just a placeholder.
# Train_hybrid.py masks these out of lambda's per-cell teacher.
SCALAR_SENTINELS = {"lambda": LAMBDA_NOTRELLIS_SENTINEL}
# Display-only ranges for the training log.
SCALAR_DISPLAY_RANGES = {
    "chroma_scale": (0.6, 1.5),
    "lambda": (8.0, 25.0),
}


def parse_config_name(name: str) -> dict:
    """Parse a zenjpeg config name into its categorical + scalar axes.

    Returns a dict with keys:
      - `color`, `sub`, `sa`, `trellis_on` — categorical (form cells)
      - `lambda`, `chroma_scale`            — scalar prediction targets
    """
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {name}")
    color = m.group("color")
    sub = m.group("sub")
    lam_raw = m.group("lam")
    cs_raw = m.group("cs")
    sa = m.group("sa") is not None
    trellis_on = lam_raw is not None
    lam_val = LAMBDA_NOTRELLIS_SENTINEL
    if trellis_on:
        lam_val = int(lam_raw) / 10.0
    cs_val = int(cs_raw) / 100.0
    return {
        "color": color,
        "sub": sub,
        "sa": sa,
        "trellis_on": trellis_on,
        "lambda": lam_val,
        "chroma_scale": cs_val,
    }
