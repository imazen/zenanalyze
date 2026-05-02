"""
Codec config for zenjpeg's hybrid-heads picker.

Used by `zentrain/tools/train_hybrid.py` (and the other tools/
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
      python3 <zenanalyze>/zentrain/tools/train_hybrid.py \\
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
FEATURES = Path("benchmarks/zq_pareto_features_2026-05-01.tsv")

# Where to write the trained model + summary:
OUT_JSON = Path("benchmarks/zq_bytes_hybrid_2026-05-01.json")
OUT_LOG = Path("benchmarks/zq_bytes_hybrid_2026-05-01.log")


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
#   - palette_fits_in_256 / palette_log2_size — bool/int gating
#     features that are nearly redundant with distinct_color_bins
#   - patch_fraction (slow exact form) — superseded by patch_fraction_fast
#   - composite likelihoods (text/screen/natural_likelihood) — derived
#     from raw signals already in the set; redundant input to picker
#   - skin_tone_fraction — too narrow for codec-config picking
#   - alpha_present (bool) — usually constant per partition
#   - effective_bit_depth — corpus-wide constant 8
# v2.2-clean schema. Tighter than v2.2-pruned (60 features). Built by:
#
#   1. Starting from v2.1 (35 features). Drop `feat_distinct_color_bins`
#      (palette-codec feature, structurally irrelevant for zenjpeg).
#   2. Add only ablation-top-rated dimension features (5):
#      pixel_count (#1 ablation, +4.89pp), log_pixels, aspect_min_over_max,
#      log_padded_pixels_8 (encoded surface area), channel_count.
#      Drop the rest of the 12 dimension features and all 11 log
#      derivatives — Spearman correlation showed redundancy and
#      ablation showed only the above are load-bearing.
#   3. Add only ablation-top-rated percentiles (~10):
#      laplacian_variance p50/75/99/peak (p50 was #4 in entire schema),
#      aq_map p75/90/95/99 (drop p50 — Spearman ρ=0.962 with mean,
#      redundant), noise_floor_y p50/90 (drop p25 — Spearman ρ=0.956
#      with parent), quant_survival_y p10 (worst-block survival → trellis ROI).
#   4. Drop UV-side percentiles entirely — parent quant_survival_uv and
#      noise_floor_uv both had negative ablation Δ on subset100.
#   5. Drop other negatives that are zenjpeg-relevant but small-corpus
#      noisy (high_freq_energy_ratio, flat_color_block_ratio,
#      grayscale_score). On full 347-image retrain these likely become
#      weak-positive; revisit then.
#
# Original v2.2-pruned dropped too aggressively (val 4.55% vs 3.25%
# baseline). v2.2-clean keeps the load-bearing redundancy.
# v2.2-pruned schema. Built by:
#   1. Starting from v2.2 (v2.1 + 12 dimension features + 26 percentile
#      features + 11 log/padded variants — see zenanalyze#42).
#   2. Dropping 7 negative-ablation features: distinct_color_bins,
#      high_freq_energy_ratio, flat_color_block_ratio, quant_survival_uv,
#      noise_floor_uv, noise_floor_uv_p50, grayscale_score
#      (each had Δ ≤ −0.05pp under permutation importance — model
#      overfit on noise from these inputs).
#   3. Dropping 9 redundant log/derivative variants (log2_pixels,
#      log10_pixels, log_pixels_rounded, sqrt_pixels, log_bitmap_bytes,
#      log_min_dim, log_max_dim, log_padded_pixels_16/32/64) — the
#      logs-only retrain showed the MLP performs worse without
#      `pixel_count`, so the linear dims carry the load and the extra
#      log scales are noise. Keep only `log_pixels` + `log_padded_pixels_8`
#      as cushion for extreme out-of-distribution sizes.
#   4. Keeping all 4 linear dimension features (pixel_count, min_dim,
#      max_dim, bitmap_bytes) — pixel_count alone was +4.89pp ablation
#      impact, the dominant signal in the schema.
KEEP_FEATURES = [
    # ---------- v2.1 inheritance (34 of 35 — drop palette-only feature) ----------
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
    "feat_cb_horiz_sharpness",
    "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness",
    "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness",
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
    # DROPPED: feat_line_art_score (zenanalyze 0.1.0 deleted composite
    # likelihoods + LineArtScore — ids 27/28/29/45 reserved).
    # DROPPED: feat_distinct_color_bins — palette codec only, structurally
    # irrelevant for zenjpeg (true-color JPEG doesn't care about palette
    # fits-in-N).
    "feat_palette_density",
    "feat_alpha_used_fraction",
    "feat_alpha_bimodal_score",
    # ---------- v2.2 dimension features (4 of 12 — ablation-validated only) ----------
    "feat_pixel_count",            # #1 ablation impact (+4.89pp)
    "feat_log_pixels",             # smooth resolution axis — restored 2026-05-02 after tiny-MLP expressivity review
    "feat_aspect_min_over_max",    # bounded strip detection
    # DROPPED: feat_log_padded_pixels_8 (zenanalyze 0.1.0 deleted the
    # log_padded_pixels_{8,16,32,64} variants — ids reserved).
    "feat_channel_count",          # discrete RGB/RGBA distinction
    # DROPPED: min_dim/max_dim/bitmap_bytes/log_aspect_abs/block_misalignment{8,16,32,64}
    # — redundant with the above per Spearman + ablation.
    # DROPPED: 11 log derivatives (log2/log10/log_pixels_rounded/sqrt/
    #   log_bitmap_bytes/log_min_dim/log_max_dim/log_padded_pixels_{16,32,64})
    # — logs-only retrain (4.10% val) confirmed pixel_count carries the
    # MLP signal; logs are redundant for tree teachers and noise for MLP students.
    # ---------- v2.2 percentile features (~10, ablation-validated) ----------
    # AqMap: drop p50 (Spearman ρ=0.962 with mean, redundant). p75/90/95/99
    # all clear ≥0.05pp ablation.
    "feat_aq_map_p75",
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    "feat_aq_map_p99",
    # NoiseFloorY: drop p25 (Spearman ρ=0.956 with parent). p50 was #9 in
    # ablation; p90 captures top-tail noise.
    "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p90",
    # Laplacian: keep all 5. Spearman showed no redundancy. p50 was #4 in
    # ablation, p75 was #7. peak captures rare extreme edges.
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance_p90",
    "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak",
    # QuantSurvivalY: keep p10 only — "worst-block survival" → trellis ROI.
    # Parent + p25/50/75 are tightly correlated (ρ > 0.85); p10 is the
    # one quantile that adds new information.
    "feat_quant_survival_y_p10",
    # DROPPED: all noise_floor_uv and quant_survival_uv percentiles —
    # parents were negative-Δ ablation; UV signal handled by chroma
    # sharpness features instead.
    # ---------- 2 new features kept in zenanalyze 0.1.0 (ids 116, 120) ----------
    # ChromaKurtosis (117), UniformitySmooth (118), FlatColorSmooth (119)
    # were retired 2026-05-01 — Tier-0 redundant on cross-codec ablation.
    "feat_luma_kurtosis",            # #116 — quartic moment of |∇²L|
    "feat_gradient_fraction_smooth", # #120 — per-block low-AC ratio
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
