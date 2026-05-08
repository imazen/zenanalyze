"""Example zensim metric trainer config — single-cell regression.

Mirrors the per-codec picker config pattern (e.g.
`zenjpeg_picker_config.py`) but adapted for zensim's single-output
regression instead of N-categorical-cells.

Usage:
    python3 -m zentrain.tools.zensim_metric_train \
        --config zentrain.examples.zensim_metric_config

The config exports module-level constants the trainer reads.
"""
from pathlib import Path

# --- Inputs ----------------------------------------------------------------

# Primary training data: zensim-validate `--features-csv` rewrite
# (header: ref_basename, human_score, f0..f299). 218k base + 122k e1
# fill = 340,206 rows. SSIM2 column is `human_score` (here = ssim2/100).
SYNTH_FEATURES = Path(
    "/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv.features.parquet"
)

# Auxiliary supervision (mixed-supervision recipe per V0_4 2026-04-30):
# format `(name, path, train_weight, val_frac)`. val_frac=0.25 holds
# 25% of unique refs as val-only.
AUXILIARY_DATASETS = [
    # (name, csv_path, train_weight, val_frac)
    ("kadid", Path("/mnt/v/zen/zensim-training/2026-05-07/v06-features/kadid_features.csv"), 0.3, 0.25),
    ("tid",   Path("/mnt/v/zen/zensim-training/2026-05-07/v06-features/tid_features.csv"),   0.3, 0.25),
    # CID22 is held out 100% — the gold standard. Per user 2026-05-08:
    # "cid22 is vastly more trusted for validation than other data sets
    # as the others are NOT compression tuned. ssim2 was tuned on cid22
    # so the cid22 training set synthetic alignment with ssim2 is the
    # most powerful authority."
    ("cid22", Path("/mnt/v/zen/zensim-training/2026-05-07/v06-features/cid22_features.csv"), 0.0, 1.0),
]

# Optional: zenanalyze TSV for `dct_hf` feature appender (V0_6 dct_hf,
# the published-baseline leader on CID22 per 4metric_overnight_FINAL_2026-05-01.md).
ZENANALYZE_TSV = Path(
    "/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1.tsv")
ZENANALYZE_FEATURES = [
    "dct_compressibility_y",
    "dct_compressibility_uv",
    "high_freq_energy_ratio",
]

# --- Architecture ----------------------------------------------------------

HIDDEN = [64]              # single hidden layer of 64 LeakyReLU
N_FEATURES_BASE = 228      # zensim's basic + peak features (post-strip)
# When `ZENANALYZE_FEATURES` non-empty, n_features = 228 + len(...)

# --- Training --------------------------------------------------------------

TARGET_PRIMARY = "score_ssim2"   # ssim2 IS CID22-MOS-aligned per CID22 paper §4
EPOCHS = 200
BATCH_SIZE = 16384
LR = 1e-3
WEIGHT_DECAY = 1e-5
RANK_WEIGHT = 0.5

# Magnitude-matching aux loss (v04-smooth recipe). λ * |α·|target|−|pred|.
# Set λ=0 to disable.
MAGNITUDE_MATCH_LAMBDA = 0.001
MAGNITUDE_MATCH_ALPHA = 30.0

# Low-band oversample (V0_7 / sampler-bias) — v07 e1-ablation found
# 0.5 was the most-explored value but every fraction regressed on
# human-MOS axes. Default 0 (no bias) until re-validated.
LOW_BAND_OVERSAMPLE = 0.0

# TV regularizer (per-curve adjacent-q monotonicity). 0 disables.
TV_WEIGHT = 0.0

# --- Validation ------------------------------------------------------------

VAL_POLICY = "min"   # "min" picks epoch where the min-over-datasets val
                     # SROCC is highest. CID22 is the gold standard, so
                     # this approximates "best-on-CID22-without-blowing
                     # -up-other-holdouts".

# --- Output ----------------------------------------------------------------

OUT_DIR = Path("/mnt/v/output/zensim/synthetic-v2/runs")
TAG = "v07_zentrain_ssim2_64h"

# Bake target — ZNPR v3 per 2026-05-08 user direction "everyone uses v3".
# Until zenpredict 0.2.0 ships v3 reader, set to 2 for deployable bakes.
ZNPR_VERSION = 3

# Flip output → distance scale (so zensim's runtime score_mapping
# `100 - a*d^b` with a=1, b=1 returns ssim2-scale predictions).
FLIP_OUTPUT = True

# --- Bake metadata stamped into the .bin -----------------------------------

BAKE_METADATA = {
    "zensim.profile": "zensim-preview-v0.7",
    "train.recipe": "v07_zentrain_ssim2_mixed_supervision",
    "train.synth_csv": str(SYNTH_FEATURES),
    "train.aux_datasets": ",".join(name for name, *_ in AUXILIARY_DATASETS),
    "train.zenanalyze_features": ",".join(ZENANALYZE_FEATURES),
}
