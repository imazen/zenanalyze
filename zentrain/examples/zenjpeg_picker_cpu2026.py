"""zenjpeg picker config — 2026-06-24 CPU corpus (ssim2 target, new name@hex8 features).

Current cell format: `<enc>_<trel>_small_<chroma>` (24 cells = 4 encoders x 3 trellis x 2 chroma).
Pure categorical (no scalar axes — the trellis lambda is baked into the `trel` label). Trained against
ssim2 from the 2026-06-24 Hetzner CPU corpus, with the freshly re-extracted 97 name@hex8 features.
KEEP_FEATURES is the full feature set; run feature_ablation.py to prune.
"""
import re
import csv
from pathlib import Path

PARETO = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/pareto/pareto_zenjpeg_2026-06-24.parquet")
FEATURES = Path("/mnt/v/output/imazen-26-features/sdr_features_2026-06-24.tsv")
OUT_JSON = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenjpeg_picker_cpu2026.json")
OUT_LOG = Path("/mnt/v/zen/zensim-training/2026-06-24-cpu/models/zenjpeg_picker_cpu2026.log")

METRIC_COLUMN = "ssim2"
METRIC_DIRECTION = "higher_better"


def _feature_columns():
    """All feature columns (name@hex8) present in the features TSV — id/meta cols excluded."""
    with open(FEATURES) as f:
        header = f.readline().rstrip("\n").split("\t")
    meta = {"image_path", "image_sha", "split", "content_class", "source",
            "size_class", "width", "height"}
    return [c for c in header if c not in meta]


# Ablation from correlation_cleanup.py (2026-06-24, dense corpus, threshold 0.99):
#   5 redundant (>=0.99 Spearman — tree learners route signal through the anchor, ignore the rest)
#   4 corpus-constant (no alpha + always 3-channel on this SDR RGB corpus — zero signal)
# Kept: is_grayscale / palette_fits_in_256 (low-variance binaries, but real chroma-decision signal).
_DROP_FEATURES = {
    "noise_floor_y_p1@a564e727",      # ~ aq_map_p1
    "info_weight_p90@4f50b8b5",       # ~ aq_map_p90
    "bitmap_bytes@4c898cd0",          # ~ log_pixels
    "pixel_count@d49a1919",           # ~ log_pixels
    "noise_floor_y_p10@3f55a530",     # ~ noise_floor_y
    "alpha_present@f307e9da",         # constant 0.0
    "alpha_used_fraction@ec61d7c5",   # constant 0.0
    "alpha_bimodal_score@1ec8c890",   # constant 0.0
    "channel_count@aa8817e3",         # constant 3.0
}
# NEGATIVE RESULT (2026-06-24): dropping the 9 _DROP_FEATURES slightly HURT — 6.07% vs 5.87% full,
# and it ADDED an OVERFIT violation. The 0.99-Spearman "redundant" features still gave the HistGB teacher
# useful alternate split points; the constants were free to keep. So the full set wins. Kept for the record.
KEEP_FEATURES = _feature_columns()

# Dense at the high-q end per the sweep discipline; the trainer interpolates the rd-optimal cell per target.
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

# Pure-categorical form cells; no scalar prediction heads.
CATEGORICAL_AXES = ["enc", "trel", "chroma"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}

OUTPUT_SPECS = {
    "bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"},
}
SPARSE_OVERRIDES: list = []

_CONFIG_RE = re.compile(
    r"^(?P<enc>gls|jp3|moz|pw4)_(?P<trel>t0|tr[0-9.]+(?:\+dc)?)_small_(?P<chroma>420|444|422)$"
)


def parse_config_name(name: str) -> dict:
    """Parse `<enc>_<trel>_small_<chroma>` into its 3 categorical axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable config name: {name}")
    return {"enc": m.group("enc"), "trel": m.group("trel"), "chroma": m.group("chroma")}
