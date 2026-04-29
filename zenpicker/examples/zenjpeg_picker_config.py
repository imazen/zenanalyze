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

# zenjpeg's pareto sweep harness produces these:
PARETO = Path("benchmarks/zq_pareto_2026-04-29.tsv")
FEATURES = Path("benchmarks/zq_pareto_features_2026-04-29.tsv")

# Where to write the trained model + summary:
OUT_JSON = Path("benchmarks/zq_bytes_hybrid_2026-04-29.json")
OUT_LOG = Path("benchmarks/zq_bytes_hybrid_2026-04-29.log")


# ---------- Schema ----------

# 8-feature reduced schema (validated by zenjpeg's PR #129 ablation;
# 11 of 19 zenanalyze features dropped including the entire Tier 2
# chroma sliding-window pass + alpha + palette tiers). See the
# zenpicker README's "Documentation map" → FOR_NEW_CODECS.md for
# how the codec runs its own ablation if a different feature set
# is preferred.
KEEP_FEATURES = [
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
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
