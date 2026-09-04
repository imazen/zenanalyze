"""zenavif AVIF autotune picker — the 2026-09-04 canonical DOE view.

CELL GRAMMAR  ``<backend>-s<speed>[-<knobs>][-bd10]``
  backend ∈ {svt, rav}   svt = zenav1-svt, rav = zenrav1e (zenavif's default)
  speed   ∈ 1..7 (svt; 8-10 alias 7 and are dropped) / 1..10 (rav)
  knobs   = the '-'-joined sorted SVT deviation set ('' = the encoder default)
  bd10    = 10-bit; absent = 8-bit (the default), encoded BY ABSENCE

Built from `/mnt/v/zen/avif-autotune-2026-09-04/` — the union of every scored
AVIF DOE wave (Stage-A a0r/a1/a2/ag, Stage-B6, the naive native control, the
era-delta a1/b1/c1 wave, and the backend-race brnat/brsdr), joined by
`zenmetrics/scripts/jobsys/avif_autotune_view.py`. See that view's
`_MANIFEST.json` for per-source row counts, sha256s and every exclusion.

READ BEFORE USING THE OUTPUT — the four limits that are properties of the DATA,
not of the model:

1. **backend and chroma are PERFECTLY COLLINEAR.** Every svt cell is 4:2:0 and
   every zenrav1e cell is 4:4:4 (measured: 1,114 av1C boxes, 0 exceptions),
   because `zenmetrics-cli/src/sweep/encode.rs::avif_config_from_knobs` pins
   `Yuv420` for svt and leaves zenravif on zenavif's `Yuv444` default — no
   chroma knob is wired for AVIF at all. The picker CANNOT separate them; a
   "backend" pick here is a (backend x chroma) pick. Splitting them needs the
   rank-1 gap: a zenrav1e 4:2:0 arm.
2. **`encode_ms` is MODELLED, not measured.** No fleet path persists a duration.
   The column comes from the 2026-09-03 speed instrument's alpha+beta fits and
   inherits its limits: single-threaded wall time, q45-anchored, 5 of 32 sources
   have a per-source fit (the pooled linear model is flagged failed on 20/20
   arms, beta spread up to 24.3x). The time head therefore learns a MODEL, and
   any time-budget gate built on it is only as good as that model.
3. **`ssim2` is the only corpus-wide quality response** and it is
   sharpness-blind — which is exactly the knob family (`shp*`) the DOE rejects.
   "costs bits at matched ssim2" and "not worth enabling" are different claims;
   only the first is measured.
4. **There is no odd-origin holdout.** All 32 references are EVEN-origin by
   construction (the corpus was k-means-selected under `--parity 0`), so the
   canonical {1,3,5} / {7,9} buckets are structurally empty. `SPLIT_RULE`
   below invokes the registered even-only sub-split (DATA_SPLITS.md L158).
   The `eval8` leg (6 origins) is a leg-side holdout, not the canonical one.
"""
from pathlib import Path

VIEW = Path("/mnt/v/zen/avif-autotune-2026-09-04")
PARETO = VIEW / "pareto_avif_autotune.parquet"
FEATURES = VIEW / "features_avif_autotune.tsv"
OUT_JSON = VIEW / "models" / "zenavif_autotune_v1_full.json"
OUT_LOG = VIEW / "models" / "zenavif_autotune_v1_full.log"

METRIC_COLUMN = "ssim2"
METRIC_DIRECTION = "higher_better"
TIME_COLUMN = "encode_ms"

# The corpus is even-origin BY CONSTRUCTION; the canonical val/test buckets are
# structurally empty. Registered sub-split of the TRAIN bucket:
# {0,2,4,6} = train, {8} = the leg-side eval holdout.
SPLIT_RULE = "even_only_eval8"

# zenanalyze that extracted `features_avif_autotune.tsv`. The G-CROP budget
# table and the native table were both produced by
# `zenanalyze/target/release/examples/extract_features_imazen26_crops`, which
# calls `analyze_features_rgb8` with `FeatureSet::SUPPORTED` and
# `AnalysisQuery::new` (config_hash 0) — the SAME entry point zenavif's
# `auto_tune` own-pass uses, so a runtime pass reproduces these values.
ANALYSIS_PROVENANCE = {"analyzer_version": "0.2.0", "feature_config_hash": 0}


def _feature_columns():
    with open(FEATURES) as f:
        header = f.readline().rstrip("\n").split("\t")
    meta = {"image_path", "image_sha", "split", "content_class", "source",
            "crop_label", "size_class", "width", "height"}
    return [c for c in header if c not in meta]


KEEP_FEATURES = _feature_columns()

# Sweep discipline: step 5 across 0..70, step 2 across 70..100 — denser through
# the perceptibility band where 1-2 points decide a production pick.
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["backend", "speed", "knobs", "bd"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}

OUTPUT_SPECS = {"bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"}}
SPARSE_OVERRIDES: list = []


def parse_config_name(name: str) -> dict:
    """`<backend>-s<speed>[-<knobs>][-bd10]` -> its four categorical axes."""
    parts = name.split("-")
    if len(parts) < 2 or parts[0] not in ("svt", "rav") or not parts[1].startswith("s"):
        raise ValueError(f"unparseable config name: {name}")
    backend, speed = parts[0], parts[1]
    rest = parts[2:]
    bd = "bd10" if rest and rest[-1] == "bd10" else "bd8"
    if bd == "bd10":
        rest = rest[:-1]
    return {
        "backend": backend,
        "speed": speed,
        "knobs": "-".join(rest) if rest else "default",
        "bd": bd,
    }
