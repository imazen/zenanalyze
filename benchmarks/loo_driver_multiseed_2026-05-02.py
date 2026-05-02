"""Multi-seed extension of the LOO retrain driver.

Same idea as `loo_driver_2026-05-02.py` but for each candidate feature
runs paired with/without across N seeds, then aggregates mean / stddev
of the deltas. The trainer's --seed override re-keys both the
train/val image-level split AND the MLP init, so each seed gives an
independent estimate.

Usage:
    python3 benchmarks/loo_driver_multiseed_2026-05-02.py
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ZA_ROOT = Path("/home/lilith/work/zen/zenanalyze")
ZW_ROOT = Path("/home/lilith/work/zen/zenwebp")
TRAIN = ZA_ROOT / "zentrain/tools/train_hybrid.py"

OUT_DIR = Path("/tmp/loo_multiseed_2026-05-02")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TSV = ZA_ROOT / "benchmarks" / "loo_retrain_multiseed_2026-05-02.tsv"
RAW_TSV = ZA_ROOT / "benchmarks" / "loo_retrain_multiseed_raw_2026-05-02.tsv"

SEEDS = [0, 1, 2, 3, 4]

# Focus on the mixed-signal candidates from the single-seed run plus
# the 3 high-confidence movers (as cross-check of variance estimates).
CANDIDATES = [
    # Mixed signals (this is the question we need answered):
    "feat_log_pixels",
    "feat_aq_map_p10",
    "feat_palette_density",
    "feat_noise_floor_y_p50",
    # High-confidence (cross-check the single-seed N=1 magnitude):
    "feat_bitmap_bytes",
    "feat_log_padded_pixels_64",
    # Already-confirmed restorations (cross-check):
    "feat_log_padded_pixels_8",
    "feat_log_padded_pixels_16",
]

# Frozen base copy (matches loo_driver_2026-05-02.py — keep in sync).
BASE_KEEP_FEATURES = [
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
    "feat_cb_horiz_sharpness",
    "feat_noise_floor_y_p25",
    "feat_noise_floor_uv",
    "feat_chroma_complexity",
    "feat_quant_survival_y_p75",
    "feat_aq_map_std",
    "feat_gradient_fraction",
    "feat_noise_floor_y_p75",
    "feat_high_freq_energy_ratio",
    "feat_colourfulness",
    "feat_quant_survival_uv",
    "feat_luma_kurtosis",
    "feat_gradient_fraction_smooth",
]

PARETO_TSV = ZW_ROOT / "benchmarks/zenwebp_pareto_2026-05-01_combined.tsv"
FEATURES_TSV = ZW_ROOT / "benchmarks/zenwebp_pareto_features_2026-05-01_combined_filled.tsv"


CONFIG_TEMPLATE = '''# Auto-generated multi-seed LOO config. DO NOT EDIT.
from __future__ import annotations
import re
from pathlib import Path

PARETO = Path({pareto!r})
FEATURES = Path({features!r})
OUT_JSON = Path({out_json!r})
OUT_LOG = Path({out_log!r})

KEEP_FEATURES = {keep_features!r}

ZQ_TARGETS = list(range(30, 70, 5)) + list(range(70, 96, 2))

CATEGORICAL_AXES = ["method", "segments"]
SCALAR_AXES = ["sns_strength", "filter_strength", "filter_sharpness"]
SCALAR_SENTINELS = {{}}
SCALAR_DISPLAY_RANGES = {{
    "sns_strength": (0, 100),
    "filter_strength": (0, 100),
    "filter_sharpness": (0, 7),
}}

_CONFIG_RE = re.compile(
    r"^m(?P<method>\\d+)_seg(?P<seg>\\d+)_cm(?P<cm>[01])"
    r"_sns(?P<sns>\\d+)_fs(?P<fs>\\d+)_sh(?P<sh>\\d+)"
    r"_pl(?P<pl>\\d+)_mp(?P<mp>[01])$"
)
_CONFIG_RE_V01 = re.compile(
    r"^m(?P<method>\\d+)_seg(?P<seg>\\d+)"
    r"_sns(?P<sns>\\d+)_fs(?P<fs>\\d+)_sh(?P<sh>\\d+)$"
)


def parse_config_name(name: str) -> dict:
    m = _CONFIG_RE.match(name)
    if m:
        return {{
            "method": int(m.group("method")),
            "segments": int(m.group("seg")),
            "sns_strength": float(m.group("sns")),
            "filter_strength": float(m.group("fs")),
            "filter_sharpness": float(m.group("sh")),
        }}
    m = _CONFIG_RE_V01.match(name)
    if m:
        return {{
            "method": int(m.group("method")),
            "segments": int(m.group("seg")),
            "sns_strength": float(m.group("sns")),
            "filter_strength": float(m.group("fs")),
            "filter_sharpness": float(m.group("sh")),
        }}
    raise ValueError("unparseable: " + name)
'''


def write_config(name: str, keep: list[str]) -> Path:
    cfg_path = OUT_DIR / f"{name}.py"
    out_json = OUT_DIR / f"{name}_model.json"
    out_log = OUT_DIR / f"{name}_train.log"
    body = CONFIG_TEMPLATE.format(
        pareto=str(PARETO_TSV),
        features=str(FEATURES_TSV),
        out_json=str(out_json),
        out_log=str(out_log),
        keep_features=keep,
    )
    cfg_path.write_text(body)
    return cfg_path


_STUDENT_RE = re.compile(
    r"Student:\s+mean\s+([0-9.]+)%\s+argmin_acc\s+([0-9.]+)%"
)


def run_train(config_name: str, seed: int, log_path: Path) -> tuple[float | None, float | None]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OUT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(TRAIN),
        "--codec-config", config_name,
        "--activation", "leakyrelu",
        "--seed", str(seed),
    ]
    started = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(
            cmd,
            cwd=str(ZW_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    if result.returncode != 0:
        sys.stderr.write(f"  [FAIL seed={seed}] {config_name} rc={result.returncode} ({elapsed:.0f}s)\n")
        return None, None
    text = log_path.read_text()
    m = _STUDENT_RE.search(text)
    if not m:
        sys.stderr.write(f"  [PARSE FAIL seed={seed}] {config_name} ({elapsed:.0f}s)\n")
        return None, None
    return float(m.group(1)), float(m.group(2))


def main() -> int:
    base = list(BASE_KEEP_FEATURES)
    n_pairs = len(CANDIDATES) * len(SEEDS) * 2

    sys.stderr.write(
        f"[loo-multi] base feature count: {len(base)}\n"
        f"[loo-multi] candidates: {len(CANDIDATES)}, seeds: {SEEDS}\n"
        f"[loo-multi] total retrains: {n_pairs}\n\n"
    )

    raw_rows: list[list] = []
    summary_rows: list[dict] = []

    started_all = time.time()

    for fi, feat in enumerate(CANDIDATES, 1):
        slug = feat.replace("feat_", "")
        sys.stderr.write(f"\n[loo-multi {fi}/{len(CANDIDATES)}] === {feat} ===\n")

        keep_with = list(base) if feat in base else list(base) + [feat]
        keep_without = [f for f in base if f != feat]

        cfg_with = write_config(f"loo_with_{slug}", keep_with)
        cfg_without = write_config(f"loo_without_{slug}", keep_without)

        per_seed = []
        for seed in SEEDS:
            log_with = OUT_DIR / f"loo_with_{slug}_seed{seed}.log"
            wm, wa = run_train(cfg_with.stem, seed, log_with)
            log_without = OUT_DIR / f"loo_without_{slug}_seed{seed}.log"
            om, oa = run_train(cfg_without.stem, seed, log_without)

            if wm is None or om is None:
                sys.stderr.write(f"  [SKIP seed={seed}] one side failed\n")
                continue

            d_oh = om - wm
            d_ac = oa - wa
            sys.stderr.write(
                f"  seed={seed}: with=({wm:.2f}%, {wa:.1f}%) "
                f"without=({om:.2f}%, {oa:.1f}%)  "
                f"ΔOH={d_oh:+.2f}pp  ΔAC={d_ac:+.2f}pp\n"
            )
            raw_rows.append([feat, seed, wm, om, d_oh, wa, oa, d_ac])
            per_seed.append((d_oh, d_ac))

        if per_seed:
            ohs = [x[0] for x in per_seed]
            acs = [x[1] for x in per_seed]
            mean_oh = statistics.mean(ohs)
            mean_ac = statistics.mean(acs)
            std_oh = statistics.stdev(ohs) if len(ohs) > 1 else 0.0
            std_ac = statistics.stdev(acs) if len(acs) > 1 else 0.0
            summary_rows.append({
                "feature": feat,
                "n_seeds_ok": len(per_seed),
                "mean_delta_overhead_pp": mean_oh,
                "stddev_delta_overhead_pp": std_oh,
                "mean_delta_argmin_pp": mean_ac,
                "stddev_delta_argmin_pp": std_ac,
            })
            elapsed = time.time() - started_all
            sys.stderr.write(
                f"  → mean ΔOH={mean_oh:+.2f}±{std_oh:.2f}pp  "
                f"mean ΔAC={mean_ac:+.2f}±{std_ac:.2f}pp  "
                f"({elapsed:.0f}s elapsed)\n"
            )

    # Raw TSV
    with open(RAW_TSV, "w") as f:
        f.write("\t".join([
            "feature", "seed",
            "with_overhead_pct", "without_overhead_pct", "delta_overhead_pp",
            "with_argmin_pct", "without_argmin_pct", "delta_argmin_pp",
        ]) + "\n")
        for r in raw_rows:
            vals = []
            for v in r:
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    # Summary TSV
    cols = [
        "feature", "n_seeds_ok",
        "mean_delta_overhead_pp", "stddev_delta_overhead_pp",
        "mean_delta_argmin_pp", "stddev_delta_argmin_pp",
    ]
    with open(RESULTS_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in summary_rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")

    sys.stderr.write(
        f"\n[loo-multi] DONE.\n"
        f"  raw TSV: {RAW_TSV}\n"
        f"  summary TSV: {RESULTS_TSV}\n"
        f"  total wall: {(time.time() - started_all):.0f}s\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
