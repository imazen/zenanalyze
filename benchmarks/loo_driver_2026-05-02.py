"""Hand-rolled Tier 3 LOO retrain driver.

For each candidate feature:
1. Generate two temp picker configs — base and base±{feature}.
2. Run `train_hybrid.py` for each, capturing stderr.
3. Parse the "Student metrics: argmin mean overhead X.YZ% argmin_acc Y.YZ%"
   line that train_hybrid prints at the end of student training.
4. Write a results TSV with paired with/without metrics + Δ.

No backgrounding, no clever wait loops — every retrain runs in foreground
so the driver's process tree is the only thing the runtime needs to track.
The previous agent-spawned attempt died on a bg-wait pattern.

Usage:
    python3 benchmarks/loo_driver_2026-05-02.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ZA_ROOT = Path("/home/lilith/work/zen/zenanalyze")
ZW_ROOT = Path("/home/lilith/work/zen/zenwebp")
TRAIN = ZA_ROOT / "zentrain/tools/train_hybrid.py"

OUT_DIR = Path("/tmp/loo_handrolled_2026-05-02")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TSV = ZA_ROOT / "benchmarks" / "loo_retrain_2026-05-02.tsv"

# Features we want a paired with/without verdict on. All exist in
# zenwebp's combined_filled TSV.
CANDIDATES = [
    # Just-restored — does the restoration earn its keep?
    "feat_log_pixels",
    "feat_log_padded_pixels_8",
    "feat_log_padded_pixels_16",
    "feat_log_padded_pixels_32",
    "feat_log_padded_pixels_64",
    # Culled but possibly worth restoring (rgb8-channel-constant linear
    # transform — should be MLP-recoverable, but verify).
    "feat_bitmap_bytes",
    # Algebraic ratio — Tier 0 says cull, tiny MLP can't compute division.
    "feat_palette_density",
    # 4-codec Tier 0 cluster against feat_noise_floor_y / feat_aq_map_p50.
    # Independent metrics that happen to correlate.
    "feat_aq_map_p10",
    "feat_noise_floor_y_p10",
    "feat_noise_floor_y_p50",
]

# zenwebp current schema's 36 KEEP features (pulled at driver-write time;
# this is a frozen copy so the LOO test is reproducible regardless of
# what the live picker config file currently says).
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

# zenwebp pareto + features (existing artifacts — no re-encoding).
PARETO_TSV = ZW_ROOT / "benchmarks/zenwebp_pareto_2026-05-01_combined.tsv"
FEATURES_TSV = ZW_ROOT / "benchmarks/zenwebp_pareto_features_2026-05-01_combined_filled.tsv"


CONFIG_TEMPLATE = '''# Auto-generated LOO config. DO NOT EDIT by hand.
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
    """Write a temp picker config to OUT_DIR and return its path."""
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
    r"Student metrics: argmin mean overhead\s+([0-9.]+)%\s+argmin_acc\s+([0-9.]+)%"
)


def run_train(config_name: str, log_path: Path) -> tuple[float | None, float | None]:
    """Invoke train_hybrid.py with the given temp config and return
    (mean_overhead_pct, argmin_acc_pct) parsed from the stderr summary line.
    Returns (None, None) on failure."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OUT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(TRAIN),
        "--codec-config",
        config_name,
        "--activation",
        "leakyrelu",
    ]
    started = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(
            cmd,
            cwd=str(ZW_ROOT),  # so the relative pareto/features paths in the temp config resolve
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    if result.returncode != 0:
        sys.stderr.write(f"  [FAIL] {config_name} rc={result.returncode} in {elapsed:.1f}s — see {log_path}\n")
        return None, None
    log_text = log_path.read_text()
    m = _STUDENT_RE.search(log_text)
    if not m:
        sys.stderr.write(f"  [PARSE FAIL] {config_name} — no Student metrics line in {log_path}\n")
        return None, None
    mean_pct = float(m.group(1))
    argmin_acc = float(m.group(2))
    sys.stderr.write(
        f"  [OK] {config_name}: mean_overhead={mean_pct:.2f}%  argmin_acc={argmin_acc:.1f}%  ({elapsed:.0f}s)\n"
    )
    return mean_pct, argmin_acc


def main() -> int:
    base = list(BASE_KEEP_FEATURES)

    sys.stderr.write(
        f"[loo] base feature count: {len(base)}\n"
        f"[loo] candidates: {len(CANDIDATES)}\n"
        f"[loo] paired retrains: {len(CANDIDATES) * 2}\n\n"
    )

    rows: list[dict[str, float | str | None]] = []

    for i, feat in enumerate(CANDIDATES, 1):
        sys.stderr.write(f"\n[loo {i}/{len(CANDIDATES)}] === {feat} ===\n")

        # 'with' = base ∪ {feat}; 'without' = base \ {feat}.
        keep_with = list(base)
        if feat not in keep_with:
            keep_with.append(feat)
        keep_without = [f for f in base if f != feat]

        # If both lists are identical (feat not in base AND not addable
        # because it'd be a no-op... shouldn't happen here), skip.
        if keep_with == keep_without:
            sys.stderr.write(f"  SKIP — base unchanged\n")
            continue

        slug = feat.replace("feat_", "")

        cfg_with = write_config(f"loo_with_{slug}", keep_with)
        log_with = OUT_DIR / f"loo_with_{slug}_train.log"
        with_mean, with_argmin = run_train(cfg_with.stem, log_with)

        cfg_without = write_config(f"loo_without_{slug}", keep_without)
        log_without = OUT_DIR / f"loo_without_{slug}_train.log"
        without_mean, without_argmin = run_train(cfg_without.stem, log_without)

        delta_mean = (
            without_mean - with_mean
            if with_mean is not None and without_mean is not None
            else None
        )
        delta_argmin = (
            without_argmin - with_argmin
            if with_argmin is not None and without_argmin is not None
            else None
        )

        rows.append(
            {
                "codec": "zenwebp",
                "feature": feat,
                "with_mean_overhead_pct": with_mean,
                "without_mean_overhead_pct": without_mean,
                "delta_mean_overhead_pp": delta_mean,
                "with_argmin_acc_pct": with_argmin,
                "without_argmin_acc_pct": without_argmin,
                "delta_argmin_acc_pp": delta_argmin,
            }
        )

    # Write TSV
    cols = [
        "codec",
        "feature",
        "with_mean_overhead_pct",
        "without_mean_overhead_pct",
        "delta_mean_overhead_pp",
        "with_argmin_acc_pct",
        "without_argmin_acc_pct",
        "delta_argmin_acc_pp",
    ]
    with open(RESULTS_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
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

    sys.stderr.write(f"\n[loo] DONE. Wrote {RESULTS_TSV}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
