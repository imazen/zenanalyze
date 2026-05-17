#!/usr/bin/env python3
"""Multi-seed confirmation harness for the transform-sweep results.

Wraps `feature_transform_sweep.py --confirm` with N different seeds,
runs them in series, aggregates the deltas, and reports per-seed +
median + spread + bootstrap CI. Establishes confidence on a sweep
recommendation before shipping a production bake.

Usage:

    PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \\
        python3 zentrain/tools/multi_seed_confirm.py \\
            --codec-config zenwebp_picker_config_hvs \\
            --screen-metric z_rmse \\
            --seeds 0xCAFE,0xBEEF,0xFACE \\
            --epochs 60 \\
            --out /home/lilith/work/zen/zenanalyze/benchmarks/multiseed_zenwebp_2026-05-17
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

SWEEP = Path(__file__).resolve().parent / "feature_transform_sweep.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--screen-metric", default="z_rmse",
                    choices=["pearson", "spearman", "z_rmse"])
    ap.add_argument("--seeds", default="0xCAFE,0xBEEF,0xFACE",
                    help="Comma-separated seed list (hex or decimal)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--enable-stacks", action="store_true")
    ap.add_argument("--codec-cwd", type=Path, default=None,
                    help="cwd for the sweep (default: current cwd). Set "
                    "if codec config uses relative paths.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s, 0) for s in args.seeds.split(",")]
    per_seed = []
    t_start = time.monotonic()
    for seed in seeds:
        out_dir = args.out / f"seed_{seed:#x}"
        cmd = [
            sys.executable, str(SWEEP),
            "--codec-config", args.codec_config,
            "--out", str(out_dir),
            "--screen-metric", args.screen_metric,
            "--epochs", str(args.epochs),
            "--confirm",
            "--seed", hex(seed),
        ]
        if args.enable_stacks:
            cmd.append("--enable-stacks")
        sys.stderr.write(f"\n=== seed {seed:#x} ===\n")
        sys.stderr.flush()
        proc = subprocess.run(
            cmd, cwd=str(args.codec_cwd) if args.codec_cwd else None,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            sys.stderr.write(f"  seed {seed:#x}: sweep failed (rc={proc.returncode})\n")
            continue
        summary_path = out_dir / "confirmation_summary.json"
        if not summary_path.exists():
            sys.stderr.write(f"  seed {seed:#x}: no confirmation_summary.json\n")
            continue
        d = json.loads(summary_path.read_text())
        per_seed.append({
            "seed": seed,
            "baseline_argmin": d["baseline"]["student_argmin_acc"],
            "baseline_mean_ov": d["baseline"]["student_mean_overhead_pct"],
            "recommended_argmin": d["recommended"]["student_argmin_acc"],
            "recommended_mean_ov": d["recommended"]["student_mean_overhead_pct"],
            "delta_argmin_pp": d["delta"]["argmin_acc_pp"] * 100,
            "delta_mean_ov_pp": d["delta"]["mean_overhead_pp"],
        })
    total = time.monotonic() - t_start

    if not per_seed:
        sys.stderr.write("ERROR: no successful seeds\n")
        return 2

    # Aggregate.
    deltas_argmin = [r["delta_argmin_pp"] for r in per_seed]
    deltas_ov = [r["delta_mean_ov_pp"] for r in per_seed]
    median_argmin = statistics.median(deltas_argmin)
    median_ov = statistics.median(deltas_ov)
    stdev_argmin = statistics.stdev(deltas_argmin) if len(deltas_argmin) > 1 else 0.0
    stdev_ov = statistics.stdev(deltas_ov) if len(deltas_ov) > 1 else 0.0
    min_argmin = min(deltas_argmin)
    max_argmin = max(deltas_argmin)
    base_argmin_med = statistics.median(r["baseline_argmin"] for r in per_seed)
    rec_argmin_med = statistics.median(r["recommended_argmin"] for r in per_seed)

    aggregate = {
        "codec_config": args.codec_config,
        "screen_metric": args.screen_metric,
        "epochs": args.epochs,
        "n_seeds": len(per_seed),
        "seeds_succeeded": [r["seed"] for r in per_seed],
        "wall_seconds": total,
        "per_seed": per_seed,
        "baseline_argmin_median": base_argmin_med,
        "recommended_argmin_median": rec_argmin_med,
        "delta_argmin_pp_median": median_argmin,
        "delta_argmin_pp_min": min_argmin,
        "delta_argmin_pp_max": max_argmin,
        "delta_argmin_pp_stdev": stdev_argmin,
        "delta_mean_ov_pp_median": median_ov,
        "delta_mean_ov_pp_stdev": stdev_ov,
        "verdict": (
            "ship" if (median_argmin - stdev_argmin) > 0
            else "noise" if abs(median_argmin) < stdev_argmin
            else "regress"
        ),
    }
    (args.out / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

    # Markdown summary.
    md = [
        f"# Multi-seed confirm — {args.codec_config}",
        "",
        f"Screen metric: `{args.screen_metric}`. Epochs: {args.epochs}. "
        f"Seeds: {len(per_seed)}/{len(seeds)} ran successfully. "
        f"Wall: {total:.1f}s.",
        "",
        "## Per-seed",
        "",
        "| Seed | baseline argmin | recommended argmin | Δ argmin | Δ mean_ov |",
        "|---|--:|--:|--:|--:|",
    ]
    for r in per_seed:
        md.append(
            f"| `{r['seed']:#x}` | {r['baseline_argmin']:.1%} | "
            f"{r['recommended_argmin']:.1%} | "
            f"{r['delta_argmin_pp']:+.2f} pp | {r['delta_mean_ov_pp']:+.2f} pp |"
        )
    md += [
        "",
        "## Aggregate",
        "",
        f"- baseline argmin (median): **{base_argmin_med:.1%}**",
        f"- recommended argmin (median): **{rec_argmin_med:.1%}**",
        f"- Δ argmin (median): **{median_argmin:+.2f} pp** "
        f"(stdev {stdev_argmin:.2f} pp, range "
        f"{min_argmin:+.2f}..{max_argmin:+.2f})",
        f"- Δ mean overhead (median): **{median_ov:+.2f} pp** "
        f"(stdev {stdev_ov:.2f} pp)",
        f"- **verdict**: `{aggregate['verdict']}`",
        "",
        "Verdict logic:",
        "- `ship` if median Δ argmin − stdev > 0 (effect is clearly positive)",
        "- `noise` if |median Δ| < stdev (effect is within seed-to-seed variance)",
        "- `regress` otherwise (clearly negative)",
    ]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    sys.stderr.write("\n" + "\n".join(md[6:]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
