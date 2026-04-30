#!/usr/bin/env python3
"""
Multi-seed LeakyReLU vs ReLU comparison — distilled, production-equivalent.

Drives `train_hybrid.py` N times per activation (default 3 seeds) using the
full HistGradientBoostingRegressor teacher + MLP student pipeline. Aggregates
safety_report.diagnostics.argmin (mean / p50 / p90 / p95 / p99 / max /
argmin_acc) across seeds and reports mean ± std side-by-side.

Per-seed per-row val overhead distributions are written to TSV for future
violin / KDE plots.

Outputs:
  - /mnt/v/output/zenpicker/leakyrelu_seeds_<date>/relu_seed*_overheads.tsv
  - /mnt/v/output/zenpicker/leakyrelu_seeds_<date>/leakyrelu_seed*_overheads.tsv
  - /mnt/v/output/zenpicker/leakyrelu_seeds_<date>/comparison_report.md

Usage:
    python3 tools/leakyrelu_seeds_runner.py \
        --codec-config zenjpeg_picker_config \
        --hidden 192,192,192 \
        --seeds 0xCAFE,0xBEEF,0xFACE
"""

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # zenanalyze worktree root
ZENJPEG_BENCH = Path("/home/lilith/work/zen/zenjpeg/benchmarks")
TRAIN_HYBRID = REPO_ROOT / "zenpicker" / "tools" / "train_hybrid.py"


def parse_seed(s: str) -> int:
    s = s.strip()
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def run_one(
    seed: int,
    activation: str,
    codec_config: str,
    hidden: str,
    output_dir: Path,
    suffix: str,
) -> dict:
    """Runs a single train_hybrid invocation. Returns parsed
    safety_report.diagnostics.argmin.val plus the path to the
    per-row TSV."""
    overhead_tsv = output_dir / f"{activation}_seed{seed:#x}_overheads.tsv"
    env = os.environ.copy()
    # Force into zenjpeg/benchmarks/ so relative TSV paths resolve.
    env["PYTHONPATH"] = str(REPO_ROOT / "zenpicker" / "examples") + ":" + env.get(
        "PYTHONPATH", ""
    )
    cmd = [
        sys.executable,
        str(TRAIN_HYBRID),
        "--codec-config",
        codec_config,
        "--hidden",
        hidden,
        "--seed",
        str(seed),
        "--activation",
        activation,
        "--out-suffix",
        suffix,
        "--dump-overheads",
        str(overhead_tsv),
        "--allow-unsafe",
    ]
    sys.stderr.write(f"\n[run] activation={activation} seed={seed:#x}\n  {' '.join(cmd)}\n")
    sys.stderr.flush()
    proc = subprocess.run(
        cmd,
        cwd=ZENJPEG_BENCH.parent,  # zenjpeg/ — TSVs at zenjpeg/benchmarks/ resolve
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        # 1 just means safety violations triggered --strict (we passed --allow-unsafe so that's fine);
        # anything else is a real error
        sys.stderr.write(proc.stderr[-2000:])
        raise RuntimeError(f"train_hybrid failed exit={proc.returncode}")

    # Find the produced JSON. train_hybrid writes to OUT_JSON with a suffix.
    # The codec config's OUT_JSON path is in zenjpeg/benchmarks/, with
    # suffix appended before .json.
    base = ZENJPEG_BENCH / f"zq_bytes_hybrid_v2_1{suffix}.json"
    if not base.exists():
        # Fall back: glob for it
        candidates = sorted(ZENJPEG_BENCH.glob(f"zq_bytes_hybrid_v2_1*{suffix}.json"))
        if not candidates:
            raise RuntimeError(f"could not find output JSON; tried {base}")
        base = candidates[-1]
    model = json.loads(base.read_text())
    sr = model.get("safety_report", {})
    diag = sr.get("diagnostics", {})
    argmin_val = diag.get("argmin", {}).get("val", {})
    return {
        "activation": activation,
        "seed": seed,
        "json_path": str(base),
        "overhead_tsv": str(overhead_tsv) if overhead_tsv.exists() else None,
        "metrics": argmin_val,
    }


def aggregate(rows: list[dict]) -> dict:
    """rows: list of per-seed result dicts for one activation."""
    fields = ("mean_pct", "p50_pct", "p90_pct", "p95_pct", "p99_pct", "max_pct", "argmin_acc")
    out: dict[str, dict] = {}
    for f in fields:
        vals = [r["metrics"].get(f) for r in rows if r["metrics"].get(f) is not None]
        if not vals:
            continue
        out[f] = {
            "mean": statistics.fmean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "values": vals,
        }
    return out


def fmt(v: float, prec: int = 2) -> str:
    return f"{v:.{prec}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--hidden", default="192,192,192")
    ap.add_argument(
        "--seeds",
        default="0xCAFE,0xBEEF,0xFACE",
        help="Comma-separated seeds (decimal or 0x-prefixed hex).",
    )
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    seeds = [parse_seed(s) for s in args.seeds.split(",")]
    sys.stderr.write(f"seeds: {[hex(s) for s in seeds]}\n")

    today = datetime.date.today().isoformat()
    out_dir = Path(args.output_dir or f"/mnt/v/output/zenpicker/leakyrelu_seeds_{today}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {"relu": [], "leakyrelu": []}
    for activation in ("relu", "leakyrelu"):
        for seed in seeds:
            suffix = f"_act_{activation}_seed_{seed:x}"
            r = run_one(
                seed=seed,
                activation=activation,
                codec_config=args.codec_config,
                hidden=args.hidden,
                output_dir=out_dir,
                suffix=suffix,
            )
            results[activation].append(r)
            m = r["metrics"]
            sys.stderr.write(
                f"  → mean {m.get('mean_pct', float('nan')):.2f}%  "
                f"p95 {m.get('p95_pct', float('nan')):.2f}%  "
                f"p99 {m.get('p99_pct', float('nan')):.2f}%  "
                f"acc {m.get('argmin_acc', 0):.1%}\n"
            )

    agg = {act: aggregate(results[act]) for act in results}
    raw = {act: results[act] for act in results}
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=str))
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    # Markdown report.
    lines: list[str] = []
    lines.append(f"# LeakyReLU vs ReLU — multi-seed distilled comparison\n")
    lines.append(f"Date: {today}  •  hidden={args.hidden}  •  seeds={[hex(s) for s in seeds]}\n")
    lines.append("\n## Methodology\n")
    lines.append(
        "Both arms use the full production train_hybrid.py pipeline:\n"
        "HistGradientBoostingRegressor teacher per cell, then MLP student\n"
        "trained on the teacher's soft targets. The only delta is hidden\n"
        "activation (sklearn ReLU vs PyTorch LeakyReLU(0.01)). Same data,\n"
        "same image-level holdout, same hidden sizes. Each metric reported\n"
        "as `mean ± stdev` across the seeds listed above.\n"
    )
    lines.append("\n## Headline numbers (val)\n")
    lines.append("| Metric | ReLU | LeakyReLU | Δ (mean) |\n|---|---|---|---:|\n")
    for f, label in [
        ("mean_pct", "Mean overhead"),
        ("p50_pct", "p50 overhead"),
        ("p90_pct", "p90 overhead"),
        ("p95_pct", "p95 overhead"),
        ("p99_pct", "p99 overhead"),
        ("max_pct", "max overhead"),
        ("argmin_acc", "argmin accuracy"),
    ]:
        r = agg["relu"].get(f)
        l = agg["leakyrelu"].get(f)
        if r is None or l is None:
            continue
        unit = "%" if f.endswith("_pct") else ""
        scale = 1 if f.endswith("_pct") else 100
        prec = 2 if f.endswith("_pct") else 2
        if f == "argmin_acc":
            unit = "pp"
        delta = (l["mean"] - r["mean"]) * (1 if f.endswith("_pct") else 100)
        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {label} | {fmt(r['mean'] * scale, prec)}{unit} ± {fmt(r['stdev'] * scale, prec)} | "
            f"{fmt(l['mean'] * scale, prec)}{unit} ± {fmt(l['stdev'] * scale, prec)} | "
            f"{sign}{fmt(delta, prec)}{unit} |\n"
        )

    lines.append("\n## Per-seed values\n")
    lines.append("| Activation | Seed | Mean | p50 | p90 | p95 | p99 | max | argmin_acc |\n|---|---|---|---|---|---|---|---|---|\n")
    for activation in ("relu", "leakyrelu"):
        for r in results[activation]:
            m = r["metrics"]
            lines.append(
                f"| {activation} | {r['seed']:#x} | "
                f"{m.get('mean_pct', float('nan')):.2f}% | "
                f"{m.get('p50_pct', float('nan')):.2f}% | "
                f"{m.get('p90_pct', float('nan')):.2f}% | "
                f"{m.get('p95_pct', float('nan')):.2f}% | "
                f"{m.get('p99_pct', float('nan')):.2f}% | "
                f"{m.get('max_pct', float('nan')):.2f}% | "
                f"{m.get('argmin_acc', float('nan')):.1%} |\n"
            )

    lines.append("\n## Per-row overhead TSVs\n")
    lines.append("Each TSV has columns: `image, size_class, zq, pick, actual_best, overhead`. Feed into `seaborn.violinplot(x='activation', y='overhead', data=df)` for distribution plots.\n\n")
    for activation in ("relu", "leakyrelu"):
        for r in results[activation]:
            if r["overhead_tsv"]:
                lines.append(f"- `{r['overhead_tsv']}`\n")

    report_path = out_dir / "comparison_report.md"
    report_path.write_text("".join(lines))
    print("".join(lines))
    sys.stderr.write(f"\nwrote {report_path}\n")


if __name__ == "__main__":
    sys.exit(main() or 0)
