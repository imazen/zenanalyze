#!/usr/bin/env python3
"""
Multi-seed student-trainer comparison for `train_hybrid.py` (zenanalyze#68).

Drives `train_hybrid.py` once per (arm × seed) on ONE codec config — identical
data, identical origin even/odd split, identical hidden shape — and aggregates
`safety_report.diagnostics.argmin` (val: mean / p50 / p90 / p95 / p99 / max /
argmin_acc) across seeds as `mean ± stdev`, side by side per arm. The arms are
the trainer choices `--activation` exposes:

    relu               sklearn `MLPRegressor` (Glorot init, alpha=1e-4 L2, single thread)
    leakyrelu          PyTorch student (Kaiming init, no L2)
    leakyrelu+wd=1e-4  PyTorch student with Adam weight_decay=1e-4 (#68 hypothesis 2)

Arm syntax: `<activation>[+wd=<float>]`. Every arm's `student_backend` record
(what the bake actually ran) is echoed into the report so the numbers can never
be mis-attributed.

Usage (run from zentrain/tools with the codec config importable):

    PYTHONPATH=../examples:.:<zenmetrics>/scripts/picker \\
    python3 leakyrelu_seeds_runner.py \\
        --codec-config zenwebp_canonical_picker_config \\
        --hidden 128,128,128 --seeds 0xCAFE,0xBEEF,0xFACE \\
        --arms relu,leakyrelu,leakyrelu+wd=1e-4 \\
        --output-dir ~/tmp/backend-gap --report benchmarks/train_hybrid_backend_gap_<date>.md

`--reuse` skips an (arm, seed) whose model JSON already exists (so a run that
was produced by hand with the same `--out-suffix` convention,
`_{arm_slug}_seed_{seed:x}`, is picked up instead of re-trained — sklearn relu
runs are the expensive ones). Per-row val overhead TSVs land in `--output-dir`
for distribution plots; the markdown report is the committed artifact.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TRAIN_HYBRID = HERE / "train_hybrid.py"

_ARM = re.compile(r"^(?P<act>[a-z_]+)(?:\+wd=(?P<wd>[0-9.eE+-]+))?$")


def parse_seed(s: str) -> int:
    s = s.strip()
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def parse_arm(spec: str) -> dict:
    m = _ARM.match(spec.strip())
    if not m:
        raise SystemExit(f"bad arm {spec!r}: want <activation>[+wd=<float>]")
    wd = float(m.group("wd")) if m.group("wd") else 0.0
    slug = m.group("act") + (f"_wd{m.group('wd')}" if m.group("wd") else "")
    return {"spec": spec.strip(), "activation": m.group("act"), "weight_decay": wd, "slug": slug}


def out_json_for(codec_config: str, suffix: str) -> Path:
    """train_hybrid writes `OUT_JSON` with `--out-suffix` appended to the stem."""
    mod = importlib.import_module(codec_config)
    base = Path(mod.OUT_JSON)
    return base.with_name(base.stem + suffix + base.suffix)


def run_one(arm: dict, seed: int, codec_config: str, hidden: str, output_dir: Path, reuse: bool) -> dict:
    suffix = f"_{arm['slug']}_seed_{seed:x}"
    model_json = out_json_for(codec_config, suffix)
    overhead_tsv = output_dir / f"{arm['slug']}_seed{seed:#x}_overheads.tsv"
    if reuse and model_json.exists():
        sys.stderr.write(f"[reuse] {model_json}\n")
    else:
        cmd = [
            sys.executable, str(TRAIN_HYBRID),
            "--codec-config", codec_config,
            "--hidden", hidden,
            "--seed", str(seed),
            "--activation", arm["activation"],
            "--out-suffix", suffix,
            "--dump-overheads", str(overhead_tsv),
            "--allow-unsafe",
        ]
        if arm["weight_decay"]:
            cmd += ["--weight-decay", repr(arm["weight_decay"])]
        sys.stderr.write(f"\n[run] arm={arm['spec']} seed={seed:#x}\n  {' '.join(cmd)}\n")
        sys.stderr.flush()
        log = output_dir / f"{arm['slug']}_seed{seed:#x}.log"
        with open(log, "w") as lf:
            proc = subprocess.run(cmd, cwd=HERE, env=os.environ.copy(), stdout=lf, stderr=subprocess.STDOUT, text=True)
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"train_hybrid failed exit={proc.returncode}; see {log}")
        if not model_json.exists():
            raise RuntimeError(f"train_hybrid produced no {model_json}; see {log}")
    model = json.loads(model_json.read_text())
    diag = model.get("safety_report", {}).get("diagnostics", {})
    argmin = diag.get("argmin", {})
    return {
        "arm": arm["spec"],
        "slug": arm["slug"],
        "seed": seed,
        "json_path": str(model_json),
        "overhead_tsv": str(overhead_tsv) if overhead_tsv.exists() else None,
        "student_backend": model.get("student_backend") or diag.get("student_backend"),
        "teacher": diag.get("argmin", {}).get("teacher_val") or model.get("teacher_metrics"),
        "metrics": argmin.get("val", {}),
        "metrics_test": argmin.get("test", {}),
    }


FIELDS = ("mean_pct", "p50_pct", "p90_pct", "p95_pct", "p99_pct", "max_pct", "argmin_acc")


def aggregate(rows: list[dict], key: str = "metrics") -> dict:
    out: dict[str, dict] = {}
    for f in FIELDS:
        vals = [r[key].get(f) for r in rows if r[key].get(f) is not None]
        if not vals:
            continue
        out[f] = {
            "mean": statistics.fmean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
            "values": vals,
        }
    return out


def _cell(agg: dict, f: str) -> str:
    v = agg.get(f)
    if v is None:
        return "—"
    if f == "argmin_acc":
        return f"{v['mean'] * 100:.1f} ± {v['stdev'] * 100:.1f} pp"
    return f"{v['mean']:.2f} ± {v['stdev']:.2f} %"


def render(args, arms: list[dict], seeds: list[int], results: dict[str, list[dict]], agg: dict, agg_test: dict, today: str) -> str:
    L: list[str] = []
    L.append("# train_hybrid student trainers — multi-seed comparison (zenanalyze#68)")
    L.append("")
    L.append(
        f"Date: {today} · codec config `{args.codec_config}` · hidden `{args.hidden}` · "
        f"seeds {[hex(s) for s in seeds]} · arms {[a['spec'] for a in arms]}"
    )
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append(
        "Every arm runs the full production `train_hybrid.py` pipeline (HistGradientBoosting "
        "teacher per cell → MLP student on the teacher's soft targets, per-head normalisation "
        "on) on the same Pareto + features files, the same canonical origin even/odd split "
        "(`origin_split.py`) and the same hidden shape; only the student trainer differs. "
        "Metrics are the student's val argmin diagnostics vs the reachable per-row optimum, "
        "`mean ± stdev` over the seeds."
    )
    L.append("")
    L.append("| arm | backend | init | L2 |")
    L.append("|---|---|---|---|")
    for a in arms:
        rs = results[a["slug"]]
        sb = (rs[0].get("student_backend") or {}) if rs else {}
        l2 = sb.get("l2") or {}
        L.append(f"| `{a['spec']}` | {sb.get('backend', '?')} | {sb.get('init', '?')} | {l2.get('kind', '?')} = {l2.get('value', '?')} |")
    L.append("")
    L.append("## Headline numbers (val, mean ± stdev over seeds)")
    L.append("")
    L.append("| metric | " + " | ".join(f"`{a['spec']}`" for a in arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    labels = [("mean_pct", "mean overhead"), ("p50_pct", "p50 overhead"), ("p90_pct", "p90 overhead"),
              ("p95_pct", "p95 overhead"), ("p99_pct", "p99 overhead"), ("max_pct", "max overhead"),
              ("argmin_acc", "argmin accuracy")]
    for f, label in labels:
        L.append(f"| {label} | " + " | ".join(_cell(agg[a["slug"]], f) for a in arms) + " |")
    L.append("")
    if any(agg_test[a["slug"]] for a in arms):
        L.append("## Held-out test (7/9 origins), mean ± stdev over seeds")
        L.append("")
        L.append("| metric | " + " | ".join(f"`{a['spec']}`" for a in arms) + " |")
        L.append("|---|" + "---|" * len(arms))
        for f, label in labels:
            L.append(f"| {label} | " + " | ".join(_cell(agg_test[a["slug"]], f) for a in arms) + " |")
        L.append("")
    L.append("## Per-seed values (val)")
    L.append("")
    L.append("| arm | seed | mean | p50 | p90 | p95 | p99 | max | argmin_acc | model |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for a in arms:
        for r in results[a["slug"]]:
            m = r["metrics"]
            L.append(
                f"| `{a['spec']}` | {r['seed']:#x} | "
                + " | ".join(f"{m.get(f, float('nan')):.2f}%" for f in FIELDS[:-1])
                + f" | {m.get('argmin_acc', float('nan')):.1%} | `{Path(r['json_path']).name}` |"
            )
    L.append("")
    L.append("## Per-row overhead TSVs")
    L.append("")
    L.append("Columns `image, size_class, zq, pick, actual_best, overhead`; one per (arm, seed) under "
             f"`{args.output_dir}` (not committed).")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec-config", required=True)
    ap.add_argument("--hidden", default="192,192,192")
    ap.add_argument("--seeds", default="0xCAFE,0xBEEF,0xFACE", help="comma-separated seeds (decimal or 0x hex)")
    ap.add_argument("--arms", default="relu,leakyrelu,leakyrelu+wd=1e-4", help="comma-separated arm specs")
    ap.add_argument("--output-dir", required=True, help="per-run logs + per-row overhead TSVs")
    ap.add_argument("--report", default=None, help="markdown report path (default <output-dir>/comparison_report.md)")
    ap.add_argument("--reuse", action="store_true", help="skip (arm, seed) runs whose model JSON already exists")
    args = ap.parse_args()

    seeds = [parse_seed(s) for s in args.seeds.split(",") if s.strip()]
    arms = [parse_arm(a) for a in args.arms.split(",") if a.strip()]
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    results: dict[str, list[dict]] = {a["slug"]: [] for a in arms}
    for a in arms:
        for seed in seeds:
            r = run_one(a, seed, args.codec_config, args.hidden, out_dir, args.reuse)
            results[a["slug"]].append(r)
            m = r["metrics"]
            sys.stderr.write(
                f"  → {a['spec']} seed {seed:#x}: mean {m.get('mean_pct', float('nan')):.2f}%  "
                f"p99 {m.get('p99_pct', float('nan')):.2f}%  acc {m.get('argmin_acc', 0):.1%}\n"
            )
    agg = {slug: aggregate(rs) for slug, rs in results.items()}
    agg_test = {slug: aggregate(rs, "metrics_test") for slug, rs in results.items()}
    (out_dir / "raw.json").write_text(json.dumps(results, indent=2, default=str))
    (out_dir / "aggregate.json").write_text(json.dumps({"val": agg, "test": agg_test}, indent=2))
    report = render(args, arms, seeds, results, agg, agg_test, today)
    report_path = Path(args.report).expanduser() if args.report else out_dir / "comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    sys.stdout.write(report)
    sys.stderr.write(f"\nwrote {report_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
