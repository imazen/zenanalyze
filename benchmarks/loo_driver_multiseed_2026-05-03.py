"""Cross-codec multi-seed LOO retrain driver, 2026-05-03.

Generalizes `loo_driver_multiseed_2026-05-02.py` from a single zenwebp
hardcoded run into a `--codec-config` driven sweep that works for any
of the 4 zen codecs (zenwebp / zenjpeg / zenavif / zenjxl).

Per-codec mode: load the picker config module, take its
`KEEP_FEATURES` as both the base set AND the candidate set (full LOO
over the active feature roster), then for each candidate run paired
with/without retrains across N seeds against the codec's full Pareto
TSV. Aggregates per-feature mean ± σ ΔOH and ΔAC.

This is *analysis only*. The codec configs are not modified. The
generated per-experiment configs live in /tmp/loo_multiseed_2026-05-03/
and override PARETO/FEATURES/OUT_JSON/OUT_LOG to absolute paths so the
trainer doesn't depend on cwd.

Usage:
    python3 benchmarks/loo_driver_multiseed_2026-05-03.py \
        --codec zenwebp --jobs 8

Outputs (per codec):
    benchmarks/loo_<codec>_multiseed_2026-05-03.tsv      summary
    benchmarks/loo_<codec>_multiseed_raw_2026-05-03.tsv  per-seed raw

Exit code 0 on success, non-zero if any feature failed all seeds.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ZA_ROOT = Path("/home/lilith/work/zen/zenanalyze")
TRAIN = ZA_ROOT / "zentrain/tools/train_hybrid.py"
EXAMPLES = ZA_ROOT / "zentrain/examples"
OUT_DIR = Path("/tmp/loo_multiseed_2026-05-03")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [51966, 0xC0DEC0DE, 0xBEEFCAFE, 0xFEEDFACE, 0xDEADBEEF]

CODECS = {
    "zenwebp": {
        "config_module": "zenwebp_picker_config",
        "cwd": Path("/home/lilith/work/zen/zenwebp"),
        "feature_override": None,
    },
    "zenjpeg": {
        "config_module": "zenjpeg_picker_config",
        "cwd": Path("/home/lilith/work/zen/zenjpeg"),
        # Picker config points at a non-existent _2026-05-01.tsv; the
        # full-schema features TSV that actually exists is _parallel.tsv.
        "feature_override": Path(
            "/home/lilith/work/zen/zenjpeg/benchmarks/zq_pareto_features_2026-05-01_parallel.tsv"
        ),
    },
    "zenavif": {
        "config_module": "zenavif_picker_config",
        "cwd": Path("/home/lilith/work/zen/zenavif"),
        "feature_override": None,
    },
    "zenjxl": {
        "config_module": "zenjxl_picker_config",
        "cwd": Path("/home/lilith/work/zen/zenjxl"),
        "feature_override": None,
    },
}


_STUDENT_RE = re.compile(
    r"Student:\s+mean\s+([0-9.]+)%\s+argmin_acc\s+([0-9.]+)%"
)


def load_codec_config(name: str):
    """Import codec config and resolve its declared paths to absolute."""
    sys.path.insert(0, str(EXAMPLES))
    cwd_save = os.getcwd()
    os.chdir(str(CODECS[name]["cwd"]))
    try:
        mod = importlib.import_module(CODECS[name]["config_module"])
        # Snapshot the static export values; resolve to absolute
        # against the codec's repo root so generated configs are
        # cwd-independent.
        cwd = CODECS[name]["cwd"]
        pareto = Path(mod.PARETO)
        if not pareto.is_absolute():
            pareto = (cwd / pareto).resolve()
        feat_override = CODECS[name]["feature_override"]
        if feat_override is not None:
            features = feat_override
        else:
            features = Path(mod.FEATURES)
            if not features.is_absolute():
                features = (cwd / features).resolve()
        return {
            "pareto": pareto,
            "features": features,
            "keep_features": list(mod.KEEP_FEATURES),
            "module": mod,
        }
    finally:
        os.chdir(cwd_save)
        # Don't remove from sys.path — module is cached.


CONFIG_TEMPLATE = '''# Auto-generated multi-seed LOO config. DO NOT EDIT.
# Inherits everything else from {base_module} via re-export, then
# overrides PARETO/FEATURES/KEEP_FEATURES/OUT_JSON/OUT_LOG.
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, {examples_dir!r})

from {base_module} import *  # noqa: F401,F403

PARETO = Path({pareto!r})
FEATURES = Path({features!r})
OUT_JSON = Path({out_json!r})
OUT_LOG = Path({out_log!r})

KEEP_FEATURES = {keep_features!r}
'''


def write_config(codec: str, name: str, keep: list[str], pareto: Path, features: Path) -> Path:
    cfg_path = OUT_DIR / f"{codec}__{name}.py"
    out_json = OUT_DIR / f"{codec}__{name}_model.json"
    out_log = OUT_DIR / f"{codec}__{name}_train.log"
    body = CONFIG_TEMPLATE.format(
        base_module=CODECS[codec]["config_module"],
        examples_dir=str(EXAMPLES),
        pareto=str(pareto),
        features=str(features),
        out_json=str(out_json),
        out_log=str(out_log),
        keep_features=keep,
    )
    cfg_path.write_text(body)
    return cfg_path


def run_train(codec: str, config_name: str, seed: int, log_path: Path) -> tuple[float | None, float | None]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OUT_DIR) + os.pathsep + str(EXAMPLES) + os.pathsep + env.get("PYTHONPATH", "")
    # Pin loky/joblib worker count per-training to avoid OOM when N
    # trainings run in parallel on a single box. Each training spawns
    # a `joblib.Parallel(n_jobs=-1, backend="loky")` pool for teacher
    # fits; without this cap, N trainings × all-cores-each blows out
    # RAM. LOO_LOKY_CPUS env var lets the user override (default 2).
    env["LOKY_MAX_CPU_COUNT"] = env.get("LOO_LOKY_CPUS", "2")
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
            cwd=str(CODECS[codec]["cwd"]),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    if result.returncode != 0:
        sys.stderr.write(f"  [FAIL] {config_name} seed={seed} rc={result.returncode} ({elapsed:.0f}s)\n")
        return None, None
    text = log_path.read_text()
    m = _STUDENT_RE.search(text)
    if not m:
        sys.stderr.write(f"  [PARSE FAIL] {config_name} seed={seed} ({elapsed:.0f}s)\n")
        return None, None
    return float(m.group(1)), float(m.group(2))


def slug(feat: str) -> str:
    return feat.replace("feat_", "")


def run_codec(codec: str, seeds: list[int], jobs: int, only_features: list[str] | None = None) -> int:
    cfg = load_codec_config(codec)
    pareto = cfg["pareto"]
    features = cfg["features"]
    base = cfg["keep_features"]

    if only_features is None:
        candidates = list(base)
    else:
        candidates = [f for f in only_features if f in base]
        if not candidates:
            sys.stderr.write(f"[{codec}] No requested features in active KEEP_FEATURES, aborting\n")
            return 1

    sys.stderr.write(
        f"\n[{codec}] === LOO multiseed sweep ===\n"
        f"  pareto:     {pareto}\n"
        f"  features:   {features}\n"
        f"  KEEP_FEATURES base size: {len(base)}\n"
        f"  candidates: {len(candidates)} ({'subset' if only_features else 'full'})\n"
        f"  seeds: {seeds} ({len(seeds)} seeds)\n"
        f"  parallel jobs: {jobs}\n"
        f"  total trainings: {len(candidates) * len(seeds) * 2}\n"
    )

    # Pre-write all configs so we can dispatch in parallel.
    config_specs = []  # list of (feat, seed, label, cfg_name, log_path)
    for feat in candidates:
        s = slug(feat)
        keep_with = list(base) if feat in base else list(base) + [feat]
        keep_without = [f for f in base if f != feat]
        cfg_with = write_config(codec, f"with_{s}", keep_with, pareto, features)
        cfg_without = write_config(codec, f"without_{s}", keep_without, pareto, features)
        for seed in seeds:
            log_with = OUT_DIR / f"{codec}__with_{s}_seed{seed}.log"
            log_without = OUT_DIR / f"{codec}__without_{s}_seed{seed}.log"
            config_specs.append((feat, seed, "with", cfg_with.stem, log_with))
            config_specs.append((feat, seed, "without", cfg_without.stem, log_without))

    # results[(feat, seed)] = {"with": (m, ac), "without": (m, ac)}
    results: dict[tuple[str, int], dict[str, tuple[float, float] | None]] = {}

    started_all = time.time()
    completed_count = 0
    total = len(config_specs)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {}
        for feat, seed, label, cfg_name, log_path in config_specs:
            fut = ex.submit(run_train, codec, cfg_name, seed, log_path)
            futures[fut] = (feat, seed, label)

        for fut in as_completed(futures):
            feat, seed, label = futures[fut]
            m, ac = fut.result()
            key = (feat, seed)
            if key not in results:
                results[key] = {}
            results[key][label] = (m, ac) if m is not None else None
            completed_count += 1
            if completed_count % max(1, jobs) == 0 or completed_count == total:
                elapsed = time.time() - started_all
                rate = completed_count / max(1.0, elapsed)
                eta = (total - completed_count) / rate if rate > 0 else 0
                sys.stderr.write(
                    f"  [{codec}] {completed_count}/{total} "
                    f"({elapsed:.0f}s, ETA {eta:.0f}s)\n"
                )

    # Aggregate and write outputs.
    raw_rows = []
    summary_rows = []
    baseline_argmin: list[float] = []  # `with` argmins across all candidates+seeds

    for feat in candidates:
        per_seed = []
        for seed in seeds:
            entry = results.get((feat, seed), {})
            w = entry.get("with")
            o = entry.get("without")
            if w is None or o is None:
                continue
            wm, wa = w
            om, oa = o
            d_oh = om - wm
            d_ac = oa - wa
            raw_rows.append([feat, seed, wm, om, d_oh, wa, oa, d_ac])
            per_seed.append((d_oh, d_ac))
            baseline_argmin.append(wa)

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

    # Write TSVs
    raw_tsv = ZA_ROOT / "benchmarks" / f"loo_{codec}_multiseed_raw_2026-05-03.tsv"
    summary_tsv = ZA_ROOT / "benchmarks" / f"loo_{codec}_multiseed_2026-05-03.tsv"

    with open(raw_tsv, "w") as f:
        f.write("\t".join([
            "feature", "seed",
            "with_overhead_pct", "without_overhead_pct", "delta_overhead_pp",
            "with_argmin_pct", "without_argmin_pct", "delta_argmin_pp",
        ]) + "\n")
        for r in raw_rows:
            vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
            f.write("\t".join(vals) + "\n")

    cols = [
        "feature", "n_seeds_ok",
        "mean_delta_overhead_pp", "stddev_delta_overhead_pp",
        "mean_delta_argmin_pp", "stddev_delta_argmin_pp",
    ]
    with open(summary_tsv, "w") as f:
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

    elapsed = time.time() - started_all
    if baseline_argmin:
        amin = min(baseline_argmin)
        amax = max(baseline_argmin)
        amean = statistics.mean(baseline_argmin)
        sys.stderr.write(
            f"\n[{codec}] DONE in {elapsed:.0f}s ({elapsed/60:.1f} min).\n"
            f"  baseline argmin (with-feature) range: {amin:.1f}% .. {amax:.1f}% (mean {amean:.1f}%)\n"
            f"  raw TSV:     {raw_tsv}\n"
            f"  summary TSV: {summary_tsv}\n"
        )
    else:
        sys.stderr.write(f"\n[{codec}] FAILED — no successful trainings.\n")
        return 1
    return 0


def refresh_workongoing():
    marker = ZA_ROOT / ".workongoing"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    marker.write_text(f"{ts} claude-multiseed-loo loo across 4 codecs\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codec",
        choices=list(CODECS.keys()) + ["all"],
        required=True,
        help="Which codec to sweep (or 'all').",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(hex(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (decimal or 0x-hex).",
    )
    parser.add_argument(
        "--jobs", type=int, default=8,
        help="Parallel training jobs.",
    )
    parser.add_argument(
        "--only-features", type=str, default=None,
        help="Comma-separated feature subset (smoke test).",
    )
    args = parser.parse_args()

    seeds = [int(s, 0) for s in args.seeds.split(",")]
    only = args.only_features.split(",") if args.only_features else None

    refresh_workongoing()

    targets = list(CODECS.keys()) if args.codec == "all" else [args.codec]
    rc = 0
    for codec in targets:
        refresh_workongoing()
        r = run_codec(codec, seeds, args.jobs, only)
        if r != 0:
            rc = r
        refresh_workongoing()
    return rc


if __name__ == "__main__":
    sys.exit(main())
