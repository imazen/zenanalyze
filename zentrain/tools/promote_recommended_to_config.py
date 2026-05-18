#!/usr/bin/env python3
"""Generate a v2 codec config that adopts the sweep's recommended transforms.

Reads a `recommended_transforms.py` (output of feature_transform_sweep.py)
and a base codec config name, emits a v2 config module that wraps the
base + overrides FEATURE_TRANSFORMS / FEATURE_TRANSFORM_PARAMS.

## Multi-seed lock (2026-05-17+)

The single-seed `feature_transform_sweep.py --confirm` deltas were
demonstrated to be unreliable (zenjpeg: +3.74 pp single-seed →
−6.81 pp 3-seed median; zenavif: +2.57 pp → +1.65 pp noise). To
prevent that failure class from leaking into production v2 configs,
this script now requires a `--multiseed-aggregate <path>` pointing
at a `multi_seed_confirm.py` `aggregate.json` and refuses to write
unless the verdict is `ship`.

Pass `--force-regress` / `--force-noise` to override per failure
class — both flags annotate the generated config's docstring with
a prominent WARNING banner so the override is visible to anyone
who later opens the file.

Usage:

    # Normal: ship verdict required
    python3 zentrain/tools/promote_recommended_to_config.py \\
        --recommended benchmarks/multiseed_zenwebp_v14_2026-05-17/seed_0xcafe/recommended_transforms.py \\
        --multiseed-aggregate benchmarks/multiseed_zenwebp_v14_2026-05-17/aggregate.json \\
        --base zenwebp_picker_config \\
        --out zentrain/examples/zenwebp_picker_config_v2.py \\
        --label "v14+z_rmse"

    # Override for diagnostic / reproducibility artifacts (rare):
    python3 zentrain/tools/promote_recommended_to_config.py \\
        ... --force-noise   # for verdict=noise
        ... --force-regress # for verdict=regress
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _parse_recommended(path: Path) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Read the recommended file and extract its top-level dict literals."""
    tree = ast.parse(path.read_text())
    transforms: dict[str, str] = {}
    params: dict[str, list[float]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "FEATURE_TRANSFORMS":
                transforms = ast.literal_eval(node.value)
            elif name == "FEATURE_TRANSFORM_PARAMS":
                params = ast.literal_eval(node.value)
    return transforms, params


def _format_dict(d: dict[str, Any], value_fmt) -> str:
    if not d:
        return "{}"
    lines = ["{"]
    for k, v in d.items():
        lines.append(f"    {k!r}: {value_fmt(v)},")
    lines.append("}")
    return "\n".join(lines)


def _check_multiseed(
    aggregate_path: Path,
    force_regress: bool,
    force_noise: bool,
) -> dict:
    """Load the multi_seed_confirm aggregate.json and gate promotion
    on `verdict`. Returns the loaded aggregate dict (so the caller
    can attach the actual stats to the generated docstring)."""
    if not aggregate_path.exists():
        raise SystemExit(
            f"--multiseed-aggregate {aggregate_path} not found. Run "
            f"`multi_seed_confirm.py` before promoting to a v2 config — "
            f"single-seed --confirm results are not sufficient (zenjpeg "
            f"and zenavif both invalidated single-seed wins via 3-seed "
            f"runs in benchmarks/multiseed_*_v14_2026-05-17/)."
        )
    agg = json.loads(aggregate_path.read_text())
    verdict = agg.get("verdict")
    if verdict is None:
        raise SystemExit(
            f"aggregate.json at {aggregate_path} has no `verdict` field; "
            f"likely written by an older multi_seed_confirm.py. Re-run."
        )
    if verdict == "ship":
        return agg
    if verdict == "regress" and force_regress:
        return agg
    if verdict == "noise" and force_noise:
        return agg
    # Refuse.
    n_seeds = agg.get("n_seeds", "?")
    median = agg.get("delta_argmin_pp_median", "?")
    stdev = agg.get("delta_argmin_pp_stdev", "?")
    flag_for_verdict = {"regress": "--force-regress", "noise": "--force-noise"}
    flag_hint = flag_for_verdict.get(verdict, "(no override for this verdict)")
    raise SystemExit(
        f"REFUSING TO PROMOTE: multi-seed verdict is {verdict!r}.\n"
        f"  aggregate: {aggregate_path}\n"
        f"  n_seeds: {n_seeds}\n"
        f"  delta_argmin median: {median} pp\n"
        f"  delta_argmin stdev:  {stdev} pp\n"
        f"\n"
        f"This is the failure class that hit zenjpeg + zenavif on\n"
        f"2026-05-17: single-seed --confirm said ship; 3-seed median\n"
        f"said regress/noise. Refusing prevents shipping a transform\n"
        f"set that doesn't survive seed variance.\n"
        f"\n"
        f"If you need this artifact for diagnostic / reproducibility\n"
        f"reasons, pass `{flag_hint}` — the generated config will\n"
        f"carry a prominent WARNING banner so reviewers see the\n"
        f"override at a glance."
    )


def _format_aggregate_banner(agg: dict, verdict: str) -> str:
    """Compose the WARNING banner inserted into a force-promoted
    config's docstring. Includes the verdict + key numbers so anyone
    grepping the file can see immediately why it shouldn't ship."""
    n = agg.get("n_seeds", "?")
    med = agg.get("delta_argmin_pp_median")
    stdev = agg.get("delta_argmin_pp_stdev")
    range_min = agg.get("delta_argmin_pp_min")
    range_max = agg.get("delta_argmin_pp_max")
    med_s = f"{med:+.2f} pp" if isinstance(med, (int, float)) else str(med)
    stdev_s = f"{stdev:.2f} pp" if isinstance(stdev, (int, float)) else str(stdev)
    range_s = (
        f"{range_min:+.2f}..{range_max:+.2f}"
        if isinstance(range_min, (int, float)) and isinstance(range_max, (int, float))
        else f"{range_min}..{range_max}"
    )
    return (
        f"!!! DO NOT BAKE INTO PRODUCTION !!!\n\n"
        f"This config was force-promoted with `--force-{verdict}`. The\n"
        f"multi-seed lock verdict was **{verdict}**, not `ship`:\n"
        f"  n_seeds: {n}\n"
        f"  delta_argmin pp (median): {med_s}\n"
        f"  delta_argmin pp (stdev):  {stdev_s}\n"
        f"  delta_argmin pp (range):  {range_s}\n\n"
        f"Kept only as a reproducibility / diagnostic artifact. To\n"
        f"un-force-promote: re-run feature_transform_sweep.py with a\n"
        f"different methodology (more seeds, pruned features, etc.)\n"
        f"and ship that result instead.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recommended", type=Path, required=True)
    ap.add_argument("--multiseed-aggregate", type=Path, required=True,
                    help="aggregate.json from multi_seed_confirm.py. The "
                    "promotion is gated on its `verdict` field being "
                    "`ship`; pass --force-regress or --force-noise to "
                    "override (with a WARNING banner in the output).")
    ap.add_argument("--base", required=True,
                    help="Base codec config module name (no .py suffix)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output v2 config path")
    ap.add_argument("--label", default="",
                    help="Optional verdict label included in the docstring")
    ap.add_argument("--out-suffix", default="v2_2026-05-17",
                    help="Filename suffix for OUT_JSON / OUT_LOG paths")
    ap.add_argument("--force-regress", action="store_true",
                    help="Override the gate when verdict == 'regress'. "
                    "Inserts a prominent WARNING banner in the generated "
                    "config's docstring.")
    ap.add_argument("--force-noise", action="store_true",
                    help="Override the gate when verdict == 'noise'. "
                    "Inserts a prominent WARNING banner.")
    args = ap.parse_args()

    agg = _check_multiseed(
        args.multiseed_aggregate, args.force_regress, args.force_noise,
    )
    verdict = agg.get("verdict", "?")

    transforms, params = _parse_recommended(args.recommended)
    if not transforms:
        raise SystemExit(f"no FEATURE_TRANSFORMS dict in {args.recommended}")

    # Strip the codec prefix to derive a clean module name for the
    # OUT_JSON / OUT_LOG (e.g. zenwebp_picker_config -> zenwebp_hybrid).
    base = args.base
    codec_short = base.replace("_picker_config", "")

    # Detect SCALAR_AXES / SCALAR_SENTINELS / ... imports by trying each;
    # missing ones are skipped via the `# noqa: F401` line.
    common_exports = [
        "PARETO", "FEATURES",
        "ZQ_TARGETS", "CATEGORICAL_AXES", "SCALAR_AXES",
        "SCALAR_SENTINELS", "SCALAR_DISPLAY_RANGES",
        "FEATURE_GROUPS", "TIME_COLUMN",
        "OUTPUT_SPECS", "SPARSE_OVERRIDES",
        "parse_config_name", "KEEP_FEATURES",
    ]
    # Read the base config to verify what's exported.
    base_path = args.out.parent / f"{base}.py"
    if not base_path.exists():
        raise SystemExit(f"base config not found: {base_path}")
    base_src = base_path.read_text()
    actually_exported = [name for name in common_exports if (
        f"\n{name} =" in base_src or f"\n{name}:" in base_src
        or f"\ndef {name}" in base_src
    )]
    if "KEEP_FEATURES" not in actually_exported:
        # Always need KEEP_FEATURES.
        actually_exported.append("KEEP_FEATURES")
    import_list = ",\n    ".join(actually_exported)

    label_line = f" ({args.label})" if args.label else ""

    def fmt_str(v):
        return repr(v)

    def fmt_params(v):
        return repr(list(v))

    transforms_block = _format_dict(transforms, fmt_str)
    params_block = _format_dict(params, fmt_params)

    # Multi-seed verdict block. For a `ship` verdict we just record
    # the lock numbers; for force-promoted (regress / noise) we lead
    # with the WARNING banner so a reader sees it immediately.
    if verdict == "ship":
        verdict_block = (
            f"Multi-seed verdict: **ship** "
            f"(n_seeds={agg.get('n_seeds', '?')}, "
            f"delta_argmin pp median {agg.get('delta_argmin_pp_median', '?')}, "
            f"stdev {agg.get('delta_argmin_pp_stdev', '?')}).\n"
            f"Aggregate: {args.multiseed_aggregate}\n"
        )
    else:
        verdict_block = (
            _format_aggregate_banner(agg, verdict)
            + f"\nAggregate: {args.multiseed_aggregate}\n"
        )

    out_text = f'''"""\
{codec_short} picker config v2 — adopts the 2026-05-17 sweep winners{label_line}.

{verdict_block}

Wraps {base} + overrides FEATURE_TRANSFORMS / FEATURE_TRANSFORM_PARAMS
with the per-feature winners from the feature_transform_sweep.py
recommendation at:

  {args.recommended}

Generated by zentrain/tools/promote_recommended_to_config.py.
"""

from __future__ import annotations

from pathlib import Path

from {base} import (  # noqa: F401
    {import_list},
)

OUT_JSON = Path("benchmarks/{codec_short}_hybrid_{args.out_suffix}.json")
OUT_LOG = Path("benchmarks/{codec_short}_hybrid_{args.out_suffix}.log")

FEATURE_TRANSFORMS = {transforms_block}

FEATURE_TRANSFORM_PARAMS = {params_block}
'''
    args.out.write_text(out_text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
