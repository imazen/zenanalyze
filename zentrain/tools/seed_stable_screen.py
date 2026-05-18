#!/usr/bin/env python3
"""Build a seed-stable transform recommendation by majority-voting
across N already-run sweep seed directories.

Background: `feature_transform_sweep.py --confirm` produces a single-
seed recommendation that can be invalidated by `multi_seed_confirm.py`
(e.g. zenjpeg's v14+z_rmse: single-seed +3.74 pp → 3-seed -6.81 pp
median). Cause: the screen step picks the highest-scoring transform
per feature on a single train/val split; with thin data per cell
(zenavif: 1.1 val rows/config), different seeds rank different
transforms first, and the picked transform overfits to the seed's
particular split.

This tool reads N already-run seed sweep directories (each containing
a `recommended_transforms.py`) and emits a majority-vote
`recommended_transforms.py`. A feature is included iff a strict
majority of seeds (default: >50%) agreed on the same transform; if
no clear majority exists, the feature is dropped (i.e. left at
identity). This yields a smaller but more reliable recommendation
set.

Then run `feature_transform_sweep.py --confirm` (or
`multi_seed_confirm.py`) on a codec config that adopts the majority-
vote recommendations.

Usage:

    python3 zentrain/tools/seed_stable_screen.py \\
        --seed-dirs benchmarks/multiseed_zenjpeg_v14_2026-05-17/seed_0xcafe \\
                    benchmarks/multiseed_zenjpeg_v14_2026-05-17/seed_0xbeef \\
                    benchmarks/multiseed_zenjpeg_v14_2026-05-17/seed_0xface \\
        --out benchmarks/seed_stable_zenjpeg_2026-05-17/recommended_transforms.py
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path


def _read_dict(path: Path, name: str) -> dict:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dirs", nargs="+", type=Path, required=True,
                    help="Per-seed sweep output directories (each has "
                    "recommended_transforms.py)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output recommended_transforms.py path")
    ap.add_argument("--min-agreement", type=float, default=0.51,
                    help="Minimum fraction of seeds that must agree on "
                    "the same transform (default 0.51 = strict majority)")
    ap.add_argument("--include-params", action="store_true",
                    help="Also pick params from the seed whose transform "
                    "won the majority vote (averaged across agreeing "
                    "seeds for numeric params)")
    args = ap.parse_args()

    n_seeds = len(args.seed_dirs)
    per_seed_transforms = []
    per_seed_params = []
    for sd in args.seed_dirs:
        rec = sd / "recommended_transforms.py"
        t = _read_dict(rec, "FEATURE_TRANSFORMS")
        p = _read_dict(rec, "FEATURE_TRANSFORM_PARAMS")
        per_seed_transforms.append(t)
        per_seed_params.append(p)
        if not t:
            print(f"  WARN: no FEATURE_TRANSFORMS in {rec}")

    # Union of all feature names recommended in any seed.
    all_feats = set()
    for t in per_seed_transforms:
        all_feats.update(t.keys())

    # Majority vote per feature.
    stable_transforms: dict[str, str] = {}
    stable_params: dict[str, list[float]] = {}
    n_dropped_minority = 0
    n_dropped_coverage = 0
    for feat in sorted(all_feats):
        votes = [t.get(feat) for t in per_seed_transforms]
        present = [v for v in votes if v is not None]
        if len(present) / n_seeds < args.min_agreement:
            n_dropped_coverage += 1
            continue
        c = Counter(present)
        top, top_n = c.most_common(1)[0]
        if top_n / n_seeds >= args.min_agreement:
            stable_transforms[feat] = top
            if args.include_params:
                # Average params across the agreeing seeds.
                agreeing_params = [
                    per_seed_params[i].get(feat)
                    for i, v in enumerate(votes)
                    if v == top and per_seed_params[i].get(feat)
                ]
                if agreeing_params and all(
                    len(p) == len(agreeing_params[0]) for p in agreeing_params
                ):
                    n = len(agreeing_params[0])
                    avg = [
                        sum(p[k] for p in agreeing_params) / len(agreeing_params)
                        for k in range(n)
                    ]
                    stable_params[feat] = avg
        else:
            n_dropped_minority += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '"""Seed-stable transform recommendation.',
        '',
        f'Built by zentrain/tools/seed_stable_screen.py from {n_seeds} seed dirs:',
    ]
    for sd in args.seed_dirs:
        lines.append(f'  - {sd}')
    lines += [
        '',
        f'Strict-majority threshold: ≥{args.min_agreement:.2f} agreement.',
        f'Features recommended in any seed: {len(all_feats)}',
        f'Features in stable output: {len(stable_transforms)}',
        f'Dropped (no clear majority): {n_dropped_minority}',
        f'Dropped (low coverage):      {n_dropped_coverage}',
        '"""',
        '',
        'FEATURE_TRANSFORMS = {',
    ]
    for k, v in stable_transforms.items():
        lines.append(f'    {k!r}: {v!r},')
    lines.append('}')
    if args.include_params and stable_params:
        lines += ['', 'FEATURE_TRANSFORM_PARAMS = {']
        for k, vs in stable_params.items():
            params_repr = '[' + ', '.join(f'{p:.6g}' for p in vs) + ']'
            lines.append(f'    {k!r}: {params_repr},')
        lines.append('}')

    args.out.write_text('\n'.join(lines) + '\n')

    print(f'Read {n_seeds} seed dirs')
    print(f'Features recommended in any seed: {len(all_feats)}')
    print(f'  stable (majority agreed): {len(stable_transforms)}')
    print(f'  dropped (no clear majority): {n_dropped_minority}')
    print(f'  dropped (low coverage):      {n_dropped_coverage}')
    print(f'Wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
