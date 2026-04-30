#!/usr/bin/env python3
"""
Human-readable health report for any baked picker artifact.

Reads either a training-side JSON (with `safety_report` if produced
by current train_hybrid.py) or a manifest JSON next to a `.bin`
(with the lifted `safety_report` block from `bake_picker.py`), and
prints the diagnostics block in a form that's quick to read at a
glance.

Falls back gracefully on legacy bakes that pre-date `safety_report`:
prints what's there (architecture, schema_hash, config count) plus
a static MLP weight scan computed from the JSON layers themselves.

Usage:

    python3 diagnose_picker.py --model benchmarks/<bake>.json
    python3 diagnose_picker.py --manifest models/<picker>.manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def _format_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.2f}%"


def _format_float(x: float | None, fmt: str = ".4f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return format(x, fmt)


def diagnose_legacy(model: dict) -> None:
    """Compute a minimal health report for a JSON without safety_report."""
    layers = model.get("layers", [])
    n_inputs = model.get("n_inputs")
    n_outputs = model.get("n_outputs")
    n_cells = model.get("hybrid_heads_manifest", {}).get("n_cells")
    print(f"  n_inputs:  {n_inputs}")
    print(f"  n_outputs: {n_outputs}")
    if n_cells is not None:
        print(f"  n_cells:   {n_cells}  (hybrid heads: {len(layers)} layers)")

    print("\n  Static MLP weight scan:")
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        nan_w = bool(np.isnan(W).any())
        inf_w = bool(np.isinf(W).any())
        absw = np.abs(W).flatten()
        med = float(np.median(absw)) if absw.size else 0.0
        mx = float(absw.max()) if absw.size else 0.0
        ratio = mx / max(med, 1e-12)
        flag = "  ⚠" if nan_w or inf_w or ratio > 1000.0 else ""
        print(
            f"    layer {i}: shape={W.shape}, |W| max/med={mx:.3f}/{med:.3f} "
            f"ratio={ratio:.1f}, |b| max={float(np.abs(b).max()):.3f}, "
            f"nan_in_W={nan_w}, inf_in_W={inf_w}{flag}"
        )


def render_report(report: dict) -> None:
    """Pretty-print a `safety_report` block."""
    passed = report.get("passed")
    icon = "✓" if passed else "⚠"
    print(f"\n  Safety: {icon} {'PASSED' if passed else 'FAILED'}")
    violations = report.get("violations") or []
    if violations:
        print(f"\n  Violations ({len(violations)}):")
        for v in violations:
            print(f"    • {v}")

    diag = report.get("diagnostics") or {}
    argmin = diag.get("argmin") or {}
    train = argmin.get("train") or {}
    val = argmin.get("val") or {}
    if val:
        gap = (val.get("mean_pct") or 0.0) - (train.get("mean_pct") or 0.0)
        print("\n  Argmin metrics:")
        print(
            f"    train: mean={_format_pct(train.get('mean_pct'))}  "
            f"argmin_acc={_format_pct((train.get('argmin_acc') or 0) * 100)}  "
            f"p99={_format_pct(train.get('p99_pct'))}  max={_format_pct(train.get('max_pct'))}"
        )
        print(
            f"    val:   mean={_format_pct(val.get('mean_pct'))}  "
            f"argmin_acc={_format_pct((val.get('argmin_acc') or 0) * 100)}  "
            f"p99={_format_pct(val.get('p99_pct'))}  max={_format_pct(val.get('max_pct'))}"
        )
        print(f"    train→val gap: {gap:+.2f}pp")

    by_zq = diag.get("by_zq") or {}
    if by_zq:
        print("\n  Per-zq breakdown (top 5 worst by p99):")
        ordered = sorted(
            by_zq.items(),
            key=lambda kv: kv[1].get("p99_pct", 0.0),
            reverse=True,
        )
        for zq, m in ordered[:5]:
            print(
                f"    zq={zq:>3}  n={m.get('n', 0):>4}  "
                f"mean={_format_pct(m.get('mean_pct'))}  "
                f"p90={_format_pct(m.get('p90_pct'))}  "
                f"p99={_format_pct(m.get('p99_pct'))}  "
                f"max={_format_pct(m.get('max_pct'))}"
            )

    by_size = diag.get("by_size") or {}
    if by_size:
        print("\n  Per-size-class breakdown:")
        for sz, m in sorted(by_size.items()):
            print(
                f"    {sz:8s}  n={m.get('n', 0):>4}  "
                f"mean={_format_pct(m.get('mean_pct'))}  "
                f"p99={_format_pct(m.get('p99_pct'))}  "
                f"max={_format_pct(m.get('max_pct'))}"
            )

    worst = diag.get("worst_case") or []
    if worst:
        print(f"\n  Worst-case rows (top {min(len(worst), 5)}):")
        for w in worst[:5]:
            print(
                f"    {w.get('overhead_pct', 0):.1f}%  "
                f"zq={w.get('zq')}  size={w.get('size_class')}  "
                f"pick={w.get('pick')}  best={w.get('actual_best')}"
            )

    per_cell = diag.get("per_cell") or []
    if per_cell:
        print("\n  Per-cell calibration (top 5 by |delta|):")
        ordered = sorted(
            per_cell,
            key=lambda c: abs(c.get("calibration_delta") or 0.0),
            reverse=True,
        )
        for c in ordered[:5]:
            d = c.get("calibration_delta")
            print(
                f"    cell {c.get('cell'):>2}: {c.get('label', '?'):30s}  "
                f"n_train_rows={c.get('n_val_reach_rows', 0):>4}  "
                f"n_member_configs={c.get('n_member_configs', 0):>2}  "
                f"calibration={_format_float(d, '+.3f')}"
            )

    mlp = diag.get("mlp") or {}
    if mlp:
        print("\n  MLP health:")
        print(
            f"    dead neurons: {mlp.get('n_dead_neurons', 0)}/"
            f"{mlp.get('n_total_hidden_neurons', 0)} "
            f"({(mlp.get('dead_neuron_fraction') or 0) * 100:.1f}%)"
        )
        print(f"    max layer weight ratio: {mlp.get('max_layer_weight_ratio', 0):.0f}")
        print(
            f"    nan_in_weights={mlp.get('nan_in_weights')}  "
            f"inf_in_weights={mlp.get('inf_in_weights')}  "
            f"nan_in_predictions={mlp.get('nan_in_predictions')}"
        )

    fb = diag.get("feature_bounds") or {}
    if fb:
        print(f"\n  Feature bounds: {len(fb)} feature(s) with [p01, p99] recorded")
        # Show 3 most-skewed (largest std/range deviation from mean)
        skewed = sorted(
            (
                (name, s)
                for name, s in fb.items()
                if s.get("std") is not None and s.get("p99") is not None
            ),
            key=lambda kv: kv[1]["p99"] - kv[1]["p01"],
            reverse=True,
        )
        for name, s in skewed[:3]:
            print(
                f"    {name:30s}  range[{_format_float(s.get('p01'), '+.3f')}, "
                f"{_format_float(s.get('p99'), '+.3f')}]  "
                f"mean={_format_float(s.get('mean'), '+.3f')}  "
                f"std={_format_float(s.get('std'), '.3f')}"
            )

    rs = diag.get("reach_safety") or {}
    if rs.get("by_zq"):
        n_zq = len(rs["by_zq"])
        threshold = rs.get("threshold", 0.99)
        # Worst zq band by safe-cell count
        ordered = sorted(
            rs["by_zq"].items(),
            key=lambda kv: sum(1 for s in kv[1].get("safe", []) if s),
        )
        if ordered:
            worst_zq, worst_m = ordered[0]
            n_safe = sum(1 for s in worst_m.get("safe", []) if s)
            print(
                f"\n  Reach gate (threshold={threshold}): {n_zq} zq bands, "
                f"worst zq={worst_zq} has only {n_safe} safe cell(s)"
            )

    thresholds = report.get("thresholds")
    if thresholds:
        print("\n  Thresholds in effect:")
        for k, v in sorted(thresholds.items()):
            print(f"    {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", type=Path, help="training-side model JSON")
    grp.add_argument("--manifest", type=Path, help=".manifest.json next to a .bin")
    args = ap.parse_args()

    path = args.model or args.manifest
    src = json.loads(path.read_text())
    print(f"diagnose_picker: {path}")
    print("-" * 70)

    sr = src.get("safety_report")
    if not sr:
        print("  (legacy bake — no `safety_report` block; printing static info)")
        diagnose_legacy(src)
        return 0

    render_report(sr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
