#!/usr/bin/env python3
"""Per-image oracle on v05c zenjxl data: what's the BEST static-cell win rate
the existing data supports, before any picker is even involved?

This is the upper bound for "what could a perfect picker over v05c achieve."
If the upper bound is good, the bottleneck is picker training (a free fix).
If the upper bound is poor, we need a new sweep with expanded knobs.

Per-image, per-distance:
- Default cell: effort=7, noise=false (matches the v0.5 picker A/B baseline)
- Compute the best cell that:
  1. Doesn't regress zensim by >0.05pp (parity)
  2. Doesn't slow down encode by >5% (parity-while-faster constraint)
  3. Minimizes bytes
- Aggregate the win rate, average improvement.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from statistics import mean, median


def load(tsv):
    rows = []
    with open(tsv) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                if not r.get("encoded_bytes") or not r.get("score_zensim"):
                    continue
                r["bytes"] = int(r["encoded_bytes"])
                r["ms"] = float(r["encode_ms"])
                r["zensim"] = float(r["score_zensim"])
                r["knobs"] = json.loads(r["knob_tuple_json"])
                if r["knobs"].get("noise") is True:
                    continue  # ignore noise-on; v0.5 picker default uses noise=False
            except Exception:
                continue
            rows.append(r)
    return rows


def cell_key(knobs):
    e = knobs.get("effort", "?")
    return f"e{e}"


def main(tsv):
    rows = load(tsv)
    by_id = defaultdict(list)
    for r in rows:
        d = r["knobs"].get("distance", -1)
        by_id[(r["image_path"], d)].append(r)

    n_cells = 0
    safe_wins = 0
    any_wins = 0
    safe_dbytes = []
    safe_dzensim = []
    safe_dms = []
    any_dbytes = []
    any_dzensim = []
    any_dms = []
    safe_cell_counts = defaultdict(int)
    any_cell_counts = defaultdict(int)

    for (img, dist), cells in by_id.items():
        default = next((c for c in cells if c["knobs"].get("effort") == 7), None)
        if default is None:
            continue
        n_cells += 1

        speed_safe = [
            c for c in cells
            if c["knobs"].get("effort") != 7
            and c["bytes"] < default["bytes"] * 0.99
            and c["zensim"] >= default["zensim"] - 0.05
            and c["ms"] <= default["ms"] * 1.05
        ]
        if speed_safe:
            speed_safe.sort(key=lambda c: c["bytes"])
            best = speed_safe[0]
            safe_wins += 1
            safe_dbytes.append(100 * (best["bytes"] - default["bytes"]) / default["bytes"])
            safe_dzensim.append(best["zensim"] - default["zensim"])
            safe_dms.append(100 * (best["ms"] - default["ms"]) / default["ms"])
            safe_cell_counts[cell_key(best["knobs"])] += 1

        any_better = [
            c for c in cells
            if c["knobs"].get("effort") != 7
            and c["bytes"] < default["bytes"] * 0.99
            and c["zensim"] >= default["zensim"] - 0.05
        ]
        if any_better:
            any_better.sort(key=lambda c: c["bytes"])
            best = any_better[0]
            any_wins += 1
            any_dbytes.append(100 * (best["bytes"] - default["bytes"]) / default["bytes"])
            any_dzensim.append(best["zensim"] - default["zensim"])
            any_dms.append(100 * (best["ms"] - default["ms"]) / default["ms"])
            any_cell_counts[cell_key(best["knobs"])] += 1

    print(f"## v05c zenjxl ORACLE — upper bound for picker-over-existing-data")
    print(f"   {n_cells} (image, distance) pairs analyzed (noise=False only)")
    print(f"   Default: effort=7, noise=False")
    print()
    print(f"   Speed-safe wins (-1% bytes, ≥-0.05pp zensim, ≤+5% ms):")
    if safe_wins:
        print(f"     count: {safe_wins}/{n_cells} = {100*safe_wins/n_cells:.1f}%")
        print(f"     mean Δbytes:   {mean(safe_dbytes):+.2f}%")
        print(f"     median Δbytes: {median(safe_dbytes):+.2f}%")
        print(f"     mean Δzensim:  {mean(safe_dzensim):+.3f}pp")
        print(f"     mean Δms:      {mean(safe_dms):+.1f}%")
        print(f"     top winning cells:")
        for cell, n in sorted(safe_cell_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"       {n:4d}x  {cell}")
    else:
        print("     no wins")
    print()
    print(f"   Speed-unconstrained wins (-1% bytes, ≥-0.05pp zensim):")
    if any_wins:
        print(f"     count: {any_wins}/{n_cells} = {100*any_wins/n_cells:.1f}%")
        print(f"     mean Δbytes:   {mean(any_dbytes):+.2f}%")
        print(f"     median Δbytes: {median(any_dbytes):+.2f}%")
        print(f"     mean Δzensim:  {mean(any_dzensim):+.3f}pp")
        print(f"     mean Δms:      {mean(any_dms):+.1f}%")
        print(f"     top winning cells:")
        for cell, n in sorted(any_cell_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"       {n:4d}x  {cell}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/lilith/sweep-data/zenjxl_v05c.tsv")
