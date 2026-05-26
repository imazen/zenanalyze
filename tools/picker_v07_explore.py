#!/usr/bin/env python3
"""Explore v07-specific knobs — does force_strategy / progressive / max_strategy_size /
gaborish / patches / lz77 / lf_frame open new Pareto cells?

For each (image, distance), find the best safe alternative cell using
the FULL v07 cell taxonomy. Compare to v06 baseline picker (which can
only choose effort × biters × ziters, not v07's knobs).
"""
from __future__ import annotations

# DEDUP-B3 deprecation banner — added 2026-05-26 (B3 audit).
# This script is RETIRED per docs/ecosystem_cleanliness_review_2026-05-17.md
# (none of the v* picker scripts under tools/ are imported by the
# canonical trainer (zentrain/tools/train_hybrid.py) or covered by CI).
# Source kept for audit + as template — NOT a live training path.
import sys as _b3_sys
_b3_sys.stderr.write(
    "WARNING: picker_v07_explore.py is RETIRED (DEDUP-B3 audit, 2026-05-26).\n"
    "         v0.7 knob-exploration probe (one-off R&D analysis).\n"
    "         Use: tools/v14_metapicker_train.py or zentrain/tools/train_hybrid.py for real training; one-off probes belong as throwaway notebooks.\n"
    "         Source kept for audit; not on the live training path.\n"
)

import csv, json, math, sys
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean

import numpy as np

V06_TSV = Path("/home/lilith/sweep-data/zenjxl_v06.tsv")
V07_DIR = Path("/home/lilith/sweep-data/v07")
FEATURES_TSV = Path(
    "/home/lilith/work/zen/zenjxl/benchmarks/zenjxl_features_v04full_2026-05-04.tsv"
)

DEFAULT_EFFORT = 7
ZENSIM_TOL = 0.05
SPEED_TOL = 1.05
BYTES_GAIN = 0.99


def load_v06():
    rows = []
    with open(V06_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                if not r["encoded_bytes"] or not r["score_zensim"]:
                    continue
                k = json.loads(r["knob_tuple_json"])
                if k.get("noise") is True:
                    continue
                rows.append({
                    "src": "v06",
                    "image": r["image_path"].rsplit("/", 1)[-1],
                    "distance": round(float(k["distance"]), 4),
                    "effort": int(k["effort"]),
                    "biters": int(k.get("butteraugli_iters", 0)),
                    "ziters": int(k.get("zensim_iters", 0)),
                    "force_strategy": None,
                    "max_strategy_size": None,
                    "progressive": "single",
                    "gaborish": True,
                    "patches": None,
                    "lz77": None,
                    "lf_frame": None,
                    "pixel_domain_loss": True,
                    "bytes": int(r["encoded_bytes"]),
                    "ms": float(r["encode_ms"]),
                    "zensim": float(r["score_zensim"]),
                })
            except Exception:
                continue
    return rows


def load_v07():
    rows = []
    if not V07_DIR.exists():
        return rows
    for tsv in sorted(V07_DIR.glob("*.tsv")):
        with open(tsv) as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                try:
                    if not r["encoded_bytes"] or not r["score_zensim"]:
                        continue
                    k = json.loads(r["knob_tuple_json"])
                    if k.get("noise") is True:
                        continue
                    rows.append({
                        "src": "v07",
                        "image": r["image_path"].rsplit("/", 1)[-1],
                        "distance": round(float(k["distance"]), 4),
                        "effort": int(k["effort"]),
                        "biters": int(k.get("butteraugli_iters", 0)),
                        "ziters": int(k.get("zensim_iters", 0)),
                        "force_strategy": k.get("force_strategy"),
                        "max_strategy_size": k.get("max_strategy_size"),
                        "progressive": k.get("progressive", "single"),
                        "gaborish": k.get("gaborish", True),
                        "patches": k.get("patches"),
                        "lz77": k.get("lz77"),
                        "lf_frame": k.get("lf_frame"),
                        "pixel_domain_loss": k.get("pixel_domain_loss", True),
                        "bytes": int(r["encoded_bytes"]),
                        "ms": float(r["encode_ms"]),
                        "zensim": float(r["score_zensim"]),
                    })
                except Exception:
                    continue
    return rows


def is_safe(c, default):
    return (
        c["bytes"] < default["bytes"] * BYTES_GAIN
        and c["ms"] <= default["ms"] * SPEED_TOL
        and c["zensim"] >= default["zensim"] - ZENSIM_TOL
    )


def main():
    v06 = load_v06()
    v07 = load_v07()
    print(f"[v06] {len(v06)} cells", file=sys.stderr)
    print(f"[v07] {len(v07)} cells", file=sys.stderr)
    if not v07:
        print("v07 dir empty; copy v07 chunks first")
        return

    # For each (image, distance), see if v07-only knobs offer Pareto wins
    # over v06's best (effort × biters × ziters)
    by_id = defaultdict(lambda: {"v06": {}, "v07": {}})
    for r in v06:
        cell = (r["effort"], r["biters"], r["ziters"])
        key = (r["image"], r["distance"])
        by_id[key]["v06"][cell] = r
    for r in v07:
        cell = (r["effort"], r["biters"], r["ziters"], r["force_strategy"],
                r["max_strategy_size"], r["progressive"], r["gaborish"],
                r["patches"], r["lz77"], r["lf_frame"], r["pixel_domain_loss"])
        key = (r["image"], r["distance"])
        by_id[key]["v07"][cell] = r

    # Per (img, dist) where BOTH sweeps have data:
    #   - default = v06's effort=7 biters=0 ziters=0
    #   - v06_best = best safe alt over v06 cells
    #   - v07_best = best safe alt over v07 cells (which may include v06-equivalent + extras)
    #   - delta = v07_best.bytes - v06_best.bytes; if v07 wins, the v07 knobs helped
    n_evaluated = 0
    n_v07_wins_strict = 0  # v07 strictly better than v06
    n_tie = 0
    v07_win_dbytes = []
    v07_win_cells = Counter()
    v06_only_picks = []  # bytes saved by v06 picker
    v07_only_picks = []  # bytes saved by v07 picker (vs default)
    v07_extra_savings = []  # extra bytes saved by v07 over v06

    for (img, dist), groups in by_id.items():
        v06_cells = groups["v06"]
        v07_cells = groups["v07"]
        if not v06_cells or not v07_cells:
            continue
        # default cell from v06 (effort=7, biters=0, ziters=0)
        default = v06_cells.get((DEFAULT_EFFORT, 0, 0))
        if default is None:
            continue
        n_evaluated += 1

        # v06's best safe alt
        v06_best_bytes = default["bytes"]
        v06_best = default
        for c, d in v06_cells.items():
            if c == (DEFAULT_EFFORT, 0, 0):
                continue
            if is_safe(d, default) and d["bytes"] < v06_best_bytes:
                v06_best_bytes = d["bytes"]
                v06_best = d

        # v07's best safe alt (any cell in v07 that's safe vs v06's default)
        v07_best_bytes = default["bytes"]
        v07_best = default
        v07_best_cell = None
        for c, d in v07_cells.items():
            if is_safe(d, default) and d["bytes"] < v07_best_bytes:
                v07_best_bytes = d["bytes"]
                v07_best = d
                v07_best_cell = c

        # Compare
        v06_savings = (default["bytes"] - v06_best_bytes) / default["bytes"]
        v07_savings = (default["bytes"] - v07_best_bytes) / default["bytes"]
        v06_only_picks.append(v06_savings)
        v07_only_picks.append(v07_savings)

        if v07_best_bytes < v06_best_bytes * 0.999:
            # v07 strictly better
            n_v07_wins_strict += 1
            v07_win_dbytes.append((v07_best_bytes - v06_best_bytes) / default["bytes"] * 100)
            v07_extra_savings.append((v06_best_bytes - v07_best_bytes) / default["bytes"] * 100)
            # Record what knobs v07 used (excluding shared effort/biters/ziters)
            if v07_best_cell:
                _, _, _, fs, mss, prog, gab, pat, lz, lf, pdl = v07_best_cell
                tag_parts = []
                if fs is not None: tag_parts.append(f"fs={fs}")
                if mss is not None: tag_parts.append(f"mss={mss}")
                if prog != "single": tag_parts.append(f"prog={prog}")
                if gab is False: tag_parts.append("gab=F")
                if pat is True: tag_parts.append("pat=T")
                if lz is False: tag_parts.append("lz=F")
                if lf is True: tag_parts.append("lf=T")
                if pdl is False: tag_parts.append("pdl=F")
                tag = "|".join(tag_parts) if tag_parts else "(no v07-extra knobs)"
                v07_win_cells[tag] += 1
        elif abs(v07_best_bytes - v06_best_bytes) <= v06_best_bytes * 0.001:
            n_tie += 1

    print(f"\n## v06-vs-v07 head-to-head on {n_evaluated} (image, distance) cells")
    print(f"   Default: effort=7, biters=0, ziters=0 (v06's default)")
    print(f"   v06's safe-alt mean savings: {mean(v06_only_picks)*100:+.3f}%")
    print(f"   v07's safe-alt mean savings: {mean(v07_only_picks)*100:+.3f}%")
    print(f"   v07 strictly better than v06: {n_v07_wins_strict} ({100*n_v07_wins_strict/n_evaluated:.1f}%)")
    if v07_extra_savings:
        print(f"     mean extra bytes saved by v07: -{mean(v07_extra_savings):.3f}%")
    print(f"   tied: {n_tie}")
    print()
    print(f"## Top v07-knob combos that beat v06")
    for tag, n in v07_win_cells.most_common(15):
        print(f"   {n:4d}x  {tag}")


if __name__ == "__main__":
    main()
