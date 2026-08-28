#!/usr/bin/env python3
"""
Per-family effort-cap DEGRADATION table + per-effort COST table from a canonical
per-codec picker dataset (zenanalyze#85, the meta-picker's "per-family
degradation" half).

The meta picker masks families whose cheapest effort exceeds the budget and
penalises each survivor by the RD cost of being capped to its *feasible*
effort (`feasible = highest effort with cost ≤ budget`):

    degradation(family, cap) = RD(cap) − RD(reference effort)

in a DECLARED metric space. This tool measures both halves, per family, from
the canonical 2026-06-27 datasets (`s3://zentrain/canonical/2026-06-27/<codec>/`,
one row per image × q × knob cell, with `encoded_bytes`, `encode_ms`, and
`score_*`):

- **degradation** — for every (image, quality target `t`, effort cap `E`):
  `bytes_at(E) = min encoded_bytes over rows with effort ≤ E and score ≥ t`
  (any cell / q that reaches the target), and
  `δ(E) = ln bytes_at(E) − ln bytes_at(E_ref)` — the log-bytes penalty for
  being capped at `E`. Aggregated over images as mean / p50 / p90 per
  (size_class, target, cap), plus the fraction of images for which *some*
  row at that cap reaches the target (reach). Units: natural log of bytes —
  `100·(e^δ − 1)` is the byte overhead in %, the space `argmin_masked`'s
  `per_output` offsets want for a log-bytes router.
- **cost** — per (effort, size_class): `encode_ms / MP` p50 / p90 over all
  rows at that effort, and a per-effort `ln ms = a + b·ln px` fit over every
  row — the `cost(family, effort, pixels)` the feasibility mask needs
  (`AllowedFamilies::viable` takes the codec's own estimate; this is what it
  should be calibrated from).

Effort is parsed from the `cell` label with `--effort-re` (a regex with a named
group `effort`; the default table covers the 2026-06-27 vocabularies:
zenwebp `vp8-m{2,4,6}_…`, zenjxl `vd-e{1..9}_…`, zenavif `s{2..8}…`). Rows
whose cell does not match are dropped and counted. Pass every split you have
(train / validate / test) — the statistic is per-codec, not per-split.

Output: a TSV (`--out`) with `section=degradation|cost` rows and a markdown
summary (`--md`). Both are meant to be committed under `benchmarks/`.

Usage:
    python3 fit_family_degradation.py --codec zenwebp \\
        --canonical ~/tmp/canonical/zenwebp_lossy/{train,validate,test}.parquet \\
        --out benchmarks/family_degradation_zenwebp_2026-08-28.tsv \\
        --md  benchmarks/family_degradation_zenwebp_2026-08-28.md
"""
from __future__ import annotations

import argparse
import datetime
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenmetrics_sweep_adapter import size_class  # noqa: E402

EFFORT_RE = {
    "zenwebp": r"^vp8l?-m(?P<effort>\d+)",
    "zenjxl": r"^(?:vd|mod)-e(?P<effort>\d+)",
    "zenavif": r"^s(?P<effort>\d+)",
}
# Knob semantics per codec: (label prefix, higher-number-means-LESS-effort). rav1e's
# `speed` runs the other way from webp `method` / jxl `effort`: s2 is the slow,
# best-RD end and s8 the fast end, so the ranking is inverted and a "cap" is a
# floor on the speed number ("cap ≥ s6" = speeds 6..8 allowed).
EFFORT_KNOB = {
    "zenwebp": ("m", False),
    "zenjxl": ("e", False),
    "zenavif": ("s", True),
}
DEFAULT_TARGETS = "40,50,60,70,75,80,85,90,95"


def load(paths: list[Path], metric: str) -> dict:
    cols = ["image_path", "cell", "encoded_bytes", "encode_ms", "width", "height", metric]
    parts = []
    for p in paths:
        parts.append(pq.read_table(p, columns=cols))
    import pyarrow as pa

    t = pa.concat_tables(parts)
    return {
        "image_path": np.asarray(t["image_path"].to_pylist(), dtype=object),
        "cell": np.asarray(t["cell"].to_pylist(), dtype=object),
        "bytes": t["encoded_bytes"].to_numpy().astype(np.float64),
        "ms": t["encode_ms"].to_numpy().astype(np.float64),
        "px": t["width"].to_numpy().astype(np.float64) * t["height"].to_numpy().astype(np.float64),
        "width": t["width"].to_numpy().astype(np.int64),
        "height": t["height"].to_numpy().astype(np.int64),
        "score": t[metric].to_numpy().astype(np.float64),
    }


MISSING = -(10**6)


def effort_of_cells(cells: np.ndarray, pattern: str, invert: bool) -> tuple[np.ndarray, int]:
    """Per-row effort RANK (higher = more effort, so a cap is `rank <= cap`);
    `MISSING` for cells the pattern does not match. With `invert` the knob
    number is negated (rav1e speed), so `orig = -rank`."""
    rx = re.compile(pattern)
    uniq = sorted(set(cells.tolist()))
    lookup = {}
    unmatched = 0
    for c in uniq:
        m = rx.match(c)
        if m:
            v = int(m.group("effort"))
            lookup[c] = -v if invert else v
        else:
            lookup[c] = MISSING
            unmatched += 1
    if unmatched:
        sys.stderr.write(f"{unmatched} of {len(uniq)} cells do not match {pattern!r}: "
                         f"{[c for c in uniq if lookup[c] == MISSING][:8]}\n")
    return np.asarray([lookup[c] for c in cells], dtype=np.int64), unmatched


def degradation(d: dict, effort: np.ndarray, targets: list[float], ref: int) -> dict:
    """{(size_class, target, cap): {"n": images with a reference-reachable target,
    "reach": images reaching at cap, "delta": [ln bytes_at(cap) − ln bytes_at(ref)]}}"""
    efforts = sorted(set(int(e) for e in effort if e != MISSING))
    order = np.argsort(d["image_path"], kind="stable")
    paths = d["image_path"][order]
    bounds = np.flatnonzero(np.r_[True, paths[1:] != paths[:-1], True])
    out: dict = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        e = effort[idx]
        keep = e != MISSING
        idx, e = idx[keep], e[keep]
        if idx.size == 0:
            continue
        sc = size_class(int(d["width"][idx[0]]), int(d["height"][idx[0]]))
        by = d["bytes"][idx]
        s = d["score"][idx]
        for t in targets:
            reach_t = s >= t
            if not reach_t.any():
                continue
            ref_mask = reach_t & (e <= ref)
            if not ref_mask.any():
                continue
            b_ref = by[ref_mask].min()
            for cap in efforts:
                key = (sc, t, cap)
                cell = out.setdefault(key, {"n": 0, "reach": 0, "delta": []})
                cell["n"] += 1
                m = reach_t & (e <= cap)
                if m.any():
                    cell["reach"] += 1
                    cell["delta"].append(math.log(by[m].min()) - math.log(b_ref))
    return out


def cost(d: dict, effort: np.ndarray) -> tuple[dict, dict]:
    """per (effort, size_class): ms/MP p50/p90; per effort: ln ms = a + b ln px fit."""
    efforts = sorted(set(int(e) for e in effort if e != MISSING))
    scs = np.asarray([size_class(int(w), int(h)) for w, h in zip(d["width"], d["height"])], dtype=object)
    ms_per_mp = d["ms"] / (d["px"] / 1e6)
    table: dict = {}
    fits: dict = {}
    for e in efforts:
        me = effort == e
        for sc in sorted(set(scs[me].tolist())):
            m = me & (scs == sc)
            v = ms_per_mp[m]
            table[(e, sc)] = {"n": int(m.sum()), "p50": float(np.median(v)), "p90": float(np.percentile(v, 90))}
        x = np.log(d["px"][me])
        y = np.log(np.maximum(d["ms"][me], 1e-3))
        A = np.c_[np.ones_like(x), x]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        fits[e] = {"a": float(coef[0]), "b": float(coef[1]), "rmse_ln": float(np.sqrt(np.mean(resid**2))), "n": int(me.sum())}
    return table, fits


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", required=True, help="zenwebp | zenjxl | zenavif | … (picks the default --effort-re)")
    ap.add_argument("--canonical", nargs="+", type=Path, required=True)
    ap.add_argument("--effort-re", default=None, help="regex with a named group `effort` applied to `cell`")
    ap.add_argument("--effort-label", default=None, help="knob letter for labels (default per codec: m / e / s)")
    ap.add_argument("--effort-invert", action="store_true", default=None,
                    help="higher knob number = LESS effort (rav1e speed); default per codec")
    ap.add_argument("--metric", default="score_zensim")
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="comma-separated quality targets in the metric's units")
    ap.add_argument("--reference-effort", type=int, default=None, help="knob number of the reference; default: the most-effort setting present")
    ap.add_argument("--out", type=Path, required=True, help="TSV")
    ap.add_argument("--md", type=Path, default=None, help="markdown summary")
    args = ap.parse_args(argv)

    pattern = args.effort_re or EFFORT_RE.get(args.codec)
    if not pattern:
        raise SystemExit(f"no default --effort-re for codec {args.codec!r}; pass one")
    knob_label, knob_invert = EFFORT_KNOB.get(args.codec, ("e", False))
    label = args.effort_label or knob_label
    invert = knob_invert if args.effort_invert is None else args.effort_invert
    targets = [float(t) for t in args.targets.split(",") if t.strip()]
    d = load(args.canonical, args.metric)
    effort, unmatched = effort_of_cells(d["cell"], pattern, invert)
    efforts = sorted(set(int(e) for e in effort if e != MISSING))  # ranks, ascending effort
    if not efforts:
        raise SystemExit("no cell matched the effort pattern")

    def orig(rank: int) -> int:
        return -rank if invert else rank

    def name(rank: int) -> str:
        return f"{label}{orig(rank)}"

    cap_word = "cap ≥" if invert else "cap ≤"
    ref = (-args.reference_effort if invert else args.reference_effort) if args.reference_effort is not None else max(efforts)
    if ref not in efforts:
        raise SystemExit(f"reference effort {name(ref)} not present; efforts = {[name(e) for e in efforts]}")
    n_images = len(set(d["image_path"].tolist()))
    deg = degradation(d, effort, targets, ref)
    ctab, cfit = cost(d, effort)

    today = datetime.date.today().isoformat()
    head = [
        f"# family degradation + cost table — codec={args.codec} metric={args.metric} knob={label} invert={invert} reference={name(ref)}",
        f"# git={_git()} date={today} rows={len(d['bytes'])} images={n_images} efforts={[name(e) for e in efforts]} (ascending effort) unmatched_cells={unmatched}",
        f"# inputs={' '.join(str(p) for p in args.canonical)}",
        "# degradation: delta = ln(min bytes reaching target with effort<=cap) - ln(min bytes reaching target with effort<=ref); over images; reach = images with any row at cap reaching the target; cap = knob number (for an inverted knob a cap allows that number and every FASTER one)",
        "# cost: encode_ms per megapixel per (effort, size_class); fit ln(ms) = a + b*ln(pixels) per effort",
    ]
    lines = list(head)
    lines.append("section\tcodec\tsize_class\ttarget\tcap_effort\tn\treach_frac\tdelta_mean_ln\tdelta_p50_ln\tdelta_p90_ln\tdelta_mean_pct")
    size_order = {"tiny": 0, "small": 1, "medium": 2, "large": 3}
    for (sc, t, cap), c in sorted(deg.items(), key=lambda kv: (size_order.get(kv[0][0], 9), kv[0][1], kv[0][2])):
        dl = np.asarray(c["delta"]) if c["delta"] else np.asarray([np.nan])
        mean = float(np.nanmean(dl))
        lines.append(
            f"degradation\t{args.codec}\t{sc}\t{t:g}\t{orig(cap)}\t{c['n']}\t{c['reach'] / c['n']:.4f}\t"
            f"{mean:.5f}\t{float(np.nanmedian(dl)):.5f}\t{float(np.nanpercentile(dl, 90)):.5f}\t{100 * (math.exp(mean) - 1):.3f}"
        )
    lines.append("section\tcodec\tsize_class\teffort\tn\tms_per_mp_p50\tms_per_mp_p90\tfit_a\tfit_b\tfit_rmse_ln")
    for (e, sc), c in sorted(ctab.items(), key=lambda kv: (kv[0][0], size_order.get(kv[0][1], 9))):
        f = cfit[e]
        lines.append(f"cost\t{args.codec}\t{sc}\t{orig(e)}\t{c['n']}\t{c['p50']:.3f}\t{c['p90']:.3f}\t{f['a']:.4f}\t{f['b']:.4f}\t{f['rmse_ln']:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")

    if args.md:
        M: list[str] = []
        M.append(f"# {args.codec}: effort-cap degradation and per-effort cost ({today})")
        M.append("")
        M.append(f"Source: canonical 2026-06-27 `{args.codec}` ({len(d['bytes'])} rows, {n_images} images, "
                 f"efforts {[name(e) for e in efforts]} (ascending effort), reference **{name(ref)}**, metric `{args.metric}`). "
                 f"Generated by `zentrain/tools/fit_family_degradation.py` → `{args.out}`.")
        M.append("")
        M.append(f"## Degradation from capping effort (all sizes pooled, ln-bytes; % = byte overhead vs the reference {name(ref)})")
        M.append("")
        M.append("| target | " + " | ".join(f"{cap_word} {name(e)}" for e in efforts if e != ref) + " |")
        M.append("|---:|" + "---:|" * (len(efforts) - 1))
        for t in targets:
            cells = []
            for e in efforts:
                if e == ref:
                    continue
                dl = []
                n = reach = 0
                for (sc, tt, cap), c in deg.items():
                    if tt == t and cap == e:
                        dl += c["delta"]
                        n += c["n"]
                        reach += c["reach"]
                if n == 0:
                    cells.append("—")
                    continue
                mean = float(np.mean(dl)) if dl else float("nan")
                p90 = float(np.percentile(dl, 90)) if dl else float("nan")
                cells.append(f"{100 * (math.exp(mean) - 1):+.2f}% (p90 {100 * (math.exp(p90) - 1):+.1f}%, reach {reach / n:.0%})")
            M.append(f"| {t:g} | " + " | ".join(cells) + " |")
        M.append("")
        M.append("## Encode cost per effort (ms / MP, all sizes pooled; fit ln ms = a + b·ln px)")
        M.append("")
        M.append("| effort | n | ms/MP p50 (tiny / small / medium) | fit a | fit b | rmse (ln) |")
        M.append("|---:|---:|---|---:|---:|---:|")
        for e in efforts:
            f = cfit[e]
            per = " / ".join(f"{ctab[(e, sc)]['p50']:.1f}" if (e, sc) in ctab else "—" for sc in ("tiny", "small", "medium", "large") if any(k[1] == sc for k in ctab))
            M.append(f"| {name(e)} | {f['n']} | {per} | {f['a']:.3f} | {f['b']:.3f} | {f['rmse_ln']:.3f} |")
        M.append("")
        M.append(
            "ms/MP for `tiny` is dominated by the per-call fixed cost (a 64² image is 0.004 MP), "
            "so read the fit (`ln ms = a + b·ln px`) for the size dependence and the p50s for the "
            "relative cost between efforts. Per-size-class degradation rows are in the TSV "
            "(`section=degradation`)."
        )
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(M) + "\n")
    sys.stderr.write(f"wrote {args.out}" + (f" and {args.md}" if args.md else "") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
