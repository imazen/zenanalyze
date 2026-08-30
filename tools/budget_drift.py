#!/usr/bin/env python3
"""
Aggregate `examples/budget_drift_grid` raw rows into per-feature, per-size drift
distributions — the evidence for "should analysis thoroughness scale with image
size?".

Input is the raw TSV the example writes (one row per class × side × crop × arm ×
feature). Nothing is aggregated in the Rust side, so the raw values stay
auditable and every statistic here is reproducible from them.

Three normalisations, because they answer different questions:

  abs   |full - default|                     — the raw move.
  rel   |full - default| / max(|default|,e)  — the move as a fraction of the
                                               value. Blows up near zero, so
                                               it is reported but not decided on.
  sigma |full - default| / stdev_across_cells(default, at this side)
                                             — THE decision-relevant one. A
                                               model sees z-scored inputs, so a
                                               drift of 0.5 sigma moves the input
                                               half a standard deviation of what
                                               the model was trained to see,
                                               whatever the units are.

Attribution: the grid varies `pixel_budget` and `hf_max_blocks` independently
(`pbfull` / `hffull` arms), so each feature is labelled by which knob drives it.

Convergence: for the pixel-budget ladder (default → 1M → 2M → 4M → 8M → full),
a feature is `converging` only if |value(arm) - value(full)| is non-increasing
across the ladder on essentially every cell. That is the test for whether a
"sample at least X" contract could ever be sound for that feature.

Usage:
    tools/budget_drift.py --grid benchmarks/budget_drift_grid.tsv \\
        [--native benchmarks/budget_drift_native.tsv] [--out report.md]
    tools/budget_drift.py --grid ... --format tsv --out summary.tsv
"""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict

# Ladder orders for the convergence test. Each holds the *other* knob at its
# default, so a ladder measures one knob. A feature is tested on the ladder of
# whichever knob drives it — testing an hf-driven feature along the pixel-budget
# ladder passes vacuously, because its value is constant there.
PB_LADDER = ["default", "pb1m", "pb2m", "pb4m", "pb8m", "pbfull"]
HF_LADDER = ["default", "hf2k", "hf4k", "hf16k", "hf64k", "hffull"]
EPS = 1e-12


def load(path):
    """-> {(class, side, crop, label, feature): {arm: value}}"""
    cells = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key = (r["class"], int(r["side"]), r["crop"], r["label"], r["feature"])
            cells[key][r["arm"]] = float(r["value"])
    return cells


def finite(x):
    return x is not None and not (math.isnan(x) or math.isinf(x))


def pct(vals, q):
    """Nearest-rank percentile on a sorted copy; `q` in [0, 100]."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = min(len(s) - 1, max(0, int(math.ceil(q / 100.0 * len(s))) - 1))
    return s[k]


def summarize(cells, classes=None):
    """-> {(side, feature): stats}, one row per feature per size."""
    # Group values by (side, feature) so the sigma normaliser can use the
    # across-cell spread of the DEFAULT arm at that size.
    by_sf = defaultdict(list)  # (side, feature) -> [(key, arms)]
    for key, arms in cells.items():
        cls = key[0]
        if classes and cls not in classes:
            continue
        by_sf[(key[1], key[4])].append((key, arms))

    out = {}
    for (side, feat), entries in by_sf.items():
        defaults = [a.get("default") for _, a in entries]
        defaults = [d for d in defaults if finite(d)]
        # Spread of the feature across images at this size — the scale a model
        # would z-score against. Population stdev; needs >= 2 cells.
        spread = statistics.pstdev(defaults) if len(defaults) >= 2 else float("nan")

        abs_d, rel_d, nan_flip = [], [], 0
        pb_only, hf_only = [], []
        converging = 0
        conv_eligible = 0
        for _, arms in entries:
            d, f = arms.get("default"), arms.get("full")
            if not finite(d) or not finite(f):
                # One arm finite and the other not is itself a budget effect.
                if finite(d) != finite(f):
                    nan_flip += 1
                continue
            delta = abs(f - d)
            abs_d.append(delta)
            rel_d.append(delta / max(abs(d), EPS))
            if finite(arms.get("pbfull")):
                pb_only.append(abs(arms["pbfull"] - d))
            if finite(arms.get("hffull")):
                hf_only.append(abs(arms["hffull"] - d))
            # Convergence, tested on the ladder of the knob that actually moves
            # this feature on this cell. A ladder along which the value never
            # changes is NOT evidence of convergence — it is evidence the ladder
            # is the wrong axis — so those cells are excluded, not counted as
            # passing.
            for ladder_arms in (PB_LADDER, HF_LADDER):
                ladder = [arms.get(a) for a in ladder_arms]
                if not all(finite(v) for v in ladder):
                    continue
                target = ladder[-1]
                errs = [abs(v - target) for v in ladder]
                if errs[0] == 0.0:
                    continue  # constant along this ladder — wrong axis
                conv_eligible += 1
                # Non-increasing within a small tolerance for f32 wobble.
                tol = 1e-6 * max(abs(target), 1.0)
                if all(errs[i + 1] <= errs[i] + tol for i in range(len(errs) - 1)):
                    converging += 1

        if not abs_d:
            continue
        med_abs, p95_abs = statistics.median(abs_d), pct(abs_d, 95)
        out[(side, feat)] = dict(
            n=len(abs_d),
            spread=spread,
            med_abs=med_abs,
            p95_abs=p95_abs,
            med_rel=statistics.median(rel_d),
            p95_rel=pct(rel_d, 95),
            med_sigma=med_abs / spread if finite(spread) and spread > 0 else float("nan"),
            p95_sigma=p95_abs / spread if finite(spread) and spread > 0 else float("nan"),
            moved=sum(1 for d in abs_d if d != 0.0),
            nan_flip=nan_flip,
            # Attribution uses the MEAN, not the median: several features are
            # unmoved on most cells and move hugely on a few, and a median of 0
            # would label those "no driver".
            pb_share=statistics.fmean(pb_only) if pb_only else float("nan"),
            hf_share=statistics.fmean(hf_only) if hf_only else float("nan"),
            conv=converging,
            conv_n=conv_eligible,
        )
    return out


def band(p95_sigma):
    """The three groups the brief asks for, on the sigma normaliser."""
    if not finite(p95_sigma):
        return "unscored"
    if p95_sigma < 0.05:
        return "under-noise"
    if p95_sigma < 0.5:
        return "material"
    return "enormous"


def driver(s):
    pb, hf = s["pb_share"], s["hf_share"]
    if not finite(pb) and not finite(hf):
        return "?"
    pb = pb if finite(pb) else 0.0
    hf = hf if finite(hf) else 0.0
    if pb == 0.0 and hf == 0.0:
        return "none"
    if hf > 4 * pb:
        return "hf_max_blocks"
    if pb > 4 * hf:
        return "pixel_budget"
    return "both"


def decisions_report(path, fh):
    """Fold `examples/budget_decision_ab` rows into the report.

    Two things are reported per consumer per size, and the second is what stops
    the first from being misread: the **change rate**, and the **number of
    distinct decisions the consumer emits at all**. A 0 % change rate on a
    consumer that emits one constant decision is not robustness — it is a
    degenerate picker, and nothing about sampling could have moved it.
    """
    rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
    consumers = sorted({r["consumer"] for r in rows})
    sides = sorted({int(r["side"]) for r in rows})

    print("\n## Does the decision change?\n", file=fh)
    print("`distinct` = how many different decisions that consumer emitted at that "
          "size across every image and target. When it is 1, the consumer is "
          "saturated and a 0 % change rate carries no information about "
          "robustness.\n", file=fh)
    print("| consumer | side | decisions | changed | rate | distinct | most common |",
          file=fh)
    print("|---|--:|--:|--:|--:|--:|---|", file=fh)
    for c in consumers:
        for side in sides:
            sub = [r for r in rows if r["consumer"] == c and int(r["side"]) == side]
            if not sub:
                continue
            ch = sum(int(r["changed"]) for r in sub)
            dist = defaultdict(int)
            for r in sub:
                dist[r["decision_default"]] += 1
            top, topn = max(dist.items(), key=lambda kv: kv[1])
            print(f"| {c} | {side} | {len(sub)} | {ch} | {100 * ch / len(sub):.2f}% | "
                  f"{len(dist)} | {top} ({100 * topn / len(sub):.0f}%) |", file=fh)

    # Regret, where the harness produced it (zenjpeg's outputs are log-bytes).
    reg = [r for r in rows if r.get("regret") not in (None, "-", "")]
    if reg:
        print("\n### Regret on a flip\n", file=fh)
        print("The model's own predicted fractional byte penalty for the "
              "default-budget pick, priced under the fully-sampled feature vector. "
              "It is the picker's own objective, **not** a measured encode.\n",
              file=fh)
        print("| consumer | side | flips | median | p95 | max |", file=fh)
        print("|---|--:|--:|--:|--:|--:|", file=fh)
        for c in sorted({r["consumer"] for r in reg}):
            for side in sides:
                v = [abs(float(r["regret"])) for r in reg
                     if r["consumer"] == c and int(r["side"]) == side
                     and r["changed"] == "1"]
                if not v:
                    continue
                print(f"| {c} | {side} | {len(v)} | {statistics.median(v) * 100:.4f}% | "
                      f"{pct(v, 95) * 100:.4f}% | {max(v) * 100:.4f}% |", file=fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--native", help="native whole-image control TSV")
    ap.add_argument("--decisions", help="budget_decision_ab TSV")
    ap.add_argument("--out")
    ap.add_argument("--format", choices=["md", "tsv"], default="md")
    ap.add_argument("--classes", help="comma list to restrict to")
    args = ap.parse_args()

    classes = set(args.classes.split(",")) if args.classes else None
    cells = load(args.grid)
    stats = summarize(cells, classes)
    sides = sorted({s for s, _ in stats})
    feats = sorted({f for _, f in stats})
    fh = open(args.out, "w") if args.out else sys.stdout

    if args.format == "tsv":
        cols = ["side", "feature", "n", "spread", "med_abs", "p95_abs", "med_rel",
                "p95_rel", "med_sigma", "p95_sigma", "moved", "nan_flip",
                "pb_share", "hf_share", "conv", "conv_n", "band", "driver"]
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for side in sides:
            for f in feats:
                s = stats.get((side, f))
                if not s:
                    continue
                w.writerow([side, f] + [s[c] for c in cols[2:-2]] +
                           [band(s["p95_sigma"]), driver(s)])
        if args.out:
            fh.close()
        return

    print("# Budget drift: feature value vs sampling budget, by image size\n", file=fh)
    print(f"Grid: `{args.grid}`  ·  classes: "
          f"{','.join(sorted(classes)) if classes else 'all'}\n", file=fh)

    # ---- headline: how many features move, per size, per band
    print("## How many features move, by size\n", file=fh)
    print("`sigma` = |full − default| / (across-image stdev of the default value at "
          "that size) — the move measured in units of the spread a model z-scores "
          "against.\n", file=fh)
    print("| side | features | moved at all | p95σ ≥ 0.05 | p95σ ≥ 0.5 | max p95σ |",
          file=fh)
    print("|---|--:|--:|--:|--:|--:|", file=fh)
    for side in sides:
        rows = [stats[(side, f)] for f in feats if (side, f) in stats]
        moved = sum(1 for r in rows if r["moved"] > 0)
        mat = sum(1 for r in rows if finite(r["p95_sigma"]) and r["p95_sigma"] >= 0.05)
        enorm = sum(1 for r in rows if finite(r["p95_sigma"]) and r["p95_sigma"] >= 0.5)
        mx = max((r["p95_sigma"] for r in rows if finite(r["p95_sigma"])), default=float("nan"))
        print(f"| {side} | {len(rows)} | {moved} | {mat} | {enorm} | {mx:.3f} |", file=fh)

    # ---- per-feature detail at the largest size
    big = sides[-1]
    print(f"\n## Per-feature drift at {big}² (sorted by p95σ)\n", file=fh)
    print("| feature | n | med σ | p95 σ | med rel | p95 rel | med abs | driver | conv | band |",
          file=fh)
    print("|---|--:|--:|--:|--:|--:|--:|---|--:|---|", file=fh)
    rows = [(f, stats[(big, f)]) for f in feats if (big, f) in stats]
    rows.sort(key=lambda kv: (-(kv[1]["p95_sigma"] if finite(kv[1]["p95_sigma"]) else -1)))
    for f, s in rows:
        conv = f"{s['conv']}/{s['conv_n']}" if s["conv_n"] else "-"
        print(f"| {f} | {s['n']} | {s['med_sigma']:.3f} | {s['p95_sigma']:.3f} | "
              f"{s['med_rel']:.2e} | {s['p95_rel']:.2e} | {s['med_abs']:.3e} | "
              f"{driver(s)} | {conv} | {band(s['p95_sigma'])} |", file=fh)

    # ---- convergence verdict
    print("\n## Convergence along the pixel-budget ladder\n", file=fh)
    print("A feature is *monotone-converging* only if |value(arm) − value(pbfull)| is "
          "non-increasing across default → 1M → 2M → 4M → 8M → full on a cell. "
          "Design option 2 (a 'sample at least X' floor) is only sound for features "
          "that converge on essentially every cell.\n", file=fh)
    print("| side | features with drift | converge on all cells | converge on ≥95% | never |",
          file=fh)
    print("|---|--:|--:|--:|--:|", file=fh)
    for side in sides:
        rows = [stats[(side, f)] for f in feats
                if (side, f) in stats and stats[(side, f)]["moved"] > 0]
        allc = sum(1 for r in rows if r["conv_n"] and r["conv"] == r["conv_n"])
        p95c = sum(1 for r in rows if r["conv_n"] and r["conv"] / r["conv_n"] >= 0.95)
        never = sum(1 for r in rows if r["conv_n"] and r["conv"] == 0)
        print(f"| {side} | {len(rows)} | {allc} | {p95c} | {never} |", file=fh)

    if args.decisions:
        decisions_report(args.decisions, fh)

    if args.native:
        nat = summarize(load(args.native))
        print("\n## Mosaic control: native whole images vs mosaics\n", file=fh)
        print("If the mosaic inflated drift, native cells at the same side would drift "
              "less. Median over features of the per-feature median σ:\n", file=fh)
        print("| side | source | features | median of med σ | median of p95 σ |", file=fh)
        print("|---|---|--:|--:|--:|", file=fh)
        for side in sorted({s for s, _ in nat}):
            for label, table in (("native", nat), ("mosaic", stats)):
                rows = [r for (sd, _), r in table.items()
                        if sd == side and finite(r["med_sigma"])]
                if not rows:
                    continue
                print(f"| {side} | {label} | {len(rows)} | "
                      f"{statistics.median([r['med_sigma'] for r in rows]):.4f} | "
                      f"{statistics.median([r['p95_sigma'] for r in rows]):.4f} |", file=fh)

    if args.out:
        fh.close()


if __name__ == "__main__":
    main()
