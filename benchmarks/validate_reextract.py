#!/usr/bin/env python3
"""Validate a feature re-extraction against a prior parquet: confirm ONLY the
expected features changed (and everything else is byte-identical), aligned by the
content key so row order can't confound it.

Usage:
  validate_reextract.py OLD.parquet NEW.parquet [--expect feat_a,feat_b,...]

Exit 0 if the set of features that actually changed == the --expect set (and the
shapes/keys line up); exit 1 otherwise. Prints a per-changed-feature summary
(rows changed, max relative deviation).
"""
import sys
import argparse
import numpy as np
import pyarrow.parquet as pq

KEY = ["image_sha", "crop_label", "size_class", "width", "height"]


def load_sorted(path):
    t = pq.read_table(path)
    names = t.column_names
    # Stable sort by the content key so old/new align regardless of thread order.
    import pyarrow.compute as pc

    idx = pc.sort_indices(t, sort_keys=[(k, "ascending") for k in KEY if k in names])
    return t.take(idx), names


def colvals(t, name):
    return t.column(name).to_numpy(zero_copy_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--expect", default="", help="comma-separated features expected to change")
    args = ap.parse_args()
    expect = set(f for f in args.expect.split(",") if f)

    old, on = load_sorted(args.old)
    new, nn = load_sorted(args.new)

    print(f"old: {old.num_rows} rows x {len(on)} cols")
    print(f"new: {new.num_rows} rows x {len(nn)} cols")
    ok = True
    if on != nn:
        print(f"!! SCHEMA differs: only-old={set(on)-set(nn)} only-new={set(nn)-set(on)}")
        ok = False
    if old.num_rows != new.num_rows:
        print(f"!! ROW COUNT differs: {old.num_rows} vs {new.num_rows}")
        ok = False

    # Keys must align row-for-row after the sort.
    for k in KEY:
        if k in on and k in nn:
            if not np.array_equal(colvals(old, k), colvals(new, k)):
                print(f"!! KEY column {k} not aligned after sort — cannot compare")
                return 2

    feats = [c for c in on if c.startswith("feat_")]
    changed = {}
    for c in feats:
        o = colvals(old, c).astype(np.float64)
        n = colvals(new, c).astype(np.float64)
        both_nan = np.isnan(o) & np.isnan(n)
        diff = (o != n) & ~both_nan
        nd = int(diff.sum())
        if nd:
            den = np.maximum.reduce([np.abs(o), np.abs(n), np.ones_like(o)])
            rel = np.where(diff, np.abs(o - n) / den, 0.0)
            changed[c] = (nd, float(np.nanmax(rel)))

    print(f"\n{len(changed)} / {len(feats)} features changed:")
    for c, (nd, mx) in sorted(changed.items(), key=lambda kv: -kv[1][1]):
        tag = "expected" if c in expect else "!! UNEXPECTED"
        print(f"  {c:<34} {nd:>8} rows ({100*nd/old.num_rows:5.1f}%)  max-rel {mx:8.4%}  [{tag}]")

    got = set(changed)
    missing = expect - got
    unexpected = got - expect
    if missing:
        print(f"\n!! EXPECTED but did NOT change: {sorted(missing)}")
        ok = False
    if unexpected:
        print(f"\n!! UNEXPECTED changes (drift?): {sorted(unexpected)}")
        ok = False
    if ok and expect:
        print(f"\nOK — exactly the {len(expect)} expected features changed; the other "
              f"{len(feats)-len(changed)} are byte-identical.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
