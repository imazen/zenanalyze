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


def _has_ver(c):
    """True if `c` ends with a zenanalyze `@<8 hex>` version qualifier."""
    at = c.rfind("@")
    return (
        at != -1
        and len(c) - at - 1 == 8
        and all(ch in "0123456789abcdef" for ch in c[at + 1:].lower())
    )


def _is_feat(c):
    """A feature column: legacy bare `feat_<name>` OR qualified `<name>@hex8`."""
    return c.startswith("feat_") or _has_ver(c)


def _canonical(c):
    """Canonical feature name, independent of a `feat_` prefix and an `@hex8` version
    qualifier — so a bare re-extraction and a qualified one align column-for-column."""
    if c.startswith("feat_"):
        c = c[len("feat_"):]
    if _has_ver(c):
        c = c[: c.rfind("@")]
    return c.lower()


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
    expect = set(_canonical(f) for f in args.expect.split(",") if f)

    old, on = load_sorted(args.old)
    new, nn = load_sorted(args.new)

    print(f"old: {old.num_rows} rows x {len(on)} cols")
    print(f"new: {new.num_rows} rows x {len(nn)} cols")
    ok = True
    # META (non-feature) columns must match exactly; FEATURE columns match by CANONICAL
    # name so a bare `feat_X` parquet and a qualified `X@hex8` re-extraction align.
    meta_old = [c for c in on if not _is_feat(c)]
    meta_new = [c for c in nn if not _is_feat(c)]
    if meta_old != meta_new:
        print(f"!! META columns differ: only-old={set(meta_old)-set(meta_new)} "
              f"only-new={set(meta_new)-set(meta_old)}")
        ok = False
    feat_old = {_canonical(c): c for c in on if _is_feat(c)}
    feat_new = {_canonical(c): c for c in nn if _is_feat(c)}
    only_old, only_new = set(feat_old) - set(feat_new), set(feat_new) - set(feat_old)
    if only_old or only_new:
        print(f"!! FEATURE set differs (by canonical name): only-old={sorted(only_old)} "
              f"only-new={sorted(only_new)}")
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

    common = sorted(set(feat_old) & set(feat_new))
    changed = {}
    for canon in common:
        o = colvals(old, feat_old[canon]).astype(np.float64)
        n = colvals(new, feat_new[canon]).astype(np.float64)
        both_nan = np.isnan(o) & np.isnan(n)
        diff = (o != n) & ~both_nan
        nd = int(diff.sum())
        if nd:
            den = np.maximum.reduce([np.abs(o), np.abs(n), np.ones_like(o)])
            rel = np.where(diff, np.abs(o - n) / den, 0.0)
            changed[canon] = (nd, float(np.nanmax(rel)))

    print(f"\n{len(changed)} / {len(common)} features changed:")
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
    if ok:
        print(f"\nOK — {len(changed)} feature(s) changed"
              + (f" (exactly the {len(expect)} expected)" if expect else "")
              + f"; the other {len(common) - len(changed)} are byte-identical across the rename.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
