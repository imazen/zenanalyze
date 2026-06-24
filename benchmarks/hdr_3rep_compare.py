#!/usr/bin/env python3
"""Compare the 3 HDR feature representations (extend / clip / tonemap).

Question: which representation should zenanalyze offer for HDR content?

Design of the sweep: each SDR image is turned into a synthetic-PQ HDR image by
pushing its top `highlight-pct` pixels into super-white up to `headroom`x diffuse
white. We then extract features under 3 representations and compare each to the
SDR baseline for the SAME image:

  extend   - linear-light, super-white extends unbounded (NOT clipped)
  clip     - linear-light + with_diffuse_white_clip (super-white hard-clipped)
  tonemap  - BT.2446A SDR-display rendition via zenpixels-convert

Metrics:
  A. content-feature drift from SDR baseline (invariance)  -> lower = more SDR-like
  B. headroom stability (drift slope as headroom grows)    -> flatter = more stable
  C. highlight-descriptor carriage (do the 6 hl_* fire?)
  D. cross-image discriminability (does the rep preserve content separation?)
"""
import sys, csv, math, statistics as st
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/v/output/zenanalyze-feature-axes/hdr_3rep_sweep_2026-06-23.tsv"
META = {"stem", "content_class", "variant", "headroom", "width", "height"}

rows = list(csv.DictReader(open(PATH), delimiter="\t"))
feat_cols = [c for c in rows[0] if c not in META]
hl_cols = [c for c in feat_cols if "highlight" in c]
print(f"loaded {len(rows)} rows, {len(feat_cols)} feature cols, {len(hl_cols)} highlight cols")

def f(r, c):
    try: return float(r[c])
    except (ValueError, KeyError): return float("nan")

# index: stem -> variant -> headroom -> row
idx = defaultdict(lambda: defaultdict(dict))
for r in rows:
    idx[r["stem"]][r["variant"]][r["headroom"]] = r

# Detect HDR-only feature cols: ~0 for every SDR baseline row (depth-tier outputs).
# Exclude them from the *content* invariance metric (they don't exist in SDR).
sdr_rows = [v["sdr"]["1"] for v in idx.values() if "sdr" in v and "1" in v["sdr"]]
hdr_only = set()
for c in feat_cols:
    if all(abs(f(r, c)) < 1e-9 for r in sdr_rows):
        hdr_only.add(c)
content_cols = [c for c in feat_cols if c not in hdr_only]
print(f"content cols (shared w/ SDR): {len(content_cols)}; HDR-only (excluded): {len(hdr_only)}")
print(f"  HDR-only examples: {sorted(hdr_only)[:8]}")

def rel_drift(r_test, r_base, cols):
    """mean over cols of |test - base| / (|base| + scale)."""
    ds = []
    for c in cols:
        a, b = f(r_test, c), f(r_base, c)
        if math.isnan(a) or math.isnan(b): continue
        denom = abs(b) + 1e-6 + 0.01 * abs(a)  # relative w/ small absolute floor
        ds.append(abs(a - b) / denom)
    return st.median(ds) if ds else float("nan")

REPS = ["hdr", "hdr_clip", "hdr_tonemap"]
HRS = sorted({r["headroom"] for r in rows if r["variant"] in REPS}, key=float)

print("\n=== A/B. Content-feature drift from SDR baseline (median over images) ===")
print("     (lower = more SDR-invariant; flat across headroom = more stable)")
hdr = f"{'headroom':>10s} " + "".join(f"{rp:>14s}" for rp in REPS)
print(hdr)
for h in HRS:
    cells = []
    for rp in REPS:
        ds = []
        for stem, vv in idx.items():
            if "sdr" not in vv or "1" not in vv["sdr"]: continue
            if rp not in vv or h not in vv[rp]: continue
            ds.append(rel_drift(vv[rp][h], vv["sdr"]["1"], content_cols))
        cells.append(st.median(ds) if ds else float("nan"))
    print(f"{h:>10s} " + "".join(f"{c:>14.4f}" for c in cells))

print("\n=== C. Highlight-descriptor carriage (mean over all rep rows) ===")
print("     (expect: hdr==hdr_clip carry the source highlight signal; tonemap=0)")
for rp in REPS:
    vals = {c: [] for c in hl_cols}
    for stem, vv in idx.items():
        if rp not in vv: continue
        for h, r in vv[rp].items():
            for c in hl_cols:
                x = f(r, c)
                if not math.isnan(x): vals[c].append(x)
    print(f"  {rp:12s} " + "  ".join(f"{c.replace('feat_highlight_','hl_')}={st.mean(v) if v else 0:.3f}" for c, v in vals.items()))

# Verify hdr vs hdr_clip highlight identity (clip must not touch the depth tier).
print("\n=== C'. hdr vs hdr_clip highlight-descriptor identity (max abs diff) ===")
mx = 0.0
for stem, vv in idx.items():
    if "hdr" not in vv or "hdr_clip" not in vv: continue
    for h in vv["hdr"]:
        if h not in vv["hdr_clip"]: continue
        for c in hl_cols:
            d = abs(f(vv["hdr"][h], c) - f(vv["hdr_clip"][h], c))
            if not math.isnan(d): mx = max(mx, d)
print(f"  max |hdr - hdr_clip| over highlight cols = {mx:.2e}  (should be ~0: clip doesn't touch depth tier)")

print("\n=== D. Cross-image discriminability at headroom=max (std over images) ===")
print("     (collapse toward 0 = rep is washing out content differences)")
hmax = HRS[-1]
key_feats = ["feat_variance", "feat_edge_density", "feat_chroma_complexity",
             "feat_dct_energy" if any("dct_energy" in c for c in feat_cols) else feat_cols[6]]
key_feats = [c for c in key_feats if c in feat_cols]
print(f"{'feature':>26s} " + "".join(f"{rp:>14s}" for rp in ["sdr"] + REPS))
for c in key_feats:
    cells = []
    for rp in ["sdr"] + REPS:
        h = "1" if rp == "sdr" else hmax
        xs = [f(vv[rp][h], c) for vv in idx.values() if rp in vv and h in vv[rp]]
        xs = [x for x in xs if not math.isnan(x)]
        cells.append(st.pstdev(xs) if len(xs) > 1 else float("nan"))
    print(f"{c:>26s} " + "".join(f"{v:>14.2f}" for v in cells))
