#!/usr/bin/env python3
"""Test the weakest claim (metric D): does the variance/spread COLLAPSE under a
representation mean LOST separability, or merely SHRUNK spread?

A monotone contractive map (e.g. a global tone-curve) shrinks std but preserves
the inter-image RANK and the pairwise-distance STRUCTURE — and a GBDT/MLP
separates images on rank/structure, not absolute spread. So the honest test of
"does tonemap destroy discriminability" is: how well does each representation
preserve the SDR baseline's inter-image ordering and geometry?

Two probes, per content feature, across the 60 images, at headroom 8:
  1. Spearman rank corr(rep, sdr)  -> does it keep WHICH images are high/low?
  2. Mantel-style corr of the pairwise-distance matrices (z-scored feature space)
     -> does it keep the overall image-to-image geometry a model would split on?

High corr => spread shrank but separability survives (std-collapse is benign).
Low  corr => the representation actually scrambles the structure (collapse is real).
"""
import sys, math, statistics as st
from collections import defaultdict
import pyarrow.parquet as pq

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/v/output/zenanalyze-feature-axes/hdr_3rep_sweep_2026-06-23.parquet"
HR = "8"      # headroom to compare at (stored as int 1/2/4/8 in the parquet)
SDR_HR = "1"  # the baseline rows carry headroom 1
META = {"stem", "content_class", "variant", "headroom", "width", "height"}

t = pq.read_table(PATH)
cols = t.column_names
rows = [dict(zip(cols, r)) for r in zip(*[t.column(c).to_pylist() for c in cols])]
feat_cols = [c for c in cols if c not in META]

def hk(r):  # normalize headroom -> "1"/"2"/"4"/"8" regardless of int/float storage
    return str(int(float(r["headroom"])))

# index stem -> variant -> headroom -> row
idx = defaultdict(lambda: defaultdict(dict))
for r in rows:
    idx[r["stem"]][r["variant"]][hk(r)] = r

def fval(r, c):
    try: return float(r[c])
    except (TypeError, ValueError): return float("nan")

# content cols = non-zero in every SDR baseline (drop HDR-only depth outputs)
sdr_rows = [v["sdr"][SDR_HR] for v in idx.values() if "sdr" in v and SDR_HR in v["sdr"]]
content_cols = [c for c in feat_cols
                if not all(abs(fval(r, c)) < 1e-9 for r in sdr_rows)]

stems = [s for s, v in idx.items()
         if "sdr" in v and SDR_HR in v["sdr"]
         and all(rp in v and HR in v[rp] for rp in ("hdr", "hdr_clip", "hdr_tonemap"))]
print(f"n images={len(stems)}, content features={len(content_cols)}, headroom={HR}")

def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def pearson(a, b):
    n = len(a)
    if n < 2: return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0: return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)

def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))

REPS = ["hdr", "hdr_clip", "hdr_tonemap"]

# --- Probe 1: per-feature Spearman(rep, sdr), median + how many features "scramble"
print("\n=== Probe 1: median per-feature Spearman rank corr vs SDR (across images) ===")
print("    (1.0 = identical ordering; <0.8 = ordering meaningfully scrambled)")
print(f"{'rep':>14s}{'median rho':>12s}{'min rho':>10s}{'feats rho<0.8':>15s}")
sp_by_rep = {}
for rp in REPS:
    rhos = []
    for c in content_cols:
        sdr_x = [fval(idx[s]["sdr"][SDR_HR], c) for s in stems]
        rep_x = [fval(idx[s][rp][HR], c) for s in stems]
        if any(math.isnan(v) for v in sdr_x + rep_x): continue
        rho = spearman(sdr_x, rep_x)
        if not math.isnan(rho): rhos.append((rho, c))
    sp_by_rep[rp] = rhos
    vals = [r for r, _ in rhos]
    nscr = sum(1 for r in vals if r < 0.8)
    print(f"{rp:>14s}{st.median(vals):>12.4f}{min(vals):>10.4f}{nscr:>12d}/{len(vals)}")

# --- Probe 2: Mantel-style pairwise-distance-matrix correlation vs SDR
# z-score each content feature by the SDR baseline's mean/std, build per-image
# vectors, compute pairwise Euclidean distances, correlate upper triangles.
print("\n=== Probe 2: pairwise inter-image distance-matrix corr vs SDR (z-scored) ===")
print("    (1.0 = same image-to-image geometry a model would split on)")
# standardization stats from SDR baseline
stat = {}
for c in content_cols:
    xs = [fval(idx[s]["sdr"][SDR_HR], c) for s in stems]
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2: continue
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    if sd > 0: stat[c] = (m, sd)
zc = list(stat.keys())

def zvec(r):
    return [(fval(r, c) - stat[c][0]) / stat[c][1] for c in zc]

def distmat_triu(vecs):
    n = len(vecs)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(sum((vecs[i][k] - vecs[j][k]) ** 2 for k in range(len(vecs[i]))))
            out.append(d)
    return out

sdr_vecs = [zvec(idx[s]["sdr"][SDR_HR]) for s in stems]
sdr_tri = distmat_triu(sdr_vecs)
print(f"{'rep':>14s}{'dist-mat corr':>15s}")
for rp in REPS:
    rep_vecs = [zvec(idx[s][rp][HR]) for s in stems]
    rep_tri = distmat_triu(rep_vecs)
    print(f"{rp:>14s}{pearson(sdr_tri, rep_tri):>15.4f}")

# --- the 3 worst-scrambled features for tonemap (where ordering breaks most)
print("\n=== tonemap's most-scrambled features (lowest Spearman vs SDR) ===")
worst = sorted(sp_by_rep["hdr_tonemap"])[:6]
for rho, c in worst:
    print(f"  {rho:+.3f}  {c}")
