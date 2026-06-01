#!/usr/bin/env python3
"""Dial-quality + byte-efficiency characterization for the zenjpeg Profile-A dial.

From the picker sweep (per image × 36 cells × 29 q: encoded_bytes + zensim-A),
for a grid of zensim-A targets T, per image:
  - ORACLE = min-bytes (cell,q) with zensim-A >= T  (the least-byte-waste optimum)
  - achieved zensim at that point (q-granularity precision: achieved - T)
  - a NAIVE fixed-config baseline (4:2:0, effort 1, no prog/sharp) for contrast
Reports, per target, the cross-image distribution (mean/std/p50/p75/p90/p95) of
bytes (the "byte spread") and quality-hit error, plus a calibration Z-RMSE.

Bake-independent (characterizes the achievable frontier + the metric's behavior);
the picker-vs-oracle waste layer is added separately once a bake is chosen.
"""
import pyarrow.parquet as pq, numpy as np, json, collections, sys

PK = '/mnt/v/zen/picker-dense-full-2026-05-27/parquet/picker_dense_full_zenjpeg_A_sourcefeat_FIXED.parquet'
t = pq.read_table(PK, columns=['image_basename','q','knob_tuple_json','encoded_bytes','score_zensim'])
img = np.array(t.column('image_basename').to_pylist())
bytes_ = np.asarray(t.column('encoded_bytes').to_pylist(), float)
zser = np.asarray(t.column('score_zensim').to_pylist(), float)
knob = np.array(t.column('knob_tuple_json').to_pylist())
NAIVE = '{"effort":1,"progressive":false,"sharp_yuv":false,"subsampling":"420"}'

images = sorted(set(img.tolist()))
# index rows per image
by_img = collections.defaultdict(list)
for i in range(len(img)):
    by_img[img[i]].append(i)

def pctl(a): 
    a=np.asarray(a,float); 
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
               p50=float(np.percentile(a,50)), p75=float(np.percentile(a,75)),
               p90=float(np.percentile(a,90)), p95=float(np.percentile(a,95)))

targets = [30,40,50,55,60,65,70,75,80,85,90]
rows=[]
for T in targets:
    orc_bytes=[]; naive_bytes=[]; qerr=[]; reach_frac=0; n=0
    for im in images:
        idx = by_img[im]
        z = zser[idx]; b = bytes_[idx]; k = knob[idx]
        reach = z >= T
        n += 1
        if reach.any():
            j = idx[np.where(reach)[0][np.argmin(b[reach])]]   # min-byte reaching cell/q
            orc_bytes.append(bytes_[j]); qerr.append(zser[j]-T); reach_frac+=1
        # naive baseline: same cell, min-byte q reaching T
        nm = (k==NAIVE) & reach
        if nm.any():
            naive_bytes.append(b[np.where(nm)[0][np.argmin(b[nm])]])
    if not orc_bytes: continue
    ob=pctl(orc_bytes); ne=pctl(naive_bytes) if naive_bytes else None; qe=pctl(qerr)
    # Z-RMSE-style calibration of achieved vs target (per-image sigma=corpus std)
    qarr=np.asarray(qerr); zr=float(np.sqrt((qarr**2).mean()))
    rows.append((T, reach_frac/n, ob, ne, qe, zr))
    waste = (ne['p50']/ob['p50']-1)*100 if ne else float('nan')
    print(f"T={T:3d}  reach={reach_frac/n:4.0%}  oracle_bytes p50={ob['p50']:8.0f} ±{ob['std']:7.0f} (p75={ob['p75']:.0f} p95={ob['p95']:.0f})  "
          f"qhit_err mean={qe['mean']:+5.2f}±{qe['std']:4.2f} (RMSE {zr:4.2f})  naive_vs_oracle_waste@p50={waste:+5.1f}%")

import pickle
pickle.dump(rows, open('/mnt/v/zen/dial-quality-2026-06-01/oracle_frontier.pkl','wb'))
print("\nsaved oracle_frontier.pkl")
