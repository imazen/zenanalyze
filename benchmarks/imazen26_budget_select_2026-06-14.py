#!/usr/bin/env python3
"""Budget-first, diversity-maximizing rendition selection for the imazen-26
training corpus.

- Eligible = images whose leading filename number is EVEN (odd = hold-out);
  variants keep the id so the split propagates.
- Coverage floor: a ~thumbnail of every eligible image (cheap, guarantees every
  image appears at least once).
- Diversity spend: greedy farthest-point sampling (FPS) over even-id full-image
  RESIZE renditions in zenanalyze feature space, adding the most-distinct
  rendition each step. FPS is incremental, so the emitted order IS the priority
  ranking — ANY gigapixel budget is a prefix (truncate at the cumulative-GP you
  can afford). Coverage (p95 distance from the whole even pool to the selected
  set) is logged at GP milestones.

Resizes only (crops are a lower-priority second pass). The existing Lanczos3
features are a selection proxy; actual renders use zenresize Mitchell-sharp.
Variant op-chain filename: `<id>.scale<W>x<H>`  (id preserved).
"""
import argparse
import os
import re
import sys
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def leadnum(p):
    m = re.match(r"^(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-budget-gp", type=float, default=2.0)
    ap.add_argument("--milestones", default="0.25,0.5,1.0,1.5,2.0")
    ap.add_argument("--thumb-max", type=int, default=128)
    ap.add_argument("--out", default="/mnt/v/output/imazen-26-features/imazen26_train_variants_2026-06-14.tsv")
    a = ap.parse_args()

    pf = pq.ParquetFile(a.parquet)
    feats = [n for n in pf.schema.names if n.startswith("feat_")]
    t = pq.read_table(a.parquet,
                      columns=["image_path", "crop_label", "size_class", "width", "height", "content_class"] + feats)
    t = t.filter(pc.equal(t["crop_label"], "full"))  # full-image resizes only
    paths = np.array(t["image_path"].to_pylist())
    w = t["width"].to_numpy().astype(np.int64)
    h = t["height"].to_numpy().astype(np.int64)
    cc = np.array(t["content_class"].to_pylist())
    ev = np.array([(lambda n: n is not None and n % 2 == 0)(leadnum(p)) for p in paths])

    M = np.empty((t.num_rows, len(feats)), np.float64)
    for j, name in enumerate(feats):
        col = t[name].to_numpy(zero_copy_only=False).astype(np.float64)
        med = np.nanmedian(col)
        med = med if np.isfinite(med) else 0.0
        M[:, j] = np.where(np.isfinite(col), col, med)

    M, paths, w, h, cc = M[ev], paths[ev], w[ev], h[ev], cc[ev]
    keep = M.std(0) > 1e-9
    M = M[:, keep]
    mu, sd = M.mean(0), M.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (M - mu) / sd
    mp = (w * h) / 1e6
    maxdim = np.maximum(w, h)
    N = len(paths)
    print(f"# even-id full-image resize candidates: {N}  ({len(np.unique(paths))} images)  feats={Z.shape[1]}", file=sys.stderr)

    # ---- thumbnail floor: per image, the rendition nearest thumb_max from below
    sel = np.zeros(N, bool)
    order = []  # (idx, cumulative_mp_after)
    by_img = {}
    for i in range(N):
        by_img.setdefault(paths[i], []).append(i)
    cum = 0.0
    for img, idxs in by_img.items():
        idxs = sorted(idxs, key=lambda i: maxdim[i])
        # largest maxdim <= thumb_max, else smallest available
        pick = next((i for i in reversed(idxs) if maxdim[i] <= a.thumb_max), idxs[0])
        sel[pick] = True
        cum += mp[pick]
        order.append((pick, cum))
    floor_mp = cum
    print(f"# thumbnail floor: {sel.sum()} renditions, {floor_mp:.1f} MP", file=sys.stderr)

    # ---- greedy FPS to max budget; record FPS order + cumulative MP
    # min distance from every candidate to the current selected set
    mind = np.full(N, np.inf)
    selidx = np.where(sel)[0]
    for c in selidx:
        d = np.linalg.norm(Z - Z[c], axis=1)
        np.minimum(mind, d, out=mind)
    budget = a.max_budget_gp * 1000.0
    while cum < budget:
        affordable = (~sel) & (mp <= (budget - cum))
        if not affordable.any():
            break
        cand = np.where(affordable, mind, -1.0)
        i = int(cand.argmax())
        if mind[i] <= 0:
            break
        sel[i] = True
        cum += mp[i]
        order.append((i, cum))
        np.minimum(mind, np.linalg.norm(Z - Z[i], axis=1), out=mind)

    # ---- coverage at milestones: p95 of every even candidate's distance to the
    # selected prefix reaching that GP
    miles = [float(x) for x in a.milestones.split(",")]
    print("budget_gp\tvariants\tcoverage_p50\tcoverage_p95\tcoverage_max", file=sys.stderr)
    for B in miles:
        Bmp = B * 1000.0
        chosen = [idx for idx, c in order if c <= Bmp]
        if not chosen:
            continue
        cs = np.array(chosen)
        # distance of all N candidates to the chosen prefix (min over chosen)
        md = np.full(N, np.inf)
        # chunked to bound memory
        for c in cs:
            np.minimum(md, np.linalg.norm(Z - Z[c], axis=1), out=md)
        print(f"{B}\t{len(chosen)}\t{np.percentile(md,50):.3f}\t{np.percentile(md,95):.3f}\t{md.max():.3f}", file=sys.stderr)

    # ---- emit the full FPS-ordered variant manifest (truncate at any GP)
    with open(a.out, "w") as f:
        f.write("rank\tcumulative_gp\timage_path\tscale_w\tscale_h\tmegapixels\tcontent_class\tvariant_name\n")
        for rank, (i, c) in enumerate(order):
            stem = re.sub(r"\.[^.]+$", "", os.path.basename(paths[i]))
            num = leadnum(paths[i])
            vname = f"{num}.scale{int(w[i])}x{int(h[i])}"
            f.write(f"{rank}\t{c/1000.0:.4f}\t{paths[i]}\t{int(w[i])}\t{int(h[i])}\t{mp[i]:.4f}\t{cc[i]}\t{vname}\n")
    print(f"# wrote {len(order)} ordered variants -> {a.out} (truncate at any cumulative_gp)", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
