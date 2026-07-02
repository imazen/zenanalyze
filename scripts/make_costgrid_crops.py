#!/usr/bin/env python3
"""Build the real-photo crop set for examples/per_tier_cost_grid.rs.

Sweep-discipline sizes {64, 256, 1024, 2048, 4096} squared, 4 distinct crops per
size so content isn't constant. Deterministic (sorted file walks, fixed offsets;
no RNG) — rerunning reproduces the set byte-for-byte from the same corpora.

Sources (real content, NOT synthetic gradients):
  64/256   crops of clean-picker-corpus-2026-06-26 renditions (web-corpus PNGs)
  1024/2048 center crops of MIT-Adobe FiveK expert_c photos (min dim >= size)
  4096     2x2 mosaics of four DISTINCT FiveK 2048^2 crops each (no single
           source photo has min dim >= 4096; a mosaic keeps every pixel real
           photo content at the cost of two seam lines per axis)

Output: /mnt/v/output/zenanalyze/costgrid-crops-2026-07-02/<side>/c<i>.png
plus _MANIFEST.tsv (source path, offsets, sha256) in the root.
"""

import glob
import hashlib
import os

from PIL import Image

OUT = "/mnt/v/output/zenanalyze/costgrid-crops-2026-07-02"
PICKER = "/mnt/v/output/clean-picker-corpus-2026-06-26"
FIVEK = "/mnt/v/input/fivek/expert_c"
N_PER_SIZE = 4

manifest = []


def crop_center(img, side, dx=0, dy=0):
    w, h = img.size
    x = (w - side) // 2 + dx
    y = (h - side) // 2 + dy
    x = max(0, min(x, w - side))
    y = max(0, min(y, h - side))
    return img.crop((x, y, x + side, y + side)), x, y


def save(img, side, idx, srcdesc, x, y):
    d = os.path.join(OUT, str(side))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"c{idx}.png")
    img.save(p, "PNG")
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    manifest.append((str(side), f"c{idx}.png", srcdesc, str(x), str(y), sha))
    print(f"{side:5d} c{idx}: {srcdesc} @({x},{y}) sha={sha}")


def picker_sources(min_dim, n, stride=97):
    """Every n-th eligible rendition from the sorted corpus (distinct origins)."""
    files = sorted(glob.glob(os.path.join(PICKER, "*.png")))
    out, seen_origin, i = [], set(), 0
    while len(out) < n and i < len(files):
        f = files[i]
        i += stride
        origin = os.path.basename(f).split(".")[0]
        if origin in seen_origin:
            continue
        im = Image.open(f)
        if min(im.size) >= min_dim and im.mode == "RGB":
            out.append(f)
            seen_origin.add(origin)
    assert len(out) == n, f"only {len(out)} picker sources >= {min_dim}"
    return out


def fivek_sources(min_dim, n, start=0, stride=211):
    files = sorted(glob.glob(os.path.join(FIVEK, "*.jpg")))
    out, i = [], start
    while len(out) < n and i < start + stride * 400:
        f = files[i % len(files)]
        i += stride
        im = Image.open(f)
        if min(im.size) >= min_dim and im.mode == "RGB" and f not in out:
            out.append(f)
    assert len(out) == n, f"only {len(out)} fivek sources >= {min_dim}"
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    # 64 + 256: web-corpus renditions
    for side in (64, 256):
        for idx, f in enumerate(picker_sources(side + 32, N_PER_SIZE)):
            img = Image.open(f).convert("RGB")
            c, x, y = crop_center(img, side, dx=(idx - 1) * side // 3)
            save(c, side, idx, os.path.relpath(f, "/mnt/v"), x, y)
    # 1024 + 2048: fivek photos
    for side, start in ((1024, 0), (2048, 13)):
        for idx, f in enumerate(fivek_sources(side, N_PER_SIZE, start=start)):
            img = Image.open(f).convert("RGB")
            c, x, y = crop_center(img, side)
            save(c, side, idx, os.path.relpath(f, "/mnt/v"), x, y)
    # 4096: 2x2 mosaics of distinct 2048^2 fivek crops (16 distinct photos)
    srcs = fivek_sources(2048, N_PER_SIZE * 4, start=101, stride=173)
    for idx in range(N_PER_SIZE):
        canvas = Image.new("RGB", (4096, 4096))
        parts = []
        for j, f in enumerate(srcs[idx * 4 : idx * 4 + 4]):
            img = Image.open(f).convert("RGB")
            c, _, _ = crop_center(img, 2048)
            canvas.paste(c, ((j % 2) * 2048, (j // 2) * 2048))
            parts.append(os.path.basename(f))
        save(canvas, 4096, idx, "mosaic4:" + "+".join(parts), 0, 0)

    with open(os.path.join(OUT, "_MANIFEST.tsv"), "w") as f:
        f.write("side\tfile\tsource\tx\ty\tsha256_16\n")
        for row in manifest:
            f.write("\t".join(row) + "\n")
    print(f"manifest: {os.path.join(OUT, '_MANIFEST.tsv')} ({len(manifest)} crops)")


if __name__ == "__main__":
    main()
