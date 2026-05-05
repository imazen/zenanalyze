#!/usr/bin/env python3
"""Per-content-class consistency check: how well does zensim correlate
with SSIM2 and butteraugli within each content class? If zensim
diverges from the other two on screens/synthetic, it's a screen-content
blind spot.

Uses v06 sweep data (no new compute needed). Bins by source path prefix:
- cid22-train, clic-1024-train, clic-train, kodak → photo
- gb82-photo → photo
- gb82-screen → screen
- kadid10k → photo (mostly Kodak refs)
- synthetic → synthetic
- size-dense-renders → photo (resampled photos)
- variants/<src>/<file> → inherit class from <src>
- gen-screen / gen-doc / gen-chart / gen-line → synthetic non-photo

Output: per-class SROCC matrix + scatter snippets where zensim disagrees most.
"""
from __future__ import annotations
import csv, json, math, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr, pearsonr

SWEEP_TSV = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/lilith/sweep-data/zenjxl_v06.tsv")
OUTPUT_REPORT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/zensim_per_class_v06.md")


def classify(src: str) -> str:
    """Map source path to content class."""
    s = src.lower()
    if s.startswith("variants/"):
        s = s.split("/", 2)[1] if "/" in s else s
    if "gb82-screen" in s or "gen-screen" in s:
        return "screen"
    if "gen-doc" in s:
        return "document"
    if "gen-chart" in s:
        return "chart"
    if "gen-line" in s:
        return "line-art"
    if s.startswith("synthetic") or "checker" in s or "noise_" in s or "thin_lines" in s or "gen-synthetic" in s:
        return "synthetic"
    if s.startswith("kadid10k"):
        return "photo"  # KADID refs are photographic
    if s.startswith(("cid22", "clic", "kodak", "gb82-photo", "size-dense-renders", "variants")):
        return "photo"
    return "other"


def main():
    rows = []
    with open(SWEEP_TSV) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        # which metric cols are present
        cols = [c for c in rdr.fieldnames if c.startswith("score_")]
        print(f"[sweep] available metrics: {cols}", file=sys.stderr)
        for r in rdr:
            try:
                if not r["encoded_bytes"] or not r["score_zensim"]:
                    continue
                # Get content class from image path
                img = r["image_path"]
                # Strip /workspace/sweep/stage-XXX/ prefix
                clean = img.split("stage-", 1)[-1].split("/", 1)[-1] if "stage-" in img else img.split("/")[-1]
                # Convert "variants__cid22-train__1277396__sz256.png" → "variants/cid22-train/..."
                clean_path = clean
                if clean.startswith("variants__"):
                    parts = clean.split("__", 2)
                    if len(parts) == 3:
                        clean_path = f"variants/{parts[1]}/{parts[2]}"
                else:
                    parts = clean.split("__", 1)
                    if len(parts) == 2:
                        clean_path = f"{parts[0]}/{parts[1]}"
                cls = classify(clean_path)
                row = {"image": clean_path, "class": cls,
                       "zensim": float(r["score_zensim"])}
                for col in cols:
                    if col != "score_zensim" and r.get(col):
                        try:
                            row[col.replace("score_", "")] = float(r[col])
                        except ValueError:
                            pass
                rows.append(row)
            except Exception:
                continue

    print(f"[sweep] {len(rows)} cells", file=sys.stderr)

    # Class distribution
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)
    print(f"[classes] {dict((k, len(v)) for k, v in by_class.items())}", file=sys.stderr)

    # For each class with >=100 samples, compute SROCC zensim vs each other metric
    other_metrics = ["ssim2", "butteraugli_max", "butteraugli_pnorm3"]

    lines = ["# zensim per-content-class consistency vs SSIM2/butteraugli — v06 sweep\n\n"]
    lines.append(f"- Source data: v06 sweep, {len(rows)} (image, knob) cells\n")
    lines.append(f"- Per-class sample counts:\n")
    for cls, n in sorted(by_class.items(), key=lambda x: -len(x[1])):
        lines.append(f"  - {cls}: {len(n)} cells\n")
    lines.append("\n")

    # Pairwise SROCC: zensim vs each metric, within each class
    lines.append("## Pairwise rank correlation (Spearman) of zensim vs other metrics, per class\n\n")
    lines.append("Higher zensim = better quality. Higher SSIM2 = better quality. Lower butter = better quality (sign-flipped to align).\n\n")
    lines.append("| class | n | zensim×ssim2 | zensim×butter_max | zensim×butter_p3 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")

    for cls, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
        if len(items) < 50:
            continue  # skip too-small classes
        z = np.array([r["zensim"] for r in items])
        out_row = f"| {cls} | {len(items)} |"
        for m in other_metrics:
            vals = []
            for r in items:
                if m in r:
                    vals.append(r[m])
                else:
                    vals.append(np.nan)
            v = np.array(vals)
            # Filter NaN
            mask = ~np.isnan(v)
            if mask.sum() < 30:
                out_row += " n/a |"
                continue
            z_clean = z[mask]; v_clean = v[mask]
            # Sign convention: butter is "lower better" so flip for correlation (so positive correlation means agreement)
            sign = -1 if m.startswith("butter") else 1
            rho, _ = spearmanr(z_clean, sign * v_clean)
            out_row += f" {rho:+.4f} |"
        lines.append(out_row + "\n")

    # CROSS-class comparison: which classes have anomalously LOW zensim correlation?
    lines.append("\n## Interpretation\n\n")
    lines.append("- High SROCC (>0.95) = zensim agrees with that metric on this class. Zensim works there.\n")
    lines.append("- Low SROCC (<0.85) = zensim diverges from that metric. Possible blind spot.\n")
    lines.append("- If photo class shows >0.95 across the board but screen/synthetic show <0.85, zensim is photo-tuned and less reliable on those classes.\n")

    # Per-class metric MAGNITUDE: are screens scored systematically differently?
    lines.append("\n## Per-class score distribution\n\n")
    lines.append("| class | n | zensim mean | zensim std | ssim2 mean | butter_max mean |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for cls, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
        if len(items) < 50:
            continue
        z = [r["zensim"] for r in items]
        s = [r["ssim2"] for r in items if "ssim2" in r]
        b = [r["butteraugli_max"] for r in items if "butteraugli_max" in r]
        zm = mean(z) if z else 0
        zsd = float(np.std(z)) if z else 0
        sm = mean(s) if s else 0
        bm = mean(b) if b else 0
        lines.append(f"| {cls} | {len(items)} | {zm:.2f} | {zsd:.2f} | {sm:.2f} | {bm:.3f} |\n")

    with open(OUTPUT_REPORT, "w") as f:
        f.writelines(lines)
    print(f"\n[wrote] {OUTPUT_REPORT}", file=sys.stderr)
    print("".join(lines))


if __name__ == "__main__":
    main()
