#!/usr/bin/env python3
"""v10 multi-codec analyzer — per (image, target zensim band), which codec wins
on bytes? Reports head-to-head bytes ratios and per-class codec preferences.

Pareto: for each image and each target_zensim band (e.g. 70, 75, 80, 85),
find the smallest encode (across all codecs and all knob settings) that
hits the target zensim. Output: codec routing table.
"""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, median

DATA = {
    "zenjxl": Path("/home/lilith/sweep-data/v10_zenjxl.tsv"),
    "zenavif": Path("/home/lilith/sweep-data/v10_zenavif.tsv"),
    "zenwebp": Path("/home/lilith/sweep-data/v10_zenwebp.tsv"),
}
OUTPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/picker_v10_multicodec_report.md")

# Quality bands: float zensim midpoint
BANDS = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL = 1.5  # ±1.5 zensim from midpoint


def load(path: Path, codec: str):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            try:
                if not r.get('encoded_bytes') or not r.get('score_zensim'):
                    continue
                rows.append({
                    'codec': codec,
                    'image': r['image_path'].rsplit('/', 1)[-1],
                    'bytes': int(r['encoded_bytes']),
                    'ms': float(r['encode_ms']),
                    'zensim': float(r['score_zensim']),
                })
            except Exception:
                continue
    return rows


def main():
    all_rows = []
    for codec, path in DATA.items():
        loaded = load(path, codec)
        all_rows.extend(loaded)
        print(f"[load] {codec}: {len(loaded)} rows", file=sys.stderr)

    # Group by image
    by_image = defaultdict(list)
    for r in all_rows:
        by_image[r['image']].append(r)
    print(f"[load] {len(by_image)} unique images", file=sys.stderr)

    # For each image, find best (smallest bytes) per (codec, band)
    # then compare codecs head-to-head.
    head_to_head = defaultdict(lambda: defaultdict(list))  # band -> codec -> [bytes]
    codec_winner_count = defaultdict(Counter)              # band -> Counter(codec)
    head_to_head_pairs = defaultdict(list)                 # band -> [(jxl, avif, webp)]

    for image, rows in by_image.items():
        for band in BANDS:
            best_per_codec = {}  # codec -> bytes
            for r in rows:
                if abs(r['zensim'] - band) <= BAND_TOL:
                    if r['codec'] not in best_per_codec or r['bytes'] < best_per_codec[r['codec']]:
                        best_per_codec[r['codec']] = r['bytes']
            if len(best_per_codec) < 2:
                continue  # need at least two codecs to compare
            # Record per-codec bytes
            for codec, b in best_per_codec.items():
                head_to_head[band][codec].append(b)
            # Find winner
            winner = min(best_per_codec, key=best_per_codec.get)
            codec_winner_count[band][winner] += 1
            # Pair record (only if all 3 present)
            if len(best_per_codec) == 3:
                head_to_head_pairs[band].append({
                    'image': image,
                    'jxl': best_per_codec.get('zenjxl', 0),
                    'avif': best_per_codec.get('zenavif', 0),
                    'webp': best_per_codec.get('zenwebp', 0),
                })

    # ── Report ──
    lines = ["# v10 multi-codec picker analysis (zenjxl / zenavif / zenwebp)\n\n"]
    lines.append(f"- Source: v10 sweep, {len(all_rows)} rows, {len(by_image)} unique images\n")
    lines.append(f"- Bands: zensim midpoints {BANDS} ± {BAND_TOL}\n\n")

    lines.append("## Per-band winner distribution\n\n")
    lines.append("Number of images where each codec produced the smallest encode meeting the band.\n\n")
    lines.append("| band | n_images | jxl wins | avif wins | webp wins | jxl% | avif% | webp% |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for band in BANDS:
        c = codec_winner_count[band]
        total = sum(c.values())
        if total == 0: continue
        jxl, avif, webp = c.get('zenjxl', 0), c.get('zenavif', 0), c.get('zenwebp', 0)
        lines.append(
            f"| {band:.0f} | {total} | {jxl} | {avif} | {webp} | "
            f"{100*jxl/total:.1f}% | {100*avif/total:.1f}% | {100*webp/total:.1f}% |\n"
        )

    lines.append("\n## Median bytes per band (across images with all 3 codecs)\n\n")
    lines.append("| band | n_imgs | jxl med | avif med | webp med | jxl/avif | jxl/webp | avif/webp |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for band in BANDS:
        pairs = head_to_head_pairs[band]
        if not pairs: continue
        jxls = sorted(p['jxl'] for p in pairs)
        avifs = sorted(p['avif'] for p in pairs)
        webps = sorted(p['webp'] for p in pairs)
        n = len(pairs)
        med = lambda xs: xs[len(xs)//2]
        jm, am, wm = med(jxls), med(avifs), med(webps)
        lines.append(
            f"| {band:.0f} | {n} | {jm} | {am} | {wm} | {jm/am:.3f} | {jm/wm:.3f} | {am/wm:.3f} |\n"
        )

    lines.append("\n## Mean Δbytes vs single-codec baseline (lower = better)\n\n")
    lines.append("If we always picked the SAME codec for every image, what would mean bytes be?\n")
    lines.append("vs the multi-codec oracle (best per image)?\n\n")
    lines.append("| band | n | jxl-only | avif-only | webp-only | oracle | oracle vs best-single |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for band in BANDS:
        pairs = head_to_head_pairs[band]
        if not pairs: continue
        jxl_mean = mean(p['jxl'] for p in pairs)
        avif_mean = mean(p['avif'] for p in pairs)
        webp_mean = mean(p['webp'] for p in pairs)
        oracle_mean = mean(min(p['jxl'], p['avif'], p['webp']) for p in pairs)
        best_single = min(jxl_mean, avif_mean, webp_mean)
        delta = (oracle_mean / best_single - 1) * 100
        lines.append(
            f"| {band:.0f} | {len(pairs)} | {jxl_mean:.0f} | {avif_mean:.0f} | {webp_mean:.0f} | "
            f"{oracle_mean:.0f} | {delta:+.2f}% |\n"
        )

    lines.append("\n## Interpretation\n\n")
    lines.append("- 'Oracle vs best-single' shows how much we'd save with a perfect per-image codec router.\n")
    lines.append("- Negative % means routing helps (oracle smaller than always picking the same codec).\n")
    lines.append("- A small gap (~0-2%) means one codec dominates this band; a large gap (5%+) means routing matters.\n")

    with open(OUTPUT, 'w') as f:
        f.writelines(lines)
    print(f"[wrote] {OUTPUT}", file=sys.stderr)
    print(''.join(lines))


if __name__ == '__main__':
    main()
