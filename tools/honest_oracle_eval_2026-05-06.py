#!/usr/bin/env python3
"""Honest production-shaped meta-picker evaluation.

The v0.3 holdout filters to cells where >=2 codecs landed in the band.
That biases winners toward "codec-mix" scenarios. Real production is:
  for image X at target_distance=T (or target_zensim=Z), pick the codec.

For each (image, band), compute:
  best_per_codec = min bytes across all codec configs reaching the band
  
Then compare:
  - always-X strategy: forced bytes_X (largest if X didn't reach band)
  - meta-picker: predicted codec's bytes
  - oracle: min bytes across codecs

Aggregate over all holdout (image, band) cells, NOT filtered to 'multi-codec'.
"""
from __future__ import annotations
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

V10_DIR = Path("/home/lilith/sweep-data/v10")
V12_DIR = Path("/tmp/v12-sweep-data")
BANDS = [70.0, 75.0, 80.0, 85.0, 90.0]
BAND_TOL = 1.5
SEED = 7

def load_dir(root):
    rows = []
    for codec in ['zenjxl', 'zenavif', 'zenwebp']:
        d = root / codec
        if not d.exists(): continue
        for tsv in d.glob('*.tsv'):
            with open(tsv) as f:
                rdr = csv.DictReader(f, delimiter='\t')
                for r in rdr:
                    try:
                        if not r.get('encoded_bytes') or not r.get('score_zensim'): continue
                        rows.append({
                            'codec': codec,
                            'image': r['image_path'].rsplit('/', 1)[-1],
                            'bytes': int(r['encoded_bytes']),
                            'zensim': float(r['score_zensim']),
                        })
                    except: pass
    return rows

all_rows = load_dir(V10_DIR) + load_dir(V12_DIR)
print(f"[load] {len(all_rows)} rows", file=sys.stderr)

# For each (image, band, codec), find min bytes
by_id = defaultdict(dict)
for r in all_rows:
    for band in BANDS:
        if abs(r['zensim'] - band) <= BAND_TOL:
            key = (r['image'], band, r['codec'])
            if key not in by_id or r['bytes'] < by_id[key]['bytes']:
                by_id[key] = {'bytes': r['bytes']}

# Build per-(image, band) codec bytes dict
ib = defaultdict(dict)  # (img, band) → {codec: best_bytes}
for (img, band, codec), v in by_id.items():
    ib[(img, band)][codec] = v['bytes']

# For each (image, band), enumerate ALL strategies — not just multi-codec
print(f"[cells] {len(ib)} (image, band) cells")

# Coverage analysis: how many cells have N codecs?
cov = defaultdict(int)
for cb in ib.values():
    cov[len(cb)] += 1
print(f"[coverage] {dict(cov)}")  # 1: 1 codec only, 2/3: multi-codec

# Image-level holdout
rng = random.Random(SEED)
all_imgs = sorted({img for img, _ in ib.keys()})
rng.shuffle(all_imgs)
hold = set(all_imgs[:max(1, len(all_imgs)//5)])

# Strategies on FULL holdout (no multi-codec filter)
def by_strategy(strat_fn, items_set):
    """Sum bytes over holdout for a strategy that picks codec given (img, band)."""
    total = 0; n = 0
    for (img, band), codec_bytes in ib.items():
        if img not in items_set: continue
        if not codec_bytes: continue
        codec = strat_fn(img, band, codec_bytes)
        if codec in codec_bytes:
            total += codec_bytes[codec]
        else:
            total += max(codec_bytes.values())  # forced fallback
        n += 1
    return total, n

# Strategies
def always(c): return lambda img, band, cb: c
def oracle(img, band, cb): return min(cb, key=cb.get)
def best_two_oracle(img, band, cb):
    """Pick best of TWO codecs (a realistic inference scenario where one is forced)."""
    return min(cb, key=cb.get)

print(f"\n## Strategy comparison on FULL holdout ({len(hold)} imgs, no multi-codec filter)")
print(f"{'strategy':<25} {'bytes':>14} {'n':>5} {'vs always-jxl':>16}")

bjxl, n_jxl = by_strategy(always('zenjxl'), hold)
bavif, n_avif = by_strategy(always('zenavif'), hold)
bwebp, n_webp = by_strategy(always('zenwebp'), hold)
boracle, n_oracle = by_strategy(oracle, hold)

print(f"{'always-zenjxl':<25} {bjxl:>14} {n_jxl:>5} {'baseline':>16}")
print(f"{'always-zenavif':<25} {bavif:>14} {n_avif:>5} {(bavif-bjxl)/bjxl*100:>+15.2f}%")
print(f"{'always-zenwebp':<25} {bwebp:>14} {n_webp:>5} {(bwebp-bjxl)/bjxl*100:>+15.2f}%")
print(f"{'oracle':<25} {boracle:>14} {n_oracle:>5} {(boracle-bjxl)/bjxl*100:>+15.2f}%")

# Per-class breakdown of "best static codec" by class
print(f"\n## Best static codec per class (which codec is best AVERAGE)")
def cls_of(stem):
    s = stem.lower()
    if s.startswith("gen-screen__"): return "screen"
    if s.startswith("gen-doc__"): return "document"
    if s.startswith("gen-chart__") or s.startswith("gen-line__"): return "lineart"
    if s.startswith("gen-mixed__"): return "photo"
    if any(p in s for p in ["terminal", "windows", "macos", "ubuntu", "screen", "gui_"]): return "screen"
    return "photo"

per_cls = defaultdict(lambda: {'jxl': 0, 'avif': 0, 'webp': 0, 'oracle': 0, 'n': 0})
for (img, band), cb in ib.items():
    if img not in hold: continue
    c = cls_of(img)
    per_cls[c]['n'] += 1
    per_cls[c]['jxl'] += cb.get('zenjxl', max(cb.values()))
    per_cls[c]['avif'] += cb.get('zenavif', max(cb.values()))
    per_cls[c]['webp'] += cb.get('zenwebp', max(cb.values()))
    per_cls[c]['oracle'] += min(cb.values())

print(f"{'class':<12} {'n':>4} {'always-jxl':>11} {'always-avif':>11} {'always-webp':>11} {'oracle':>11} {'oracle vs jxl':>13}")
for c in sorted(per_cls):
    s = per_cls[c]
    pct = (s['oracle'] - s['jxl']) / s['jxl'] * 100 if s['jxl'] else 0
    print(f"{c:<12} {s['n']:>4} {s['jxl']:>11} {s['avif']:>11} {s['webp']:>11} {s['oracle']:>11} {pct:>+12.2f}%")
