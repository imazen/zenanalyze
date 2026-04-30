#!/usr/bin/env python3
"""
Size-invariance probe for a baked picker JSON.

Size invariance is a *safety property* of the picker (see
SAFETY_PLANE.md → "Size invariance is a safety property"). The picker
is a feature-vector-in / argmin-out function — it has no notion of
image dimensions at runtime, only what the codec packs into the
feature vector via `size_class` one-hot, `log_pixels`, and the
zq×feature cross-terms. The picker MUST produce trustworthy picks at
every (width, height), not just at the four sample sizes
{tiny, small, medium, large} we sampled at training time.

This probe answers: given the same image content at four sizes, does
the picker's argmin cell stay stable across `size_class`? At each
`target_zq`? A stable argmin across all four sizes for a given image
is the canonical "size invariant" outcome — the codec is making the
*same encoder-config decision* regardless of resize. If the cell
flips drastically when an image is resized, the picker's pick at
that (image, zq) is sensitive to size in a way the safety plane
needs to know about.

# Inputs

This script consumes a features TSV produced by the codec's pareto
sweep harness — same shape `train_hybrid.py` reads:

    image_path<TAB>size_class<TAB>width<TAB>height<TAB>feat_<col>...

The harness must have run with `--features-only` over a corpus that
covers all four size_classes per image. For zenjpeg that's:

    cargo run --release -p zenjpeg --features 'target-zq trellis' \\
        --example zq_pareto_calibrate -- --features-only \\
        --corpus /home/lilith/work/codec-eval/codec-corpus/...

(See FOR_NEW_CODECS.md Step 1.5 for the codec-side discipline that
makes this TSV exist.)

By default the probe samples ~10 images that cover all four sizes;
override with `--n-images` and `--seed`. For each image it builds
the same feature vector `train_hybrid.py` builds (raw feats +
size_oh + log_px + zq cross-terms + icc placeholder) and runs the
forward pass via the Python numpy reference (no Rust dependency).

# Stability metric

For each image i and target_zq z:
    cells_i_z = {argmin_cell(i, sz, z) for sz in
                  {tiny, small, medium, large}}
    stable_i_z = (|cells_i_z| == 1)

Image-level stability:
    stable_i = mean over z of stable_i_z

Corpus-level stability:
    stability_pct = 100 * mean over (i, z) of stable_i_z

`--strict` exits 1 when `stability_pct < --threshold` (default 90.0).
This is the post-bake gate counterpart to `train_hybrid.py`'s
in-trainer `PER_SIZE_TAIL` and `DATA_STARVED_SIZE` violations.

# Why this matters

A picker that flips its cell pick when the same image is fed at a
different size is making encoder-config decisions on signal that
*should* be size-invariant. The right response is one of:
  1. Retrain with denser per-size coverage (DATA_STARVED_SIZE gate)
  2. Re-engineer the cross-term layout so size enters more
     gracefully into the feature vector
  3. Accept that some content (e.g. tiny screen captures) genuinely
     wants a different config than the same content at 4K — but
     surface the rate so it's not silently the picker's failure mode

# Usage

    python3 size_invariance_probe.py \\
        --model benchmarks/zq_bytes_hybrid_v2_1.json \\
        --features-tsv benchmarks/zq_pareto_features_2026-04-29_v2_1.tsv \\
        [--n-images 10] [--threshold 90.0] [--strict]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIZE_CLASSES = ["tiny", "small", "medium", "large"]
SIZE_INDEX = {s: i for i, s in enumerate(SIZE_CLASSES)}


def relu_forward(x: np.ndarray, layers: list[dict]) -> np.ndarray:
    """ReLU MLP forward pass — final layer identity. Matches
    train_hybrid.py / adversarial_probe.py."""
    a = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        z = a @ W + b
        a = z if i == last else np.maximum(z, 0.0)
    return a


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return (x - mean) / safe_scale


def build_engineered_input(
    raw_feats: np.ndarray,
    size_class: str,
    width: int,
    height: int,
    zq: int,
) -> np.ndarray:
    """Reproduces train_hybrid.py's `xe` construction.

    Layout (must match `build_dataset` in train_hybrid.py):
        f                              (n_feat,)
        size_oh                        (4,)
        [log_px, log_px², zq_norm,
         zq_norm², zq_norm × log_px]   (5,)
        zq_norm × f                    (n_feat,)
        [icc_bytes_placeholder]        (1,)
    """
    log_px = math.log(max(1, width * height))
    zq_norm = zq / 100.0
    size_oh = np.zeros(len(SIZE_CLASSES), dtype=np.float64)
    size_oh[SIZE_INDEX[size_class]] = 1.0
    return np.concatenate([
        raw_feats.astype(np.float64),
        size_oh,
        np.array([log_px, log_px * log_px, zq_norm, zq_norm * zq_norm, zq_norm * log_px]),
        zq_norm * raw_feats.astype(np.float64),
        np.array([0.0]),
    ])


def load_features_tsv(path: Path, feat_cols: list[str]):
    """Load features TSV indexed by (image_path, size_class).

    Returns: {(image, size): {"feats": ndarray, "w": int, "h": int}}
    """
    out = {}
    with open(path) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        all_cols = [c for c in rdr.fieldnames if c.startswith("feat_")]
        # Order MUST match the bake's feat_cols (the picker's first
        # n_feat inputs are positionally defined).
        missing = [c for c in feat_cols if c not in all_cols]
        if missing:
            raise SystemExit(
                f"features TSV is missing {len(missing)} columns the bake "
                f"declares: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        for r in rdr:
            key = (r["image_path"], r["size_class"])
            try:
                w = int(r["width"])
                h = int(r["height"])
                feats = np.array([float(r[c]) for c in feat_cols], dtype=np.float64)
            except (ValueError, KeyError):
                continue
            out[key] = {"feats": feats, "w": w, "h": h}
    return out


def pick_argmin(model: dict, feat_vec: np.ndarray, n_cells: int) -> int:
    """Run the forward pass and return the argmin cell index over the
    bytes-log subrange. No reach mask — we want the picker's *raw*
    pick, identical to what the codec runtime would compute before
    the constraint mask is applied."""
    mean = np.asarray(model["scaler_mean"], dtype=np.float64)
    scale = np.asarray(model["scaler_scale"], dtype=np.float64)
    x = standardize(feat_vec, mean, scale).reshape(1, -1)
    out = relu_forward(x, model["layers"]).ravel()
    bytes_log = out[:n_cells]
    return int(np.argmin(bytes_log))


def select_fixture_images(
    feats_by_key: dict, n_images: int, seed: int
) -> list[str]:
    """Pick images that have features at all four size_classes.
    Sampling without replacement, deterministic for `seed`."""
    by_image = defaultdict(set)
    for (img, sz), _ in feats_by_key.items():
        by_image[img].add(sz)
    full = sorted(
        img for img, szs in by_image.items()
        if all(s in szs for s in SIZE_CLASSES)
    )
    if not full:
        raise SystemExit(
            "no images in features TSV cover all four size_classes — "
            "the codec's pareto sweep harness must run at "
            "tiny / small / medium / large per image. See "
            "FOR_NEW_CODECS.md Step 1.5."
        )
    rng = random.Random(seed)
    rng.shuffle(full)
    return full[:n_images]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path,
                    help="trained model JSON (output of train_hybrid.py)")
    ap.add_argument("--features-tsv", required=True, type=Path,
                    help="features TSV (output of codec's --features-only "
                    "sweep harness; must cover all four size_classes per image)")
    ap.add_argument("--n-images", type=int, default=10,
                    help="number of fixture images to probe (default 10)")
    ap.add_argument("--seed", type=int, default=0xC0FFEE,
                    help="seed for fixture sampling (default 0xC0FFEE)")
    ap.add_argument("--zq-targets", type=str, default=None,
                    help="comma-separated target_zq values to probe. "
                    "Default: read from model JSON's training_objective "
                    "or the canonical zenjpeg grid 0..70 step 5 + 70..100 step 2")
    ap.add_argument("--threshold", type=float, default=90.0,
                    help="stability percentage floor for --strict (default 90.0)")
    ap.add_argument("--strict", action="store_true",
                    help="Auto-enabled in CI. Exit code 1 when "
                    "stability_pct < threshold.")
    args = ap.parse_args()

    model = json.loads(args.model.read_text())
    feat_cols = model.get("feat_cols")
    if not feat_cols:
        raise SystemExit("model JSON has no feat_cols — re-bake with current trainer")
    n_cells = (
        model.get("hybrid_heads_manifest", {}).get("n_cells")
        or int(model.get("n_outputs", 0))
    )
    if not n_cells:
        raise SystemExit("model JSON has no n_cells / n_outputs")

    if args.zq_targets:
        zq_targets = [int(z) for z in args.zq_targets.split(",")]
    else:
        zq_targets = list(range(0, 70, 5)) + list(range(70, 101, 2))

    sys.stderr.write(
        f"size_invariance_probe: model={args.model} n_cells={n_cells} "
        f"feat_cols={len(feat_cols)} zq_targets={len(zq_targets)}\n"
    )

    feats = load_features_tsv(args.features_tsv, feat_cols)
    images = select_fixture_images(feats, args.n_images, args.seed)
    sys.stderr.write(
        f"  loaded {len(feats)} (image, size) feature rows; "
        f"sampled {len(images)} fixtures covering all 4 sizes\n\n"
    )

    # Per-(image, zq), gather the 4 picks (one per size_class).
    n_pairs = 0
    n_stable = 0
    per_image = {}  # img -> stability fraction
    flips = []  # (img, zq, picks_per_size) for the worst flips
    for img in images:
        per_image[img] = {"stable": 0, "n": 0, "by_zq": {}}
        for zq in zq_targets:
            picks = []
            for sz in SIZE_CLASSES:
                row = feats.get((img, sz))
                if row is None:
                    picks.append(None)
                    continue
                xe = build_engineered_input(
                    row["feats"], sz, row["w"], row["h"], zq
                )
                picks.append(pick_argmin(model, xe, n_cells))
            if any(p is None for p in picks):
                continue
            unique = set(picks)
            stable = len(unique) == 1
            n_pairs += 1
            n_stable += int(stable)
            per_image[img]["n"] += 1
            per_image[img]["stable"] += int(stable)
            per_image[img]["by_zq"][zq] = picks
            if not stable:
                flips.append((img, zq, picks, len(unique)))

    if n_pairs == 0:
        sys.stderr.write(
            "no (image, zq) pairs produced picks at all 4 sizes — "
            "check that the features TSV covers tiny/small/medium/large "
            "for the sampled images.\n"
        )
        return 1 if args.strict else 0

    stability_pct = 100.0 * n_stable / n_pairs

    # Per-image table.
    sys.stderr.write(
        f"{'image':<60s}  {'stable/n':>10s}  pct\n"
    )
    for img in images:
        rec = per_image[img]
        if rec["n"] == 0:
            sys.stderr.write(f"  {img:<58s}  {'(skipped)':>10s}  --\n")
            continue
        pct = 100.0 * rec["stable"] / rec["n"]
        sys.stderr.write(
            f"  {img:<58s}  {rec['stable']:>4d}/{rec['n']:<4d}   {pct:5.1f}%\n"
        )

    sys.stderr.write(
        f"\nCorpus stability: {n_stable}/{n_pairs} = {stability_pct:.2f}% "
        f"(threshold {args.threshold:.1f}%)\n"
    )

    # Top flips (worst non-stable cases by cell-count diversity).
    if flips:
        flips.sort(key=lambda f: -f[3])
        sys.stderr.write("\nTop flips (most cells visited across sizes):\n")
        for img, zq, picks, k in flips[:6]:
            picks_str = ", ".join(
                f"{sz}={p}" for sz, p in zip(SIZE_CLASSES, picks)
            )
            sys.stderr.write(
                f"  {img}  @zq{zq}  {k} distinct cells: {picks_str}\n"
            )

    failed = stability_pct < args.threshold
    if failed:
        sys.stderr.write(
            f"\n[FAIL] stability {stability_pct:.2f}% < threshold {args.threshold:.1f}%\n"
        )
    else:
        sys.stderr.write(
            f"\n[OK] stability {stability_pct:.2f}% >= threshold {args.threshold:.1f}%\n"
        )

    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
