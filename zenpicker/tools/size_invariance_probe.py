#!/usr/bin/env python3
"""
Size-invariance probe for a baked picker JSON.

Size invariance is a *safety property* of the picker (see
SAFETY_PLANE.md → "Size invariance is a safety property"). The picker
must produce *trustworthy picks at every (width, height)* — not just
at the four sample sizes (`tiny / small / medium / large`) we
sampled at training time.

# What "size invariance" actually means for a size-aware picker

The picker is **trained size-aware** — `size_class` one-hot,
`log_pixels`, and zq×feature cross-terms are explicit inputs. So
it's *intentionally* allowed to pick *different* cells at different
sizes; doing so is the point of including those features. Demanding
identical argmin across the four sizes contradicts training intent.

The right "invariance" criterion is therefore not pick identity but
**per-size correctness**: the picker should pick a *good* cell at
every size, even if it's a different good cell per size. We measure
this by:

  1. **Principal criterion** — read
     `safety_report.diagnostics.by_size` (training-time per-size
     mean overhead) and check that the worst size_class is no more
     than `--max-size-overhead-ratio` × the best (default 3.0). One
     size_class being dramatically worse than another is a real bug
     (often: data starvation at that size — see
     `DATA_STARVED_SIZE` in `train_hybrid.py`).
  2. **Diagnostic** — argmin-identical-across-sizes rate. Reports
     how often the same cell is chosen across all four sizes for
     the same image at the same target_zq. Useful as a baseline
     check on whether size-awareness is doing anything at all
     (a value near 100% means the picker isn't using size_class
     much; a value near 0% means it's heavily size-driven). Not a
     hard gate by default; set `--argmin-identical-floor` non-zero
     to enable.

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

The model JSON must carry `safety_report.diagnostics.by_size` —
trained with current `train_hybrid.py` so the per-size aggregates
are populated. Bakes that pre-date the safety machinery emit a
warning and skip the principal criterion.

# Strict gate

`--strict` exits 1 when:
  - per-size worst/best mean overhead ratio > `--max-size-overhead-ratio`,
    OR
  - argmin-identical rate < `--argmin-identical-floor` (when that
    threshold is set non-zero)

This is the post-bake counterpart to `train_hybrid.py`'s in-trainer
`PER_SIZE_TAIL` and `DATA_STARVED_SIZE` violations. Auto-strict in CI.

# Usage

    python3 size_invariance_probe.py --strict \\
        --model benchmarks/zq_bytes_hybrid_v2_1.json \\
        --features-tsv benchmarks/zq_pareto_features_2026-04-29_v2_1.tsv
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
    ap.add_argument(
        "--max-size-overhead-ratio",
        type=float,
        default=3.0,
        help="Principal pass criterion: ratio of worst-size mean "
        "overhead to best-size mean overhead, read from "
        "safety_report.diagnostics.by_size. Default 3.0 — no "
        "size_class may be more than 3× worse than the best. This "
        "catches real per-size regressions (the picker is bad at "
        "one size_class) without penalizing intentional size-aware "
        "cell variation. Requires a model JSON with safety_report.",
    )
    ap.add_argument(
        "--argmin-identical-floor",
        type=float,
        default=0.0,
        help="Diagnostic gate (default 0.0 = disabled). Was the only "
        "criterion in earlier versions; kept for explicit users. The "
        "picker is *trained* with size_class as an input feature, so "
        "demanding identical argmin across sizes contradicts training "
        "intent. Set to a non-zero value (e.g. 25.0) only when probing "
        "a baseline that is supposed to be size-invariant by design.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit code 1 when the principal criterion fails (or, "
        "when --argmin-identical-floor > 0, that secondary gate also).",
    )
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
        f"\n[diagnostic] argmin-identical-across-sizes: "
        f"{n_stable}/{n_pairs} = {stability_pct:.2f}%\n"
        f"             (the picker is *trained* size-aware via the "
        f"size_class one-hot, so per-size variation is *expected*; this\n"
        f"             rate matters only as a baseline check on whether "
        f"size-awareness is doing anything at all.)\n"
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

    # Principal criterion: per-size correctness from safety_report.
    sr = model.get("safety_report") or {}
    by_size = (sr.get("diagnostics") or {}).get("by_size") or {}
    if not by_size:
        sys.stderr.write(
            "\n[WARN] model JSON has no safety_report.diagnostics.by_size — "
            "principal criterion (per-size mean-overhead ratio) cannot run. "
            "Re-bake with current train_hybrid.py to populate it.\n"
        )
        ratio_failed = True
        ratio = float("nan")
    else:
        means = {sz: by_size[sz].get("mean_pct") for sz in SIZE_CLASSES if sz in by_size}
        means = {sz: v for sz, v in means.items() if v is not None}
        if not means:
            sys.stderr.write(
                "[WARN] safety_report.by_size has no per-size mean values; "
                "principal criterion skipped.\n"
            )
            ratio_failed = True
            ratio = float("nan")
        else:
            best = min(means.values())
            worst = max(means.values())
            ratio = worst / max(best, 1e-9)
            sys.stderr.write(
                "\n[principal] per-size mean overhead "
                "(from training-time safety_report):\n"
            )
            for sz in SIZE_CLASSES:
                if sz in means:
                    flag = "  ←worst" if means[sz] == worst else (
                        "  ←best" if means[sz] == best else ""
                    )
                    sys.stderr.write(
                        f"    {sz:8s}  mean={means[sz]:6.2f}%{flag}\n"
                    )
            sys.stderr.write(
                f"\n             worst/best ratio = {ratio:.2f}× "
                f"(threshold {args.max_size_overhead_ratio:.2f}×)\n"
            )
            ratio_failed = ratio > args.max_size_overhead_ratio

    # Secondary gate (off by default).
    identical_failed = (
        args.argmin_identical_floor > 0.0
        and stability_pct < args.argmin_identical_floor
    )

    failed = ratio_failed or identical_failed
    if failed:
        msgs = []
        if ratio_failed:
            msgs.append(
                f"per-size ratio {ratio:.2f}× > "
                f"{args.max_size_overhead_ratio:.2f}× (one size_class "
                "is much worse than another — re-train with more rows "
                "for the lagging size, or tighten "
                "DATA_STARVED_SIZE / PER_SIZE_TAIL thresholds)"
            )
        if identical_failed:
            msgs.append(
                f"argmin-identical {stability_pct:.2f}% < "
                f"{args.argmin_identical_floor:.1f}% (size-invariant "
                "baseline check)"
            )
        sys.stderr.write("\n[FAIL] " + "; ".join(msgs) + "\n")
    else:
        sys.stderr.write(
            f"\n[OK] per-size ratio {ratio:.2f}× ≤ "
            f"{args.max_size_overhead_ratio:.2f}× — picker overheads "
            "are coherent across size_classes.\n"
        )

    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
