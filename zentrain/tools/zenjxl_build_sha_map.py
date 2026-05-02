#!/usr/bin/env python3
"""Build a SHA→path map from the picker-train manifest used by
``jxl-encoder/examples/lossy_pareto_calibrate.rs``.

The manifest is a TSV with columns: sha256, split, content_class,
source, size_bytes, path. The pareto oracle keys by `image_sha`, so
we need a {sha: path} map to re-extract features.

Usage:
    python3 zenjxl_build_sha_map.py \\
        --manifest /home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv \\
        --out /home/lilith/work/zen/zenjxl/benchmarks/zenjxl_image_sha_paths.json

By default it walks the picker-train manifest. Pass `--shas FILE` (a
plain text list of sha hashes, one per line) to subset the output to
only the shas actually referenced by the oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/home/lilith/work/codec-corpus/picker-train/manifest_v1_100.tsv"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            os.path.expanduser(
                "~/work/zen/zenjxl/benchmarks/zenjxl_image_sha_paths.json"
            )
        ),
    )
    ap.add_argument(
        "--shas",
        type=Path,
        help="Optional path to a plain-text list of shas to subset",
    )
    args = ap.parse_args()

    sha_filter = None
    if args.shas:
        sha_filter = {ln.strip() for ln in args.shas.read_text().splitlines() if ln.strip()}
        print(f"[sha-map] filtering to {len(sha_filter)} shas", file=sys.stderr)

    out: dict[str, str] = {}
    with open(args.manifest) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            sha = r.get("sha256")
            path = r.get("path")
            if not sha or not path:
                continue
            if sha_filter is not None and sha not in sha_filter:
                continue
            out[sha] = path

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"[sha-map] manifest={args.manifest} mapped={len(out)} -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
