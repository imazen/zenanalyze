#!/usr/bin/env python3
"""
zen-metrics sweep TSV → zentrain Pareto TSV adapter (v0.4 picker arc).

Why this exists
---------------
The `zen-metrics sweep` subcommand (turbo-metrics 0.3.0) emits a flat
TSV with columns:

  image_path | codec | q | knob_tuple_json | encoded_bytes |
  encode_ms | decode_ms | score_zensim | score_ssim2 [| score_dssim]

zentrain's training pipeline (feature_ablation.py, correlation_cleanup.py,
train_hybrid.py) consumes a Pareto TSV with columns:

  image_path | size_class | width | height | config_id | config_name |
  <axis cols> | bytes | zensim | encode_ms [| decode_ms]

This adapter rewrites the columns without re-encoding. Knob tuples
become axis columns + a stable `config_name`, the metric column is
renamed, and `size_class` is synthesized from the image's pixel
count (using the same buckets as zenwebp_picker_config). Images are
keyed by their flattened path (the on-disk filename in the sweep
directory) for join-compatibility with the corresponding features
TSV emitted by `refresh_features.py`.

Usage
-----
    python3 zenmetrics_sweep_adapter.py \\
        --input s3://zentrain/sweep-v04-2026-05-04/zenavif_pareto_v04_full.tsv \\
        --codec zenavif \\
        --output ~/work/zen/zenavif/benchmarks/zenavif_pareto_2026-05-04_v04_full.tsv

If --input starts with s3:// it's pulled from R2 first. The metric
selection (zensim vs ssim2) defaults to zensim for zenavif and
ssim2 for zenjxl (matching the metric used in the v0.3 picker
configs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Mirror size buckets from zenwebp_picker_config: tiny / small /
# medium / large by max(width, height). Keep in sync with whichever
# upstream picker config consumes this output.
SIZE_BUCKETS = [
    ("tiny", 64),
    ("small", 256),
    ("medium", 1024),
    ("large", 1 << 30),
]


def size_class(width: int, height: int) -> str:
    md = max(width, height)
    for name, hi in SIZE_BUCKETS:
        if md <= hi:
            return name
    return "large"


def stable_config_id(name: str) -> int:
    """Stable, deterministic hash mod 2**31. Same knob tuple → same id."""
    h = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False) & 0x7FFF_FFFF


def synth_zenavif_config_name(q: int, knobs: dict) -> str:
    """zenavif config_name format: s{speed}_q{q}_t{tune}.

    Compatible with the v0.4 collapsed cell taxonomy
    (CATEGORICAL_AXES = ['speed']).
    """
    speed = knobs.get("speed", "?")
    tune = knobs.get("tune", 0)
    return f"s{speed}_q{q}_t{tune}"


def synth_zenjxl_config_name(q: int, knobs: dict) -> str:
    """zenjxl config_name format: e{effort}_d{distance}_q{q}.

    `q` here is the public-API quality knob (0..100); `distance` is
    the alternate quality dial. The v0.4 picker config keeps the
    mapping public-API-only (no with_internal_params).
    """
    effort = knobs.get("effort", "?")
    distance = knobs.get("distance", "?")
    return f"e{effort}_d{distance}_q{q}"


def fetch_input(input_arg: str, work_dir: Path) -> Path:
    """If --input is an s3:// path, sync it down via aws-cli using R2
    creds from ~/.config/cloudflare/r2-credentials. Otherwise return
    the path unchanged."""
    if not input_arg.startswith("s3://"):
        return Path(input_arg).expanduser()
    cred_path = Path("~/.config/cloudflare/r2-credentials").expanduser()
    if not cred_path.exists():
        raise SystemExit(f"adapter: missing R2 creds at {cred_path}")
    creds = {}
    for line in cred_path.read_text().splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        creds[k.strip()] = v.strip()
    env = dict(os.environ)
    env["AWS_ACCESS_KEY_ID"] = creds["R2_ACCESS_KEY_ID"]
    env["AWS_SECRET_ACCESS_KEY"] = creds["R2_SECRET_ACCESS_KEY"]
    endpoint = f"https://{creds['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    local = work_dir / "input.tsv"
    cmd = [
        "aws", "s3", "cp", input_arg, str(local),
        "--endpoint-url", endpoint, "--region", "auto",
    ]
    subprocess.run(cmd, env=env, check=True)
    return local


def adapt(
    src: Path,
    dst: Path,
    *,
    codec: str,
    metric: str,
    sources_dir: Path | None,
) -> tuple[int, int, set[str]]:
    """Rewrite zen-metrics sweep TSV at `src` to zentrain Pareto schema
    at `dst`.

    The image dimensions are read from the source corpus directory if
    provided; otherwise width/height columns are written as `0` and
    size_class as `unknown` (callers should provide --sources-dir).
    """
    if codec == "zenavif":
        synth = synth_zenavif_config_name
        axis_cols = ["speed", "tune"]
        score_col = f"score_{metric}"
    elif codec == "zenjxl":
        synth = synth_zenjxl_config_name
        axis_cols = ["effort", "distance"]
        score_col = f"score_{metric}"
    else:
        raise SystemExit(f"adapter: unknown codec {codec}")

    # Optionally cache image dimensions to avoid re-decoding for each row.
    dims_cache: dict[str, tuple[int, int]] = {}

    def get_dims(image_basename: str) -> tuple[int, int]:
        """Read PNG width/height from the PNG IHDR chunk. Cheap, no PIL
        required."""
        if image_basename in dims_cache:
            return dims_cache[image_basename]
        if sources_dir is None:
            return (0, 0)
        path = sources_dir / image_basename
        if not path.exists():
            dims_cache[image_basename] = (0, 0)
            return (0, 0)
        try:
            with open(path, "rb") as f:
                head = f.read(24)
            if head[:8] != b"\x89PNG\r\n\x1a\n":
                dims_cache[image_basename] = (0, 0)
                return (0, 0)
            w = int.from_bytes(head[16:20], "big")
            h = int.from_bytes(head[20:24], "big")
            dims_cache[image_basename] = (w, h)
            return (w, h)
        except OSError:
            dims_cache[image_basename] = (0, 0)
            return (0, 0)

    rows_in = 0
    rows_out = 0
    unique_imgs: set[str] = set()

    out_fields = [
        "image_path", "size_class", "width", "height",
        "config_id", "config_name", "q",
    ] + axis_cols + ["bytes", "zensim", "encode_ms", "decode_ms"]

    with open(src, "r", newline="") as f_in, open(dst, "w", newline="") as f_out:
        rdr = csv.DictReader(f_in, delimiter="\t")
        if score_col not in rdr.fieldnames:
            raise SystemExit(
                f"adapter: expected {score_col} column in {src}; got {rdr.fieldnames}"
            )
        wrt = csv.DictWriter(f_out, fieldnames=out_fields, delimiter="\t",
                             extrasaction="ignore")
        wrt.writeheader()
        for row in rdr:
            rows_in += 1
            image_path_raw = row.get("image_path") or ""
            image_basename = os.path.basename(image_path_raw)
            if not image_basename:
                continue
            unique_imgs.add(image_basename)
            q_raw = row.get("q")
            if q_raw is None or q_raw == "":
                continue
            try:
                q = int(q_raw)
                bytes_ = int(row["encoded_bytes"])
                zensim = float(row[score_col])
            except (ValueError, KeyError, TypeError):
                continue
            knobs = json.loads(row["knob_tuple_json"])
            config_name = synth(q, knobs)
            w, h = get_dims(image_basename)
            sc = size_class(w, h) if w > 0 else "unknown"
            out_row = {
                "image_path": image_basename,
                "size_class": sc,
                "width": w,
                "height": h,
                "config_id": stable_config_id(config_name),
                "config_name": config_name,
                "q": q,
                "bytes": bytes_,
                "zensim": f"{zensim:.6f}",
                "encode_ms": row.get("encode_ms", ""),
                "decode_ms": row.get("decode_ms", ""),
            }
            for ax in axis_cols:
                out_row[ax] = knobs.get(ax, "")
            wrt.writerow(out_row)
            rows_out += 1
    return rows_in, rows_out, unique_imgs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Path or s3:// URL of the zen-metrics sweep TSV")
    p.add_argument("--codec", required=True, choices=["zenavif", "zenjxl"])
    p.add_argument("--output", required=True,
                   help="Output Pareto TSV path (zentrain schema)")
    p.add_argument("--metric", default=None,
                   help="Score column to use (zensim or ssim2). "
                        "Defaults: zenavif=zensim, zenjxl=ssim2.")
    p.add_argument("--sources-dir", default=None,
                   help="Path to source images directory; used for "
                        "looking up width/height (lightweight PNG IHDR "
                        "read). If omitted, dims = 0 and size_class = 'unknown'.")
    args = p.parse_args()

    metric = args.metric
    if metric is None:
        metric = "zensim" if args.codec == "zenavif" else "ssim2"

    sources_dir = Path(args.sources_dir).expanduser() if args.sources_dir else None
    if sources_dir is not None and not sources_dir.is_dir():
        raise SystemExit(f"adapter: --sources-dir {sources_dir} is not a directory")

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = fetch_input(args.input, tmp_path)
        rows_in, rows_out, unique_imgs = adapt(
            src, out_path,
            codec=args.codec,
            metric=metric,
            sources_dir=sources_dir,
        )

    print(f"adapter: {args.codec} {rows_in} rows in → {rows_out} rows out, "
          f"{len(unique_imgs)} unique images, metric={metric}")
    print(f"  output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
