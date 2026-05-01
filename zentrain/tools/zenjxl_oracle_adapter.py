#!/usr/bin/env python3
"""
zenjxl Pareto-oracle adapter — synthesizes a zentrain-compatible TSV
from the existing JXL pareto sweep.

Why this exists
---------------
The JXL oracle at ``/mnt/v/output/jxl-encoder/picker-oracle-2026-04-30/``
was produced by ``jxl-encoder/examples/lossy_pareto_calibrate.rs`` which
emits one row per (image, knob_combination, distance) sample and keys
its rows by `image_sha` rather than `image_path`. zentrain's tools
(`feature_ablation.py`, `correlation_cleanup.py`, `train_hybrid.py`)
expect a TSV keyed by `(image_path, size_class)` with `config_id`,
`config_name`, `bytes`, `zensim` columns.

This adapter rewrites the columns without re-encoding:

  - `config_name` — synthesized from the encoder-knob columns ONLY
    (distance is the quality axis, analogous to q in zenwebp/zenjpeg
    sweeps; ZQ_TARGETS interpolate it on the fly).
        Lossy:    c{cell_id}_ac{ac_intensity}_ec{enhanced_clustering}
                  _g{gaborish}_p{patches}_kil{k_info_loss_mul}
                  _kaq{k_ac_quant}_ed8{entropy_mul_dct8}
        Lossless: c{cell_id}_lz{lz77_method}_sq{squeeze}_p{patches}
                  _rct{nb_rcts_to_try}_wp{wp_num_param_sets}
                  _tmb{tree_max_buckets}_tnp{tree_num_properties}
                  _tsf{tree_sample_fraction}
  - `config_id` — stable hash of the knob tuple, modulo 2**31 (so it
    fits a positive int32 — what feature_ablation densely remaps).
  - `image_path` — synthesized as ``sha:<first-16-chars-of-sha>`` so
    the trainer's `(image_path, size_class)` join key works against
    the matching synthesised features TSV.
  - `zensim` — populated from the source `ssim2` column. **This is a
    metric substitution: the JXL sweep measured ssim2 (Butteraugli's
    sibling), not zenanalyze/zenpipe's `zensim` (XYB-Butteraugli).
    Numerically distinct.** A `# metric_name=ssim2` comment is
    written into the output header. Cross-codec aggregation against
    sweeps that use real `zensim` will mis-rank — see the picker
    config docstring.
  - `size_class` — passes through if present, else `"native"`.

Outputs
-------
  ~/work/zen/zenjxl/benchmarks/zenjxl_lossy_pareto_2026-05-01.tsv
  ~/work/zen/zenjxl/benchmarks/zenjxl_lossless_pareto_2026-05-01.tsv

Usage
-----
    python3 zenjxl_oracle_adapter.py \\
        --input /mnt/v/output/jxl-encoder/picker-oracle-2026-04-30 \\
        --out-dir ~/work/zen/zenjxl/benchmarks \\
        --date 2026-05-01

The script handles both lossy and lossless oracles in one pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

LOSSY_KNOBS = [
    "cell_id",
    "ac_intensity",
    "enhanced_clustering",
    "gaborish",
    "patches",
    "k_info_loss_mul",
    "k_ac_quant",
    "entropy_mul_dct8",
]

LOSSLESS_KNOBS = [
    "cell_id",
    "lz77_method",
    "squeeze",
    "patches",
    "nb_rcts_to_try",
    "wp_num_param_sets",
    "tree_max_buckets",
    "tree_num_properties",
    "tree_sample_fraction",
]


def synth_lossy_config_name(row: dict) -> str:
    return (
        f"c{row['cell_id']}_ac{row['ac_intensity']}"
        f"_ec{row['enhanced_clustering']}_g{row['gaborish']}"
        f"_p{row['patches']}_kil{row['k_info_loss_mul']}"
        f"_kaq{row['k_ac_quant']}_ed8{row['entropy_mul_dct8']}"
    )


def synth_lossless_config_name(row: dict) -> str:
    return (
        f"c{row['cell_id']}_lz{row['lz77_method']}_sq{row['squeeze']}"
        f"_p{row['patches']}_rct{row['nb_rcts_to_try']}"
        f"_wp{row['wp_num_param_sets']}_tmb{row['tree_max_buckets']}"
        f"_tnp{row['tree_num_properties']}_tsf{row['tree_sample_fraction']}"
    )


def stable_config_id(name: str) -> int:
    """Stable, deterministic hash modulo 2**31. SHA-256 first 8 bytes
    masked to int32-positive. Same knob tuple → same id across runs."""
    h = hashlib.sha256(name.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big", signed=False)
    return n & 0x7FFF_FFFF


def synth_image_path(sha: str) -> str:
    return f"sha:{sha[:16]}"


def adapt(
    src: Path,
    dst: Path,
    *,
    knobs: list[str],
    config_synth,
    has_zensim: bool,
    metric_label: str,
) -> tuple[int, int, set[str]]:
    """Rewrite `src` → `dst` with adapted column schema.

    Returns (rows_in, rows_out, unique_image_shas).
    """
    rows_in = 0
    rows_out = 0
    unique_shas: set[str] = set()

    with open(src, "r", newline="") as f_in:
        rdr = csv.DictReader(f_in, delimiter="\t")
        # The lossy sweep has ssim2; the lossless one does not (lossless
        # sweeps optimise bytes only, no quality dimension).
        if has_zensim and "ssim2" not in rdr.fieldnames:
            raise SystemExit(
                f"adapter: expected ssim2 column in {src}; got {rdr.fieldnames}"
            )
        for k in knobs + ["image_sha", "width", "height", "bytes"]:
            if k not in rdr.fieldnames:
                raise SystemExit(
                    f"adapter: expected {k} column in {src}; got {rdr.fieldnames}"
                )

        # Write a sidecar .meta file with the metric annotation —
        # zentrain's csv.DictReader doesn't skip '#' comment lines,
        # so we keep the TSV body clean and put provenance alongside.
        meta_dst = dst.with_suffix(dst.suffix + ".meta")
        with open(meta_dst, "w") as f_meta:
            f_meta.write(f"metric_name={metric_label}\n")
            f_meta.write(f"adapted_from={src}\n")
            f_meta.write(f"config_name_format={'|'.join(knobs)}\n")

        with open(dst, "w", newline="") as f_out:
            out_cols = [
                "image_path",
                "image_sha",
                "split",
                "content_class",
                "size_class",
                "width",
                "height",
                "config_id",
                "config_name",
                "bytes",
            ]
            if has_zensim:
                out_cols.append("zensim")
            f_out.write("\t".join(out_cols) + "\n")

            for row in rdr:
                rows_in += 1
                sha = row["image_sha"]
                unique_shas.add(sha)
                size_class = row.get("size_class") or "native"
                cname = config_synth(row)
                cid = stable_config_id(cname)
                vals = [
                    synth_image_path(sha),
                    sha,
                    row.get("split", ""),
                    row.get("content_class", ""),
                    size_class,
                    row["width"],
                    row["height"],
                    str(cid),
                    cname,
                    row["bytes"],
                ]
                if has_zensim:
                    vals.append(row["ssim2"])
                f_out.write("\t".join(vals) + "\n")
                rows_out += 1

    return rows_in, rows_out, unique_shas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/v/output/jxl-encoder/picker-oracle-2026-04-30"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.path.expanduser("~/work/zen/zenjxl/benchmarks")),
    )
    ap.add_argument("--date", default="2026-05-01")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    lossy_src = args.input / "lossy_pareto_2026-04-30.tsv"
    lossless_src = args.input / "lossless_pareto_2026-04-30.tsv"
    lossy_dst = args.out_dir / f"zenjxl_lossy_pareto_{args.date}.tsv"
    lossless_dst = args.out_dir / f"zenjxl_lossless_pareto_{args.date}.tsv"

    print(f"[adapter] lossy  {lossy_src} -> {lossy_dst}", file=sys.stderr)
    rin, rout, shas = adapt(
        lossy_src,
        lossy_dst,
        knobs=LOSSY_KNOBS,
        config_synth=synth_lossy_config_name,
        has_zensim=True,
        metric_label="ssim2",
    )
    print(
        f"[adapter] lossy:    rows_in={rin} rows_out={rout} "
        f"unique_shas={len(shas)}",
        file=sys.stderr,
    )

    print(f"[adapter] lossless {lossless_src} -> {lossless_dst}", file=sys.stderr)
    rin_l, rout_l, shas_l = adapt(
        lossless_src,
        lossless_dst,
        knobs=LOSSLESS_KNOBS,
        config_synth=synth_lossless_config_name,
        has_zensim=False,
        metric_label="(lossless: no quality axis)",
    )
    print(
        f"[adapter] lossless: rows_in={rin_l} rows_out={rout_l} "
        f"unique_shas={len(shas_l)}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
