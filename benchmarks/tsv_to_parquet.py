"""Convert a TSV pareto / features file to Parquet (zstd-compressed).

Use this on big pareto sweeps (>50 MB TSV) so picker training,
LOO retrains, and Tier 0 / Tier 1.5 ablation can all load the
same data ~36× faster and at ~16× smaller disk footprint.
On the zenwebp pareto (3.4 GB / 21.8M rows):

    csv.DictReader (current train_hybrid path):  68 s
    pyarrow.csv.read_csv on the same TSV:         3 s
    Parquet (zstd):                               2 s, 0.21 GB on disk

Usage:
    python3 benchmarks/tsv_to_parquet.py [--keep-tsv] <path-to-tsv> [<path-to-tsv> ...]

By default the TSV is replaced by an .parquet file with the same
stem; pass `--keep-tsv` to leave the source file in place.

The picker configs (zenwebp_picker_config.py etc.) auto-detect
Parquet by extension via train_hybrid.py's `_read_table_records`
helper — once the file is converted, point `PARETO` /
`FEATURES` at the .parquet path and you're done.

See `~/work/claudehints/topics/parquet-vs-tsv.md` for the
project-wide convention.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.csv as pa_csv
import pyarrow.parquet as pq


def convert(tsv_path: Path, keep_tsv: bool = False) -> Path:
    if not tsv_path.is_file():
        sys.stderr.write(f"  [SKIP] not a file: {tsv_path}\n")
        return tsv_path
    out = tsv_path.with_suffix(".parquet")
    if out.exists():
        sys.stderr.write(f"  [SKIP] already exists: {out}\n")
        return out

    src_size = tsv_path.stat().st_size
    sys.stderr.write(f"  reading {tsv_path} ({src_size / 1e9:.2f} GB)...\n")
    t0 = time.time()
    table = pa_csv.read_csv(
        tsv_path,
        parse_options=pa_csv.ParseOptions(delimiter="\t"),
        # Allow integer columns to coexist with empty cells (NULL).
        convert_options=pa_csv.ConvertOptions(strings_can_be_null=True),
    )
    read_elapsed = time.time() - t0
    sys.stderr.write(
        f"  parsed {table.num_rows} rows × {table.num_columns} cols "
        f"in {read_elapsed:.1f} s\n"
    )

    sys.stderr.write(f"  writing {out} (zstd compression)...\n")
    t0 = time.time()
    pq.write_table(table, out, compression="zstd", compression_level=3)
    write_elapsed = time.time() - t0
    out_size = out.stat().st_size
    sys.stderr.write(
        f"  wrote {out_size / 1e9:.2f} GB in {write_elapsed:.1f} s "
        f"(compression ratio {src_size / out_size:.1f}×)\n"
    )

    if not keep_tsv:
        tsv_path.unlink()
        sys.stderr.write(f"  removed source TSV: {tsv_path}\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="+",
        help="TSV files to convert. Output written alongside as .parquet.",
    )
    ap.add_argument(
        "--keep-tsv",
        action="store_true",
        help="Leave source TSV in place (default: remove after successful conversion).",
    )
    args = ap.parse_args()
    started = time.time()
    for p in args.paths:
        convert(Path(p), keep_tsv=args.keep_tsv)
    sys.stderr.write(f"\nTotal wall: {time.time() - started:.1f} s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
