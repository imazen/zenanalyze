#!/usr/bin/env python3
"""Rename a feature parquet's bare ``feat_<name>`` columns to their qualified
``<name>@hex8`` zenanalyze-api contract identity — **values and schema metadata
untouched** — for migrating existing feature data to the qualified schema without
re-extraction (the "edit parquets" path).

The mapping is the committed ``benchmarks/feature_qualified_names.tsv`` — the
golden-blessed output of ``zenanalyze::versioning::feature_qualified_names()`` (the
single source of truth; replicating the hash off-Rust would risk a silent mismatch).
A ``feat_<name>`` column whose bare name isn't in the table is left unchanged (a
non-versioned / unknown column — the safe direction). Meta columns (no ``feat_``
prefix) are never touched.

Usage:
    qualify_parquet_columns.py IN.parquet OUT.parquet
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

_TSV = Path(__file__).resolve().parent / "feature_qualified_names.tsv"


def _mapping() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _TSV.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        bare, qualified = line.split("\t", 1)
        out[bare] = qualified
    return out


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: qualify_parquet_columns.py IN.parquet OUT.parquet")
    inp, outp = sys.argv[1], sys.argv[2]
    table = pq.read_table(inp)  # preserves schema-level KV metadata
    m = _mapping()
    new_names, renamed, already = [], 0, 0
    for c in table.column_names:
        if c.startswith("feat_") and c[len("feat_"):] in m:
            new_names.append(m[c[len("feat_"):]])
            renamed += 1
        else:
            if "@" in c:
                already += 1
            new_names.append(c)
    table = table.rename_columns(new_names)
    pq.write_table(table, outp, compression="zstd")
    print(
        f"{inp} -> {outp}\n"
        f"  renamed {renamed} feature columns feat_<name> -> <name>@hex8 "
        f"({already} already qualified); {table.num_rows} rows x {table.num_columns} cols "
        f"(values + KV metadata untouched)"
    )


if __name__ == "__main__":
    main()
