#!/usr/bin/env python3
"""Inventory every snapshotted public declaration against named source trees.

Matches are evidence for review, not proof that similarly named items are the
same API. The accompanying readiness document records qualified workflows.
"""

import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer", action="append", default=[], metavar="LABEL=PATH")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = []
    for consumer in args.consumer:
        label, directory = consumer.split("=", 1)
        directory = Path(directory)
        paths = subprocess.run(
            ["rg", "--files", "-g", "*.rs", str(directory)],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        for filename in paths:
            path = Path(filename)
            text = path.read_text()
            # Exclude docs-only mentions and keep exact source/line receipts.
            for number, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith(("//", "/*", "*")):
                    continue
                sources.append((label, path.relative_to(directory), number,
                                set(re.findall(r"\b[A-Za-z_]\w*\b", line))))

    rows = []
    for suffix in ("", ".features", ".internal"):
        snapshot = root / "docs/public-api" / f"zenpredict{suffix}.txt"
        for number, line in enumerate(snapshot.read_text().splitlines(), 1):
            match = re.search(
                r"\bpub (?:(?:const )?fn |struct |enum |type |const |mod |trait )?",
                line,
            )
            if not match:
                continue
            # Generic paths may contain spaces (`<const K: usize>`) and
            # lifetimes; keep parsing until the declaration's arguments/type.
            path = ""
            depth = 0
            for char in line[match.end():]:
                if char == "<":
                    depth += 1
                elif char == ">":
                    depth -= 1
                elif depth == 0:
                    if char.isspace() or char in "(=":
                        break
                    path += char
            symbol = path.rstrip(":").split("::")[-1]
            refs = [f"{label}:{file}:{n}" for label, file, n, tokens in sources
                    if symbol in tokens]
            rows.append((f"{snapshot.name}:{number}", line, refs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("# zenpredict public declaration census\n\n")
        out.write("One row per declaration in supported, feature and hidden snapshots. "
                  "References are lexical candidates, including test/tool code; "
                  "they require the workflow qualification in docs/PREDICT_READINESS_2026-11-05.md. "
                  "An empty cell is not proof that wire/runtime support can be removed.\n\n")
        out.write("| Snapshot receipt | Declaration | Candidate source references |\n|---|---|---|\n")
        for receipt, declaration, refs in rows:
            declaration = declaration.replace("|", "\\|")
            out.write(f"| {receipt} | `{declaration}` | {'; '.join(refs) or 'No source match'} |\n")
    print(f"{len(rows)} public declarations inventoried into {args.output}")


if __name__ == "__main__":
    main()
