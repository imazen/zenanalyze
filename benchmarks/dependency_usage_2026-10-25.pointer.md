# Ecosystem usage evidence — 2026-10-25

Findings: [ecosystem use and approaches to reconsider](../docs/ECOSYSTEM_USAGE_2026-10-25.md).

Private evidence on **dev**, checked 2026-10-25:

- `/mnt/v/output/zenanalyze/dependency-audit-2026-10-25/evidence-final.parquet`
- SHA-256: `d292fe85760e9f9ef353e05c23f9ec0686a521378e4a8eab4959dc38dab9bc1b`
- Parquet with zstd; each original JSONL row is retained as text. Decoding
  the Parquet reconstructed every input file's SHA-256 exactly.
- Input JSONL, ref summaries and fetch results remain in the same directory.
  They include private checkout metadata and are not public artifacts.
- No R2 or tower mirror was created for this source census.

Tool: `tools/audit_dependency_refs.py` at `03dd46a3`, including its `compact` mode.
Every row binds its repository,
commit, ref names, manifests and source lines. Captured refs are observations
from this date, not a claim about future branch heads.

Commands, run with the host's resource wrapper:

```sh
python3 tools/audit_dependency_refs.py --roots "$HOME/work" "$HOME/work/zen" --output "$EVIDENCE"
python3 tools/audit_dependency_refs.py --manifest-graph --roots "$HOME/work" "$HOME/work/zen" --output "$GRAPH"
python3 tools/audit_dependency_refs.py compact "$PARQUET" "$SOURCE_JSONL" "$GRAPH_JSONL"
```

Additional passes restrict `--only-repos` to dependent repositories and use
`--source-pattern` for the symbol census. The zenjpeg symbol supplement was
run on i265 because dev's checkout was busy. No checkout update was needed.
Repository discovery, active-lock exclusions and object-read errors are in
the per-pass `summary.json` files.

Focused validation on i265:

```sh
cargo build --locked -p zenpredict-bake --bin zenpredict-inspect
cargo test --locked --features api,hdr --test zenpicker_live_route -- --nocapture
cargo run --locked --features hdr --example list_features -- --variants
```

The live route result was 0 model / 75 heuristic. Six immutable public model
artifacts were read through zenpredict-inspect. The report distinguishes
these observations from unmeasured rate-distortion or deployment claims.
