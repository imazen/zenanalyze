# API audit evidence — 2026-10-28

Findings: [pre-publication API audit](../docs/API_NECESSITY_2026-10-28.md).

Evidence is preserved on **dev**, checked 2026-10-28, at:

`/mnt/v/output/zenanalyze/api-audit-2026-10-28/`

The `callers-complete/` directory contains immutable-ref source evidence,
checkout discovery and coverage/error/exclusion summaries. Its refs are dated
observations, not freshly fetched origin heads. Existing locks and uncommitted
changes were excluded. The first pass without an available jj executable is
not the completed census; `callers-complete.log` records the successful pass.

`probe.log`, `transforms-no-std.log`, `transforms-std.log`, `clippy.log` and
`tests-*.log` preserve the compiled probes and focused validation. `SHA256SUMS`
records their hashes and those of the census inputs. The source baseline is
`208ee7ec`; the two new examples and their justfile recipes are committed with
this pointer. Compilation used an isolated three-member workspace containing
zenpredict, zenpredict-bake and zenanalyze-api, with opt-level 2 and debug info
disabled; runtime library source was unchanged.

Reproduce with the host resource wrapper around `just api-necessity-probe`,
`just api-necessity-transforms`, or `just api-necessity-callers <output> <roots>`.
Do not commit raw census data: it contains local checkout metadata. No model,
numeric expectation or API was changed, and no new RD/performance result is claimed.
