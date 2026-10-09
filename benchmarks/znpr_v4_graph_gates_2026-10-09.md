# ZNPR v4 graph runtime — gate record (2026-10-09)

Tree: zenanalyze `zenpredict` / `zenpredict-bake` at the `perf(zenpredict): vectorize the
Input-node scaler` commit of the ZNPR v4 PR. Old side everywhere: zenanalyze `main` at
`e6c72f99390c190c102275c3b0e4d41fe0a734a4` (the last commit before v4). Host: `dev`
(Ryzen 9 9950X3D, Zen 5), rustc 1.99.0 stable, shared box, every heavy job under
`run-heavy` (peak RSS ≤ 4.23 GiB, the zensim test suite). Spec: `zenpredict/docs/ZNPR_V4_GRAPH.md`.

## 1. Bit-identity vs the pre-graph runtime (`zenpredict/tools/graph-parity`)

The harness links both zenpredict copies in one binary. For each file it compares:
- the load verdict: a summary of dims, transforms and per-layer weight/bias hashes, or
  the error's `Debug` text;
- `predict`, `predict_transformed`, `predict_with_specs` and `argmin_masked`, by bit
  pattern, over realistic (`mean ± 3·scale`), wide-range, special-value (NaN, ±inf,
  subnormal, ±MAX) and wrong-length vectors.

| input set | files | loaded | vectors | output values compared | mismatches |
|---|---:|---:|---:|---:|---:|
| unique (sha256) ZNPR files under `~/work/zen`: repo fixtures, zensim/zenjpeg/zenavif/zenjxl/jxl-encoder bakes | 260 | 127 | 510,624 | 13,202,598 | 0 |
| rev4 production bake `production-f16.bin` (sha256 `f803b74c…`), `ZPGRAPH_VECTORS=50000` | 1 | 1 | 100,006 | 100,002 | 0 |
| every 77th of 38,870 unique rev4-featpot bakes, `ZPGRAPH_VECTORS=300` | 504 | 504 | 305,424 | 303,408 | 0 |
| `synth 2000`: random v3 chains baked with BOTH composers (bytes required identical) | 2,000 | 2,000 | 1,611,726 | 27,170,156 | 0 |

The 133 files that don't load are the 127 ZNPR v2 files (rejected as `UnsupportedVersion`)
plus 6 stale v3 bakes whose metadata fails validation. Both sides reject all 133 with
identical errors.

Negative control: multiplying the new scaler by `1.0000001` made 50 of 50 synth cases
mismatch.

```text
just graph-parity <files…>     # = cargo run --release -- synth 2000; … check <files…>
```

Log sha256:
- synth `f1a040ae…`
- production `afa20959…`
- fixtures `60fcf379…`
- rev4 sample `92e25d45…`

## 2. zensim against this zenpredict (not committed to zensim)

zensim `main` `40ebb3b8`, built in a throwaway jj workspace with the patch passed on the
command line (no manifest edit):

```text
cargo test -p zensim --all-features \
  --config 'patch."https://github.com/imazen/zenanalyze".zenpredict.path="<this tree>/zenpredict"' \
  --config 'patch."https://github.com/imazen/zenanalyze".zenpredict-bake.path="<this tree>/zenpredict-bake"'
```

- Tests: 910 passed, 0 failed, 27 ignored (all pre-existing).
- Scores: CID22-512 validation, 41 references × 4 synthetic distortions = 164 pairs, f64
  score bits from the pinned zenpredict (`05de3cbc`) vs this tree. Both sets are
  identical:
  - `BakeScorer` over the production bake (`ZENSIM_FORMULA_REV=5`): sha256 `ff6d712e…`
  - default profile B: sha256 `df324959…`

## 3. Gated head (E33 arm B)

`zenpredict-bake/examples/gated_head.json` (7 nodes: Input, Gather d, Gather r, Dense+ReLU
no bias, Dense+Exp no bias, Mul, Dense).

- `tests/graph.rs::gated_head_is_exactly_zero_when_d_is_zero`: output bits == `+0.0` for
  20,000 finite `r`, drawn from any finite bit pattern, ±50 and ±1e30.
- `gated_head_matches_reference`: bit-identical to a naive per-node evaluator on 5,000
  random inputs.
- `random_graphs_match_reference_evaluator`: 400 random DAGs over every op, activation
  and dtype.
- Mutation check: freeing dead inputs before allocating a node's slot fails 8 tests.

## 4. Speed (old chain vs graph executor, same bytes)

Summary: the production bake is 1.7 % faster (2.2–2.5 % in the round-1 rerun); synthetic
shapes land within −7 % … +2 % across runs on this shared box, so the wall-clock rows below
for them are noise-limited and the `perf stat` counters are the stable comparison.

zenbench, paired and interleaved, `just graph-parity-bench '<production bake>'`. 95% CI
vs old:

| model | old | new | 95% CI |
|---|---:|---:|---|
| rev4 production bake | 42.9 µs | 41.9 µs | [−1.8%, −1.7%] |
| 944→128→1 f32 | 6.3 µs | 6.6 µs | [+0.0%, +7.3%] (261 noisy rounds) |
| 228→384→1 i8 | 5.5 µs | 5.4 µs | [−6.8%, −1.8%] |
| 51→64→24 f16 | 3.4 µs | 3.5 µs | [+0.2%, +0.4%] |

Wall clock on this shared box swings by several percent between runs on the 944-wide
shape: three earlier runs gave [−0.1, +1.4], [+3.3, +4.3] and [−6.4, −2.2]. So that row
was settled with deterministic counters instead. Method: `perf stat -r 10`, pinned to one
core, `examples/predict_loop.rs`, per predict (setup amortized equally on both sides):

| model | instructions old → new | cycles old → new |
|---|---|---|
| 944→128→1 f32 | 195,100 → 192,780 (−1.2%) | 33,611 → 32,145 / 33,657 → 32,451 (−4.4% / −3.6%) |
| 228→384→1 i8 | 151,800 → 151,200 (−0.4%) | 32,730 → 32,620 (−0.3%) |
| 51→64→24 f16 | 118,700 → 118,700 (0.0%) | 17,430 → 17,480 (+0.3%) |
| production bake | 1,271,000 → 1,270,000 (−0.1%) | 225,700 → 223,600 (−0.9%) |

The first full bench of the executor showed the 944-wide shape slower. That traced to the
Input-node scaler losing vectorization: four indexed slices meant bounds checks in the
loop. It is now a zipped `#[autoversion(v3)]` loop, bit-identical (`simd_parity_tests`).

Logs:
- `bench_final2` sha256 `cc5e3514…`
- `perfstat` sha256 `fd22f49c…`

## 5. Fuzz

`just fuzz-graph 900 8` (nightly, cargo-fuzz, 8 workers × 900 s per target):
- `graph_from_bytes`: seeded from `fuzz/seeds/graph_from_bytes/`. 1,219,631 runs in the
  surviving worker's final stats; the other workers' logs were overwritten by the second
  target.
- `graph_structured`: 24,006,486 runs across 8 workers, coverage 1,905 edges / 8,331
  features.
- 0 crashes, 0 artifacts.
- Corpus (26,114 files, 104 MB) mirrored to `/mnt/v/fuzzes/zenpredict/{corpus,artifacts,regression,seeds}`.

`tests/fuzz_regression.rs` replays seeds/ and regression/ on stable.

## 6. Hygiene

- `cargo fmt --check`; `cargo clippy -D warnings` (all features / no default features /
  harness); `cargo test` with all features and with no default features (526 passed,
  0 failed); `wasm32-unknown-unknown` no_std build; `ZEN_API_DOC=check` snapshot
  current.
- `cargo semver-checks --baseline-root <main checkout>`: zenpredict and zenpredict-bake
  both "no semver update required".
- `RUSTDOCFLAGS=-D warnings cargo doc` fails on 14 pre-existing private-item / HTML-tag
  links (feature_transform, knob_veto, picker_safety, rescue, unachievable_zone, cli
  repack docs). None of them are in v4 code.

## Fix round 1 (review FIX-FIRST, same day)

Changes: zenpredict-viz refuses v4 (`load_chain_model`; P1); `limits::MAX_TOTAL_ELEMS`
op-weighted compute budget (P2); `NodeView` per-variant `#[non_exhaustive]` + `Clone` + `u32`
indices + `width()`, public `NodeEntry`, `BakeNode` constructors, `GraphBakeRequest`,
`deny_unknown_fields` on graph JSON (P2 API shapes); refusal tests, golden Exp/Softplus bits,
differential `graph_structured` fuzz target (P3). The production bake uses 53,888 of
`MAX_TOTAL_WEIGHTS` and 678 of `MAX_TOTAL_ELEMS`.

| gate | result |
|---|---|
| parity | 0 mismatches: 260 fixtures (list rebuilt by sha256 — 79 representative paths had vanished with another lane's removed workspace; all 260 contents found elsewhere), production bake 50k vectors, 504 rev4 bakes, synth 2,000 / 27,170,156 values. Logs `a8c392c7` / `afa20959` / `92e25d45` / `f1a040ae` |
| zensim | zensim `main` `713d2d73` + `--config <patch.toml>`: `cargo test -p zensim --all-features` 912 passed / 0 failed / 27 ignored (`ef49cf7c`); BakeScorer (production bake) and profile-B scores on 164 pairs bit-identical to the unpatched build (hashes `ff6d712e…`, `df324959…`, unchanged from round 0) |
| tests | 703 pass across zenpredict / zenpredict-bake (all features, no_std, no_std+std) and zenpredict-viz (`onnx-export`); wasm32 builds for zenpredict (no_std) and zenpredict-viz |
| semver | `cargo semver-checks` vs main: both crates "no semver update required" (adding `Copy` to `LayerView` was flagged major, so `Copy` is queued instead) |
| bench | zenbench (`9682a7eb`): production 44.5 → 43.8 µs, CI [−2.5 %, −2.2 %]; 944×128 f32 [−1.0 %, +2.0 %]; 228×384 i8 [−1.2 %, +0.2 %]; 51×64×24 f16 [+0.2 %, +0.4 %]. `perf stat -r 10` per predict (`81487b14`): cycles 944×128 −2.5 %, 228×384 −0.2 %, 51×64×24 +0.04 %, production −0.6 %; instructions −1.2 % … −0.0 % |
| fuzz smoke | 8 workers × 120 s per target: `graph_from_bytes` 1,039,574 runs, `graph_structured` (now differential vs the reference evaluator, replaying the 22,016-file corpus first) 1,608,567 runs; 0 crashes, 0 mismatches. Corpus mirrored to `/mnt/v/fuzzes/zenpredict` (26,763 files) |
| found and fixed | the differential seed replay failed under `--no-default-features`: a no_std zenpredict accumulates Dense with `a * b + c` (documented); the reference now takes the build's rule explicitly |
