# Dispatch axes — what `analyze_features` runs for a given request

Source of truth: `analyze_features` and `analyze_specialized_raw` in
`src/lib.rs`, and the gating `FeatureSet` constants in `src/feature.rs`.
This page codifies what is already implemented so #50's "lazy by request"
(Sub-C) and "don't analyze when the picker isn't used" (Sub-G) stop being
re-investigated. `tests/docs_dispatch_axes.rs` fails if a dispatch knob is
added to `analyze_specialized_raw` without being named here.

## The request decides the work

`analyze_features(slice, query)` looks only at `query.features()` (plus the
two `AnalysisQuery` mode bits) and derives every gate below **before** any
pixel is touched. Nothing runs "because it is in the default set" — a tier
runs only if at least one requested feature needs it. The corollary for
callers is in
[`zentrain/FOR_NEW_CODECS.md`](../zentrain/FOR_NEW_CODECS.md#step-75--request-only-what-the-bake-consumes):
request exactly the bake's `feat_cols`, nothing more.

## Const-bool axes (4 → 16 monomorphized arms)

`analyze_specialized_raw::<PAL, T2, T3, ALPHA>` is instantiated for every
combination; the `dispatch!` macro in `analyze_features` picks the arm. Each
axis is a whole pass whose inner loop benefits from being compiled out.

| axis | true when the request intersects | pass it gates |
|---|---|---|
| `PAL` | `PAL_NEEDED_BY` (= `PALETTE_FEATURES`) | palette scan (full or quick, see below) |
| `T2` | `TIER2_FEATURES` | Tier 2 three-row sliding-window Cb/Cr sharpness (`tier2_chroma::populate_tier2`) |
| `T3` | `T3_NEEDED_BY` (= `TIER3_FEATURES`) | Tier 3 luma histogram + (optionally) the sampled 8×8 DCT walk (`tier3::populate_tier3`) |
| `ALPHA` | `ALPHA_FEATURES` | alpha pass (`alpha::scan_alpha`, reads source bytes directly) |

Tier 1 has no axis: it is always run (every practical request wants at least
one Tier 1 feature; a fifth axis would double the table for no measured
caller).

## Runtime sub-gates (booleans passed into the arm)

These are ordinary `bool` parameters of `analyze_specialized_raw`, evaluated
once per call. They are cheap branches around a pass or a sub-pass, so they do
not need monomorphization.

| parameter | true when | effect |
|---|---|---|
| `palette_full_required` | request intersects `PALETTE_FULL_FEATURES` (`DistinctColorBins` / `Chao1` / `PaletteDensity`) | full-image `scan_palette`; otherwise `scan_palette_quick` (early-exits once the running count passes 256 — typically within ~10 rows on photos) |
| `palette_wants_grayscale` | `GrayscaleScore` requested | enables the per-pixel max/min gate inside the full palette scan |
| `tier1_wants_laplacian` | any `LaplacianVariance*` (mean, P50/P75/P90/P99, Peak) | runs the separate Laplacian SIMD row pass |
| `tier1_full_kernel` | request intersects `TIER1_FULL_FEATURES` | enables the `Variance` / `Colourfulness` / `EdgeSlopeStdev` / `LaplacianVariance` accumulators in `accumulate_row_simd` |
| `tier1_wants_skin` | request intersects `TIER1_SKIN_FEATURES` | BT.601 chroma matrix + Chai-Ngan skin thresholds (peeled off the full kernel: 12 vmovups + 13 broadcasts of spill traffic when bundled) |
| `run_depth` | request intersects `DEPTH_FEATURES` (empty unless the `hdr` cargo feature is on) | `tier_depth::scan_depth`, reads source samples directly; SDR sources short-circuit to the canonical profile (no walk) |
| `run_dct` | request intersects `DCT_NEEDED_BY` | the ~0.97 ms/MP 8×8 DCT walk inside Tier 3; with `T3 && !run_dct` only the luma-histogram pass runs (entropy / line-art) |
| `run_strict_gray` | `IsGrayscale` requested | `grayscale::scan_strict_grayscale` — walks rows, exits at the first non-gray pixel (~6 µs on colored content, a few ms on a truly gray 4 MP image) |
| `run_xyb444` / `run_xyb_bq` / `run_csl` | `Xyb444ColorLoss` / `XybBquarterChromaLoss` / `ChromaSubsampleDctLoss` requested (`experimental`) | re-walk the `RowStream` for the XYB / subsample color-loss features |
| `run_linear_light` | `query.linear_light()` | `RowStream::new_normalized_linear` — every content tier reads diffuse-white-normalized linear RGB; off by default (zero-copy fast paths) |
| `run_clip` | `run_linear_light && query.diffuse_white_clip()` | clamp the content tiers at diffuse white |

Two more parameters are budgets, not gates: `pixel_budget`
(`DEFAULT_PIXEL_BUDGET`) and `hf_max_blocks` (`DEFAULT_HF_MAX_BLOCKS`).
`analyze_features` always passes the crate-invariant defaults (the
`full-budgets` cargo feature const-folds them to MAX / 4096); only the test /
oracle override path varies them.

**They bind only above ~1 MP, and raising them is measured to not be worth it.**
Both are fixed *absolute* caps, so the sampled fraction shrinks as images grow
(4096² is 16.8 MP against a 500 k budget, ~3 % sampled). Below 1024² they do not
bind at all — every feature is bit-identical across every budget arm. Above it,
65 of 117 features move, but fully sampling costs **24× at 4096²** and changes
the shipped zenjpeg pick **0 %** of the time there. Full study, including why a
per-feature convergence floor is unsound (only 4 of 65 moving features converge
monotonically): `benchmarks/budget_scaling_2026-08-30.md`.

Note which knob matters if you do touch this: **`hf_max_blocks` dominates**.
Every one of the 25 largest feature drifts at 4096² is driven by the tier-3 8×8
block cap, not by `pixel_budget`.

## Pass order inside an arm

1. `scan_alpha` (if `ALPHA`) — source bytes.
2. `scan_depth` (if `run_depth`) — source bytes.
3. `RowStream::new` / `new_normalized_linear` — Native (zero-copy) vs Convert.
4. Palette scan (if `PAL`; full vs quick per `palette_full_required`).
5. `populate_dimensions` — descriptor math, always (~10 ns).
6. Tier 1 (`extract_tier1_into_dispatch`, `u8` or `f32` per `run_linear_light`).
7. Tier 2 (if `T2`, width/height ≥ 3).
8. Tier 3 (if `T3`; DCT walk if `run_dct`).
9. Strict grayscale (if `run_strict_gray`).
10. XYB / chroma-subsample color-loss re-walks (`experimental`, if requested).

`RawAnalysis::into_results` then keeps only the requested features.

## What each pass costs (so you know what a request buys)

Per-**feature** cost is the wrong granularity — features share passes, so
requesting any one feature of a tier runs that tier. Measured per-tier solo
cost at 4 MP RGB8 SDR (7950X, `examples/per_tier_cost.rs`,
`benchmarks/per_tier_cost_2026-06-18.md`):

| tier | 4 MP |
|---|--:|
| alpha | 2.3 ms |
| tier2 (chroma sharpness) | 2.4 ms |
| tier1 | 3.6 ms |
| tier3 histogram | 3.6 ms |
| palette (full) | 4.5 ms |
| tier3 DCT | 5.6 ms |
| depth (after the SDR short-circuit was restored) | 2.3 ms |

Size scaling (α + β·pixels per cumulative tier subset, 64²–4096², real photo
crops): `examples/per_tier_cost_grid.rs` →
`benchmarks/feature_cost_grid_2026-07-02.tsv`. That grid is the size sweep
#50's Sub-A asked for, at tier granularity.

Per-**feature** granularity (solo + leave-one-out, same five sizes, photo AND
screen crops): `examples/per_feature_cost_grid.rs` →
`benchmarks/per_feature_cost_grid_2026-08-28.tsv`, joined to the downstream
consumption inventory in `docs/feature-consumption.md` ("Cost vs use") by
`tools/feature_inventory.py --cost`. Solo cost is dominated by the pass floor
(~2 ms for any single Tier-3 feature at 2048², aarch64) — the LOO column is the
one that tells you what dropping a feature from a request actually saves.

> **Correction (2026-08-28): the "2.7 ms + 0.98 ns/px" fit for `SUPPORTED` is
> an artifact, not a per-call floor.** The measured 64² call is **0.63 ms**,
> four times smaller than that α. The fit is dominated by the two largest sides
> and the cost is not affine in pixels — the sampling budgets cap several
> passes, so marginal cost per pixel falls 26× across the sweep (16.2 ns/px
> between 64² and 256², 0.62 ns/px between 2048² and 4096²). Read the measured
> cell for your size, not the fit. Full table, per content class, before/after:
> [`benchmarks/perf_2026-08-28.md`](../benchmarks/perf_2026-08-28.md).
>
> That file also carries the 2026-08-28 optimizations (whole-pass 1.04× at
> 1 MP, 1.11× at 4 MP, up to 1.35× at 16 MP, byte-identical output) and the
> note that the `screen` class in every grid before that date was sourced from
> `gb82`, which is the *photographic* set.

## What is NOT implemented (and why)

- **Implication short-circuits (#50 Sub-D)** — e.g. `is_grayscale → skip Tier
  2 and the UV half of Tier 3`. Not done: the analyzer would then emit
  *implied* values (zeros / saturations) that differ from computed ones, and
  every shipped picker was trained on computed values. Needs an explicit
  opt-in on `AnalysisQuery` plus re-validation of the pickers.
- **Cross-call caching (#50 Sub-E)** — not in the analyzer. The features are a
  pure function of `(pixels, query)`; a codec that analyzes the same image
  twice (verify / rescue paths) holds its own `AnalysisResults` — see the
  caller-side pattern in `FOR_NEW_CODECS.md`. An analyzer-owned cache would
  add a hash of the pixels to every one-shot call.
- **Adaptive second pass (#46)** — gated on corpus evidence, see that issue.
