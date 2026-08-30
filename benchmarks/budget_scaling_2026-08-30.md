# Should analysis thoroughness scale with image size? — 2026-08-30

**Verdict: no. Leave the budgets alone.** Fully-sampled analysis costs **24× more
at 4096²** and changes the shipped zenjpeg pick **0 % of the time there** — the
picker emits one constant cell at that size. Where picks do change (1024²/2048²)
the model's own predicted byte penalty is a median **0.26 % / 1.11 %**. The
expected byte saving across the corpus is ~**0.01–0.06 %** for a 4–24× analysis
cost increase.

**Two findings that came out of this measurement matter more than the budget
question, and both are actionable today.** They are written up first, below,
because the budget answer is "change nothing" and these are not:

- **[A. The zenjpeg picker's byte model breaks down above 2048²](#a-the-zenjpeg-picker-breaks-down-above-2048)** — it predicts a median
  **151 GB** for a 4096² JPEG whose real size is **2.3–8.7 MB**, and collapses to
  a single output cell.
- **[B. The three shipped zenpicker routers reuse nothing from current `main`](#b-the-shipped-cross-codec-routers-reuse-nothing-from-current-main)** —
  one drifted feature column makes `default_route` return `Ok(None)` for every
  offer, on every image, silently.

Neither is caused by sampling budgets; both were found by pointing the
instruments at real consumers.

Host: Apple M4 Pro, 12 cores, macOS 26.5.2, aarch64. Release profile
(`lto = "thin"`, `codegen-units = 1`), **no `-C target-cpu=native`**. Every cargo
command under `nice -n 19 … -j 4`. Commit `6afb6f118b20`.

Instruments (all committed, all deterministic — the raw rows below regenerate
from these with one command each):

| what | how |
|---|---|
| feature drift | `examples/budget_drift_grid.rs` |
| decision change | `examples/budget_decision_ab.rs` |
| cost | `benches/budget_cost.rs` |
| aggregation | `tools/budget_drift.py` |

```text
nice -n 19 cargo run --release --features hdr --example budget_drift_grid
nice -n 19 cargo run --release --features hdr --example budget_drift_grid   # DRIFT_NATIVE=1
nice -n 19 cargo run --release --features hdr,api --example budget_decision_ab
nice -n 19 cargo bench --features hdr --bench budget_cost
python3 tools/budget_drift.py --grid <grid.tsv> --native <native.tsv> --decisions <ab.tsv>
```

Committed data: the aggregated per-feature drift table
[`budget_drift_summary_2026-08-30.tsv`](budget_drift_summary_2026-08-30.tsv)
(117 features × 5 sizes) and the raw zenjpeg cell scores
[`budget_zenjpeg_cell_scores_2026-08-30.tsv`](budget_zenjpeg_cell_scores_2026-08-30.tsv)
(5 760 rows — the evidence for §A). The raw drift grids (26 MB + 11 MB) are
**not** committed; they regenerate deterministically from the harness above.

**The "default" arm is the shipping path, verified, not a proxy for it.**
`__analyze_internal` at `(500 000, 1 024)` was compared feature-by-feature with
`f32::to_bits()` against `analyze_features_rgb8(FeatureSet::SUPPORTED)` over
4 content classes × {1024, 2048, 4096}²: **zero differing features**. Without
that, every number here would be measuring the harness rather than the crate.

## A. The zenjpeg picker breaks down above 2048²

`zenjpeg/zenjpeg/src/encode/picker.rs` predicts log-bytes for each of 36
categorical cells and takes the argmin. Dumping the **shape** of those 36 scores
at the shipping budgets, over 1 152 (image × target) combinations per size —
6 content classes × 8 crops × 24 targets from zq 5 to 97 — against the **measured**
size of the same buffer encoded as JPEG on the same host:

| side | distinct cells picked | median predicted best-cell bytes | measured real JPEG (q50 – q95) | model error |
|---|--:|--:|--:|--:|
| 64 | 16 / 36 | 834 B | — | plausible |
| 256 | 18 / 36 | 4.23 KB | — | plausible |
| 1024 | 15 / 36 | 66.9 KB | 80 KB – 565 KB | plausible |
| 2048 | **3 / 36** | 1.87 MB | 400 KB – 2.28 MB | plausible |
| 4096 | **1 / 36** | **151 GB** | 1.74 MB – 8.71 MB | **~20 000–65 000×** |

The measured column is `image` 0.25's baseline JPEG encoder over the same mosaics
the picker was fed, across the four cost-grid classes (`benchmarks/` regenerable;
photo 4096² is 2.26 MB at q50, 4.67 MB at q85, 8.71 MB at q95).

Two distinct failures, and they start at different sizes:

1. **Output diversity collapses at 2048².** 15 distinct cells at 1024², 3 at
   2048², and at 4096² the picker returns `420/base/plain/balanced` for *every*
   input at *every* target from zq 5 to 97 — all six content classes, all 1 152
   combinations. A picker that ignores both content and quality target is not
   picking.
2. **The byte prediction explodes at 4096².** Growth in median predicted bytes
   per 4× pixel step: 1024→2048 is **28×** and 2048→4096 is **80 700×**, where
   the real ratio is ≈ 4× at both steps. The model is extrapolating off a cliff,
   almost certainly because it was trained on smaller renditions and the size
   inputs (`log_pixels`, `pixel_count`, `bitmap_bytes`) leave its fitted range.

It is **not** a near-tie that argmin resolves arbitrarily — the model is
confidently wrong. Spread between the worst and best cell *widens* with size
(median bytes ratio 1.45 at 1024², 1.85 at 2048², **7.28** at 4096²), and the
top-2 margin widens with it (1.016 → 1.066 → 1.170).

The bake declares no `feature_bounds` (`model.feature_bounds()` is empty), so
`first_out_of_distribution` — the OOD rescue seam the picker already calls — is
dormant and cannot catch this. Populating bounds at bake time would convert a
silent 20 000× error into a clean `None`, which the encoder already handles by
keeping its heuristic config.

Reproduce:
```bash
AB_DUMP_SCORES=1 cargo run --release --features hdr,api --example budget_decision_ab
```

**Bearing on the budget question:** the zenjpeg picker's 0 % decision-change rate
at 4096² in §2 below is this bug, not robustness. No sampling policy can improve
a decision that is already constant, so 4K is moot for the budget question
regardless of how it is answered.

## B. The shipped cross-codec routers reuse nothing from current `main`

Each of `zenpicker/benchmarks/zenpicker_router_{lossy,lossless,gate}_v0.1.bin`
declares 101 qualified `name@hex8` feature columns. **100 match this build. One
does not:**

```
routers want:  chroma_subsample_dct_loss@48f0f976
this build:    chroma_subsample_dct_loss@fabc9776
```

`Select::Features` is all-or-nothing by design, so one drifted column misses the
whole want-set. `MetaPicker::route` maps "the offer cannot satisfy a model's
columns" to `Ok(None)` — correctly, since the caller is meant to re-extract — so
**`zenpicker::default_route` returns `Ok(None)` for every offer this build can
produce, at every target, on every image.** Consumers fall back to `family_rule`,
the no-features prior, and cannot distinguish that from "no family survived the
mask".

The gate is doing exactly what it exists for: refusing to feed a compiled model a
re-defined column, which is the entire reason `Select::Features` exists rather
than `Select::Names`. The defect is that the bakes are stale relative to current
feature definitions, so the refusal is total rather than occasional — and silent.

Reproduce:
```bash
grep -a -o -E "[a-z_0-9]{4,}@[0-9a-f]{8}" \
  zenpicker/benchmarks/zenpicker_router_lossy_v0.1.bin | sort -u
```
and diff against `zenanalyze::versioning::feature_qualified_names()`. All three
bins carry the same stale column. End to end:
`AB_NO_BRIDGE=1 cargo run --release --features hdr,api --example budget_decision_ab`
returns `none` for every routed decision.

Fix: re-bake the three routers, and add a CI tripwire asserting every shipped
bake's declared columns resolve against `feature_qualified_names()`. The tripwire
is the part that matters — one assert would have caught this at the commit that
redefined `chroma_subsample_dct_loss`, and it generalises to every bake in the
tree (zenjpeg's `feature_order.txt`, zenavif's `rav1e_picker_v0_1_1.bin`, these
three routers). Filed as imazen/zenanalyze#88; blocks the decision
`docs/meta-picker-degradation-2026-08-28.md` is waiting on, since degradation
offsets threaded into a router that never runs buy nothing.

**Bearing on the budget question:** to measure routing at all, `budget_decision_ab`
re-qualifies that one column to the hash the routers expect. That is precisely
the silent substitution `Select::Features` exists to prevent, done deliberately
in a measurement harness and flagged in the source as something that must never
be copied into production. It is sound *there* only because it is applied
identically to both arms, so it cannot manufacture or mask a difference between
them — it only makes the router answer.

## What was measured

`DEFAULT_PIXEL_BUDGET` is a fixed *absolute* cap (500 000 px) and
`DEFAULT_HF_MAX_BLOCKS` is a fixed *absolute* 8×8-block cap (1 024), so the
sampled *fraction* shrinks as images grow: a 4096² image is 16.8 MP against a
500 k budget, ~3 % sampled. Each cell was analyzed under a ladder of
`(pixel_budget, hf_max_blocks)` arms through the `#[doc(hidden)]` re-extraction
backdoor, with the two knobs varied **independently** so drift could be
attributed to the right one.

Content: real codec-corpus images only, no synthetic gradients — `photo`
(CID22 / clic2025), `photohard` (gb82), `screen` (gb82-sc), `mixed`,
`screenwide` (gb82-sc + qoi-benchmark web screenshots), `lineart` (6 synthetic-
graphics / diagram sources picked by measured signature, not filename). 8 crops
per class per size.

Drift is reported in three normalisations. The one that decides things is
**σ = |full − default| / (across-image stdev of the default value at that size)**,
because a model sees z-scored inputs: a drift of 0.5 σ moves an input half a
standard deviation of the range the model was trained on, whatever its units.
Relative drift is also reported but blows up near zero (one feature shows
p95 rel = 3 × 10¹⁰ on a value that is legitimately 0), so it is not decided on.

## 1. Feature drift

Below 1024², **every feature is bit-identical across every arm** — the budgets
simply do not bind (64² = 4 096 px, 256² = 65 536 px, both under the 500 k cap).
That confirms the predicted shape rather than assuming it.

| side | features | moved at all | p95 σ ≥ 0.05 | p95 σ ≥ 0.5 | max p95 σ |
|---|--:|--:|--:|--:|--:|
| 64 | 87 | 0 | 0 | 0 | 0.000 |
| 256 | 117 | 0 | 0 | 0 | 0.000 |
| 1024 | 117 | 65 | 54 | 11 | 1.533 |
| 2048 | 117 | 66 | 64 | 33 | 3.729 |
| 4096 | 117 | 65 | 61 | 38 | 4.759 |

So drift is real and grows with size: at 4096², 65 of 117 features move, 38 of
them by more than half a standard deviation of their own corpus spread. The
worst are the tier-3 percentile family — `noise_floor_y_p75` (p95 σ = 4.05),
`noise_floor_uv_p75` (3.66), `aq_map_p75` (2.60), `patch_fraction` (2.38).

**52 of 117 features never move at any depth.** Those are safe under any
sampling regime — the geometry/derived features, the palette family, alpha, and
the depth tier.

### The dominant knob is `hf_max_blocks`, not `pixel_budget`

This corrects the premise the investigation started from, which centred on
`DEFAULT_PIXEL_BUDGET`. Attributing each feature's drift to whichever knob
produced it (`pbfull` vs `hffull` arms, each holding the other at default):

| side | moving features | driven by `hf_max_blocks` | driven by `pixel_budget` |
|---|--:|--:|--:|
| 1024 | 65 | 39 | 26 |
| 2048 | 66 | 40 | 26 |
| 4096 | 65 | 39 | 26 |

Both knobs matter, but the hf-driven group carries the large magnitudes — every
one of the top 25 features by p95 σ at 4096² is hf-driven. The tier-3 DCT block
cap is doing more to feature values than the tier-1/2 stripe budget is. Any
future work here should start with `hf_max_blocks`.

### Mosaic control

No local source is 4096², so the large sides are mosaics of distinct centre
crops, and a mosaic is spatially more heterogeneous than one photograph — which
could overstate drift. Re-running over whole native images (never resampled)
says it does not:

| side | source | features | median of med σ | median of p95 σ |
|---|---|--:|--:|--:|
| 1024 | native (63 imgs) | 85 | 0.0046 | 0.1182 |
| 1024 | mosaic | 79 | 0.0061 | 0.1512 |
| 2048 | native (2 imgs) | 70 | 0.0413 | 0.0680 |
| 2048 | mosaic | 74 | 0.0389 | 0.3564 |

Central tendency agrees closely at both sides. The p95 columns diverge at 2048²,
where the native control is only 2 images — too thin to read. **The local corpus
cannot control the mosaic at 4096² at all**; that caveat stands on the 4096² row
of every table here.

## 2. Does the decision change?

Feature drift only matters if it changes an output. The shipped cross-codec
routers were driven through their **real** entry point (`zenpicker::default_route`,
in this workspace); the zenjpeg picker and zenwebp classifier are mirrors of
their source (this crate cannot depend on crates that depend on it).

`distinct` is the number of different decisions the consumer emitted at that
size across every image and target. **It is what stops a 0 % change rate from
being misread**: a consumer emitting one constant decision cannot change, and
that is not robustness.

| consumer | side | decisions | changed | rate | distinct | most common |
|---|--:|--:|--:|--:|--:|---|
| meta_picker | 64 | 1152 | 0 | 0.00% | 1 | none (100%) |
| meta_picker | 256 | 1152 | 0 | 0.00% | 4 | Webp (38%) |
| meta_picker | 1024 | 1152 | 39 | 3.39% | 3 | Jxl (53%) |
| meta_picker | 2048 | 1152 | 8 | 0.69% | 2 | Avif (64%) |
| meta_picker | 4096 | 1152 | 45 | 3.91% | 2 | Jxl (58%) |
| zenjpeg_picker | 64 | 1152 | 0 | 0.00% | 16 | 420/prog/sharp/fast (18%) |
| zenjpeg_picker | 256 | 1152 | 0 | 0.00% | 18 | 444/prog/plain/max (14%) |
| zenjpeg_picker | 1024 | 1152 | 109 | 9.46% | 15 | 420/prog/sharp/balanced (17%) |
| zenjpeg_picker | 2048 | 1152 | 66 | 5.73% | 3 | 420/base/plain/balanced (73%) |
| zenjpeg_picker | 4096 | 1152 | 0 | 0.00% | **1** | 420/base/plain/balanced (100%) |
| zenwebp_classifier | 64..4096 | 48 ea | 0 | 0.00% | 1–2 | Photo / Icon |

Three things to read off this:

1. **The zenjpeg picker's 0 % at 4096² is the §A bug, not robustness.** It emits
   one constant cell there, so nothing about sampling could have moved it — which
   also makes 4K moot for the budget question either way. Full diagnosis,
   including the 151 GB byte prediction, in [§A](#a-the-zenjpeg-picker-breaks-down-above-2048).
2. **zenwebp's classifier never changes.** Expected: it is three live threshold
   rules on four features (`skin_tone_fraction`, `edge_slope_stdev`,
   `flat_color_block_ratio`, `distinct_color_bins`) with wide margins, and most
   images sit far from the cliffs at 0.15 / 35.0 / 0.50 / 4096.
3. The meta-picker changes 0.7–3.9 % above 1024², from a 2–3 way decision.

### Different is not better: the regret

For zenjpeg the bake's outputs are predicted **log-bytes**, so it can be asked
what the default-budget pick costs, priced under the better-informed fully
sampled feature vector. That is the picker's own objective, so a regret near zero
means the flip is churn by the picker's own accounting.

| side | flips | median | p95 | max |
|---|--:|--:|--:|--:|
| 1024 | 109 / 1152 | 0.2604 % | 1.3132 % | 4.1738 % |
| 2048 | 66 / 1152 | 1.1101 % | 2.8035 % | 3.6664 % |
| 4096 | 0 / 1152 | — | — | — |

Decomposing the 175 flips by which knob moved:

| knob moved | flips | share |
|---|--:|--:|
| effort | 127 | 72.6 % |
| progressive | 85 | 48.6 % |
| sharp_yuv | 61 | 34.9 % |
| **subsampling** | **60** | **34.3 %** |

Subsampling is the knob with real byte consequences, and flips that move it
carry median 1.18 % regret vs 0.29 % for flips that do not. But a subsampling
flip occurs in only 60 of 5 760 decisions at ≥ 1024² (**1.04 %**), so the
expected corpus-wide byte saving from full sampling is roughly
`0.0104 × 1.18 % ≈ 0.012 %`.

**Limitations of the regret number, stated plainly.** It is the model's opinion,
not a measured encode. Worse, it is a model trained on default-budget features
being asked to price full-budget features — which is exactly the off-distribution
substitution this whole area is about. A ground-truth encode-and-score would
break that circularity and was **not** run: it needs zenjpeg (236 deps, a C FFI
sub-crate) plus a perceptual metric, on a host with 17 GiB free. Given the
expected saving is ~0.01 % and the cost is 4–24×, a ground-truth measurement
would have to be off by two orders of magnitude to change the recommendation.
That is the reasoning for stopping here, not a claim that the question is closed.

## 3. What does full sampling cost?

Measured at each size, never extrapolated. Both budget arms go through the
**same** entry point so "budget cost" is not confounded with "dispatch cost";
`public` is `analyze_features_rgb8` at default budgets, and it tracks `default`
to within noise at every size, confirming the internal path is a fair proxy.

| side | default | hffull | full | full ÷ default |
|---|--:|--:|--:|--:|
| 256 | 1.8 ms | 1.8 ms | 1.8 ms | 1.00× |
| 1024 | 5.9 ms | 20.2 ms | 24.1 ms | **4.08×** |
| 2048 | 8.1 ms | 68.7 ms | 95.7 ms | **11.8×** |
| 4096 | 16.1 ms | 257.0 ms | 381.6 ms | **23.7×** |

The hf knob alone accounts for most of it (257 of 382 ms at 4096²) — consistent
with it being the dominant drift driver.

### Against encode time on the same bytes, same host

| side | analysis (default) | analysis (full) | `image` JPEG q85 | `image` PNG |
|---|--:|--:|--:|--:|
| 1024 | 5.9 ms | 24.1 ms | 12.0 ms | 6.0 ms |
| 4096 | 16.1 ms | 381.6 ms | 139.4 ms | 73.5 ms |

At 4096², default analysis is **12 % of a fast JPEG encode**; fully-sampled
analysis is **2.7× that encode**. These are the `image` crate's encoders — a
baseline JPEG with no trellis and PNG at default effort — which sit at the *fast*
end. Against a heavier production codec (mozjpeg at high effort, AVIF, JXL)
analysis would be a smaller share, not a larger one. No number is quoted for a
codec that was not run here.

## 4. Which design does the data support?

Three candidates were on the table.

**Option 2 — a convergence knob ("legal above some sampling floor") — is ruled
out by the data.** A feature qualifies only if its error against the fully
sampled value is monotonically non-increasing along the ladder of the knob that
drives it, on essentially every cell. Testing each feature on the ladder of *its
own* driver (testing an hf-driven feature along the pixel-budget ladder passes
vacuously, since its value is constant there):

| at 4096² | count |
|---|--:|
| never move — safe at any depth | 52 |
| move, converge on **all** cells | **4** |
| move, converge on **some** cells | 59 |
| move, converge on **no** cells | 2 |

Only 4 of 65 moving features converge cleanly, and 3 of those 4 have p95 σ =
0.000 — they barely move at all. As predicted, the percentile- and max-type
statistics dominate the non-converging group: a p75 of a sampled population is
not a monotone function of sample count, and no amount of extra sampling makes
it one. **There is no useful set of features for which a sampling floor is
sound.**

**Option 1 — thoroughness in the identity (`edge_density@v3.deep`) — is the only
sound one**, and the codebase already agrees: the `full-budgets` cargo feature
carries the note *"A model trained on full-budget features must be served them at
inference"*, which is exactly option 1 expressed as a build flag. If depth is
ever varied, it belongs in `feature_defs_version` / the qualified name so a
mismatched model misses instead of silently eating substituted inputs.

**Option 3 — a size-conditioned budget as a release decision — is technically
sound** (bump the version, retrain once) but the data says it would buy ~0.01 %
bytes for 4–24× analysis cost, and nothing at all at 4096² where the zenjpeg
picker is already constant.

### Recommendation

**None of the three. Leave the budgets as they are.** Three independent reasons,
any one of which is sufficient:

1. The cost is 24× at 4K against an expected saving of ~0.01 % bytes.
2. At the size where drift is largest, the biggest consumer emits a constant, so
   there is no decision left to improve.
3. Every model in the tree was fit at the current depth. Changing depth without
   retraining is the silent-substitution failure the contract's
   `Select::Features` gate exists to prevent; changing it *with* a retrain of
   every consumer costs far more than 0.01 % bytes is worth.

If anyone revisits this, the ordering the data implies is: fix the degenerate
zenjpeg picker first (it is a bigger lever than sampling), then look at
`hf_max_blocks`, not `pixel_budget`.

## Smaller corrections

The two consequential findings are §A and §B at the top. These are the minor
ones, also independent of the budget question.

### `benches/tier_isolation.rs` has an inverted comment

Lines 68–71 say the default `experimental` feature "raises the sampling budget to
full (pixel_budget = usize::MAX)". It does not — that is `full-budgets`, which is
**off** by default (`Cargo.toml`: `default = ["experimental"]`, `full-budgets = []`).
The comment then concludes the 1024² and 2048² cases "would do the same amount of
work" without it, which is backwards: with default features the budgets *do*
bind, and it is the *capped* case that flattens the size axis. Corrected in the
same change as this report.

### Corrections to the crate index

- `~/.claude/CLAUDE.md` lists `zenwebp_picker_v0.1.bin` as "wired into a codec in
  production". It does not exist — `find zenwebp -name '*.bin'` outside `target/`
  returns nothing, and `zenwebp/CLAUDE.md` records its removal on 2026-07-01
  (`5d6df59`). zenwebp's live content decision is the hand-threshold classifier
  in `src/encoder/analysis/classifier.rs`, not a model.
- That classifier requests 10 features but only **4** reach a decision;
  `screen_content`, `text_likelihood`, `natural_likelihood` and `line_art_score`
  are hardcoded to `0.0` by its only populated constructor because those features
  were culled from zenanalyze, so every rule reading them is dead. Six of the ten
  requested features are extracted and discarded.

## Harness limitations

- **4096² content is necessarily mosaicked** — no local source is that large, and
  the native control cannot reach it.
- **`screen` aliases at large sizes.** Its 8-source pool divides the mosaic tile
  count, so all 8 crops are identical at 2048² and 4096² (effective n = 1). Use
  `screenwide` (22 sources) for screen-content percentiles there. The class is
  kept as-is because the cost grid and the `feature_bits` bitlock measure it.
- **`lineart` is thin** — 6 sources, two of them Frymire variants.
- **The 64² meta-picker row is a harness artifact.** `__analyze_internal` has no
  mirror-tile recovery (that lives in the public `analyze_features`), so ~30
  percentile features are NaN at 64² and the router declines. Production would
  recover them. It does not affect any conclusion: budgets cannot bind at 64².
- **zenjpeg and zenwebp are mirrors, not the real code.** Each asserts what it
  can about its source artifact and cites the file it mirrors, but a change over
  there could make these stale.
- **zenavif's `auto_tune` was not covered.** Its bake wants 43 columns, several of
  which no longer exist in this build and are zero-filled by the real code, and
  it expands to a 96-dim engineered vector plus two JSON LUTs. Mirroring that
  faithfully would more likely measure the transcription than the behaviour.
