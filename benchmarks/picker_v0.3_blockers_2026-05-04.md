# Picker v0.3 train arc — structural blockers, 2026-05-04

Investigation of the "train + bake + A/B + ship v0.3 picker for zenwebp,
zenavif, zenjxl" mission, against the fresh 2026-05-03 zen-metrics
sweep TSVs at `s3://zentrain/sweep-2026-05-03/` (also local at
`~/work/zen/<codec>/benchmarks/<codec>_pareto_2026-05-04_extended.tsv`).

**Verdict: cannot proceed end-to-end with the current pipeline. Three
structural blockers, all of which apply identically to all 3 codecs.**

## Blocker 1 — Pareto TSV schema mismatch

`zentrain/tools/train_hybrid.py::load_pareto` (lines 550–635) expects
the columns:

```
image_path, size_class, width, height, config_id, config_name,
bytes, <METRIC_COLUMN>  (default "zensim")
```

The fresh sweep TSV (`zen-metrics-cli` 0.3.0 `sweep` subcommand)
produces:

```
image_path, codec, q, knob_tuple_json, encoded_bytes, encode_ms,
decode_ms, score_zensim, score_ssim2, score_dssim
```

Missing fields the trainer requires:
- `size_class` — categorical bucket the trainer joins against the
  features TSV `(image_path, size_class, width, height)` key.
- `width`, `height` — used both for the join key and as scaling hints.
- `config_id` — stable integer config identifier; the trainer keys
  per-config Pareto frontiers on this.
- `config_name` — passed to `parse_config_name(name)` to extract
  categorical + scalar axes.
- `bytes` — renamed to `encoded_bytes` in the new TSV.
- `zensim` (or other METRIC_COLUMN) — renamed to `score_zensim` in
  the new TSV.

There is no current adapter from new → old schema in the trainer or
in zentrain/tools. The new schema also has no `size_class` concept at
all: the sweep treats each image as a single (image, q, knob) row,
not as `(image, size_class, q, knob)`.

## Blocker 2 — No matching features TSV for the sweep corpus

The fresh sweep corpus is `~/work/zentrain-corpus/mlp-tune-fast/`,
587 images across 7 sub-corpora:

```
cid22-train         clic-1024-train    clic-train
gb82-photo          gb82-screen        kadid10k
size-dense-renders
```

The most recent existing features TSVs are:

| Codec   | Features file                                                             | Image count | Stems matching sweep |
|---------|---------------------------------------------------------------------------|------------:|---------------------:|
| zenwebp | `zenwebp_features_expanded_2026-05-02.tsv`                                |         597 |                  250 |
| zenavif | `zenavif_features_expanded_2026-05-02.tsv` / `_merged_2026-05-02.tsv`     |   ~597/~700 |                  ≤TBD |
| zenjxl  | `zenjxl_features_multiaxis_only_2026-05-02.tsv`                           |   varies    |                  ≤TBD |

For zenwebp specifically:
- 587 unique sweep images
- 250 stems match an entry in the features TSV (via basename-stem
  comparison, stripping the `<corpus>__` sweep prefix)
- 337 sweep images (~57%) have no feature row at all

The remaining corpora — `kadid10k` (large, distortion-heavy),
`clic-1024-train` (synthetic upscales), and `size-dense-renders`
(synthetic per-image size sweep) — were added to the sweep on
2026-05-03 but have not been pushed through the feature extractors.

Per the global CLAUDE.md sweep discipline rules, training a picker
on the 250-image overlap subset is not OK either:
- Drops kadid10k entirely (the screen-content / distortion class).
- Drops size-dense-renders, the size-axis training data the v0.3
  picker is supposed to learn from.
- Reduces effective n per (corpus, q, knob) cell below the
  "≥ 50 per content class for percentile fits" floor.

## Blocker 3 — Sweep grid is a single binary knob per codec

The sweep emits 2 cells per codec (`{"method":[4,6]}`,
`{"speed":[6,8]}`, `{"effort":[3,7]}`). The existing v0.1/v0.2
pickers are designed around 6 cells per codec (e.g. zenwebp
`method × segments`) plus 3 per-cell scalar heads
(`sns_strength`, `filter_strength`, `filter_sharpness`).

A picker trained against the new 2-cell grid would:
- Have a different output layout (`1 × 2 × bytes_log` rather than
  `4 × 6 × {bytes_log, sns, filter, sharp}`).
- Not be drop-in compatible with the existing
  `src/encoder/picker/spec.rs` runtime (schema_hash mismatch on
  load).
- Cover only the binary `method`/`speed`/`effort` knob — none of
  the per-cell scalars the production runtime expects.

This is fine **if** the v0.3 plan is "pick `method ∈ {4,6}` from
features" as a minimum-viable picker shipped via external A/B
harness rather than in-runtime, and the user is OK with the older
in-runtime picker continuing to ship for sns/filter/sharpness. But
it is not what the existing trainer + bake + picker spec are wired
to consume.

## What landed today (committed)

- imazen/zenanalyze main, commit `f44556c8` —
  `loo: multi-seed LOO zenjpeg + cross-codec consensus update`.
  Picks up the prior `claude-multiseed-loo` agent's uncommitted
  benchmark results + the `sweep_2026-05-04_results.md` summary
  doc that documents the new sweep itself.

## What's needed before a v0.3 picker arc can complete

In the order they unblock the next step:

1. **Schema bridge**, one of:
   - Adapter in `zentrain/tools/` that converts the new sweep TSV
     to the old schema. Stable per-row `config_id` from
     `hash(knob_tuple_json + q)`. `config_name` synthesized as
     `q{q}_<knob_kv>` (e.g. `q75_method4`). `size_class` derived
     from `(width, height)` bucketing. `bytes` aliased from
     `encoded_bytes`. `zensim` aliased from `score_zensim`.
   - Or extend `train_hybrid.py::load_pareto` to consume the new
     schema natively (zen-metrics 0.3.0+ format).

2. **Refresh features TSVs** to cover the full `mlp-tune-fast` corpus
   (587 images, 7 sub-corpora). Each codec's feature extractor must
   run against the corpus. `zentrain/tools/refresh_features.py` is
   the entry point but its CODECS recipes point at older manifest
   paths — they need updating to take
   `~/work/zentrain-corpus/mlp-tune-fast/manifest.tsv` (which itself
   needs to be authored — list of all 587 image paths).

3. **Sweep grid scope decision**, one of:
   - (a) Accept the 2-cell binary-knob grid and ship a v0.3 picker
     that lives entirely in the external A/B harness, with codec
     runtimes unchanged (per "DO NOT touch in-codec picker glue"
     rule). The output is just `pick(image, q) → (knob,)` — a small
     bake.
   - (b) Re-run the sweep with a richer grid — at minimum 6 cells
     per codec — so the v0.3 trained picker can subsume v0.2.
     This is overnight-budget work and re-stages everything from
     scratch.

(a) is consistent with "codecs stay DUMB, integration is external"
and is achievable in a single session once blockers 1 and 2 are
resolved. (b) is a bigger arc.

## Recommendation

Stop here. The blockers above are not "training failures" in the
sense of "metric column missing, retry on a different data slice"
— they are pipeline-shape mismatches that require deliberate user
input on the v0.3 design (single-knob external picker vs. richer
in-runtime replacement). Pushing forward without that decision
risks shipping a `.bin` whose output shape doesn't match anything
the codec runtimes can consume, and which won't beat the v0.1
baseline on bytes (because it predicts a binary knob, not the
3-scalar v0.1 head set).

Hand-back questions for the user:

1. Is v0.3 intended as "external picker over `__expert`" (in which
   case the schema bridge is a 1-day task and the grid is fine)?
   Or as a drop-in replacement for v0.2 (in which case the sweep
   needs re-running with a richer grid)?
2. Should we author a `mlp-tune-fast/manifest.tsv` and cut
   per-codec features against it before any further training? This
   work belongs in zentrain (Tier 2 centralized extractor), and
   the existing per-codec extractors all need to be re-pointed at
   the new corpus.
3. Is the held-out A/B target still `cid22-val/` (41 imgs), and is
   the comparator the v0.1 baked .bin, the bucket-table, or both?

— Lilith River, 2026-05-04
