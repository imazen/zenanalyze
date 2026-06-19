# Dense percentiles: the decision + how to scientifically choose among them

The `feat/dense-percentiles` branch adds **90 features (ids 122-211)** — 13-point
percentile sweeps of four block-statistic distributions (LaplacianVariance,
AqMap, NoiseFloor{Y,UV}, QuantSurvival{Y,UV}), all `experimental`-gated. The
question: which of these duplicate-looking signals do we keep, and on what
basis.

## The decision: search space, not a merge

**Do not merge 90 percentile features onto the default surface.** The dense grid
is a *training-time search space* for finding which percentiles carry signal —
not a runtime shipping set. Every feature on the default surface is a runtime
cost on every codec encode and another column in the train↔runtime contract that
has to stay pinned. Keep the dense grid `experimental`-gated, run the selection
below against a real codec target, and **promote only the surviving subset** to
the default surface. This is also the gate on 1.0: converge the feature set, then
freeze.

## The evidence refutes the naive prior

The intuitive assumption — "adjacent percentiles of one distribution are
collinear, so collapse each to a median" — is **wrong on this data**. Measured on
the imazen-26 features parquet (246k rows; Spearman ρ, eff-rank = PCA components
for X% variance; `benchmarks/percentile_redundancy_evidence_2026-06-19.py`):

```
per base distribution        #perc  mean|ρ|  eff-rank@95%  eff-rank@99%
aq_map                          8     0.41         4             7
laplacian_variance              8     0.37         5             7
noise_floor_uv                  4     0.65         3             4
noise_floor_y                   7     0.52         4             6
quant_survival_uv               4     0.44         4   (FULL)    4
quant_survival_y                6     0.54         5             6

GLOBAL: 37 existing percentile features -> eff-rank @90%=12  @95%=17  @99%=26
```

Read that carefully:
- **Within a distribution, percentiles are only moderately correlated** (mean
  |ρ| 0.37-0.65, not >0.9). `quant_survival_uv`'s 4 percentiles are *full rank* —
  every one independent. The median and the p90 of an image's per-block cost
  distribution measure genuinely different things (typical vs worst-case block),
  and the shape varies across images. You cannot collapse to one point per
  distribution without losing signal.
- **Globally the percentile space is ~17-dimensional** (at 95% variance, from 37
  features). It compresses ~2×, not ~10×.
- **The real redundancy is cross-distribution**: the *same* percentile across
  different bases is highly correlated — p50: `aq_map_p50`/`noise_floor_y_p50`/
  `quant_survival_y_p50` ρ∈[0.82, 0.99]; p75 ρ∈[0.96, 0.98]. These "median /
  upper-tail block cost" bands are the dup clusters (matching the existing
  `feature_redundancy_clusters` "median block cost" group), not the within-grid
  neighbors.

So the dense 13-point grids over-sample (the in-between points P15/P20/P55…
interpolate existing ones), but the percentile *idea* carries multi-dimensional
shape signal that a sparse-but-spread grid must preserve.

## The method (relevance ∧ redundancy — mRMR / cluster-LOGO)

Variance-redundancy (above) is **necessary but not sufficient**: it finds that
the space is ~17-dimensional, but a feature can be variance-independent yet
target-irrelevant (pure noise), or low-variance yet decisive for one knob. The
keep-set is decided by *relevance to the codec outcome*, with redundancy as the
tie-breaker. Per the repo's blessed ablation methodology
(`zenmetrics/docs/ML_FRAMEWORK_AND_PICKER_ABLATION_2026-06-09.md`):

1. **Extract** the full dense grid + base percentiles on a *codec sweep* corpus
   (per `(image, codec, q, knob)` with `zensim`/`bytes`/`butteraugli` targets and
   the encoded bytes persisted). This is the missing dataset — neither the
   imazen-26 features (no target) nor the existing sweeps (no dense columns) have
   both; the dense branch must be re-extracted over a sweep corpus.
2. **Redundancy-cluster** |Spearman ρ| ≥ 0.95 (hierarchical). Confirmed runnable;
   collapses the cross-distribution bands.
3. **Relevance via GBDT cluster-LOGO.** Train a GBDT (forust;
   `zenpicker-train/src/bin/picker_tree_ab.rs`) on the target and measure
   **cluster-level** (leave-one-group-out) permutation importance — NOT
   per-feature, because ρ≥0.95 twins mask each other when permuted singly.
   Conditional on the **surviving knob and the zq band** (relevance is not
   global). Do output ablation first (pin dead knobs) — it's the bigger lever.
4. **Select (mRMR):** keep clusters whose LOGO importance clears the RD-tolerance
   bar; the representative per kept cluster is its highest-individual-relevance
   member. This naturally keeps a *spread* of percentile bands (low-tail /
   median / upper-tail / peak) rather than one point.
5. **Validate:** retrain on the pruned set; held-out RD (encode→metric bytes at
   matched quality) within tolerance of the full set. Judge by RD, not training
   loss.
6. **Promote** the survivors to the default surface; drop or keep-experimental
   the rest. Bump `feature_defs_version()` if any promoted feature's definition
   changed.

**Expected outcome** (hypothesis, to be confirmed by step 3-5): from ~127
percentile features (90 dense + ~37 base) down to **~10-20** that carry
independent predictive signal — a sparse spread of bands × the most-relevant base
distribution per band, plus the few full-rank ones (`quant_survival_uv`).

## What this needs next
- A dense-grid + target sweep (re-extract on `feat/dense-percentiles` over a
  codec-corpus sweep; persist encoded bytes + all metric variants per the
  persistence discipline).
- Run `picker_tree_ab.rs` cluster-LOGO per knob/zq-band; produce the promote-list.
- Then the branch resolves: promote the survivors, retire the rest — and the
  feature set is one step closer to a 1.0 freeze.
