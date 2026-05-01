# Feature ablation methodology — pit-of-success guide

Every codec picker bake will eventually face the question: which of the 100+ features in `FeatureSet::SUPPORTED` are pulling their weight, and which are dead code on the analyzer hot path? This doc is the recipe — what to run, in which order, and how to read the output.

If you skip ahead to "run LOO and trust the result," **you will get the wrong answer**. Two things have already burned us:

1. **Constants masquerade as redundancies.** A feature with one value across the corpus reports Δ = 0 from every ablation tool — not because it's unimportant, but because it can't be tested. Confused #60 (block_misalignment) into a deprecation candidate when the corpus had no non-aligned images.
2. **Mathematical redundancies aren't always cross-codec.** `feat_min_dim` and `feat_max_dim` are zero-Δ on zenjpeg (the model substitutes `feat_pixel_count` + size one-hot) but +0.10–0.14pp on zenwebp (the m4/m5/m6 method choice depends on aspect shape directly). Single-codec ablation overgeneralizes.

The pipeline below is designed to fail loudly on both.

---

## Tier 0 — pre-flight: corpus coverage check (`correlation_cleanup.py`)

**What:** Pairwise Spearman correlation across all features in the per-image features TSV. Three buckets:

- **Constants** (`n_unique = 1`) — features with zero variance across the corpus. **Cannot draw any conclusion about these.** Fix the corpus before measuring.
- **Low-variance** (`n_unique < 5`) — Spearman is rank-based; tied ranks dominate. Redundancy claims here are weak.
- **True redundancy** (|Spearman| ≥ 0.99) — features that are mathematical or near-mathematical transforms of each other. Tree learners route through one and ignore the rest.

**Run:**

```bash
PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
    python3 <zenanalyze>/zentrain/tools/correlation_cleanup.py \
        --codec-config <codec>_picker_config \
        --threshold 0.99
```

**Interpret:**

- If the **constants** report has more than ~5 features, **stop and fix the corpus** before any further ablation. Add aspect / non-power-of-2 sizes / HDR / alpha samples until the constants list is small. Issue #61 documents the failure mode.
- The **redundancy** report's drop list is the *upper bound* on what's safe to deprecate — but not a deprecation list yet (Tier 3 confirms cross-codec).

**Cost:** seconds, no model training.

**Pit-of-success default:** threshold 0.99. Below 0.99 you start dropping correlated-but-distinct features (`cb_sharpness` ≈ 0.92 with `cr_sharpness` — related, not redundant). 0.99+ is "the same signal in different units."

---

## Tier 1 — fast screening: permutation importance (`feature_ablation.py --method permutation`)

**What:** Train HistGB once with all features. For each feature column, shuffle the values across rows; measure how much the model's score drops. Repeat with the original column restored, move to the next.

**Run:**

```bash
PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
    python3 <zenanalyze>/zentrain/tools/feature_ablation.py \
        --codec-config <codec>_picker_config \
        --method permutation
```

**Interpret:**

- Δ < 1e-4 → feature didn't earn its keep on this corpus / model. Candidate for further investigation.
- Δ ≥ 1e-3 → load-bearing. Don't touch.
- Δ between → ambiguous; defer to Tier 3 LOO.

**Cost:** ~1× train time + N forward passes. ~30s–5min at our corpus sizes.

**Caveat:** Shares LOO's multicollinearity blind spot. If A and B are correlated, both shuffle to ~zero importance because the model just routes through whichever survived. Catches constants and true redundancies; bad for "which of these correlated peers carries unique signal."

**Hybrid-heads pickers (per-cell scalars):** `feature_ablation.py` trains one HistGB per *config*, which explodes when the picker has scalar heads — train_hybrid's v0.2 schema has 110 K configs at 7 scalars × 10 cells. Use `student_permutation.py` instead (Tier 1.5 below) for these models — it operates on the trained student MLP directly, ~30 s wall regardless of config count.

---

## Tier 1.5 — hybrid-heads fast path (`student_permutation.py`)

**What:** Permutation importance against a *trained student MLP* (the JSON output of `train_hybrid.py`), no teacher retraining. For each feature column, shuffle values on the val set, run the student forward pass, measure overhead delta vs the unpermuted baseline.

**When:** Use this whenever the codec uses train_hybrid's hybrid-heads schema (per-cell bytes + scalar regression heads). The same `feature_ablation.py --method permutation` works for older single-head HistGB pickers; both compute the same kind of importance.

**Run:**

```bash
PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
    python3 <zenanalyze>/zentrain/tools/student_permutation.py \
        --codec-config <codec>_picker_config \
        --model-json benchmarks/<codec>_picker_v*.json \
        --output benchmarks/<codec>_student_perm.json \
        --n-repeats 5
```

**Interpret:** Same thresholds as Tier 1 — Δ_pp < 0.05 is cull-candidate territory, Δ_pp < 0 is "model overfit on this feature." Shares the same multicollinearity caveat. Same downstream gates: confirm with Tier 3 LOO and Tier 4 cross-codec before deprecating.

**Cost:** O(n_features × n_val_rows × forward_pass_cost). ~30 s for 43 features × 1.7 K val rows × a 96→192³→80 MLP. Independent of n_configs.

**Caveat (corpus-sensitivity, same as Tier 1):** Shuffling a feature whose validation rows are constant across the active corpus reports Δ ≈ 0 *whether or not the feature carries signal in general*. Always run Tier 0 first.

---

## Tier 2 — group ablation (`feature_group_ablation.py`)

**What:** Drop entire semantic clusters together — all `laplacian_variance_p*` percentiles, all chroma sharpness variants, all noise floor families. Catches the case where any single feature in a percentile family is replaceable but the family as a whole carries information.

**Run:**

```bash
PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
    python3 <zenanalyze>/zentrain/tools/feature_group_ablation.py \
        --codec-config <codec>_picker_config
```

**Interpret:** A group with Δ ≪ sum-of-individual-Δ is **synergistic** — features within carry overlapping signal but the group as a whole matters. A group with Δ ≈ sum-of-individual-Δ is **independent** — each feature contributes separately.

**Cost:** group-count × train time.

---

## Tier 3 — LOO retrain (gold standard, expensive)

**What:** For each feature, retrain HistGB on the remaining N-1 features. Measure overhead delta against baseline. This is the script's `--method loo` mode — what zenjpeg's v2.2 ablation runs.

**Run:**

```bash
PYTHONPATH=<zenanalyze>/zentrain/examples:<zenanalyze>/zentrain/tools \
    python3 <zenanalyze>/zentrain/tools/feature_ablation.py \
        --codec-config <codec>_picker_config \
        --method loo
```

**Interpret:** This is the deprecation gate. A feature should drop only if Tier 0 (correlation) + Tier 1 (permutation) + Tier 3 (LOO) all agree it's zero-impact, **and** the agreement holds across at least 2 codecs (per Tier 4 below).

**Cost:** N retrains. ~10× permutation. Use only on the post-Tier-1 candidate set, not as the discovery tool.

---

## Tier 4 — cross-codec confirmation

**What:** Repeat Tiers 0–3 against at least one other codec's picker oracle TSV. A feature deprecation that holds for zenjpeg might fail for zenwebp (`feat_min_dim` is the example: zero-Δ on zenjpeg, +0.14pp on zenwebp).

**Recipe:**

1. Each codec runs the full pipeline (this doc) against its own oracle.
2. Cross-tabulate per-feature Δ values across codecs (see `benchmarks/zenjpeg_ablation_2026-05-01.{log,json}` and `benchmarks/zenwebp_ablation_2026-05-01.{log,json}` for examples).
3. **Deprecate only features that are zero across all consumers.** Single-codec drops are fragile.

For features that disagree across codecs, document the asymmetry — they're real signals just not universal.

---

## What "deprecate today" means

A feature is safe to mark `#[deprecated]` and slate for removal in the next major release iff:

- ✅ Tier 0: in a high-Spearman cluster with a kept canonical anchor (mathematically redundant), AND
- ✅ Tier 1: permutation Δ < 1e-4 on every codec measured, AND
- ✅ Tier 3: LOO Δ < 0.05pp on every codec measured, AND
- ✅ Tier 4: at least 2 codecs measured.

If the failure mode is "constant on this corpus" (Tier 0 constants bucket), **do not deprecate** — fix the corpus and re-measure first. The feature might carry signal we can't see.

---

## Worked example — issue #59 / #60 / #61 timeline

**Setup:** the jxl-encoder team ran a single-codec LOO ablation, found 16 mathematical transforms of `feat_pixel_count` at Δ ≤ 1e-5, and proposed deprecating them as #59. Also flagged 4 `block_misalignment_*` variants at Δ ≤ 2e-5 as #60.

**Tier 0 cross-check (zenjpeg corpus):**
- 17 features in #59 split: 16 form a true mathematical redundancy cluster anchored by `feat_pixel_count`. The 17th (`feat_aspect_min_over_max`) is a constant on the zenjpeg corpus (all-square images).
- All 4 of #60's `block_misalignment_*` are constants on the zenjpeg corpus (all-power-of-2 sizes).
- 16 other features are ALSO constants — 19 corpus-coverage gaps in total. Filed as #61.

**Tier 4 cross-check (zenwebp corpus):**
- 13 of the 17 #59 features confirm as redundant (the pure mathematical transforms: log/sqrt/padded/bitmap_bytes).
- `feat_min_dim` (+0.14pp), `feat_max_dim` (+0.10pp), `feat_aspect_min_over_max` (+0.10pp) **are NOT redundant on zenwebp** — they carry signal even with `feat_pixel_count` available, because zenwebp's m4/m5/m6 method choice depends on shape. KEEP.
- `feat_log_aspect_abs` is borderline (–0.02 LOO, +0.06 permutation). KEEP conservatively.
- All 4 `block_misalignment_*` confirm zero Δ on zenwebp **but the corpus has the same coverage gap** — the result is consistent with "zero on this corpus" rather than "zero everywhere."

**Decision:**

- **#59 — partial deprecation.** 13 of 17 candidates flagged `#[deprecated]`. The 4 dim/aspect features stay. Stable feature ids 57, 60, 62, 94–104 are reserved (never recycled).
- **#60 — defer.** Confirmed agreement across codecs but ALL agreement was on corpora that don't exercise non-power-of-2 sizes. Re-measure after #61 lands.
- **#61 — corpus rerun blocking.** Add aspect / non-power-of-2 / HDR / alpha samples to zenjpeg's sweep harness defaults; re-run the full pipeline; revisit #60.

The same shape applies to every future deprecation proposal: cross-codec, post-corpus-fix, all four tiers in agreement, or it's not safe to deprecate.

---

## Tooling reference

| Tool | Method | Cost | Output |
|---|---|---|---|
| `correlation_cleanup.py` | Spearman pairwise + cluster | seconds | constants / low-variance / redundancy buckets |
| `feature_ablation.py --method permutation` | shuffle one column at a time (per-config HistGB teachers) | ~min, but breaks on hybrid-heads at high n_configs | Δ per feature, ranked |
| `student_permutation.py` | shuffle one column at a time (trained student MLP forward pass) | ~30 s, hybrid-heads native | Δ per feature, ranked |
| `feature_ablation.py --method loo` | retrain N times, drop one feature each | ~10× perm | Δ per feature, gold-standard |
| `feature_group_ablation.py` | drop semantic groups | group-count × train | Δ per group |

All four read the per-image features TSV declared by the codec's `*_picker_config.py` (`FEATURES = Path(...)`). Output paths follow the codec config's `OUT_*` conventions; ablation artifacts ship next to the trained model.

---

## Cross-references

- [`PRINCIPLES.md`](PRINCIPLES.md) — cross-codec data discipline (sweep four dimensions, per-image zensim ceiling, OOD bounds).
- [`FOR_NEW_CODECS.md`](FOR_NEW_CODECS.md) — onboarding flow; the ablation step plugs in after the picker has trained at least once.
- [`/benchmarks/zenjpeg_ablation_2026-05-01.log`](../benchmarks/zenjpeg_ablation_2026-05-01.log) — zenjpeg LOO baseline, 100-image SDR subset (corpus has known gaps; see #61).
- [`/benchmarks/zenwebp_ablation_2026-05-01.log`](../benchmarks/zenwebp_ablation_2026-05-01.log) — zenwebp LOO + permutation, full corpus.
- Issues [#59](https://github.com/imazen/zenanalyze/issues/59), [#60](https://github.com/imazen/zenanalyze/issues/60), [#61](https://github.com/imazen/zenanalyze/issues/61).
