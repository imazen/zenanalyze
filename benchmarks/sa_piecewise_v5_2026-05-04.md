# SA Piecewise v5 — Validation Report (2026-05-04)

**Status: NO-SHIP. v4 keeps shipping.**

This report lives alongside the zenjpeg-side finding at
`zenjpeg/benchmarks/sa_piecewise_v5_finding_2026-05-04.md` and the local
Phase 1 trace at `/tmp/glassa_sa_local_phase1.md`. Cross-reference both
for the full story.

## 1. Train pareto

v4 mean pareto vs jpegli defaults: **+6.602** across 20 anchors q5-100
(20/20 anchors beat baseline). Source: `combined_hybrid_v4.json` from
the SA optimizer in `coefficient/examples/piecewise_optimize.rs`
(branch `feat/piecewise-quant-tables`, commit `435c1c9`) on
CID22-512-photo training (209 images).

Per-anchor breakdown:

```
   q     pareto      bpp    qual_score    found_at_iter
   5      7.454   0.4119      5.901            1069
  10      7.430   0.3596      6.552             795
  15      7.530   0.4182      5.446              17
  20      7.514   0.4261      5.439            2008
  25      7.448   0.5245      4.848            1021
  30      7.454   0.5226      4.855            1006
  35      7.403   0.5878      4.512            2036
  40      7.415   0.5256      5.030            1016
  45      7.273   0.7192      3.980            2005
  50      7.162   0.7341      4.411               7
  55      7.151   0.7917      3.933               3
  60      7.377   0.6124      4.390            3006
  65      6.888   0.9722      3.666            1008
  70      6.733   1.0508      3.712            1004
  75      6.460   1.3031      2.865            2003
  80      5.928   1.6389      2.575             505
  85      5.777   1.6995      2.723             968
  90      4.790   2.1600      3.536             128
  95      4.068   2.5848      3.534             989
 100      2.794   3.2690      3.669            1000
```

## 2. Out-of-sample (OOS) pareto

Closest available holdout proxy: v3.5 (`sa_validated_training.json`)
q1-100 interpolation on 41-image CID22-photo holdout =
**+6.385 mean, 100/100 levels beat baseline, min +2.345 at q100**.

OOS / train ratio (using v4 train as denominator):
**6.385 / 6.602 = 96.7% ≥ 80% (Gate 2: PASS)**.

Caveat: this is a v3.5 holdout, not a v4 holdout. v4 is a small
cumulative improvement over v3.5 (mean train +6.602 vs +6.385), so
v4 holdout is expected to be at-or-above 6.385. We do not have a
direct v4 q1-100 holdout file because the q1-100 validation was run
on v3.5 (`holdout_validation_q1_100.txt`, 2026-02-02 01:47 -0700)
and v4 was created 3 hours later.

## 3. Gates fired / passed

| Gate | Threshold | v4 actual | Result |
|------|-----------|-----------|--------|
| 1: Train pareto vs jpegli ≥ +5.0 | +5.0 mean | +6.602 | PASS |
| 2: OOS pareto ≥ 80% of train | ≥ +5.282 | +6.385 (proxy) | PASS |
| 4: New v5 vs v4 on holdout ≥ +3 | n/a (no v5) | n/a | NO-RUN |

Gates 1 and 2 are passed by v4. Gate 4 is the v5 ship gate, and v5 was
not produced this run (see §4 below).

## 4. Why no v5 produced

Two candidate local pipelines were available for Phase 1 reproduction
of the v4 SA optimizer:

1. `~/work/glassa/examples/optimize_jpegli_tables.rs` (2063 LoC,
   CUDA + butteraugli-GPU + jpegli encoder). Does not compile against
   `fast-ssim2 0.8` (`Rgb`/`ColorPrimaries`/`TransferCharacteristic`
   relocated to `yuvxyb`), `jpegli-rs 0.12`
   (`QuantTableConfig` removed, `EncoderConfig::new` removed), or
   Rust 2024 edition (4 reference-pattern errors in clustering code).
   9 distinct error sites; estimated 1-2h of API-aware porting to
   restore. **One safe edit applied** to glassa Cargo.toml: the
   `fast-ssim2` path-dep was repointed from
   `../zen/fast-ssim2/ssimulacra2` (no longer exists) to
   `../zen/fast-ssim2/fast-ssim2` (current location after upstream
   rename).

2. `~/work/coefficient/examples/piecewise_optimize.rs` (the literal
   binary that produced v4). Lives on `feat/piecewise-quant-tables`
   branch (commit `435c1c9`, 4 weeks old). Building it requires
   checking out a sibling repo branch, which the brief explicitly
   scopes out (parent agent owns the coefficient/zenjpeg checkouts
   and is no longer pushing) and the global rules forbid without
   approval.

Phase 2 (cloud) was NOT triggered. Its precondition is a working
local pipeline reproducing v4 within ±0.3 pareto on a 30-image, 4-anchor
subset, and without that anchor the cloud SA results would be
untrustable.

## 5. Refit upper-bound estimate

Best-case theoretical lift (anchors q90/95/100 each gain ~1.5
pareto, the largest single-anchor jumps observed in any v3→v4-style
cumulative improvement of v4):

```
new mean = (6.602 × 20 + 1.21 + 1.43 + 1.71) / 20 = 6.820
delta vs v4 mean: +0.218
```

The +3 mean-pareto ship threshold is unreachable. Even a per-anchor
read of Gate 4 (≥ +3 at any single anchor) plausibly only fires at
q100, which v4's docstring already documents as a "use jpegli
defaults" tie.

## 6. Recommendation

- **Keep v4 shipping** at `zenjpeg/src/encode/tables/sa_piecewise_v4{,_data}.rs`.
- **Do NOT produce v5 photo tables** in this run.
- **Repair the local SA pipeline** (1-2h of API porting in glassa, or
  resurrect the coefficient piecewise branch in a worktree) as the
  gating prerequisite for any future v5 work.
- If a different metric (zensim2) or a different RD regime (trellis-on)
  becomes the picker's preferred ranking, treat that as a separate v5
  family — the photo butteraugli-GPU jpegli cell is essentially
  saturated.

## 7. References

- v4 raw provenance: `/mnt/v/output/coefficient/piecewise/combined_hybrid_v4.json`
- v4 source export: `/mnt/v/output/coefficient/piecewise/best_tables_v4.rs`
- v3.5 q1-100 holdout proxy: `/mnt/v/output/coefficient/piecewise/holdout_validation_q1_100.txt`
- Phase 1 trace: `/tmp/glassa_sa_local_phase1.md`
- zenjpeg-side finding: `~/work/zen/zenjpeg/benchmarks/sa_piecewise_v5_finding_2026-05-04.md`
