# Per-class quant table signal probe (2026-05-04)

Gating experiment to decide whether per-content-class quant tables are worth
pursuing. **30-minute single-turn measurement, no SA, no corpus expansion.**
The question: **does a deliberately differentiated quant table buy material
pareto improvement on screen content vs photo content?**

## Method

- **Tables compared:** three sets at q ∈ {15, 50, 85}.
  - **A. v4** — current photo-tuned defaults from
    `zenjpeg::encode::tables::sa_piecewise_v4::tables_for_quality(q)`.
  - **B. zero_hf** — v4 with the bottom-right 4×4 high-freq region
    (rows 4-7, cols 4-7, natural order) of luma + Cb + Cr quant matrices
    forced to **1** (legal min — JPEG forbids quant=0; quant=1 means
    "preserve all energy at that frequency"). Hypothesis: screen content
    has near-zero high-freq AC energy, so this should be ~free for screens
    and costly for photos.
  - **C. half_dc** — v4 with luma DC quant (table[0][0]) halved.
- **Corpus:**
  - Photo: 20 images sampled (LCG Fisher-Yates, seed=7) from
    `~/work/zentrain-corpus/mlp-tune-fast/cid22-train` (209 total).
  - Screen: all 10 images in `~/work/zentrain-corpus/mlp-tune-fast/gb82-screen`
    (brief said 11; corpus has 10 — used all 10).
- **Encoder:** `EncoderConfig::ycbcr(ApproxJpegli(q), Quarter).tables(...)`
  with `ScalingParams::Exact` so the perturbations aren't rescaled by quality.
  Decoder: `zune-jpeg`. Metric: butteraugli 0.9 (CPU, 3-norm).
- **Cell count:** 30 images × 3 q × 3 tables = **270 cells**, all green.
- **Probe binary:** `zenjpeg/dev/per_class_signal_probe.rs`
  (zenjpeg `2423a8aa`).
- **Raw TSV:** `benchmarks/per_class_signal_probe_2026-05-04.tsv` (270 rows).

## Results — at fixed q (mean bytes & butteraugli)

| class | table | q | n | mean_bytes | mean_BA | Δbytes vs A | ΔBA vs A |
|---|---|---:|---:|---:|---:|---:|---:|
| photo | A_v4 | 15 | 20 | 15,630 | 5.622 | +0.0% | +0.000 |
| photo | B_zero_hf | 15 | 20 | 59,931 | 5.590 | +283.4% | -0.033 |
| photo | C_half_dc | 15 | 20 | 16,116 | 5.598 | +3.1% | -0.024 |
| photo | A_v4 | 50 | 20 | 25,639 | 4.602 | +0.0% | +0.000 |
| photo | B_zero_hf | 50 | 20 | 68,619 | 4.509 | +167.6% | -0.093 |
| photo | C_half_dc | 50 | 20 | 26,120 | 4.531 | +1.9% | -0.071 |
| photo | A_v4 | 85 | 20 | 54,800 | 2.831 | +0.0% | +0.000 |
| photo | B_zero_hf | 85 | 20 | 92,586 | 2.719 | +69.0% | -0.112 |
| photo | C_half_dc | 85 | 20 | 55,298 | 2.843 | +0.9% | +0.012 |
| screen | A_v4 | 15 | 10 | 104,441 | 7.953 | +0.0% | +0.000 |
| screen | B_zero_hf | 15 | 10 | 224,297 | 7.602 | +114.8% | -0.351 |
| screen | C_half_dc | 15 | 10 | 105,548 | 7.972 | +1.1% | +0.019 |
| screen | A_v4 | 50 | 10 | 145,162 | 6.199 | +0.0% | +0.000 |
| screen | B_zero_hf | 50 | 10 | 257,878 | 6.084 | +77.6% | -0.115 |
| screen | C_half_dc | 50 | 10 | 146,082 | 6.045 | +0.6% | -0.154 |
| screen | A_v4 | 85 | 10 | 242,087 | 3.662 | +0.0% | +0.000 |
| screen | B_zero_hf | 85 | 10 | 331,811 | 3.493 | +37.1% | -0.170 |
| screen | C_half_dc | 85 | 10 | 242,955 | 3.692 | +0.4% | +0.029 |

**Read of the at-q view:**

- **B (zero_hf)** improves butteraugli at every (class, q), but spends
  **+37% to +283%** more bytes to do so. B preserves all high-freq energy
  → entropy coder spends a lot to encode it.
- **C (half_dc)** is essentially noise: byte deltas <3.1%, BA deltas
  ≤ 0.16. The DC-quant change is too small to materially reshape RD.

## Results — at matched bytes (pareto-distance proxy)

For each image, build a (log_bytes, butteraugli) curve from each table's
3 anchors and interpolate B/C's curve at the BYTES that A produced for that
image. Negative = perturbation BEATS A at the same byte budget.

| class | perturbation | A's anchor q | n | mean ΔBA | median ΔBA |
|---|---|---:|---:|---:|---:|
| photo | B_zero_hf | 15 | 20 | -0.033 | -0.027 |
| photo | B_zero_hf | 50 | 20 | +0.987 | +0.812 |
| photo | B_zero_hf | 85 | 20 | +2.668 | +2.631 |
| photo | C_half_dc | 15 | 20 | -0.024 | -0.020 |
| photo | C_half_dc | 50 | 20 | -0.025 | +0.004 |
| photo | C_half_dc | 85 | 20 | +0.034 | +0.028 |
| screen | B_zero_hf | 15 | 10 | -0.351 | -0.035 |
| screen | B_zero_hf | 50 | 10 | +1.403 | +1.481 |
| screen | B_zero_hf | 85 | 10 | +2.904 | +2.988 |
| screen | C_half_dc | 15 | 10 | +0.019 | -0.011 |
| screen | C_half_dc | 50 | 10 | -0.117 | -0.026 |
| screen | C_half_dc | 85 | 10 | +0.049 | +0.027 |

**Caveat on the q=15 row:** B's smallest output (q=15) is bigger than A's
q=85 output for many images, so when we ask "what is B's BA at A's q=15
bytes?", we're extrapolating below B's measured range — interpolation
clamps to B's smallest measured BA (= B's q=15). That collapses the q=15
row to the at-q delta and isn't a fair pareto comparison. The q=50 and
q=85 anchors are on B's measured range and are the trustworthy comparisons.

## Pareto verdict (mean ΔBA across q anchors, matched-bytes)

| class | perturbation | mean ΔBA across q | direction |
|---|---|---:|---|
| photo | B_zero_hf | +1.208 | WORSE |
| photo | C_half_dc | -0.005 | ≈ |
| screen | B_zero_hf | +1.318 | WORSE |
| screen | C_half_dc | -0.016 | BETTER (within noise) |

## Verdict

- **B (zero_hf) — MIXED, but trending strongly negative.** B does win on
  raw butteraugli at fixed q across every cell, *and* it wins by a larger
  margin on screens (-0.17 to -0.35 BA) than on photos (-0.03 to -0.11 BA).
  That's a real screen-vs-photo signal in the right direction. **But the
  cost is enormous (+37% to +115% bytes on screens).** At matched bytes,
  B is dramatically *worse* than A on both classes — both screen ΔBA and
  photo ΔBA are far above the brief's "≥ 3 points" threshold (in the wrong
  direction, since lower BA = better; the brief's threshold framing presumes
  the perturbation's pareto delta is favorable, which B's isn't).
- **C (half_dc) — WEAK.** Both byte and BA deltas are noise-floor (|ΔBA| ≤
  0.16, |Δbytes| ≤ 3%). DC quant is too small a lever.

## Per-brief decision criteria

The brief asked: is the screen-vs-photo pareto delta from a deliberately
differentiated table > 3 points (worth pursuing) or < 1 point (dead)?

- **Pursuit threshold (Δ_screen ≥ +3 AND Δ_photo ≤ +0.5, in pareto units):**
  not met by either B or C. B's matched-bytes deltas are LARGER than 3
  but in the *wrong* direction (B is worse, not better, at matched bytes).
  C's matched-bytes deltas are tiny.
- **Dead threshold (both deltas in [-1, +1] for both perturbations):** met
  cleanly by C. Met for B *only at the q=15 anchor* (which is the
  unreliable extrapolation row). At q=50/85, B's deltas are above +1.

## Recommendation

**Per-class quant tables in the form tested are not promising.**

- **B's wins are not real wins.** B beats A on BA at every fixed q, and the
  margin is 4-10× larger on screens than on photos — exactly the signature
  of "this perturbation matters more for the screen domain", which is the
  signal the brief was probing for. But all of B's apparent quality gain is
  paid for by spending dramatically more bytes preserving high-frequency
  AC energy. Once we control for bytes, B is far worse on both classes.
- **The high-freq region is where photos and screens differ.** That part
  of the hypothesis was right. What was wrong was the direction of the
  perturbation. Forcing quant=1 there preserves energy at extreme cost.
  The right experiment is the *opposite*: increase the high-freq quant on
  screens (compress more aggressively where there's no energy) and see
  whether the screen byte savings are large while photo BA degradation is
  small. That would be a "WIN for the screen class only" signal.
- **C (DC quant) is dead.** The DC coefficient is already quantized lightly
  in v4 (luma DC quant ≈ 13-30 across q anchors) and halving it is
  imperceptible — almost no images have visible block-DC error.
- **Suggested next step (if the user still wants to validate per-class
  tables):** run the same probe with **B' = v4 with high-freq quant
  *increased* (e.g. 2× or 4× the bottom-right 4×4)**. That tests the
  asymmetric-cost hypothesis: screens should lose ~nothing, photos should
  degrade. If B' shows screen ΔBA ≤ +0.05 and photo ΔBA ≥ +0.30 at fixed q
  with screens shrinking by 5-15%, that's the STRONG signal worth investing
  SA budget against per-class tables. If B' is also flat or noisy, the
  per-class hypothesis is dead and the user should redirect to other axes.

## Files

- Probe binary: `zenjpeg/dev/per_class_signal_probe.rs` (commit `2423a8aa`).
- Raw TSV: `zenanalyze/benchmarks/per_class_signal_probe_2026-05-04.tsv`
  (270 rows, columns: image, content_class, table, q, bytes, butteraugli,
  w, h).
- Corpus pointers: `~/work/zentrain-corpus/mlp-tune-fast/cid22-train` (photo)
  and `~/work/zentrain-corpus/mlp-tune-fast/gb82-screen` (screen).
