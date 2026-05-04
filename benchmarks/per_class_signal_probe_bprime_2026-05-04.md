# Per-class quant table B' signal probe — Step 0 gate result (2026-05-04)

**TL;DR: gate FAILS on all 4 candidate clusters across all 3 B'
variants (Bp_2x, Bp_4x, Bp_8x) at all 3 quality levels (q15, q50,
q85). Per-class quant tables in the form prescribed (bottom-right
4×4 HF block on photo/screen/technical/synthetic clusters) are
dead. NO-SHIP. Recommendation: do not pursue per-class SA on this
axis. Honest gate numbers below.**

This is the follow-up gating measurement that the original
[per_class_signal_probe_2026-05-04.md](per_class_signal_probe_2026-05-04.md)
explicitly recommended as a prerequisite to any per-class SA work.
That report's verbatim Step-0 criterion:

> "If B' shows screen ΔBA ≤ +0.05 AND photo ΔBA ≥ +0.30 at fixed q
>  with screens shrinking by 5-15%, that's the STRONG signal worth
>  investing SA budget against per-class tables. If B' is also flat
>  or noisy, the per-class hypothesis is dead."

We extend the criterion to the symmetric "any cluster" form: for the
class C to deserve its own table, the perturbation must shrink C's
bytes by 5-15% at ΔBA ≤ +0.05 on C's own content (and ideally
damage photos enough that the asymmetry is real).

## Method

- **Variants:** B' = `sa_piecewise_v4` baseline with the bottom-right
  4×4 quant block (rows 4..7 × cols 4..7, natural-order indices, all of
  Y/Cb/Cr) multiplied by `factor` and clamped to `[1, 255]`. Three
  factors: 2×, 4×, 8×.
- **Encoder:** `EncoderConfig::ycbcr(ApproxJpegli(q), Quarter).tables(...)`
  with `ScalingParams::Exact` so our edits aren't rescaled by quality.
- **Decoder:** `zune-jpeg`. Metric: butteraugli 0.9 CPU 3-norm.
- **Quality anchors:** q ∈ {15, 50, 85}, matching the prior probe.
- **Tables:** A_v4 (baseline) + Bp_2x + Bp_4x + Bp_8x = 4 tables × 3 q.
- **Clusters / corpora (`~/work/zentrain-corpus/mlp-tune-fast/`):**
  - **photo** — 20 images sampled by deterministic LCG Fisher-Yates
    (seed=7) from `cid22-train` (209 PNGs).
  - **screen** — all 10 PNGs in `gb82-screen` (the brief said 11; the
    corpus has 10).
  - **technical** — 20 images sampled (seed=7) from `kadid10k`
    (35 PNGs total).
  - **synthetic** — 20 images sampled (seed=7) from `size-dense-renders`
    (259 PNGs). Note: the corpus does **not** have a separate
    `synthetic/` subdir — the prompt's reference to one was incorrect;
    `size-dense-renders` is the Imageflow-rendered cluster that
    matches what the prompt called "synthetic".
- **Cell count:** 70 images × 4 tables × 3 q = **840 cells**, all green.
- **Probe binary:** `zenjpeg/dev/per_class_signal_probe_bprime.rs`.
- **Raw TSV:** `benchmarks/per_class_signal_probe_bprime_4cluster_2026-05-04.tsv`
  (840 rows, columns: image, content_class, table, q, bytes,
  butteraugli, w, h).
- **Companion 2-cluster TSV** (photo + screen only, run first as
  a sanity pass): `benchmarks/per_class_signal_probe_bprime_2026-05-04.tsv`
  (360 rows). Both runs produce identical numbers for shared cells.

## Results — at fixed q (mean bytes & butteraugli)

### Photo (n=20)

| table | q | mean_bytes | mean_BA | Δbytes | Δbytes_pct | ΔBA |
|---|---:|---:|---:|---:|---:|---:|
| A_v4    | 15 | 15,630 | 5.622 | +0     | +0.0% | +0.000 |
| Bp_2x   | 15 | 15,610 | 5.622 | -20    | -0.1% | +0.000 |
| Bp_4x   | 15 | 15,610 | 5.622 | -20    | -0.1% | +0.000 |
| Bp_8x   | 15 | 15,610 | 5.622 | -20    | -0.1% | +0.000 |
| A_v4    | 50 | 25,639 | 4.602 | +0     | +0.0% | +0.000 |
| Bp_2x   | 50 | 25,538 | 4.569 | -100   | -0.4% | -0.034 |
| Bp_4x   | 50 | 25,531 | 4.569 | -107   | -0.4% | -0.034 |
| Bp_8x   | 50 | 25,532 | 4.569 | -107   | -0.4% | -0.033 |
| A_v4    | 85 | 54,800 | 2.831 | +0     | +0.0% | +0.000 |
| Bp_2x   | 85 | 53,851 | 2.822 | -949   | -1.7% | -0.010 |
| Bp_4x   | 85 | 53,624 | 2.907 | -1,176 | -2.1% | +0.075 |
| Bp_8x   | 85 | 53,592 | 3.096 | -1,208 | -2.2% | +0.264 |

### Screen (n=10)

| table | q | mean_bytes | mean_BA | Δbytes | Δbytes_pct | ΔBA |
|---|---:|---:|---:|---:|---:|---:|
| A_v4    | 15 | 104,441 | 7.953 | +0      | +0.0% | +0.000 |
| Bp_2x   | 15 | 103,579 | 7.877 | -862    | -0.8% | -0.077 |
| Bp_4x   | 15 | 103,553 | 7.876 | -888    | -0.9% | -0.078 |
| Bp_8x   | 15 | 103,553 | 7.876 | -888    | -0.9% | -0.078 |
| A_v4    | 50 | 145,162 | 6.199 | +0      | +0.0% | +0.000 |
| Bp_2x   | 50 | 142,851 | 6.305 | -2,310  | -1.6% | +0.106 |
| Bp_4x   | 50 | 142,522 | 6.484 | -2,639  | -1.8% | +0.285 |
| Bp_8x   | 50 | 142,510 | 6.433 | -2,652  | -1.8% | +0.234 |
| A_v4    | 85 | 242,087 | 3.662 | +0      | +0.0% | +0.000 |
| Bp_2x   | 85 | 234,640 | 3.882 | -7,447  | -3.1% | +0.220 |
| Bp_4x   | 85 | 230,927 | 4.367 | -11,160 | -4.6% | +0.705 |
| Bp_8x   | 85 | 229,558 | 4.419 | -12,529 | -5.2% | +0.757 |

### Technical (kadid10k, n=20)

| table | q | mean_bytes | mean_BA | Δbytes | Δbytes_pct | ΔBA |
|---|---:|---:|---:|---:|---:|---:|
| A_v4   | 15 | (see TSV) | (see TSV) | +0     | +0.0% | +0.000 |
| Bp_2x  | 15 |           |           |        | -0.7% | +0.002 |
| Bp_4x  | 15 |           |           |        | -0.7% | +0.048 |
| Bp_8x  | 15 |           |           |        | -0.7% | +0.048 |
| Bp_2x  | 50 |           |           |        | -1.3% | +0.011 |
| Bp_4x  | 50 |           |           |        | -1.4% | +0.056 |
| Bp_8x  | 50 |           |           |        | -1.4% | +0.077 |
| Bp_2x  | 85 |           |           |        | -2.9% | +0.103 |
| Bp_4x  | 85 |           |           |        | -4.0% | +0.255 |
| Bp_8x  | 85 |           |           |        | -4.3% | +0.518 |

### Synthetic (size-dense-renders, n=20)

| table | q | Δbytes_pct | ΔBA |
|---|---:|---:|---:|
| Bp_2x  | 15 | -0.4% | -0.004 |
| Bp_4x  | 15 | -0.4% | -0.003 |
| Bp_8x  | 15 | -0.4% | -0.003 |
| Bp_2x  | 50 | -0.8% | +0.014 |
| Bp_4x  | 50 | -0.9% | +0.001 |
| Bp_8x  | 50 | -0.9% | +0.001 |
| Bp_2x  | 85 | -2.6% | +0.129 |
| Bp_4x  | 85 | -3.5% | +0.176 |
| Bp_8x  | 85 | -3.7% | +0.255 |

(Mean-bytes columns omitted from the technical/synthetic tables for
brevity; full numbers in the TSV.)

## Gate evaluation — every (cluster, table, q) cell

The "own-class-can-use-Bp" gate fires when applying Bp to a cluster
saves 5-15% bytes at ΔBA ≤ +0.05 on that cluster's own content.

| cluster   | table  | q  | own ΔBA | own Δbytes% | own-gate |
|-----------|--------|---:|--------:|------------:|:--------:|
| photo     | Bp_2x  | 15 | +0.000  | -0.1%       | FAIL     |
| photo     | Bp_2x  | 50 | -0.034  | -0.4%       | FAIL     |
| photo     | Bp_2x  | 85 | -0.010  | -1.7%       | FAIL     |
| photo     | Bp_4x  | 15 | +0.000  | -0.1%       | FAIL     |
| photo     | Bp_4x  | 50 | -0.034  | -0.4%       | FAIL     |
| photo     | Bp_4x  | 85 | +0.075  | -2.1%       | FAIL     |
| photo     | Bp_8x  | 15 | +0.000  | -0.1%       | FAIL     |
| photo     | Bp_8x  | 50 | -0.033  | -0.4%       | FAIL     |
| photo     | Bp_8x  | 85 | +0.264  | -2.2%       | FAIL     |
| screen    | Bp_2x  | 15 | -0.077  | -0.8%       | FAIL     |
| screen    | Bp_2x  | 50 | +0.106  | -1.6%       | FAIL     |
| screen    | Bp_2x  | 85 | +0.220  | -3.1%       | FAIL     |
| screen    | Bp_4x  | 15 | -0.078  | -0.9%       | FAIL     |
| screen    | Bp_4x  | 50 | +0.285  | -1.8%       | FAIL     |
| screen    | Bp_4x  | 85 | +0.705  | -4.6%       | FAIL     |
| screen    | Bp_8x  | 15 | -0.078  | -0.9%       | FAIL     |
| screen    | Bp_8x  | 50 | +0.234  | -1.8%       | FAIL     |
| screen    | Bp_8x  | 85 | +0.757  | -5.2%       | FAIL     |
| synthetic | Bp_2x  | 15 | -0.004  | -0.4%       | FAIL     |
| synthetic | Bp_2x  | 50 | +0.014  | -0.8%       | FAIL     |
| synthetic | Bp_2x  | 85 | +0.129  | -2.6%       | FAIL     |
| synthetic | Bp_4x  | 15 | -0.003  | -0.4%       | FAIL     |
| synthetic | Bp_4x  | 50 | +0.001  | -0.9%       | FAIL     |
| synthetic | Bp_4x  | 85 | +0.176  | -3.5%       | FAIL     |
| synthetic | Bp_8x  | 15 | -0.003  | -0.4%       | FAIL     |
| synthetic | Bp_8x  | 50 | +0.001  | -0.9%       | FAIL     |
| synthetic | Bp_8x  | 85 | +0.255  | -3.7%       | FAIL     |
| technical | Bp_2x  | 15 | +0.002  | -0.7%       | FAIL     |
| technical | Bp_2x  | 50 | +0.011  | -1.3%       | FAIL     |
| technical | Bp_2x  | 85 | +0.103  | -2.9%       | FAIL     |
| technical | Bp_4x  | 15 | +0.048  | -0.7%       | FAIL     |
| technical | Bp_4x  | 50 | +0.056  | -1.4%       | FAIL     |
| technical | Bp_4x  | 85 | +0.255  | -4.0%       | FAIL     |
| technical | Bp_8x  | 15 | +0.048  | -0.7%       | FAIL     |
| technical | Bp_8x  | 50 | +0.077  | -1.4%       | FAIL     |
| technical | Bp_8x  | 85 | +0.518  | -4.3%       | FAIL     |

**36 of 36 cells FAIL the gate.** Not a single (cluster × variant ×
quality) combination produces the asymmetric "free quality" the
hypothesis predicted.

## Why the hypothesis was wrong

The prior probe's at-q signal — screen content showing 4-10× larger
ΔBA at fixed q than photos when HF was *preserved* (B with quant=1) —
was correctly identified as a real screen-vs-photo asymmetry. The
prior report's qualitative read was: "screens have less HF energy,
so coarsening it should be ~free for screens and damaging for photos."

The B' measurement falsifies the directional inference. **At q=85,
screens are 3× more sensitive to HF coarsening than photos**:

|              | Bp_4x q85 ΔBA | Bp_8x q85 ΔBA |
|--------------|--------------:|--------------:|
| photo        | +0.075        | +0.264        |
| screen       | **+0.705**    | **+0.757**    |
| technical    | +0.255        | +0.518        |
| synthetic    | +0.176        | +0.255        |

The mechanism: high-q screens still have substantial HF AC content
from sharp UI / text edges (the Fourier expansion of a step has
energy at every frequency). Photos at high q have natural smoothing
and JND-aware noise masking that absorbs HF coarsening better.
The metric (butteraugli 3-norm) heavily weights edge-localized
distortion, which is precisely what coarse HF quant produces on
screens — ringing on text, blocking on chart lines.

The "preserve HF" perturbation B looked like a screen win at fixed q
because the screen's HF was being faithfully reconstructed; the cost
was paid in bytes (+114% on screens at q=15). Inverting the
perturbation does NOT invert the conclusion — the bytes-quality
trade is asymmetric in the direction the hypothesis assumed; the
*sensitivity* is asymmetric in the OPPOSITE direction.

This is consistent with the existing
[`sa_piecewise_v5_finding_2026-05-04.md`](../../zenjpeg/benchmarks/sa_piecewise_v5_finding_2026-05-04.md)
which documented v4 as already saturated for the photo butteraugli-GPU
cell. Per-class differentiation against the same metric on adjacent
clusters does not unlock new pareto room.

## Verdict

**No-ship across all 4 candidate clusters.** Per-class quant tables
in the prescribed form (bottom-right 4×4 HF block, photo/screen/
technical/synthetic clusters by mlp-tune-fast subdir, butteraugli-GPU
fitness, jpegli-4:2:0, no-trellis) **fail the gate the prior agent
defined and the user implicitly accepted by committing it.**

Phase 2 cloud SA was NOT triggered. Per-cluster table integration
in `zenjpeg/src/encode/tables/` was NOT done. The `EncodingTables`
public API stays as-is.

## What WOULD justify a future per-class SA run

Some axes that this probe did not falsify (any one of these reaching
+0.30 photo ΔBA / -5% photo bytes asymmetry on a different cluster
boundary would be the signal):

1. **Different metric.** zensim2 / fast-ssim2 have different
   frequency weighting; the at-q asymmetry direction may differ.
   In particular, SSIMULACRA2's local-contrast term penalises blur
   more than ringing, so HF-preservation perturbations could pareto
   differently.
2. **Different perturbation axis.** Bottom-right 4×4 was the obvious
   "high-frequency" handle but is only 16/64 = 25% of the matrix.
   The mid-frequency band (rows 2-5 × cols 2-5, 16 entries
   overlapping the corner block) is what carries most photographic
   detail. Probing mid-band differentiation might give a real
   asymmetry.
3. **Different cluster cut.** Photo vs screen is a coarse axis. The
   `axis_class` column we used groups by source corpus, not by
   intrinsic content features. Re-clustering on entropy /
   edge-density / variance / patch-fingerprint features (per
   zenanalyze's own classifier feature set) might surface a sharper
   signal where one cluster's HF response really does differ from
   another's.
4. **Different chroma subsampling.** All probes were 4:2:0; 4:4:4
   keeps full chroma resolution and could change the screen-vs-photo
   asymmetry on UI screenshots where chroma carries text edges.

Any of (1)-(4) could be tested with the same probe binary by editing
the `make_table_bprime` factor / region or swapping the metric — each
is a 30-minute experiment, not a 6-10 hour SA run. The discipline
is: re-run the cheap signal probe before spending SA budget.

## Files

- Probe binary: `zenjpeg/dev/per_class_signal_probe_bprime.rs`
- Raw 4-cluster TSV (840 rows): `per_class_signal_probe_bprime_4cluster_2026-05-04.tsv`
- Raw 2-cluster TSV (360 rows, photo + screen first pass): `per_class_signal_probe_bprime_2026-05-04.tsv`
- Prior at-q probe: [`per_class_signal_probe_2026-05-04.md`](per_class_signal_probe_2026-05-04.md) + companion TSV
- Prior NO-SHIP analysis: [`zenjpeg/benchmarks/sa_piecewise_v5_finding_2026-05-04.md`](../../zenjpeg/benchmarks/sa_piecewise_v5_finding_2026-05-04.md)
- Analysis script: `/tmp/analyze_bprime.py` (not committed; quoted
  numbers reproduce from the raw TSV with a re-run of that script)
