# Picker overhead — bytes / time / zensim (in-sample teacher, val split)

When picker disagrees with oracle, what does the user actually lose
across the three dimensions that matter (size, encode speed, achieved
quality)?

**Methodology**: in-sample HistGB-per-cell teacher trained on the
train slice of each codec's pareto, evaluated on the image-level val
holdout (seed 0xCAFE, 20% of images). Per row: oracle = argmin(actual
bytes), pick = argmin(predicted bytes). Compute delta_bytes,
delta_encode_ms, delta_zensim. Reproduces via
`tools/picker_overhead_3d.py`.

The teacher is a proxy for the production MLP picker (both are
bytes-targeted, both have full pareto access during training). The
production MLP's argmin_acc is slightly different (it has joint-codec
trunk context) but the SHAPE of the miss distribution is similar.

## Results

| Codec   | argmin_acc | bytes mean | bytes p90 | bytes p99 | bytes max | time p90 | time p99 | zensim mean | zensim p90 | zensim max |
|---------|----------:|----------:|---------:|---------:|---------:|--------:|--------:|-----------:|-----------:|-----------:|
| zenjpeg |     58.5% |      1.8% |      5.7% |    17.5% |    95.7% |   +4 ms |  +43 ms |     +0.22  |     +0.82  |    +21     |
| zenwebp |     41.7% |      2.5% |      7.5% |    16.1% |    86.6% |    n/a  |    n/a  |     +0.20  |     +0.82  |    +22     |
| zenavif |     28.9% |      3.5% |     10.6% |    27.1% |    88.5% | +592 ms |  +10 s  |     +0.63  |     +2.01  |    +11     |

## Reading the numbers

1. **Bytes overhead**: p50 ≈ 0% in all three codecs — when picker is "wrong"
   half the time it lands on a cell that ties the oracle in bytes. Tail
   bounded: p99 ≤ 27%, max ~90% on rare images.

2. **Encode time delta is BIMODAL** (mean often NEGATIVE because the
   oracle min-bytes pick frequently uses high-effort encoding; picker
   sometimes picks a faster lighter config). The slow tail exists but
   is rare: zenjpeg p99 +43 ms, zenavif p99 +10 s.

3. **Zensim delta is POSITIVE** (zensim_pick > zensim_oracle on
   average) — picker tends to OVER-DELIVER quality vs the requested
   zq target. p50 = 0 (often exactly matches), p90 +0.8 to +2.0
   zensim above oracle. So when picker disagrees the user gets MORE
   quality, not less, at the cost of a few percent more bytes.

## Why "wrong picks" cost so little

zenjpeg has 12 output cells; many cells are near-ties on bytes at any
given (image, zq). The picker model predicts log-bytes, then argmin
picks the smallest predicted. A near-tie noisy pick lands on a
sibling cell with similar bytes but possibly different
chroma/quantization characteristics. The siblings share roughly the
same bytes; they differ in zensim (chroma 444 vs 420) and time
(progressive vs baseline).

## Caveats

- This is the TEACHER (HistGB-per-cell), not the trained MLP. The MLP
  has joint context that the teacher lacks. Production argmin_acc for
  zenjpeg is 54.5% (MLP) vs 58.5% (teacher); zenwebp 51.2% (MLP) vs
  41.7% (teacher); zenavif 31.0% (MLP) vs 28.9% (teacher). The
  shape of misses transfers; absolute numbers shift ±5-10 pp.
- zenwebp's pareto has no encode_ms column — speed delta unavailable.
- zensim_delta uses log(zensim) per cell from `metric_log` returned
  by `build_dataset(emit_metric_head=True)`; converted back to linear
  via `exp()`. Sign-preserved.
