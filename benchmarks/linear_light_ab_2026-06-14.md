# Linear-light opt-in prototype — per-feature A/B (2026-06-14)

The "prototype opt-in mode" measurement for the linear-light decision (follow-on
to `linear_light_precision_review_2026-06-14.md`). Goal: measure, per feature,
how much computing the SDR luma-domain tier-1 features in **linear light** and at
**higher precision** changes them — so we decide which features earn a production
linear kernel *before* building f32 SIMD kernels.

## Method

`examples/linear_light_ab.rs` reimplements the four luma-domain tier-1 features
(variance, edge density, uniformity, laplacian variance) as a scalar f32 kernel
over a precomputed luma plane in [0,255], mirroring tier1.rs's reductions exactly
(full-image, not stripe-sampled). The **only** thing that varies between arms is
how the luma plane is built, so each axis isolates one effect:

- **`sdr_fidelity_gamma_vs_shipped`** — my gamma kernel vs the shipped
  `analyze_features`. Confirms the reimplementation tracks the real feature
  (full-image vs stripe-sampled → expect high correlation, not equality). This
  is the control: if fidelity is high, the linear-vs-gamma delta below is a real
  domain effect, not a kernel artifact.
- **`sdr_linear_effect_gamma_vs_linear`** — my gamma vs my linear
  (`srgb_to_linear(v/255)·255`, same [0,255] scale). The linear-light effect,
  sampling held constant.
- **`hdr_precision_full16_vs_narrow8`** — on the 16-bit PQ renditions:
  linear-from-16-bit vs linear-from-narrowed-8-bit (`round(code·255/65535)`).
  Isolates whether genuine >8-bit precision moves the features (same linear
  domain, only precision differs).

Metrics per axis × feature: Pearson correlation (does the *ranking* survive),
median relative |Δ| and max relative |Δ| (effect *magnitude*).

Corpus: imazen-26 display-SDR PNGs (`*.sdr.png`) for the SDR axes; the 76 16-bit
PQ renditions (`*.hdr.png`) for the precision axis. Sample: 200 SDR + 76 HDR.

## Results

200 SDR + 76 HDR images. Committed: `benchmarks/linear_light_ab_2026-06-14.tsv`.

**Control — `sdr_fidelity_gamma_vs_shipped`** (my gamma kernel vs shipped):

| feature | Pearson | median rel Δ | max rel Δ |
|---|--:|--:|--:|
| variance | 0.9945 | 1.6% | 43.7% |
| edge_density | 0.9985 | 2.8% | 67.9% |
| uniformity | 0.9971 | 1.4% | 39.0% |
| laplacian_variance | 0.9985 | 3.0% | 46.7% |

High correlation (≥0.994), low median Δ — the reimplementation tracks the real
feature. (Max-Δ outliers are individual high-frequency images where full-image
diverges from stripe sampling; expected, doesn't affect the controlled axes.)

**`sdr_linear_effect_gamma_vs_linear`** (the linear-light effect):

| feature | Pearson | median rel Δ | max rel Δ |
|---|--:|--:|--:|
| variance | **0.8715** | **21.5%** | 81.7% |
| edge_density | 0.9587 | **25.9%** | 78.8% |
| uniformity | **0.9166** | **18.8%** | 83.0% |
| laplacian_variance | 0.9786 | **19.1%** | 85.0% |

**`hdr_precision_full16_vs_narrow8`** (the higher-precision effect):

| feature | Pearson | median rel Δ | max rel Δ |
|---|--:|--:|--:|
| variance | 1.0000 | 0.03% | 1.0% |
| edge_density | 0.9971 | 0.0% | 22.2% |
| uniformity | 1.0000 | 0.0% | 0.0% |
| laplacian_variance | 0.9999 | 3.5% | 55.1% |

## Interpretation

**Linear light is a large, consistent effect; higher precision is not.**

- **Linear light moves every feature 19–26% at the median**, and for `variance`
  and `uniformity` it reorders images (Pearson 0.87 / 0.92) — i.e. it changes not
  just magnitudes but which images rank as high-variance / uniform, which is what
  a picker keys off. `edge_density` / `laplacian_variance` shift in magnitude
  (median ~20–26%) but keep their ranking (Pearson 0.96–0.98). So linear light is
  decisively *not* a no-op — it is worth a production path.
- **Higher precision barely registers**: median ~0% across the board with the 16-
  vs 8-bit narrowing, the rankings essentially identical (Pearson ≥0.997). The
  only exception is `laplacian_variance` (median 3.5%, max 55%) and a small
  `edge_density` tail — both on high-frequency content where 8-bit quantization
  bites the fine ∇²/gradient signal. For variance/uniformity, 8-bit is plenty.

**What this means for the production opt-in (next chunk):**

1. Build the linear-light path; **prioritize `variance` + `uniformity`** (ranking
   changes most), then `edge_density` / `laplacian_variance`.
2. **Skip f32-from-source for most of it.** Since precision barely matters,
   the linear path can reuse an i12/i16 linear intermediate (à la
   zenresize's `srgb_u8_to_linear_i12`) — linearize once, keep the existing
   integer-style reductions — rather than a full f32-from-source kernel rewrite.
   Reserve the precision-preserving f32 source path for `laplacian_variance` /
   `edge_density` *if* the high-frequency tail proves to matter downstream.
3. This is a feature-value change → still gated behind the opt-in
   (`AnalysisQuery` flag), A/B'd against fitted consumers (encode-sweep picker
   accuracy, not just feature deltas) before any default flip. The deltas here
   say the *features* change a lot; whether that *improves picks* is the
   downstream A/B that justifies flipping the default.

Caveat: this measures the four luma-domain tier-1 features. Chroma-domain
(chroma_complexity, colourfulness, skin) and tier-3 DCT features may behave
differently in linear light — extend the kernel before generalizing the verdict.

## Reproduce

```bash
cargo run --release --features experimental --example linear_light_ab -- \
  --sdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
  --hdr-dir /mnt/v/output/imazen-26-hdr-2026-06-14 \
  --out /mnt/v/output/imazen-26-features/linear_light_ab_2026-06-14.tsv --limit 200
```
