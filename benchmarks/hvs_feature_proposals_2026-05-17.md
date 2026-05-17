# HVS-derived feature proposals for zenanalyze (2026-05-17)

## Executive summary

Eight candidates, ranked by ROI. Top three — **Y/Cb–Cr covariance**,
**IW-SSIM local information weight**, **orientation energy ratio** —
capture codec-divergence axes that current features only weakly
approximate. Two more (spectral slope, 3-band CSF energy) plug
frequency-band gaps; three are speculative and likely redundant.

## Existing-features inventory

zenanalyze ships ~102 features (`src/feature.rs:271` `features_table!`).
HVS coverage today: local energy/spread (`variance`,
`laplacian_variance`+%iles, `aq_map_*`, `variance_spread`); edges
(`edge_density`, `edge_slope_stdev`, `gradient_fraction`); frequency
(`high_freq_energy_ratio`, `dct_compressibility_y/uv`,
`quant_survival_*`); chroma (`chroma_complexity`, `cb/cr_*_sharpness`,
`colourfulness`); distribution (`luma_kurtosis`,
`luma_histogram_entropy`); content class (`patch_fraction(_fast)`,
`skin_tone_fraction`); HDR/gamut and geometry.

**Missing:** orientation-resolved energy, multi-band CSF decomposition,
per-pixel information content (Wang–Li 2011), spectral slope, Oklab
chroma spread, Y–chroma covariance.

## Per-candidate proposals

### 1. `feat_chroma_luma_covariance` (Y-Cb, Y-Cr Pearson) — **best ROI**

Two scalars in `[−1, 1]` from Tier-1 running sums.
**Why:** AVIF/AV1 chroma-from-luma wins big when chroma tracks luma;
JPEG can't exploit it. **Cost:** ~0.1 ms/MP — pure accumulator
extension. **Risk:** low; orthogonal to all current features.

### 2. `feat_info_weight_mean` / `_p90` — IW-SSIM weights

Wang–Li 2011: `w(x) = log2(1 + σ²_p(x) / σ²_e)`, σ²_p from a 5×5
Gaussian, σ²_e from existing `noise_floor_y`. Emit mean + p90.
**Why:** measures where the eye attends; high spread predicts AVIF/JXL
adaptive-quant wins. zensim's `iw_pool.rs` already runs this internally.
**Cost:** ~1.5 ms/MP; piggybacks on the laplacian accumulator.
**Risk:** medium overlap with `aq_map_p75/p90` — different transfer
function; ablate via LOO.
Cite: Wang & Li, IEEE TIP 2011.

### 3. `feat_orientation_energy_ratio` — directional anisotropy

Four oriented Sobel kernels (0°/45°/90°/135°) on stripe-sampled luma;
emit `max(Σ_θ) / mean(Σ_θ)` (axis-aligned vs isotropic).
**Why:** AV1/JXL have directional transforms; JPEG doesn't.
Text/UI is anisotropic, photos isotropic. `edge_slope_stdev` measures
magnitude spread, not direction.
**Cost:** ~1.0 ms/MP — reuses edge-sweep stripe.
**Risk:** partial overlap with `patch_fraction` for screen content; low.
Cite: Freeman & Adelson 1991, IEEE TPAMI.

### 4. `feat_spectral_slope_y` — 1/f exponent per block

Fit `log|F(r)| = a − β·log r` over 4–6 radial bins of the per-block
DCT magnitude already in `block_acs`; emit mean β.
**Why:** continuous photo-vs-graphic axis; `high_freq_energy_ratio` is a
single cut. Steep β favours JXL VarDCT; flat β favours wavelet codecs.
**Cost:** ≤ 0.5 ms/MP at default `hf_max_blocks=1024`.
**Risk:** medium overlap with `dct_compressibility_y`.
Cite: Field, JOSA A 1987.

### 5. `feat_csf_band_energy_{lo,mid,hi}` — three CSF-weighted bands

Laplacian-pyramid band energies at 3 scales (2×/4×/8× downsample) on
luma, weighted by a fixed castleCSF gain table.
**Why:** AVIF wins mid-frequency, WebP loop-filter wins low,
JPEG near-DC; the picker can route. Single-ratio `high_freq_energy_ratio`
can't.
**Cost:** ~2.5 ms/MP (one shared pyramid pass).
**Risk:** medium; spatial-domain band-pass vs DCT-zigzag — likely
complementary.
Cite: Mantiuk et al. SIGGRAPH 2024 castleCSF.

### 6–8. Lower-ROI candidates (one-line each)

- **`feat_oklab_chroma_spread`** — `√(var(a*)+var(b*))` on Oklab stripe
  samples. Captures perceptually-uniform chroma variance; XYB / CfL
  signal. Cost ~2 ms/MP (cbrt). **High overlap risk** with
  `chroma_complexity`; ablate first. Cite Ottosson 2020.
- **`feat_log_luma_adaptation_spread`** — p99−p1 of `log(1+Y_linear)` on
  a 32×32 box-blur. CVVDP masking normalizer. ~2 ms/MP; medium overlap
  with `variance_spread`. Cite Mantiuk 2024 §3.2.
- **`feat_phase_congruency_mean`** — Kovesi 2-scale × 4-orient log-Gabor
  PC. Strong line-art discriminator but ~8–12 ms/MP and high overlap
  with `patch_fraction` (AUC 0.880 already). **Likely reject.** Cite
  Kovesi, Videre 1999.

## Ranked table

| # | Feature | Expected lift | Cost ms/MP | Redundancy | Effort |
|--|--|--|--|--|--|
| 1 | `chroma_luma_covariance` | med-high | 0.1 | low | low |
| 2 | `info_weight_mean/_p90` | med-high | 1.5 | medium | medium |
| 3 | `orientation_energy_ratio` | medium | 1.0 | low | medium |
| 4 | `spectral_slope_y` | medium | 0.5 | medium | medium |
| 5 | `csf_band_energy_{lo,mid,hi}` | medium | 2.5 | medium | high |
| 6 | `oklab_chroma_spread` | low-med | 2.0 | high | medium |
| 7 | `log_luma_adaptation_spread` | low-med | 2.0 | medium | medium |
| 8 | `phase_congruency_mean` | low (after #3) | 8–12 | high | high |

Ship #1 first (one-day work, near-free). Validate #2–#4 with the same
transform-sweep harness that produced the 2026-05-17 results. Decide
#5–#8 from ablation outcomes.

## What NOT to add

**Full butteraugli / HDR-VDP-3 / CVVDP front-ends.** Even
preprocessing-only subsets cost 20–60 ms/MP — way over budget.

**Per-pixel CIEDE2000.** ~50 ns/pixel of trig. Oklab Δab² captures most
of the signal at ~5 ns/pixel (proposal #6).

**Saliency / foveation maps.** Cheap proxies reduce to features we
already have; deep variants need ONNX. Revisit when zentract integrates.

**Multi-scale steerable pyramid (Simoncelli).** Covers #3, #4, #5
combined but at ~15 ms/MP. Use the cheap approximations first.

**Full FSIM PC+GM.** PC rejected (#8); GM is `laplacian_variance` +
`edge_density` already.

Sources:
- [ColorVideoVDP (Mantiuk et al. 2024)](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2024_ColorVideoVDP.pdf)
- [Phase Congruency (Kovesi 2003)](https://www.peterkovesi.com/papers/phasecorners.pdf)
- [IW-SSIM (Wang & Li, IEEE TIP 2011)](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)
- [Oklab (Ottosson 2020)](https://bottosson.github.io/posts/oklab/)
- zensim IW estimator: `/home/lilith/work/zen/zensim/zensim/src/iw_pool.rs`
