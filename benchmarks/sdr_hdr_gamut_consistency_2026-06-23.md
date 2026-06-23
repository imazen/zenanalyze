# SDR / HDR / gamut feature-consistency — first cut (2026-06-23)

Settles the MLP-architecture question for HDR/wide-gamut content (one
regime-**conditioned** trunk vs **separate** models vs **deeper**) with a
controlled measurement instead of a prior.

## Method

`examples/sdr_hdr_gamut_sweep.rs`. For 40 real sRGB sources (imazen-26,
max-dim 1024) synthesize matched variants of **identical content**:

- `sdr` — sRGB, default gamma analysis (baseline).
- `hdr_h{1,2,4,8}` — same sRGB primaries; brightest **5 %** of pixels boosted
  ×N above diffuse white (203 nits), PQ-encoded, linear-light analysis. `h1`
  (no boost) is the OETF round-trip baseline; `h2/4/8` extend the highlights.
- `widegamut` — same values reinterpreted in Bt2020 primaries, gamma.

Data: `/mnt/v/output/zenanalyze-feature-axes/sdr_hdr_gamut_sweep_40img_2026-06-23.tsv`
(240 rows × 110 features). Synthetic-content companion:
`examples/sdr_hdr_consistency.rs` (95/99 invariant).

## Results

**1. Below diffuse white the OETF holds on real content.** 94/99 *content*
features are SDR/HDR-invariant within 3 % (`hdr_h1` vs `sdr`); only the
depth/regime features differ, by design (they measure the envelope). Matches
the synthetic probe's 95/99. → the below-diffuse-white manifold is **shared**.

**2. Super-white extends a majority of content features.** 60/99 shift >3 % at
`hdr_h8`: `variance` 2.4×, `laplacian_variance` 8×, `cb_peak_sharpness` 12×,
`dct_compressibility_y` 3.5×, `edge_slope_stdev` 3.2×. So HDR content features
are **not** axis-invariant — the regime must be modelled.

**3. The shift is mostly *conditionable* on a scalar headroom.** Per ramping
feature, R² of `hdr_feat ~ a·sdr_baseline + b·headroom_stops + c` over the
ladder (160 points):

| feature | h8 ramp | R²(sdr) | R²(sdr+headroom) |
|---|--:|--:|--:|
| `variance` | 2.4× | 0.125 | **0.900** |
| `edge_slope_stdev` | 3.2× | 0.102 | **0.875** |
| `dct_compressibility_y` | 3.5× | 0.389 | **0.790** |
| `quant_survival_y` | 0.66× | 0.827 | **0.948** |
| `info_weight_p90` | 0.43× | 0.531 | **0.876** |
| `cb_peak_sharpness` | 12× | 0.274 | 0.498 |
| `laplacian_variance` | 8× | 0.389 | 0.671 |
| `cr_peak_sharpness` | 4× | 0.240 | 0.350 |

**39/60 ramping features reach R²>0.90** with `(sdr + scalar headroom)`. The
~21 that don't are the **peak-sharpness / specular family** — their shift
depends on the highlight *distribution*, not just total headroom.

**4. Gamut is even more separable.** Only 7 features shift >3 % under Bt2020
(the chroma family + `chroma_luma_covariance_*`); the gamut **regime** features
`gamut_coverage_srgb/p3` flag it cleanly (rel-dev → 1.0). Luma-derived features
are gamut-invariant.

## Conclusion (architecture)

The data backs **one shared trunk, conditioned on the regime feature vector —
not separate models, not raw-deeper**:

- 94/99 content features are already shared below diffuse white (OETF).
- Of the 60 that extend, 39 are cleanly conditionable by `(baseline +
  headroom)`; the rest want the *fuller* regime vector
  (`hdr_pixel_fraction` + the `peak`/`p99` split), not separate models.
- Separate HDR models are doubly wrong: HDR data is scarce (76 sources) *and*
  the base manifold is shared — they'd overfit and relearn it.
- Gamut conditioning is nearly free (`gamut_coverage_*`).

→ **FiLM-condition the trunk on `{headroom_stops, hdr_pixel_fraction, peak/p99,
gamut_coverage_*}`** (the Phase-3 FiLM item), not a scalar.

## Caveats / next cut

- 40-image first cut — scale to a k-means-stratified ~150 and report held-out.
- **Fixed 5 % highlight fraction** — so this cannot test whether
  `hdr_pixel_fraction` rescues the peak-sharpness stragglers. A second sweep
  **varying highlight size** (1 %, 5 %, 20 %) at fixed headroom is the decisive
  test for those ~21 features.
- Wide-gamut variant reinterprets sRGB values as Bt2020 (synthetic saturation);
  a real wide-gamut source corpus would firm up the gamut axis.
