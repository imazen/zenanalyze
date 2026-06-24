# HDR content-feature representation: which one zenanalyze should offer

**Date:** 2026-06-23
**Data:** `/mnt/v/output/zenanalyze-feature-axes/hdr_3rep_sweep_2026-06-23.{parquet,tsv}`
(60 imazen-26 source photos, max-dim 768, highlight-pct 5, headroom ladder ×{1,2,4,8})
**Harness:** `examples/sdr_hdr_gamut_sweep.rs` (path-dep zenpixels-convert 0.2.15 +
`hdr-experimental` for the tone-map arm — a *local, uncommitted* experiment; the
crate's default deps stay at the registry `0.2.14`).

## Question

For an HDR image, what should the **content-feature** tiers (variance, edges,
chroma sharpness, DCT energy, …) read? Three candidate representations:

| arm | what it does | API |
|---|---|---|
| **extend** | linear-light, super-white (>diffuse-white) extends unbounded | `with_linear_light(true)` |
| **clip** | linear-light, super-white hard-clipped to diffuse white | `with_linear_light(true).with_diffuse_white_clip(true)` |
| **tonemap** | BT.2446A SDR-display rendition (highlights rolled off into 0–100 nit) | `zenpixels_convert::ConvertPlan::new_with_hdr_peak` → RGB8 |

Each synthetic-PQ image pushes its top 5 % of pixels into super-white up to
`headroom`× diffuse white; we compare each arm's features to the **same image's
SDR baseline**.

## Result — clip wins decisively; tonemap is actively harmful for features

### A. Content-feature drift from the SDR baseline (median over images; lower = more SDR-like)

| headroom | extend | **clip** | tonemap |
|---:|---:|---:|---:|
| 1 | 0.0001 | 0.0001 | 0.665 |
| 2 | 0.029 | 0.025 | 0.591 |
| 4 | 0.058 | 0.030 | 0.469 |
| 8 | 0.082 | **0.031** | 0.389 |

- **extend** drift *climbs with headroom* (→8 %): super-white inflates the
  content features, and how much depends on how bright the highlights are — i.e.
  **not headroom-stable**.
- **clip** drift is small and *flat* (≈3 %, plateaus): it is **SDR-invariant up to
  the hard-clip of the top few %** (clip maps super-white→pure white, which the
  SDR original retained as gradation — hence the small non-zero residual, not a bug).
- **tonemap** sits 39–67 % away from SDR: BT.2446A re-renders the *whole* tonal
  range, so even the displayable base shifts massively. It is a different image,
  not an SDR-consistent reading of the same one.

### B. Highlight-descriptor carriage (the 6 `highlight_*` depth-tier features)

`extend` and `hdr_clip` produce **byte-identical** highlight descriptors
(`max|extend − clip| = 0.0` across all 6) — clipping the content path does **not**
touch the source-direct depth tier, so the highlight texture signal is carried
*separately and intact* regardless of clipping. `tonemap` reads **all zeros** (it
goes through the SDR `analyze_features_rgb8` path with no depth tier): its highlight
texture is instead baked — lossily — into its (heavily-shifted) content features.

This is the **clip-and-separate** thesis, now measured: clip the content path for
SDR-invariance, and let the 6 dedicated descriptors carry the highlight signal.

### C. Cross-image discriminability (std of a feature across the 60 images, headroom 8)

| feature | sdr | extend | **clip** | tonemap |
|---|---:|---:|---:|---:|
| variance | 977 | 1944 | 886 | **139** |
| edge_density | 0.15 | 0.15 | 0.15 | **0.06** |
| chroma_complexity | 0.07 | 0.10 | 0.07 | **0.03** |

The damning column is **tonemap**: BT.2446A's global range-compression collapses
the spread (variance 7×, edges 2.5×). **clip** tracks the SDR spread (886 ≈ 977);
**extend** inflates and adds noise (1944). Std alone is a weak proxy, though — the
*Robustness check* below confirms tonemap is genuinely *scrambling* the inter-image
structure (distance-geometry corr 0.197 vs SDR), not merely shrinking it, while
clip preserves it (0.965).

## Recommendation

**Ship clip-and-separate. Do not adopt tone-mapping for feature extraction.**

1. **Content tiers** read the **clip** representation (`with_diffuse_white_clip`):
   SDR-invariant, headroom-stable, and discriminability-preserving. This is what
   commit `929ad11` already landed (chunk 1/3).
2. **Highlight texture** is carried by the 6 source-direct `highlight_*`
   descriptors (ids 212–217), independent of the clip. Already landed.
3. **tonemap is rejected for features.** It is a *display* operation (great for
   showing an HDR image on an SDR screen) but the wrong thing to feed a feature
   extractor: it is not SDR-consistent and it collapses inter-image separation.
   It remains useful in `zenpixels-convert` as a rendition path; it should not
   become zenanalyze's HDR content representation.

## Robustness check — does C's spread-collapse mean lost *separability*?

Metric C (std-across-images) is the weakest of the three: a *monotone contractive*
map shrinks std but can preserve rank, and a GBDT/MLP splits on rank/structure, not
absolute spread — so "std collapsed 7×" does not by itself prove lost
discriminability. I tested that weak claim directly (`hdr_3rep_validity_rank.py`,
output `hdr_3rep_validity_rank_2026-06-23.txt`), headroom 8, 60 images:

| representation | per-feature rank ρ vs SDR (median) | inter-image distance-geometry corr vs SDR |
|---|---:|---:|
| extend | 0.972 | 0.724 |
| **clip** | **0.981** | **0.965** |
| tonemap | 0.915 | **0.197** |

- **Per-feature rank** (Spearman, marginal): all three keep *most* individual
  features' ordering (ρ ≥ 0.91 median) — so the collapse is **not** simple
  per-feature monotone shrinkage a model could trivially undo.
- **Joint geometry** (Pearson corr of the 60×60 pairwise z-scored distance
  matrices, standardised by SDR's scale) is the decisive probe: **clip 0.965**
  (an SDR-calibrated model sees essentially SDR's structure) vs **tonemap 0.197**
  (the inter-image geometry is *scrambled*, not merely shrunk). Tonemap's
  worst-scrambled features are exactly the texture/HF ones its roll-off destroys:
  `noise_floor_y`, `laplacian_variance`, `quant_survival`, `edge_slope_stdev`,
  `patch_fraction`.

This **upgrades** the C claim from "spread shrank" (weak) to "the inter-image
structure an SDR-trained model relies on is preserved by clip and destroyed by
tonemap" (a proper separability statement). **Precise wording:** because the
distances are z-scored by *SDR* stats, this measures **drop-in compatibility with
an SDR-calibrated model** — not intrinsic undiscriminability. A tonemap-*native*
model might recover separability, but then you've surrendered the single-model
reuse that is clip's whole architectural advantage.

## Confidence grading per claim

| claim | confidence | basis |
|---|---|---|
| extend is headroom-unstable (drift → 8 %) | **High** | monotone, large, definitional (super-white extends the value range) |
| clip is headroom-stable (drift flat ~3 %) | **High** | by construction — clip caps at diffuse white, so features can't depend on the clipped-away super-white |
| clip ≈ SDR for the displayable base | **High** (as a bound) | ~3 % is a small upper bound; the exact value is metric-dependent |
| highlight descriptors are clip-invariant (extend ≡ clip, diff = 0.0) | **Very high** | code-path fact — depth tier reads source, not the clipped stream; exact-zero diff |
| clip preserves SDR inter-image geometry; tonemap scrambles it | **High** | distance-matrix corr 0.965 vs 0.197 — a large, unambiguous gap |
| **single SDR-trained MLP generalises to HDR-clip content** | **Hypothesis** | feature geometry (0.965) only; **no model was trained or scored** |
| tonemap collapse *magnitude* (7× std / 0.197 geometry) | **Medium** | direction robust; magnitude confounded by the aggressive peak declaration + single content slice |

## Threats to validity

**Construct — does the test exercise real HDR?**
- **Synthetic, not real HDR.** "HDR" here = SDR photos with their top-5 % luma
  pixels multiplied in linear light and re-encoded to PQ. There is **no genuine
  super-white detail** (clipped skies, specular texture, light-source structure) a
  real HDR capture carries. So the *highlight-texture-preservation* argument — the
  one reason tonemap could theoretically win — is **under-tested**: there is little
  real highlight detail for clip to lose or tonemap to save. The invariance and
  geometry findings (about the *displayable base*) are unaffected; the
  highlight-texture tradeoff is **not** settled by this run.
- **Tonemap peak is a confound.** I declared source_peak = 203 × headroom (1624
  nits at hr 8), but only 5 % of pixels reach it; a content-adaptive / MaxFALL-like
  peak would compress far less. The *direction* of the collapse is structural; the
  *magnitude* is inflated by the aggressive peak. (Real HDR files do declare MaxCLL
  peaks, so the declaration isn't unrealistic — but here headroom and declared-peak
  were tied, which a real corpus separates.)

**Internal — do the metrics support the claims?**
- Metric C was **upgraded, not eliminated** — the rank/geometry test backs it, but
  its SDR-z-scoring encodes the specific question "does an SDR model generalise,"
  stated precisely above.
- Metric A's drift denominator `|a−b|/(|b|+1e-6+0.01|a|)` is ad-hoc: the *ordering*
  is robust to it, the specific percentages (3 / 8 / 39–67 %) are not.
- "Content features" were auto-detected as non-zero across these 60 SDR images
  (99/116); a feature coincidentally ~0 on this corpus could be mis-excluded.

**External — does it generalise?**
- **One content domain** — imazen-26 personal photos (general/interiors/nature).
  **No screen, line-art, synthetic, or document content**, which the project sweep
  discipline requires. Photo-only.
- **One highlight fraction (5 %).** A 1 % specular glint vs a 20 % blown sky stress
  the representations very differently (clip flattens ~nothing at 1 %, a lot at
  20 %). Untested.
- **One resolution** (max-dim 768, Lanczos3). Downscaling lowpasses highlight
  detail (weakening the texture test further) and features are size-dependent — one
  point on a size curve.

**Statistical**
- **n = 60, no confidence intervals.** Point estimates only. The geometry gap
  (0.965 vs 0.197) is large enough to survive n = 60 sampling error; the 3 %-vs-8 %
  drift gap is smaller and not interval-bounded.
- **No downstream-task validation.** Every conclusion is about *feature statistics*
  — never picker/MLP accuracy, bytes, or zensim. No model was trained on any
  representation; "clip wins" is inferred from invariance + geometry, and the MLP
  implication is explicitly a hypothesis.

## What would change the conclusion / what to test next

1. **Real HDR corpus** with genuine super-white detail (not a synthetic boost) —
   the only way to settle whether clip's hard-clip discards highlight texture a
   tonemap rendition keeps. If it does, the answer could shift to "clip + richer
   highlight descriptors" or a per-content-class choice.
2. **Train the model.** Decisive test of the single-MLP hypothesis: train a head on
   SDR features, evaluate on the HDR-clip parquet vs an HDR-tonemap parquet. The
   geometry result *predicts* clip generalises and tonemap doesn't — predict ≠
   measure.
3. **Vary highlight-pct {1, 5, 20} + add screen/line-art content** — harden
   external validity.
4. **Separate declared-peak from headroom** so the tonemap magnitude is not
   confounded.

**Bottom line.** The *recommendation* (ship clip-and-separate; don't make tonemap
the default feature path) rests on the High-confidence findings — clip is
headroom-stable, preserves SDR inter-image geometry, and the highlight separation
is an exact code-path fact — and is robust to every weak metric above: even halving
tonemap's collapse magnitude leaves it incompatible with SDR-model reuse. What is
**not** yet earned: declaring the HDR-MLP-architecture question closed (needs a
trained model), or claiming clip loses no real highlight texture (needs real HDR
content).

## Reproducibility

- Data: `/mnt/v/output/zenanalyze-feature-axes/hdr_3rep_sweep_2026-06-23.{parquet,tsv}`
  (840 rows = 60 images × {sdr, widegamut, 4×[extend, clip, tonemap]}).
- Analysis (in-repo): `benchmarks/hdr_3rep_compare.py` (A/B/C) +
  `benchmarks/hdr_3rep_validity_rank.py` (robustness). Each takes the parquet path
  as argv[1] (defaults to the /mnt/v path above). Captured validity output:
  `/mnt/v/output/zenanalyze-feature-axes/hdr_3rep_validity_rank_2026-06-23.txt`.
- Harness: `examples/sdr_hdr_gamut_sweep.rs` at commit `401a04b` (extend+clip arms);
  the tonemap arm ran out-of-tree against path-dep zenpixels-convert 0.2.15
  `hdr-experimental` (the crate's committed deps stay on registry 0.2.14).
