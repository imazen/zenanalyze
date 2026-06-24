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

The damning column is **tonemap**: BT.2446A's global range-compression is a
*contractive* map in feature space — it makes different images look more alike
(variance spread collapses 7×, edges 2.5×). A picker exists to tell images apart;
tonemap washes out exactly the separation it needs. **clip** tracks the SDR spread
(886 ≈ 977); **extend** inflates and adds noise (1944).

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

## Caveats

- Synthetic HDR (top-5 % pushed to super-white, declared peak = headroom×diffuse).
  The tonemap arm's absolute drift is *amplified* by declaring a high peak on
  mostly-SDR content — but the **cross-image discriminability collapse** (C) is a
  structural property of contractive tone-curves and is not a synthesis artifact.
- One content slice (imazen-26 photos) at one highlight fraction (5 %). A
  follow-up varying highlight-pct {1, 5, 20} and adding screen/line-art content
  would harden the C-table, but the A/B/C ordering (clip ≫ extend ≫ tonemap for
  feature use) is already unambiguous.
