# v07 knob exploration — patches/gaborish are real Pareto gainers (2026-05-05)

Compared v07's cell taxonomy (with force_strategy, max_strategy_size,
progressive, gaborish, patches, lz77, lf_frame, pixel_domain_loss
extras) against v06's (effort × distance × butteraugli_iters ×
zensim_iters) on the 265 (image, distance) cells where both sweeps
have data.

## Headline

| metric | value |
|---|---|
| (img, dist) cells with both sweeps | 265 |
| v06 best safe-alt mean savings | +5.26% bytes |
| **v07 best safe-alt mean savings** | **+8.16% bytes** |
| v07 strictly beats v06 | 117 cells (44.2%) |
| Mean extra savings on v07 wins | -8.83% bytes |

## Top winning v07-extra knob combos

| count | combo |
|---:|---|
| 42 | gaborish=False + patches=True |
| 18 | patches=True alone |
| 11 | patches=True + pixel_domain_loss=False |
| 11 | gaborish=False + patches=True + pixel_domain_loss=False |
| 8 | (no v07-extra knobs — v06 cells with v07-default values won) |
| 5 | force_strategy=0 (DCT8) |
| 4 | lf_frame=True |
| 4 | max_strategy_size=16 |
| 4 | force_strategy=4 (DCT4x4) |

## Surprising findings

- **`patches=True` is the dominant winner**, even though prior single-image
  testing suggested it was a no-op. With v07's diverse image subset the
  knob actually helps on ~30% of (image, distance) cells.
- **`gaborish=False` is second**. Disabling the post-filter saves bytes
  on screen/synthetic content where the smoothing it does isn't useful.
- **`force_strategy` is rarely picked** but when it is, DCT8 (0) and
  DCT4x4 (4) are the picks. Other strategies don't help.
- `progressive`, `lz77`, `lf_frame` had minimal impact in the sweep.

## Path forward: v08 sweep

Combine v06's main grid (4 efforts × 14 distances × 3 biters × 2
ziters) with v07's winners (patches, gaborish, pixel_domain_loss) as
new knob axes:

- effort × distance × biters × ziters × patches × gaborish × pdl
- = 4 × 14 × 3 × 2 × 2 × 2 × 2 = **2688 cells/image**
- × 500 clustered images = **1.34M cells**

That's 8x v06 size — too expensive for a full sweep. Better strategy:
sweep just the patches × gaborish × pdl combinations × a SMALLER grid
(e.g. 4 efforts × 8 distances × 2 biters × 1 ziter = 64 base cells),
giving 64 × 8 = 512 cells × 500 imgs = 256k cells. Comparable to v06.

This v08 grid lets the picker learn when patches/gaborish/pdl help
across diverse content, capturing the +2.9pp byte savings v07 hinted at.
