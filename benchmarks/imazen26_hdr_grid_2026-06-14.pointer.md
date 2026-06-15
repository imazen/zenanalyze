# imazen-26 HDR size grid — renditions + features (2026-06-14)

The size-grid follow-on to `imazen26_hdr_2026-06-14` (native HDR). Bit-depth-
preserving, **linear-light** downscales of the 76 16-bit PQ HDR sources,
mirroring the SDR Mitchell-sharp render grid, with zenanalyze features (depth
tier live) extracted per rendition — so the HDR training corpus spans sizes,
not just native resolution (per the dense-size sweep discipline).

## Method (correctness-critical)

`zenresize::resize_u16` **hardcodes the sRGB transfer** to linearize, which is
wrong for PQ/HLG and would corrupt highlights. So `examples/extract_hdr_size_grid.rs`
instead:

1. Decodes each `.hdr.png` (16-bit, native-endian u16) + reads its `cICP`.
2. Linearizes with the **correct** transfer (`linear_srgb::tf::pq_to_linear`,
   or `hlg_to_linear`).
3. Resizes in **linear f32** — `zenresize` `Filter::Mitchell` + `resize_sharpen`
   — downscale only, ~16 log-spaced longest edges (128..3072) per source.
4. Re-encodes linear → 16-bit PQ/HLG (`linear_to_pq`, clamped to [0,1]).
5. Extracts features via the bit-depth/transfer-aware `analyze_features`
   (`RGB16` + transfer + primaries).

**Native size is read straight from the source u16 (no transfer round-trip)** —
verified byte-identical (all 110 features) to the `imazen26_hdr_features`
native rows, so the grid's native path reproduces the validated extraction.

**Renditions** are written via the `image` crate (correct pixels + endianness),
then a `cICP` chunk is spliced after IHDR so they stay self-describing — verified
on a sample: 16-bit RGB, chunks `[IHDR, cICP, IDAT, IEND]`, cICP `(1,16,0,1)` =
BT.709/PQ/full-range, all CRCs valid (incl. the spliced chunk), decodes cleanly.
Pixel/IDAT bytes are left untouched by the splice.

### Sharpen caveat (documented, not a bug)

The f32 path sharpens in **linear** light (`resize_f32` runs unsharp on the
linear output), whereas the SDR `resize()` path sharpens in **gamma** (on the u8
output, clamped at 255). On HDR content linear unsharp can lift highlights, but
the effect is modest (~3–6% on `peak_luminance_nits`, measured sharpen=0 vs 10).
The larger peak-vs-size swing is the depth tier's **stride-sampled** peak being
resolution-dependent (a 4000px native undersamples the brightest pixels; small
downscales sample a larger fraction) — honest feature behavior the dense-size
corpus is meant to capture, not corruption. Renditions are deliberate Mitchell+
sharpen downscales; their features describe them exactly.

## Features (canonical)

`/mnt/v/output/imazen-26-features/imazen26_hdr_grid_features_2026-06-14.parquet`
— **1216 rows × 116 cols** (6 meta + 110 features), sha256-16 `bb835b307e20fff2`,
Tower-mirrored (`/mnt/tower/output/imazen-26-features/`). 76 sources × ~16 sizes
(downscale-only; fewer for smaller sources). Schema = the native HDR file's
(`variant_name / content_class / width / height / primaries / transfer / feat_*`).

- `hdr_present` = true for all 1216 rows.
- peak luminance **215–1776 nits** (median 635), headroom **1.43–4.47 stops**
  (median 2.99); 100 distinct widths, 90–5712 px.
- Variant naming preserves the leading id + size, e.g.
  `1064_..._3000x4000.scale1200x1600` (native keeps the source stem).

## Renditions

`/mnt/v/output/imazen-26-hdr-grid-2026-06-14/` — **1140 16-bit PQ `.hdr.png`**
(the downscales; native already lives in `imazen-26-hdr-2026-06-14/`), 7.8 GB,
cICP-tagged. Block storage only (regenerate from the command below); not
Tower-mirrored (bytes regenerate).

## Regenerate

```bash
cargo run --release --features experimental,hdr --example extract_hdr_size_grid -- \
  --hdr-dir      /mnt/v/output/imazen-26-hdr-2026-06-14 \
  --features-out /mnt/v/output/imazen-26-features/imazen26_hdr_grid_features_<DATE>.tsv \
  --out-dir      /mnt/v/output/imazen-26-hdr-grid-<DATE> \
  --sharpen 10
```

Hold-out (per corpus convention) is still by **even leading filename id**; the
leading id is preserved through the `.scale` suffix so the split carries over.
