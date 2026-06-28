<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# zenpicker

Codec-family meta-picker. Given image features + a quality target + an allowed-family mask, picks one of `{jpeg, webp, jxl, avif, png, gif}`. Per-codec pickers (separate ZNPR v3 bakes shipped by the codec crate) handle config selection within the chosen family.

`#![forbid(unsafe_code)]`. `no_std + alloc` capable. Built on top of [zenpredict](https://github.com/imazen/zenanalyze/tree/main/zenpredict) — wraps a `Predictor` whose output dimension equals the number of families. AGPL-3.0-only / Commercial dual license.

## Where it sits

```
       features (zenanalyze) + target_zq + caller constraints
                              │
                              ▼
                 ┌──────────────────────┐
                 │ zenpicker            │   one ZNPR v3 model;
                 │  meta-picker         │   N_outputs = N families
                 └──────────┬───────────┘
                            │ chosen family
                            ▼
                 ┌──────────────────────┐
                 │ Per-codec picker     │   one .bin per family,
                 │  (zenpredict model)  │   shipped from the codec
                 │  → cell + scalars    │   crate
                 └──────────┬───────────┘
                            ▼
                   concrete EncoderConfig
```

The meta-picker emits a `CodecFamily`; it does **not** know how to resolve a family into a concrete encoder config. That's the per-codec picker's job — same ZNPR v3 format, separate `.bin`, baked from a sweep over that codec's config grid.

## Quick start

```toml
[dependencies]
zenpicker = "0.1.0"
zenpredict = "0.2.0"   # the runtime; provides `Model` and the ZNPR v3 parser
```

```rust,ignore
use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker};
use zenpredict::Model;

// `Model::from_bytes*` copies the bake into an owned, internally-aligned
// buffer, so a plain `include_bytes!` works as-is (no `#[repr(align)]` wrapper).
const META_BIN: &[u8] = include_bytes!("meta_picker_v1.bin");
// The schema hash chosen at bake time (read it with `zenpredict-inspect`),
// compiled in so a stale/mismatched bake fails loudly at load.
const MY_SCHEMA_HASH: u64 = 0x0123_4567_89ab_cdef;

let model = Model::from_bytes_with_schema(META_BIN, MY_SCHEMA_HASH)?;
let mut meta = MetaPicker::new(&model);  // borrows the Model; Model must outlive it
meta.validate_family_order()?;          // hard-fail if bake disagrees with enum

let allowed = AllowedFamilies::all()
    .deny(CodecFamily::Gif)              // caller bans GIF for this request
    .deny(CodecFamily::Png);
let chosen = meta.pick(&features, &allowed)?;
match chosen {
    Some(CodecFamily::Webp) => /* dispatch to per-codec webp picker */,
    Some(CodecFamily::Jxl)  => /* … */,
    None                    => /* nothing allowed; caller fallback */,
    // …
}
```

## Family order is a load-time contract

The output index of the meta-picker model maps 1:1 to a `CodecFamily` discriminant. The bake declares the order via the `zenpicker.family_order` metadata key (UTF-8, comma-separated lowercase labels). `MetaPicker::validate_family_order` reads that key on a parsed `Model` and refuses if it doesn't match the runtime's `ALL_LABELS_CSV`.

Adding a `CodecFamily` variant is a breaking change for any baked meta-picker that existed before — bake a fresh meta-picker that includes the new family before deploying.

## Companion crates

- **[zenpredict](https://github.com/imazen/zenanalyze/tree/main/zenpredict)** — the runtime this crate composes on. Owns the ZNPR v3 binary format, the parser, the forward pass, the masked-argmin math, the metadata blob, and the `Predictor`. zenpicker adds: family enum + family-order validation + `AllowedFamilies` mask sugar.
- **[zentrain](https://github.com/imazen/zenanalyze/tree/main/zentrain)** — the Python training pipeline that produces the `.bin` artifact a meta-picker (or a per-codec picker) loads. Train with `cells = families` and `output_layout = bytes_log` only (purely categorical, no scalar heads). The bake's metadata block must include `zenpicker.family_order`.
- **[zenanalyze](https://github.com/imazen/zenanalyze)** — the feature extractor that produces the input vector both this meta-picker and the per-codec pickers consume.

## Status

v0.1 establishes the crate boundary and the API shape. Baking an actual cross-codec meta-picker is downstream work — once a labelled training set exists where each row maps `(image features, target_zq) → best family`, run zentrain's `train_hybrid.py` with cells = families and `output_layout` of `bytes_log` only.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · [resamplescope-rs] |
| Pickers & ML | [zenanalyze] · [zenpredict] · **zenpicker** |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
