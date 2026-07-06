# zenpicker [![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/zenanalyze/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/zenpicker?style=flat-square)](https://crates.io/crates/zenpicker) [![lib.rs](https://img.shields.io/crates/v/zenpicker?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpicker) [![docs.rs](https://img.shields.io/docsrs/zenpicker?style=flat-square)](https://docs.rs/zenpicker) [![MSRV](https://img.shields.io/badge/MSRV-1.93-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/badge/license-AGPL--3.0%20%2F%20Commercial-blue?style=flat-square)](#license)

Cross-codec **family router**: given image features (a zenanalyze `Offer`) + a quality target + the formats you can emit, it picks the best of `{jpeg, webp, jxl, avif, png, gif}` — **one-shot, no trial encodes**. Per-codec pickers (separate ZNPR bakes shipped by each codec crate) then choose the config *within* the chosen family.

`#![forbid(unsafe_code)]`, `no_std + alloc` capable. Built on top of [zenpredict](https://github.com/imazen/zenanalyze/tree/main/zenpredict) (the ZNPR runtime). AGPL-3.0-only / Commercial dual license.

## Where it sits

```
       features (zenanalyze) + target_zq + caller constraints
                              │
                              ▼
                 ┌──────────────────────┐
                 │ zenpicker            │   gate + lossy +
                 │  family router       │   lossless routers
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

## The decision

```text
lossless?  — caller asked, or target ≥ ~96 (near-perfect → store exact)
viable     — formats you allow  ∩  formats the image needs (alpha / HDR — content_capability)
                                 ∩  formats fast enough for the latency budget (viable)
order      — best-first among the viable; take the top
```

The **order** is the only place judgment lives, and there are two ways to get it:

- **`default_route` (the shipped default)** — runs the baked routers: a **6-pairwise-discriminant
  linear lossy router** (held-out **7.16% mean / 22.05% p90** extra bytes vs the perfect oracle on
  the zensim-A retrain, 7f4d914 — the *median* pick is the oracle), an auto-gate (lossy vs
  lossless), and a lossless family router. Image-aware, one-shot, and the lossy router's weights
  are readable (the size-dependence is isolated to the jxl-vs-avif pair; the rest is pure content).
- **`family_rule` (the no-model fallback)** — a fixed codec-reality prior (`JXL > AVIF > WebP > JPEG
  > GIF` lossy; `JXL > WebP > PNG > GIF` lossless), now confirmed by the data. Use it when you have
  no features, or want an obviously-correct, model-free path.

Both mask to **any subset of available formats** — one, several, or none — so the pick is always sane.

## Quick start

```toml
[dependencies]
zenpicker = "0.1.0"
zenpredict = "0.2.0"   # the runtime; provides `Model` and the ZNPR v3 parser
```

```rust,ignore
use zenpicker::{CodecFamily, QualityTarget, default_route};
use zenpredict::EncodeMode;

// `offer` is a zenanalyze-api `Offer` (the image's features). Mask to the formats you can emit:
let decision = default_route(
    &offer,
    QualityTarget::Zq(82.0),
    &[CodecFamily::Jpeg, CodecFamily::Webp, CodecFamily::Avif], // available formats
    EncodeMode::QueuedBalanced,   // no latency budget; RealtimeFastest would drop slow codecs
    None,                         // latency_ms
    &[0; CodecFamily::COUNT],     // per-family encode-time estimates (used by realtime modes)
)?;
match decision {
    Some(d) => { let _family = d.family(); /* → dispatch to that codec's per-codec picker */ }
    None    => { /* nothing available can encode it, or no features — fall back to family_rule */ }
}
```

No features (or want an obviously-correct, model-free path)? Use the prior — just capability + the codec-reality order, no offer-materialization:

```rust,ignore
use zenpicker::{family_rule, AllowedFamilies, CodecFamily, QualityTarget};
use zenpredict::EncodeMode;

let family: Option<CodecFamily> = family_rule(
    &offer, QualityTarget::Zq(82.0),
    AllowedFamilies::from_allowed([CodecFamily::Webp, CodecFamily::Avif]),
    EncodeMode::QueuedBalanced, None, &[0; CodecFamily::COUNT],
);
```

For a hot loop, hold one `MetaPicker::default_routers()` and call `.route(..)` repeatedly — the parsed models are process-static (`OnceLock`); only the per-call `Predictor` scratch is rebuilt.

## `pick` vs `route`

`MetaPicker::pick(features, allowed)` is a raw masked-argmin over a **family-score** model — for per-codec pickers, the auto-gate, or the lossless router. It **refuses the pairwise lossy router** (which emits per-pair margins, not family scores) with `MetaPickerError::PairwiseRouterNeedsRoute`: use `route` / `default_route` for the lossy family choice, which round-robins the margins into a family.

## Family order is a load-time contract

For a **family-score** model (the auto-gate, the lossless router, per-codec pickers) the output index maps 1:1 to a `CodecFamily` discriminant. The bake declares the order via the `zenpicker.family_order` metadata key (UTF-8, comma-separated lowercase labels). `MetaPicker::validate_family_order` reads that key on a parsed `Model` and refuses if it doesn't match the runtime's `ALL_LABELS_CSV`.

(The **pairwise lossy router** is the exception: its 6 outputs are codec *pairs*, not families — `route` round-robins them into a family. It records the pair order in `zenpicker.lossy_pairwise`, and `pick` refuses it.)

Adding a `CodecFamily` variant is a breaking change for any baked meta-picker that existed before — bake a fresh meta-picker that includes the new family before deploying.

## Companion crates

- **[zenpredict](https://github.com/imazen/zenanalyze/tree/main/zenpredict)** — the runtime this crate composes on. Owns the ZNPR v3 binary format, the parser, the forward pass, the masked-argmin math, the metadata blob, and the `Predictor`. zenpicker adds: family enum + family-order validation + `AllowedFamilies` mask sugar.
- **[zentrain](https://github.com/imazen/zenanalyze/tree/main/zentrain)** — the Python training pipeline + `bake_picker.py` (the `BakeRequestJson` emitter). The **lossy router** is fit as 6 pairwise linear discriminants (zenmetrics `scripts/picker/pairwise_discriminants.py`) and baked **f32** via `bake_picker.py`; the auto-gate, lossless, and per-codec pickers are trained via `train_hybrid.py`. Bakes carry `zenpicker.family_order` (+ `zenpicker.lossy_pairwise` for the lossy router).
- **[zenanalyze](https://github.com/imazen/zenanalyze)** — the feature extractor that produces the input vector both this meta-picker and the per-codec pickers consume.

## Status

The shipped routers (`MetaPicker::default_routers`) are baked and wired: an **f32 6-pairwise-discriminant lossy router** (held-out **7.16% mean / 22.05% p90** extra bytes vs the perfect oracle on the zensim-A retrain (7f4d914) — the median pick is the oracle, the loss is a thin tail concentrated on tiny images), plus **i8** auto-gate + lossless family routers, all trained on confound-corrected sweep data (no re-sweep). `default_route` is the one-call entry, masked by available format. Methodology + the per-percentile RD distribution: zenmetrics `docs/HOW_THE_PICKER_DECIDES.md`.

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
