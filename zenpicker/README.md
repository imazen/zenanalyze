# zenpicker ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpicker?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpicker?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpicker) ![docs.rs](https://img.shields.io/docsrs/zenpicker?style=flat-square) ![License](https://img.shields.io/crates/l/zenpicker?style=flat-square)

Codec-family meta-picker. Given image features + a quality target + an allowed-family mask, picks one of `{jpeg, webp, jxl, avif, png, gif}`. Per-codec pickers (separate ZNPR v2 bakes shipped by the codec crate) handle config selection within the chosen family.

`#![forbid(unsafe_code)]`. `no_std + alloc` capable. Built on top of [zenpredict](../zenpredict/) — wraps a `Predictor` whose output dimension equals the number of families. AGPL-3.0-only / Commercial dual license.

## Where it sits

```
       features (zenanalyze) + target_zq + caller constraints
                              │
                              ▼
                 ┌──────────────────────┐
                 │ zenpicker            │   one ZNPR v2 model;
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

The meta-picker emits a `CodecFamily`; it does **not** know how to resolve a family into a concrete encoder config. That's the per-codec picker's job — same ZNPR v2 format, separate `.bin`, baked from a sweep over that codec's config grid.

## Quick start

```rust,ignore
use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker};
use zenpredict::Model;

#[repr(C, align(16))]
struct Aligned<const N: usize>([u8; N]);
const META_BIN: &[u8] = &Aligned(*include_bytes!("meta_picker_v1.bin")).0;

let model = Model::from_bytes_with_schema(META_BIN, MY_SCHEMA_HASH)?;
let mut meta = MetaPicker::new(model);
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

- **[zenpredict](../zenpredict/)** — the runtime this crate composes on. Owns the ZNPR v2 binary format, the parser, the forward pass, the masked-argmin math, the metadata blob, and the `Predictor`. zenpicker adds: family enum + family-order validation + `AllowedFamilies` mask sugar.
- **[zentrain](../zentrain/)** — the Python training pipeline that produces the `.bin` artifact a meta-picker (or a per-codec picker) loads. Train with `cells = families` and `output_layout = bytes_log` only (purely categorical, no scalar heads). The bake's metadata block must include `zenpicker.family_order`.
- **[zenanalyze](../)** — the feature extractor that produces the input vector both this meta-picker and the per-codec pickers consume.

## Status

v0.1 establishes the crate boundary and the API shape. Baking an actual cross-codec meta-picker is downstream work — once a labelled training set exists where each row maps `(image features, target_zq) → best family`, run zentrain's `train_hybrid.py` with cells = families and `output_layout` of `bytes_log` only.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial.
