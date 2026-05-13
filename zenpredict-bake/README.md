# zenpredict-bake ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenanalyze/ci.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/zenpredict-bake?style=flat-square) [![lib.rs](https://img.shields.io/crates/v/zenpredict-bake?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/zenpredict-bake) ![docs.rs](https://img.shields.io/docsrs/zenpredict-bake?style=flat-square) ![License](https://img.shields.io/crates/l/zenpredict-bake?style=flat-square)

Bake-side composer for the [zenpredict](../zenpredict/) ZNPR v3 binary format. Compose a `Vec<u8>` that round-trips through [`zenpredict::Model::from_bytes`](https://docs.rs/zenpredict/) — used by zentrain's Python pipeline (via the `zenpredict-bake` CLI binary) and by anyone writing Rust integration tests against a freshly-baked model.

`#![forbid(unsafe_code)]`. `no_std + alloc` compatible (the `std` feature only adds `std::error::Error` impls). MIT / Apache-2.0 dual license — same as zenpredict.

## Why a separate crate?

The bake side pulls in `serde` + `serde_json` + a hand-rolled JSON visitor. Together those are ~30–40 % of zenpredict's monomorphization budget at 0.1.x — a cost every codec consumer paid for code zero of them called.

`zenpredict-bake = "0.1"` ships the composer + JSON baker + CLI binaries. `zenpredict = "0.2"` is the lean runtime. Codec runtimes (`include_bytes!` + parse + predict) depend only on `zenpredict`; trainers and tooling depend on `zenpredict-bake`. Build times for downstream codecs drop measurably.

## Binaries

- **`zenpredict-bake`** — reads a JSON `BakeRequestJson` on stdin / from a file, writes a v3 `.bin`. The Python trainer at `zentrain/tools/bake_picker.py` shells out to this.
- **`zenpredict-inspect`** — loads a v3 `.bin` and emits a JSON dump of header + sections + per-layer dim/dtype/activation + metadata. Useful for spot-checking a bake before shipping it.

## Rust API

Three entry points, ordered roughly by ergonomics:

```rust,ignore
// 1. Builder — fluent, default-empty optional sections.
use zenpredict_bake::{BakeLayer, BakeRequest, bake};
use zenpredict::{Activation, WeightDtype};

let layers = [BakeLayer {
    in_dim: 2, out_dim: 3,
    activation: Activation::Identity,
    dtype: WeightDtype::F32,
    weights: &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    biases: &[0.0, 0.0, 5.0],
}];
let bytes = BakeRequest::builder(0, 0, &[0.0, 0.0], &[1.0, 1.0], &layers)
    .bake()
    .unwrap();
```

```rust,ignore
// 2. Direct struct — best when you have every section ready.
let req = BakeRequest {
    schema_hash: 0xfeed,
    flags: 0,
    scaler_mean: &mean,
    scaler_scale: &scale,
    layers: &layers,
    feature_bounds: &fb,
    metadata: &md,
    output_specs: &os,
    discrete_sets: &ds,
    sparse_overrides: &so,
};
let bytes = bake(&req).unwrap();
```

```rust,ignore
// 3. JSON in — for language-agnostic toolchains.
use zenpredict_bake::bake_from_json_str;
let bytes = bake_from_json_str(json_string).unwrap();
```

## Format

The `bake` function emits ZNPR v3. Wire layout, validation, and migration from earlier formats are documented in [zenpredict's README](../zenpredict/README.md) and [`zenpredict/src/model.rs`](../zenpredict/src/model.rs).

v2 bins from earlier ships migrate via [`zentrain/tools/migrate_znpr_v2_to_v3.py`](../zentrain/tools/migrate_znpr_v2_to_v3.py) (header rewrite only — layer payloads are byte-identical).

## Features

| Feature | Default | What it gates |
|---|---|---|
| `std` | yes | `std::error::Error` impls on `BakeError` / `BakeJsonError` |

## License

MIT OR Apache-2.0, at your option. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
