//! ZNPR v3 bake-side composer.
//!
//! Produces a `Vec<u8>` that round-trips through [`zenpredict::Model::from_bytes`].
//! Used by:
//!
//! - **Test fixtures.** Synthesize a small valid model in memory, parse it,
//!   run inference, compare against a reference forward pass.
//! - **Placeholder weights.** zensim's `PreviewV0_4` profile uses the rust
//!   composer at crate-init time to materialize a small "identity-ish" MLP
//!   for offline tests where no real bake is shipped.
//! - **Round-trip checks.** The Python baker (`zentrain/tools/bake_picker.py`)
//!   emits v3 bytes; bake_roundtrip checks compare the Python forward-pass
//!   output to the Rust-loader output. The Rust-side composer here lets the
//!   round-trip check fabricate adversarial inputs without touching the
//!   Python baker.
//!
//! For real shipped bakes — the kind that go in `include_bytes!("zenjpeg_picker_*.bin")`
//! — the canonical baker is the Python pipeline in zentrain. This crate is the
//! Rust mirror, kept in lockstep with the parser's wire format so we can
//! produce valid bakes from pure-Rust integration tests.
//!
//! # Crate split rationale (0.2.0)
//!
//! `zenpredict-bake` is a separate crate from [`zenpredict`] so consumers that
//! only load pre-baked `.bin` blobs via `include_bytes!` don't link
//! `serde_json` (and its monomorph overhead) into their runtime. Codec runtime
//! depends on `zenpredict`; trainers and tooling depend on `zenpredict-bake`.
//!
//! [`zenpredict`]: https://docs.rs/zenpredict
//! [`zenpredict::Model::from_bytes`]: https://docs.rs/zenpredict/latest/zenpredict/struct.Model.html#method.from_bytes

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod composer;
pub mod json;

pub use composer::{
    BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, BakeRequestBuilder, bake,
};
pub use json::{
    ActivationJson, BakeJsonError, BakeLayerJson, BakeRequestJson, DtypeJson, FeatureBoundJson,
    MetadataEntryJson, OutputSpecJson, OutputTransformJson, SparseOverrideJson, bake_from_json,
    bake_from_json_str,
};
