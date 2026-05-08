//! Rust-side ZNPR v2 byte-stream composer.
//!
//! Produces a `Vec<u8>` that round-trips through
//! [`crate::Model::from_bytes`]. Used by:
//!
//! - **Test fixtures.** Synthesize a small valid model in memory,
//!   parse it, run inference, compare against a numpy reference.
//! - **Placeholder weights.** zensim's V0_4 profile uses this at
//!   crate-init time to materialize a small "identity-ish" MLP for
//!   offline tests where no real bake is shipped.
//! - **Round-trip checks.** The Python baker
//!   (`zenanalyze/zenpicker/tools/bake_picker.py`) emits v2 bytes;
//!   `bake_roundtrip_check.py` compares Python forward-pass output
//!   to Rust-loader output. The Rust-side composer here lets the
//!   round-trip check fabricate adversarial inputs without
//!   touching the Python baker.
//!
//! For real shipped bakes — the kind that go in
//! `include_bytes!("zenjpeg_picker_*.bin")` — the canonical baker
//! is the Python pipeline. This module is the Rust mirror, kept in
//! lockstep on layout but not on the training-side conveniences
//! (loss functions, calibration metrics, manifest emission).

pub mod json;
mod v2;

pub use json::{
    ActivationJson, BakeJsonError, BakeLayerJson, BakeRequestJson, DtypeJson, FeatureBoundJson,
    MetadataEntryJson, MetadataKindJson, bake_from_json, bake_from_json_str,
};
pub use v2::{BakeError, BakeLayer, BakeMetadataEntry, BakeRequest, bake_v2};
