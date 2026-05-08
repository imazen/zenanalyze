//! # zenpredict — zero-copy MLP runtime
//!
//! Parse a packed binary model (ZNPR v3 format), run scaler +
//! layer-by-layer forward pass, surface typed metadata, run masked
//! argmin for codec-config selection.
//!
//! Two consumer shapes:
//!
//! 1. **Codec picker** (zenjpeg / zenwebp / zenavif / zenjxl) — owns
//!    a [`Predictor`], builds a feature vector + an [`AllowedMask`],
//!    calls [`Predictor::argmin_masked`] to pick a config.
//! 2. **Perceptual scorer** (zensim V0_4) — owns a [`Predictor`],
//!    feeds a feature vector through [`Predictor::predict`], reads
//!    the first output as a scalar distance.
//!
//! The decision math (masked argmin, top-K, score transforms,
//! additive offsets, fallback policy) is generic — codecs use it,
//! anything else with a "pick one of N predicted scores" shape can
//! use it too.
//!
//! ## Lifecycle
//!
//! Real consumers `include_bytes!` a baked `.bin`:
//!
//! ```ignore
//! let bytes: &'static [u8] = include_bytes!("zenjpeg_picker_v2.2.bin");
//! let model = zenpredict::Model::from_bytes(bytes)?;
//! let mut predictor = zenpredict::Predictor::new(model);
//!
//! let features = my_codec::extract_features(&analysis, target_zq);
//! let mask_data = my_codec::allowed_configs(&caller_constraints);
//! let mask = zenpredict::AllowedMask::new(&mask_data);
//! let pick = predictor.argmin_masked(
//!     &features,
//!     &mask,
//!     zenpredict::ScoreTransform::Exp,
//!     None,
//! )?;
//! ```
//!
//! Self-contained working example using the [`bake`] module:
//!
//! ```rust
//! use zenpredict::bake::{BakeLayer, BakeRequest, bake_v2};
//! use zenpredict::{Activation, Model, Predictor, WeightDtype};
//!
//! // Bake a 2-input → 3-output identity-ish model.
//! let scaler_mean = [0.0f32, 0.0];
//! let scaler_scale = [1.0f32, 1.0];
//! let weights = [
//!     1.0f32, 0.0, 0.0, // input 0 → outs
//!     0.0, 1.0, 0.0,    // input 1 → outs
//! ];
//! let biases = [0.0f32, 0.0, 5.0];
//! let layers = [BakeLayer {
//!     in_dim: 2,
//!     out_dim: 3,
//!     activation: Activation::Identity,
//!     dtype: WeightDtype::F32,
//!     weights: &weights,
//!     biases: &biases,
//! }];
//! let bytes = bake_v2(&BakeRequest {
//!     schema_hash: 0,
//!     flags: 0,
//!     scaler_mean: &scaler_mean,
//!     scaler_scale: &scaler_scale,
//!     layers: &layers,
//!     feature_bounds: &[],
//!     metadata: &[],
//!     output_specs: &[],
//!     discrete_sets: &[],
//!     sparse_overrides: &[],
//! }).unwrap();
//!
//! // Load and predict. Real consumers wrap the bytes in
//! // `#[repr(C, align(16))]` to guarantee zero-copy alignment;
//! // the `bake_v2` output is 16-aligned by virtue of being a
//! // freshly-allocated `Vec` (heap allocations are at least
//! // 8-aligned on every supported target — usually 16).
//! let model = Model::from_bytes(&bytes).unwrap();
//! let mut p = Predictor::new(model);
//! let out = p.predict(&[3.0, 4.0]).unwrap();
//! assert_eq!(out, &[3.0, 4.0, 5.0]);
//! ```
//!
//! ## Depth and size are unconstrained
//!
//! The format puts no fixed limits on the network's shape. Number
//! of layers, layer widths, and input / output dimensions are all
//! `u32` in the binary header — the practical limit is whatever the
//! `n_inputs * out_dim` multiplications in `usize` can handle. Tests
//! exercise single-layer, ten-layer, 1024-wide-hidden, and
//! mixed-dtype-per-layer (i8 → f16 → f32) shapes.
//!
//! Scratch buffers are sized to `max(n_inputs, max_layer_out_dim) *
//! sizeof(f32)`, computed by [`Model::scratch_len`]. A 64-input
//! 1024-hidden model needs 4 KB of scratch — trivially small.
//!
//! ## Format stability
//!
//! ZNPR v3 — fixed `#[repr(C)]` header + offset table + zero-copy
//! data sections + a TLV metadata blob + optional per-output
//! [`OutputSpec`] / discrete-set / sparse-override sections. See
//! [`Header`] and [`LayerEntry`] for the wire layout, and the source
//! of `model.rs` for the documented byte offsets. Earlier formats
//! (v1, v2) are not supported by this crate; older bakes need to be
//! rebaked through the v3 baker.
//!
//! ## Storage
//!
//! Weights are stored as f32, f16, or i8. f16 conversion is built
//! in (no `half` dep). i8 carries one f32 scale per output neuron.
//!
//! ## no_std
//!
//! `default-features = false` keeps the crate `no_std + alloc`. The
//! `std` feature adds `std::error::Error` impls and `f32::exp` for
//! [`ScoreTransform::Exp`]; `no_std` builds without an exp
//! implementation degrade `Exp` to a tied score (see
//! [`ScoreTransform`] docs). Add an `alloc-libm` feature in a
//! future patch if the no_std-with-Exp consumer ever surfaces.
//!
//! ## Crate boundary
//!
//! `zenpredict` is the Rust runtime. The training pipeline lives at
//! `zenanalyze/zenpicker/` (Python) — pareto sweep, teacher fit,
//! distill, ablation, holdout probes. The two are versioned and
//! released independently; the format (`ZNPR v2`) is the contract
//! between them.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod argmin;
mod bounds;
mod error;
mod feature_transform;
mod inference;
mod metadata;
mod model;
pub mod output_spec;
mod predictor;
pub mod rescue;
mod safety;

#[cfg(feature = "bake")]
pub mod bake;

pub use argmin::{
    AllowedMask, ArgminOffsets, ScoreTransform, argmin_masked, argmin_masked_in_range,
    argmin_masked_top_k, argmin_masked_top_k_in_range, argmin_masked_top_k_with_scorer,
    argmin_masked_with_scorer, pick_with_confidence, pick_with_confidence_in_range, threshold_mask,
};
pub use bounds::{
    FeatureBound, OutputBound, first_out_of_distribution, output_first_out_of_distribution,
};
pub use error::PredictError;
pub use feature_transform::{FeatureTransform, apply_feature_transforms};
pub use inference::f16_bits_to_f32;
pub use metadata::{Metadata, MetadataEntry, MetadataType, keys};
pub use model::{
    Activation, FORMAT_VERSION, Header, LEAKY_RELU_ALPHA, LayerEntry, LayerView, Model, Section,
    WeightDtype, WeightStorage,
};
pub use output_spec::{OutputSpec, OutputTransform, OutputValue, SparseOverride, apply_spec};
pub use predictor::Predictor;
pub use rescue::{RescueDecision, RescuePolicy, RescueStrategy, should_rescue};
pub use safety::{CellHint, FallbackEntry, SafetyCompact, SafetyProfile, fallback_for};

#[cfg(test)]
mod tests;
