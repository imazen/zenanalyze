//! # zenpredict — zero-copy MLP runtime
//!
//! Parse a packed binary model (ZNPR v2 format), run scaler +
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
//! ```ignore
//! let bytes: &'static [u8] = include_bytes!("zenjpeg_picker_v2.2.bin");
//! let model = zenpredict::Model::from_bytes(bytes)?;
//! let mut predictor = zenpredict::Predictor::new(model);
//!
//! let features = my_codec::extract_features(&analysis, target_zq);
//! let mask = my_codec::allowed_configs(&caller_constraints);
//! let pick = predictor.argmin_masked(
//!     &features,
//!     &mask,
//!     zenpredict::ScoreTransform::Exp,
//!     None,
//! )?;
//! ```
//!
//! ## Format stability
//!
//! ZNPR v2 — fixed `#[repr(C)]` header + offset table + zero-copy
//! data sections + a TLV metadata blob. See [`model`] for the byte
//! layout. v1 (the original 32-byte-header positional layout) is
//! not supported by this crate; older bakes need to be rebaked
//! through the v2 baker.
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
mod inference;
mod metadata;
mod model;
mod predictor;
pub mod rescue;

#[cfg(feature = "bake")]
pub mod bake;

pub use argmin::{
    AllowedMask, ArgminOffsets, ScoreTransform, argmin_masked, argmin_masked_in_range,
    argmin_masked_top_k, argmin_masked_top_k_in_range, pick_with_confidence,
    pick_with_confidence_in_range, threshold_mask,
};
pub use bounds::{FeatureBound, first_out_of_distribution};
pub use error::PredictError;
pub use metadata::{Metadata, MetadataEntry, MetadataType, keys};
pub use model::{
    Activation, FORMAT_VERSION, Header, LEAKY_RELU_ALPHA, LayerEntry, LayerView, Model, Section,
    WeightDtype, WeightStorage,
};
pub use predictor::Predictor;
pub use rescue::{RescueDecision, RescuePolicy, RescueStrategy, should_rescue};

#[cfg(test)]
mod tests;
