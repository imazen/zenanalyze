//! # zenpredict — zero-copy MLP runtime
//!
//! Parse a packed binary model (ZNPR v3 layer chain or v4 op graph),
//! run scaler + forward pass, surface typed metadata, run masked
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
//! let mut predictor = zenpredict::Predictor::new(&model);
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
//! Self-contained working example using the sibling `zenpredict-bake`
//! crate to mint a bake in-memory:
//!
//! ```rust,ignore
//! // requires zenpredict-bake = "0.1"
//! use zenpredict_bake::{BakeLayer, BakeRequest, bake};
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
//! let bytes = BakeRequest::builder(0, 0, &scaler_mean, &scaler_scale, &layers)
//!     .bake()
//!     .unwrap();
//!
//! // Load and predict. Real consumers wrap the bytes in
//! // `#[repr(C, align(16))]` to guarantee zero-copy alignment;
//! // the `bake` output is 16-aligned by virtue of being a
//! // freshly-allocated `Vec` (heap allocations are at least
//! // 8-aligned on every supported target — usually 16).
//! let model = Model::from_bytes(&bytes).unwrap();
//! let mut p = Predictor::new(&model);
//! let out = p.predict(&[3.0, 4.0]).unwrap();
//! assert_eq!(out, &[3.0, 4.0, 5.0]);
//! ```
//!
//! ## Depth and size
//!
//! Shape fields are `u32` in the binary, bounded at load by
//! [`limits`]: widths by `MAX_DIM` (65,536), layers by `MAX_LAYERS`,
//! graph nodes by `MAX_NODES`, compute by `MAX_TOTAL_WEIGHTS` (2^24
//! multiply-adds per forward pass) and scratch by `MAX_SCRATCH_ELEMS`
//! — each 100×–1000× above any shipped bake. Tests exercise
//! single-layer, ten-layer, 1024-wide-hidden, mixed-dtype-per-layer
//! (i8 → f16 → f32) and random-DAG shapes.
//!
//! Scratch is one f32 arena per [`Predictor`], packed by liveness at
//! load: a chain needs about `n_inputs + 2 × max_hidden` floats (a
//! 64-input 1024-hidden model, ~8 KB). [`Model::scratch_len`] reports the
//! widest vector in the network.
//!
//! ## Op graphs (ZNPR v4)
//!
//! A v4 bake replaces the layer chain with a static op graph —
//! [`NodeView`] lists the ops (`Input`, `Dense`, `Activation`, `Gather`,
//! `Add`, `Mul`, `Concat`) — so new head shapes need no runtime change.
//! Every model, v3 included, runs on one graph executor (a v3 chain is
//! `Input → Dense → …`). Walk a model's graph with [`Model::nodes`];
//! [`Model::is_layer_chain`] says whether [`Model::layers`] describes the
//! whole network. Spec: `docs/ZNPR_V4_GRAPH.md`.
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
//! `std` feature adds only `std::error::Error` impls; all numeric
//! work — including `f32::exp` for [`ScoreTransform::Exp`] — runs
//! identically on no_std via the unconditional `libm` dependency, so
//! there is no degraded transform path.
//!
//! ## Crate boundary
//!
//! `zenpredict` is the Rust runtime. The training pipeline lives at
//! `zenanalyze/zenpicker/` (Python) — pareto sweep, teacher fit,
//! distill, ablation, holdout probes. The two are versioned and
//! released independently; the format (`ZNPR v3`) is the contract
//! between them.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod argmin;
mod bounds;
mod directed_search;
mod encode_strategy;
mod error;
mod feature_transform;
mod graph;
mod inference;
mod knob_veto;
pub mod limits;
mod metadata;
mod model;
pub mod output_spec;
mod picker_safety;
mod predictor;
#[cfg(feature = "advanced")]
pub mod rescue;
#[cfg(feature = "advanced")]
mod safety;
mod unachievable_zone;
pub mod wire;

// Default-surface picker selection kit: the masked argmin/top-K
// ranking plus the runtime constraint masks (`mask_at_least` =
// target-quality floor, `mask_at_most` = perf/cost ceiling). Top-K
// lives here — not in each consumer — so the masking / score-transform
// / NaN / tie-break contract is defined once. The verify *loop* (rank
// → encode → measure → pick) is the codec's to compose over these.
pub use argmin::{
    AllowedMask, ArgminOffsets, ScoreTransform, argmin_masked, argmin_masked_in_range,
    argmin_masked_top_k, argmin_masked_top_k_in_range, mask_at_least, mask_at_most,
};
#[cfg(feature = "advanced")]
pub use argmin::{
    argmin_masked_top_k_with_scorer, argmin_masked_with_scorer, pick_with_confidence,
    pick_with_confidence_in_range,
};
pub use bounds::{FeatureBound, first_out_of_distribution};
#[cfg(feature = "advanced")]
pub use bounds::{OutputBound, output_first_out_of_distribution};
pub use error::PredictError;
pub use feature_transform::{FeatureTransform, apply_feature_transforms};
pub use inference::f16_bits_to_f32;
// Feature-gated knob-veto safety bounds: on the default surface alongside
// the rest of the picker selection kit (argmin/AllowedMask/mask_at_*),
// since `apply_knob_vetoes` is the same shape — a pre-argmin masking pass.
pub use knob_veto::{
    KNOB_VETOES_KEY, KnobVeto, VetoOp, apply_knob_vetoes, knob_vetoes_from_metadata,
    parse_knob_vetoes,
};
pub use metadata::{Metadata, MetadataEntry, MetadataType, keys};
// Size-discriminated unachievable-zone fallbacks: same default-surface
// placement as `knob_veto` — both are pre-argmin picker-selection overlays
// the codec composes around its argmin (here a `resolve`-then-skip-argmin
// short-circuit; there a masking pass).
pub use unachievable_zone::{
    UNACHIEVABLE_ZONES_KEY, UnachievableZone, UnachievableZones, ZoneFallback,
    parse_unachievable_zones, unachievable_zones_from_metadata,
};
// Picker safety pipeline: the canonical composition of the pre-argmin safety
// overlays (zone resolve → knob vetoes) into one call, so every codec enforces
// the bake-validated bounds in the same order. The post-encode rescue step
// lives in `rescue` (advanced); `picker_safety`'s module docs document the
// full 4-step pipeline tying them together.
pub use directed_search::{QualityTarget, Trial, best_trial, next_trial};
pub use encode_strategy::{EncodeBudget, EncodeMode, PickerStrategy};
pub use graph::NodeView;
pub use model::{
    Activation, EXP_INPUT_CLAMP, FORMAT_VERSION, GRAPH_FORMAT_VERSION, Header, LEAKY_RELU_ALPHA,
    LayerEntry, LayerView, Model, SOFTPLUS_THRESHOLD, Section, WeightDtype, WeightStorage,
};
pub use output_spec::{OutputSpec, OutputTransform, SparseOverride};
#[cfg(feature = "advanced")]
pub use output_spec::{OutputValue, apply_spec};
pub use picker_safety::{PreArgminDecision, resolve_pre_argmin};
pub use predictor::Predictor;
#[cfg(feature = "advanced")]
pub use rescue::{RescueDecision, RescuePolicy, RescueStrategy, should_rescue};
#[cfg(feature = "advanced")]
pub use safety::{CellHint, FallbackEntry, SafetyCompact, SafetyProfile, fallback_for};

#[cfg(test)]
mod tests;
