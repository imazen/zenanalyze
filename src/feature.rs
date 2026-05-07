//! Public API surface: a stable, opaque, set-composable feature
//! interface that lets multiple codecs share one analysis pass.
//!
//! **Stability contract: there is no 0.2.x.** Every change in 0.1.x
//! is additive — new variants on `#[non_exhaustive]` enums, new
//! parallel functions, never a signature change to a shipped item.
//! See `CLAUDE.md`.
//!
//! Stability contract:
//! - [`AnalysisFeature`] discriminants are **immutable once shipped**.
//!   A retired variant keeps its `u16` slot forever; new variants get
//!   the next sequential number. Never reuse a number — that would
//!   silently break callers persisting [`FeatureSet`] bits to disk or
//!   wire.
//! - [`FeatureSet`] storage is opaque. Future versions may grow the
//!   backing bitset; the public ops are `const fn` set math.
//! - [`FeatureValue`] is `#[non_exhaustive]` — future structured
//!   feature types (small histograms, vectors) can land via additional
//!   variants without a major bump.
//!
//! Composition pattern:
//! ```ignore
//! // Each codec exposes the features it cares about.
//! const JPEG_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::EdgeDensity)
//!     .with(AnalysisFeature::DctCompressibilityY);
//! const WEBP_FEATURES: FeatureSet = FeatureSet::new()
//!     .with(AnalysisFeature::Variance)
//!     .with(AnalysisFeature::AlphaPresent)
//!     .with(AnalysisFeature::AlphaBimodalScore);
//!
//! // Orchestrator unions and runs once.
//! let needed = JPEG_FEATURES.union(WEBP_FEATURES);
//! // analyze(slice, &AnalysisQuery::new(needed)) → AnalysisResults …
//! ```
//!
//! Design note — typed-feature path (deferred):
//!
//! A separate sealed-trait `Feature` + `IsActive` system was
//! prototyped alongside this enum, with one zero-sized struct per
//! feature (`pub struct Variance(_)`). It would let callers write
//! `results.get_typed::<Variance>() -> Option<f32>` (no runtime
//! variant match) and would make `FeatureSet::just_typed::<F>()`
//! refuse to compile when `F` is a retired feature. Deferred to a
//! later round so we can settle the dynamic surface first; the macro
//! that generates the witnesses is straightforward when the enum
//! variants are stable. **Action item before adding it:** decide
//! whether the 30 extra public type names are worth the compile-time
//! gating, or whether `#[deprecated]` warnings on the enum variant +
//! `None` at runtime is enough.

/// Single-source-of-truth macro: generates the [`AnalysisFeature`]
/// enum, its `id` / `from_u16` / `is_active` / `name` impls, the
/// internal [`RawAnalysis`] dense struct that the SIMD tiers write
/// into, the `RawAnalysis::into_results` translator, and the
/// [`FeatureSet::SUPPORTED`] preset — all in lockstep so adding a
/// feature is a one-line edit at the invocation site.
///
/// Per-feature row syntax:
/// `$(#[$attr:meta])* $Variant = $id : $type => $field`
/// The `$field` ident is the snake_case name (also used for
/// `AnalysisFeature::name`'s string return via `stringify!`).