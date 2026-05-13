//! Shared wire-format constants for the ZNPR v3 binary format.
//!
//! These offsets and sizes are byte-exact and used by both the parser
//! ([`crate::Model::from_bytes`]) and the composer in the sibling
//! `zenpredict-bake` crate. Keeping them in one module guarantees the
//! two sides can't drift — a parser-side offset change must update
//! the composer in lockstep and vice versa.
//!
//! See [`crate::Header`] for the full byte layout documentation.

/// Size of the fixed-format header in bytes.
pub const HEADER_SIZE: usize = 128;

/// Size of one [`crate::LayerEntry`] in bytes.
pub const LAYER_ENTRY_SIZE: usize = 48;

/// Header byte offset of the `scaler_mean` [`crate::Section`].
pub const SECTION_OFF_SCALER_MEAN: usize = 32;

/// Header byte offset of the `scaler_scale` [`crate::Section`].
pub const SECTION_OFF_SCALER_SCALE: usize = 40;

/// Header byte offset of the `layer_table` [`crate::Section`].
pub const SECTION_OFF_LAYER_TABLE: usize = 48;

/// Header byte offset of the `feature_bounds` [`crate::Section`].
pub const SECTION_OFF_FEATURE_BOUNDS: usize = 56;

/// Header byte offset of the `metadata` [`crate::Section`].
pub const SECTION_OFF_METADATA: usize = 64;

/// Header byte offset of the `output_specs` [`crate::Section`] (v3+).
pub const SECTION_OFF_OUTPUT_SPECS: usize = 72;

/// Header byte offset of the `discrete_sets` [`crate::Section`] (v3+).
pub const SECTION_OFF_DISCRETE_SETS: usize = 80;

/// Header byte offset of the `sparse_overrides` [`crate::Section`] (v3+).
pub const SECTION_OFF_SPARSE_OVERRIDES: usize = 88;
