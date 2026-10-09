//! Resource limits enforced by the parser.
//!
//! These constants bound the parser's resource use against
//! adversarial input. The numbers are picked to comfortably exceed
//! every realistic shipped bake (V0_18 zensim is 93 KB, the largest
//! shipped picker is the zenavif rav1e v0.1.1 at 217 KB; the deepest
//! shipped layer count is 4) while staying small enough that a
//! gigabyte-claiming header fails fast.
//!
//! ## Hardening pattern
//!
//! Every limit is enforced **before** memory is allocated against
//! the value it bounds. `MAX_BAKE_BYTES` rejects the byte slice
//! itself before the header is read. `MAX_DIM` / `MAX_LAYERS` reject
//! the parsed-but-untrusted header values before the scratch
//! allocation in [`crate::Predictor::new`] tries to materialize them.
//! Whole-bake LZ4 decompression at load time additionally bounds
//! the decompressed payload to `MAX_BAKE_BYTES` minus header.
//!
//! ## Picking numbers
//!
//! 64 MB / 64 K / 256 are 1000x – 10000x larger than any production
//! bake. They're upper bounds that protect web / fuzz / no_std-alloc
//! consumers from runaway parsing without restricting legitimate
//! research / picker work. Adjust if a real bake ever approaches
//! the limit (none should).

/// Maximum byte length of a ZNPR `.bin` accepted by [`crate::Model::from_bytes`].
/// 64 MiB — every shipped bake is < 1 MiB; the limit exists to bound
/// fuzz / adversarial input.
pub const MAX_BAKE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum value of a per-layer or scaler dimension (`n_inputs`,
/// `n_outputs`, `in_dim`, `out_dim`). 65,536 — every shipped bake's
/// largest dim is 384; the limit caps multiplications like
/// `in_dim * out_dim` against `usize` overflow on 32-bit targets.
pub const MAX_DIM: usize = 65_536;

/// Maximum layer count. 256 — every shipped bake has ≤ 4 layers;
/// the limit exists so that `layer_table` allocations are bounded.
pub const MAX_LAYERS: usize = 256;

/// Maximum node count of a ZNPR v4 op graph. 1024 — the E33 gated head
/// is 7 nodes and a lowered 4-layer chain is 5; the limit bounds the
/// node-table allocation and the per-node validation loop.
pub const MAX_NODES: usize = 1024;

/// Maximum arity of one graph node (only `Concat` takes more than two
/// inputs). Bounds the per-node input walk.
pub const MAX_NODE_INPUTS: usize = 64;

/// Maximum total multiply-adds per forward pass: the sum of
/// `in_dim * out_dim` over every Dense node (v4) or layer (v3, after
/// lowering). 2^24 — production zensim is 944 × 128 = 120,832.
///
/// File size does not bound compute on its own: sections may alias, so
/// a crafted file can point many nodes at one weight blob. This limit
/// does.
pub const MAX_TOTAL_WEIGHTS: usize = 1 << 24;

/// Maximum f32 elements of the liveness-packed scratch arena a
/// [`crate::Predictor`] allocates (2^22 = 16 MiB). Computed and checked
/// at load, before anything is allocated against it.
pub const MAX_SCRATCH_ELEMS: usize = 1 << 22;
