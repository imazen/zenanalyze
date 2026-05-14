//! Hidden-unit reorder for bake-time compression optimization.
//!
//! Sorts each interior dimension (between two adjacent layers) by
//! L2 norm of the upstream layer's output columns. Permutes the
//! upstream layer's output cols + biases AND the downstream layer's
//! input rows in lockstep — the forward pass is mathematically
//! identical to the un-permuted bake, but row-major LZ4 / zstd find
//! substantially longer zero runs when dead hidden units cluster at
//! one end of the matrix.
//!
//! Always-on, deterministic (L2-asc), zero wire-format support.
//! Worst case (fully-live matrix) is identity — no compression
//! regression.
//!
//! Measured on V0_18 (228 → 384 → 1, 74 % dead HUs after τ=0.005
//! zerobias):
//!
//! - Layer-0 LZ4 (no reorder): 32,628 bytes
//! - Layer-0 LZ4 + HU L2-asc reorder: 13,807 bytes (−57.7 %)
//! - Layer-0 zstd-22 (no reorder): 18,524 bytes
//! - Layer-0 zstd-22 + HU L2-asc reorder: 11,272 bytes (−39 %)
//!
//! The win scales with the dead-HU fraction. Fully-active matrices
//! see ~0 % delta; never a regression.
//!
//! See `benchmarks/reorder_lz4_zstd_eval_2026-05-13.md` in the zensim
//! repo for the full experiment grid.

use alloc::vec::Vec;

use zenpredict::{Activation, WeightDtype};

use crate::composer::BakeLayer;

/// Owned (heap-allocated) copy of a `BakeLayer`. The composer's
/// public API takes borrowed slices, but HU reorder needs to create
/// permuted versions of the weight arrays — those owned arrays live
/// here.
pub(crate) struct OwnedBakeLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Activation,
    pub dtype: WeightDtype,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

impl OwnedBakeLayer {
    fn from_borrowed(layer: &BakeLayer<'_>) -> Self {
        Self {
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
            activation: layer.activation,
            dtype: layer.dtype,
            weights: layer.weights.to_vec(),
            biases: layer.biases.to_vec(),
        }
    }

    pub(crate) fn as_borrowed(&self) -> BakeLayer<'_> {
        BakeLayer {
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            activation: self.activation,
            dtype: self.dtype,
            weights: &self.weights,
            biases: &self.biases,
        }
    }
}

/// Reorder hidden units in every interior dimension by L2 norm
/// ascending. See module docs for rationale.
///
/// Returns owned copies of every layer with permutations applied.
/// The first layer's inputs and the last layer's outputs are NOT
/// reordered — only interior (hidden) dimensions, since reordering
/// the user-facing input or output dims would break caller
/// expectations.
///
/// Single-layer bakes (`layers.len() == 1`) have no interior dims;
/// returns owned copies with the original ordering preserved (still
/// allocates — caller paid for the bake step regardless).
pub(crate) fn apply_hu_reorder(layers: &[BakeLayer<'_>]) -> Vec<OwnedBakeLayer> {
    let mut owned: Vec<OwnedBakeLayer> =
        layers.iter().map(OwnedBakeLayer::from_borrowed).collect();

    // Interior dim between layer i and layer i+1 has size
    // = layers[i].out_dim = layers[i+1].in_dim. Permute layer i's
    // output cols + biases AND layer i+1's input rows in lockstep.
    for i in 0..owned.len().saturating_sub(1) {
        let interior_dim = owned[i].out_dim;
        debug_assert_eq!(owned[i].out_dim, owned[i + 1].in_dim);

        // Compute L2-norm-squared per output column of layer i.
        // sqrt skipped — relative order is preserved by the square.
        let mut norms_sq: Vec<f32> = alloc::vec![0.0f32; interior_dim];
        let in_dim_i = owned[i].in_dim;
        for r in 0..in_dim_i {
            let row_start = r * interior_dim;
            for c in 0..interior_dim {
                let w = owned[i].weights[row_start + c];
                norms_sq[c] += w * w;
            }
        }

        // Permutation π[new] = old. Sort indices by norm ascending.
        // For ties (e.g., entirely dead cols), break by old index to
        // keep the sort deterministic across runs.
        let mut perm: Vec<u32> = (0..interior_dim as u32).collect();
        perm.sort_by(|&a, &b| {
            let na = norms_sq[a as usize];
            let nb = norms_sq[b as usize];
            na.partial_cmp(&nb).unwrap_or(core::cmp::Ordering::Equal).then(a.cmp(&b))
        });

        // Apply permutation to layer i: out cols + biases.
        // new[r, c_new] = old[r, perm[c_new]]
        let old_weights_i = owned[i].weights.clone();
        let old_biases_i = owned[i].biases.clone();
        for r in 0..in_dim_i {
            let row_start = r * interior_dim;
            for (c_new, &c_old) in perm.iter().enumerate() {
                owned[i].weights[row_start + c_new] =
                    old_weights_i[row_start + c_old as usize];
            }
        }
        for (c_new, &c_old) in perm.iter().enumerate() {
            owned[i].biases[c_new] = old_biases_i[c_old as usize];
        }

        // Apply permutation to layer i+1: input rows.
        // new[r_new, o] = old[perm[r_new], o]
        let out_dim_ip1 = owned[i + 1].out_dim;
        let old_weights_ip1 = owned[i + 1].weights.clone();
        for (r_new, &r_old) in perm.iter().enumerate() {
            let new_row_start = r_new * out_dim_ip1;
            let old_row_start = (r_old as usize) * out_dim_ip1;
            for o in 0..out_dim_ip1 {
                owned[i + 1].weights[new_row_start + o] =
                    old_weights_ip1[old_row_start + o];
            }
        }
    }

    owned
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpredict::{Activation, WeightDtype};

    /// Single layer — no interior dims. Output should match input
    /// exactly (the function still allocates owned copies but the
    /// data is preserved).
    #[test]
    fn single_layer_is_identity() {
        let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let biases = [0.1f32, 0.2, 0.3];
        let layers = [BakeLayer {
            in_dim: 2,
            out_dim: 3,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &weights,
            biases: &biases,
        }];
        let permuted = apply_hu_reorder(&layers);
        assert_eq!(permuted.len(), 1);
        assert_eq!(permuted[0].weights, weights);
        assert_eq!(permuted[0].biases, biases);
    }

    /// Two-layer net with one HU dead (L2 = 0). The dead HU should
    /// move to position 0 (L2-asc). Layer 1 row 0 should be the
    /// row that was at the dead HU's position.
    #[test]
    fn dead_hu_clusters_at_start() {
        // 2 → 3 → 1: layer 0 weights = [r0c0 r0c1 r0c2 | r1c0 r1c1 r1c2]
        // Make col 1 (HU 1) all zeros — it's dead.
        let w0 = [
            1.0f32, 0.0, 3.0,
            4.0,    0.0, 6.0,
        ];
        let b0 = [0.1f32, 0.99, 0.3];
        let w1 = [10.0f32, 11.0, 12.0]; // layer 1 rows: [r0o0, r1o0, r2o0]
        let b1 = [0.0f32];
        let layers = [
            BakeLayer {
                in_dim: 2,
                out_dim: 3,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w0,
                biases: &b0,
            },
            BakeLayer {
                in_dim: 3,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w1,
                biases: &b1,
            },
        ];
        let permuted = apply_hu_reorder(&layers);

        // Dead HU (originally col 1) should now be at position 0
        // (L2 = 0, smallest).
        // After permutation, col 0 in new layer 0 = col 1 of old (all zeros).
        let new_w0 = &permuted[0].weights;
        assert_eq!(new_w0[0], 0.0); // r0, new col 0 = old col 1 = 0
        assert_eq!(new_w0[3], 0.0); // r1, new col 0 = old col 1 = 0
        assert_eq!(permuted[0].biases[0], 0.99); // bias for dead col

        // Layer 1: row 0 should be the OLD row 1 (which corresponded
        // to the dead HU).
        assert_eq!(permuted[1].weights[0], 11.0);
    }

    /// Round-trip equivalence: forward pass on reordered bake must
    /// equal forward pass on original bake. This is the canonical
    /// correctness check — easy to break if you forget to permute
    /// biases or the downstream layer.
    #[test]
    fn forward_pass_is_invariant() {
        // 2 → 4 → 1 with deliberate variance per HU.
        let w0 = [
            1.5f32, 0.0, -2.3, 0.7,
            0.4,    0.0,  1.1, -0.9,
        ];
        let b0 = [0.5f32, 1.0, -0.2, 0.3];
        let w1 = [2.0f32, -1.5, 0.7, 3.3]; // 4 rows × 1 col
        let b1 = [0.1f32];
        let layers = [
            BakeLayer {
                in_dim: 2,
                out_dim: 4,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w0,
                biases: &b0,
            },
            BakeLayer {
                in_dim: 4,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w1,
                biases: &b1,
            },
        ];

        let permuted = apply_hu_reorder(&layers);

        // Manual forward pass for both versions; results must be
        // bit-equal.
        let input = [0.8f32, -1.2];
        let out_orig = forward_f32(&layers, &input);
        let permuted_borrowed: Vec<BakeLayer<'_>> =
            permuted.iter().map(OwnedBakeLayer::as_borrowed).collect();
        let out_permuted = forward_f32(&permuted_borrowed, &input);
        assert_eq!(out_orig, out_permuted);
    }

    /// Three-layer net round-trip — exercises the interior-dim loop
    /// twice and confirms both reorderings compose correctly.
    #[test]
    fn forward_pass_invariant_three_layers() {
        // 2 → 3 → 4 → 1
        let w0 = [
            0.5f32, -1.0, 2.5,
            1.2,     0.0, -0.3,
        ];
        let b0 = [0.1f32, 0.0, -0.5];
        let w1 = [
             1.0f32, 0.0, -1.5,  2.0,
            -0.5,    0.7,  0.0, -0.3,
             0.2,   -1.1,  0.8,  1.4,
        ];
        let b1 = [0.0f32, 0.6, -0.1, 0.4];
        let w2 = [1.0f32, -0.7, 0.3, 1.5];
        let b2 = [0.2f32];
        let layers = [
            BakeLayer {
                in_dim: 2,
                out_dim: 3,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w0,
                biases: &b0,
            },
            BakeLayer {
                in_dim: 3,
                out_dim: 4,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w1,
                biases: &b1,
            },
            BakeLayer {
                in_dim: 4,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w2,
                biases: &b2,
            },
        ];

        let permuted = apply_hu_reorder(&layers);
        let permuted_borrowed: Vec<BakeLayer<'_>> =
            permuted.iter().map(OwnedBakeLayer::as_borrowed).collect();

        // Try a few inputs to make sure both interior permutations
        // round-trip correctly.
        for input in [
            [1.0f32, 0.0],
            [-2.0, 3.0],
            [0.5, -0.5],
            [0.0, 0.0],
        ] {
            let out_orig = forward_f32(&layers, &input);
            let out_permuted = forward_f32(&permuted_borrowed, &input);
            assert_eq!(out_orig, out_permuted, "input={input:?}");
        }
    }

    fn forward_f32(layers: &[BakeLayer<'_>], input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for layer in layers {
            let mut next = alloc::vec![0.0f32; layer.out_dim];
            for o in 0..layer.out_dim {
                let mut acc = layer.biases[o];
                for (i, &x) in current.iter().enumerate() {
                    acc += x * layer.weights[i * layer.out_dim + o];
                }
                next[o] = match layer.activation {
                    Activation::Identity => acc,
                    Activation::Relu => acc.max(0.0),
                    Activation::LeakyRelu => {
                        if acc < 0.0 {
                            acc * 0.01
                        } else {
                            acc
                        }
                    }
                };
            }
            current = next;
        }
        current
    }
}
