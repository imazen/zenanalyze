//! Bake-time auto-optimizer: sweep candidate (feature_order,
//! output_order, compressed) combinations and pick the smallest
//! output.
//!
//! All candidates produce mathematically-identical predict outputs
//! (load-time inverse permutation + decompression are lossless), so
//! the optimizer is free to pick whichever is smallest with no quality
//! impact.
//!
//! ## Candidate matrix
//!
//! For input feature permutation (rows of layer 0):
//! - identity (no reorder)
//! - row L2 norm ascending — clusters low-magnitude rows together,
//!   helpful for row-aligned sparsity
//! - row L2 norm descending — symmetric, sometimes better when
//!   bake-internal layout interacts with section alignment
//!
//! For output permutation (cols of layer last):
//! - identity
//! - col L2 norm ascending
//! - col L2 norm descending
//!
//! For compression: try both compressed and uncompressed; the
//! smaller wins. For tiny bakes the LZ4 header overhead can exceed
//! the savings; for picker-sized bakes compression always wins.
//!
//! Total candidates: 3 × 3 × 2 = 18. Each bake() call is ~1 ms on a
//! V_X-shape bake; ~18 ms total — invisible at build time.
//!
//! ## What this does NOT do
//!
//! - Hidden-unit (interior dim) reorder. That's already done
//!   unconditionally inside `bake()` via the `hu_reorder` module —
//!   the auto-optimizer doesn't need to touch it.
//! - Hierarchical clustering or other expensive permutations. The
//!   agent's 2026-05-13 reorder eval found L2-asc to dominate
//!   clustering at lower cost; we ship L2-only and skip cluster.
//! - Multi-codec / multi-bake co-optimization. Each bake is optimized
//!   independently.

use alloc::vec::Vec;

use crate::composer::{BakeError, BakeRequest, bake};

/// Compute candidate row permutations for the first layer's input
/// dim. Returns `Vec<Option<Vec<u32>>>` — None = identity baseline,
/// Some(perm) = a candidate to try.
pub(crate) fn input_permutation_candidates(req: &BakeRequest<'_>) -> Vec<Option<Vec<u32>>> {
    let n_in = req.layers[0].in_dim;
    let out_dim = req.layers[0].out_dim;
    let weights = req.layers[0].weights;

    let mut candidates: Vec<Option<Vec<u32>>> = Vec::with_capacity(3);
    candidates.push(None); // identity baseline

    // Row L2-norm-squared: norm[r] = sum_o W[r, o]^2.
    let mut row_norms = alloc::vec![0.0f32; n_in];
    for r in 0..n_in {
        let mut acc = 0.0f32;
        for o in 0..out_dim {
            let w = weights[r * out_dim + o];
            acc += w * w;
        }
        row_norms[r] = acc;
    }

    // Ascending: perm[bake_pos] = caller_idx where caller_idx is the
    // row with bake_pos-th smallest norm.
    let mut asc: Vec<u32> = (0..n_in as u32).collect();
    asc.sort_by(|&a, &b| {
        row_norms[a as usize]
            .partial_cmp(&row_norms[b as usize])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    candidates.push(Some(asc.clone()));

    // Descending: reverse order.
    let mut desc = asc;
    desc.reverse();
    candidates.push(Some(desc));

    candidates
}

/// Compute candidate column permutations for the last layer's output
/// dim. None = identity; Some = candidate.
pub(crate) fn output_permutation_candidates(req: &BakeRequest<'_>) -> Vec<Option<Vec<u32>>> {
    let last = req.layers.last().expect("at least one layer");
    let n_out = last.out_dim;
    let in_dim = last.in_dim;
    let weights = last.weights;

    let mut candidates: Vec<Option<Vec<u32>>> = Vec::with_capacity(3);
    candidates.push(None); // identity baseline

    // Skip output reorder when n_out == 1 (trivial — single output
    // gives nothing to reorder). Saves 2/3 of the candidate matrix
    // for scorer-shaped bakes like zensim.
    if n_out <= 1 {
        return candidates;
    }

    // Col L2-norm-squared: norm[c] = sum_r W[r, c]^2.
    let mut col_norms = alloc::vec![0.0f32; n_out];
    for r in 0..in_dim {
        for c in 0..n_out {
            let w = weights[r * n_out + c];
            col_norms[c] += w * w;
        }
    }

    let mut asc: Vec<u32> = (0..n_out as u32).collect();
    asc.sort_by(|&a, &b| {
        col_norms[a as usize]
            .partial_cmp(&col_norms[b as usize])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    candidates.push(Some(asc.clone()));

    let mut desc = asc;
    desc.reverse();
    candidates.push(Some(desc));

    candidates
}

/// Bake the request multiple times across candidate
/// `(feature_order, output_order, compressed)` combinations and
/// return the smallest output.
///
/// All candidates yield mathematically-identical predict outputs.
/// The compositor's HU reorder (interior dims) is always-on
/// regardless of which candidate is chosen.
///
/// ## Behaviour
///
/// 1. Generate input-permutation candidates from layer-0 row L2 norms.
/// 2. Generate output-permutation candidates from last-layer col L2
///    norms (skipped when `n_outputs == 1`).
/// 3. For each (input_perm, output_perm, compressed) combination —
///    18 candidates max, 6 for single-output bakes — call `bake()`
///    and measure output length.
/// 4. Return the smallest output. Ties broken by preferring identity
///    permutations (smaller decode pipeline at load time).
///
/// ## Cost
///
/// `bake()` takes ~1 ms on a V_X-shape model; total optimizer cost
/// is ~6–18 ms. Negligible at build time, never run at deployment.
///
/// ## When to call
///
/// Use as a drop-in replacement for `bake()` whenever you're emitting
/// a ship bake. Skip when iterating during development (compose +
/// inspect cycles) — `bake()` directly is fine and emits the
/// HU-reordered bake without the candidate sweep.
pub fn bake_optimized(req: &BakeRequest<'_>) -> Result<Vec<u8>, BakeError> {
    let f_candidates = input_permutation_candidates(req);
    let o_candidates = output_permutation_candidates(req);

    let mut best: Option<Vec<u8>> = None;
    let mut best_score: Option<(usize, u8)> = None; // (bytes, identity_pref)

    for f_perm in &f_candidates {
        for o_perm in &o_candidates {
            for compress in [true, false] {
                // Skip permutation in BakeRequest by setting it only
                // when Some.
                let mut local = clone_bake_request(req);
                local.feature_order = f_perm.as_deref();
                local.output_order = o_perm.as_deref();
                local.compressed = compress;
                let bytes = bake(&local)?;
                let identity_pref = u8::from(f_perm.is_none())
                    + u8::from(o_perm.is_none());
                let score = (bytes.len(), 255 - identity_pref * 64);
                if best_score.map_or(true, |b| score < b) {
                    best_score = Some(score);
                    best = Some(bytes);
                }
            }
        }
    }

    Ok(best.expect("at least one candidate always succeeds"))
}

/// Shallow-copy a `BakeRequest`, preserving all `&[T]` borrows.
fn clone_bake_request<'a>(req: &BakeRequest<'a>) -> BakeRequest<'a> {
    BakeRequest {
        schema_hash: req.schema_hash,
        flags: req.flags,
        scaler_mean: req.scaler_mean,
        scaler_scale: req.scaler_scale,
        layers: req.layers,
        feature_bounds: req.feature_bounds,
        metadata: req.metadata,
        output_specs: req.output_specs,
        discrete_sets: req.discrete_sets,
        sparse_overrides: req.sparse_overrides,
        feature_order: req.feature_order,
        output_order: req.output_order,
        compressed: req.compressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composer::BakeLayer;
    use zenpredict::{Activation, WeightDtype};

    #[test]
    fn optimizer_picks_smallest() {
        // Tiny synthetic 8→8→1 model. The optimizer should explore
        // candidates and never produce something larger than the
        // baseline identity-no-compression bake.
        let weights_0 = vec![0.5f32; 64];
        let biases_0 = vec![0.1f32; 8];
        let weights_1 = vec![0.3f32; 8];
        let biases_1 = vec![0.05f32];
        let layers = vec![
            BakeLayer {
                in_dim: 8,
                out_dim: 8,
                activation: Activation::Relu,
                dtype: WeightDtype::F32,
                weights: &weights_0,
                biases: &biases_0,
            },
            BakeLayer {
                in_dim: 8,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &weights_1,
                biases: &biases_1,
            },
        ];
        let scaler_mean = vec![0.0f32; 8];
        let scaler_scale = vec![1.0f32; 8];
        let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);
        let baseline = bake(&req).unwrap();
        let optimized = bake_optimized(&req).unwrap();
        assert!(
            optimized.len() <= baseline.len(),
            "optimizer made it bigger? baseline={} optimized={}",
            baseline.len(),
            optimized.len()
        );
    }

    #[test]
    fn optimizer_preserves_predict_output() {
        use zenpredict::{Model, Predictor};
        let weights_0: Vec<f32> = (0..64).map(|i| (i as f32 * 0.123).sin() * 0.4).collect();
        let biases_0 = vec![0.0f32; 8];
        let weights_1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.789).cos() * 0.3).collect();
        let biases_1 = vec![0.0f32];
        let layers = vec![
            BakeLayer {
                in_dim: 8,
                out_dim: 8,
                activation: Activation::Relu,
                dtype: WeightDtype::F32,
                weights: &weights_0,
                biases: &biases_0,
            },
            BakeLayer {
                in_dim: 8,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &weights_1,
                biases: &biases_1,
            },
        ];
        let scaler_mean = vec![0.0f32; 8];
        let scaler_scale = vec![1.0f32; 8];
        let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);

        let baseline = bake(&req).unwrap();
        let optimized = bake_optimized(&req).unwrap();

        #[repr(C, align(16))]
        struct Aligned(Vec<u8>);
        let m_base = Model::from_bytes(&Aligned(baseline).0).unwrap();
        let m_opt = Model::from_bytes(&Aligned(optimized).0).unwrap();
        let mut p_base = Predictor::new(&m_base);
        let mut p_opt = Predictor::new(&m_opt);
        let features = [0.5f32, -0.3, 0.1, 0.8, -0.2, 0.6, -0.5, 0.0];
        let out_base = p_base.predict(&features).unwrap().to_vec();
        let out_opt = p_opt.predict(&features).unwrap().to_vec();
        assert_eq!(out_base, out_opt);
    }
}
