//! Bake-time auto-optimizer: sweep candidate (feature_order,
//! output_order, compressed) combinations and pick the smallest
//! output. Optionally run a bounded swap-based local search from the
//! best heuristic candidate.
//!
//! All candidates produce mathematically-identical predict outputs
//! (load-time inverse permutation + decompression are lossless), so
//! the optimizer is free to pick whichever is smallest with zero
//! quality impact.
//!
//! ## Why a richer search than "L2-asc / L2-desc / identity"
//!
//! LZ4 finds 4-byte matches within a 64 KB sliding window. The
//! compressibility of a row-major weight tensor depends on whether
//! adjacent rows share byte patterns at the same column offsets —
//! particularly zero positions and small-magnitude i8 buckets after
//! zerobias. L2-norm sorting is one proxy for this (low-energy rows
//! have more zeros) but it's not the only one. Useful complements:
//!
//! - **Sort by zero count**: explicitly group rows by how many
//!   zero-bytes they have after i8 requant. Rows with the same zero
//!   density cluster together; LZ4 finds long matches between them.
//! - **Sort by sign distribution**: rows whose nonzero weights are
//!   predominantly positive vs negative byte-encode differently;
//!   grouping by sign ratio tightens i8 byte ranges within a cluster.
//! - **Greedy nearest-neighbor**: start from any row, then iteratively
//!   append the row most similar (by Hamming distance on the i8
//!   quantization sign+zero pattern). Produces a chain where each row
//!   is its closest neighbor's closest neighbor — strong locality.
//! - **Bounded pairwise-swap hill climb**: from the best heuristic,
//!   propose random swaps; keep if they shrink, revert otherwise.
//!   Hours of compute would do a real search; ~50 ms gives a
//!   noticeable improvement on shapes where the heuristics are
//!   suboptimal.
//!
//! ## Cost
//!
//! `bake()` takes ~1 ms on a V_X-shape model; the optimizer's default
//! budget is ~50 candidates plus 100 swap evaluations = ~150 ms.
//! Invisible at build time, never run at deployment.
//!
//! ## What this does NOT do
//!
//! - Hidden-unit (interior dim) reorder. That's unconditional in
//!   `bake()` via the `hu_reorder` module.
//! - Exhaustive search. Even 228! is astronomical; we settle for
//!   "best of a representative set + a few hundred hill-climb steps."
//! - Cross-bake / co-optimization. Each bake is optimized
//!   independently.

extern crate alloc;
use alloc::vec::Vec;

use crate::composer::{BakeError, BakeRequest, bake};

/// Tunable knobs for [`bake_optimized_with`].
#[derive(Clone, Copy, Debug)]
pub struct OptimizeConfig {
    /// Whether to evaluate input-feature reorder candidates.
    /// Default true. Set false if you know the bake's inputs are
    /// already in a meaningful caller-natural order and shouldn't
    /// be permuted (rare).
    pub explore_input_order: bool,
    /// Whether to evaluate output reorder candidates.
    /// Default true. Skipped automatically when `n_outputs == 1`.
    pub explore_output_order: bool,
    /// Maximum number of pairwise-swap iterations after the best
    /// heuristic is selected. Each iteration evaluates a small batch
    /// of random swap proposals. Set to 0 to disable local search.
    /// Default 600 — enough to comfortably fill a 1 s budget on
    /// V_X-shape models (~1 ms per bake), giving the swap optimizer
    /// real coverage of the n=228 search space.
    pub local_search_iters: usize,
    /// Random-seed for the local-search swap proposals. Fixed seed
    /// makes the optimizer's output deterministic across runs.
    /// Default 0xDEADBEEF.
    pub local_search_seed: u64,
    /// Cap on total bake() invocations the optimizer is allowed to
    /// make, including heuristics + local search. Hard ceiling that
    /// shadows `local_search_iters` when needed. Default 2048 — at
    /// ~1 ms per bake, that's ~2 s wall, but the local-search
    /// budget itself defaults to ~600 iters so a typical V_X bake
    /// completes in ~700 ms with room to spare.
    pub max_evaluations: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            explore_input_order: true,
            explore_output_order: true,
            // ~600 swap evals = ~600 ms; combined with ~10-30
            // heuristic evals the optimizer pushes about 1 s total
            // on V_X-shape bakes. Matches the user-stated bake-time
            // budget rule: "at least a full second to optimize".
            local_search_iters: 600,
            local_search_seed: 0xDEAD_BEEF,
            max_evaluations: 2048,
        }
    }
}

/// Drop-in replacement for [`bake`] that auto-explores permutation
/// + compression candidates and returns the smallest output.
///
/// Uses [`OptimizeConfig::default`]. For tighter control over the
/// search budget, use [`bake_optimized_with`].
pub fn bake_optimized(req: &BakeRequest<'_>) -> Result<Vec<u8>, BakeError> {
    bake_optimized_with(req, &OptimizeConfig::default())
}

/// Like [`bake_optimized`] but with caller-provided search budget.
pub fn bake_optimized_with(
    req: &BakeRequest<'_>,
    cfg: &OptimizeConfig,
) -> Result<Vec<u8>, BakeError> {
    let n_in = req.layers[0].in_dim;
    let n_out = req.layers.last().expect("nonempty layers").out_dim;

    // ── Generate heuristic candidates. ──
    let input_candidates: Vec<Option<Vec<u32>>> = if cfg.explore_input_order {
        input_heuristics(req)
    } else {
        alloc::vec![None]
    };
    let output_candidates: Vec<Option<Vec<u32>>> = if cfg.explore_output_order && n_out > 1 {
        output_heuristics(req)
    } else {
        alloc::vec![None]
    };

    // ── Sweep the heuristic matrix × {compressed, uncompressed}. ──
    let mut evaluations: usize = 0;
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_score: Option<(usize, u8)> = None;
    let mut best_input: Option<Vec<u32>> = None;
    let mut best_output: Option<Vec<u32>> = None;
    let mut best_compressed: bool = false;

    for f_perm in &input_candidates {
        for o_perm in &output_candidates {
            for compress in [true, false] {
                if evaluations >= cfg.max_evaluations {
                    break;
                }
                let mut local = clone_request(req);
                local.feature_order = f_perm.as_deref();
                local.output_order = o_perm.as_deref();
                local.compressed = compress;
                let bytes = bake(&local)?;
                evaluations += 1;
                let identity_pref =
                    u8::from(f_perm.is_none()) + u8::from(o_perm.is_none());
                let score = (bytes.len(), 255 - identity_pref * 64);
                if best_score.map_or(true, |b| score < b) {
                    best_score = Some(score);
                    best_bytes = Some(bytes);
                    best_input = f_perm.clone();
                    best_output = o_perm.clone();
                    best_compressed = compress;
                }
            }
        }
    }

    // ── Local search: pairwise swaps on the best heuristic. ──
    if cfg.local_search_iters > 0 && evaluations < cfg.max_evaluations {
        let mut rng = SmallRng::new(cfg.local_search_seed);

        // Try improving the INPUT permutation (rows of layer 0).
        if cfg.explore_input_order && n_in > 1 {
            let mut perm = best_input
                .clone()
                .unwrap_or_else(|| identity_perm(n_in));
            let mut current_size = best_bytes.as_ref().unwrap().len();
            let iters = cfg.local_search_iters.min(cfg.max_evaluations - evaluations);
            for _ in 0..iters {
                if evaluations >= cfg.max_evaluations {
                    break;
                }
                let i = (rng.next_u32() as usize) % n_in;
                let j = (rng.next_u32() as usize) % n_in;
                if i == j {
                    continue;
                }
                perm.swap(i, j);
                let mut local = clone_request(req);
                local.feature_order = Some(&perm);
                local.output_order = best_output.as_deref();
                local.compressed = best_compressed;
                let bytes = bake(&local)?;
                evaluations += 1;
                if bytes.len() < current_size {
                    current_size = bytes.len();
                    best_bytes = Some(bytes);
                    best_input = Some(perm.clone());
                } else {
                    perm.swap(i, j); // revert
                }
            }
        }

        // Try improving the OUTPUT permutation (cols of last layer).
        if cfg.explore_output_order && n_out > 1 && evaluations < cfg.max_evaluations {
            let mut perm = best_output
                .clone()
                .unwrap_or_else(|| identity_perm(n_out));
            let mut current_size = best_bytes.as_ref().unwrap().len();
            let iters = cfg.local_search_iters.min(cfg.max_evaluations - evaluations);
            for _ in 0..iters {
                if evaluations >= cfg.max_evaluations {
                    break;
                }
                let i = (rng.next_u32() as usize) % n_out;
                let j = (rng.next_u32() as usize) % n_out;
                if i == j {
                    continue;
                }
                perm.swap(i, j);
                let mut local = clone_request(req);
                local.feature_order = best_input.as_deref();
                local.output_order = Some(&perm);
                local.compressed = best_compressed;
                let bytes = bake(&local)?;
                evaluations += 1;
                if bytes.len() < current_size {
                    current_size = bytes.len();
                    best_bytes = Some(bytes);
                    best_output = Some(perm.clone());
                } else {
                    perm.swap(i, j); // revert
                }
            }
        }

        // Final pass: try toggling compression with the best perms.
        // The optimal compression flag can shift after local search
        // when the post-permutation byte structure changes.
        if evaluations < cfg.max_evaluations {
            for compress in [true, false] {
                if compress == best_compressed {
                    continue;
                }
                let mut local = clone_request(req);
                local.feature_order = best_input.as_deref();
                local.output_order = best_output.as_deref();
                local.compressed = compress;
                let bytes = bake(&local)?;
                // Final pass — no further evaluations after this
                // block, so we don't increment the counter.
                if bytes.len() < best_bytes.as_ref().unwrap().len() {
                    best_bytes = Some(bytes);
                    best_compressed = compress;
                }
            }
        }
    }

    Ok(best_bytes.expect("at least one candidate always succeeds"))
}

// ── Heuristic candidate generators. ───────────────────────────────

fn input_heuristics(req: &BakeRequest<'_>) -> Vec<Option<Vec<u32>>> {
    let n_in = req.layers[0].in_dim;
    let out_dim = req.layers[0].out_dim;
    let weights = req.layers[0].weights;

    let mut out: Vec<Option<Vec<u32>>> = Vec::with_capacity(8);
    out.push(None); // identity

    // L2-asc / L2-desc.
    let row_norms_sq = row_norms_sq(weights, n_in, out_dim);
    let mut asc: Vec<u32> = (0..n_in as u32).collect();
    asc.sort_by(|&a, &b| {
        row_norms_sq[a as usize]
            .partial_cmp(&row_norms_sq[b as usize])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let desc: Vec<u32> = asc.iter().rev().copied().collect();
    out.push(Some(asc.clone()));
    out.push(Some(desc));

    // Zero-count clustering on the i8 estimate. Rows are quantized
    // mentally: count weights with |w| < layer_max * 0.005 (the
    // zerobias threshold). Sort by zero-count ascending.
    let layer_max = weights.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    let zerobias_cut = layer_max * 0.005;
    let mut zero_counts: Vec<(u32, u32)> = (0..n_in as u32)
        .map(|r| {
            let mut zc: u32 = 0;
            for o in 0..out_dim {
                let w = weights[r as usize * out_dim + o];
                if w.abs() < zerobias_cut {
                    zc += 1;
                }
            }
            (r, zc)
        })
        .collect();
    zero_counts.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    out.push(Some(zero_counts.iter().map(|p| p.0).collect()));
    out.push(Some(zero_counts.iter().rev().map(|p| p.0).collect()));

    // Sign-distribution sort: positive_count - negative_count.
    let mut sign_balance: Vec<(u32, i32)> = (0..n_in as u32)
        .map(|r| {
            let mut bal: i32 = 0;
            for o in 0..out_dim {
                let w = weights[r as usize * out_dim + o];
                if w > 0.0 {
                    bal += 1;
                } else if w < 0.0 {
                    bal -= 1;
                }
            }
            (r, bal)
        })
        .collect();
    sign_balance.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    out.push(Some(sign_balance.iter().map(|p| p.0).collect()));

    // Greedy nearest-neighbour chain on Hamming distance of the
    // sign+zero pattern. Costly for large n; cap n_in to avoid
    // worst-case quadratic blowup on multi-thousand-input bakes.
    if n_in <= 1024 {
        out.push(Some(greedy_nn_chain(weights, n_in, out_dim, zerobias_cut)));
    }

    out
}

fn output_heuristics(req: &BakeRequest<'_>) -> Vec<Option<Vec<u32>>> {
    let last = req.layers.last().expect("nonempty layers");
    let n_out = last.out_dim;
    let in_dim = last.in_dim;
    let weights = last.weights;

    let mut out: Vec<Option<Vec<u32>>> = Vec::with_capacity(4);
    out.push(None);

    let col_norms_sq = col_norms_sq(weights, in_dim, n_out);
    let mut asc: Vec<u32> = (0..n_out as u32).collect();
    asc.sort_by(|&a, &b| {
        col_norms_sq[a as usize]
            .partial_cmp(&col_norms_sq[b as usize])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let desc: Vec<u32> = asc.iter().rev().copied().collect();
    out.push(Some(asc.clone()));
    out.push(Some(desc));

    // Greedy nearest-neighbour on output cols (transposed view).
    if n_out <= 1024 {
        let layer_max = weights.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
        let zerobias_cut = layer_max * 0.005;
        out.push(Some(greedy_nn_chain_cols(
            weights, in_dim, n_out, zerobias_cut,
        )));
    }

    out
}

fn row_norms_sq(weights: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut out = alloc::vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let mut s = 0.0f32;
        for c in 0..n_cols {
            let w = weights[r * n_cols + c];
            s += w * w;
        }
        out[r] = s;
    }
    out
}

fn col_norms_sq(weights: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut out = alloc::vec![0.0f32; n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let w = weights[r * n_cols + c];
            out[c] += w * w;
        }
    }
    out
}

/// Greedy nearest-neighbour chain by Hamming distance on the
/// per-row "is this byte (likely) zero?" bitmap. Starts from the
/// row with the most zeros (highest sparsity). Each step picks the
/// remaining row closest to the previous one. Produces a chain
/// where adjacent rows share zero positions — good LZ4 locality.
fn greedy_nn_chain(weights: &[f32], n_rows: usize, n_cols: usize, cut: f32) -> Vec<u32> {
    // Build zero-pattern bitmap (one bit per col, packed into u64s).
    let words_per_row = n_cols.div_ceil(64);
    let mut bitmap: Vec<u64> = alloc::vec![0u64; n_rows * words_per_row];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let w = weights[r * n_cols + c];
            if w.abs() < cut {
                bitmap[r * words_per_row + c / 64] |= 1u64 << (c % 64);
            }
        }
    }
    // Count zeros per row; start from the row with the most.
    let pop = |row: usize| -> u32 {
        let s = row * words_per_row;
        bitmap[s..s + words_per_row]
            .iter()
            .map(|w| w.count_ones())
            .sum()
    };
    let mut visited = alloc::vec![false; n_rows];
    let mut chain: Vec<u32> = Vec::with_capacity(n_rows);
    let start = (0..n_rows).max_by_key(|&r| pop(r)).unwrap_or(0);
    chain.push(start as u32);
    visited[start] = true;
    let mut cur = start;
    for _ in 1..n_rows {
        // Pick the unvisited row with minimum Hamming distance to cur.
        let cur_start = cur * words_per_row;
        let mut best: usize = 0;
        let mut best_dist: u32 = u32::MAX;
        for r in 0..n_rows {
            if visited[r] {
                continue;
            }
            let r_start = r * words_per_row;
            let mut d: u32 = 0;
            for w in 0..words_per_row {
                d += (bitmap[cur_start + w] ^ bitmap[r_start + w]).count_ones();
            }
            if d < best_dist {
                best_dist = d;
                best = r;
            }
        }
        chain.push(best as u32);
        visited[best] = true;
        cur = best;
    }
    chain
}

fn greedy_nn_chain_cols(weights: &[f32], n_rows: usize, n_cols: usize, cut: f32) -> Vec<u32> {
    // Same as greedy_nn_chain but for columns of a row-major matrix.
    // Build bitmap per column (n_rows bits per column, packed).
    let words_per_col = n_rows.div_ceil(64);
    let mut bitmap: Vec<u64> = alloc::vec![0u64; n_cols * words_per_col];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let w = weights[r * n_cols + c];
            if w.abs() < cut {
                bitmap[c * words_per_col + r / 64] |= 1u64 << (r % 64);
            }
        }
    }
    let pop = |col: usize| -> u32 {
        let s = col * words_per_col;
        bitmap[s..s + words_per_col]
            .iter()
            .map(|w| w.count_ones())
            .sum()
    };
    let mut visited = alloc::vec![false; n_cols];
    let mut chain: Vec<u32> = Vec::with_capacity(n_cols);
    let start = (0..n_cols).max_by_key(|&c| pop(c)).unwrap_or(0);
    chain.push(start as u32);
    visited[start] = true;
    let mut cur = start;
    for _ in 1..n_cols {
        let cur_start = cur * words_per_col;
        let mut best: usize = 0;
        let mut best_dist: u32 = u32::MAX;
        for c in 0..n_cols {
            if visited[c] {
                continue;
            }
            let c_start = c * words_per_col;
            let mut d: u32 = 0;
            for w in 0..words_per_col {
                d += (bitmap[cur_start + w] ^ bitmap[c_start + w]).count_ones();
            }
            if d < best_dist {
                best_dist = d;
                best = c;
            }
        }
        chain.push(best as u32);
        visited[best] = true;
        cur = best;
    }
    chain
}

fn identity_perm(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

fn clone_request<'a>(req: &BakeRequest<'a>) -> BakeRequest<'a> {
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

/// xoshiro256** small-state PRNG (deterministic, no_std-friendly).
/// Used by the local-search swap proposer.
struct SmallRng {
    state: [u64; 4],
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        // Use splitmix64 to expand seed into 4 state words.
        let mut sm = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self {
            state: [next(), next(), next(), next()],
        }
    }
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composer::BakeLayer;
    use zenpredict::{Activation, WeightDtype};

    #[test]
    fn optimizer_picks_smallest_or_equal_to_baseline() {
        let weights_0 = alloc::vec![0.5f32; 64];
        let biases_0 = alloc::vec![0.1f32; 8];
        let weights_1 = alloc::vec![0.3f32; 8];
        let biases_1 = alloc::vec![0.05f32];
        let layers = alloc::vec![
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
        let scaler_mean = alloc::vec![0.0f32; 8];
        let scaler_scale = alloc::vec![1.0f32; 8];
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
        let weights_0: alloc::vec::Vec<f32> =
            (0..64).map(|i| (i as f32 * 0.123).sin() * 0.4).collect();
        let biases_0 = alloc::vec![0.0f32; 8];
        let weights_1: alloc::vec::Vec<f32> =
            (0..8).map(|i| (i as f32 * 0.789).cos() * 0.3).collect();
        let biases_1 = alloc::vec![0.0f32];
        let layers = alloc::vec![
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
        let scaler_mean = alloc::vec![0.0f32; 8];
        let scaler_scale = alloc::vec![1.0f32; 8];
        let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);
        let baseline = bake(&req).unwrap();
        let optimized = bake_optimized(&req).unwrap();
        #[repr(C, align(16))]
        struct Aligned(alloc::vec::Vec<u8>);
        let m_base = Model::from_bytes(&Aligned(baseline).0).unwrap();
        let m_opt = Model::from_bytes(&Aligned(optimized).0).unwrap();
        let mut p_base = Predictor::new(&m_base);
        let mut p_opt = Predictor::new(&m_opt);
        let features = [0.5f32, -0.3, 0.1, 0.8, -0.2, 0.6, -0.5, 0.0];
        let out_base = p_base.predict(&features).unwrap().to_vec();
        let out_opt = p_opt.predict(&features).unwrap().to_vec();
        assert_eq!(out_base, out_opt);
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let weights_0: alloc::vec::Vec<f32> =
            (0..64).map(|i| (i as f32 * 0.31).sin()).collect();
        let biases_0 = alloc::vec![0.0f32; 8];
        let weights_1: alloc::vec::Vec<f32> =
            (0..16).map(|i| (i as f32 * 0.27).cos()).collect();
        let biases_1 = alloc::vec![0.0f32; 2];
        let layers = alloc::vec![
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
                out_dim: 2,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &weights_1,
                biases: &biases_1,
            },
        ];
        let scaler_mean = alloc::vec![0.0f32; 8];
        let scaler_scale = alloc::vec![1.0f32; 8];
        let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);
        // Two runs at the same seed must produce byte-equal output.
        let cfg = OptimizeConfig::default();
        let a = bake_optimized_with(&req, &cfg).unwrap();
        let b = bake_optimized_with(&req, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn budget_zero_skips_local_search() {
        let weights_0 = alloc::vec![0.5f32; 64];
        let biases_0 = alloc::vec![0.1f32; 8];
        let weights_1 = alloc::vec![0.3f32; 8];
        let biases_1 = alloc::vec![0.05f32];
        let layers = alloc::vec![
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
        let scaler_mean = alloc::vec![0.0f32; 8];
        let scaler_scale = alloc::vec![1.0f32; 8];
        let req = BakeRequest::new(0, 0, &scaler_mean, &scaler_scale, &layers);
        let mut cfg = OptimizeConfig::default();
        cfg.local_search_iters = 0;
        let _ = bake_optimized_with(&req, &cfg).unwrap();
    }
}
