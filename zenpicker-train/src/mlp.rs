//! A minimal multi-output MLP trainer — the Rust port of zentrain's
//! `_train_torch_leakyrelu_student` (in `zentrain/tools/train_hybrid.py`).
//!
//! Topology + schedule mirror the Python student so the Rust port is
//! apples-to-apples comparable:
//!
//! - `hidden_layer_sizes` Linear layers, each followed by
//!   `LeakyReLU(negative_slope = 0.01)`, then a final Linear to `n_out`.
//! - He/Kaiming-uniform init (the PyTorch `nn.Linear` default).
//! - Adam optimizer, MSE loss over the flat output vector.
//! - Internal row-shuffled validation slice (`val_frac`, default 0.1)
//!   for early stopping; `n_iter_no_change` patience, `tol`.
//!
//! NaN targets (unreachable cells in the picker formulation) are masked
//! out of the loss per-element — they contribute neither gradient nor
//! denominator. This is the Rust analogue of zentrain feeding NaN-free
//! teacher rows; we keep NaN in the matrix and mask, which is simpler
//! and matches the picker's "reach[cell]" semantics exactly.
//!
//! Deterministic given a seed: the RNG (init + shuffles + the internal
//! val split) is a seeded SplitMix64/xoshiro pair, so a fixed seed
//! reproduces the same fit.

// The matmul / backprop / Adam loops index several parallel flat arrays
// (`w`, `dst`, `delta`, `gw[li]`, …) by the same counter; the
// range-loop form is clearer here than zipped iterators over the
// multiple slices, and these are the numeric hot loops.
#![allow(clippy::needless_range_loop)]

/// Hyperparameters for one MLP fit. Field defaults mirror the zentrain
/// student (`lr = 2e-3`, `batch = 512`, `max_iter = 500`,
/// `n_iter_no_change = 30`, `tol = 1e-6`, `val_frac = 0.1`,
/// `leaky_slope = 0.01`).
#[derive(Debug, Clone)]
pub struct MlpConfig {
    /// Hidden layer widths, e.g. `vec![128, 128]`.
    pub hidden: Vec<usize>,
    pub lr: f64,
    pub batch_size: usize,
    pub max_iter: usize,
    pub n_iter_no_change: usize,
    pub tol: f64,
    pub val_frac: f64,
    pub leaky_slope: f64,
    pub seed: u64,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden: vec![128, 128],
            lr: 2e-3,
            batch_size: 512,
            max_iter: 500,
            n_iter_no_change: 30,
            tol: 1e-6,
            val_frac: 0.1,
            leaky_slope: 0.01,
            seed: 0x5eed_1234_u64,
        }
    }
}

/// Tiny seeded RNG (SplitMix64) — deterministic, no external dep, fine
/// for weight init + shuffles.
#[derive(Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform f64 in `[-a, a)`.
    fn uniform(&mut self, a: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * a
    }
    /// Fisher–Yates shuffle.
    fn shuffle(&mut self, v: &mut [usize]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// One dense layer: `out = W·x + b`, `W` is `out_dim × in_dim`
/// (row-major), bias length `out_dim`.
#[derive(Clone)]
struct Linear {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f64>,
    b: Vec<f64>,
    // Adam moments.
    mw: Vec<f64>,
    vw: Vec<f64>,
    mb: Vec<f64>,
    vb: Vec<f64>,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, rng: &mut SplitMix64) -> Self {
        // PyTorch nn.Linear default init: uniform(-1/sqrt(in), 1/sqrt(in)).
        let bound = 1.0 / (in_dim as f64).sqrt();
        let mut w = vec![0.0f64; out_dim * in_dim];
        for x in &mut w {
            *x = rng.uniform(bound);
        }
        let mut b = vec![0.0f64; out_dim];
        for x in &mut b {
            *x = rng.uniform(bound);
        }
        Self {
            in_dim,
            out_dim,
            w,
            b,
            mw: vec![0.0; out_dim * in_dim],
            vw: vec![0.0; out_dim * in_dim],
            mb: vec![0.0; out_dim],
            vb: vec![0.0; out_dim],
        }
    }

    /// Forward one sample; `x` len `in_dim`, writes `out` len `out_dim`.
    fn forward(&self, x: &[f64], out: &mut [f64]) {
        for o in 0..self.out_dim {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut acc = self.b[o];
            for i in 0..self.in_dim {
                acc += row[i] * x[i];
            }
            out[o] = acc;
        }
    }
}

/// A fitted MLP. Weights are natural-unit (no folded scaler — the
/// caller folds the input standardizer into the bake's scaler).
#[derive(Clone)]
pub struct Mlp {
    layers: Vec<Linear>,
    leaky_slope: f64,
    pub n_in: usize,
    pub n_out: usize,
    /// Final internal-validation MSE (for the search to rank on).
    pub best_val_loss: f64,
    pub n_iter: usize,
}

impl Mlp {
    /// Forward one sample, returning the `n_out` outputs. Hidden layers
    /// get LeakyReLU; the final layer is linear (identity).
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let mut cur = x.to_vec();
        let n_layers = self.layers.len();
        for (li, layer) in self.layers.iter().enumerate() {
            let mut out = vec![0.0f64; layer.out_dim];
            layer.forward(&cur, &mut out);
            if li + 1 < n_layers {
                for v in &mut out {
                    if *v < 0.0 {
                        *v *= self.leaky_slope;
                    }
                }
            }
            cur = out;
        }
        cur
    }

    /// The fitted weight/bias arrays per layer, as
    /// `(in_dim, out_dim, weights, biases[out])`.
    ///
    /// zenpredict's forward pass indexes weights as
    /// `w[in_idx * out_dim + out_idx]` (input-major, i.e. `in_dim ×
    /// out_dim` row-major). Our internal `Linear` stores `out_dim ×
    /// in_dim` (`w[out*in_dim + in]`), so we TRANSPOSE here. Layer `i`
    /// activation is LeakyReLU for all but the last (identity).
    pub fn layers_for_bake(&self) -> Vec<(usize, usize, Vec<f32>, Vec<f32>)> {
        self.layers
            .iter()
            .map(|l| {
                let mut w = vec![0.0f32; l.in_dim * l.out_dim];
                for o in 0..l.out_dim {
                    for i in 0..l.in_dim {
                        w[i * l.out_dim + o] = l.w[o * l.in_dim + i] as f32;
                    }
                }
                (
                    l.in_dim,
                    l.out_dim,
                    w,
                    l.b.iter().map(|&x| x as f32).collect(),
                )
            })
            .collect()
    }
}

/// Train an MLP on standardized inputs `x_std` (row-major
/// `n × n_in`) to multi-output targets `y` (row-major `n × n_out`).
/// `y` may contain NaN for "no target" cells (unreachable in the picker
/// formulation); those elements are masked out of the loss/gradient.
///
/// Returns the fitted [`Mlp`] (best-val-loss snapshot restored).
pub fn train_mlp(
    x_std: &[f64],
    y: &[f64],
    n: usize,
    n_in: usize,
    n_out: usize,
    cfg: &MlpConfig,
) -> Mlp {
    let mut rng = SplitMix64::new(cfg.seed);

    // Build layers: n_in -> h0 -> h1 -> ... -> n_out.
    let mut dims: Vec<usize> = Vec::with_capacity(cfg.hidden.len() + 2);
    dims.push(n_in);
    dims.extend(cfg.hidden.iter().copied());
    dims.push(n_out);
    let mut layers: Vec<Linear> = Vec::with_capacity(dims.len() - 1);
    for w in dims.windows(2) {
        layers.push(Linear::new(w[0], w[1], &mut rng));
    }

    // Internal row-shuffled val split for early stopping.
    let mut perm: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut perm);
    let n_val = ((n as f64) * cfg.val_frac).round() as usize;
    let n_val = n_val.clamp(if n > 1 { 1 } else { 0 }, n.saturating_sub(1).max(1));
    let val_idx: Vec<usize> = perm[..n_val].to_vec();
    let tr_idx: Vec<usize> = perm[n_val..].to_vec();

    let mut model = Mlp {
        layers,
        leaky_slope: cfg.leaky_slope,
        n_in,
        n_out,
        best_val_loss: f64::INFINITY,
        n_iter: 0,
    };

    let mut best_layers = model.layers.clone();
    let mut best_val = f64::INFINITY;
    let mut bad_epochs = 0usize;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let mut adam_t = 0u64;

    let batch = cfg.batch_size.max(1);

    for epoch in 0..cfg.max_iter {
        // Shuffle training rows each epoch.
        let mut order = tr_idx.clone();
        rng.shuffle(&mut order);

        let mut bi = 0;
        while bi < order.len() {
            let hi = (bi + batch).min(order.len());
            let batch_rows = &order[bi..hi];
            adam_t += 1;
            train_minibatch(
                &mut model.layers,
                cfg.leaky_slope,
                x_std,
                y,
                n_in,
                n_out,
                batch_rows,
                cfg.lr,
                beta1,
                beta2,
                eps,
                adam_t,
            );
            bi = hi;
        }

        // Validation MSE (masked).
        let v = masked_mse(
            &model.layers,
            cfg.leaky_slope,
            x_std,
            y,
            n_in,
            n_out,
            &val_idx,
        );
        model.n_iter = epoch + 1;
        if v < best_val - cfg.tol {
            best_val = v;
            best_layers = model.layers.clone();
            bad_epochs = 0;
        } else {
            bad_epochs += 1;
        }
        if bad_epochs >= cfg.n_iter_no_change {
            break;
        }
    }

    model.layers = best_layers;
    model.best_val_loss = best_val;
    model
}

/// One Adam minibatch step. Forward + backprop through LeakyReLU stack
/// with masked-MSE (NaN targets skipped). Updates `layers` in place.
#[allow(clippy::too_many_arguments)]
fn train_minibatch(
    layers: &mut [Linear],
    slope: f64,
    x_std: &[f64],
    y: &[f64],
    n_in: usize,
    n_out: usize,
    rows: &[usize],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    adam_t: u64,
) {
    let n_layers = layers.len();
    // Gradient accumulators per layer.
    let mut gw: Vec<Vec<f64>> = layers.iter().map(|l| vec![0.0; l.w.len()]).collect();
    let mut gb: Vec<Vec<f64>> = layers.iter().map(|l| vec![0.0; l.b.len()]).collect();

    // Count of valid (non-NaN) target elements in the batch, for the
    // 1/(batch * n_out_valid) MSE normalization. We normalize by the
    // number of contributing rows so the gradient scale matches a
    // per-row mean-over-outputs MSE (zentrain's `((pred-y)**2).mean()`).
    let mut valid_rows = 0usize;

    for &r in rows {
        let x = &x_std[r * n_in..(r + 1) * n_in];
        // Forward, caching pre-activation (z) and activation (a) per layer.
        let mut acts: Vec<Vec<f64>> = Vec::with_capacity(n_layers + 1);
        let mut zs: Vec<Vec<f64>> = Vec::with_capacity(n_layers);
        acts.push(x.to_vec());
        for (li, layer) in layers.iter().enumerate() {
            let mut z = vec![0.0f64; layer.out_dim];
            layer.forward(acts.last().unwrap(), &mut z);
            zs.push(z.clone());
            if li + 1 < n_layers {
                for v in &mut z {
                    if *v < 0.0 {
                        *v *= slope;
                    }
                }
            }
            acts.push(z);
        }

        // Output-layer delta = dL/dz_last. MSE over valid outputs.
        let pred = acts.last().unwrap();
        let target = &y[r * n_out..(r + 1) * n_out];
        let mut n_valid_out = 0usize;
        for o in 0..n_out {
            if target[o].is_finite() {
                n_valid_out += 1;
            }
        }
        if n_valid_out == 0 {
            continue;
        }
        valid_rows += 1;
        // d(mean_o (pred-y)^2)/d pred_o = 2/n_valid_out * (pred-y), masked.
        let inv = 2.0 / n_valid_out as f64;
        let mut delta: Vec<f64> = vec![0.0; n_out];
        for o in 0..n_out {
            if target[o].is_finite() {
                delta[o] = inv * (pred[o] - target[o]);
            }
        }

        // Backprop.
        let mut cur_delta = delta;
        for li in (0..n_layers).rev() {
            let layer = &layers[li];
            let a_in = &acts[li]; // input to this layer
            // Accumulate grads: gw[o*in + i] += delta[o]*a_in[i]; gb[o] += delta[o].
            for o in 0..layer.out_dim {
                let d = cur_delta[o];
                if d != 0.0 {
                    let base = o * layer.in_dim;
                    let gwl = &mut gw[li];
                    for i in 0..layer.in_dim {
                        gwl[base + i] += d * a_in[i];
                    }
                    gb[li][o] += d;
                }
            }
            if li > 0 {
                // Propagate delta to previous layer through W then the
                // LeakyReLU derivative at that layer's pre-activation.
                let prev_z = &zs[li - 1];
                let mut prev_delta = vec![0.0f64; layer.in_dim];
                for o in 0..layer.out_dim {
                    let d = cur_delta[o];
                    if d == 0.0 {
                        continue;
                    }
                    let base = o * layer.in_dim;
                    for i in 0..layer.in_dim {
                        prev_delta[i] += d * layer.w[base + i];
                    }
                }
                for i in 0..layer.in_dim {
                    let g = if prev_z[i] >= 0.0 { 1.0 } else { slope };
                    prev_delta[i] *= g;
                }
                cur_delta = prev_delta;
            }
        }
    }

    if valid_rows == 0 {
        return;
    }
    let scale = 1.0 / valid_rows as f64;
    // Adam update per layer.
    let bc1 = 1.0 - beta1.powi(adam_t as i32);
    let bc2 = 1.0 - beta2.powi(adam_t as i32);
    for li in 0..n_layers {
        let layer = &mut layers[li];
        for k in 0..layer.w.len() {
            let g = gw[li][k] * scale;
            layer.mw[k] = beta1 * layer.mw[k] + (1.0 - beta1) * g;
            layer.vw[k] = beta2 * layer.vw[k] + (1.0 - beta2) * g * g;
            let mhat = layer.mw[k] / bc1;
            let vhat = layer.vw[k] / bc2;
            layer.w[k] -= lr * mhat / (vhat.sqrt() + eps);
        }
        for k in 0..layer.b.len() {
            let g = gb[li][k] * scale;
            layer.mb[k] = beta1 * layer.mb[k] + (1.0 - beta1) * g;
            layer.vb[k] = beta2 * layer.vb[k] + (1.0 - beta2) * g * g;
            let mhat = layer.mb[k] / bc1;
            let vhat = layer.vb[k] / bc2;
            layer.b[k] -= lr * mhat / (vhat.sqrt() + eps);
        }
    }
}

/// Masked MSE over `rows`: mean over (row, valid-output) of `(pred-y)^2`.
fn masked_mse(
    layers: &[Linear],
    slope: f64,
    x_std: &[f64],
    y: &[f64],
    n_in: usize,
    n_out: usize,
    rows: &[usize],
) -> f64 {
    let n_layers = layers.len();
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for &r in rows {
        let x = &x_std[r * n_in..(r + 1) * n_in];
        let mut cur = x.to_vec();
        for (li, layer) in layers.iter().enumerate() {
            let mut out = vec![0.0f64; layer.out_dim];
            layer.forward(&cur, &mut out);
            if li + 1 < n_layers {
                for v in &mut out {
                    if *v < 0.0 {
                        *v *= slope;
                    }
                }
            }
            cur = out;
        }
        let target = &y[r * n_out..(r + 1) * n_out];
        for o in 0..n_out {
            if target[o].is_finite() {
                let d = cur[o] - target[o];
                sum += d * d;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        f64::INFINITY
    } else {
        sum / cnt as f64
    }
}
