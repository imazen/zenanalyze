//! Forward-pass kernel.
//!
//! For each layer:
//!   1. Initialize accumulator with biases (broadcast).
//!   2. For each input element x[i], add x[i] * W[i, :] to the
//!      accumulator. This is a SAXPY across all output dims.
//!   3. Apply activation in-place.
//!
//! The SAXPY-style inner loop is what `magetypes::f32x8` wants —
//! each output dim is independent so we vectorize trivially across
//! the output axis. archmage tokens at the outer-layer boundary
//! prevent LLVM from un-inlining the per-tier dispatch.

use crate::model::{Activation, LayerView, Model, WeightStorage};

/// Run the full forward pass: scale inputs, then layer-by-layer.
///
/// `scratch_a` and `scratch_b` are reused across layers. They must
/// each be at least `max(layer.out_dim, n_inputs)` long. `output`
/// must be exactly `n_outputs` long.
pub fn forward(
    model: &Model<'_>,
    features: &[f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    output: &mut [f32],
) {
    let n_inputs = model.n_inputs();
    debug_assert_eq!(features.len(), n_inputs);
    debug_assert_eq!(output.len(), model.n_outputs());

    // Scale inputs: x' = (x - mean) * scale.
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    let cur = &mut scratch_a[..n_inputs];
    for i in 0..n_inputs {
        cur[i] = (features[i] - mean[i]) * scale[i];
    }

    let mut input_buf: &mut [f32] = scratch_a;
    let mut output_buf: &mut [f32] = scratch_b;

    let layers = model.layers();
    let last_idx = layers.len() - 1;

    for (idx, layer) in layers.iter().enumerate() {
        let in_dim = layer.in_dim;
        let out_dim = layer.out_dim;

        let dst: &mut [f32] = if idx == last_idx {
            &mut output[..out_dim]
        } else {
            &mut output_buf[..out_dim]
        };
        let src = &input_buf[..in_dim];

        layer_forward(layer, src, dst);

        if idx != last_idx {
            // Swap buffers for the next layer.
            core::mem::swap(&mut input_buf, &mut output_buf);
        }
    }
}

/// Compute `dst = activation(W^T * src + b)` for one layer.
fn layer_forward(layer: &LayerView<'_>, src: &[f32], dst: &mut [f32]) {
    let out_dim = layer.out_dim;
    let in_dim = layer.in_dim;
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(dst.len(), out_dim);
    debug_assert_eq!(layer.biases.len(), out_dim);

    // Initialize accumulator with biases.
    dst.copy_from_slice(layer.biases);

    match &layer.weights {
        WeightStorage::F32(w) => saxpy_matmul_f32(src, w, dst, in_dim, out_dim),
        #[cfg(feature = "f16")]
        WeightStorage::F16(w) => saxpy_matmul_f16(src, w, dst, in_dim, out_dim),
    }

    apply_activation(dst, layer.activation);
}

/// `dst[o] += sum_i src[i] * W[i, o]` for the f32-stored case.
///
/// Inner loop adds `src[i] * W[i, :]` to `dst[:]` in chunks of 8.
/// Embarrassingly parallel across the output dim. The fixed-size
/// `[f32; 8]` chunk loads let LLVM auto-vectorize this to one f32x8
/// FMA per iteration on AVX2/AVX-512 and 2× f32x4 on NEON/WASM. We
/// can drop in `#[magetypes]` dispatch later for guaranteed
/// vectorization across tiers — for now scalar autovec is plenty
/// fast for the model sizes we ship (15K weights → ~15 µs).
///
/// Layout: weights are input-major, so `W[i, :]` is
/// `&w[i * out_dim..(i + 1) * out_dim]` — contiguous in memory.
fn saxpy_matmul_f32(src: &[f32], w: &[f32], dst: &mut [f32], in_dim: usize, out_dim: usize) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let weight_chunk: &[f32; 8] = row[base..base + 8].try_into().unwrap();
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                acc_chunk[k] = s.mul_add(weight_chunk[k], acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                dst[tail_start + k] = s.mul_add(row[tail_start + k], dst[tail_start + k]);
            }
        }
    }
}

#[cfg(feature = "f16")]
fn saxpy_matmul_f16(
    src: &[f32],
    w: &[half::f16],
    dst: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(w.len(), in_dim * out_dim);
    let chunks = out_dim / 8;
    let tail = out_dim % 8;

    for i in 0..in_dim {
        let s = src[i];
        if s == 0.0 {
            continue;
        }
        let row = &w[i * out_dim..(i + 1) * out_dim];

        for c in 0..chunks {
            let base = c * 8;
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                let wf = row[base + k].to_f32();
                acc_chunk[k] = s.mul_add(wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = row[tail_start + k].to_f32();
                dst[tail_start + k] = s.mul_add(wf, dst[tail_start + k]);
            }
        }
    }
}

fn apply_activation(buf: &mut [f32], act: Activation) {
    match act {
        Activation::Identity => {}
        Activation::Relu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}
