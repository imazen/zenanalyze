//! Forward-pass kernel.
//!
//! For each layer:
//! 1. Initialize the accumulator with the layer's biases (broadcast).
//! 2. For each input element `x[i]`, add `x[i] * W[i, :]` to the
//!    accumulator. Embarrassingly parallel across the output dim.
//! 3. Apply activation in-place.
//!
//! The SAXPY-style inner loop is what `magetypes::f32x8` wants when
//! SIMD dispatch lands. The fixed-size `[f32; 8]` chunk loads let
//! LLVM auto-vectorize this to one `f32x8` FMA per iteration on
//! AVX2/AVX-512 and 2× `f32x4` on NEON/WASM today.

use crate::error::PredictError;
use crate::model::{Activation, LEAKY_RELU_ALPHA, LayerView, Model, WeightStorage};

/// Run the full forward pass: scale inputs, then layer-by-layer.
///
/// `scratch_a` and `scratch_b` are reused across layers. They must
/// each be at least [`Model::scratch_len`](crate::Model::scratch_len)
/// long. `output` must be exactly `n_outputs` long.
pub fn forward(
    model: &Model<'_>,
    features: &[f32],
    scratch_a: &mut [f32],
    scratch_b: &mut [f32],
    output: &mut [f32],
) -> Result<(), PredictError> {
    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    if features.len() != n_inputs {
        return Err(PredictError::FeatureLenMismatch {
            expected: n_inputs,
            got: features.len(),
        });
    }
    if output.len() != n_outputs {
        return Err(PredictError::FeatureLenMismatch {
            expected: n_outputs,
            got: output.len(),
        });
    }
    let need = model.scratch_len();
    if scratch_a.len() < need || scratch_b.len() < need {
        return Err(PredictError::FeatureLenMismatch {
            expected: need,
            got: scratch_a.len().min(scratch_b.len()),
        });
    }

    // Scale inputs: x' = (x - mean) / scale.
    //
    // sklearn's `StandardScaler.scale_` IS the standard deviation
    // (np.sqrt(var_)); transform divides by it. Mirror that here so
    // the runtime feeds the network inputs scaled the same way the
    // training-time eval did.
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    for i in 0..n_inputs {
        scratch_a[i] = (features[i] - mean[i]) / scale[i];
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
            core::mem::swap(&mut input_buf, &mut output_buf);
        }
    }
    Ok(())
}

fn layer_forward(layer: &LayerView<'_>, src: &[f32], dst: &mut [f32]) {
    let out_dim = layer.out_dim;
    let in_dim = layer.in_dim;
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(dst.len(), out_dim);
    debug_assert_eq!(layer.biases.len(), out_dim);

    match &layer.weights {
        WeightStorage::F32(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f32(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::F16(w) => {
            dst.copy_from_slice(layer.biases);
            saxpy_matmul_f16(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::I8 { weights, scales } => {
            // Per-output `scales[o]` only applies to the SAXPY
            // accumulator, not the bias. Zero dst, accumulate raw,
            // then `dst[o] = bias[o] + scales[o] * dst[o]`.
            for v in dst.iter_mut() {
                *v = 0.0;
            }
            saxpy_matmul_i8(src, weights, dst, in_dim, out_dim);
            debug_assert_eq!(scales.len(), out_dim);
            for o in 0..out_dim {
                dst[o] = layer.biases[o] + scales[o] * dst[o];
            }
        }
    }

    apply_activation(dst, layer.activation);
}

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
                acc_chunk[k] = fma(s, weight_chunk[k], acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                dst[tail_start + k] = fma(s, row[tail_start + k], dst[tail_start + k]);
            }
        }
    }
}

fn saxpy_matmul_f16(src: &[f32], w: &[u16], dst: &mut [f32], in_dim: usize, out_dim: usize) {
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
                let wf = f16_bits_to_f32(row[base + k]);
                acc_chunk[k] = fma(s, wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = f16_bits_to_f32(row[tail_start + k]);
                dst[tail_start + k] = fma(s, wf, dst[tail_start + k]);
            }
        }
    }
}

fn saxpy_matmul_i8(src: &[f32], w: &[i8], dst: &mut [f32], in_dim: usize, out_dim: usize) {
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
            let weight_chunk: &[i8; 8] = row[base..base + 8].try_into().unwrap();
            let acc_chunk: &mut [f32; 8] = (&mut dst[base..base + 8]).try_into().unwrap();
            for k in 0..8 {
                let wf = weight_chunk[k] as f32;
                acc_chunk[k] = fma(s, wf, acc_chunk[k]);
            }
        }
        if tail > 0 {
            let tail_start = chunks * 8;
            for k in 0..tail {
                let wf = row[tail_start + k] as f32;
                dst[tail_start + k] = fma(s, wf, dst[tail_start + k]);
            }
        }
    }
}

/// IEEE-754 binary16 → binary32 converter. Pure integer bit math —
/// works in `no_std` and at compile time. Same answer as
/// `_mm256_cvtph_ps`, one element at a time.
#[inline]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let exp = (h & 0x7c00) >> 10;
    let mant = h & 0x03ff;
    let bits = if exp == 0 {
        if mant == 0 {
            0
        } else {
            // Subnormal — promote to f32 normal.
            let k = 31 - mant.leading_zeros();
            let shift = 10 - k;
            let normalized_mant = (mant << shift) & 0x3ff;
            let f32_exp = k + 103;
            (f32_exp << 23) | (normalized_mant << 13)
        }
    } else if exp == 0x1f {
        0x7f80_0000 | (mant << 13)
    } else {
        ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(sign | bits)
}

/// `a * b + c`. Uses `f32::mul_add` (single-rounding fma) when the
/// `std` feature is on; falls back to `a * b + c` (two roundings)
/// for `no_std + alloc` builds. The numerical difference is in the
/// last bit and well below MLP training noise; the perf difference
/// is one fused instruction vs two.
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        a * b + c
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
        Activation::LeakyRelu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v *= LEAKY_RELU_ALPHA;
                }
            }
        }
    }
}
