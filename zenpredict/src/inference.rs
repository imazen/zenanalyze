//! Forward-pass executor.
//!
//! Runs a model's op graph (see [`crate::graph`]) node by node in file
//! order. A v3 layer chain arrives lowered to `Input → Dense → … →
//! Dense`; each Dense node runs [`layer_forward`], the v3 per-layer code:
//!
//! 1. Initialize the accumulator with the layer's biases (broadcast).
//! 2. For each input element `x[i]`, add `x[i] * W[i, :]` to the
//!    accumulator. Embarrassingly parallel across the output dim.
//! 3. Apply activation in-place.
//!
//! The SAXPY-style inner loop is what `magetypes::f32x8` wants when
//! SIMD dispatch lands. The fixed-size `[f32; 8]` chunk loads let
//! LLVM auto-vectorize this to one `f32x8` FMA per iteration on
//! AVX2/AVX-512 and 2× `f32x4` on NEON/WASM today.
//!
//! Every node except the last writes into its load-planned slot of one
//! f32 arena; the last node writes straight into the caller's output.

use crate::error::PredictError;
use crate::graph::{GraphNode, NodeKind, concat_inputs, gather_indices};
use crate::model::{
    Activation, EXP_INPUT_CLAMP, LEAKY_RELU_ALPHA, LayerView, Model, SOFTPLUS_THRESHOLD,
    WeightStorage,
};

#[cfg(feature = "simd")]
use archmage::autoversion;

/// Run the full forward pass: scale inputs, then every graph node.
///
/// `arena` must be at least `model.arena_len()` long and `output`
/// exactly `n_outputs` long. Nothing is allocated.
pub(crate) fn forward(
    model: &Model,
    features: &[f32],
    arena: &mut [f32],
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
    let need = model.arena_len();
    if arena.len() < need {
        return Err(PredictError::FeatureLenMismatch {
            expected: need,
            got: arena.len(),
        });
    }

    let nodes = &model.graph().nodes;
    let last = nodes.len() - 1;
    for (i, node) in nodes.iter().enumerate() {
        let width = node.width as usize;
        if i == last {
            let io = Inputs {
                left: arena,
                right: &[],
                right_base: usize::MAX,
            };
            run_node(model, nodes, node, &io, features, &mut output[..width])?;
        } else {
            // Planned slots never overlap a live input, so the node's own
            // slot splits the arena into a writable middle and two
            // readable sides.
            let start = node.slot as usize;
            let (left, rest) = arena.split_at_mut(start);
            let (dst, right) = rest.split_at_mut(width);
            let io = Inputs {
                left,
                right,
                right_base: start + width,
            };
            run_node(model, nodes, node, &io, features, dst)?;
        }
    }
    Ok(())
}

/// Read-only view of the arena around the slot being written.
struct Inputs<'x> {
    left: &'x [f32],
    right: &'x [f32],
    right_base: usize,
}

impl Inputs<'_> {
    #[inline]
    fn get<'n>(&'n self, nodes: &[GraphNode], j: u32) -> &'n [f32] {
        let n = &nodes[j as usize];
        let (off, len) = (n.slot as usize, n.width as usize);
        if off < self.right_base {
            &self.left[off..off + len]
        } else {
            let o = off - self.right_base;
            &self.right[o..o + len]
        }
    }
}

fn run_node(
    model: &Model,
    nodes: &[GraphNode],
    node: &GraphNode,
    io: &Inputs<'_>,
    features: &[f32],
    dst: &mut [f32],
) -> Result<(), PredictError> {
    match node.kind {
        NodeKind::Input => {
            // Scale inputs: x' = (x - mean) / scale.
            //
            // Zero-variance columns: sklearn's `_handle_zeros_in_scale`
            // replaces `scale=0` with `1.0` so the column passes through as
            // `(x - mean)`. Mirror that defensively.
            scale_inputs(features, model.scaler_mean(), model.scaler_scale(), dst);
        }
        NodeKind::Dense { input, ref layer } => {
            let view = model.materialize_offsets(layer);
            layer_forward(&view, io.get(nodes, input), dst)?;
        }
        NodeKind::Activation { input, activation } => {
            dst.copy_from_slice(io.get(nodes, input));
            apply_activation(dst, activation);
        }
        NodeKind::Gather {
            input,
            indices,
            contiguous_start,
        } => {
            let src = io.get(nodes, input);
            match contiguous_start {
                Some(start) => {
                    let start = start as usize;
                    dst.copy_from_slice(&src[start..start + dst.len()]);
                }
                None => {
                    let idx = gather_indices(indices, dst.len(), model.raw_bytes());
                    for (d, &k) in dst.iter_mut().zip(idx) {
                        *d = src[k as usize];
                    }
                }
            }
        }
        NodeKind::Add { a, b } => add_elementwise(io.get(nodes, a), io.get(nodes, b), dst),
        NodeKind::Mul { a, b } => mul_elementwise(io.get(nodes, a), io.get(nodes, b), dst),
        NodeKind::Concat { inputs } => {
            let mut at = 0;
            for &j in concat_inputs(inputs, model.raw_bytes()) {
                let src = io.get(nodes, j);
                dst[at..at + src.len()].copy_from_slice(src);
                at += src.len();
            }
        }
    }
    Ok(())
}

fn layer_forward(layer: &LayerView<'_>, src: &[f32], dst: &mut [f32]) -> Result<(), PredictError> {
    let out_dim = layer.out_dim;
    let in_dim = layer.in_dim;
    debug_assert_eq!(src.len(), in_dim);
    debug_assert_eq!(dst.len(), out_dim);
    debug_assert!(layer.biases.is_empty() || layer.biases.len() == out_dim);

    // A v4 Dense node may omit its bias; that computes exactly as if every
    // bias were `+0.0` (same fill, same `0.0 + s * acc` I8 post-scale).
    let has_bias = !layer.biases.is_empty();
    match &layer.weights {
        WeightStorage::F32(w) => {
            init_bias(dst, layer.biases, has_bias);
            saxpy_matmul_f32(src, w, dst, in_dim, out_dim);
        }
        WeightStorage::F16(w) => {
            init_bias(dst, layer.biases, has_bias);
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
            if has_bias {
                for o in 0..out_dim {
                    dst[o] = layer.biases[o] + scales[o] * dst[o];
                }
            } else {
                for o in 0..out_dim {
                    dst[o] = 0.0 + scales[o] * dst[o];
                }
            }
        }
    }

    apply_activation(dst, layer.activation);
    Ok(())
}

#[inline]
fn init_bias(dst: &mut [f32], biases: &[f32], has_bias: bool) {
    if has_bias {
        dst.copy_from_slice(biases);
    } else {
        dst.fill(0.0);
    }
}

/// `dst[i] = (x[i] - mean[i]) / scale[i]`, `scale == 0` treated as 1.
/// Subtraction and division are correctly rounded per lane, so the
/// `#[autoversion]` variant (8-wide under `+avx2`) is bit-identical to the
/// scalar loop. Zipped iterators keep bounds checks out of the loop so it
/// vectorizes at all.
#[cfg_attr(feature = "simd", autoversion(v3))]
fn scale_inputs(x: &[f32], mean: &[f32], scale: &[f32], dst: &mut [f32]) {
    for (((d, &v), &m), &s) in dst.iter_mut().zip(x).zip(mean).zip(scale) {
        let safe_s = if s == 0.0 { 1.0 } else { s };
        *d = (v - m) / safe_s;
    }
}

// Elementwise binary ops. IEEE-754 add and multiply are correctly
// rounded per lane at any vector width, so the `#[autoversion]` variant
// (vectorized under `+avx2`) is bit-identical to the scalar one; no
// reduction, no reassociation. Gated by `simd_parity_tests`.
#[cfg_attr(feature = "simd", autoversion(v3))]
fn add_elementwise(a: &[f32], b: &[f32], dst: &mut [f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *d = x + y;
    }
}

#[cfg_attr(feature = "simd", autoversion(v3))]
fn mul_elementwise(a: &[f32], b: &[f32], dst: &mut [f32]) {
    for ((d, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *d = x * y;
    }
}

// The three `saxpy_matmul_*` kernels carry `#[autoversion]` under the
// `simd` feature. It is a **signature-only** transform: archmage re-emits
// the body verbatim as `<name>_v3` under
// `#[target_feature(enable = "avx2,fma")]` plus `<name>_scalar` (the body
// as written), and replaces the original name with a runtime dispatcher.
// Nothing below this comment changes.
//
// Why: baseline x86-64 has no FMA, so the `fma()` helper's `f32::mul_add`
// lowers to an out-of-line software `fmaf` call — 41% of `bake_verdict`'s
// cycles (`perf`, 2026-08-04). Inside the `+fma` variant the same call is
// one `vfmadd` instruction, and LLVM auto-vectorizes the fixed-size
// `[f32; 8]` chunk into `vfmadd231ps`.
//
// Why that is bit-identical, not "close enough":
//   * `f32::mul_add` is IEEE-754 `fusedMultiplyAdd` — a single correctly
//     rounded result. `vfmadd` computes that same operation. Software and
//     hardware FMA agree on every input, including subnormals and NaN
//     payloads. (Rewriting to `a * b + c` would round twice and is
//     deliberately NOT done — see `fma` below.)
//   * The inner loop is a SAXPY, not a reduction: lane `k` accumulates
//     only into `dst[k]`. Widening 8 independent FMAs into one vector FMA
//     reassociates nothing.
// `dispatcher_and_scalar_variant_agree_bitwise` in `simd_parity_tests`
// holds this empirically over random and adversarial inputs.
#[cfg_attr(feature = "simd", autoversion(v3))]
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

#[cfg_attr(feature = "simd", autoversion(v3))]
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

#[cfg_attr(feature = "simd", autoversion(v3))]
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
///
/// **Do not "optimize" the `std` arm to `a * b + c`.** Every shipped
/// zensim score and every baked picker threshold was produced through
/// the single-rounding path; two roundings would move results in the
/// last bit across the whole corpus. The cost of `mul_add` on a CPU
/// without `+fma` enabled is an out-of-line software `fmaf` call — the
/// fix for that is the `simd` feature's `#[autoversion]` on the kernels
/// above (same operation, one instruction), not a change of operation.
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
        Activation::Exp => {
            for v in buf.iter_mut() {
                *v = exp_clamped(*v);
            }
        }
        Activation::Softplus => {
            for v in buf.iter_mut() {
                *v = softplus(*v);
            }
        }
    }
}

/// [`Activation::Exp`]. `libm` on every build (std too): a platform
/// `expf` is not guaranteed to return the same bits everywhere. NaN stays
/// NaN; ±inf clamp to `exp(±EXP_INPUT_CLAMP)`.
#[inline]
pub(crate) fn exp_clamped(x: f32) -> f32 {
    libm::expf(x.clamp(-EXP_INPUT_CLAMP, EXP_INPUT_CLAMP))
}

/// [`Activation::Softplus`], PyTorch `Softplus(beta=1, threshold=20)`.
/// `libm` on every build.
#[inline]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > SOFTPLUS_THRESHOLD {
        x
    } else {
        libm::log1pf(libm::expf(x))
    }
}

/// Bit-identity gate for the `simd` feature.
///
/// `#[autoversion]` leaves `<kernel>_scalar` (the body exactly as written,
/// no target features) next to the dispatcher, so the two can be compared
/// directly on the same inputs. Every element of every output must match
/// **by bit pattern** — not by tolerance. A tolerance here would let the
/// two-rounding rewrite this crate forbids slip through unnoticed.
///
/// On a machine without AVX2+FMA the dispatcher selects `_scalar` and the
/// comparison is trivially true; the gate is meaningful on the CI runners
/// and dev boxes that do have it (`archmage::X64V3Token::summon()` reports
/// which case ran).
#[cfg(all(test, feature = "simd"))]
mod simd_parity_tests {
    use super::{saxpy_matmul_f16, saxpy_matmul_f16_scalar};
    use super::{saxpy_matmul_f32, saxpy_matmul_f32_scalar};
    use super::{saxpy_matmul_i8, saxpy_matmul_i8_scalar};
    use archmage::{ScalarToken, SimdToken};

    /// Deterministic xorshift — no rand dependency in the parity gate, so
    /// the exact byte sequence it exercises is reproducible from the seed.
    struct Rng(u64);
    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        /// Wide dynamic range on purpose: exercises subnormal, huge, and
        /// cancellation-prone magnitudes where a rounding difference
        /// would actually show up.
        fn next_f32(&mut self) -> f32 {
            let m = (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let e = (self.next_u32() % 60) as i32 - 30;
            m * (2.0f32).powi(e)
        }
    }

    /// Shapes chosen to cover both the 8-wide chunk loop and every
    /// possible scalar-tail length (`out_dim % 8` = 0..7), plus the real
    /// model widths zensim ships (944-in / 128-hidden) and a 1×1 edge.
    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (3, 7),
        (8, 8),
        (5, 9),
        (16, 13),
        (7, 15),
        (944, 128),
        (128, 1),
        (128, 3),
    ];

    #[test]
    fn dispatcher_and_scalar_variant_agree_bitwise() {
        let tier = if archmage::X64V3Token::summon().is_some() {
            "v3 (AVX2+FMA)"
        } else {
            "scalar (no AVX2+FMA on this host)"
        };
        let st = ScalarToken::summon().expect("ScalarToken is always available");

        let mut rng = Rng(0x5eed_1234_9876_abcd);
        for &(in_dim, out_dim) in SHAPES {
            let src: Vec<f32> = (0..in_dim).map(|_| rng.next_f32()).collect();
            let bias: Vec<f32> = (0..out_dim).map(|_| rng.next_f32()).collect();

            // f32 weights
            let w32: Vec<f32> = (0..in_dim * out_dim).map(|_| rng.next_f32()).collect();
            let mut a = bias.clone();
            let mut b = bias.clone();
            saxpy_matmul_f32(&src, &w32, &mut a, in_dim, out_dim);
            saxpy_matmul_f32_scalar(st, &src, &w32, &mut b, in_dim, out_dim);
            assert_bits(&a, &b, "f32", in_dim, out_dim, tier);

            // f16 weights (raw binary16 bit patterns, full range)
            let w16: Vec<u16> = (0..in_dim * out_dim)
                .map(|_| (rng.next_u32() & 0xffff) as u16)
                .collect();
            let mut a = bias.clone();
            let mut b = bias.clone();
            saxpy_matmul_f16(&src, &w16, &mut a, in_dim, out_dim);
            saxpy_matmul_f16_scalar(st, &src, &w16, &mut b, in_dim, out_dim);
            assert_bits(&a, &b, "f16", in_dim, out_dim, tier);

            // i8 weights
            let w8: Vec<i8> = (0..in_dim * out_dim)
                .map(|_| (rng.next_u32() & 0xff) as u8 as i8)
                .collect();
            let mut a = vec![0.0f32; out_dim];
            let mut b = vec![0.0f32; out_dim];
            saxpy_matmul_i8(&src, &w8, &mut a, in_dim, out_dim);
            saxpy_matmul_i8_scalar(st, &src, &w8, &mut b, in_dim, out_dim);
            assert_bits(&a, &b, "i8", in_dim, out_dim, tier);
        }
    }

    /// Adversarial inputs: zeros (the `s == 0.0` skip branch), infinities,
    /// NaN, and subnormals — the places where a fused-vs-unfused or
    /// reassociated implementation diverges first.
    #[test]
    fn dispatcher_agrees_on_special_values() {
        let tier = if archmage::X64V3Token::summon().is_some() {
            "v3 (AVX2+FMA)"
        } else {
            "scalar"
        };
        let st = ScalarToken::summon().expect("ScalarToken is always available");
        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE / 4.0, // subnormal
            f32::MAX,
            -f32::MAX,
            1e-30,
            1e30,
        ];
        let in_dim = specials.len();
        let out_dim = 11; // 8-chunk + 3-tail
        let src: Vec<f32> = specials.to_vec();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| specials[i % specials.len()])
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| specials[i % specials.len()]).collect();

        let mut a = bias.clone();
        let mut b = bias.clone();
        saxpy_matmul_f32(&src, &w, &mut a, in_dim, out_dim);
        saxpy_matmul_f32_scalar(st, &src, &w, &mut b, in_dim, out_dim);
        assert_bits(&a, &b, "f32/special", in_dim, out_dim, tier);
    }

    /// The graph executor's elementwise Add / Mul: the dispatcher (vector
    /// under `+avx2`) and the scalar variant must agree bit for bit on
    /// random, tail-length and special-value operands.
    #[test]
    fn elementwise_dispatcher_agrees_bitwise() {
        use super::{add_elementwise, add_elementwise_scalar};
        use super::{mul_elementwise, mul_elementwise_scalar};
        let tier = if archmage::X64V3Token::summon().is_some() {
            "v3 (AVX2+FMA)"
        } else {
            "scalar"
        };
        let st = ScalarToken::summon().expect("ScalarToken is always available");
        let specials = [
            0.0f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE / 4.0,
            f32::MAX,
            -f32::MAX,
        ];
        let mut rng = Rng(0x00e1_e3e4_7a5e_0001);
        for len in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 64, 129, 944] {
            let a: Vec<f32> = (0..len)
                .map(|i| {
                    if i % 5 == 0 {
                        specials[(i / 5) % specials.len()]
                    } else {
                        rng.next_f32()
                    }
                })
                .collect();
            let b: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let (mut x, mut y) = (vec![0.0f32; len], vec![0.0f32; len]);
            add_elementwise(&a, &b, &mut x);
            add_elementwise_scalar(st, &a, &b, &mut y);
            assert_bits(&x, &y, "add", len, 1, tier);
            mul_elementwise(&a, &b, &mut x);
            mul_elementwise_scalar(st, &a, &b, &mut y);
            assert_bits(&x, &y, "mul", len, 1, tier);
        }
    }

    /// The Input-node scaler: dispatcher vs scalar, bit for bit, including
    /// zero scales (treated as 1) and special values.
    #[test]
    fn scaler_dispatcher_agrees_bitwise() {
        use super::{scale_inputs, scale_inputs_scalar};
        let tier = if archmage::X64V3Token::summon().is_some() {
            "v3 (AVX2+FMA)"
        } else {
            "scalar"
        };
        let st = ScalarToken::summon().expect("ScalarToken is always available");
        let mut rng = Rng(0x5ca1_e000_0000_0001);
        for len in [1usize, 7, 8, 9, 31, 228, 944] {
            let x: Vec<f32> = (0..len)
                .map(|i| match i % 9 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => -0.0,
                    _ => rng.next_f32(),
                })
                .collect();
            let mean: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();
            let scale: Vec<f32> = (0..len)
                .map(|i| if i % 5 == 0 { 0.0 } else { rng.next_f32() })
                .collect();
            let (mut a, mut b) = (vec![0.0f32; len], vec![0.0f32; len]);
            scale_inputs(&x, &mean, &scale, &mut a);
            scale_inputs_scalar(st, &x, &mean, &scale, &mut b);
            assert_bits(&a, &b, "scaler", len, 1, tier);
        }
    }

    fn assert_bits(a: &[f32], b: &[f32], what: &str, in_dim: usize, out_dim: usize, tier: &str) {
        assert_eq!(a.len(), b.len());
        for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what} {in_dim}x{out_dim} lane {k}: dispatcher [{tier}] gave {x} \
                 (0x{:08x}), scalar gave {y} (0x{:08x}) — the SIMD variant is NOT \
                 bit-identical",
                x.to_bits(),
                y.to_bits()
            );
        }
    }
}

#[cfg(test)]
mod f16_tests {
    use super::f16_bits_to_f32;

    fn check(bits: u16, expected: f32, what: &str) {
        let got = f16_bits_to_f32(bits);
        if expected.is_nan() {
            assert!(
                got.is_nan(),
                "{what}: bits=0x{bits:04x} expected NaN, got {got}"
            );
        } else {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "{what}: bits=0x{bits:04x} got {got} ({:08x}), expected {expected} ({:08x})",
                got.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn zeros_and_signs() {
        check(0x0000, 0.0, "+0");
        check(0x8000, -0.0, "-0");
    }

    #[test]
    fn ones() {
        check(0x3c00, 1.0, "+1.0");
        check(0xbc00, -1.0, "-1.0");
        check(0x4000, 2.0, "+2.0");
        check(0xc000, -2.0, "-2.0");
    }

    #[test]
    fn fractions() {
        // 0.5 = exp=14 (bias-15 → -1), mant=0
        check(0x3800, 0.5, "0.5");
        // 1/3 representable: 0x3555 = 0.333251953125
        check(0x3555, 0.333_251_95, "approx 1/3");
    }

    #[test]
    fn subnormals() {
        // Smallest positive subnormal: 0x0001 = 2^-24 ≈ 5.96e-8
        check(0x0001, 5.960_464_5e-8, "smallest +subnormal");
        // Largest subnormal: 0x03ff
        check(0x03ff, 6.097_555e-5, "largest +subnormal");
        check(0x8001, -5.960_464_5e-8, "smallest -subnormal");
    }

    #[test]
    fn extremes() {
        // Smallest positive normal: 0x0400 = 2^-14 ≈ 6.10e-5
        check(0x0400, 6.103_515_6e-5, "smallest +normal");
        // Largest normal: 0x7bff = 65504.0
        check(0x7bff, 65504.0, "largest +normal");
    }

    #[test]
    fn inf_nan() {
        check(0x7c00, f32::INFINITY, "+inf");
        check(0xfc00, f32::NEG_INFINITY, "-inf");
        let nan = f16_bits_to_f32(0x7e00);
        assert!(nan.is_nan(), "0x7e00 should be NaN");
    }
}

#[cfg(test)]
mod activation_tests {
    use super::{apply_activation, exp_clamped, softplus};
    use crate::model::{Activation, EXP_INPUT_CLAMP, SOFTPLUS_THRESHOLD};

    #[test]
    fn exp_is_clamped_and_finite() {
        let top = libm::expf(EXP_INPUT_CLAMP);
        let bottom = libm::expf(-EXP_INPUT_CLAMP);
        assert!(top.is_finite() && bottom > 0.0);
        assert_eq!(exp_clamped(f32::INFINITY).to_bits(), top.to_bits());
        assert_eq!(exp_clamped(f32::MAX).to_bits(), top.to_bits());
        assert_eq!(exp_clamped(1.0e6).to_bits(), top.to_bits());
        assert_eq!(exp_clamped(f32::NEG_INFINITY).to_bits(), bottom.to_bits());
        assert!(exp_clamped(f32::NAN).is_nan());
        assert_eq!(exp_clamped(0.0), 1.0);
        assert_eq!(exp_clamped(-0.0), 1.0);
        // A gated product with a zero gate stays exactly zero.
        assert_eq!(
            (0.0 * exp_clamped(f32::INFINITY)).to_bits(),
            0.0f32.to_bits()
        );
    }

    #[test]
    fn exp_matches_f64_reference() {
        let mut x = -EXP_INPUT_CLAMP;
        while x <= EXP_INPUT_CLAMP {
            let got = exp_clamped(x) as f64;
            let want = (x as f64).exp();
            assert!(
                ((got - want) / want).abs() < 2.0e-7,
                "exp({x}) = {got}, want {want}"
            );
            x += 0.0137;
        }
    }

    #[test]
    fn softplus_matches_reference_and_threshold() {
        let mut x = -40.0f32;
        while x <= SOFTPLUS_THRESHOLD {
            let got = softplus(x) as f64;
            let want = (x as f64).exp().ln_1p();
            assert!(
                (got - want).abs() <= 2.0e-7 * want.max(1e-30),
                "softplus({x}) = {got}, want {want}"
            );
            assert!(got >= 0.0);
            x += 0.0173;
        }
        // Above the threshold: identity, exactly.
        for x in [20.000002f32, 25.0, 1.0e10, f32::MAX, f32::INFINITY] {
            assert_eq!(softplus(x).to_bits(), x.to_bits());
        }
        assert_eq!(softplus(-200.0), 0.0);
        assert_eq!(softplus(f32::NEG_INFINITY), 0.0);
        assert!(softplus(f32::NAN).is_nan());
    }

    #[test]
    fn apply_activation_covers_every_variant() {
        let src = [-2.0f32, -0.0, 0.0, 0.5, 30.5];
        for act in [
            Activation::Identity,
            Activation::Relu,
            Activation::LeakyRelu,
            Activation::Exp,
            Activation::Softplus,
        ] {
            let mut buf = src;
            apply_activation(&mut buf, act);
            for (i, (&got, &x)) in buf.iter().zip(&src).enumerate() {
                let want = match act {
                    Activation::Identity => x,
                    Activation::Relu => {
                        if x < 0.0 {
                            0.0
                        } else {
                            x
                        }
                    }
                    Activation::LeakyRelu => {
                        if x < 0.0 {
                            x * 0.01
                        } else {
                            x
                        }
                    }
                    Activation::Exp => exp_clamped(x),
                    Activation::Softplus => softplus(x),
                };
                assert_eq!(got.to_bits(), want.to_bits(), "{act:?}[{i}]");
            }
        }
        // ReLU keeps -0.0 (as v3 always has).
        let mut z = [-0.0f32];
        apply_activation(&mut z, Activation::Relu);
        assert_eq!(z[0].to_bits(), (-0.0f32).to_bits());
    }
}
