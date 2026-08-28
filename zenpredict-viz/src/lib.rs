//! WASM bridge for the zenpredict-viz interactive bake visualizer.
//!
//! Exposes three entry points:
//!
//! - [`parse_bake`] — parses ZNPR v3 bytes and returns a JSON-serializable
//!   summary (header, layer dims, scaler, L0 importance, metadata key list).
//!   Forms the data backbone for the "summary", "scaler", "importance",
//!   and "weights" panels.
//!
//! - [`forward_with_taps`] — runs the standardize → layer-by-layer
//!   forward pass and returns intermediate activations at every stage.
//!   Backs the "live forward pass" panel's waterfall.
//!
//! - [`layer_weights`] — returns a single layer's weights (dequantized
//!   to f32 if I8 storage). Backs the "weights" panel's per-layer heatmap.
//!
//! The Rust side handles standardize + MLP layers (the math that lives
//! in `zenpredict::inference`). Zensim-side calibration stages
//! (`zentrain.tanh_output_head`, `zentrain.output_calibration_spline`,
//! `zentrain.per_sample_alpha_head`, `zentrain.per_codec_calibration`)
//! are decoded in JS from the metadata blob — their wire formats are
//! small and well-documented, and keeping them out of this crate keeps
//! the WASM module dependency-free of zensim.

use serde::Serialize;
use wasm_bindgen::prelude::*;
use zenpredict::output_spec::apply_spec;
use zenpredict::{
    Activation, FeatureTransform, LEAKY_RELU_ALPHA, LayerView, Model, WeightStorage,
    apply_feature_transforms, f16_bits_to_f32,
};

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "panic_console")]
    console_error_panic_hook::set_once();
}

#[derive(Serialize)]
pub struct BakeSummary {
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_layers: usize,
    pub schema_hash: u64,
    pub bake_bytes: usize,
    pub scaler_mean: Vec<f32>,
    pub scaler_scale: Vec<f32>,
    pub layers: Vec<LayerSummary>,
    pub metadata_keys: Vec<MetadataKey>,
    pub l0_importance: Vec<f32>,
    /// Feature names from `zentrain.feature_columns` when the bake carries
    /// them (picker bakes do); empty otherwise. Index-aligned with the
    /// caller's feature vector, so the UI prefers these over the zensim
    /// n_inputs-based layout and the `f<idx>` fallback.
    pub feature_names: Vec<String>,
    /// Number of features the CALLER passes: `n_inputs` unless the bake's
    /// `feature_transforms` change the width (Drop / expanders).
    pub caller_input_width: usize,
    pub has_feature_transforms: bool,
    pub has_output_specs: bool,
}

#[derive(Serialize)]
pub struct LayerSummary {
    pub idx: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: String,
    pub dtype: String,
    pub bias_min: f32,
    pub bias_max: f32,
    pub bias_mean: f32,
    pub weight_min: f32,
    pub weight_max: f32,
    pub weight_mean: f32,
    pub i8_scales: Option<Vec<f32>>,
}

#[derive(Serialize)]
pub struct MetadataKey {
    pub key: String,
    pub kind: String,
    pub value_len: usize,
    /// Hex-encoded raw bytes — the JS side decodes known keys
    /// (`zentrain.tanh_output_head` etc.) into typed structures.
    pub value_hex: String,
}

#[wasm_bindgen]
pub fn parse_bake(bytes: &[u8]) -> Result<JsValue, JsError> {
    let summary = parse_bake_native(bytes).map_err(|e| JsError::new(&e))?;
    serde_wasm_bindgen::to_value(&summary).map_err(|e| JsError::new(&format!("{e}")))
}

/// Native entry point used by tests + native callers.
pub fn parse_bake_native(bytes: &[u8]) -> Result<BakeSummary, String> {
    let model = Model::from_bytes(bytes).map_err(|e| format!("{e}"))?;
    Ok(build_summary(&model, bytes.len()))
}

fn build_summary(model: &Model, bake_bytes: usize) -> BakeSummary {
    let n_inputs = model.n_inputs();
    let n_outputs = model.n_outputs();
    let n_layers = model.n_layers();
    let schema_hash = model.schema_hash();
    let scaler_mean = model.scaler_mean().to_vec();
    let scaler_scale = model.scaler_scale().to_vec();

    let mut l0_importance = vec![0.0_f32; n_inputs];
    let mut layers: Vec<LayerSummary> = Vec::with_capacity(n_layers);
    for (idx, layer) in model.layers().enumerate() {
        let dtype = match &layer.weights {
            WeightStorage::F32(_) => "f32",
            WeightStorage::F16(_) => "f16",
            WeightStorage::I8 { .. } => "i8",
        };
        let (weight_min, weight_max, weight_mean) = weight_stats(&layer);
        let bias_min = layer.biases.iter().copied().fold(f32::INFINITY, f32::min);
        let bias_max = layer
            .biases
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let bias_mean = layer.biases.iter().copied().sum::<f32>() / layer.biases.len() as f32;
        let i8_scales = if let WeightStorage::I8 { scales, .. } = &layer.weights {
            Some(scales.to_vec())
        } else {
            None
        };
        if idx == 0 {
            accumulate_l0_importance(&layer, &scaler_scale, &mut l0_importance);
        }
        layers.push(LayerSummary {
            idx,
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
            activation: activation_name(layer.activation).to_string(),
            dtype: dtype.to_string(),
            bias_min,
            bias_max,
            bias_mean,
            weight_min,
            weight_max,
            weight_mean,
            i8_scales,
        });
    }

    let metadata_keys = model
        .metadata()
        .iter()
        .map(|e| MetadataKey {
            key: e.key.to_string(),
            kind: format!("{:?}", e.kind),
            value_len: e.value.len(),
            value_hex: hex(e.value),
        })
        .collect();

    let feature_names: Vec<String> = model.feature_columns().map(str::to_string).collect();

    BakeSummary {
        n_inputs,
        n_outputs,
        n_layers,
        schema_hash,
        bake_bytes,
        scaler_mean,
        scaler_scale,
        layers,
        metadata_keys,
        l0_importance,
        feature_names,
        caller_input_width: model.caller_input_width(),
        has_feature_transforms: model.has_nontrivial_feature_transforms(),
        has_output_specs: model.has_output_specs(),
    }
}

fn activation_name(a: Activation) -> &'static str {
    match a {
        Activation::Identity => "identity",
        Activation::Relu => "relu",
        Activation::LeakyRelu => "leaky_relu",
        _ => "unknown",
    }
}

fn weight_stats(layer: &LayerView<'_>) -> (f32, f32, f32) {
    let in_dim = layer.in_dim;
    let out_dim = layer.out_dim;
    let n = in_dim * out_dim;
    let (mut min_v, mut max_v, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0_f64);
    let iter = WeightIter::new(layer);
    for v in iter {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
        sum += v as f64;
    }
    let mean = (sum / n as f64) as f32;
    (min_v, max_v, mean)
}

fn accumulate_l0_importance(layer: &LayerView<'_>, scaler_scale: &[f32], out: &mut [f32]) {
    let out_dim = layer.out_dim;
    // Σ_h |W0[i, h]| per input row, in f64 so the order of summation
    // cannot move the heatmap.
    let row_abs_sum = |row_idx: usize| -> f64 {
        let base = row_idx * out_dim;
        match &layer.weights {
            WeightStorage::F32(w) => w[base..base + out_dim]
                .iter()
                .map(|&v| (v as f64).abs())
                .sum(),
            WeightStorage::F16(w) => w[base..base + out_dim]
                .iter()
                .map(|&h| (f16_bits_to_f32(h) as f64).abs())
                .sum(),
            WeightStorage::I8 { weights, scales } => weights[base..base + out_dim]
                .iter()
                .zip(scales.iter())
                .map(|(&q, &s)| (q as f64 * s as f64).abs())
                .sum(),
        }
    };
    for (i, (dst, &scale)) in out.iter_mut().zip(scaler_scale).enumerate() {
        *dst = (row_abs_sum(i) * scale as f64) as f32;
    }
}

/// Iterate dequantized weight values in row-major (input-major) order.
struct WeightIter<'a> {
    layer: WeightView<'a>,
    pos: usize,
    end: usize,
}

enum WeightView<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    I8 {
        weights: &'a [i8],
        scales: &'a [f32],
        out_dim: usize,
    },
}

impl<'a> WeightIter<'a> {
    fn new(layer: &'a LayerView<'a>) -> Self {
        let end = layer.in_dim * layer.out_dim;
        let view = match &layer.weights {
            WeightStorage::F32(w) => WeightView::F32(w),
            WeightStorage::F16(w) => WeightView::F16(w),
            WeightStorage::I8 { weights, scales } => WeightView::I8 {
                weights,
                scales,
                out_dim: layer.out_dim,
            },
        };
        Self {
            layer: view,
            pos: 0,
            end,
        }
    }
}

impl<'a> Iterator for WeightIter<'a> {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.pos >= self.end {
            return None;
        }
        let p = self.pos;
        let v = match &self.layer {
            WeightView::F32(w) => w[p],
            WeightView::F16(w) => f16_bits_to_f32(w[p]),
            WeightView::I8 {
                weights,
                scales,
                out_dim,
            } => {
                let o = p % out_dim;
                weights[p] as f32 * scales[o]
            }
        };
        self.pos += 1;
        Some(v)
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(NIBBLE[(b >> 4) as usize] as char);
        s.push(NIBBLE[(b & 0x0f) as usize] as char);
    }
    s
}

const NIBBLE: &[u8; 16] = b"0123456789abcdef";

/// Per-stage forward-pass result for the live waterfall.
///
/// Stages are recorded in apply order: [optional feature_transforms] →
/// standardized → layer 0 pre/post → layer 1 pre/post → ... → final
/// raw output → [optional output_specs]. Zensim-specific post-MLP
/// stages (tanh pin, spline, per-codec) are not applied here — the JS
/// side handles them with the parsed metadata payloads.
///
/// The MLP math mirrors `zenpredict::inference` operation for
/// operation (same standardize, the same `f32::mul_add` SAXPY in the
/// same input-major order with the same zero-input skip, the same i8
/// `bias + scale · acc` epilogue, the same activations), so `output`
/// is bit-identical to `Predictor::predict` / `predict_transformed` —
/// `tests/forward_parity.rs` asserts equality of the bit patterns.
#[derive(Serialize)]
pub struct ForwardTaps {
    /// The caller's features after the bake's `feature_transforms`
    /// (`None` when the bake has none — the standardize stage then
    /// reads the caller's vector directly).
    pub transformed: Option<Vec<f32>>,
    pub standardized: Vec<f32>,
    pub layer_stages: Vec<LayerStage>,
    /// Raw MLP output — what `Predictor::predict` returns.
    pub output: Vec<f32>,
    /// `output` after the bake's per-output specs (transform, clamp,
    /// discrete snap, sentinel) — what `Predictor::predict_with_specs`
    /// returns; `None` per entry where the runtime reports "use the
    /// codec default". `None` overall when the bake has no specs.
    pub specs_applied: Option<Vec<Option<f32>>>,
}

#[derive(Serialize)]
pub struct LayerStage {
    pub idx: usize,
    /// Pre-activation values (matmul + bias, before activation function).
    pub pre_activation: Vec<f32>,
    /// Post-activation values (after activation). Equal to
    /// `pre_activation` when activation is `Identity` (typical for the
    /// final layer).
    pub post_activation: Vec<f32>,
}

#[wasm_bindgen]
pub fn forward_with_taps(bytes: &[u8], features: Vec<f32>) -> Result<JsValue, JsError> {
    let taps = forward_with_taps_native(bytes, &features).map_err(|e| JsError::new(&e))?;
    serde_wasm_bindgen::to_value(&taps).map_err(|e| JsError::new(&format!("{e}")))
}

/// Native entry point used by tests + native callers. Returns the
/// `ForwardTaps` struct directly instead of going through JsValue.
///
/// `features` is the CALLER-width vector (`Model::caller_input_width`),
/// exactly what a codec hands to `Predictor::predict_transformed`.
pub fn forward_with_taps_native(bytes: &[u8], features: &[f32]) -> Result<ForwardTaps, String> {
    let model = Model::from_bytes(bytes).map_err(|e| format!("{e}"))?;
    let n_inputs = model.n_inputs();
    let caller_width = model.caller_input_width();
    if features.len() != caller_width {
        return Err(format!(
            "feature length mismatch: expected {caller_width}, got {}",
            features.len()
        ));
    }

    // Stage 0 (optional): feature transforms, mirroring
    // `Predictor::predict_transformed` — scalar transforms per feature,
    // or the expanding pipeline when an expander variant is present.
    let transformed: Option<Vec<f32>> = match model.feature_transforms() {
        None => None,
        Some(transforms) => {
            if transforms.len() != features.len() {
                return Err(format!(
                    "feature_transforms length {} != caller width {}",
                    transforms.len(),
                    features.len()
                ));
            }
            let params = model.feature_transform_params();
            if model.has_expander_feature_transforms() {
                let params = params.ok_or_else(|| {
                    "bake has an expander feature transform but no feature_transform_params"
                        .to_string()
                })?;
                let mut dst = vec![0.0_f32; model.expanded_input_dim()];
                let mut offset = 0usize;
                for (i, t) in transforms.iter().enumerate() {
                    let p: &[f32] = &params[i];
                    let arity = t.output_arity(p);
                    let written =
                        t.apply_expanding(features[i], p, &mut dst[offset..offset + arity]);
                    debug_assert_eq!(written, arity);
                    offset += arity;
                }
                Some(dst)
            } else {
                let mut dst = vec![0.0_f32; features.len()];
                match params {
                    Some(params) => {
                        for i in 0..features.len() {
                            dst[i] = transforms[i].apply_with_params(features[i], &params[i]);
                        }
                    }
                    None => apply_feature_transforms(transforms, features, &mut dst)
                        .map_err(|e| format!("{e}"))?,
                }
                Some(dst)
            }
        }
    };
    let mlp_input: &[f32] = transformed.as_deref().unwrap_or(features);
    if mlp_input.len() != n_inputs {
        return Err(format!(
            "transformed width {} != model n_inputs {n_inputs}",
            mlp_input.len()
        ));
    }

    // Stage 1: standardize — `(x - mean) / scale`, zero scale ⇒ 1.0
    // (sklearn's `_handle_zeros_in_scale`), as in `inference::forward`.
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    let mut current: Vec<f32> = (0..n_inputs)
        .map(|i| {
            let s = scale[i];
            let safe_s = if s == 0.0 { 1.0 } else { s };
            (mlp_input[i] - mean[i]) / safe_s
        })
        .collect();
    let standardized = current.clone();

    // Stages 2..=n_layers: forward through each layer, recording pre/post-activation.
    let mut layer_stages: Vec<LayerStage> = Vec::with_capacity(model.n_layers());
    for (idx, layer) in model.layers().enumerate() {
        let out_dim = layer.out_dim;
        let in_dim = layer.in_dim;
        if current.len() != in_dim {
            return Err(format!(
                "layer {idx}: input width {} != in_dim {in_dim}",
                current.len()
            ));
        }
        let pre = layer_pre_activation(&layer, &current, in_dim, out_dim);
        let pre_activation = pre.clone();
        let mut post = pre;
        apply_activation(&mut post, layer.activation);
        layer_stages.push(LayerStage {
            idx,
            pre_activation,
            post_activation: post.clone(),
        });
        current = post;
    }

    // Stage n+1 (optional): per-output specs — `Predictor::predict_with_specs`.
    let specs_applied = if model.has_output_specs() {
        let pool = model.discrete_sets();
        Some(
            model
                .output_specs()
                .iter()
                .zip(current.iter())
                .map(|(spec, &raw)| apply_spec(spec, raw, pool).value())
                .collect(),
        )
    } else {
        None
    };

    Ok(ForwardTaps {
        transformed,
        standardized,
        layer_stages,
        output: current,
        specs_applied,
    })
}

/// `bias + Σ_i src[i] · W[i, :]`, accumulated exactly like
/// `zenpredict::inference::layer_forward`: input-major SAXPY with a
/// correctly-rounded `mul_add` per lane and the `src[i] == 0` skip
/// (which matters for ±inf / NaN weights), and for i8 the raw
/// accumulation followed by `bias + scale · acc`.
fn layer_pre_activation(
    layer: &LayerView<'_>,
    src: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    match &layer.weights {
        WeightStorage::F32(w) => {
            let mut dst = layer.biases.to_vec();
            for i in 0..in_dim {
                let s = src[i];
                if s == 0.0 {
                    continue;
                }
                let row = &w[i * out_dim..(i + 1) * out_dim];
                for o in 0..out_dim {
                    dst[o] = s.mul_add(row[o], dst[o]);
                }
            }
            dst
        }
        WeightStorage::F16(w) => {
            let mut dst = layer.biases.to_vec();
            for i in 0..in_dim {
                let s = src[i];
                if s == 0.0 {
                    continue;
                }
                let row = &w[i * out_dim..(i + 1) * out_dim];
                for o in 0..out_dim {
                    dst[o] = s.mul_add(f16_bits_to_f32(row[o]), dst[o]);
                }
            }
            dst
        }
        WeightStorage::I8 { weights, scales } => {
            let mut acc = vec![0.0_f32; out_dim];
            for i in 0..in_dim {
                let s = src[i];
                if s == 0.0 {
                    continue;
                }
                let row = &weights[i * out_dim..(i + 1) * out_dim];
                for o in 0..out_dim {
                    acc[o] = s.mul_add(row[o] as f32, acc[o]);
                }
            }
            (0..out_dim)
                .map(|o| layer.biases[o] + scales[o] * acc[o])
                .collect()
        }
    }
}

/// Same branch structure as `zenpredict::inference::apply_activation`.
/// An activation this viz build doesn't know about passes the
/// pre-activation values through unchanged; `activation_name` renders
/// "unknown" for the same variant, so post == pre signals "not
/// applied" rather than silently faking a curve.
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
        _ => {}
    }
}

/// Feature-transform tokens as the bake declares them (one per caller
/// feature, `zentrain.feature_transforms`), for the scaler panel's
/// "transforms" toggle. Empty when the bake has none.
#[wasm_bindgen]
pub fn feature_transform_tokens(bytes: &[u8]) -> Result<Vec<String>, JsError> {
    feature_transform_tokens_native(bytes).map_err(|e| JsError::new(&e))
}

/// Native entry point used by tests + native callers.
pub fn feature_transform_tokens_native(bytes: &[u8]) -> Result<Vec<String>, String> {
    let model = Model::from_bytes(bytes).map_err(|e| format!("{e}"))?;
    Ok(model
        .feature_transforms()
        .map(|ts| {
            ts.iter()
                .map(|t: &FeatureTransform| t.as_token().to_string())
                .collect()
        })
        .unwrap_or_default())
}

/// Return a single layer's dequantized weight matrix (in_dim × out_dim,
/// row-major as `[i * out_dim + o]`). For the "weights" panel.
#[wasm_bindgen]
pub fn layer_weights(bytes: &[u8], layer_idx: usize) -> Result<Vec<f32>, JsError> {
    layer_weights_native(bytes, layer_idx).map_err(|e| JsError::new(&e))
}

/// Native entry point used by tests + native callers.
pub fn layer_weights_native(bytes: &[u8], layer_idx: usize) -> Result<Vec<f32>, String> {
    let model = Model::from_bytes(bytes).map_err(|e| format!("{e}"))?;
    if layer_idx >= model.n_layers() {
        return Err(format!(
            "layer index {layer_idx} out of range (n_layers={})",
            model.n_layers()
        ));
    }
    let layer = model.layer(layer_idx);
    let n = layer.in_dim * layer.out_dim;
    let mut out = Vec::with_capacity(n);
    match &layer.weights {
        WeightStorage::F32(w) => out.extend_from_slice(w),
        WeightStorage::F16(w) => out.extend(w.iter().map(|&h| f16_bits_to_f32(h))),
        WeightStorage::I8 { weights, scales } => {
            for i in 0..layer.in_dim {
                let base = i * layer.out_dim;
                for o in 0..layer.out_dim {
                    out.push(weights[base + o] as f32 * scales[o]);
                }
            }
        }
    }
    Ok(out)
}
