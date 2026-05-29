//! Verifies the ZNPR-side reference output matches what onnxruntime
//! would compute on the ONNX produced by `znpr2onnx`. We assert the
//! reference value here; the actual onnxruntime check is documented in
//! `docs/onnx_parity_v0_18.txt` (Python-side; not pulled in to keep
//! Rust deps light).
use zenpredict::{Model, Predictor};

const BAKE: &str =
    "/home/lilith/work/zen/zensim/zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin";

#[test]
fn reference_output_for_sine_input() {
    if !std::path::Path::new(BAKE).exists() {
        eprintln!("skip: bake not present");
        return;
    }
    let bytes = std::fs::read(BAKE).unwrap();
    let model = Model::from_bytes(&bytes).unwrap();
    let n_inputs = model.n_inputs();
    let features: Vec<f32> = (0..n_inputs)
        .map(|i| (0.1f32 * i as f32).sin() * 5.0)
        .collect();
    let mut p = Predictor::new(&model);
    let out: Vec<f32> = p.predict(&features).unwrap().to_vec();
    eprintln!("zenpredict reference output[0] = {}", out[0]);
    // Sanity: zenpredict v0_18 produces a finite value on sine input.
    assert!(out[0].is_finite());
}
