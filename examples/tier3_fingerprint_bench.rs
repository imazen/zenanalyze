//! Microbench for the SIMD-optimized tier 3 fingerprint kernels —
//! `block_signature_dhash` and `quant_survival`. Times `analyze_features`
//! end-to-end at 1 / 4 / 16 Mpx with the requested feature set narrowed
//! to just those two so the per-call cost reflects the kernels under
//! test (plus the tier 3 DCT pass that produces their inputs).

use std::time::Instant;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

fn make_random(n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    let mut s: u32 = 0xdeadbeef;
    for b in buf.iter_mut() {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        *b = (s >> 16) as u8;
    }
    buf
}

fn bench(label: &str, iters: usize, mut f: impl FnMut()) {
    for _ in 0..3 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let dt = t0.elapsed();
    let avg_us = dt.as_micros() as f64 / iters as f64;
    println!("{:60} {:8.1} µs/call ({} iters)", label, avg_us, iters);
}

fn run_at(width: u32, height: u32, query: &AnalysisQuery, label: &str) {
    let n = (width as usize) * (height as usize) * 3;
    let buf = make_random(n);
    let stride = (width as usize) * 3;
    let iters = match width * height {
        x if x <= 1 << 20 => 200,
        x if x <= 1 << 22 => 50,
        _ => 10,
    };
    bench(&format!("{label} ({width}×{height})"), iters, || {
        let s = PixelSlice::new(&buf, width, height, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, query).unwrap();
    });
}

fn main() {
    // Three queries to measure the DCT-gate impact:
    let fingerprint_set = FeatureSet::new()
        .with(AnalysisFeature::PatchFractionFast)
        .with(AnalysisFeature::QuantSurvivalY)
        .with(AnalysisFeature::QuantSurvivalUv);
    let entropy_only = FeatureSet::new().with(AnalysisFeature::LumaHistogramEntropy);
    let entropy_and_dct = FeatureSet::new()
        .with(AnalysisFeature::LumaHistogramEntropy)
        .with(AnalysisFeature::HighFreqEnergyRatio);

    let q_fp = AnalysisQuery::new(fingerprint_set);
    let q_entropy = AnalysisQuery::new(entropy_only);
    let q_both = AnalysisQuery::new(entropy_and_dct);

    for (w, h) in &[(1024, 1024), (2048, 2048), (4096, 4096)] {
        run_at(*w, *h, &q_fp, "fingerprint+quant_survival (DCT on)");
        run_at(
            *w,
            *h,
            &q_entropy,
            "luma_histogram_entropy ONLY (DCT skipped)",
        );
        run_at(*w, *h, &q_both, "entropy + HighFreqEnergyRatio (DCT on)");
    }
}
