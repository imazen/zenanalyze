//! Marginal-cost measurement for the HVS-derived features added
//! 2026-05-17.
//!
//! Compares two queries:
//!
//! - `baseline`: the existing Tier-1 FULL features (Variance,
//!   Colourfulness, EdgeSlopeStdev, LaplacianVariance) plus the
//!   Tier-3 DCT means (AqMapMean / NoiseFloorY) — i.e. everything
//!   already paid for to make the HVS additions piggyback feasible.
//!
//! - `hvs`: `baseline` PLUS the five new HVS features.
//!
//! The delta is the marginal cost of computing the new features on
//! top of an already-paid Tier 1 + Tier 3 pass.

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

fn time_query(buf: &[u8], w: u32, h: u32, q: &AnalysisQuery, iters: usize) -> f64 {
    let stride = (w as usize) * 3;
    // Warmup
    for _ in 0..5 {
        let s = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, q).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let s = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, q).unwrap();
    }
    let dt = t0.elapsed();
    dt.as_secs_f64() * 1e6 / iters as f64
}

fn baseline_set() -> FeatureSet {
    let mut s = FeatureSet::new();
    s = s.with(AnalysisFeature::Variance);
    s = s.with(AnalysisFeature::EdgeDensity);
    // Tier-1 FULL accumulators already paid for.
    s = s.with(AnalysisFeature::Colourfulness);
    s = s.with(AnalysisFeature::LaplacianVariance);
    s = s.with(AnalysisFeature::EdgeSlopeStdev);
    // Tier-3 DCT pass already running.
    s = s.with(AnalysisFeature::AqMapMean);
    s = s.with(AnalysisFeature::AqMapStd);
    s = s.with(AnalysisFeature::NoiseFloorY);
    s = s.with(AnalysisFeature::HighFreqEnergyRatio);
    s
}

fn run_size(w: u32, h: u32, iters: usize, label: &str) {
    let buf = make_random((w * h * 3) as usize);
    let base = baseline_set();
    let mut hvs = base;
    hvs = hvs.with(AnalysisFeature::ChromaLumaCovarianceCb);
    hvs = hvs.with(AnalysisFeature::ChromaLumaCovarianceCr);
    hvs = hvs.with(AnalysisFeature::InfoWeightMean);
    hvs = hvs.with(AnalysisFeature::InfoWeightP90);
    hvs = hvs.with(AnalysisFeature::OrientationEnergyRatio);

    let t_base = time_query(&buf, w, h, &AnalysisQuery::new(base), iters);
    let t_hvs = time_query(&buf, w, h, &AnalysisQuery::new(hvs), iters);
    let delta_us = t_hvs - t_base;
    let mp = (w * h) as f64 / 1_000_000.0;
    println!(
        "{label}: {w}×{h} ({mp:.2} MP)  baseline {t_base:>7.1} µs  +HVS5 {t_hvs:>7.1} µs  Δ = {delta_us:>+7.1} µs  ({:>+5.2} ms/MP)",
        delta_us / 1000.0 / mp
    );
}

fn main() {
    println!("baseline = Tier-1 FULL + Tier-3 DCT pass already running");
    println!(
        "HVS5 = ChromaLumaCovariance{{Cb,Cr}} + InfoWeight{{Mean,P90}} + OrientationEnergyRatio"
    );
    println!();
    run_size(64, 64, 200, "tiny");
    run_size(256, 256, 100, "small");
    run_size(1024, 1024, 30, "medium");
    run_size(2048, 2048, 10, "large");
    run_size(4096, 4096, 5, "huge");
}
