//! Grayscale-classifier worst-case timing: measures `is_grayscale`
//! against (a) random RGB content (early-exits on row 1) and (b) a
//! truly grayscale image (walks every row, no early exit).
//!
//! Build:
//! ```sh
//! cargo build --release --example grayscale_perf
//! ```

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

fn make_truly_gray(w: usize, h: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(w * h * 3);
    let mut s: u32 = 0xc0ffee;
    for _ in 0..(w * h) {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let v = (s >> 16) as u8;
        buf.extend_from_slice(&[v, v, v]);
    }
    buf
}

fn time_one(buf: &[u8], w: u32, h: u32, iters: usize, query: &AnalysisQuery) -> f64 {
    let stride = (w as usize) * 3;
    for _ in 0..3 {
        let s = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, query).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let s = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, query).unwrap();
    }
    (t0.elapsed().as_micros() as f64) / iters as f64
}

fn run(label: &str, w: u32, h: u32, gray: bool, iters: usize) {
    let buf = if gray {
        make_truly_gray(w as usize, h as usize)
    } else {
        make_random((w as usize) * (h as usize) * 3)
    };

    let q_only_gray = AnalysisQuery::new(FeatureSet::new().with(AnalysisFeature::IsGrayscale));
    let q_empty = AnalysisQuery::new(FeatureSet::new());

    let solo = time_one(&buf, w, h, iters, &q_only_gray);
    let baseline = time_one(&buf, w, h, iters, &q_empty);
    let kernel = solo - baseline;

    println!(
        "{:<40} {}x{}  empty={:.0}µs  solo={:.0}µs  kernel≈{:.0}µs",
        label, w, h, baseline, solo, kernel
    );
}

fn main() {
    println!("=== Colored / random RGB (early-exit row 1) ===");
    run("colored 1 MP", 1024, 1024, false, 100);
    run("colored 4 MP", 2048, 2048, false, 50);
    run("colored 16 MP (4K)", 4096, 4096, false, 20);

    println!();
    println!("=== Truly grayscale (R=G=B everywhere — walks every pixel) ===");
    run("grayscale 1 MP", 1024, 1024, true, 100);
    run("grayscale 4 MP", 2048, 2048, true, 50);
    run("grayscale 16 MP (4K)", 4096, 4096, true, 20);
}
