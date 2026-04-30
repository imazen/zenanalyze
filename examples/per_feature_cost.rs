//! Per-feature timing harness: measures the wall-clock cost of each
//! `AnalysisFeature` along two axes —
//!
//! * **Solo cost**: time to run `analyze_features` with only that feature
//!   requested. Captures all dependency overhead (Tier 1/2/3 dispatch,
//!   alpha pass, depth pass, derived likelihoods) the feature pulls in.
//!   Answers "if I only need this feature, what does it cost?"
//!
//! * **Leave-one-out (LOO) marginal**: time delta between
//!   `analyze_features(SUPPORTED)` and `analyze_features(SUPPORTED \ F)`.
//!   Answers "if I'm already computing everything, what does this
//!   feature add?" Negative or near-zero values mean the feature shares
//!   passes with others and adds no marginal cost.
//!
//! Both measurements at 1 MP and 4 MP. Output: markdown table on
//! stdout, sorted by 4 MP solo cost descending.
//!
//! Build:
//!
//! ```sh
//! cargo build --release --features experimental,composites \
//!     --example per_feature_cost
//! ```
//!
//! Run:
//!
//! ```sh
//! ./target/release/examples/per_feature_cost > /tmp/per_feature_cost.md
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

fn time_query(buf: &[u8], width: u32, height: u32, query: &AnalysisQuery, iters: usize) -> f64 {
    let stride = (width as usize) * 3;
    // Warmup
    for _ in 0..3 {
        let s = PixelSlice::new(buf, width, height, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, query).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let s = PixelSlice::new(buf, width, height, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = zenanalyze::analyze_features(s, query).unwrap();
    }
    let dt = t0.elapsed();
    dt.as_micros() as f64 / iters as f64
}

struct Sample {
    name: &'static str,
    solo_1m: f64,
    solo_4m: f64,
    loo_1m: f64,
    loo_4m: f64,
}

fn run_at(buf: &[u8], w: u32, h: u32, iters: usize) -> (f64, Vec<(AnalysisFeature, f64, f64)>) {
    let supported = FeatureSet::SUPPORTED;
    let baseline = time_query(buf, w, h, &AnalysisQuery::new(supported), iters);
    let mut rows = Vec::new();
    for f in supported.iter() {
        let solo = time_query(buf, w, h, &AnalysisQuery::new(FeatureSet::new().with(f)), iters);
        let loo = time_query(buf, w, h, &AnalysisQuery::new(supported.without(f)), iters);
        rows.push((f, solo, loo));
    }
    (baseline, rows)
}

fn main() {
    let buf_1m = make_random(1024 * 1024 * 3);
    let buf_4m = make_random(2048 * 2048 * 3);

    eprintln!("warming up + measuring 1 MP baseline...");
    let (base_1m, rows_1m) = run_at(&buf_1m, 1024, 1024, 30);
    eprintln!("1 MP baseline: {:.0} µs", base_1m);

    eprintln!("warming up + measuring 4 MP baseline...");
    let (base_4m, rows_4m) = run_at(&buf_4m, 2048, 2048, 15);
    eprintln!("4 MP baseline: {:.0} µs", base_4m);

    // Stitch
    let mut samples: Vec<Sample> = Vec::with_capacity(rows_1m.len());
    for ((f1, s1, l1), (f4, s4, l4)) in rows_1m.iter().zip(rows_4m.iter()) {
        debug_assert_eq!(f1, f4);
        samples.push(Sample {
            name: f1.name(),
            solo_1m: *s1,
            solo_4m: *s4,
            loo_1m: base_1m - *l1,
            loo_4m: base_4m - *l4,
        });
    }

    // Sort by 4 MP solo cost descending.
    samples.sort_by(|a, b| b.solo_4m.partial_cmp(&a.solo_4m).unwrap());

    println!("# Per-feature timing (zenanalyze, 7950X release build)");
    println!();
    println!("Baselines: 1 MP `SUPPORTED` = {:.0} µs, 4 MP `SUPPORTED` = {:.0} µs.", base_1m, base_4m);
    println!();
    println!("- **Solo**: time when this feature is the only one requested. Includes all tier dispatch + dependencies.");
    println!("- **LOO (leave-one-out)**: `SUPPORTED` baseline minus `SUPPORTED \\ F` time. Negative = noise; 0 = shares pass with another feature; positive = unique cost in the full pipeline.");
    println!();
    println!("| Feature | Solo 1 MP (µs) | Solo 4 MP (µs) | LOO 1 MP (µs) | LOO 4 MP (µs) |");
    println!("|---|---:|---:|---:|---:|");
    for s in &samples {
        println!(
            "| `{}` | {:.0} | {:.0} | {:+.0} | {:+.0} |",
            s.name, s.solo_1m, s.solo_4m, s.loo_1m, s.loo_4m
        );
    }
}
