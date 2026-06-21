//! Minimal harness for callgrind / flamegraph profiling of the full feature pass
//! (`analyze_features(FeatureSet::SUPPORTED)`), so the non-transcendental hotspots
//! (RowStream traversal, Tier-1 SIMD reductions, Tier-3 DCT, gamut) show up at the
//! function level.
//!
//! Callgrind (instruction-level, deterministic; no `target-cpu=native`):
//!   cargo build --release --features experimental,hdr --example profile_analyze
//!   valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out \
//!     target/release/examples/profile_analyze 3
//!   callgrind_annotate /tmp/cg.out
//!
//! Arg 1 = iterations (default 30 for flamegraph; use ~3 under callgrind).

use std::hint::black_box;
use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

fn main() {
    let side = 1024u32;
    let n = (side * side) as usize;
    // Deterministic pseudo-random RGB8 (matches per_tier_cost's content).
    let mut s = 0xC0FFEEu32;
    let buf: Vec<u8> = (0..n * 3)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 16) as u8
        })
        .collect();
    let stride = (side * 3) as usize;
    let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(30);

    for _ in 0..iters {
        let slice = PixelSlice::new(
            black_box(&buf),
            side,
            side,
            stride,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap();
        let _ = black_box(analyze_features(black_box(slice), black_box(&q)));
    }
}
