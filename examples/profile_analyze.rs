//! Harness for profiling the full feature pass
//! (`analyze_features(FeatureSet::SUPPORTED)`).
//!
//! Two modes:
//!
//! * **callgrind / DHAT** (function attribution, heap) — single size, few iters:
//!   ```text
//!   cargo build --release --features experimental,hdr --example profile_analyze
//!   valgrind --tool=callgrind --cache-sim=no --branch-sim=no \
//!     --callgrind-out-file=/tmp/cg.out target/release/examples/profile_analyze 3 1024
//!   callgrind_annotate /tmp/cg.out
//!   ```
//!
//! * **wall-clock size sweep** (α + β·pixels fit) — pass `sweep`:
//!   ```text
//!   target/release/examples/profile_analyze sweep
//!   ```
//!
//! Args: `profile_analyze [iters] [side]` or `profile_analyze sweep`.
//! No `target-cpu=native` (release default) — runtime SIMD dispatch is what
//! ships.

use std::hint::black_box;
use std::time::Instant;
use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

/// Deterministic pseudo-random RGB8 of `side × side`.
fn make_image(side: u32) -> Vec<u8> {
    let n = (side * side) as usize;
    let mut s = 0xC0FFEEu32;
    (0..n * 3)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 16) as u8
        })
        .collect()
}

/// Run `iters` analyze passes over a `side × side` image; return ns/call.
fn time_pass(side: u32, iters: usize, q: &AnalysisQuery) -> f64 {
    let buf = make_image(side);
    let stride = (side * 3) as usize;
    // Warmup (page-in, branch-predictor, code cache).
    for _ in 0..3 {
        let slice = PixelSlice::new(&buf, side, side, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let _ = black_box(analyze_features(black_box(slice), q));
    }
    let t = Instant::now();
    for _ in 0..iters {
        let slice = PixelSlice::new(
            black_box(&buf),
            side,
            side,
            stride,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap();
        let _ = black_box(analyze_features(black_box(slice), black_box(q)));
    }
    t.elapsed().as_nanos() as f64 / iters as f64
}

fn main() {
    let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let arg1 = std::env::args().nth(1).unwrap_or_default();

    // Real-image mode: `profile_analyze <path.jpg> [iters]`. Random pixels bias
    // the palette/DCT hotspots (random saturates the 32K colour table → maximal
    // scatter cache-misses); a decoded photo gives the true content profile.
    if std::path::Path::new(&arg1).is_file() {
        let img = image::open(&arg1).expect("decode image").to_rgb8();
        let (w, h) = img.dimensions();
        let buf = img.into_raw();
        let stride = (w * 3) as usize;
        let iters: usize = std::env::args()
            .nth(2)
            .and_then(|a| a.parse().ok())
            .unwrap_or(3);
        for _ in 0..iters {
            let slice =
                PixelSlice::new(black_box(&buf), w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
            let _ = black_box(analyze_features(black_box(slice), black_box(&q)));
        }
        eprintln!(
            "profiled real image {arg1} ({w}×{h}, {} Mpx) ×{iters}",
            (w as f64 * h as f64) / 1e6
        );
        return;
    }

    if arg1 == "sweep" {
        // Sizes spanning tiny (fixed-overhead-dominated) → 4K (per-pixel-
        // dominated). Iters scaled down as size grows to keep total time sane.
        let sizes: [(u32, usize); 6] = [
            (64, 2000),
            (128, 1000),
            (256, 400),
            (512, 120),
            (1024, 40),
            (2048, 10),
        ];
        println!(
            "{:>6}  {:>12}  {:>10}  {:>12}",
            "side", "ns/call", "Mpx", "ns/px"
        );
        let mut pts: Vec<(f64, f64)> = Vec::new(); // (pixels, ns/call)
        for (side, iters) in sizes {
            let ns = time_pass(side, iters, &q);
            let px = (side as f64) * (side as f64);
            println!(
                "{side:>6}  {ns:>12.0}  {:>10.3}  {:>12.3}",
                px / 1e6,
                ns / px
            );
            pts.push((px, ns));
        }
        // The cost is NOT globally α + β·px: each tier subsamples above its own
        // budget (tier3 ≈ 1024 blocks ≈ 256², tier1 ≈ 500K px ≈ 707²), so above
        // those knees the work is capped and ns/px falls as 1/px. A global
        // linear fit is the wrong model. Fit α + β·px only on the two smallest
        // sizes, where every tier is still at full density — that β is the true
        // per-pixel cost; the larger sizes are the sampling-capped regime.
        let (x0, y0) = pts[0];
        let (x1, y1) = pts[1];
        let beta = (y1 - y0) / (x1 - x0);
        let alpha = y0 - beta * x0;
        println!(
            "\nfull-density fit (≤128², every tier uncapped): ns/call ≈ {:.1} µs + {beta:.2} ns/px · pixels",
            alpha / 1000.0
        );
        println!(
            "  → fixed-overhead α ≈ {:.1} µs  |  full-density per-pixel β ≈ {beta:.1} ns/px",
            alpha / 1000.0
        );
        println!(
            "  larger sizes are sampling-capped: ns/px falls (256²→2048²: {:.1}→{:.1}) as tier budgets engage.",
            pts[2].1 / pts[2].0,
            pts[5].1 / pts[5].0
        );
        return;
    }

    // callgrind / DHAT single-size mode.
    let iters: usize = arg1.parse().ok().unwrap_or(30);
    let side: u32 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1024);
    let ns = time_pass(side, iters, &q);
    eprintln!(
        "{side}×{side}: {ns:.0} ns/call ({:.3} ns/px)",
        ns / (side as f64 * side as f64)
    );
}
