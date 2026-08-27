//! SIMD-tier isolation: the native top tier vs the same code forced to scalar.
//!
//! zenanalyze's hot kernels are the tier-1 feature extractors in `src/tier1.rs`,
//! declared `#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]`. This
//! crate had no benchmarks at all, so nothing measured what those are worth on
//! any architecture — a kernel slower than its own scalar fallback would be
//! invisible. (The same gap in linear-srgb was hiding a real regression.)
//!
//! Run: `cargo bench --bench tier_isolation`
//! Do NOT build with `-C target-cpu=native`: that pins the tier at compile
//! time, after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenbench::prelude::*;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

/// Noise + patches. A gradient would give degenerate DCT/edge statistics and
/// understate exactly the high-frequency feature extractors being measured.
fn make_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            rgb[i + 1] = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            rgb[i + 2] = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
        }
    }
    rgb
}

fn bench_tiers(suite: &mut Suite) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!(
            "[tier_isolation] no toggleable SIMD tier on this target, or the tier is \
             compile-time guaranteed (drop -C target-cpu=native). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_isolation] comparing {TIER_NAME} vs forced scalar");

    // NOTE: the default `experimental` feature raises the sampling budget to
    // full (pixel_budget = usize::MAX). With it off, extraction stride-samples
    // to ~500k pixels and the 1024x1024 and 2048x2048 cases would do the same
    // amount of work — which would silently flatten the size axis.
    for &(label, w, h) in &[("512x512", 512usize, 512usize), ("2048x2048", 2048, 2048)] {
        let rgb: &'static [u8] = Box::leak(make_rgb(w, h).into_boxed_slice());
        let query: &'static AnalysisQuery =
            Box::leak(Box::new(AnalysisQuery::new(FeatureSet::SUPPORTED)));
        suite.compare(format!("analyze/{label}"), |g| {
            g.throughput(Throughput::Bytes((w * h * 3) as u64));
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.with_input(move || set_simd(simd)).run(move |_| {
                        zenanalyze::analyze_features_rgb8(rgb, w as u32, h as u32, query)
                    })
                });
            }
        });
    }
    set_simd(true);
}

zenbench::main!(bench_tiers);
