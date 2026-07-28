//! Per-TIER SIMD isolation: each analysis pass measured on its own.
//!
//! `tier_isolation.rs` runs `analyze_features_rgb8` with the full `SUPPORTED`
//! set. That is a single aggregate number, so a tier whose kernels are SLOWER
//! than their own scalar fallback is invisible — the faster tiers average it
//! away. That exact failure mode was found and fixed in garb, zensim, zentone,
//! zenpng and zenresize during the 2026-07-28 aarch64 sweep.
//!
//! Requesting one tier's `FeatureSet` at a time gates the other passes off
//! (per the tier architecture in CLAUDE.md), so each measurement is dominated
//! by that tier's kernels.
//!
//! NOTE: on aarch64 NEON is BASELINE, so the "scalar" arm is still fully
//! autovectorized by LLVM. A ratio near 1.00 does NOT mean SIMD is missing —
//! it means both arms compiled to equivalent work.
//!
//! Run: `cargo bench --bench tier_kernels`
//! Do NOT pass `-C target-cpu=native`: that pins the tier at compile time,
//! after which it cannot be disabled and this bench skips rather than
//! silently reporting the SIMD path under both labels.

use zenanalyze::__bench_sets as sets;
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
/// understate exactly the high-frequency extractors being measured.
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
            "[tier_kernels] no toggleable SIMD tier on this target, or the tier \
             is compile-time guaranteed (drop -C target-cpu=native). Skipping."
        );
        return;
    }
    set_simd(true);
    eprintln!("[tier_kernels] comparing {TIER_NAME} vs forced scalar, per analysis pass");

    // 1024x1024: past the stride-sampling budget so the size axis is real,
    // and large enough that per-call overhead is not the story.
    let (w, h) = (1024usize, 1024usize);
    let rgb: &'static [u8] = Box::leak(make_rgb(w, h).into_boxed_slice());

    let passes: &[(&str, FeatureSet)] = &[
        ("tier1_full", sets::TIER1_FULL),
        ("tier1_extras", sets::TIER1_EXTRAS),
        ("tier2_chroma", sets::TIER2),
        ("tier3_dct", sets::TIER3),
        ("palette", sets::PALETTE),
        ("depth", sets::DEPTH),
    ];

    for &(label, fs) in passes {
        let query: &'static AnalysisQuery = Box::leak(Box::new(AnalysisQuery::new(fs)));
        suite.compare(format!("pass/{label}"), |g| {
            g.throughput(Throughput::Bytes((w * h * 3) as u64));
            for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
                g.bench(arm, move |b| {
                    b.iter(move || {
                        // Inside the closure: zenbench interleaves the arms.
                        set_simd(simd);
                        zenanalyze::analyze_features_rgb8(rgb, w as u32, h as u32, query)
                    })
                });
            }
        });
    }
    set_simd(true);
}

zenbench::main!(bench_tiers);
