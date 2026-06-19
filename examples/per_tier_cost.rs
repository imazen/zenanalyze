//! Per-TIER cost — the right granularity (features share combined passes, so
//! per-feature cost is meaningless: requesting any one feature runs its whole
//! tier). Each row requests ONE representative feature that gates a tier, so the
//! delta from the tier1 baseline is that tier's added shared-pass cost.
//!
//! Run: cargo run --release --features experimental,hdr --example per_tier_cost

use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisFeature as AF, AnalysisQuery, FeatureSet};
use zenbench::prelude::*;
use zenpixels::{PixelDescriptor, PixelSlice};

fn img(n: usize, seed: u32) -> &'static [u8] {
    let mut s = seed;
    let v: Vec<u8> = (0..n * 3)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 16) as u8
        })
        .collect();
    Box::leak(v.into_boxed_slice())
}

fn main() {
    // One representative feature per tier (each gates a distinct pass).
    let tiers: &[(&str, AF)] = &[
        ("tier1(variance)", AF::Variance),
        ("tier2(cb_sharpness)", AF::CbSharpness),
        ("tier3_hist(entropy)", AF::LumaHistogramEntropy),
        ("tier3_dct(hf_ratio)", AF::HighFreqEnergyRatio),
        ("palette(distinct)", AF::DistinctColorBins),
        ("alpha(present)", AF::AlphaPresent),
        ("depth(peak_nits)", AF::PeakLuminanceNits),
    ];

    let result = zenbench::run(|suite| {
        for (sz, side) in [("1MP", 1024u32), ("4MP", 2048u32)] {
            let n = (side * side) as usize;
            let buf = img(n, 0xC0FFEE ^ n as u32);
            let stride = (side * 3) as usize;
            let mk = move || {
                PixelSlice::new(buf, side, side, stride, PixelDescriptor::RGB8_SRGB).unwrap()
            };

            suite.compare(format!("per_tier_{sz}"), |g| {
                g.config().max_rounds(40);
                g.throughput(Throughput::Elements(n as u64));
                for (label, feat) in tiers {
                    let q = AnalysisQuery::new(FeatureSet::just(*feat));
                    g.bench(*label, move |b| {
                        b.iter(|| black_box(analyze_features(black_box(mk()), &q)))
                    });
                }
                let q_all = AnalysisQuery::new(FeatureSet::SUPPORTED);
                g.bench("ALL", move |b| {
                    b.iter(|| black_box(analyze_features(black_box(mk()), &q_all)))
                });
            });
        }
    });
    zenbench::postprocess_result(&result);
    let out = "/mnt/v/output/imazen-26-features/per_tier_cost_2026-06-18.json";
    let _ = result.save(out);
    eprintln!("saved {out}");
}
