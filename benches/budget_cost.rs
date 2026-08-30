//! What does fully-sampled analysis COST, measured at each size?
//!
//! The companion to `examples/budget_drift_grid` (which measures what fully
//! sampled analysis *changes*). Together they answer "should analysis
//! thoroughness scale with image size, and what would it cost?".
//!
//! Both arms go through the **same** entry point — `zenanalyze::__analyze_internal`,
//! the `#[doc(hidden)]` re-extraction backdoor — with only the two sampling
//! budgets differing. That matters: the public `analyze_features` dispatcher
//! skips whole passes whose outputs were not requested, while the internal path
//! runs every tier unconditionally, so timing one against the other would
//! confound "budget cost" with "dispatch cost". A third arm times the public
//! entry point at default budgets as the shipping reference point.
//!
//! Arms per size:
//!
//! | arm | what |
//! |---|---|
//! | `default` | `__analyze_internal` at 500 k px / 1 024 blocks — the shipping budgets |
//! | `full` | `__analyze_internal` at `usize::MAX` px / 262 144 blocks |
//! | `hffull` | `__analyze_internal` at 500 k px / 262 144 blocks — the hf knob alone, which the drift grid finds is the dominant driver |
//! | `public` | `analyze_features_rgb8` at default budgets — the reference |
//!
//! Content is real: centre crops / mosaics of codec-corpus images through
//! `examples/common/mod.rs`, the same loader the drift grid and cost grid use,
//! so a cost number here lines up with a drift number there on the same bytes.
//! A synthetic noise buffer would misstate the palette and histogram passes and
//! flatter the DCT pass.
//!
//! Run: `nice -n 19 cargo bench --features hdr --bench budget_cost`
//! Do NOT build with `-C target-cpu=native` — runtime SIMD dispatch is what ships.
//!
//! Env: `ZENANALYZE_CORPUS_DIR` (default `../codec-corpus`), `COST_SIDES`
//! (comma list, default `256,1024,2048,4096`), `COST_CLASS` (default `photo`).

use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenbench::prelude::*;
use zenpixels::{PixelDescriptor, PixelSlice};

#[path = "../examples/common/mod.rs"]
mod common;

const HF_UNCAPPED: usize = 262_144;
const DEFAULT_PIXEL_BUDGET: usize = 500_000;
const DEFAULT_HF_MAX_BLOCKS: usize = 1_024;

fn run_internal(rgb: &[u8], side: u32, pb: usize, hf: usize) {
    let iq = AnalysisQuery::__internal_with_overrides(FeatureSet::SUPPORTED, pb, hf);
    let slice = PixelSlice::new(
        rgb,
        side,
        side,
        side as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap();
    std::hint::black_box(zenanalyze::__analyze_internal(slice, &iq)).unwrap();
}

fn bench_budgets(suite: &mut Suite) {
    let corpus = std::path::PathBuf::from(
        std::env::var("ZENANALYZE_CORPUS_DIR").unwrap_or_else(|_| "../codec-corpus".into()),
    );
    let want_class = std::env::var("COST_CLASS").unwrap_or_else(|_| "photo".into());
    let sides: Vec<u32> = std::env::var("COST_SIDES")
        .unwrap_or_else(|_| "256,1024,2048,4096".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let classes = common::classes(&corpus);
    let Some(class) = classes.iter().find(|c| c.name == want_class) else {
        eprintln!("[budget_cost] no class `{want_class}`; skipping");
        return;
    };

    for side in sides {
        // Leaked so the closures can be 'static; one buffer per size, built once
        // outside the timed region.
        let (buf, _) = common::build(side, class, 0);
        let rgb: &'static [u8] = Box::leak(buf.into_boxed_slice());
        let query: &'static AnalysisQuery =
            Box::leak(Box::new(AnalysisQuery::new(FeatureSet::SUPPORTED)));

        suite.compare(format!("analyze/{want_class}/{side}"), |g| {
            g.throughput(Throughput::Elements((side as u64) * (side as u64)));
            g.bench("default", move |b| {
                b.iter(|| run_internal(rgb, side, DEFAULT_PIXEL_BUDGET, DEFAULT_HF_MAX_BLOCKS))
            });
            g.bench("hffull", move |b| {
                b.iter(|| run_internal(rgb, side, DEFAULT_PIXEL_BUDGET, HF_UNCAPPED))
            });
            g.bench("full", move |b| {
                b.iter(|| run_internal(rgb, side, usize::MAX, HF_UNCAPPED))
            });
            g.bench("public", move |b| {
                b.iter(|| {
                    std::hint::black_box(zenanalyze::analyze_features_rgb8(rgb, side, side, query))
                })
            });
        });
    }
}

/// An encode-time anchor at the same size, on the same bytes, on this host — so
/// "analysis costs 382 ms" can be read as a fraction of something.
///
/// **These are the `image` crate's encoders (0.25, a dev-dependency here), not a
/// zen codec, and they sit at the FAST end of the spectrum**: baseline JPEG with
/// no trellis / no adaptive quantization, and PNG at its default filter+deflate
/// effort. A production web encoder (mozjpeg at high effort, WebP, AVIF, JXL)
/// costs substantially more per pixel than either. The ratio computed against
/// these is therefore the *least* favourable one for analysis — against a
/// heavier codec, analysis is a smaller share, not a larger one. No number for a
/// codec that was not run here is reported.
fn bench_encode_anchor(suite: &mut Suite) {
    let corpus = std::path::PathBuf::from(
        std::env::var("ZENANALYZE_CORPUS_DIR").unwrap_or_else(|_| "../codec-corpus".into()),
    );
    let classes = common::classes(&corpus);
    let Some(class) = classes.iter().find(|c| c.name == "photo") else {
        return;
    };
    for side in [1024u32, 4096] {
        let (buf, _) = common::build(side, class, 0);
        let rgb: &'static [u8] = Box::leak(buf.into_boxed_slice());
        suite.compare(format!("encode_anchor/photo/{side}"), |g| {
            g.throughput(Throughput::Elements((side as u64) * (side as u64)));
            g.bench("image_jpeg_q85", move |b| {
                b.iter(|| {
                    let mut out = Vec::with_capacity(1 << 20);
                    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85);
                    enc.encode(rgb, side, side, image::ExtendedColorType::Rgb8)
                        .unwrap();
                    std::hint::black_box(out.len())
                })
            });
            g.bench("image_png_default", move |b| {
                b.iter(|| {
                    let mut out = Vec::with_capacity(1 << 22);
                    {
                        let enc = image::codecs::png::PngEncoder::new(&mut out);
                        image::ImageEncoder::write_image(
                            enc,
                            rgb,
                            side,
                            side,
                            image::ExtendedColorType::Rgb8,
                        )
                        .unwrap();
                    }
                    std::hint::black_box(out.len())
                })
            });
        });
    }
}

fn all(suite: &mut Suite) {
    bench_budgets(suite);
    bench_encode_anchor(suite);
}

zenbench::main!(all);
