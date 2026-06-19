//! Linear-light analysis — integration tests.
//!
//! The diffuse-white-normalized linear path now lives in
//! [`crate::row_stream::RowStream::new_normalized_linear`]: when
//! `AnalysisQuery::with_linear_light(true)` is set, the row stream emits
//! normalized-linear RGB8 (decode → ×diffuse-white anchor → display range), so
//! EVERY content tier reads the same envelope-normalized bytes in its existing
//! combined pass — there is no separate linear pass. These tests exercise that
//! end-to-end via `analyze_features`.

use crate::analyze_features;
use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

fn gray_ramp_rgb8(w: usize, h: usize, max_v: u8) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 3;
            let v = (x as u32 * max_v as u32 / (w as u32 - 1)) as u8;
            buf[p] = v;
            buf[p + 1] = v;
            buf[p + 2] = v;
        }
    }
    buf
}

/// Variance under the diffuse-white-normalized linear-light path.
fn linear_variance(slice: PixelSlice<'_>) -> f32 {
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance)).with_linear_light(true);
    analyze_features(slice, &q)
        .unwrap()
        .get_f32(AnalysisFeature::Variance)
        .unwrap()
}

/// Variance under the default gamma path.
fn gamma_variance(slice: PixelSlice<'_>) -> f32 {
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
    analyze_features(slice, &q)
        .unwrap()
        .get_f32(AnalysisFeature::Variance)
        .unwrap()
}

/// Shadows are where sRGB diverges most from linear, so a dark ramp (0..63) has
/// far lower variance in linear light than in gamma.
#[test]
fn linear_light_collapses_shadow_variance() {
    let (w, h) = (64usize, 64usize);
    let buf = gray_ramp_rgb8(w, h, 63);
    let mk =
        || PixelSlice::new(&buf, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let lin = linear_variance(mk());
    let gamma = gamma_variance(mk());
    assert!(
        lin.is_finite() && lin > 0.0,
        "linear var positive, got {lin}"
    );
    assert!(
        lin < 0.5 * gamma,
        "linear-light should collapse shadow variance: linear {lin} vs gamma {gamma}"
    );
}

/// End-to-end: the opt-in flag changes the reported Variance.
#[test]
fn linear_light_flag_changes_variance_end_to_end() {
    let (w, h) = (64usize, 64usize);
    let buf = gray_ramp_rgb8(w, h, 63);
    let mk =
        || PixelSlice::new(&buf, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let gamma = gamma_variance(mk());
    let linear = linear_variance(mk());
    assert!(
        linear < gamma * 0.5,
        "flag must lower shadow variance: gamma {gamma} linear {linear}"
    );
}

/// A uniform field reads ~zero variance through the normalized-linear stream.
#[test]
fn uniform_field_is_zero_variance() {
    let buf = vec![137u8; 40 * 40 * 3];
    let slice = PixelSlice::new(&buf, 40, 40, 40 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let lin = linear_variance(slice);
    assert!(lin.abs() < 1e-2, "uniform field variance ~0, got {lin}");
}

/// The normalized-linear path handles non-RGB8 layouts (here RGBA8) — the
/// converter does the strip-alpha + linearize.
#[test]
fn rgba8_supported() {
    let buf = vec![80u8; 32 * 32 * 4];
    let slice = PixelSlice::new(&buf, 32, 32, 32 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    assert!(linear_variance(slice).is_finite());
}

/// The anchor is read from a **signaled** `ColorContext.diffuse_white`. A PQ
/// envelope authored at 100 nits matches SDR-normal only when the slice signals
/// 100; the BT.2408 default (203) mis-normalizes it. Proves the streaming
/// decode honors the signal.
#[test]
fn honors_signaled_diffuse_white() {
    use linear_srgb::tf::{linear_to_pq, srgb_to_linear};
    use std::sync::Arc;
    use zenpixels::{ColorContext, DiffuseWhite, TransferFunction};

    let (w, h) = (96u32, 64u32);
    let sdr = gray_ramp_rgb8(w as usize, h as usize, 200);
    const DW: f32 = 100.0; // non-default authoring white
    let pq16: Vec<u16> = sdr
        .iter()
        .map(|&c| {
            let pq = linear_to_pq(srgb_to_linear(c as f32 / 255.0) * DW / 10000.0);
            (pq.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
        })
        .collect();
    let pq_bytes: Vec<u8> = pq16.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let hdr_desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);

    let v_sdr = linear_variance(
        PixelSlice::new(&sdr, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap(),
    );
    let ctx = Arc::new(ColorContext::default().with_diffuse_white(DiffuseWhite::new(DW)));
    let v_signaled = linear_variance(
        PixelSlice::new(&pq_bytes, w, h, (w * 6) as usize, hdr_desc)
            .unwrap()
            .with_color_context(ctx),
    );
    let v_default =
        linear_variance(PixelSlice::new(&pq_bytes, w, h, (w * 6) as usize, hdr_desc).unwrap());
    assert!(
        (v_sdr - v_signaled).abs() <= 0.10 * v_sdr.max(1.0),
        "signaled anchor must match SDR: signaled {v_signaled} vs sdr {v_sdr}"
    );
    assert!(
        (v_sdr - v_default).abs() > 0.15 * v_sdr.max(1.0),
        "default (203) must mis-normalize a 100-nit envelope: default {v_default} vs sdr {v_sdr}"
    );
}
