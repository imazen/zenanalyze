//! Linear-light analysis — integration tests.
//!
//! The diffuse-white-normalized linear path now lives in
//! [`crate::row_stream::RowStream::new_normalized_linear`]: when
//! `AnalysisQuery::with_linear_light(true)` is set, the row stream emits
//! display-range bytes (decode → ×diffuse-white anchor → sRGB OETF → ×255), so
//! EVERY content tier reads the same envelope-normalized bytes in its existing
//! combined pass — there is no separate linear pass. The sRGB OETF re-encode makes
//! the displayable range perceptually identical to the default gamma path (SDR scores
//! the same either way), while the f32 fetch lets super-white extend past 255 so HDR
//! highlights survive. These tests exercise that end-to-end via `analyze_features`.

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

/// Below diffuse white the linear-light path re-encodes through the sRGB OETF, so an
/// SDR scene scores **the same** under linear-light as under the default gamma path
/// (to sRGB round-trip precision). This is the deliberate "same content → same score"
/// choice for the displayable range — an SDR scene reads identically whether delivered
/// as SDR or carried in an HDR envelope. A dark ramp (0..63, where sRGB bends hardest
/// away from linear) is the strongest case: before the OETF re-encode this path fed raw
/// linear and *collapsed* shadow variance; the re-encode pins that artifact shut.
#[test]
fn sdr_scores_same_under_linear_light_below_diffuse_white() {
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
    // 5e-3 is the cross-platform float-noise budget (REL_TOLERANCE); the actual gap is
    // ~1e-7 (sRGB EOTF∘OETF round-trip), vs the old behavior where linear was < 0.5·gamma.
    assert!(
        (lin - gamma).abs() <= 5.0e-3 * gamma.max(1.0),
        "SDR must score the same below diffuse white (sRGB round-trip): linear {lin} vs gamma {gamma}"
    );
}

/// End-to-end: the opt-in flag is a near-identity for SDR below diffuse white (proven
/// above), so it earns its keep on **HDR super-white** — content the default gamma path
/// narrows and display-clips. A PQ image split between diffuse-white (203 nits) and 3×
/// (609 nits) reads large Variance under linear-light (super-white survives) where the
/// gamma path crushes the highlight half toward the dark display range.
#[test]
fn linear_light_flag_changes_variance_for_hdr_superwhite() {
    use linear_srgb::tf::linear_to_pq;
    use zenpixels::TransferFunction;

    let (w, h) = (64u32, 64u32);
    let q = |nits: f32| (linear_to_pq(nits / 10000.0).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let (pl, ph) = (q(203.0), q(609.0));
    let mut buf = Vec::with_capacity((w * h * 6) as usize);
    for _y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { pl } else { ph };
            for _c in 0..3 {
                buf.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    let desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
    let var = |linear: bool| -> f32 {
        let mut q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
        if linear {
            q = q.with_linear_light(true);
        }
        analyze_features(
            PixelSlice::new(&buf, w, h, (w * 6) as usize, desc).unwrap(),
            &q,
        )
        .unwrap()
        .get_f32(AnalysisFeature::Variance)
        .unwrap()
    };
    let gamma = var(false); // default path narrows PQ → display-clips the highlight half
    let linear = var(true); // linear-light keeps super-white → real contrast
    assert!(
        linear > 1000.0 && linear - gamma > 500.0,
        "flag must surface HDR super-white the gamma path crushes: gamma {gamma} linear {linear}"
    );
}

/// The diffuse-white-normalized linear path is a ROW-STREAM transform
/// (`RowStream::new_normalized_linear`), so EVERY content tier (1/2/3 + palette) reads
/// its bytes — not just `Variance`. SDR below diffuse white is a near-identity by design
/// (proven above), so the tier-wide coverage shows on **HDR super-white**: an 8×8
/// checkerboard between diffuse-white (203 nits) and super-white (812 nits) must move
/// *many* tier features, where the default gamma path display-clips the bright half flat.
#[test]
fn linear_light_moves_all_tiers_on_hdr_superwhite() {
    use linear_srgb::tf::linear_to_pq;
    use zenpixels::TransferFunction;

    let (w, h) = (128u32, 128u32);
    let q = |nits: f32| (linear_to_pq(nits / 10000.0).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let (lo, hi) = (q(203.0), q(812.0));
    let mut buf = Vec::with_capacity((w * h * 6) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = if ((x / 8) + (y / 8)) % 2 == 0 { lo } else { hi };
            for _c in 0..3 {
                buf.extend_from_slice(&v.to_ne_bytes());
            }
        }
    }
    let desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
    let mk = || PixelSlice::new(&buf, w, h, (w * 6) as usize, desc).unwrap();
    let g = analyze_features(mk(), &AnalysisQuery::new(FeatureSet::SUPPORTED)).unwrap();
    let l = analyze_features(
        mk(),
        &AnalysisQuery::new(FeatureSet::SUPPORTED).with_linear_light(true),
    )
    .unwrap();
    let mut moved = Vec::new();
    for f in FeatureSet::SUPPORTED.iter() {
        if let (Some(a), Some(b)) = (g.get_f32(f), l.get_f32(f))
            && a.is_finite()
            && b.is_finite()
            && (a - b).abs() > 1e-4 * a.abs().max(1.0)
        {
            moved.push(f.name());
        }
    }
    std::eprintln!(
        "linear-light moved {} features on HDR: {:?}",
        moved.len(),
        moved
    );
    assert!(
        moved.len() > 5,
        "linear-light must move many tier features on HDR super-white; moved {}: {:?}",
        moved.len(),
        moved
    );
    assert!(
        moved.contains(&"variance"),
        "Variance must move under linear-light on HDR super-white"
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

/// Foundation for the f32 HDR-correct tier path: `RowStream::fetch_f32_into`
/// preserves super-white HDR as `> 255.0`, where the u8 `fetch_into` path
/// hard-clips to 255. This is exactly the information the (in-progress) f32 tier
/// kernels need to see the full envelope instead of the display-clipped range.
#[test]
fn fetch_f32_preserves_superwhite_where_u8_clips() {
    use crate::row_stream::RowStream;
    use linear_srgb::tf::linear_to_pq;
    use zenpixels::TransferFunction;

    let (w, h) = (8u32, 2u32);
    // Each pixel sits at 2× the default diffuse-white (203 nits) in linear — above
    // display-white. PQ stores linear normalized to the 10000-nit peak.
    let above = 2.0 * 203.0 / 10000.0;
    let pix16 = (linear_to_pq(above).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let bytes: Vec<u8> = (0..(w * h * 3)).flat_map(|_| pix16.to_ne_bytes()).collect();
    let desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
    let mk = || PixelSlice::new(&bytes, w, h, (w * 6) as usize, desc).unwrap();

    let mut f32row = vec![0f32; (w * 3) as usize];
    RowStream::new_normalized_linear(mk())
        .unwrap()
        .fetch_f32_into(0, &mut f32row);
    let mut u8row = vec![0u8; (w * 3) as usize];
    RowStream::new_normalized_linear(mk())
        .unwrap()
        .fetch_into(0, &mut u8row);

    // 2× diffuse-white ≈ 345 in the f32 path (sRGB OETF of linear 2.0, ×255); the u8
    // path clamps to 255. Either way super-white survives as > 255 in f32.
    assert!(
        f32row[0] > 300.0,
        "f32 fetch must preserve super-white (>255), got {}",
        f32row[0]
    );
    assert_eq!(u8row[0], 255, "u8 fetch must clamp super-white to 255");
}

/// End-to-end HDR-correct proof: the f32 tier-1 path SEES super-white. A PQ image
/// split between diffuse-white and 3× diffuse-white highlights has large
/// linear-light Variance, where the old RGB8-clamp path collapsed both to 255 → ~0.
#[test]
fn linear_light_variance_sees_superwhite_highlights() {
    use linear_srgb::tf::linear_to_pq;
    use zenpixels::TransferFunction;

    let (w, h) = (64u32, 64u32);
    // PQ16 image: left half at `lo` nits, right half at `hi` nits (gray, R=G=B).
    let mk_pq = |lo_nits: f32, hi_nits: f32| -> Vec<u8> {
        let q = |nits: f32| (linear_to_pq(nits / 10000.0).clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        let (pl, ph) = (q(lo_nits), q(hi_nits));
        let mut buf = Vec::with_capacity((w * h * 6) as usize);
        for _y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { pl } else { ph };
                for _c in 0..3 {
                    buf.extend_from_slice(&v.to_ne_bytes());
                }
            }
        }
        buf
    };
    let desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
    let lin_var = |buf: &[u8]| -> f32 {
        let q =
            AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance)).with_linear_light(true);
        analyze_features(
            PixelSlice::new(buf, w, h, (w * 6) as usize, desc).unwrap(),
            &q,
        )
        .unwrap()
        .get_f32(AnalysisFeature::Variance)
        .unwrap()
    };
    // Uniform diffuse-white (203 nits) — control: ~0 variance either way.
    let var_flat = lin_var(&mk_pq(203.0, 203.0));
    assert!(
        var_flat < 1.0,
        "uniform diffuse-white ~0 variance, got {var_flat}"
    );
    // Diffuse-white vs 3× diffuse-white (609 nits) highlights — in u8 both clamp to
    // 255 (≈0 var); the f32 path keeps the 609-nit half at ~411 (sRGB OETF of linear
    // 3.0, ×255) → large variance.
    let var_hi = lin_var(&mk_pq(203.0, 609.0));
    assert!(
        var_hi > 1000.0,
        "super-white highlights must raise linear-light variance (f32 path, not clamped), got {var_hi}"
    );
}
