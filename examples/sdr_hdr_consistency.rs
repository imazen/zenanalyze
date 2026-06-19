//! SDR/HDR consistency probe — the second goal axis.
//!
//! Requirement: SDR content wrapped in an HDR envelope (here: SDR-white-anchored
//! PQ u16, diffuse white 203 nits) must produce the SAME features as the same
//! content tagged SDR — with the diffuse-white-normalized linear-light path on
//! for both. `sdr_in_hdr_envelope_matches_sdr_normal` proves this for Variance;
//! this probe measures EVERY feature so the gap is visible.
//!
//! Today the linear-light override only re-derives Variance on normalized
//! linear; the other content tiers still run on the PQ→RGB8 narrowing, which
//! collapses the SDR-in-PQ envelope. This probe quantifies exactly which
//! features are already SDR/HDR-invariant and which still need the normalized
//! representation.
//!
//! Run: cargo run --release --features experimental,hdr --example sdr_hdr_consistency

use linear_srgb::tf::{linear_to_pq, srgb_to_linear};
use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice, TransferFunction};

const W: u32 = 192;
const H: u32 = 128;
const DIFFUSE_NITS: f32 = 203.0;
const PQ_PEAK_NITS: f32 = 10000.0;

fn make_sdr() -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 3) as usize];
    for y in 0..H {
        for x in 0..W {
            let p = ((y * W + x) * 3) as usize;
            let base = (x * 255 / (W - 1)) as u8;
            let tex = (((x * 11 + y * 5) % 9) as u8).wrapping_mul(7);
            v[p] = base.saturating_add(tex);
            v[p + 1] = base;
            v[p + 2] = base.saturating_sub(tex / 2);
        }
    }
    v
}

fn main() {
    let sdr = make_sdr();
    // Same content as SDR-white-anchored PQ u16.
    let pq16: Vec<u16> = sdr
        .iter()
        .map(|&c| {
            let lin = srgb_to_linear(c as f32 / 255.0);
            let pq = linear_to_pq(lin * DIFFUSE_NITS / PQ_PEAK_NITS);
            (pq.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
        })
        .collect();
    let pq_bytes: Vec<u8> = pq16.iter().flat_map(|&v| v.to_ne_bytes()).collect();

    let mk_sdr =
        || PixelSlice::new(&sdr, W, H, (W * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let hdr_desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
    let mk_hdr = || PixelSlice::new(&pq_bytes, W, H, (W * 6) as usize, hdr_desc).unwrap();

    let q = AnalysisQuery::new(FeatureSet::SUPPORTED).with_linear_light(true);
    let rs = analyze_features(mk_sdr(), &q).expect("sdr");
    let rh = analyze_features(mk_hdr(), &q).expect("hdr");

    let rel = |a: f32, b: f32| {
        let d = (a - b).abs();
        if d == 0.0 { 0.0 } else { d / a.abs().max(1e-6) }
    };
    // Exclude features that differ by construction, not by envelope:
    //  - HDR/depth (ids 32-39,46,47) MEASURE the envelope on purpose.
    //  - bitmap_bytes (60) is the format byte count (RGB8 vs RGB16 = 2×).
    let is_depth = |id: u16| matches!(id, 32..=39 | 46 | 47 | 60);

    let tol = 0.03;
    let (mut inv, mut tot) = (0u32, 0u32);
    let mut diverging: Vec<String> = Vec::new();
    for f in FeatureSet::SUPPORTED.iter() {
        let id = f.id();
        if is_depth(id) {
            continue;
        }
        let (vs, vh) = (rs.get_f32(f), rh.get_f32(f));
        if let (Some(vs), Some(vh)) = (vs, vh) {
            tot += 1;
            let r = rel(vs, vh);
            if r <= tol {
                inv += 1;
            } else {
                diverging.push(format!("{:<28} sdr={:>11.5} hdr={:>11.5}  rel={:.2}", format!("{f:?}"), vs, vh, r));
            }
        }
    }

    println!("SDR-in-PQ-envelope vs SDR-normal, linear-light ON, content features only:");
    println!("  {inv}/{tot} features SDR/HDR-invariant (rel ≤ {tol})\n");
    if !diverging.is_empty() {
        println!("--- still envelope-dependent (need normalized-linear content tiers) ---");
        for d in &diverging {
            println!("{d}");
        }
    }
}
