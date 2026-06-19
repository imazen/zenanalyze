//! Consistency matrix — EVERY feature, across channel types, SDR.
//!
//! Goal: SDR feature values must be identical regardless of source channel
//! type (a lossless u8→u16 promotion or a u8→f32 sRGB encode carries no new
//! information, so it must not move any feature). This is the broad version of
//! `channel_consistency_probe` — that one hand-picked 9 SDR features; this runs
//! `feature_vector_all` so a divergence in ANY feature (incl. experimental/hdr
//! when those cargo features are on) is visible.
//!
//! Axes:
//!   A) u8  RGB8_SRGB                      — baseline
//!   B) u16 RGB16_SRGB, (c<<8)|c (=c*257)  — LOSSLESS promote, want ==A
//!   E) f32 RGBF32 sRGB, c/255.0           — 3rd channel type,   want ==A
//!
//! Height is deliberately NOT a multiple of 8 (260) so the tier1 scalar tail
//! stripe runs alongside the SIMD full stripes — if the two disagree on the
//! luma definition, every channel type inherits the same mix, so this axis
//! won't catch it (the kernel-level test in tier1.rs does). It DOES catch any
//! channel-type-dependent narrowing bug.
//!
//! Run: cargo run --release --features experimental,hdr --example consistency_matrix

use zenanalyze::{feature_count, feature_ids, feature_name, feature_vector_all};
use zenpixels::{PixelDescriptor, PixelSlice, TransferFunction};

const W: u32 = 320;
const H: u32 = 260; // not %8 → exercises the scalar tail stripe

fn make_u8() -> Vec<u8> {
    // Mixed content so most features are non-trivial: horizontal ramp (gradient),
    // a textured band (edges/noise), a flat block (uniformity/flat-color), and a
    // chroma gradient (chroma complexity).
    let mut v = vec![0u8; (W * H * 3) as usize];
    for y in 0..H {
        for x in 0..W {
            let p = ((y * W + x) * 3) as usize;
            let base = (x * 255 / (W - 1)) as u8;
            if y < H / 4 {
                // textured band
                let tex = (((x * 13 + y * 7) % 11) as u8).wrapping_mul(6);
                v[p] = base.saturating_add(tex);
                v[p + 1] = base;
                v[p + 2] = base.saturating_sub(tex / 2);
            } else if y < H / 2 {
                // flat mid-gray block
                v[p] = 128;
                v[p + 1] = 128;
                v[p + 2] = 128;
            } else {
                // chroma ramp
                v[p] = base;
                v[p + 1] = 255 - base;
                v[p + 2] = (base / 2).wrapping_add(64);
            }
        }
    }
    v
}

fn vec_all(slice: PixelSlice<'_>) -> Vec<f32> {
    let n = feature_count();
    let mut out = vec![0.0f32; n];
    assert!(feature_vector_all(slice, &mut out), "feature_vector_all failed");
    out
}

fn main() {
    let n = feature_count();
    let mut ids = vec![0u16; n];
    feature_ids(&mut ids);

    let u8buf = make_u8();
    let a = vec_all(PixelSlice::new(&u8buf, W, H, (W * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap());

    let b_u16: Vec<u16> = u8buf.iter().map(|&c| ((c as u16) << 8) | c as u16).collect();
    let b_bytes: Vec<u8> = b_u16.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let b = vec_all(PixelSlice::new(&b_bytes, W, H, (W * 6) as usize, PixelDescriptor::RGB16_SRGB).unwrap());

    let e_f32: Vec<f32> = u8buf.iter().map(|&c| c as f32 / 255.0).collect();
    let e_bytes: Vec<u8> = e_f32.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let e_desc = PixelDescriptor::RGBF32.with_transfer(TransferFunction::Srgb);
    let e = vec_all(PixelSlice::new(&e_bytes, W, H, (W * 12) as usize, e_desc).unwrap());

    // rel divergence vs A, per feature
    let rel = |va: f32, vx: f32| {
        let d = (va - vx).abs();
        if d == 0.0 { 0.0 } else { d / va.abs().max(1e-6) }
    };
    let tol = 1e-5;

    // Format/precision-descriptive features legitimately differ by channel
    // type — they REPORT the container, not the content. bitmap_bytes is the
    // byte count (1/2/4 bpc); effective_bit_depth reports storage depth for
    // floats (8 for u8, 8 for losslessly-promoted u16, 32 for f32) by design.
    // Everything else is a content feature and MUST be channel-type-invariant.
    let format_descriptive = |name: &str| matches!(name, "bitmap_bytes" | "effective_bit_depth");

    let (mut content_n, mut content_ok) = (0u32, 0u32);
    let mut content_bugs: Vec<String> = Vec::new();
    let mut expected: Vec<String> = Vec::new();
    for i in 0..n {
        let (va, vb, ve) = (a[i], b[i], e[i]);
        let beq = rel(va, vb) <= tol;
        let eeq = rel(va, ve) <= tol;
        let name = feature_name(ids[i]).unwrap_or("?");
        let line = format!(
            "{:<28} {:>12.6} {:>12.6} {:>12.6}  {}{}",
            format!("{}({})", name, ids[i]), va, vb, ve,
            if beq { "" } else { "B≠" }, if eeq { "" } else { "E≠" },
        );
        if format_descriptive(name) {
            if !beq || !eeq { expected.push(line); }
        } else {
            content_n += 1;
            if beq && eeq { content_ok += 1; } else { content_bugs.push(line); }
        }
    }

    println!("--- CONTENT features: must be channel-type-invariant ---");
    if content_bugs.is_empty() {
        println!("  all {content_ok}/{content_n} content features identical across u8/u16/f32 ✓");
    } else {
        println!("{:<28} {:>12} {:>12} {:>12}", "feature(id)", "A:u8", "B:u16loss", "E:f32srgb");
        for d in &content_bugs { println!("{d}"); }
        println!("  ✗ {}/{} content features DIVERGE — channel-type-dependent bug",
            content_bugs.len(), content_n);
    }
    println!("\n--- format/precision-descriptive (expected to differ by design) ---");
    for d in &expected { println!("{d}"); }

    assert!(content_bugs.is_empty(), "content-feature channel-type divergence");
}
