//! Throughput A/B for the linear-intermediate **representation** choice (i12/i16
//! vs u16 vs f32) for zenanalyze's tier-1 luma kernels. zenanalyze is hot-path,
//! so the representation is decided by measured throughput, not precision
//! arguments. Interleaved/paired via zenbench (kills thermal/turbo bias).
//!
//! The decision hinges on two effects the bench separates:
//!  - **math lane width** — i16/u16 pack 2× per SIMD register vs f32, BUT
//!    variance/laplacian need the *squared* term which widens i16→i64 and eats
//!    the advantage; in-width ops (min/max for uniformity, threshold compares
//!    for edges) keep the full 2×. So we bench both a widening reduction
//!    (`variance_reduce`) and an in-width reduction (`minmax_reduce`).
//!  - **linearize cost** — gamma→linear is the opt-in's only added work. LUT
//!    (gather) vs SIMD-friendly compute, for 8-bit (256-LUT) and 16-bit PQ
//!    (64k-LUT), into each target type. `pq_direct` = no linearize (work in PQ
//!    code space), the floor.
//!
//! NB: these kernels are plain autovectorized loops, NOT the hand-tuned
//! magetypes kernels tier1 ships — so treat the numbers as first-order (a
//! hand-tuned i16 kernel could widen the gap). Good enough to pick a direction
//! and flag whether a hand-tuned bake-off is warranted.
//!
//! Run: cargo run --release --example bench_linear_repr

use linear_srgb::tf::{pq_to_linear, srgb_to_linear};
use zenbench::prelude::*;

// Fixed-point BT.601 luma (libwebp/coefficient lineage, sum 220) — the integer
// luma matrix tier3 already uses.
const QR: i64 = 66;
const QG: i64 = 129;
const QB: i64 = 25;

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

fn prng(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        s
    }
}

// ---- widening reduction: variance (sum + sum of squares) ----
fn var_f32(p: &[f32]) -> f32 {
    let (mut s, mut sq) = (0.0f32, 0.0f32);
    for &x in p {
        s += x;
        sq = x.mul_add(x, sq);
    }
    sq - s
}
fn var_i16(p: &[i16]) -> i64 {
    let (mut s, mut sq) = (0i64, 0i64);
    for &x in p {
        let v = x as i64;
        s += v;
        sq += v * v;
    }
    sq - s
}
fn var_u16(p: &[u16]) -> u64 {
    let (mut s, mut sq) = (0u64, 0u64);
    for &x in p {
        let v = x as u64;
        s += v;
        sq += v * v;
    }
    sq.wrapping_sub(s)
}

// ---- in-width reduction: min/max (uniformity-style block extent) ----
fn minmax_f32(p: &[f32]) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in p {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}
fn minmax_i16(p: &[i16]) -> i16 {
    let (mut lo, mut hi) = (i16::MAX, i16::MIN);
    for &x in p {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}
fn minmax_u16(p: &[u16]) -> u16 {
    let (mut lo, mut hi) = (u16::MAX, u16::MIN);
    for &x in p {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}

// ---- SDR linearize + luma (build a linear luma checksum) ----
fn lin_lut_f32(r: &[u8], g: &[u8], b: &[u8], lut: &[f32; 256]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..r.len() {
        acc += 0.299 * lut[r[i] as usize] + 0.587 * lut[g[i] as usize] + 0.114 * lut[b[i] as usize];
    }
    acc
}
fn lin_lut_i16(r: &[u8], g: &[u8], b: &[u8], lut: &[i16; 256]) -> i64 {
    let mut acc = 0i64;
    for i in 0..r.len() {
        let l = QR * lut[r[i] as usize] as i64
            + QG * lut[g[i] as usize] as i64
            + QB * lut[b[i] as usize] as i64;
        acc += l >> 8;
    }
    acc
}
fn lin_compute_f32(r: &[u8], g: &[u8], b: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..r.len() {
        let lr = srgb_to_linear(r[i] as f32 / 255.0);
        let lg = srgb_to_linear(g[i] as f32 / 255.0);
        let lb = srgb_to_linear(b[i] as f32 / 255.0);
        acc += 0.299 * lr + 0.587 * lg + 0.114 * lb;
    }
    acc
}

// ---- HDR (u16 PQ) linearize + luma ----
fn pq_direct_i16(r: &[u16], g: &[u16], b: &[u16]) -> i64 {
    // No linearize: luma on PQ code values (perceptual-ish, the floor).
    let mut acc = 0i64;
    for i in 0..r.len() {
        let l = QR * (r[i] >> 4) as i64 + QG * (g[i] >> 4) as i64 + QB * (b[i] >> 4) as i64;
        acc += l >> 8;
    }
    acc
}
fn pq_lut_f32(r: &[u16], g: &[u16], b: &[u16], lut: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..r.len() {
        acc += 0.299 * lut[r[i] as usize] + 0.587 * lut[g[i] as usize] + 0.114 * lut[b[i] as usize];
    }
    acc
}
fn pq_lut_i16(r: &[u16], g: &[u16], b: &[u16], lut: &[i16]) -> i64 {
    let mut acc = 0i64;
    for i in 0..r.len() {
        let l = QR * lut[r[i] as usize] as i64
            + QG * lut[g[i] as usize] as i64
            + QB * lut[b[i] as usize] as i64;
        acc += l >> 8;
    }
    acc
}
fn pq_compute_f32(r: &[u16], g: &[u16], b: &[u16]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..r.len() {
        let lr = pq_to_linear(r[i] as f32 / 65535.0);
        let lg = pq_to_linear(g[i] as f32 / 65535.0);
        let lb = pq_to_linear(b[i] as f32 / 65535.0);
        acc += 0.299 * lr + 0.587 * lg + 0.114 * lb;
    }
    acc
}

fn main() {
    let result = zenbench::run(|suite| {
        for (sz, w, h) in [("1MP", 1024usize, 1024usize), ("4MP", 2048usize, 2048usize)] {
            let n = w * h;
            let mut rng = prng(0xC0FFEE ^ (n as u32));

            // luma planes (same content, three types) for the reductions
            let luma_f32 = leak(
                (0..n)
                    .map(|_| (rng() >> 16 & 0x3FF) as f32)
                    .collect::<Vec<_>>(),
            );
            let luma_i16 = leak(luma_f32.iter().map(|&x| x as i16).collect::<Vec<_>>());
            let luma_u16 = leak(luma_f32.iter().map(|&x| x as u16).collect::<Vec<_>>());

            // RGB8 planes + sRGB LUTs for SDR linearize
            let r8 = leak((0..n).map(|_| (rng() >> 16) as u8).collect::<Vec<_>>());
            let g8 = leak((0..n).map(|_| (rng() >> 16) as u8).collect::<Vec<_>>());
            let b8 = leak((0..n).map(|_| (rng() >> 16) as u8).collect::<Vec<_>>());
            let lut_f32: &'static [f32; 256] = {
                let mut a = [0.0f32; 256];
                for (v, e) in a.iter_mut().enumerate() {
                    *e = srgb_to_linear(v as f32 / 255.0) * 255.0;
                }
                Box::leak(Box::new(a))
            };
            let lut_i16: &'static [i16; 256] = {
                let mut a = [0i16; 256];
                for (v, e) in a.iter_mut().enumerate() {
                    *e = (srgb_to_linear(v as f32 / 255.0) * 4095.0 + 0.5) as i16;
                }
                Box::leak(Box::new(a))
            };

            // u16 PQ planes + 64k PQ→linear LUTs for HDR linearize
            let pr = leak(
                (0..n)
                    .map(|_| (rng() >> 8 & 0xFFFF) as u16)
                    .collect::<Vec<_>>(),
            );
            let pg = leak(
                (0..n)
                    .map(|_| (rng() >> 8 & 0xFFFF) as u16)
                    .collect::<Vec<_>>(),
            );
            let pb = leak(
                (0..n)
                    .map(|_| (rng() >> 8 & 0xFFFF) as u16)
                    .collect::<Vec<_>>(),
            );
            let pqlut_f32 = leak(
                (0..65536)
                    .map(|c| pq_to_linear(c as f32 / 65535.0) * 255.0)
                    .collect::<Vec<_>>(),
            );
            let pqlut_i16 = leak(
                (0..65536)
                    .map(|c| (pq_to_linear(c as f32 / 65535.0) * 4095.0 + 0.5) as i16)
                    .collect::<Vec<_>>(),
            );

            suite.compare(format!("variance_reduce_{sz}"), |gp| {
                gp.config().max_rounds(60);
                gp.throughput(Throughput::Elements(n as u64));
                gp.bench("f32", |b| b.iter(|| black_box(var_f32(luma_f32))));
                gp.bench("i16_widen_i64", |b| b.iter(|| black_box(var_i16(luma_i16))));
                gp.bench("u16_widen_u64", |b| b.iter(|| black_box(var_u16(luma_u16))));
            });

            suite.compare(format!("minmax_reduce_{sz}"), |gp| {
                gp.config().max_rounds(60);
                gp.throughput(Throughput::Elements(n as u64));
                gp.bench("f32", |b| b.iter(|| black_box(minmax_f32(luma_f32))));
                gp.bench("i16_inwidth", |b| {
                    b.iter(|| black_box(minmax_i16(luma_i16)))
                });
                gp.bench("u16_inwidth", |b| {
                    b.iter(|| black_box(minmax_u16(luma_u16)))
                });
            });

            suite.compare(format!("sdr_linearize_{sz}"), |gp| {
                gp.config().max_rounds(60);
                gp.throughput(Throughput::Elements(n as u64));
                gp.bench("lut_f32", |b| {
                    b.iter(|| black_box(lin_lut_f32(r8, g8, b8, lut_f32)))
                });
                gp.bench("lut_i16", |b| {
                    b.iter(|| black_box(lin_lut_i16(r8, g8, b8, lut_i16)))
                });
                gp.bench("compute_f32", |b| {
                    b.iter(|| black_box(lin_compute_f32(r8, g8, b8)))
                });
            });

            suite.compare(format!("hdr_linearize_{sz}"), |gp| {
                gp.config().max_rounds(60);
                gp.throughput(Throughput::Elements(n as u64));
                gp.bench("pq_direct_i16", |b| {
                    b.iter(|| black_box(pq_direct_i16(pr, pg, pb)))
                });
                gp.bench("pq_lut_f32", |b| {
                    b.iter(|| black_box(pq_lut_f32(pr, pg, pb, pqlut_f32)))
                });
                gp.bench("pq_lut_i16", |b| {
                    b.iter(|| black_box(pq_lut_i16(pr, pg, pb, pqlut_i16)))
                });
                gp.bench("pq_compute_f32", |b| {
                    b.iter(|| black_box(pq_compute_f32(pr, pg, pb)))
                });
            });
        }
    });

    zenbench::postprocess_result(&result);
    let out = "/mnt/v/output/imazen-26-features/bench_linear_repr_2026-06-14.json";
    if let Err(e) = result.save(out) {
        eprintln!("save failed: {e}");
    } else {
        eprintln!("saved {out}");
    }
}
