//! "Native kernel format" bench: how the analysis-compute type affects speed.
//! tier1/2 deinterleave RGB8 → f32 and run f32x8 SIMD (garb only emits f32
//! planes); tier3 uses i32 fixed-point. This benches the content-tier *core*
//! (RGB8 → luma → variance) across the candidate native formats, plus the
//! in-width op (min/max, uniformity-style). Multi-accumulator loops (K=16) so
//! the f32 reduction actually vectorizes (the autovec-serial trap from
//! bench_repr_handtuned). NB autovec, not the hand-tuned garb+f32x8 production
//! path — relative format cost, not absolute tier1 throughput.
//!
//! Run: cargo run --release --example bench_kernel_format

use zenbench::prelude::*;

const K: usize = 16; // independent accumulators

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

// ---- RGB8 → luma → variance, end to end, per native format ----
fn var_rgb8_f32(rgb: &[u8]) -> f64 {
    let mut s = [0.0f32; K];
    let mut sq = [0.0f32; K];
    let mut it = rgb.chunks_exact(K * 3);
    for c in &mut it {
        for j in 0..K {
            let l =
                0.299 * c[j * 3] as f32 + 0.587 * c[j * 3 + 1] as f32 + 0.114 * c[j * 3 + 2] as f32;
            s[j] += l;
            sq[j] += l * l;
        }
    }
    let (mut st, mut sqt) = (0.0f64, 0.0f64);
    for j in 0..K {
        st += s[j] as f64;
        sqt += sq[j] as f64;
    }
    sqt - st
}
fn var_rgb8_i32(rgb: &[u8]) -> i64 {
    // (77R + 150G + 29B) >> 8 — tier3's fixed-point luma (sum 256), then i64 acc.
    let mut s = [0i64; K];
    let mut sq = [0i64; K];
    let mut it = rgb.chunks_exact(K * 3);
    for c in &mut it {
        for j in 0..K {
            let l =
                (77 * c[j * 3] as i32 + 150 * c[j * 3 + 1] as i32 + 29 * c[j * 3 + 2] as i32) >> 8;
            let li = l as i64;
            s[j] += li;
            sq[j] += li * li;
        }
    }
    let (mut st, mut sqt) = (0i64, 0i64);
    for j in 0..K {
        st += s[j];
        sqt += sq[j];
    }
    sqt - st
}
fn var_rgb8_i16(rgb: &[u8]) -> i64 {
    // i16 luma, square promotes to i32 (≤255²), accumulate i64.
    let mut s = [0i64; K];
    let mut sq = [0i64; K];
    let mut it = rgb.chunks_exact(K * 3);
    for c in &mut it {
        for j in 0..K {
            let l = ((77 * c[j * 3] as i32 + 150 * c[j * 3 + 1] as i32 + 29 * c[j * 3 + 2] as i32)
                >> 8) as i16;
            let li = l as i64;
            s[j] += li;
            sq[j] += li * li;
        }
    }
    let (mut st, mut sqt) = (0i64, 0i64);
    for j in 0..K {
        st += s[j];
        sqt += sq[j];
    }
    sqt - st
}

// ---- in-width op: min/max over a luma plane (uniformity block extent) ----
fn minmax_u8(p: &[u8]) -> u8 {
    let (mut lo, mut hi) = (u8::MAX, u8::MIN);
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
fn minmax_f32(p: &[f32]) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in p {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo
}

fn main() {
    let result = zenbench::run(|suite| {
        for (sz, n) in [("1MP", 1024usize * 1024), ("4MP", 2048usize * 2048)] {
            let mut s: u32 = 0xA11FACE ^ n as u32;
            let mut next = || {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                (s >> 16) as u8
            };
            let rgb = leak((0..n * 3).map(|_| next()).collect::<Vec<u8>>());
            let luma_u8 = leak((0..n).map(|_| next()).collect::<Vec<u8>>());
            let luma_i16 = leak(luma_u8.iter().map(|&x| x as i16).collect::<Vec<_>>());
            let luma_f32 = leak(luma_u8.iter().map(|&x| x as f32).collect::<Vec<_>>());

            suite.compare(format!("rgb8_to_variance_{sz}"), |g| {
                g.config().max_rounds(60);
                g.throughput(Throughput::Elements(n as u64));
                g.bench("f32", |b| b.iter(|| black_box(var_rgb8_f32(rgb))));
                g.bench("i32_fixedpoint", |b| {
                    b.iter(|| black_box(var_rgb8_i32(rgb)))
                });
                g.bench("i16_luma", |b| b.iter(|| black_box(var_rgb8_i16(rgb))));
            });

            suite.compare(format!("luma_minmax_{sz}"), |g| {
                g.config().max_rounds(60);
                g.throughput(Throughput::Elements(n as u64));
                g.bench("u8", |b| b.iter(|| black_box(minmax_u8(luma_u8))));
                g.bench("i16", |b| b.iter(|| black_box(minmax_i16(luma_i16))));
                g.bench("f32", |b| b.iter(|| black_box(minmax_f32(luma_f32))));
            });
        }
    });
    zenbench::postprocess_result(&result);
    let out = "/mnt/v/output/imazen-26-features/bench_kernel_format_2026-06-18.json";
    if let Err(e) = result.save(out) {
        eprintln!("save failed: {e}");
    } else {
        eprintln!("saved {out}");
    }
}
