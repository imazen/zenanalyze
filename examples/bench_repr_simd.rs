//! REAL hand-tuned SIMD variance bake-off — f32x8 vs i32x8 via magetypes
//! (`#[magetypes]` + `incant!`), the apples-to-apples the autovec
//! `bench_repr_handtuned` couldn't do. Both use 4 independent accumulators to
//! break the reduction dependency chain (the lesson from the autovec run, where
//! plain multi-accumulator f32 still wouldn't vectorize).
//!
//! Why i32x8, not i16x16: the variance *square* needs i32 lanes regardless
//! (luma² overflows i16), and magetypes exposes i16→i32 widening only on the
//! per-platform concrete types, not the portable generic API. So the integer
//! reduction lives in i32x8 either way; i16x16 would only halve the *load*
//! count. This bench measures whether the integer reduction beats f32 with both
//! hand-tuned — if it does and loads dominate, per-platform i16x16 is the
//! follow-on.
//!
//! Run: cargo run --release --features experimental --example bench_repr_simd

use archmage::prelude::*;
use zenbench::prelude::*;

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

// ---- f32x8: 4 accumulators, f32 lanes reduced to f64 at the end ----
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn var_f32_simd(token: Token, p: &[f32]) -> f64 {
    let mut s = [f32x8::zero(token); 4];
    let mut sq = [f32x8::zero(token); 4];
    let mut it = p.chunks_exact(32);
    for c in &mut it {
        for k in 0..4 {
            let v = f32x8::from_slice(token, &c[k * 8..k * 8 + 8]);
            s[k] += v;
            sq[k] = v.mul_add(v, sq[k]);
        }
    }
    let (mut st, mut sqt) = (0.0f64, 0.0f64);
    for k in 0..4 {
        st += s[k].reduce_add() as f64;
        sqt += sq[k].reduce_add() as f64;
    }
    for &x in it.remainder() {
        st += x as f64;
        sqt += (x as f64) * (x as f64);
    }
    sqt - st
}

// ---- i32x8: 4 accumulators, i32 lanes flushed to i64 every 120 chunks ----
// luma is i12 (0..4095) so luma² ≤ 16.7M fits i32; per lane 120·16.7M ≈ 2.0e9 <
// i32::MAX → flush (via to_array, summing lanes into i64) before overflow.
#[magetypes(define(i32x8), v4, v3, neon, wasm128, scalar)]
fn var_i32_simd(token: Token, p: &[i32]) -> i64 {
    const FLUSH: usize = 120;
    let mut s = [i32x8::zero(token); 4];
    let mut sq = [i32x8::zero(token); 4];
    let (mut st, mut sqt) = (0i64, 0i64);
    let mut cnt = 0usize;
    let mut it = p.chunks_exact(32);
    for c in &mut it {
        for k in 0..4 {
            let v = i32x8::from_slice(token, &c[k * 8..k * 8 + 8]);
            s[k] += v;
            sq[k] += v * v;
        }
        cnt += 1;
        if cnt == FLUSH {
            for k in 0..4 {
                for lane in s[k].to_array() {
                    st += lane as i64;
                }
                for lane in sq[k].to_array() {
                    sqt += lane as i64;
                }
                s[k] = i32x8::zero(token);
                sq[k] = i32x8::zero(token);
            }
            cnt = 0;
        }
    }
    for k in 0..4 {
        for lane in s[k].to_array() {
            st += lane as i64;
        }
        for lane in sq[k].to_array() {
            sqt += lane as i64;
        }
    }
    for &x in it.remainder() {
        let v = x as i64;
        st += v;
        sqt += v * v;
    }
    sqt - st
}

fn var_f32(p: &[f32]) -> f64 {
    incant!(var_f32_simd(p))
}
fn var_i32(p: &[i32]) -> i64 {
    incant!(var_i32_simd(p))
}

fn main() {
    let result = zenbench::run(|suite| {
        for (sz, n) in [("1MP", 1024usize * 1024), ("4MP", 2048usize * 2048)] {
            let mut s: u32 = 0xC0FFEE ^ n as u32;
            let mut next = || {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                s >> 12 & 0xFFF // i12 range 0..4095
            };
            let lf = leak((0..n).map(|_| next() as f32).collect::<Vec<_>>());
            let li = leak(lf.iter().map(|&x| x as i32).collect::<Vec<_>>());

            suite.compare(format!("variance_simd_{sz}"), |g| {
                g.config().max_rounds(80);
                g.throughput(Throughput::Elements(n as u64));
                g.bench("f32x8", |b| b.iter(|| black_box(var_f32(lf))));
                g.bench("i32x8", |b| b.iter(|| black_box(var_i32(li))));
            });
        }
    });
    zenbench::postprocess_result(&result);
    let out = "/mnt/v/output/imazen-26-features/bench_repr_simd_2026-06-14.json";
    if let Err(e) = result.save(out) {
        eprintln!("save failed: {e}");
    } else {
        eprintln!("saved {out}");
    }
}
