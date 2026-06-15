//! Hand-tuned variance bake-off: settles whether the autovec f32 disadvantage
//! (the 7–17× gap in bench_linear_repr) survives a *hand-tuned* f32 kernel, or
//! was purely the serial FP-reduction artifact. Variance is the contested case
//! (the squared term widens i16→i64); minmax/linearize were already decisive.
//!
//! Variants (all on the same i12-range luma plane, 0..4095):
//!
//! - `f32_serial` — single accumulator (LLVM can't reassociate FP → serial)
//! - `f32_multi16` — 16 independent accumulators (≈ tier1's hand-tuned f32x8)
//! - `i16_i64_direct` — widen each to i64, 16 accumulators (simple integer)
//! - `i16_i32_flush` — i32 accumulators, flush to i64 every 120 chunks (i16's
//!   best: hot loop stays i32, no per-element i64 widen)
//! - `u16_i64_direct` — unsigned counterpart
//!
//! Run: cargo run --release --example bench_repr_handtuned

use zenbench::prelude::*;

const K: usize = 16;

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

fn var_f32_serial(p: &[f32]) -> f32 {
    let (mut s, mut sq) = (0.0f32, 0.0f32);
    for &x in p {
        s += x;
        sq = x.mul_add(x, sq);
    }
    sq - s
}

fn var_f32_multi(p: &[f32]) -> f32 {
    let mut s = [0.0f32; K];
    let mut sq = [0.0f32; K];
    let mut it = p.chunks_exact(K);
    for c in &mut it {
        for j in 0..K {
            s[j] += c[j];
            sq[j] = c[j].mul_add(c[j], sq[j]);
        }
    }
    let (mut st, mut sqt) = (0.0f32, 0.0f32);
    for j in 0..K {
        st += s[j];
        sqt += sq[j];
    }
    for &x in it.remainder() {
        st += x;
        sqt = x.mul_add(x, sqt);
    }
    sqt - st
}

fn var_i16_i64_direct(p: &[i16]) -> i64 {
    let mut s = [0i64; K];
    let mut sq = [0i64; K];
    let mut it = p.chunks_exact(K);
    for c in &mut it {
        for j in 0..K {
            let v = c[j] as i64;
            s[j] += v;
            sq[j] += v * v;
        }
    }
    let (mut st, mut sqt) = (0i64, 0i64);
    for j in 0..K {
        st += s[j];
        sqt += sq[j];
    }
    for &x in it.remainder() {
        let v = x as i64;
        st += v;
        sqt += v * v;
    }
    sqt - st
}

fn var_i16_i32_flush(p: &[i16]) -> i64 {
    // i12 max 4095 → 4095² ≈ 16.7M; 120·16.7M ≈ 2.0e9 < i32::MAX. Flush before.
    const FLUSH: usize = 120;
    let mut s_acc = [0i32; K];
    let mut sq_acc = [0i32; K];
    let (mut st, mut sqt) = (0i64, 0i64);
    let mut cnt = 0usize;
    let mut it = p.chunks_exact(K);
    for c in &mut it {
        for j in 0..K {
            let v = c[j] as i32;
            s_acc[j] += v;
            sq_acc[j] += v * v;
        }
        cnt += 1;
        if cnt == FLUSH {
            for j in 0..K {
                st += s_acc[j] as i64;
                sqt += sq_acc[j] as i64;
                s_acc[j] = 0;
                sq_acc[j] = 0;
            }
            cnt = 0;
        }
    }
    for j in 0..K {
        st += s_acc[j] as i64;
        sqt += sq_acc[j] as i64;
    }
    for &x in it.remainder() {
        let v = x as i64;
        st += v;
        sqt += v * v;
    }
    sqt - st
}

fn var_u16_i64_direct(p: &[u16]) -> u64 {
    let mut s = [0u64; K];
    let mut sq = [0u64; K];
    let mut it = p.chunks_exact(K);
    for c in &mut it {
        for j in 0..K {
            let v = c[j] as u64;
            s[j] += v;
            sq[j] += v * v;
        }
    }
    let (mut st, mut sqt) = (0u64, 0u64);
    for j in 0..K {
        st += s[j];
        sqt += sq[j];
    }
    for &x in it.remainder() {
        let v = x as u64;
        st += v;
        sqt += v * v;
    }
    sqt.wrapping_sub(st)
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
            let li = leak(lf.iter().map(|&x| x as i16).collect::<Vec<_>>());
            let lu = leak(lf.iter().map(|&x| x as u16).collect::<Vec<_>>());

            suite.compare(format!("variance_handtuned_{sz}"), |g| {
                g.config().max_rounds(80);
                g.throughput(Throughput::Elements(n as u64));
                g.bench("f32_serial", |b| b.iter(|| black_box(var_f32_serial(lf))));
                g.bench("f32_multi16", |b| b.iter(|| black_box(var_f32_multi(lf))));
                g.bench("i16_i64_direct", |b| {
                    b.iter(|| black_box(var_i16_i64_direct(li)))
                });
                g.bench("i16_i32_flush", |b| {
                    b.iter(|| black_box(var_i16_i32_flush(li)))
                });
                g.bench("u16_i64_direct", |b| {
                    b.iter(|| black_box(var_u16_i64_direct(lu)))
                });
            });
        }
    });
    zenbench::postprocess_result(&result);
    let out = "/mnt/v/output/imazen-26-features/bench_repr_handtuned_2026-06-14.json";
    if let Err(e) = result.save(out) {
        eprintln!("save failed: {e}");
    } else {
        eprintln!("saved {out}");
    }
}
