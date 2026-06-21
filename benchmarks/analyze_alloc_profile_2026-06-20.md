# `analyze_features` allocation profile — per-row scratch hoist (2026-06-20)

Deeper profile of the full feature pass after the transcendental work
(`transcendental_breakdown_2026-06-21.md`) showed SDR transcendentals were
already < 0.1 ns/px. This pass attacks the **non-transcendental** cost at the
function level with callgrind (instruction counts) + DHAT (heap).

## Method

Harness: `examples/profile_analyze.rs` — runs
`analyze_features(FeatureSet::SUPPORTED)` over a deterministic pseudo-random
1024×1024 RGB8 image, N times. No `target-cpu=native` (release default).

```
cargo build --release --features experimental,hdr --example profile_analyze
valgrind --tool=callgrind --cache-sim=no --branch-sim=no \
  --callgrind-out-file=/tmp/cg.out target/release/examples/profile_analyze 3
callgrind_annotate /tmp/cg.out
valgrind --tool=dhat --dhat-out-file=/tmp/dhat.out \
  target/release/examples/profile_analyze 3
```

Commit: parent `2329a9a7` (edge rsqrt work). Host: WSL2, x86-64 AVX-512.

## Finding

callgrind put **`__memset_avx2` at 10.7%** of all instructions — surprising for
a per-image pass. DHAT resolved it: **11,014 heap allocations over 3 iters ≈
3,671 per `analyze_features` call**, almost all of them `vec![0.0; width]`
scratch re-allocated *every sampled row*:

| site | allocs / call | each | what |
|---|--:|--:|---|
| `tier1::accumulate_laplacian_simd` | ~1,341/3 ≈ 447 rows × 3 | ~4 KB | `prev_l` / `cur_l` / `next_l` luma scratch |
| `tier2_chroma::process_row_group_simd` | ~256 groups × 9 | ~4 KB | `y0/cb0/cr0/y1/cb1/cr1/y2/cb2/cr2` deinterleave scratch |

Both buffers depend only on `width` (constant across rows) and are fully
overwritten before any read — classic hoist candidates.

## Fix

Hoist the scratch to the driver, allocate once, pass `&mut` into the kernel:
- tier1: 3 `&mut [f32]` threaded `extract_tier1_into_dispatch` →
  `accumulate_laplacian_dispatch` → `accumulate_laplacian_simd`; allocation
  gated on `dispatch.wants_laplacian`.
- tier2: a `RowGroupScratch` struct allocated once in
  `image_sharpness_breakdown`, sub-sliced to `row0_len` / `row12_len` in the
  kernel.

Numerically identical (no value changes) — verified by the `golden_is_stable`
content-hash tripwire (unchanged) + all 212 lib tests.

## Measured (3 iters, 1 MP)

| metric | before | after | Δ |
|---|--:|--:|--:|
| total instructions | 424.5 M | **377.3 M** | **−11.1%** |
| `__memset_avx2` | 45.3 M (10.7%) | 0.58 M (0.15%) | **−98.7%** |
| heap allocs | 11,014 (3,671/call) | 115 (38/call) | **−99.0%** |

## New hotspot ranking (self %, post-hoist)

All remaining cost is now genuine compute — the allocation waste is gone.

| function | self % | what | next-lever notes |
|---|--:|---|---|
| `tier1::accumulate_row_simd_v3` | 27.1% | per-pixel variance/chroma/edge/skin stats | already AVX2/512; algorithmic only |
| `palette::scan_and_count_gray_v3` | 11.7% | distinct-colour bin counting | candidate — full-image scan |
| `tier3::dct_stats` | 9.9% | 8×8 DCT statistics | per-block FMA, near-optimal |
| `tier1::accumulate_laplacian_simd_v3` | 6.8% | laplacian_variance | recomputes luma 3× per row (rolling-buffer opportunity) |
| `tier3::populate_tier3` | 3.4% | DCT orchestration | |
| `tier2_chroma::process_row_group_simd_v3` | 3.2% | Cb/Cr 2nd-diff sharpness | |

Next opportunities, in order: (1) palette gray-scan (12%, full-image), (2) the
laplacian's 3×-redundant per-row luma recompute (the kernel re-derives luma for
prev/cur/next every row, where `cur` of row *i* is `prev` of row *i+1*).

## Size sweep (wall-clock, `profile_analyze sweep`)

The per-pixel cost is **not** globally linear — each tier subsamples above its
own budget, so above those knees the work is capped and ns/px falls as 1/px.

| side | Mpx | ns/call | ns/px | regime |
|--:|--:|--:|--:|---|
| 64 | 0.004 | 130 µs | 31.8 | full density (all tiers uncapped) |
| 128 | 0.016 | 497 µs | 30.4 | full density |
| 256 | 0.066 | 2.26 ms | 34.4 | tier3 at its 1024-block cap |
| 512 | 0.262 | 4.41 ms | 16.8 | tier3 + tier2 capped |
| 1024 | 1.049 | 7.98 ms | 7.6 | tier1 near its 500K-px cap |
| 2048 | 4.194 | 12.4 ms | 3.0 | all tiers sampling-capped |

Full-density fit (≤128², every tier uncapped):
**ns/call ≈ 7.7 µs fixed + 29.9 ns/px · pixels**.

* **Fixed per-call overhead is small** (~7.7 µs: result struct + RowStream setup
  + tier dispatch) — not a focus target.
* **Full-density per-pixel cost ≈ 30 ns/px** is paid in full for images up to
  ~256² — i.e. **web thumbnails / small images**, the web-focused case. That's
  exactly where tier1 `accumulate_row` (27%), palette gray-scan (12%) and tier3
  `dct_stats` (10%) all run at full density, so kernel optimisations there help
  the small-image path most.
* **Large images are already sampling-bounded** (1 MP → 7.6 ns/px, 4 MP → 3.0)
  — the analyzer does budget-capped work, so kernel speedups help proportionally
  less there (most pixels are skipped).
