# Tier 1 dispatch and linked size — 2026-11-23

The adopted leaf change shares the Wasm128 edge/chroma-gradient kernel across
SKIN variants, keeping BT601 and FULL specialized. Statistics and skin remain
fused. Wasm scalar retains the original fused implementation. Native code is
unchanged after conditional compilation: the complete linked `.text` sections
compare byte-for-byte on both i265 and r5600g, including both benchmark binaries.
All v4, v3, NEON, Wasm128 and scalar dispatch entries remain. Public APIs,
Cargo defaults, arithmetic order, reductions, flush boundaries and tails are
unchanged. Laplacian specialization remains unchanged.

**Result:** the leaf saves 11,414 stripped Wasm bytes (1.44%) and 1,729 gzipped
bytes (0.76%). There is no native size or speed improvement. Six native
sharing variants were rejected after measurement because they slowed Tier 1
or scalar analysis. Keeping the original native body avoids accepting those
regressions. This is a bounded Wasm size adoption, not the originally hoped-for
large native reduction. Wasm execution speed was not benchmarked.

## Baselines and inherited HDR state

The requested initial baseline `0bc01d3f` was fully measured before leaf edits.
Origin subsequently advanced to `1777b1a2`, adopting zenpixels/convert
`fde59b96632557592cffa9b7301acd8db96ed078`. The implementation is rebased on
that revision. Its source commit is `4d5799376b00eddc8d938e191bef51fb89ea9c91`. All leaf deltas below compare
unedited `1777b1a2` with this implementation; initial-baseline numbers are
shown separately so upstream gains are not credited to the leaf.

**Inherited failure, explicitly reproduced on unedited upstream:** the
reference-mode golden assertion reports `feat_dct_compressibility_y[44]`
`56.020832` against pinned `55.6875`: deviation 0.595% exceeds the unchanged
0.5% tolerance. Comparing `0bc01d3f` with unedited `1777b1a2` finds exactly
64 changed AVX2 linear-light entries (66 scalar); gamma-light entries do not
change. These are changed entries, not 64 failing tests. One reference-mode
assertion exceeds tolerance. Both upstream and candidate reproduce exit 101
and the identical assertion and complete output bits. The ordinary portable
mode golden test passes. The owner explicitly directed landing the pure leaf
with this inherited state documented.

The isolated upstream copy was verified against all 1,110 tracked blobs of
`1777b1a2`, with zero differences, before execution. Candidate-versus-upstream
dumps are bit-identical, including those exact 64 linear-light entries and
DCT compressibility. `src/hdr_compat.rs`, `Cargo.toml`, every golden file/value,
and every tolerance remain exactly as upstream. No dependency or HDR repair
is included.

## Linked size and attribution

All sizes are bytes. Native attribution uses `tier_kernels`, default features
plus `_dev`, i265, Rust 1.99.1 / LLVM 23.1.1, release thin LTO, one codegen unit,
debug enabled and `RUSTFLAGS='-C strip=none'`. Executable paths are taken from
Cargo JSON artifact records. Transfer size uses GNU strip.

| Measurement | Initial 0bc01d3f | Upstream 1777b1a2 | Leaf AFTER | Leaf saving |
|---|---:|---:|---:|---:|
| Native benchmark .text | 2,632,118 | 2,501,318 | 2,501,318 | 0 (0.00%) |
| Native stripped ELF | 3,180,768 | 3,038,840 | 3,038,840 | 0 (0.00%) |
| Wasm stripped | 831,462 | 794,739 | 783,325 | 11,414 (1.44%) |
| Wasm gzip -n -9 | 229,086 | 227,032 | 225,303 | 1,729 (0.76%) |
| Wasm code payload | 772,514 | 735,097 | 723,883 | 11,214 (1.53%) |

Native row symbols remain **16 bodies / 83,047 bytes**; Laplacian remains
**4 bodies / 12,392 bytes**. Named bodies are counted by unique address/size,
not aliases. LTO inlines additional tier bodies into dispatch; named symbol
counts are not a dispatch-coverage claim. Native `__simd_inner_*` counts remain
**0 → 0**. `objdump -d -C` reports **0 → 0 zenanalyze-owned vmaskmov**.
Both consumers contain two masked moves in the unchanged dependency function
`zenpixels_convert::convert::convert_kernels::__arcane_multiply_color_channels_tier_v3`.
There was no owned masked-move defect to repair.

Wasm sizes use a temporary cdylib calling `analyze_features_rgb8` with
`FeatureSet::SUPPORTED`, folding every typed result/absence into an exported
u64 checksum. It uses default zenanalyze features, wasm-bindgen 0.2.130,
Rust 1.99.0 / LLVM 23.1.1 on i134, thin LTO and one codegen unit;
`wasm32-unknown-unknown`, `RUSTFLAGS='-C target-feature=+simd128'`.
`wasm-tools strip --all` removes custom sections; `gzip -n -9` supplies the
compressed size. This measures the linked probe, not JavaScript packaging.
Source, actual lockfiles and raw/stripped/gzipped artifacts are archived.
A separate WASI+simd128 probe executes the analyzer with wasmtime for parity.

```sh
RUSTFLAGS='-C strip=none' cargo build --release --bench tier_kernels --features _dev
size -A "$BINARY"
nm -CS --size-sort --defined-only "$BINARY"
objdump -d -C "$BINARY"
strip -o stripped-elf "$BINARY"
RUSTFLAGS='-C target-feature=+simd128' cargo build --release \
  --manifest-path "$PROBE/Cargo.toml" --target wasm32-unknown-unknown
wasm-tools strip --all probe.wasm -o stripped.wasm
gzip -n -9 -c stripped.wasm > stripped.wasm.gz
```

## r5600g benchmark results

Both groups ran through `remote_bench.sh r5600g zenanalyze ...`, serialized
with the shared worker flock and externally pinned using **taskset -c 4**.
All invocations use **--rounds=30 --cpu-policy=unrestricted --format=llm**.
Zenbench reports **UNVERIFIED_UNRESTRICTED**: frequency/continuous quietness
are not verified. Completed packets record effective thread affinity `[4]`.
No `target-cpu=native`. Worker Rust 1.98.0 / LLVM 22.1.8, thin LTO, one codegen
unit, `RUSTFLAGS='-C strip=none'`; nice 19 and run-heavy 10 GiB/three jobs.

Unchanged kernel input is deterministic noise plus patches at 1024²;
isolation uses 512² and 2048² with the shipped sampling budgets. The default
feature build lacks `hdr`: the existing depth arm requests an empty feature
set but still executes the analyzer's unconditional Tier 1 base scan. It is
not a depth-kernel speed measurement. No odd-width timing arm was added;
executed exact probes cover odd widths and poisoned padding.

Initial numbers are the median of two independent process medians. Upstream
and AFTER are the median of six process medians, from three fixed ABBA blocks
(BEFORE, AFTER, AFTER, BEFORE) per group. Each process has 30 rounds per arm.
The initial series is not interleaved with the final series and is contextual.
No rounds or outliers were removed. Times below are microseconds per call;
Δ is throughput, `100 * (BEFORE / AFTER - 1)`.

| Group / arm | Initial | Upstream BEFORE | AFTER | Δ throughput |
|---|---:|---:|---:|---:|
| isolation / 2048x2048 / scalar | 56,799.191 | 57,144.582 | 57,240.978 | -0.168% |
| isolation / 2048x2048 / v3(avx2) | 17,143.979 | 17,073.239 | 17,002.707 | +0.415% |
| isolation / 512x512 / scalar | 28,719.968 | 28,621.071 | 28,768.387 | -0.512% |
| isolation / 512x512 / v3(avx2) | 6,214.376 | 6,177.831 | 6,279.758 | -1.623% |
| kernels / depth / scalar | 10,204.133 | 10,033.278 | 10,009.040 | +0.242% |
| kernels / depth / v3(avx2) | 3,224.693 | 3,234.060 | 3,206.165 | +0.870% |
| kernels / palette / scalar | 11,228.670 | 11,175.693 | 11,261.883 | -0.765% |
| kernels / palette / v3(avx2) | 4,173.375 | 4,114.885 | 4,138.806 | -0.578% |
| kernels / tier1_extras / scalar | 37,116.908 | 36,656.851 | 36,931.536 | -0.744% |
| kernels / tier1_extras / v3(avx2) | 6,577.417 | 6,487.931 | 6,538.332 | -0.771% |
| kernels / tier1_full / scalar | 27,286.796 | 27,327.620 | 27,144.292 | +0.675% |
| kernels / tier1_full / v3(avx2) | 6,502.664 | 6,390.201 | 6,358.548 | +0.498% |
| kernels / tier2_chroma / scalar | 12,439.237 | 12,341.787 | 12,389.309 | -0.384% |
| kernels / tier2_chroma / v3(avx2) | 3,951.175 | 3,917.528 | 3,933.408 | -0.404% |
| kernels / tier3_dct / scalar | 18,823.824 | 18,767.434 | 18,841.234 | -0.392% |
| kernels / tier3_dct / v3(avx2) | 5,590.392 | 5,518.070 | 5,544.251 | -0.472% |

The final native `.text` is byte-identical to baseline in both worker
executables. The timing scatter therefore is not evidence of a native leaf
speedup. Some paired intervals exclude zero despite identical executable
code: unrestricted shared-host/process effects remain. We report those
intervals rather than labeling every negative measurement a pass or claiming
statistical proof of zero slowdown.

Six nearest before/after process pairs per arm, log throughput effects,
20,000 bootstrap resamples, seed 20261123; percentile 95% intervals:

| Group / arm | Paired Δ throughput | 95% interval |
|---|---:|---:|
| isolation / 2048x2048 / scalar | -0.265% | [-1.291%, +0.795%] |
| isolation / 2048x2048 / v3(avx2) | +0.362% | [-0.229%, +0.934%] |
| isolation / 512x512 / scalar | -0.296% | [-0.813%, +0.405%] |
| isolation / 512x512 / v3(avx2) | -1.276% | [-2.211%, -0.281%] |
| kernels / depth / scalar | +0.058% | [-1.015%, +1.127%] |
| kernels / depth / v3(avx2) | +0.294% | [-1.318%, +1.790%] |
| kernels / palette / scalar | -0.485% | [-1.680%, +0.612%] |
| kernels / palette / v3(avx2) | -0.997% | [-2.447%, +0.177%] |
| kernels / tier1_extras / scalar | -0.767% | [-1.373%, -0.243%] |
| kernels / tier1_extras / v3(avx2) | -1.155% | [-2.253%, -0.185%] |
| kernels / tier1_full / scalar | +0.318% | [-1.455%, +1.472%] |
| kernels / tier1_full / v3(avx2) | +0.388% | [-1.049%, +1.991%] |
| kernels / tier2_chroma / scalar | +0.099% | [-0.580%, +1.107%] |
| kernels / tier2_chroma / v3(avx2) | -0.137% | [-1.183%, +0.858%] |
| kernels / tier3_dct / scalar | -0.735% | [-1.529%, -0.125%] |
| kernels / tier3_dct / v3(avx2) | -0.668% | [-2.310%, +1.240%] |

Runnable reproduction (preserve each revision's binary before rebuilding):

```sh
cargo bench --features _dev --bench tier_kernels --bench tier_isolation --no-run
"$BINARY" --rounds=30 --cpu-policy=unrestricted --format=llm
```

## Rejected native candidates

These private experiments are not in the landing. Every one preserved exact
feature bits; throughput rejected them. All raw data and candidate source
snapshots remain in the evidence archive.

| Candidate | Native .text | Stripped Wasm | Measured rejection example |
|---|---:|---:|---|
| Runtime luma weights | 2,354,102 | 672,773 | AVX2 full paired −3.06%, 95% [−4.51%, −1.55%] |
| Separate skin pass | 2,372,358 | 686,739 | extras paired −4.66% AVX2 / −6.65% scalar |
| Shared edges, all tiers | 2,466,518 | 779,955 | scalar extras paired −4.16% |
| TypeId-selected fused scalar | 2,486,246 | 783,317 | scalar extras paired −2.22% |
| Explicit inline scalar | 2,486,918 | 783,317 | scalar extras paired −3.67% |
| Explicit outlined scalar | 2,493,830 | 783,317 | scalar extras paired −1.97% |

## Exact output and checks

**96,216 / 96,216 entries compare bit-for-bit with unedited 1777b1a2:**

- Scalar, AVX2 and executed AVX-512: 6,072 golden entries each, 117 features ×
  26 corpus images × gamma/linear light. This includes SDR, PQ/HLG, wide gamut,
  float/16-bit input, alpha and super-white cases.
- Scalar, AVX2, executed AVX-512 and executed Wasm128: 19,500 public-API
  probe entries each. Widths 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257, 1031;
  heights 1 and 19; six poisoned padding bytes per RGB row; Bt709/Bt2020/P3;
  gamma/linear modes; all four FULL/SKIN combinations plus SUPPORTED.
  Explicit absence, f32 bits and integer/bool values are compared.
- Six SIMD math tests pass, including unchanged
  `STABLE_GOLDEN_HASH = 0x6b404d4c5e62e664`.

AVX-512 executes on dev. Bare magetypes `v4` requires consumer
`cfg(feature="avx512")`, while the default manifest enables the dependency's
AVX-512 feature. Extra validation uses a Rustc wrapper that adds this cfg only
to zenanalyze, on both unedited upstream and candidate, without editing Cargo
features or dependencies. Named v4 bodies are verified in `nm`; default v4
selection then executes on capable hardware. Initial ceiling-only v4 captures
are archived under that label and are not counted as actual v4 execution.
Each candidate tier is compared to the same tier of upstream; no cross-tier
bit equality is inferred from tolerances.

Required gates pass: fmt `--check -p zenanalyze`; clippy `--all-targets -- -D
warnings`; release `--lib --tests` (**211 passed, four existing ignored**);
unknown-Wasm check; `just api-doc-check` (**one passed**); doctests (**zero
executed, three existing ignored**). Extra release tests with
`--features experimental,hdr,_dev` pass **232 tests, four existing ignored**.
The explicit reference-mode golden still fails identically on upstream and
candidate; it is not skipped or weakened. Local Cargo emitted a nonfatal
registry-cache permission warning; all build/check commands returned zero.

run-heavy observed peaks: final native build 0.79 GiB, required gates 0.74 GiB,
Wasm size probe 0.45 GiB, r5600g build/benchmark batch 0.74 GiB. These are
measured process-tree RSS observations, not extrapolated allocation claims.

## Largest linked analyzer symbols

Top 40 named executable symbols, `nm -CS --defined-only`, sorted by size.
BEFORE and AFTER attribution agree exactly, so one table represents both.
`zenanalyze::` is omitted; full records and disassembly are archived.

| Bytes | Symbol |
|---:|---|
| 100,307 | `tier1::accumulate_row_dispatch::<f32>` |
| 94,014 | `tier1::accumulate_row_dispatch::<u8>` |
| 29,666 | `<feature::RawAnalysis>::into_results` |
| 27,699 | `tier3::populate_tier3::<f32>` |
| 27,593 | `tier3::populate_tier3::<u8>` |
| 15,989 | `tier1::extract_tier1_into_dispatch::<f32>` |
| 15,249 | `tier1::extract_tier1_into_dispatch::<u8>` |
| 11,957 | `tier2_chroma::populate_tier2::<f32>` |
| 11,402 | `chroma_subsample_loss::block_subsample_dct_loss` |
| 11,389 | `tier2_chroma::populate_tier2::<u8>` |
| 10,636 | `tier2_chroma::__arcane_process_row_group_simd_v3::<f32>` |
| 8,908 | `analyze_features` |
| 8,301 | `tier2_chroma::__arcane_process_row_group_simd_v3::<u8>` |
| 7,202 | `tier1::__arcane_accumulate_row_simd_v3::<false, true, true, f32>` |
| 7,109 | `tier1::__arcane_accumulate_row_simd_v3::<false, true, true, u8>` |
| 6,987 | `tier1::__arcane_accumulate_row_simd_v3::<true, true, true, u8>` |
| 6,924 | `tier1::__arcane_accumulate_row_simd_v3::<true, true, true, f32>` |
| 6,490 | `tier1::__arcane_accumulate_row_simd_v3::<false, true, false, f32>` |
| 6,312 | `tier1::__arcane_accumulate_row_simd_v3::<false, true, false, u8>` |
| 6,173 | `tier1::__arcane_accumulate_row_simd_v3::<true, true, false, f32>` |
| 6,139 | `tier1::__arcane_accumulate_row_simd_v3::<true, true, false, u8>` |
| 5,898 | `<row_stream::RowStream>::new` |
| 5,888 | `grayscale::__arcane_scan_strict_grayscale_impl_v3` |
| 5,556 | `tier3::__arcane_dct2d_8_three_planes_simd_v3` |
| 5,544 | `tier3::dct2d_8_three_planes_simd_scalar` |
| 4,888 | `tier1::__arcane_accumulate_row_simd_v3::<false, false, true, f32>` |
| 4,772 | `tier1::__arcane_accumulate_row_simd_v3::<true, false, true, f32>` |
| 4,683 | `palette::__arcane_scan_and_count_no_gray_v4` |
| 4,603 | `palette::__arcane_scan_and_count_no_gray_v4x` |
| 4,407 | `xyb_color_loss::xyb_bquarter_chroma_loss_from_stream` |
| 4,291 | `palette::__arcane_scan_and_count_gray_v4` |
| 4,096 | `palette::__arcane_scan_and_count_gray_v4x` |
| 4,093 | `tier1::__arcane_accumulate_row_simd_v3::<false, false, false, f32>` |
| 3,982 | `tier1::__arcane_accumulate_row_simd_v3::<true, false, false, f32>` |
| 3,416 | `tier1::__arcane_accumulate_row_simd_v3::<true, false, true, u8>` |
| 3,388 | `<row_stream::RowStream>::borrow_row` |
| 3,333 | `tier1::__arcane_accumulate_row_simd_v3::<false, false, true, u8>` |
| 3,275 | `<row_stream::RowStream>::fetch_f32_into` |
| 3,190 | `<row_stream::RowStream>::new_normalized_linear` |
| 3,177 | `<row_stream::RowStream>::fetch_into` |

Raw evidence: tower coefficient storage, `output/zenanalyze/2026-11-23-r5600g-dispatch-4d579937/`. This contains initial baseline captures, unedited-upstream source verification, binaries, source and lock identities, exact dumps, rejected experiments, immutable completed benchmark packets, probe/harness sources and check logs. Temporary targets and probes are cleaned after evidence preservation.
