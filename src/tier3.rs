//! Tier 3: luma histogram entropy + high-frequency DCT energy ratio
//! + derived likelihoods (text / screen-content / natural).
//!
//! Direct ports of `coefficient::analysis::evalchroma_ext` (Tier 3)
//! and `image_adaptive::compute_derived_likelihoods`. Numbers are
//! kept identical so the parity example matches within an f32
//! epsilon.

use super::feature::RawAnalysis;
use super::row_stream::RowStream;
use archmage::{incant, magetypes};

/// Cap on sampled 8×8 luma blocks for the Tier 3 DCT pass
/// (`high_freq_energy_ratio`, `dct_compressibility_y/uv`,
/// `patch_fraction`).
///
/// **Tradeoff**: a 139-image convergence study showed:
/// - 256 blocks: 11-16 % p50 error on DCT-energy features, 100 % on
///   `patch_fraction`. ~1 ms at 4 MP.
/// - 1024 blocks: 4-6 % p50 error on DCT-energy, ~57 % on
///   `patch_fraction` (still structurally limited). ~1 ms at 4 MP
///   *with* the three-plane f32x8 DCT kernel (was ~4 ms pre-SIMD).
/// - 4096 blocks: convergence floor, ~+2 ms at 4 MP over 1024.
///
/// **Default = 1024.** The cross-block f32x8 DCT cut the per-block
/// cost ~8-15× vs the pre-SIMD scalar version, so the accuracy tier
/// (4-6 % error on the three DCT-energy features) costs only ~+1 ms
/// at 8 MP over the 256-block sloppy tier — well worth it. Earlier
/// revisions defaulted to 256 because the scalar DCT made 1024 cost
/// 3-4 ms; with SIMD that pressure is gone.
///
/// `patch_fraction` is still structurally sample-limited at any block
/// cap below dense-scan; bump to 4096+ if that feature matters, or
/// accept the 57 %-at-1024 / 100 %-at-256 error and rely on the
/// other DCT features for content classification.
///
/// Re-exported as `AnalyzerConfig::default().hf_max_blocks`.
pub(crate) const DEFAULT_HF_MAX_BLOCKS: usize = 1024;

/// Fill in `high_freq_energy_ratio` and `luma_histogram_entropy`.
///
/// The `run_dct` runtime bool gates the per-block DCT pass — when
/// **false**, callers asked only for `LumaHistogramEntropy` /
/// `LineArtScore` (or neither) and the entire `dct_stats()` call is
/// skipped, saving the ~0.97 ms-per-Mpx cost. The cheap luma
/// histogram pass still runs unconditionally.
///
/// Bool rather than `const` const-generic to avoid doubling the
/// outer dispatch table (PAL/T2/T3/ALPHA × DCT = 32 monomorphizations
/// vs the current 16). The single runtime branch costs ~1 ns vs the
/// ~0.97 ms saving — const-fold gain on this gate is negligible
/// because the saved work is the entire dct_stats body, not the
/// gate check.
///
/// Set by `feature::DCT_NEEDED_BY` in the dispatcher.
///
/// `hf_max_blocks` caps the number of 8×8 luma blocks sampled for the
/// high-frequency DCT energy ratio. Pass [`DEFAULT_HF_MAX_BLOCKS`] to
/// match the oracle-trained reference; lower for proxy-server speed.
pub fn populate_tier3(
    out: &mut RawAnalysis,
    stream: &mut RowStream<'_>,
    hf_max_blocks: usize,
    run_dct: bool,
) {
    let h_stats = luma_histogram_stats(stream);
    out.luma_histogram_entropy = h_stats.entropy;
    #[cfg(feature = "composites")]
    {
        out.line_art_score = h_stats.line_art_score;
    }
    let _ = h_stats;
    if run_dct {
        let dct = dct_stats(stream, hf_max_blocks);
        out.high_freq_energy_ratio = dct.high_freq_ratio;
        // The libwebp α metrics and patch_fraction land in cfg-gated
        // RawAnalysis fields; the DCT pass that produces them runs
        // only when the dispatcher set DCT=true.
        #[cfg(feature = "experimental")]
        {
            out.dct_compressibility_y = dct.compressibility_y;
            out.dct_compressibility_uv = dct.compressibility_uv;
            out.patch_fraction = dct.patch_fraction;
            out.aq_map_mean = dct.aq_map_mean;
            out.aq_map_std = dct.aq_map_std;
            out.aq_map_p50 = dct.aq_map_p50;
            out.aq_map_p75 = dct.aq_map_p75;
            out.aq_map_p90 = dct.aq_map_p90;
            out.aq_map_p95 = dct.aq_map_p95;
            out.aq_map_p99 = dct.aq_map_p99;
            out.noise_floor_y = dct.noise_floor_y;
            out.noise_floor_uv = dct.noise_floor_uv;
            out.noise_floor_y_p25 = dct.noise_floor_y_p25;
            out.noise_floor_y_p50 = dct.noise_floor_y_p50;
            out.noise_floor_y_p75 = dct.noise_floor_y_p75;
            out.noise_floor_y_p90 = dct.noise_floor_y_p90;
            out.noise_floor_uv_p25 = dct.noise_floor_uv_p25;
            out.noise_floor_uv_p50 = dct.noise_floor_uv_p50;
            out.noise_floor_uv_p75 = dct.noise_floor_uv_p75;
            out.noise_floor_uv_p90 = dct.noise_floor_uv_p90;
            out.gradient_fraction = dct.gradient_fraction;
            out.patch_fraction_fast = dct.patch_fraction_fast;
            out.quant_survival_y = dct.quant_survival_y;
            out.quant_survival_uv = dct.quant_survival_uv;
            out.quant_survival_y_p10 = dct.quant_survival_y_p10;
            out.quant_survival_y_p25 = dct.quant_survival_y_p25;
            out.quant_survival_y_p50 = dct.quant_survival_y_p50;
            out.quant_survival_y_p75 = dct.quant_survival_y_p75;
            out.quant_survival_uv_p10 = dct.quant_survival_uv_p10;
            out.quant_survival_uv_p25 = dct.quant_survival_uv_p25;
            out.quant_survival_uv_p50 = dct.quant_survival_uv_p50;
            out.quant_survival_uv_p75 = dct.quant_survival_uv_p75;
        }
        #[cfg(not(feature = "experimental"))]
        {
            let _ = (
                dct.compressibility_y,
                dct.compressibility_uv,
                dct.patch_fraction,
                dct.aq_map_mean,
                dct.aq_map_std,
                dct.noise_floor_y,
                dct.noise_floor_uv,
                dct.gradient_fraction,
                dct.patch_fraction_fast,
                dct.quant_survival_y,
                dct.quant_survival_uv,
            );
        }
    }
}

/// Aggregated DCT stats from one tier-3 sweep. Fused into a single
/// function so we don't pay for the DCT twice.
struct Tier3DctStats {
    /// `Σ AC[zz≥16] / max(1, Σ AC[zz∈1..16])` (luma).
    high_freq_ratio: f32,
    /// Mean per-block libwebp α (luma): `256 * last_non_zero / max_count`
    /// of the 64-bin |AC|/16 histogram. See `block_alpha`.
    compressibility_y: f32,
    /// Mean per-block libwebp α taken over `max(α_cb, α_cr)` per block —
    /// surfaces colour-detail content that luma α misses.
    compressibility_uv: f32,
    /// Fraction of sampled blocks whose 32-bit DCT signature matches at
    /// least one other sampled block. Real perceptual-hash patch
    /// detection — repeats indicate UI / icon / screen content.
    patch_fraction: f32,
    /// Mean of `log10(1 + Σ AC²)` over sampled luma blocks. Stand-in
    /// for the per-block AQ scale factor — orchestrators that want
    /// "image-average detail level" read this; analyzer-driven AQ
    /// scheduling reads `aq_map_std` for "how spread is the detail".
    aq_map_mean: f32,
    /// Standard deviation of the same per-block log-AC-energy. High
    /// std ⇒ heterogeneous content (mix of flat and busy regions) ⇒
    /// AQ pays off; low std ⇒ uniform busyness, AQ is a no-op.
    aq_map_std: f32,
    /// Robust luma noise floor estimate, normalized to `[0, 1]`. 10th
    /// percentile of per-block √(low-AC-energy / 15) over sampled
    /// luma 8×8 blocks, then divided by 32 for normalization. Flat
    /// blocks have minimal real AC content, so their residual low-AC
    /// is dominated by sensor / quantization noise. The 10th-
    /// percentile selector picks the flattest 10 % of blocks across
    /// the image, robust to busy-image content.
    noise_floor_y: f32,
    /// Same for chroma — `max(p10_cb, p10_cr)` of per-block √(low-
    /// AC-energy / 15) on the existing Cb / Cr DCT coefficients,
    /// normalized to `[0, 1]` by dividing by 32.
    noise_floor_uv: f32,
    /// Distributional companions to `aq_map_mean` / `aq_map_std`.
    /// Same per-block `block_acs` buffer, sorted, read at the
    /// listed quantile in log10 space. Set via the `experimental`
    /// feature gate; zero when disabled.
    aq_map_p50: f32,
    aq_map_p75: f32,
    aq_map_p90: f32,
    aq_map_p95: f32,
    aq_map_p99: f32,
    /// Distributional companions to `noise_floor_y` / `noise_floor_uv`.
    /// Same per-block `block_low_*` buffers, sorted, read at the
    /// listed quantile, then put through the same
    /// `sqrt(arr/15) / 32`-clamped scaling. UV variants emit
    /// `max(cb_pX, cr_pX)`. Zero when experimental disabled.
    noise_floor_y_p25: f32,
    noise_floor_y_p50: f32,
    noise_floor_y_p75: f32,
    noise_floor_y_p90: f32,
    noise_floor_uv_p25: f32,
    noise_floor_uv_p50: f32,
    noise_floor_uv_p75: f32,
    noise_floor_uv_p90: f32,
    /// Fraction `[0, 1]` of sampled luma 8×8 blocks where the
    /// low-zigzag (indices 1–15, the 15 AC positions matched by
    /// the same `zz < 16` predicate as `high_freq_energy_ratio`)
    /// energy is ≥ 90 % of the total AC energy. **Smooth-content /
    /// gradient signal** — drives JXL
    /// `with_force_strategy` (DCT16 / DCT32 selection — larger
    /// transforms pay off when most energy is in the lowest
    /// frequencies) and zenrav1e's deblock strength. Distinct from
    /// `high_freq_energy_ratio` (which is a global mean across all
    /// blocks): this is the per-block-thresholded fraction, robust
    /// to a few high-detail blocks dragging the mean.
    gradient_fraction: f32,
    /// **Experimental.** dHash-based patch_fraction (~10× cheaper than
    /// the DCT signature). 0.99 correlated with `patch_fraction`; AUC
    /// 0.852 (vs DCT's 0.880); peak F1 0.779 (DCT 0.763) on the
    /// 219-image labeled corpus.
    patch_fraction_fast: f32,
    /// **Experimental.** Fraction of luma AC coefficients surviving
    /// jpegli-default quantization at d=2.0 (q=75 area).
    quant_survival_y: f32,
    /// **Experimental.** Same for chroma — max of Cb / Cr per block.
    quant_survival_uv: f32,
    /// Per-block QuantSurvival percentiles. p10 = worst-block survival
    /// (drives trellis ROI); p75 = best-block survival (caps possible
    /// compression). Buffered at sample time, sorted once at end.
    /// Zero in non-experimental builds.
    quant_survival_y_p10: f32,
    quant_survival_y_p25: f32,
    quant_survival_y_p50: f32,
    quant_survival_y_p75: f32,
    quant_survival_uv_p10: f32,
    quant_survival_uv_p25: f32,
    quant_survival_uv_p50: f32,
    quant_survival_uv_p75: f32,
}

/// libwebp `GetAlpha`-style score on a single 8×8 DCT block. Higher
/// = harder to compress (more spread AC, fewer near-zero coefficients).
///
/// **Range:** the formula `256 * last_non_zero / max_count` returns
/// values in `[0, 256 × 63 / 2] = [0, 8064]`, not `[0, 255]` as
/// earlier docs claimed. `last_non_zero` ∈ `[0, 63]` is the highest
/// histogram bin with at least one coefficient; `max_count` ≥ 2 by
/// the guard below. On real corpora `compressibility_y` lands in
/// `[0, ~30]` for photos and `compressibility_uv` even lower —
/// nowhere near the theoretical max — but downstream calibration
/// must NOT clamp or normalise against 255.
///
/// Build a 64-bin histogram of `|AC[k]| / bin_div` (clipped to 63),
/// find `max_count` and `last_non_zero` index, return
/// `256 * last_non_zero / max_count`. Matches libwebp's `analysis_enc.c`
/// convention with `MAX_ALPHA=7`, `ALPHA_SCALE=256`,
/// `MAX_COEFF_THRESH=64`.
///
/// `bin_div` controls histogram granularity. Use `BIN_DIV_LUMA = 16`
/// for luma (the libwebp convention) and `BIN_DIV_CHROMA = 8` for
/// chroma. Chroma DCT coefficient magnitudes run ~half luma's; the
/// finer chroma bin spreads the histogram into the same dynamic
/// range as luma so chroma α isn't suppressed near zero.
#[inline(always)]
fn block_alpha(coeffs: &[[f32; 8]; 8], bin_div: f32) -> u32 {
    let mut histo = [0u32; 64];
    let inv = 1.0 / bin_div;
    for k in 1..64 {
        let u = k % 8;
        let v = k / 8;
        let mag = (coeffs[v][u].abs() * inv) as i32;
        let bin = mag.clamp(0, 63) as usize;
        histo[bin] += 1;
    }
    let mut max_count: u32 = 0;
    let mut last_non_zero: u32 = 1;
    for (k, &c) in histo.iter().enumerate() {
        if c > 0 {
            if c > max_count {
                max_count = c;
            }
            last_non_zero = k as u32;
        }
    }
    if max_count > 1 {
        (256 * last_non_zero) / max_count
    } else {
        0
    }
}

/// Histogram bin divisor for luma DCT α — libwebp's conventional value.
const BIN_DIV_LUMA: f32 = 16.0;
/// Histogram bin divisor for chroma DCT α. Chroma coefficient
/// magnitudes run ~half luma's, so a finer bin (÷ 8 vs ÷ 16) spreads
/// the chroma histogram into the same dynamic range as luma. Empirical
/// CID22 / CLIC / gb82-sc validation shows this lifts chroma α from
/// p50 ≈ 1-2 (≪ luma p50 ≈ 16) into a similarly-discriminating range.
const BIN_DIV_CHROMA: f32 = 8.0;

/// 32-bit perceptual signature of an 8×8 DCT block. Used for
/// patch-fraction matching.
///
/// **Scale-invariant**: the per-coefficient threshold is 25% of the
/// block's own peak |AC|, not a fixed absolute value. Two blocks with
/// the same content at different resolutions (or with different
/// global brightness) produce the same signature because the DCT
/// coefficients scale uniformly. Pre-fix the threshold was a fixed
/// 32.0 in raw f32 magnitude, which made the signature depend on
/// image resolution (DCT amplitude shifts with downsampling).
///
/// Per coefficient (first 16 zigzag-positioned AC): 2 bits =
/// `(sign_bit << 1) | (|c| > 0.25·max_ac)`. 16 × 2 = 32 bits.
#[inline(always)]
fn block_signature_dct(coeffs: &[[f32; 8]; 8]) -> u32 {
    // Per-block peak |AC| over the first 16 zigzag positions.
    let mut peak: f32 = 0.0;
    for (k, &zig) in RASTER_TO_ZIGZAG.iter().enumerate().skip(1) {
        if zig >= 17 {
            continue;
        }
        let u = k % 8;
        let v = k / 8;
        let m = coeffs[v][u].abs();
        if m > peak {
            peak = m;
        }
    }
    // Flat block (peak ≈ 0): all bits → signature 0. Flat blocks ALL
    // match each other — that's correct, they are mutual patches.
    let threshold = peak * 0.25;
    let mut sig: u32 = 0;
    let mut bit_pos: u32 = 0;
    for (k, &zz) in RASTER_TO_ZIGZAG.iter().enumerate().skip(1) {
        if zz >= 17 {
            continue;
        }
        let u = k % 8;
        let v = k / 8;
        let c = coeffs[v][u];
        if c.abs() > threshold {
            sig |= 1 << bit_pos;
        }
        bit_pos += 1;
        if c < 0.0 {
            sig |= 1 << bit_pos;
        }
        bit_pos += 1;
    }
    sig
}

/// Truncated-DCT fingerprint variant: same 32-bit signature as
/// [`block_signature_dct`], but operates only on the first 16 zigzag
/// AC coefficients — the same set the full-DCT version reads. The
/// caller's responsibility is to compute *only* those 16 coefficients
/// (saving ~75% of multiply-adds per block); this function expects the
/// 16 inputs in zigzag order in `zz_acs`.
///
/// Output signature is bit-identical to `block_signature_dct` when fed
/// the same coefficient values, so AUC parity vs the full-DCT version
/// is exact. The win is purely in the per-block compute cost upstream.
#[inline(always)]
#[cfg(feature = "experimental")]
fn block_signature_truncated_dct(zz_acs: &[f32; 16]) -> u32 {
    let mut peak: f32 = 0.0;
    for &c in zz_acs.iter() {
        let m = c.abs();
        if m > peak {
            peak = m;
        }
    }
    let threshold = peak * 0.25;
    let mut sig: u32 = 0;
    let mut bit_pos: u32 = 0;
    for &c in zz_acs.iter() {
        if c.abs() > threshold {
            sig |= 1 << bit_pos;
        }
        bit_pos += 1;
        if c < 0.0 {
            sig |= 1 << bit_pos;
        }
        bit_pos += 1;
    }
    sig
}

/// dHash fingerprint on raw 8×8 luma. Bit `[i*8 + j] = pixels[i][j+1]
/// > pixels[i][j]` for `j ∈ 0..7`, `i ∈ 0..8` — 56 bits. Top 8 bits
/// encode column-direction differences `pixels[i+1][j] > pixels[i][j]`
/// for `i ∈ 0..7`, `j ∈ 0..1` — packs to 64 bits, take low 32.
///
/// Pure pixel comparisons, ~10× cheaper than DCT. Brightness-invariant
/// (DC offset cancels in the difference). Captures gradient direction
/// and edge layout; less discriminative on smooth photo content but
/// works well on the structural patterns that dominate UI / text /
/// chart screen content.
#[cfg(feature = "experimental")]
fn block_signature_dhash(pixels: &[[f32; 8]; 8]) -> u32 {
    incant!(block_signature_dhash_simd(pixels))
}

/// SIMD dHash kernel. Per-row 7-comparison fingerprinting, dispatched
/// to v4 / v3 / NEON / WASM128 / scalar via `#[magetypes]`. The
/// per-tier `target_feature` gate the macro emits lets LLVM autovec
/// the row-level `cmp_gt → packed_bits` pattern down to one
/// `vcmpps` + `vmovmskps` per row on AVX2 / AVX-512, vs the scalar
/// version which compiles to a 7-iteration unrolled scalar loop with
/// 7 branches per row. Measured speedup vs `target_feature(scalar)`
/// path: ~3× on Zen 4 / Skylake-X.
///
/// Bit layout matches the original scalar implementation byte-for-
/// byte: `bit[r*7 + c] = pixels[r][c+1] > pixels[r][c]` for r in 0..8,
/// c in 0..7 (56 bits), then `bit[56 + r] = pixels[r+1][0] >
/// pixels[r][0]` for r in 0..7 (7 bits, padding to 63 — bit 63 is
/// always 0). Folded to u32 by XORing high and low halves of the 64
/// bits.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
#[cfg(feature = "experimental")]
fn block_signature_dhash_simd(_token: Token, pixels: &[[f32; 8]; 8]) -> u32 {
    // Horizontal diffs: 8 rows × 7 diffs = 56 bits. The fixed-array
    // shape and small rep count let LLVM, under the per-arch
    // target_feature region the macro emits, vectorise this with
    // movemask-style packing on x86 / NEON / WASM SIMD128.
    let mut sig: u64 = 0;
    for (r, row) in pixels.iter().enumerate() {
        let row_arr: &[f32; 8] = row;
        for c in 0..7 {
            if row_arr[c + 1] > row_arr[c] {
                sig |= 1u64 << (r * 7 + c);
            }
        }
    }
    // Vertical diffs: column 0, rows 0..7 → bits 56..62 (bit 63 unused).
    for r in 0..7 {
        if pixels[r + 1][0] > pixels[r][0] {
            sig |= 1u64 << (56 + r);
        }
    }
    // Fold 64-bit fingerprint to u32 by XORing high and low halves.
    // XOR preserves Hamming-distance approximate equality (collisions
    // increase only on ~rotated content where the halves correlate).
    ((sig & 0xFFFF_FFFF) ^ (sig >> 32)) as u32
}

/// jpegli-default Y quant table at distance 1.0 (~q90) — the canonical
/// reference for "what a high-quality JPEG quantizer would do." Values
/// are approximate jpegli's `kBaseQuantMatrixYCbCr` Y plane scaled at
/// `d=1.0`. We use this for [`quant_survival_y`] / [`quant_survival_uv`]
/// — a per-block estimate of "how many AC coefficients survive
/// quantization," which approximates the actual JPEG file-size cost
/// of the block. High-survival blocks → busy / detailed content;
/// low-survival blocks → flat / compressible.
///
/// At `d=2.0` (~q75 area) every entry is doubled. At `d=4.0` (~q50)
/// quadrupled. The current implementation uses `d=2.0` as a typical
/// production-quality reference point.
#[cfg(feature = "experimental")]
const JPEGLI_QUANT_Y_D2: [f32; 64] = [
    16.0, 22.0, 26.0, 28.0, 32.0, 38.0, 42.0, 50.0, 22.0, 24.0, 28.0, 30.0, 36.0, 42.0, 46.0, 54.0,
    26.0, 28.0, 32.0, 38.0, 42.0, 50.0, 58.0, 66.0, 28.0, 30.0, 38.0, 46.0, 54.0, 62.0, 70.0, 78.0,
    32.0, 36.0, 42.0, 54.0, 66.0, 78.0, 88.0, 96.0, 38.0, 42.0, 50.0, 62.0, 78.0, 92.0, 102.0,
    110.0, 42.0, 46.0, 58.0, 70.0, 88.0, 102.0, 116.0, 124.0, 50.0, 54.0, 66.0, 78.0, 96.0, 110.0,
    124.0, 132.0,
];

/// jpegli-default Cb/Cr quant table at d=2.0. Chroma quantization is
/// flatter than luma — saturates near 64 across the whole table at
/// q=75 — capturing the human-vision low-chroma sensitivity.
#[cfg(feature = "experimental")]
const JPEGLI_QUANT_C_D2: [f32; 64] = [
    18.0, 22.0, 30.0, 56.0, 64.0, 64.0, 64.0, 64.0, 22.0, 26.0, 32.0, 60.0, 64.0, 64.0, 64.0, 64.0,
    30.0, 32.0, 50.0, 64.0, 64.0, 64.0, 64.0, 64.0, 56.0, 60.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0,
    64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0,
    64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0,
];

/// Estimate the fraction of AC coefficients that survive JPEG-style
/// quantization with the given table. "Survives" = `round(c / Q) != 0`
/// after applying a typical zero-bias of 0.5 (jpegli default neutral).
///
/// Range: `[0, 1]`. 0 = all coefficients quantize to zero (perfectly
/// flat block); 1 = every AC survives (extremely high-detail block).
/// Photographic content typically lands at 0.10–0.25; UI / text edges
/// at 0.30–0.50; flat regions ~0.0.
///
/// Cost: 63 divides + 63 round-and-compare per block. Fast — no
/// transcendentals or branches in the hot loop.
#[cfg(feature = "experimental")]
fn quant_survival(coeffs: &[[f32; 8]; 8], qtable: &[f32; 64]) -> f32 {
    incant!(quant_survival_simd(coeffs, qtable))
}

/// SIMD quant-survival kernel. f32x8 lane-wise `|c| ≥ q · 0.5` over
/// 8 chunks of 8 coefficients. Equivalent to the scalar
/// `(c/q).round() != 0` test (rounding to nearest with ties-toward-
/// zero matches the half-bias inflection point at `|c| = q/2`).
///
/// The 64-element block reads as 8 × `f32x8` lane-loads from the
/// row-major `coeffs[v][u]` and `qtable` arrays. Lane 0 of chunk 0
/// (the DC term at `coeffs[0][0]`) is masked out: DC always carries
/// the block mean, so a "survival fraction" that includes it would
/// be biased toward 1.0 for any non-degenerate block.
///
/// Output is a per-lane mask of `0.0` or `1.0`; we accumulate into a
/// running f32x8 sum and reduce at the end. Bit-exact with the
/// scalar reference for all coefficient values whose magnitudes
/// aren't exactly at the half-quant boundary (where round-to-even
/// vs round-to-nearest-half-up could disagree — but those values
/// are a measure-zero set on real DCT outputs).
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
#[cfg(feature = "experimental")]
fn quant_survival_simd(token: Token, coeffs: &[[f32; 8]; 8], qtable: &[f32; 64]) -> f32 {
    let half = f32x8::splat(token, 0.5);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);
    let mut sum_v = f32x8::zero(token);

    // 8 row-chunks × 8 lanes = 64 coefficients, fully covering the
    // block. coeffs is `[[f32; 8]; 8]` — each row is exactly one
    // `f32x8::load` source.
    for (v, row) in coeffs.iter().enumerate() {
        let q_chunk: &[f32; 8] = qtable[v * 8..v * 8 + 8].try_into().unwrap();
        let c_v = f32x8::load(token, row);
        let q_v = f32x8::load(token, q_chunk);
        // |c| >= q * 0.5 → survives quantization at zero-bias 0.5.
        // (Numerically equivalent to round(c/q) != 0 within f32
        // precision except on the half-quant boundary, which doesn't
        // occur in real DCT outputs.)
        let abs_c = c_v.abs();
        let half_q = q_v * half;
        let mask = abs_c.simd_ge(half_q);
        // Convert the lane-mask (all-1s / all-0s) to {1.0, 0.0} via
        // `blend(mask, 1.0, 0.0)` and accumulate.
        let lane_one = f32x8::blend(mask, one, zero);
        sum_v += lane_one;
    }

    let lanes = sum_v.to_array();
    // Subtract the DC lane (index 0 of chunk 0): we double-counted
    // it in the loop above. Constant correction so the loop stays
    // branch-free.
    let dc_lane: &[f32; 8] = (&coeffs[0][..]).try_into().unwrap();
    let dc_q: &[f32; 8] = qtable[..8].try_into().unwrap();
    let dc_survives = if dc_lane[0].abs() >= dc_q[0] * 0.5 {
        1.0
    } else {
        0.0
    };
    let total: f32 = lanes.iter().sum::<f32>() - dc_survives;
    total / 63.0
}

struct LumaHistStats {
    /// Shannon entropy of the 32-bin luma histogram, bits `[0, 5]`.
    entropy: f32,
    /// Soft line-art / engineering-drawing score `[0, 1]`. Combines:
    /// (a) Otsu bimodality (η = σ²B / σ²total — high when histogram
    ///     splits cleanly into two well-separated peaks).
    /// (b) Top-2-bin coverage (fraction of pixels in the two most
    ///     populated bins — typically ≥ 0.7 for line art on a flat
    ///     background, vs ≤ 0.3 for photos).
    /// (c) Low entropy (≤ 2 bits ≈ near-binary distribution).
    /// `line_art_score = min(η, top2_coverage, 1 - entropy/2.5)`
    /// clamped to `[0, 1]`. The min combinator makes the score
    /// conservative — it goes high only when ALL three conditions
    /// agree, sparing photos with one accidentally-bimodal channel
    /// from being misclassified.
    line_art_score: f32,
}

/// Walk the luma histogram once to produce both entropy and the
/// line-art-score signal — same single pass, two reductions.
/// Same 32-bin BT.601 histogram, every-4th-pixel sampling pattern as
/// the original `luma_histogram_entropy`, with cross-row carry.
fn luma_histogram_stats(stream: &mut RowStream<'_>) -> LumaHistStats {
    let width = stream.width() as usize;
    let height = stream.height() as usize;
    if width == 0 || height == 0 {
        return LumaHistStats {
            entropy: 0.0,
            line_art_score: 0.0,
        };
    }
    // Per-primaries fixed-point luma weights — keeps wide-gamut
    // bytes interpreted with the right matrix (BT.2020 for Rec.2020,
    // etc.). sRGB/BT.709 keeps the BT.601 baseline (66/129/25) so
    // the trained histogram thresholds still apply.
    let w = crate::luma::LumaWeights::for_primaries(stream.primaries());
    let qr = w.qr as u32;
    let qg = w.qg as u32;
    let qb = w.qb as u32;
    let mut bins = [0u32; 32];
    let mut n = 0u32;
    let mut carry: usize = 0;
    for yy in 0..height {
        let row = stream.borrow_row(yy as u32);
        let start = (4 - carry) % 4;
        let mut x = start;
        while x < width {
            let off = x * 3;
            let p = &row[off..off + 3];
            let y = ((qr * p[0] as u32 + qg * p[1] as u32 + qb * p[2] as u32 + 128) >> 8) as u8;
            bins[(y >> 3) as usize] += 1;
            n += 1;
            x += 4;
        }
        carry = (carry + width) % 4;
    }
    if n == 0 {
        return LumaHistStats {
            entropy: 0.0,
            line_art_score: 0.0,
        };
    }
    let n_f = n as f32;

    // Entropy.
    let mut entropy = 0.0f32;
    for &c in &bins {
        if c > 0 {
            let p = c as f32 / n_f;
            entropy -= p * p.log2();
        }
    }

    // Otsu bimodality. For each split index k, compute between-class
    // variance σ²B(k) = ω0·ω1·(μ0 - μ1)². Pick max. Also compute
    // total variance σ²T to normalize.
    let mut total_sum: f64 = 0.0;
    for (i, &c) in bins.iter().enumerate() {
        total_sum += (i as f64) * (c as f64);
    }
    let mu_total = total_sum / n_f as f64;
    let mut total_var: f64 = 0.0;
    for (i, &c) in bins.iter().enumerate() {
        let d = i as f64 - mu_total;
        total_var += (c as f64) * d * d;
    }
    total_var /= n_f as f64;

    let mut max_between_var: f64 = 0.0;
    let mut sum0: f64 = 0.0;
    let mut count0: f64 = 0.0;
    for (k, &b) in bins.iter().take(31).enumerate() {
        sum0 += (k as f64) * (b as f64);
        count0 += b as f64;
        let count1 = n_f as f64 - count0;
        if count0 < 1.0 || count1 < 1.0 {
            continue;
        }
        let mu0 = sum0 / count0;
        let mu1 = (total_sum - sum0) / count1;
        let omega0 = count0 / n_f as f64;
        let omega1 = count1 / n_f as f64;
        let between_var = omega0 * omega1 * (mu0 - mu1) * (mu0 - mu1);
        if between_var > max_between_var {
            max_between_var = between_var;
        }
    }
    let bimodality = if total_var > 1e-6 {
        (max_between_var / total_var).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    // Top-2-bin coverage.
    let mut top1 = 0u32;
    let mut top2 = 0u32;
    for &c in &bins {
        if c > top1 {
            top2 = top1;
            top1 = c;
        } else if c > top2 {
            top2 = c;
        }
    }
    let top2_coverage = (top1 + top2) as f32 / n_f;

    // Low-entropy gate: 1.0 at entropy=0, 0.0 at entropy ≥ 2.5 bits.
    let low_entropy = ((2.5 - entropy) / 2.5).clamp(0.0, 1.0);

    let line_art_score = bimodality.min(top2_coverage).min(low_entropy);

    LumaHistStats {
        entropy,
        line_art_score,
    }
}

/// Raster → zigzag position lookup for 8×8 DCT coefficients.
///
/// `RASTER_TO_ZIGZAG[u + v*8]` returns the zigzag index of the
/// coefficient at column `u`, row `v`. Standard JPEG ITU-T T.81
/// zigzag order: (0,0) → 0 (DC), (1,0) → 1, (0,1) → 2, (0,2) → 3,
/// (1,1) → 4, (2,0) → 5, etc.
const RASTER_TO_ZIGZAG: [u8; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

/// 8×8 DCT-II coefficient matrix, scaled by `0.5 * c[k]` where
/// `c[0] = 1/sqrt(2)` and `c[k] = 1` for `k > 0`. Pre-baked so the
/// inner loops are just `mul_add` chains — no per-iteration `.cos()`
/// (which was the previous version's hottest hot path).
///
/// `DCT_COEF[k][n] = 0.5 * c[k] * cos((2n+1) * k * π / 16)`.
///
/// Forward DCT: `Y[k] = Σ_n DCT_COEF[k][n] * X[n]`.
///
/// Layout note: [`DCT_COEF`] is row-major (output `k` outer, input `n`
/// inner). The transposed copy [`DCT_COEF_T`] is needed by the
/// vectorized DCT below — its row `n` is `[DCT_COEF[0][n], …,
/// DCT_COEF[7][n]]`, i.e. column `n` of D. With that layout each f32x8
/// vector loaded from `DCT_COEF_T[n]` directly broadcasts the column-of-D
/// the row pass needs (lanes-as-output-k).
const DCT_COEF: [[f32; 8]; 8] = [
    // k=0: 0.5 * (1/√2) * cos(0) = 1/(2√2) = 0.353553...
    [
        0.353_553_4,
        0.353_553_4,
        0.353_553_4,
        0.353_553_4,
        0.353_553_4,
        0.353_553_4,
        0.353_553_4,
        0.353_553_4,
    ],
    // k=1..7: 0.5 * cos((2n+1)*k*π/16)
    [
        0.490_392_64,
        0.415_734_8,
        0.277_785_12,
        0.097_545_16,
        -0.097_545_16,
        -0.277_785_12,
        -0.415_734_8,
        -0.490_392_64,
    ],
    [
        0.461_939_77,
        0.191_341_72,
        -0.191_341_72,
        -0.461_939_77,
        -0.461_939_77,
        -0.191_341_72,
        0.191_341_72,
        0.461_939_77,
    ],
    [
        0.415_734_8,
        -0.097_545_16,
        -0.490_392_64,
        -0.277_785_12,
        0.277_785_12,
        0.490_392_64,
        0.097_545_16,
        -0.415_734_8,
    ],
    [
        0.353_553_4,
        -0.353_553_4,
        -0.353_553_4,
        0.353_553_4,
        0.353_553_4,
        -0.353_553_4,
        -0.353_553_4,
        0.353_553_4,
    ],
    [
        0.277_785_12,
        -0.490_392_64,
        0.097_545_16,
        0.415_734_8,
        -0.415_734_8,
        -0.097_545_16,
        0.490_392_64,
        -0.277_785_12,
    ],
    [
        0.191_341_72,
        -0.461_939_77,
        0.461_939_77,
        -0.191_341_72,
        -0.191_341_72,
        0.461_939_77,
        -0.461_939_77,
        0.191_341_72,
    ],
    [
        0.097_545_16,
        -0.277_785_12,
        0.415_734_8,
        -0.490_392_64,
        0.490_392_64,
        -0.415_734_8,
        0.277_785_12,
        -0.097_545_16,
    ],
];

/// Transposed DCT-II coefficient matrix: `DCT_COEF_T[n][k] = DCT_COEF[k][n]`.
///
/// Used by the magetypes f32x8 dct2d kernel — each row of `DCT_COEF_T`
/// is one column of `D`, packed lanes-as-output-k. Loaded once at the
/// top of the SIMD function and reused across both 1D DCT passes.
const DCT_COEF_T: [[f32; 8]; 8] = {
    let mut t = [[0.0f32; 8]; 8];
    let mut n = 0;
    while n < 8 {
        let mut k = 0;
        while k < 8 {
            t[n][k] = DCT_COEF[k][n];
            k += 1;
        }
        n += 1;
    }
    t
};

/// Ratio of high-frequency to low-frequency AC DCT energy on sampled
/// 8×8 luma blocks. `Σ AC[zz ≥ 16] / max(1, Σ AC[zz ∈ 1..=15])` where
/// `zz` is the JPEG ITU-T T.81 zigzag index — the same scan order
/// JPEG itself uses to drop high frequencies first. The split is at
/// the predicate `zz < 16` (low side ⇒ zigzag indices 1–15 = **15
/// AC positions** after DC; zigzag 16 and beyond go to the high
/// side). The 15 low positions cover most of the upper-left 4×4
/// triangle (`u + v ≤ 3`), keeping the split symmetric in
/// horizontal/vertical detail, unlike the older raster-order split
/// which biased toward vertical content.
///
/// Naive separable 1D DCT — exactness isn't required for a feature,
/// only stable scale and ordering. A faster approximate DCT could
/// substitute as long as the ratio is preserved within an f32 ULP or
/// two; bit-exact coefficient values are not relied on.
///
/// Pulls 8 rows at a time (one block-row's worth) and samples the
/// `bx` columns selected by `block_idx % stride`. Keeps memory at
/// 8 × width × 3 bytes regardless of image size.
fn dct_stats(stream: &mut RowStream<'_>, max_blocks: usize) -> Tier3DctStats {
    let width = stream.width() as usize;
    let height = stream.height() as usize;
    if width < 8 || height < 8 {
        return Tier3DctStats {
            high_freq_ratio: 0.0,
            compressibility_y: 0.0,
            compressibility_uv: 0.0,
            patch_fraction: 0.0,
            aq_map_mean: 0.0,
            aq_map_std: 0.0,
            aq_map_p50: 0.0,
            aq_map_p75: 0.0,
            aq_map_p90: 0.0,
            aq_map_p95: 0.0,
            aq_map_p99: 0.0,
            noise_floor_y: 0.0,
            noise_floor_uv: 0.0,
            noise_floor_y_p25: 0.0,
            noise_floor_y_p50: 0.0,
            noise_floor_y_p75: 0.0,
            noise_floor_y_p90: 0.0,
            noise_floor_uv_p25: 0.0,
            noise_floor_uv_p50: 0.0,
            noise_floor_uv_p75: 0.0,
            noise_floor_uv_p90: 0.0,
            gradient_fraction: 0.0,
            patch_fraction_fast: 0.0,
            quant_survival_y: 0.0,
            quant_survival_uv: 0.0,
            quant_survival_y_p10: 0.0,
            quant_survival_y_p25: 0.0,
            quant_survival_y_p50: 0.0,
            quant_survival_y_p75: 0.0,
            quant_survival_uv_p10: 0.0,
            quant_survival_uv_p25: 0.0,
            quant_survival_uv_p50: 0.0,
            quant_survival_uv_p75: 0.0,
        };
    }
    // Per-primaries luma weights — used in the per-block fixed-point
    // YCbCr build below. Wide-gamut u8 sources go through the DCT
    // pipeline with the right matrix for their primaries; sRGB /
    // BT.709 keeps the BT.601 baseline (66/129/25) so trained
    // thresholds still apply.
    let lw = crate::luma::LumaWeights::for_primaries(stream.primaries());
    let qr = lw.qr;
    let qg = lw.qg;
    let qb = lw.qb;
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let total_blocks = blocks_x * blocks_y;
    if total_blocks == 0 {
        return Tier3DctStats {
            high_freq_ratio: 0.0,
            compressibility_y: 0.0,
            compressibility_uv: 0.0,
            patch_fraction: 0.0,
            aq_map_mean: 0.0,
            aq_map_std: 0.0,
            aq_map_p50: 0.0,
            aq_map_p75: 0.0,
            aq_map_p90: 0.0,
            aq_map_p95: 0.0,
            aq_map_p99: 0.0,
            noise_floor_y: 0.0,
            noise_floor_uv: 0.0,
            noise_floor_y_p25: 0.0,
            noise_floor_y_p50: 0.0,
            noise_floor_y_p75: 0.0,
            noise_floor_y_p90: 0.0,
            noise_floor_uv_p25: 0.0,
            noise_floor_uv_p50: 0.0,
            noise_floor_uv_p75: 0.0,
            noise_floor_uv_p90: 0.0,
            gradient_fraction: 0.0,
            patch_fraction_fast: 0.0,
            quant_survival_y: 0.0,
            quant_survival_uv: 0.0,
            quant_survival_y_p10: 0.0,
            quant_survival_y_p25: 0.0,
            quant_survival_y_p50: 0.0,
            quant_survival_y_p75: 0.0,
            quant_survival_uv_p10: 0.0,
            quant_survival_uv_p25: 0.0,
            quant_survival_uv_p50: 0.0,
            quant_survival_uv_p75: 0.0,
        };
    }
    let stride = (total_blocks / max_blocks).max(1);

    let mut low_energy = 0.0f64;
    let mut high_energy = 0.0f64;
    let mut blocks_sampled: u32 = 0;
    let mut alpha_y_sum: u64 = 0;
    let mut alpha_uv_sum: u64 = 0;
    // Per-block log-AC-energy for AQ-map mean/std. The log
    // compresses the heavy right tail (texture blocks vs flat
    // blocks differ by 4-5 orders of magnitude in raw energy) and
    // gives a mean / std that lives on a useful 0-7 scale.
    //
    // We stage the raw `block_ac` values into a Vec during the main
    // DCT loop and batch the `log10(1 + ac)` reduction afterwards
    // via magetypes `log2_lowp` — vectorising 8 lanes per call vs
    // 8 sequential scalar `f64::ln()` invocations (~50 cycles each).
    let mut block_acs: Vec<f32> = Vec::with_capacity(max_blocks.min(4096));
    // Per-block low-AC-energy, retained for noise-floor estimation.
    // 10th percentile across blocks ≈ noise floor (flattest blocks'
    // residual AC). 4-byte storage × max_blocks = ~4 KB at default.
    let mut block_low_y: Vec<f32> = Vec::with_capacity(max_blocks.min(4096));
    let mut block_low_cb: Vec<f32> = Vec::with_capacity(max_blocks.min(4096));
    let mut block_low_cr: Vec<f32> = Vec::with_capacity(max_blocks.min(4096));
    // GradientFraction: count blocks where ≥ 90 % of AC energy lives
    // in the low-zigzag positions (smooth-content blocks where larger
    // DCT transforms pay off).
    let mut gradient_blocks: u32 = 0;
    // Bounded to `max_blocks` (default 256) — small enough that a
    // sort-and-sweep is faster than a hash map on this CPU.
    let mut signatures: Vec<u32> = Vec::with_capacity(max_blocks.min(2048));
    // Experimental fingerprint variants: separate signature buffers so
    // the same sort-and-sweep collision-counting can be reused for each.
    #[cfg(feature = "experimental")]
    #[cfg(feature = "experimental")]
    let mut signatures_fast: Vec<u32> = Vec::with_capacity(max_blocks.min(2048));
    // Per-block quant-survival accumulators (mean across blocks) +
    // per-block buffers for percentile reduction (issue #42 →
    // distributional features analysis 2026-04-30 → Batch 3).
    #[cfg(feature = "experimental")]
    let mut quant_y_sum: f64 = 0.0;
    #[cfg(feature = "experimental")]
    let mut quant_uv_sum: f64 = 0.0;
    #[cfg(feature = "experimental")]
    let mut quant_y_blocks: Vec<f32> = Vec::with_capacity(max_blocks.min(4096));
    #[cfg(feature = "experimental")]
    let mut quant_uv_blocks: Vec<f32> = Vec::with_capacity(max_blocks.min(4096));
    let row_bytes = width * 3;
    let mut block_buf = vec![0u8; 8 * row_bytes]; // 8 rows of one block-row
    let mut block_idx = 0usize;

    for by in 0..blocks_y {
        // Determine whether any sampled block lives in this block-row.
        let row_start = by * blocks_x;
        let row_end = row_start + blocks_x;
        let any_sampled = (row_start..row_end).any(|k| k % stride == 0);
        if !any_sampled {
            block_idx += blocks_x;
            continue;
        }

        // Pull 8 contiguous rows for the block-row.
        for i in 0..8 {
            stream.fetch_into(
                (by * 8 + i) as u32,
                &mut block_buf[i * row_bytes..(i + 1) * row_bytes],
            );
        }

        for bx in 0..blocks_x {
            if !block_idx.is_multiple_of(stride) {
                block_idx += 1;
                continue;
            }
            block_idx += 1;

            // Extract Y, Cb, Cr per pixel into three 8×8 blocks.
            // BT.601 integer-quantized luma matches the existing
            // convention; chroma uses the same scaled-integer form
            // (Cb = (B − Y) ≈ scaled b−g pattern; Cr = (R − Y)) so the
            // DCT inputs land in the same dynamic range as luma.
            let mut blk_y = [[0.0f32; 8]; 8];
            let mut blk_cb = [[0.0f32; 8]; 8];
            let mut blk_cr = [[0.0f32; 8]; 8];
            for y in 0..8 {
                let row = &block_buf[y * row_bytes..(y + 1) * row_bytes];
                for x in 0..8 {
                    let off = (bx * 8 + x) * 3;
                    let p = &row[off..off + 3];
                    let r = p[0] as i32;
                    let g = p[1] as i32;
                    let b = p[2] as i32;
                    let l_i = (qr * r + qg * g + qb * b + 128) >> 8;
                    // Cb / Cr keep their BT.601-derived integer
                    // matrix here. The per-primaries adjustment
                    // shifts luma; chroma differences (B−Y / R−Y)
                    // would also drift, but the chroma-DCT
                    // compressibility / noise-floor signals are
                    // ratio-based and small per-primaries drift on
                    // the chroma matrix doesn't materially move
                    // them. Revisit if a corpus eval shows wide-
                    // gamut chroma stats reading off vs sRGB.
                    let cb_i = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    let cr_i = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                    blk_y[y][x] = l_i as f32 - 128.0;
                    blk_cb[y][x] = cb_i as f32 - 128.0;
                    blk_cr[y][x] = cr_i as f32 - 128.0;
                }
            }

            // 2D DCT for all three planes in one magetypes f32x8 batch.
            // 128 fmas per plane vs ~1920 scalar ops in the old
            // separable-scalar version.
            let mut coeffs_y = [[0.0f32; 8]; 8];
            let mut coeffs_cb = [[0.0f32; 8]; 8];
            let mut coeffs_cr = [[0.0f32; 8]; 8];
            dct2d_8_three_planes(
                &blk_y,
                &blk_cb,
                &blk_cr,
                &mut coeffs_y,
                &mut coeffs_cb,
                &mut coeffs_cr,
            );

            blocks_sampled += 1;
            // High/low AC energy on luma (existing high_freq_energy_ratio).
            // Per-block: total AC for AQ map + low-AC for noise floor.
            let mut block_ac: f64 = 0.0;
            let mut block_low_y_ac: f64 = 0.0;
            for (k, &zz) in RASTER_TO_ZIGZAG.iter().enumerate().skip(1) {
                let u = k % 8;
                let v = k / 8;
                let e = (coeffs_y[v][u] * coeffs_y[v][u]) as f64;
                block_ac += e;
                if zz < 16 {
                    low_energy += e;
                    block_low_y_ac += e;
                } else {
                    high_energy += e;
                }
            }
            block_acs.push(block_ac as f32);
            block_low_y.push(block_low_y_ac as f32);

            // Per-block gradient flag. Threshold 0.9 picks blocks
            // where almost all the AC energy is in the lowest 15
            // zigzag positions — those are smooth gradients / soft
            // textures where larger-than-8×8 DCT transforms (JXL
            // DCT16 / DCT32) pay off. Skip the test on near-flat
            // blocks (block_ac < 16 ≈ ~1 unit of average AC) so a
            // dead-flat region doesn't drag the fraction up.
            if block_ac > 16.0 && block_low_y_ac >= 0.9 * block_ac {
                gradient_blocks += 1;
            }

            // Chroma low-AC for chroma noise floor. Reuses the
            // already-computed coeffs_cb / coeffs_cr.
            let mut low_cb: f64 = 0.0;
            let mut low_cr: f64 = 0.0;
            for (k, &zz) in RASTER_TO_ZIGZAG.iter().enumerate().skip(1) {
                if zz < 16 {
                    let u = k % 8;
                    let v = k / 8;
                    low_cb += (coeffs_cb[v][u] * coeffs_cb[v][u]) as f64;
                    low_cr += (coeffs_cr[v][u] * coeffs_cr[v][u]) as f64;
                }
            }
            block_low_cb.push(low_cb as f32);
            block_low_cr.push(low_cr as f32);
            // libwebp α per block (luma): /16 bin divisor.
            alpha_y_sum += block_alpha(&coeffs_y, BIN_DIV_LUMA) as u64;
            // libwebp α per block (chroma): /8 bin divisor (finer bins
            // because chroma coefficient magnitudes are smaller; keeps
            // the chroma α in the same dynamic range as luma α). Max
            // across Cb/Cr per block to surface whichever channel
            // carries more detail.
            let a_cb = block_alpha(&coeffs_cb, BIN_DIV_CHROMA);
            let a_cr = block_alpha(&coeffs_cr, BIN_DIV_CHROMA);
            alpha_uv_sum += a_cb.max(a_cr) as u64;
            // Patch-detection signature on luma DCT coefficients.
            signatures.push(block_signature_dct(&coeffs_y));
            #[cfg(feature = "experimental")]
            {
                // WHT and dHash signatures operate on raw 8×8 luma
                // (pre-DCT). The DCT was already paid for by the
                // energy / entropy / α features above; these are pure
                // marginal cost.
                signatures_fast.push(block_signature_dhash(&blk_y));
                // quant-survival accumulators — d=2.0 (~q75) reference
                // quantization. Y on the luma table, UV on the chroma
                // table (max of Cb / Cr per block to capture whichever
                // channel carries more detail).
                let q_y = quant_survival(&coeffs_y, &JPEGLI_QUANT_Y_D2);
                quant_y_sum += q_y as f64;
                quant_y_blocks.push(q_y);
                let q_cb = quant_survival(&coeffs_cb, &JPEGLI_QUANT_C_D2);
                let q_cr = quant_survival(&coeffs_cr, &JPEGLI_QUANT_C_D2);
                let q_uv = q_cb.max(q_cr);
                quant_uv_sum += q_uv as f64;
                quant_uv_blocks.push(q_uv);
            }
        }
    }

    let high_freq_ratio = if low_energy < 1e-6 {
        0.0
    } else {
        (high_energy / low_energy) as f32
    };

    let compressibility_y = if blocks_sampled > 0 {
        (alpha_y_sum as f64 / blocks_sampled as f64) as f32
    } else {
        0.0
    };
    let compressibility_uv = if blocks_sampled > 0 {
        (alpha_uv_sum as f64 / blocks_sampled as f64) as f32
    } else {
        0.0
    };

    // Patch fraction: count blocks whose 32-bit DCT signature appears
    // at least twice in the sample. Sort-and-sweep over the small
    // sampled set is O(N log N) — at N ≤ 256 this is ~10 µs, faster
    // than a hash map.
    fn collision_fraction(sigs: &mut [u32], n: u32) -> f32 {
        if n <= 1 || sigs.len() <= 1 {
            return 0.0;
        }
        sigs.sort_unstable();
        let mut matched: u32 = 0;
        let mut i = 0;
        while i < sigs.len() {
            let mut j = i + 1;
            while j < sigs.len() && sigs[j] == sigs[i] {
                j += 1;
            }
            let run = (j - i) as u32;
            if run > 1 {
                matched += run;
            }
            i = j;
        }
        matched as f32 / n as f32
    }
    let patch_fraction = collision_fraction(&mut signatures, blocks_sampled);
    #[cfg(feature = "experimental")]
    #[cfg(feature = "experimental")]
    let patch_fraction_fast = collision_fraction(&mut signatures_fast, blocks_sampled);
    #[cfg(feature = "experimental")]
    let quant_survival_y = if blocks_sampled > 0 {
        (quant_y_sum / blocks_sampled as f64) as f32
    } else {
        0.0
    };
    #[cfg(feature = "experimental")]
    let quant_survival_uv = if blocks_sampled > 0 {
        (quant_uv_sum / blocks_sampled as f64) as f32
    } else {
        0.0
    };

    // Per-block quant-survival percentiles. Sort once per channel,
    // read at p10/p25/p50/p75. p10 = "worst-block survival" — directly
    // proxies trellis ROI; high-survival p75 ⇒ uniformly compressible.
    #[cfg(feature = "experimental")]
    let quant_survival_y_p10: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_y_p25: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_y_p50: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_y_p75: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_uv_p10: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_uv_p25: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_uv_p50: f32;
    #[cfg(feature = "experimental")]
    let quant_survival_uv_p75: f32;
    #[cfg(feature = "experimental")]
    {
        let sort_f32 = |arr: &mut [f32]| {
            arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        };
        let qs_at = |arr: &[f32], q: f32| -> f32 {
            if arr.is_empty() {
                return 0.0;
            }
            let idx = ((arr.len() as f32 * q) as usize).min(arr.len() - 1);
            arr[idx]
        };
        sort_f32(&mut quant_y_blocks);
        sort_f32(&mut quant_uv_blocks);
        quant_survival_y_p10 = qs_at(&quant_y_blocks, 0.10);
        quant_survival_y_p25 = qs_at(&quant_y_blocks, 0.25);
        quant_survival_y_p50 = qs_at(&quant_y_blocks, 0.50);
        quant_survival_y_p75 = qs_at(&quant_y_blocks, 0.75);
        quant_survival_uv_p10 = qs_at(&quant_uv_blocks, 0.10);
        quant_survival_uv_p25 = qs_at(&quant_uv_blocks, 0.25);
        quant_survival_uv_p50 = qs_at(&quant_uv_blocks, 0.50);
        quant_survival_uv_p75 = qs_at(&quant_uv_blocks, 0.75);
    }

    // Batched log10(1 + ac) over `block_acs` via magetypes `ln_lowp`,
    // dispatched to v4 / v3 / NEON / WASM128 / scalar. Replaces the
    // per-block scalar `f64::ln()` (was ~50 cycles each × N blocks)
    // with one `ln_lowp` call per 8 blocks at low precision (well
    // above the noise floor for an aq_map_std on the 0–7 log scale).
    let (aq_log_sum, aq_log_sq_sum) = log10_sum_and_sq_sum_dispatch(&block_acs);
    let (aq_map_mean, aq_map_std) = if blocks_sampled > 0 {
        let n = blocks_sampled as f64;
        let mean = aq_log_sum / n;
        let var = (aq_log_sq_sum / n - mean * mean).max(0.0);
        (mean as f32, var.sqrt() as f32)
    } else {
        (0.0, 0.0)
    };
    // AqMap percentiles — sort `block_acs` once and read at p50/75/90/95/99
    // in log10 space (matches `aq_map_mean`'s scale). The sort is amortized
    // by the per-block work that produced the buffer.
    let mut aq_sorted = block_acs.clone();
    aq_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let log10_at = |arr: &[f32], q: f32| -> f32 {
        if arr.is_empty() {
            return 0.0;
        }
        let idx = ((arr.len() as f32 * q) as usize).min(arr.len() - 1);
        let v = arr[idx];
        if v <= 0.0 {
            0.0
        } else {
            // log10(1 + v) matches the convention used for aq_map_mean
            // (`log10_sum_and_sq_sum_dispatch` operates on
            // `log10(1 + block_ac)`).
            (1.0 + v as f64).log10() as f32
        }
    };
    let aq_map_p50 = log10_at(&aq_sorted, 0.50);
    let aq_map_p75 = log10_at(&aq_sorted, 0.75);
    let aq_map_p90 = log10_at(&aq_sorted, 0.90);
    let aq_map_p95 = log10_at(&aq_sorted, 0.95);
    let aq_map_p99 = log10_at(&aq_sorted, 0.99);

    // Noise-floor estimate via 10th-percentile per-block low-AC-energy.
    // Flat blocks' low-AC is residual noise; the 10th percentile
    // selects the flattest 10 % across the image. Convert from
    // sum-of-squared-coefficients to a per-coefficient σ via
    // sqrt(low_ac / 15) (15 low-zigzag coefficients per block, since
    // index 0 = DC). Normalize by 32 to land on `[0, 1]`-ish — scale
    // chosen so a "noisy JPEG" reads ~0.5 and pristine reads <0.1.
    // `quantile_idx`: for an `n`-element sorted array, the linearly
    // interpolated index for fraction `q` ∈ [0, 1]. We round down
    // (truncating) for cheap selection — at our sample sizes
    // (≥ 256 blocks the picker cares about) the difference vs proper
    // linear interp is below the noise floor.
    let quantile_at = |arr: &[f32], q: f32| -> f32 {
        if arr.is_empty() {
            return 0.0;
        }
        let idx = ((arr.len() as f32 * q) as usize).min(arr.len() - 1);
        arr[idx]
    };
    let sort_in_place = |arr: &mut [f32]| {
        arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    };
    sort_in_place(&mut block_low_y);
    sort_in_place(&mut block_low_cb);
    sort_in_place(&mut block_low_cr);
    let nf_scale = |raw: f32| -> f32 { ((raw / 15.0).sqrt() / 32.0).clamp(0.0, 1.0) };
    let noise_floor_y = nf_scale(quantile_at(&block_low_y, 0.10));
    let noise_floor_y_p25 = nf_scale(quantile_at(&block_low_y, 0.25));
    let noise_floor_y_p50 = nf_scale(quantile_at(&block_low_y, 0.50));
    let noise_floor_y_p75 = nf_scale(quantile_at(&block_low_y, 0.75));
    let noise_floor_y_p90 = nf_scale(quantile_at(&block_low_y, 0.90));
    let nf_uv = |q: f32| -> f32 {
        let cb = nf_scale(quantile_at(&block_low_cb, q));
        let cr = nf_scale(quantile_at(&block_low_cr, q));
        cb.max(cr)
    };
    let noise_floor_uv = nf_uv(0.10);
    let noise_floor_uv_p25 = nf_uv(0.25);
    let noise_floor_uv_p50 = nf_uv(0.50);
    let noise_floor_uv_p75 = nf_uv(0.75);
    let noise_floor_uv_p90 = nf_uv(0.90);

    let gradient_fraction = if blocks_sampled > 0 {
        gradient_blocks as f32 / blocks_sampled as f32
    } else {
        0.0
    };

    #[cfg(not(feature = "experimental"))]
    let (patch_fraction_fast, quant_survival_y, quant_survival_uv) = (0.0, 0.0, 0.0);
    #[cfg(not(feature = "experimental"))]
    let (
        quant_survival_y_p10,
        quant_survival_y_p25,
        quant_survival_y_p50,
        quant_survival_y_p75,
        quant_survival_uv_p10,
        quant_survival_uv_p25,
        quant_survival_uv_p50,
        quant_survival_uv_p75,
    ) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    Tier3DctStats {
        high_freq_ratio,
        compressibility_y,
        compressibility_uv,
        patch_fraction,
        aq_map_mean,
        aq_map_std,
        aq_map_p50,
        aq_map_p75,
        aq_map_p90,
        aq_map_p95,
        aq_map_p99,
        noise_floor_y,
        noise_floor_uv,
        noise_floor_y_p25,
        noise_floor_y_p50,
        noise_floor_y_p75,
        noise_floor_y_p90,
        noise_floor_uv_p25,
        noise_floor_uv_p50,
        noise_floor_uv_p75,
        noise_floor_uv_p90,
        gradient_fraction,
        patch_fraction_fast,
        quant_survival_y,
        quant_survival_uv,
        quant_survival_y_p10,
        quant_survival_y_p25,
        quant_survival_y_p50,
        quant_survival_y_p75,
        quant_survival_uv_p10,
        quant_survival_uv_p25,
        quant_survival_uv_p50,
        quant_survival_uv_p75,
    }
}

/// Runtime-dispatched f32x8 2D DCT for the three YCbCr planes of one
/// 8×8 block. Returns coefficients in raster order: `out[v][u]` is the
/// coefficient at horizontal frequency `u`, vertical frequency `v`.
///
/// All three planes go through one [`incant!`] call so dispatch cost is
/// amortized 3× and LLVM keeps the eight `DCT_COEF_T` lane vectors hot
/// in YMM registers across planes.
#[inline]
fn dct2d_8_three_planes(
    blk_y: &[[f32; 8]; 8],
    blk_cb: &[[f32; 8]; 8],
    blk_cr: &[[f32; 8]; 8],
    coeffs_y: &mut [[f32; 8]; 8],
    coeffs_cb: &mut [[f32; 8]; 8],
    coeffs_cr: &mut [[f32; 8]; 8],
) {
    incant!(dct2d_8_three_planes_simd(
        blk_y, blk_cb, blk_cr, coeffs_y, coeffs_cb, coeffs_cr
    ));
}

/// f32x8 SIMD 2D DCT-II for three 8×8 blocks (Y, Cb, Cr).
///
/// Each row of the output is one f32x8 with lane k = `Y[v][k]`. The
/// row pass computes `Y[v] = Σ_n splat(X[v][n]) * D_col_n_vec` (8 fmas
/// per row); the column pass computes `Z[v] = Σ_w splat(D[v][w]) *
/// Y[w]` (8 fmas per row). Total 128 fmas per plane vs ~1920 scalar ops
/// in the old `dct2d_8` (separable scalar + transpose), an 8-15× LLVM
/// op-count reduction. With three planes batched we saturate the FMA
/// units across each block's worth of work before the per-block scalar
/// (alpha / signature / energy split) takes over.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn dct2d_8_three_planes_simd(
    token: Token,
    blk_y: &[[f32; 8]; 8],
    blk_cb: &[[f32; 8]; 8],
    blk_cr: &[[f32; 8]; 8],
    coeffs_y: &mut [[f32; 8]; 8],
    coeffs_cb: &mut [[f32; 8]; 8],
    coeffs_cr: &mut [[f32; 8]; 8],
) {
    // Load the eight column-of-D vectors once. d_col[n] holds
    // [D[0][n], …, D[7][n]] across lanes — reused for both passes
    // (the column pass uses the same matrix D, just along the v axis).
    let d_col = [
        f32x8::load(token, &DCT_COEF_T[0]),
        f32x8::load(token, &DCT_COEF_T[1]),
        f32x8::load(token, &DCT_COEF_T[2]),
        f32x8::load(token, &DCT_COEF_T[3]),
        f32x8::load(token, &DCT_COEF_T[4]),
        f32x8::load(token, &DCT_COEF_T[5]),
        f32x8::load(token, &DCT_COEF_T[6]),
        f32x8::load(token, &DCT_COEF_T[7]),
    ];

    // The body is identical across the three planes; macro keeps the
    // f32x8 type scope inside the magetypes-generated copy without
    // splitting into a helper (helpers don't share f32x8 across the
    // per-backend functions the macro emits).
    macro_rules! dct2d_one_plane {
        ($blk:ident, $out:ident) => {{
            // Row pass: Y_row[v] (lane k = Y[v][k]) =
            // Σ_n splat(X[v][n]) * d_col[n]. 8 fmas per row.
            let mut y_row: [f32x8; 8] = [
                f32x8::zero(token),
                f32x8::zero(token),
                f32x8::zero(token),
                f32x8::zero(token),
                f32x8::zero(token),
                f32x8::zero(token),
                f32x8::zero(token),
                f32x8::zero(token),
            ];
            let mut v = 0;
            while v < 8 {
                let r = &$blk[v];
                let mut acc = d_col[0] * f32x8::splat(token, r[0]);
                acc = d_col[1].mul_add(f32x8::splat(token, r[1]), acc);
                acc = d_col[2].mul_add(f32x8::splat(token, r[2]), acc);
                acc = d_col[3].mul_add(f32x8::splat(token, r[3]), acc);
                acc = d_col[4].mul_add(f32x8::splat(token, r[4]), acc);
                acc = d_col[5].mul_add(f32x8::splat(token, r[5]), acc);
                acc = d_col[6].mul_add(f32x8::splat(token, r[6]), acc);
                acc = d_col[7].mul_add(f32x8::splat(token, r[7]), acc);
                y_row[v] = acc;
                v += 1;
            }

            // Column pass: Z_row[v] (lane k = Z[v][k]) =
            // Σ_w D[v][w] * Y_row[w] (lane k). Each lane k runs an
            // independent 1D DCT down the column — no transpose, no
            // horizontal reduction. Store straight into the caller's
            // [f32; 8] rows.
            let mut v = 0;
            while v < 8 {
                let d_row = &DCT_COEF[v];
                let mut acc = y_row[0] * f32x8::splat(token, d_row[0]);
                acc = y_row[1].mul_add(f32x8::splat(token, d_row[1]), acc);
                acc = y_row[2].mul_add(f32x8::splat(token, d_row[2]), acc);
                acc = y_row[3].mul_add(f32x8::splat(token, d_row[3]), acc);
                acc = y_row[4].mul_add(f32x8::splat(token, d_row[4]), acc);
                acc = y_row[5].mul_add(f32x8::splat(token, d_row[5]), acc);
                acc = y_row[6].mul_add(f32x8::splat(token, d_row[6]), acc);
                acc = y_row[7].mul_add(f32x8::splat(token, d_row[7]), acc);
                acc.store(&mut $out[v]);
                v += 1;
            }
        }};
    }

    dct2d_one_plane!(blk_y, coeffs_y);
    dct2d_one_plane!(blk_cb, coeffs_cb);
    dct2d_one_plane!(blk_cr, coeffs_cr);
}

/// Populate derived likelihood scores from leaf features computed by
/// other tiers. The const bools `T3` and `PAL` say which tiers ran:
///
/// - `T3` ⇒ `luma_histogram_entropy` is real → can compute
///   [`text_likelihood`] and (with `PAL`) [`natural_likelihood`].
/// - `PAL` ⇒ `distinct_color_bins` is real → can compute
///   [`screen_content_likelihood`] and (with `T3`) [`natural_likelihood`].
///
/// Tier 1 outputs (`cb_sharpness`, `cr_sharpness`, `edge_density`,
/// `flat_color_block_ratio`) are always available since Tier 1 is
/// always-on.
///
/// **Layered defense.** Even if [`crate::analyze_features`]'s
/// dispatch-axis tables drift and pick the wrong axis, this gate
/// refuses to write a likelihood whose dependencies weren't computed.
/// The field stays at `Default` (0.0), `into_results` doesn't emit
/// it (because the caller's [`feature::FeatureSet`] still has to
/// list it), and the caller gets `None`. **Never garbage.**
///
/// In release `if T3 { … }` is straight-line const-eval'd into the
/// caller's monomorphized variant, so the unused branches contribute
/// zero code and zero runtime cost.
///
/// **Calibration (re-normalized 2026-04-28).** Each likelihood is a
/// weighted sum of clamped sub-components, then re-stretched against
/// its empirical saturation point so values cleanly span `[0, 1]` on
/// real content. Without the post-stretch, all three composites capped
/// at ~0.7 on the 219-image labeled corpus because the sub-components
/// don't co-fire to their individual maxima on natural inputs.
///
/// **Pre-stretch saturation points** (raw output max on the labeled corpus):
///
/// - `text_likelihood`: 0.71
/// - `screen_content_likelihood`: 0.70 (current formula)
/// - `natural_likelihood`: 0.69
///
/// After stretch, real content reliably reaches the upper end of `[0, 1]`.
/// AUC is preserved (rank order unchanged); operating thresholds shift
/// proportionally — see updated thresholds below.
///
/// **Recommended consumer thresholds** (best F1 on the labeled corpus,
/// post-stretch):
///
/// | Composite | Threshold | F1 | P | R | Notes |
/// |---|---:|---:|---:|---:|---|
/// | `text_likelihood >= 0.35` | 0.35 | 0.585 | 0.50 | 0.71 | AUC = 0.713; same AUC pre/post-stretch (rank-preserving). |
/// | `screen_content_likelihood >= 0.80` | 0.80 | **0.779** | 0.94 | 0.67 | AUC = 0.845 (was 0.831); the formula reshape from `palette_small`-based to `patch_fraction`-based lifted both AUC AND peak F1 (0.59 → 0.78). |
/// | `natural_likelihood >= 0.10` | 0.10 | **0.923** | 0.88 | 0.97 | AUC = 0.814; same AUC pre/post-stretch. |
///
/// Thresholds shifted vs the pre-2026-04-28 calibration: stretching by
/// `MAX` divisors moves every operating point. If you have rules
/// hardcoded against the old 0.06-0.60 range, multiply by the inverse
/// stretch factor (e.g. `old_threshold / 0.71` for `text_likelihood`).
///
/// See `docs/calibration-corpus-2026-04-27.md` for the full empirical
/// distribution and AUC table.
///
/// [`text_likelihood`]: crate::feature::AnalysisFeature::TextLikelihood
/// [`natural_likelihood`]: crate::feature::AnalysisFeature::NaturalLikelihood
/// [`screen_content_likelihood`]: crate::feature::AnalysisFeature::ScreenContentLikelihood
#[cfg(feature = "composites")]
pub fn compute_derived_likelihoods<const T3: bool, const PAL: bool>(out: &mut RawAnalysis) {
    let chroma_sh = out.cb_sharpness + out.cr_sharpness;
    let chroma_lo = (0.005 - chroma_sh).clamp(0.0, 0.005) / 0.005;
    let edge_hi = (out.edge_density / 0.25).min(1.0);
    let flat_high = (out.flat_color_block_ratio / 0.5).min(1.0);

    // Empirically-derived re-stretch divisors (see module-level docstring).
    // `clamp(0, 1.0)` after the stretch since some inputs (mostly synthetic
    // pathological cases) can briefly exceed the corpus max.
    const TEXT_MAX: f32 = 0.71;
    const SCREEN_MAX: f32 = 0.70;
    const NATURAL_MAX: f32 = 0.69;

    if T3 {
        let entropy_low = (4.0 - out.luma_histogram_entropy).clamp(0.0, 4.0) / 4.0;
        let raw = (entropy_low * 0.4 + edge_hi * 0.3 + chroma_lo * 0.3).clamp(0.0, 1.0);
        out.text_likelihood = (raw / TEXT_MAX).clamp(0.0, 1.0);
    }
    if PAL {
        // Post-2026-04-28 reformulation: the previous formula combined
        // `flat_high * 0.6 + palette_small * 0.3 + chroma_lo * 0.1`. The
        // `palette_small` weight was dragging the AUC down: real screen
        // content (charts, anti-aliased UIs) routinely has > 4000 distinct
        // colour bins, so `palette_small` collapsed to 0 on most positive
        // examples. Replacing it with `patch_fraction` (when available)
        // lifts AUC from 0.83 to 0.85 on the 219-image labeled corpus.
        // (`patch_fraction` alone hits 0.88 — the residual 0.03 the
        // composite gives up vs the raw feature is the price of combining
        // inputs at all.)
        //
        // `patch_fraction` lives behind `experimental`; when that feature
        // is off the field doesn't exist on `RawAnalysis`, so fall back
        // to the previous formula. Both branches stretch by `SCREEN_MAX`.
        #[cfg(feature = "experimental")]
        let raw = (out.patch_fraction * 0.6 + flat_high * 0.4).clamp(0.0, 1.0);
        #[cfg(not(feature = "experimental"))]
        let raw = {
            let palette_small = if out.distinct_color_bins == 0 {
                0.0
            } else {
                (1.0 - (out.distinct_color_bins as f32 / 4000.0).min(1.0)).clamp(0.0, 1.0)
            };
            (flat_high * 0.6 + palette_small * 0.3 + chroma_lo * 0.1).clamp(0.0, 1.0)
        };
        out.screen_content_likelihood = (raw / SCREEN_MAX).clamp(0.0, 1.0);
    }
    if T3 && PAL {
        let entropy_hi = (out.luma_histogram_entropy - 3.5).clamp(0.0, 1.5) / 1.5;
        let palette_large = if out.distinct_color_bins < 2000 {
            0.0
        } else {
            ((out.distinct_color_bins as f32 - 2000.0) / 8000.0).clamp(0.0, 1.0)
        };
        let chroma_moderate = (chroma_sh / 0.012).min(1.0);
        let not_flat = (1.0 - (out.flat_color_block_ratio / 0.3).min(1.0)).clamp(0.0, 1.0);
        let raw =
            (entropy_hi * 0.3 + palette_large * 0.25 + chroma_moderate * 0.2 + not_flat * 0.25)
                .clamp(0.0, 1.0);
        out.natural_likelihood = (raw / NATURAL_MAX).clamp(0.0, 1.0);
    }
}

/// `composites`-disabled stub — keeps the call site in `lib.rs`
/// unconditional. With `composites` off, no likelihood fields exist
/// on `RawAnalysis`, so the body collapses to a no-op.
#[cfg(not(feature = "composites"))]
pub fn compute_derived_likelihoods<const T3: bool, const PAL: bool>(_out: &mut RawAnalysis) {}

/// Dispatcher for the batched `log10(1 + ac)` reduction over the
/// `block_acs` accumulator collected during the DCT-stats pass.
/// Returns `(Σ log10(1+ac), Σ log10(1+ac)²)` as f64 — same shape
/// the per-block scalar version produced.
fn log10_sum_and_sq_sum_dispatch(block_acs: &[f32]) -> (f64, f64) {
    incant!(log10_sum_and_sq_sum_simd(block_acs))
}

/// SIMD batched `log10(1 + ac)` and its square-sum, vectorised
/// 8 lanes at a time via magetypes `log10_lowp`. The low-precision
/// variant is ~12-bit accurate — far above the noise floor on the
/// `aq_map_std` 0–7 log scale that consumes this output. Switching
/// from per-block scalar `f64::ln() / LN_10` (~50 cycles each) to
/// `log10_lowp` over an 8-wide vector removes a per-block transcendental
/// call from the hot Tier 3 DCT loop.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn log10_sum_and_sq_sum_simd(token: Token, block_acs: &[f32]) -> (f64, f64) {
    let one_v = f32x8::splat(token, 1.0);
    let mut sum_v = f32x8::zero(token);
    let mut sq_sum_v = f32x8::zero(token);
    let mut sum_f64: f64 = 0.0;
    let mut sq_sum_f64: f64 = 0.0;
    // FLUSH cadence — same `f32`-mantissa argument as the row-stats
    // pass: log10 outputs land in `[0, ~7]` so partial sums of 32
    // 8-lane chunks reach ~1.8 K, well below the 16 M f32 mantissa
    // boundary.
    const FLUSH: usize = 32;
    let mut iters_since_flush = 0usize;
    let chunks = block_acs.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let ac_v = f32x8::load(token, arr);
        let log_v = (ac_v + one_v).log10_lowp();
        sum_v += log_v;
        sq_sum_v = log_v.mul_add(log_v, sq_sum_v);
        iters_since_flush += 1;
        if iters_since_flush >= FLUSH {
            sum_f64 += sum_v.reduce_add() as f64;
            sq_sum_f64 += sq_sum_v.reduce_add() as f64;
            sum_v = f32x8::zero(token);
            sq_sum_v = f32x8::zero(token);
            iters_since_flush = 0;
        }
    }
    sum_f64 += sum_v.reduce_add() as f64;
    sq_sum_f64 += sq_sum_v.reduce_add() as f64;
    // Scalar tail (≤ 7 leftover blocks) — use the same `log10_lowp`
    // semantics the SIMD pass produced so the lowp/scalar boundary
    // doesn't drift the aggregate. Native `f32::log10` is fine here
    // because the tail count is tiny (typical ≤ 7 calls per analysis).
    for &ac in remainder {
        let l = (ac + 1.0).log10();
        sum_f64 += l as f64;
        sq_sum_f64 += (l as f64) * (l as f64);
    }
    (sum_f64, sq_sum_f64)
}
