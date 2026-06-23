//! Source-direct HDR / bit-depth tier.
//!
//! All other tiers read RGB8 (via `RowStream::Native` or row-by-row
//! conversion through `zenpixels-convert::RowConverter`). That works
//! for SDR features whose threshold contract is calibrated on display-
//! space bytes, but it **destroys** the HDR signal on PQ / HLG / linear-
//! light f32 sources because `RowConverter` does not tonemap — it
//! clips into the [0, 1] sRGB-display range. The bytes that reach the
//! standard tiers from a 4000-nit PQ HDR image are bit-for-bit
//! identical to the bytes from a 100-nit-clipped SDR image.
//!
//! This tier reads the source samples directly via `PixelSlice::row`
//! (same pattern as the alpha pass), decodes them through the
//! descriptor's `TransferFunction` to linear nits, and measures HDR-
//! aware statistics:
//!
//! - **Peak luminance** (max nits across sampled pixels)
//! - **P99 luminance** (robust against single hot pixels — single-bin
//!   histogram resolves to ~3% of full range)
//! - **HDR headroom in stops** (`log2(peak / sdr_reference)`)
//! - **HDR pixel fraction** (sampled pixels above the SDR threshold)
//! - **Wide-gamut peak** (per-channel max in linear light — feeds
//!   future gamut-clipping decisions)
//! - **Effective bit depth** (sample-distribution probe: how many
//!   bottom bits carry information vs being u8-promotion zeros)
//!
//! ## Reference white convention
//!
//! "Nits" here is a **convention**, not a measurement — we don't have
//! pixel-level mastering metadata. The convention tracks the transfer
//! function's de-facto signal-to-nits mapping:
//!
//! | Transfer | Linear 1.0 maps to | Rationale |
//! |---|---|---|
//! | `Srgb` / `Bt709` / `Gamma22` | 80 nits | sRGB display reference (IEC 61966-2-1). |
//! | `Linear` | 80 nits | Treat scene-referred f32 as display-referred without metadata. |
//! | `Pq` | 10 000 nits | SMPTE ST 2084 absolute reference. |
//! | `Hlg` | 1 000 nits | Nominal peak for typical HLG broadcasts. |
//! | `Unknown` | 80 nits | Conservative SDR fallback. |
//!
//! These numbers are stable across the 0.1.x line; if a downstream
//! consumer needs a different mapping, file an issue — at that point
//! we'd thread metadata through the descriptor rather than changing
//! the convention silently.
//!
//! ## Why a separate tier
//!
//! Other tiers iterate over RGB8 rows from `RowStream`. This one
//! iterates over the **source bytes**, decodes per channel-type
//! (u8 / u16 / f32), and applies the transfer EOTF in f32. There's
//! no shared inner loop with Tier 1/2/3 — the data layout, sample
//! type, and math are all different. Sharing a pass would either
//! require RGB8 conversion (defeats the purpose) or specializing the
//! existing tiers across 4 channel types × 5 transfers (massive
//! monomorphization explosion). One small dedicated tier is the
//! clean answer.

use archmage::{incant, magetypes};
use linear_srgb::tf;
use zenpixels::{ChannelType, ColorPrimaries, PixelSlice, TransferFunction};

/// Reference peak luminance per transfer function. See module docs for
/// the convention.
const PEAK_SRGB_NITS: f32 = 80.0;
const PEAK_PQ_NITS: f32 = 10_000.0;
const PEAK_HLG_NITS: f32 = 1_000.0;

/// Threshold above which a sample counts as "HDR" — a rendered
/// approximation of the SDR boundary on a typical sRGB display.
const SDR_THRESHOLD_NITS: f32 = 100.0;

/// Output of the depth tier. Default (all-zero) is what gets written
/// when the tier is gated off, so every field's "absent" semantics
/// must read as "no signal" (zero peak / zero headroom / zero
/// fraction).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DepthStats {
    /// Peak luminance in nits over sampled pixels.
    pub peak_nits: f32,
    /// 99th-percentile luminance in nits.
    pub p99_nits: f32,
    /// HDR headroom in stops: `log2(peak_nits / 80)`. SDR ⇒ ~0.
    pub headroom_stops: f32,
    /// Fraction of sampled pixels above [`SDR_THRESHOLD_NITS`].
    pub hdr_pixel_fraction: f32,
    /// Largest single-channel linear value across sampled pixels.
    /// Useful as a "this image will clip on sRGB primaries" signal —
    /// values >> 1.0 indicate wide-gamut content.
    pub wide_gamut_peak: f32,
    /// Fraction of sampled pixels with at least one channel > 1.0
    /// in linear light (i.e., outside the source-primaries' gamut
    /// that maps to sRGB-display [0, 255]).
    pub wide_gamut_fraction: f32,
    /// Effective bit depth: smallest power-of-2 quantization grid
    /// the sampled values populate. Common values: 8, 10, 12, 14, 16.
    /// For u8 sources always 8. For u16 sources, probes the
    /// low-byte distribution to detect u8-promoted content.
    pub effective_bit_depth: u32,
    /// `true` iff [`peak_nits`] >> the SDR threshold AND the source
    /// transfer function is genuinely HDR-capable (PQ / HLG /
    /// Linear-with-out-of-range values). Catches the hard case the
    /// standard tiers miss: a PQ-encoded image whose tonemapped
    /// rendition looks like SDR but whose source carries far more
    /// dynamic range.
    pub hdr_present: bool,
    /// Fraction of sampled pixels whose linear-RGB, when projected
    /// from the source primaries into BT.709 / sRGB primaries, has
    /// every channel within `[-ε, 1 + ε]`. `1.0` ⇒ image is safely
    /// downcastable to sRGB primaries (codecs save bits by encoding
    /// nclx-Bt709 / sRGB ICC instead of the wider declared gamut).
    /// For sRGB-declared sources this is trivially `1.0`; the signal
    /// is load-bearing for P3 / Rec.2020 / AdobeRGB sources whose
    /// pixels happen to all live in the sRGB sub-gamut.
    pub gamut_coverage_srgb: f32,
    /// Same as [`gamut_coverage_srgb`] for the Display P3 sub-gamut.
    /// Useful when the source is declared Rec.2020 — `1.0` here
    /// means the source is downcastable to P3 (a smaller container
    /// than Rec.2020 but wider than sRGB).
    pub gamut_coverage_p3: f32,
    /// Mean luminance (nits) of the super-white highlight pixels (above
    /// [`SDR_THRESHOLD_NITS`]). `0` when there are no highlights.
    pub highlight_luma_mean: f32,
    /// Std-dev of highlight luminance (nits).
    pub highlight_luma_std: f32,
    /// Mean linear saturation `(max−min)/max` of the highlight pixels.
    pub highlight_chroma_mean: f32,
    /// Std-dev of highlight saturation.
    pub highlight_chroma_std: f32,
    /// Strong-edge density within the highlight region — adjacent-sample
    /// relative-luma jumps `> 10 %` (H + V), per highlight pixel.
    pub highlight_edge_count: f32,
    /// Fraction `[0, 1]` of highlight edges that are horizontal, `H / (H + V)`.
    pub highlight_orientation_ratio: f32,
}

/// Histogram-bin count for percentile estimation. 256 bins on a
/// linear nits scale gives ~3% relative resolution at the SDR
/// threshold and ~40-nit absolute resolution at 10 000 nits — fine
/// for codec dispatch decisions.
const HIST_BINS: usize = 256;

/// Logarithmic histogram bin: maps `nits ∈ [0, ~10 000]` to
/// `[0, 256)` via `log2(1 + nits)` so SDR detail isn't squashed by
/// the HDR tail.
#[inline]
fn nits_to_bin(nits: f32) -> usize {
    if !nits.is_finite() || nits <= 0.0 {
        return 0;
    }
    // log2(1 + 10000) ≈ 13.29; scale to [0, HIST_BINS).
    let v = (1.0_f32 + nits).log2() / 14.0;
    let i = (v * HIST_BINS as f32) as usize;
    i.min(HIST_BINS - 1)
}

#[inline]
fn bin_to_nits(bin: usize) -> f32 {
    // Inverse of nits_to_bin: nits = exp2((bin / HIST_BINS) * 14) - 1.
    (((bin as f32 + 0.5) / HIST_BINS as f32) * 14.0).exp2() - 1.0
}

/// Apply a transfer function's EOTF (signal → linear) to a sample
/// already normalized to `[0, 1]` (or wider, for HDR-out-of-range
/// f32). Output is normalized linear (1.0 = the transfer's reference
/// peak — see `peak_nits_for`).
#[inline]
fn eotf(tf_kind: TransferFunction, signal: f32) -> f32 {
    match tf_kind {
        TransferFunction::Linear => signal,
        TransferFunction::Srgb | TransferFunction::Unknown => tf::srgb_to_linear(signal),
        TransferFunction::Bt709 => tf::bt709_to_linear(signal),
        TransferFunction::Gamma22 => signal.max(0.0).powf(2.2),
        TransferFunction::Pq => tf::pq_to_linear(signal),
        TransferFunction::Hlg => tf::hlg_to_linear(signal),
        _ => signal, // Defensive: non_exhaustive enum ⇒ any future variant.
    }
}

/// Apply the transfer EOTF to a whole `[f32]` slice **in place** (signal →
/// linear), via linear-srgb's public auto-dispatched SIMD slices. Element-wise, so
/// it matches the scalar [`eotf`] per element — within the SIMD `f32` polynomial's
/// few-ULP of the scalar's `f64`-intermediate (well inside the feature tolerance),
/// and deterministic on FMA arches. `Linear` (and any future variant) is a no-op,
/// matching `eotf`'s `Linear => signal`. The per-pixel HDR loop reaches this only
/// for PQ / HLG / Linear (the SDR fast path returns earlier for the rest), but the
/// full match keeps it correct if that ever changes.
fn eotf_slice(tf_kind: TransferFunction, values: &mut [f32]) {
    use linear_srgb::default as d;
    match tf_kind {
        TransferFunction::Linear => {}
        TransferFunction::Srgb | TransferFunction::Unknown => d::srgb_to_linear_slice(values),
        TransferFunction::Bt709 => d::bt709_to_linear_slice(values),
        TransferFunction::Gamma22 => {
            for v in values.iter_mut() {
                *v = v.max(0.0); // match the scalar `signal.max(0.0).powf(2.2)`
            }
            d::gamma_to_linear_slice(values, 2.2);
        }
        TransferFunction::Pq => d::pq_to_linear_slice(values),
        TransferFunction::Hlg => d::hlg_to_linear_slice(values),
        _ => {}
    }
}

/// For each `nits`, compute the histogram bin *value*
/// `log2(1 + max(nits,0)) · (HIST_BINS/14)` (the caller floors + clamps + scatters)
/// — the SIMD `log2_midp` form of [`nits_to_bin`]'s scalar `log2`. ~11× faster than
/// libm `log2`; `log2_midp`'s ~1e-7 vs libm almost never flips the integer bin
/// (within the feature tolerance). The 0..8 tail uses scalar libm `log2` to match
/// `nits_to_bin` exactly. `nits`/`out` equal length.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn nits_bin_value_slice(token: Token, nits: &[f32], out: &mut [f32]) {
    const SCALE: f32 = HIST_BINS as f32 / 14.0;
    let scale = f32x8::splat(token, SCALE);
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::splat(token, 0.0);
    let n = nits.len();
    let nchunks = n / 8;
    for c in 0..nchunks {
        let off = c * 8;
        let arr: &[f32; 8] = nits[off..off + 8].try_into().unwrap();
        let v = (f32x8::load(token, arr).max(zero) + one).log2_midp() * scale;
        let mut buf = [0.0f32; 8];
        v.store(&mut buf);
        out[off..off + 8].copy_from_slice(&buf);
    }
    for i in nchunks * 8..n {
        out[i] = (1.0 + nits[i].max(0.0)).log2() * SCALE;
    }
}

/// Reference peak in nits for a transfer function — see module docs.
#[inline]
fn peak_nits_for(tf_kind: TransferFunction) -> f32 {
    match tf_kind {
        TransferFunction::Pq => PEAK_PQ_NITS,
        TransferFunction::Hlg => PEAK_HLG_NITS,
        _ => PEAK_SRGB_NITS,
    }
}

/// 3×3 matrices that take linear RGB in the source primaries and
/// project it into linear sRGB-primaries RGB. Pre-computed from the
/// ITU-R / SMPTE primaries chromaticities + D65 whitepoint via
/// `M = M_xyz_to_srgb · M_src_to_xyz`. Source: standard derivation;
/// the same numbers appear in libjxl, ffmpeg, and `colour-science`.
///
/// Values stored row-major: `[r_out, g_out, b_out]` from
/// `(r_lin, g_lin, b_lin)` via `out = M · in`.
const M_DISPLAYP3_TO_SRGB: [[f32; 3]; 3] = [
    [1.224_940_2, -0.224_940_4, 0.000_000_0],
    [-0.042_056_9, 1.042_057_1, 0.000_000_0],
    [-0.019_637_6, -0.078_636_1, 1.098_273_7],
];
const M_BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5, 1.132_899_9, -0.008_349_4],
    [-0.018_150_8, -0.100_578_9, 1.118_729_7],
];
const M_ADOBERGB_TO_SRGB: [[f32; 3]; 3] = [
    [1.398_287_7, -0.398_287_8, 0.000_000_0],
    [0.000_000_0, 1.000_000_0, 0.000_000_0],
    [0.000_000_0, -0.042_969_2, 1.042_969_3],
];
/// Same shape, projecting wider primaries into Display P3 (the
/// "smallest wide gamut" — useful when the source is BT.2020 and we
/// want to know if a P3-down container would suffice).
const M_BT2020_TO_DISPLAYP3: [[f32; 3]; 3] = [
    [1.343_578_8, -0.282_855_8, -0.060_722_6],
    [-0.077_876_4, 1.083_393_2, -0.005_516_5],
    [0.000_307_5, -0.027_209_2, 1.026_901_8],
];

/// Identity 3×3.
const M_IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Multiply 3×3 matrix `m` by 3-vector `(r, g, b)`.
#[inline]
fn mat3_mul(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

/// Pick the matrix that takes source-primaries linear RGB into sRGB
/// (BT.709) primaries linear RGB. Returns `None` if the source is
/// already in sRGB primaries (caller's hot path skips the multiply).
#[inline]
fn primaries_to_srgb_matrix(src: ColorPrimaries) -> Option<&'static [[f32; 3]; 3]> {
    match src {
        ColorPrimaries::Bt709 => None,
        ColorPrimaries::DisplayP3 => Some(&M_DISPLAYP3_TO_SRGB),
        ColorPrimaries::Bt2020 => Some(&M_BT2020_TO_SRGB),
        ColorPrimaries::AdobeRgb => Some(&M_ADOBERGB_TO_SRGB),
        _ => None, // Unknown / future ⇒ no projection (assume already in target gamut).
    }
}

/// Pick the matrix that takes source-primaries linear RGB into
/// Display P3 primaries linear RGB. P3 is the "next-smallest" gamut
/// after sRGB; useful for "is this Rec.2020 source actually just P3
/// content?" downcast detection.
#[inline]
fn primaries_to_displayp3_matrix(src: ColorPrimaries) -> &'static [[f32; 3]; 3] {
    match src {
        ColorPrimaries::DisplayP3 => &M_IDENTITY,
        ColorPrimaries::Bt2020 => &M_BT2020_TO_DISPLAYP3,
        // sRGB / Bt709 / AdobeRGB are subsets of P3 in practice
        // (some AdobeRGB greens push slightly outside P3, but the
        // common case is fully covered). For codec dispatch we
        // treat anything sRGB-or-narrower as P3-coverable.
        ColorPrimaries::Bt709 | ColorPrimaries::AdobeRgb => &M_IDENTITY,
        _ => &M_IDENTITY,
    }
}

/// `true` iff `tf` can carry above-SDR signal levels. Used to gate
/// `hdr_present` so a sRGB image with a single hot pixel from
/// rounding doesn't trip the flag.
#[inline]
fn is_hdr_capable_tf(tf_kind: TransferFunction) -> bool {
    matches!(
        tf_kind,
        TransferFunction::Pq | TransferFunction::Hlg | TransferFunction::Linear
    )
}

/// Decode one source sample to `[0, 1]`-normalized signal (may be
/// wider than 1.0 for f32 HDR linear inputs). `bytes` points at the
/// start of the sample; caller guarantees `bytes.len() >= ch.byte_size()`.
#[inline]
fn read_sample(ch: ChannelType, bytes: &[u8]) -> f32 {
    match ch {
        ChannelType::U8 => bytes[0] as f32 / 255.0,
        ChannelType::U16 => u16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 65535.0,
        ChannelType::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        ChannelType::F16 => {
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]);
            f16_to_f32(raw)
        }
        // ChannelType is non_exhaustive — defensive fallback.
        _ => 0.0,
    }
}

/// Bit-for-bit IEEE 754 binary16 → binary32 conversion. Inlined to
/// keep the hot loop tight; called only on the (rare) F16 path.
#[inline]
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 0x1) as u32;
    let exp = ((half >> 10) & 0x1f) as u32;
    let mant = (half & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal — renormalize.
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf / NaN.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 127 - 15) & 0xff) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Walk source samples directly, accumulating depth statistics. Stride-
/// samples rows so total walked pixels ≈ `pixel_budget`.
pub(crate) fn scan_depth(slice: &PixelSlice<'_>, pixel_budget: usize) -> DepthStats {
    let desc = slice.descriptor();
    let width = slice.width() as usize;
    let height = slice.rows() as usize;
    if width == 0 || height == 0 {
        return DepthStats::default();
    }

    let layout = desc.layout();
    let ch = desc.channel_type();
    let color_channels = desc.color_model().color_channels() as usize;
    if color_channels == 0 {
        return DepthStats::default();
    }
    let total_channels = layout.channels();
    let bpp = total_channels * ch.byte_size();
    let ch_bytes = ch.byte_size();

    let tf_kind = desc.transfer();
    let peak_nits_unit = peak_nits_for(tf_kind);
    let hdr_capable = is_hdr_capable_tf(tf_kind);

    // SDR fast path — the load-bearing performance path. ANY non-HDR transfer
    // (sRGB / Bt709 / Gamma22 / Linear / Unknown) is bounded to [0, 1] linear,
    // so it carries no HDR / wide-gamut signal and its depth profile IS the
    // canonical SDR display reference. Skipping the per-pixel EOTF + gamut +
    // histogram walk here is what keeps the overwhelmingly common SDR case
    // ~free. It applies to u8 / u16 / f32 alike, so peak/p99 stay CONSISTENT
    // across channel types (all report the 80-nit display reference) AND fast —
    // only true HDR (PQ / HLG) walks pixels for the real content luminance.
    //
    // peak/p99 are display-referred for SDR by design: the depth tier answers
    // "what dynamic range does this content need" → SDR / 80 nits; the actual
    // brightness distribution is Tier 1/3's job (Variance / LumaHistogram).
    if !hdr_capable {
        // effective_bit_depth still distinguishes losslessly-promoted u16
        // (low byte == high byte everywhere ⇒ 8) from genuine ≥9-bit, via a
        // CHEAP byte-only probe — no EOTF / gamut / histogram, so it's a small
        // fraction of the walk it replaces.
        let effective_bit_depth = match ch {
            ChannelType::U8 => 8,
            ChannelType::F32 => 32,
            ChannelType::F16 => 16,
            ChannelType::U16 => sdr_u16_effective_depth(slice, bpp, width, height, pixel_budget),
            _ => 8,
        };
        let trivial_srgb_cover = if matches!(desc.primaries, ColorPrimaries::Bt709) {
            1.0
        } else {
            0.0
        };
        let trivial_p3_cover = if matches!(
            desc.primaries,
            ColorPrimaries::Bt709 | ColorPrimaries::DisplayP3
        ) {
            1.0
        } else {
            0.0
        };
        return DepthStats {
            peak_nits: PEAK_SRGB_NITS,
            p99_nits: PEAK_SRGB_NITS,
            headroom_stops: 0.0,
            hdr_pixel_fraction: 0.0,
            wide_gamut_peak: 1.0,
            wide_gamut_fraction: 0.0,
            effective_bit_depth,
            hdr_present: false,
            gamut_coverage_srgb: trivial_srgb_cover,
            gamut_coverage_p3: trivial_p3_cover,
            // SDR content in an HDR-capable container — no super-white highlights.
            ..DepthStats::default()
        };
    }

    // Stride-sample rows, same shape as alpha pass. Within a row we
    // also stride-sample pixels for very wide images so the budget
    // covers a representative slice of the height too.
    let pixels_per_row = width.max(1);
    let target_rows = (pixel_budget / pixels_per_row).max(1).min(height);
    let row_step = (height / target_rows).max(1);

    let mut hist = [0u32; HIST_BINS];
    let mut total: u32 = 0;
    let mut hdr_pixels: u32 = 0;
    let mut wide_gamut_pixels: u32 = 0;
    let mut peak_nits: f32 = 0.0;
    let mut wide_gamut_peak: f32 = 0.0;

    // Highlight descriptors — accumulated over the super-white mask (pixels with
    // luma > SDR_THRESHOLD_NITS). `0` everywhere when the image has no highlights.
    let mut hl_count: u32 = 0;
    let mut hl_luma_sum = 0.0f64;
    let mut hl_luma_sq = 0.0f64;
    let mut hl_chroma_sum = 0.0f64;
    let mut hl_chroma_sq = 0.0f64;
    let mut hl_h_edges: u32 = 0;
    let mut hl_v_edges: u32 = 0;
    // Per-sampled-row luma + highlight-mask, kept one row back for the H/V edge
    // scan (a 2-row sliding window over the sampling grid).
    let mut cur_luma: Vec<f32> = Vec::with_capacity(width);
    let mut cur_hdr: Vec<bool> = Vec::with_capacity(width);
    let mut prev_luma: Vec<f32> = Vec::new();
    let mut prev_hdr: Vec<bool> = Vec::new();
    // A "strong edge" is a >10% relative luma jump between adjacent highlights.
    const HL_EDGE_REL: f32 = 0.10;

    // Gamut downcast counters. For each pixel, project the linear
    // RGB from source primaries to {sRGB, P3} and count pixels whose
    // every channel stays within [GAMUT_LO, GAMUT_HI] = [-0.005, 1.005].
    // Tolerance absorbs small numerical noise from the matrix walk.
    let m_to_srgb = primaries_to_srgb_matrix(desc.primaries);
    let m_to_p3 = primaries_to_displayp3_matrix(desc.primaries);
    // The EOTF (signal → linear) is applied a ROW at a time through linear-srgb's
    // public auto-dispatched SIMD slices (`default::pq_to_linear_slice` etc.) — a
    // magetypes f32x16 polynomial, ~11× faster than the per-pixel scalar EOTF and
    // ~3× faster than precomputing a u16 LUT at this budget (measured). We gather a
    // row's normalized samples into `lin_buf`, EOTF the whole slice, then walk the
    // linearized values for the per-pixel statistics. `nch` colour channels/pixel.
    let nch = color_channels.min(4);
    let mut lin_buf: Vec<f32> = Vec::with_capacity(width * nch);
    // Per-row nits + bin-value scratch for the SIMD `log2` histogram binning.
    let mut nits_buf: Vec<f32> = Vec::with_capacity(width);
    let mut binv_buf: Vec<f32> = Vec::with_capacity(width);
    // Bt709 ⊆ both sRGB(=Bt709) and P3, so every Bt709-source pixel is trivially
    // in both gamuts — skip the per-pixel matrix projections (exact: 1.0).
    let trivial_gamut_bt709 = matches!(desc.primaries, ColorPrimaries::Bt709);
    let mut srgb_in: u32 = 0;
    let mut p3_in: u32 = 0;
    const GAMUT_LO: f32 = -0.005;
    const GAMUT_HI: f32 = 1.005;
    #[inline]
    fn in_gamut(r: f32, g: f32, b: f32) -> bool {
        let range = GAMUT_LO..=GAMUT_HI;
        range.contains(&r) && range.contains(&g) && range.contains(&b)
    }

    // For BT.2020-ish luma weights — close enough for cross-primary
    // luminance approximation. The exact primaries vary, but the
    // relative weight on green dominates so the choice barely shifts
    // the peak / P99.
    const WL_R: f32 = 0.2627;
    const WL_G: f32 = 0.6780;
    const WL_B: f32 = 0.0593;

    // Effective-bit-depth probe (integer sources only): track the OR
    // of the low byte across samples. If it stays at 0 the source is
    // u8-promoted; otherwise we have at least 9-bit content. Counts
    // distinct low-byte values via a 256-bin presence flag for a
    // sharper estimate up to ~10/12-bit.
    let mut low_byte_seen = [false; 256];
    let mut low_byte_distinct: u32 = 0;
    // Byte-replication flag: a losslessly u8→u16-promoted source has
    // (c<<8)|c for every sample, so the low byte equals the high byte
    // everywhere. The distinct-low-byte heuristic alone can't see this (a
    // promoted ramp has many distinct low bytes and looks like ≥14-bit) — this
    // is the content-independent signature the field docs note is needed.
    let mut byte_replicated = true;
    let probe_bits = matches!(ch, ChannelType::U16);

    let mut y = 0usize;
    while y < height {
        let row = slice.row(y as u32);
        // Pixel stride within a row: 1 for narrow images, larger for
        // very wide images so we don't blow past the budget on a
        // single row.
        let row_pixel_stride = ((width as u32) / 1024).max(1) as usize;

        // Pass 1 — gather this row's sampled pixels' normalized colour samples into
        // `lin_buf`, and run the low-byte / byte-replication probe (which needs the
        // raw u16 bytes, not the linear value).
        lin_buf.clear();
        let mut x = 0usize;
        while x < width {
            let off = x * bpp;
            if off + color_channels * ch_bytes > row.len() {
                break;
            }
            for c in 0..nch {
                lin_buf.push(read_sample(ch, &row[off + c * ch_bytes..]));
            }
            if probe_bits && color_channels >= 1 {
                let low = row[off]; // little-endian: byte 0 is the low byte
                let high = row[off + 1];
                if low != high {
                    byte_replicated = false; // genuine ≥9-bit content
                }
                if !low_byte_seen[low as usize] {
                    low_byte_seen[low as usize] = true;
                    low_byte_distinct += 1;
                }
            }
            x += row_pixel_stride;
        }

        // Pass 2 — SIMD EOTF the whole row's gathered samples (signal → linear).
        eotf_slice(tf_kind, &mut lin_buf);

        // Pass 3 — per-pixel statistics from the linearized buffer. The luminance
        // histogram is deferred: gather `nits` here, then bin the row in one SIMD
        // `log2_midp` pass (Pass 4) instead of a per-pixel scalar `log2`.
        let n_pixels = lin_buf.len() / nch;
        nits_buf.clear();
        cur_luma.clear();
        cur_hdr.clear();
        for pi in 0..n_pixels {
            let linears = &lin_buf[pi * nch..pi * nch + nch];
            let mut linear_max = 0.0f32;
            let mut linear_min = f32::INFINITY;
            for &l in linears {
                if l > linear_max {
                    linear_max = l;
                }
                if l < linear_min {
                    linear_min = l;
                }
            }
            let linear_luma = if color_channels >= 3 {
                WL_R * linears[0] + WL_G * linears[1] + WL_B * linears[2]
            } else {
                // Grayscale — luma = the single channel, already linear.
                linears[0]
            };

            let nits = linear_luma * peak_nits_unit;
            if nits > peak_nits {
                peak_nits = nits;
            }
            if linear_max > wide_gamut_peak {
                wide_gamut_peak = linear_max;
            }
            if linear_max > 1.0 {
                wide_gamut_pixels += 1;
            }
            let is_hl = nits > SDR_THRESHOLD_NITS;
            if is_hl {
                hdr_pixels += 1;
                hl_count += 1;
                hl_luma_sum += nits as f64;
                hl_luma_sq += (nits as f64) * (nits as f64);
                // Linear saturation: 0 = achromatic, 1 = one channel fully dominant.
                let sat = if linear_max > 1e-6 {
                    ((linear_max - linear_min) / linear_max).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                hl_chroma_sum += sat as f64;
                hl_chroma_sq += (sat as f64) * (sat as f64);
            }
            cur_luma.push(linear_luma);
            cur_hdr.push(is_hl);
            // Gamut-coverage projections (only meaningful for ≥ 3
            // colour channels — grayscale by construction has the
            // pixel sitting on the achromatic axis, in every gamut).
            if color_channels >= 3 {
                if trivial_gamut_bt709 {
                    // Bt709 ⊆ sRGB and ⊆ P3 — in both by construction.
                    srgb_in += 1;
                    p3_in += 1;
                } else {
                    let (sr_r, sr_g, sr_b) = match m_to_srgb {
                        Some(m) => mat3_mul(m, linears[0], linears[1], linears[2]),
                        None => (linears[0], linears[1], linears[2]),
                    };
                    if in_gamut(sr_r, sr_g, sr_b) {
                        srgb_in += 1;
                    }
                    let (p3_r, p3_g, p3_b) = mat3_mul(m_to_p3, linears[0], linears[1], linears[2]);
                    if in_gamut(p3_r, p3_g, p3_b) {
                        p3_in += 1;
                    }
                }
            } else {
                // Achromatic pixel — by definition it sits in every
                // gamut. Counting both buckets keeps the fraction
                // meaningful for grayscale-in-RGB sources.
                srgb_in += 1;
                p3_in += 1;
            }
            nits_buf.push(nits);
            total += 1;
        }

        // Pass 4 — SIMD `log2_midp` bin values for the whole row, then scatter into
        // the histogram (the data-dependent index keeps the scatter scalar).
        binv_buf.resize(nits_buf.len(), 0.0);
        incant!(nits_bin_value_slice(&nits_buf, &mut binv_buf));
        for &v in &binv_buf {
            // `v ≥ 0` (nits ≥ 0), so `as usize` floors; clamp to the top bin.
            hist[(v as usize).min(HIST_BINS - 1)] += 1;
        }

        // Highlight edges: a >10% relative luma jump between two adjacent
        // highlight samples. Horizontal = within this row; vertical = same column
        // against the previous sampled row.
        for pi in 1..cur_hdr.len() {
            if cur_hdr[pi] && cur_hdr[pi - 1] {
                let (a, b) = (cur_luma[pi], cur_luma[pi - 1]);
                if (a - b).abs() > HL_EDGE_REL * a.max(b).max(1e-6) {
                    hl_h_edges += 1;
                }
            }
        }
        if !prev_hdr.is_empty() {
            let n = cur_hdr.len().min(prev_hdr.len());
            for pi in 0..n {
                if cur_hdr[pi] && prev_hdr[pi] {
                    let (a, b) = (cur_luma[pi], prev_luma[pi]);
                    if (a - b).abs() > HL_EDGE_REL * a.max(b).max(1e-6) {
                        hl_v_edges += 1;
                    }
                }
            }
        }
        std::mem::swap(&mut prev_luma, &mut cur_luma);
        std::mem::swap(&mut prev_hdr, &mut cur_hdr);

        y += row_step;
    }

    if total == 0 {
        return DepthStats::default();
    }
    let total_f = total as f32;

    // P99: walk the histogram from the top; pick the bin where the
    // cumulative count first reaches 1% of total.
    let target = (total / 100).max(1);
    let mut cum: u32 = 0;
    let mut p99_bin = 0usize;
    for b in (0..HIST_BINS).rev() {
        cum += hist[b];
        if cum >= target {
            p99_bin = b;
            break;
        }
    }
    let p99_nits = bin_to_nits(p99_bin).min(peak_nits);

    let headroom_stops = if peak_nits > 0.0 {
        (peak_nits / PEAK_SRGB_NITS).max(1.0).log2()
    } else {
        0.0
    };

    // Effective bit depth.
    let effective_bit_depth = match ch {
        ChannelType::U8 => 8,
        ChannelType::F32 | ChannelType::F16 => {
            // f32 / f16: report the storage depth. A finer probe
            // (sample-quantization grid analysis) is deferred — the
            // storage depth is the more useful per-codec signal
            // (drives jxl modular bit width and avif encode_depth).
            if matches!(ch, ChannelType::F32) {
                32
            } else {
                16
            }
        }
        ChannelType::U16 if byte_replicated => 8, // losslessly promoted u8
        ChannelType::U16 => effective_depth_from_low_byte(low_byte_distinct, total),
        // Defensive: ChannelType is non_exhaustive.
        _ => 0,
    };

    // hdr_present: peak well above SDR AND a non-trivial fraction of
    // pixels are above the threshold AND the transfer function can
    // carry HDR. Counting pixels avoids tripping on a single rounding
    // outlier.
    let hdr_pixel_fraction = hdr_pixels as f32 / total_f;
    let wide_gamut_fraction = wide_gamut_pixels as f32 / total_f;
    let hdr_present =
        hdr_capable && peak_nits > 1.5 * SDR_THRESHOLD_NITS && hdr_pixel_fraction > 0.001;
    let gamut_coverage_srgb = srgb_in as f32 / total_f;
    let gamut_coverage_p3 = p3_in as f32 / total_f;

    // Highlight descriptors from the super-white accumulators. All `0` when the
    // image carries no highlights (`hl_count == 0`), so SDR content reads zeros.
    let (
        highlight_luma_mean,
        highlight_luma_std,
        highlight_chroma_mean,
        highlight_chroma_std,
        highlight_edge_count,
        highlight_orientation_ratio,
    ) = if hl_count > 0 {
        let n = hl_count as f64;
        let lmean = hl_luma_sum / n;
        let lstd = (hl_luma_sq / n - lmean * lmean).max(0.0).sqrt();
        let cmean = hl_chroma_sum / n;
        let cstd = (hl_chroma_sq / n - cmean * cmean).max(0.0).sqrt();
        let total_edges = (hl_h_edges + hl_v_edges) as f32;
        let edge_density = total_edges / hl_count as f32;
        let orient = if total_edges > 0.0 {
            hl_h_edges as f32 / total_edges
        } else {
            0.0
        };
        (
            lmean as f32,
            lstd as f32,
            cmean as f32,
            cstd as f32,
            edge_density,
            orient,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    DepthStats {
        peak_nits,
        p99_nits,
        headroom_stops,
        hdr_pixel_fraction,
        wide_gamut_peak,
        wide_gamut_fraction,
        effective_bit_depth,
        hdr_present,
        gamut_coverage_srgb,
        gamut_coverage_p3,
        highlight_luma_mean,
        highlight_luma_std,
        highlight_chroma_mean,
        highlight_chroma_std,
        highlight_edge_count,
        highlight_orientation_ratio,
    }
}

/// Map low-byte-distinct count to effective bit depth. Reasoning: a
/// u8-promoted u16 has every low byte equal to its high byte, so the
/// distinct low-byte count caps at 256 — but it only reaches 256 if
/// the high byte is uniformly distributed. Genuine 10/12/14/16-bit
/// content, by contrast, has the low byte sweeping uniformly through
/// 256 values regardless of high-byte distribution, so a small image
/// region exhibits ~256 distinct low bytes quickly.
///
/// In practice the discriminator that catches u8-promoted-to-u16 is
/// simpler: in u8-promoted u16 every low byte is *also* a high byte
/// of the same sample, so `low == high` for every sample. Detecting
/// that is a separate test we don't do here — the distinct count is
/// the conservative signal: <16 ⇒ 8-bit-class content, 16..64 ⇒
/// 10-bit, 64..192 ⇒ 12-bit, ≥192 ⇒ 14-bit-or-finer.
#[inline]
fn effective_depth_from_low_byte(distinct: u32, total: u32) -> u32 {
    // Tiny samples can't tell us anything — fall back to storage depth.
    if total < 64 {
        return 16;
    }
    match distinct {
        0..=15 => 8,
        16..=63 => 10,
        64..=191 => 12,
        _ => 14, // ≥ 192 distinct low-byte values ⇒ effectively 14-bit or finer
    }
}

/// Cheap effective-bit-depth probe for u16 SDR sources, used by the SDR fast
/// path. Byte reads only — NO EOTF / gamut matrix / histogram — so it costs a
/// small fraction of the full luminance walk. Byte-replication (low byte ==
/// high byte for every sample) is the losslessly-promoted-u8 signature ⇒ 8;
/// otherwise the distinct-low-byte estimate. Strides identically to the main
/// walk so the sample set matches.
fn sdr_u16_effective_depth(
    slice: &PixelSlice<'_>,
    bpp: usize,
    width: usize,
    height: usize,
    pixel_budget: usize,
) -> u32 {
    let pixels_per_row = width.max(1);
    let target_rows = (pixel_budget / pixels_per_row).max(1).min(height);
    let row_step = (height / target_rows).max(1);
    let row_pixel_stride = ((width as u32) / 1024).max(1) as usize;
    let mut low_byte_seen = [false; 256];
    let (mut distinct, mut total, mut byte_replicated) = (0u32, 0u32, true);
    let mut y = 0usize;
    while y < height {
        let row = slice.row(y as u32);
        let mut x = 0usize;
        while x < width {
            let off = x * bpp;
            if off + 1 >= row.len() {
                break;
            }
            let (low, high) = (row[off], row[off + 1]);
            if low != high {
                byte_replicated = false;
            }
            if !low_byte_seen[low as usize] {
                low_byte_seen[low as usize] = true;
                distinct += 1;
            }
            total += 1;
            x += row_pixel_stride;
        }
        y += row_step;
    }
    if total == 0 {
        16
    } else if byte_replicated {
        8 // losslessly promoted u8
    } else {
        effective_depth_from_low_byte(distinct, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::PixelDescriptor;

    /// The measurement that picked the HDR-scan EOTF strategy: per-pixel scalar PQ
    /// EOTF vs a 65536-entry LUT vs linear-srgb's public auto-dispatched SIMD slice
    /// (`default::pq_to_linear_slice`). The slice wins (~11× scalar, ~3× the LUT) at
    /// the 100k-pixel budget, so the scan uses it (row-batched). Prints `PQEOTFPERF`.
    #[test]
    fn pq_eotf_strategy_perf() {
        use std::time::Instant;
        let tf = TransferFunction::Pq;
        const NSAMPLES: usize = 300_000; // ~100k pixels × 3 channels
        // u64 math so the LCG doesn't overflow 32-bit usize on i686.
        let samples: Vec<u16> = (0..NSAMPLES)
            .map(|i| ((i as u64 * 2_654_435 + 1) % 65536) as u16)
            .collect();
        const ITERS: u32 = 20;
        let (mut t_pp, mut t_lut, mut t_simd) = (0u128, 0u128, 0u128);
        let mut sink = 0.0f32;
        for _ in 0..ITERS {
            // Scalar per-pixel EOTF.
            let a = Instant::now();
            let mut s = 0.0f32;
            for &v in &samples {
                s += eotf(tf, v as f32 / 65535.0);
            }
            t_pp += a.elapsed().as_nanos();
            sink += s;

            // 65536-entry LUT (build + lookup).
            let b = Instant::now();
            let mut lut = vec![0.0f32; 65536];
            for (i, e) in lut.iter_mut().enumerate() {
                *e = eotf(tf, i as f32 / 65535.0);
            }
            let mut s2 = 0.0f32;
            for &v in &samples {
                s2 += lut[v as usize];
            }
            t_lut += b.elapsed().as_nanos();
            sink += s2;

            // linear-srgb's public auto-dispatched SIMD EOTF slice (gather + SIMD).
            let c = Instant::now();
            let mut buf: Vec<f32> = samples.iter().map(|&v| v as f32 / 65535.0).collect();
            linear_srgb::default::pq_to_linear_slice(&mut buf);
            sink += buf.iter().copied().sum::<f32>();
            t_simd += c.elapsed().as_nanos();
        }
        let per = |t: u128| t as f64 / (ITERS as f64 * NSAMPLES as f64);
        println!(
            "PQEOTFPERF ns/sample: scalar={:.2} lut_incl_build={:.2} simd_slice={:.2}  \
             (lut {:.2}x, simd {:.2}x)  sink={sink}",
            per(t_pp),
            per(t_lut),
            per(t_simd),
            per(t_pp) / per(t_lut).max(1e-9),
            per(t_pp) / per(t_simd).max(1e-9),
        );
        assert!(sink.is_finite());
    }

    /// ns/pixel of the full HDR depth scan (PQ u16, 1MP) + the nits_to_bin log2's
    /// share (scalar libm log2 vs a SIMD log2_midp stand-in over the same count).
    /// Prints `HDRDEPTHPERF` — the data for "where to focus next".
    #[test]
    fn hdr_depth_scan_perf() {
        use std::time::Instant;
        let (w, h) = (1024usize, 1024usize);
        let desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
        // Deterministic-ish u16 PQ content, varied so the scan walks (not uniform).
        let mut buf = vec![0u8; w * h * 6];
        for (i, px) in buf.chunks_exact_mut(2).enumerate() {
            let v = ((i as u64 * 2_654_435 + 1) % 65536) as u16;
            px.copy_from_slice(&v.to_le_bytes());
        }
        let s = PixelSlice::new(&buf, w as u32, h as u32, w * 6, desc).unwrap();
        let npx = (w * h) as f64;
        const ITERS: u32 = 30;
        let mut t = 0u128;
        let mut sink = 0.0f32;
        for _ in 0..ITERS {
            let a = Instant::now();
            let d = scan_depth(&s, 500_000);
            t += a.elapsed().as_nanos();
            sink += d.peak_nits + d.p99_nits;
        }
        // Isolate the nits_to_bin libm log2 cost over the sampled count (~500k).
        let sampled = 500_000u32.min((w * h) as u32);
        let nits_vals: Vec<f32> = (0..sampled).map(|i| (i % 10000) as f32).collect();
        let mut t_log = 0u128;
        let mut bsink = 0usize;
        for _ in 0..ITERS {
            let a = Instant::now();
            for &n in &nits_vals {
                bsink += nits_to_bin(n);
            }
            t_log += a.elapsed().as_nanos();
        }
        println!(
            "HDRDEPTHPERF 1MP PQ-u16: scan_total={:.2} ns/px | nits_to_bin_log2={:.2} ns/sampled-px (~{:.2} ns/px)  sink={sink} {bsink}",
            t as f64 / (ITERS as f64 * npx),
            t_log as f64 / (ITERS as f64 * sampled as f64),
            t_log as f64 / (ITERS as f64 * npx),
        );
        assert!(sink.is_finite());
    }

    /// The row-batched SIMD-slice EOTF scan must be size-independent: a large image
    /// (many rows of gather→slice) and a small one give the same peak for uniform
    /// content. Guards the gather/EOTF/scatter restructure against row-boundary or
    /// indexing bugs.
    #[test]
    fn hdr_scan_consistent_across_sizes() {
        const V: u16 = 40_000;
        let desc = PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq);
        let mk = |w: usize, h: usize| -> Vec<u8> {
            let vb = V.to_le_bytes();
            let mut buf = vec![0u8; w * h * 6];
            for px in buf.chunks_exact_mut(6) {
                px[0..2].copy_from_slice(&vb);
                px[2..4].copy_from_slice(&vb);
                px[4..6].copy_from_slice(&vb);
            }
            buf
        };
        let big = mk(256, 256); // 196_608 samples > 65536 ⇒ LUT path
        let small = mk(64, 48); // 9_216 samples < 65536 ⇒ per-pixel path
        let s_big = PixelSlice::new(&big, 256, 256, 256 * 6, desc).unwrap();
        let s_small = PixelSlice::new(&small, 64, 48, 64 * 6, desc).unwrap();
        let d_big = scan_depth(&s_big, 100_000);
        let d_small = scan_depth(&s_small, 100_000);
        assert_eq!(
            d_big.peak_nits, d_small.peak_nits,
            "u16 EOTF LUT path must match the per-pixel path bit-for-bit"
        );
        assert_eq!(d_big.wide_gamut_peak, d_small.wide_gamut_peak);
    }

    #[test]
    fn solid_srgb_canonical_sdr_profile_consistent_across_channel_types() {
        // SDR fast path: u8 AND a lossless u16 promotion of the same bytes both
        // short-circuit to the canonical SDR display profile — fast AND
        // consistent. peak/p99 are the 80-nit display reference (display-
        // referred for SDR by design; the content brightness is a Tier 1/3
        // signal), and the cheap byte-replication probe still reads the
        // promoted u16 as 8-bit.
        let buf = vec![128u8; 32 * 32 * 3];
        let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.peak_nits, PEAK_SRGB_NITS);
        assert_eq!(d.p99_nits, PEAK_SRGB_NITS);
        assert_eq!(d.effective_bit_depth, 8);
        assert!(!d.hdr_present);
        assert_eq!(d.headroom_stops, 0.0);
        assert_eq!(d.wide_gamut_fraction, 0.0);

        // lossless u8→u16: (c<<8)|c == c*257 → same canonical profile.
        let buf16: Vec<u8> = buf
            .iter()
            .flat_map(|&c| (((c as u16) << 8) | c as u16).to_ne_bytes())
            .collect();
        let s16 = PixelSlice::new(&buf16, 32, 32, 32 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
        let d16 = scan_depth(&s16, 100_000);
        assert_eq!(d16.peak_nits, PEAK_SRGB_NITS);
        assert_eq!(d16.p99_nits, PEAK_SRGB_NITS);
        assert_eq!(d.peak_nits, d16.peak_nits); // channel-type consistent
        assert_eq!(d16.effective_bit_depth, 8); // byte-replication ⇒ 8
    }

    #[test]
    fn solid_white_srgb_u8_at_sdr_reference() {
        // sRGB white via the SDR fast path: the canonical 80-nit display
        // reference (exact constant, no walk).
        let buf = vec![255u8; 32 * 32 * 3];
        let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.peak_nits, PEAK_SRGB_NITS);
        assert_eq!(d.hdr_pixel_fraction, 0.0);
        assert!(!d.hdr_present);
    }

    #[test]
    fn solid_pq_full_signal_is_high_dynamic_range() {
        // PQ code 1.0 ⇒ linear 1.0 ⇒ 10000 nits per ST 2084.
        let mut buf = vec![0u8; 32 * 32 * 3 * 4];
        let one = 1.0_f32.to_le_bytes();
        for px in buf.chunks_exact_mut(12) {
            px[0..4].copy_from_slice(&one);
            px[4..8].copy_from_slice(&one);
            px[8..12].copy_from_slice(&one);
        }
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Pq);
        let s = PixelSlice::new(&buf, 32, 32, 32 * 12, desc).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!(
            (d.peak_nits - 10_000.0).abs() < 5.0,
            "peak={} expected ~10000",
            d.peak_nits
        );
        assert!(d.headroom_stops > 6.0, "headroom={}", d.headroom_stops);
        assert_eq!(d.hdr_pixel_fraction, 1.0);
        assert!(d.hdr_present);
    }

    #[test]
    fn solid_hlg_full_signal_is_hdr_at_1000_nits() {
        // HLG signal 1.0 ⇒ linear 1.0 ⇒ 1000 nits (our convention).
        let mut buf = vec![0u8; 16 * 16 * 12];
        let one = 1.0_f32.to_le_bytes();
        for px in buf.chunks_exact_mut(12) {
            px[0..4].copy_from_slice(&one);
            px[4..8].copy_from_slice(&one);
            px[8..12].copy_from_slice(&one);
        }
        let desc = PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Hlg);
        let s = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!(
            (d.peak_nits - 1_000.0).abs() < 1.0,
            "peak={} expected ~1000",
            d.peak_nits
        );
        assert!(d.hdr_present);
    }

    #[test]
    fn linear_f32_above_one_is_wide_gamut_signal() {
        // Linear-light f32 with values above 1.0 ⇒ wide-gamut peak.
        let mut buf = vec![0u8; 16 * 16 * 12];
        let two = 2.0_f32.to_le_bytes();
        for px in buf.chunks_exact_mut(12) {
            px[0..4].copy_from_slice(&two);
            px[4..8].copy_from_slice(&two);
            px[8..12].copy_from_slice(&two);
        }
        let s = PixelSlice::new(&buf, 16, 16, 16 * 12, PixelDescriptor::RGBF32_LINEAR).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!((d.wide_gamut_peak - 2.0).abs() < 1e-3);
        assert_eq!(d.wide_gamut_fraction, 1.0);
    }

    #[test]
    fn u8_promoted_u16_reads_as_8bit_effective_depth() {
        // u16 samples = u8 * 257 with very few distinct high bytes.
        // The distinct-low-byte probe should map to 8-bit.
        let mut buf = vec![0u8; 16 * 16 * 6];
        for (i, px) in buf.chunks_exact_mut(2).enumerate() {
            // Cycle through only 4 high-byte values across the image.
            let v = ((i % 4) * 64) as u8;
            let u = (v as u16) * 257;
            px.copy_from_slice(&u.to_le_bytes());
        }
        let s = PixelSlice::new(&buf, 16, 16, 16 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.effective_bit_depth, 8);
    }

    #[test]
    fn genuine_16bit_u16_reads_as_high_effective_depth() {
        // Sweep u16 values uniformly — many distinct low bytes ⇒
        // effective_bit_depth ≥ 14.
        let mut buf = vec![0u8; 64 * 64 * 6];
        let mut state = 0xC001_u32;
        for px in buf.chunks_exact_mut(2) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let u = (state & 0xFFFF) as u16;
            px.copy_from_slice(&u.to_le_bytes());
        }
        let s = PixelSlice::new(&buf, 64, 64, 64 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert!(
            d.effective_bit_depth >= 14,
            "expected ≥14, got {}",
            d.effective_bit_depth
        );
    }

    #[test]
    fn empty_slice_returns_default_stats() {
        let buf: Vec<u8> = Vec::new();
        let s = PixelSlice::new(&buf, 0, 0, 0, PixelDescriptor::RGB8_SRGB).unwrap();
        let d = scan_depth(&s, 100_000);
        assert_eq!(d.peak_nits, 0.0);
        assert_eq!(d.effective_bit_depth, 0);
        assert!(!d.hdr_present);
    }
}
