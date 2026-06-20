//! Feature **content-hash versioning** — so features serialized to disk today
//! can be safely reused for training years from now.
//!
//! # The model
//!
//! Each feature gets a [`feature_version_hash`] derived from a **golden test**:
//! run the feature over a fixed, ordered synthetic corpus and hash the resulting
//! value vector, mixed with the crate's **caret-compatibility root** (`0.MINOR`
//! for a 0.x crate, `MAJOR` for ≥1.0 — the same boundary Cargo uses to decide
//! whether two versions unify). The properties this buys:
//!
//! * **Only changed features change.** A localized math change moves exactly the
//!   features it touches; everything else keeps its hash, so serialized vectors
//!   stay reusable feature-by-feature instead of all-or-nothing.
//! * **The caret-root is a hard reset.** Within a `^`-compatible line the golden
//!   values govern; across the boundary the maintainer is *signaling* a possible
//!   break, so hashes differ even if the golden values happen to match — bounding
//!   the "golden set must fully characterize each feature" burden to one line.
//! * **Self-maintaining, not hand-bumped.** [`golden_is_stable`] fails when the
//!   current code stops reproducing the committed golden (beyond a tolerance).
//!   The maintainer re-blesses the golden, which changes the hash — you cannot
//!   silently ship a behaviour change.
//!
//! # Platform handling
//!
//! The hash is computed from the **committed golden text** (`versioning_golden.tsv`),
//! not from a fresh extraction, so it is identical on every platform — x86, ARM,
//! i686, wasm all read the same bytes. The stability test re-extracts and
//! compares *values* within [`REL_TOLERANCE`], which absorbs cross-platform float
//! noise; that tolerance is precisely "platforms close enough to share a
//! serialized vector." (The cleaner long game is bit-exact determinism — pinned
//! `libm`, fixed reduction order — which removes the tolerance entirely.)
//!
//! # Coverage
//!
//! The corpus spans tiny/photo/screen/line-art/palette/alpha content **and HDR**
//! — PQ `RGB16`, Bt2020 wide-gamut, and super-white `RGBF32_LINEAR` cases that
//! fire the depth tier under the `hdr` feature — under the default gamma config.
//! (Linear-light is a *separate* reuse axis, `zenanalyze_api`'s `config_hash`,
//! not folded into the code-version hash; the corpus can grow a second config
//! pass when more features have linear paths.) The keystone is the
//! `every_feature_varies` lint: a feature constant across the
//! corpus is *unversioned* (a math change preserving the constant wouldn't move
//! the hash), so every feature must take ≥2 distinct values. That lint is only
//! satisfiable by a genuinely diverse corpus, so it mechanically enforces the
//! coverage rather than trusting good intentions.
//!
//! [`golden_is_stable`]: #the-tripwire

use crate::feature::AnalysisFeature;
use std::sync::OnceLock;

/// The committed golden: `feat_<name>\t<v0>\t<v1>\t…` per feature, one line each,
/// values in corpus order. Re-bless by running the tests with
/// `ZENANALYZE_BLESS_GOLDEN=1` and committing the rewritten file.
const GOLDEN: &str = include_str!("versioning_golden.tsv");

/// Relative tolerance for the value-stability check — the cross-platform float
/// budget below which two extractions are deemed the same version. Tunable; the
/// sensitivity test proves a deliberate change exceeds it.
pub const REL_TOLERANCE: f32 = 1.0e-4;

/// The caret-compatibility root of a semver string: `"0.2"` for `0.2.7`, `"1"`
/// for `1.5.3`, `"2"` for `2.0.0`. The boundary across which reuse is forbidden
/// outright (and the boundary Cargo unifies within).
#[must_use]
pub fn caret_root(version: &str) -> &str {
    let major = version.split('.').next().unwrap_or("");
    if major == "0" {
        match version.match_indices('.').nth(1) {
            Some((i, _)) => &version[..i],
            None => version,
        }
    } else {
        major
    }
}

/// FNV-1a-64 over bytes — a stable, dependency-free digest.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `fnv1a(caret_root ++ golden_row)` for `feature`.
fn hash_row(version: &str, row_values_text: &str) -> u64 {
    let mut buf = caret_root(version).as_bytes().to_vec();
    buf.push(b'\t');
    buf.extend_from_slice(row_values_text.as_bytes());
    fnv1a(&buf)
}

/// The content-hash version of `feature`.
///
/// Stable across platforms (computed from the committed golden text, not a live
/// extraction) and across patch/minor drift within a caret line that stays inside
/// [`REL_TOLERANCE`]; changes when the maintainer re-blesses the golden after a
/// real behaviour change, or when the caret root bumps. Stamp it alongside
/// serialized features so a future training run can check, feature-by-feature,
/// whether its current analyzer would produce compatible values. `None` if the
/// feature has no golden row (a build whose feature set predates it).
#[must_use]
pub fn feature_version_hash(feature: AnalysisFeature) -> Option<u64> {
    version_hash_for_name(feature.name())
}

/// Like [`feature_version_hash`] but keyed by the feature's canonical **name** —
/// the form serialized in an [`crate::OwnedOffer`] / on disk — so a serialized
/// name list can be re-stamped without re-resolving each [`AnalysisFeature`].
#[must_use]
pub fn feature_version_hash_by_name(name: &str) -> Option<u64> {
    version_hash_for_name(name)
}

/// The parsed golden as `(feat_<name>, values_text)` rows, parsed once.
fn golden_rows() -> &'static Vec<(String, String)> {
    static ROWS: OnceLock<Vec<(String, String)>> = OnceLock::new();
    ROWS.get_or_init(|| {
        GOLDEN
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                l.split_once('\t')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
            })
            .collect()
    })
}

fn version_hash_for_name(name: &str) -> Option<u64> {
    let key = format!("feat_{name}");
    let row = golden_rows().iter().find(|(k, _)| *k == key)?;
    Some(hash_row(crate::analyzer_version(), &row.1))
}

/// A stable digest of the **value-affecting input framing** of `slice`: the
/// descriptor's transfer function, primaries, and alpha mode, plus any signaled
/// diffuse-white. The *same pixels* under different primaries (per-primaries luma
/// weights), transfer (HDR EOTF), alpha mode (premultiplied vs straight), or
/// diffuse-white (the PQ anchor) produce different features — so this is the
/// third leg of a serialized feature set's reuse key, alongside the per-feature
/// [`feature_version_hash`] (code) and `AnalysisQuery::config_hash` (config).
///
/// Bit depth / channel type is deliberately **not** hashed: u8 / u16 / f32 SDR
/// converge to identical features by the consistency contract, so mixing depth in
/// would split a reuse key that should match. Only framing that changes values is
/// folded in. A mismatch costs an own-pass, never a wrong reuse, so the digest
/// errs toward distinguishing (e.g. unsignaled vs signaled-default diffuse-white
/// hash apart even where the PQ anchor would coincide).
#[must_use]
pub fn descriptor_hash(slice: &zenpixels::PixelSlice<'_>) -> u64 {
    let dw_nits = slice
        .color_context()
        .and_then(|c| c.diffuse_white)
        .map(|d| d.nits());
    descriptor_hash_of(&slice.descriptor(), dw_nits)
}

/// The framing digest for an explicit descriptor + optional signaled diffuse-white
/// (in nits) — the same hash [`descriptor_hash`] derives from a [`zenpixels::PixelSlice`],
/// for callers that have the descriptor directly (e.g. the RGB8 fast path, which is
/// `descriptor_hash_of(&PixelDescriptor::RGB8_SRGB, None)`).
#[must_use]
pub fn descriptor_hash_of(
    descriptor: &zenpixels::PixelDescriptor,
    diffuse_white_nits: Option<f32>,
) -> u64 {
    // Discriminants are stable (`#[repr(u8)]` in zenpixels); alpha `None` → 0,
    // distinct from every real `AlphaMode` (which start at 1). Unsignaled
    // diffuse-white → 0 bits, distinct from any signaled value.
    let dw = diffuse_white_nits
        .map_or(0u64, |n| u64::from(n.to_bits()))
        .to_le_bytes();
    let bytes = [
        descriptor.transfer as u8,
        descriptor.primaries as u8,
        descriptor.alpha.map_or(0u8, |a| a as u8),
        dw[0],
        dw[1],
        dw[2],
        dw[3],
        dw[4],
        dw[5],
        dw[6],
        dw[7],
    ];
    fnv1a(&bytes)
}

/// The framing digest of the **RGB8 sRGB fast path** —
/// `descriptor_hash_of(&PixelDescriptor::RGB8_SRGB, None)`. This is the framing of
/// every [`crate::analyze_features_rgb8`] extraction (and the
/// `extract_features_for_picker` example), so the in-workspace extractor can stamp
/// provenance without naming a descriptor.
#[must_use]
pub fn rgb8_srgb_descriptor_hash() -> u64 {
    descriptor_hash_of(&zenpixels::PixelDescriptor::RGB8_SRGB, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{AnalysisQuery, FeatureSet, FeatureValue};
    use zenpixels::PixelDescriptor;

    // ----------------------------------------------------------------------
    // The corpus — deterministic synthetic images exercising every feature.
    // ----------------------------------------------------------------------

    /// One corpus case: owns its pixel bytes + descriptor so it can be analyzed
    /// through whichever entry fits (RGB8 fast path or the generic `PixelSlice`).
    struct Case {
        w: u32,
        h: u32,
        bytes: Vec<u8>,
        descriptor: PixelDescriptor,
        rgb8_fast: bool,
        diffuse_white: Option<f32>,
    }

    impl Case {
        /// Every [`FeatureSet::SUPPORTED`] feature present in this build, in
        /// ascending feature-id order, under the given config.
        fn extract(&self, linear_light: bool) -> Vec<(u16, FeatureValue)> {
            let query = AnalysisQuery::new(FeatureSet::SUPPORTED).with_linear_light(linear_light);
            let results = if self.rgb8_fast {
                crate::analyze_features_rgb8(&self.bytes, self.w, self.h, &query)
            } else {
                let stride = self.bytes.len() / self.h as usize;
                let mut slice = zenpixels::PixelSlice::new(
                    &self.bytes,
                    self.w,
                    self.h,
                    stride,
                    self.descriptor,
                )
                .expect("valid corpus slice");
                if let Some(dw) = self.diffuse_white {
                    use std::sync::Arc;
                    use zenpixels::{ColorContext, DiffuseWhite};
                    slice = slice.with_color_context(Arc::new(
                        ColorContext::default().with_diffuse_white(DiffuseWhite::new(dw)),
                    ));
                }
                crate::analyze_features(slice, &query).expect("corpus analysis")
            };
            let mut out = Vec::new();
            for f in FeatureSet::SUPPORTED.iter() {
                if let Some(v) = results.get(f) {
                    out.push((f.id(), v));
                }
            }
            out
        }
    }

    fn rgb8(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Case {
        let mut bytes = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                bytes.extend_from_slice(&f(x, y));
            }
        }
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGB8_SRGB,
            rgb8_fast: true,
            diffuse_white: None,
        }
    }

    fn rgba8(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Case {
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                bytes.extend_from_slice(&f(x, y));
            }
        }
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGBA8_SRGB,
            rgb8_fast: false,
            diffuse_white: None,
        }
    }

    /// A PQ-encoded RGB16 HDR case from an sRGB authoring image at `diffuse_white`
    /// nits, with the top quarter pushed into the HDR range by `peak_scale`.
    fn hdr_pq(w: u32, h: u32, diffuse_white: f32, peak_scale: f32) -> Case {
        use linear_srgb::tf::{linear_to_pq, srgb_to_linear};
        use zenpixels::TransferFunction;
        let mut u16s: Vec<u16> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let base = (x * 255 / w.max(1)) as f32 / 255.0;
                let hdr_boost = if y < h / 4 { peak_scale } else { 1.0 };
                for _ in 0..3 {
                    let lin = srgb_to_linear(base) * diffuse_white / 10000.0 * hdr_boost;
                    let pq = linear_to_pq(lin).clamp(0.0, 1.0);
                    u16s.push((pq * 65535.0 + 0.5) as u16);
                }
            }
        }
        let bytes: Vec<u8> = u16s.iter().flat_map(|&v| v.to_le_bytes()).collect();
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGB16.with_transfer(TransferFunction::Pq),
            rgb8_fast: false,
            diffuse_white: Some(diffuse_white),
        }
    }

    /// Saturated primaries in **Bt2020** PQ — pure-hue pixels that fall outside
    /// the sRGB (and partly the P3) gamut, so the gamut-coverage / wide-gamut
    /// depth features take non-trivial values instead of the in-gamut constant.
    fn hdr_wide_gamut(w: u32, h: u32) -> Case {
        use linear_srgb::tf::linear_to_pq;
        use zenpixels::{ColorPrimaries, TransferFunction};
        let mut u16s: Vec<u16> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let lum = 0.05 + 0.45 * (x as f32 / w.max(1) as f32);
                let rgb_lin = match (x + y) % 3 {
                    0 => [lum, 0.0, 0.0],
                    1 => [0.0, lum, 0.0],
                    _ => [0.0, 0.0, lum],
                };
                for &c in &rgb_lin {
                    let pq = linear_to_pq(c).clamp(0.0, 1.0);
                    u16s.push((pq * 65535.0 + 0.5) as u16);
                }
            }
        }
        let bytes: Vec<u8> = u16s.iter().flat_map(|&v| v.to_le_bytes()).collect();
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGB16
                .with_transfer(TransferFunction::Pq)
                .with_primaries(ColorPrimaries::Bt2020),
            rgb8_fast: false,
            diffuse_white: Some(203.0),
        }
    }

    /// RGBF32 **linear-light** with super-white highlights (channel > 1.0) — the
    /// only source kind whose EOTF exceeds 1.0, so `wide_gamut_fraction`
    /// (linear_max > 1.0) and `wide_gamut_peak` take non-zero values. (PQ/HLG
    /// decode to [0,1], so they can't exercise these.)
    fn hdr_linear_float(w: u32, h: u32) -> Case {
        let mut f32s: Vec<f32> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let bright = if y < h / 3 {
                    1.5 + 2.5 * (x as f32 / w as f32)
                } else {
                    0.3
                };
                f32s.extend_from_slice(&[bright, bright * 0.6, 0.2]);
            }
        }
        let bytes: Vec<u8> = f32s.iter().flat_map(|&v| v.to_le_bytes()).collect();
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGBF32_LINEAR,
            rgb8_fast: false,
            diffuse_white: None,
        }
    }

    /// Generic RGB `u16` case with explicit transfer + primaries — covers the
    /// descriptor code-paths (Native-vs-Convert, per-primaries luma, transfer
    /// decode) that change a feature's value for the *same* logical content.
    fn u16_rgb(
        w: u32,
        h: u32,
        transfer: zenpixels::TransferFunction,
        primaries: zenpixels::ColorPrimaries,
        diffuse_white: Option<f32>,
        f: impl Fn(u32, u32) -> [u16; 3],
    ) -> Case {
        let mut u16s: Vec<u16> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                u16s.extend_from_slice(&f(x, y));
            }
        }
        let bytes: Vec<u8> = u16s.iter().flat_map(|&v| v.to_le_bytes()).collect();
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGB16
                .with_transfer(transfer)
                .with_primaries(primaries),
            rgb8_fast: false,
            diffuse_white,
        }
    }

    /// RGBA8 with **premultiplied** alpha — exercises the premult alpha path
    /// (distinct from the straight-alpha `rgba8` cases above).
    fn rgba8_premult(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Case {
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                bytes.extend_from_slice(&f(x, y));
            }
        }
        Case {
            w,
            h,
            bytes,
            descriptor: PixelDescriptor::RGBA8_SRGB
                .with_alpha(Some(zenpixels::AlphaMode::Premultiplied)),
            rgb8_fast: false,
            diffuse_white: None,
        }
    }

    /// The ordered corpus. Order is part of the contract — only **append**;
    /// reordering rewrites every hash.
    fn corpus() -> Vec<Case> {
        let lcg = |seed: u32| seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        vec![
            rgb8(8, 8, |_, _| [0, 0, 0]),         // solid black
            rgb8(16, 16, |_, _| [128, 128, 128]), // solid gray
            rgb8(8, 8, |_, _| [255, 255, 255]),   // solid white
            rgb8(64, 48, |x, _| {
                let v = (x * 255 / 63) as u8;
                [v, v, v]
            }), // gray h-gradient
            rgb8(48, 64, |_, y| {
                let v = (y * 255 / 63) as u8;
                [v, 255 - v, 128]
            }), // color v-gradient
            rgb8(64, 64, |x, y| {
                if (x / 2 + y / 2) % 2 == 0 {
                    [240, 240, 240]
                } else {
                    [16, 16, 16]
                }
            }), // checkerboard
            rgb8(32, 48, |x, _| {
                if x % 2 == 0 {
                    [255, 200, 50]
                } else {
                    [30, 40, 200]
                }
            }), // v-lines
            rgb8(48, 32, |_, y| {
                if y % 2 == 0 {
                    [255, 255, 255]
                } else {
                    [0, 0, 0]
                }
            }), // h-lines
            rgb8(64, 64, move |x, y| {
                let s = lcg(x.wrapping_mul(7919).wrapping_add(y.wrapping_mul(104_729)));
                [
                    (s & 0xff) as u8,
                    ((s >> 8) & 0xff) as u8,
                    ((s >> 16) & 0xff) as u8,
                ]
            }), // noise
            rgb8(32, 32, |x, y| {
                let j = (x.wrapping_add(y) % 16) as u8;
                [230u8.saturating_add(j / 2), 175 + j, 150 + j]
            }), // skin tone
            rgb8(64, 64, |x, y| {
                if x % 12 == 0 || y % 12 == 0 {
                    [10, 10, 10]
                } else {
                    [250, 250, 250]
                }
            }), // line art
            rgb8(64, 64, |x, y| {
                let idx = ((x / 16) + 4 * (y / 16)) % 16;
                [(idx * 16) as u8, (255 - idx * 16) as u8, (idx * 8) as u8]
            }), // 16-color palette
            rgb8(
                64,
                48,
                |x, _| if x < 32 { [220, 20, 20] } else { [20, 20, 220] },
            ), // chroma split
            rgb8(4, 4, |x, y| [(x * 60) as u8, (y * 60) as u8, 90]), // tiny
            rgb8(128, 16, |x, _| {
                let v = (x * 255 / 127) as u8;
                [v, 90, 255 - v]
            }), // wide
            rgb8(256, 256, move |x, y| {
                let s = lcg(x.wrapping_mul(31).wrapping_add(y.wrapping_mul(131)));
                [
                    ((x ^ y) & 0xff) as u8,
                    (s & 0xff) as u8,
                    ((s >> 16) & 0xff) as u8,
                ]
            }), // large textured
            rgba8(32, 32, |x, y| [(x * 8) as u8, (y * 8) as u8, 128, 255]), // opaque
            rgba8(32, 32, |x, y| {
                let a = ((x + y) * 4).min(255) as u8;
                [200, 100, 50, a]
            }), // semi-transparent
            hdr_pq(64, 48, 203.0, 20.0),          // HDR PQ bright
            hdr_pq(64, 48, 100.0, 4.0),           // HDR PQ dim
            hdr_wide_gamut(64, 48),               // Bt2020 saturated — gamut-coverage features
            hdr_linear_float(48, 48),             // RGBF32 super-white — wide_gamut_fraction/peak
            // --- descriptor path coverage: same logical content, framings the
            //     code branches on (so a per-path change is versioned). ---
            u16_rgb(
                64,
                48,
                zenpixels::TransferFunction::Srgb,
                zenpixels::ColorPrimaries::Bt709,
                None,
                |x, _| {
                    let v = (x * 65535 / 63) as u16;
                    [v, v / 2, 65535 - v]
                },
            ), // Native u16 SDR — versions the u8==u16==f32 consistency
            u16_rgb(
                64,
                48,
                zenpixels::TransferFunction::Pq,
                zenpixels::ColorPrimaries::DisplayP3,
                Some(203.0),
                |x, y| {
                    let lum = (3000 + x * 600) as u16;
                    match (x + y) % 3 {
                        0 => [lum, 200, 200],
                        1 => [200, lum, 200],
                        _ => [200, 200, lum],
                    }
                },
            ), // DisplayP3 PQ saturated — P3 luma + gamut_coverage_p3 path
            u16_rgb(
                64,
                48,
                zenpixels::TransferFunction::Hlg,
                zenpixels::ColorPrimaries::Bt2020,
                Some(203.0),
                |x, y| {
                    let v = ((x ^ y) * 700 % 60000) as u16;
                    [v, 30000u16.saturating_sub(v / 2), v / 3]
                },
            ), // HLG Bt2020 — the HLG decode path
            rgba8_premult(32, 32, |x, y| {
                let a = ((x + y) * 6).min(255) as u8;
                let s = |c: u32| (a as u32 * c / 255) as u8;
                [s(200), s(80), a / 2, a] // premultiplied colour (already × alpha)
            }), // premultiplied alpha path
        ]
    }

    /// Per-feature ordered value vectors across the corpus under the **default
    /// (gamma) config**. The linear-light config is a *separate* reuse axis
    /// (`zenanalyze_api`'s `config_hash`), not folded into the code-version hash;
    /// extend this with a second pass (`case.extract(true)`) once enough features
    /// have linear paths to make the extra golden columns worth their bytes
    /// (today only `variance` differs, so they would be ~99% redundant).
    fn extract_matrix() -> Vec<(String, Vec<FeatureValue>)> {
        let cases = corpus();
        let mut order: Vec<u16> = Vec::new();
        let mut rows: std::collections::BTreeMap<u16, Vec<FeatureValue>> =
            std::collections::BTreeMap::new();
        for linear in [false] {
            for case in &cases {
                for (id, v) in case.extract(linear) {
                    if !rows.contains_key(&id) {
                        order.push(id);
                    }
                    rows.entry(id).or_default().push(v);
                }
            }
        }
        order.sort_unstable();
        order.dedup();
        order
            .into_iter()
            .map(|id| {
                let name = format!("feat_{}", crate::feature_name(id).unwrap_or("?"));
                (name, rows.remove(&id).unwrap_or_default())
            })
            .collect()
    }

    /// **Type-canonical** text for one value — discrete types as exact integers
    /// (so a large count round-trips losslessly and an off-by-one is detectable),
    /// `f32` at fixed precision, `NaN` as a sentinel (a real value — e.g. peak
    /// luminance on an SDR image).
    fn format_one(v: FeatureValue) -> String {
        match v {
            FeatureValue::U32(n) => n.to_string(),
            FeatureValue::U64(n) => n.to_string(),
            FeatureValue::Bool(b) => u8::from(b).to_string(),
            FeatureValue::F32(x) if x.is_nan() => "nan".to_string(),
            FeatureValue::F32(x) => format!("{x:.7e}"),
        }
    }

    fn format_values(values: &[FeatureValue]) -> String {
        values
            .iter()
            .map(|&v| format_one(v))
            .collect::<Vec<_>>()
            .join("\t")
    }

    fn golden_text_rows() -> Vec<(String, Vec<String>)> {
        GOLDEN
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut it = l.split('\t');
                let name = it.next().unwrap_or("").to_string();
                (name, it.map(str::to_string).collect())
            })
            .collect()
    }

    /// Per-feature `f32` relative-tolerance overrides for the few features whose
    /// large reductions / transcendentals (entropy `log`, gamut matrices, big
    /// pixel sums) drift more across platforms than the global [`REL_TOLERANCE`].
    /// Empty until measured on real ARM/i686/wasm runs — the honest way to set
    /// these is from observed spread, not a guess (and bit-exact determinism via
    /// pinned `libm` would retire them entirely). Keyed by `feat_<name>`.
    const F32_TOLERANCE_OVERRIDES: &[(&str, f32)] = &[];

    fn f32_tolerance(feat_key: &str) -> f32 {
        F32_TOLERANCE_OVERRIDES
            .iter()
            .find(|(k, _)| *k == feat_key)
            .map_or(REL_TOLERANCE, |(_, t)| *t)
    }

    fn rel_close(a: f32, b: f32, tol: f32) -> bool {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() && b.is_nan();
        }
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
    }

    /// Tolerance keyed on the value's TYPE (and, for `f32`, the feature). Discrete
    /// features (counts, bit depths, bools) are computed as integers with no float
    /// noise, so they must match **exactly** — a single relative tolerance would
    /// miss an off-by-one in a large count (e.g. 32768→32769 is 3e-5 relative,
    /// under `REL_TOLERANCE`). Only `f32` features get a relative tolerance, with
    /// an optional per-feature override ([`F32_TOLERANCE_OVERRIDES`]).
    fn value_matches(recomputed: FeatureValue, golden_text: &str, f32_tol: f32) -> bool {
        match recomputed {
            FeatureValue::F32(x) => {
                if golden_text == "nan" {
                    x.is_nan()
                } else {
                    golden_text
                        .parse::<f32>()
                        .map(|e| rel_close(x, e, f32_tol))
                        .unwrap_or(false)
                }
            }
            other => format_one(other) == golden_text,
        }
    }

    /// THE KEYSTONE. Every feature must take ≥2 distinct values across the corpus
    /// — a constant feature is unversioned, since a math change preserving the
    /// constant would never move its hash. Also forces corpus diversity.
    #[test]
    fn every_feature_varies() {
        let matrix = extract_matrix();
        assert!(!matrix.is_empty(), "corpus produced no features");
        let mut constant = Vec::new();
        for (name, values) in &matrix {
            let mut bits: Vec<u32> = values.iter().map(|v| v.to_f32().to_bits()).collect();
            bits.sort_unstable();
            bits.dedup();
            if bits.len() < 2 {
                constant.push(name.clone());
            }
        }
        assert!(
            constant.is_empty(),
            "these features never vary across the corpus → they are unversioned; \
             add a case that exercises them: {constant:?}"
        );
    }

    /// THE TRIPWIRE. Current code must reproduce the committed golden within
    /// tolerance; when it doesn't, a behaviour changed — re-bless with
    /// `ZENANALYZE_BLESS_GOLDEN=1` and review. Never relax the tolerance to pass.
    #[test]
    fn golden_is_stable() {
        let matrix = extract_matrix();

        if std::env::var_os("ZENANALYZE_BLESS_GOLDEN").is_some() {
            let mut out = String::new();
            for (name, values) in &matrix {
                out.push_str(&format!("{name}\t{}\n", format_values(values)));
            }
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/versioning_golden.tsv");
            std::fs::write(path, out).expect("write golden");
            eprintln!("blessed golden: {path} ({} features)", matrix.len());
            return;
        }

        let golden: std::collections::HashMap<String, Vec<String>> =
            golden_text_rows().into_iter().collect();
        assert!(
            !golden.is_empty(),
            "golden is empty — bless it first: ZENANALYZE_BLESS_GOLDEN=1 cargo test --all-features golden_is_stable"
        );
        let mut drift = Vec::new();
        for (name, values) in &matrix {
            let Some(expected) = golden.get(name) else {
                drift.push(format!("{name}: present in build but missing from golden"));
                continue;
            };
            if expected.len() != values.len() {
                drift.push(format!(
                    "{name}: len {} != golden {}",
                    values.len(),
                    expected.len()
                ));
                continue;
            }
            for (i, (v, e)) in values.iter().zip(expected).enumerate() {
                if !value_matches(*v, e, f32_tolerance(name)) {
                    drift.push(format!("{name}[{i}]: {} vs golden {}", format_one(*v), e));
                    break;
                }
            }
        }
        assert!(
            drift.is_empty(),
            "feature outputs drifted from the committed golden — a behaviour \
             changed. Re-bless (ZENANALYZE_BLESS_GOLDEN=1) and review:\n{}",
            drift.join("\n")
        );
    }

    /// Proves the version detects change: a changed value moves the hash, and the
    /// caret root participates.
    #[test]
    fn version_hash_is_change_sensitive() {
        let line0 = GOLDEN
            .lines()
            .find(|l| !l.trim().is_empty())
            .expect("golden non-empty");
        let (_, vals0) = line0.split_once('\t').expect("golden row has values");
        let perturbed = format!("{vals0}9");
        assert_ne!(
            hash_row("0.2.0", vals0),
            hash_row("0.2.0", &perturbed),
            "a changed value must change the hash"
        );
        assert_ne!(
            hash_row("0.2.0", vals0),
            hash_row("1.0.0", vals0),
            "the caret root must participate in the hash"
        );
    }

    /// The public API resolves a hash for every present feature.
    #[test]
    fn every_present_feature_has_a_version_hash() {
        for f in FeatureSet::SUPPORTED.iter() {
            assert!(
                feature_version_hash(f).is_some(),
                "feature `{}` has no golden row — re-bless the golden",
                f.name()
            );
        }
    }

    #[test]
    fn caret_root_is_the_compat_boundary() {
        assert_eq!(caret_root("0.2.7"), "0.2");
        assert_eq!(caret_root("0.2.7-rc1"), "0.2");
        assert_eq!(caret_root("0.3.0"), "0.3"); // 0.x minor IS the breaking position
        assert_eq!(caret_root("1.5.3"), "1"); // ≥1.0: minor is non-breaking
        assert_eq!(caret_root("2.0.0"), "2");
        assert_ne!(caret_root("0.2.0"), caret_root("2.0.0")); // not the bare digit
    }
}
