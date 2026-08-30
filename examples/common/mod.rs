//! Deterministic real-content fixtures from the sibling `codec-corpus`
//! checkout, shared by every example that needs "a `side`×`side` RGB8 buffer of
//! real content of class C".
//!
//! Not an example itself — Cargo only auto-discovers `examples/*.rs` and
//! `examples/*/main.rs`, so this module is compiled only where it is
//! `#[path = "common/mod.rs"] mod common;`-included. Consumers today:
//! `per_feature_cost_grid.rs` (cost grid) and `feature_bits.rs` (the byte-exact
//! optimization lock). One owner for the loader, per the no-duplicates rule.
//!
//! **Content classes** map to directories of the codec-corpus checkout. No
//! synthetic gradients — every pixel is real content; sizes larger than a
//! source are built by mosaicking DISTINCT centre crops (2×2 / 4×4 / 8×8),
//! which costs seam lines but keeps the content real.
//!
//! | class | sources | what it is |
//! |---|---|---|
//! | `photo` | `CID22` (≥512²) for 64/256, `clic2025` (≥1024²) for 1024+ | ordinary photography |
//! | `photohard` | `gb82` (576², 25 images) | photography chosen to resist metric gaming — fine facial detail, low-contrast sky, digital noise |
//! | `screen` | `gb82-sc` (8 of 10 images have min dim ≥ 512) | screenshots / UI / anti-aliased text — sharp edges and flat regions |
//! | `mixed` | alternating `photo` and `screen` tiles | the two regimes inside one image |
//!
//! **`screen` used to read `gb82`, which is the *photographic* set** (see the
//! codec-corpus README: "GB82 … compact photographic benchmarking" vs "GB82-SC
//! … screen content & screenshot compression"). Any grid produced before
//! 2026-08-28 measured photo-vs-photo under a photo/screen heading. `gb82` is
//! still measured, under its correct name `photohard`.
//!
//! No dedicated line-art class: the only line-art-ish sources in the local
//! checkout are `gb82-sc/graph.png` (796×481) and `windows95.png` (640×480),
//! both below the 512 tile floor, and `imazen-26` is manifest-only here (the
//! images live in R2). Line art is represented inside `screen` by the text and
//! UI content of `codec_wiki` / `terminal` / `windows` rather than by a class of
//! its own — stated so nobody reads a line-art conclusion out of this grid.

use std::path::{Path, PathBuf};

/// Sides the grids sweep. Every larger side must be an integer multiple of the
/// class's tile side.
#[allow(dead_code)] // used by per_feature_cost_grid, not by feature_bits
pub const ALL_SIDES: [u32; 5] = [64, 256, 1024, 2048, 4096];

pub struct Class {
    pub name: &'static str,
    /// `(max_side_served, tile_side, sources)` — the first tier whose
    /// `max_side >= side` wins; sides larger than `tile_side` mosaic distinct
    /// centre crops of that tier's sources.
    pub tiers: Vec<(u32, u32, Vec<PathBuf>)>,
    /// Interleave with this class's tiles (checkerboard by tile index) to build
    /// a mixed-content image. `None` for the homogeneous classes.
    pub interleave_with: Option<Box<Class>>,
}

fn walk_pngs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk_pngs(&p, out);
        } else if p.extension().is_some_and(|e| e == "png") {
            out.push(p);
        }
    }
}

/// Width/height straight out of the IHDR — avoids decoding every candidate.
fn png_dims(p: &Path) -> Option<(u32, u32)> {
    let bytes = std::fs::read(p).ok()?;
    if bytes.len() < 24 || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((w, h))
}

/// Every PNG under `dir` whose smaller side is at least `min_dim`, sorted
/// (deterministic — the grids index into this by position).
pub fn sources(dir: &Path, min_dim: u32) -> Vec<PathBuf> {
    let mut all = Vec::new();
    walk_pngs(dir, &mut all);
    all.into_iter()
        .filter(|p| png_dims(p).is_some_and(|(w, h)| w.min(h) >= min_dim))
        .collect()
}

fn photo_class(corpus: &Path) -> Class {
    let cid22 = sources(&corpus.join("CID22"), 512);
    let clic = sources(&corpus.join("clic2025"), 1024);
    assert!(!cid22.is_empty(), "no CID22 PNGs ≥ 512 under {corpus:?}");
    assert!(!clic.is_empty(), "no clic2025 PNGs ≥ 1024 under {corpus:?}");
    Class {
        name: "photo",
        tiers: vec![(512, 512, cid22), (u32::MAX, 1024, clic)],
        interleave_with: None,
    }
}

fn screen_class(corpus: &Path) -> Class {
    let sc = sources(&corpus.join("gb82-sc"), 512);
    assert!(!sc.is_empty(), "no gb82-sc PNGs ≥ 512 under {corpus:?}");
    Class {
        name: "screen",
        tiers: vec![(u32::MAX, 512, sc)],
        interleave_with: None,
    }
}

pub fn classes(corpus: &Path) -> Vec<Class> {
    let gb82 = sources(&corpus.join("gb82"), 512);
    assert!(!gb82.is_empty(), "no gb82 PNGs ≥ 512 under {corpus:?}");
    vec![
        photo_class(corpus),
        Class {
            name: "photohard",
            tiers: vec![(u32::MAX, 512, gb82)],
            interleave_with: None,
        },
        screen_class(corpus),
        Class {
            name: "mixed",
            tiers: photo_class(corpus).tiers,
            interleave_with: Some(Box::new(screen_class(corpus))),
        },
    ]
}

/// Classes **in addition to** [`classes`], for grids that need more per-class
/// crop diversity or a line-art class than the four cost-grid classes give.
///
/// Deliberately NOT folded into [`classes`]: `examples/feature_bits.rs` (the
/// byte-exact optimization lock) iterates `classes()`, so adding to it would
/// silently change the lock's fixture set. Callers opt in.
///
/// | class | sources | why it exists |
/// |---|---|---|
/// | `screenwide` | `gb82-sc` + `qoi-benchmark/screenshot_web` | `screen`'s 8-source pool makes every mosaic crop **identical** at 2048² and 4096² (`n·n` tiles per crop is a multiple of 8, so `k % len` repeats), so `screen` has an effective sample size of 1 there. 22 sources breaks the aliasing. |
/// | `lineart` | 6 synthetic-graphics / diagram sources | the four cost-grid classes have no line-art class at all. |
///
/// **`lineart` is a thin pool — 6 sources, two of them Frymire variants.** It is
/// enough to show whether line art behaves like the other classes, not enough
/// for a tight per-class percentile. Sources were selected by measured signature
/// (high `flat_color_block_ratio`, low `distinct_color_bins` / `gradient_fraction`),
/// not by filename.
#[allow(dead_code)] // used by budget_drift_grid, not by the cost grid or the lock
pub fn extra_classes(corpus: &Path) -> Vec<Class> {
    let mut wide = sources(&corpus.join("gb82-sc"), 512);
    wide.extend(sources(&corpus.join("qoi-benchmark/screenshot_web"), 512));
    assert!(wide.len() > 10, "screenwide pool too small: {}", wide.len());

    // Explicit list: these are picked out of three different corpus dirs by
    // content signature, not by a directory that happens to hold only line art.
    let lineart: Vec<PathBuf> = [
        "imageflow/test_inputs/rings2.png",
        "imageflow/test_inputs/frymire.png",
        "imageflow/test_inputs/frymire-srgb.png",
        "png-conformance/wm_upload_wikimedia_org_c8a458b0cef3d942.png",
        "image-rs/test-images/png/bugfixes/debug_triangle_corners_widescreen.png",
        "image-rs/test-images/png/iptc.png",
    ]
    .iter()
    .map(|r| corpus.join(r))
    .filter(|p| p.exists())
    .collect();
    assert!(
        lineart.len() >= 5,
        "line-art pool is {} sources; expected 6 — corpus checkout incomplete?",
        lineart.len()
    );

    vec![
        Class {
            name: "screenwide",
            tiers: vec![(u32::MAX, 512, wide)],
            interleave_with: None,
        },
        Class {
            name: "lineart",
            tiers: vec![(u32::MAX, 512, lineart)],
            interleave_with: None,
        },
    ]
}

fn tier_for(class: &Class, side: u32) -> &(u32, u32, Vec<PathBuf>) {
    class
        .tiers
        .iter()
        .find(|(max_side, _, _)| *max_side >= side)
        .expect("no tier serves this side")
}

/// A `side`×`side` RGB8 buffer: an `n`×`n` mosaic of centre crops of distinct
/// sources, `n = side / tile`. Crop `crop_idx` starts at source `crop_idx*n*n`
/// and wraps when the pool is exhausted. Returns `(buffer, distinct_tiles)`.
///
/// For a `mixed` class the tile at `(ty, tx)` comes from the primary class when
/// `(ty + tx)` is even and from `interleave_with` when it is odd.
pub fn build(side: u32, class: &Class, crop_idx: usize) -> (Vec<u8>, usize) {
    let (_, tile_side, _) = tier_for(class, side);
    let tile = (*tile_side).min(side);
    let n = side / tile;
    assert_eq!(n * tile, side, "side {side} not a multiple of tile {tile}");
    let mut buf = vec![0u8; (side as usize) * (side as usize) * 3];
    let per_crop = (n * n) as usize;
    let start = crop_idx * per_crop;
    let mut used = 0usize;
    for ty in 0..n {
        for tx in 0..n {
            let alt = class
                .interleave_with
                .as_deref()
                .filter(|_| (ty + tx) % 2 == 1);
            let srcs = match alt {
                Some(other) => &tier_for(other, side).2,
                None => &tier_for(class, side).2,
            };
            let k = start + (ty * n + tx) as usize;
            let src = &srcs[k % srcs.len()];
            used += 1;
            let img = image::open(src)
                .unwrap_or_else(|e| panic!("{src:?}: {e}"))
                .to_rgb8();
            let (w, h) = (img.width(), img.height());
            assert!(
                w >= tile && h >= tile,
                "{src:?} is {w}×{h}, smaller than the {tile} tile"
            );
            let x0 = (w - tile) / 2;
            let y0 = (h - tile) / 2;
            let crop = image::imageops::crop_imm(&img, x0, y0, tile, tile).to_image();
            let raw = crop.as_raw();
            for row in 0..tile as usize {
                let dst_y = (ty * tile) as usize + row;
                let dst_x = (tx * tile) as usize;
                let dst = (dst_y * side as usize + dst_x) * 3;
                let src_off = row * tile as usize * 3;
                buf[dst..dst + tile as usize * 3]
                    .copy_from_slice(&raw[src_off..src_off + tile as usize * 3]);
            }
        }
    }
    (buf, used)
}
