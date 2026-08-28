//! Per-feature cost grid on REAL content across the sweep-discipline sizes —
//! the measurement zenanalyze#41 ("per-feature cost vs use") and #50 Sub-A
//! ("per-feature cost map at all sizes, per content class") both ask for.
//!
//! `examples/per_feature_cost.rs` measures the same two quantities on ONE
//! synthetic-noise buffer at 1 MP / 4 MP. This grid measures them per feature at
//! {64, 256, 1024, 2048, 4096}² on real photo and screen content, so the fixed
//! per-call intercept (tiny) and the per-pixel slope (large) are both observed
//! instead of assumed, and content-dependent early exits (palette quick-scan,
//! grayscale) show up as a photo/screen split:
//!
//! * **solo**  — `analyze_features` with only that feature requested (what the
//!   feature costs on its own, dependencies included);
//! * **loo**   — `SUPPORTED` baseline minus `SUPPORTED \ F` (what the feature
//!   adds when everything else is already computed; ≤ 0 = shares a pass, noise).
//!
//! Content (no synthetic gradients — sweep rule): tiles are centre crops of
//! codec-corpus images, mosaicked when no source is large enough (2×2 / 4×4 /
//! 8×8 of DISTINCT crops; every pixel stays real content at the cost of seam
//! lines — same trade `scripts/make_costgrid_crops.py` made). Photo: CID22
//! (512²) for 64/256, clic2025 (min dim ≥ 1024) for 1024+. Screen: gb82 (576²),
//! 512-tiles mosaicked for 1024+ (25 sources, so 4096² repeats tiles).
//! Deterministic: sorted walks, fixed offsets, no RNG.
//!
//! Output is a raw TSV (one row per class × side × crop × feature) that
//! `tools/feature_inventory.py --cost <tsv>` aggregates (median over crops,
//! α + β·pixels fit, cost × consumption cross-reference) into
//! `docs/feature-consumption.md`.
//!
//! Run (build with the same cargo features as the inventory universe):
//!
//! ```text
//! nice -n 19 cargo run --release --features hdr --example per_feature_cost_grid
//! ```
//!
//! Env: `ZENANALYZE_CORPUS_DIR` (default `../codec-corpus`), `PFC_SIDES`
//! (comma list, default all five), `PFC_CLASSES` (default `photo,screen`),
//! `PFC_CROPS` (crops per class × side, default 2), `PFC_OUT` (default
//! `benchmarks/per_feature_cost_grid_<date>.tsv`), `PFC_APPEND=1` (append rows
//! without a header, for size-by-size runs).
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

const ALL_SIDES: [u32; 5] = [64, 256, 1024, 2048, 4096];

struct Class {
    name: &'static str,
    /// (max side served by these sources, tile side, sources) — first entry whose
    /// `max_side >= side` wins; larger sides mosaic `tile` crops of it.
    tiers: Vec<(u32, u32, Vec<PathBuf>)>,
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

fn png_dims(p: &Path) -> Option<(u32, u32)> {
    let bytes = std::fs::read(p).ok()?;
    if bytes.len() < 24 || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((w, h))
}

fn sources(dir: &Path, min_dim: u32) -> Vec<PathBuf> {
    let mut all = Vec::new();
    walk_pngs(dir, &mut all);
    all.into_iter()
        .filter(|p| png_dims(p).is_some_and(|(w, h)| w.min(h) >= min_dim))
        .collect()
}

fn classes(corpus: &Path) -> Vec<Class> {
    let cid22 = sources(&corpus.join("CID22"), 512);
    let clic = sources(&corpus.join("clic2025"), 1024);
    let gb82 = sources(&corpus.join("gb82"), 512);
    assert!(!cid22.is_empty(), "no CID22 PNGs ≥ 512 under {corpus:?}");
    assert!(!clic.is_empty(), "no clic2025 PNGs ≥ 1024 under {corpus:?}");
    assert!(!gb82.is_empty(), "no gb82 PNGs ≥ 512 under {corpus:?}");
    vec![
        Class {
            name: "photo",
            tiers: vec![(512, 512, cid22), (u32::MAX, 1024, clic)],
        },
        Class {
            name: "screen",
            tiers: vec![(u32::MAX, 512, gb82)],
        },
    ]
}

/// RGB8 `side`×`side` buffer: a `side/tile` × `side/tile` mosaic of centre crops
/// of distinct sources, starting at source index `start` (crop `i` uses
/// sources `i*n*n ..`, wrapping when the pool is exhausted).
fn build(side: u32, class: &Class, crop_idx: usize) -> (Vec<u8>, usize) {
    let (_, tile_side, srcs) = class
        .tiers
        .iter()
        .find(|(max_side, _, _)| *max_side >= side)
        .expect("tier");
    let tile = (*tile_side).min(side);
    let n = side / tile;
    assert_eq!(n * tile, side, "side {side} not a multiple of tile {tile}");
    let mut buf = vec![0u8; (side as usize) * (side as usize) * 3];
    let per_crop = (n * n) as usize;
    let start = crop_idx * per_crop;
    let mut used = 0usize;
    for ty in 0..n {
        for tx in 0..n {
            let k = start + (ty * n + tx) as usize;
            let src = &srcs[k % srcs.len()];
            used += 1;
            let img = image::open(src)
                .unwrap_or_else(|e| panic!("{src:?}: {e}"))
                .to_rgb8();
            let (w, h) = (img.width(), img.height());
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

/// Median wall-clock ns of `analyze_features(query)`: ≥ 5 runs and ≥ 100 ms of
/// measurement (≤ 300 runs), after 2 warm-ups.
fn median_ns(buf: &[u8], side: u32, query: &AnalysisQuery) -> f64 {
    let stride = side as usize * 3;
    let run = || {
        let s = PixelSlice::new(
            std::hint::black_box(buf),
            side,
            side,
            stride,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap();
        std::hint::black_box(analyze_features(s, query)).unwrap();
    };
    run();
    run();
    let mut times = Vec::with_capacity(16);
    let t0 = Instant::now();
    while times.len() < 5 || (t0.elapsed().as_millis() < 100 && times.len() < 300) {
        let t = Instant::now();
        run();
        times.push(t.elapsed().as_nanos() as f64);
    }
    times.sort_by(f64::total_cmp);
    let n = times.len();
    if n % 2 == 1 {
        times[n / 2]
    } else {
        (times[n / 2 - 1] + times[n / 2]) / 2.0
    }
}

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86_400) as i64;
    // civil-from-days (Howard Hinnant), proleptic Gregorian
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

fn cmd(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("ZENANALYZE_CORPUS_DIR").unwrap_or_else(|_| "../codec-corpus".to_string()),
    );
    let sides: Vec<u32> = env_list("PFC_SIDES", "64,256,1024,2048,4096")
        .iter()
        .map(|s| s.parse().expect("PFC_SIDES: u32"))
        .collect();
    for s in &sides {
        assert!(ALL_SIDES.contains(s), "side {s} not in {ALL_SIDES:?}");
    }
    let want_classes = env_list("PFC_CLASSES", "photo,screen");
    let n_crops: usize = std::env::var("PFC_CROPS")
        .ok()
        .map(|v| v.parse().expect("PFC_CROPS: usize"))
        .unwrap_or(2);
    let append = std::env::var("PFC_APPEND").is_ok_and(|v| v == "1");
    let out = std::env::var("PFC_OUT")
        .unwrap_or_else(|_| format!("benchmarks/per_feature_cost_grid_{}.tsv", date()));

    let classes: Vec<Class> = classes(&corpus)
        .into_iter()
        .filter(|c| want_classes.iter().any(|w| w == c.name))
        .collect();
    assert!(!classes.is_empty(), "PFC_CLASSES matched nothing");

    let supported = FeatureSet::SUPPORTED;
    let feats: Vec<AnalysisFeature> = supported.iter().collect();

    let mut f = if append {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&out)
            .unwrap_or_else(|e| panic!("append {out}: {e}"))
    } else {
        let mut f = std::fs::File::create(&out).unwrap_or_else(|e| panic!("create {out}: {e}"));
        writeln!(
            f,
            "# zenanalyze per-feature cost grid — analyze_features RGB8_SRGB, real content\n\
             # git={} host={} arch={} date={} corpus={} crops/cell={n_crops} features={} (FeatureSet::SUPPORTED of this build)\n\
             # sides={sides:?} classes={want_classes:?}; photo=CID22 512² (64/256) + clic2025 1024² centre crops (1024+, 2×2/4×4 mosaics); screen=gb82 512² centre crops (mosaicked 2×2..8×8 for 1024+)\n\
             # baseline_ns = median of SUPPORTED; solo_ns = median with only this feature requested; loo_ns = baseline − median(SUPPORTED \\ feature) (≤ 0 → shares a pass / noise)\n\
             # median over ≥ 5 runs and ≥ 100 ms per cell; tiles = number of distinct source crops mosaicked",
            cmd("git", &["rev-parse", "--short", "HEAD"]),
            cmd("hostname", &[]),
            std::env::consts::ARCH,
            date(),
            corpus.display(),
            feats.len(),
        )
        .unwrap();
        writeln!(
            f,
            "class\tside\tpixels\tcrop\ttiles\tfeature_id\tfeature\tbaseline_ns\tsolo_ns\tloo_ns"
        )
        .unwrap();
        f
    };

    for class in &classes {
        for &side in &sides {
            for crop_idx in 0..n_crops {
                let (buf, tiles) = build(side, class, crop_idx);
                let t_cell = Instant::now();
                let baseline = median_ns(&buf, side, &AnalysisQuery::new(supported));
                for &feat in &feats {
                    let solo = median_ns(&buf, side, &AnalysisQuery::new(FeatureSet::just(feat)));
                    let without =
                        median_ns(&buf, side, &AnalysisQuery::new(supported.without(feat)));
                    writeln!(
                        f,
                        "{}\t{side}\t{}\t{crop_idx}\t{tiles}\t{}\tfeat_{}\t{baseline:.0}\t{solo:.0}\t{:.0}",
                        class.name,
                        (side as u64) * (side as u64),
                        feat.id(),
                        feat.name(),
                        baseline - without
                    )
                    .unwrap();
                }
                f.flush().unwrap();
                eprintln!(
                    "{} {side}² crop{crop_idx} ({tiles} tiles): baseline {:.3} ms, {} features in {:.1} s",
                    class.name,
                    baseline / 1e6,
                    feats.len(),
                    t_cell.elapsed().as_secs_f64()
                );
            }
        }
    }
    eprintln!("wrote {out}");
}
