//! **Does analysis thoroughness need to scale with image size?** Measures how
//! far each feature's value moves between the crate-invariant sampling budgets
//! and effectively-full sampling, across the sweep-discipline size grid.
//!
//! Motivation: [`DEFAULT_PIXEL_BUDGET`] is a fixed *absolute* cap (500 k px), so
//! the sampled *fraction* shrinks as images grow — a 4096² image is 16.8 MP
//! against a 500 k budget, ~3 % sampled. `benchmarks/perf_2026-08-28.md`
//! measures the consequence on the cost side: marginal cost falls from
//! 16.22 ns/px (64→256) to 0.62 ns/px (2048→4096), 26× less analysis per pixel
//! at 4K. This example measures the consequence on the *value* side.
//!
//! The two budgets are crate invariants, not caller knobs — they fold into
//! `feature_defs_version`, NOT `config_hash` — so sampling harder returns a
//! *different number under the same qualified name*. This grid is the evidence
//! base for deciding whether that difference matters; it changes no default.
//!
//! ## Arms
//!
//! Each cell is analyzed under several `(pixel_budget, hf_max_blocks)` arms via
//! the `#[doc(hidden)]` re-extraction backdoor
//! [`AnalysisQuery::__internal_with_overrides`] + `zenanalyze::__analyze_internal`.
//! The two knobs cap *different passes* (pixel_budget: tier 1 stripe step, tier 2
//! triplet stride, alpha + depth row stride; hf_max_blocks: the tier 3 8×8 DCT
//! block count), so they are varied independently to attribute drift correctly:
//!
//! | arm | pixel_budget | hf_max_blocks | what it isolates |
//! |---|---|---|---|
//! | `default` | 500 000 | 1 024 | the shipping reference |
//! | `pb1m` … `pb8m` | 1–8 M | 1 024 | the pixel-budget convergence curve |
//! | `pbfull` | `usize::MAX` | 1 024 | pixel-budget effect alone |
//! | `hf2k` … `hf64k` | 500 000 | 2 k–64 k | the hf-block convergence curve |
//! | `hffull` | 500 000 | 262 144 | hf-block effect alone |
//! | `full` | `usize::MAX` | 262 144 | the fully-sampled reference |
//!
//! Both ladders are needed: `hf_max_blocks` turns out to drive most of the
//! drift, so a convergence test run only along the pixel-budget ladder would
//! pass vacuously for the majority of moving features.
//!
//! 262 144 = every 8×8 block of a 4096² image, i.e. uncapped at every measured
//! size.
//!
//! ## Content
//!
//! Real codec-corpus content only (no synthetic gradients) via
//! `examples/common/mod.rs`. The four cost-grid classes (`photo`, `photohard`,
//! `screen`, `mixed`) are the *same loader on the same bytes* the cost grid
//! measures, so cost and drift are directly comparable. Two more come from
//! `common::extra_classes`:
//!
//! * `screenwide` — `screen`'s 8-source pool aliases against the mosaic tile
//!   count, making every crop **identical** at 2048² and 4096² (effective n = 1
//!   there). 22 sources fixes that; prefer `screenwide` for any screen-content
//!   percentile at the large sides.
//! * `lineart` — the cost grid has no line-art class. 6 sources, thin; enough to
//!   show whether line art behaves differently, not enough for a tight percentile.
//!
//! **Mosaic caveat, and the control for it.** No local source is 4096², so the
//! large sides are mosaics of distinct centre crops. A mosaic is spatially more
//! heterogeneous than one large photograph, and sampling error grows with
//! heterogeneity — so a mosaic could *overstate* drift. `DRIFT_NATIVE=1` re-runs
//! the grid over whole native images (never resampled — each is centre-cropped
//! to a square at its own size and bucketed to the largest swept side that
//! fits), which is the control: if native drift matches mosaic drift at the same
//! side, the mosaic is not driving the result. The local corpus supports this
//! control at 1024² (63 images) and barely at 2048² (2), and not at all at 4096².
//!
//! ## Output
//!
//! One TSV row per (class, side, crop, arm, feature), carrying the raw f32.
//! Aggregation — median / p95 of |Δ|, of relative Δ, and of Δ in units of the
//! feature's across-image spread — is `tools/budget_drift.py`; nothing is
//! aggregated here so the raw values stay auditable.
//!
//! ```text
//! nice -n 19 cargo run --release --features hdr --example budget_drift_grid
//! ```
//!
//! Env: `ZENANALYZE_CORPUS_DIR` (default `../codec-corpus`), `DRIFT_SIDES`
//! (comma list, default all five), `DRIFT_CLASSES`, `DRIFT_CROPS` (default 8),
//! `DRIFT_OUT`, `DRIFT_NATIVE=1` (native whole-image control, see above),
//! `DRIFT_ARMS` (comma list, default all).
use std::io::Write;
use std::path::PathBuf;

use zenanalyze::feature::{AnalysisQuery, FeatureSet};
use zenpixels::{PixelDescriptor, PixelSlice};

#[path = "common/mod.rs"]
mod common;
use common::{ALL_SIDES, Class, build, classes, extra_classes, sources};

/// Every 8×8 block of a 4096² image — uncapped at every size this grid sweeps.
const HF_UNCAPPED: usize = 262_144;
/// The crate-invariant defaults, mirrored here so the arm table is explicit.
/// (They are `pub(crate)` in the crate, so an example cannot name them.)
const DEFAULT_PIXEL_BUDGET: usize = 500_000;
const DEFAULT_HF_MAX_BLOCKS: usize = 1_024;

/// `(name, pixel_budget, hf_max_blocks)`.
const ARMS: &[(&str, usize, usize)] = &[
    ("default", DEFAULT_PIXEL_BUDGET, DEFAULT_HF_MAX_BLOCKS),
    ("pb1m", 1_000_000, DEFAULT_HF_MAX_BLOCKS),
    ("pb2m", 2_000_000, DEFAULT_HF_MAX_BLOCKS),
    ("pb4m", 4_000_000, DEFAULT_HF_MAX_BLOCKS),
    ("pb8m", 8_000_000, DEFAULT_HF_MAX_BLOCKS),
    ("pbfull", usize::MAX, DEFAULT_HF_MAX_BLOCKS),
    ("hf2k", DEFAULT_PIXEL_BUDGET, 2_048),
    ("hf4k", DEFAULT_PIXEL_BUDGET, 4_096),
    ("hf16k", DEFAULT_PIXEL_BUDGET, 16_384),
    ("hf64k", DEFAULT_PIXEL_BUDGET, 65_536),
    ("hffull", DEFAULT_PIXEL_BUDGET, HF_UNCAPPED),
    ("full", usize::MAX, HF_UNCAPPED),
];

fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Analyze `buf` under every selected arm, writing one row per feature.
#[allow(clippy::too_many_arguments)]
fn emit_cell(
    out: &mut impl Write,
    class: &str,
    side: u32,
    crop: usize,
    label: &str,
    buf: &[u8],
    w: u32,
    h: u32,
    arms: &[String],
) {
    let stride = w as usize * 3;
    for (arm, pb, hf) in ARMS {
        if !arms.iter().any(|a| a == arm) {
            continue;
        }
        let iq = AnalysisQuery::__internal_with_overrides(FeatureSet::SUPPORTED, *pb, *hf);
        let slice = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
        let res = zenanalyze::__analyze_internal(slice, &iq)
            .unwrap_or_else(|e| panic!("{class} {side} crop{crop} arm {arm}: {e:?}"));
        for f in FeatureSet::SUPPORTED.iter() {
            // `None` = not computed at this size (a percentile / windowed
            // estimator below its sample floor). Emitted as `nan` so the
            // aggregator can distinguish "absent" from "absent in one arm
            // only", which would itself be a budget effect worth seeing.
            let v = res.get_f32(f).unwrap_or(f32::NAN);
            // Full f32 precision: the whole point is small differences.
            writeln!(
                out,
                "{class}\t{side}\t{crop}\t{label}\t{arm}\t{pb}\t{hf}\tfeat_{}\t{:.9e}",
                f.name(),
                v
            )
            .unwrap();
        }
    }
}

/// Whole native images, centre-cropped to a square at their own size — the
/// control for the mosaic caveat. Only sources whose square side lands within
/// `[min_side, max_side]` are used, bucketed to the nearest swept side.
fn native_control(out: &mut impl Write, corpus: &std::path::Path, arms: &[String]) {
    let mut pool: Vec<(String, PathBuf)> = Vec::new();
    for (class, dir, min_dim) in [
        ("photo", "clic2025", 1024u32),
        ("screen", "gb82-sc", 1024),
        ("photohard", "gb82", 512),
    ] {
        for p in sources(&corpus.join(dir), min_dim) {
            pool.push((class.to_string(), p));
        }
    }
    for (class, path) in pool {
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("{path:?}: {e}"))
            .to_rgb8();
        let (w, h) = (img.width(), img.height());
        let sq = w.min(h);
        // Bucket to the largest swept side that fits, so the control is
        // comparable to a mosaic cell of the same nominal side.
        let Some(&side) = ALL_SIDES.iter().rev().find(|&&s| s <= sq) else {
            continue;
        };
        if side < 512 {
            continue; // below the mosaic tile floor — nothing to compare against
        }
        let x0 = (w - side) / 2;
        let y0 = (h - side) / 2;
        let crop = image::imageops::crop_imm(&img, x0, y0, side, side).to_image();
        let name = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        eprintln!("  native {class} {side}² {name}");
        emit_cell(
            out,
            &class,
            side,
            0,
            &format!("native:{name}"),
            crop.as_raw(),
            side,
            side,
            arms,
        );
    }
}

fn main() {
    let corpus = PathBuf::from(
        std::env::var("ZENANALYZE_CORPUS_DIR").unwrap_or_else(|_| "../codec-corpus".into()),
    );
    let arms = env_list(
        "DRIFT_ARMS",
        &ARMS
            .iter()
            .map(|(n, _, _)| *n)
            .collect::<Vec<_>>()
            .join(","),
    );
    let out_path = std::env::var("DRIFT_OUT")
        .unwrap_or_else(|_| "benchmarks/budget_drift_grid.tsv".to_string());
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    writeln!(
        out,
        "class\tside\tcrop\tlabel\tarm\tpixel_budget\thf_max_blocks\tfeature\tvalue"
    )
    .unwrap();

    if std::env::var("DRIFT_NATIVE").is_ok_and(|v| v == "1") {
        eprintln!("native whole-image control (mosaic control arm)");
        native_control(&mut out, &corpus, &arms);
        out.flush().unwrap();
        eprintln!("wrote {out_path}");
        return;
    }

    let want_sides = env_list("DRIFT_SIDES", &ALL_SIDES.map(|s| s.to_string()).join(","));
    let want_classes = env_list(
        "DRIFT_CLASSES",
        "photo,photohard,screen,mixed,screenwide,lineart",
    );
    let crops: usize = std::env::var("DRIFT_CROPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let mut all: Vec<Class> = classes(&corpus);
    all.extend(extra_classes(&corpus));
    for class in all
        .iter()
        .filter(|c| want_classes.iter().any(|w| w == c.name))
    {
        for side in ALL_SIDES
            .iter()
            .copied()
            .filter(|s| want_sides.iter().any(|w| w == &s.to_string()))
        {
            for crop in 0..crops {
                eprintln!("{} {side}² crop{crop}", class.name);
                let (buf, _tiles) = build(side, class, crop);
                emit_cell(
                    &mut out, class.name, side, crop, "mosaic", &buf, side, side, &arms,
                );
            }
        }
    }
    out.flush().unwrap();
    eprintln!("wrote {out_path}");
}
