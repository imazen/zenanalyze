//! Size x tier-subset cost grid on REAL photo crops — the P0 measurement for
//! zenavif's feature-hints program (zenavif docs/FEATURE_HINTS_PLAN.md).
//!
//! Complements `per_tier_cost.rs` (per-tier marginal cost on synthetic noise at
//! 1/4 MP): this grid measures CUMULATIVE tier subsets across the
//! sweep-discipline sizes {64, 256, 1024, 2048, 4096}^2 with 4 distinct real
//! crops per size (built by `scripts/make_costgrid_crops.py`), then fits
//! `total_ns = alpha + beta * pixels` per subset — the fixed per-call overhead
//! (alpha) is what decides whether per-64x64-superblock analysis is affordable.
//!
//! Subsets gate passes via representative features (requesting any feature of a
//! pass runs the whole pass — see per_tier_cost.rs):
//!   t1          full Tier-1 kernel (FULL + SKIN stripe scan), nothing else
//!   t1t2        + Tier 2 (3-row sliding-window Cb/Cr sharpness)
//!   t1t2t3      + Tier 3 (luma histogram + sampled 8x8 DCT block pass)
//!   t1t2t3_pal  + Palette (full-image distinct-color scan)
//!   full        FeatureSet::SUPPORTED (adds the Alpha pass; ~0 on RGB8)
//!
//! Run: cargo run --release --example per_tier_cost_grid
//! Env: ZENANALYZE_COSTGRID_DIR (default /mnt/v/output/zenanalyze/costgrid-crops-2026-07-02)
//!      COSTGRID_OUT (default benchmarks/feature_cost_grid_<date>.tsv)

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisFeature as AF, AnalysisQuery, FeatureSet};
use zenbench::prelude::*;
use zenpixels::{PixelDescriptor, PixelSlice};

const SIZES: [u32; 5] = [64, 256, 1024, 2048, 4096];
const N_CROPS: usize = 4;

fn subsets() -> Vec<(&'static str, FeatureSet)> {
    // Tier-1: every feature whose work rides the Tier-1 stripe scan (FULL +
    // SKIN kernels + the always-on base accumulators).
    let t1 = FeatureSet::new()
        .with(AF::Variance)
        .with(AF::Colourfulness)
        .with(AF::LaplacianVariance)
        .with(AF::EdgeSlopeStdev)
        .with(AF::ChromaLumaCovarianceCb)
        .with(AF::ChromaLumaCovarianceCr)
        .with(AF::OrientationEnergyRatio)
        .with(AF::SkinToneFraction)
        .with(AF::EdgeDensity)
        .with(AF::Uniformity)
        .with(AF::ChromaComplexity)
        .with(AF::FlatColorBlockRatio)
        .with(AF::GrayscaleScore);
    let t2 = t1.with(AF::CbSharpness).with(AF::CrSharpness);
    let t3 = t2
        .with(AF::LumaHistogramEntropy)
        .with(AF::HighFreqEnergyRatio)
        .with(AF::DctCompressibilityY)
        .with(AF::DctCompressibilityUV)
        .with(AF::AqMapMean)
        .with(AF::AqMapStd)
        .with(AF::NoiseFloorY)
        .with(AF::NoiseFloorUV)
        .with(AF::QuantSurvivalY)
        .with(AF::QuantSurvivalUv)
        .with(AF::PatchFraction)
        .with(AF::GradientFraction)
        .with(AF::InfoWeightMean)
        .with(AF::SpectralSlopeY);
    let t3p = t3
        .with(AF::DistinctColorBins)
        .with(AF::PaletteFitsIn256)
        .with(AF::PaletteLog2Size);
    vec![
        ("t1", t1),
        ("t1t2", t2),
        ("t1t2t3", t3),
        ("t1t2t3_pal", t3p),
        ("full", FeatureSet::SUPPORTED),
    ]
}

struct Crop {
    side: u32,
    idx: usize,
    bytes: Vec<u8>,
}

fn load_crops(dir: &PathBuf) -> Vec<Crop> {
    let mut crops = Vec::new();
    for side in SIZES {
        for idx in 0..N_CROPS {
            let p = dir.join(side.to_string()).join(format!("c{idx}.png"));
            let img = image::open(&p)
                .unwrap_or_else(|e| {
                    panic!("missing crop {p:?}: {e}\nrun scripts/make_costgrid_crops.py first")
                })
                .to_rgb8();
            assert_eq!((img.width(), img.height()), (side, side), "{p:?}");
            crops.push(Crop {
                side,
                idx,
                bytes: img.into_raw(),
            });
        }
    }
    crops
}

fn main() {
    let dir = PathBuf::from(
        std::env::var("ZENANALYZE_COSTGRID_DIR")
            .unwrap_or_else(|_| "/mnt/v/output/zenanalyze/costgrid-crops-2026-07-02".to_string()),
    );
    let crops: &'static [Crop] = Box::leak(load_crops(&dir).into_boxed_slice());
    let subs = subsets();
    for (name, s) in &subs {
        eprintln!("subset {name}: {} features", s.len());
    }

    // COSTGRID_SPOTCHECK=1: plain Instant loops (no zenbench) to cross-check
    // the harness's medians before numbers land in docs.
    if std::env::var("COSTGRID_SPOTCHECK").is_ok() {
        for side in SIZES {
            for crop in crops.iter().filter(|c| c.side == side && c.idx == 0) {
                for (name, set) in &subs {
                    let q = AnalysisQuery::new(*set);
                    let stride = (side * 3) as usize;
                    let run = || {
                        let s = PixelSlice::new(
                            black_box(&crop.bytes[..]),
                            side,
                            side,
                            stride,
                            PixelDescriptor::RGB8_SRGB,
                        )
                        .unwrap();
                        black_box(analyze_features(s, &q)).unwrap()
                    };
                    for _ in 0..3 {
                        run();
                    }
                    let mut times = Vec::new();
                    let t_total = std::time::Instant::now();
                    while times.len() < 10
                        || (t_total.elapsed().as_secs_f64() < 0.4 && times.len() < 3000)
                    {
                        let t = std::time::Instant::now();
                        run();
                        times.push(t.elapsed().as_secs_f64() * 1e9);
                    }
                    times.sort_by(f64::total_cmp);
                    println!(
                        "spotcheck sz{side} {name}/c0: min={:.0}ns med={:.0}ns n={}",
                        times[0],
                        times[times.len() / 2],
                        times.len()
                    );
                }
            }
        }
        return;
    }

    let result = zenbench::run(|suite| {
        for side in SIZES {
            let px = (side as u64) * (side as u64);
            suite.compare(format!("sz{side}"), |g| {
                g.config().max_rounds(40);
                g.throughput(Throughput::Elements(px));
                for crop in crops.iter().filter(|c| c.side == side) {
                    for (name, set) in subsets() {
                        let q = AnalysisQuery::new(set);
                        let buf: &'static [u8] = &crop.bytes;
                        let stride = (side * 3) as usize;
                        let label = format!("{name}/c{}", crop.idx);
                        g.bench(label, move |b| {
                            b.iter(|| {
                                let s = PixelSlice::new(
                                    black_box(buf),
                                    side,
                                    side,
                                    stride,
                                    PixelDescriptor::RGB8_SRGB,
                                )
                                .unwrap();
                                black_box(analyze_features(s, &q))
                            })
                        });
                    }
                }
            });
        }
    });
    zenbench::postprocess_result(&result);

    // ---- Collect medians, fit total = alpha + beta*px per subset. ----
    // rows[(subset, side)] -> Vec<median_ns> over crops
    let mut rows: BTreeMap<(String, u32), Vec<f64>> = BTreeMap::new();
    for cmp in &result.comparisons {
        let side: u32 = cmp.group_name.trim_start_matches("sz").parse().unwrap();
        for b in &cmp.benchmarks {
            let subset = b.name.split('/').next().unwrap().to_string();
            rows.entry((subset, side))
                .or_default()
                .push(b.summary.median);
        }
    }

    let git = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let host = std::process::Command::new("hostname")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let date = chrono_free_date();
    let out = std::env::var("COSTGRID_OUT")
        .unwrap_or_else(|_| format!("benchmarks/feature_cost_grid_{date}.tsv"));
    let mut f = std::fs::File::create(&out).expect("create out tsv");
    writeln!(
        f,
        "# zenanalyze feature cost grid — analyze_features RGB8, real photo crops\n\
         # git={git} host={host} date={date} crops_dir={} crops/size={N_CROPS}\n\
         # sizes={SIZES:?} subsets=t1,t1t2,t1t2t3,t1t2t3_pal,full (cumulative pass gating)\n\
         # median_ns = per-crop zenbench median; fit: total_ns = alpha + beta*pixels (least squares over all size x crop points)",
        dir.display()
    )
    .unwrap();
    writeln!(f, "subset\tside\tpixels\tmedian_ns_c0\tmedian_ns_c1\tmedian_ns_c2\tmedian_ns_c3\tmedian_ns_med\tns_per_px_med").unwrap();

    let mut fits: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for (name, _) in &subs {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for side in SIZES {
            let px = (side as f64) * (side as f64);
            let meds = &rows[&(name.to_string(), side)];
            let mut sorted = meds.clone();
            sorted.sort_by(f64::total_cmp);
            let med = (sorted[1] + sorted[2]) / 2.0; // median of 4
            let cols = meds
                .iter()
                .map(|v| format!("{v:.0}"))
                .collect::<Vec<_>>()
                .join("\t");
            writeln!(
                f,
                "{name}\t{side}\t{px:.0}\t{cols}\t{med:.0}\t{:.4}",
                med / px
            )
            .unwrap();
            for v in meds {
                xs.push(px);
                ys.push(*v);
            }
        }
        // least squares y = a + b x
        let n = xs.len() as f64;
        let sx: f64 = xs.iter().sum();
        let sy: f64 = ys.iter().sum();
        let sxx: f64 = xs.iter().map(|v| v * v).sum();
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
        let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let a = (sy - b * sx) / n;
        fits.insert(name.to_string(), (a, b));
    }
    writeln!(f, "#\n# fit total_ns = alpha + beta*pixels").unwrap();
    writeln!(
        f,
        "# subset\talpha_us\tbeta_ns_per_px\tms_at_1MP\tms_at_4MP\tus_at_64x64"
    )
    .unwrap();
    for (name, (a, b)) in &fits {
        writeln!(
            f,
            "# {name}\t{:.1}\t{:.4}\t{:.3}\t{:.3}\t{:.1}",
            a / 1e3,
            b,
            (a + b * 1_048_576.0) / 1e6,
            (a + b * 4_194_304.0) / 1e6,
            (a + b * 4096.0) / 1e3
        )
        .unwrap();
    }
    println!("\nwrote {out}");
    for (name, (a, b)) in &fits {
        println!(
            "{name:>10}: alpha = {:>8.1} us   beta = {:.4} ns/px   (1MP: {:.2} ms, 64x64: {:.1} us)",
            a / 1e3,
            b,
            (a + b * 1_048_576.0) / 1e6,
            (a + b * 4096.0) / 1e3
        );
    }
    let _ = result.save(format!(
        "/mnt/v/output/zenanalyze/per_tier_cost_grid_{date}.json"
    ));
}

/// yyyy-mm-dd without a chrono dep.
fn chrono_free_date() -> String {
    let out = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .expect("date");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}
