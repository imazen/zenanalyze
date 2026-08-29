//! **Byte-exact feature-vector lock** — the gate every optimization has to pass.
//!
//! Feature values are training inputs. A baked picker's coefficients were fit
//! against the exact numbers this crate produced, so a 1-ULP change from a
//! "pure refactor" silently invalidates every shipped bake without failing a
//! single existing test: `versioning::golden_is_stable` compares at
//! `REL_TOLERANCE = 0.5 %`, which is ~2^20 ULP of headroom. This tool closes
//! that gap by comparing `f32::to_bits()`.
//!
//! It is deliberately **not** a `#[test]`. Byte-exactness across architectures
//! is not a property this crate has or wants — nine SIMD-reduced features are
//! documented to diverge per SIMD tier
//! (`docs/feature-cross-platform-divergence-2026-06-20.md`), which is exactly
//! why the committed golden is tolerance-based and blessed on x86-64. A
//! byte-exact table is a **per-host** artifact: you bless it before an
//! optimization and check it after, on the same machine. Wiring that into CI
//! would either fail on every non-blessing arch or need a runtime skip, and
//! runtime skips are banned. `just bitlock` / `just bitlock-bless` are the
//! caller-visible entry points.
//!
//! ```text
//! just bitlock-bless                 # write benchmarks/feature_bits_<arch>.tsv
//! …optimize…
//! just bitlock                       # exit 1 on the first bit that moved
//! ```
//!
//! ## Fixtures
//!
//! Real content first: `photo` / `photohard` / `screen` / `mixed` (see
//! `examples/common/mod.rs`) at 64² / 256² / 1024². Real crops do not, however,
//! reach every kernel — so four synthetic buffers cover the paths they miss:
//! the RGBA8 alpha pass and `RowStream::StripAlpha8`, the full palette scan
//! (≤ 256 distinct colours, which real photos never take), the strict-grayscale
//! early exit, and the degenerate all-one-colour statistics.
//!
//! Both `AnalysisQuery` configs are exercised per fixture — the default gamma
//! pass and the linear-light pass — because the linear tier has its own
//! kernels.

use std::collections::BTreeMap;
use std::path::PathBuf;

use zenanalyze::analyze_features;
use zenanalyze::feature::{AnalysisQuery, FeatureSet, FeatureValue};
use zenpixels::{PixelDescriptor, PixelSlice};

#[path = "common/mod.rs"]
mod common;
use common::{build, classes};

const SIDES: [u32; 3] = [64, 256, 1024];

/// A fixture: a name, the pixel bytes, dimensions, and how to describe them.
struct Fixture {
    name: String,
    buf: Vec<u8>,
    w: u32,
    h: u32,
    desc: PixelDescriptor,
}

impl Fixture {
    fn rgb8(name: String, buf: Vec<u8>, w: u32, h: u32) -> Self {
        Self {
            name,
            buf,
            w,
            h,
            desc: PixelDescriptor::RGB8_SRGB,
        }
    }
}

/// `w`×`h` RGBA8 with a varying alpha ramp — exercises the alpha pass and the
/// `StripAlpha8` RowStream path, which no RGB8 crop can reach.
fn synth_rgba8(w: u32, h: u32) -> Fixture {
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let a = (((x * 7 + y * 13) % 256) as u8).max(1);
            buf.extend_from_slice(&[
                ((x * 3) % 256) as u8,
                ((y * 5) % 256) as u8,
                ((x ^ y) % 256) as u8,
                a,
            ]);
        }
    }
    Fixture {
        name: format!("synth_rgba8_{w}x{h}"),
        buf,
        w,
        h,
        desc: PixelDescriptor::RGBA8_SRGB,
    }
}

/// ≤ 256 distinct colours: takes the FULL palette scan instead of the quick
/// early-exit one, and makes `PaletteFitsIn256` / `PaletteLog2Size` non-trivial.
fn synth_palette(w: u32, h: u32) -> Fixture {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = (((x / 4) + (y / 4) * 16) % 200) as u8;
            buf.extend_from_slice(&[i, i.wrapping_mul(3), 255 - i]);
        }
    }
    Fixture::rgb8(format!("synth_palette_{w}x{h}"), buf, w, h)
}

/// Strictly gray (R == G == B everywhere): the `scan_strict_grayscale` walk
/// runs to completion instead of exiting at the first coloured pixel.
fn synth_gray(w: u32, h: u32) -> Fixture {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = (((x * x + y * y) / 3) % 256) as u8;
            buf.extend_from_slice(&[v, v, v]);
        }
    }
    Fixture::rgb8(format!("synth_gray_{w}x{h}"), buf, w, h)
}

/// One colour everywhere — zero variance, zero edges: the degenerate branch of
/// every normalized statistic (divide-by-zero guards, Pearson degeneracy).
fn synth_solid(w: u32, h: u32) -> Fixture {
    Fixture::rgb8(
        format!("synth_solid_{w}x{h}"),
        vec![137u8; (w * h * 3) as usize],
        w,
        h,
    )
}

fn fixtures(corpus: &std::path::Path) -> Vec<Fixture> {
    let mut out = Vec::new();
    for class in classes(corpus) {
        for &side in &SIDES {
            let (buf, _tiles) = build(side, &class, 0);
            out.push(Fixture::rgb8(
                format!("{}_{side}", class.name),
                buf,
                side,
                side,
            ));
        }
    }
    out.push(synth_rgba8(129, 97));
    out.push(synth_palette(200, 200));
    out.push(synth_gray(200, 200));
    out.push(synth_solid(64, 64));
    out
}

/// Canonical, lossless text for one value. `f32` goes out as its raw bit
/// pattern — that is the whole point — with the decimal beside it so a diff is
/// readable. `NaN` keeps its exact payload (it is a real value here: peak
/// luminance on an SDR source).
fn encode(v: FeatureValue) -> String {
    match v {
        FeatureValue::F32(x) => format!("f32:0x{:08x}\t{x:e}", x.to_bits()),
        FeatureValue::U32(n) => format!("u32:{n}\t{n}"),
        FeatureValue::U64(n) => format!("u64:{n}\t{n}"),
        FeatureValue::Bool(b) => format!("bool:{}\t{b}", u8::from(b)),
        other => format!("other:{other:?}\t{other:?}"),
    }
}

/// `fixture/config/feature -> encoded value`, in a stable order.
fn extract_all(fixtures: &[Fixture]) -> BTreeMap<String, String> {
    let supported = FeatureSet::SUPPORTED;
    let mut rows = BTreeMap::new();
    for fx in fixtures {
        let stride = fx.w as usize * fx.desc.bytes_per_pixel();
        let slice = PixelSlice::new(&fx.buf, fx.w, fx.h, stride, fx.desc)
            .unwrap_or_else(|e| panic!("{}: {e:?}", fx.name));
        for linear in [false, true] {
            let q = AnalysisQuery::new(supported).with_linear_light(linear);
            let r = analyze_features(slice.clone(), &q)
                .unwrap_or_else(|e| panic!("{} linear={linear}: {e}", fx.name));
            let cfg = if linear { "linear" } else { "gamma" };
            for f in supported.iter() {
                let Some(v) = r.get(f) else { continue };
                rows.insert(
                    format!("{}\t{cfg}\t{:03}\tfeat_{}", fx.name, f.id(), f.name()),
                    encode(v),
                );
            }
        }
    }
    rows
}

fn render(rows: &BTreeMap<String, String>) -> String {
    let mut s = String::new();
    for (k, v) in rows {
        s.push_str(k);
        s.push('\t');
        s.push_str(v);
        s.push('\n');
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let corpus = PathBuf::from(
        std::env::var("ZENANALYZE_CORPUS_DIR").unwrap_or_else(|_| "../codec-corpus".to_string()),
    );
    let path = std::env::var("BITLOCK_OUT")
        .unwrap_or_else(|_| format!("benchmarks/feature_bits_{}.tsv", std::env::consts::ARCH));

    let fx = fixtures(&corpus);
    let rows = extract_all(&fx);
    let header = format!(
        "# zenanalyze byte-exact feature lock — arch={} features={} fixtures={} rows={}\n\
         # Per-HOST artifact: SIMD-reduced features diverge per tier across arches by design.\n\
         # columns: fixture<TAB>config<TAB>id<TAB>feature<TAB>typed_bits<TAB>decimal(informational)\n",
        std::env::consts::ARCH,
        FeatureSet::SUPPORTED.len(),
        fx.len(),
        rows.len(),
    );

    if args.iter().any(|a| a == "--bless") {
        std::fs::write(&path, format!("{header}{}", render(&rows)))
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
        eprintln!(
            "blessed {} rows over {} fixtures -> {path}",
            rows.len(),
            fx.len()
        );
        return;
    }

    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}: {e} — bless it first with `just bitlock-bless`"));
    let mut expected = BTreeMap::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(cols.len() >= 5, "malformed row: {line}");
        expected.insert(cols[..4].join("\t"), cols[4..].join("\t"));
    }

    let mut bad = 0usize;
    for (k, v) in &rows {
        match expected.get(k) {
            Some(e) if e == v => {}
            Some(e) => {
                if bad < 25 {
                    println!("MOVED  {k}\n  was {e}\n  now {v}");
                }
                bad += 1;
            }
            None => {
                if bad < 25 {
                    println!("NEW    {k}\t{v}");
                }
                bad += 1;
            }
        }
    }
    for k in expected.keys() {
        if !rows.contains_key(k) {
            if bad < 25 {
                println!("GONE   {k}");
            }
            bad += 1;
        }
    }
    if bad == 0 {
        eprintln!(
            "bitlock OK: {} values byte-identical over {} fixtures ({})",
            rows.len(),
            fx.len(),
            path
        );
    } else {
        eprintln!(
            "bitlock FAILED: {bad} of {} values differ (showing at most 25). \
             If the change was meant to alter values, that is a feature-definition \
             change: bump the definition, re-bless the versioning golden, and \
             retrain every affected bake — do NOT re-bless this lock to make it pass.",
            rows.len()
        );
        std::process::exit(1);
    }
}
