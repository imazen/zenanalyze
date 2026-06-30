//! End-to-end check of the baked cross-codec router: load the 3 real ZNPR `.bin`
//! (lossy / lossless / gate), build a zenanalyze-api `Offer` from one variant's qualified
//! features, and route at a few quality targets.
//!
//!   cargo run --release --features api --example route_demo -- \
//!       router_lossy.bin router_lossless.bin router_gate.bin demo_variant.tsv
//!
//! `demo_variant.tsv` is one `name@hex8<TAB>value` line per feature.
use std::fs;
use zenanalyze_api::{FeatureResult, NamedFeature, Offer, Provenance};
use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker, QualityTarget};
use zenpredict::{EncodeMode, Model};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: route_demo <lossy.bin> <lossless.bin> <gate.bin> <features.tsv>");
        std::process::exit(2);
    }
    let lossy = Aligned(fs::read(&a[1]).unwrap());
    let lossless = Aligned(fs::read(&a[2]).unwrap());
    let gate = Aligned(fs::read(&a[3]).unwrap());
    let lm = Model::from_bytes(&lossy.0).unwrap();
    let llm = Model::from_bytes(&lossless.0).unwrap();
    let gm = Model::from_bytes(&gate.0).unwrap();
    let mut r = MetaPicker::new(&lm).with_router(&gm, &llm);

    let txt = fs::read_to_string(&a[4]).unwrap();
    let cells: Vec<FeatureResult> = txt
        .lines()
        .filter_map(|l| {
            let mut p = l.split('\t');
            let name = p.next()?;
            let v: f32 = p.next()?.trim().parse().ok()?;
            Some(FeatureResult::new(NamedFeature::parse(name)?, v))
        })
        .collect();
    println!("offer: {} qualified features", cells.len());
    let offer = Offer::new(&cells, Provenance::new("demo"));
    let est = [0u32; CodecFamily::COUNT];

    for t in [
        QualityTarget::Zq(60.0),
        QualityTarget::Zq(85.0),
        QualityTarget::Zq(97.0),
        QualityTarget::Lossless,
    ] {
        match r
            .route(
                &offer,
                t,
                AllowedFamilies::all(),
                EncodeMode::QueuedBalanced,
                None,
                &est,
            )
            .unwrap()
        {
            Some(d) => println!(
                "  {:?} -> family={:?} lossless={} ranked={:?}",
                t,
                d.family(),
                d.lossless(),
                d.ranked()
            ),
            None => println!("  {:?} -> no viable family", t),
        }
    }
}
