// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Print named zenanalyze features for image files, one TSV row per image:
//! `path\t<feature>...` with a header row. For offline Zq-head table builds
//! (zenwebp census wave 2026-08-27) and any harness that must consume the
//! SAME feature values the in-binary heads see.
//!
//!   cargo run --release --example print_features -- \
//!     grayscale_score,flat_color_block_ratio a.png b.png

use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet, FeatureValue};

fn feature_by_snake(name: &str) -> AnalysisFeature {
    AnalysisFeature::from_name(name)
        .unwrap_or_else(|| panic!("unknown feature `{name}` — see AnalysisFeature::from_name"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let names: Vec<String> = args
        .next()
        .expect("first arg: comma-separated feature snake names")
        .split(',')
        .map(str::to_string)
        .collect();
    let feats: Vec<AnalysisFeature> = names.iter().map(|n| feature_by_snake(n)).collect();
    let mut set = FeatureSet::just(feats[0]);
    for f in &feats[1..] {
        set = set.with(*f);
    }
    println!("path\t{}", names.join("\t"));
    for path in args {
        let img = image::open(&path).expect("decode").to_rgb8();
        let (w, h) = (img.width(), img.height());
        let a = zenanalyze::try_analyze_features_rgb8(img.as_raw(), w, h, &AnalysisQuery::new(set.clone()))
            .expect("analyze");
        let vals: Vec<String> = feats
            .iter()
            .map(|f| {
                if let Some(v) = a.get_f32(*f) {
                    return format!("{v}");
                }
                match a.get(*f) {
                    Some(FeatureValue::U32(x)) => format!("{x}"),
                    Some(FeatureValue::F32(x)) => format!("{x}"),
                    _ => "nan".into(),
                }
            })
            .collect();
        println!("{path}\t{}", vals.join("\t"));
    }
}
