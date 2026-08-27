//! `docs/dispatch-axes.md` must name every dispatch knob of
//! `analyze_specialized_raw` (zenanalyze#50 Sub-C/G documentation
//! deliverable). Adding a const-bool axis or a runtime gate to the
//! specialization site without documenting it fails here, and so does
//! dropping a knob from the doc.

use std::fs;
use std::path::Path;

fn read(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The parameter names and const generics of `fn analyze_specialized_raw`
/// in `src/lib.rs`, parsed from the source text.
fn dispatch_knobs() -> (Vec<String>, Vec<String>) {
    let src = read("src/lib.rs");
    let start = src
        .find("fn analyze_specialized_raw<")
        .expect("src/lib.rs defines fn analyze_specialized_raw<...>");
    let sig_end = src[start..]
        .find(") ->")
        .expect("analyze_specialized_raw signature ends with `) ->`");
    let sig = &src[start..start + sig_end];

    let generics_start = sig.find('<').unwrap();
    let generics_end = sig.find('>').unwrap();
    let consts: Vec<String> = sig[generics_start + 1..generics_end]
        .split(',')
        .filter_map(|g| {
            let g = g.trim();
            g.strip_prefix("const ")
                .map(|rest| rest.split(':').next().unwrap().trim().to_string())
        })
        .collect();

    let params_start = sig.find('(').unwrap();
    let params: Vec<String> = sig[params_start + 1..]
        .split(',')
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let name = p.split(':').next().unwrap().trim();
            if name == "slice" {
                None // the input, not a knob
            } else {
                Some(name.to_string())
            }
        })
        .collect();
    (consts, params)
}

#[test]
fn signature_parse_is_sane() {
    let (consts, params) = dispatch_knobs();
    assert_eq!(
        consts,
        ["PAL", "T2", "T3", "ALPHA"],
        "const-bool axes changed — update this test and the doc"
    );
    assert!(
        params.len() >= 10,
        "expected the runtime knobs, got {params:?}"
    );
    for expected in [
        "pixel_budget",
        "hf_max_blocks",
        "run_dct",
        "run_strict_gray",
        "run_linear_light",
    ] {
        assert!(
            params.iter().any(|p| p == expected),
            "missing {expected} in {params:?}"
        );
    }
}

#[test]
fn every_dispatch_knob_is_documented() {
    let doc = read("docs/dispatch-axes.md");
    let (consts, params) = dispatch_knobs();
    let mut missing = Vec::new();
    for c in &consts {
        if !doc.contains(&format!("`{c}`")) {
            missing.push(c.clone());
        }
    }
    for p in &params {
        if !doc.contains(&format!("`{p}`")) {
            missing.push(p.clone());
        }
    }
    assert!(
        missing.is_empty(),
        "docs/dispatch-axes.md does not name these analyze_specialized_raw knobs (as `code`): {missing:?}"
    );
}

#[test]
fn documented_gating_sets_exist_in_feature_rs() {
    // The doc names the FeatureSet constants that decide each axis; they must
    // still exist under those names in src/feature.rs.
    let doc = read("docs/dispatch-axes.md");
    let feature_rs = read("src/feature.rs");
    for name in [
        "PAL_NEEDED_BY",
        "TIER2_FEATURES",
        "T3_NEEDED_BY",
        "ALPHA_FEATURES",
        "PALETTE_FULL_FEATURES",
        "TIER1_FULL_FEATURES",
        "TIER1_SKIN_FEATURES",
        "DEPTH_FEATURES",
        "DCT_NEEDED_BY",
        "DEFAULT_PIXEL_BUDGET",
        "DEFAULT_HF_MAX_BLOCKS",
    ] {
        assert!(
            doc.contains(&format!("`{name}`")),
            "doc no longer mentions `{name}`"
        );
        assert!(
            feature_rs.contains(&format!("const {name}:")),
            "src/feature.rs no longer defines `{name}` — the doc is stale"
        );
    }
}
