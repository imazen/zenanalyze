//! Print every feature this build can compute, one per line, as the
//! `feat_<name>` column names training data and bakes use.
//!
//! This is the "universe" input for `tools/feature_inventory.py`
//! (`--universe`), which reports which features no shipped bake consumes.
//! Build with the cargo features the inventory should cover (the crate's
//! `experimental` is on by default; add `hdr` for the depth tier):
//!
//! ```text
//! cargo run --release --example list_features --features hdr > /path/universe.txt
//! cargo run --release --example list_features -- --ids      # `id<TAB>feat_name`
//! cargo run --release --example list_features -- --variants # `id<TAB>feat_name<TAB>Variant`
//! ```
//!
//! `--variants` adds the `AnalysisFeature` variant identifier, which the
//! inventory uses to render the per-family `FeatureSet` preset proposals as
//! compilable Rust.

use zenanalyze::feature::FeatureSet;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let with_variants = args.iter().any(|a| a == "--variants");
    let with_ids = with_variants || args.iter().any(|a| a == "--ids");
    for f in FeatureSet::SUPPORTED.iter() {
        if with_variants {
            println!("{}\tfeat_{}\t{f:?}", f.id(), f.name());
        } else if with_ids {
            println!("{}\tfeat_{}", f.id(), f.name());
        } else {
            println!("feat_{}", f.name());
        }
    }
}
