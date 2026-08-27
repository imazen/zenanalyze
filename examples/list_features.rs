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
//! ```

use zenanalyze::feature::FeatureSet;

fn main() {
    let with_ids = std::env::args().skip(1).any(|a| a == "--ids");
    for f in FeatureSet::SUPPORTED.iter() {
        if with_ids {
            println!("{}\tfeat_{}", f.id(), f.name());
        } else {
            println!("feat_{}", f.name());
        }
    }
}
