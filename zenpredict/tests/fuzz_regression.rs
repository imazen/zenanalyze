//! Fuzz regression suite: replays every curated seed (`fuzz/seeds/*/`)
//! and every fixed crash (`fuzz/regression/`) through the exact exercise
//! code the graph fuzz targets run (`fuzz/fuzz_targets/graph_core.rs`,
//! pulled in with `include!`), as a plain `cargo test` — no nightly.
//! A panic here is a regression of a previously-fixed bug, or a seed the
//! runtime no longer handles.
//!
//! To add a crash: drop the minimized file into `fuzz/regression/`.

use std::fs;
use std::path::{Path, PathBuf};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/fuzz/fuzz_targets/graph_core.rs"
));

fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file() && p.extension().is_none_or(|x| x != "md"))
        .collect();
    out.sort();
    out
}

#[test]
fn graph_seeds_load_and_run() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/graph_from_bytes");
    let seeds = files_in(&dir);
    assert!(
        seeds.len() >= 4,
        "expected the curated v4 seeds in {}",
        dir.display()
    );
    for p in &seeds {
        let bytes = fs::read(p).unwrap();
        // Curated seeds are valid bakes: they must load, not just not panic.
        let model = zenpredict::Model::from_bytes(&bytes)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(model.version(), zenpredict::GRAPH_FORMAT_VERSION);
        // Differential, as the graph_structured target runs. The
        // reference follows this build's multiply-add rule (fused with
        // `std`, `a * b + c` without).
        exercise_model_bytes(&bytes, Some(cfg!(feature = "std")));
    }
}

#[test]
fn regression_inputs_do_not_panic() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/regression");
    for p in files_in(&dir) {
        run_model_bytes(&fs::read(&p).unwrap());
    }
}
