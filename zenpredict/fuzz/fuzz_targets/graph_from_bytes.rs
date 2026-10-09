//! Fuzz target: raw bytes through the ZNPR v4 graph path. Seeded from
//! `fuzz/seeds/graph_from_bytes/` (real v4 bakes), so byte mutations land
//! in the node table, node sections and graph validation rather than
//! dying at the header. Load → node views → `Predictor::try_new` →
//! `predict` must never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

include!("graph_core.rs");

fuzz_target!(|bytes: &[u8]| {
    run_model_bytes(bytes);
});
