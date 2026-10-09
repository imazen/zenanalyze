Minimized crash inputs from the zenpredict fuzz targets, after the fix landed.
`tests/fuzz_regression.rs` replays every file here (and every curated seed under
`../seeds/`) through the same code the fuzz targets run, on stable, in
`cargo test`. Name files `crash-<sha1>`; keep each under 8 KB.
