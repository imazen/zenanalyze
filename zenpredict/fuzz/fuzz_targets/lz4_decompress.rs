//! Fuzz target: `lz4_block::decompress_into` must never panic.
//!
//! Splits the fuzz input into a `decompressed_len: u16` prefix +
//! the rest as the compressed payload. The output buffer size is
//! capped at 64 KiB (matches the practical V_X weight layer size +
//! gives the decoder plenty of room without exploding fuzz memory).
//!
//! Any panic in the decoder — arithmetic overflow, slice OOB,
//! infinite loop on adversarial offsets, copy-overlap edge cases —
//! is a fuzz hit. Returns `Result`s are fine; the contract is
//! "no panic on any byte string."

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    // First two bytes pick the output buffer size (mod 64 KiB).
    let out_len = u16::from_le_bytes([data[0], data[1]]) as usize;
    let mut out = vec![0u8; out_len.min(64 * 1024)];
    let _ = zenpredict::lz4_block::decompress_into(&data[2..], &mut out);
});
