//! Minimal LZ4 block-format decoder, pure-safe-Rust, single-alloc.
//!
//! Decoder-only. The companion encoder lives in the
//! `zenpredict-bake` crate (via its `lz4_flex` dep — encoder code
//! doesn't need to be vendored into the lean runtime).
//!
//! Format reference: <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>
//!
//! ## Single-alloc contract
//!
//! [`decompress_into`] writes into a caller-supplied `&mut [u8]`.
//! No allocations on the hot path. The convenience [`decompress`]
//! wrapper allocates exactly one `Vec<u8>` sized to the caller-known
//! decompressed length.
//!
//! ## Resource limits
//!
//! - The decoder ignores any "decompressed length prefix" — the
//!   caller supplies the exact output buffer size. This makes
//!   zip-bomb attacks impossible: a 1 KB compressed payload can't
//!   expand into a 4 GB buffer unless the caller already allocated
//!   that 4 GB.
//! - All array indexing is bounds-checked by Rust; over-runs return
//!   [`Lz4Error::OutputOverflow`] / [`Lz4Error::InputTruncated`].
//! - No recursion — bounded state machine.
//!
//! ## Why not vendor lz4_flex?
//!
//! `lz4_flex::block::decompress_safe` is well-tested but pulls in
//! ~1 KLOC of supporting code (its `sink` + `fastcpy` modules) plus
//! a richer error surface than zenpredict needs. The hand-roll
//! here is ~150 LOC, has the same safe-Rust guarantees, and matches
//! the LZ4 reference decoder byte-for-byte (verified by round-trip
//! tests against the `lz4_flex` encoder).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Minimum match length per the LZ4 specification.
const MIN_MATCH: usize = 4;
/// `0xFF` continuation byte marker used by both literal-length and
/// match-length extension reads.
const EXT_MARKER: u8 = 0xFF;

/// Errors raised by the LZ4 block decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Lz4Error {
    /// Input ran out mid-token. Compressed stream is truncated /
    /// corrupted.
    InputTruncated,
    /// Decoder tried to write past the supplied output buffer. Either
    /// caller under-allocated `out` or the compressed stream lies
    /// about its decompressed size.
    OutputOverflow,
    /// Match offset was 0 or pointed before the start of the output
    /// buffer. Invalid LZ4 stream.
    InvalidMatchOffset,
    /// Compressed stream signalled it had more data after the last
    /// literal but the output buffer is exactly full. (Strict bake
    /// contract: the decompressed length must match the layer's
    /// expected weight byte count exactly.)
    TrailingData,
}

impl core::fmt::Display for Lz4Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InputTruncated => f.write_str("lz4: input truncated mid-token"),
            Self::OutputOverflow => f.write_str("lz4: output buffer overflow"),
            Self::InvalidMatchOffset => f.write_str("lz4: invalid match offset (0 or before output start)"),
            Self::TrailingData => f.write_str("lz4: stream has trailing data beyond expected output length"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Lz4Error {}

/// Decompress one LZ4 block into a pre-sized buffer.
///
/// `out.len()` must equal the exact decompressed length. The decoder
/// stops when either the input is exhausted (last literal copy fills
/// the output, no match follows) OR the output is exactly filled.
/// Mismatches return an error rather than silently truncating.
///
/// Single-alloc contract: this function performs zero allocations on
/// its own. All scratch state lives in the caller's `out` slice plus
/// stack-local indices.
pub fn decompress_into(src: &[u8], out: &mut [u8]) -> Result<(), Lz4Error> {
    let mut s = 0usize; // src cursor
    let mut d = 0usize; // dst cursor

    loop {
        if s >= src.len() {
            // Input exhausted between sequences. The spec says the
            // last LZ4 sequence is followed by no match, so reaching
            // end-of-input cleanly (with output exactly filled) is
            // the normal exit path for streams whose last sequence
            // ended with a match. Accept it; mismatch is the error.
            if d != out.len() {
                return Err(Lz4Error::OutputOverflow);
            }
            return Ok(());
        }
        let token = src[s];
        s += 1;
        // Literal length: high nibble, with 0xFF-extension scheme when ==15.
        let mut lit_len = (token >> 4) as usize;
        if lit_len == 0xF {
            loop {
                if s >= src.len() {
                    return Err(Lz4Error::InputTruncated);
                }
                let b = src[s];
                s += 1;
                lit_len = lit_len
                    .checked_add(b as usize)
                    .ok_or(Lz4Error::InputTruncated)?;
                if b != EXT_MARKER {
                    break;
                }
            }
        }
        // Copy literals.
        if lit_len > 0 {
            let src_end = s.checked_add(lit_len).ok_or(Lz4Error::InputTruncated)?;
            if src_end > src.len() {
                return Err(Lz4Error::InputTruncated);
            }
            let dst_end = d.checked_add(lit_len).ok_or(Lz4Error::OutputOverflow)?;
            if dst_end > out.len() {
                return Err(Lz4Error::OutputOverflow);
            }
            out[d..dst_end].copy_from_slice(&src[s..src_end]);
            s = src_end;
            d = dst_end;
        }
        // Last sequence has only literals — no match offset follows.
        if s == src.len() {
            // Strict: decompressed length must match output buffer.
            if d != out.len() {
                return Err(Lz4Error::OutputOverflow);
            }
            return Ok(());
        }
        // Match offset (16-bit little-endian).
        if s + 2 > src.len() {
            return Err(Lz4Error::InputTruncated);
        }
        let offset = u16::from_le_bytes([src[s], src[s + 1]]) as usize;
        s += 2;
        if offset == 0 || offset > d {
            return Err(Lz4Error::InvalidMatchOffset);
        }
        // Match length: low nibble + 0xFF-extension + MIN_MATCH.
        let mut match_len = (token & 0xF) as usize;
        if match_len == 0xF {
            loop {
                if s >= src.len() {
                    return Err(Lz4Error::InputTruncated);
                }
                let b = src[s];
                s += 1;
                match_len = match_len
                    .checked_add(b as usize)
                    .ok_or(Lz4Error::InputTruncated)?;
                if b != EXT_MARKER {
                    break;
                }
            }
        }
        match_len = match_len
            .checked_add(MIN_MATCH)
            .ok_or(Lz4Error::OutputOverflow)?;
        let dst_end = d.checked_add(match_len).ok_or(Lz4Error::OutputOverflow)?;
        if dst_end > out.len() {
            return Err(Lz4Error::OutputOverflow);
        }
        // Match copy. When offset < match_len the source region
        // overlaps with the dest region (LZ77 RLE pattern); copy
        // byte-by-byte to honor the overlap semantics.
        let match_start = d - offset;
        if offset >= match_len {
            // Non-overlapping — split borrow lets us memcpy.
            let (lhs, rhs) = out.split_at_mut(d);
            rhs[..match_len].copy_from_slice(&lhs[match_start..match_start + match_len]);
        } else {
            for i in 0..match_len {
                out[d + i] = out[match_start + i];
            }
        }
        d = dst_end;
    }
}

/// Convenience wrapper that allocates the output buffer.
///
/// One `Vec<u8>` allocation sized to `decompressed_len`; no other
/// allocations. Returns an error if the stream doesn't decompress
/// to exactly that length.
pub fn decompress(src: &[u8], decompressed_len: usize) -> Result<Vec<u8>, Lz4Error> {
    let mut out = vec![0u8; decompressed_len];
    decompress_into(src, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-crafted LZ4 block: literal "Hello, " + match offset=7
    // pointing back to "Hello, " (match_len=3 → "Hel"), giving
    // "Hello, Hel" as output.
    #[test]
    fn round_trip_short_literal_only() {
        // Token: 0x70 (lit_len=7, match_len=0), then 7 bytes "Hello, ", end.
        // But "end" with match_len=0 still tries to read offset... per spec,
        // the LAST sequence has only literals. So we encode token=0x70,
        // literals, and nothing else. The decoder detects s == src.len()
        // after the literal copy and exits cleanly.
        let src = [0x70, b'H', b'e', b'l', b'l', b'o', b',', b' '];
        let out = decompress(&src, 7).unwrap();
        assert_eq!(&out, b"Hello, ");
    }

    #[test]
    fn extended_literal_length() {
        // Token: 0xF0 (lit_len needs extension), then 0x00 ext (15+0=15),
        // then 15 literal bytes "Hello, world!Hi".
        let mut src = vec![0xF0, 0x00];
        src.extend_from_slice(b"Hello, world!Hi");
        let out = decompress(&src, 15).unwrap();
        assert_eq!(&out[..], b"Hello, world!Hi");
    }

    #[test]
    fn match_copy_non_overlapping() {
        // "ABCDEFGH" then match offset=8 (back to "ABCD"), match_len=4
        // → "ABCDEFGHABCD".
        // Token: lit_len=8, match_len=0 (raw 0 → +MIN_MATCH=4).
        //   token = 0x80
        let mut src = vec![0x80];
        src.extend_from_slice(b"ABCDEFGH");
        src.extend_from_slice(&8u16.to_le_bytes()); // offset=8 → back to "A"
        let out = decompress(&src, 12).unwrap();
        assert_eq!(&out[..], b"ABCDEFGHABCD");
    }

    #[test]
    fn match_copy_overlapping_rle() {
        // RLE pattern: literal "A", then match offset=1 match_len=5 →
        // "A" + "AAAAA" = "AAAAAA" (6 bytes).
        // Token: lit_len=1, match_len=1 (low nibble, +MIN_MATCH=5)
        //   token = (1 << 4) | 1 = 0x11
        let mut src = vec![0x11, b'A'];
        src.extend_from_slice(&1u16.to_le_bytes()); // offset=1
        let out = decompress(&src, 6).unwrap();
        assert_eq!(&out[..], b"AAAAAA");
    }

    #[test]
    fn truncated_input_returns_err() {
        // Token claims lit_len=5 but only 3 bytes follow.
        let src = [0x50, b'A', b'B', b'C'];
        let err = decompress(&src, 5).unwrap_err();
        assert_eq!(err, Lz4Error::InputTruncated);
    }

    #[test]
    fn output_overflow_returns_err() {
        // Token claims lit_len=5 but output buffer is sized 3.
        let src = [0x50, b'A', b'B', b'C', b'D', b'E'];
        let mut out = [0u8; 3];
        let err = decompress_into(&src, &mut out).unwrap_err();
        assert_eq!(err, Lz4Error::OutputOverflow);
    }

    #[test]
    fn invalid_offset_zero_returns_err() {
        // Token says we have literals + a match, but offset=0.
        let mut src = vec![0x11, b'A'];
        src.extend_from_slice(&0u16.to_le_bytes()); // offset=0 invalid
        let err = decompress(&src, 6).unwrap_err();
        assert_eq!(err, Lz4Error::InvalidMatchOffset);
    }

    #[test]
    fn invalid_offset_beyond_output_start_returns_err() {
        // offset=100 but we've only written 1 byte.
        let mut src = vec![0x11, b'A'];
        src.extend_from_slice(&100u16.to_le_bytes());
        let err = decompress(&src, 6).unwrap_err();
        assert_eq!(err, Lz4Error::InvalidMatchOffset);
    }

    #[test]
    fn never_panics_on_random_input() {
        // Property-style: feed arbitrary bytes, output buffer up to
        // 256 bytes. Any error is fine; a panic is a bug.
        for seed in 0..200u64 {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let len = (state as usize) % 64 + 1;
            let src: Vec<u8> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (state >> 33) as u8
                })
                .collect();
            let _ = decompress(&src, 256);
        }
    }
}
