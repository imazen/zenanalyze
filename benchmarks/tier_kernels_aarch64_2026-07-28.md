# Per-pass NEON isolation — 2026-07-28

Platform: Apple Silicon (aarch64, NEON), darwin 25.5.0
Bench: `benches/tier_kernels.rs` (zenbench, interleaved arms), 1024×1024 noise+patches input
Method: archmage `NeonToken::dangerously_disable_token_process_wide` toggles the tier between
arms in one process. Built without `-C target-cpu=native` so the tier stays toggleable.

## Why this exists

`benches/tier_isolation.rs` runs `analyze_features_rgb8` with the full `SUPPORTED` set — a
single aggregate number. That cannot show one analysis pass whose kernels are *slower* than
their own scalar fallback, because the faster passes average it away. That exact failure mode
was found and fixed in garb, zensim, zentone, zenpng and zenresize during the same aarch64
sweep, so zenanalyze's passes were checked individually rather than inferred from the
end-to-end ratio.

Requesting one pass's `FeatureSet` at a time gates the others off (per the tier architecture
in CLAUDE.md), so each measurement is dominated by that pass's kernels.

## Result: no losers

| pass | NEON | scalar | ratio |
|---|---|---|---|
| tier1_full | 4.9 ms | 7.3 ms | 1.49× |
| tier1_extras | 5.1 ms | 8.6 ms | 1.69× |
| tier2_chroma | 4.3 ms | 6.2 ms | 1.44× |
| tier3_dct | 5.2 ms | 6.7 ms | 1.29× |
| palette | 4.7 ms | 5.9 ms | 1.26× |
| depth | 3.8 ms | 5.0 ms | 1.32× |

## Reading these numbers

Each pass still carries the shared per-call cost (RowStream setup, row conversion), so these
ratios are **diluted** — they are lower bounds on the kernels' own speedups, not measurements
of the kernels in isolation. They are sufficient for the question asked (is any pass
NEON-negative?) and not sufficient for tuning an individual kernel.

On aarch64 NEON is *baseline*, so the "scalar" arm is the magetypes scalar tier **with LLVM
autovectorization**, not unvectorized code. A ratio near 1.00 would therefore not indicate a
missing kernel; it would indicate both arms compiled to equivalent work. Nothing here is near
1.00, so every pass's explicit SIMD is earning its keep over what the autovectorizer manages
on its own.
