# Zerobias argmin-agreement sweep — 2026-05-18

Synthetic-input argmin-agreement vs τ=0 baseline, N=2000 deterministic
feature vectors from `argmin_vs_tau` example. Measured on the v0.4
production bake candidates (see `benchmarks/picker_bakes_2026-05-18/`).

## What "argmin agreement" measures

Each picker outputs a vector of predicted log-bytes, one per cell.
**Argmin** = the cell index `i` minimizing that vector — "which
encoder config will produce the smallest file for this image".
**Agreement** = the fraction of inputs where two pickers pick the
same cell. NOT the same as `argmin_acc` (which compares against an
oracle), but a strict upper bound on argmin-acc degradation.

## Results

### Low-τ regime (the useful one for picker bakes)

All variants include `+compress +optimize`. The shipped v0.4 bakes
use **τ=0** for lossless argmin (100% agreement vs F32 trainer model).

| Codec   | τ=0 (shipped) | τ=0.0005 | τ=0.001 | τ=0.002 | τ=0.003 | τ=0.005 (earlier draft) | τ=0.01 |
|---------|--------------:|---------:|--------:|--------:|--------:|------------------------:|-------:|
| zenjpeg |    **100.00%** | 100.00% |  99.80% |  99.60% |  98.60% |                  95.10% | 64.50% |
| zenwebp |    **100.00%** | 100.00% |  99.90% |  99.70% |  99.45% |                  96.80% | 66.50% |
| zenavif |    **100.00%** |  99.90% |  99.40% |  97.70% |  98.35% |                  98.80% | 84.10% |

### High-τ regime (kept for posterity — not usable)

| Codec   | τ=0.02 | τ=0.05 | τ=0.1 |
|---------|-------:|-------:|------:|
| zenjpeg |  71.5% |  24.0% | 15.7% |
| zenwebp |  59.7% |  16.4% |  0.5% |
| zenavif |  73.4% |  71.6% | 30.1% |

### Bake size by τ (zenjpeg — most-affected codec)

| Knob set                            | bytes | Δ vs i8 baseline |
|-------------------------------------|------:|-----------------:|
| baseline (i8 only)                  | 55 980 |                0 |
| **τ=0 + cmp + opt (shipped)**       | 53 606 |  -2 374 (-4.2%) |
| τ=0.001 + cmp + opt                 | 53 545 |          -2 435 |
| τ=0.002 + cmp + opt                 | 53 462 |          -2 518 |
| τ=0.003 + cmp + opt                 | 53 328 |          -2 652 |
| τ=0.005 + cmp + opt (earlier draft) | 52 925 |          -3 055 |
| τ=0.01 + cmp + opt                  | 52 589 |          -3 391 |

Even τ=0.005's nominal extra -681 bytes vs τ=0 costs 5 pp of agreement
on zenjpeg. The trade is bad past τ=0 — almost all of the size win
comes from lz4 + permutation-optimize, not zerobias.

## Bake-size delta — shipped τ=0 + compress + optimize (final)

| Codec   | baseline B | shipped B | Δ     |
|---------|-----------:|----------:|------:|
| zenjpeg |     55 980 |    53 606 | -4.2% |
| zenwebp |     54 232 |    52 288 | -3.6% |
| zenavif |     52 036 |    50 845 | -2.3% |

## Verdict

**Ship τ=0 + compress + optimize.** Captures the 2.3-4.2% size shrink
with 100% argmin agreement vs the F32 trainer model. Anything past
τ=0.001 starts giving back picks for trivial extra bytes; τ ≥ 0.005
is plainly worse. The joint-pretrain weight distribution has a
narrower range than V_18 zensim's, so the zensim-calibrated τ=0.005
default zeros too aggressively here. Each picker shape should sweep
its own τ-vs-agreement curve before adopting a non-zero default.

## Reproduce

```bash
cargo build --release -p zenpredict-bake --example argmin_vs_tau

# Stage the BakeRequestJson intermediate (Python translator):
python3 tools/bake_picker.py \
  --model benchmarks/joint_finetune_2026-05-18/finetune_0xbabe_zenjpeg_picker.json \
  --out /tmp/stage.bin \
  --bake-json-out /tmp/zenjpeg_bake_request.json \
  --bake-bin ./target/release/zenpredict-bake --no-manifest

# Sweep tau:
./target/release/examples/argmin_vs_tau /tmp/zenjpeg_bake_request.json
```
