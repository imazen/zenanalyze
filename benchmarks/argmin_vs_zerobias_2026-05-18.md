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

| Codec   | τ=0.005 (shipped) | τ=0.01 | τ=0.02 | τ=0.05 | τ=0.1 |
|---------|------------------:|-------:|-------:|-------:|------:|
| zenjpeg |             95.1% |  64.5% |  71.5% |  24.0% | 15.7% |
| zenwebp |             96.8% |  66.5% |  59.7% |  16.4% |  0.5% |
| zenavif |             98.8% |  84.1% |  73.4% |  71.6% | 30.1% |

## Bake-size delta at the shipped τ=0.005 + compress + optimize

| Codec   | baseline B | shipped B | Δ      |
|---------|-----------:|----------:|-------:|
| zenjpeg |     55 980 |    52 925 | -5.5%  |
| zenwebp |     54 232 |    52 144 | -3.9%  |
| zenavif |     52 036 |    50 703 | -2.6%  |

(τ alone doesn't shrink the .bin — lz4 realizes the zero-density gain.
Tested separately: bare `compressed=true` shrinks 4-5% on optimized
weights but expands on raw i8 noise per CHANGELOG; the combination is
what wins.)

## Verdict

τ=0.005 is the right shipped value. Higher τ shrinks bytes only
marginally more (≤2 KB) but breaks argmin agreement dramatically.
The joint-pretrain weight distribution doesn't tolerate aggressive
zerobias as well as V_18 zensim did (which was calibrated against
this same τ=0.005 for 87.5% zero density).

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
