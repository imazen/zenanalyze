# Picker bakes 2026-05-18 — v0.4

Production-candidate .bin pickers from the 5-seed joint-pretrain run.
Median-seed-by-val-argmin picks; **zerobias_tau=0 (lossless) + lz4 +
permutation optimize** applied — preserves 100% argmin agreement vs the
F32 trainer model.

| Codec   | File | Bytes | Source seed | Methodology | Val argmin |
|---------|---|--:|---|---|--:|
| zenjpeg | `zenjpeg_picker_v0.4_2026-05-18.bin` | 53 606 | 0xBABE finetune | joint pretrain + 80-epoch head-finetune | 54.50% |
| zenwebp | `zenwebp_picker_v0.4_2026-05-18.bin` | 52 288 | 0xBABE finetune | joint pretrain + 80-epoch head-finetune | 51.18% |
| zenavif | `zenavif_picker_v0.4_2026-05-18.bin` | 50 845 | 0xDEAD joint-only | joint pretrain (finetune lost on this codec) | 31.04% |

## Why τ=0 (no zerobias)

Earlier draft of this bake set used τ=0.005 (the V_18-zensim-calibrated
default). The argmin-vs-τ sweep at `benchmarks/argmin_vs_zerobias_2026-05-18.md`
showed it broke argmin agreement on 1.2-4.9% of inputs for only
~200-650 bytes extra shrink. Bad trade. τ=0 keeps argmin agreement at
100% and still captures the lz4+optimize gains.

## Size optimization stack — zenjpeg

| Knob set                              | bytes | Δ vs baseline | argmin agree |
|---------------------------------------|------:|--------------:|-------------:|
| baseline (i8 only)                    | 55 980 |              0 |        100% |
| **τ=0 + cmp + opt (shipped)**         | **53 606** | **-2 374 (-4.2%)** | **100%** |
| τ=0.001 + cmp + opt                   | 53 545 |        -2 435 |        99.8% |
| τ=0.002 + cmp + opt                   | 53 462 |        -2 518 |        99.6% |
| τ=0.005 + cmp + opt (earlier draft)   | 52 925 |        -3 055 |        95.1% |
| τ=0.01 + cmp + opt                    | 52 589 |        -3 391 |        64.5% |

The marginal savings past τ=0 are <600 bytes total even at the
agreement-cratering τ=0.005. lz4 + permutation-optimize do all the
real work. The joint-pretrain weight distribution doesn't tolerate
zerobias well — V_18 zensim's distribution (which the τ=0.005 default
was calibrated against) had a much wider range and could absorb the
zeroing without large output shifts.

## Caveat: union-input layout, no `extra_axes` in metadata

The trunk is joint-trained on a union-feature input layout:

```
[union_feat_values  (U=65 floats)]
[presence_mask      (U=65 bools)]
[size_onehot        (4 bools)]
[log_pixels         (1 float)]
[zq_norm            (1 float)]
[codec_onehot       (C=3 bools)]
                       = 139 inputs total
```

Caller must construct this vector manually per `docs/external-caller-pattern.md`.
The bake's `extra_axes` block was emitted with anonymous `aux_*`
placeholders because `bake_picker.py` doesn't yet recognize the
union-input layout — codec wrapper code must hard-code the slot
positions (see external-caller-pattern.md pseudocode).

## How to use

```bash
# Inspect a bake:
./target/release/zenpredict-inspect <file>.bin

# Re-bake with different flags / seed / config:
python3 tools/bake_picker.py \
  --model benchmarks/joint_finetune_2026-05-18/finetune_<SEED>_<codec>_picker.json \
  --out <output>.bin \
  --zerobias-tau 0 --compress --optimize \
  --bake-bin ./target/release/zenpredict-bake \
  --no-manifest
```
