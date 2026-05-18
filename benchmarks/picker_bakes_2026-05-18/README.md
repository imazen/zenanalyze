# Picker bakes 2026-05-18 — v0.4

Production-candidate .bin pickers from the 5-seed joint-pretrain run.
Median-seed-by-val-argmin picks; zerobias_tau=0.005 + lz4 + permutation
optimize applied.

| Codec   | File | Bytes | Source seed | Methodology | Val argmin |
|---------|---|--:|---|---|--:|
| zenjpeg | `zenjpeg_picker_v0.4_2026-05-18.bin` | 52 939 | 0xBABE finetune | joint pretrain + 80-epoch head-finetune | 54.50% |
| zenwebp | `zenwebp_picker_v0.4_2026-05-18.bin` | 52 158 | 0xBABE finetune | joint pretrain + 80-epoch head-finetune | 51.18% |
| zenavif | `zenavif_picker_v0.4_2026-05-18.bin` | 50 836 | 0xDEAD joint-only | joint pretrain (finetune lost on this codec) | 31.04% |

## Size optimization stack

Per-codec shrink from baseline (i8, no flags):

| Knob set                              | bytes | Δ vs baseline |
|---------------------------------------|------:|--------------:|
| baseline (i8)                         | 55 984 |              0 |
| +zerobias τ=0.005                     | 55 980 |          ~0   |
| +zerobias τ=0.02                      | 55 980 |          ~0   |
| +compress (lz4)                       | 53 616 |        -2 368 |
| +optimize (permutation hillclimb)     | 53 612 |        -2 372 |
| +zerobias 0.005 +compress             | 53 149 |        -2 835 |
| **+zerobias 0.005 +compress +optimize (shipped)** | **52 929** | **-3 055 (-5.5%)** |
| +zerobias 0.02 +compress +optimize    | 51 568 |        -4 416 |
| (f16 baseline, for reference)         | 103 260 |        +47 276 |

Zerobias alone doesn't shrink because lz4 is what realizes the
zero-density gain. The joint-pretrain weight distribution has a
narrower range than the V_18 zensim bake that τ=0.005 was calibrated
against, so zerobias alone removes only ~6% of weights at the
canonical τ.

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
  --zerobias-tau 0.005 --compress --optimize \
  --bake-bin ./target/release/zenpredict-bake \
  --no-manifest
```
