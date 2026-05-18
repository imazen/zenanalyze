# Joint-trunk pretraining beats per-codec methodology — measured SOTA (2026-05-17)

## TL;DR

A **single shared-trunk MLP trained jointly on combined codec data**
beats every per-codec methodology measured this session. The
restored `train_multi_codec.py` (commit `849caba`, deleted in the
v3.2 revert and resurrected for this experiment) outputs per-codec
JSON bakes that consume a UNION feature schema. 5-seed measurement:

| Codec | Per-codec best | Joint median | Δ |
|---|--:|--:|--:|
| zenjpeg | 30% | **52.7%** | **+22 pp** |
| zenwebp | **45.3%** (v14 ship) | 43.7% | −1.6 pp |
| zenavif | 20.2% (ultraprune) | **31.0%** | **+10.8 pp** |

Joint training transfers signal from data-rich codecs to data-poor
ones. zenwebp (the data-richest) doesn't benefit. zenjpeg and zenavif
both see massive lifts.

## Per-seed details

5-seed PyTorch shared-trunk training (hidden=192,96), 100 epochs each.

| Seed | zenjpeg | zenwebp | zenavif |
|---|--:|--:|--:|
| `0xCAFE` | 56.7% | 48.9% | 24.0% |
| `0xBEEF` | 52.7% | 54.0% | 31.9% |
| `0xFACE` | 56.0% | 43.7% | 29.3% |
| `0xDEAD` | 47.5% | 38.3% | 31.0% |
| `0xBABE` | 50.6% | 38.9% | 32.8% |
| **median** | **52.7%** | **43.7%** | **31.0%** |
| stdev (sample) | 3.7 pp | 7.0 pp | 3.5 pp |

zenavif's stdev tightens dramatically from 7.30 (ultraprune) to 3.5
(joint) — joint training gives zenavif a more robust picker too.

zenwebp's stdev widens (5.41 unpruned → 7.0 joint), and median drops
slightly. The joint trunk's "average codec" signal slightly dilutes
zenwebp's task-specific signal. Stay on per-codec for zenwebp.

## Why it works

The picker's input is the same set of 102 zenanalyze features for
every codec. The OUTPUT differs per codec (different config grids,
different cells). Per-codec training learns image-feature → codec-
configs in isolation. Joint training learns image-feature → embedding
that's USEFUL FOR ALL CODECS, then a per-codec head maps embedding →
that codec's configs.

The trunk sees:
- zenjpeg: 29,185 decision rows
- zenwebp: 13,360 decision rows
- zenavif: 5,205 decision rows
- combined: 47,750 decision rows

zenavif's 5,205 rows + the trunk's awareness of patterns from the
other 42,545 rows = better generalization than zenavif's 5,205 alone.

## Why zenwebp doesn't benefit

zenwebp has 13,360 rows / 72 configs / 6 cells = 31 rows/config on
average. Per-codec training already saturates the achievable
information density. Adding other codecs' data brings task-irrelevant
patterns that slightly dilute zenwebp's task-specific embedding.

zenjpeg has 29,185 rows but 120 configs and 12 cells = 20 rows/cell
on average. Sparser than zenwebp per-cell — joint training helps.
zenavif: 5,205 rows / 200 configs / 10 cells = 2.6 rows/config —
extremely sparse, joint training helps massively.

## Wire format consideration

The joint trainer outputs per-codec JSONs where the input is the
UNION schema: `[union_feat_values (65), presence_mask (65),
size_onehot (4), log_px, zq_norm, codec_onehot (3)] = 139 inputs`.

Each codec's runtime needs to assemble this union vector before
calling `Predictor::predict()`. That was what the deleted
v3.2 multi_codec_schema section did automatically. Without v3.2,
the caller (external orchestrator OR the codec's encoder)
needs to:
1. Compute the codec's 102 natural features
2. Map them to the union schema slots (per the joint trainer's
   `union_slot_for_codec_feat` array, which lives in the bake's
   metadata)
3. Set presence mask, size onehot, etc.
4. Feed to Predictor::predict

This is the "external-caller pattern" from the deleted
`docs/multi-codec-external-caller-pattern.md`. It's tractable.

## Recommendation

1. **Re-enable joint-trunk pretraining** as the primary methodology
   for the data-poor codecs (zenjpeg, zenavif).
2. **Keep zenwebp on per-codec methodology** — it's already at its
   sweet spot.
3. **DO NOT re-add the v3.2 multi_codec_schema runtime section.**
   The external-caller pattern is sufficient. Each codec's bake is
   still a normal v3 bin; the only requirement is that its runtime
   knows how to build the union input vector.
4. **Restore the external-caller pattern docs** (deleted in the
   revert) — they describe the runtime contract that consumers need.

## Production path

For zenjpeg + zenavif:
1. Joint-train (`train_multi_codec.py`) → per-codec JSONs (UNION schema)
2. Bake each JSON with `bake_picker.py` → per-codec .bin (UNION input shape)
3. Each codec's encoder builds the union vector before predict
4. Multi-seed verdict confirms shipability

For zenwebp:
1. Stay on per-codec `zenwebp_picker_config` v14+z_rmse (already ships +24.54 pp)

## Files

- `zentrain/tools/train_multi_codec.py` (restored from commit 849caba)
- `benchmarks/joint_pretrain_2026-05-17/jointpre_*.json` — seed 0xCAFE baseline
- `benchmarks/joint_pretrain_multiseed_2026-05-17/joint_*.json` — 4 more seeds + summaries
- `benchmarks/joint_pretrain_breakthrough_2026-05-17.md` — this doc

## Next steps (queued, not done in this session)

- Restore `docs/multi-codec-external-caller-pattern.md` documenting
  the per-codec runtime contract
- Restore `distill_multi_to_per_codec.py` for comparison — joint
  training vs distillation are different techniques
- Try fine-tuning per-codec heads from the joint pretrain (the
  approach the user originally suggested: pretrain, then per-codec
  fine-tune layer-K weights)
- Try different trunk depths (96-only vs 192-96 vs 256-128-64)
- Try joint training with zenjxl included (4 codecs instead of 3)
