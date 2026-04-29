# zenpicker training pipeline

Codec-agnostic training scripts. Each codec writes a small **codec
config module** (see [`examples/zenjpeg_picker_config.py`](../examples/zenjpeg_picker_config.py))
declaring its TSV paths, feature schema, zq target grid, and
config-name parser. Then runs these scripts against that config.

## Files

| File | Purpose |
|---|---|
| `_picker_lib.py` | Shared library: disk-cached dataset construction, parallel teacher training (joblib), HISTGB_FAST/FULL presets, subsample helper, StepTimer |
| `train_hybrid.py` | **Recommended.** Hybrid-heads training: N categorical cells + K scalar prediction heads per cell. Codec supplies the cell taxonomy via `parse_config_name` |
| `train_distill.py` | Categorical-only distillation (legacy v1.x picker shape). Per-config HistGB teacher → small shared MLP student |
| `train_distill_reduced.py` | Same as `train_distill.py` but with a reduced feature subset declared in the codec config |
| `feature_ablation.py` | Single-feature LOO ablation across `KEEP_FEATURES` |
| `feature_group_ablation.py` | Group ablation (drop entire tier-aligned groups) |
| `validate_schema.py` | Re-train production HistGB with a chosen feature subset, report held-out metrics |
| `capacity_sweep.py` | Architecture × cross-term-recipe sweep over the student MLP |

The [`tools/bake_picker.py`](../../tools/bake_picker.py) script (parent
directory, not this one) consumes the JSON output of any of the
training scripts and emits the v1 binary blob + manifest.

## Codec config contract

A codec config module exports:

```python
from pathlib import Path

PARETO        = Path(...)          # Pareto sweep TSV
FEATURES      = Path(...)          # Analyzer features TSV
OUT_JSON      = Path(...)          # Trained model JSON output
OUT_LOG       = Path(...)          # Training summary text output
KEEP_FEATURES = [...]              # list[str] of `feat_*` columns
ZQ_TARGETS    = [...]              # list[int] of target_zq values

def parse_config_name(name: str) -> dict:
    """Parse a config_name into categorical + scalar axes."""
    ...
```

`parse_config_name` returns a dict where keys partition into
**categorical axes** (hashable values; combined to form cells) and
**scalar axes** (floats; per-cell prediction targets, with a
sentinel value for cells where the axis is N/A).

For the zenjpeg reference, see [`examples/zenjpeg_picker_config.py`](../examples/zenjpeg_picker_config.py).

## End-to-end

```bash
# 1. Codec runs its Pareto sweep harness, producing TSVs.
cd zenjpeg-checkout
cargo run --release -p zenjpeg --features 'target-zq trellis' \
    --example zq_pareto_calibrate

# 2. Train the hybrid-heads picker against zenjpeg's config taxonomy.
PYTHONPATH=zenanalyze/zenpicker/examples:zenanalyze/zenpicker/tools \
    python3 zenanalyze/zenpicker/tools/train_hybrid.py \
        --codec-config zenjpeg_picker_config

# 3. Bake to v1 binary.
python3 zenanalyze/tools/bake_picker.py \
    --model benchmarks/zq_bytes_hybrid_2026-04-29.json \
    --out   models/zenjpeg_picker_v2.0_hybrid.bin \
    --dtype f16

# 4. Verify round-trip.
python3 zenanalyze/tools/bake_roundtrip_check.py \
    --model benchmarks/zq_bytes_hybrid_2026-04-29.json \
    --dtype f16
```

End-to-end retrain on a 16-core box, with the dataset cache warm:
~3 minutes wall-clock (was ~25 minutes before the parallel-teachers
refactor).

## Iteration vs production

`_picker_lib.py` exports two HistGB hyperparameter presets:

| Preset | `max_iter` | `max_depth` | When |
|---|---:|---:|---|
| `HISTGB_FAST` | 100 | 4 | Architecture sweeps, ablation runs, anything you'll re-run |
| `HISTGB_FULL` | 400 | 8 | Production bake — only the final pre-bake training run |

The picker ablation work confirmed feature-importance ranking is
stable between FAST and FULL; absolute mean overhead drops by ~1pp
when switching FAST→FULL. Use FAST for everything except the final
bake.

## Adding a new codec

1. Copy `examples/zenjpeg_picker_config.py` to `examples/<your_codec>_picker_config.py`.
2. Edit paths, KEEP_FEATURES, ZQ_TARGETS, and `parse_config_name` for your codec's config name pattern.
3. Add `examples/` to `PYTHONPATH` and run the training scripts as above.

The runtime crate (zenpicker) doesn't change. Your codec's encoder
crate ships the produced `.bin` via `include_bytes!` and a
compile-time `CELLS` table matching the manifest. See
[FOR_NEW_CODECS.md](../FOR_NEW_CODECS.md) for the codec-side wiring
walkthrough.
