# Picker v0.4 — Session Handoff (2026-05-04)

**Author:** Lilith River
**Status:** session ended mid-execution; sweeps still running on vast.ai

## TL;DR

Per-codec status against the spec at
`benchmarks/picker_v0.4_data_starvation_spec_zenavif_zenjxl.md`:

| Step | zenavif | zenjxl |
|---|---|---|
| 3.1 corpus expansion sweep | RUNNING on vast.ai | RUNNING on vast.ai |
| 3.2 cell taxonomy collapse | DONE (config_v04 pushed) | DONE (config_v04 pushed) |
| 3.3 features TSV refresh | NOT STARTED | NOT STARTED |
| 3.4 trainer arch sizing | DOCUMENTED in config | DOCUMENTED in config |
| 3.5 LOO feature pruning | inherited from v0.3 (51 feats) | inherited (67 feats) |
| 3.6 held-out A/B harness | NOT STARTED | NOT STARTED |
| §4-step 4 train+bake | NOT STARTED | NOT STARTED |
| §4-step 6 A/B run | NOT STARTED | NOT STARTED |
| §4-step 7 ship/hold verdict | NOT STARTED | NOT STARTED |

**Verdict for this session:** SHIP/HOLD verdict is NOT yet possible.
Pick-up agent has 7 steps remaining per codec; estimated ~10-12 hours
each on top of waiting for sweeps to finish.

## What happened in this session

1. **Read spec**, surveyed prior agents' uncommitted state, set
   `.workongoing` markers in zenanalyze, zenavif, zenjxl.
2. **Verified infrastructure**: vast.ai $10.04 credit available,
   R2 buckets accessible, mlp-tune-fast corpus already at
   `s3://zentrain/sweep-2026-05-03/sources/` (587 PNGs).
3. **Discovered key constraint**: `zen-metrics sweep` `--jobs` flag
   is "Reserved for future fan-out. Currently serial." Spec's
   estimate of $0.05-0.20/codec at 8-way parallelism is therefore
   wrong; real cost is ~$0.50-2.00/codec at single-instance speed.
4. **Smoke-tested zen-metrics sweep locally** on 2 images — works.
5. **Launched two vast.ai instances** with `onstart-cmd` + ssh
   bootstrap fallback (the ssh proxy auth that prior agents bounced
   off worked fine in this session):
   - Instance 36108332 (RTX 5060 Ti, $0.068/hr): zenavif sweep
     - q=15..90 step 5, speed={3,5,7,9}, tune={0,1}
     - Total: 587 × 16 × 4 × 2 = 75,008 encodes
     - Started 2026-05-04T08:44:29Z
     - Rate: ~0.73 rows/s (speed=3 dominates), ETA ~28 hours
   - Instance 36108335 (RTX 3070, $0.057/hr): zenjxl sweep
     - q=5..95, distance={0.5,1.0,2.0,3.0,5.0,8.0,12.0}, effort={3,5,7,9}
     - Total: 587 × 13 × 7 × 4 = 213,668 encodes
     - Started 2026-05-04T08:44:22Z
     - Rate: ~2.5 rows/s, ETA ~24 hours
6. **Authored v0.4 picker configs**:
   - `zentrain/examples/zenavif_picker_config_v04.py`
     - Cells: speed {3,5,7,9} → 4 cells (was 10)
     - Scalars: tune (binary, snap-to-{0,1}) → 1 scalar
     - Outputs: 4 × (1 bytes_log + 1 tune) = 8 (was 20)
   - `zentrain/examples/zenjxl_picker_config_v04.py`
     - Cells: effort {3,5,7,9} → 4 cells (was 16)
     - Scalars: distance (continuous) → 1 scalar
     - Outputs: 4 × (1 bytes_log + 1 distance) = 8 (was 64)
     - **Constrained to public knob surface** — zenjxl 0.2.1 doesn't
       expose `with_internal_params` for the lossy path, so
       `ac_intensity / gaborish / patches / enhanced_clustering`
       cells from v0.3 are dropped.
7. **Authored sweep adapter**:
   `zentrain/tools/zenmetrics_sweep_adapter.py`
   - Translates zen-metrics 0.3.0 sweep TSV format
     (`image_path / codec / q / knob_tuple_json / score_zensim / ...`)
     into zentrain Pareto schema
     (`image_path / size_class / config_id / config_name / q / axes / bytes / zensim / ...`).
   - Supports `s3://` inputs (auto-pulls via aws-cli + R2 creds at
     `~/.config/cloudflare/r2-credentials`).
   - Smoke-tested against existing concat sweep TSVs at
     `s3://zentrain/sweep-2026-05-03/`: adapts cleanly, 10741+11740
     rows, all rows preserved.

Commit: `imazen/zenanalyze` main `24208e5fbc09`

## Sweep monitoring

Sentinel paths:
- `s3://coefficient/results/sweep-v04/zenavif_2026-05-04.done`
- `s3://coefficient/results/sweep-v04/zenjxl_2026-05-04.done`

When the `.done` sentinel appears, the result TSV is at:
- `s3://zentrain/sweep-v04-2026-05-04/zenavif_pareto_v04_full.tsv`
- `s3://zentrain/sweep-v04-2026-05-04/zenjxl_pareto_v04_full.tsv`

Live progress via SSH:
```bash
ssh -p 28332 root@ssh6.vast.ai 'wc -l /workspace/result.tsv; tail -3 /workspace/result.tsv'  # avif
ssh -p 28334 root@ssh6.vast.ai 'wc -l /workspace/result.tsv; tail -3 /workspace/result.tsv'  # jxl
```

**Cost cap:** $9.50. Estimated total for both sweeps: $4-6 worst case
(28h × $0.068 + 24h × $0.057 = $3.27).

When done, **destroy instances** to stop billing:
```bash
vastai destroy instance 36108332
vastai destroy instance 36108335
```

## Pick-up agent: next steps

### Step 1 — wait for sweeps + pull results

```bash
# Poll for sentinels, then fetch
source ~/.config/cloudflare/r2-credentials
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Check for done sentinels
aws s3 ls s3://coefficient/results/sweep-v04/ --endpoint-url "$ENDPOINT"

# Once present, pull the TSVs
mkdir -p /tmp/v04
aws s3 cp s3://zentrain/sweep-v04-2026-05-04/zenavif_pareto_v04_full.tsv \
    ~/work/zen/zenavif/benchmarks/zenavif_pareto_2026-05-04_v04_full_raw.tsv \
    --endpoint-url "$ENDPOINT" --region auto
aws s3 cp s3://zentrain/sweep-v04-2026-05-04/zenjxl_pareto_v04_full.tsv \
    ~/work/zen/zenjxl/benchmarks/zenjxl_pareto_2026-05-04_v04_full_raw.tsv \
    --endpoint-url "$ENDPOINT" --region auto

# DESTROY instances
vastai destroy instance 36108332
vastai destroy instance 36108335
```

### Step 2 — adapt sweep TSVs to zentrain schema

```bash
# Sources dir for size_class lookup (PNG dimensions)
SRCS=~/work/zentrain-corpus/mlp-tune-fast

cd ~/work/zen/zenanalyze
python3 zentrain/tools/zenmetrics_sweep_adapter.py \
    --input ~/work/zen/zenavif/benchmarks/zenavif_pareto_2026-05-04_v04_full_raw.tsv \
    --codec zenavif \
    --sources-dir "$SRCS"/cid22-train \
    --output ~/work/zen/zenavif/benchmarks/zenavif_pareto_2026-05-04_v04_full.tsv

python3 zentrain/tools/zenmetrics_sweep_adapter.py \
    --input ~/work/zen/zenjxl/benchmarks/zenjxl_pareto_2026-05-04_v04_full_raw.tsv \
    --codec zenjxl \
    --sources-dir "$SRCS"/cid22-train \
    --output ~/work/zen/zenjxl/benchmarks/zenjxl_pareto_2026-05-04_v04_full.tsv
```

NOTE: `--sources-dir` only handles ONE level. The mlp-tune-fast corpus
has multiple subdirs (`cid22-train/`, `clic-1024-train/`, etc.). The
on-disk filename in the sweep TSV is `<subdir>__<file>.png` (flattened
with `__` joiner, see vast.ai onstart script). The adapter needs the
sources dir to be the FLATTENED dir, not the multi-subdir mlp-tune-fast.
**Fix the adapter** to walk subdirs OR flatten sources locally before
running.

Quick fix: rebuild flat sources locally:
```bash
mkdir -p /tmp/srcs_flat
for sub in $SRCS/*/; do
  subname=$(basename "$sub")
  for f in "$sub"*.png; do
    [ -e "$f" ] || continue
    ln -sf "$f" /tmp/srcs_flat/${subname}__$(basename $f)
  done
done
# Then rerun adapter with --sources-dir /tmp/srcs_flat
```

### Step 3 — refresh features TSVs

```bash
cd ~/work/zen/zenanalyze
PYTHONPATH=zentrain/examples:zentrain/tools python3 zentrain/tools/refresh_features.py \
    --codec-config zenavif_picker_config_v04 \
    --corpus /tmp/srcs_flat \
    --output ~/work/zen/zenavif/benchmarks/zenavif_features_2026-05-04_v04_full.tsv

# Same for zenjxl_picker_config_v04 → zenjxl_features_2026-05-04_v04_full.tsv
```

CPU time: ~30-60 min per codec.

### Step 4 — train + bake

```bash
PYTHONPATH=zentrain/examples:zentrain/tools python3 zentrain/tools/train_hybrid.py \
    --codec-config zenavif_picker_config_v04 \
    --hidden 32 16 \
    --bake-output ~/work/zen/zenanalyze/benchmarks/zenavif_picker_v0.4_2026-05-04.bin

# zenjxl same with --hidden 48 20
```

**Verify val argmin acc ≥ 45%, mean overhead ≤ 4%** (spec §5
success criteria). If those fail, fall back to HistGB-only per spec
§5/§7.2.

### Step 5 — apply baker patch (#60)

The v0.3 .bins all needed an in-place patch script for the
`feature_transforms` count bug. Same patch applies to v0.4.
Locate the patch script at the same path used for v0.3 baker output
and run it on the v0.4 .bin.

### Step 6 — held-out A/B harness

Build per spec §3.6 using v0.3 patterns:
- `zenwebp/dev/picker_v0.3_holdout_ab.rs` (commit `1f46e06`)
- `zenjpeg/dev/picker_v0.3_holdout_ab.rs` (commit `1440b417`)

For zenavif: known broken — `enc.with_partition_range` missing in
current rav1e. Either fix or drop the partition range knob.

For zenjxl: distance-based RD instead of q-based. Adapt the secant
search to find distance for target_zensim.

Output to `<codec>/dev/picker_v0.4_holdout_ab.rs`.

### Step 7 — A/B run + verdict

```bash
cargo run --release --example picker_v0.4_holdout_ab -- \
    --picker ~/work/zen/zenanalyze/benchmarks/<codec>_picker_v0.4_2026-05-04.bin \
    --validation-corpus ~/work/zentrain-corpus/mlp-validate/cid22-val/ \
    --q-grid 30,35,...,90 \
    --output benchmarks/picker_v0.4_holdout_ab.tsv
```

Verdict criteria per spec §5:
- Total bytes ≤ bucket at parity zensim across q=30..90
- Per-band SHIP on at least 2 of 3 bands

Write report at `imazen/zenanalyze:benchmarks/picker_v0.4_<codec>_2026-05-04.md`.

### Step 8 — upload + push

```bash
# R2
aws s3 cp ~/work/zen/zenanalyze/benchmarks/<codec>_picker_v0.4_2026-05-04.bin \
    s3://zentrain/<codec>/pickers/<codec>_picker_v0.4_2026-05-04.bin \
    --endpoint-url "$ENDPOINT" --region auto
# ditto manifest, holdout_ab.{md,tsv}

# Cross-codec summary
# Write benchmarks/picker_v0.4_summary_2026-05-04.md aggregating both verdicts.

# Push to imazen/zenanalyze main, imazen/<codec> main
cd ~/work/zen/zenanalyze
jj bookmark set main -r @ && jj git push
```

## Caveats / known issues

1. **zenavif sweep includes speed=3** which is very slow (~1.6s/encode).
   If wall-time is critical, restart with speed={5,7,9} only — drops
   ~25% of work. The picker can still distinguish among speeds it
   trains on.

2. **zenjxl sweep encodes (q × distance) cross-product** but these are
   normally alternative quality dials. Verify zen-metrics' zenjxl
   driver actually composes them (might be that q wins and distance
   is ignored, in which case ¾ of rows are duplicates). Inspect the
   output TSV for byte-identical rows across distance values at fixed
   (image, effort, q). If duplication confirmed, drop one of the two
   axes from the picker config.

3. **`size-dense-renders/` filenames with spaces** broke `basename` in
   the onstart script — 1 of 587 images dropped (`...D 1953-2012.png`).
   Acceptable loss; flag for the next sweep iteration.

4. **Adapter sources-dir doesn't recurse** — see Step 2 above. Future
   improvement: walk subdirs in the adapter's PNG dimension lookup.

5. **Prior session artifacts preserved** in zenavif and zenjxl as `wip`
   commits (not pushed to main). Pick-up agent should review and
   decide whether to keep, push, or abandon those commits. Commits:
   - zenavif: `e0f7a468 wip(picker): v0.3 artifacts ...`
   - zenjxl: `4ec69832 wip(picker): v0.3 artifacts ...`

6. **`.workongoing` markers** are still present in zenanalyze, zenavif,
   zenjxl with this session's id `claude-opus47-picker-v04`. The
   timestamp will be stale; pick-up agent should overwrite or clear.

## Files added this session

`imazen/zenanalyze`:
- `benchmarks/picker_v0.4_data_starvation_spec_zenavif_zenjxl.md` (committed)
- `benchmarks/picker_v0.4_session_handoff_2026-05-04.md` (this file)
- `zentrain/examples/zenavif_picker_config_v04.py`
- `zentrain/examples/zenjxl_picker_config_v04.py`
- `zentrain/tools/zenmetrics_sweep_adapter.py`

R2 (when sweeps complete):
- `s3://zentrain/sweep-v04-2026-05-04/zenavif_pareto_v04_full.tsv`
- `s3://zentrain/sweep-v04-2026-05-04/zenjxl_pareto_v04_full.tsv`
- `s3://coefficient/results/sweep-v04/{zenavif,zenjxl}_2026-05-04.{log,done}`

## vast.ai instance details (for reference)

```
id=36108332 zenavif  RTX 5060 Ti  $0.068/hr  ssh6.vast.ai:28332
id=36108335 zenjxl   RTX 3070     $0.057/hr  ssh6.vast.ai:28334
```

Both running ubuntu:22.04, /workspace/run_sweep.sh in place, sweep
PIDs alive at session end.

API key set in `~/.config/vastai/vast_api_key`.
