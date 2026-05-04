# Picker v0.4 — Address zenavif + zenjxl Data Starvation

**Audience:** another team or agent picking up this work cross-session.
**Author:** Lilith River
**Status:** spec — ready to execute
**Background:** v0.3 zenwebp + zenjpeg pickers shipped at −4.60% / −9.65% bytes Δ vs bucket. zenavif + zenjxl pickers were trained but val metrics show data starvation. This spec lays out a multi-pronged fix.

---

## 1. Problem statement

| Codec | val argmin acc | mean overhead | train→val gap | Diagnosis |
|---|---:|---:|---:|---|
| zenwebp v0.3 | 58.7% | 1.39% | +0.13pp | shipped (−4.60%) |
| zenjpeg v0.3 | 59.0% | 2.17% | (parity) | shipped (−9.65%) |
| **zenavif v0.3** | **23.3%** | **5.60%** | **+2.08pp** | **OVERFIT, data-starved** |
| **zenjxl v0.3** | **11.2%** | **4.57%** | **+0.81pp** | **HEAVY data-starve (~2× uniform-prior baseline)** |

zenavif: 200 source imgs × 89.6k Pareto rows × 10 cells × 20 outputs (rav1e_phase1a sweep, narrow grid).
zenjxl: 100 source imgs × 610k Pareto rows × 16 cells × 64 outputs (synthesized oracle, no rich knob coverage).

Held-out A/B was not run for either; the val numbers above predict HOLD.

---

## 2. Root causes (multi-factor)

1. **Corpus too small.** zenavif 200, zenjxl 100 vs zenwebp 514 / zenjpeg 348.
2. **Cell taxonomy too rich for the data.** zenjxl 16 cells × 4 scalars/cell = 64 outputs at 100 imgs ≈ 1.6 imgs per output dim. zenwebp shipped at 24 outputs / 514 imgs = 21.4× more data per output.
3. **Sparse cell occupancy.** Per-cell row counts are low — many cells effectively starved.
4. **Knob grid mismatch.** The original sweeps for these codecs were authored for different purposes and don't cover the cell taxonomy evenly.
5. **Trainer architecture sized for zenwebp's data shape.** 100→128² student MLP (~37k params) is overkill for ~1k effective training rows after the Pareto+train split.

These compound. Fixes need to attack multiple sides simultaneously.

---

## 3. Multi-pronged fix plan

### 3.1 Corpus expansion (HIGHEST LEVERAGE)

Targets:
- zenavif: 200 → ≥ 600 imgs (3× growth)
- zenjxl: 100 → ≥ 600 imgs (6× growth)

Source: `s3://zentrain/sources/` mirror of `~/work/zentrain-corpus/mlp-tune-fast/` (587 imgs, behaviorally clustered, deduped). Already uploaded. License-clean.

Implementation: run `zen-metrics sweep` (already shipped in `zen-metrics-v0.3.0`) per codec on the full 587-image corpus.

```bash
# zenavif
~/work/turbo-metrics/target/release/zen-metrics sweep \
    --codec zenavif \
    --sources ~/work/zentrain-corpus/mlp-tune-fast/ \
    --q-grid 15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90 \
    --knob-grid '{"speed":[3,5,7,9],"tune":[0,1]}' \
    --metrics zensim,ssim2 \
    --output ~/work/zen/zenavif/benchmarks/zenavif_pareto_2026-05-XX_v04_full.tsv \
    --jobs 8

# zenjxl
~/work/turbo-metrics/target/release/zen-metrics sweep \
    --codec zenjxl \
    --sources ~/work/zentrain-corpus/mlp-tune-fast/ \
    --q-grid 5,10,15,20,25,30,40,50,60,70,80,90,95 \
    --knob-grid '{"distance":[0.5,1.0,2.0,3.0,5.0,8.0,12.0],"effort":[3,5,7,9],"gaborish":[0,1]}' \
    --metrics zensim,ssim2 \
    --output ~/work/zen/zenjxl/benchmarks/zenjxl_pareto_2026-05-XX_v04_full.tsv \
    --jobs 8
```

Wall time on 7950X (per local v0.3 sweep): ~3 hr per codec at 8-way parallel.

### 3.2 Cell taxonomy collapse

Re-think each codec's cell vs scalar split with the constraint of total outputs ≤ ~25 (matching zenwebp v0.3's 24).

**zenavif** (current 10 cells × 2 scalars = 20 outputs):
- Cells: `(speed)` only — 4 cells `{3, 5, 7, 9}`
- Scalars: `q`, `qm`, `vaq_strength` — 3 outputs/cell
- Total: 4 cells × (1 bytes_log + 3 scalars) = **16 outputs**
- Rationale: speed is the dominant categorical axis; tune_still is binary and rarely Pareto-discriminating.

**zenjxl** (current 16 × 4 = 64 outputs):
- Cells: `(effort, ac_intensity)` only — 4 cells `{(3,compact), (5,compact), (5,full), (7,full)}` (drop unused combos)
- Scalars: `distance`, `gaborish`, `patches`, `enhanced_clustering` — 4 outputs/cell
- Total: 4 cells × (1 bytes_log + 4 scalars) = **20 outputs**
- Rationale: gaborish/patches/enhanced_clustering are binary toggles; better as scalar predictions (then snap-to-{0,1}) than cell axes.

Both end up at zenwebp-comparable output dim count, and at 600 imgs the data ratio improves dramatically.

### 3.3 Per-codec features TSV refresh

Required because the current features TSVs cover ~43% of mlp-tune-fast.

```bash
cd ~/work/zen/zenanalyze
PYTHONPATH=zentrain/examples:zentrain/tools python3 zentrain/tools/refresh_features.py \
    --codec-config <codec>_picker_config \
    --corpus ~/work/zentrain-corpus/mlp-tune-fast/ \
    --output ~/work/zen/<codec>/benchmarks/<codec>_pareto_features_2026-05-XX_full.tsv
```

Per-codec ~30-60 min CPU.

### 3.4 Trainer architecture sizing

Apply Hsu et al. 10× params rule:
- Target: ≥ 10 train rows per learnable parameter
- For ~600 imgs × ~80 q×knob configs = ~48k rows, at 80/20 split → ~38k train rows
- Max params: 3,800 (very tight) to 5,000 (moderate)

Recommended student arch per codec:
- zenavif: `100→32→16` ≈ 4k params → matches data
- zenjxl: `100→48→20` ≈ 6.5k params → near limit

If still overfit at this arch:
- Drop MLP distillation entirely; ship the HistGB teacher directly via a tree-based picker format. **This is the right escape valve for genuine data-starve.**
- Or: ship at the architecture sized for the actual data, accept lower expressivity.

### 3.5 Multi-seed LOO feature pruning per-codec

The post-LOO `KEEP_FEATURES` lists should be applied:
- zenavif: 67 → 52 features (per #43 LOO consensus, already documented in handoff)
- zenjxl: 67 → 64 features

This both reduces overfit risk and removes engineered axes that don't carry codec-specific signal.

### 3.6 Held-out A/B harness per codec

Use the zenwebp + zenjpeg patterns at:
- `~/work/zen/zenwebp/dev/picker_v0.3_holdout_ab.rs` (commit `1f46e06`)
- `~/work/zen/zenjpeg/dev/picker_v0.3_holdout_ab.rs` (commit `1440b417`)

Plus the template doc at `imazen/zenanalyze:benchmarks/holdout_ab_harness_template.md`.

For zenavif:
- Build harness using `zenavif::EncodeRequest` with `__expert`-feature-gated knobs
- Bucket-table baseline: existing `Preset::Auto` at target_zensim
- Targets: q-grid 30..90 step 5

For zenjxl:
- Use jxl-encoder's local `__expert` features
- Distance-based RD instead of q-based — adapt the binary search to find distance that hits target_zensim

---

## 4. Implementation order (sequential within codec, parallel across codecs)

Per codec:
1. **Refresh features** on full mlp-tune-fast (~30-60 min CPU)
2. **Re-sweep** with proper grid (~3 hr CPU on 7950X with `--jobs 8`)
3. **Update picker config** — collapsed cell taxonomy, post-LOO KEEP_FEATURES, smaller student arch
4. **Train + bake** (~10-30 min)
5. **Build held-out A/B harness** in `<codec>/dev/picker_v0.4_holdout_ab.rs` (~3-4 hr Rust work; template-driven)
6. **Run A/B** (~15-30 min wall encode)
7. **Verdict + ship/hold**

Total per codec: ~10-12 hr. Both can run on the same workstation if sequential, or split across two sessions.

---

## 5. Success criteria

- val argmin acc ≥ 45% (vs zenwebp 58.7%, zenjpeg 59.0%)
- val mean overhead ≤ 4% (vs zenwebp 1.39%, zenjpeg 2.17%)
- Held-out A/B: total bytes ≤ bucket at parity zensim, across q=30..90 (full range)
- per-band SHIP verdict on at least 2 of 3 bands

If criteria aren't met after this fix:
- HistGB-only picker (no MLP) as fallback — ship the ensemble directly
- Or accept the codec as "no picker" until corpus grows further

---

## 6. Pre-existing infrastructure (already shipped, ready to use)

- `zen-metrics-v0.3.0` released, 7-platform binaries on github.com/imazen/turbo-metrics/releases
- `zen-metrics sweep` subcommand in `~/work/turbo-metrics/target/release/zen-metrics`
- ZNPR v3 format with `OutputSpec`/`sentinel`/`sparse_overrides`/`feature_transforms`
- Per-head loss normalization in `zentrain/tools/train_hybrid.py` (commit `7e2534a`)
- Semantic FEATURE_TRANSFORMS + OUTPUT_SPECS in codec configs (commit `1e0f365`)
- Multi-seed LOO results: `~/work/zen/zenanalyze/benchmarks/loo_<codec>_multiseed_2026-05-03.tsv`
- Three-bucket R2 layout (`coefficient` / `zentrain` / `codec-corpus`) with credentials at `~/.config/cloudflare/r2-credentials`
- mlp-tune-fast corpus at `~/work/zentrain-corpus/mlp-tune-fast/` (587 imgs, R2-uploaded as `s3://zentrain/sources/` or `s3://zentrain/sweep-2026-05-03/sources/` — verify path)
- Held-out validation set: `~/work/zentrain-corpus/mlp-validate/cid22-val/` (41 imgs)

---

## 7. Open questions / decision points

1. **Cell taxonomy collapse**: how aggressive? More cells = better fit at high data, fewer = better at low data. Recommended starting point above; tune empirically.
2. **HistGB-only fallback**: when do we give up on MLP? Recommended threshold: if val argmin still <30% after the corpus expansion, ship HistGB ensemble directly (smaller .bin, no distillation). zenpredict would need a tree format addition (~1 day work).
3. **Knob grid**: do the recommended grids (above) cover the codecs' Pareto-optimal regions? Verify by checking the existing rav1e_phase1a + zenjxl_lossy_pareto for which knob combinations land on the Pareto frontier most often. Drop combinations that never appear.
4. **Cloud vs local**: USE vast.ai. The previous agents flagged an SSH proxy auth bug, but it's resolvable (see §11 vast.ai usage). The user explicitly authorized vast.ai spend up to $9.50; current credit $10.04 untouched.
5. **zenjxl with_internal_params**: zenjxl 0.2.1 published doesn't expose `with_internal_params` for the lossy path — knob coverage is constrained to public knobs. Either (a) accept the constraint and tune within public surface, (b) extend zenjxl's lossy path locally (path-dep), (c) wait for next zenjxl release. Option (a) is fastest.
6. **zenavif `tune_still`**: the existing rav1e_phase1a sweep used `tune_still=1` (still-image mode) uniformly. If cell taxonomy includes `tune_still` as an axis, need to actually vary it. Otherwise drop from the cell taxonomy.

---

## 8. Pre-known caveats

1. **Baker `feature_transforms` count bug** (#60) — currently a manual in-place patch script, applied to every v0.3 .bin so far. Real fix tracked separately. Apply the same patch to v0.4 .bins until baker is fixed.
2. **zenwebp picker compile broken at HEAD** — runtime.rs references retired `AnalysisFeature::NaturalLikelihood` and `ScreenContentLikelihood`. Doesn't block this work (codecs stay dumb, picker is external), but worth noting if anyone tries to build the in-codec path.
3. **zenavif auto-tune compile broken at HEAD** — `enc.with_partition_range` doesn't exist in current rav1e. Same — doesn't block external picker.
4. **vast.ai SSH proxy bug** — known issue from prior session, blocks vast.ai instance bootstrap. If cloud is needed, work around manually or skip vast.ai.

---

## 9. Time budget

~30-40 hr total across both codecs. Split per-codec or session as convenient. Recommended split:

- **Session A** (zenavif): refresh features + re-sweep + train + bake + harness + A/B = ~12-15 hr
- **Session B** (zenjxl): same shape = ~15-18 hr (longer because of distance-based RD harness adaptation)

Or one focused agent with 30-40 hr budget can do both sequentially.

---

## 10. End-of-work artifacts

Per codec, the work is "done" when these all exist:

- `s3://zentrain/<codec>/pickers/<codec>_picker_v0.4_<date>.bin` (sized to data per §3.4)
- `s3://zentrain/<codec>/pickers/<codec>_picker_v0.4_<date>.manifest.json`
- `s3://zentrain/<codec>/pickers/picker_v0.4_holdout_ab.{md,tsv}` (full q=30..90 verdict)
- `imazen/zenanalyze:benchmarks/picker_v0.4_<codec>_<date>.md` (training + A/B report)
- `imazen/<codec>:dev/picker_v0.4_holdout_ab.rs` (harness)
- Updated cross-codec summary at `imazen/zenanalyze:benchmarks/picker_v0.4_summary_<date>.md`
- All commits on `imazen/zenanalyze` and `imazen/<codec>` main branches per jj-on-main convention

If the verdict is HOLD, ship the report anyway with a clear "what's the next blocker" — that's still production-ready signal.

---

## 11. vast.ai usage (REQUIRED, not optional)

The user explicitly directed: **use vast.ai for data fill-in**. Prior agents bounced off an SSH proxy bug. Resolutions, in order of preference:

### 11.1 Approach A — bypass SSH entirely with onstart-cmd (RECOMMENDED)

Vast.ai instances accept an `--onstart-cmd` that runs as root immediately on boot. No SSH required. Workflow:

```bash
# 1. Search for cheap GPU box (CUDA-capable; we use it for GPU metrics)
vastai search offers \
    'rentable=true cpu_ram>=16 cpu_cores>=8 dph_total<0.06 \
     gpu_name in [RTX_3060,RTX_3060_Ti,RTX_3070,RTX_4060]' \
    --order dph_total -o dph_total --raw

# 2. Build the onstart script that does EVERYTHING (no SSH needed)
cat > /tmp/vastai_sweep_onstart.sh <<'EOF'
#!/bin/bash
set -e
exec > /workspace/sweep.log 2>&1
echo "starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Install minimal deps
apt-get update && apt-get install -y curl awscli

# Pull pre-built zen-metrics binary from GitHub release
cd /workspace
curl -L -o zen-metrics.tar.gz \
    "https://github.com/imazen/turbo-metrics/releases/download/zen-metrics-v0.3.0/zen-metrics-x86_64-unknown-linux-gnu.tar.gz"
tar xzf zen-metrics.tar.gz
chmod +x zen-metrics
mv zen-metrics /usr/local/bin/

# Set R2 creds (passed via env from --env on instance create)
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Pull source corpus from R2
mkdir -p sources
aws s3 sync "s3://zentrain/sweep-2026-05-03/sources/" sources/ \
    --endpoint-url "$ENDPOINT" --region auto

# Run sweep (codec + grid passed as env vars)
zen-metrics sweep \
    --codec "$CODEC" \
    --sources sources/ \
    --q-grid "$Q_GRID" \
    --knob-grid "$KNOB_GRID" \
    --metrics zensim,ssim2 \
    --output result.tsv \
    --jobs 8

# Push result to R2
aws s3 cp result.tsv \
    "s3://zentrain/sweep-2026-05-XX/${CODEC}_pareto_v04.tsv" \
    --endpoint-url "$ENDPOINT" --region auto
aws s3 cp /workspace/sweep.log \
    "s3://coefficient/results/sweep-v04/${CODEC}.log" \
    --endpoint-url "$ENDPOINT" --region auto

# Self-destruct: leaves a sentinel file, dispatcher polls and destroys
echo "DONE_$(date -u +%Y%m%dT%H%M%SZ)" > /workspace/done.sentinel
aws s3 cp /workspace/done.sentinel \
    "s3://coefficient/results/sweep-v04/${CODEC}.done" \
    --endpoint-url "$ENDPOINT" --region auto
EOF

# 3. Create instance with onstart + env
vastai create instance <offer-id> \
    --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
    --disk 64 \
    --onstart-cmd "$(cat /tmp/vastai_sweep_onstart.sh)" \
    --env "-e CODEC=zenavif -e Q_GRID=15,20,...,90 -e KNOB_GRID='{\"speed\":[3,5,7,9],\"tune\":[0,1]}' \
           -e R2_ACCOUNT_ID=$R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY" \
    --label sweep-zenavif-v04
```

The dispatcher polls `s3://coefficient/results/sweep-v04/<codec>.done` for the sentinel file. Once present, fetch results from R2 and `vastai destroy instance <id>`. **No SSH required.** This bypasses the proxy bug entirely.

### 11.2 Approach B — SSH proxy workaround (if onstart-cmd insufficient)

The prior agent flagged: "vast.ai SSH proxy never authenticated despite both global ssh-key registration and per-instance attach for both RSA + ED25519 keys (inner sshd's `Accepted key … Connection closed by authenticating user` pattern, a known vast.ai PAM/proxy issue)."

Workarounds:
- Use `vastai ssh-url <id>` to get the proxy URL, but run commands via `vastai execute <id> "command..."` instead of interactive ssh
- The `execute` subcommand bypasses the proxy SSH path entirely
- Or: use `vastai logs <id>` for read-only inspection while `onstart-cmd` runs

### 11.3 Approach C — rent CPU-only boxes (fallback)

If GPU instances have provisioning issues, vast.ai also offers CPU-only boxes at <$0.02/hr. Same tooling works (zen-metrics CLI runs CPU metrics fine). Slower but reliable.

### 11.4 Per-codec budget

| Codec | Imgs | Configs/img | Encode-ms/cfg | Wall (8-way) | Cost @ $0.05/hr |
|---|---|---|---|---|---|
| zenavif | 587 | 256 (16q × 4 speed × 2 tune × 2 align) | 200 | 1.0 hr | **$0.05** |
| zenjxl | 587 | 364 (13q × 7 dist × 4 effort) | 300 | 2.2 hr | **$0.11** |

Total for both: **~$0.20**. Far under the $9.50 hard cap.

### 11.5 Hard rules for vast.ai use
- **DO NOT halt on SSH issues** — use onstart-cmd approach (§11.1) which doesn't require SSH at all
- **DO destroy instances** after work completes; `vastai show instances-v1` should show 0 when done
- **DO NOT exceed $9.50** total spend; budget per-codec is $0.05-0.20 so this should never be close
- **DO push results to R2** before self-destruct (sentinel file pattern in §11.1)
- **DO check the GitHub release** for the actual binary asset name — the brief assumes `zen-metrics-x86_64-unknown-linux-gnu.tar.gz` but verify via `gh release view zen-metrics-v0.3.0 --repo imazen/turbo-metrics --json assets`

### 11.6 Fallback: build from source on the box
If the GitHub release binary doesn't work or assets don't match the expected name:
```bash
# Replace the binary-pull section of the onstart-cmd with:
apt-get install -y git build-essential pkg-config libssl-dev libclang-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
git clone --depth 1 https://github.com/imazen/turbo-metrics.git
cd turbo-metrics
cargo build --release --bin zen-metrics --features sweep,gpu
cp target/release/zen-metrics /usr/local/bin/
```
Adds ~10-15 min build time per instance. Acceptable.
