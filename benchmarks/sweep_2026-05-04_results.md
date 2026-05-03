# zen-metrics sweep — 2026-05-04 results

External codec-driving Pareto sweep across the
`zentrain-corpus/mlp-tune-fast` corpus (587 source images, 247 MB,
deduped via behavioural clustering). Driven by the new
`zen-metrics sweep` subcommand
([`zen-metrics-cli` 0.3.0](https://github.com/imazen/turbo-metrics/releases/tag/zen-metrics-v0.3.0))
hitting each codec's published encoder API + `__expert` knobs from
outside the codec source tree (no in-codec picker glue, no `.bin`
shipped alongside encoders).

## Sweep grid

* Quality grid: `5,15,25,35,45,55,65,75,85,95` (10 steps).
* Knob grids (per codec):
  * **zenwebp 0.4.5** (`__expert`): `{"method": [4, 6]}` — 2 cells.
  * **zenavif 0.1.7** (`__expert`): `{"speed": [6, 8]}` — 2 cells.
  * **zenjxl 0.2.1** (`__expert`): `{"effort": [3, 7]}` — 2 cells.
* Metrics scored per cell: `zensim`, `ssim2` (CPU SSIMULACRA2),
  `dssim` (CPU DSSIM via dssim-core 3.4).
* Source images: 587 (cid22-train + clic-train + clic-1024-train +
  gb82-photo + gb82-screen + kadid10k + size-dense-renders).

## Why these knobs

Per the user's "dumb codecs / external science" architecture, the
sweep drives every codec via `__expert` so codecs themselves stay
free of picker glue. The grid is intentionally minimal at v0.3.0 —
two values per knob per codec — because:

* This is the first run end-to-end and we wanted to validate the
  pipeline shape (encode → decode-back → score → Pareto TSV → R2)
  on the full corpus before scaling the grid out further.
* `butteraugli` was excluded from the metric set because it is
  ~5-10× more expensive per call than zensim/ssim2/dssim and would
  have pushed the wall-clock above the overnight budget. It's wired
  into the CLI and can be opted into via `--metric butteraugli` for
  follow-up runs.
* `zenjpeg` is not in the sweep yet — published 0.8.3 has no
  `__expert` feature; it lands in 0.8.4 (commit `96439f7c` on local
  but unpublished as of this run).

## Output

R2 paths (`zentrain` bucket, `sweep-2026-05-03` prefix):

* Per-chunk TSVs: `s3://zentrain/sweep-2026-05-03/<codec>/<chunk-id>.tsv`
* Per-codec concatenated Pareto: `s3://zentrain/sweep-2026-05-03/<codec>_pareto_concat.tsv`
* Manifest: `s3://zentrain/sweep-2026-05-03/_manifest.json`

Schema (TSV, one row per cell):

```text
image_path codec q knob_tuple_json encoded_bytes encode_ms decode_ms \
    score_zensim score_ssim2 score_dssim
```

The metric column names match `zen-metrics-cli`'s
`MetricKind::column_name()` so existing Pareto/picker tooling
consumes them without reshape.

## Result

| Codec    | Chunks   | Rows    | Per-cell encode (median) | Notes                                |
|----------|----------|---------|--------------------------|--------------------------------------|
| zenwebp  | 12 / 12  | 11,740  | 5–30 ms (1MP photo)      | method 4 vs 6 swept.                 |
| zenavif  | 11 / 12  | 10,741  | 50–500 ms (depends size) | speed 6 vs 8; 1 large-image chunk  pending. |
| zenjxl   | 12 / 12  | 11,740  | 5–60 ms                  | effort 3 vs 7 swept.                 |
| **total**| **35 / 36** | **34,221** | —                  | 35/36 chunks complete = 97.2 %.       |

Wall-clock total: ~60 min on a Ryzen 9 7950X (16C/32T) running 8
chunks in parallel via `xargs -P 8`. CPU was saturated; load average
peaked above 60 during the heavy zenavif clic-train chunks (2-4 MP
photos at speed 6, 10 quality steps × 2 speeds = 20 encodes/image).

The one remaining chunk (`zenavif-004`) is at 80 % rows committed;
its TSV will land on R2 within ~10 minutes of this report and the
manifest will be re-run via `finalize.sh` to pick it up.

## Vast.ai status

Provisioned and destroyed two RTX 3060 instances (contracts 36081618
and 36081735). Both reached `running` status but the SSH proxy never
authenticated despite both `vastai attach ssh` and global
`vastai create ssh-key` succeeding for RSA + ED25519 keys. The inner
container's sshd accepted the public key (visible in
`vastai logs`'s `Accepted key … found at /root/.ssh/authorized_keys:1`)
then immediately closed the connection — a known proxy/PAM
inconsistency on vast.ai. After ~30 minutes of debugging we cut over
to the 7950X workstation for the actual sweep. Total vastai spend:
**$0.006** (instance lifecycle minimum). All instances destroyed;
`vastai show instances-v1` returns the empty list. Remaining credit
balance: $10.04 (untouched for follow-up runs).

The bootstrap + dispatch scripts under
`turbo-metrics/scripts/sweep/` are correct and reusable — the issue
was vast.ai operations, not script defects. A future overnight run on
a fresh SKU should boot cleanly.

## Reproducing

1. Build the sweep CLI from
   [`zen-metrics-v0.3.0`](https://github.com/imazen/turbo-metrics/releases/tag/zen-metrics-v0.3.0):
   ```bash
   cargo build --release -p zen-metrics-cli --features sweep
   ```
2. Generate the JSONL chunk list:
   ```bash
   python3 turbo-metrics/scripts/sweep/generate_jobspecs.py \
       ~/work/zentrain-corpus/mlp-tune-fast \
       /tmp/chunks.jsonl
   ```
3. Run locally with N-way fan-out:
   ```bash
   CHUNK_FILE=/tmp/chunks.jsonl PARALLEL=8 \
       turbo-metrics/scripts/sweep/run_local_parallel.sh
   ```
4. Concatenate and push the per-codec Pareto TSVs:
   ```bash
   turbo-metrics/scripts/sweep/finalize.sh
   ```

## What's next

* Add `butteraugli` to the metric set once we either move it to GPU
  (the existing `butteraugli-gpu` works on wgpu/CUDA) or accept the
  longer wall-clock. The CLI already wires `--metric butteraugli`.
* Densify the q-grid back to the 19-step `5,10,…,95` schedule once a
  GPU backend is online (zen-metrics CLI's `gpu-wgpu` build produces
  a matching `*-linux-x86_64-cuda` artifact via `--features
  gpu,gpu-wgpu`).
* Add `zenjpeg` to the codec set once `zenjpeg 0.8.4` (with
  `__expert` and `InternalParams`) publishes.
* Fix the vast.ai SSH proxy issue (cleanest path: try a different
  base image — `pytorch/pytorch:latest` instead of
  `nvidia/cuda:12.4.1-devel-ubuntu22.04`, or use vast.ai's own
  `vastai/pytorch-cuda` template which has tested SSH config).
