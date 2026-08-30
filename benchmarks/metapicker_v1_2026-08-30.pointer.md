# metapicker_v1 (criterion-8 codec-family meta-picker) — pointer + wiring state

## Artifacts (block storage; >30 KB, never in git)

- **Bake**: `/mnt/v/output/zensim/metapicker-2026-08-30/metapicker_v1.bin`
  (104,193 B, sha256
  `4479ef9c874ebf1c83d059794d2959e725c1ea9d1aa77a5721c7a26a31624e34`;
  sibling `.toml` manifest).
- **Inputs**: `meta_{train,validate,test}.parquet` in the same directory
  (2,946,036 / 1,764,808 / 1,031,816 rows, 0 missing renditions, sha256s in
  `_MANIFEST.json`). Builder:
  zensim `scripts/canonical_corpus/build_metapicker_input_2026-08-30.py`.
- **k-seed wave**: `.../kseed/` and `.../kseed_b/` (per-seed bake + `.toml` +
  train/eval logs + `kseed_spread.tsv`). Merged table committed at
  `benchmarks/metapicker_kseed_spread_2026-08-30.tsv`.
- **Slot→feature identity**: `benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv`
  (committed; see "The identity gap" below).

## Contract (self-describing via bake metadata)

62 inputs = **61 source features ⊕ `zq_norm`** (the caller's requested quality
÷ 100; the codec's per-encode `q` is NOT an input — no q-leakage); 7 outputs =
`bytes_log` per **family×mode cell**:

| idx | cell | family | mode |
|---|---|---|---|
| 0 | `zenavif_lossy` | Avif | Lossy |
| 1 | `zenjpeg_lossy` | Jpeg | Lossy |
| 2 | `zenjxl_lossless` | Jxl | Lossless |
| 3 | `zenjxl_lossy` | Jxl | Lossy |
| 4 | `zenpng_lossless` | Png | Lossless |
| 5 | `zenwebp_lossless` | Webp | Lossless |
| 6 | `zenwebp_lossy` | Webp | Lossy |

Keys: `zenpicker_train.cell_labels`, `.image_feature_names`, `.input_order`.
The pick is a **masked argmin** over reachable cells (smaller `bytes_log` =
fewer bytes).

## Training recipe (reproduced BYTE-IDENTICALLY)

```sh
zenpicker-train --input meta_train.parquet --out metapicker_v1.bin \
                --mode mlp --hidden 128,128 --seed 0
```
`--hidden` disables the bounded grid search and fits exactly that topology at
`MlpConfig::default().lr` = 2e-3 — i.e. grid candidate #2, the winner. Default
`--val-frac 0.2`, grouped-by-image split, no input shaping, no distillation.
64,596 (image,zq) rows → 51,688 train / 12,908 held out.

**Verified 2026-08-30**: this command reproduces `metapicker_v1.bin`
**byte-for-byte** (same sha256) with an identical `[heldout]` block to every
digit. Training is bit-deterministic, so the k-seed wave below varies *only*
the seed.

## Honest panel (origin-validate view)

```sh
zenpicker-train --input meta_validate.parquet --eval-bake <bake>.bin \
                --val-frac 1.0 --baselines
```

`--val-frac 1.0` scores the ENTIRE held-out view (38,696 (image,zq) rows,
245,585 reachable (row,cell) pairs). The original v1 panel used
`--val-frac 0.999`, which `grouped_split_picker`'s rounding turns into "all
but one image" = 38,668 rows / 245,402 pairs — **measured, and the only
difference**: every reported statistic is identical to 4 dp either way.

<!-- BAND -->

## Wiring state

**Landed, INERT** (zenanalyze `782ee433`, `7224d61c`, `563e3579`, `61ece73d`).
The registered wiring problem was real: v1's cells are FAMILY×MODE (7) while
`zenpicker::CodecFamily` is a 6-enum with no mode axis, so
`MetaPicker::pick` — which reads `CodecFamily::ALL[output_index]` — would
mis-map every output. The adapter is a **separate, additive** surface:

- `CellContract::from_model` — reads the three metadata keys and validates
  them against the model's real `n_outputs` / `caller_input_width`; refuses on
  any disagreement.
- `CellContract::build_input(zq_norm, source)` — **the** contract mapping.
- `CellPicker::from_znpr_bytes[_with_schema]` — owns the parsed bake +
  validated contract; refuses a bake without a well-formed contract.
- `CellPicker::predict_cells(input, allowed, reachable)` — one forward pass →
  7 cell scores + the masked argmin cell.
- `CellPicker::meta_picker()`, `FamilyModeCell`, `CellMode`, `CellPrediction`,
  the three key consts, `MetaPickerError::CellContract`.

Nothing existing changed. `default_route` / `MetaPicker::{route, pick,
default_routers}` and the three shipped routers are untouched, and a test
asserts each shipped router is **refused** as a cell bake. The whole path is
`no_std + alloc` clean (builds for `aarch64-unknown-none`). Flipping a cell
bake into the shipped route remains a user-gated decision.

`examples/cell_pick_demo.rs` runs the path end to end on real analyzer values.

## Touch-once contract test

`zenpicker/tests/metapicker_v1_contract.rs` (6 tests, run with
`just metapicker-v1-test`). The registered touch-once test asserts the
contract mapping is a **bijection**: each of the 61 declared source features is
requested from the caller's source **exactly once**, `zq_norm` is placed
exactly once at its declared index and never requested from the source, no
name outside the contract is read, and every input slot carries the value
belonging to the name declared there (the probe source is injective by
construction). Alongside: the registered 7-cell shape, a missing feature
failing loudly, the schema-hash gate, the masked-argmin forward pass, and the
slot-identity cross-check.

The bake is located via `ZENPICKER_METAPICKER_V1_BAKE`; **unset ⇒ the tests
FAIL LOUDLY**, never self-skip. CI has no block storage and makes the skip
decision explicitly (`-- --skip metapicker_v1_`), so the chain is visible:
CI workflow → justfile → test.

## The identity gap (open, upstream)

The bake declares its source features as **positional placeholders**
(`feat_0..feat_60`), so the bake ALONE cannot say which analyzer feature
belongs in which slot; it also carries no `zentrain.feature_columns`, so
`Model::feature_columns()` is empty and `MetaPicker::feature_request()`
returns `None` for it — i.e. **v1 cannot participate in zenanalyze-api `Offer`
negotiation**.

Cause, located: zensim
`scripts/canonical_corpus/build_metapicker_input_2026-08-30.py:59`
renames the qualified source columns to `feat_<j>` and records the original
names nowhere (not in the parquets, not in `_MANIFEST.json`). The upstream
`clean_features.tsv` carries them in exactly the qualified `name@hex8` form
the contract wants.

`benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv` recovers all 61 by
re-running the builder's own deterministic rule against the sha256-pinned
source TSV, and a test keeps it in lockstep with the bake. **The fix for
future bakes belongs upstream** (the builder preserving the names, or a
feature-name passthrough in `zenpicker-train`); it was NOT applied here
because changing the bake metadata would have voided the byte-identity
reproduction gate above.

## Test view

`meta_test.parquet` (origins {7,9}) is **UNREAD** — reserved for the
touch-once terminal read at ship proposal.

## Measurement cost (finding)

One `--eval-bake` pass over the full validate view takes **1125 s**, almost
all of it the O(n²) KROCC + PWRC inside `compute_panel_lowmem` at 245 k pairs
— **neither of which `--eval-bake` prints**. A rank-only panel option would
cut a k-seed wave from hours to minutes. Not done here (it would have mixed
binaries mid-wave); registered for the next pass.
