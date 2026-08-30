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
- **k-seed wave**: `.../kseed/`, `.../kseed_b/`, `.../kseed_c/` (three lanes
  run in parallel; per-seed bake + `.toml` + train/eval logs +
  `kseed_spread.tsv` in each). Merged table committed at
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

## k-seed spread (k = 5, 2026-08-30)

The grid winner retrained at 5 seeds with the identical recipe — only `--seed`
varies — then each bake scored on the full origin-validate view. Per-seed rows
+ shas: [`benchmarks/metapicker_kseed_spread_2026-08-30.tsv`](metapicker_kseed_spread_2026-08-30.tsv).

| seed | sha256 (12) | internal heldout argmin | **val argmin** | **val ovh mean** | p50 | **p90** | val SROCC |
|---|---|---|---|---|---|---|---|
| 0 (= shipped v1) | `4479ef9c874e` | **0.70212** | 0.7500 | 0.0447 | 0.0000 | 0.1453 | 0.9869 |
| 1 | `f47d81b9fbe8` | 0.68283 | 0.7519 | 0.0456 | 0.0000 | 0.1469 | 0.9861 |
| 2 | `5bf753c9cfd1` | 0.68175 | 0.7545 | 0.0445 | 0.0000 | 0.1453 | 0.9843 |
| 3 | `b88671d4e030` | 0.69670 | 0.7519 | 0.0431 | 0.0000 | 0.1444 | 0.9854 |
| **4 (SELECTED)** | `4485cf37da8f` | 0.69546 | **0.7551** | **0.0427** | 0.0000 | **0.1370** | 0.9859 |

**Band (k = 5):**

| metric | median | sd | min–max |
|---|---|---|---|
| argmin accuracy | **0.7519** | 0.0021 | 0.7500 – 0.7551 |
| overhead mean | **0.0445** | 0.0012 | 0.0427 – 0.0456 |
| overhead p50 | **0.0000** | 0.0000 | 0.0000 – 0.0000 |
| overhead p90 | **0.1453** | 0.0039 | 0.1370 – 0.1469 |
| bytes SROCC | **0.9859** | 0.0010 | 0.9843 – 0.9869 |
| internal heldout argmin | 0.6955 | 0.0090 | 0.6817 – 0.7021 |

**The picker is seed-STABLE on every decision metric.** argmin spans 0.51 pp
(sd 0.0021), overhead mean spans 0.29 pp (sd 0.0012), the median pick costs
ZERO extra bytes at every seed, and p90 spans 0.99 pp. The v1 numbers are not a
lucky draw — but they are the *bottom* of the band (see below).

**Selection rule applied.** The registered rule is the trainer's own
(`search.rs`): highest held-out **argmin accuracy**, ties broken by the
bytes-log SROCC. Run on the **honest origin-validate panel** (the registered
honest surface; the trainer's internal split is a sub-split of the *train*
view's images) it selects **seed 4**, sha256
`4485cf37da8f0f79c4f0a1b79bed20117d2c2555c46bbc3669511779270c3b44`
(`kseed_b/metapicker_v1_s4.bin`, 104,193 B): argmin **0.7551**, overhead
**4.27 % mean / 0 p50 / 13.70 % p90**, bytes-SROCC 0.9859. Against the shipped
v1 that is +0.0051 argmin, −0.20 pp mean overhead and −0.83 pp p90.

**⚠ The two held-out surfaces rank the seeds in OPPOSITE order.** By the
trainer's *internal* split the order is `s0 > s3 > s4 > s1 > s2`; by the honest
origin-validate panel it is `s4 > s2 > (s1 = s3) > s0`. The internal-best seed
is the honest-**worst**, and the internal-worst seed is the honest-second-best.
So the surface the bounded grid search ranks on does **not** track the honest
one in direction at this margin — a k-seed wave selected on the internal split
picks the wrong bake. **Select on the honest panel; report the internal one as
a comparator only.** (This also means v1's original grid-winner status came
from the internal split; on the honest panel v1 is the band floor. All five
seeds clear the baseline gate by ≥4.4×, so this is a refinement, not a defect.)

**Baseline gate re-run at the band's WORST seed (seed 0 = shipped v1).**
PASSES: picker 4.47 % mean / 14.53 % p90 vs the best fixed choice, always-avif,
at **19.75 % / 52.13 %** — **4.4× better**, median pick free. The fixed-policy
table is bake-independent (a property of the dataset + oracle) and came out
byte-identical at every seed, so this gate holds at every seed; at the selected
seed 4 the ratio is 4.6×.

| fixed policy (its own best reachable cell) | mean | p50 | p90 | argmin | coverage |
|---|---|---|---|---|---|
| always-avif | **0.1975** | 0.0000 | 0.5213 | 0.5156 | **0.9444** |
| always-jxl | 0.7795 | 0.3535 | 2.3123 | 0.2180 | 1.0000 |
| always-webp | 0.8052 | 0.3795 | 2.1829 | 0.1705 | 1.0000 |
| always-jpeg | 0.9421 | 0.4266 | 1.4689 | 0.0497 | 0.8878 |
| always-png | 5.4751 | 1.8689 | 11.7862 | 0.0462 | 1.0000 |

The ledger's coarse-grid 20.4 % / 55.1 % **holds** on the dense grid (19.75 % /
52.13 %). **Correction to the ledger**: avif does *not* reach every (image,zq)
cell — coverage **0.9444** (jpeg 0.8878, jxl-lossy 0.7745, webp-lossy 0.7398);
only the three lossless-bearing families reach everywhere, and avif's 19.75 %
is measured on the 94.44 % it can reach. Against the best *full-coverage* fixed
family (jxl, 77.95 % mean) the picker is 17.4× better.

**Ship status: the flip is still user-gated.** Seed 4 is the rule's answer, not
a decision — swapping the shipped bake is a user call, and the wiring stays
inert either way.

## Wiring state

**Landed, INERT** (zenanalyze `782ee433` wiring, `7224d61c` touch-once,
`ec66b1b` the baseline gate at the owner, `563e3579` slot identity, `61ece73d`
demo, `9a3d241` + `f31baee` contract-validation tests, `fa0c1bc` + `f479df1`
docs).
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

Because those 6 are skipped on a runner, `zenpicker/tests/cell_contract.rs`
covers the *refusal* half on synthetic in-process bakes — 13 tests, no external
artifact, so every contract check runs on every CI run: a bad cell label, a
repeated `(family, mode)`, a cell count that disagrees with `n_outputs`, an
input order that is not a bijection or has no `zq_norm` or names an undeclared
feature, `zq_norm` declared as a source feature, a short *or* long input
vector, a wrong-width reach mask, and the schema gate.

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
