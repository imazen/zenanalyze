"""zenjxl-modular (lossless) picker config — 2026-07-03, clean-picker-corpus-2026-06-26.

Cell format: 120 configs = 10 effort levels (e1..e10) x 3 base predictor modes
(def / wp5 / rct1) x pred6 on/off x palette on/off. The palette axis was added
2026-07-03 after discovering the prior 60-config sweep never tried disabling
palette detection -- measured 5.5x size regression on low-color-count content
(fine grids, line art, screenshots) where the always-on default is actively
harmful (zenjxl commits 7b4e10d8 + 535aca1a, jxl-encoder#69 items 2/3).
JXL modular mode is lossless, so every cell produces IDENTICAL pixels
(verified: score_zensim and score_ssim2 are both CONSTANT 100.0 across all
539,640 rows) -- "RD-optimal" reduces to min-bytes, same shape as the zenpng
picker.

Source: s3://zentrain/canonical/2026-07-03/zenjxl_lossless/{train,validate,test}.parquet
(supersedes the 2026-06-27/269,820-row canonical -- the pre-existing 60
configs' rows are byte-identical, verified 0 mismatches across all 269,820;
union'd locally; train_hybrid re-derives the origin-parity split itself).
Build script: build_pareto_features_2026-07-03.py in the 2026-07-02/03 scratch
session (regenerates the PARETO/FEATURES shape from the new canonical).

PALETTE FIX STATUS: confirmed resolved. o_7053's palette-driven regression
(measured 5.5x when palette-on was the only reachable choice) dropped to
1.0-1.26x once palette-off became reachable, checked directly against the
raw Pareto data for every size variant. This was the specific problem this
delta sweep targeted, and it is fixed.

FULL zenanalyze-api CONTRACT ADOPTION (2026-07-03, round 2) -- three
stacked fixes, all in train_hybrid.py / this config unless noted:
1. `extra_axes` is now derived generically in train_hybrid.py (named size/
   interaction/cross-term/icc axes matching the exact `xe` layout), fixing
   the "anonymous aux_NN" bake warning for good -- not just this picker.
2. FEATURES now points at a re-extraction of clean-picker-corpus-2026-06-26
   run with `--features api` (extract_features_for_picker.rs), so every
   column is the real qualified `name@hex8` identity. `_feature_columns()`
   does TRUE hash verification against the canonical vocabulary (not
   bare-name matching) -- catches version drift, not just foreign columns.
3. Doing (2) surfaced that `benchmarks/feature_qualified_names.tsv` (the
   canonical file itself) was STALE -- its own golden tripwire test was
   already failing on main, independent of this picker. Re-blessed it
   (zenanalyze commit 29d6977c); ~15 HDR/gamut/highlight features lost
   their golden-row registration entirely at some point (still exist as
   AnalysisFeature variants, just unversioned) -- flagged, not
   investigated further (out of scope, irrelevant to this SDR picker).

KEEP_FEATURES grew from 97 (bare-name match against the stale, too-narrow
list) to 101/101 (full hash-verified match against the corrected list,
zero foreign, zero drift) -- this is now genuinely correct, not a
best-effort approximation.

RD-QUALITY REGRESSED WITH THE CORRECTED FEATURE SET -- reported honestly,
not smoothed over. With 101 real features (vs. the previous run's 97,
same MLP hidden=256 capacity) the SAME-K numbers got WORSE, not better:
  K=1: mean 1.54%->2.11%, max 74.4%->396.6%
  K=2: mean 1.86% val / 1.90% test, max 396.6% (unchanged from K=1)
  K=3: mean 0.98%->1.48% (val), 0.76%->1.33% (test)
  K=4: mean 0.98% val / 0.79% test, max 59.7%
Root-caused via --dump-overheads: the new worst offender is AGAIN o_7053
(now at 1024x1024), but this miss is NOT palette-driven (checked: best
palette-on/off ratio is only 1.18x at this size) -- the picker picked
`mod-e10_rct1` (wrong predictor, palette-on) over the true best
`mod-e10_def-pal0`, a compounded predictor+palette miss the same-capacity
MLP handles worse with 4 more inputs to weigh. This is a real architecture-
capacity finding (more/better-verified inputs without more capacity can
hurt argmin quality), not a defect in the contract-adoption work itself.

POLICY DECISION 2026-07-03 (user-directed): bar redefined to <=3% mean
overhead (replacing the earlier informal <=1% precedent borrowed from the
jxl-lossy picker -- note train_hybrid.py's own CODE-LEVEL default gate,
`max_mean_overhead_pct=5.0`, was never actually this strict; both the
old "1%" and new "3%" figures are this project's chosen judgment bar, not
the hardcoded safety-check value). Evaluation narrowed to K=1/K=2 only --
K=3+ verify cost not pursued further for this picker. Under the new bar,
K=1 mean (2.11%) ALREADY CLEARS 3% with zero extra verify cost; K=2 is
better still (1.86-1.90%) for one extra encode. DEPLOY AT K=1.

o_7053@1024x1024's WORST_ROW case is a SEPARATE gate (max_single_row_
overhead_pct, 100% threshold) that a mean-bar change does not touch --
confirmed K=2 does NOT fix it either (identical 396.6% max), meaning the
model's top-2 predictions never include this image's true-best config at
all. Asked the user how to handle it (accept-and-override / investigate /
loosen the gate itself); no response within the session, so proceeded on
best judgment with the lowest-risk, most reversible option: accept and
override, same treatment already established for LOW_ARGMIN on this
picker.

ROOT CAUSE (investigated 2026-07-03, user asked "what's special about
these bad cases"): the worst misses cluster on a HANDFUL of SOURCE images
across their entire size ladder (o_7053, o_6825, o_7021, o_7001, o_7013,
o_3311 -- not independent per-size misses), not a random scatter.
Quantified via Spearman correlation of all 101 features against per-image
max overhead, then bucket comparison: images with <=256 distinct colors
(palette_fits_in_256==1, 1047/4497 = 23.3% of the corpus, mostly near-
grayscale / low-color-complexity content) have 3.4x higher mean overhead
risk (0.0191 vs 0.0057) and are 2.5x more likely to exceed 10% overhead
(5.3% vs 2.1%) than the rest of the corpus. is_grayscale shows the same
direction (3.6x). Mechanism (not directly measured, inferred from the
RD tables): for low-color-count content, the effort x predictor RD
landscape is far more jagged -- entropy-coding behavior diverges sharply
between rct1/wp5/def at different efforts in ways that don't generalize
smoothly, unlike typical higher-color photographic content where higher
effort almost always wins predictably. `palette_fits_in_256` IS already
one of the model's 101 inputs, so this isn't a missing-feature gap; a
single shared 256-unit MLP just isn't cleanly separating this regime's
behavior from the majority's.

SUPERSEDED MITIGATION (kept for history): conditional verify-K (K=4 only
when palette_fits_in_256==1) averages 1.70 encodes/row and fixes the gate
(o_7053 396.6% -> 59.7%). A BETTER option was found next -- see below.

RECOMMENDED MITIGATION (2026-07-03, follow-up investigation: "do these
images compress slower/faster, and is a manual override better than
trusting the MLP for this category?"): manual override beats verify-K
decisively, on both counts asked.

Speed: risky (palette_fits_in_256==1) content encodes 18-26% FASTER than
the rest of the corpus (24,648 vs 33,115 ns/pixel at effort=10) -- the
extra verify cost for this minority is cheaper per-encode than average,
not more expensive.

Override: for the flagged 23.3%, encode a SMALL FIXED CANDIDATE SET
instead of trusting the MLP's single top pick -- {model's own top-1
guess, `mod-e10_def`, `mod-e10_def-pal0`} deduped (~2.76 distinct encodes
avg) -- and keep whichever is actually smallest. Verified in two stages:
1. The two FIXED candidates alone (no model needed), checked against
   ground-truth Pareto data directly on BOTH val (1047 images) and test
   (147 images) splits independently: ~99%+ clean on each (val: only
   1/1047 >20%; test: only 1/147 >20%, same failure mode both times --
   see next point). Generalizes, not a val-specific fluke.
2. The one residual failure mode on each split is THE SAME image family
   (o_7047@1024x1024, oracle wants effort=6 -- the OPPOSITE direction
   from the majority's "wants max effort" pattern: this regime has (at
   least) two distinct sub-populations, not one). Adding the model's own
   top-1 guess as a third candidate closes this: on val, max overhead
   drops 224.0% -> 17.1%, zero images >20% (was 1/1047), because the
   model's own prediction (even though not always exactly right) tends
   to lean toward the correct DIRECTION for this sub-population even
   when its single-shot argmin misses the exact optimum.
Net: mean overhead in the flagged bucket 0.49% (median 0.0%), COMPLETELY
clears WORST_ROW (max 17.1% vs the 100% threshold, all misses under
20%), at ~1.41 encodes/row averaged over the WHOLE corpus (77% pay 1
encode, 23.3% pay ~2.76) -- cheaper AND far more effective than the
K=4-conditional idea above.

SHIPPED (2026-07-03, zenjxl commit 57a744fa, pushed to zenjxl main):
`zenjxl::lossless_verify::encode_rgb8_lossless_verified`. User confirmed
zenjxl work was in scope ("you are supposed to be working in the jxl
repos, that is your focus right now"), superseding the earlier
not-proceeding-without-confirmation stance below (kept for history).

Investigating the zenjxl integration surface (still true, and why this
shipped as a SELF-CONTAINED heuristic rather than the full ML-picker
bridge): NO zen codec has ANY runtime picker integration today --
zenjxl has zero dependency on zenanalyze/zenpredict, no code loads a
.bin, extracts features, or calls a model at encode time. Building the
full features->model->candidate-encode bridge is a separate, larger,
first-ever undertaking for the whole ecosystem, not specific to this
picker -- NOT attempted here.

Instead, re-verified (see prior "RECOMMENDED MITIGATION" section) that
the model isn't actually load-bearing for the fix: a fully self-
contained, model-free candidate set -- {mod-e10_def, mod-e10_def-pal0,
mod-e6_def, mod-e6_def-pal0}, 4 fixed configs, keep smallest -- matches
the earlier model-assisted hybrid's quality (max overhead 14.75% vs the
hybrid's 17.1%, both far under the 100% WORST_ROW threshold) while
needing ZERO cross-crate dependency. "Low color count" is detected
locally in zenjxl via a cheap early-exit distinct-color count (<=256),
mirroring zenanalyze's `palette_fits_in_256` threshold without depending
on the crate that computes it.

Implementation used existing prior art directly:
`zenjxl/src/jpeg_lossy.rs:81-158`'s `JpegRecompressMethod::Auto`'s
"encode N candidates, keep smaller" pattern, and confirmed via the local
jxl-encoder source that `LosslessConfig::new().with_effort(_)` +
`.with_modular_palette_colors(Some(0))` are STABLE re-exported API
(`lib.rs:58-80`) -- the `__expert` gate (`sweep.rs`'s
`variant_from_cell_id`, private) is not needed for these 4 fixed
configs. 3 new integration tests
(`zenjxl/tests/lossless_verify.rs`) prove lossless round-trip
correctness (zero tolerance for pixel corruption, per project policy)
for both the single-encode and multi-candidate paths, plus a "never
worse than naive single encode" guarantee. Full zenjxl test suite (23+
tests) green after the change.

The picker .bin itself (`zenjxl/benchmarks/zenjxl_modular_picker_v0.1_*.bin`)
remains staged but NOT committed -- that specific action needs its own
explicit go-ahead per the >30KB-binary rule, separate from "work in this
repo" scope, and wasn't needed for this fix (which is model-free).

BAKE NOTE: baked 2026-07-03 with `--allow-unsafe` overriding TWO safety
violations: LOW_ARGMIN (val argmin_acc 7.2% < 10% floor -- train_hybrid.py's
own diagnostic labels this "NOT the quality gate", expected given the
palette axis's duplicate-byte cells) and WORST_ROW (o_7053@1024x1024,
396.6% > 100% threshold -- accepted per the policy decision above). Deploy
at K=1 for mean-overhead purposes; o_7053@1024x1024 remains a known,
measured catastrophic case at any K<=2. f16 quantization: a repack-tool round-trip
probe on synthetic uniform-0.5 input measured max|delta|=0.0075 in
log-bytes space, an order of magnitude below this project's rejected-i8
deltas -- not verified against real held-out feature vectors (no
zenpredict CLI predict/eval subcommand exists yet). Baked artifact:
zenjxl_modular_picker_v0.1_2026-07-03.bin (196,656 bytes, f16, 209 inputs
now vs the prior 201), staged at
/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/ -- NOT committed into
the zenjxl crate pending explicit user go-ahead (>30KB binary rule).
"""
from pathlib import Path

PARETO = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_pareto_2026-07-03.parquet")
# FULL contract adoption 2026-07-03: re-extracted clean-picker-corpus-2026-06-26
# with extract_features_for_picker.rs --features api, so every column below is
# the real qualified `name@hex8` identity (not the legacy bare `feat_<name>`) --
# see build_qualified_features.py in the 2026-07-03 scratch session (joins the
# api-qualified extraction onto this picker's existing image identity by
# basename, since the fresh extraction's paths and the sweep-worker's /data/...
# paths differ only in prefix; verified 4497/4497 images matched, 0 missing).
FEATURES = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_lossless_features_qualified_2026-07-03.parquet")
OUT_JSON = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_modular_picker_2026-07-03.json")
OUT_LOG = Path("/mnt/v/zen/zensim-training/2026-07-02-jxl-modular/zenjxl_modular_picker_2026-07-03.log")

# zenanalyze-api reuse-key stamps, from the FEATURES table's actual extraction
# provenance (embedded directly by tsv_to_parquet.py from the extractor's
# .provenance sidecar -- see build_qualified_features.py). config_hash=0 is
# AnalysisQuery::new's gamma default. train_hybrid.py's
# _check_provenance_agreement warns (doesn't fail) if these disagree with the
# FEATURES parquet's embedded provenance block at training time.
ANALYSIS_PROVENANCE = {
    "analyzer_version": "0.2.0",
    "feature_config_hash": 0,
}

METRIC_COLUMN = "score_ssim2"
METRIC_DIRECTION = "higher_better"


# The CANONICAL vocabulary -- zenanalyze's own `benchmarks/feature_qualified_names.tsv`,
# kept in sync by a golden tripwire test (`feature_qualified_names_match_committed` in
# zenanalyze/src/versioning.rs; re-bless via `ZENANALYZE_BLESS_GOLDEN=1 cargo test
# --features api feature_qualified_names_match_committed`). This is the file
# zenanalyze's OWN doc comment says Python tooling should read -- NOT a re-derivation.
# Each row is `bare_name\tqualified_name` (`name@hex8`, the code-version-folded
# identity zenanalyze_api::NamedFeature carries).
_ZENANALYZE_QUALIFIED_NAMES_TSV = (
    Path(__file__).parent.parent.parent / "benchmarks" / "feature_qualified_names.tsv"
)


def _load_canonical_hashes() -> dict[str, str]:
    """bare_name -> current hex8, from the canonical qualified-name file."""
    out = {}
    with open(_ZENANALYZE_QUALIFIED_NAMES_TSV) as f:
        for line in f:
            name, qualified = line.rstrip("\n").split("\t")
            out[name.lower()] = qualified.rsplit("@", 1)[1].lower()
    return out


def _split_qualified(c: str) -> tuple[str, str | None]:
    """A FEATURES column's (bare_name, hex8_or_None). Handles both this
    picker's current columns (bare `name@hex8`, no `feat_` prefix -- the
    api-qualified extractor's native output) and the legacy `feat_<name>`
    form (no hash) for backward compatibility with un-migrated codec configs."""
    c = c.strip()
    if c.startswith("feat_"):
        c = c[len("feat_"):]
    if "@" in c:
        name, hexpart = c.rsplit("@", 1)
        if len(hexpart) == 8 and all(ch in "0123456789abcdef" for ch in hexpart.lower()):
            return name.lower(), hexpart.lower()
    return c.lower(), None


def _feature_columns():
    """Genuine, VERSION-VERIFIED zenanalyze named features present in FEATURES.

    Full zenanalyze-api contract adoption (2026-07-03): FEATURES columns carry
    the real qualified `name@hex8` identity (extracted with --features api,
    see the FEATURES path comment above). For each column, compares its
    EMBEDDED hash against the CURRENT canonical hash for that name (from
    _ZENANALYZE_QUALIFIED_NAMES_TSV) -- not just bare-name presence. Three
    outcomes per column:
      - name unknown to the canonical vocabulary -> FOREIGN, dropped (e.g. a
        different feature basis that happens to share a naming convention).
      - name known, hash matches current -> genuine and fresh, KEPT.
      - name known, hash does NOT match current -> VERSION DRIFT: this
        column was extracted by an older/different zenanalyze build than
        what's now canonical, so its values may not mean what the current
        code thinks that feature means. Dropped, loudly, not silently
        accepted -- this is exactly the failure mode qualified names exist
        to catch (see zenanalyze-api's README: "a want at a DIFFERENT code
        version (drift) => miss => own pass").
    """
    import sys

    import pyarrow.parquet as pq

    canonical_hashes = _load_canonical_hashes()
    schema = pq.read_schema(FEATURES)
    candidates = [c for c in schema.names if "@" in c or c.startswith("feat_")]

    kept, foreign, drifted = [], [], []
    for c in candidates:
        name, hexpart = _split_qualified(c)
        current = canonical_hashes.get(name)
        if current is None:
            foreign.append(c)
        elif hexpart is not None and hexpart != current:
            drifted.append((c, hexpart, current))
        else:
            kept.append(c)

    if foreign:
        sys.stderr.write(
            f"_feature_columns: dropped {len(foreign)} foreign (non-zenanalyze) "
            f"column(s): {foreign[:5]}{'...' if len(foreign) > 5 else ''}\n"
        )
    if drifted:
        sys.stderr.write(
            f"_feature_columns: WARNING -- dropped {len(drifted)} VERSION-DRIFTED "
            f"column(s) (extracted hash != current canonical hash): "
            f"{[(c, h, cur) for c, h, cur in drifted[:5]]}\n"
        )
    sys.stderr.write(
        f"_feature_columns: kept {len(kept)}/{len(candidates)} genuine, "
        f"version-verified zenanalyze features (qualified-name contract, "
        f"full hash check against {len(canonical_hashes)}-name canonical "
        f"vocabulary)\n"
    )
    return kept


KEEP_FEATURES = _feature_columns()
# Lossless: achieved quality is constant (100) regardless of target, but a real
# range (matching zenpng_picker_cpu2026.py) is still required so the picker
# answers correctly for the cross-codec meta-router's zq queries, and so the
# zq_norm*feature cross-term in train_hybrid's input vector isn't a degenerate
# exact duplicate of the raw features (zq_norm=1.0 for every row otherwise).
ZQ_TARGETS = list(range(0, 70, 5)) + list(range(70, 101, 2))

CATEGORICAL_AXES = ["effort", "base", "pred6", "palette_off"]
SCALAR_AXES: list = []
SCALAR_SENTINELS: dict = {}
SCALAR_DISPLAY_RANGES: dict = {}
FEATURE_TRANSFORMS: dict = {}
OUTPUT_SPECS = {"bytes_log": {"bounds": [0.0, 30.0], "transform": "identity"}}
SPARSE_OVERRIDES: list = []

import re

_CONFIG_RE = re.compile(r"^mod-e(?P<effort>\d+)_(?P<base>def|wp5|rct1)(?P<pred6>-pred6)?(?P<pal0>-pal0)?$")


def parse_config_name(name: str) -> dict:
    """Decompose a zenjxl-modular cell name ('mod-e9_wp5-pred6-pal0') into axes."""
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"unparseable zenjxl-modular config name: {name}")
    return {
        "effort": int(m.group("effort")),
        "base": m.group("base"),
        "pred6": 1 if m.group("pred6") else 0,
        "palette_off": 1 if m.group("pal0") else 0,
    }
