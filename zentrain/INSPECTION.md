# Inspecting a baked picker

How to look inside a `.bin` and catch hidden pathologies before they
ship. Companion to [`PRINCIPLES.md`](PRINCIPLES.md) — the principles
doc says what's invariant; this one walks the tools that verify
your bake conforms.

## Tools

| | What it does |
|---|---|
| `zenpredict-inspect <model.bin>` (Rust binary) | Loads the bake and dumps everything the runtime sees as a single JSON document on stdout: shape, weights / biases / scaler, feature_bounds, metadata. With `--weights` includes the full weight arrays. Used as the data source for the Python tool below |
| `zentrain/tools/inspect_picker.py` (Python) | Surfaces pick distribution, tree approximation, boundary-stress pathologies, and tree-vs-MLP divergence over a corpus |

Build the binary once per workspace bump:

```bash
cargo build --release -p zenpredict --bin zenpredict-inspect
```

The Python tool finds it on PATH or the workspace `target/release/`
directory; pass `--inspect-bin <path>` to override.

## Modes (in the order you should run them)

### 1. `summary` — sanity check the bake

```bash
python3 zentrain/tools/inspect_picker.py path/to/picker.bin --mode summary
```

Prints:

- File size, format version, flags, schema_hash
- Per-layer dims + activation + dtype + weight magnitude stats (`abs_max`, `abs_p50`, `abs_p99`, `abs_mean`, `near_zero_fraction`)
- Total weight count + sparsity
- `feature_bounds` populated? Finite bounds vs ±∞ engineered axes
- Every metadata entry (key, kind, value preview)

**Pathology checks (reads-on-summary):**

| Symptom | What it means |
|---|---|
| Layer near-zero fraction > 50 % | Trainer over-regularized; many neurons are dead. Re-train with lower L2 or fewer hidden units |
| `abs_max ≫ abs_p99` (e.g. 100× ratio) | Outlier weight survived training. Could be a healthy "I really need this feature" — or it could be the trainer overfitting one corpus image. Cross-check with `mode picks` for cell sensitivity |
| `feature_bounds` count < n_inputs OR all `±inf` | Bake didn't populate OOD bounds. Codec's `first_out_of_distribution` is a no-op. Re-bake with `safety_report.diagnostics.feature_bounds` populated |
| Missing `zentrain.bake_name` / `zentrain.schema_version_tag` | Bake-side metadata wasn't emitted. Codec startup logs can't tell which bake is loaded. Re-bake with `bake_picker.py` defaults (it emits these unconditionally) |
| Missing `zentrain.hybrid_heads_layout` for hybrid-heads pickers | Codec doesn't know where the bytes head ends and scalar heads begin. Most inspect modes fall back to "treat all outputs as cells", but a real codec MUST know |

### 2. `tree-approx` — what's the picker actually doing

```bash
python3 zentrain/tools/inspect_picker.py picker.bin \
    --mode tree-approx \
    --corpus benchmarks/zq_pareto_features_2026-04-30.tsv \
    --depth 4 \
    --dot /tmp/picker_tree.dot

# Render:
dot -Tsvg /tmp/picker_tree.dot -o /tmp/picker_tree.svg
```

Fits a depth-`N` decision tree on `(feature vector → picker argmin
cell)` pairs over the corpus. Reports:

- The tree as text (sklearn `export_text`)
- Agreement % vs the MLP on the training corpus
- Optional graphviz `.dot` for visualization

**What it tells you:**

- **Which features matter.** Tree splits at the top of the tree are
  the highest-IG features for the picker's argmin. If the tree
  depends mostly on features 1-2 of a 50-feature schema, the picker
  is probably under-using its inputs.
- **Whether the MLP earns its complexity.** Agreement ≥ 95 % at
  depth 4-6 means a tree could replace the MLP; the MLP isn't
  paying for the parameters it spends. Expected agreement is
  roughly 75-90 % at depth 6 — the MLP captures non-axis-aligned
  decisions a tree can't.
- **What the cell partition looks like.** The text rendering of the
  tree shows feature thresholds and which cell wins in each leaf.
  Hand-readable.

**Pathology checks:**

| Symptom | What it means |
|---|---|
| Tree depth-4 agreement = 100 % | The MLP is just an axis-aligned classifier. Replace with a tree (smaller, faster, debuggable) — or check that you're not training a degenerate corpus |
| Tree splits on engineered axes (`zq_norm`, `log_pixels`) only | Picker is ignoring image features. Either the corpus doesn't reward image-feature use, or the MLP didn't learn to. Both are bad signs |
| One cell dominates every leaf | One cell is universally optimal in your corpus. The picker isn't differentiating. Re-check the corpus diversity |

### 3. `picks` — pick distribution + dead-cell detection

```bash
python3 zentrain/tools/inspect_picker.py picker.bin \
    --mode picks \
    --corpus features.tsv \
    --target-zq 85
```

Prints:

- Per-cell hit count + bar chart (over the corpus)
- Confidence-distribution: `p50`, `p10`, `p01` of the top-2 score gap
- Count of low-confidence picks (gap < `--low-confidence-threshold`,
  default 0.05 in log-bytes space)

**Pathology checks (built in — flagged automatically):**

| Symptom | What it means |
|---|---|
| **DEAD CELL** (cell never picked) | Cell carries weight-cost but contributes nothing. Re-train with smaller categorical-axis enumeration, or expand corpus to include the conditions that should select it |
| **DOMINANT** (cell > 50 % of picks) | Cell saturates the corpus — picker is barely doing work. Often signals corpus skew (e.g. only-photo training set always picking the photo cell) |
| Low-confidence picks > 30 % of corpus | Picker is uncertain on a third of your corpus. Either the corpus is genuinely ambiguous (multi-near-tie cells), or the MLP under-fit. Cross-check with `mode tree-approx` |

### 4. `stress` — boundary pathologies (synthetic edge inputs)

```bash
python3 zentrain/tools/inspect_picker.py picker.bin \
    --mode stress \
    --target-zq 85
```

Synthesizes feature vectors at every `(p01, midpoint, p99)` corner
of the bake's `feature_bounds` envelope (3^n combinations for
small schemas; for n_feat > 8 falls back to axis-aligned single-edge
perturbations to keep the grid manageable).

**Pathology checks (auto-flagged):**

| Symptom | What it means |
|---|---|
| Same cell picked at every corner | Picker isn't differentiating boundary inputs. The MLP defaulted to its prior on out-of-distribution corners — training corpus didn't span this far |
| Median confidence > 1.0 at corners | Picker is HIGHLY confident at OOD inputs. It SHOULD be uncertain. Indicates linear extrapolation past the training envelope. **High-risk**: production traffic that lands at corners (e.g. extreme aspect ratios, pure-screen content) gets picks the model has no business being confident in |
| Fewer than 30 % of cells fire at corners | Corner inputs all converge to a small handful of cells — model has effectively collapsed boundary behavior |
| Predicted bytes_log lands outside `output_bounds` p01..p99 at any corner | Pure hallucination — the MLP is producing a value the training run never observed. Check via `output_first_out_of_distribution(&pred, model.output_bounds())` after each corner predict. Codecs that hit this in production should refuse the pick and route to `KnownGoodFallback` |

### 5. `diverge` — tree-vs-MLP disagreement, stratified by confidence

```bash
python3 zentrain/tools/inspect_picker.py picker.bin \
    --mode diverge \
    --corpus features.tsv \
    --depth 6
```

Fits the depth-`N` tree as in `tree-approx`, then on the same corpus
counts where MLP and tree disagree. Splits the disagreements by
MLP's top-2 confidence (top-quartile = high-confidence
disagreements; bottom-quartile = low-confidence).

**Pathology checks:**

| Symptom | What it means |
|---|---|
| Zero disagreements at any depth | The MLP is shaped exactly like a tree of that depth. Replace with the tree |
| Many disagreements at high MLP confidence | The MLP confidently picks something the tree can't justify. **Two interpretations:** (a) the MLP found real non-axis-aligned signal a tree can't capture (good — MLP is earning its complexity); (b) the MLP memorized corpus quirks (bad — overfitting). Distinguish by checking held-out generalization separately |
| Disagreements concentrated in low-confidence band | Expected — near-tie picks split arbitrarily between MLP and tree. Not a red flag |

## Recommended sequence

For a fresh bake about to ship:

```
1. summary           — sanity check shape + metadata
2. picks --corpus    — sanity check pick distribution on validation set
3. stress            — boundary pathologies (free; no corpus needed)
4. tree-approx       — what's it doing; emit dot for the PR
5. diverge           — earn-its-complexity check
```

For an already-shipped bake suspected of misbehavior on production
traffic:

```
1. summary           — confirm schema_hash matches deployed
2. picks --corpus production_features.tsv  — what cells fire on real traffic
3. stress            — does the picker behave at boundary?
4. tree-approx       — read the picker's logic; compare to expectations
```

## Caveats / known limitations

1. **Engineered axis reconstruction is legacy-zenjpeg-shaped.**
   `prepare_input_for_picker` reconstructs the
   `n_feat + 4 size + 5 poly + n_feat zq×feat + 1 icc` layout. New
   layouts need their own reconstructor. If `n_inputs` doesn't match
   that legacy shape, the tool warns and runs on raw `feat_cols`
   only — picker output may be nonsense. Address by extending
   `prepare_input_for_picker` per the codec's `extra_axes` declaration.

2. **Tree-approx mimics the categorical-bytes head, not scalar heads.**
   Scalar predictions (chroma_scale, lambda) aren't
   visualized — that's a different shape (regression per cell). For
   scalar diagnostics, run `mode picks` and inspect `predict()`
   output directly via the Python forward.

3. **Doesn't currently surface SHAP-style per-prediction attribution.**
   Filed as a follow-up; the tree-approx + stress modes
   together cover most pathology cases without SHAP's overhead.

4. **`stress` corner grid is exponential in n_feat.** For `n_feat
   > 8` the tool falls back to axis-aligned perturbations. The full
   corner grid (3^15 ≈ 14M points for the v2.1 schema) would crush
   the forward pass; axis-aligned is the practical compromise.

## Cross-references

- [`PRINCIPLES.md`](PRINCIPLES.md) — the data discipline this tool verifies
- [`SAFETY_PLANE.md`](SAFETY_PLANE.md) — what the codec does when picks are pathological
- [`tools/feature_ablation.py`](tools/feature_ablation.py) — global feature importance via permutation (complementary to `tree-approx`)
- [`tools/size_invariance_probe.py`](tools/size_invariance_probe.py) — size-stability gate
- [`zenpredict/src/bin/zenpredict_inspect.rs`](../zenpredict/src/bin/zenpredict_inspect.rs) — the Rust dump binary
- [`tools/inspect_picker.py`](tools/inspect_picker.py) — the Python inspector
