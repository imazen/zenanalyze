//! The **family×mode cell contract** — the output layout a
//! `zenpicker-train` meta-picker bake declares about itself, and the
//! inert load/forward path for it.
//!
//! ## Why this is separate from [`MetaPicker::pick`](crate::MetaPicker::pick)
//!
//! [`pick`](crate::MetaPicker::pick) is a **family-score** path: output
//! index `i` *is* `CodecFamily::ALL[i]`, so the model must score exactly
//! [`CodecFamily::COUNT`] outputs. A `zenpicker-train` meta-picker does
//! not: its cells are `family × {lossy, lossless}` pairs, declared by
//! name in the bake's own metadata, and there are as many as the
//! training corpus had (7 for `metapicker_v1` — the lossless families
//! are a subset of the lossy ones). Feeding such a bake to `pick` would
//! read `CodecFamily::ALL[idx]` off a cell index and mis-map every
//! output. So the cell layout gets its own type, which **refuses**
//! anything that does not declare a well-formed contract.
//!
//! ## The contract
//!
//! Three UTF-8 metadata keys, all written by `zenpicker-train`:
//!
//! | key | shape |
//! |---|---|
//! | [`CELL_LABELS_KEY`] | one `<family>_<mode>` label per output |
//! | [`IMAGE_FEATURE_NAMES_KEY`] | the source-feature names the model consumes |
//! | [`INPUT_ORDER_KEY`] | the full input vector's names — every image feature exactly once, plus [`ZQ_NORM_INPUT`] exactly once |
//!
//! [`CellContract::from_model`] validates all of it against the parsed
//! model's real widths and refuses on any mismatch;
//! [`CellContract::build_input`] is the one mapping that turns named
//! source features + a normalized quality target into the model's input
//! vector, reading each contract name exactly once and nothing else.
//!
//! ## Inert
//!
//! Nothing here is reachable from [`default_route`](crate::default_route),
//! [`MetaPicker::route`](crate::MetaPicker::route), or
//! [`MetaPicker::default_routers`](crate::MetaPicker::default_routers) —
//! the shipped routers are untouched. A caller opts in explicitly by
//! constructing a [`CellPicker`].

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use zenpredict::argmin::{AllowedMask, ScoreTransform};
use zenpredict::{Model, Predictor};

use crate::{AllowedFamilies, CodecFamily, MetaPicker, MetaPickerError};

/// Metadata key: the per-output cell labels, `\n`- or `,`-separated
/// `<family>_<mode>` (e.g. `zenavif_lossy`). Written by `zenpicker-train`.
pub const CELL_LABELS_KEY: &str = "zenpicker_train.cell_labels";

/// Metadata key: the model's source-feature names, `\n`- or
/// `,`-separated, in the order the trainer read them.
pub const IMAGE_FEATURE_NAMES_KEY: &str = "zenpicker_train.image_feature_names";

/// Metadata key: the FULL input-vector names in input order —
/// [`IMAGE_FEATURE_NAMES_KEY`] plus [`ZQ_NORM_INPUT`].
pub const INPUT_ORDER_KEY: &str = "zenpicker_train.input_order";

/// The one non-image input: the caller's requested quality target
/// divided by 100. The codec's per-encode `q` is **not** an input — `q`
/// is the decision the picker informs, so there is no q-leakage.
pub const ZQ_NORM_INPUT: &str = "zq_norm";

/// Lossy or lossless — the second axis of a cell label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum CellMode {
    Lossy,
    Lossless,
}

impl CellMode {
    /// The label this mode carries in a cell name.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lossy => "lossy",
            Self::Lossless => "lossless",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "lossy" => Some(Self::Lossy),
            "lossless" => Some(Self::Lossless),
            _ => None,
        }
    }
}

/// One output cell: a [`CodecFamily`] paired with a [`CellMode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyModeCell {
    family: CodecFamily,
    mode: CellMode,
    label: String,
}

impl FamilyModeCell {
    /// The codec family this cell routes to.
    #[inline]
    pub fn family(&self) -> CodecFamily {
        self.family
    }

    /// Lossy or lossless.
    #[inline]
    pub fn mode(&self) -> CellMode {
        self.mode
    }

    /// The label verbatim as the bake declared it (e.g. `zenavif_lossy`).
    #[inline]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Parse one `<family>_<mode>` label. The family half accepts the
    /// crate name (`zenavif`) or the bare [`CodecFamily::label`]
    /// (`avif`) — `zenpicker-train` writes the crate name.
    fn parse(label: &str) -> Option<Self> {
        let (fam_raw, mode_raw) = label.rsplit_once('_')?;
        let mode = CellMode::parse(mode_raw)?;
        let bare = fam_raw.strip_prefix("zen").unwrap_or(fam_raw);
        let family = CodecFamily::ALL.into_iter().find(|f| f.label() == bare)?;
        Some(Self {
            family,
            mode,
            label: label.to_string(),
        })
    }
}

/// Split a `\n`- or `,`-separated metadata list the way
/// [`Model::feature_columns`] does.
fn split_list(raw: &str) -> Vec<&str> {
    raw.split(['\n', ','])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn read_list<'m>(model: &'m Model, key: &str) -> Result<Vec<&'m str>, MetaPickerError> {
    let raw = model
        .metadata()
        .get_utf8(key)
        .map_err(|e| MetaPickerError::CellContract(format!("{key}: {e:?}")))?;
    let items = split_list(raw);
    if items.is_empty() {
        return Err(MetaPickerError::CellContract(format!("{key}: empty")));
    }
    Ok(items)
}

/// The validated family×mode contract a cell-layout bake declares.
///
/// Built by [`from_model`](Self::from_model), which refuses a model whose
/// declared names do not match its real input/output widths, whose cell
/// labels do not parse, or whose input order is not exactly "every image
/// feature once, plus `zq_norm` once".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellContract {
    cells: Vec<FamilyModeCell>,
    image_features: Vec<String>,
    input_order: Vec<String>,
    zq_index: usize,
}

impl CellContract {
    /// Read + validate the contract from a parsed bake.
    ///
    /// Checks, in order — each failure is a
    /// [`CellContract`](MetaPickerError::CellContract) error naming what
    /// disagreed:
    ///
    /// 1. all three metadata keys are present and non-empty;
    /// 2. every cell label parses to a known `(family, mode)` and no cell
    ///    repeats;
    /// 3. the cell count equals the model's `n_outputs`;
    /// 4. the image-feature names are unique and none of them is
    ///    [`ZQ_NORM_INPUT`];
    /// 5. the input order holds every image feature **exactly once** plus
    ///    exactly one [`ZQ_NORM_INPUT`], and nothing else;
    /// 6. the input order's length equals the model's
    ///    [`caller_input_width`](Model::caller_input_width) — the width a
    ///    caller must supply, which is not `n_inputs` on a
    ///    dead-column-pruned bake.
    pub fn from_model(model: &Model) -> Result<Self, MetaPickerError> {
        let mut cells = Vec::new();
        for label in read_list(model, CELL_LABELS_KEY)? {
            let cell = FamilyModeCell::parse(label).ok_or_else(|| {
                MetaPickerError::CellContract(format!(
                    "{CELL_LABELS_KEY}: cell {label:?} is not <family>_<lossy|lossless> \
                     over {{{ALL}}}",
                    ALL = crate::ALL_LABELS_CSV
                ))
            })?;
            if cells
                .iter()
                .any(|c: &FamilyModeCell| c.family == cell.family && c.mode == cell.mode)
            {
                return Err(MetaPickerError::CellContract(format!(
                    "{CELL_LABELS_KEY}: cell {label:?} repeats an earlier (family, mode)"
                )));
            }
            cells.push(cell);
        }
        if cells.len() != model.n_outputs() {
            return Err(MetaPickerError::CellContract(format!(
                "{CELL_LABELS_KEY}: {} cells but the model scores {} outputs",
                cells.len(),
                model.n_outputs()
            )));
        }

        let image_features: Vec<String> = read_list(model, IMAGE_FEATURE_NAMES_KEY)?
            .into_iter()
            .map(str::to_string)
            .collect();
        for (i, name) in image_features.iter().enumerate() {
            if name == ZQ_NORM_INPUT {
                return Err(MetaPickerError::CellContract(format!(
                    "{IMAGE_FEATURE_NAMES_KEY}: {ZQ_NORM_INPUT} is the quality input, \
                     not a source feature"
                )));
            }
            if image_features[..i].contains(name) {
                return Err(MetaPickerError::CellContract(format!(
                    "{IMAGE_FEATURE_NAMES_KEY}: {name:?} appears more than once"
                )));
            }
        }

        let input_order: Vec<String> = read_list(model, INPUT_ORDER_KEY)?
            .into_iter()
            .map(str::to_string)
            .collect();
        let width = model.caller_input_width();
        if input_order.len() != width {
            return Err(MetaPickerError::CellContract(format!(
                "{INPUT_ORDER_KEY}: {} names but the model takes {width} inputs",
                input_order.len()
            )));
        }
        if input_order.len() != image_features.len() + 1 {
            return Err(MetaPickerError::CellContract(format!(
                "{INPUT_ORDER_KEY}: {} names for {} image features + {ZQ_NORM_INPUT}",
                input_order.len(),
                image_features.len()
            )));
        }
        // Exactly one zq_norm, and every image feature exactly once —
        // the bijection `build_input` relies on.
        let mut zq_index = None;
        let mut seen = alloc::vec![0usize; image_features.len()];
        for (i, name) in input_order.iter().enumerate() {
            if name == ZQ_NORM_INPUT {
                if zq_index.replace(i).is_some() {
                    return Err(MetaPickerError::CellContract(format!(
                        "{INPUT_ORDER_KEY}: {ZQ_NORM_INPUT} appears more than once"
                    )));
                }
                continue;
            }
            match image_features.iter().position(|f| f == name) {
                Some(k) => seen[k] += 1,
                None => {
                    return Err(MetaPickerError::CellContract(format!(
                        "{INPUT_ORDER_KEY}: {name:?} is not in {IMAGE_FEATURE_NAMES_KEY}"
                    )));
                }
            }
        }
        let Some(zq_index) = zq_index else {
            return Err(MetaPickerError::CellContract(format!(
                "{INPUT_ORDER_KEY}: no {ZQ_NORM_INPUT} input"
            )));
        };
        if let Some(k) = seen.iter().position(|&n| n != 1) {
            return Err(MetaPickerError::CellContract(format!(
                "{INPUT_ORDER_KEY}: {:?} appears {} times, expected exactly once",
                image_features[k], seen[k]
            )));
        }

        Ok(Self {
            cells,
            image_features,
            input_order,
            zq_index,
        })
    }

    /// The output cells, in output-index order.
    #[inline]
    pub fn cells(&self) -> &[FamilyModeCell] {
        &self.cells
    }

    /// The source-feature names the model consumes (no `zq_norm`).
    #[inline]
    pub fn image_features(&self) -> &[String] {
        &self.image_features
    }

    /// Every input's name in input order (image features + `zq_norm`).
    #[inline]
    pub fn input_order(&self) -> &[String] {
        &self.input_order
    }

    /// The position of [`ZQ_NORM_INPUT`] within [`input_order`](Self::input_order).
    #[inline]
    pub fn zq_index(&self) -> usize {
        self.zq_index
    }

    /// The families this contract can route to (deduplicated, declared order).
    pub fn families(&self) -> AllowedFamilies {
        AllowedFamilies::from_allowed(self.cells.iter().map(FamilyModeCell::family))
    }

    /// **The contract mapping.** Materialize the model's input vector
    /// from named source features plus a normalized quality target.
    ///
    /// Walks [`input_order`](Self::input_order) once and, for each name,
    /// either places `zq_norm` (at [`zq_index`](Self::zq_index)) or calls
    /// `source` with that name. Because the contract was validated as a
    /// bijection at load, **each source-feature name is passed to
    /// `source` exactly once, `zq_norm` is placed exactly once, and no
    /// name outside the contract is ever requested.**
    ///
    /// `source` returns `None` for a feature it cannot supply, which is a
    /// [`CellContract`](MetaPickerError::CellContract) error naming the
    /// missing feature — never a silent zero.
    pub fn build_input<F>(&self, zq_norm: f32, mut source: F) -> Result<Vec<f32>, MetaPickerError>
    where
        F: FnMut(&str) -> Option<f32>,
    {
        let mut out = Vec::with_capacity(self.input_order.len());
        for name in &self.input_order {
            if out.len() == self.zq_index {
                out.push(zq_norm);
                continue;
            }
            let v = source(name).ok_or_else(|| {
                MetaPickerError::CellContract(format!("source feature {name:?} not supplied"))
            })?;
            out.push(v);
        }
        Ok(out)
    }
}

/// A meta-picker bake loaded against the family×mode cell contract.
///
/// Owns the parsed [`Model`] (zenpredict copies the bake into its own
/// buffer, so any byte slice works) and the validated
/// [`CellContract`]. Construction **refuses** a bake that does not
/// declare a well-formed contract, so a family-score bake — or a
/// pairwise router — can never be read as cells.
pub struct CellPicker {
    model: Model,
    contract: CellContract,
}

impl CellPicker {
    /// Parse ZNPR bytes and validate the cell contract.
    pub fn from_znpr_bytes(bytes: &[u8]) -> Result<Self, MetaPickerError> {
        let model = Model::from_bytes(bytes).map_err(MetaPickerError::Predict)?;
        Self::from_model(model)
    }

    /// Parse ZNPR bytes, checking the bake's `schema_hash` against
    /// `expected` **before** any section parsing, then validate the cell
    /// contract. Use this when the caller compiles in the hash of the
    /// bake it was built against.
    pub fn from_znpr_bytes_with_schema(
        bytes: &[u8],
        expected_schema: u64,
    ) -> Result<Self, MetaPickerError> {
        let model = Model::from_bytes_with_schema(bytes, expected_schema)
            .map_err(MetaPickerError::Predict)?;
        Self::from_model(model)
    }

    fn from_model(model: Model) -> Result<Self, MetaPickerError> {
        let contract = CellContract::from_model(&model)?;
        Ok(Self { model, contract })
    }

    /// The validated contract.
    #[inline]
    pub fn contract(&self) -> &CellContract {
        &self.contract
    }

    /// The parsed model — for metadata, widths, or a caller-owned
    /// [`Predictor`].
    #[inline]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Borrow this bake as a [`MetaPicker`] — the bridge into the
    /// existing runtime type for metadata reads and diagnostics.
    ///
    /// Note that [`MetaPicker::pick`] is **not** meaningful on a cell
    /// bake (it would read `CodecFamily::ALL[cell_index]`); use
    /// [`predict_cells`](Self::predict_cells) for the routing decision.
    #[inline]
    pub fn meta_picker(&self) -> MetaPicker<'_> {
        MetaPicker::new(&self.model)
    }

    /// **The predict entry.** One forward pass → every cell's score plus
    /// the masked argmin cell.
    ///
    /// The model predicts `bytes_log` per cell (smaller is better), so
    /// the pick is an **argmin** over the cells that survive both masks:
    ///
    /// - `allowed` — the caller's format mask, applied per cell via the
    ///   cell's [`family`](FamilyModeCell::family);
    /// - `reachable` — an optional per-cell mask (one `bool` per cell, in
    ///   output order) for cells that can hit the requested quality.
    ///   `None` treats every cell as reachable. This is the same mask the
    ///   trainer's held-out panel argmins over.
    ///
    /// `input` must be [`CellContract::build_input`]'s output (or the
    /// same layout); a wrong width is a
    /// [`Predict`](MetaPickerError::Predict) error, never a silent
    /// prefix read. Feature transforms are applied when the bake carries
    /// non-trivial ones — the same branch the trainer's evaluator takes.
    ///
    /// `Ok(pred)` with `pred.pick() == None` when every cell is masked out.
    ///
    /// Builds one [`Predictor`]'s scratch per call — the same cost shape
    /// [`crate::default_route`] documents. A meta-pick happens once per
    /// image, so this is not a hot loop; if it ever becomes one, hold the
    /// [`Predictor`] yourself over [`model`](Self::model).
    pub fn predict_cells<'c>(
        &'c self,
        input: &[f32],
        allowed: &AllowedFamilies,
        reachable: Option<&[bool]>,
    ) -> Result<CellPrediction<'c>, MetaPickerError> {
        let n_cells = self.contract.cells.len();
        if let Some(r) = reachable
            && r.len() != n_cells
        {
            return Err(MetaPickerError::CellContract(format!(
                "reachable mask has {} entries, expected {n_cells}",
                r.len()
            )));
        }
        let mut predictor = Predictor::new(&self.model);
        let scores: Vec<f32> = if self.model.has_nontrivial_feature_transforms() {
            predictor.predict_transformed(input)
        } else {
            predictor.predict(input)
        }
        .map_err(MetaPickerError::Predict)?
        .to_vec();
        if scores.len() != n_cells {
            return Err(MetaPickerError::OutputShape {
                expected: n_cells,
                got: scores.len(),
            });
        }

        let mask_flags: Vec<bool> = self
            .contract
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| allowed.is_allowed(c.family) && reachable.is_none_or(|r| r[i]))
            .collect();
        let mask = AllowedMask::new(&mask_flags);
        let pick =
            zenpredict::argmin::argmin_masked(&scores, &mask, ScoreTransform::Identity, None)
                .map(|i| &self.contract.cells[i]);
        Ok(CellPrediction { scores, pick })
    }
}

/// One cell-layout forward pass: every cell's predicted `bytes_log` plus
/// the masked argmin cell.
#[derive(Debug, Clone, PartialEq)]
pub struct CellPrediction<'c> {
    scores: Vec<f32>,
    pick: Option<&'c FamilyModeCell>,
}

impl<'c> CellPrediction<'c> {
    /// Predicted `bytes_log` per cell, in output order (smaller is better).
    #[inline]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// The winning cell, or `None` when every cell was masked out.
    #[inline]
    pub fn pick(&self) -> Option<&'c FamilyModeCell> {
        self.pick
    }

    /// The winning cell's family — the routing decision.
    #[inline]
    pub fn family(&self) -> Option<CodecFamily> {
        self.pick.map(FamilyModeCell::family)
    }

    /// The winning cell's mode.
    #[inline]
    pub fn mode(&self) -> Option<CellMode> {
        self.pick.map(FamilyModeCell::mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_labels_parse_family_and_mode() {
        let c = FamilyModeCell::parse("zenavif_lossy").expect("zenavif_lossy");
        assert_eq!(c.family(), CodecFamily::Avif);
        assert_eq!(c.mode(), CellMode::Lossy);
        assert_eq!(c.label(), "zenavif_lossy");

        // the bare family label is accepted too
        let c = FamilyModeCell::parse("png_lossless").expect("png_lossless");
        assert_eq!(c.family(), CodecFamily::Png);
        assert_eq!(c.mode(), CellMode::Lossless);

        // every family x mode the enum knows
        for fam in CodecFamily::ALL {
            for mode in [CellMode::Lossy, CellMode::Lossless] {
                let label = format!("zen{}_{}", fam.label(), mode.label());
                let c = FamilyModeCell::parse(&label).expect("well-formed label");
                assert_eq!((c.family(), c.mode()), (fam, mode));
            }
        }
    }

    #[test]
    fn malformed_cell_labels_are_refused() {
        for bad in [
            "zenavif",          // no mode
            "zenavif_lossyish", // unknown mode
            "zenheic_lossy",    // unknown family
            "_lossy",           // empty family
            "zenavif_",         // empty mode
        ] {
            assert!(
                FamilyModeCell::parse(bad).is_none(),
                "{bad:?} must not parse as a cell"
            );
        }
    }

    // The shipped routers are FAMILY-score / pairwise bakes with no cell
    // metadata: reading one as a cell contract must fail loudly rather
    // than mis-map output indices onto CodecFamily::ALL.
    #[cfg(feature = "std")]
    #[test]
    fn shipped_routers_are_refused_as_cell_bakes() {
        for (name, bytes) in [
            ("lossy", crate::ROUTER_LOSSY),
            ("lossless", crate::ROUTER_LOSSLESS),
            ("gate", crate::ROUTER_GATE),
        ] {
            let err = CellPicker::from_znpr_bytes(bytes)
                .err()
                .unwrap_or_else(|| panic!("{name} router must not load as a cell bake"));
            assert!(
                matches!(err, MetaPickerError::CellContract(_)),
                "{name}: expected a CellContract error, got {err:?}"
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn wiring_is_inert_for_the_shipped_routers() {
        // default_routers() still builds and still validates the family
        // order — nothing in this module perturbs the shipped path.
        let mut mp = MetaPicker::default_routers();
        assert!(mp.predictor().model().n_outputs() > 0);
    }
}
