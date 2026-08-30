//! The `metapicker_v1` bake against the family×mode cell contract —
//! including the registered **touch-once** test: every one of the bake's
//! source features and `zq_norm` is consumed **exactly once** through the
//! contract mapping, and nothing outside the contract is ever read.
//!
//! ## Locating the bake
//!
//! The bake is 104 KB, so it lives in block storage and **never** in git
//! (`benchmarks/metapicker_v1_2026-08-30.pointer.md` is the tracked
//! reference). These tests read `ZENPICKER_METAPICKER_V1_BAKE` — a path to
//! the `.bin` — and **fail loudly when it is unset or unreadable**. There
//! is no runtime self-skip: the skip decision belongs to the caller and is
//! visible in the whole chain —
//!
//! - `just metapicker-v1-test [bake=<path>]` runs them, and
//! - CI (which has no block storage) passes `--skip metapicker_v1_`
//!   explicitly.
//!
//! A test that silently passes without testing anything is worse than one
//! that loudly fails.

use zenpicker::{
    AllowedFamilies, CellMode, CellPicker, CodecFamily, MetaPickerError, ZQ_NORM_INPUT,
};

/// The env var that points at the bake. Unset ⇒ these tests FAIL.
const BAKE_ENV: &str = "ZENPICKER_METAPICKER_V1_BAKE";

/// The registered v1 contract (campaign ledger, criterion 8 +
/// `benchmarks/metapicker_v1_2026-08-30.pointer.md`): 7 family×mode cells,
/// declared order.
const EXPECTED_CELLS: [(&str, CodecFamily, CellMode); 7] = [
    ("zenavif_lossy", CodecFamily::Avif, CellMode::Lossy),
    ("zenjpeg_lossy", CodecFamily::Jpeg, CellMode::Lossy),
    ("zenjxl_lossless", CodecFamily::Jxl, CellMode::Lossless),
    ("zenjxl_lossy", CodecFamily::Jxl, CellMode::Lossy),
    ("zenpng_lossless", CodecFamily::Png, CellMode::Lossless),
    ("zenwebp_lossless", CodecFamily::Webp, CellMode::Lossless),
    ("zenwebp_lossy", CodecFamily::Webp, CellMode::Lossy),
];

/// 61 source features ⊕ `zq_norm` = 62 inputs (the registered contract).
const EXPECTED_N_IMAGE_FEATURES: usize = 61;
const EXPECTED_N_INPUTS: usize = 62;

fn load() -> CellPicker {
    let path = std::env::var(BAKE_ENV).unwrap_or_else(|_| {
        panic!(
            "{BAKE_ENV} is not set. Point it at the metapicker_v1 bake (104 KB, block \
             storage — see benchmarks/metapicker_v1_2026-08-30.pointer.md), e.g. \
             {BAKE_ENV}=/mnt/v/output/zensim/metapicker-2026-08-30/metapicker_v1.bin, \
             or `just metapicker-v1-test`. To run the rest of the suite without it, skip \
             these explicitly with `-- --skip metapicker_v1_`."
        )
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{BAKE_ENV}={path}: {e}"));
    CellPicker::from_znpr_bytes(&bytes)
        .unwrap_or_else(|e| panic!("{path}: does not satisfy the cell contract: {e}"))
}

/// A plausible, deterministic feature value derived from the name (FNV-1a
/// folded into 24 bits, so every value is exactly representable in `f32`).
/// Used where only *some* well-formed input is needed. The touch-once test
/// does NOT use this — it needs a source that is injective by construction,
/// see `injective_source_value`.
fn value_of(name: &str) -> f32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    ((h >> 40) as u32) as f32
}

#[test]
fn metapicker_v1_contract_is_the_registered_shape() {
    let picker = load();
    let c = picker.contract();

    assert_eq!(
        c.image_features().len(),
        EXPECTED_N_IMAGE_FEATURES,
        "registered contract is {EXPECTED_N_IMAGE_FEATURES} source features"
    );
    assert_eq!(
        c.input_order().len(),
        EXPECTED_N_INPUTS,
        "registered contract is {EXPECTED_N_IMAGE_FEATURES} source features + zq_norm"
    );
    assert_eq!(
        picker.model().caller_input_width(),
        EXPECTED_N_INPUTS,
        "the width a caller must supply must equal the declared input order"
    );
    // v1 is not dead-column pruned, so every declared input actually reaches
    // the net. (A pruned bake would keep caller_input_width at 62 and forward
    // fewer — legal, and the contract would still hold — but it is not what
    // this bake is, and saying so pins the property.)
    assert_eq!(
        picker.model().n_inputs(),
        picker.model().caller_input_width(),
        "metapicker_v1 is expected to be unpruned: every declared input is forwarded"
    );

    let got: Vec<(&str, CodecFamily, CellMode)> = c
        .cells()
        .iter()
        .map(|x| (x.label(), x.family(), x.mode()))
        .collect();
    assert_eq!(
        got,
        EXPECTED_CELLS.to_vec(),
        "cell labels / (family, mode) mapping drifted from the registered v1 contract"
    );

    // The reason this adapter exists: the cell count is NOT the family
    // count, so MetaPicker::pick (which reads CodecFamily::ALL[idx]) is
    // structurally wrong for this bake.
    assert_ne!(
        picker.model().n_outputs(),
        CodecFamily::COUNT,
        "if these ever coincide, re-check that pick() is still refused for cell bakes"
    );

    // Every family the contract can route to is a real CodecFamily, and
    // `families()` is their deduplicated set.
    let fams = c.families();
    for (_, fam, _) in EXPECTED_CELLS {
        assert!(fams.is_allowed(fam), "{fam:?} missing from families()");
    }
    assert!(
        !fams.is_allowed(CodecFamily::Gif),
        "v1 has no gif cell; families() must not claim one"
    );
}

/// **THE TOUCH-ONCE TEST.** Build the model's input vector through the
/// contract mapping with an instrumented source, then assert the mapping is
/// a bijection: each of the 61 source features is requested exactly once,
/// `zq_norm` is placed exactly once and never requested from the source,
/// nothing outside the contract is read, and every slot holds the value
/// belonging to the name declared at that slot.
#[test]
fn metapicker_v1_touch_once_every_contract_input_consumed_exactly_once() {
    let picker = load();
    let c = picker.contract();

    // The source value is the feature's position in the declared feature
    // list (+1). Injective BY CONSTRUCTION — a position is unique — so the
    // positional check at the end genuinely detects a mis-ordered mapping,
    // with no hash-collision luck involved. A name outside the contract has
    // no position, so the source returns None for it and `build_input`
    // turns that into a loud error.
    let injective_source_value = |name: &str| -> Option<f32> {
        c.image_features()
            .iter()
            .position(|f| f == name)
            .map(|k| (k as f32) + 1.0)
    };

    let zq: f32 = 0.82;
    let mut asked: Vec<String> = Vec::new();
    let input = c
        .build_input(zq, |name| {
            asked.push(name.to_string());
            injective_source_value(name)
        })
        .expect("the contract mapping must build the input vector");

    // 1. Width — exactly the declared input order, nothing padded or dropped.
    assert_eq!(input.len(), c.input_order().len());
    assert_eq!(input.len(), picker.model().caller_input_width());

    // 2. TOUCH-ONCE: every declared source feature requested exactly once.
    for f in c.image_features() {
        let n = asked.iter().filter(|a| a.as_str() == f.as_str()).count();
        assert_eq!(n, 1, "source feature {f:?} was requested {n} times, want 1");
    }

    // 3. NOTHING OUTSIDE THE CONTRACT was read — including zq_norm, which
    //    the mapping places itself and must never ask the source for.
    for a in &asked {
        assert!(
            c.image_features().iter().any(|f| f == a),
            "the mapping requested {a:?}, which is not a contract source feature"
        );
        assert_ne!(
            a.as_str(),
            ZQ_NORM_INPUT,
            "zq_norm is placed by the contract, never requested from the source"
        );
    }

    // 4. Total request count == feature count ⇒ a bijection (2 + 3 + 4
    //    together leave no room for a duplicate, an omission, or an extra).
    assert_eq!(
        asked.len(),
        c.image_features().len(),
        "the mapping made {} requests for {} source features",
        asked.len(),
        c.image_features().len()
    );

    // 5. zq_norm appears exactly once in the declared order, at zq_index.
    let zq_slots: Vec<usize> = c
        .input_order()
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_str() == ZQ_NORM_INPUT)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(zq_slots, vec![c.zq_index()], "zq_norm must appear once");

    // 6. POSITION: every slot carries the value of the name declared there.
    for (i, name) in c.input_order().iter().enumerate() {
        if i == c.zq_index() {
            assert_eq!(input[i], zq, "slot {i} must hold zq_norm");
        } else {
            assert_eq!(
                Some(input[i]),
                injective_source_value(name),
                "slot {i} holds the wrong feature's value (declared {name:?})"
            );
        }
    }
}

#[test]
fn metapicker_v1_missing_source_feature_fails_loudly() {
    let picker = load();
    let c = picker.contract();
    let drop_me = c.image_features()[7].clone();

    let err = c
        .build_input(0.82, |name| {
            (name != drop_me.as_str()).then(|| value_of(name))
        })
        .expect_err("a source that cannot supply a contract feature must error");
    match err {
        MetaPickerError::CellContract(msg) => assert!(
            msg.contains(&drop_me),
            "the error must name the missing feature; got {msg:?}"
        ),
        other => panic!("expected a CellContract error, got {other:?}"),
    }
}

#[test]
fn metapicker_v1_forward_pass_and_masked_argmin() {
    let picker = load();
    let c = picker.contract();
    let n_cells = c.cells().len();
    let input = c
        .build_input(0.82, |name| Some(value_of(name)))
        .expect("input vector");

    // Unmasked over the contract's own families: 7 finite scores, one pick.
    let pred = picker
        .predict_cells(&input, &c.families(), None)
        .expect("forward pass");
    assert_eq!(pred.scores().len(), n_cells);
    assert!(
        pred.scores().iter().all(|s| s.is_finite()),
        "cell scores must be finite: {:?}",
        pred.scores()
    );
    let pick = pred.pick().expect("an unmasked pick");
    assert!(c.cells().contains(pick));
    // The pick IS the argmin over the allowed cells.
    let best = c
        .cells()
        .iter()
        .enumerate()
        .filter(|(_, cell)| c.families().is_allowed(cell.family()))
        .min_by(|a, b| pred.scores()[a.0].partial_cmp(&pred.scores()[b.0]).unwrap())
        .map(|(i, _)| i)
        .expect("some allowed cell");
    assert_eq!(
        pred.pick().map(|p| p.label()),
        Some(c.cells()[best].label())
    );

    // A one-family mask can only ever pick that family.
    for fam in [CodecFamily::Avif, CodecFamily::Jpeg, CodecFamily::Webp] {
        let only = AllowedFamilies::none().allow(fam);
        let p = picker
            .predict_cells(&input, &only, None)
            .expect("masked forward pass");
        assert_eq!(
            p.family(),
            Some(fam),
            "mask to {fam:?} picked something else"
        );
    }

    // A family with no cell in this contract ⇒ nothing to pick.
    let gif_only = AllowedFamilies::none().allow(CodecFamily::Gif);
    assert!(
        picker
            .predict_cells(&input, &gif_only, None)
            .expect("forward pass")
            .pick()
            .is_none(),
        "v1 has no gif cell, so a gif-only mask must pick nothing"
    );

    // Empty mask ⇒ no pick, still no error.
    assert!(
        picker
            .predict_cells(&input, &AllowedFamilies::none(), None)
            .expect("forward pass")
            .pick()
            .is_none()
    );

    // The reach mask is honoured per cell (the same mask the trainer's
    // held-out panel argmins over): allow exactly one cell and it must win.
    for target in 0..n_cells {
        let mut reach = vec![false; n_cells];
        reach[target] = true;
        let p = picker
            .predict_cells(&input, &AllowedFamilies::all(), Some(&reach))
            .expect("forward pass");
        assert_eq!(
            p.pick().map(|x| x.label()),
            Some(c.cells()[target].label()),
            "reach mask allowing only cell {target} must pick it"
        );
    }

    // A wrong-width reach mask is refused, never silently truncated.
    let short = vec![true; n_cells - 1];
    assert!(matches!(
        picker.predict_cells(&input, &AllowedFamilies::all(), Some(&short)),
        Err(MetaPickerError::CellContract(_))
    ));

    // A wrong-width input is refused, never read as a prefix.
    let mut short_input = input.clone();
    short_input.pop();
    assert!(
        picker
            .predict_cells(&short_input, &c.families(), None)
            .is_err(),
        "a short input vector must error, not score a prefix"
    );
}

#[test]
fn metapicker_v1_schema_gate_refuses_a_wrong_hash() {
    let path = std::env::var(BAKE_ENV).unwrap_or_else(|_| {
        panic!("{BAKE_ENV} is not set (see the module docs / `just metapicker-v1-test`)")
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{BAKE_ENV}={path}: {e}"));

    let real = CellPicker::from_znpr_bytes(&bytes)
        .expect("loads")
        .model()
        .schema_hash();
    assert!(
        CellPicker::from_znpr_bytes_with_schema(&bytes, real).is_ok(),
        "the bake's own schema hash must pass the gate"
    );
    assert!(
        CellPicker::from_znpr_bytes_with_schema(&bytes, real ^ 0xdead_beef).is_err(),
        "a wrong schema hash must be refused at load"
    );
}

/// The bake's declared source-feature names are **positional placeholders**
/// (`feat_0..feat_60`), so the bake alone does not say which zenanalyze
/// feature belongs in which slot — the identity was dropped upstream by the
/// meta-input builder. `benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv`
/// recovers it. This test keeps the two in lockstep: one recovered row per
/// declared slot, in the same order, each naming a well-formed
/// zenanalyze-api qualified identity — so a caller can resolve the contract
/// mapping against a real `Offer`.
#[test]
fn metapicker_v1_recovered_feature_slots_resolve_the_contract() {
    let picker = load();
    let c = picker.contract();
    let tsv = include_str!("../../benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv");

    let rows: Vec<(usize, &str, &str)> = tsv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .skip(1) // header
        .map(|l| {
            let mut f = l.split('\t');
            let slot: usize = f
                .next()
                .expect("slot")
                .trim()
                .parse()
                .expect("slot is a number");
            let input = f.next().expect("input_name").trim();
            let feat = f.next().expect("zenanalyze_feature").trim();
            (slot, input, feat)
        })
        .collect();

    assert_eq!(
        rows.len(),
        c.image_features().len(),
        "the recovered slot table and the bake's declared feature list disagree on length"
    );
    for (i, (slot, input, feat)) in rows.iter().enumerate() {
        assert_eq!(*slot, i, "slot column must be dense and in order");
        assert_eq!(
            *input,
            c.image_features()[i].as_str(),
            "slot {i}: recovered input_name does not match the bake's declared name"
        );
        // A zenanalyze-api qualified identity: `<name>@<8 lowercase hex>`.
        let (name, hash) = feat
            .rsplit_once('@')
            .unwrap_or_else(|| panic!("slot {i}: {feat:?} is not a qualified name@hex8"));
        assert!(!name.is_empty(), "slot {i}: empty feature name in {feat:?}");
        assert!(
            hash.len() == 8
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "slot {i}: {hash:?} is not 8 lowercase hex digits"
        );
        #[cfg(feature = "api")]
        assert!(
            zenanalyze_api::NamedFeature::parse(feat).is_some(),
            "slot {i}: zenanalyze_api::NamedFeature::parse rejected {feat:?}"
        );
    }

    // Every recovered zenanalyze name is distinct — a duplicate would mean two
    // slots claim the same analyzer feature.
    let mut seen: Vec<&str> = rows.iter().map(|(_, _, f)| *f).collect();
    seen.sort_unstable();
    let n = seen.len();
    seen.dedup();
    assert_eq!(
        seen.len(),
        n,
        "the recovered slot table names a feature twice"
    );

    // Documents the gap this file exists to close: the bake's OWN names carry
    // no analyzer identity. If a future bake ever declares qualified names
    // directly, these assertions are the signal to retire the recovery table.
    assert!(
        c.image_features().iter().all(|f| !f.contains('@')),
        "the bake now declares qualified feature names — retire \
         benchmarks/metapicker_v1_feature_slots_2026-08-30.tsv and read them from the bake"
    );
    // ... and it carries no `zentrain.feature_columns` either, so the
    // zenanalyze-api negotiation path sees nothing: `feature_columns()` is
    // empty and `MetaPicker::feature_request()` is None. That is exactly why
    // v1 cannot consume a shared `Offer` today.
    assert_eq!(
        picker.model().feature_columns().count(),
        0,
        "v1 is expected to carry no zentrain.feature_columns"
    );
    #[cfg(feature = "api")]
    {
        let mp = picker.meta_picker();
        assert!(
            mp.feature_request().is_none(),
            "with no qualified feature columns, feature_request() must be None \
             (never a vacuously-satisfied empty request)"
        );
    }
}
