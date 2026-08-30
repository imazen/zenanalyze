use super::*;
use alloc::string::ToString;
use alloc::vec::Vec;

// Build a NamedFeature from its qualified literal (no alloc — the literal IS the identity).
fn nf(qualified: &'static str) -> NamedFeature<'static> {
    NamedFeature::parse(qualified).expect("valid qualified literal")
}
fn sample() -> ([FeatureResult<'static>; 3], &'static str) {
    (
        [
            FeatureResult::new(nf("variance@11111111"), 0.5),
            FeatureResult::new(nf("edge_density@abcdef01"), 12.0),
            FeatureResult::new(nf("uniformity@00000009"), 0.9),
        ],
        "0.2.7",
    )
}

#[test]
fn name_validation() {
    assert!(NamedFeature::is_valid_name("variance"));
    assert!(NamedFeature::is_valid_name("chroma_luma_covariance_cb"));
    assert!(!NamedFeature::is_valid_name(""));
    assert!(!NamedFeature::is_valid_name("Variance")); // uppercase
    assert!(!NamedFeature::is_valid_name("edge-density")); // hyphen
    assert!(!NamedFeature::is_valid_name("a@b")); // @
}

#[test]
fn fold_is_deterministic_and_mixes() {
    assert_eq!(NamedFeature::fold_hash(0), 0);
    assert_eq!(NamedFeature::fold_hash(0x0000_0000_dead_beef), 0xdead_beef);
    assert_eq!(NamedFeature::fold_hash(0x1234_5678_0000_0000), 0x1234_5678);
    assert_eq!(NamedFeature::fold_hash(0x0000_0001_0000_0001), 0); // both halves participate
}

#[test]
fn named_feature_is_one_string_with_lazy_splits() {
    let f = nf("variance@b4a1c2d3");
    assert_eq!(f.qualified_name(), "variance@b4a1c2d3"); // THE identity
    assert_eq!(f.name(), "variance"); // lazy split, no alloc
    assert_eq!(f.version_hash(), 0xb4a1_c2d3); // lazy split, no alloc
    assert_eq!(f.to_string(), "variance@b4a1c2d3"); // Display = qualified
    assert_eq!(NamedFeature::parse("variance@b4a1c2d3"), Some(f));
    assert_eq!(NamedFeature::try_from("variance@b4a1c2d3").unwrap(), f);
    assert_eq!(NamedFeature::from_qualified("variance@b4a1c2d3"), f);
    assert_eq!(
        NamedFeature::qualified_for("uniformity", 9),
        "uniformity@00000009"
    );
    assert_eq!(nf("uniformity@00000009").version_hash(), 9);
}

#[test]
fn parse_is_const_and_strict() {
    // const, so a preset is validated at build time
    const PRESET: NamedFeature<'static> = match NamedFeature::parse("variance@b4a1c2d3") {
        Some(f) => f,
        None => panic!("valid preset"),
    };
    assert_eq!(PRESET.version_hash(), 0xb4a1_c2d3);
    // is_valid_name / fold_hash are const too (these items compile only if so)
    const _: bool = NamedFeature::is_valid_name("variance");
    const _: u32 = NamedFeature::fold_hash(0xdead_beef_0000_0000);

    assert_eq!(NamedFeature::parse("variance@B4A1C2D3"), None); // uppercase hex
    assert_eq!(NamedFeature::parse("variance@b4a1c2d"), None); // 7 digits
    assert_eq!(NamedFeature::parse("variance@b4a1c2d3e"), None); // 9 digits
    assert_eq!(NamedFeature::parse("variance@xyzxyzxy"), None); // non-hex
    assert_eq!(NamedFeature::parse("Variance@b4a1c2d3"), None); // bad name
    assert_eq!(NamedFeature::parse("variance"), None); // no @
}

#[test]
fn value_native_types_and_canonical_f32() {
    // canonical f32 projection matches zenanalyze's FeatureValue::to_f32
    assert_eq!(Value::from(0.5f32).to_f32(), 0.5);
    assert_eq!(Value::from(4096u32).to_f32(), 4096.0);
    assert_eq!(Value::from(true).to_f32(), 1.0);
    assert_eq!(Value::from(false).to_f32(), 0.0);
    let big = 1_000_000_000_000u64;
    assert_eq!(Value::from(big).to_f32(), big as f64 as f32); // via f64, near 2^53

    // FeatureResult preserves the native type, projects the canonical f32
    let r = FeatureResult::new(nf("pixel_count@00000001"), 4096u32);
    assert_eq!(r.float(), 4096.0); // canonical currency
    assert_eq!(r.value(), Value::U32(4096)); // native preserved
    let b = FeatureResult::new(nf("hdr_present@00000002"), true);
    assert_eq!(b.float(), 1.0);
    assert_eq!(b.value(), Value::Bool(true));
    // the f32-literal path still infers f32 (only float with Into<Value>)
    assert_eq!(
        FeatureResult::new(nf("variance@11111111"), 0.5).value(),
        Value::F32(0.5)
    );
}

#[test]
fn offer_get_reuse_satisfies_and_provenance() {
    let (feats, av) = sample();
    let offer = Offer::new(&feats, Provenance::new(av).with_descriptor(9));

    assert_eq!(offer.get("variance").map(|f| f.float()), Some(0.5));
    assert!(offer.get("absent").is_none());
    assert_eq!(offer.features().len(), 3);
    let p = offer.provenance();
    assert_eq!(p.analyzer_version(), "0.2.7");
    assert_eq!(p.config_hash(), 0);
    assert_eq!(p.descriptor_hash(), 9);
    assert_eq!(p, Provenance::new("0.2.7").with_descriptor(9));

    let wants = [nf("edge_density@abcdef01"), nf("variance@11111111")];
    let req = Request::new(Select::Features(&wants));
    assert!(matches!(req.select(), Select::Features(_)));
    assert!(offer.satisfies(&req));
    assert_eq!(offer.reuse_for(&req), Some(alloc::vec![12.0, 0.5]));

    let drift = [nf("variance@ffffffff")];
    let drift_req = Request::new(Select::Features(&drift));
    assert!(!offer.satisfies(&drift_req));
    assert_eq!(offer.reuse_for(&drift_req), None);

    let all = Request::new(Select::All);
    assert!(offer.satisfies(&all));
    assert_eq!(offer.reuse_for(&all), Some(alloc::vec![0.5, 12.0, 0.9]));
}

#[test]
fn get_by_name_classifies_a_reuse_miss() {
    // the failure rationale, derived from get() alone (no Reuse type) — by bare name,
    // compare versions: present-and-equal ⇒ reusable, present-but-different ⇒ drift, absent ⇒ missing
    let (feats, av) = sample();
    let offer = Offer::new(&feats, Provenance::new(av));

    let fresh = nf("variance@11111111");
    assert!(
        offer
            .get(fresh.name())
            .is_some_and(|f| f.feature().version_hash() == fresh.version_hash())
    ); // reusable

    let drift = nf("edge_density@deadbeef");
    assert!(
        offer
            .get(drift.name())
            .is_some_and(|f| f.feature().version_hash() != drift.version_hash())
    ); // present, code-version drift

    assert!(offer.get("noise_floor_y").is_none()); // missing
}

#[test]
fn schema_hash_is_order_independent_and_sensitive() {
    let (feats, av) = sample();
    let prov = Provenance::new(av).with_descriptor(9);
    let base = Offer::new(&feats, prov).schema_hash();

    let reordered = [feats[2], feats[0], feats[1]];
    assert_eq!(Offer::new(&reordered, prov).schema_hash(), base);

    let mut drift = feats;
    drift[0] = FeatureResult::new(nf("variance@99999999"), 0.5);
    assert_ne!(Offer::new(&drift, prov).schema_hash(), base);

    // a VALUE change alone does NOT move schema_hash (identity, not data)
    let mut reval = feats;
    reval[0] = FeatureResult::new(nf("variance@11111111"), 999.0);
    assert_eq!(Offer::new(&reval, prov).schema_hash(), base);

    assert_ne!(
        Offer::new(
            &feats,
            Provenance::new(av).with_config(1).with_descriptor(9)
        )
        .schema_hash(),
        base
    );
    assert_ne!(
        Offer::new(&feats, Provenance::new(av).with_descriptor(8)).schema_hash(),
        base
    );
}

#[test]
fn block_round_trips_native_values() {
    let feats = [
        FeatureResult::new(nf("variance@11111111"), 0.5),
        FeatureResult::new(nf("pixel_count@22222222"), 16_777_217u32), // 2^24+1, NOT f32-exact
        FeatureResult::new(nf("hdr_present@33333333"), true),
    ];
    let offer = Offer::new(&feats, Provenance::new("0.2.7").with_descriptor(9));
    let owned = OwnedOffer::parse(&offer.to_block()).expect("parse");

    assert_eq!(
        owned.provenance(),
        Provenance::new("0.2.7").with_descriptor(9)
    );
    // native type + full precision survives the text round trip
    assert_eq!(owned.get("variance").unwrap().value(), Value::F32(0.5));
    assert_eq!(
        owned.get("pixel_count").unwrap().value(),
        Value::U32(16_777_217)
    );
    assert_eq!(owned.get("hdr_present").unwrap().value(), Value::Bool(true));
    // …whereas the canonical f32 projection rounds 2^24+1 down — that's why native matters
    assert_eq!(owned.get("pixel_count").unwrap().float(), 16_777_216.0);
    assert_eq!(owned.features().len(), 3);
    assert_eq!(owned.schema_hash(), offer.schema_hash());
}

#[test]
fn owned_offer_negotiates_directly() {
    let (feats, av) = sample();
    let offer = Offer::new(&feats, Provenance::new(av));
    let owned = OwnedOffer::parse(&offer.to_block()).unwrap();

    // a deserialized offer (no original to re-run) negotiates exactly like the borrowed one
    let wants = [nf("variance@11111111"), nf("edge_density@abcdef01")];
    let req = Request::new(Select::Features(&wants));
    assert!(owned.satisfies(&req));
    assert_eq!(owned.reuse_for(&req), offer.reuse_for(&req));
    assert_eq!(owned.reuse_for(&req), Some(alloc::vec![0.5, 12.0]));

    // features() lends the owned cells zero-cost; as_ref bridges back to a borrowed Offer
    let cells = owned.features();
    let names: Vec<&str> = cells.iter().map(OwnedFeatureResult::name).collect();
    assert_eq!(names, alloc::vec!["variance", "edge_density", "uniformity"]);
    let frs: Vec<FeatureResult> = cells.iter().map(OwnedFeatureResult::as_ref).collect();
    assert_eq!(
        Offer::new(&frs, owned.provenance())
            .get("variance")
            .unwrap()
            .float(),
        0.5
    );
}

#[test]
fn owned_offer_from_parts() {
    // the parquet/TSV path: build from deserialized cells + provenance, no text block
    let cells = alloc::vec![
        OwnedFeatureResult::new("variance@11111111", 0.5),
        OwnedFeatureResult::new("pixel_count@22222222", 4096u32),
    ];
    let owned = OwnedOffer::new(cells, Provenance::new("0.2.7").with_descriptor(9));
    assert_eq!(
        owned.provenance(),
        Provenance::new("0.2.7").with_descriptor(9)
    );
    assert_eq!(owned.get("variance").unwrap().float(), 0.5);
    assert_eq!(owned.get("pixel_count").unwrap().value(), Value::U32(4096));
    // negotiates like any offer
    let wants = [nf("variance@11111111")];
    assert!(owned.satisfies(&Request::new(Select::Features(&wants))));
}

#[test]
fn owned_feature_is_the_owned_twin() {
    // own from parts (parsed/deserialized row)
    let of = OwnedFeatureResult::new("pixel_count@22222222", 16_777_217u32);
    assert_eq!(of.qualified_name(), "pixel_count@22222222");
    assert_eq!(of.name(), "pixel_count");
    assert_eq!(of.version_hash(), 0x2222_2222);
    assert_eq!(of.value(), Value::U32(16_777_217)); // native precision kept
    assert_eq!(of.float(), 16_777_216.0); // canonical f32 rounds 2^24+1

    // own a borrowed result, and lend it back via as_ref
    let borrowed = FeatureResult::new(nf("variance@11111111"), 0.5);
    let owned: OwnedFeatureResult = borrowed.into();
    assert_eq!(owned.as_ref(), borrowed);
}

#[test]
fn parse_rejects_bad_blocks() {
    assert_eq!(
        OwnedOffer::parse("garbage").unwrap_err(),
        FormatError::UnknownFormat
    );
    assert_eq!(
        OwnedOffer::parse("zenanalyze-features/1\nconfig_hash=0\n").unwrap_err(),
        FormatError::MissingHeader
    );
    // a malformed feature key is rejected
    assert_eq!(
        OwnedOffer::parse(
            "zenanalyze-features/1\nanalyzer_version=0.2.0\nconfig_hash=0\ndescriptor_hash=9\n\
             [features]\nvariance@xyz=0.5\n"
        )
        .unwrap_err(),
        FormatError::BadLine
    );
    // unknown headers are ignored forward-compatibly
    let p = OwnedOffer::parse(
        "zenanalyze-features/1\nanalyzer_version=0.2.0\nconfig_hash=0\ndescriptor_hash=9\n\
         schema_hash=123\nfuture=whatever\n[features]\nvariance@00000001=0.5\n",
    )
    .expect("ignores unknown headers");
    assert_eq!(p.get("variance").unwrap().float(), 0.5);
}

#[test]
fn catalog_offers_and_unionizes() {
    let avail = [nf("variance@11111111"), nf("edge_density@abcdef01")];
    let cat = Catalog::new(&avail);
    assert_eq!(cat.available().len(), 2);
    assert!(cat.offers(&nf("variance@11111111")));
    assert!(!cat.offers(&nf("variance@22222222"))); // present, wrong version
    assert!(cat.has_name("variance"));
    assert!(!cat.has_name("peak_luminance_nits")); // compiled out

    let wants = [nf("variance@22222222"), nf("peak_luminance_nits@55555555")];
    assert_eq!(
        cat.unmet(&wants),
        alloc::vec!["variance", "peak_luminance_nits"]
    );

    let b_feats = [nf("edge_density@00000001"), nf("uniformity@00000002")];
    let a = Request::new(Select::All);
    let b = Request::new(Select::Features(&b_feats));
    assert_eq!(
        cat.union(&[a, b]),
        alloc::vec!["variance", "edge_density", "uniformity"]
    );
}

#[test]
fn format_error_is_an_error() {
    let e: &dyn core::error::Error = &FormatError::BadLine;
    assert_eq!(alloc::format!("{e}"), "a malformed line");
}

// ─────────────────── Select::Names — version-agnostic matching ────────────────────

/// `Names` matches by bare name at whatever version the offer carries — the point of the
/// variant. The same want as a version-pinned `Features` MISSES on a code drift.
#[test]
fn names_select_matches_across_a_code_drift_where_features_misses() {
    let feats = [
        FeatureResult::new(nf("variance@11111111"), 0.5f32),
        FeatureResult::new(nf("edge_density@22222222"), 12.0f32),
    ];
    let offer = Offer::new(&feats, Provenance::new("0.2.7"));

    // Wanted at a DIFFERENT code version than the offer carries.
    let pinned = [nf("variance@ffffffff"), nf("edge_density@eeeeeeee")];
    let pinned_req = Request::new(Select::Features(&pinned));
    assert!(
        !offer.satisfies(&pinned_req),
        "a drift must miss when pinned"
    );
    assert!(offer.reuse_for(&pinned_req).is_none());

    // The same two features by bare name reuse regardless of version.
    let by_name = ["variance", "edge_density"];
    let name_req = Request::new(Select::Names(&by_name));
    assert!(offer.satisfies(&name_req));
    assert_eq!(offer.reuse_for(&name_req), Some(alloc::vec![0.5, 12.0]));

    // Order follows the request, and an absent name is still a miss (never a silent zero).
    let reordered = ["edge_density", "variance"];
    assert_eq!(
        offer.reuse_for(&Request::new(Select::Names(&reordered))),
        Some(alloc::vec![12.0, 0.5])
    );
    let absent = ["variance", "noise_floor_y"];
    assert!(
        offer
            .reuse_for(&Request::new(Select::Names(&absent)))
            .is_none()
    );
}

/// The owned twin negotiates `Names` identically, and `Catalog::union` resolves it.
#[test]
fn names_select_works_on_owned_offers_and_in_union() {
    let owned = OwnedOffer::new(
        alloc::vec![
            OwnedFeatureResult::new("variance@11111111", 0.5f32),
            OwnedFeatureResult::new("pixel_count@33333333", 4096u32),
        ],
        Provenance::new("0.2.7"),
    );
    let wants = ["pixel_count"];
    let req = Request::new(Select::Names(&wants));
    assert!(owned.satisfies(&req));
    assert_eq!(owned.reuse_for(&req), Some(alloc::vec![4096.0]));

    let available = [nf("variance@11111111"), nf("pixel_count@33333333")];
    let cat = Catalog::new(&available);
    let pinned = [nf("variance@11111111")];
    assert_eq!(
        cat.union(&[
            Request::new(Select::Names(&wants)),
            Request::new(Select::Features(&pinned)),
        ]),
        alloc::vec!["pixel_count", "variance"]
    );
}

// ─────────────────────── the intended model, end to end ────────────────────

/// **The whole flow, in 0.1.0 verbs plus `Select::Names`.** The host runs one pass and
/// *gives* the codec the data; the codec answers yes/no; on "no" it learns exactly which
/// wants were missing and runs its own scan.
///
/// This test is the reason `zenanalyze-api` has no extraction trait. Every step below is
/// `Offer` / `Request` / `Select` — data the host already has. A `&dyn` provider would let
/// the codec reach *back* into an analyzer to pull values, which inverts the direction of
/// control and is a step this flow never takes.
#[test]
fn push_model_answers_yes_no_and_names_the_gaps() {
    let (features, version) = sample();
    let offer = Offer::new(&features, Provenance::new(version));

    // ── yes ── everything the codec wants is present, at whatever version the host ran.
    let covered = ["variance", "edge_density"];
    let req = Request::new(Select::Names(&covered));
    assert!(offer.satisfies(&req), "the shared pass covered this codec");
    assert_eq!(offer.reuse_for(&req), Some(alloc::vec![0.5, 12.0]));

    // ── no ── one want is absent, so the answer is a clean miss, not a partial vector.
    let wants = ["variance", "skin_tone_fraction", "edge_density"];
    let req = Request::new(Select::Names(&wants));
    assert!(!offer.satisfies(&req));
    assert_eq!(
        offer.reuse_for(&req),
        None,
        "all-or-nothing: never a silent hole"
    );

    // ── which ones ── `Offer::get` (0.1.0) classifies every want without a re-run, so the
    // codec's own scan can be narrowed to exactly the gap. No catalog type is involved:
    // this is a question about an OFFER, not about what some build could produce.
    let missing: Vec<&str> = wants
        .iter()
        .copied()
        .filter(|w| offer.get(w).is_none())
        .collect();
    assert_eq!(missing, alloc::vec!["skin_tone_fraction"]);

    // A present-but-drifted want is a DIFFERENT answer than an absent one — `get` returns
    // `Some` and the version hash disagrees. That distinction is what lets a codec choose
    // between "re-scan" and "accept at another version".
    let drifted = [nf("variance@ffffffff")];
    assert!(!offer.satisfies(&Request::new(Select::Features(&drifted))));
    let cell = offer.get("variance").expect("present by name");
    assert_ne!(cell.feature().version_hash(), drifted[0].version_hash());
}
