//! Load `meta_picker_v0_4.bin` baked from coefficient's cross-codec
//! substrate (90,087 rows × 257 sources × 6 codec families:
//! avif/gif/jpeg/jxl/png/webp) and demonstrate a pick via the
//! high-level `MetaPicker` API.
//!
//! v0.4 was trained against a Spearman-pruned 68-feature schema
//! (down from 107 in v0.2/v0.3 — kitchen-sink hurt mean overhead).
//! Hidden 192×192×192 (the playbook's recommended size at this
//! scale). Student argmin_acc 87.8%, mean_overhead 17.73%.
//!
//! Run from the workspace root:
//!     cargo run -p zenpicker --example load_meta_picker_v0_1

use zenpicker::{AllowedFamilies, CodecFamily, MetaPicker};
use zenpredict::Model;

const BIN_PATH: &str =
    "/home/lilith/oracle-d2-store/oracle-d2/picker/zenpicker_output/meta_picker_v0_4.bin";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(BIN_PATH)?;
    eprintln!("Loaded {} bytes from {}", bytes.len(), BIN_PATH);

    let model = Model::from_bytes(&bytes)?;
    eprintln!(
        "  n_inputs={}, n_outputs={}, schema_hash=0x{:016x}",
        model.n_inputs(),
        model.n_outputs(),
        model.schema_hash()
    );

    let mut meta = MetaPicker::new(model);
    meta.validate_family_order()?;
    eprintln!(
        "  family order (parsed): {:?}",
        meta.family_at_output().unwrap()
    );

    // Sample 1: zero-feature input (just confirming the forward pass +
    // argmin work end-to-end).
    let dummy = vec![0.0f32; meta.predictor().n_inputs()];

    let pick = meta.pick(&dummy, &AllowedFamilies::all())?;
    eprintln!("\n  pick(zeros, all allowed) -> {:?}", pick);

    // Sample 2: ban everything but png. Exercises the bake's order
    // mapping.
    let only_png = AllowedFamilies::none().allow(CodecFamily::Png);
    let pick = meta.pick(&dummy, &only_png)?;
    eprintln!("  pick(zeros, only png allowed) -> {:?}", pick);

    // Sample 3: ban gif (which the bake doesn't have anyway). Should
    // behave the same as "all allowed".
    let no_gif = AllowedFamilies::all().deny(CodecFamily::Gif);
    let pick = meta.pick(&dummy, &no_gif)?;
    eprintln!("  pick(zeros, no gif) -> {:?}", pick);

    // Sample 4: only gif allowed. v0.4 includes a gif cell so this
    // returns Some(Gif). (v0.1's 5-cell bake returned None here.)
    let only_gif = AllowedFamilies::none().allow(CodecFamily::Gif);
    let pick = meta.pick(&dummy, &only_gif)?;
    eprintln!("  pick(zeros, only gif allowed) -> {:?}", pick);

    // Sample 5: pick under a time/size tradeoff. Synthetic per-family
    // encode-time estimates (ms at this image size) — in production
    // these come from the bake-time α + β·MPx fits.
    let predicted_ms = make_synthetic_ms_table();
    eprintln!("\n  synthetic encode-ms per family at source MPx:");
    for (i, fam) in CodecFamily::ALL.iter().enumerate() {
        eprintln!("    {:<5} {:>6.1} ms", fam.label(), predicted_ms[i]);
    }
    for &bytes_per_ms in &[0.0_f32, 100.0, 1_000.0, 100_000.0] {
        let pick = meta.pick_with_time_cost(
            &dummy,
            &AllowedFamilies::all(),
            &predicted_ms,
            bytes_per_ms,
        )?;
        eprintln!(
            "  pick_with_time_cost(bytes_per_ms={:>7.0}) -> {:?}",
            bytes_per_ms, pick
        );
    }

    eprintln!("\nMetaPicker roundtrip OK against the v0.4 bake.");
    Ok(())
}

/// Synthetic per-family encode-time estimates at the source's
/// resolution. Numbers are illustrative — production callers fit
/// `α + β·MPx` per codec from the lossless rows that have measured
/// timing (see coefficient's `encode_time_models.json`).
fn make_synthetic_ms_table() -> [f32; CodecFamily::COUNT] {
    let mut out = [0.0_f32; CodecFamily::COUNT];
    out[CodecFamily::Jpeg.index()] = 50.0; // jpeg fast
    out[CodecFamily::Webp.index()] = 80.0; // webp default
    out[CodecFamily::Jxl.index()] = 800.0; // jxl-modular slow lossless
    out[CodecFamily::Avif.index()] = 1500.0; // avif lossy at decent speed
    out[CodecFamily::Png.index()] = 15.0; // png fast
    out[CodecFamily::Gif.index()] = 40.0; // gif palette quantize
    out
}
