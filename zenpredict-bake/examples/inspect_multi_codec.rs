//! Smoke-test helper: load a `.bin` and print whether it carries the
//! ZNPR v3.2 `multi_codec_schema` section, plus a brief dump of each
//! per-codec map. Used by the joint-bake smoke pipeline:
//!
//! ```bash
//! train_multi_codec.py ... --joint-bake joint.json
//! bake_picker.py --model joint.json --out joint.bin --dtype f32
//! cargo run --release -p zenpredict-bake --example inspect_multi_codec \
//!     -- joint.bin
//! ```
//!
//! Also exercises a synthetic `predict_multi_codec(0, ...)` call on
//! codec 0 to confirm the runtime forward pass works on the bake.

use std::env;
use std::fs;
use std::process::ExitCode;

use zenpredict::{Model, Predictor};

#[repr(C, align(16))]
struct Aligned(Vec<u8>);

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: inspect_multi_codec <model.bin>");
        return ExitCode::from(2);
    }
    let bytes = match fs::read(&args[1]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {}: {e}", args[1]);
            return ExitCode::from(2);
        }
    };
    let aligned = Aligned(bytes);
    let model = match Model::from_bytes(&aligned.0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Model::from_bytes failed: {e:?}");
            return ExitCode::from(3);
        }
    };
    println!(
        "Loaded model: schema_hash=0x{:016x} n_inputs={} n_outputs={}",
        model.schema_hash(),
        model.n_inputs(),
        model.n_outputs()
    );

    if !model.has_multi_codec_schema() {
        println!("multi_codec_schema: ABSENT (single-codec bake)");
        return ExitCode::SUCCESS;
    }
    let schema = match model.multi_codec_schema() {
        Some(s) => s,
        None => {
            eprintln!("has_multi_codec_schema=true but parse returned None");
            return ExitCode::from(4);
        }
    };
    println!(
        "multi_codec_schema: union_feat_count={} n_codecs={}",
        schema.union_feat_count, schema.n_codecs
    );
    for (i, m) in schema.per_codec.iter().enumerate() {
        println!(
            "  codec[{}] {:<10} slots={:<3} output_range=[{},{}) head_n_cells={} head_n_heads={}",
            i,
            m.codec_name,
            m.union_slot_for_codec_feat.len(),
            m.output_range.0,
            m.output_range.1,
            m.head_meta.n_cells,
            m.head_meta.n_heads,
        );
    }

    // Quick forward-pass smoke for codec 0: feed zeros for its natural
    // feature vector + zero size_class / log_pixels / zq_norm. We just
    // want to confirm predict_multi_codec doesn't error and returns a
    // sized slice. Real correctness is exercised in the integration
    // tests (json_round_trip_with_multi_codec_schema etc.).
    let n_feats_codec0 = schema.per_codec[0].union_slot_for_codec_feat.len();
    let feats = vec![0.0_f32; n_feats_codec0];
    let mut p = Predictor::new(&model);
    match p.predict_multi_codec(0, &feats, 0, 0.0, 0.0) {
        Ok(out) => println!(
            "predict_multi_codec(0, zeros) → output_len={} first3={:?}",
            out.len(),
            &out[..out.len().min(3)]
        ),
        Err(e) => {
            eprintln!("predict_multi_codec failed: {e:?}");
            return ExitCode::from(5);
        }
    }
    ExitCode::SUCCESS
}
