//! Qualify an existing bake with independently prepared compact inputs.
//! Usage: input_program_parity MODEL PROGRAM_JSON CASES_JSON OUTPUT
//! Each case is {"source":[...],"prepared":[...]}; prepared values are before
//! existing feature transforms/scaling. No labels or quality fitting are read.
use serde::Deserialize;
use std::{error::Error, fs};
use zenpredict::{Model, Predictor};
use zenpredict_bake::input_program::{InputProgramJson, attach_input_program};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    source: Vec<f32>,
    prepared: Vec<f32>,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: input_program_parity MODEL PROGRAM_JSON CASES_JSON OUTPUT".into());
    }
    let original = fs::read(&args[1])?;
    let program: InputProgramJson = serde_json::from_slice(&fs::read(&args[2])?)?;
    let program = program.build()?;
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(&args[3])?)?;
    if cases.is_empty() {
        return Err("at least one parity case is required".into());
    }
    let attached = attach_input_program(&original, &program)?;
    let baseline = Model::from_bytes(&original)?;
    let composed = Model::from_bytes(&attached)?;
    let mut old = Predictor::new(&baseline).unwrap();
    let mut new = Predictor::new(&composed).unwrap();
    let mut scratch = vec![0.; program.scratch_len()];
    let mut compact = vec![0.; program.outputs().len()];
    for (i, case) in cases.iter().enumerate() {
        program.apply(&case.source, &mut scratch, &mut compact)?;
        if compact
            .iter()
            .map(|v| v.to_bits())
            .ne(case.prepared.iter().map(|v| v.to_bits()))
        {
            return Err(format!("case {i}: prepared inputs differ").into());
        }
        let a = old.predict_transformed(&case.prepared)?;
        let b = new.predict_from_source(&case.source)?;
        if a.iter()
            .map(|v| v.to_bits())
            .ne(b.iter().map(|v| v.to_bits()))
        {
            return Err(format!("case {i}: scores differ").into());
        }
    }
    // Never replace an existing artifact. Successful output is a v4 bake.
    use std::io::Write;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[4])?;
    output.write_all(&attached)?;
    println!(
        "{} cases: bit-identical prepared inputs and predictions; v{} -> v{}; {} source columns -> {} compact inputs",
        cases.len(),
        baseline.version(),
        composed.version(),
        composed.n_source_inputs(),
        composed.n_inputs()
    );
    Ok(())
}
