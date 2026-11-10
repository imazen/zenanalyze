//! Exercise the process boundary, not just the JSON library entry point.
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use zenpredict::{Model, Predictor};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("input-program-{}-{id}", std::process::id()));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn cli(args: &[&std::ffi::OsStr]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_zenpredict"))
        .args(args)
        .output()
        .unwrap()
}
#[test]
fn source_model_bakes_inspects_and_repacks_through_public_cli() {
    let scratch = Scratch::new();
    let input = scratch.0.join("input.json");
    let baked = scratch.0.join("model.bin");
    let repacked = scratch.0.join("repacked.bin");
    fs::write(&input,r#"{"schema_hash":123,"scaler_mean":[0],"scaler_scale":[1],"input_program":{"source_count":2,"nodes":[{"op":"source","index":0},{"op":"source","index":1},{"op":"product","left":0,"right":1}],"outputs":[2]},"layers":[{"in_dim":1,"out_dim":1,"activation":"softplus","dtype":"f32","weights":[1],"biases":[0]}]}"#).unwrap();
    let output = cli(&["bake".as_ref(), input.as_os_str(), baked.as_os_str()]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("metadata_entries=1"));
    let output = cli(&["inspect".as_ref(), baked.as_os_str()]);
    assert!(output.status.success());
    let inspected: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(inspected["version"], 4);
    assert_eq!(inspected["n_source_inputs"], 2);
    assert_eq!(inspected["input_program"]["nodes"][2]["op"], "product");
    let output = cli(&[
        "repack".as_ref(),
        baked.as_os_str(),
        repacked.as_os_str(),
        "--dtype".as_ref(),
        "f32".as_ref(),
        "--compress".as_ref(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let a = Model::from_bytes(&fs::read(&baked).unwrap()).unwrap();
    let b = Model::from_bytes(&fs::read(&repacked).unwrap()).unwrap();
    assert_eq!(a.input_program(), b.input_program());
    assert_eq!(b.version(), 4);
    let mut old = Predictor::new(&a).unwrap();
    let mut new = Predictor::new(&b).unwrap();
    for x in [[0., 1.], [2., 3.], [0.5, 0.25]] {
        assert_eq!(
            old.predict_from_source(&x).unwrap(),
            new.predict_from_source(&x).unwrap()
        );
    }
}
#[test]
fn malformed_output_is_rejected_before_creating_a_bake() {
    let scratch = Scratch::new();
    let input = scratch.0.join("bad.json");
    let output = scratch.0.join("bad.bin");
    fs::write(&input,r#"{"schema_hash":0,"scaler_mean":[0],"scaler_scale":[1],"metadata":[{"key":"zentrain.feature_transforms","type":"bytes","hex":"ff"}],"layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[1],"biases":[0]}]}"#).unwrap();
    let result = cli(&["bake".as_ref(), input.as_os_str(), output.as_os_str()]);
    assert!(!result.status.success());
    assert!(!output.exists());
}
