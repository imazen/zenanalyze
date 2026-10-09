//! ZNPR v4 graph bakes load in zenpredict, but every viz entry point walks
//! `layers()` as a chain. Each must refuse a graph with a clean error —
//! never panic, never return numbers for a function it doesn't compute.

use zenpredict_viz::{forward_with_taps_native, layer_weights_native, parse_bake_native};

fn gated_head() -> Vec<u8> {
    let bytes = zenpredict_bake::bake_from_json_str(include_str!(
        "../../zenpredict-bake/examples/gated_head.json"
    ))
    .expect("bake gated_head.json");
    // It is a valid model: the refusal must come from the viz, not the parser.
    assert_eq!(
        zenpredict::Model::from_bytes(&bytes).unwrap().version(),
        zenpredict::GRAPH_FORMAT_VERSION
    );
    bytes
}

fn assert_refused(what: &str, r: Result<impl core::fmt::Debug, String>) {
    match r {
        Err(e) => assert!(e.contains("not supported by zenpredict-viz"), "{what}: {e}"),
        Ok(v) => panic!("{what} accepted a v4 graph: {v:?}"),
    }
}

#[test]
fn parse_bake_refuses_graphs() {
    assert_refused(
        "parse_bake_native",
        parse_bake_native(&gated_head()).map(|_| ()),
    );
}

#[test]
fn forward_with_taps_refuses_graphs() {
    let x = [0.5f32, -0.25, 1.0, 0.5, -1.0, 2.0];
    assert_refused(
        "forward_with_taps_native",
        forward_with_taps_native(&gated_head(), &x).map(|_| ()),
    );
}

#[test]
fn layer_weights_refuses_graphs() {
    assert_refused(
        "layer_weights_native",
        layer_weights_native(&gated_head(), 0),
    );
}

/// The ONNX exporter (a binary) exits non-zero with the same message and
/// writes no file.
#[cfg(feature = "onnx-export")]
#[test]
fn znpr2onnx_refuses_graphs() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("graph_refusal_onnx");
    std::fs::create_dir_all(&dir).unwrap();
    let bin = dir.join("gated.bin");
    let onnx = dir.join("gated.onnx");
    let _ = std::fs::remove_file(&onnx);
    std::fs::write(&bin, gated_head()).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_znpr2onnx"))
        .arg(&bin)
        .arg(&onnx)
        .output()
        .expect("run znpr2onnx");
    assert_eq!(
        out.status.code(),
        Some(1),
        "znpr2onnx must refuse a v4 graph"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not supported by zenpredict-viz"),
        "{stderr}"
    );
    assert!(!onnx.exists(), "no ONNX file may be written");
}
