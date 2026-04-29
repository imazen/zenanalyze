#!/usr/bin/env python3
"""
Bake an sklearn MLPRegressor JSON dump into the zenpicker v1 binary format.

Inputs:
- `--model JSON`    : sklearn-side model file (see fields below)
- `--manifest JSON` : optional companion file describing per-output
                      config metadata. When present, gets re-emitted
                      into a separate `<base>.manifest.json` next to
                      the `.bin`. The codec crate compares its
                      compile-time CONFIGS table against this.
- `--out FILE.bin`  : output path
- `--dtype f32|f16` : weight storage dtype (default f32)

Required model JSON fields:
  n_inputs, n_outputs, scaler_mean, scaler_scale, layers,
  feat_cols (used for schema_hash), activation (string: "relu")

Each `layers[i]` has `W` (shape [in_dim, out_dim]) and `b` (length out_dim).
The first n_layers-1 layers use the model's `activation`; the final layer
is identity (regression head).

Output binary v1 layout matches `zenpicker::Model::from_bytes`:

    Header (32 bytes):
      magic         "ZNPK"
      version u16   1
      header_size u16  32
      n_inputs u32
      n_outputs u32
      n_layers u32
      schema_hash u64
      flags u32
    Scaler:
      f32[n_inputs] mean
      f32[n_inputs] scale
    Per-layer:
      in_dim u32
      out_dim u32
      activation u8     (0=Identity, 1=ReLU)
      weight_dtype u8   (0=F32, 1=F16)
      reserved u8 u8
      W (row-major, in_dim major), in_dim*out_dim values in dtype
      b (f32) out_dim values
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"ZNPK"
FORMAT_VERSION = 1
HEADER_SIZE = 32

ACTIVATION_IDENTITY = 0
ACTIVATION_RELU = 1
DTYPE_F32 = 0
DTYPE_F16 = 1


def schema_hash(feat_cols: list[str], extra_axes: list[str], version_tag: str) -> int:
    """Stable u64 hash of the input schema.

    The codec crate compares this against its compile-time
    expectation. Mismatch → hard error at load. Bumping
    `version_tag` forces a re-bake when the schema changes shape.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(version_tag.encode("utf-8"))
    h.update(b"\x00")
    for c in feat_cols:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"||")
    for a in extra_axes:
        h.update(a.encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little")


def derive_extra_axes(n_inputs: int, feat_cols: list[str]) -> list[str]:
    """Reverse-engineer which engineered features the model used.

    The distill / shared-MLP fit scripts use this layout:
      19 raw feats + 4 size onehot + 5 zq/log_pixels polynomials
      + 19 zq×feat crosses + 1 icc_bytes = 48 inputs

    The shared-MLP-only path used 4+1+1=6 inputs of polynomials and
    no icc, giving 49. Codify the layouts here so the schema_hash is
    stable per layout family.
    """
    n_feat = len(feat_cols)
    if n_inputs == n_feat + 4 + 5 + n_feat + 1:
        return (
            ["size_tiny", "size_small", "size_medium", "size_large"]
            + ["log_pixels", "log_pixels_sq", "zq_norm", "zq_norm_sq", "zq_norm_x_log_pixels"]
            + [f"zq_x_{c}" for c in feat_cols]
            + ["icc_bytes"]
        )
    raise SystemExit(
        f"unrecognized input layout: {n_inputs} inputs vs {n_feat} feat cols. "
        f"add the layout to derive_extra_axes() and bump SCHEMA_VERSION_TAG."
    )


SCHEMA_VERSION_TAG = "zenpicker.v1.shared-mlp.distill+icc"


def encode_dtype(arr: np.ndarray, dtype: str) -> bytes:
    if dtype == "f32":
        return arr.astype("<f4").tobytes()
    elif dtype == "f16":
        return arr.astype("<f2").tobytes()
    raise ValueError(f"unknown dtype {dtype!r}")


def write_layer(out: bytes, W: np.ndarray, b: np.ndarray, activation: int, dtype: str) -> bytes:
    in_dim, out_dim = W.shape
    out += struct.pack("<II", in_dim, out_dim)
    out += struct.pack("<BBBB", activation, DTYPE_F32 if dtype == "f32" else DTYPE_F16, 0, 0)
    out += encode_dtype(W, dtype)
    # Pad to keep f32 biases 4-aligned. f32 weights are already 4-aligned;
    # f16 weights might end on a 2-aligned boundary if the count is odd.
    n_weights = in_dim * out_dim
    if dtype == "f16" and n_weights % 2 == 1:
        out += b"\x00\x00"  # 2 bytes pad to reach 4-alignment
    out += encode_dtype(b, "f32")
    return out


def bake(model_path: Path, out_path: Path, dtype: str, manifest_path: Path | None) -> None:
    model = json.loads(model_path.read_text())
    n_inputs = int(model["n_inputs"])
    layers = model["layers"]
    n_layers = len(layers)
    feat_cols = list(model["feat_cols"])
    activation = model.get("activation", "relu")
    # Tolerate both "n_outputs" (shared-MLP fit) and "n_configs"
    # (distill fit; same meaning, different field name in the JSON).
    n_outputs = int(
        model.get("n_outputs")
        if model.get("n_outputs") is not None
        else model.get("n_configs", len(layers[-1]["b"]))
    )
    if activation != "relu":
        raise SystemExit(f"unsupported activation {activation!r}; only relu is wired today")

    extra_axes = derive_extra_axes(n_inputs, feat_cols)
    sh = schema_hash(feat_cols, extra_axes, SCHEMA_VERSION_TAG)

    # Header.
    blob = b""
    blob += MAGIC
    blob += struct.pack("<HH", FORMAT_VERSION, HEADER_SIZE)
    blob += struct.pack("<III", n_inputs, n_outputs, n_layers)
    blob += struct.pack("<Q", sh)
    blob += struct.pack("<I", 0)  # flags
    assert len(blob) == HEADER_SIZE

    # Scaler.
    blob += np.array(model["scaler_mean"], dtype="<f4").tobytes()
    blob += np.array(model["scaler_scale"], dtype="<f4").tobytes()

    # Layers — last layer identity, all earlier ReLU.
    last_idx = n_layers - 1
    for i, layer in enumerate(layers):
        W = np.array(layer["W"], dtype=np.float32)
        b = np.array(layer["b"], dtype=np.float32)
        if W.ndim != 2:
            raise SystemExit(f"layer {i} W has bad ndim {W.ndim}")
        in_dim, out_dim = W.shape
        if i == 0 and in_dim != n_inputs:
            raise SystemExit(f"layer 0 in_dim {in_dim} != header n_inputs {n_inputs}")
        if i == last_idx and out_dim != n_outputs:
            raise SystemExit(f"final layer out_dim {out_dim} != header n_outputs {n_outputs}")
        if b.shape != (out_dim,):
            raise SystemExit(f"layer {i} bias shape {b.shape} != (out_dim={out_dim},)")
        act = ACTIVATION_IDENTITY if i == last_idx else ACTIVATION_RELU
        blob = write_layer(blob, W, b, act, dtype)

    out_path.write_bytes(blob)
    sys.stderr.write(
        f"baked {out_path} ({len(blob)} bytes), schema_hash=0x{sh:016x}, "
        f"dtype={dtype}, n_inputs={n_inputs}, n_outputs={n_outputs}, "
        f"n_layers={n_layers}\n"
    )

    # Manifest — re-emit the per-output config metadata that the codec
    # crate consumes to build its compile-time CONFIGS table. Only
    # written when the input model carries a `config_names` field
    # (the distill / shared-MLP scripts do).
    if "config_names" in model:
        manifest_out = manifest_path or out_path.with_suffix(".manifest.json")
        cfg_names = model["config_names"]
        manifest = {
            "schema_hash": f"0x{sh:016x}",
            "schema_version_tag": SCHEMA_VERSION_TAG,
            "feat_cols": feat_cols,
            "extra_axes": extra_axes,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "configs": {str(k): v for k, v in cfg_names.items()},
        }
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        sys.stderr.write(f"wrote manifest {manifest_out}\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="sklearn-side JSON dump")
    ap.add_argument("--out", required=True, type=Path, help="output .bin path")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"], help="weight storage dtype")
    ap.add_argument("--manifest", type=Path, help="explicit manifest output path (default: <out>.manifest.json)")
    args = ap.parse_args(argv)
    bake(args.model, args.out, args.dtype, args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
