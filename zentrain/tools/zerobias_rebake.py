#!/usr/bin/env python3
"""Apply zerobias to a ZNPR bake's weights and re-quantize.

**Soft-deprecated since zenpredict-bake 0.1.1.** New bakes should set
`zerobias_tau` (and optionally `compressed`, `optimize`) directly in
the BakeRequestJson — `bake_picker.py --zerobias-tau 0.005 --compress`
or `--optimize` produces a zerobiased + compressed bake in one step,
no post-process required. This script remains as the canonical "I
have a baked `.bin` and want to apply zerobias to it without
re-training" tool — useful for measuring τ sensitivity across an
existing bake, or for rescuing shipped bakes whose source training
config is gone. Prefer the JSON pipeline for new bakes.

Reads any v2 or v3 ZNPR `.bin`, extracts each layer's weights as f32
(dequantizing F16 / I8 as needed), applies per-output-column zerobias
at threshold τ, then re-emits the bake using the **same layer dtypes**
as the input (so the output is a drop-in replacement).

For I8 layers: the re-quantization recomputes per-output scales from
the zero-biased f32 weights. Many small weights collapse to exactly 0,
giving 87+ % zero density in the i8 byte stream at τ=0.005 (per the
2026-05-13 RLE/zerobias eval). This is the data shape the
forthcoming `compressed-weights` lz4 path will exploit.

For F32 / F16 layers: zerobias produces exact-zero floats; the wire
representation is unchanged in size but the bytes gain compressibility.

The output bake's format version (v2 / v3) matches the input.

Usage:
    python3 zerobias_rebake.py <in.bin> <out.bin> --tau 0.005 [--check]

`--check` runs N random feature vectors through both bakes (input +
output) using zenpredict's parsing rules, reports max |Δ| in the
final layer's first output. Score-stability target is |Δ| < 0.01 for
the V_X 228→384→1 shape at τ ≤ 0.05; larger Δ indicates a re-quant
rounding bug.

Provenance reminder per zensim/CLAUDE.md: a non-zero-τ bake **must**
land alongside a methodology doc documenting τ + cross-corpus SROCC
(KADID, TID, CID22, AIC-3, AIC-4, KonJND).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_SIZE = 128
LAYER_ENTRY_SIZE = 48
ZNPR_MAGIC = b"ZNPR"

# Header offsets (shared v2/v3 layout through byte 72)
OFF_VERSION = 4
OFF_FLAGS = 6
OFF_N_INPUTS = 8
OFF_N_OUTPUTS = 12
OFF_N_LAYERS = 16
OFF_SCHEMA_HASH = 24
OFF_SCALER_MEAN = 32
OFF_SCALER_SCALE = 40
OFF_LAYER_TABLE = 48
OFF_FEATURE_BOUNDS = 56
OFF_METADATA = 64
OFF_OUTPUT_SPECS = 72
OFF_DISCRETE_SETS = 80
OFF_SPARSE_OVERRIDES = 88


def f32_to_f16_bits(f: float) -> int:
    """Match zenpredict-bake/src/composer.rs f32_to_f16_bits.

    Round-to-nearest-even via numpy's f16 cast (matches IEEE 754).
    """
    return int(np.float32(f).astype(np.float16).view(np.uint16).item())


def parse_bake(path: Path):
    data = path.read_bytes()
    if data[0:4] != ZNPR_MAGIC:
        raise ValueError(f"bad magic on {path}")
    version = struct.unpack_from("<H", data, OFF_VERSION)[0]
    if version not in (2, 3):
        raise ValueError(f"unsupported version {version}")
    n_inputs = struct.unpack_from("<I", data, OFF_N_INPUTS)[0]
    n_outputs = struct.unpack_from("<I", data, OFF_N_OUTPUTS)[0]
    n_layers = struct.unpack_from("<I", data, OFF_N_LAYERS)[0]
    schema_hash = struct.unpack_from("<Q", data, OFF_SCHEMA_HASH)[0]
    mean_off, mean_len = struct.unpack_from("<II", data, OFF_SCALER_MEAN)
    scale_off, scale_len = struct.unpack_from("<II", data, OFF_SCALER_SCALE)
    lt_off, lt_len = struct.unpack_from("<II", data, OFF_LAYER_TABLE)
    fb_off, fb_len = struct.unpack_from("<II", data, OFF_FEATURE_BOUNDS)
    md_off, md_len = struct.unpack_from("<II", data, OFF_METADATA)
    out_off, out_len = struct.unpack_from("<II", data, OFF_OUTPUT_SPECS)
    ds_off, ds_len = struct.unpack_from("<II", data, OFF_DISCRETE_SETS)
    so_off, so_len = struct.unpack_from("<II", data, OFF_SPARSE_OVERRIDES)

    scaler_mean = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=mean_off).copy()
    scaler_scale = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=scale_off).copy()

    layers = []
    for i in range(n_layers):
        e_off = lt_off + i * LAYER_ENTRY_SIZE
        in_dim, out_dim = struct.unpack_from("<II", data, e_off)
        act, dtype_byte, _flags = struct.unpack_from("<BBH", data, e_off + 8)
        w_off, w_len = struct.unpack_from("<II", data, e_off + 12)
        s_off, _s_len = struct.unpack_from("<II", data, e_off + 20)
        b_off, _b_len = struct.unpack_from("<II", data, e_off + 28)
        n_w = in_dim * out_dim
        if dtype_byte == 0:  # F32
            weights = np.frombuffer(data, dtype=np.float32, count=n_w, offset=w_off).reshape(in_dim, out_dim).copy()
        elif dtype_byte == 1:  # F16
            raw = np.frombuffer(data, dtype=np.uint16, count=n_w, offset=w_off).copy()
            weights = raw.view(np.float16).astype(np.float32).reshape(in_dim, out_dim)
        elif dtype_byte == 2:  # I8 with per-output f32 scales
            raw = np.frombuffer(data, dtype=np.int8, count=n_w, offset=w_off).reshape(in_dim, out_dim).astype(np.float32)
            scales = np.frombuffer(data, dtype=np.float32, count=out_dim, offset=s_off).copy()
            weights = raw * scales[None, :]
        else:
            raise ValueError(f"unknown dtype {dtype_byte}")
        biases = np.frombuffer(data, dtype=np.float32, count=out_dim, offset=b_off).copy()
        layers.append({
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "activation": int(act),
            "dtype": int(dtype_byte),
            "weights_f32": weights,
            "biases": biases,
        })

    feature_bounds = bytes(data[fb_off : fb_off + fb_len]) if fb_len else b""
    metadata = bytes(data[md_off : md_off + md_len]) if md_len else b""
    output_specs = bytes(data[out_off : out_off + out_len]) if out_len else b""
    discrete_sets = bytes(data[ds_off : ds_off + ds_len]) if ds_len else b""
    sparse_overrides = bytes(data[so_off : so_off + so_len]) if so_len else b""

    return {
        "version": version,
        "flags": struct.unpack_from("<H", data, OFF_FLAGS)[0],
        "n_inputs": int(n_inputs),
        "n_outputs": int(n_outputs),
        "schema_hash": schema_hash,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "layers": layers,
        "feature_bounds": feature_bounds,
        "metadata": metadata,
        "output_specs": output_specs,
        "discrete_sets": discrete_sets,
        "sparse_overrides": sparse_overrides,
    }


def apply_zerobias(weights: np.ndarray, tau: float, per_column: bool = False) -> np.ndarray:
    """Zerobias on a 2D [in_dim, out_dim] weight matrix.

    Default is per-LAYER (single threshold against the layer's global
    max-abs weight). Matches the 2026-05-13 RLE/zerobias eval's
    threshold semantics, which produced 87.5 % i8 zero density at
    τ=0.005 with -0.0001 CID22 SROCC.

    `per_column=True` switches to per-output thresholding (each column
    uses its own max). More structurally aligned with the per-output
    i8 quantization but less aggressive — preserves small weights in
    small-magnitude output columns. Trades higher per-column accuracy
    for less compressibility.
    """
    if not (tau > 0.0):
        return weights.copy()
    out = weights.copy()
    if per_column:
        max_ref = np.max(np.abs(out), axis=0)  # [out_dim]
        cut = tau * max_ref[None, :]
    else:
        max_ref = float(np.max(np.abs(out)))
        cut = tau * max_ref
    out[np.abs(out) < cut] = 0.0
    return out


def pad_to(buf: bytearray, alignment: int):
    while len(buf) % alignment != 0:
        buf.append(0)


def write_section(buf: bytearray, off: int, section_off: int, section_len: int):
    struct.pack_into("<II", buf, off, section_off, section_len)


def emit_bake(parsed: dict) -> bytes:
    """Re-emit a bake from a parsed dict, preserving format version + layer dtypes."""
    version = parsed["version"]
    n_inputs = parsed["n_inputs"]
    n_outputs = parsed["n_outputs"]
    n_layers = len(parsed["layers"])

    buf = bytearray(HEADER_SIZE + n_layers * LAYER_ENTRY_SIZE)
    buf[0:4] = ZNPR_MAGIC
    struct.pack_into("<H", buf, OFF_VERSION, version)
    struct.pack_into("<H", buf, OFF_FLAGS, parsed["flags"])
    struct.pack_into("<I", buf, OFF_N_INPUTS, n_inputs)
    struct.pack_into("<I", buf, OFF_N_OUTPUTS, n_outputs)
    struct.pack_into("<I", buf, OFF_N_LAYERS, n_layers)
    struct.pack_into("<Q", buf, OFF_SCHEMA_HASH, parsed["schema_hash"])

    # scaler_mean
    pad_to(buf, 4)
    sm_off = len(buf)
    buf.extend(parsed["scaler_mean"].astype(np.float32).tobytes())
    write_section(buf, OFF_SCALER_MEAN, sm_off, n_inputs * 4)
    # scaler_scale
    pad_to(buf, 4)
    ss_off = len(buf)
    buf.extend(parsed["scaler_scale"].astype(np.float32).tobytes())
    write_section(buf, OFF_SCALER_SCALE, ss_off, n_inputs * 4)

    # layer table: header LayerEntry[n] starts at HEADER_SIZE (offset 128).
    # We've already reserved space for the table; now we lay out per-layer
    # payloads (weights, scales, biases) and back-patch the entries.
    lt_off = HEADER_SIZE
    write_section(buf, OFF_LAYER_TABLE, lt_off, n_layers * LAYER_ENTRY_SIZE)

    for li, layer in enumerate(parsed["layers"]):
        entry_off = HEADER_SIZE + li * LAYER_ENTRY_SIZE
        struct.pack_into("<I", buf, entry_off, layer["in_dim"])
        struct.pack_into("<I", buf, entry_off + 4, layer["out_dim"])
        buf[entry_off + 8] = layer["activation"]
        buf[entry_off + 9] = layer["dtype"]
        # flags + reserved zero

        # Weights section
        dtype = layer["dtype"]
        if dtype == 0:  # F32
            pad_to(buf, 4)
            w_off = len(buf)
            buf.extend(layer["weights_f32"].astype(np.float32).tobytes())
            w_len = layer["weights_f32"].size * 4
        elif dtype == 1:  # F16
            pad_to(buf, 2)
            w_off = len(buf)
            for w in layer["weights_f32"].astype(np.float32).flatten():
                bits = f32_to_f16_bits(w)
                buf.extend(struct.pack("<H", bits))
            w_len = layer["weights_f32"].size * 2
        else:  # I8
            # Recompute per-output scales from the (possibly biased) f32.
            max_col = np.max(np.abs(layer["weights_f32"]), axis=0)
            scales = np.where(max_col == 0, 1.0, max_col / 127.0).astype(np.float32)
            w_off = len(buf)
            for r in range(layer["in_dim"]):
                for c in range(layer["out_dim"]):
                    s = scales[c]
                    if s == 0:
                        q = 0
                    else:
                        q = int(round(layer["weights_f32"][r, c] / s))
                        q = max(-128, min(127, q))
                    buf.append(q & 0xFF)
            w_len = layer["weights_f32"].size
        write_section(buf, entry_off + 12, w_off, w_len)

        # Scales section (I8 only)
        if dtype == 2:
            pad_to(buf, 4)
            s_off = len(buf)
            buf.extend(scales.tobytes())
            write_section(buf, entry_off + 20, s_off, layer["out_dim"] * 4)

        # Biases
        pad_to(buf, 4)
        b_off = len(buf)
        buf.extend(layer["biases"].astype(np.float32).tobytes())
        write_section(buf, entry_off + 28, b_off, layer["out_dim"] * 4)

    # Trailing sections (feature_bounds, metadata, output_specs, discrete_sets, sparse_overrides)
    if parsed["feature_bounds"]:
        pad_to(buf, 4)
        off = len(buf)
        buf.extend(parsed["feature_bounds"])
        write_section(buf, OFF_FEATURE_BOUNDS, off, len(parsed["feature_bounds"]))
    if parsed["metadata"]:
        off = len(buf)
        buf.extend(parsed["metadata"])
        write_section(buf, OFF_METADATA, off, len(parsed["metadata"]))
    if version >= 3:
        if parsed["output_specs"]:
            pad_to(buf, 4)
            off = len(buf)
            buf.extend(parsed["output_specs"])
            write_section(buf, OFF_OUTPUT_SPECS, off, len(parsed["output_specs"]))
        if parsed["discrete_sets"]:
            pad_to(buf, 4)
            off = len(buf)
            buf.extend(parsed["discrete_sets"])
            write_section(buf, OFF_DISCRETE_SETS, off, len(parsed["discrete_sets"]))
        if parsed["sparse_overrides"]:
            pad_to(buf, 4)
            off = len(buf)
            buf.extend(parsed["sparse_overrides"])
            write_section(buf, OFF_SPARSE_OVERRIDES, off, len(parsed["sparse_overrides"]))

    return bytes(buf)


# DEDUP-B2 (2026-05-26): forward delegated to
# `_predict_lib.forward_from_layers` (canonical home — sibling to
# `_picker_lib` and `_metapicker_lib`). The lib auto-detects the
# `weights_f32` / `biases` schema + the numeric `activation` codes
# (1=ReLU, 2=LeakyReLU) + mirrors the `dtype==2` i8 re-quantization
# the saved bake will apply. Bit-identical to the pre-extraction impl
# (proven by test_predict_lib::test_zerobias_rebake_*).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _predict_lib import forward_from_layers as _predict_forward_from_layers  # noqa: E402


def forward(features: np.ndarray, mean, scale, layers):
    return _predict_forward_from_layers(features, mean, scale, layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path", type=Path)
    ap.add_argument("out_path", type=Path)
    ap.add_argument("--tau", type=float, default=0.005,
                    help="zerobias threshold (default 0.005 = sweet spot per 2026-05-13 eval)")
    ap.add_argument("--per-column", action="store_true",
                    help="per-output-column threshold (default: per-layer global max)")
    ap.add_argument("--output-dtype", choices=["preserve", "f32", "f16", "i8"], default="preserve",
                    help="re-quantize every layer to this dtype on output (default: same as input)")
    ap.add_argument("--check", action="store_true",
                    help="parse output + compare 64 random predictions against input bake")
    args = ap.parse_args()

    print(f"reading {args.in_path}")
    in_data = parse_bake(args.in_path)
    print(f"  v{in_data['version']}, {in_data['n_inputs']} in / {in_data['n_outputs']} out, "
          f"{len(in_data['layers'])} layers")

    # Zero density before
    before_zeros = 0
    before_total = 0
    for layer in in_data["layers"]:
        before_zeros += int(np.sum(layer["weights_f32"] == 0.0))
        before_total += layer["weights_f32"].size
    print(f"  pre-zerobias zero density: {before_zeros / before_total * 100:.2f} %")

    # Apply zerobias
    for layer in in_data["layers"]:
        layer["weights_f32"] = apply_zerobias(layer["weights_f32"], args.tau, args.per_column)

    # Optional output dtype override
    if args.output_dtype != "preserve":
        dtype_byte = {"f32": 0, "f16": 1, "i8": 2}[args.output_dtype]
        for layer in in_data["layers"]:
            layer["dtype"] = dtype_byte

    after_zeros = 0
    for layer in in_data["layers"]:
        after_zeros += int(np.sum(layer["weights_f32"] == 0.0))
    print(f"  post-zerobias (τ={args.tau}) zero density: {after_zeros / before_total * 100:.2f} %")

    out_bytes = emit_bake(in_data)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_bytes(out_bytes)
    print(f"wrote {args.out_path} ({len(out_bytes)} bytes; in was {args.in_path.stat().st_size})")

    if args.check:
        # Re-parse output and run inference comparison
        out_data = parse_bake(args.out_path)
        rng = np.random.default_rng(0xC0FFEE)
        feats = rng.normal(size=(64, in_data["n_inputs"])).astype(np.float32)
        # Compare against the original (un-zero-biased) input model
        orig = parse_bake(args.in_path)
        y_in = forward(feats, orig["scaler_mean"], orig["scaler_scale"], orig["layers"])
        y_out = forward(feats, out_data["scaler_mean"], out_data["scaler_scale"], out_data["layers"])
        delta = np.abs(y_in - y_out).max()
        rel = delta / max(np.abs(y_in).max(), 1e-9)
        print(f"  check: max |Δ| = {delta:.6f}  (rel {rel:.2e})")
        # Loose tolerance per V_X 228→384→1 shape — meaningful SROCC change
        # requires cross-corpus eval, not random-input divergence.
        if rel > 0.10:
            print("  WARN: divergence > 10% — recheck τ / re-quantization", file=sys.stderr)


if __name__ == "__main__":
    main()
