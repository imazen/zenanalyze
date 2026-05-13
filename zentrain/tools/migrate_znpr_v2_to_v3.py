#!/usr/bin/env python3
"""ZNPR v2 → v3 migration tool.

zenpredict 0.2.0 drops v2 parsing entirely. Existing baked .bin files
(zensim V0_18, zenwebp picker v0.1, zenavif rav1e picker v0.1.1, every
zenjxl / zenpicker / content-classifier bake) are stuck in v2 unless
they're rewritten.

The v3 wire format keeps v2's first 72 header bytes intact and packs
three optional sections (`output_specs`, `discrete_sets`,
`sparse_overrides`) into bytes that v2 used as `reserved`. Layer
payloads (weights, scales, biases), the scaler arrays, the feature
bounds blob, and the metadata blob are unchanged.

So the migration is mechanical:
  1. Parse the v2 header + sections.
  2. Rewrite the header with `version = 3`, the three new section
     offsets/lengths zeroed, and the `reserved` tail shrunk from
     `[u32; 12]` to `[u32; 8]`.
  3. Copy the remaining file contents byte-for-byte.

The output v3 .bin loads cleanly under zenpredict 0.2.0 and produces
bit-identical inference results to the v2 input under zenpredict 0.1.0.

Verify with `--check`: re-parses the v3 output, runs N random feature
vectors through both models, and reports the max |Δ|. Should be 0.0
for F32/I8 layers (lossless reencode). F16 layers may differ in the
last bit if your check harness rounds f32→f16 differently than the
original bake; that's not a migration bug.

Usage:
    python3 migrate_znpr_v2_to_v3.py <in.bin> <out.bin>
    python3 migrate_znpr_v2_to_v3.py <in.bin> <out.bin> --check
    python3 migrate_znpr_v2_to_v3.py --batch <manifest.tsv>

Batch mode reads a TSV of `in_path\tout_path` pairs (no header) and
migrates each. Use this for the imazen hard-fork rollout to convert
every embedded .bin in one pass.
"""
import argparse
import struct
import sys
from pathlib import Path

HEADER_SIZE = 128
LAYER_ENTRY_SIZE = 48
ZNPR_MAGIC = b"ZNPR"

# Header field offsets (absolute, little-endian throughout)
OFF_MAGIC = 0
OFF_VERSION = 4
OFF_FLAGS = 6
OFF_N_INPUTS = 8
OFF_N_OUTPUTS = 12
OFF_N_LAYERS = 16
OFF_PAD0 = 20
OFF_SCHEMA_HASH = 24
OFF_SCALER_MEAN = 32
OFF_SCALER_SCALE = 40
OFF_LAYER_TABLE = 48
OFF_FEATURE_BOUNDS = 56
OFF_METADATA = 64
# v3-only:
OFF_OUTPUT_SPECS = 72
OFF_DISCRETE_SETS = 80
OFF_SPARSE_OVERRIDES = 88
OFF_RESERVED = 96  # [u32; 8] in v3 (vs [u32; 12] in v2)


def parse_v2_header(data: bytes) -> dict:
    """Return parsed v2 header as a dict. Raise on malformed input."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"file too short for header: {len(data)} < {HEADER_SIZE}")
    if data[0:4] != ZNPR_MAGIC:
        raise ValueError(f"bad magic: {data[0:4]!r}")
    version = struct.unpack_from("<H", data, OFF_VERSION)[0]
    if version != 2:
        raise ValueError(f"expected v2, got v{version}")
    return {
        "magic": data[0:4],
        "version": 2,
        "flags": struct.unpack_from("<H", data, OFF_FLAGS)[0],
        "n_inputs": struct.unpack_from("<I", data, OFF_N_INPUTS)[0],
        "n_outputs": struct.unpack_from("<I", data, OFF_N_OUTPUTS)[0],
        "n_layers": struct.unpack_from("<I", data, OFF_N_LAYERS)[0],
        "schema_hash": struct.unpack_from("<Q", data, OFF_SCHEMA_HASH)[0],
        "scaler_mean": struct.unpack_from("<II", data, OFF_SCALER_MEAN),
        "scaler_scale": struct.unpack_from("<II", data, OFF_SCALER_SCALE),
        "layer_table": struct.unpack_from("<II", data, OFF_LAYER_TABLE),
        "feature_bounds": struct.unpack_from("<II", data, OFF_FEATURE_BOUNDS),
        "metadata": struct.unpack_from("<II", data, OFF_METADATA),
    }


def build_v3_header(h: dict) -> bytes:
    """Construct a 128-byte v3 header from a parsed v2 header dict."""
    out = bytearray(HEADER_SIZE)
    out[0:4] = ZNPR_MAGIC
    struct.pack_into("<H", out, OFF_VERSION, 3)
    struct.pack_into("<H", out, OFF_FLAGS, h["flags"])
    struct.pack_into("<I", out, OFF_N_INPUTS, h["n_inputs"])
    struct.pack_into("<I", out, OFF_N_OUTPUTS, h["n_outputs"])
    struct.pack_into("<I", out, OFF_N_LAYERS, h["n_layers"])
    # pad0 stays zero
    struct.pack_into("<Q", out, OFF_SCHEMA_HASH, h["schema_hash"])
    struct.pack_into("<II", out, OFF_SCALER_MEAN, *h["scaler_mean"])
    struct.pack_into("<II", out, OFF_SCALER_SCALE, *h["scaler_scale"])
    struct.pack_into("<II", out, OFF_LAYER_TABLE, *h["layer_table"])
    struct.pack_into("<II", out, OFF_FEATURE_BOUNDS, *h["feature_bounds"])
    struct.pack_into("<II", out, OFF_METADATA, *h["metadata"])
    # v3 additions — all zero-length (absent) for a clean migration.
    # output_specs, discrete_sets, sparse_overrides at OFF_72/80/88.
    # OFF_RESERVED..128 stays zero.
    return bytes(out)


def migrate(in_bytes: bytes) -> bytes:
    """Migrate a v2 .bin to v3. Returns the new bytes."""
    h = parse_v2_header(in_bytes)
    new_header = build_v3_header(h)
    # Body (everything after the 128-byte header) is byte-identical.
    return new_header + in_bytes[HEADER_SIZE:]


def check_roundtrip(v2_path: Path, v3_path: Path, n_samples: int = 64) -> bool:
    """Optional: load both bakes via numpy parser, run N random inputs,
    compare predictions. Returns True if max |Δ| < 1e-5 for f32/i8 layers.

    The v2 parser here mirrors `score_unified_with_bake.py:parse_bake_v2`
    so we don't need zenpredict to verify.
    """
    import numpy as np

    def parse_any_znpr(path: Path):
        data = path.read_bytes()
        if data[0:4] != ZNPR_MAGIC:
            raise ValueError(f"bad magic on {path}")
        version = struct.unpack_from("<H", data, OFF_VERSION)[0]
        if version not in (2, 3):
            raise ValueError(f"unsupported version {version}")
        n_inputs = struct.unpack_from("<I", data, OFF_N_INPUTS)[0]
        n_layers = struct.unpack_from("<I", data, OFF_N_LAYERS)[0]
        mean_off, _ = struct.unpack_from("<II", data, OFF_SCALER_MEAN)
        scale_off, _ = struct.unpack_from("<II", data, OFF_SCALER_SCALE)
        lt_off, _ = struct.unpack_from("<II", data, OFF_LAYER_TABLE)
        mean = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=mean_off).copy()
        scale = np.frombuffer(data, dtype=np.float32, count=n_inputs, offset=scale_off).copy()
        layers = []
        for i in range(n_layers):
            e_off = lt_off + i * LAYER_ENTRY_SIZE
            in_dim, out_dim = struct.unpack_from("<II", data, e_off)
            act, dtype_byte, _flags = struct.unpack_from("<BBH", data, e_off + 8)
            w_off, _w_len = struct.unpack_from("<II", data, e_off + 12)
            s_off, _s_len = struct.unpack_from("<II", data, e_off + 20)
            b_off, _b_len = struct.unpack_from("<II", data, e_off + 28)
            n_w = in_dim * out_dim
            if dtype_byte == 0:  # F32
                w = np.frombuffer(data, dtype=np.float32, count=n_w, offset=w_off).reshape(in_dim, out_dim).copy()
            elif dtype_byte == 1:  # F16
                raw = np.frombuffer(data, dtype=np.uint16, count=n_w, offset=w_off).copy()
                w = raw.view(np.float16).astype(np.float32).reshape(in_dim, out_dim)
            elif dtype_byte == 2:  # I8 with per-output scales
                raw = np.frombuffer(data, dtype=np.int8, count=n_w, offset=w_off).reshape(in_dim, out_dim).astype(np.float32)
                scales = np.frombuffer(data, dtype=np.float32, count=out_dim, offset=s_off).copy()
                w = raw * scales[None, :]
            else:
                raise ValueError(f"unknown weight_dtype {dtype_byte}")
            biases = np.frombuffer(data, dtype=np.float32, count=out_dim, offset=b_off).copy()
            layers.append((act, w, biases))
        return mean, scale, layers

    def forward(features, mean, scale, layers):
        x = (features - mean[None, :]) / scale[None, :]
        for i, (act, w, b) in enumerate(layers):
            x = x @ w + b[None, :]
            if act == 2 and i < len(layers) - 1:  # LeakyReLU
                x = np.where(x > 0, x, x * 0.01)
            elif act == 1 and i < len(layers) - 1:  # ReLU
                x = np.maximum(x, 0)
        return x

    m2, s2, l2 = parse_any_znpr(v2_path)
    m3, s3, l3 = parse_any_znpr(v3_path)
    assert m2.shape == m3.shape and np.array_equal(m2, m3), "scaler mismatch"
    rng = np.random.default_rng(0xC0FFEE)
    feats = rng.normal(size=(n_samples, len(m2))).astype(np.float32)
    out2 = forward(feats, m2, s2, l2)
    out3 = forward(feats, m3, s3, l3)
    delta = np.abs(out2 - out3).max()
    print(f"  check: max |Δ| = {delta:.2e} over {n_samples} samples × {out2.shape[1]} outputs")
    return delta < 1e-5


def migrate_file(in_path: Path, out_path: Path, check: bool) -> bool:
    print(f"migrate {in_path}")
    in_bytes = in_path.read_bytes()
    try:
        out_bytes = migrate(in_bytes)
    except ValueError as e:
        print(f"  SKIP: {e}", file=sys.stderr)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out_bytes)
    print(f"  -> {out_path} ({len(out_bytes)} bytes, {len(in_bytes)} in)")
    if check:
        ok = check_roundtrip(in_path, out_path)
        if not ok:
            print("  FAIL: roundtrip delta exceeded 1e-5", file=sys.stderr)
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="ZNPR v2 → v3 hard-fork migration")
    ap.add_argument("in_path", type=Path, nargs="?")
    ap.add_argument("out_path", type=Path, nargs="?")
    ap.add_argument("--check", action="store_true",
                    help="verify roundtrip: parse v3 output, compare predictions to v2 input on 64 random samples")
    ap.add_argument("--batch", type=Path,
                    help="TSV manifest of in\\tout pairs (no header); migrate each")
    args = ap.parse_args()

    if args.batch:
        if args.in_path or args.out_path:
            ap.error("--batch is mutually exclusive with positional args")
        failed = 0
        with args.batch.open() as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"BAD LINE: {line!r}", file=sys.stderr)
                    failed += 1
                    continue
                in_path, out_path = Path(parts[0]), Path(parts[1])
                if not migrate_file(in_path, out_path, args.check):
                    failed += 1
        sys.exit(1 if failed else 0)

    if not args.in_path or not args.out_path:
        ap.error("usage: migrate_znpr_v2_to_v3.py <in.bin> <out.bin> [--check]")
    sys.exit(0 if migrate_file(args.in_path, args.out_path, args.check) else 1)


if __name__ == "__main__":
    main()
