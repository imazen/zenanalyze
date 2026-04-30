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


DEFAULT_SCHEMA_VERSION_TAG = "zenpicker.v1.generic"


def derive_extra_axes(n_inputs: int, feat_cols: list[str], model: dict) -> list[str]:
    """Determine the engineered-axes list for schema_hash computation.

    Resolution order, most explicit wins:

    1. **`model["extra_axes"]`** — codec explicitly declares the
       names of every input column past the raw feature columns.
       Most reusable; recommended for new codecs.

    2. **Built-in known layouts** — the legacy zenjpeg
       `shared-mlp.distill+icc` layout
       (n_feat + 4 size + 5 polynomials + n_feat zq×feat + 1 icc).
       Bumping `SCHEMA_VERSION_TAG` to a layout name forces re-bake.

    3. **Fallback** — synthesize unnamed `aux_<i>` axes. The
       schema_hash will still be stable per (n_inputs, feat_cols),
       but the codec must hash the same axis list at compile time.
       Emit a warning so the codec author notices.
    """
    explicit = model.get("extra_axes")
    if explicit is not None:
        n_aux = n_inputs - len(feat_cols)
        if len(explicit) != n_aux:
            raise SystemExit(
                f"model.extra_axes has {len(explicit)} entries but n_inputs - n_feats = {n_aux}"
            )
        return list(explicit)

    n_feat = len(feat_cols)
    # Legacy zenjpeg layout: 8/19 feats + 4 size + 5 poly + n_feat cross + 1 icc.
    if n_inputs == n_feat + 4 + 5 + n_feat + 1:
        return (
            ["size_tiny", "size_small", "size_medium", "size_large"]
            + ["log_pixels", "log_pixels_sq", "zq_norm", "zq_norm_sq", "zq_norm_x_log_pixels"]
            + [f"zq_x_{c}" for c in feat_cols]
            + ["icc_bytes"]
        )

    # Fallback: synthesize anonymous names; codec must hash these
    # exact strings at compile time.
    sys.stderr.write(
        f"WARNING: bake_picker received n_inputs={n_inputs} with no `extra_axes` "
        f"in model JSON and no built-in layout match (n_feat={n_feat}). "
        f"Using anonymous 'aux_*' axis names — codec must use the same.\n"
    )
    n_aux = n_inputs - n_feat
    return [f"aux_{i:02d}" for i in range(n_aux)]


def schema_version_tag(model: dict) -> str:
    """The model JSON may declare its own version tag. Falls back to
    the crate-wide default when omitted."""
    return model.get("schema_version_tag", DEFAULT_SCHEMA_VERSION_TAG)


# Kept for backwards compatibility with older callers that import
# this constant. New code should call `schema_version_tag(model)`.
SCHEMA_VERSION_TAG = DEFAULT_SCHEMA_VERSION_TAG


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


def bake(
    model_path: Path,
    out_path: Path,
    dtype: str,
    manifest_path: Path | None,
    allow_unsafe: bool = False,
) -> None:
    model = json.loads(model_path.read_text())
    # Safety gate — refuse to bake a model whose training-side
    # diagnostics flagged a danger. Reviewers can still inspect the
    # JSON (we don't delete it); --allow-unsafe overrides for
    # intentional violations. Bakes without `safety_report` (older
    # JSONs) are tolerated with a warning so legacy data still works.
    sr = model.get("safety_report")
    if sr is None:
        sys.stderr.write(
            "WARNING: model JSON has no `safety_report` block. Re-train "
            "with the current train_hybrid.py to get safety diagnostics.\n"
        )
    elif not sr.get("passed", True):
        violations = sr.get("violations") or []
        sys.stderr.write(
            "\n" + "=" * 70 + "\n"
            f"  ⚠ BAKE REFUSED — model has {len(violations)} unresolved safety violation(s)\n"
            + "=" * 70 + "\n"
        )
        for v in violations:
            sys.stderr.write(f"  • {v}\n")
        sys.stderr.write("=" * 70 + "\n")
        if not allow_unsafe:
            sys.stderr.write(
                "Pass --allow-unsafe to bake anyway (only when the violation "
                "is intentional and reviewed).\n"
            )
            raise SystemExit(2)
        sys.stderr.write("--allow-unsafe set — baking despite violations.\n")
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

    extra_axes = derive_extra_axes(n_inputs, feat_cols, model)
    sh = schema_hash(feat_cols, extra_axes, schema_version_tag(model))

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
    # (the distill / shared-MLP / hybrid scripts all do).
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
        # Hybrid-heads (v0.2) manifest passthrough — describes the
        # categorical-vs-scalar layout of the n_outputs vector. Codec
        # crate's compile-time CONFIGS table uses this to slice
        # `predict()` output into bytes_log / chroma_scale / lambda /
        # … sub-ranges. Optional: pure-categorical models omit it.
        if "hybrid_heads_manifest" in model:
            manifest["hybrid_heads"] = model["hybrid_heads_manifest"]
        # Safety-profile passthrough — `safety_profile`,
        # `training_objective`, and `reach_safety` are emitted by
        # train_hybrid.py for both size_optimal and zensim_strict
        # bakes. The codec consumer reads `reach_safety.by_zq[<zq>].safe`
        # and ANDs it into its constraint mask before argmin.
        for key in ("safety_profile", "training_objective", "reach_safety"):
            if key in model:
                manifest[key] = model[key]
        # Safety-report passthrough — codec runtime can refuse to load
        # a bake whose `safety_report.passed` is false even after the
        # bake-time --allow-unsafe override (defense in depth).
        if "safety_report" in model:
            manifest["safety_report"] = model["safety_report"]
            # Lift feature_bounds to the top-level of the manifest so
            # codecs can compile a `FEATURE_BOUNDS: &[(f32, f32)]` const
            # from a stable path. The full per-percentile dict stays
            # inside safety_report.diagnostics; the lifted form is a
            # compact list aligned to feat_cols, picking (p01, p99) by
            # default. Codecs that want different bounds can read the
            # full dict instead.
            fb = (
                model["safety_report"]
                .get("diagnostics", {})
                .get("feature_bounds")
            )
            if fb:
                lifted = []
                for col in feat_cols:
                    s = fb.get(col)
                    if s and s.get("p01") is not None and s.get("p99") is not None:
                        lifted.append({"feat": col, "low": s["p01"], "high": s["p99"]})
                    else:
                        # If we have no usable percentiles, fall back
                        # to (-inf, +inf) so the runtime gate doesn't
                        # spuriously reject this column.
                        lifted.append({"feat": col, "low": float("-inf"), "high": float("inf")})
                manifest["feature_bounds_p01_p99"] = lifted
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        sys.stderr.write(f"wrote manifest {manifest_out}\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path, help="sklearn-side JSON dump")
    ap.add_argument("--out", required=True, type=Path, help="output .bin path")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"], help="weight storage dtype")
    ap.add_argument("--manifest", type=Path, help="explicit manifest output path (default: <out>.manifest.json)")
    ap.add_argument(
        "--allow-unsafe",
        action="store_true",
        help="Bake even when the model JSON's safety_report.passed is false. "
        "Use only when the violation is intentional and reviewed.",
    )
    args = ap.parse_args(argv)
    bake(args.model, args.out, args.dtype, args.manifest, allow_unsafe=args.allow_unsafe)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
